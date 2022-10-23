import inspect, traceback
import time
from mimetypes import init
from typing import Callable, List, Tuple, Optional, Union

import numpy as np
import torch
import torchvision
import torchvision.transforms as T

import PIL
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import LMSDiscreteScheduler
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import deprecate, logging
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

from sdgrpcserver import images
from sdgrpcserver.pipeline.old_schedulers.scheduling_utils import OldSchedulerMixin
from sdgrpcserver.pipeline.lpw_text_embedding import LPWTextEmbedding

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class UnifiedMode(object):

    def __init__(self, **_):
        self.t_start = 0

    def generateLatents(self):
        raise NotImplementedError('Subclasses must implement')

    def prepUnetInput(self, latent_model_input):
        return latent_model_input, "unet"

    def latentStep(self, latents, i, t, steppos):
        return latents

class Txt2imgMode(UnifiedMode):

    def __init__(self, pipeline, generator, height, width, latents_dtype, batch_total, **kwargs):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        super().__init__(**kwargs)

        self.device = pipeline.device
        self.scheduler = pipeline.scheduler

        self.generators = generator if isinstance(generator, list) else [generator] * batch_total

        self.latents_device = "cpu" if self.device.type == "mps" else self.device
        self.latents_dtype = latents_dtype
        self.latents_shape = (
            batch_total, 
            pipeline.unet.in_channels, 
            height // 8, 
            width // 8
        )

    def generateLatents(self):
        # Unlike in other pipelines, latents need to be generated in the target device
        # for 1-to-1 results reproducibility with the CompVis implementation.
        # However this currently doesn't work in `mps`.
        latents = torch.cat([
            torch.randn((1, *self.latents_shape[1:]), generator=generator, device=self.latents_device, dtype=self.latents_dtype) 
            for generator in self.generators
        ], dim=0)
        
        latents = latents.to(self.device)

        # scale the initial noise by the standard deviation required by the scheduler
        if isinstance(self.scheduler, OldSchedulerMixin): 
            return latents * self.scheduler.sigmas[0]
        else:
            return latents * self.scheduler.init_noise_sigma

    def prepUnetInput(self, latent_model_input):
        return latent_model_input, "unet"

class Img2imgMode(UnifiedMode):

    def __init__(self, pipeline, generator, init_image, latents_dtype, batch_total, num_inference_steps, strength, **kwargs):
        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")
        
        super().__init__(**kwargs)

        self.device = pipeline.device
        self.scheduler = pipeline.scheduler
        self.pipeline = pipeline

        self.generators = generator if isinstance(generator, list) else [generator] * batch_total

        self.latents_dtype = latents_dtype
        self.batch_total = batch_total
        
        self.offset = self.scheduler.config.get("steps_offset", 0)
        self.init_timestep = int(num_inference_steps * strength) + self.offset
        self.init_timestep = min(self.init_timestep, num_inference_steps)
        self.t_start = max(num_inference_steps - self.init_timestep + self.offset, 0)

        if isinstance(init_image, PIL.Image.Image):
            self.init_image = self.preprocess(init_image)
        else:
            self.init_image = self.preprocess_tensor(init_image)

    def preprocess(self, image):
        w, h = image.size
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
        image = image.resize((w, h), resample=PIL.Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return 2.0 * image - 1.0

    def preprocess_tensor(self, tensor):
        # Make sure it's BCHW not just CHW
        if tensor.ndim == 3: tensor = tensor[None, ...]
        # Strip any alpha
        tensor = tensor[:, [0,1,2]]
        # Adjust to -1 .. 1
        tensor = 2.0 * tensor - 1.0
        # Done
        return tensor

    def _convertToLatents(self, image, mask=None):
        """
        Convert an RGB image in standard tensor (BCHW, 0..1) format
        to latents, optionally with pre-masking

        The output will have a batch for each generator / batch_total
        
        If passed, mask must be the same res as the image, and
        is "black replace, white keep" format (0R1K)
        """
        if image.shape[0] != 1: 
            print(
                "Warning: image passed to convertToLatents has more than one batch. "
                "This is probably a mistake"
            )

        if mask is not None and mask.shape[0] != 1: 
            print(
                "Warning: mask passed to convertToLatents has more than one batch. "
                "This is probably a mistake"
            )

        image = image.to(device=self.device, dtype=self.latents_dtype)
        if mask is not None: image = image * (mask > 0.5)

        dist = self.pipeline.vae.encode(image).latent_dist

        latents = torch.cat([
            dist.sample(generator=generator)
            for generator in self.generators
        ], dim=0)

        latents = 0.18215 * latents

        return latents

    def _buildInitialLatents(self):
        return self._convertToLatents(self.init_image)

    def _getSchedulerNoiseTimestep(self, i, t = None):
        """Figure out the timestep to pass to scheduler.add_noise
        If it's an old-style scheduler:
          - return the index as a single integer tensor

        If it's a new-style scheduler:
          - if we know the timestep use it
          - otherwise look up the timestep in the scheduler
          - either way, return a tensor * batch_total on our device
        """
        if isinstance(self.scheduler, OldSchedulerMixin): 
            return torch.tensor(i)
        else:
            timesteps = t if t != None else self.scheduler.timesteps[i]
            return torch.tensor([timesteps] * self.batch_total, device=self.device)

    def _addInitialNoise(self, latents):
        # NOTE: We run K_LMS in float32, because it seems to have problems with float16
        noise_dtype=torch.float32 if isinstance(self.scheduler, LMSDiscreteScheduler) else self.latents_dtype

        self.image_noise = torch.cat([
            torch.randn((1, *latents.shape[1:]), generator=generator, device=self.device, dtype=noise_dtype)
            for generator in self.generators
        ])

        result = self.scheduler.add_noise(latents.to(noise_dtype), self.image_noise, self._getSchedulerNoiseTimestep(self.t_start))
        return result.to(self.latents_dtype) # Old schedulers return float32, and we force K_LMS into float32, but we need to return float16

    def generateLatents(self):
        init_latents = self._buildInitialLatents()
        init_latents = self._addInitialNoise(init_latents)
        return init_latents

class MaskProcessorMixin(object):

    def preprocess_mask(self, mask, inputIs0K1R=True):
        """
        Load a mask from a PIL image
        """
        mask = mask.convert("L")
        mask = np.array(mask).astype(np.float32) / 255.0
        return self.preprocess_mask_tensor(torch.from_numpy(mask), inputIs0K1R=inputIs0K1R)

    def preprocess_mask_tensor(self, tensor, inputIs0K1R=True):
        """
        Preprocess a tensor in 1CHW 0..1 format into a mask in
        11HW 0..1 0R1K format
        """

        if tensor.ndim == 3: tensor = tensor[None, ...]
        # Create a single channel, from whichever the first channel is (L or R)
        tensor = tensor[:, [0]]
        # Invert if input is 0K1R
        if inputIs0K1R: tensor = 1 - tensor
        # Done
        return tensor

    def mask_to_latent_mask(self, mask):
        # Downsample by a factor of 1/8th
        mask = T.functional.resize(mask, [mask.shape[2]//8, mask.shape[3]//8], T.InterpolationMode.NEAREST)
        # And make 4 channel to match latent shape
        mask = mask[:, [0, 0, 0, 0]]
        # Done
        return mask

    def round_mask(self, mask, threshold=0.5):
        """
        Round mask to either 0 or 1 based on threshold (by default, evenly)
        """
        mask = mask.clone()
        mask[mask >= threshold] = 1
        mask[mask < 1] = 0
        return mask

    def round_mask_high(self, mask):
        """
        Round mask so anything above 0 is rounded up to 1
        """
        return self.round_mask(mask, 0.001)

    def round_mask_low(self, mask):
        """
        Round mask so anything below 1 is rounded down to 0
        """
        return self.round_mask(mask, 0.999)



class OriginalInpaintMode(Img2imgMode, MaskProcessorMixin):

    def __init__(self, mask_image, **kwargs):
        super().__init__(**kwargs)

        if isinstance(mask_image, PIL.Image.Image):
            self.mask_image = self.preprocess_mask(mask_image)
        else:
            self.mask_image = self.preprocess_mask_tensor(mask_image)

        self.mask = self.mask_image.to(device=self.device, dtype=self.latents_dtype)
        self.mask = torch.cat([self.mask] * self.batch_total)

    def generateLatents(self):
        init_latents = self._buildInitialLatents()

        self.init_latents_orig = init_latents

        init_latents = self._addInitialNoise(init_latents)
        return init_latents

    def latentStep(self, latents, i, t, steppos):
        # masking
        init_latents_proper = self.scheduler.add_noise(self.init_latents_orig, self.image_noise, torch.tensor([t]))
        return (init_latents_proper * self.mask) + (latents * (1 - self.mask))

class RunwayInpaintMode(UnifiedMode):

    def __init__(self, pipeline, generator, init_image, mask_image, latents_dtype, batch_total, num_inference_steps, strength, **kwargs):
        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")
        
        super().__init__(**kwargs)

        self.device = pipeline.device
        self.scheduler = pipeline.scheduler
        self.pipeline = pipeline

        self.generators = generator if isinstance(generator, list) else [generator] * batch_total

        if isinstance(init_image, PIL.Image.Image):
            mask, masked_image = self._prepare_mask_and_masked_image(init_image, mask_image)
        else:
            mask, masked_image = self._prepare_mask_and_masked_image_tensor(init_image, mask_image)

        height = mask_image.shape[2]
        width = mask_image.shape[3]

        self.latents_device = "cpu" if self.device.type == "mps" else self.device
        self.latents_dtype = latents_dtype
        self.num_channels_latents = self.pipeline.vae.config.latent_channels
        self.latents_shape = (
            batch_total, 
            self.num_channels_latents, 
            height // 8, 
            width // 8
        )

        self.batch_total = batch_total
        
        self.offset = self.scheduler.config.get("steps_offset", 0)
        self.init_timestep = int(num_inference_steps * strength) + self.offset
        self.init_timestep = min(self.init_timestep, num_inference_steps)
        self.t_start = max(num_inference_steps - self.init_timestep + self.offset, 0)


        # -- Stage 2 prep --

        # prepare mask and masked_image
        mask = mask.to(device=self.device, dtype=self.latents_dtype)
        masked_image = masked_image.to(device=self.device, dtype=self.latents_dtype)

        # resize the mask to latents shape as we concatenate the mask to the latents
        mask = torch.nn.functional.interpolate(mask, size=(height // 8, width // 8))

        # encode the mask image into latents space so we can concatenate it to the latents
        masked_image_latent_dist = self.pipeline.vae.encode(masked_image).latent_dist
        
        masked_image_latents = torch.cat([
            masked_image_latent_dist.sample(generator=generator)
            for generator in self.generators
        ], dim=0)
        
        masked_image_latents = 0.18215 * masked_image_latents

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        mask = mask.repeat(batch_total, 1, 1, 1)
        #masked_image_latents = masked_image_latents.repeat(num_images_per_prompt, 1, 1, 1)

        do_classifier_free_guidance=True # TODO - not this

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )

        num_channels_mask = mask.shape[1]
        num_channels_masked_image = masked_image_latents.shape[1]

        if self.num_channels_latents + num_channels_mask + num_channels_masked_image != self.pipeline.inpaint_unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.inpaint_unet`: {self.pipeline.inpaint_unet.config} expects"
                f" {self.pipeline.inpaint_unet.config.in_channels} but received `num_channels_latents`: {self.num_channels_latents} +"
                f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                f" = {self.num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                " `pipeline.unet` or your `mask_image` or `image` input."
            )

        self.mask = mask
        self.masked_image_latents = masked_image_latents




    def _prepare_mask_and_masked_image(image, mask):
        image = np.array(image.convert("RGB"))
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        mask = np.array(mask.convert("L"))
        mask = mask.astype(np.float32) / 255.0
        mask = mask[None, None]
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)

        masked_image = image * (mask < 0.5)

        return mask, masked_image

    def _prepare_mask_and_masked_image_tensor(self, image_tensor, mask_tensor):
        # Make sure it's BCHW not just CHW
        if image_tensor.ndim == 3: image_tensor = image_tensor[None, ...]
        # Strip any alpha
        image_tensor = image_tensor[:, [0,1,2]]
        # Adjust to -1 .. 1
        image_tensor = 2.0 * image_tensor - 1.0

        # Make sure it's BCHW not just CHW
        if mask_tensor.ndim == 3: mask_tensor = mask_tensor[None, ...]
        # Ensure single channel
        mask_tensor = mask_tensor[:, [0]]
        # Harden
        mask_tensor[mask_tensor < 0.5] = 0
        mask_tensor[mask_tensor >= 0.5] = 1

        masked_image_tensor = image_tensor * (mask_tensor < 0.5)

        return mask_tensor, masked_image_tensor


    def generateLatents(self):
        latents = torch.cat([
            torch.randn((1, *self.latents_shape[1:]), generator=generator, device=self.latents_device, dtype=self.latents_dtype) 
            for generator in self.generators
        ], dim=0)

        # scale the initial noise by the standard deviation required by the scheduler
        if isinstance(self.scheduler, OldSchedulerMixin): 
            return latents * self.scheduler.sigmas[0]
        else:
            return latents * self.scheduler.init_noise_sigma

    def prepUnetInput(self, latent_model_input):
        return torch.cat([latent_model_input, self.mask, self.masked_image_latents], dim=1), "inpaint_unet"

class EnhancedInpaintMode(Img2imgMode, MaskProcessorMixin):

    def __init__(self, mask_image, num_inference_steps, strength, **kwargs):
        # Check strength
        if strength < 0 or strength > 2:
            raise ValueError(f"The value of strength should in [0.0, 2.0] but is {strength}")

        # When strength > 1, we start allowing the protected area to change too. Remember that and then set strength
        # to 1 for parent class
        self.fill_with_shaped_noise = strength >= 1.0
        self.mask_scale = min(2 - strength, 1)
        strength = min(strength, 1)

        super().__init__(strength=strength, num_inference_steps=num_inference_steps, **kwargs)

        self.num_inference_steps = num_inference_steps

        # Load mask in 1K0D, 1CHW L shape
        if isinstance(mask_image, PIL.Image.Image):
            self.mask = self.preprocess_mask(mask_image)
        else:
            self.mask = self.preprocess_mask_tensor(mask_image)

        self.mask = self.mask.to(device=self.device, dtype=self.latents_dtype)

        # Remove any excluded pixels (0)
        high_mask = self.round_mask_high(self.mask)
        self.init_latents_orig = self._convertToLatents(self.init_image, high_mask)
        #low_mask = self.round_mask_low(self.mask)
        #blend_mask = self.mask * self.mask_scale

        self.latent_mask = self.mask_to_latent_mask(self.mask)
        self.latent_mask = torch.cat([self.latent_mask] * self.batch_total)

        self.latent_high_mask = self.round_mask_high(self.latent_mask)
        self.latent_low_mask = self.round_mask_low(self.latent_mask)
        self.latent_blend_mask = self.latent_mask * self.mask_scale


    def _matchToSD(self, tensor, targetSD):
        # Normalise tensor to -1..1
        tensor=tensor-tensor.min()
        tensor=tensor.div(tensor.max())
        tensor=tensor*2-1

        # Caculate standard deviation
        sd = tensor.std()
        return tensor * targetSD / sd

    def _matchToSamplerSD(self, tensor):
        if isinstance(self.scheduler, OldSchedulerMixin): 
            targetSD = self.scheduler.sigmas[0]
        else:
            targetSD = self.scheduler.init_noise_sigma

        return _matchToSD(self, tensor, targetSD)

    def _matchNorm(self, tensor, like, cf=1):
        # Normalise tensor to 0..1
        tensor=tensor-tensor.min()
        tensor=tensor.div(tensor.max())

        # Then match range to like
        norm_range = (like.max() - like.min()) * cf
        norm_min = like.min() * cf
        return tensor * norm_range + norm_min


    def _fillWithShapedNoise(self, init_latents):
        # HERE ARE ALL THE THINGS THAT GIVE BETTER OR WORSE RESULTS DEPENDING ON THE IMAGE:
        noise_mask_factor=1 # (1) How much to reduce noise during mask transition
        lmask_mode=3 # 3 (high_mask) seems consistently good. Options are 0 = none, 1 = low mask, 2 = mask as passed, 3 = high mask
        nmask_mode=0 # 1 or 3 seem good, 3 gives good blends slightly more often
        fft_norm_mode="ortho" # forward, backward or ortho. Doesn't seem to affect results too much

        # 0 == normal, matched to latent, 1 == cauchy, matched to latent, 2 == log_normal, 3 == standard normal, mean=0, std=1
        # 0 sometimes gives the best result, but sometimes it gives artifacts
        noise_mode=0

        # 0 == to sampler requested std deviation, 1 == to original image distribution
        match_mode=2

        def latent_mask_for_mode(mode):
            if mode == 1: return self.latent_low_mask
            elif mode == 2: return self.latent_mask
            else: return self.latent_high_mask

        # Current theory: if we can match the noise to the image latents, we get a nice well scaled color blend between the two.
        # The nmask mostly adjusts for incorrect scale. With correct scale, nmask hurts more than it helps

        # noise_mode = 0 matches well with nmask_mode = 0
        # nmask_mode = 1 or 3 matches well with noise_mode = 1 or 3

        # Only consider the portion of the init image that aren't completely masked
        masked_latents = init_latents

        if lmask_mode > 0:
            latent_mask = latent_mask_for_mode(lmask_mode)
            masked_latents = masked_latents * latent_mask

        batch_noise = []

        for generator, split_latents in zip(self.generators, masked_latents.split(1)):
            # Generate some noise TODO: This might affect the seed?
            noise = torch.empty_like(split_latents)
            if noise_mode == 0 and noise_mode < 1: noise = noise.normal_(generator=generator, mean=split_latents.mean(), std=split_latents.std())
            elif noise_mode == 1 and noise_mode < 2: noise = noise.cauchy_(generator=generator, median=split_latents.median(), sigma=split_latents.std())
            elif noise_mode == 2:
                noise = noise.log_normal_(generator=generator)
                noise = noise - noise.mean()
            elif noise_mode == 3: noise = noise.normal_(generator=generator)
            elif noise_mode == 4:
                if isinstance(self.scheduler, OldSchedulerMixin):
                    targetSD = self.scheduler.sigmas[0]
                else:
                    targetSD = self.scheduler.init_noise_sigma

                noise = noise.normal_(generator=generator, mean=0, std=targetSD)

            # Make the noise less of a component of the convolution compared to the latent in the unmasked portion
            if nmask_mode > 0:
                noise_mask = latent_mask_for_mode(nmask_mode)
                noise = noise.mul(1-(noise_mask * noise_mask_factor))

            # Color the noise by the latent
            noise_fft = torch.fft.fftn(noise.to(torch.float32), norm=fft_norm_mode)
            latent_fft = torch.fft.fftn(split_latents.to(torch.float32), norm=fft_norm_mode)
            convolve = noise_fft.mul(latent_fft)
            noise = torch.fft.ifftn(convolve, norm=fft_norm_mode).real.to(self.latents_dtype)

            # Stretch colored noise to match the image latent
            if match_mode == 0: noise = self._matchToSamplerSD(noise)
            elif match_mode == 1: noise = self._matchNorm(noise, split_latents, cf=1)
            elif match_mode == 2: noise = self._matchToSD(noise, 1)

            batch_noise.append(noise)

        noise = torch.cat(batch_noise, dim=0)

        # And mix resulting noise into the black areas of the mask
        return (init_latents * self.latent_mask) + (noise * (1 - self.latent_mask))

    def generateLatents(self):
        # Build initial latents from init_image the same as for img2img
        init_latents = self._buildInitialLatents()       
        # If strength was >=1, filled exposed areas in mask with new, shaped noise
        if self.fill_with_shaped_noise: init_latents = self._fillWithShapedNoise(init_latents)
        # Add the initial noise
        init_latents = self._addInitialNoise(init_latents)
        # And return
        return init_latents

    def latentStep(self, latents, i, t, steppos):
        # The type shifting here is due to note in Img2img._addInitialNoise
        init_latents_proper = self.scheduler.add_noise(self.init_latents_orig.to(self.image_noise.dtype), self.image_noise, self._getSchedulerNoiseTimestep(i, t))
        init_latents_proper = init_latents_proper.to(latents.dtype)

        iteration_mask = self.latent_blend_mask.gt(steppos).to(self.latent_blend_mask.dtype)

        return (init_latents_proper * iteration_mask) + (latents * (1 - iteration_mask))       


class EnhancedRunwayInpaintMode(EnhancedInpaintMode):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        do_classifier_free_guidance=True # TODO - not this

        inpaint_mask = 1 - self.latent_high_mask[:, [0]]
        self.inpaint_mask = torch.cat([inpaint_mask] * 2) if do_classifier_free_guidance else inpaint_mask

        self.masked_image_latents = self.init_latents_orig

        self.masked_image_latents = (
            torch.cat([self.masked_image_latents] * 2) if do_classifier_free_guidance else self.self.masked_image_latents
        )

    def prepUnetInput(self, latent_model_input):
        return torch.cat([latent_model_input, self.inpaint_mask, self.masked_image_latents], dim=1), "inpaint_unet"

    def latentStep(self, latents, i, t, steppos):
        if False: return super().latentStep(latents, i, t, steppos)
        return latents

class DynamicModuleDiffusionPipeline(DiffusionPipeline):

    def __init__(self, *args, **kwargs):
        self._moduleMode = "all"
        self._moduleDevice = torch.device("cpu")

    def register_modules(self, **kwargs):
        self._modules = set(kwargs.keys())
        self._modulesDyn = set(("vae", "text_encoder", "unet", "inpaint_unet", "safety_checker"))
        self._modulesStat = self._modules - self._modulesDyn

        super().register_modules(**kwargs)

    def set_module_mode(self, mode):
        self._moduleMode = mode
        self.to(self._moduleDevice)

    def to(self, torch_device, forceAll=False):
        if torch_device is None:
            return self

        module_names, _ = self.extract_init_dict(dict(self.config))

        self._moduleDevice = torch.device(torch_device)

        moveNow = self._modules if (self._moduleMode == "all" or forceAll) else self._modulesStat

        for name in moveNow:
            module = getattr(self, f"_{name}" if name in self._modulesDyn else name)
            if isinstance(module, torch.nn.Module):
                module.to(torch_device)
        
        return self

    @property
    def device(self) -> torch.device:
        return self._moduleDevice

    def prepmodule(self, name, module):
        if self._moduleMode == "all":
            return module

        # We assume if this module is on a device of the right type we put it there
        # (How else would it get there?)
        if self._moduleDevice.type == module.device.type:
            return module

        if name in self._modulesStat:
            return module

        for name in self._modulesDyn:
            other = getattr(self, f"_{name}")
            if other is not module: other.to("cpu")
        
        module.to(self._moduleDevice)
        return module

    @property 
    def vae(self):
        return self.prepmodule("vae", self._vae)
    
    @vae.setter 
    def vae(self, value):
        self._vae = value

    @property 
    def text_encoder(self):
        return self.prepmodule("text_encoder", self._text_encoder)
    
    @text_encoder.setter 
    def text_encoder(self, value):
        self._text_encoder = value

    @property 
    def unet(self):
        return self.prepmodule("unet", self._unet)
    
    @unet.setter 
    def unet(self, value):
        self._unet = value

    @property 
    def inpaint_unet(self):
        a = self.prepmodule("inpaint_unet", self._inpaint_unet)
        return a
    
    @inpaint_unet.setter 
    def inpaint_unet(self, value):
        self._inpaint_unet = value

    @property 
    def safety_checker(self):
        return self.prepmodule("safety_checker", self._safety_checker)
    
    @safety_checker.setter 
    def safety_checker(self, value):
        self._safety_checker = value


class NoisePredictor:

    def __init__(self, pipeline, mode, text_embeddings, do_classifier_free_guidance, guidance_scale):
        self.pipeline = pipeline
        self.mode = mode
        self.text_embeddings = text_embeddings
        self.do_classifier_free_guidance = do_classifier_free_guidance
        self.guidance_scale = guidance_scale

    def step(self, latents, i, t, sigma = None):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

        latent_model_input, unet_klass = self.mode.prepUnetInput(latent_model_input)

        if isinstance(self.pipeline.scheduler, OldSchedulerMixin): 
            if not sigma: sigma = self.pipeline.scheduler.sigmas[i] 
            # the model input needs to be scaled to match the continuous ODE formulation in K-LMS
            latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
        else:
            latent_model_input = self.pipeline.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        noise_pred = getattr(self.pipeline, unet_klass)(latent_model_input, t, encoder_hidden_states=self.text_embeddings).sample

        # perform guidance
        if self.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

        return noise_pred

class BasicTextEmbedding():

    def __init__(self, pipe, **kwargs):
        self.pipe = pipe
    
    def _get_embeddedings(self, strings, label):
        strings = [string[0][0] for string in strings]
        tokenizer = self.pipe.tokenizer

        # get prompt text embeddings
        text_inputs = tokenizer(
            strings,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        if text_input_ids.shape[-1] > tokenizer.model_max_length:
            removed_text = tokenizer.batch_decode(text_input_ids[:, tokenizer.model_max_length :])
            logger.warning(
                f"The following part of your {label} input was truncated because CLIP can only handle sequences up to "
                f"{tokenizer.model_max_length} tokens: {removed_text}"
            )
            text_input_ids = text_input_ids[:, :tokenizer.model_max_length]

        text_embeddings = self.pipe.text_encoder(text_input_ids.to(self.pipe.device))[0]

        return text_embeddings

    def get_text_embeddings(self, prompt, uncond_prompt):
        """Prompt and negative a both expected to be lists of strings, and matching in length"""
        text_embeddings = self._get_embeddedings(prompt, "prompt")
        uncond_embeddings = self._get_embeddedings(uncond_prompt, "negative prompt") if uncond_prompt else None

        return (text_embeddings, uncond_embeddings)

class UnifiedPipelinePrompt():
    def __init__(self, prompts):
        self._weighted = False
        self._prompt = self.parse_prompt(prompts)

    def check_tuples(self, list):
        for item in list:
            if not isinstance(item, tuple) or len(item) != 2 or not isinstance(item[0], str) or not isinstance(item[1], float):
                raise ValueError(f"Expected a list of (text, weight) tuples, but got {item} of type {type(item)}")
            if item[1] != 1.0:
                self._weighted = True

    def parse_single_prompt(self, prompt):
        """Parse a single prompt - no lists allowed.
        Prompt is either a text string, or a list of (text, weight) tuples"""

        if isinstance(prompt, str):
            return [(prompt, 1.0)]
        elif isinstance(prompt, list) and isinstance(prompt[0], tuple):
            self.check_tuples(prompt)
            return prompt

        raise ValueError(f"Expected a string or a list of tuples, but got {type(prompt)}")

    def parse_prompt(self, prompts):
        try:
            return [self.parse_single_prompt(prompts)]
        except:
            if isinstance(prompts, list): return [self.parse_single_prompt(prompt) for prompt in prompts]

            raise ValueError(
                f"Expected a string, a list of strings, a list of (text, weight) tuples or "
                f"a list of a list of tuples. Got {type(prompts)} instead."
            )
                
    @property
    def batch_size(self):
        return len(self._prompt)
    
    @property
    def weighted(self):
        return self._weighted

    def as_tokens(self):
        return self._prompt

    def as_unweighted_string(self):
        return [" ".join([token[0] for token in prompt]) for prompt in self._prompt]

class UnifiedPipeline(DynamicModuleDiffusionPipeline):
    r"""
    Pipeline for unified image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/CompVis/stable-diffusion-v1-4) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """


    
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: SchedulerMixin,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        inpaint_unet: Optional[UNet2DConditionModel] = None,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            inpaint_unet=inpaint_unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                a number is provided, uses as many slices as `attention_head_dim // slice_size`. In this case,
                `attention_head_dim` must be a multiple of `slice_size`.
        """
        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = self.unet.config.attention_head_dim // 2
        self.unet.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        r"""
        Disable sliced attention computation. If `enable_attention_slicing` was previously invoked, this method will go
        back to computing attention in one step.
        """
        # set slice_size = `None` to disable `attention slicing`
        self.enable_attention_slicing(None)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str], List[Tuple[str, float]], List[List[Tuple[str, float]]]],
        height: int = 512,
        width: int = 512,
        init_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        mask_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        outmask_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        strength: float = 0.0,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str], List[Tuple[str, float]], List[List[Tuple[str, float]]]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        max_embeddings_multiples: Optional[int] = 3,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        run_safety_checker: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]` or `List[Tuple[str, double]]` or List[List[Tuple[str, double]]]):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
                Alternatively, a list of torch generators, who's length must exactly match the length of the prompt, one
                per batch. This allows batch-size-idependant consistency (except where schedulers that use generators are 
                used)
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        prompt = UnifiedPipelinePrompt(prompt)
        batch_size = prompt.batch_size

        # Match the negative prompt length to the batch_size
        if negative_prompt is None:
            negative_prompt = UnifiedPipelinePrompt([[("", 1.0)]] * batch_size)
        else:
            negative_prompt = UnifiedPipelinePrompt(negative_prompt)

        if batch_size != negative_prompt.batch_size:
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )

        batch_total = batch_size * num_images_per_prompt

        # Match the generator list to the batch_total
        if isinstance(generator, list) and len(generator) != batch_total:
            raise ValueError(
                f"Generator passed as a list, but list length does not match "
                f"batch size {batch_size} * number of images per prompt {num_images_per_prompt}, i.e. {batch_total}"
            )

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if (mask_image != None and init_image == None):
            raise ValueError(f"Can't pass a mask without an image")

        if (outmask_image != None and init_image == None):
            raise ValueError(f"Can't pass a outmask without an image")

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        text_embedding_calculator = BasicTextEmbedding(self) #LPWTextEmbedding(self, max_embeddings_multiples=max_embeddings_multiples)

        # get unconditional embeddings for classifier free guidance
        text_embeddings, uncond_embeddings = text_embedding_calculator.get_text_embeddings(
            prompt=prompt.as_tokens(),
            uncond_prompt=negative_prompt.as_tokens() if do_classifier_free_guidance else None,
        )

        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            bs_embed, seq_len, _ = uncond_embeddings.shape
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # Get the latents dtype based on the text_embeddings dtype
        latents_dtype = text_embeddings.dtype

        # Calculate operating mode based on arguments
        if mask_image != None: mode_class = EnhancedRunwayInpaintMode
        elif init_image != None: mode_class = Img2imgMode
        else: mode_class = Txt2imgMode

        mode = mode_class(
            pipeline=self, 
            generator=generator,
            width=width, height=height,
            init_image=init_image, mask_image=mask_image,
            latents_dtype=latents_dtype,
            batch_total=batch_total,
            num_inference_steps=num_inference_steps,
            strength=strength
        ) 

        print(f"Mode {mode.__class__} with strength {strength}")

        # Build the noise predictor. We move this into it's own class so it can be
        # passed into a scheduler if they need to re-call
        noise_predictor = NoisePredictor(
            pipeline=self, 
            mode=mode,
            text_embeddings=text_embeddings, 
            do_classifier_free_guidance=do_classifier_free_guidance, guidance_scale=guidance_scale
        )

        # Get the initial starting point - either pure random noise, or the source image with some noise depending on mode
        latents = mode.generateLatents()

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        accepts_noise_predictor = "noise_predictor" in set(inspect.signature(self.scheduler.step).parameters.keys())

        extra_step_kwargs = {}
        if accepts_eta: extra_step_kwargs["eta"] = eta
        if accepts_generator: extra_step_kwargs["generator"] = generator[0] if isinstance(generator, list) else generator
        if accepts_noise_predictor: extra_step_kwargs["noise_predictor"] = noise_predictor.step

        t_start = mode.t_start

        timesteps_tensor = self.scheduler.timesteps[t_start:].to(self.device)

        for i, t in enumerate(self.progress_bar(timesteps_tensor)):
            t_index = t_start + i

            # predict the noise residual
            noise_pred = noise_predictor.step(latents, t_index, t)

            # compute the previous noisy sample x_t -> x_t-1

            if isinstance(self.scheduler, OldSchedulerMixin): 
                latents = self.scheduler.step(noise_pred, t_index, latents, **extra_step_kwargs).prev_sample
            else:
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            latents = mode.latentStep(latents, t_index, t, i / (timesteps_tensor.shape[0] + 1))

            if False:
                stage_latents = 1 / 0.18215 * latents
                stage_image = self.vae.decode(stage_latents).sample
                stage_image = (stage_image / 2 + 0.5).clamp(0, 1).cpu()

                for j, pngBytes in enumerate(images.toPngBytes(stage_image)):
                    with open(f"/tests/out/debug-{j}-{i}.png", "wb") as f:
                        f.write(pngBytes)

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)

        if strength <= 1 and outmask_image != None:
            outmask = torch.cat([outmask_image] * batch_total)
            outmask = outmask[:, [0,1,2]]
            outmask = outmask.to(self.device)

            source =  torch.cat([init_image] * batch_total)
            source = source[:, [0,1,2]]
            source = source.to(self.device)

            image = source * (1-outmask) + image * outmask

        numpyImage = image.cpu().permute(0, 2, 3, 1).numpy()

        if run_safety_checker:
            # run safety checker
            safety_cheker_input = self.feature_extractor(self.numpy_to_pil(numpyImage), return_tensors="pt").to(self.device)
            numpyImage, has_nsfw_concept = self.safety_checker(images=numpyImage, clip_input=safety_cheker_input.pixel_values.to(text_embeddings.dtype))
        else:
            has_nsfw_concept = [False] * numpyImage.shape[0]

        if output_type == "pil":
            image = self.numpy_to_pil(image)
        elif output_type == "tensor":
            image = torch.from_numpy(numpyImage).permute(0, 3, 1, 2)
        else:
            image = numpyImage

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
