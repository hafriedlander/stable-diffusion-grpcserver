import inspect, traceback, math
import time
from mimetypes import init

from typing import Callable, List, Tuple, Iterable, Optional, Union, Literal, NewType

from packaging import version

import numpy as np
import torch

import torchvision.transforms as T

from torch.profiler import profile, record_function, ProfilerActivity

import PIL

from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPModel, CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import LMSDiscreteScheduler, PNDMScheduler
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import deprecate, logging
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker


from sdgrpcserver import images
from sdgrpcserver.pipeline.text_embedding import *
from sdgrpcserver.pipeline.kschedulers.scheduling_utils import KSchedulerMixin

from sdgrpcserver.pipeline.randtools import *
from sdgrpcserver.pipeline.scheduler_wrappers import *
from sdgrpcserver.pipeline.unet_wrappers.types import *
from sdgrpcserver.pipeline.unet_wrappers.clipguided import *
from sdgrpcserver.pipeline.unet_wrappers.graft import *
from sdgrpcserver.pipeline.unet_wrappers.hires_fix import *

from sdgrpcserver.pipeline.attention_replacer import replace_cross_attention
from sdgrpcserver.pipeline.models.memory_efficient_cross_attention import has_xformers, MemoryEfficientCrossAttention
from sdgrpcserver.pipeline.models.structured_cross_attention import StructuredCrossAttention

try:
    from nonfree import tome_patcher
except:
    tome_patcher = None

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

enabled_debug_latents = set([
    #"initial",
    #"step",
    #"mask",
    #"shapednoise",
    #"initnoise",
    #"blendin",
    #"blendout",
    # "small",
])

def write_debug_latents(vae, label, i, latents):
    if label not in enabled_debug_latents: return

    stage_latents = 1 / 0.18215 * latents
    stage_image = vae.decode(stage_latents).sample
    stage_image = (stage_image / 2 + 0.5).clamp(0, 1).cpu()

    for j, pngBytes in enumerate(images.toPngBytes(stage_image)):
        with open(f"/tests/out/debug-nup-{label}-{j}-{i}.png", "wb") as f:
            f.write(pngBytes)






def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value

class UNetWithEmbeddings():

    def __init__(self, unet, text_embeddings):
        self.unet = unet
        self.text_embeddings = text_embeddings

    def __call__(self, latents, t):
        return self.unet(latents, t, encoder_hidden_states=self.text_embeddings).sample

class UNetGuidedOuter_Seperated():
    def __init__(self, unet_g, unet_u, unet_f, guidance_scale, batch_total):
        self.unet_g = unet_g
        self.unet_u = unet_u
        self.unet_f = unet_f
        self.guidance_scale = guidance_scale
        self.batch_total = batch_total

    def __call__(self, latents, t):
        noise_pred_g = self.unet_g(latents, t)
        noise_pred_u = self.unet_u(latents, t)

        # Detect if some deeper unet inside self.unet has already 
        # done guidance (CLIP guidance will sometimes do this)
        # if noise_pred.shape[0] > self.batch_total:
        noise_pred = noise_pred_u + self.guidance_scale * (noise_pred_g - noise_pred_u)
        return noise_pred

class UNetGuidedOuter():
    def __init__(self, unet_g, unet_u, unet_f, guidance_scale, batch_total):
        self.unet_g = unet_g
        self.unet_u = unet_u
        self.unet_f = unet_f
        self.guidance_scale = guidance_scale
        self.batch_total = batch_total

    def __call__(self, latents, t):
        latents = torch.cat([latents] * 2)

        noise_pred = self.unet_f(latents, t)
        noise_pred_u, noise_pred_g = noise_pred.chunk(2)

        # Detect if some deeper unet inside self.unet has already 
        # done guidance (CLIP guidance will sometimes do this)
        # if noise_pred.shape[0] > self.batch_total:
        noise_pred = noise_pred_u + self.guidance_scale * (noise_pred_g - noise_pred_u)
        return noise_pred


class UnifiedMode(object):

    def __init__(self, **kwargs):
        pass

    def generateLatents(self):
        raise NotImplementedError('Subclasses must implement')

    def wrap_unet(self, unet: NoisePredictionUNet) -> NoisePredictionUNet:
        return unet

    def wrap_guidance_unet(
        self, 
        unet_g: NoisePredictionUNet, 
        unet_u: NoisePredictionUNet,
        unet_f: NoisePredictionUNet,
        guidance_scale: float, 
        batch_total: int
    ) -> NoisePredictionUNet:
        return UNetGuidedOuter(unet_g, unet_u, unet_f, guidance_scale, batch_total)

    def wrap_k_unet(self, unet: KDiffusionSchedulerUNet) -> KDiffusionSchedulerUNet:
        return unet
    
    def wrap_d_unet(self, unet: DiffusersSchedulerUNet) -> DiffusersSchedulerUNet:
        return unet

class Txt2imgMode(UnifiedMode):

    def __init__(self, pipeline, scheduler, generators, height, width, latents_dtype, batch_total, **kwargs):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        super().__init__(**kwargs)

        self.device = pipeline.device
        self.scheduler = scheduler
        self.generators = generators
        self.pipeline = pipeline

        self.latents_dtype = latents_dtype
        self.latents_shape = (
            batch_total, 
            pipeline.unet.in_channels, 
            height // pipeline.vae_scale_factor, 
            width // pipeline.vae_scale_factor
        )

        self.unet_sample_size = pipeline.get_unet_sample_size(pipeline.unet)

    def generateLatents(self):
        # Always start off by creating the same size noise, to try and give
        # a more consistent result when resolution is changed
        mid_latents = batched_randn(
            [*self.latents_shape[:2], self.unet_sample_size, self.unet_sample_size], 
            self.generators, 
            device=self.device, 
            dtype=self.latents_dtype
        )

        latents = batched_randn(
            self.latents_shape, 
            self.generators, 
            device=self.device, 
            dtype=self.latents_dtype
        )

        off2 = (latents.shape[2] - mid_latents.shape[2]) // 2
        off3 = (latents.shape[3] - mid_latents.shape[3]) // 2

        # Insert mid_latents over latents
        latents[:, :, off2:off2+mid_latents.shape[2], off3:off3+mid_latents.shape[3]] = mid_latents

        # scale the initial noise by the standard deviation required by the scheduler
        return self.scheduler.prepare_initial_latents(latents)

class Img2imgMode(UnifiedMode):

    def __init__(self, pipeline, scheduler, generators, init_image, latents_dtype, batch_total, num_inference_steps, strength, **kwargs):
        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")
        
        super().__init__(**kwargs)

        self.device = pipeline.device
        self.scheduler = scheduler
        self.generators = generators
        self.pipeline = pipeline

        self.latents_dtype = latents_dtype
        self.batch_total = batch_total
        
        if isinstance(init_image, PIL.Image.Image):
            init_image = images.fromPIL(init_image)

        self.init_image = self.preprocess_tensor(init_image)

    def preprocess_tensor(self, tensor):
        # Make sure it's BCHW not just CHW
        if tensor.ndim == 3: tensor = tensor[None, ...]
        # Strip any alpha
        tensor = tensor[:, [0,1,2]]
        # Adjust to -1 .. 1
        tensor = 2.0 * tensor - 1.0
        # TODO: resize & crop if it doesn't match width & height

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

        latents = latents.to(self.device)
        latents = 0.18215 * latents

        return latents

    def _buildInitialLatents(self):
        return self._convertToLatents(self.init_image)

    def _addInitialNoise(self, latents):
        # NOTE: We run K_LMS in float32, because it seems to have problems with float16
        noise_dtype=torch.float32 if isinstance(self.scheduler, LMSDiscreteScheduler) else self.latents_dtype

        self.image_noise = batched_randn(latents.shape, self.generators, self.device, noise_dtype)

        result = self.scheduler.add_noise(latents.to(noise_dtype), self.image_noise, self.scheduler.start_timestep)
        return result.to(self.latents_dtype) # Old schedulers return float32, and we force K_LMS into float32, but we need to return float16

    def generateLatents(self):
        init_latents = self._buildInitialLatents()
        init_latents = self._addInitialNoise(init_latents)
        return init_latents

class MaskProcessorMixin(object):

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

class EnhancedInpaintMode(Img2imgMode, MaskProcessorMixin):

    def __init__(self, mask_image, num_inference_steps, strength, **kwargs):
        # Check strength
        if strength < 0 or strength > 2:
            raise ValueError(f"The value of strength should in [0.0, 2.0] but is {strength}")

        # When strength > 1, we start allowing the protected area to change too. Remember that and then set strength
        # to 1 for parent class
        self.fill_with_shaped_noise = strength >= 1.0

        self.shaped_noise_strength = min(2 - strength, 1)
        self.mask_scale = 1

        strength = min(strength, 1)

        super().__init__(strength=strength, num_inference_steps=num_inference_steps, **kwargs)

        self.num_inference_steps = num_inference_steps

        # Load mask in 1K0D, 1CHW L shape
        if isinstance(mask_image, PIL.Image.Image):
            mask_image = images.fromPIL(mask_image.convert("L"))

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
        if isinstance(self.scheduler, KSchedulerMixin): 
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


    def _fillWithShapedNoise(self, init_latents, noise_mode=5):
        """
        noise_mode sets the noise distribution prior to convolution with the latent

        0: normal, matched to latent, 1: cauchy, matched to latent, 2: log_normal, 
        3: standard normal (mean=0, std=1), 4: normal to scheduler SD
        5: random shuffle (does not convolve afterwards)
        """

        # HERE ARE ALL THE THINGS THAT GIVE BETTER OR WORSE RESULTS DEPENDING ON THE IMAGE:
        noise_mask_factor=1 # (1) How much to reduce noise during mask transition
        lmask_mode=3 # 3 (high_mask) seems consistently good. Options are 0 = none, 1 = low mask, 2 = mask as passed, 3 = high mask
        nmask_mode=0 # 1 or 3 seem good, 3 gives good blends slightly more often
        fft_norm_mode="ortho" # forward, backward or ortho. Doesn't seem to affect results too much

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
            # Generate some noise
            noise = torch.zeros_like(split_latents)
            if noise_mode == 0 and noise_mode < 1: noise = noise.normal_(generator=generator, mean=split_latents.mean(), std=split_latents.std())
            elif noise_mode == 1 and noise_mode < 2: noise = noise.cauchy_(generator=generator, median=split_latents.median(), sigma=split_latents.std())
            elif noise_mode == 2:
                noise = noise.log_normal_(generator=generator)
                noise = noise - noise.mean()
            elif noise_mode == 3: noise = noise.normal_(generator=generator)
            elif noise_mode == 4:
                if isinstance(self.scheduler, KSchedulerMixin):
                    targetSD = self.scheduler.sigmas[0]
                else:
                    targetSD = self.scheduler.init_noise_sigma

                noise = noise.normal_(generator=generator, mean=0, std=targetSD)
            elif noise_mode == 5:
                # Seed the numpy RNG from the batch generator, so it's consistent
                npseed = torch.randint(low=0, high=torch.iinfo(torch.int32).max, size=[1], generator=generator, device=generator.device, dtype=torch.int32).cpu()
                npgen = np.random.default_rng(npseed.numpy())
                # Fill each channel with random pixels selected from the good portion
                # of the channel. I wish there was a way to do this in PyTorch :shrug:
                channels = []
                for channel in split_latents.split(1, dim=1):
                    good_pixels = channel.masked_select(latent_mask[[0], [0]].ge(0.5))
                    np_mixed = npgen.choice(good_pixels.cpu().numpy(), channel.shape)
                    channels.append(torch.from_numpy(np_mixed).to(noise.device).to(noise.dtype))

                # In noise mode 5 we don't convolve. The pixel shuffled noise is already extremely similar to the original in tone. 
                # We allow the user to request some portion is uncolored noise to allow outpaints that differ greatly from original tone
                # (with an increasing risk of image discontinuity)
                noise = noise.to(generator.device).normal_(generator=generator).to(noise.device)
                noise = noise * (1-self.shaped_noise_strength) + torch.cat(channels, dim=1) * self.shaped_noise_strength

                batch_noise.append(noise)
                continue

            elif noise_mode == 6:
                noise = torch.ones_like(split_latents)


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

        write_debug_latents(self.pipeline.vae, "shapednoise", 0, init_latents)

        # Add the initial noise
        init_latents = self._addInitialNoise(init_latents)

        write_debug_latents(self.pipeline.vae, "initnoise", 0, init_latents)

        # And return
        return init_latents

    def _blend(self, u, orig, next):
            steppos = u
            steppos = steppos.clamp(0, 0.9999)

            iteration_mask = self.latent_blend_mask.gt(steppos).to(next.dtype)

            return (orig * iteration_mask) + (next * (1 - iteration_mask))       

    def wrap_k_unet(self, unet: KDiffusionSchedulerUNet) -> KDiffusionSchedulerUNet:
        def wrapped_unet(latents, sigma, u) -> PX0Tensor:
            px0 = unet(latents, sigma, u=u)
            orig = self.init_latents_orig.to(px0.dtype)

            return cast(PX0Tensor, self._blend(u, orig, px0))

        return wrapped_unet

    def wrap_d_unet(self, unet: DiffusersSchedulerUNet) -> DiffusersSchedulerUNet:
        def wrapped_unet(latents, t, u):
            xt = unet(latents, t, u=u)
            orig = self.init_latents_orig.to(self.image_noise.dtype)
            orig = self.scheduler.add_noise(orig, self.image_noise, t)
            orig = orig.to(xt.dtype)

            return self._blend(u, orig, xt)

        return wrapped_unet


class EnhancedRunwayInpaintMode(EnhancedInpaintMode):

    def __init__(self, do_classifier_free_guidance, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Runway inpaint unet need mask in B1HW 0K1R format
        self.inpaint_mask = 1 - self.latent_high_mask[:, [0]]
        self.masked_image_latents = self.init_latents_orig

        # Since these are passed into the unet, they need doubling
        # just like the latents are if we are doing CFG
        if do_classifier_free_guidance:
            self.inpaint_mask_cfg = torch.cat([self.inpaint_mask] * 2)
            self.masked_image_latents_cfg = torch.cat([self.masked_image_latents] * 2)

        self.inpaint_mask_cache = {}
        self.masked_lantents_cache = {}

    def _fillWithShapedNoise(self, init_latents):
        return super()._fillWithShapedNoise(init_latents, noise_mode=5)

    def wrap_unet(self, unet: NoisePredictionUNet) -> NoisePredictionUNet:
        def wrapped_unet(latents: XtTensor, t):
            if latents.shape[0] == self.inpaint_mask.shape[0]:
                inpaint_mask = self.inpaint_mask
                masked_latents = self.masked_image_latents
            else:
                inpaint_mask = self.inpaint_mask_cfg
                masked_latents = self.masked_image_latents_cfg

            # TODO: This is for the hi-res adjustment, it shouldn't be here
            if masked_latents.shape != latents.shape:
                offset2 = (masked_latents.shape[2] - latents.shape[2]) // 2
                offset3 = (masked_latents.shape[3] - latents.shape[3]) // 2

                inpaint_mask = inpaint_mask[:, :, offset2:offset2+latents.shape[2], offset3:offset3+latents.shape[3]]
                masked_latents = masked_latents[:, :, offset2:offset2+latents.shape[2], offset3:offset3+latents.shape[3]]
                
            expanded_latents = torch.cat([
                latents, 
                # Note: these specifically _do not_ get scaled by the 
                # current timestep sigma like the latents above have been
                inpaint_mask,
                masked_latents
            ], dim=1)

            return unet(expanded_latents, t)

        return wrapped_unet

    def wrap_k_unet(self, unet: KDiffusionSchedulerUNet) -> KDiffusionSchedulerUNet:
        return unet

    def wrap_d_unet(self, unet: DiffusersSchedulerUNet) -> DiffusersSchedulerUNet:
        return unet

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
        

UnifiedPipelinePromptType = Union[
    str,                           # Just a single string, for a batch of 1
    List[str],                     # A list of strings, for a batch of len(prompt)
    List[Tuple[str, float]],       # A list of (part, weight) token tuples, for a batch of 1
    List[List[Tuple[str, float]]], # A list of lists of (part, weight) token tuples, for a batch of len(prompt)
    UnifiedPipelinePrompt          # A pre-parsed prompt
]

UnifiedPipelineImageType = Union[
    torch.FloatTensor, PIL.Image.Image
]

class UnifiedPipeline(DiffusionPipeline):
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


    def _check_scheduler_config(self, scheduler):
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


    def _check_unet_config(self, unet):
        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely .If you're checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

    def get_unet_sample_size(self, unet):
        return getattr(unet.config, "sample_size", 64)
    
    def get_unet_pixel_size(self, unet):
        return self.get_unet_sample_size(unet) * self.vae_scale_factor

    vae: AutoencoderKL
    text_encoder: CLIPTextModel
    tokenizer: CLIPTokenizer
    unet: UNet2DConditionModel
    scheduler: SchedulerMixin
    safety_checker: StableDiffusionSafetyChecker
    feature_extractor: CLIPFeatureExtractor
    clip_model: Optional[CLIPModel]
    clip_tokenizer: Optional[CLIPTokenizer]
    inpaint_unet: Optional[UNet2DConditionModel]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: SchedulerMixin,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        clip_model: Optional[CLIPModel] = None,
        clip_tokenizer: Optional[CLIPTokenizer] = None,
        inpaint_unet: Optional[UNet2DConditionModel] = None,
    ):
        super().__init__()

        self._check_scheduler_config(scheduler)
        self._check_unet_config(unet)

        # Grafted inpaint uses an inpaint_unet from a different model than the primary model 
        # as guidance to produce a nicer inpaint that EnhancedInpaintMode otherwise can
        self._grafted_inpaint = False
        self._graft_factor = 0.8
        self._structured_diffusion = False

        # Some models use a different text embedding layer
        self._text_embedding_layer = "final"

        if clip_tokenizer is None: clip_tokenizer = tokenizer

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            inpaint_unet=inpaint_unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            clip_model=clip_model,
            clip_tokenizer=clip_tokenizer
        )

        # Pull out VAE scale factor
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.clip_defaults = {
            "vae_cutouts": 2,
            "approx_cutouts": 2,
            "no_cutouts": False,
            "guidance_scale": 0.0,
            "guidance_base": "guided",
            "gradient_length": 15,
            "gradient_threshold": 0.01,
            "gradient_maxloss": 1.0,
        }

        if self.clip_model is not None:
            set_requires_grad(self.text_encoder, False)
            set_requires_grad(self.clip_model, False)
            set_requires_grad(self.unet, False)
            set_requires_grad(self.vae, False)

    def set_options(self, options):
        for key, value in options.items():
            if key == "grafted_inpaint": 
                self._grafted_inpaint = bool(value)
            elif key == "graft_factor": 
                self._graft_factor = float(value)
            elif key == "xformers" and value:
                print(
                    "XFormers is always enabled if it is available now, "
                    "you don't need to specify it in your engines.yaml"
                )
            elif key == "tome" and bool(value):
                print("Warning: ToMe isn't finished, and shouldn't be used")
                if tome_patcher:
                    tome_patcher.apply_tome(self.unet)
                    self.unet.r = int(value)
                else:
                    print("Warning: you asked for ToMe, but nonfree packages are not available")
            elif key == "structured_diffusion" and value:
                print("Warning: structured diffusion isn't finished, and shouldn't be used")
                replace_cross_attention(target=self.unet, crossattention=StructuredCrossAttention, name="unet")
                if self.inpaint_unet: replace_cross_attention(target=self.inpaint_unet, crossattention=StructuredCrossAttention, name="inpaint_unet")
                self._structured_diffusion = True
            elif key == "clip":
                for subkey, subval in value.items():
                    if subkey == "unet_grad":
                        set_requires_grad(self.unet, bool(subval))
                    elif subkey == "vae_grad":
                        set_requires_grad(self.vae, bool(subval))
                    elif subkey == "vae_cutouts":
                        self.clip_defaults["vae_cutouts"] = int(subval)
                    elif subkey == "approx_cutouts":
                        self.clip_defaults["approx_cutouts"] = int(subval)
                    elif subkey == "no_cutouts":
                        self.clip_defaults["no_cutouts"] = bool(subval)
                    elif subkey == "guidance_scale": 
                        self.clip_defaults["guidance_scale"] = float(subval)
                    elif subkey == "guidance_base": 
                        self.clip_defaults["guidance_base"] = str(subval)
                    elif subkey == "gradient_length":
                        self.clip_defaults["gradient_length"] = int(subval)
                    elif subkey == "gradient_threshold":
                        self.clip_defaults["gradient_threshold"] = float(subval)
                    elif subkey == "gradient_maxloss":
                        self.clip_defaults["gradient_maxloss"] = float(subval)
                    else:
                        raise ValueError(f"Unknown option {subkey}: {subval} passed as part of clip settings")
            elif key == "clip_vae_grad":
                set_requires_grad(self.vae, bool(value))
            elif key == "text_embedding_layer":
                self._text_embedding_layer = str(value)
            else:
                raise ValueError(f"Unknown option {key}: {value} passed to UnifiedPipeline")

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
            if isinstance(self.unet.config.attention_head_dim, int):
                # half the attention head size is usually a good trade-off between
                # speed and memory
                slice_size = self.unet.config.attention_head_dim // 2
            else:
                # if `attention_head_dim` is a list, take the smallest head size
                slice_size = min(self.unet.config.attention_head_dim)

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
        prompt: UnifiedPipelinePromptType,
        height: int = 512,
        width: int = 512,
        init_image: Optional[UnifiedPipelineImageType] = None,
        mask_image: Optional[UnifiedPipelineImageType] = None,
        outmask_image: Optional[UnifiedPipelineImageType] = None,
        strength: float = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[UnifiedPipelinePromptType] = None,
        num_images_per_prompt: Optional[int] = 1,
        prediction_type: Optional[str] = "epsilon",
        eta: Optional[float] = None,
        churn: Optional[float] = None,
        churn_tmin: Optional[float] = None,
        churn_tmax: Optional[float] = None,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        karras_rho: Optional[float] = None,
        scheduler_noise_type: Optional[SCHEDULER_NOISE_TYPE] = "normal",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        max_embeddings_multiples: Optional[int] = 3,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        run_safety_checker: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        clip_guidance_scale: Optional[float] = None,
        clip_guidance_base: Optional[CLIP_GUIDANCE_BASE] = None,
        clip_gradient_length: Optional[int] = None,
        clip_gradient_threshold: Optional[float] = None,
        clip_gradient_maxloss: Optional[float] = None,
        clip_prompt: Optional[Union[str, List[str]]] = None,
        vae_cutouts: Optional[int] = None,
        approx_cutouts: Optional[int] = None,
        no_cutouts: Optional[Union[str, bool]] = None,
        scheduler = None,
        hires_fix = 2,
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

        # If scheduler wasn't passed in, use pipeline default
        if scheduler is None: scheduler = self.scheduler

        # Check CLIP before overwritting
        if clip_guidance_scale is not None and self.clip_model is None:
            print("Warning: CLIP guidance passed to a pipeline without a CLIP model. It will be ignored.")

        # Set defaults for clip
        if clip_guidance_scale is None: clip_guidance_scale = self.clip_defaults["guidance_scale"]
        if clip_guidance_base is None: clip_guidance_base = self.clip_defaults["guidance_base"]
        if clip_gradient_length is None: clip_gradient_length = self.clip_defaults["gradient_length"]
        if clip_gradient_threshold is None: clip_gradient_threshold = self.clip_defaults["gradient_threshold"]
        if clip_gradient_maxloss is None: clip_gradient_maxloss = self.clip_defaults["gradient_maxloss"]
        if vae_cutouts is None: vae_cutouts = self.clip_defaults["vae_cutouts"]
        if approx_cutouts is None: approx_cutouts = self.clip_defaults["approx_cutouts"]
        if no_cutouts is None: no_cutouts = self.clip_defaults["no_cutouts"]

        # Parse prompt and calculate batch size
        prompt = UnifiedPipelinePrompt(prompt)
        batch_size = prompt.batch_size

        # Match the negative prompt length to the batch_size
        if negative_prompt is None:
            negative_prompt = UnifiedPipelinePrompt([[("", 1.0)]] * batch_size)
        else:
            negative_prompt = UnifiedPipelinePrompt(negative_prompt)

        if batch_size != negative_prompt.batch_size:
            raise ValueError(
                f"negative_prompt has batch size {negative_prompt.batch_size}, but " 
                f"prompt has batch size {batch_size}. They need to match."
            )

        if clip_guidance_scale > 0:
            if clip_prompt is None:
                clip_prompt = prompt
            else:
                clip_prompt = UnifiedPipelinePrompt(clip_prompt)

            if batch_size != clip_prompt.batch_size:
                raise ValueError(
                    f"clip_prompt has batch size {clip_prompt.batch_size}, but " 
                    f"prompt has batch size {batch_size}. They need to match."
                )

        batch_total = batch_size * num_images_per_prompt

        if isinstance(generator, list):
            generators = generator
            # Match the generator list to the batch_total
            if len(generators) != batch_total:
                raise ValueError(
                    f"Generator passed as a list, but list length does not match "
                    f"batch size {batch_size} * number of images per prompt {num_images_per_prompt}, i.e. {batch_total}"
                )
        else:
            generators = [generator] * batch_total

        inspect.getmodule(scheduler).torch = TorchRandOverride(generators)

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

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0


        # Get the latents dtype based on the text_embeddings dtype
        text_embedding_calculator = BasicTextEmbedding(self, layer=self._text_embedding_layer)
        text_embeddings, uncond_embeddings = text_embedding_calculator.get_embeddings(
            prompt=prompt,
            uncond_prompt=negative_prompt if do_classifier_free_guidance else None,
        )
        latents_dtype = text_embeddings.dtype

        mode_classes = []

        # Calculate operating mode based on arguments
        if mask_image != None: 
            if self.inpaint_unet is not None: 
                mode_classes.append(EnhancedRunwayInpaintMode)
                if self._grafted_inpaint:
                    mode_classes.append(EnhancedInpaintMode)
            else:
                mode_classes.append(EnhancedInpaintMode)
        elif init_image != None: 
            mode_classes.append(Img2imgMode)
        else: 
            mode_classes.append(Txt2imgMode)

        # Get the underlying unet.
        # TODO: inpaint_unet is dependant on the right mode being used

        if mask_image is not None and self.inpaint_unet is not None: 
            unet = self.inpaint_unet
        else:
            unet = self.unet

        unet_sample_size = self.get_unet_sample_size(unet)
        unet_pixel_size = self.get_unet_pixel_size(unet)

        # if self._structured_diffusion:
        #     text_embedding_calculator = StructuredTextEmbedding(self, "align_seq")
        # else:
        #     # AFAIK there's no scenario where just BasicTextEmbedding is better than LPWTextEmbedding
        #     # text_embedding_calculator = BasicTextEmbedding(self)
        #     text_embedding_calculator = LPWTextEmbedding(self, max_embeddings_multiples=max_embeddings_multiples)

        # # get unconditional embeddings for classifier free guidance
        # text_embeddings, uncond_embeddings = text_embedding_calculator.get_embeddings(
        #     prompt=prompt,
        #     uncond_prompt=negative_prompt if do_classifier_free_guidance else None,
        # )

        text_embeddings = text_embedding_calculator.repeat(text_embeddings, num_images_per_prompt)
        # unet_g is the guided unet, for when we aren't doing CFG, or we want to run seperately to unet_u
        unet_g = UNetWithEmbeddings(unet, text_embeddings)

        if do_classifier_free_guidance:
            uncond_embeddings = text_embedding_calculator.repeat(uncond_embeddings, num_images_per_prompt)
            # unet_u is the unguided unet, for when we want to run seperately to unet_g
            unet_u = UNetWithEmbeddings(unet, uncond_embeddings)
            # unet_f is the fused unet, for running CFG in a single execution
            unet_f = UNetWithEmbeddings(unet, torch.cat([uncond_embeddings, text_embeddings]))

        if isinstance(scheduler, SchedulerMixin):
            scheduler_wrapper = DiffusersScheduler
        elif isinstance(scheduler, KSchedulerMixin):
            scheduler_wrapper = DiffusersKScheduler
        else:
            scheduler_wrapper = KDiffusionScheduler
        
        scheduler = scheduler_wrapper(scheduler, generators, self.device, latents_dtype, self.vae)


        print(f"Modes: {mode_classes} with strength {strength}")

        modes = [mode_class(
            pipeline=self, 
            scheduler=scheduler,
            generators=generators,
            width=width, height=height,
            init_image=init_image, 
            mask_image=mask_image,
            latents_dtype=latents_dtype,
            batch_total=batch_total,
            num_inference_steps=num_inference_steps,
            strength=strength,
            do_classifier_free_guidance=do_classifier_free_guidance
        ) for mode_class in mode_classes]

        if self.clip_model is not None and clip_guidance_scale > 0:
            max_length = min(
                self.clip_tokenizer.model_max_length,
                self.text_encoder.config.max_position_embeddings
            )

            clip_text_input = self.clip_tokenizer(
                clip_prompt.as_unweighted_string(),
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(self.device)

            text_embeddings_clip = self.clip_model.get_text_features(clip_text_input)
            text_embeddings_clip = text_embeddings_clip / text_embeddings_clip.norm(p=2, dim=-1, keepdim=True)
            # duplicate text embeddings clip for each generation per prompt
            text_embeddings_clip = text_embeddings_clip.repeat_interleave(num_images_per_prompt, dim=0)

            modes = [ClipGuidedMode(
                wrapped_mode=mode,
                scheduler=scheduler,
                pipeline=self,
                dtype=latents_dtype,
                guidance_scale=guidance_scale,
                text_embeddings_clip=text_embeddings_clip,
                clip_guidance_scale=clip_guidance_scale,
                clip_guidance_base=clip_guidance_base,
                clip_gradient_length=clip_gradient_length,
                clip_gradient_threshold=clip_gradient_threshold,
                clip_gradient_maxloss=clip_gradient_maxloss,
                vae_cutouts=vae_cutouts,
                approx_cutouts=approx_cutouts,
                no_cutouts=no_cutouts,
                generator=generator,
            ) for mode in modes]

        if do_classifier_free_guidance:
            unet_gs = [mode.wrap_unet(unet_g) for mode in modes]
            unet_us = [mode.wrap_unet(unet_u) for mode in modes]
            unet_fs = [mode.wrap_unet(unet_f) for mode in modes]

            if isinstance(scheduler, DiffusersSchedulerBase):
                unet_gs = [scheduler.wrap_scaled_unet(unet_g) for unet_g in unet_gs]
                unet_us = [scheduler.wrap_scaled_unet(unet_u) for unet_u in unet_us]
                unet_fs = [scheduler.wrap_scaled_unet(unet_f) for unet_f in unet_fs]

            unets = [
                mode.wrap_guidance_unet(unet_g, unet_u, unet_f, guidance_scale, batch_total)
                for mode, unet_g, unet_u, unet_f in zip(modes, unet_gs, unet_us, unet_fs)
            ]
        else:
            unet_gs = [mode.wrap_unet(unet_g) for mode in modes]

            if isinstance(scheduler, DiffusersSchedulerBase):
                unet_gs = [scheduler.wrap_scaled_unet(unet_g) for unet_g in unet_gs]

            unets = unet_gs

        scheduler.set_eps_unets(unets)

        timestep_args = {}
        if init_image is not None: timestep_args["strength"] = strength
        if prediction_type: timestep_args["prediction_type"] = prediction_type
        if eta: timestep_args["eta"] = eta
        if churn: timestep_args["churn"] = churn
        if churn_tmin: timestep_args["churn_tmin"] = churn_tmin
        if churn_tmax: timestep_args["churn_tmax"] = churn_tmax
        if sigma_min: timestep_args["sigma_min"] = sigma_min
        if sigma_max: timestep_args["sigma_max"] = sigma_max
        if karras_rho: timestep_args["karras_rho"] = karras_rho
        timestep_args["noise_type"] = scheduler_noise_type

        scheduler.set_timesteps(num_inference_steps, **timestep_args)

        if callback:
            scheduler.set_callback(callback, callback_steps)

        if isinstance(scheduler, KDiffusionScheduler):
            scheduler.unets = [
                mode.wrap_k_unet(unet)
                for mode, unet in zip(modes, scheduler.unets)
            ]
        else:
            scheduler.unets = [
                mode.wrap_d_unet(unet)
                for mode, unet in zip(modes, scheduler.unets)
            ]

        if len(scheduler.unets) == 1:
            scheduler.unet = scheduler.unets[0]
        else:
            scheduler.unet = GraftUnets(scheduler.unets, generators, self._graft_factor)


        # Get the initial starting point - either pure random noise, or the source image with some noise depending on mode
        # We only actually use the latents for the first mode, but UnifiedMode instances expect it to be called
        latents = [mode.generateLatents() for mode in modes][0]
        write_debug_latents(self.vae, "initial", 0, latents)

        if width <= unet_pixel_size and height <= unet_pixel_size: hires_fix = False

        if hires_fix:
            print("HiRes Fix Mode 2")
            scheduler.unet = hires_unet_wrapper(
                scheduler.unet, 
                [*latents.shape[:2], unet_sample_size, unet_sample_size],
                generators,
            )
            latents = torch.concat([latents, latents])

        # MAIN LOOP
        latents = scheduler.loop(latents, self.progress_bar)

        if hires_fix == 2:
           latents = latents.chunk(2)[1]

        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)

        if outmask_image != None:
            outmask = torch.cat([outmask_image] * batch_total)
            outmask = outmask[:, [0,1,2]]
            outmask = outmask.to(image.device)

            source =  torch.cat([init_image] * batch_total)
            source = source[:, [0,1,2]]
            source = source.to(image.device)

            image = source * (1-outmask) + image * outmask

        numpyImage = image.cpu().permute(0, 2, 3, 1).numpy()

        if run_safety_checker and self.safety_checker is not None:
            # run safety checker
            safety_cheker_input = self.feature_extractor(self.numpy_to_pil(numpyImage), return_tensors="pt").to(self.device)
            numpyImage, has_nsfw_concept = self.safety_checker(images=numpyImage, clip_input=safety_cheker_input.pixel_values.to(latents_dtype))
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
