import os, glob
from typing import Callable, List, Tuple, Optional, Union, Literal, NewType

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import sdgrpcserver.k_diffusion as K

import PIL
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.models import AutoencoderKL
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from sdgrpcserver.pipeline.kschedulers import *
from sdgrpcserver.pipeline.text_embedding import *

from sdgrpcserver.pipeline.unified_pipeline import UnifiedPipelinePrompt, UnifiedPipelinePromptType, UnifiedPipelineImageType


# Based on https://t.co/aNqKfn1Mxl - license for that code is below

# ---------------------------------------------------------------------------

# Copyright (c) 2022 Emily Shepperd <nshepperd@gmail.com>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


class NoiseLevelAndTextConditionedUpscaler(torch.nn.Module):
    def __init__(self, inner_model, sigma_data=1., embed_dim=256):
        super().__init__()
        self.inner_model = inner_model
        self.sigma_data = sigma_data
        self.low_res_noise_embed = K.layers.FourierFeatures(1, embed_dim, std=2)

    def forward(self, input, sigma, low_res, low_res_sigma, c, **kwargs):
        cross_cond, cross_cond_padding, pooler = c
        c_in = 1 / (low_res_sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_noise = low_res_sigma.log1p()[:, None]
        c_in = K.utils.append_dims(c_in, low_res.ndim)
        low_res_noise_embed = self.low_res_noise_embed(c_noise)
        low_res_in = F.interpolate(low_res, scale_factor=2, mode='nearest') * c_in
        mapping_cond = torch.cat([low_res_noise_embed, pooler], dim=1)
        return self.inner_model(input, sigma, unet_cond=low_res_in, mapping_cond=mapping_cond, cross_cond=cross_cond, cross_cond_padding=cross_cond_padding, **kwargs)
    
    @classmethod
    def from_pretrained(cls, source_path, pooler_dim=768, train=False, torch_dtype="float32", torch_device="cpu"):
        config = K.config.load_config(open(glob.glob(os.path.join(source_path, "*.json"))[0]))
        model = K.config.make_model(config)
        model = NoiseLevelAndTextConditionedUpscaler(
            model,
            sigma_data=config['model']['sigma_data'],
            embed_dim=config['model']['mapping_cond_dim'] - pooler_dim,
        )
        ckpt = torch.load(glob.glob(os.path.join(source_path, "*.pth"))[0], map_location='cpu')
        model.load_state_dict(ckpt['model_ema'])
        model = K.config.make_denoiser_wrapper(config)(model)
        if not train:
            model = model.eval().requires_grad_(False)
        return model.to(torch_device, dtype=torch_dtype)


class CFGUpscaler(torch.nn.Module):
    def __init__(self, model, uc, cond_scale):
        super().__init__()
        self.inner_model = model
        self.uc = uc
        self.cond_scale = cond_scale

    def forward(self, x, sigma, low_res, low_res_sigma, c):
        if self.cond_scale in (0.0, 1.0):
          # Shortcut for when we don't need to run both.
          if self.cond_scale == 0.0:
            c_in = self.uc
          elif self.cond_scale == 1.0:
            c_in = c
          return self.inner_model(x, sigma, low_res=low_res, low_res_sigma=low_res_sigma, c=c_in)
          
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        low_res_in = torch.cat([low_res] * 2)
        low_res_sigma_in = torch.cat([low_res_sigma] * 2)
        c_in = [torch.cat([uc_item, c_item]) for uc_item, c_item in zip(self.uc, c)]
        uncond, cond = self.inner_model(x_in, sigma_in, low_res=low_res_in, low_res_sigma=low_res_sigma_in, c=c_in).chunk(2)
        return uncond + (cond - uncond) * self.cond_scale

class CLIPTokenizerTransform:
    def __init__(self, tokenizer, max_length=77):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, text):
        indexer = 0 if isinstance(text, str) else ...
        tok_out = self.tokenizer(text, truncation=True, max_length=self.max_length,
                                 return_length=True, return_overflowing_tokens=False,
                                 padding='max_length', return_tensors='pt')
        input_ids = tok_out['input_ids'][indexer]
        attention_mask = 1 - tok_out['attention_mask'][indexer]
        return input_ids, attention_mask


class CLIPEmbedder(torch.nn.Module):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, transformer): #, device="cuda"):
        super().__init__()
        self.transformer = transformer

    @property
    def device(self):
        return self.transformer.device

    def forward(self, tok_out):
        input_ids, cross_cond_padding = tok_out
        clip_out = self.transformer(input_ids=input_ids.to(self.device), output_hidden_states=True)
        return clip_out.hidden_states[-1], cross_cond_padding.to(self.device), clip_out.pooler_output


# Model configuration values
SD_C = 4 # Latent dimension
SD_F = 8 # Latent patch size (pixels per latent)
SD_Q = 0.18215 # sd_model.scale_factor; scaling for latents in first stage models

class UpscalerPipeline(DiffusionPipeline):
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
        scheduler: SchedulerMixin,
        upscaler: NoiseLevelAndTextConditionedUpscaler,
        **kwargs,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            upscaler=upscaler,
        )

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        pass

    def disable_attention_slicing(self):
        pass

    def set_options(self, options):
        for key, value in options.items():
            raise ValueError(f"Unknown option {key}: {value} passed to UnifiedPipeline")

    def _random(self, generators, size, dtype, device):
        latents = torch.cat([
            torch.randn((1, *size[1:]), generator=generator, device=generator.device, dtype=dtype) 
            for generator in generators
        ], dim=0)
        
        return latents.to(device)

    @property
    def device(self) -> torch.device:
        r"""
        Returns:
            `torch.device`: The torch device on which the pipeline is located.
        """
        module_names, _ = self.extract_init_dict(dict(self.config))
        for name in module_names.keys():
            module = getattr(self, name)
            if isinstance(module, torch.nn.Module):
                device = getattr(module, 'device', None)
                if device:
                    return torch.device("cpu") if device == torch.device("meta") else device
        return torch.device("cpu")
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: UnifiedPipelinePromptType,
        init_image: Optional[UnifiedPipelineImageType] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        strength = 0.0,
        **kwargs
    ):

        tok_up = CLIPTokenizerTransform(self.tokenizer)
        text_encoder_up = CLIPEmbedder(self.text_encoder)

        prompt = UnifiedPipelinePrompt(prompt)
        batch_size = prompt.batch_size
        batch_total = batch_size * num_images_per_prompt
        negative_prompt = UnifiedPipelinePrompt([[("", 1.0)]] * batch_size)

        if isinstance(generator, list) and len(generator) != batch_total:
            raise ValueError(
                f"Generator passed as a list, but list length does not match "
                f"batch size {batch_size} * number of images per prompt {num_images_per_prompt}, i.e. {batch_total}"
            )

        c = text_encoder_up(tok_up(num_images_per_prompt * prompt.as_unweighted_string()))
        uc = text_encoder_up(tok_up(batch_total * [""]))

        dtype = c[0].dtype
        device = self.device

        if isinstance(init_image, PIL.Image.Image):
            init_image = T.functional.to_tensor(init_image)

        init_image = init_image.to(device, dtype=dtype)

        if init_image.ndim == 3: init_image = init_image[None, ...]
        init_image = init_image[:, [0,1,2]]
        init_image = 2.0 * init_image - 1.0

        dist = self.vae.encode(init_image).latent_dist

        low_res_latent = torch.cat([
            dist.sample(generator=batch_generator)
            for batch_generator in generator
        ], dim=0)

        low_res_latent = low_res_latent * SD_Q

        [_, C, H, W] = low_res_latent.shape

        # Noise levels from stable diffusion.
        sigma_min, sigma_max = 0.029167532920837402, 14.614642143249512

        model_wrap = CFGUpscaler(self.upscaler, uc, cond_scale=guidance_scale)
        low_res_sigma = torch.full([batch_total], strength, device=device, dtype=dtype)
        x_shape = [batch_total, C, 2*H, 2*W]

        if isinstance(self.scheduler, EulerDiscreteScheduler):
            sampler = 'k_euler'
        elif isinstance(self.scheduler, EulerAncestralDiscreteScheduler):
            sampler = 'k_euler_ancestral'
        elif isinstance(self.scheduler, DPM2AncestralDiscreteScheduler):
            sampler = 'k_dpm_2_ancestral'
        else:
            sampler='k_dpm_fast'
   
        def do_sample(noise, extra_args):
            # We take log-linear steps in noise-level from sigma_max to sigma_min, using one of the k diffusion samplers.
            sigmas = torch.linspace(np.log(sigma_max), np.log(sigma_min), num_inference_steps+1).exp().to(device, dtype=dtype)
            if sampler == 'k_euler':
                return K.sampling.sample_euler(model_wrap, noise * sigma_max, sigmas, extra_args=extra_args)
            elif sampler == 'k_euler_ancestral':
                return K.sampling.sample_euler_ancestral(model_wrap, noise * sigma_max, sigmas, extra_args=extra_args, eta=eta)
            elif sampler == 'k_dpm_2_ancestral':
                return K.sampling.sample_dpm_2_ancestral(model_wrap, noise * sigma_max, sigmas, extra_args=extra_args, eta=eta)
            elif sampler == 'k_dpm_fast':
                return K.sampling.sample_dpm_fast(model_wrap, noise * sigma_max, sigma_min, sigma_max, num_inference_steps, extra_args=extra_args, eta=eta)

        latent_noised = low_res_latent + strength * self._random(generator, low_res_latent, dtype, device)
        extra_args = {'low_res': latent_noised, 'low_res_sigma': low_res_sigma, 'c': c}

        noise = self._random(generator, x_shape, dtype, device)

        up_latents = do_sample(noise, extra_args)

        print("X")

        pixels = self.vae.decode(up_latents/SD_Q).sample # equivalent to sd_model.decode_first_stage(up_latents)
        pixels = pixels.add(1).div(2).clamp(0,1).cpu()

        print("Y", pixels.device)

        return (pixels, [False] * pixels.shape[0])
