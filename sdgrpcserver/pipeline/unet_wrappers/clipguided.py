from typing import Callable, List, Tuple, Iterable, Optional, Union, Literal, NewType

import numpy as np
import torch
import torchvision.transforms as T

from sdgrpcserver import resize_right
from sdgrpcserver.pipeline.vae_approximator import VaeApproximator
from sdgrpcserver.pipeline.scheduler_wrappers import *
from sdgrpcserver.pipeline.unet_wrappers.types import *

class MakeCutouts(torch.nn.Module):
    def __init__(self, cut_size: int, generators: List[torch.Generator], cut_power: float = 1.0):
        super().__init__()

        self.cut_size = cut_size
        self.cut_power = cut_power
        self.generators = generators

    def forward(self, pixel_values, num_cutouts):
        sideY, sideX = pixel_values.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []

        # To maintain batch-independance, we split out per batch, since each batch has a different generator
        # This also keeps the output grouped by batch (b1c1, b1c2, b1c3, ..., b2c1, b2c2, ...) whereas otherwise
        # it would be grouped by cutout (b1c1, b2c1, b3c1, ....., b1c2, b2c2, ...)
        for generator, batch_pixels in zip(self.generators, pixel_values.split(1)):
            for _ in range(num_cutouts):
                size = int(torch.rand([], generator=generator, device=generator.device) ** self.cut_power * (max_size - min_size) + min_size)
                offsetx = torch.randint(0, sideX - size + 1, (), generator=generator, device=generator.device)
                offsety = torch.randint(0, sideY - size + 1, (), generator=generator, device=generator.device)
                cutout = batch_pixels[:, :, offsety : offsety + size, offsetx : offsetx + size]

                cutouts.append(resize_right.resize(cutout, out_shape=(self.cut_size, self.cut_size), pad_mode='reflect').to(cutout.dtype))
        
        return torch.cat(cutouts)


def spherical_dist_loss(x, y):
    x = torch.nn.functional.normalize(x, dim=-1)
    y = torch.nn.functional.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

CLIP_NO_CUTOUTS_TYPE = Literal[False, True, "vae", "approx"]
CLIP_GUIDANCE_BASE = Literal["guided", "mixed"]

class ClipGuidedMode:

    def __init__(self, wrapped_mode, scheduler, pipeline, dtype, guidance_scale, text_embeddings_clip, clip_guidance_scale, clip_guidance_base, clip_gradient_length, clip_gradient_threshold, clip_gradient_maxloss, vae_cutouts, approx_cutouts, no_cutouts, generator):
        self.wrapped_mode = wrapped_mode

        self.scheduler = scheduler
        self.pipeline = pipeline
        self.dtype = dtype
        self.guidance_scale = guidance_scale

        self.text_embeddings_clip = text_embeddings_clip
        self.clip_guidance_scale = clip_guidance_scale
        self.clip_guidance_base = clip_guidance_base
        self.clip_gradient_length = clip_gradient_length
        self.clip_gradient_threshold = clip_gradient_threshold
        self.clip_gradient_maxloss = clip_gradient_maxloss
        self.vae_cutouts = vae_cutouts
        self.approx_cutouts = approx_cutouts
        self.no_cutouts = no_cutouts

        self.normalize = T.Normalize(mean=self.pipeline.feature_extractor.image_mean, std=self.pipeline.feature_extractor.image_std)
        self.normalizeB = T.Normalize(mean=self.pipeline.feature_extractor.image_mean[2], std=self.pipeline.feature_extractor.image_std[2])

        self.vae_scale_factor = pipeline.vae_scale_factor

        self.make_cutouts = MakeCutouts(self.pipeline.feature_extractor.size // self.vae_scale_factor, generators=generator)
        self.make_cutouts_rgb = MakeCutouts(self.pipeline.feature_extractor.size, generators=generator)

        self.approx_decoder = VaeApproximator(device = self.pipeline.device, dtype = dtype)

        self.generator = generator

        self.lossavg = []
        self.flatloss = False

        self._cache_g = None

    def generateLatents(self):
        return self.wrapped_mode.generateLatents()

    def wrap_unet(self, unet: NoisePredictionUNet) -> NoisePredictionUNet:
        return self.wrapped_mode.wrap_unet(unet)

    def wrap_guidance_unet(
        self, 
        unet_g: NoisePredictionUNet, 
        unet_u: NoisePredictionUNet,
        unet_f: NoisePredictionUNet,
        guidance_scale: float, 
        batch_total: int
    ) -> NoisePredictionUNet:
        child_wrapped_unet = self.wrapped_mode.wrap_guidance_unet(unet_g, unet_u, unet_f, guidance_scale, batch_total)

        def wrapped_unet_ford(latents, t):
            if self._has_flatloss():
                return child_wrapped_unet(latents, t)
            else:
                if self.clip_guidance_base == "guided":
                    noise_pred_u = unet_u(latents, t)
                    noise_pred_g, grads = self.model_d_fn(unet_g, latents, t)

                    noise_pred = noise_pred_u + guidance_scale * (noise_pred_g - noise_pred_u)
                else:
                    noise_pred, grads = self.model_d_fn(child_wrapped_unet, latents, t)


                if isinstance(self.scheduler, DiffusersKScheduler):
                    sigma = self.scheduler.scheduler.t_to_sigma(t)
                    return noise_pred - grads * sigma

                else:
                    # Todo: Don't duplicate this bit from cond_fn
                    alpha_prod_t = self.scheduler.scheduler.alphas_cumprod[t]
                    beta_prod_t = 1 - alpha_prod_t
                    return noise_pred - torch.sqrt(beta_prod_t) * grads

        def wrapped_unet_fork(latents, t):
            if self._has_flatloss():
                return child_wrapped_unet(latents, t)
            else:
                if self.section == "g":
                    noise_pred_g = unet_g(latents, t)
                    self._cache_g = (noise_pred_g, t)
                    return noise_pred_g
                else:
                    noise_pred_g = None

                    if self._cache_g:
                        noise_pred_g, cache_t = self._cache_g
                        if cache_t != t: noise_pred_g = None

                    if noise_pred_g is None: 
                        return child_wrapped_unet(latents, t)

                    else:
                        noise_pred_u = unet_u(latents, t)
                        return noise_pred_u + guidance_scale * (noise_pred_g  - noise_pred_u)

        if isinstance(self.scheduler, DiffusersSchedulerBase):
            return wrapped_unet_ford
        else:
            return wrapped_unet_fork

    def _has_flatloss(self):
        # Check to see if loss gradient has flattened out (i.e. guidance is no longer reducing loss) 
        if not self.flatloss and len(self.lossavg) > self.clip_gradient_length:
            # Calculate gradient for loss
            x = np.linspace(0,1,self.clip_gradient_length)
            X = np.vstack([x, np.ones(len(x))]).T
            y = np.asarray(self.lossavg[-self.clip_gradient_length:])
            
            try:
                m, c = np.linalg.lstsq(X, y, rcond=None)[0]
                if abs(m) < self.clip_gradient_threshold and c < self.clip_gradient_maxloss:
                    # print("Flat", len(self.lossavg), m, c)
                    self.flatloss = True
            except np.linalg.LinAlgError:
                pass

        return self.flatloss

    def wrap_k_unet(self, unet: KDiffusionSchedulerUNet) -> KDiffusionSchedulerUNet:
        child_wrapped_unet = self.wrapped_mode.wrap_k_unet(unet)

        def wrapped_unet(latents, sigma, u):
            if self._has_flatloss():
                return child_wrapped_unet(latents, sigma, u=u)
            else:
                if self.clip_guidance_base == "guided":
                    self.section="g"
                    _, grads = self.model_k_fn(child_wrapped_unet, latents, sigma, u=u)
                    self.section="merged"
                    res = child_wrapped_unet(latents, sigma, u=u)
                else:
                    self.section="merged"
                    res, grads = self.model_k_fn(child_wrapped_unet, latents, sigma, u=u)

                return res + grads * (sigma**2)


        return wrapped_unet
    
    def wrap_d_unet(self, unet: DiffusersSchedulerUNet) -> DiffusersSchedulerUNet:
        return self.wrapped_mode.wrap_d_unet(unet)
    
    @torch.enable_grad()
    def model_k_fn(self, unet, latents, sigma, u):
        latents = latents.detach().requires_grad_()
        sample = unet(latents, sigma, u=u)

        # perform guidance - in a seperate function as this contains the bits where grad is needed
        grads = self.cond_fn(
            latents,
            sample,
            self.text_embeddings_clip,
            # Adjust to match to non-KScheduler effect. Determined through experiment.
            # self.clip_guidance_scale / 10 
            self.clip_guidance_scale,
            self.vae_cutouts,
            self.approx_cutouts,
            self.no_cutouts
        )

        return sample, grads

    @torch.enable_grad()
    def model_d_fn(self, unet, latents, t):

        latents = latents.detach().requires_grad_()
        noise_pred = unet(latents, t)

        sample = self.scheduler.predict_x0(latents, noise_pred, t)

        # perform guidance - in a seperate function as this contains the bits where grad is needed
        grads = self.cond_fn(
            latents,
            sample,
            self.text_embeddings_clip,
            self.clip_guidance_scale,
            self.vae_cutouts,
            self.approx_cutouts,
            self.no_cutouts
        )

        return noise_pred, grads

    @torch.enable_grad()
    def cond_fn(
        self,
        latents,
        sample,
        text_embeddings_clip,
        clip_guidance_scale,
        vae_cutouts,
        approx_cutouts,
        no_cutouts,
    ):
        num_cutouts = vae_cutouts + approx_cutouts
        batch_total = latents.shape[0]

        if no_cutouts:
            if no_cutouts == "approx":
                image = self.approx_decoder(sample)
                image = T.Resize(self.pipeline.feature_extractor.size)(image)
            else:
                sample = T.Resize(self.pipeline.feature_extractor.size // self.vae_scale_factor)(sample)
                sample = 1 / 0.18215 * sample
                image = self.pipeline.vae.decode(sample).sample

        else:
            image = None

            if approx_cutouts:
                out_shape=(
                    sample.shape[2]*self.vae_scale_factor, 
                    sample.shape[3]*self.vae_scale_factor
                )
                image = self.approx_decoder(sample)
                image = resize_right.resize(image, out_shape=out_shape, pad_mode='reflect').to(sample.dtype)
                image = self.make_cutouts_rgb(image, approx_cutouts)

            if vae_cutouts:
                sample2 = self.make_cutouts(sample, vae_cutouts)
                sample2 = 1 / 0.18215 * sample2
                image2 = self.pipeline.vae.decode(sample2).sample

                if image is None:
                    image = image2
                else:
                    # image and image2 are batch-grouped. We need to interleave them so they are still batch groups after
                    # [b1c1, b1c2, b2c1, b2c2] + [b1c3, b1c4, b2c3, b2c4] => [b1c1, b1c2, b1c3, b1c4, b2c1, b2c2, b2c3, b2c4]
                    # First, split into 5D b cut c h w tensors
                    image = image.view(batch_total, approx_cutouts, *image.shape[-3:])
                    image2 = image.view(batch_total, vae_cutouts, *image2.shape[-3:])
                    # Now stack on the cut dimension, so we get a 6D tensor b cutgroup cut c h w
                    image = torch.stack([image, image2], dim=1)
                    # Then collapse down into 4D b*cut c h w
                    image = image.view(batch_total * num_cutouts, *image.shape[-3:])

        image = (image / 2 + 0.5).clamp(0, 1)
        image = self.normalize(image)

        image_embeddings_clip = self.pipeline.clip_model.get_image_features(image)

        if no_cutouts:
            loss = spherical_dist_loss(image_embeddings_clip, text_embeddings_clip).mean()
        else:
            text_embeddings_input = text_embeddings_clip.repeat_interleave(num_cutouts, dim=0)
            dists = spherical_dist_loss(image_embeddings_clip, text_embeddings_input)
            dists = dists.view([num_cutouts, latents.shape[0], -1])
            loss = dists.sum(2).mean(0).sum() 

        self.lossavg.append(float(loss))

        return -torch.autograd.grad(loss * (clip_guidance_scale * 500), latents)[0]
