from dataclasses import dataclass
from typing import Callable, Literal, cast

import numpy as np
import torch
import torchvision.transforms as T

from sdgrpcserver import resize_right
from sdgrpcserver.k_diffusion import utils as k_utils
from sdgrpcserver.pipeline.common_scheduler import (
    DiffusersKScheduler,
    DiffusersSchedulerBase,
)
from sdgrpcserver.pipeline.unet.cfg import CFGChildUnets
from sdgrpcserver.pipeline.unet.types import (
    DiffusersSchedulerUNet,
    EpsTensor,
    KDiffusionSchedulerUNet,
    NoisePredictionUNet,
    ScheduleTimestep,
    XtTensor,
)
from sdgrpcserver.pipeline.vae_approximator import VaeApproximator

CLIP_NO_CUTOUTS_TYPE = Literal[False, True, "vae", "approx"]
CLIP_GUIDANCE_BASE = Literal["guided", "mixed"]


@dataclass
class ClipGuidanceConfig:
    guidance_scale: float = 0
    guidance_base: CLIP_GUIDANCE_BASE = "guided"
    gradient_length: int = 15
    gradient_threshold: float = 0.01
    gradient_maxloss: float = 1.0
    vae_cutouts: int = 2
    approx_cutouts: int = 2
    no_cutouts: CLIP_NO_CUTOUTS_TYPE = False


class MakeCutouts(torch.nn.Module):
    def __init__(
        self, cut_size: int, generators: list[torch.Generator], cut_power: float = 1.0
    ):
        super().__init__()

        self.cut_size = cut_size
        self.cut_power = cut_power
        self.generators = generators

    def randint(self, max, generator, min=0):
        return torch.randint(
            min,
            max,
            (),
            generator=generator,
            device=generator.device,
        )

    def forward(self, pixel_values: torch.Tensor, num_cutouts: int):
        sideY, sideX = pixel_values.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        cut_shape = (self.cut_size, self.cut_size)

        # Number of batches must match number of generators provided
        assert pixel_values.shape[0] == len(self.generators)

        # To maintain batch-independance, we split out per batch, since each batch
        # has a different generator. This also keeps the output grouped by batch
        # (b1c1, b1c2, b1c3, ..., b2c1, b2c2, ...) whereas otherwise it would be
        # grouped by cutout (b1c1, b2c1, b3c1, ....., b1c2, b2c2, ...)
        for generator, batch_pixels in zip(self.generators, pixel_values.split(1)):
            for _ in range(num_cutouts):
                size = torch.rand([], generator=generator, device=generator.device)
                size = int(size**self.cut_power * (max_size - min_size) + min_size)
                offsetx = self.randint(sideX - size + 1, generator)
                offsety = self.randint(sideY - size + 1, generator)
                cutout = batch_pixels[
                    :, :, offsety : offsety + size, offsetx : offsetx + size
                ]

                cutouts.append(
                    resize_right.resize(cutout, out_shape=cut_shape, pad_mode="reflect")
                )

        return torch.cat(cutouts)


def spherical_dist_loss(x, y):
    x = torch.nn.functional.normalize(x, dim=-1)
    y = torch.nn.functional.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


class ClipGuidedMode:
    def __init__(
        self,
        wrapped_mode,
        scheduler,
        pipeline,
        dtype: torch.dtype,
        text_embeddings_clip: torch.Tensor,
        config: ClipGuidanceConfig,
        generators: list[torch.Generator],
        reversible_ctx: Callable,
        **kwargs
    ):
        self.wrapped_mode = wrapped_mode

        self.scheduler = scheduler
        self.pipeline = pipeline
        self.dtype = dtype

        self.text_embeddings_clip = text_embeddings_clip
        self.config = config
        self.reversible_ctx = reversible_ctx

        self.normalize = T.Normalize(
            mean=self.pipeline.feature_extractor.image_mean,
            std=self.pipeline.feature_extractor.image_std,
        )
        self.normalizeB = T.Normalize(
            mean=self.pipeline.feature_extractor.image_mean[2],
            std=self.pipeline.feature_extractor.image_std[2],
        )

        self.vae_scale_factor = pipeline.vae_scale_factor

        feature_extractor_size = self.pipeline.feature_extractor.size
        if isinstance(feature_extractor_size, dict):
            feature_extractor_size = feature_extractor_size["shortest_edge"]

        self.make_cutouts = MakeCutouts(
            feature_extractor_size // self.vae_scale_factor,
            generators=generators,
        )
        self.make_cutouts_rgb = MakeCutouts(
            feature_extractor_size, generators=generators
        )

        self.approx_decoder = VaeApproximator(
            device=self.pipeline.execution_device, dtype=dtype
        )

        self.generators = generators

        self.lossavg = []
        self.flatloss = False

        # For K-diffusion samplers, we might want to force running just the guided stem
        # We cache the result to avoid re-calulating
        self._guided_stem_only = False

    def _has_flatloss(self):
        # Check to see if loss gradient has flattened out
        # (i.e. guidance is no longer reducing loss)
        if not self.flatloss and len(self.lossavg) > self.config.gradient_length:
            # Calculate gradient for loss
            x = np.linspace(0, 1, self.config.gradient_length)
            X = np.vstack([x, np.ones(len(x))]).T
            y = np.asarray(self.lossavg[-self.config.gradient_length :])

            try:
                m, c = np.linalg.lstsq(X, y, rcond=None)[0]
                if (
                    abs(m) < self.config.gradient_threshold
                    and c < self.config.gradient_maxloss
                ):
                    # print("Flat", len(self.lossavg), m, c)
                    self.flatloss = True
            except np.linalg.LinAlgError:
                pass

        return self.flatloss

    def generateLatents(self) -> XtTensor:
        return self.wrapped_mode.generateLatents()

    def wrap_unet(self, unet: NoisePredictionUNet) -> NoisePredictionUNet:
        return self.wrapped_mode.wrap_unet(unet)

    def xformers_retry(self, callback, *args, **kwargs):
        with self.reversible_ctx():
            return callback(*args, **kwargs)

    def wrap_guidance_unet_ford(
        self,
        cfg_unets: CFGChildUnets,
        child_wrapped_unet: NoisePredictionUNet,
        guidance_scale: float,
        batch_total: int,
    ) -> NoisePredictionUNet:
        def wrapper_unet(latents: XtTensor, t: ScheduleTimestep) -> EpsTensor:
            if self.config.guidance_base == "guided":
                noise_pred_u = cfg_unets.u(latents, t)
                noise_pred_g, grads = self.xformers_retry(
                    self.model_d_fn, cfg_unets.g, latents, t
                )

                noise_pred = noise_pred_u + guidance_scale * (
                    noise_pred_g - noise_pred_u
                )
            else:
                noise_pred, grads = self.xformers_retry(
                    self.model_d_fn, child_wrapped_unet, latents, t
                )

            if isinstance(self.scheduler, DiffusersKScheduler):
                sigma = self.scheduler.scheduler.t_to_sigma(t)
                return noise_pred - grads * sigma

            else:
                alpha_prod_t = self.scheduler.scheduler.alphas_cumprod[t]
                beta_prod_t = 1 - alpha_prod_t
                return noise_pred - torch.sqrt(beta_prod_t) * grads

        return wrapper_unet

    def wrap_guidance_unet_fork(
        self,
        cfg_unets: CFGChildUnets,
        child_wrapped_unet: NoisePredictionUNet,
        guidance_scale: float,
        batch_total: int,
    ) -> NoisePredictionUNet:

        g_cache: list[EpsTensor] = []

        def wrapped_unet(latents: XtTensor, t: ScheduleTimestep) -> EpsTensor:
            if self._guided_stem_only:
                noise_pred_g = cfg_unets.g(latents, t)
                g_cache.append(noise_pred_g)
                return noise_pred_g
            else:
                if g_cache:
                    noise_pred_g = g_cache.pop()
                    noise_pred_u = cfg_unets.u(latents, t)
                    return noise_pred_u + guidance_scale * (noise_pred_g - noise_pred_u)
                else:
                    return child_wrapped_unet(latents, t)

        return wrapped_unet

    def wrap_guidance_unet(
        self,
        cfg_unets: CFGChildUnets,
        guidance_scale: float,
        batch_total: int,
    ) -> NoisePredictionUNet:
        child_wrapped_unet = self.wrapped_mode.wrap_guidance_unet(
            cfg_unets, guidance_scale, batch_total
        )

        if isinstance(self.scheduler, DiffusersSchedulerBase):
            wrapped_unet_forscheduler = self.wrap_guidance_unet_ford(
                cfg_unets, child_wrapped_unet, guidance_scale, batch_total
            )
        else:
            wrapped_unet_forscheduler = self.wrap_guidance_unet_fork(
                cfg_unets, child_wrapped_unet, guidance_scale, batch_total
            )

        def wrapped_unet(latents, t):
            if self._has_flatloss():
                return child_wrapped_unet(latents, t)
            else:
                return wrapped_unet_forscheduler(latents, t)

        return wrapped_unet

    def wrap_k_unet(self, unet: KDiffusionSchedulerUNet) -> KDiffusionSchedulerUNet:
        child_wrapped_unet = self.wrapped_mode.wrap_k_unet(unet)

        def wrapped_unet(latents, sigma, u):
            if self._has_flatloss():
                return child_wrapped_unet(latents, sigma, u=u)
            else:
                if self.config.guidance_base == "guided":
                    self._guided_stem_only = True
                    _, grads = self.xformers_retry(
                        self.model_k_fn, child_wrapped_unet, latents, sigma, u=u
                    )
                    self._guided_stem_only = False
                    res = child_wrapped_unet(latents, sigma, u=u)
                else:
                    self._guided_stem_only = False
                    res, grads = self.xformers_retry(
                        self.model_k_fn, child_wrapped_unet, latents, sigma, u=u
                    )

                # grads * sigma will fail if batchsize > 1 unless we match shape
                if isinstance(sigma, torch.Tensor):
                    sigma = k_utils.append_dims(sigma, len(res.shape))

                return res + grads * (sigma**2)

        return wrapped_unet

    def wrap_d_unet(self, unet: DiffusersSchedulerUNet) -> DiffusersSchedulerUNet:
        return self.wrapped_mode.wrap_d_unet(unet)

    @torch.enable_grad()
    def model_k_fn(self, unet, latents, sigma, u):
        latents = latents.detach().requires_grad_()
        sample = unet(latents, sigma, u=u)

        # perform guidance - in a seperate function as this contains the bits
        # where grad is needed
        grads = self.cond_fn(
            latents,
            sample,
            self.text_embeddings_clip,
            self.config.guidance_scale,
            self.config.vae_cutouts,
            self.config.approx_cutouts,
            self.config.no_cutouts,
        )

        return sample, grads

    @torch.enable_grad()
    def model_d_fn(self, unet, latents, t):
        latents = latents.detach().requires_grad_()
        noise_pred = unet(latents, t)
        sample = self.scheduler.predict_x0(latents, noise_pred, t)

        # perform guidance - in a seperate function as this contains the bits
        # where grad is needed
        grads = self.cond_fn(
            latents,
            sample,
            self.text_embeddings_clip,
            self.config.guidance_scale,
            self.config.vae_cutouts,
            self.config.approx_cutouts,
            self.config.no_cutouts,
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

        if not num_cutouts:
            if no_cutouts == "approx":
                image = self.approx_decoder(sample)
                image = T.Resize(self.pipeline.feature_extractor.size)(image)
            else:
                sample = T.Resize(
                    self.pipeline.feature_extractor.size // self.vae_scale_factor
                )(sample)
                sample = 1 / 0.18215 * sample
                image = self.pipeline.vae.decode(sample).sample

        else:
            image = None

            if approx_cutouts:
                out_shape = (
                    sample.shape[2] * self.vae_scale_factor,
                    sample.shape[3] * self.vae_scale_factor,
                )
                image = self.approx_decoder(sample)
                image = resize_right.resize(
                    image, out_shape=out_shape, pad_mode="reflect"
                )
                image = self.make_cutouts_rgb(image, approx_cutouts)

            if vae_cutouts:
                sample2 = self.make_cutouts(sample, vae_cutouts)
                sample2 = 1 / 0.18215 * sample2
                image2 = self.pipeline.vae.decode(sample2).sample

                if image is None:
                    image = image2
                else:
                    # image and image2 are batch-grouped. We need to interleave them so
                    # they are still batch groups after
                    #   [b1c1, b1c2, b2c1, b2c2] + [b1c3, b1c4, b2c3, b2c4] =>
                    #   [b1c1, b1c2, b1c3, b1c4, b2c1, b2c2, b2c3, b2c4]
                    # First, split into 5D b cut c h w tensors
                    image = image.view(batch_total, approx_cutouts, *image.shape[-3:])
                    image2 = image.view(batch_total, vae_cutouts, *image2.shape[-3:])
                    # Now stack on the cut dimension, so we get a 6D tensor
                    # b cutgroup cut c h w
                    image = torch.stack([image, image2], dim=1)
                    # Then collapse down into 4D b*cut c h w
                    image = image.view(batch_total * num_cutouts, *image.shape[-3:])

        image = cast(torch.Tensor, image)
        image = (image / 2 + 0.5).clamp(0, 1)
        image = self.normalize(image)

        image_embeddings_clip = self.pipeline.clip_model.get_image_features(image)

        if no_cutouts:
            loss = spherical_dist_loss(
                image_embeddings_clip, text_embeddings_clip
            ).mean()
        else:
            text_embeddings_input = text_embeddings_clip.repeat_interleave(
                num_cutouts, dim=0
            )
            dists = spherical_dist_loss(image_embeddings_clip, text_embeddings_input)
            dists = dists.view([num_cutouts, latents.shape[0], -1])
            loss = dists.sum(2).mean(0).sum()

        self.lossavg.append(float(loss) / latents.shape[0])

        return -torch.autograd.grad(loss * (clip_guidance_scale * 500), latents)[0]
