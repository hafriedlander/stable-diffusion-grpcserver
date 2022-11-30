from dataclasses import dataclass

import torch

from sdgrpcserver.pipeline.unet.types import (
    EpsTensor,
    NoisePredictionUNet,
    ScheduleTimestep,
    XtTensor,
)


@dataclass
class CFGChildUnets:
    g: NoisePredictionUNet
    u: NoisePredictionUNet
    f: NoisePredictionUNet

    def wrap_all(self, wrapper):
        return CFGChildUnets(
            g=wrapper(self.g),
            u=wrapper(self.u),
            f=wrapper(self.f),
        )


class CFGUnet_Seperated:
    def __init__(self, cfg_unets: CFGChildUnets, guidance_scale, batch_total):
        self.cfg_unets = cfg_unets
        self.guidance_scale = guidance_scale
        self.batch_total = batch_total

    def __call__(self, latents: XtTensor, t: ScheduleTimestep) -> EpsTensor:
        noise_pred_g = self.cfg_unets.g(latents, t)
        noise_pred_u = self.cfg_unets.u(latents, t)

        noise_pred = noise_pred_u + self.guidance_scale * (noise_pred_g - noise_pred_u)
        return noise_pred


class CFGUnet:
    def __init__(self, cfg_unets: CFGChildUnets, guidance_scale, batch_total):
        self.cfg_unets = cfg_unets
        self.guidance_scale = guidance_scale
        self.batch_total = batch_total

    def __call__(self, latents: XtTensor, t: ScheduleTimestep) -> EpsTensor:
        latents = torch.cat([latents] * 2)

        noise_pred = self.cfg_unets.f(latents, t)
        noise_pred_u, noise_pred_g = noise_pred.chunk(2)

        noise_pred = noise_pred_u + self.guidance_scale * (noise_pred_g - noise_pred_u)
        return noise_pred
