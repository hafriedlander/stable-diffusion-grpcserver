from typing import Literal, Sequence, cast

import torch

from gyre import resize_right
from gyre.pipeline.easing import Easing
from gyre.pipeline.randtools import batched_rand
from gyre.pipeline.unet.types import (
    DiffusersSchedulerUNet,
    EpsTensor,
    GenericSchedulerUNet,
    KDiffusionSchedulerUNet,
    PX0Tensor,
    ScheduleTimestep,
    XtTensor,
)

# from gyre.pipeline.unet.types import *


def match_shape(latents: torch.Tensor, target: torch.Size):
    # If it's already the right size, just return it
    if latents.shape[-len(target) :] == target:
        return latents

    # Maybe scale it?
    scale = max(target[0] / latents.shape[2], target[1] / latents.shape[3])
    if scale != 1:
        latents = resize_right.resize(latents, scale_factors=scale, pad_mode="reflect")

    # If we don't need to crop, skip that bit
    if latents.shape[-len(target) :] == target:
        return latents

    offset2 = (latents.shape[2] - target[0]) // 2
    offset3 = (latents.shape[3] - target[1]) // 2

    return latents[:, :, offset2 : offset2 + target[0], offset3 : offset3 + target[1]]


class HiresUnetEpsWrapper:
    def __init__(
        self,
        parent: "HiresUnetWrapper",
        unet: DiffusersSchedulerUNet | KDiffusionSchedulerUNet,
        target: torch.Size,
    ):
        self.parent = parent
        self.unet = unet
        self.target = target

    def __call__(self, latents: XtTensor, t: ScheduleTimestep) -> EpsTensor:
        if self.parent.mode == "lo":
            *_, h, w = latents.shape
            th, tw = self.target

            offseth = (h - th) // 2
            offsetw = (w - tw) // 2

            in_latents = latents[:, :, offseth : offseth + th, offsetw : offsetw + tw]

            res = self.unet(in_latents, t)

            expanded = torch.zeros(
                (*res.shape[:2], *latents.shape[2:]),
                dtype=latents.dtype,
                device=latents.device,
            )
            expanded[:, :, offseth : offseth + th, offsetw : offsetw + tw] = res

            return expanded

        else:
            return self.unet(latents, t)


class HiresUnetGenericWrapper(GenericSchedulerUNet):
    def __init__(
        self,
        parent: "HiresUnetWrapper",
        unet: DiffusersSchedulerUNet | KDiffusionSchedulerUNet,
        generators: list[torch.Generator],
        target: torch.Size,
        latent_debugger,
    ):
        self.parent = parent
        self.unet = unet
        self.generators = generators
        self.target = target

        self.easing = Easing(floor=0, start=0, end=0.3, easing="quartic")
        self.latent_debugger = latent_debugger

    def __call__(self, latents: XtTensor, __step, u: float) -> PX0Tensor | XtTensor:
        # Linear blend between base and graft
        p = self.easing.interp(u)

        lo_in, hi_in = latents.chunk(2)

        if isinstance(__step, torch.Tensor) and __step.shape:
            lo_t, hi_t = __step.chunk(2)
        else:
            lo_t = hi_t = __step

        self.parent.mode = "hi"
        hi = self.unet(hi_in, hi_t, u=u)

        # Early out if we're passed the graft stage
        if p >= 0.999:
            return cast(type(hi), torch.concat([lo_in, hi]))

        *_, h, w = latents.shape
        th, tw = self.target

        offseth = (h - th) // 2
        offsetw = (w - tw) // 2

        self.parent.mode = "lo"
        lo = self.unet(lo_in, lo_t, u=u)[
            :, :, offseth : offseth + th, offsetw : offsetw + tw
        ]

        # Crop hi and merge it back into lo
        hi_crop = hi[:, :, offseth : offseth + th, offsetw : offsetw + tw]

        randmap = batched_rand(lo.shape, self.generators, lo.device, lo.dtype)
        lo_merged = torch.where(randmap >= p, lo, hi_crop)

        # Scale lo and merge it back into hi
        lo_scaled = match_shape(lo, hi.shape[-2:])

        randmap = batched_rand(hi.shape, self.generators, hi.device, hi.dtype)
        hi_merged = torch.where(randmap >= p, lo_scaled, hi)

        # Expand lo back to full tensor size by wrapping with 0
        lo_expanded = torch.zeros_like(hi_merged)
        lo_expanded[:, :, offseth : offseth + th, offsetw : offsetw + tw] = lo_merged

        self.latent_debugger.log("hires_lo", int(u * 1000), lo_expanded[0:1])
        self.latent_debugger.log("hires_hi", int(u * 1000), hi_merged[0:1])

        res = torch.concat([lo_expanded, hi_merged])
        return cast(type(hi), res)


class HiresUnetWrapper:
    def __init__(
        self,
        generators: list[torch.Generator],
        target: torch.Size,
        latent_debugger,
    ):
        self.generators = generators
        self.target = target
        self.latent_debugger = latent_debugger

        self.mode: Literal["lo", "hi"] = "lo"

    def get_eps_wrapper(self, unet):
        return HiresUnetEpsWrapper(self, unet, self.target)

    def get_generic_wrapper(self, unet):
        return HiresUnetGenericWrapper(
            self, unet, self.generators, self.target, self.latent_debugger
        )
