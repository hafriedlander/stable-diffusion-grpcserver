from typing import Sequence, cast

import easing_functions
import torch

from sdgrpcserver import resize_right
from sdgrpcserver.pipeline.randtools import batched_rand
from sdgrpcserver.pipeline.unet.types import (
    DiffusersSchedulerUNet,
    GenericSchedulerUNet,
    KDiffusionSchedulerUNet,
    PX0Tensor,
    XtTensor,
)

# from sdgrpcserver.pipeline.unet.types import *


def match_shape(latents: torch.Tensor, target: torch.Size):
    # If it's already the right size, just return it
    if latents.shape == target:
        return latents

    # Maybe scale it?
    scale = max(target[2] / latents.shape[2], target[3] / latents.shape[3])
    if scale != 1:
        latents = resize_right.resize(latents, scale_factors=scale, pad_mode="reflect")

    # If we don't need to crop, skip that bit
    if latents.shape == target:
        return latents

    offset2 = (latents.shape[2] - target[2]) // 2
    offset3 = (latents.shape[3] - target[3]) // 2

    return latents[:, :, offset2 : offset2 + target[2], offset3 : offset3 + target[3]]


class HiresUnetWrapper(GenericSchedulerUNet):
    def __init__(
        self,
        unet: DiffusersSchedulerUNet | KDiffusionSchedulerUNet,
        generators: list[torch.Generator],
        target: Sequence[int],
    ):
        self.unet = unet
        self.generators = generators
        self.target = target

        self.floor = 0
        self.start = 0
        self.end = 0.3
        self.easing = easing_functions.QuarticEaseInOut(
            end=1 - self.floor, duration=1 - (self.start + self.end)  # type: ignore
        )

    def __call__(self, latents: XtTensor, __step, u: float) -> PX0Tensor | XtTensor:
        # Linear blend between base and graft
        p = self.floor + self.easing(u - self.start) if u < 1 - self.end else 1

        lo, hi = latents.chunk(2)

        if isinstance(__step, torch.Tensor):
            lo_t, hi_t = __step.chunk(2)
        else:
            lo_t = hi_t = __step

        offset2 = (latents.shape[2] - self.target[2]) // 2
        offset3 = (latents.shape[3] - self.target[3]) // 2

        hi_in = hi
        hi_out = self.unet(hi_in, hi_t, u=u)

        # Early out if we're passed the graft stage
        if p >= 0.999:
            return cast(type(hi_out), torch.concat([lo, hi_out]))

        lo_in = lo[
            :, :, offset2 : offset2 + self.target[2], offset3 : offset3 + self.target[3]
        ]
        lo_out = self.unet(lo_in, lo_t, u=u)

        # Crop hi and merge it back into lo
        hi_crop = hi_out[
            :, :, offset2 : offset2 + self.target[2], offset3 : offset3 + self.target[3]
        ]

        randmap = batched_rand(
            lo_out.shape, self.generators, lo_out.device, lo_out.dtype
        )
        lo_merged = torch.where(randmap >= p, lo_out, hi_crop)

        # Scale lo and merge it back into hi
        lo_scaled = match_shape(lo_out, hi.shape)

        randmap = batched_rand(
            hi_out.shape, self.generators, hi_out.device, hi_out.dtype
        )
        hi_merged = torch.where(randmap >= p, lo_scaled, hi_out)

        # Expand lo back to full tensor size by wrapping with 0
        lo_expanded = torch.zeros_like(hi_merged)
        lo_expanded[
            :, :, offset2 : offset2 + self.target[2], offset3 : offset3 + self.target[3]
        ] = lo_merged

        res = torch.concat([lo_expanded, hi_merged])
        return cast(type(hi_out), res)
