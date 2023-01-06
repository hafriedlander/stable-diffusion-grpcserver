from typing import cast

import torch

from gyre.pipeline.easing import Easing
from gyre.pipeline.randtools import batched_rand
from gyre.pipeline.unet.types import (
    DiffusersSchedulerUNet,
    GenericSchedulerUNet,
    KDiffusionSchedulerUNet,
    PX0Tensor,
    XtTensor,
)


class GraftUnets(GenericSchedulerUNet):
    def __init__(
        self,
        unet_root: DiffusersSchedulerUNet | KDiffusionSchedulerUNet,
        unet_top: DiffusersSchedulerUNet | KDiffusionSchedulerUNet,
        generators: list[torch.Generator],
    ):
        self.unet_root = unet_root
        self.unet_top = unet_top
        self.generators = generators

        self.easing = Easing(floor=0, start=0.1, end=0.3, easing="sine")

    def __call__(self, latents: XtTensor, __step, u: float) -> PX0Tensor | XtTensor:
        p = self.easing.interp(u)

        if p <= 0:
            return self.unet_root(latents, __step, u=u)
        elif p >= 1:
            return self.unet_top(latents, __step, u=u)

        root = self.unet_root(latents, __step, u=u)
        top = self.unet_top(latents, __step, u=u)

        # Build a map of 0..1 like latents
        randmap = batched_rand(top.shape, self.generators, top.device, top.dtype)

        # Linear blend between base and graft
        res = cast(type(top), torch.where(randmap >= p, root, top))

        return res

    @classmethod
    def merge_initial_latents(cls, left, right):
        return left

    @classmethod
    def split_result(cls, left, right):
        return right
