from typing import cast

import torch

from sdgrpcserver.pipeline.easing import Easing
from sdgrpcserver.pipeline.randtools import batched_rand
from sdgrpcserver.pipeline.unet.types import (
    DiffusersSchedulerUNet,
    GenericSchedulerUNet,
    KDiffusionSchedulerUNet,
    PX0Tensor,
    XtTensor,
)


class GraftUnets(GenericSchedulerUNet):
    def __init__(
        self,
        unets: list[DiffusersSchedulerUNet] | list[KDiffusionSchedulerUNet],
        generators: list[torch.Generator],
        graft_factor: float,
    ):
        self.unets = unets
        self.generators = generators
        self.graft_factor = graft_factor

        self.easing = Easing(floor=0, start=0.1, end=0.3, easing="sine")

    def __call__(self, latents: XtTensor, __step, u: float) -> PX0Tensor | XtTensor:
        p = self.easing.interp(u)

        if p <= 0:
            return self.unets[0](latents, __step, u=u)
        elif p >= 1:
            return self.unets[1](latents, __step, u=u)

        outputs = [unet(latents, __step, u=u) for unet in self.unets]

        # Build a map of 0..1 like latents
        randmap = batched_rand(
            outputs[-1].shape, self.generators, outputs[-1].device, outputs[-1].dtype
        )

        # Linear blend between base and graft
        res = cast(type(outputs[0]), torch.where(randmap >= p, outputs[0], outputs[1]))

        return res
