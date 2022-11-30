from typing import cast

import easing_functions
import torch

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

        self.floor = 0.1
        self.start = 0.0
        self.end = 0.2
        self.easing = easing_functions.QuarticEaseInOut(
            end=1 - self.floor, duration=1 - (self.start + self.end)  # type: ignore - easing_functions takes floats just fine
        )

    def __call__(self, latents: XtTensor, __step, u: float) -> PX0Tensor | XtTensor:
        if self.floor == 0 and u < self.start:
            return self.unets[0](latents, __step, u=u)
        elif u > 1 - self.end:
            return self.unets[1](latents, __step, u=u)

        outputs = [unet(latents, __step, u=u) for unet in self.unets]

        # Build a map of 0..1 like latents
        randmap = batched_rand(
            outputs[-1].shape, self.generators, outputs[-1].device, outputs[-1].dtype
        )

        # Linear blend between base and graft
        p = self.floor + self.easing(u - self.start)
        res = cast(type(outputs[0]), torch.where(randmap >= p, outputs[0], outputs[1]))

        return res
