from typing import cast

import torch
import torchvision.transforms as T

from sdgrpcserver import resize_right
from sdgrpcserver.pipeline.easing import Easing
from sdgrpcserver.pipeline.randtools import batched_rand
from sdgrpcserver.pipeline.unet.types import (
    DiffusersSchedulerUNet,
    GenericSchedulerUNet,
    KDiffusionSchedulerUNet,
    PX0Tensor,
    XtTensor,
)


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


class HiresUnetWrapper(GenericSchedulerUNet):
    def __init__(
        self,
        unet_natural: DiffusersSchedulerUNet | KDiffusionSchedulerUNet,
        unet_hires: DiffusersSchedulerUNet | KDiffusionSchedulerUNet,
        generators: list[torch.Generator],
        target: torch.Size,
        latent_debugger,
    ):
        self.unet_natural = unet_natural
        self.unet_hires = unet_hires
        self.generators = generators
        self.target = target

        self.easing = Easing(floor=0, start=0, end=0.4, easing="sine")
        self.latent_debugger = latent_debugger

    def __call__(self, latents: XtTensor, __step, u: float) -> PX0Tensor | XtTensor:
        # Linear blend between base and graft
        p = self.easing.interp(u)

        lo_in, hi_in = latents.chunk(2)

        if isinstance(__step, torch.Tensor) and __step.shape:
            lo_t, hi_t = __step.chunk(2)
        else:
            lo_t = hi_t = __step

        hi = self.unet_hires(hi_in, hi_t, u=u)

        # Early out if we're passed the graft stage
        if p >= 0.999:
            return cast(type(hi), torch.concat([lo_in, hi]))

        *_, h, w = latents.shape
        th, tw = self.target

        offseth = (h - th) // 2
        offsetw = (w - tw) // 2

        lo_in = lo_in[:, :, offseth : offseth + th, offsetw : offsetw + tw]
        lo = self.unet_natural(lo_in, lo_t, u=u)

        # Crop hi and merge it back into lo
        scale = min(tw / w, th / h)

        h_s = int(h * scale)
        w_s = int(w * scale)

        offseth2 = (th - h_s) // 2
        offsetw2 = (tw - w_s) // 2

        image_slice = (
            slice(0, None),
            slice(0, None),
            slice(offseth2, offseth2 + h_s),
            slice(offsetw2, offsetw2 + w_s),
        )

        hi_crop = torch.zeros_like(lo)
        # T.functional.resize(hi, [th, tw], T.InterpolationMode.NEAREST)
        hi_crop[image_slice] = T.functional.resize(
            hi, [h_s, w_s], T.InterpolationMode.NEAREST
        )

        # hi_crop = hi[:, :, offseth : offseth + th, offsetw : offsetw + tw]

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

    @classmethod
    def image_to_natural(cls, natural_size: int, image: torch.Tensor, fill=torch.zeros):
        *_, height, width = image.shape
        scale = min(natural_size / width, natural_size / height)

        height_scaled = int(height * scale)
        width_scaled = int(width * scale)

        offseth = (natural_size - height_scaled) // 2
        offsetw = (natural_size - width_scaled) // 2

        image_slice = (
            slice(0, None),
            slice(0, None),
            slice(offseth, offseth + height_scaled),
            slice(offsetw, offsetw + width_scaled),
        )

        natural_image_size = (*image.shape[:-2], natural_size, natural_size)

        natural_image = fill(natural_image_size, device=image.device, dtype=image.dtype)

        natural_image[image_slice] = resize_right.resize(
            image, scale_factors=scale, pad_mode="reflect"
        )

        return natural_image

    @classmethod
    def merge_initial_latents(cls, left, right):
        left_resized = torch.zeros_like(right)

        *_, th, tw = left.shape
        *_, h, w = right.shape

        offseth = (h - th) // 2
        offsetw = (w - tw) // 2

        left_resized[:, :, offseth : offseth + th, offsetw : offsetw + tw] = left
        return torch.concat([left_resized, right])

    @classmethod
    def split_result(cls, left, right):
        return right.chunk(2)[1]
