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

# Indexes into a shape for the height and width dimensions
# Negative indexed to work for any number of dimensions
Hi, Wi = -2, -1


def pad_like(latents, like, mode="replicate"):
    wd = like.shape[Wi] - latents.shape[Wi]
    hd = like.shape[Hi] - latents.shape[Hi]
    l = wd // 2
    r = wd - l
    t = hd // 2
    b = hd - t

    pad = torch.nn.functional.pad

    if isinstance(mode, int | float):
        return pad(latents, pad=(l, r, t, b), mode="constant", value=mode)
    else:
        return pad(latents, pad=(l, r, t, b), mode=mode)


def resize_nearest(latents, scale_factor=1):
    hs = int(latents.shape[Hi] * scale_factor)
    ws = int(latents.shape[Wi] * scale_factor)

    return T.functional.resize(latents, [hs, ws], T.InterpolationMode.NEAREST)


def scale_into(latents, target, scale):
    if scale >= 1:
        # latents = resize_right.resize(latents, scale_factors=scale, pad_mode="reflect")
        latents = resize_nearest(latents, scale)
    else:
        latents = resize_nearest(latents, scale)

    # Now crop off anything that's outside target shape, and offset if it's inside target shape

    # Positive is offset into the shape, negative is crop amount
    offh = (target.shape[Hi] - latents.shape[Hi]) // 2
    offw = (target.shape[Wi] - latents.shape[Wi]) // 2

    if offh < 0:
        latents = latents[:, :, -offh : -offh + target.shape[Hi], :]
        offh = 0

    if offw < 0:
        latents = latents[:, :, :, -offw : -offw + target.shape[Wi]]
        offw = 0

    target[
        :, :, offh : offh + latents.shape[Hi], offw : offw + latents.shape[Wi]
    ] = latents
    return target


def downscale_into(latents, target, oos_fraction):
    scale_min = min(
        target.shape[Hi] / latents.shape[Hi], target.shape[Wi] / latents.shape[Wi]
    )
    scale_max = max(
        target.shape[Hi] / latents.shape[Hi], target.shape[Wi] / latents.shape[Wi]
    )

    # At oos_fraction == 1, we want to downscale to completely contain the latent within
    # the square target - i.e. scale_min. At oos_fraction == 0 we want to downscale to
    # completely cover the square target - i.e. scale_max

    scale = scale_min * oos_fraction + scale_max * (1 - oos_fraction)
    return scale_into(latents, target, scale)


def upscale_into(latents, target, oos_fraction):
    scale_min = min(
        target.shape[Hi] / latents.shape[Hi], target.shape[Wi] / latents.shape[Wi]
    )
    scale_max = max(
        target.shape[Hi] / latents.shape[Hi], target.shape[Wi] / latents.shape[Wi]
    )

    # At oos_fraction == 1, we want to upscale to completely cover the
    # target - i.e. scale_max. At oos_fraction = 0 we want to completely
    # fit square latent into OOS targe, i.e. scale_min

    scale = scale_max * oos_fraction + scale_min * (1 - oos_fraction)
    return scale_into(latents, target, scale)


class HiresUnetWrapper(GenericSchedulerUNet):
    def __init__(
        self,
        unet_natural: DiffusersSchedulerUNet | KDiffusionSchedulerUNet,
        unet_hires: DiffusersSchedulerUNet | KDiffusionSchedulerUNet,
        generators: list[torch.Generator],
        natural_size: torch.Size,
        oos_fraction: float,
        latent_debugger,
    ):
        self.unet_natural = unet_natural
        self.unet_hires = unet_hires
        self.generators = generators
        self.natural_size = natural_size
        self.oos_fraction = oos_fraction

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
        th, tw = self.natural_size

        offseth = (h - th) // 2
        offsetw = (w - tw) // 2

        lo_in = lo_in[:, :, offseth : offseth + th, offsetw : offsetw + tw]
        lo = self.unet_natural(lo_in, lo_t, u=u)

        # Downscale hi and merge into lo
        hi_downscaled = torch.zeros_like(lo)  # Un-overlapped space is zero
        hi_downscaled = downscale_into(hi, hi_downscaled, self.oos_fraction)

        randmap = batched_rand(lo.shape, self.generators, lo.device, lo.dtype)
        lo_merged = torch.where(randmap >= p, lo, hi_downscaled)

        # Upscale lo and merge it back into hi
        lo_upscaled = hi.clone()  # Un-overlapped space copied from hi
        lo_upscaled = upscale_into(lo, lo_upscaled, self.oos_fraction)

        randmap = batched_rand(hi.shape, self.generators, hi.device, hi.dtype)
        hi_merged = torch.where(randmap >= p, lo_upscaled, hi)

        # Expand lo back to full tensor size by wrapping with 0
        lo_expanded = torch.zeros_like(hi_merged)
        lo_expanded[:, :, offseth : offseth + th, offsetw : offsetw + tw] = lo_merged

        self.latent_debugger.log("hires_lo", int(u * 1000), lo_expanded[0:1])
        self.latent_debugger.log("hires_hi", int(u * 1000), hi_merged[0:1])

        res = torch.concat([lo_expanded, hi_merged])
        return cast(type(hi), res)

    @classmethod
    def image_to_natural(
        cls,
        natural_size: int,
        image: torch.Tensor,
        oos_fraction: float,
        fill=torch.zeros,
    ):
        natural_image_size = (*image.shape[:-2], natural_size, natural_size)
        natural_image = fill(natural_image_size, device=image.device, dtype=image.dtype)

        downscale_into(image, natural_image, oos_fraction)
        return natural_image

    @classmethod
    def merge_initial_latents(cls, left, right):
        left_resized = torch.zeros_like(right)

        *_, th, tw = left.shape
        *_, h, w = right.shape

        offseth = (h - th) // 2
        offsetw = (w - tw) // 2

        left_resized[:, :, offseth : offseth + th, offsetw : offsetw + tw] = left
        right[:, :, offseth : offseth + th, offsetw : offsetw + tw] = left
        return torch.concat([left_resized, right])

    @classmethod
    def split_result(cls, left, right):
        return right.chunk(2)[1]
