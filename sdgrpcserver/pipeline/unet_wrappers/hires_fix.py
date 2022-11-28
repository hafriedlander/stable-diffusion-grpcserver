from typing import List, cast, Sequence

import torch
import easing_functions

from sdgrpcserver import resize_right

from sdgrpcserver.pipeline.randtools import *
from sdgrpcserver.pipeline.unet_wrappers.types import *

def match_shape(latents: torch.Tensor, target: List[int]):
    # If it's already the right size, just return it
    if latents.shape == target: return latents

    # Maybe scale it?
    scale = max(target[2]/latents.shape[2], target[3]/latents.shape[3])
    if scale != 1:
        latents = resize_right.resize(latents, scale_factors=scale, pad_mode='reflect').to(latents.dtype) # type: ignore

    # If we don't need to crop, skip that bit
    if latents.shape == target: return latents

    offset2 = (latents.shape[2] - target[2]) // 2
    offset3 = (latents.shape[3] - target[3]) // 2

    return latents[:, :, offset2:offset2+target[2], offset3:offset3+target[3]]
    

def hires_unet_wrapper(
    unet: GenericSchedulerUNet, 
    target: Sequence[int], 
    generators: List[torch.Generator]
    ) -> GenericSchedulerUNet:

    floor=0
    start=0
    end=0.3
    easing=easing_functions.QuarticEaseInOut(end=1-floor, duration=1-(start+end)) # type: ignore

    def wrapper(latents, t_or_sigma, u) -> Union[PX0Tensor, XtTensor]:
        # Linear blend between base and graft
        p = floor + easing(u-start) if u < 1-end else 1

        lo, hi = latents.chunk(2)
        
        if isinstance(t_or_sigma, torch.Tensor):
            lo_t, hi_t = t_or_sigma.chunk(2)
        else:
            lo_t = hi_t = t_or_sigma

        offset2 = (latents.shape[2] - target[2]) // 2
        offset3 = (latents.shape[3] - target[3]) // 2

        hi_in = hi
        hi_out = unet(hi_in, hi_t, u=u)

        # Early out if we're passed the graft stage
        if p >= 0.999: return cast(type(hi_out), torch.concat([lo, hi_out]))

        lo_in = lo[:, :, offset2:offset2+target[2], offset3:offset3+target[3]]
        lo_out = unet(lo_in, lo_t, u=u)

        # Crop hi and merge it back into lo
        hi_crop = hi_out[:, :, offset2:offset2+target[2], offset3:offset3+target[3]]

        randmap = batched_rand(lo_out.shape, generators, lo_out.device, lo_out.dtype)
        lo_merged = torch.where(randmap >= p, lo_out, hi_crop)

        # Scale lo and merge it back into hi
        lo_scaled = match_shape(lo_out, hi.shape)

        randmap = batched_rand(hi_out.shape, generators, hi_out.device, hi_out.dtype)
        hi_merged = torch.where(randmap >= p, lo_scaled, hi_out)

        # Expand lo back to full tensor size by wrapping with 0
        lo_expanded = torch.zeros_like(hi_merged)
        lo_expanded[:, :, offset2:offset2+target[2], offset3:offset3+target[3]] = lo_merged

        res = torch.concat([lo_expanded, hi_merged])
        return cast(type(hi_out), res)

    return wrapper
