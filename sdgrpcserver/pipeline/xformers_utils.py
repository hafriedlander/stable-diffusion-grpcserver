import functools

import torch
from diffusers.utils.import_utils import is_xformers_available


@functools.cache
def xformers_mea_available():
    available = False

    if is_xformers_available():
        try:
            from xformers.ops import memory_efficient_attention

            # Make sure we can run the memory efficient attention
            _ = memory_efficient_attention(
                torch.randn((1, 2, 40), device="cuda"),
                torch.randn((1, 2, 40), device="cuda"),
                torch.randn((1, 2, 40), device="cuda"),
            )
        except Exception:
            pass
        else:
            available = True

    return available
