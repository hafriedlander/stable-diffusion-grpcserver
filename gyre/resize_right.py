# Redirect to the embedded git submodule

import sys
from typing import TypeVar, cast

import numpy as np
import torch

from gyre.src.ResizeRight import interp_methods

sys.modules["interp_methods"] = interp_methods
from gyre.src.ResizeRight import resize_right  # noqa: E402

T = TypeVar("T", bound=torch.Tensor | np.ndarray)


def resize(
    input: T,
    scale_factors=None,
    out_shape=None,
    interp_method=interp_methods.cubic,
    support_sz=None,
    antialiasing=True,
    by_convs=False,
    scale_tolerance=None,
    max_numerator=10,
    pad_mode="constant",
) -> T:
    result = resize_right.resize(
        input,
        scale_factors=scale_factors,
        out_shape=out_shape,
        interp_method=interp_method,
        support_sz=support_sz,
        antialiasing=antialiasing,
        by_convs=by_convs,
        scale_tolerance=scale_tolerance,
        max_numerator=max_numerator,
        pad_mode=pad_mode,
    )

    if isinstance(result, torch.Tensor):
        result = result.to(cast(torch.Tensor, input).dtype)
    else:
        result = result.astype(cast(np.ndarray, input).dtype)

    return cast(T, result)
