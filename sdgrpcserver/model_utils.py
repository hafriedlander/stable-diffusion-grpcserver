from copy import deepcopy
from typing import Literal

import torch
from accelerate.utils import set_module_tensor_to_device


def clone_model(model, clone_tensors: Literal["share"] | str | torch.device = "share"):
    """
    Copies a model so you get a different set of instances, but they share
    all their parameters and buffers
    """

    # If this isn't actually a model, just return a deepcopy
    if not isinstance(model, torch.nn.Module):
        clone = deepcopy(model)
        if clone_tensors != "share":
            clone = clone.to(clone_tensors)
        return clone

    # Start by pulling all the Tensors out of the model, so they're not copied on deepclone
    cache = {}

    for (model_name, source) in model.named_modules():
        model_params = {}
        model_buffers = {}

        for name, param in source.named_parameters(recurse=False):
            model_params[name] = param
            source._parameters[name] = None

        for name, buffer in source.named_buffers(recurse=False):
            model_buffers[name] = buffer
            source._buffers[name] = None

        cache[model_name] = (model_params, model_buffers)

    # Deep clone the model
    clone = deepcopy(model)

    # Put the tensors back into the model
    for (model_name, dest) in model.named_modules():
        model_params, model_buffers = cache[model_name]

        for name, param in model_params.items():
            dest._parameters[name] = param
        for name, buffer in model_buffers.items():
            dest._buffers[name] = buffer

    # And into the clone
    for (model_name, dest) in clone.named_modules():
        model_params, model_buffers = cache[model_name]

        for name, param in model_params.items():
            # Even if we're not sharing, set it to shared to start with
            dest.register_parameter(name, param)

            if clone_tensors != "share":
                # explicitly copy, as set_module_tensor_to_device won't create
                # a copy if the device is already correct
                set_module_tensor_to_device(
                    dest, name, clone_tensors, param.to(clone_tensors, copy=True)
                )

        for name, buffer in model_buffers.items():
            dest.register_buffer(name, buffer)

            if clone_tensors != "share":
                set_module_tensor_to_device(
                    dest, name, clone_tensors, buffer.to(clone_tensors, copy=True)
                )

    return clone
