from copy import deepcopy
from typing import Literal

import torch
from accelerate.hooks import ModelHook, add_hook_to_module
from accelerate.utils import send_to_device, set_module_tensor_to_device


class CloneToGPUHook(ModelHook):
    def __init__(self, execution_device, params, buffers):
        self.execution_device = execution_device
        self.params = params
        self.buffers = buffers

    def pre_forward(self, module, *args, **kwargs):
        dev = self.execution_device

        for name, param in module.named_parameters(recurse=False):
            if param.device == torch.device("meta"):
                # explicitly copy, as set_module_tensor_to_device won't create
                # a copy if the device is already correct
                new_param = self.params[name].to(dev, copy=True)
                set_module_tensor_to_device(module, name, dev, new_param)

        for name, buffer in module.named_buffers(recurse=False):
            if buffer.device == torch.device("meta"):
                new_buffer = self.buffers[name].to(dev, copy=True)
                set_module_tensor_to_device(module, name, dev, new_buffer)

        return (
            send_to_device(args, dev),
            send_to_device(kwargs, dev),
        )


def clone_model(
    model, clone_tensors: Literal["share"] | str | torch.device = "share", delayed=False
):
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
    # Even if we're not sharing, set it to shared to start with
    for (model_name, dest) in clone.named_modules():
        model_params, model_buffers = cache[model_name]

        for name, param in model_params.items():
            dest.register_parameter(name, param)
        for name, buffer in model_buffers.items():
            dest.register_buffer(name, buffer)

    if clone_tensors != "share":
        for (model_name, dest) in clone.named_modules():
            model_params, model_buffers = cache[model_name]

            if delayed:
                for name in model_params.keys():
                    set_module_tensor_to_device(dest, name, "meta")
                for name in model_buffers.keys():
                    set_module_tensor_to_device(dest, name, "meta")

                add_hook_to_module(
                    dest, CloneToGPUHook(clone_tensors, model_params, model_buffers)
                )
            else:
                for name, param in model_params.items():
                    new_param = param.to(clone_tensors, copy=True)
                    set_module_tensor_to_device(dest, name, clone_tensors, new_param)
                for name, buffer in model_buffers.items():
                    new_buffer = buffer.to(clone_tensors, copy=True)
                    set_module_tensor_to_device(dest, name, clone_tensors, new_buffer)

    return clone


def clone_model_hook_reset(top):
    for _, model in top.named_modules():
        if hasattr(model, "_hf_hook") and isinstance(model._hf_hook, CloneToGPUHook):
            for name in model._hf_hook.params.keys():
                set_module_tensor_to_device(model, name, "meta")
            for name in model._hf_hook.buffers.keys():
                set_module_tensor_to_device(model, name, "meta")
