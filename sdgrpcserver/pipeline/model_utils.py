from copy import deepcopy
from typing import Literal

import torch
from accelerate.hooks import ModelHook, add_hook_to_module
from accelerate.utils import send_to_device, set_module_tensor_to_device


class CloneToGPUHook(ModelHook):
    def __init__(self, execution_device, exclusion_set, top, params, buffers):
        self.execution_device = execution_device
        self.exclusion_set = exclusion_set
        self.top = top
        self.params = params
        self.buffers = buffers

    def pre_forward(self, module, *args, **kwargs):
        if self.exclusion_set:
            self.exclusion_set.activate(self.top)

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

    def reset(self, model):
        for name in self.params.keys():
            set_module_tensor_to_device(model, name, "meta")
        for name in self.buffers.keys():
            set_module_tensor_to_device(model, name, "meta")


class GPUExclusionSet:
    def __init__(self, max_activated=-1):
        self.sets = []
        self.activated = []
        self.max_activated = max_activated

    def add(self, top):
        models = [
            model
            for _, model in top.named_modules()
            if hasattr(model, "_hf_hook") and isinstance(model._hf_hook, CloneToGPUHook)
        ]

        self.sets.append((top, models))

    def reset(self, exclude=[]):
        exclude = list(exclude)

        for top, models in self.sets:
            if top in exclude:
                continue

            for model in models:
                model._hf_hook.reset(model)

    def activate(self, top):
        # No-op if top is already the most recently activated
        if self.activated and self.activated[0] is top:
            return

        # Update the LRU activated queue
        self.activated = [model for model in self.activated if model is not top]
        self.activated.insert(0, top)
        self.activated = self.activated[: self.max_activated]

        self.reset(exclude=self.activated)


def clone_model(
    model,
    clone_tensors: Literal["share"] | str | torch.device = "share",
    exclusion_set=None,
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
        if exclusion_set:
            exclusion_set.add(clone)

        for (model_name, dest) in clone.named_modules():
            model_params, model_buffers = cache[model_name]

            if exclusion_set:
                for name in model_params.keys():
                    set_module_tensor_to_device(dest, name, "meta")
                for name in model_buffers.keys():
                    set_module_tensor_to_device(dest, name, "meta")

                add_hook_to_module(
                    dest,
                    CloneToGPUHook(
                        clone_tensors, exclusion_set, clone, model_params, model_buffers
                    ),
                )
            else:
                for name, param in model_params.items():
                    new_param = param.to(clone_tensors, copy=True)
                    set_module_tensor_to_device(dest, name, clone_tensors, new_param)
                for name, buffer in model_buffers.items():
                    new_buffer = buffer.to(clone_tensors, copy=True)
                    set_module_tensor_to_device(dest, name, clone_tensors, new_buffer)

    return clone
