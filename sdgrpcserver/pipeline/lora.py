from torch.nn import Linear, ModuleList
from torch.nn.parameter import Parameter

from sdgrpcserver.src.lora.lora_diffusion.lora import (
    LoraInjectedLinear,
    monkeypatch_lora,
    tune_lora_scale,
)


def has_lora(model):
    for module in model.modules():
        if isinstance(module, LoraInjectedLinear):
            return True
    return false


def _find_linears(
    model,
    target_replace_module={"CrossAttention", "Attention"},
    linear_classes=Linear | LoraInjectedLinear,
):
    # Get the targets we should replace all linears under
    targets = (
        module
        for module in model.modules()
        if module.__class__.__name__ in target_replace_module
    )

    # For each target find every linear_class module that isn't a child of a LoraInjectedLinear
    for target in targets:
        for fullname, module in target.named_modules():
            if isinstance(module, linear_classes):
                # Find the direct parent if this is a descendant, not a child, of target
                *path, name = fullname.rsplit(".", 1)
                parent = target.get_submodule(path[0]) if path else target
                # Skip this linear if it's a child of a LoraInjectedLinear
                if isinstance(parent, LoraInjectedLinear):
                    continue
                # Otherwise, yield it
                yield parent, name, module


def addorreplace_lora(
    model, loras, target_replace_module={"CrossAttention", "Attention"}, r: int = 4
):
    for parent, name, module in _find_linears(model, target_replace_module):
        source = module.linear if isinstance(module, LoraInjectedLinear) else module
        weight, bias = source.weight, source.bias

        _tmp = LoraInjectedLinear(
            source.in_features, source.out_features, source.bias is not None, r=r
        )

        _tmp.linear.weight = weight
        if bias is not None:
            _tmp.linear.bias = bias

        # switch the module
        parent._modules[name] = _tmp

        try:
            up_weight = loras.pop(0)
            down_weight = loras.pop(0)
        except IndexError:
            # If ran out of parameters, that means the lora doesn't match the model
            raise RuntimeError(
                f"Lora doesn't match this {model.__class__} - was it trained for it?"
            )

        parent._modules[name].lora_up.weight = Parameter(up_weight.type(weight.dtype))
        parent._modules[name].lora_down.weight = Parameter(
            down_weight.type(weight.dtype)
        )

        parent._modules[name].to(weight.device)

    # If we still have parameters left, that means the lora doesn't match the model
    if loras:
        raise RuntimeError(
            f"Loras model doesn't match this {model.__class__} - was it trained for it?"
        )


def remove_lora(model, target_replace_module=["CrossAttention", "Attention"]):
    for parent, name, module in _find_linears(
        model, target_replace_module, LoraInjectedLinear
    ):
        source = module.linear if isinstance(module, LoraInjectedLinear) else module
        weight, bias = source.weight, source.bias

        _tmp = Linear(source.in_features, source.out_features, bias is not None)

        _tmp.weight = weight
        if bias is not None:
            _tmp.bias = bias

        parent._modules[name] = _tmp
