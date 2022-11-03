from diffusers import UNet2DConditionModel
from diffusers.models.attention import SpatialTransformer, BasicTransformerBlock

from nonfree.tome_unet import ToMeUNet, ToMeSpatialTransformer, ToMeCrossAttention
from nonfree.tome_memory_efficient_cross_attention import has_xformers, ToMeMemoryEfficientCrossAttention

from sdgrpcserver.pipeline.models.memory_efficient_cross_attention import MemoryEfficientCrossAttention

def apply_tome(
    model: UNet2DConditionModel, trace_source: bool = False, prop_attn: bool = True
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """

    model.__class__ = ToMeUNet
    model.r = 0
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": False,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, SpatialTransformer):
            module.__class__ = ToMeSpatialTransformer
        if isinstance(module, BasicTransformerBlock):
            #module.__class__ = ToMeTransformerBlock
            #module._tome_info = model._tome_info
            if isinstance(module.attn1, MemoryEfficientCrossAttention):
                module.attn1.__class__ = ToMeMemoryEfficientCrossAttention
            else:
                module.attn1.__class__ = ToMeCrossAttention
            module.attn1._tome_info = model._tome_info
