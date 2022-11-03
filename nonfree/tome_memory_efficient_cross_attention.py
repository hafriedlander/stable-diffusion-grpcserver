
# A merge of the CrossAttention blocks from ToMe and MemoryEfficientCrossAttention
# (C) Hamish Friedlander 2022, All Rights Reserved. Distributable under the same license as ToMe

from typing import Tuple, Union

import torch
from diffusers.models.attention import CrossAttention
from tome.merge import bipartite_soft_matching, merge_source, merge_wavg
from tome.utils import parse_r

try:
    import xformers
    import xformers.ops
except:
    xformers = None

def has_xformers():
    return xformers is not None

class ToMeMemoryEfficientCrossAttention(CrossAttention):
    def forward(self, hidden_states, context=None, mask=None):

        # This bit from ToMe

        batch_size, sequence_length, _ = hidden_states.shape

        q = self.to_q(hidden_states)
        context = context if context is not None else hidden_states
        k = self.to_k(context)
        v = self.to_v(context)
        dim = q.shape[-1]
        r = self._tome_info["r"].pop(0)
        if r > 0:
            # Apply ToMe here
            merge, _ = bipartite_soft_matching(
                k,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, k, self._tome_info["source"]
                )
                self._tome_info["source"] = merge_source(
                    merge, v, self._tome_info["source"]
                )
            k, self._tome_info["size"] = merge_wavg(merge, k)
            v, self._tome_info["size"] = merge_wavg(merge, v)

        # This bit from MemoryEfficientCrossAttention

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=None)

        # TODO: Use this directly in the attention operation, as a bias
        if mask is not None:
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)
