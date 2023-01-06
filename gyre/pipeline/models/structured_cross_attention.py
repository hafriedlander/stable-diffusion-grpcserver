# Mostly from https://github.com/shunk031/training-free-structured-diffusion-guidance
# 
# Changes:
#   - _attention changed to _sliced_attention to match Diffusers new(?) argument structure


from typing import Optional, Tuple

import torch as th
from diffusers.models.attention import CrossAttention

from sdgrpcserver.pipeline.text_embedding.structured_text_embedding import KeyValueTensors

from einops.layers.torch import Reduce

class StructuredCrossAttention(CrossAttention):
    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: int = 0,
        struct_attention: bool = False,
    ) -> None:
        super().__init__(query_dim, context_dim, heads, dim_head, dropout)
        self.struct_attention = struct_attention

        self.max_pooling_layer = Reduce(f"b c h w -> 1 c h w", 'max')


    def struct_qkv(
        self,
        q: th.Tensor,
        context: Tuple[th.Tensor, KeyValueTensors],
        mask: Optional[th.Tensor] = None,
    ) -> th.Tensor:

        assert len(context) == 2 and isinstance(context, tuple)
        uc_context = context[0]
        context_k = context[1].k
        context_v = context[1].v

        if isinstance(context_k, list) and isinstance(context_v, list):
            return self.multi_qkv(
                q=q,
                uc_context=uc_context,
                context_k=context_k,
                context_v=context_v,
                mask=mask,
            )
        elif isinstance(context_k, th.Tensor) and isinstance(context_v, th.Tensor):
            return self.heterogenous_qkv(
                q=q,
                uc_context=uc_context,
                context_k=context_k,
                context_v=context_v,
                mask=mask,
            )
        else:
            raise NotImplementedError

    def multi_qkv(
        self,
        q: th.Tensor,
        uc_context: th.Tensor,
        context_k: th.Tensor,
        context_v: th.Tensor,
        mask: Optional[th.Tensor] = None,
    ) -> None:
        h = self.heads
        assert uc_context.size(0) == context_k[0].size(0) == context_v[0].size(0)
        true_bs = uc_context.size(0)*h

        k_uc = self.to_k(uc_context)
        v_uc = self.to_v(uc_context)

        k_c = [self.to_k(c_k) for c_k in context_k]
        v_c = [self.to_v(c_v) for c_v in context_v]
                
        q = self.reshape_heads_to_batch_dim(q)
        k_uc = self.reshape_heads_to_batch_dim(k_uc)
        v_uc = self.reshape_heads_to_batch_dim(v_uc)

        k_c = [self.reshape_heads_to_batch_dim(k) for k in k_c]
        v_c = [self.reshape_heads_to_batch_dim(v) for v in v_c]

        q_uc = q[:true_bs]
        q_c = q[true_bs:]

        sim_uc = th.matmul(q_uc, k_uc.transpose(-1, -2)) * self.scale
        sim_c = [th.matmul(q_c, k.transpose(-1, -2)) * self.scale for k in k_c]

        attn_uc = sim_uc.softmax(dim=-1)
        attn_c = [sim.softmax(dim=-1) for sim in sim_c]

        out_uc = th.matmul(attn_uc, v_uc)
        out_c = [th.matmul(attn, v) for attn, v in zip(attn_c, v_c)]

        out_c = sum(out_c) / len(v_c)

        out = th.cat([out_uc, out_c])

        return self.reshape_batch_dim_to_heads(out)
        
    def normal_qkv(
        self,
        q: th.Tensor,
        context: th.Tensor,
        mask: Optional[th.Tensor] = None,
    ) -> th.Tensor:

        batch_size, sequence_length, dim = q.shape

        k = self.to_k(context)
        v = self.to_v(context)

        q = self.reshape_heads_to_batch_dim(q)
        k = self.reshape_heads_to_batch_dim(k)
        v = self.reshape_heads_to_batch_dim(v)

        hidden_states = self._sliced_attention(q, k, v, sequence_length, dim)

        return hidden_states

    def heterogenous_qkv(
        self,
        q: th.Tensor,
        uc_context: th.Tensor,
        context_k: th.Tensor,
        context_v: th.Tensor,
        mask: Optional[th.Tensor] = None,
    ) -> th.Tensor:

        batch_size, sequence_length, dim = q.shape

        k = self.to_k(th.cat((uc_context, context_k), dim=0))
        v = self.to_v(th.cat((uc_context, context_v), dim=0))

        q = self.reshape_heads_to_batch_dim(q)
        k = self.reshape_heads_to_batch_dim(k)
        v = self.reshape_heads_to_batch_dim(v)

        hidden_states = self._sliced_attention(q, k, v, sequence_length, dim)

        return hidden_states

    def get_kv(self, context: th.Tensor) -> KeyValueTensors:
        return KeyValueTensors(k=self.to_k(context), v=self.to_v(context))

    def forward(
        self,
        x: th.Tensor,
        context: Optional[Tuple[th.Tensor, KeyValueTensors]] = None,
        mask: Optional[th.Tensor] = None,
    ) -> th.Tensor:

        q = self.to_q(x)

        if isinstance(context, tuple):
            assert len(context) == 2
            assert isinstance(context[0], th.Tensor)  # unconditioned embedding
            assert isinstance(context[1], KeyValueTensors)  # conditioned embedding

            if self.struct_attention:
                out = self.struct_qkv(q=q, context=context, mask=mask)
            else:
                uc_context = context[0]
                c_full_seq = context[1].k[0].unsqueeze(dim=0)
                print("n", c_full_seq.shape)
                out = self.normal_qkv(
                    q=q, context=th.cat((uc_context, c_full_seq), dim=0), mask=mask
                )
        else:
            ctx = context if context is not None else x
            out = self.normal_qkv(q=q, context=ctx, mask=mask)

        return self.to_out(out)
