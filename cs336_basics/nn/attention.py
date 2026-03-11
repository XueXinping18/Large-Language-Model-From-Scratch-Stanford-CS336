from __future__ import annotations

import torch
import torch.nn as nn
from einops import einsum, rearrange

from cs336_basics.nn.activations import softmax
from cs336_basics.nn.linear import Linear
from cs336_basics.nn.rope import RoPE


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled dot-product attention.

    Args:
        Q: (..., seq_len_q, d_k)
        K: (..., seq_len_k, d_k)
        V: (..., seq_len_k, d_v)
        mask: (seq_len_q, seq_len_k) boolean mask. True = keep, False = mask out.

    Returns:
        (..., seq_len_q, d_v)
    """
    inner = K.size(-1) ** (-0.5) * einsum(Q, K, "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k")
    if mask is not None:
        inner = inner.masked_fill(~mask, -torch.inf)
    return einsum(softmax(inner, -1), V, "... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v")


class CausalMultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention with optional RoPE."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPE | None = None,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            rope: Optional RoPE instance.
            token_positions: (batch, seq_len) — required if rope is provided.

        Returns:
            (batch, seq_len, d_model)
        """
        if rope:
            assert rope.d_k == self.d_k
            assert x.size(-2) == token_positions.size(-1)
        q_mono = self.q_proj(x)
        k_mono = self.k_proj(x)
        v_mono = self.v_proj(x)
        q_multihead_batch = rearrange(q_mono, "... seq_len (num_head d_k) -> ... num_head seq_len d_k", num_head=self.num_heads, d_k = self.d_k)
        k_multihead_batch = rearrange(k_mono, "... seq_len (num_head d_k) -> ... num_head seq_len d_k", num_head=self.num_heads, d_k = self.d_k)
        v_multihead_batch = rearrange(v_mono, "... seq_len (num_head d_k) -> ... num_head seq_len d_k", num_head=self.num_heads, d_k = self.d_k)
        if rope is not None:
            # add new dimension to token_positions -- 1 is to match num_head in subsequent broadcasting; this guarantees subsequent broadcasted batching
            q_multihead_batch = rope(q_multihead_batch, rearrange(token_positions, "... seq_len -> ... 1 seq_len"))
            k_multihead_batch = rope(k_multihead_batch, rearrange(token_positions, "... seq_len -> ... 1 seq_len"))
        seq_len = x.size(-2)
        mask = torch.tril(torch.ones(seq_len, seq_len,  dtype=torch.bool, device=x.device))
        attention_multihead = scaled_dot_product_attention(q_multihead_batch, k_multihead_batch, v_multihead_batch, mask)
        attention = rearrange(attention_multihead, "... num_head seq_len d_k -> ... seq_len (num_head d_k)")
        return self.o_proj(attention)