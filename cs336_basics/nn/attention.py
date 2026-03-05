from __future__ import annotations

import torch
from einops import einsum
from cs336_basics.nn.activations import softmax


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
        inner[~mask] = - torch.inf
    return einsum(softmax(inner, -1), V, "... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v")