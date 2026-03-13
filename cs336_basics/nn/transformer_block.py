from __future__ import annotations

import torch
import torch.nn as nn
from cs336_basics.nn import RMSNorm, CausalMultiHeadSelfAttention, SwiGLU, RoPE


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block.

    x -> RMSNorm -> MultiHeadAttn -> + residual -> RMSNorm -> SwiGLU FFN -> + residual
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPE | None = None,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of the pre-norm Transformer block.

        Args:
            x: (batch, seq_len, d_model) input tensor.
            rope: Optional RoPE instance for rotary positional embeddings.
            token_positions: (batch, seq_len) absolute token positions, required if rope is provided.

        Returns:
            (batch, seq_len, d_model) output tensor.
        """
        x = x + self.attn(self.ln1(x), rope=rope, token_positions=token_positions)
        x = x + self.ffn(self.ln2(x))
        return x