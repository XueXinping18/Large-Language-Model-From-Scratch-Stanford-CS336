from __future__ import annotations

import torch
import torch.nn as nn
from einops import einsum, rearrange

class RoPE(nn.Module):
    """Rotary Positional Embedding.

    Precomputes cos and sin values for all positions up to max_seq_len,
    then applies rotary transformation to input queries/keys.

    No learnable parameters — uses register_buffer with persistent=False.
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        assert d_k % 2 == 0, f"d_k must be even, got {d_k}"
        # 1. Compute frequency values: theta_i = theta^(-2i/d_k) for i in [0, d_k/2)
        # 2. Compute position indices: [0, 1, ..., max_seq_len - 1]
        # 3. Compute outer product: positions × frequencies -> (max_seq_len, d_k/2)
        # 4. Precompute cos and sin of those angles
        # 5. Register them as buffers using self.register_buffer("cos_buf", ..., persistent=False)
        #    and self.register_buffer("sdo in_buf", ..., persistent=False)
        dim_pair_freqs = theta ** (-torch.arange(0, d_k, 2, device=device).float() / d_k) # record the index of dimension pairs
        seq_locs = torch.arange(0, max_seq_len, device=device).float() # record the absolute location of each token in this sequence
        # we record angles in the order of [seq, freq] because we calculate for each token the entire spectrum of dimension pairs
        # we want row-wise access which is contiguous in memory
        angles = einsum(seq_locs, dim_pair_freqs, "seq, freq -> seq freq")
        #   RoPE 的 cos/sin 不是可学习参数，不需要梯度。所以不能用
        #   nn.Parameter。但又需要跟着设备移动，所以用 register_buffer
        #   它就是专门给不需要训练但需要跟着模块走的 tensor 设计的
        # Dimension of cos_buf: (max_seq_len, d_k / 2)
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.register_buffer("cos_buf", torch.cos(angles), persistent=False)
        self.register_buffer("sin_buf", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to input tensor.
            We avoid matrix multiplication in favor of element-wise multiplication.

        Args:
            x: (..., seq_len, d_k)
            token_positions: (..., seq_len) — integer positions

        Returns:
            (..., seq_len, d_k)

        Steps:
            1. Use token_positions to index into precomputed cos/sin buffers
            2. Split x into pairs: x_even = x[..., 0::2], x_odd = x[..., 1::2]
            3. Apply rotation:
                   out_even = x_even * cos - x_odd * sin
                   out_odd  = x_even * sin + x_odd * cos
            4. Interleave even/odd back together

        Hint: torch.stack([out_even, out_odd], dim=-1).flatten(-2)
              interleaves the pairs back into shape (..., seq_len, d_k)
        """
        # Step 1: index into precomputed buffers
        # token_positions: (..., seq_len) -> cos/sin: (..., seq_len, d_k/2)
        cos = self.cos_buf[token_positions]
        sin = self.sin_buf[token_positions]

        # Step 2: split into pairs
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # Step 3: rotate:
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        return torch.stack([out_even, out_odd], dim=-1).flatten(-2) # merge the last two dimensions