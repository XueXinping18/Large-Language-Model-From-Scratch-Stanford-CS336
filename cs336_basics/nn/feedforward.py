from __future__ import annotations

import torch
import torch.nn as nn
from einops import einsum
from cs336_basics.nn.linear import Linear


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU activation: x * sigmoid(x).

    Use torch.sigmoid for numerical stability.
    """
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network.

    SwiGLU(x) = W2 @ (SiLU(W1 @ x) * (W3 @ x))

    Three linear projections (no bias):
        w1: d_model -> d_ff   (gate projection, fed through SiLU)
        w3: d_model -> d_ff   (up projection)
        w2: d_ff -> d_model   (down projection)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))