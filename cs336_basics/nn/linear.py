from __future__ import annotations

import torch
import torch.nn as nn
from einops import einsum


class Linear(nn.Module):
    """Linear transformation without bias: y = x @ W^T."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        # variance = 2 / (d_in + d_out), truncated at ±3 sigma
        std = (2.0 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features), weight: (out_features, in_features)
        # Contract over in_features (i), keep batch dims (...) and out_features (j)
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
