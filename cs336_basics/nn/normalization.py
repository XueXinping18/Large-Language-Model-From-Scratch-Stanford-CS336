from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(d_model, device=device, dtype=dtype)
        )
        self.eps = eps
    # Here because of the presence of small epsilon and square operation, it is necessary to upcast to float32
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save original dtype, upcast x to float32
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.rsqrt(torch.mean(x ** 2, dim = -1, keepdim= True) + self.eps)
        result = (x * rms) * self.weight.to(torch.float32)
        return result.to(in_dtype)
