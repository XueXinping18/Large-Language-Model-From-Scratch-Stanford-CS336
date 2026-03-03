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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Save original dtype, upcast x to float32
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # 2. Compute RMS: sqrt(mean(x^2) + eps) over the last dimension
        rms = torch.sqrt(torch.mean(x ** 2, dim = -1, keepdim= True) + self.eps)
        # 3. Normalize: x / RMS
        norm = x / rms
        # 4. Apply learnable multiplication
        result = norm * self.weight
        # 5. Downcast back to original dtype
        return result.to(in_dtype)
