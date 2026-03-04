from __future__ import annotations

import torch
import torch.nn as nn

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
        # TODO: Create three Linear layers: self.w1, self.w2, self.w3
        #       w1: d_model -> d_ff
        #       w2: d_ff -> d_model
        #       w3: d_model -> d_ff
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Compute SwiGLU(x) = w2(silu(w1(x)) * w3(x))
        raise NotImplementedError