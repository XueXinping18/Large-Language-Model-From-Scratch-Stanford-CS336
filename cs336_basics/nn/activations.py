from __future__ import annotations

import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Numerically stable softmax.

    Args:
        x: Input tensor of arbitrary shape.
        dim: Dimension along which to apply softmax.

    Returns:
        Tensor of same shape with softmax applied along dim.

    Steps:
    """
    max = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)