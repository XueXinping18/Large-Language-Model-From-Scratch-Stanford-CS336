from __future__ import annotations

from collections.abc import Iterable

import torch


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    """Clip gradients by global L2 norm.

    If the combined L2 norm of all parameter gradients exceeds max_l2_norm,
    scale all gradients down so the total norm equals max_l2_norm.

    Modifies parameter.grad in-place.

    Args:
        parameters: Iterable of parameters whose gradients to clip.
        max_l2_norm: Maximum allowed L2 norm.

    Steps:
        1. Compute the global L2 norm across all parameter gradients
        2. If norm > max_l2_norm, scale each gradient by (max_l2_norm / norm)
    """
    # TODO: implement
    raise NotImplementedError