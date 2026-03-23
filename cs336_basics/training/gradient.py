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
    # Using detach to avoid Higher-order Gradients being generated in the computational graph in Pytorch
    # Note that detach does not copy the data but still point to the data while disabling the construction of further computational graph
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return
    norm_stack = torch.stack([torch.norm(grad, 2) for grad in grads])
    total_norm = torch.norm(norm_stack, 2)

    if total_norm > max_l2_norm:
        clip_coeff = (max_l2_norm / (total_norm + 1e-6))
        for grad in grads:
            grad.mul_(clip_coeff) # Or use *=, the in place operations to modify the data the local variable points to