from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random input-label pairs for language modeling.

    Args:
        dataset: 1D numpy array of integer token IDs.
        batch_size: Number of sequences to sample.
        context_length: Length of each sequence.
        device: PyTorch device string.

    Returns:
        (x, y) where both are LongTensors of shape (batch_size, context_length).
        x[i] is a random contiguous slice from dataset.
        y[i] is x[i] shifted by one position (the next-token prediction target).
    """
    indices = torch.randint(len(dataset) - context_length - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(dataset[i : i + context_length]) for i in indices])
    y = torch.stack([torch.from_numpy(dataset[i + 1: i + 1 + context_length]) for i in indices])
    return x.to(device), y.to(device)