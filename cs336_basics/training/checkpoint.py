from __future__ import annotations

import os
from typing import IO, BinaryIO

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """Serialize model, optimizer, and iteration to disk.

    Args:
        model: Model to save.
        optimizer: Optimizer to save.
        iteration: Current training iteration.
        out: Path or file-like object to write to.
    """
    state_to_save = {"it" : iteration, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}
    torch.save(state_to_save, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Restore model and optimizer state from a checkpoint.

    Args:
        src: Path or file-like object to read from.
        model: Model to restore state into.
        optimizer: Optimizer to restore state into.

    Returns:
        The iteration number saved in the checkpoint.
    """
    checkpoint_state = torch.load(src)
    model.load_state_dict(checkpoint_state["model_state"])
    optimizer.load_state_dict(checkpoint_state["optimizer_state"])
    return checkpoint_state["it"]