from __future__ import annotations

import torch
import torch.nn as nn

class Embedding(nn.Module):
    """Embedding lookup table: maps token IDs to dense vectors.

    Stores a weight matrix of shape (vocab_size, embedding_dim).
    Forward pass indexes into rows by token ID — no matrix multiplication.
    """

    def __init__(
            self,
            vocab_size: int, # Number of embeddings
            embedding_dim: int, # d_model
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(vocab_size, embedding_dim, device=device, dtype=dtype)
        )
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]