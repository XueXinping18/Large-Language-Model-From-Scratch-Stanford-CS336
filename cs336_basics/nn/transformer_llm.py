from __future__ import annotations

import torch
import torch.nn as nn
from cs336_basics.nn import Embedding, RMSNorm, Linear, TransformerBlock, RoPE

class TransformerLLM(nn.Module):
    """Transformer-based language model.

    Architecture: token embedding -> N transformer blocks -> RMSNorm -> linear head.
    Uses RoPE for positional encoding (shared across all layers).
    Returns raw logits (unnormalized) — no softmax applied.
    """

    def __init__(
        self,
        vocal_size: int,
        context_length: int, # used in RoPE
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocal_size
        self.max_context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.embeddings = Embedding(vocal_size, d_model, device=device, dtype=dtype)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, device=device, dtype=dtype) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocal_size, device=device, dtype=dtype)
        self.rope = RoPE(rope_theta, d_model // num_heads, context_length, device=device)

    def forward(self, in_token_ids):
        seq_len = in_token_ids.size(-1)
        # Compute token positions for actual seq_len
        token_positions = torch.arange(seq_len, device=in_token_ids.device).expand_as(in_token_ids) # expand_as don't really copy the data, just view
        embeddings = self.embeddings(in_token_ids)
        for transformer_block in self.transformer_blocks:
            embeddings = transformer_block(embeddings, self.rope, token_positions)
        # Return raw logits — no softmax (cross-entropy or other types of loss expects unnormalized logits)
        return self.lm_head(self.ln_final(embeddings))

