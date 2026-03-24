"""Decoding utilities for autoregressive text generation."""

from __future__ import annotations

import torch

from cs336_basics.nn.softmax import softmax


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Zero out logits outside the top-p (nucleus) probability mass.
        Mask the logits to -inf so that they no longer contribute to the sampling
    Args:
        logits: (vocab_size,) — unnormalized logits for a single position.
        top_p: Cumulative probability threshold (0.0 < top_p <= 1.0).

    Returns:
        (vocab_size,) — logits with low-probability tokens set to -inf.

    Steps:
        1. Compute probabilities from logits (use softmax)
        2. Sort probabilities in descending order
        3. Compute cumulative sum of sorted probabilities
        4. Find the cutoff: smallest set of tokens whose cumulative prob >= top_p
        5. Set logits of tokens outside this set to -inf
    """
    probs = softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # Find which sorted positions exceed the threshold
    # Shift by subtracting current prob so the token that crosses top_p is kept
    mask = cumulative_probs - sorted_probs > top_p
    # Get the original indices of tokens to mask out, and set them to -inf
    indices_to_remove = sorted_indices[mask]
    logits[indices_to_remove] = -torch.inf
    return logits


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Generate a completion from the model given a prompt.

    Args:
        model: Your TransformerLLM (should be in eval mode).
        prompt_ids: (seq_len,) — 1D tensor of token IDs for the prompt.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Softmax temperature. Higher = more random, lower = more greedy.
        top_p: Nucleus sampling threshold. 1.0 = no filtering.
        eos_token_id: If provided, stop generation when this token is sampled.

    Returns:
        (seq_len + generated_len,) — full sequence including prompt and generated tokens.

    Steps for each generation step:
        1. Feed current sequence through the model to get logits
        2. Take logits for the last position: logits[:, -1, :]
        3. Apply temperature: divide logits by temperature
        4. Apply top-p filtering if top_p < 1.0
        5. Compute probabilities with softmax
        6. Sample one token from the distribution (torch.multinomial)
        7. Append sampled token to the sequence
        8. Stop if eos_token_id is sampled or max_new_tokens reached

    Hint: Keep the sequence length within context_length by truncating
          from the left if needed (sliding window).
    """
    context_length = model.max_context_length
    for t in range(max_new_tokens):
        # Truncate to context window if sequence exceeds it (sliding window)
        input_ids = prompt_ids[-context_length:].unsqueeze(0)  # (1, seq_len)
        logits = model(input_ids)
        # Get logits for the last position
        last_logit = logits[0, -1, :] / temperature
        # Apply top-p filtering
        if top_p < 1.0:
            last_logit = top_p_filter(last_logit, top_p)
        prob = softmax(last_logit, -1)
        next_token = torch.multinomial(prob, num_samples=1)  # (1,)
        # Note torch.stack create a new dimension and requires all tensor to have the same shape
        # and torch.cat concat tensors along an existing dimensions
        prompt_ids = torch.cat([prompt_ids, next_token])
        if eos_token_id is not None and next_token.item() == eos_token_id:
            return prompt_ids
    return prompt_ids