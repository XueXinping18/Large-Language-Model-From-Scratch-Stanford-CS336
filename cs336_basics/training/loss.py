from __future__ import annotations

import torch


def cross_entropy(
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute average cross-entropy loss from unnormalized logits.

    Args:
        inputs: (batch_size, vocab_size) — unnormalized logits.
        targets: (batch_size,) — integer class indices.

    Returns:
        Scalar tensor with the average cross-entropy loss.

    Don't use torch.nn.functional.cross_entropy or torch.nn.CrossEntropyLoss.

    Hints:
        - Don't Use softmax. Use log-sum-exp trick for numerical stability
        - log_softmax(x)_i = x_i - log(sum(exp(x)))
        - Loss for one example = -log_softmax(x)[target]
        - Average over the batch
    """
    numerator = torch.gather(inputs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    max_per_token = torch.max(inputs, dim = -1).values
    exp = torch.exp(inputs - torch.unsqueeze(max_per_token, -1))
    loss_tensor = - (numerator - (max_per_token + torch.log(torch.sum(exp, dim=-1))))
    return torch.mean(loss_tensor)
