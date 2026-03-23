from __future__ import annotations

import math


def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Cosine learning rate schedule with linear warmup.

    Args:
        it: Current iteration number.
        max_learning_rate: Peak learning rate (alpha_max).
        min_learning_rate: Final learning rate (alpha_min).
        warmup_iters: Number of linear warmup iterations (T_w).
        cosine_cycle_iters: Total number of cosine annealing iterations (T_c).

    Returns:
        Learning rate at iteration `it`.

    Three phases:
        1. it < warmup_iters: linear warmup from 0 to max_lr
        2. warmup_iters <= it < cosine_cycle_iters: cosine decay from max_lr to min_lr
        3. it >= cosine_cycle_iters: constant min_lr
    """
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi))
    else:
        return min_learning_rate