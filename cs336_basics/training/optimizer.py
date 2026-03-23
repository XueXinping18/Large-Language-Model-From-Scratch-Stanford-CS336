from __future__ import annotations

import torch


class AdamW(torch.optim.Optimizer):
    """AdamW optimizer (Adam with decoupled weight decay).

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate.
        betas: Coefficients for computing running averages of gradient and its square.
        eps: Term added to denominator for numerical stability.
        weight_decay: Decoupled weight decay coefficient.

    Don't use torch.optim.AdamW or any existing optimizer implementation.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Perform a single optimization step.

        For each parameter:
            1. Get the gradient
            2. Update biased first moment estimate (m)
            3. Update biased second moment estimate (v)
            4. Compute bias-corrected estimates
            5. Update parameter: p = p - lr * m_hat / (sqrt(v_hat) + eps)
            6. Apply weight decay: p = p - lr * weight_decay * p
        """
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                grad = p.grad.data
                m = betas[0] * state.get("m", 0) + (1 - betas[0]) * grad
                v = betas[1] * state.get("v", 0) + (1 - betas[1]) * grad ** 2
                lr_t = lr * (1 - betas[1] ** t) ** 0.5 / (1 - betas[0] ** t)
                p.data -= lr_t * m / (v ** 0.5 + eps)
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss
