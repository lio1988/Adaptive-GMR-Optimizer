"""AdaptiveGMR — Geman-McClure robust gradient scaling optimizer."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
from torch.optim import Optimizer


class AdaptiveGMR(Optimizer):
    """
    SGD optimizer with per-parameter Geman-McClure gradient scaling.

    Each parameter maintains an adaptive sigma that tracks gradient magnitude.
    Gradients are scaled as sigma^2 / (sigma^2 + ||g||^2) before the SGD update.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        alpha: float = 0.9,
        sigma_init: float = 1.0,
        sigma_warmup_steps: int = 0,
        accumulation_steps: int = 1,
        eps: float = 1e-8,
        grad_norm_history_size: int = 10,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if sigma_init <= 0.0:
            raise ValueError(f"sigma_init must be positive, got {sigma_init}")
        if accumulation_steps < 1:
            raise ValueError(f"accumulation_steps must be >= 1, got {accumulation_steps}")

        defaults = dict(
            lr=lr,
            alpha=alpha,
            sigma_init=sigma_init,
            sigma_warmup_steps=sigma_warmup_steps,
            eps=eps,
        )
        super().__init__(params, defaults)
        self.accumulation_steps = accumulation_steps
        self.grad_norm_history_size = grad_norm_history_size
        self._accum_count = 0
        self._global_step = 0

    def __repr__(self) -> str:
        lr = self.param_groups[0]["lr"] if self.param_groups else 0.0
        return (
            f"{self.__class__.__name__}(lr={lr}, "
            f"alpha={self.param_groups[0]['alpha'] if self.param_groups else 0.0}, "
            f"params={sum(len(g['params']) for g in self.param_groups)})"
        )

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Reset gradient accumulation counter when gradients are cleared."""
        super().zero_grad(set_to_none=set_to_none)
        self._accum_count = 0

    def _get_grad(self, p: torch.Tensor) -> Optional[torch.Tensor]:
        """Return gradient tensor; handle DeepSpeed ZeRO-3 (grad None, grad_fn set)."""
        if p.grad is not None:
            return p.grad
        if getattr(p, "grad_fn", None) is not None:
            return None
        return None

    def _should_update_sigma(self) -> bool:
        """Update sigma only on the actual optimizer step (after accumulation)."""
        return (self._accum_count + 1) % self.accumulation_steps == 0

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._accum_count += 1
        is_actual_step = self._should_update_sigma()

        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["alpha"]
            sigma_init = group["sigma_init"]
            sigma_warmup_steps = group["sigma_warmup_steps"]
            eps = group["eps"]

            for p in group["params"]:
                grad = self._get_grad(p)
                if grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["sigma"] = torch.tensor(
                        sigma_init, dtype=torch.float32, device=p.device
                    )
                    state["grad_norm_history"] = []

                grad_norm_sq = grad.detach().pow(2).sum()
                grad_norm = grad_norm_sq.sqrt()

                history: List[float] = state["grad_norm_history"]
                history.append(float(grad_norm.item()))
                if len(history) > self.grad_norm_history_size:
                    history.pop(0)
                state["grad_norm_history"] = history

                if is_actual_step:
                    if self._global_step >= sigma_warmup_steps:
                        state["sigma"] = (
                            alpha * state["sigma"] + (1.0 - alpha) * grad_norm
                        )
                    else:
                        state["sigma"] = torch.tensor(
                            sigma_init, dtype=torch.float32, device=p.device
                        )

                    sigma = state["sigma"]
                    scaling = sigma.pow(2) / (sigma.pow(2) + grad_norm_sq + eps)
                    p.data.add_(grad * scaling, alpha=-lr)

        if is_actual_step:
            self._global_step += 1

        return loss

    def get_sigma_stats(self) -> Optional[Dict[str, float]]:
        """Return mean/min/max sigma across all parameters."""
        sigmas: List[float] = []
        for group in self.param_groups:
            for p in group["params"]:
                if p in self.state and "sigma" in self.state[p]:
                    sigmas.append(float(self.state[p]["sigma"].item()))
        if not sigmas:
            return None
        return {
            "mean": sum(sigmas) / len(sigmas),
            "min": min(sigmas),
            "max": max(sigmas),
        }

    def reset_sigma(self) -> None:
        """Reset all per-parameter sigmas to their group's sigma_init value."""
        for group in self.param_groups:
            sigma_init = group["sigma_init"]
            for p in group["params"]:
                if p in self.state:
                    self.state[p]["sigma"] = torch.tensor(
                        sigma_init, dtype=torch.float32, device=p.device
                    )

    def state_dict(self) -> Dict[str, Any]:
        """Return optimizer state including accumulation counters."""
        sd = super().state_dict()
        sd["accumulation_steps"] = self.accumulation_steps
        sd["accum_count"] = self._accum_count
        sd["global_step"] = self._global_step
        return sd

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load optimizer state including accumulation counters."""
        self.accumulation_steps = state_dict.pop("accumulation_steps", self.accumulation_steps)
        self._accum_count = state_dict.pop("accum_count", 0)
        self._global_step = state_dict.pop("global_step", 0)
        super().load_state_dict(state_dict)
