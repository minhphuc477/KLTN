"""Reusable gradient-flow probes for neural-symbolic training."""

from __future__ import annotations

import logging
import math
from itertools import combinations
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch
from torch import nn


def objective_gradient_diagnostics(
    objectives: Mapping[str, torch.Tensor],
    parameters: Iterable[nn.Parameter],
    *,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """Measure per-objective gradient strength and pairwise agreement.

    The caller should pass the *weighted* scalar loss terms that are actually
    summed by the optimizer.  Gradients are obtained with ``autograd.grad`` so
    this diagnostic does not populate or overwrite ``parameter.grad``.  It is
    intended for publication preflights, not every training step.

    A pairwise cosine is ``None`` when either objective has zero gradient.  A
    missing cosine must therefore be treated as a failed liveness check rather
    than as neutral agreement.
    """
    named_objectives = [(str(name), loss) for name, loss in objectives.items()]
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    if not named_objectives:
        raise ValueError("At least one objective is required for gradient diagnostics.")
    if not trainable:
        raise ValueError("At least one trainable parameter is required for gradient diagnostics.")
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError(f"eps must be finite and positive, got {eps!r}.")

    gradients: Dict[str, tuple[Optional[torch.Tensor], ...]] = {}
    norms: Dict[str, float] = {}
    active_fractions: Dict[str, float] = {}
    for name, loss in named_objectives:
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
            raise ValueError(f"Objective {name!r} must be a scalar tensor.")
        if not bool(torch.isfinite(loss.detach()).all()):
            raise FloatingPointError(f"Objective {name!r} is non-finite.")

        if loss.requires_grad:
            raw_gradients = torch.autograd.grad(
                loss,
                trainable,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
        else:
            raw_gradients = tuple(None for _ in trainable)
        detached = tuple(
            None if gradient is None else gradient.detach()
            for gradient in raw_gradients
        )
        gradients[name] = detached
        squared_norm = sum(
            float(gradient.double().square().sum().item())
            for gradient in detached
            if gradient is not None
        )
        norms[name] = math.sqrt(max(0.0, squared_norm))
        active_fractions[name] = sum(
            int(torch.count_nonzero(gradient).item())
            for gradient in detached
            if gradient is not None
        ) / float(sum(parameter.numel() for parameter in trainable))

    cosines: Dict[str, Optional[float]] = {}
    for (left_name, _), (right_name, _) in combinations(named_objectives, 2):
        denominator = norms[left_name] * norms[right_name]
        key = f"{left_name}__vs__{right_name}"
        if denominator <= float(eps):
            cosines[key] = None
            continue
        dot = sum(
            float((left.double() * right.double()).sum().item())
            for left, right in zip(gradients[left_name], gradients[right_name])
            if left is not None and right is not None
        )
        cosines[key] = max(-1.0, min(1.0, dot / denominator))

    return {
        "objective_gradient_norms": norms,
        "objective_active_parameter_fractions": active_fractions,
        "objective_gradient_cosines": cosines,
    }


class GradientProbe:
    """Context manager that records output-gradient norms for selected modules."""

    def __init__(
        self,
        modules: Iterable[Tuple[str, nn.Module]],
        logger: Optional[Any] = None,
        *,
        prefix: str = "grad",
    ) -> None:
        self.modules = [(str(name), module) for name, module in modules if isinstance(module, nn.Module)]
        self.logger = logger
        self.prefix = str(prefix).strip("/") or "grad"
        self.hooks: list[Any] = []
        self.last_stats: Dict[str, float] = {}

    @classmethod
    def for_logicnet(cls, logic_net: nn.Module, logger: Optional[Any] = None) -> "GradientProbe":
        names = (
            "tile_classifier",
            "walkability",
            "grid_pathfinder",
            "graph_pathfinder",
            "semantic_edge_encoder",
        )
        return cls(
            ((name, getattr(logic_net, name)) for name in names if hasattr(logic_net, name)),
            logger=logger,
            prefix="grad/logicnet",
        )

    def __enter__(self) -> "GradientProbe":
        for name, module in self.modules:
            self.hooks.append(module.register_full_backward_hook(self._make_hook(name)))
        return self

    def __exit__(self, _exc_type: Any, exc: Any, tb: Any) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def _make_hook(self, name: str):
        def _hook(_module: nn.Module, _grad_input: tuple[Any, ...], grad_output: tuple[Any, ...]) -> None:
            tensors = [grad for grad in grad_output if isinstance(grad, torch.Tensor)]
            if not tensors:
                return
            value = float(torch.stack([grad.detach().abs().mean() for grad in tensors]).mean().item())
            metric_name = f"{self.prefix}/{name}"
            self.last_stats[metric_name] = value
            self._log(metric_name, value)

        return _hook

    def _log(self, name: str, value: float) -> None:
        target = self.logger
        if target is None:
            return
        payload = {name: value}
        if isinstance(target, logging.Logger):
            target.info("%s=%s", name, value)
        elif hasattr(target, "log"):
            target.log(payload)
        elif hasattr(target, "log_metrics"):
            target.log_metrics(payload)
        elif hasattr(target, "add_scalar"):
            target.add_scalar(name, value)
        elif callable(target):
            target(payload)
