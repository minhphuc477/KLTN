"""Reusable gradient-flow probes for neural-symbolic training."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch import nn


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

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
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
