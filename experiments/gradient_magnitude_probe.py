#!/usr/bin/env python
"""Measure whether LogicNet losses send gradients into the latent path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.logic_net import LogicNet


def _grad_stats(tensor: torch.Tensor | None) -> Dict[str, float]:
    if tensor is None:
        return {"mean_abs": 0.0, "max_abs": 0.0, "l2": 0.0}
    detached = tensor.detach()
    return {
        "mean_abs": float(detached.abs().mean().item()),
        "max_abs": float(detached.abs().max().item()),
        "l2": float(torch.linalg.vector_norm(detached).item()),
    }


def run_probe(args: argparse.Namespace) -> Dict[str, Any]:
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)

    net = LogicNet(
        latent_dim=int(args.latent_dim),
        num_classes=int(args.num_classes),
        hidden_dim=int(args.hidden_dim),
        num_iterations=int(args.num_iterations),
        grid_pathfinder_type=str(args.pathfinder),
    ).to(device)
    net.train()

    z = torch.randn(
        int(args.batch_size),
        int(args.latent_dim),
        int(args.height),
        int(args.width),
        device=device,
        requires_grad=True,
    )

    loss, info = net(z, graph_data=None)
    retained_names = ["latent_tile_logits", "walkability", "grid_reach_loss"]
    for name in retained_names:
        value = info.get(name)
        if torch.is_tensor(value):
            value.retain_grad()

    net.zero_grad(set_to_none=True)
    if z.grad is not None:
        z.grad = None
    loss.backward()

    first_param = next(net.tile_classifier.parameters())
    result: Dict[str, Any] = {
        "seed": int(args.seed),
        "pathfinder": str(args.pathfinder),
        "loss": float(loss.detach().item()),
        "grid_reach_loss": float(info["grid_reach_loss"].detach().item()),
        "gradients": {
            "latent_z": _grad_stats(z.grad),
            "tile_classifier_first_param": _grad_stats(first_param.grad),
        },
    }
    for name in retained_names:
        value = info.get(name)
        if torch.is_tensor(value):
            result["gradients"][name] = _grad_stats(value.grad)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("results/logicnet_gradient_probe.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pathfinder", choices=("bellman_ford", "cnn"), default="bellman_ford")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=44)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--num-iterations", type=int, default=8)
    args = parser.parse_args()

    result = run_probe(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
