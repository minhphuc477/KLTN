"""Measure local chunked versus global soft Bellman-Ford scaling."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.baselines.common import json_ready, set_reproducible_seed
from src.core.logic_net import DifferentiablePathfinder


def _peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return float(torch.cuda.max_memory_allocated(device) / (1024.0**2))
    return 0.0


def _bench_grid(
    *,
    shape: Tuple[int, int],
    iterations: int,
    repeats: int,
    device: torch.device,
) -> Dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    h, w = int(shape[0]), int(shape[1])
    walkable = torch.ones(1, h, w, device=device)
    weights = torch.ones_like(walkable)
    source = torch.zeros_like(walkable)
    source[:, 0, 0] = 1.0
    model = DifferentiablePathfinder(num_iterations=iterations, temperature=0.1).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(int(repeats)):
            dist = model(walkable, weights, source)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - start) / max(1, int(repeats))
    return {
        "shape": [h, w],
        "iterations": int(iterations),
        "seconds_per_forward": float(elapsed),
        "peak_memory_mb": _peak_memory_mb(device),
        "finite_output": bool(torch.isfinite(dist).all().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-shape", type=str, default="11x16")
    parser.add_argument("--global-shapes", type=str, default="64x64,128x128,256x256")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=str, default="experiments/outputs/ablation_logicnet_scaling.json")
    args = parser.parse_args()

    set_reproducible_seed(args.seed)
    device = torch.device(args.device)

    def parse_shape(text: str) -> Tuple[int, int]:
        lhs, rhs = str(text).lower().split("x", 1)
        return int(lhs), int(rhs)

    repeats = 1 if args.dry_run else int(args.repeats)
    global_shapes = [parse_shape("32x32")] if args.dry_run else [parse_shape(item) for item in args.global_shapes.split(",") if item]
    chunk_shape = parse_shape(args.chunk_shape)
    result = {
        "config": vars(args),
        "local_chunk": _bench_grid(shape=chunk_shape, iterations=args.iterations, repeats=repeats, device=device),
        "global": [
            _bench_grid(shape=shape, iterations=args.iterations, repeats=repeats, device=device)
            for shape in global_shapes
        ],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(json_ready(result), indent=2), encoding="utf-8")
    print(json.dumps(json_ready(result), indent=2))


if __name__ == "__main__":
    main()
