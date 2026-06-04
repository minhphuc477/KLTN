"""Probe LogicNet gradient stability under diffusion-style Gaussian noise.

This is a feasibility check for DPS/DPPS-style LogicNet guidance. It measures
whether the differentiable pathfinder still provides finite, non-trivial
gradients when clean semantic room logits are corrupted toward noise.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.definitions import ROOM_HEIGHT, ROOM_WIDTH, SEMANTIC_PALETTE
from src.core.logic_net import LogicNet
from src.utils.checkpoint import safe_torch_load


@dataclass(frozen=True)
class NoiseGradientStats:
    noise_level: float
    score_mean: float
    loss_mean: float
    grad_norm_mean: float
    grad_norm_std: float
    grad_abs_mean: float
    finite_rate: float
    relative_grad_norm: float
    walkability_mean: float
    walkability_std: float


def _parse_noise_levels(raw: str) -> List[float]:
    levels = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value < 0.0:
            raise argparse.ArgumentTypeError("Noise levels must be non-negative.")
        levels.append(value)
    if not levels:
        raise argparse.ArgumentTypeError("At least one noise level is required.")
    return levels


def build_clean_solvable_room_logits(
    *,
    num_classes: int = 44,
    height: int = ROOM_HEIGHT,
    width: int = ROOM_WIDTH,
    logit_margin: float = 8.0,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one deterministic solvable room in tile-logit space."""
    if num_classes <= max(int(SEMANTIC_PALETTE["TRIFORCE"]), int(SEMANTIC_PALETTE["START"])):
        raise ValueError("num_classes is too small for the Zelda semantic palette.")
    device = device or torch.device("cpu")
    wall_id = int(SEMANTIC_PALETTE["WALL"])
    floor_id = int(SEMANTIC_PALETTE["FLOOR"])
    start_id = int(SEMANTIC_PALETTE["START"])
    goal_id = int(SEMANTIC_PALETTE["TRIFORCE"])

    tile_ids = torch.full((height, width), wall_id, device=device, dtype=torch.long)
    row = height // 2
    tile_ids[row, 1 : width - 1] = floor_id
    tile_ids[row, 1] = start_id
    tile_ids[row, width - 2] = goal_id

    logits = torch.full((1, num_classes, height, width), -float(logit_margin), device=device)
    logits.scatter_(1, tile_ids.unsqueeze(0).unsqueeze(1), float(logit_margin))

    start_mask = torch.zeros((1, height, width), device=device)
    start_mask[:, row, 1] = 1.0
    goal_mask = torch.zeros((1, height, width), device=device)
    goal_mask[:, row, width - 2] = 1.0
    return logits, start_mask, goal_mask


def _diffusion_style_corrupt(
    clean_logits: torch.Tensor,
    *,
    noise_level: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Blend clean logits with Gaussian noise using a VP-style alpha bar."""
    noise_level = float(max(0.0, min(1.0, noise_level)))
    alpha_bar = 1.0 - noise_level
    noise = torch.randn(
        clean_logits.shape,
        generator=generator,
        device=clean_logits.device,
        dtype=clean_logits.dtype,
    )
    return (alpha_bar ** 0.5) * clean_logits + ((1.0 - alpha_bar) ** 0.5) * noise


def probe_logicnet_noisy_gradients(
    logic_net: LogicNet,
    *,
    noise_levels: Sequence[float],
    samples_per_level: int = 8,
    seed: int = 0,
    device: Optional[torch.device] = None,
) -> List[NoiseGradientStats]:
    """Return gradient statistics for increasing noise levels."""
    if samples_per_level <= 0:
        raise ValueError("samples_per_level must be positive.")
    device = device or next(logic_net.parameters(), torch.empty((), device="cpu")).device
    logic_net = logic_net.to(device)
    logic_net.eval()

    clean_logits, start_mask, goal_mask = build_clean_solvable_room_logits(
        num_classes=int(logic_net.num_classes),
        device=device,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    rows: List[NoiseGradientStats] = []
    clean_grad_norm = None
    for noise_level in noise_levels:
        grad_norms = []
        grad_abs = []
        scores = []
        losses = []
        finite_count = 0
        walk_means = []
        walk_stds = []

        for _ in range(int(samples_per_level)):
            x = _diffusion_style_corrupt(
                clean_logits,
                noise_level=float(noise_level),
                generator=generator,
            ).detach().requires_grad_(True)
            score = logic_net(x, start_mask, goal_mask).mean()
            loss = 1.0 - score
            grad = torch.autograd.grad(loss, x, allow_unused=False)[0]
            finite = torch.isfinite(grad).all() and torch.isfinite(loss)
            finite_count += int(bool(finite))
            grad_norms.append(float(grad.norm().detach().cpu().item()))
            grad_abs.append(float(grad.abs().mean().detach().cpu().item()))
            scores.append(float(score.detach().cpu().item()))
            losses.append(float(loss.detach().cpu().item()))
            with torch.no_grad():
                walk = logic_net.walkability(x, is_probs=False)
                walk_means.append(float(walk.mean().detach().cpu().item()))
                walk_stds.append(float(walk.std(unbiased=False).detach().cpu().item()))

        grad_norm_t = torch.tensor(grad_norms, dtype=torch.float64)
        grad_norm_mean = float(grad_norm_t.mean().item())
        if clean_grad_norm is None and float(noise_level) <= 1e-12:
            clean_grad_norm = max(grad_norm_mean, 1e-12)
        denominator = clean_grad_norm if clean_grad_norm is not None else max(grad_norm_mean, 1e-12)
        rows.append(
            NoiseGradientStats(
                noise_level=float(noise_level),
                score_mean=float(torch.tensor(scores, dtype=torch.float64).mean().item()),
                loss_mean=float(torch.tensor(losses, dtype=torch.float64).mean().item()),
                grad_norm_mean=grad_norm_mean,
                grad_norm_std=float(grad_norm_t.std(unbiased=False).item()),
                grad_abs_mean=float(torch.tensor(grad_abs, dtype=torch.float64).mean().item()),
                finite_rate=float(finite_count / max(1, int(samples_per_level))),
                relative_grad_norm=float(grad_norm_mean / max(denominator, 1e-12)),
                walkability_mean=float(torch.tensor(walk_means, dtype=torch.float64).mean().item()),
                walkability_std=float(torch.tensor(walk_stds, dtype=torch.float64).mean().item()),
            )
        )
    return rows


def recommend_guidance_window(
    rows: Iterable[NoiseGradientStats],
    *,
    min_relative_grad: float = 0.05,
    max_relative_grad: float = 20.0,
    min_finite_rate: float = 1.0,
) -> dict[str, float | str]:
    """Classify whether full or late-stage guidance is supported by the probe."""
    rows = sorted(rows, key=lambda row: row.noise_level)
    stable = [
        row
        for row in rows
        if row.finite_rate >= min_finite_rate
        and min_relative_grad <= row.relative_grad_norm <= max_relative_grad
        and row.grad_norm_mean > 0.0
    ]
    if not stable:
        return {"strategy": "disabled", "max_stable_noise": 0.0}
    max_stable = max(row.noise_level for row in stable)
    max_seen = max(row.noise_level for row in rows) if rows else 0.0
    strategy = "full" if max_stable >= max_seen else "late"
    return {"strategy": strategy, "max_stable_noise": float(max_stable)}


def _extract_logicnet_state(checkpoint: Any, state_key: str = "logic_net_state_dict") -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if state_key in checkpoint and isinstance(checkpoint[state_key], dict):
            return checkpoint[state_key]
        for key in ("logic_net_state_dict", "state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            return checkpoint
    raise ValueError(
        "Could not find a LogicNet state dict. Expected a raw state_dict or a checkpoint "
        "with logic_net_state_dict/state_dict/model_state_dict."
    )


def _infer_logicnet_architecture(state: dict[str, torch.Tensor]) -> dict[str, int]:
    inferred: dict[str, int] = {}
    first = state.get("tile_classifier.classifier.0.weight")
    if isinstance(first, torch.Tensor) and first.dim() == 4:
        inferred["hidden_dim"] = int(first.shape[0])
        inferred["latent_dim"] = int(first.shape[1])
    final = state.get("tile_classifier.classifier.6.weight")
    if isinstance(final, torch.Tensor) and final.dim() == 4:
        inferred["num_classes"] = int(final.shape[0])
    walkability = state.get("walkability.walkability_weights")
    if isinstance(walkability, torch.Tensor) and walkability.dim() == 1:
        inferred["num_classes"] = int(walkability.shape[0])
    return inferred


def _write_json(path: Path, rows: Sequence[NoiseGradientStats], recommendation: dict[str, float | str]) -> None:
    payload = {
        "rows": [asdict(row) for row in rows],
        "recommendation": recommendation,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[NoiseGradientStats]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise-levels", type=_parse_noise_levels, default=_parse_noise_levels("0,0.1,0.25,0.5,0.75,1.0"))
    parser.add_argument("--samples-per-level", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=44)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--logicnet-checkpoint", type=Path, default=None)
    parser.add_argument("--logicnet-state-key", type=str, default="logic_net_state_dict")
    parser.add_argument("--no-infer-checkpoint-shapes", action="store_true")
    parser.add_argument("--num-iterations", type=int, default=8)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    state = None
    num_classes = 44
    latent_dim = int(args.latent_dim)
    hidden_dim = int(args.hidden_dim)
    if args.logicnet_checkpoint is not None:
        checkpoint = safe_torch_load(args.logicnet_checkpoint, map_location="cpu")
        state = _extract_logicnet_state(checkpoint, state_key=str(args.logicnet_state_key))
        if not bool(args.no_infer_checkpoint_shapes):
            inferred = _infer_logicnet_architecture(state)
            latent_dim = int(inferred.get("latent_dim", latent_dim))
            hidden_dim = int(inferred.get("hidden_dim", hidden_dim))
            num_classes = int(inferred.get("num_classes", num_classes))
    logic_net = LogicNet(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_iterations=int(args.num_iterations),
        grid_pathfinder_type="bellman_ford",
    )
    if state is not None:
        logic_net.load_state_dict(state)
    rows = probe_logicnet_noisy_gradients(
        logic_net,
        noise_levels=args.noise_levels,
        samples_per_level=int(args.samples_per_level),
        seed=int(args.seed),
        device=device,
    )
    recommendation = recommend_guidance_window(rows)
    print(json.dumps({"rows": [asdict(row) for row in rows], "recommendation": recommendation}, indent=2))
    if args.json_output is not None:
        _write_json(args.json_output, rows, recommendation)
    if args.csv_output is not None:
        _write_csv(args.csv_output, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
