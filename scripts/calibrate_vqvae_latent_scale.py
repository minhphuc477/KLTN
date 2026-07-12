#!/usr/bin/env python
"""Estimate a reproducible corpus-level VQ latent scale for diffusion training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.vqvae import create_vqvae
from src.utils.checkpoint import checkpoint_sha256, safe_torch_load
from src.zelda_data.zelda_loader import create_dataloader


def _state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise ValueError("VQ-VAE checkpoint must be a mapping.")
    for key in ("model_state_dict", "vqvae_state_dict", "state_dict"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            return candidate
    if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
        return payload
    raise ValueError("Checkpoint contains no standalone VQ-VAE state dictionary.")


def calibrate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    model = create_vqvae(
        architecture=args.architecture,
        num_classes=args.num_classes,
        codebook_size=args.codebook_size,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        top_codebook_size=args.top_codebook_size,
        top_latent_dim=args.top_latent_dim,
        use_coordconv=args.use_coordconv,
    ).to(device)
    checkpoint = safe_torch_load(str(args.checkpoint), map_location="cpu")
    incompatible = model.load_state_dict(_state_dict(checkpoint), strict=False)
    missing = [key for key in incompatible.missing_keys if key != "illegal_adjacency_matrix"]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"VQ-VAE checkpoint mismatch: missing={missing}, "
            f"unexpected={list(incompatible.unexpected_keys)}"
        )
    model.eval()

    loader = create_dataloader(
        data_dir=str(args.data_dir),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        use_vglc=True,
        normalize=True,
        room_level=True,
        load_graphs=False,
        dungeon_ids=args.dungeon_ids,
        variants=args.variants,
    )
    count = 0
    total = torch.zeros((), dtype=torch.float64, device=device)
    total_sq = torch.zeros((), dtype=torch.float64, device=device)
    room_count = 0
    with torch.inference_mode():
        for batch in loader:
            maps = batch[0] if isinstance(batch, (tuple, list)) else batch
            maps = maps.to(device)
            if maps.dim() != 4 or maps.shape[1] != 1:
                raise ValueError(f"Expected normalized [B,1,H,W] maps, got {tuple(maps.shape)}.")
            ids = (maps[:, 0] * (args.num_classes - 1)).round().long()
            one_hot = F.one_hot(ids.clamp(0, args.num_classes - 1), args.num_classes)
            one_hot = one_hot.permute(0, 3, 1, 2).float()
            latent, _indices = model.encode(one_hot)
            values = latent.to(torch.float64)
            count += values.numel()
            total += values.sum()
            total_sq += values.square().sum()
            room_count += int(values.shape[0])
            if args.max_rooms > 0 and room_count >= args.max_rooms:
                break
    if count < 2:
        raise RuntimeError("At least two latent values are required for calibration.")
    mean = float((total / count).item())
    variance = float(((total_sq - (total.square() / count)) / (count - 1)).item())
    std = math.sqrt(max(variance, 0.0))
    if not math.isfinite(std) or std <= 0.0:
        raise RuntimeError(f"Invalid latent standard deviation {std!r}.")
    return {
        "format": "hmolqd_vqvae_latent_scale_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256(args.checkpoint),
        "dungeon_ids": list(args.dungeon_ids),
        "variants": list(args.variants),
        "rooms_observed": room_count,
        "latent_values_observed": count,
        "latent_mean": mean,
        "latent_std": std,
        "recommended_latent_scale_factor": 1.0 / std,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("Data/The Legend of Zelda"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--architecture", choices=("vqvae", "vqvae2"), default="vqvae")
    parser.add_argument("--num-classes", type=int, default=44)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--top-codebook-size", type=int)
    parser.add_argument("--top-latent-dim", type=int)
    parser.add_argument("--use-coordconv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-rooms", type=int, default=0)
    parser.add_argument("--dungeon-ids", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--variants", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    report = calibrate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
