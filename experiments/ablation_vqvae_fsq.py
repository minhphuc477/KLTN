"""Ablate learned EMA VQ against finite scalar quantization.

The script reports reconstruction loss, quantizer loss, perplexity/usage, and
simple collapse indicators. It is intentionally checkpoint-optional so a
`--dry-run` validates wiring on CPU before expensive trained-model runs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.baselines.common import json_ready, load_room_grids, set_reproducible_seed
from src.core import ROOM_HEIGHT, ROOM_WIDTH
from src.core.vqvae import create_vqvae


def _one_hot(grids: Iterable[np.ndarray], *, num_classes: int, device: torch.device) -> torch.Tensor:
    arrays = [np.asarray(grid, dtype=np.int64) for grid in grids]
    if not arrays:
        raise ValueError("No grids available for VQ-VAE ablation.")
    x = torch.as_tensor(np.stack(arrays, axis=0), device=device, dtype=torch.long).clamp(0, num_classes - 1)
    return F.one_hot(x, num_classes=num_classes).permute(0, 3, 1, 2).float()


def _load_checkpoint(model: torch.nn.Module, path: Optional[str], device: torch.device) -> None:
    if not path:
        return
    payload = torch.load(path, map_location=device)
    state = payload.get("model_state_dict", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] checkpoint {path}: missing={len(missing)} unexpected={len(unexpected)}")


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, x: torch.Tensor, *, batch_size: int) -> Dict[str, Any]:
    model.eval()
    losses: List[float] = []
    q_losses: List[float] = []
    all_indices: List[torch.Tensor] = []
    for start in range(0, int(x.shape[0]), int(batch_size)):
        batch = x[start : start + int(batch_size)]
        recon, vq_loss, detail = model(batch)
        target = batch.argmax(dim=1)
        losses.append(float(F.cross_entropy(recon.float(), target, reduction="mean").item()))
        q_losses.append(float(vq_loss.detach().float().item()))
        indices = detail.get("indices") if isinstance(detail, dict) else None
        if isinstance(indices, torch.Tensor):
            all_indices.append(indices.detach().reshape(-1).cpu())

    codebook_size = int(getattr(model, "codebook_size", 0) or 0)
    metrics: Dict[str, Any] = {
        "reconstruction_ce": float(np.mean(losses)) if losses else math.nan,
        "quantizer_loss": float(np.mean(q_losses)) if q_losses else math.nan,
        "codebook_size": codebook_size,
    }
    if all_indices and codebook_size > 0:
        flat = torch.cat(all_indices).long()
        counts = torch.bincount(flat.clamp(0, codebook_size - 1), minlength=codebook_size).float()
        probs = counts / counts.sum().clamp_min(1.0)
        entropy = -(probs[probs > 0] * probs[probs > 0].log()).sum()
        used = int((counts > 0).sum().item())
        metrics.update(
            {
                "used_codes": used,
                "dead_code_rate": float(1.0 - used / max(1, codebook_size)),
                "perplexity": float(torch.exp(entropy).item()),
                "max_code_probability": float(probs.max().item()),
            }
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=44)
    parser.add_argument("--codebook-size", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--vq-checkpoint", type=str, default=None)
    parser.add_argument("--fsq-checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=str, default="experiments/outputs/ablation_vqvae_fsq.json")
    args = parser.parse_args()

    set_reproducible_seed(args.seed)
    device = torch.device(args.device)
    if args.dry_run:
        rng = np.random.default_rng(args.seed)
        grids = [
            rng.integers(0, min(args.num_classes, 8), size=(ROOM_HEIGHT, ROOM_WIDTH), dtype=np.int64)
            for _ in range(min(8, args.max_samples))
        ]
    else:
        grids = load_room_grids(args.data_dir, max_samples=args.max_samples)
    x = _one_hot(grids, num_classes=args.num_classes, device=device)

    common = dict(
        num_classes=args.num_classes,
        codebook_size=args.codebook_size,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    )
    models = {
        "vq_ema": create_vqvae(architecture="vqvae", use_ema=True, **common).to(device),
        "fsq": create_vqvae(architecture="fsq", **common).to(device),
    }
    _load_checkpoint(models["vq_ema"], args.vq_checkpoint, device)
    _load_checkpoint(models["fsq"], args.fsq_checkpoint, device)

    result = {
        "config": vars(args),
        "num_samples": int(x.shape[0]),
        "metrics": {name: evaluate_model(model, x, batch_size=args.batch_size) for name, model in models.items()},
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(json_ready(result), indent=2), encoding="utf-8")
    print(json.dumps(json_ready(result), indent=2))


if __name__ == "__main__":
    main()
