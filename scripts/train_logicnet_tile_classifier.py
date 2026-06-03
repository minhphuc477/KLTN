#!/usr/bin/env python
"""Train/evaluate LogicNet's supervised tile-classifier head on frozen VQ-VAE latents."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.logic_net import LogicNet
from src.core.vqvae import create_vqvae
from src.train_diffusion import _resolve_vqvae_architecture, _validate_vqvae_checkpoint_state
from src.utils.checkpoint import safe_torch_load
from src.zelda_data.zelda_loader import create_dataloader


def _tile_targets(real_maps: torch.Tensor, *, num_classes: int, device: torch.device) -> torch.Tensor:
    if real_maps.dim() == 4 and int(real_maps.shape[1]) == 1:
        targets = real_maps[:, 0]
    elif real_maps.dim() == 4 and int(real_maps.shape[1]) == int(num_classes):
        targets = real_maps.argmax(dim=1)
    elif real_maps.dim() == 3:
        targets = real_maps
    else:
        raise ValueError(f"Cannot derive tile targets from shape {tuple(real_maps.shape)}.")
    if targets.dtype.is_floating_point:
        max_value = float(targets.detach().max().item()) if targets.numel() else 0.0
        targets = torch.round(targets * float(num_classes - 1)) if max_value <= 1.0 else torch.round(targets)
    return targets.to(device=device, dtype=torch.long).clamp(0, int(num_classes) - 1)


def _macro_f1(pred: torch.Tensor, target: torch.Tensor, *, num_classes: int) -> float:
    pred = pred.reshape(-1).long().cpu()
    target = target.reshape(-1).long().cpu()
    idx = target * int(num_classes) + pred
    conf = torch.bincount(idx, minlength=int(num_classes) * int(num_classes)).reshape(num_classes, num_classes).float()
    tp = conf.diag()
    precision = tp / conf.sum(dim=0).clamp_min(1.0)
    recall = tp / conf.sum(dim=1).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
    present = conf.sum(dim=1) > 0
    return float(f1[present].mean().item()) if bool(present.any()) else 0.0


@torch.no_grad()
def evaluate(vqvae: torch.nn.Module, logic_net: LogicNet, loader, *, device: torch.device, num_classes: int) -> Dict[str, float]:
    vqvae.eval()
    logic_net.eval()
    correct = 0
    total = 0
    preds = []
    targets = []
    for batch in loader:
        real_maps = batch[0] if isinstance(batch, (tuple, list)) else batch
        real_maps = real_maps.to(device=device, dtype=torch.float32)
        target = _tile_targets(real_maps, num_classes=num_classes, device=device)
        z_0, _ = vqvae.encode(real_maps)
        logits = logic_net.tile_classifier(z_0)
        logits = logic_net._project_tile_logits_to_room(logits)
        pred = logits.argmax(dim=1)
        correct += int((pred == target).sum().item())
        total += int(target.numel())
        preds.append(pred.detach().cpu())
        targets.append(target.detach().cpu())
    pred_all = torch.cat([p.reshape(-1) for p in preds]) if preds else torch.empty(0, dtype=torch.long)
    target_all = torch.cat([t.reshape(-1) for t in targets]) if targets else torch.empty(0, dtype=torch.long)
    return {
        "accuracy": float(correct / max(1, total)),
        "macro_f1": _macro_f1(pred_all, target_all, num_classes=num_classes),
        "num_tiles": float(total),
    }


def train(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    arch = _resolve_vqvae_architecture(
        args.vqvae_checkpoint,
        num_classes=args.num_classes,
        latent_dim=args.latent_dim,
        hidden_dim=args.vqvae_hidden_dim,
        codebook_size=args.vqvae_codebook_size,
        architecture=args.vqvae_architecture,
        use_coordconv=args.vqvae_use_coordconv,
        mrf_penalty_weight=args.vqvae_mrf_penalty_weight,
    )
    vqvae = create_vqvae(
        architecture=arch["architecture"],
        num_classes=arch["num_classes"],
        latent_dim=arch["latent_dim"],
        hidden_dim=arch["hidden_dim"],
        codebook_size=arch["codebook_size"],
        use_coordconv=arch["use_coordconv"],
        mrf_penalty_weight=arch["mrf_penalty_weight"],
    ).to(device)
    checkpoint = safe_torch_load(args.vqvae_checkpoint, map_location="cpu")
    _validate_vqvae_checkpoint_state(args.vqvae_checkpoint, checkpoint, expected_codebook_size=int(arch["codebook_size"]))
    vqvae.load_state_dict(checkpoint["model_state_dict"])
    vqvae.eval()
    for param in vqvae.parameters():
        param.requires_grad = False

    logic_net = LogicNet(
        latent_dim=int(arch["latent_dim"]),
        hidden_dim=args.logic_hidden_dim,
        num_classes=args.num_classes,
        num_iterations=args.logic_iterations,
    ).to(device)
    for name, param in logic_net.named_parameters():
        param.requires_grad = name.startswith("tile_classifier")

    train_loader = create_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        use_vglc=True,
        normalize=True,
        room_level=True,
        load_graphs=False,
        dungeon_ids=args.train_dungeon_ids,
        variants=args.variants,
    )
    val_loader = create_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        use_vglc=True,
        normalize=True,
        room_level=True,
        load_graphs=False,
        dungeon_ids=args.val_dungeon_ids,
        variants=args.variants,
    )
    optimizer = torch.optim.AdamW((p for p in logic_net.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, int(args.epochs) + 1):
        logic_net.train()
        for batch in train_loader:
            real_maps = batch[0] if isinstance(batch, (tuple, list)) else batch
            real_maps = real_maps.to(device=device, dtype=torch.float32)
            target = _tile_targets(real_maps, num_classes=args.num_classes, device=device)
            with torch.no_grad():
                z_0, _ = vqvae.encode(real_maps)
            logits = logic_net.tile_classifier(z_0)
            logits = logic_net._project_tile_logits_to_room(logits)
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in logic_net.parameters() if p.requires_grad], args.grad_clip_norm)
            optimizer.step()
        metrics = evaluate(vqvae, logic_net, val_loader, device=device, num_classes=args.num_classes)
        print(f"epoch={epoch} val_accuracy={metrics['accuracy']:.4f} val_macro_f1={metrics['macro_f1']:.4f}")

    metrics = evaluate(vqvae, logic_net, val_loader, device=device, num_classes=args.num_classes)
    payload = {"metrics": metrics, "config": vars(args), "vqvae_architecture": arch}
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "logic_net_state_dict": logic_net.state_dict(),
            "metrics": metrics,
            "config": vars(args),
            "vqvae_architecture": arch,
        },
        args.checkpoint_out,
    )
    if not args.no_enforce_threshold and metrics["accuracy"] < float(args.min_accuracy):
        raise SystemExit(
            f"Tile classifier accuracy {metrics['accuracy']:.4f} is below threshold {float(args.min_accuracy):.4f}."
        )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", default="Data/The Legend of Zelda")
    parser.add_argument("--vqvae-checkpoint", required=True)
    parser.add_argument("--checkpoint-out", type=Path, default=Path("checkpoints/logicnet_tile_classifier.pth"))
    parser.add_argument("--metrics-out", type=Path, default=Path("results/logicnet_tile_classifier_metrics.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--min-accuracy", type=float, default=0.70)
    parser.add_argument("--no-enforce-threshold", action="store_true")
    parser.add_argument("--num-classes", type=int, default=44)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--logic-hidden-dim", type=int, default=128)
    parser.add_argument("--logic-iterations", type=int, default=30)
    parser.add_argument("--vqvae-architecture", default="vqvae")
    parser.add_argument("--vqvae-hidden-dim", type=int, default=96)
    parser.add_argument("--vqvae-codebook-size", type=int, default=256)
    parser.add_argument("--vqvae-use-coordconv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vqvae-mrf-penalty-weight", type=float, default=0.05)
    parser.add_argument("--train-dungeon-ids", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--val-dungeon-ids", type=int, nargs="+", default=[9])
    parser.add_argument("--variants", type=int, nargs="+", default=[1, 2])
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
