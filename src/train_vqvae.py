"""
H-MOLQD Stage 1: VQ-VAE Pre-training
=====================================

Trains the Semantic VQ-VAE (Block II) to reconstruct dungeon grids
before the latent diffusion model can operate on meaningful latent codes.

This MUST be run before diffusion training:
    python -m src.train_vqvae --data-dir "data/The Legend of Zelda" --epochs 300

Then pass the checkpoint to diffusion training:
    python -m src.train_diffusion --vqvae-checkpoint checkpoints/vqvae_pretrained.pth ...
"""

import sys
import argparse
import logging
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

# Ensure project root on path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.vqvae import create_vqvae, VQVAETrainer
from src.config_system import merge_config, seed_everything
from src.utils.model_capacity import count_parameters, log_capacity_guardrails
from src.zelda_data.zelda_loader import create_dataloader
from src.utils.checkpoint import (
    LATEST_RESUME_FILENAME,
    atomic_torch_save,
    enforce_checkpoint_storage_budget,
    log_checkpoint_artifact,
    prune_checkpoints,
    resolve_resume_checkpoint,
    write_checkpoint_metadata,
)

logger = logging.getLogger(__name__)


def grids_to_onehot(batch: torch.Tensor, num_classes: int = 44) -> torch.Tensor:
    """
    Convert normalised grid batch to one-hot encoding.

    Data loader returns [B, 1, H, W] with values in [0, 1].
    The normalisation divides by a fixed constant (43 = TileID.PUZZLE, the
    highest tile ID).  To recover integer IDs we multiply by 43, round, and
    clamp; this gives an exact round-trip for all dungeons.

    Returns [B, C, H, W] float32 one-hot.
    """
    tile_ids = (batch.squeeze(1) * (num_classes - 1)).round().long().clamp(0, num_classes - 1)
    onehot = F.one_hot(tile_ids, num_classes=num_classes)  # [B, H, W, C]
    return onehot.permute(0, 3, 1, 2).float()              # [B, C, H, W]


def vqvae_training_kwargs_from_resolved_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build standalone VQ-VAE trainer kwargs from the validated global config payload."""
    stage = config["vqvae"]
    dataset = config["dataset"]
    runtime = config["runtime"]
    return {
        "data_dir": dataset["data_dir"],
        "epochs": stage["epochs"],
        "batch_size": dataset["batch_size"],
        "lr": stage["learning_rate"],
        "weight_decay": stage["weight_decay"],
        "grad_clip_norm": stage["grad_clip_norm"],
        "latent_dim": stage["latent_dim"],
        "hidden_dim": stage["hidden_dim"],
        "codebook_size": stage["codebook_size"],
        "num_classes": dataset["num_classes"],
        "commitment_cost": stage["commitment_cost"],
        "rare_tile_weight": stage["rare_tile_weight"],
        "use_ema": stage["use_ema"],
        "use_coordconv": stage["use_coordconv"],
        "mrf_penalty_weight": stage["mrf_penalty_weight"],
        "dead_code_reset_interval": stage["dead_code_reset_interval"],
        "dead_code_threshold": stage["dead_code_threshold"],
        "dead_code_warmup_steps": stage["dead_code_warmup_steps"],
        "protect_active_codes_during_reset": stage["protect_active_codes_during_reset"],
        "max_dead_code_resets_per_event": stage["max_dead_code_resets_per_event"],
        "min_samples_per_epoch": dataset["min_samples_per_epoch"],
        "save_dir": stage["checkpoint_dir"],
        "save_every": stage["save_every"],
        "keep_last": stage["keep_last"],
        "num_workers": dataset["num_workers"],
        "pin_memory": dataset["pin_memory"],
        "drop_last": dataset["drop_last"],
        "use_vglc": dataset["use_vglc"],
        "normalize": dataset["normalize"],
        "room_level": dataset["room_level"],
        "seed": runtime["seed"],
        "auto_resume": runtime["auto_resume"],
        "checkpoint_storage_budget_gb": runtime["checkpoint_storage_budget_gb"],
        "checkpoint_storage_warning_fraction": runtime["checkpoint_storage_warning_fraction"],
        "checkpoint_storage_cleanup_enabled": runtime["checkpoint_storage_cleanup_enabled"],
        "checkpoint_storage_cleanup_target_fraction": runtime["checkpoint_storage_cleanup_target_fraction"],
        "resume": stage["resume_checkpoint"] or runtime["resume"],
        "device": runtime["device"],
        "verbose": runtime["verbose"],
        "quick": runtime["quick"],
    }


def _default_vqvae_training_kwargs() -> Dict[str, Any]:
    """Preserve historical standalone defaults when no YAML config is provided."""
    return {
        "data_dir": None,
        "epochs": 300,
        "batch_size": 4,
        "lr": 3e-4,
        "weight_decay": 1e-5,
        "grad_clip_norm": 1.0,
        "latent_dim": 64,
        "hidden_dim": 128,
        "codebook_size": 512,
        "num_classes": 44,
        "commitment_cost": 0.25,
        "rare_tile_weight": 5.0,
        "use_ema": True,
        "use_coordconv": True,
        "mrf_penalty_weight": 0.05,
        "dead_code_reset_interval": 100,
        "dead_code_threshold": 0.05,
        "dead_code_warmup_steps": 500,
        "protect_active_codes_during_reset": True,
        "max_dead_code_resets_per_event": 16,
        "min_samples_per_epoch": 64,
        "save_dir": "checkpoints",
        "save_every": 50,
        "keep_last": 2,
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": True,
        "use_vglc": True,
        "normalize": True,
        "room_level": True,
        "seed": None,
        "auto_resume": True,
        "checkpoint_storage_budget_gb": None,
        "checkpoint_storage_warning_fraction": 0.8,
        "checkpoint_storage_cleanup_enabled": True,
        "checkpoint_storage_cleanup_target_fraction": 0.6,
        "resume": None,
        "device": "auto",
        "verbose": False,
        "quick": False,
        "config": None,
    }


def _legacy_vqvae_overrides_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Collect only explicitly provided legacy CLI overrides."""
    overrides: Dict[str, Any] = {}

    def _set(name: str, value: Any) -> None:
        if value is None:
            return
        overrides[name] = value

    _set("data_dir", getattr(args, "data_dir", None))
    _set("epochs", getattr(args, "epochs", None))
    _set("batch_size", getattr(args, "batch_size", None))
    _set("lr", getattr(args, "lr", None))
    _set("weight_decay", getattr(args, "weight_decay", None))
    _set("grad_clip_norm", getattr(args, "grad_clip_norm", None))
    _set("latent_dim", getattr(args, "latent_dim", None))
    _set("hidden_dim", getattr(args, "hidden_dim", None))
    _set("codebook_size", getattr(args, "codebook_size", None))
    _set("num_classes", getattr(args, "num_classes", None))
    _set("commitment_cost", getattr(args, "commitment_cost", None))
    _set("rare_tile_weight", getattr(args, "rare_tile_weight", None))
    _set("use_ema", getattr(args, "use_ema", None))
    _set("use_coordconv", getattr(args, "use_coordconv", None))
    _set("mrf_penalty_weight", getattr(args, "mrf_penalty_weight", None))
    _set("dead_code_reset_interval", getattr(args, "dead_code_reset_interval", None))
    _set("dead_code_threshold", getattr(args, "dead_code_threshold", None))
    _set("dead_code_warmup_steps", getattr(args, "dead_code_warmup_steps", None))
    _set("protect_active_codes_during_reset", getattr(args, "protect_active_codes_during_reset", None))
    _set("max_dead_code_resets_per_event", getattr(args, "max_dead_code_resets_per_event", None))
    _set("min_samples_per_epoch", getattr(args, "min_samples_per_epoch", None))
    _set("save_dir", getattr(args, "save_dir", None))
    _set("save_every", getattr(args, "save_every", None))
    _set("keep_last", getattr(args, "keep_last", None))
    _set("num_workers", getattr(args, "num_workers", None))
    _set("pin_memory", getattr(args, "pin_memory", None))
    _set("drop_last", getattr(args, "drop_last", None))
    _set("use_vglc", getattr(args, "use_vglc", None))
    _set("normalize", getattr(args, "normalize", None))
    _set("room_level", getattr(args, "room_level", None))
    _set("seed", getattr(args, "seed", None))
    _set("auto_resume", getattr(args, "auto_resume", None))
    _set("checkpoint_storage_budget_gb", getattr(args, "checkpoint_storage_budget_gb", None))
    _set("checkpoint_storage_warning_fraction", getattr(args, "checkpoint_storage_warning_fraction", None))
    _set("checkpoint_storage_cleanup_enabled", getattr(args, "checkpoint_storage_cleanup_enabled", None))
    _set("checkpoint_storage_cleanup_target_fraction", getattr(args, "checkpoint_storage_cleanup_target_fraction", None))
    _set("resume", getattr(args, "resume", None))
    _set("device", getattr(args, "device", None))
    _set("verbose", getattr(args, "verbose", None))
    _set("quick", getattr(args, "quick", None))
    return overrides


def build_vqvae_training_args_from_args(args: argparse.Namespace) -> SimpleNamespace:
    """Resolve the standalone VQ-VAE CLI into the effective training namespace."""
    merged_kwargs = _default_vqvae_training_kwargs()
    config_path = getattr(args, "config", None)
    if config_path:
        resolved = merge_config(yaml_path=str(config_path), cli_overrides=None)
        merged_kwargs.update(vqvae_training_kwargs_from_resolved_config(resolved))
        merged_kwargs["config"] = str(config_path)

    merged_kwargs.update(_legacy_vqvae_overrides_from_args(args))

    if not merged_kwargs.get("data_dir"):
        raise ValueError("VQ-VAE training requires --data-dir or --config with dataset.data_dir.")

    return SimpleNamespace(**merged_kwargs)


def train_vqvae(args):
    """Full VQ-VAE pre-training loop."""

    args.epochs = 2 if bool(getattr(args, "quick", False)) else int(args.epochs)

    if args.seed is not None:
        resolved_seed = seed_everything(int(args.seed))
        logger.info("VQ-VAE trainer seeds initialized: seed=%d", resolved_seed)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    logger.info(f"Device: {device}")

    # ------------------------------------------------------------------
    # Dataset - use VGLC mode, same as diffusion training
    # ------------------------------------------------------------------
    room_level = bool(getattr(args, "room_level", True))
    base_loader = create_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(getattr(args, "num_workers", 0)),
        pin_memory=bool(getattr(args, "pin_memory", torch.cuda.is_available())),
        drop_last=bool(getattr(args, "drop_last", True)),
        use_vglc=bool(getattr(args, "use_vglc", True)),
        normalize=bool(getattr(args, "normalize", True)),
        room_level=room_level,
        load_graphs=False,
    )
    dataset = base_loader.dataset
    sample_kind = "rooms" if room_level else "dungeons"
    logger.info("Dataset: %d %s", len(dataset), sample_kind)

    if len(dataset) == 0:
        logger.error("No %s samples found! Check --data-dir path.", sample_kind)
        sys.exit(1)

    # Small dataset -> duplicate to fill an epoch with more gradient steps
    effective_size = max(len(dataset), args.min_samples_per_epoch)
    sampler = torch.utils.data.RandomSampler(
        dataset,
        replacement=True,
        num_samples=effective_size,
        generator=(torch.Generator().manual_seed(int(args.seed)) if args.seed is not None else None),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=int(getattr(args, "num_workers", 0)),
        pin_memory=bool(getattr(args, "pin_memory", torch.cuda.is_available())),
        drop_last=bool(getattr(args, "drop_last", True)),
    )
    logger.info(f"Effective samples/epoch: {effective_size}, "
                f"batches/epoch: {len(dataloader)}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = create_vqvae(
        num_classes=int(getattr(args, "num_classes", 44)),
        codebook_size=args.codebook_size,
        latent_dim=args.latent_dim,
        hidden_dim=int(getattr(args, "hidden_dim", 128)),
        commitment_cost=float(getattr(args, "commitment_cost", 0.25)),
        rare_tile_weight=float(getattr(args, "rare_tile_weight", 5.0)),
        use_ema=bool(getattr(args, "use_ema", True)),
        use_coordconv=bool(args.use_coordconv),
        mrf_penalty_weight=float(args.mrf_penalty_weight),
        dead_code_reset_interval=int(getattr(args, "dead_code_reset_interval", 100)),
        dead_code_threshold=float(getattr(args, "dead_code_threshold", 0.05)),
        dead_code_warmup_steps=int(getattr(args, "dead_code_warmup_steps", 500)),
        protect_active_codes_during_reset=bool(getattr(args, "protect_active_codes_during_reset", True)),
        max_dead_code_resets_per_event=int(getattr(args, "max_dead_code_resets_per_event", 16)),
    ).to(device)
    num_classes = int(model.num_classes)

    total_params = count_parameters(model, trainable_only=True)
    logger.info(f"VQ-VAE parameters: {total_params:,}")
    log_capacity_guardrails(
        logger,
        stage_name="VQ-VAE trainer",
        dataset_size=len(dataset),
        param_groups={"vqvae": total_params},
        recommended_config="configs/zelda_hmolqd.yaml",
        capacity_knobs="vqvae.hidden_dim, vqvae.latent_dim, vqvae.codebook_size",
    )

    trainer = VQVAETrainer(
        model,
        lr=args.lr,
        weight_decay=float(getattr(args, "weight_decay", 1e-5)),
        grad_clip_norm=float(getattr(args, "grad_clip_norm", 1.0)),
    )

    # ------------------------------------------------------------------
    # Checkpoint resume
    # ------------------------------------------------------------------
    start_epoch = 0
    resume_path = resolve_resume_checkpoint(
        explicit_path=getattr(args, "resume", None),
        checkpoint_dir=str(getattr(args, "save_dir", "checkpoints")),
        auto_resume=bool(getattr(args, "auto_resume", True)),
        latest_filename=LATEST_RESUME_FILENAME,
    )
    if resume_path is not None:
        ckpt = torch.load(str(resume_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = float(ckpt.get("best_loss", ckpt.get("loss", float("inf"))))
        logger.info(f"Resumed from {resume_path} (epoch {start_epoch})")
    else:
        best_loss = float("inf")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history = []

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_metrics = {
            "loss": 0.0,
            "recon_loss": 0.0,
            "vq_loss": 0.0,
            "illegal_adjacency_penalty": 0.0,
            "perplexity": 0.0,
        }
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # Handle (tensor, graph_dict) tuples
            if isinstance(batch, (list, tuple)):
                batch = batch[0]

            batch = batch.to(device)
            x_onehot = grids_to_onehot(batch, num_classes=num_classes)

            # Forward / backward. Some trainer implementations return only scalar loss.
            step_out = trainer.train_step(x_onehot, return_metrics=True)
            if isinstance(step_out, tuple):
                _loss, metrics = step_out
            else:
                _loss = float(step_out)
                metrics = {
                    "loss": _loss,
                    "recon_loss": 0.0,
                    "vq_loss": 0.0,
                    "illegal_adjacency_penalty": 0.0,
                    "perplexity": 0.0,
                }

            for k in epoch_metrics:
                epoch_metrics[k] += metrics.get(k, 0.0)
            num_batches += 1

            if batch_idx % max(1, len(dataloader) // 5) == 0:
                logger.debug(
                    f"  Epoch {epoch+1}/{args.epochs} | Batch {batch_idx}/{len(dataloader)} | "
                    f"loss={metrics['loss']:.4f} recon={metrics['recon_loss']:.4f} "
                    f"vq={metrics['vq_loss']:.4f} "
                    f"mrf={metrics.get('illegal_adjacency_penalty', 0.0):.4f} "
                    f"perp={metrics['perplexity']:.1f}"
                )

        # Average metrics
        for k in epoch_metrics:
            epoch_metrics[k] /= max(num_batches, 1)

        # Evaluation accuracy
        model.eval()
        eval_acc = 0.0
        eval_n = 0
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(device)
                x_onehot = grids_to_onehot(batch, num_classes=num_classes)
                info = trainer.eval_step(x_onehot)
                eval_acc += info["accuracy"]
                eval_n += 1
                if eval_n >= 5:  # cap eval batches
                    break
        eval_acc /= max(eval_n, 1)
        epoch_metrics["accuracy"] = eval_acc

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"loss={epoch_metrics['loss']:.4f} | "
            f"recon={epoch_metrics['recon_loss']:.4f} | "
            f"vq={epoch_metrics['vq_loss']:.4f} | "
            f"mrf={epoch_metrics['illegal_adjacency_penalty']:.4f} | "
            f"perplexity={epoch_metrics['perplexity']:.1f} | "
            f"accuracy={eval_acc:.3f}"
        )
        history.append({"epoch": epoch + 1, **epoch_metrics})

        # Save best
        if epoch_metrics["loss"] < best_loss:
            best_loss = epoch_metrics["loss"]
            save_path = save_dir / "vqvae_pretrained.pth"
            best_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "loss": best_loss,
                "accuracy": eval_acc,
                "perplexity": epoch_metrics["perplexity"],
            }
            atomic_torch_save(best_payload, str(save_path))
            write_checkpoint_metadata(
                str(save_path),
                model_type="vqvae",
                architecture={
                    "latent_dim": int(args.latent_dim),
                    "codebook_size": int(args.codebook_size),
                    "use_coordconv": bool(args.use_coordconv),
                    "mrf_penalty_weight": float(args.mrf_penalty_weight),
                    "dead_code_reset_interval": int(args.dead_code_reset_interval),
                    "dead_code_threshold": float(args.dead_code_threshold),
                    "dead_code_warmup_steps": int(args.dead_code_warmup_steps),
                    "protect_active_codes_during_reset": bool(args.protect_active_codes_during_reset),
                    "max_dead_code_resets_per_event": int(args.max_dead_code_resets_per_event),
                    "room_level": bool(room_level),
                },
                extra={
                    "epoch": int(epoch + 1),
                    "loss": float(best_loss),
                    "accuracy": float(eval_acc),
                },
            )
            log_checkpoint_artifact(
                logger,
                save_path,
                checkpoint_dir=save_dir,
                label=f"[BEST] Saved best model (loss={best_loss:.4f})",
            )

        # Periodic checkpoint
        resume_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "loss": epoch_metrics["loss"],
            "best_loss": float(best_loss),
            "history_length": len(history),
        }
        latest_resume = save_dir / LATEST_RESUME_FILENAME
        atomic_torch_save(resume_payload, str(latest_resume))
        write_checkpoint_metadata(
            str(latest_resume),
            model_type="vqvae_resume",
            architecture={
                "latent_dim": int(args.latent_dim),
                "codebook_size": int(args.codebook_size),
                "use_coordconv": bool(args.use_coordconv),
                "dead_code_reset_interval": int(args.dead_code_reset_interval),
                "dead_code_threshold": float(args.dead_code_threshold),
                "dead_code_warmup_steps": int(args.dead_code_warmup_steps),
                "protect_active_codes_during_reset": bool(args.protect_active_codes_during_reset),
                "max_dead_code_resets_per_event": int(args.max_dead_code_resets_per_event),
                "room_level": bool(room_level),
            },
            extra={
                "epoch": int(epoch + 1),
                "checkpoint_kind": "latest_resume",
            },
        )
        log_checkpoint_artifact(
            logger,
            latest_resume,
            checkpoint_dir=save_dir,
            label="Saved latest VQ-VAE resume checkpoint",
        )
        if (epoch + 1) % args.save_every == 0:
            periodic = save_dir / f"vqvae_resume_epoch{epoch+1:04d}.pth"
            atomic_torch_save(resume_payload, str(periodic))
            write_checkpoint_metadata(
                str(periodic),
                model_type="vqvae_resume",
                architecture={
                    "latent_dim": int(args.latent_dim),
                    "codebook_size": int(args.codebook_size),
                    "use_coordconv": bool(args.use_coordconv),
                    "mrf_penalty_weight": float(args.mrf_penalty_weight),
                    "dead_code_reset_interval": int(args.dead_code_reset_interval),
                    "dead_code_threshold": float(args.dead_code_threshold),
                    "dead_code_warmup_steps": int(args.dead_code_warmup_steps),
                    "protect_active_codes_during_reset": bool(args.protect_active_codes_during_reset),
                    "max_dead_code_resets_per_event": int(args.max_dead_code_resets_per_event),
                    "room_level": bool(room_level),
                },
                extra={
                    "epoch": int(epoch + 1),
                    "loss": float(epoch_metrics["loss"]),
                    "checkpoint_kind": "retained_resume",
                },
            )
            log_checkpoint_artifact(
                logger,
                periodic,
                checkpoint_dir=save_dir,
                label="Saved retained VQ-VAE resume checkpoint",
            )
            prune_checkpoints(
                checkpoint_dir=str(save_dir),
                pattern="vqvae_resume_epoch*.pth",
                keep_last=int(getattr(args, "keep_last", 2)),
            )
        enforce_checkpoint_storage_budget(
            logger,
            checkpoint_dir=save_dir,
            budget_gb=getattr(args, "checkpoint_storage_budget_gb", None),
            warning_fraction=float(getattr(args, "checkpoint_storage_warning_fraction", 0.8)),
            cleanup_enabled=bool(getattr(args, "checkpoint_storage_cleanup_enabled", True)),
            cleanup_target_fraction=float(getattr(args, "checkpoint_storage_cleanup_target_fraction", 0.6)),
            removable_patterns=("vqvae_resume_epoch*.pth",),
        )

    # Save training history
    hist_path = save_dir / "vqvae_training_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {hist_path}")
    logger.info(f"Best loss: {best_loss:.4f}")

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train VQ-VAE (Block II) for dungeon grid reconstruction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Optional YAML experiment config to inherit canonical dataset/runtime/VQ-VAE settings.")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to dungeon data (e.g. 'data/The Legend of Zelda')")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--codebook-size", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--commitment-cost", type=float, default=None)
    parser.add_argument("--rare-tile-weight", type=float, default=None)
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-coordconv", action=argparse.BooleanOptionalAction, default=None,
                        help="Use CoordConv in first VQ-VAE encoder layer.")
    parser.add_argument("--mrf-penalty-weight", type=float, default=None,
                        help="Weight for differentiable illegal adjacency penalty.")
    parser.add_argument("--dead-code-reset-interval", type=int, default=None,
                        help="Check for dead VQ codes every N optimizer steps.")
    parser.add_argument("--dead-code-threshold", type=float, default=None,
                        help="EMA assignment-count threshold below which a code is considered dead.")
    parser.add_argument("--dead-code-warmup-steps", type=int, default=None,
                        help="Do not reset VQ codes until at least this many optimizer steps have elapsed.")
    parser.add_argument("--protect-active-codes-during-reset", action=argparse.BooleanOptionalAction, default=None,
                        help="Never reset codes that are still active in the current batch.")
    parser.add_argument("--max-dead-code-resets-per-event", type=int, default=None,
                        help="Maximum number of VQ codes to reset in one maintenance event; 0 disables the cap.")
    parser.add_argument("--min-samples-per-epoch", type=int, default=None,
                        help="Minimum samples per epoch (upsampled for small datasets)")
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=None,
                        help="Save periodic checkpoint every N epochs")
    parser.add_argument("--keep-last", type=int, default=None,
                        help="Retain at most N full resume checkpoints besides latest_resume.pth")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--drop-last", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-vglc", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--room-level", action=argparse.BooleanOptionalAction, default=None,
                        help="Train Block II on canonical room crops instead of stitched full dungeons.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Deterministic seed for reproducible A/B runs.")
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=None,
                        help="Automatically resume from save_dir/latest_resume.pth when present.")
    parser.add_argument("--checkpoint-storage-budget-gb", type=float, default=None,
                        help="Optional checkpoint storage budget in GB for this stage.")
    parser.add_argument("--checkpoint-storage-warning-fraction", type=float, default=None,
                        help="Warn when checkpoint usage reaches this fraction of the storage budget.")
    parser.add_argument("--checkpoint-storage-cleanup-enabled", action=argparse.BooleanOptionalAction, default=None,
                        help="Automatically delete retained resume checkpoints when over budget.")
    parser.add_argument("--checkpoint-storage-cleanup-target-fraction", type=float, default=None,
                        help="Cleanup target fraction of the storage budget after automatic pruning.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default=None,
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--quick", action=argparse.BooleanOptionalAction, default=None,
                        help="Short smoke-test mode that truncates training to two epochs.")
    parser.add_argument("-v", "--verbose", action="store_true", default=None)

    raw_args = parser.parse_args()
    try:
        args = build_vqvae_training_args_from_args(raw_args)
    except ValueError as exc:
        parser.error(str(exc))
    train_vqvae(args)


if __name__ == "__main__":
    main()

