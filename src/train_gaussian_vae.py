"""
H-MOLQD Block II: Gaussian VAE pre-training.

Standalone training entrypoint for the continuous latent baseline. The file is
deliberately separate from `train_vqvae.py` so the existing VQ-VAE path stays
unchanged.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

# Ensure project root on path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_system import merge_config, seed_everything
from src.core.gaussian_vae import GaussianVAETrainer, create_gaussian_vae
from src.utils.checkpoint import (
    LATEST_RESUME_FILENAME,
    atomic_torch_save,
    enforce_checkpoint_storage_budget,
    log_checkpoint_artifact,
    prune_checkpoints,
    resolve_resume_checkpoint,
    write_checkpoint_metadata,
)
from src.utils.data_loading import dataloader_runtime_kwargs
from src.utils.model_capacity import count_parameters, log_capacity_guardrails
from src.zelda_data.zelda_loader import create_dataloader

logger = logging.getLogger(__name__)


def grids_to_onehot(batch: torch.Tensor, num_classes: int = 44) -> torch.Tensor:
    """Convert normalized grid batches to semantic one-hot tensors."""
    tile_ids = (batch.squeeze(1) * (num_classes - 1)).round().long().clamp(0, num_classes - 1)
    batch_size, height, width = tile_ids.shape
    onehot = torch.zeros(
        batch_size,
        num_classes,
        height,
        width,
        device=batch.device,
        dtype=torch.float32,
    )
    onehot.scatter_(1, tile_ids.unsqueeze(1), 1.0)
    return onehot


def split_dataset_for_gaussian_vae_validation(
    dataset,
    *,
    validation_fraction: float,
    seed: int | None,
):
    """Deterministically split the dataset into train and validation subsets."""
    dataset_size = len(dataset)
    fraction = float(max(0.0, validation_fraction))
    if dataset_size < 2 or fraction <= 0.0:
        return dataset, None

    val_size = int(round(dataset_size * fraction))
    val_size = max(1, min(dataset_size - 1, val_size))
    train_size = dataset_size - val_size
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(int(seed))
    train_subset, val_subset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )
    return train_subset, val_subset


def gaussian_vae_training_kwargs_from_resolved_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build Gaussian-VAE trainer kwargs from the validated global config payload."""
    stage = config["vqvae"]
    dataset = config["dataset"]
    runtime = config["runtime"]
    output_dir = Path(runtime["output_dir"])
    return {
        "data_dir": dataset["data_dir"],
        "epochs": stage["epochs"],
        "batch_size": dataset["batch_size"],
        "lr": stage["learning_rate"],
        "scheduler_eta_min": stage["scheduler_eta_min"],
        "weight_decay": stage["weight_decay"],
        "grad_clip_norm": stage["grad_clip_norm"],
        "latent_dim": stage["latent_dim"],
        "hidden_dim": stage["hidden_dim"],
        "num_classes": dataset["num_classes"],
        "rare_tile_weight": stage["rare_tile_weight"],
        "kl_weight": float(stage.get("kl_weight", 1.0)),
        "use_coordconv": stage["use_coordconv"],
        "mrf_penalty_weight": stage["mrf_penalty_weight"],
        "validation_fraction": stage["validation_fraction"],
        "validation_max_batches": stage["validation_max_batches"],
        "best_checkpoint_metric": stage["best_checkpoint_metric"],
        "min_samples_per_epoch": dataset["min_samples_per_epoch"],
        "save_dir": str(output_dir / "checkpoints" / "gaussian_vae"),
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


def _default_gaussian_vae_training_kwargs() -> Dict[str, Any]:
    """Preserve historical standalone defaults when no YAML config is provided."""
    return {
        "data_dir": None,
        "epochs": 300,
        "batch_size": 4,
        "lr": 3e-4,
        "scheduler_eta_min": 1e-6,
        "weight_decay": 1e-5,
        "grad_clip_norm": 1.0,
        "latent_dim": 64,
        "hidden_dim": 128,
        "num_classes": 44,
        "rare_tile_weight": 5.0,
        "kl_weight": 1.0,
        "use_coordconv": True,
        "mrf_penalty_weight": 0.05,
        "validation_fraction": 0.1,
        "validation_max_batches": 16,
        "best_checkpoint_metric": "val_loss",
        "min_samples_per_epoch": 64,
        "save_dir": "checkpoints/gaussian_vae",
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


def _legacy_gaussian_vae_overrides_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}

    def _set(name: str, value: Any) -> None:
        if value is None:
            return
        overrides[name] = value

    _set("data_dir", getattr(args, "data_dir", None))
    _set("epochs", getattr(args, "epochs", None))
    _set("batch_size", getattr(args, "batch_size", None))
    _set("lr", getattr(args, "lr", None))
    _set("scheduler_eta_min", getattr(args, "scheduler_eta_min", None))
    _set("weight_decay", getattr(args, "weight_decay", None))
    _set("grad_clip_norm", getattr(args, "grad_clip_norm", None))
    _set("latent_dim", getattr(args, "latent_dim", None))
    _set("hidden_dim", getattr(args, "hidden_dim", None))
    _set("num_classes", getattr(args, "num_classes", None))
    _set("rare_tile_weight", getattr(args, "rare_tile_weight", None))
    _set("kl_weight", getattr(args, "kl_weight", None))
    _set("use_coordconv", getattr(args, "use_coordconv", None))
    _set("mrf_penalty_weight", getattr(args, "mrf_penalty_weight", None))
    _set("validation_fraction", getattr(args, "validation_fraction", None))
    _set("validation_max_batches", getattr(args, "validation_max_batches", None))
    _set("best_checkpoint_metric", getattr(args, "best_checkpoint_metric", None))
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


def build_gaussian_vae_training_args_from_args(args: argparse.Namespace) -> SimpleNamespace:
    """Resolve the standalone Gaussian-VAE CLI into the effective training namespace."""
    merged_kwargs = _default_gaussian_vae_training_kwargs()
    config_path = getattr(args, "config", None)
    if config_path:
        resolved = merge_config(yaml_path=str(config_path), cli_overrides=None)
        merged_kwargs.update(gaussian_vae_training_kwargs_from_resolved_config(resolved))
        merged_kwargs["config"] = str(config_path)

    merged_kwargs.update(_legacy_gaussian_vae_overrides_from_args(args))

    if not merged_kwargs.get("data_dir"):
        raise ValueError("Gaussian VAE training requires --data-dir or --config with dataset.data_dir.")

    return SimpleNamespace(**merged_kwargs)


def evaluate_gaussian_vae_loader(
    model: torch.nn.Module,
    trainer: GaussianVAETrainer,
    loader: DataLoader,
    *,
    num_classes: int,
    device: torch.device,
    max_batches: int | None = None,
) -> Dict[str, float]:
    """Run a bounded evaluation pass on a loader."""
    totals = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "kl_loss": 0.0,
        "kl_loss_weighted": 0.0,
        "illegal_adjacency_penalty": 0.0,
        "accuracy": 0.0,
    }
    batches = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device)
            x_onehot = grids_to_onehot(batch, num_classes=num_classes)
            info = trainer.eval_step(x_onehot)
            for key in totals:
                totals[key] += float(info.get(key, 0.0))
            batches += 1
            if max_batches is not None and batches >= int(max_batches):
                break

    for key in totals:
        totals[key] /= max(1, batches)
    totals["batches"] = float(batches)
    return totals


def build_gaussian_vae_scheduler(
    trainer: GaussianVAETrainer,
    *,
    epochs: int,
    eta_min: float,
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    """Create the resumable epoch-level cosine scheduler for Gaussian-VAE training."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer,
        T_max=max(1, int(epochs)),
        eta_min=float(max(0.0, eta_min)),
    )


def train_gaussian_vae(args):
    """Full Gaussian-VAE pre-training loop."""

    args.epochs = 2 if bool(getattr(args, "quick", False)) else int(args.epochs)

    if args.seed is not None:
        resolved_seed = seed_everything(int(args.seed))
        logger.info("Gaussian-VAE trainer seeds initialized: seed=%d", resolved_seed)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    logger.info("Device: %s", device)

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
    train_dataset, val_dataset = split_dataset_for_gaussian_vae_validation(
        dataset,
        validation_fraction=float(getattr(args, "validation_fraction", 0.0)),
        seed=getattr(args, "seed", None),
    )
    sample_kind = "rooms" if room_level else "dungeons"
    logger.info("Dataset: %d %s", len(dataset), sample_kind)

    if len(dataset) == 0:
        logger.error("No %s samples found! Check --data-dir path.", sample_kind)
        sys.exit(1)

    effective_size = max(len(train_dataset), args.min_samples_per_epoch)
    sampler = torch.utils.data.RandomSampler(
        train_dataset,
        replacement=True,
        num_samples=effective_size,
        generator=(torch.Generator().manual_seed(int(args.seed)) if args.seed is not None else None),
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=bool(getattr(args, "drop_last", True)),
        **dataloader_runtime_kwargs(
            num_workers=int(getattr(args, "num_workers", 0)),
            pin_memory=bool(getattr(args, "pin_memory", torch.cuda.is_available())),
        ),
    )
    eval_source = val_dataset if val_dataset is not None else train_dataset
    eval_split_name = "val" if val_dataset is not None else "train"
    eval_loader = DataLoader(
        eval_source,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **dataloader_runtime_kwargs(
            num_workers=int(getattr(args, "num_workers", 0)),
            pin_memory=bool(getattr(args, "pin_memory", torch.cuda.is_available())),
        ),
    )
    logger.info("Effective samples/epoch: %d, batches/epoch: %d", effective_size, len(dataloader))
    logger.info(
        "Gaussian-VAE split: train=%d %s | %s=%d %s | best_metric=%s",
        len(train_dataset),
        sample_kind,
        eval_split_name,
        len(eval_source),
        sample_kind,
        str(getattr(args, "best_checkpoint_metric", "val_loss")),
    )

    model = create_gaussian_vae(
        num_classes=int(getattr(args, "num_classes", 44)),
        latent_dim=int(args.latent_dim),
        hidden_dim=int(getattr(args, "hidden_dim", 128)),
        rare_tile_weight=float(getattr(args, "rare_tile_weight", 5.0)),
        kl_weight=float(getattr(args, "kl_weight", 1.0)),
        use_coordconv=bool(getattr(args, "use_coordconv", True)),
        mrf_penalty_weight=float(args.mrf_penalty_weight),
    ).to(device)
    num_classes = int(model.num_classes)

    total_params = count_parameters(model, trainable_only=True)
    logger.info("Gaussian-VAE parameters: %d", total_params)
    log_capacity_guardrails(
        logger,
        stage_name="Gaussian-VAE trainer",
        dataset_size=len(dataset),
        param_groups={"gaussian_vae": total_params},
        recommended_config="configs/zelda_hmolqd.yaml",
        capacity_knobs="gaussian_vae.hidden_dim, gaussian_vae.latent_dim, gaussian_vae.kl_weight",
    )

    trainer = GaussianVAETrainer(
        model,
        lr=args.lr,
        weight_decay=float(getattr(args, "weight_decay", 1e-5)),
        grad_clip_norm=float(getattr(args, "grad_clip_norm", 1.0)),
    )
    scheduler = build_gaussian_vae_scheduler(
        trainer,
        epochs=args.epochs,
        eta_min=float(getattr(args, "scheduler_eta_min", 1e-6)),
    )

    start_epoch = 0
    resume_path = resolve_resume_checkpoint(
        explicit_path=getattr(args, "resume", None),
        checkpoint_dir=str(getattr(args, "save_dir", "checkpoints/gaussian_vae")),
        auto_resume=bool(getattr(args, "auto_resume", True)),
        latest_filename=LATEST_RESUME_FILENAME,
    )
    if resume_path is not None:
        ckpt = torch.load(str(resume_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_metric_name = str(
            ckpt.get(
                "best_metric_name",
                "val_loss" if float(getattr(args, "validation_fraction", 0.0)) > 0.0 else "train_loss",
            )
        )
        best_metric_value = float(
            ckpt.get(
                "best_metric_value",
                ckpt.get("best_loss", ckpt.get("loss", float("inf"))),
            )
        )
        logger.info("Resumed from %s (epoch %d)", resume_path, start_epoch)
    else:
        best_metric_name = "val_loss" if float(getattr(args, "validation_fraction", 0.0)) > 0.0 else "train_loss"
        best_metric_value = float("inf")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_model_type = "gaussian_vae"
    checkpoint_resume_type = "gaussian_vae_resume"
    history = []

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_metrics = {
            "loss": 0.0,
            "recon_loss": 0.0,
            "kl_loss": 0.0,
            "kl_loss_weighted": 0.0,
            "illegal_adjacency_penalty": 0.0,
        }
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]

            batch = batch.to(device)
            x_onehot = grids_to_onehot(batch, num_classes=num_classes)

            step_out = trainer.train_step(x_onehot, return_metrics=True)
            if isinstance(step_out, tuple):
                _loss, metrics = step_out
            else:
                _loss = float(step_out)
                metrics = {
                    "loss": _loss,
                    "recon_loss": 0.0,
                    "kl_loss": 0.0,
                    "kl_loss_weighted": 0.0,
                    "illegal_adjacency_penalty": 0.0,
                }

            for key in epoch_metrics:
                epoch_metrics[key] += metrics.get(key, 0.0)
            num_batches += 1

            if batch_idx % max(1, len(dataloader) // 5) == 0:
                logger.debug(
                    "  Epoch %d/%d | Batch %d/%d | loss=%.4f recon=%.4f kl=%.4f klw=%.4f mrf=%.4f",
                    epoch + 1,
                    args.epochs,
                    batch_idx,
                    len(dataloader),
                    metrics["loss"],
                    metrics["recon_loss"],
                    metrics["kl_loss"],
                    metrics["kl_loss_weighted"],
                    metrics.get("illegal_adjacency_penalty", 0.0),
                )

        for key in epoch_metrics:
            epoch_metrics[key] /= max(num_batches, 1)

        eval_metrics = evaluate_gaussian_vae_loader(
            model,
            trainer,
            eval_loader,
            num_classes=num_classes,
            device=device,
            max_batches=int(getattr(args, "validation_max_batches", 16)),
        )
        epoch_metrics["accuracy"] = float(eval_metrics["accuracy"])
        epoch_metrics["eval_split"] = eval_split_name
        epoch_metrics["eval_batches"] = int(eval_metrics["batches"])
        if eval_split_name == "val":
            epoch_metrics["val_loss"] = float(eval_metrics["loss"])
            epoch_metrics["val_recon_loss"] = float(eval_metrics["recon_loss"])
            epoch_metrics["val_kl_loss"] = float(eval_metrics["kl_loss"])
            epoch_metrics["val_kl_loss_weighted"] = float(eval_metrics["kl_loss_weighted"])
            epoch_metrics["val_illegal_adjacency_penalty"] = float(eval_metrics["illegal_adjacency_penalty"])
            epoch_metrics["val_accuracy"] = float(eval_metrics["accuracy"])
        else:
            epoch_metrics["train_eval_loss"] = float(eval_metrics["loss"])
            epoch_metrics["train_eval_recon_loss"] = float(eval_metrics["recon_loss"])
            epoch_metrics["train_eval_kl_loss"] = float(eval_metrics["kl_loss"])
            epoch_metrics["train_eval_kl_loss_weighted"] = float(eval_metrics["kl_loss_weighted"])
            epoch_metrics["train_eval_illegal_adjacency_penalty"] = float(eval_metrics["illegal_adjacency_penalty"])
            epoch_metrics["train_eval_accuracy"] = float(eval_metrics["accuracy"])

        scheduler.step()
        epoch_metrics["learning_rate"] = float(trainer.optimizer.param_groups[0]["lr"])

        checkpoint_metric_name = str(getattr(args, "best_checkpoint_metric", "val_loss"))
        if checkpoint_metric_name == "val_loss" and eval_split_name == "val":
            checkpoint_metric_value = float(epoch_metrics["val_loss"])
        else:
            checkpoint_metric_name = "train_loss"
            checkpoint_metric_value = float(epoch_metrics["loss"])

        logger.info(
            "Epoch %d/%d | loss=%.4f | recon=%.4f | kl=%.4f | klw=%.4f | mrf=%.4f | %s_loss=%.4f | %s_accuracy=%.3f | best_metric=%s:%.4f",
            epoch + 1,
            args.epochs,
            epoch_metrics["loss"],
            epoch_metrics["recon_loss"],
            epoch_metrics["kl_loss"],
            epoch_metrics["kl_loss_weighted"],
            epoch_metrics["illegal_adjacency_penalty"],
            eval_split_name,
            eval_metrics["loss"],
            eval_split_name,
            eval_metrics["accuracy"],
            checkpoint_metric_name,
            checkpoint_metric_value,
        )
        history.append({"epoch": epoch + 1, **epoch_metrics})

        if checkpoint_metric_value < best_metric_value:
            best_metric_name = checkpoint_metric_name
            best_metric_value = checkpoint_metric_value
            save_path = save_dir / "gaussian_vae_pretrained.pth"
            best_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": float(epoch_metrics["loss"]),
                "accuracy": float(eval_metrics["accuracy"]),
                "kl_loss": float(eval_metrics["kl_loss"]),
                "best_metric_name": str(best_metric_name),
                "best_metric_value": float(best_metric_value),
            }
            atomic_torch_save(best_payload, str(save_path))
            write_checkpoint_metadata(
                str(save_path),
                model_type=checkpoint_model_type,
                architecture={
                    "num_classes": int(num_classes),
                    "latent_dim": int(args.latent_dim),
                    "hidden_dim": int(getattr(args, "hidden_dim", 128)),
                    "kl_weight": float(getattr(args, "kl_weight", 1.0)),
                    "kl_normalized_by_latent_volume": True,
                    "rare_tile_weight": float(getattr(args, "rare_tile_weight", 5.0)),
                    "use_coordconv": bool(args.use_coordconv),
                    "mrf_penalty_weight": float(args.mrf_penalty_weight),
                    "num_res_blocks": 2,
                    "encoder_channel_mult": [1, 2, 4],
                    "decoder_channel_mult": [4, 2, 1],
                    "room_level": bool(room_level),
                    "scheduler": "CosineAnnealingLR",
                    "scheduler_eta_min": float(getattr(args, "scheduler_eta_min", 1e-6)),
                },
                extra={
                    "epoch": int(epoch + 1),
                    "loss": float(epoch_metrics["loss"]),
                    "accuracy": float(eval_metrics["accuracy"]),
                    "eval_split": eval_split_name,
                    "best_metric_name": str(best_metric_name),
                    "best_metric_value": float(best_metric_value),
                    **{
                        key: value
                        for key, value in epoch_metrics.items()
                        if key.startswith("val_") or key.startswith("train_eval_")
                    },
                },
            )
            log_checkpoint_artifact(
                logger,
                save_path,
                checkpoint_dir=save_dir,
                label=f"[BEST] Saved best model ({best_metric_name}={best_metric_value:.4f})",
            )

        resume_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": float(epoch_metrics["loss"]),
            "best_loss": float(best_metric_value),
            "best_metric_name": str(best_metric_name),
            "best_metric_value": float(best_metric_value),
            "history_length": len(history),
        }
        latest_resume = save_dir / LATEST_RESUME_FILENAME
        atomic_torch_save(resume_payload, str(latest_resume))
        write_checkpoint_metadata(
            str(latest_resume),
            model_type=checkpoint_resume_type,
            architecture={
                "num_classes": int(num_classes),
                "latent_dim": int(args.latent_dim),
                "hidden_dim": int(getattr(args, "hidden_dim", 128)),
                "kl_weight": float(getattr(args, "kl_weight", 1.0)),
                "kl_normalized_by_latent_volume": True,
                "rare_tile_weight": float(getattr(args, "rare_tile_weight", 5.0)),
                "use_coordconv": bool(args.use_coordconv),
                "mrf_penalty_weight": float(args.mrf_penalty_weight),
                "num_res_blocks": 2,
                "encoder_channel_mult": [1, 2, 4],
                "decoder_channel_mult": [4, 2, 1],
                "room_level": bool(room_level),
                "scheduler": "CosineAnnealingLR",
                "scheduler_eta_min": float(getattr(args, "scheduler_eta_min", 1e-6)),
            },
            extra={
                "epoch": int(epoch + 1),
                "checkpoint_kind": "latest_resume",
                "eval_split": eval_split_name,
                "best_metric_name": str(best_metric_name),
                "best_metric_value": float(best_metric_value),
                **{
                    key: value
                    for key, value in epoch_metrics.items()
                    if key.startswith("val_") or key.startswith("train_eval_")
                },
            },
        )
        log_checkpoint_artifact(
            logger,
            latest_resume,
            checkpoint_dir=save_dir,
            label="Saved latest Gaussian-VAE resume checkpoint",
        )

        if (epoch + 1) % args.save_every == 0:
            periodic = save_dir / f"gaussian_vae_resume_epoch{epoch + 1:04d}.pth"
            atomic_torch_save(resume_payload, str(periodic))
            write_checkpoint_metadata(
                str(periodic),
                model_type=checkpoint_resume_type,
                architecture={
                    "num_classes": int(num_classes),
                    "latent_dim": int(args.latent_dim),
                    "hidden_dim": int(getattr(args, "hidden_dim", 128)),
                    "kl_weight": float(getattr(args, "kl_weight", 1.0)),
                    "kl_normalized_by_latent_volume": True,
                    "rare_tile_weight": float(getattr(args, "rare_tile_weight", 5.0)),
                    "use_coordconv": bool(args.use_coordconv),
                    "mrf_penalty_weight": float(args.mrf_penalty_weight),
                    "num_res_blocks": 2,
                    "encoder_channel_mult": [1, 2, 4],
                    "decoder_channel_mult": [4, 2, 1],
                    "room_level": bool(room_level),
                    "scheduler": "CosineAnnealingLR",
                    "scheduler_eta_min": float(getattr(args, "scheduler_eta_min", 1e-6)),
                },
                extra={
                    "epoch": int(epoch + 1),
                    "checkpoint_kind": "retained_resume",
                    "eval_split": eval_split_name,
                    "best_metric_name": str(best_metric_name),
                    "best_metric_value": float(best_metric_value),
                    **{
                        key: value
                        for key, value in epoch_metrics.items()
                        if key.startswith("val_") or key.startswith("train_eval_")
                    },
                },
            )
            log_checkpoint_artifact(
                logger,
                periodic,
                checkpoint_dir=save_dir,
                label="Saved retained Gaussian-VAE resume checkpoint",
            )
            prune_checkpoints(
                checkpoint_dir=str(save_dir),
                pattern="gaussian_vae_resume_epoch*.pth",
                keep_last=int(getattr(args, "keep_last", 2)),
            )

        enforce_checkpoint_storage_budget(
            logger,
            checkpoint_dir=save_dir,
            budget_gb=getattr(args, "checkpoint_storage_budget_gb", None),
            warning_fraction=float(getattr(args, "checkpoint_storage_warning_fraction", 0.8)),
            cleanup_enabled=bool(getattr(args, "checkpoint_storage_cleanup_enabled", True)),
            cleanup_target_fraction=float(getattr(args, "checkpoint_storage_cleanup_target_fraction", 0.6)),
            removable_patterns=("gaussian_vae_resume_epoch*.pth",),
        )

    hist_path = save_dir / "gaussian_vae_training_history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info("Training history saved to %s", hist_path)
    logger.info("Best %s: %.4f", best_metric_name, best_metric_value)

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train a Gaussian VAE baseline for dungeon grid reconstruction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Optional YAML experiment config to inherit canonical dataset/runtime settings.")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to dungeon data (e.g. 'Data/The Legend of Zelda')")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--scheduler-eta-min", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--rare-tile-weight", type=float, default=None)
    parser.add_argument("--kl-weight", type=float, default=None,
                        help="KL-divergence weight for the continuous latent bottleneck.")
    parser.add_argument("--use-coordconv", action=argparse.BooleanOptionalAction, default=None,
                        help="Use CoordConv in the first encoder layer.")
    parser.add_argument("--mrf-penalty-weight", type=float, default=None,
                        help="Weight for differentiable illegal adjacency penalty.")
    parser.add_argument("--validation-fraction", type=float, default=None,
                        help="Held-out validation fraction for model selection; 0 disables a validation split.")
    parser.add_argument("--validation-max-batches", type=int, default=None,
                        help="Maximum number of validation mini-batches evaluated each epoch.")
    parser.add_argument("--best-checkpoint-metric", type=str, default=None,
                        choices=["val_loss", "train_loss"],
                        help="Metric used to select the best Gaussian-VAE checkpoint.")
    parser.add_argument("--min-samples-per-epoch", type=int, default=None,
                        help="Minimum effective samples per epoch (upsampled for small datasets)")
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
                        help="Train on canonical room crops instead of stitched full dungeons.")
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
        args = build_gaussian_vae_training_args_from_args(raw_args)
    except ValueError as exc:
        parser.error(str(exc))
    train_gaussian_vae(args)


if __name__ == "__main__":
    main()
