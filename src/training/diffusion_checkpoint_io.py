"""Checkpoint payload and file I/O helpers for diffusion training."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import torch

from src.pipeline.room_topology_conditioning import build_topology_anchor_policy_metadata
from src.training.diffusion_checkpoint_contracts import load_checkpoint_metadata_sidecar
from src.utils.checkpoint import (
    atomic_torch_save,
    log_checkpoint_artifact,
    safe_torch_load,
    write_checkpoint_metadata,
)


SafetensorsLoad = Callable[..., Mapping[str, torch.Tensor]]
SafetensorsSave = Callable[[Mapping[str, torch.Tensor], str], None]


def build_resume_checkpoint_payload(trainer: Any, metrics: Optional[Dict] = None) -> Dict[str, Any]:
    payload = {
        "epoch": trainer.epoch,
        "global_step": trainer.global_step,
        "diffusion_state_dict": trainer.diffusion.state_dict(),
        "ema_diffusion_state_dict": trainer.ema_diffusion.state_dict(),
        "condition_encoder_state_dict": trainer.condition_encoder.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "scheduler_state_dict": trainer.scheduler.state_dict(),
        "config": trainer.config.to_dict(),
        "metrics": metrics,
        "schedule_type": trainer.config.schedule_type,
    }
    grad_scaler = getattr(trainer, "_grad_scaler", None)
    if grad_scaler is not None:
        payload["grad_scaler_state_dict"] = grad_scaler.state_dict()
    if (
        bool(getattr(trainer.config, "logic_net_enabled", True))
        and getattr(trainer, "logic_net", None) is not None
    ):
        payload["logic_net_state_dict"] = trainer.logic_net.state_dict()
    if getattr(trainer, "puzzle_stage_semantics_head", None) is not None:
        payload["puzzle_stage_semantics_head_state_dict"] = (
            trainer.puzzle_stage_semantics_head.state_dict()
        )
    return payload


def build_inference_checkpoint_payload(trainer: Any, metrics: Optional[Dict] = None) -> Dict[str, Any]:
    payload = {
        "epoch": trainer.epoch,
        "global_step": trainer.global_step,
        "diffusion_state_dict": trainer.diffusion.state_dict(),
        "ema_diffusion_state_dict": trainer.ema_diffusion.state_dict(),
        "condition_encoder_state_dict": trainer.condition_encoder.state_dict(),
        "config": trainer.config.to_dict(),
        "metrics": metrics,
        "schedule_type": trainer.config.schedule_type,
    }
    if (
        bool(getattr(trainer.config, "logic_net_enabled", True))
        and getattr(trainer, "logic_net", None) is not None
    ):
        payload["logic_net_state_dict"] = trainer.logic_net.state_dict()
    if getattr(trainer, "puzzle_stage_semantics_head", None) is not None:
        payload["puzzle_stage_semantics_head_state_dict"] = (
            trainer.puzzle_stage_semantics_head.state_dict()
        )
    return payload


def prefixed_safetensors_state(
    prefix: str,
    state_dict: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {
        f"{prefix}.{key}": value.detach().cpu()
        for key, value in state_dict.items()
        if isinstance(value, torch.Tensor)
    }


def extract_prefixed_safetensors_state(
    payload: Mapping[str, torch.Tensor],
    prefix: str,
) -> Dict[str, torch.Tensor]:
    stem = f"{prefix}."
    return {
        key[len(stem) :]: value
        for key, value in payload.items()
        if isinstance(key, str) and key.startswith(stem)
    }


def _torch_checkpoint_architecture(config: Any) -> Dict[str, Any]:
    return {
        "latent_dim": int(config.latent_dim),
        "latent_scale_factor": float(getattr(config, "latent_scale_factor", 1.0)),
        "logic_initial_temperature": float(getattr(config, "logic_initial_temperature", 1.0)),
        "logic_final_temperature": float(getattr(config, "logic_final_temperature", 0.05)),
        "context_dim": int(config.context_dim),
        "num_timesteps": int(config.num_timesteps),
        "schedule_type": str(config.schedule_type),
        "diffusion_training_objective": str(
            getattr(config, "diffusion_training_objective", "diffusion")
        ),
        "denoiser_backbone": str(getattr(config, "denoiser_backbone", "unet")),
        "pag_scale": float(getattr(config, "pag_scale", 0.0)),
        "dit_depth": int(getattr(config, "dit_depth", 4)),
        "dit_patch_size": int(getattr(config, "dit_patch_size", 1)),
        "dit_mlp_ratio": float(getattr(config, "dit_mlp_ratio", 4.0)),
        "dit_activation_type": str(getattr(config, "dit_activation_type", "gelu")),
        "dit_norm_type": str(getattr(config, "dit_norm_type", "layer")),
        "num_classes": int(config.num_classes),
        "vqvae_hidden_dim": int(config.vqvae_hidden_dim),
        "vqvae_codebook_size": int(config.vqvae_codebook_size),
        "vqvae_architecture": str(getattr(config, "vqvae_architecture", "vqvae")),
        "vqvae_top_codebook_size": getattr(config, "vqvae_top_codebook_size", None),
        "vqvae_top_latent_dim": getattr(config, "vqvae_top_latent_dim", None),
        "vqvae_use_coordconv": bool(config.vqvae_use_coordconv),
    }


def _safetensors_checkpoint_architecture(config: Any) -> Dict[str, Any]:
    return {
        "latent_dim": int(config.latent_dim),
        "latent_scale_factor": float(getattr(config, "latent_scale_factor", 1.0)),
        "logic_initial_temperature": float(getattr(config, "logic_initial_temperature", 1.0)),
        "logic_final_temperature": float(getattr(config, "logic_final_temperature", 0.05)),
        "context_dim": int(config.context_dim),
        "num_timesteps": int(config.num_timesteps),
        "schedule_type": str(config.schedule_type),
        "diffusion_training_objective": str(
            getattr(config, "diffusion_training_objective", "diffusion")
        ),
        "denoiser_backbone": str(getattr(config, "denoiser_backbone", "unet")),
        "num_classes": int(config.num_classes),
    }


def save_checkpoint(
    trainer: Any,
    path: str,
    metrics: Optional[Dict] = None,
    *,
    include_optimizer: bool = True,
    has_safetensors: bool,
    save_safetensors: Optional[SafetensorsSave],
    logger: logging.Logger,
) -> None:
    checkpoint = (
        trainer._build_resume_checkpoint_payload(metrics)
        if bool(include_optimizer)
        else trainer._build_inference_checkpoint_payload(metrics)
    )
    atomic_torch_save(checkpoint, path)

    safetensors_path: Optional[Path] = None
    if has_safetensors:
        try:
            safetensors_path = Path(path).with_suffix(".safetensors")
            safetensors_payload: Dict[str, torch.Tensor] = {}
            safetensors_payload.update(
                trainer._prefixed_safetensors_state("diffusion", trainer.diffusion.state_dict())
            )
            safetensors_payload.update(
                trainer._prefixed_safetensors_state(
                    "ema_diffusion", trainer.ema_diffusion.state_dict()
                )
            )
            safetensors_payload.update(
                trainer._prefixed_safetensors_state(
                    "condition_encoder", trainer.condition_encoder.state_dict()
                )
            )
            if bool(getattr(trainer.config, "logic_net_enabled", True)):
                safetensors_payload.update(
                    trainer._prefixed_safetensors_state(
                        "logic_net", trainer.logic_net.state_dict()
                    )
                )
            if getattr(trainer, "puzzle_stage_semantics_head", None) is not None:
                safetensors_payload.update(
                    trainer._prefixed_safetensors_state(
                        "puzzle_stage_semantics_head",
                        trainer.puzzle_stage_semantics_head.state_dict(),
                    )
                )
            if save_safetensors is None:
                raise ImportError("safetensors save function is unavailable")
            save_safetensors(safetensors_payload, str(safetensors_path))
            logger.debug("Saved safetensors sidecar: %s", safetensors_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("safetensors save failed (%s); .pth checkpoint is intact.", exc)
            safetensors_path = None

    contains = (
        ["diffusion", "ema_diffusion", "condition_encoder"]
        + (["logic_net"] if bool(getattr(trainer.config, "logic_net_enabled", True)) else [])
        + (
            ["puzzle_stage_semantics_head"]
            if getattr(trainer, "puzzle_stage_semantics_head", None) is not None
            else []
        )
    )
    write_checkpoint_metadata(
        path,
        model_type="diffusion_resume" if include_optimizer else "diffusion",
        architecture=_torch_checkpoint_architecture(trainer.config),
        extra={
            "epoch": int(trainer.epoch),
            "global_step": int(trainer.global_step),
            "checkpoint_kind": "resume" if include_optimizer else "inference",
            "primary_format": "torch_pth",
            "safetensors_sidecar": (
                str(safetensors_path.name) if safetensors_path is not None else None
            ),
            "safetensors_contains_optimizer": False,
            "contains": contains + (["optimizer", "scheduler"] if include_optimizer else []),
            "vqvae_checkpoint": str(getattr(trainer.config, "vqvae_checkpoint", "") or ""),
            "topology_anchor_policy": build_topology_anchor_policy_metadata(
                semantic_role_prior_strength=trainer.config.semantic_role_prior_strength,
                semantic_puzzle_offset=trainer.config.semantic_puzzle_offset,
                topology_supervision_mode=trainer.config.topology_supervision_mode,
            ),
        },
    )
    if safetensors_path is not None:
        write_checkpoint_metadata(
            str(safetensors_path),
            model_type="diffusion_safetensors_inference",
            architecture=_safetensors_checkpoint_architecture(trainer.config),
            extra={
                "epoch": int(trainer.epoch),
                "global_step": int(trainer.global_step),
                "checkpoint_kind": "inference",
                "primary_format": "safetensors",
                "contains_optimizer": False,
                "contains": contains,
                "torch_resume_checkpoint": Path(path).name if include_optimizer else None,
            },
        )
    log_checkpoint_artifact(
        logger,
        path,
        checkpoint_dir=Path(path).parent,
        label="Saved checkpoint",
    )


def _validate_checkpoint_config(config: Any, checkpoint_config: Mapping[str, Any]) -> None:
    saved_scale = float(checkpoint_config.get("latent_scale_factor", 1.0))
    configured_scale = float(getattr(config, "latent_scale_factor", 1.0))
    if not math.isclose(saved_scale, configured_scale, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "Diffusion checkpoint latent_scale_factor mismatch: "
            f"checkpoint={saved_scale}, config={configured_scale}."
        )
    for field, default in (
        ("logic_initial_temperature", 1.0),
        ("logic_final_temperature", 0.05),
    ):
        if field not in checkpoint_config:
            continue
        saved_value = float(checkpoint_config.get(field, default))
        configured_value = float(getattr(config, field, default))
        if not math.isclose(saved_value, configured_value, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"Diffusion checkpoint {field} mismatch: "
                f"checkpoint={saved_value}, config={configured_value}."
            )


def load_checkpoint(
    trainer: Any,
    path: str,
    *,
    restore_training_state: bool,
    has_safetensors: bool,
    load_safetensors: Optional[SafetensorsLoad],
    logger: logging.Logger,
) -> Dict[str, Any]:
    if str(path).lower().endswith(".safetensors"):
        if restore_training_state:
            raise ValueError(
                "Safetensors artifacts contain model weights only and cannot be used for "
                "stateful resume. Use the matching .pth resume checkpoint or configure "
                "warm_start_checkpoint explicitly."
            )
        if not has_safetensors or load_safetensors is None:
            raise ImportError("Loading .safetensors checkpoints requires the safetensors package.")
        payload = load_safetensors(path, device=str(trainer.device))
        diffusion_state = trainer._extract_prefixed_safetensors_state(payload, "diffusion")
        ema_state = trainer._extract_prefixed_safetensors_state(payload, "ema_diffusion")
        condition_state = trainer._extract_prefixed_safetensors_state(payload, "condition_encoder")
        logic_state = trainer._extract_prefixed_safetensors_state(payload, "logic_net")
        stage_head_state = trainer._extract_prefixed_safetensors_state(
            payload,
            "puzzle_stage_semantics_head",
        )
        for name, state in (
            ("diffusion", diffusion_state),
            ("ema_diffusion", ema_state),
            ("condition_encoder", condition_state),
            ("logic_net", logic_state),
            ("puzzle_stage_semantics_head", stage_head_state),
        ):
            if state and not trainer._state_dict_is_finite(state):
                raise ValueError(
                    f"Checkpoint {path} contains non-finite values in `{name}` and cannot be loaded safely."
                )
        if not diffusion_state or not condition_state:
            raise ValueError(
                f"Safetensors checkpoint {path} must contain at least diffusion.* and "
                "condition_encoder.* weights."
            )
        trainer.diffusion.load_state_dict(diffusion_state)
        trainer.ema_diffusion.load_state_dict(ema_state or diffusion_state)
        trainer.condition_encoder.load_state_dict(condition_state)
        if logic_state and getattr(trainer, "logic_net", None) is not None:
            trainer.logic_net.load_state_dict(logic_state)
        elif logic_state:
            logger.warning(
                "Safetensors checkpoint %s contains LogicNet weights, but the current "
                "configuration disables LogicNet; leaving those weights unused.",
                path,
            )
        if stage_head_state and getattr(trainer, "puzzle_stage_semantics_head", None) is not None:
            trainer.puzzle_stage_semantics_head.load_state_dict(stage_head_state)
        elif stage_head_state:
            logger.warning(
                "Safetensors checkpoint %s contains puzzle-stage semantics weights, but "
                "the current configuration disables that ablation; leaving those weights unused.",
                path,
            )

        metadata = load_checkpoint_metadata_sidecar(path)
        architecture = dict(metadata.get("architecture", {}) or {})
        _validate_checkpoint_config(trainer.config, architecture)
        trainer._reset_training_state_for_warm_start()
        trainer._configure_guidance()
        trainer._configure_guidance(trainer.ema_diffusion)
        logger.info(
            "Warm-started model weights from tensor-only safetensors checkpoint %s; "
            "epoch/global_step and optimizer/scheduler state were reset.",
            path,
        )
        return {}

    checkpoint = safe_torch_load(path, map_location=trainer.device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint {path} must contain a mapping payload.")
    checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    if isinstance(checkpoint_config, dict):
        _validate_checkpoint_config(trainer.config, checkpoint_config)

    for key in ("diffusion_state_dict", "ema_diffusion_state_dict"):
        if key in checkpoint:
            checkpoint[key], removed = trainer._strip_embedded_guidance_logic_net_state(
                checkpoint[key]
            )
            if removed:
                logger.warning(
                    "Stripped %d legacy guidance.logic_net.* tensor(s) from `%s` while loading %s; "
                    "using `logic_net_state_dict` as the LogicNet source of truth.",
                    int(removed),
                    key,
                    path,
                )

    for key in (
        "diffusion_state_dict",
        "ema_diffusion_state_dict",
        "condition_encoder_state_dict",
        "logic_net_state_dict",
        "puzzle_stage_semantics_head_state_dict",
    ):
        if key in checkpoint and not trainer._state_dict_is_finite(checkpoint[key]):
            raise ValueError(
                f"Checkpoint {path} contains non-finite values in `{key}` and cannot be resumed safely."
            )

    required_model_keys = {
        "diffusion_state_dict",
        "condition_encoder_state_dict",
    }
    missing_model_keys = sorted(required_model_keys.difference(checkpoint))
    if missing_model_keys:
        raise ValueError(
            f"Checkpoint {path} is missing required model state: {missing_model_keys}."
        )
    if restore_training_state:
        required_resume_keys = {
            "epoch",
            "global_step",
            "optimizer_state_dict",
            "scheduler_state_dict",
        }
        if bool(getattr(trainer.config, "use_amp", False)):
            required_resume_keys.add("grad_scaler_state_dict")
        missing_resume_keys = sorted(required_resume_keys.difference(checkpoint))
        if missing_resume_keys:
            raise ValueError(
                f"Checkpoint {path} is not a complete resume artifact; missing "
                f"{missing_resume_keys}. Use warm_start_checkpoint for weights-only artifacts."
            )

    if restore_training_state:
        trainer.epoch = int(checkpoint["epoch"])
        trainer.global_step = int(checkpoint["global_step"])
        trainer._reset_gradient_accumulation()
    trainer.diffusion.load_state_dict(checkpoint["diffusion_state_dict"])
    if "ema_diffusion_state_dict" in checkpoint:
        trainer.ema_diffusion.load_state_dict(checkpoint["ema_diffusion_state_dict"])
    else:
        trainer.ema_diffusion.load_state_dict(checkpoint["diffusion_state_dict"])
        logger.warning(
            "Checkpoint %s has no ema_diffusion_state_dict; initialized EMA weights from "
            "diffusion_state_dict.",
            path,
        )
    trainer.condition_encoder.load_state_dict(checkpoint["condition_encoder_state_dict"])
    if "logic_net_state_dict" in checkpoint and getattr(trainer, "logic_net", None) is not None:
        trainer.logic_net.load_state_dict(checkpoint["logic_net_state_dict"])
    elif "logic_net_state_dict" in checkpoint:
        logger.warning(
            "Checkpoint %s contains LogicNet weights, but the current configuration "
            "disables LogicNet; leaving those weights unused.",
            path,
        )
    stage_head_state = checkpoint.get("puzzle_stage_semantics_head_state_dict")
    if stage_head_state is not None:
        if getattr(trainer, "puzzle_stage_semantics_head", None) is None:
            logger.warning(
                "Checkpoint %s contains puzzle-stage semantics weights, but the current "
                "configuration disables that ablation; leaving those weights unused.",
                path,
            )
        else:
            trainer.puzzle_stage_semantics_head.load_state_dict(stage_head_state)
    elif (
        restore_training_state
        and getattr(trainer, "puzzle_stage_semantics_head", None) is not None
    ):
        raise ValueError(
            f"Checkpoint {path} cannot resume stage-semantics training because it is missing "
            "puzzle_stage_semantics_head_state_dict. Use warm_start_checkpoint only when "
            "you intentionally want a newly initialized auxiliary head."
        )

    if not restore_training_state:
        trainer._reset_training_state_for_warm_start()

    optimizer_state_loaded = False
    if restore_training_state:
        try:
            trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            optimizer_state_loaded = True
        except ValueError as exc:
            raise ValueError(
                f"Optimizer state in resume checkpoint {path} is incompatible with the "
                f"current trainer: {exc}"
            ) from exc
    scheduler_state_loaded = False
    if restore_training_state and optimizer_state_loaded:
        try:
            trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            scheduler_state_loaded = True
        except ValueError as exc:
            raise ValueError(
                f"Scheduler state in resume checkpoint {path} is incompatible with the "
                f"current trainer: {exc}"
            ) from exc
    if restore_training_state and "grad_scaler_state_dict" in checkpoint:
        try:
            trainer._grad_scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
        except (RuntimeError, ValueError, TypeError) as exc:
            if bool(getattr(trainer.config, "use_amp", False)):
                raise ValueError(
                    f"AMP GradScaler state in resume checkpoint {path} is incompatible: {exc}"
                ) from exc
            logger.warning("Ignoring incompatible inactive GradScaler state from %s: %s", path, exc)

    trainer._configure_guidance()
    trainer._configure_guidance(trainer.ema_diffusion)
    trainer._scheduler_state_restored = bool(scheduler_state_loaded)
    trainer._scheduler_period_configured = bool(scheduler_state_loaded)
    metrics = checkpoint.get("metrics", {}) if restore_training_state else {}
    trainer._loaded_checkpoint_metrics = dict(metrics) if isinstance(metrics, dict) else {}
    if restore_training_state:
        logger.info("Resumed complete training state from %s (epoch %s)", path, trainer.epoch)
    else:
        logger.info(
            "Warm-started model weights from %s; epoch/global_step and optimizer/scheduler "
            "state were reset.",
            path,
        )
    return dict(trainer._loaded_checkpoint_metrics)
