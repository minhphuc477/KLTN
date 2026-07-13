"""Frozen VQ-VAE checkpoint compatibility contracts for diffusion training."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch


logger = logging.getLogger(__name__)


def load_checkpoint_metadata_sidecar(checkpoint_path: str | Path) -> Dict[str, Any]:
    """Load ``<checkpoint>.meta.json`` when present."""
    meta_path = Path(f"{checkpoint_path}.meta.json")
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read checkpoint metadata sidecar %s: %s", meta_path, exc)
        return {}


def validate_vqvae_checkpoint_state(
    checkpoint_path: str | Path,
    checkpoint: Dict[str, Any],
    *,
    expected_codebook_size: int,
) -> None:
    """Infer codebook size from old checkpoints without sidecars when possible."""
    state = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        return
    for key in ("quantizer.embedding.weight", "bottom_quantizer.embedding.weight"):
        value = state.get(key)
        if isinstance(value, torch.Tensor) and value.dim() >= 1:
            actual = int(value.shape[0])
            expected = int(expected_codebook_size)
            if actual != expected:
                raise ValueError(
                    f"VQ-VAE checkpoint {checkpoint_path} codebook size mismatch: "
                    f"checkpoint={actual}, config={expected}. Update vqvae_codebook_size "
                    "or choose the matching frozen VQ-VAE checkpoint."
                )
            return


def resolve_vqvae_architecture(
    checkpoint_path: Optional[str],
    *,
    num_classes: int,
    latent_dim: int,
    hidden_dim: int,
    codebook_size: int,
    architecture: str = "vqvae",
    top_codebook_size: Optional[int] = None,
    top_latent_dim: Optional[int] = None,
    use_coordconv: bool = True,
    mrf_penalty_weight: float = 0.05,
) -> Dict[str, Any]:
    """Resolve architecture metadata and reject checkpoint/config drift upstream."""
    resolved: Dict[str, Any] = {
        "architecture": str(architecture or "vqvae"),
        "num_classes": int(num_classes),
        "latent_dim": int(latent_dim),
        "hidden_dim": int(hidden_dim),
        "codebook_size": int(codebook_size),
        "top_codebook_size": top_codebook_size,
        "top_latent_dim": top_latent_dim,
        "use_coordconv": bool(use_coordconv),
        "mrf_penalty_weight": float(mrf_penalty_weight),
    }
    if checkpoint_path:
        metadata = load_checkpoint_metadata_sidecar(Path(checkpoint_path))
        checkpoint_architecture = metadata.get("architecture", {}) if isinstance(metadata, dict) else {}
        if isinstance(checkpoint_architecture, dict):
            for key in tuple(resolved):
                if key in checkpoint_architecture and checkpoint_architecture[key] is not None:
                    resolved[key] = checkpoint_architecture[key]

    return {
        "architecture": str(resolved.get("architecture", "vqvae")),
        "num_classes": int(resolved["num_classes"]),
        "latent_dim": int(resolved["latent_dim"]),
        "hidden_dim": int(resolved["hidden_dim"]),
        "codebook_size": int(resolved["codebook_size"]),
        "top_codebook_size": (
            None if resolved.get("top_codebook_size") is None else int(resolved["top_codebook_size"])
        ),
        "top_latent_dim": (
            None if resolved.get("top_latent_dim") is None else int(resolved["top_latent_dim"])
        ),
        "use_coordconv": bool(resolved["use_coordconv"]),
        "mrf_penalty_weight": float(resolved["mrf_penalty_weight"]),
    }


__all__ = [
    "load_checkpoint_metadata_sidecar",
    "resolve_vqvae_architecture",
    "validate_vqvae_checkpoint_state",
]
