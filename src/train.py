"""
Unified training orchestrator for KLTN.

This module replaces the legacy monolithic trainer with a stable staged flow:
1) Train VQ-VAE (Block II)
2) Train Latent Diffusion + Logic guidance (Blocks III-V)

Usage examples:
    python -m src.train --stage all --data-dir "Data/The Legend of Zelda"
    python -m src.train --stage vqvae --epochs-vqvae 300
    python -m src.train --stage diffusion --vqvae-checkpoint checkpoints/vqvae_pretrained.pth
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from src.train_vqvae import train_vqvae
from src.train_diffusion import DiffusionTrainingConfig, train_diffusion

logger = logging.getLogger(__name__)


def _run_vqvae_stage(args: argparse.Namespace) -> Path:
    """Run Stage 1 VQ-VAE training and return expected checkpoint path."""
    vqvae_args = SimpleNamespace(
        data_dir=args.data_dir,
        epochs=int(args.epochs_vqvae),
        batch_size=int(args.batch_size),
        lr=float(args.lr_vqvae),
        latent_dim=int(args.latent_dim),
        codebook_size=int(args.codebook_size),
        min_samples_per_epoch=int(args.min_samples_per_epoch),
        save_dir=str(args.checkpoint_dir),
        save_every=int(args.save_every_vqvae),
        resume=args.resume_vqvae,
        device=args.device,
        verbose=bool(args.verbose),
    )
    train_vqvae(vqvae_args)
    return Path(args.checkpoint_dir) / "vqvae_pretrained.pth"


def _run_diffusion_stage(args: argparse.Namespace, vqvae_checkpoint: Optional[Path]) -> None:
    """Run Stage 2 diffusion training with optional VQ-VAE checkpoint."""
    ckpt_path: Optional[str] = None
    if args.vqvae_checkpoint:
        ckpt_path = str(Path(args.vqvae_checkpoint))
    elif vqvae_checkpoint is not None:
        ckpt_path = str(vqvae_checkpoint)

    if ckpt_path is None or not Path(ckpt_path).exists():
        logger.warning(
            "No valid VQ-VAE checkpoint found. Diffusion will run with fresh/fallback VQ-VAE weights."
        )

    config = DiffusionTrainingConfig(
        data_dir=args.data_dir,
        batch_size=int(args.batch_size),
        vqvae_checkpoint=ckpt_path,
        latent_dim=int(args.latent_dim),
        epochs=int(args.epochs_diffusion),
        learning_rate=float(args.lr_diffusion),
        alpha_logic=float(args.alpha_logic),
        guidance_scale=float(args.guidance_scale),
        checkpoint_dir=str(args.checkpoint_dir),
        save_every=int(args.save_every_diffusion),
        device=args.device,
        quick=bool(args.quick),
    )
    train_diffusion(config)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified staged training for KLTN (VQ-VAE + Diffusion).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--stage", choices=["all", "vqvae", "diffusion"], default="all")
    parser.add_argument("--data-dir", type=str, default="Data/The Legend of Zelda")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--quick", action="store_true", help="Quick mode for diffusion stage.")

    parser.add_argument("--epochs-vqvae", type=int, default=300)
    parser.add_argument("--lr-vqvae", type=float, default=3e-4)
    parser.add_argument("--codebook-size", type=int, default=512)
    parser.add_argument("--min-samples-per-epoch", type=int, default=64)
    parser.add_argument("--save-every-vqvae", type=int, default=50)
    parser.add_argument("--resume-vqvae", type=str, default=None)

    parser.add_argument("--epochs-diffusion", type=int, default=100)
    parser.add_argument("--lr-diffusion", type=float, default=1e-4)
    parser.add_argument("--alpha-logic", type=float, default=0.1)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--save-every-diffusion", type=int, default=10)
    parser.add_argument("--vqvae-checkpoint", type=str, default=None)

    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    vqvae_ckpt: Optional[Path] = None
    if args.stage in {"all", "vqvae"}:
        logger.info("Stage 1/2: Training VQ-VAE")
        vqvae_ckpt = _run_vqvae_stage(args)
        logger.info("VQ-VAE stage complete")

    if args.stage in {"all", "diffusion"}:
        logger.info("Stage 2/2: Training diffusion")
        _run_diffusion_stage(args, vqvae_ckpt)
        logger.info("Diffusion stage complete")


if __name__ == "__main__":
    main()
