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

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Ensure project root on path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.vqvae import SemanticVQVAE, create_vqvae, VQVAETrainer
from src.data.zelda_loader import create_dataloader, ZeldaDungeonDataset

logger = logging.getLogger(__name__)


def grids_to_onehot(batch: torch.Tensor, num_classes: int = 44) -> torch.Tensor:
    """
    Convert normalised grid batch to one-hot encoding.

    Data loader returns [B, 1, H, W] with values in [0, 1].
    The normalisation divides by a fixed constant (43 = TileID.PUZZLE, the
    highest tile ID).  To recover integer IDs we multiply by 43, round, and
    clamp — this gives an exact round-trip for all dungeons.

    Returns [B, C, H, W] float32 one-hot.
    """
    tile_ids = (batch.squeeze(1) * (num_classes - 1)).round().long().clamp(0, num_classes - 1)
    onehot = F.one_hot(tile_ids, num_classes=num_classes)  # [B, H, W, C]
    return onehot.permute(0, 3, 1, 2).float()              # [B, C, H, W]


def train_vqvae(args):
    """Full VQ-VAE pre-training loop."""

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
    # Dataset — use VGLC mode, same as diffusion training
    # ------------------------------------------------------------------
    base_loader = create_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        shuffle=True,
        use_vglc=True,       # ← CRITICAL: must match diffusion training
        normalize=True,
        load_graphs=False,
    )
    dataset = base_loader.dataset
    logger.info(f"Dataset: {len(dataset)} dungeons")

    if len(dataset) == 0:
        logger.error("No dungeon samples found! Check --data-dir path.")
        sys.exit(1)

    # Small dataset → duplicate to fill an epoch with more gradient steps
    effective_size = max(len(dataset), args.min_samples_per_epoch)
    sampler = torch.utils.data.RandomSampler(
        dataset, replacement=True, num_samples=effective_size
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        drop_last=True,
    )
    logger.info(f"Effective samples/epoch: {effective_size}, "
                f"batches/epoch: {len(dataloader)}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = create_vqvae(
        num_classes=44,
        codebook_size=args.codebook_size,
        latent_dim=args.latent_dim,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"VQ-VAE parameters: {total_params:,}")

    trainer = VQVAETrainer(model, lr=args.lr)

    # ------------------------------------------------------------------
    # Checkpoint resume
    # ------------------------------------------------------------------
    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(f"Resumed from {args.resume} (epoch {start_epoch})")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    history = []

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_metrics = {"loss": 0.0, "recon_loss": 0.0, "vq_loss": 0.0, "perplexity": 0.0}
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # Handle (tensor, graph_dict) tuples
            if isinstance(batch, (list, tuple)):
                batch = batch[0]

            batch = batch.to(device)
            x_onehot = grids_to_onehot(batch, num_classes=44)

            # Forward / backward
            _loss, metrics = trainer.train_step(x_onehot)

            for k in epoch_metrics:
                epoch_metrics[k] += metrics.get(k, 0.0)
            num_batches += 1

            if batch_idx % max(1, len(dataloader) // 5) == 0:
                logger.debug(
                    f"  Epoch {epoch+1}/{args.epochs} | Batch {batch_idx}/{len(dataloader)} | "
                    f"loss={metrics['loss']:.4f} recon={metrics['recon_loss']:.4f} "
                    f"vq={metrics['vq_loss']:.4f} perp={metrics['perplexity']:.1f}"
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
                x_onehot = grids_to_onehot(batch, num_classes=44)
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
            f"perplexity={epoch_metrics['perplexity']:.1f} | "
            f"accuracy={eval_acc:.3f}"
        )
        history.append({"epoch": epoch + 1, **epoch_metrics})

        # Save best
        if epoch_metrics["loss"] < best_loss:
            best_loss = epoch_metrics["loss"]
            save_path = save_dir / "vqvae_pretrained.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "loss": best_loss,
                "accuracy": eval_acc,
                "perplexity": epoch_metrics["perplexity"],
            }, save_path)
            logger.info(f"  ★ Saved best model → {save_path} (loss={best_loss:.4f})")

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            periodic = save_dir / f"vqvae_epoch{epoch+1:04d}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "loss": epoch_metrics["loss"],
            }, periodic)

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
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to dungeon data (e.g. 'data/The Legend of Zelda')")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--codebook-size", type=int, default=512)
    parser.add_argument("--min-samples-per-epoch", type=int, default=64,
                        help="Minimum samples per epoch (upsampled for small datasets)")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--save-every", type=int, default=50,
                        help="Save periodic checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    train_vqvae(args)


if __name__ == "__main__":
    main()
