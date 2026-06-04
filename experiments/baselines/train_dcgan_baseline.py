"""Lightweight DCGAN baseline with Gumbel-Softmax generator outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.baselines.common import (
    BaselineEvalConfig,
    evaluate_generated_grids,
    load_room_grids,
    set_reproducible_seed,
    write_json_report,
)


class RoomGenerator(nn.Module):
    def __init__(self, *, z_dim: int, num_tiles: int, hidden: int, room_shape: Tuple[int, int]) -> None:
        super().__init__()
        self.num_tiles = int(num_tiles)
        self.room_shape = (int(room_shape[0]), int(room_shape[1]))
        out_dim = self.num_tiles * self.room_shape[0] * self.room_shape[1]
        self.net = nn.Sequential(
            nn.Linear(int(z_dim), int(hidden)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(int(hidden), int(hidden) * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(int(hidden) * 2, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.net(z)
        return logits.view(z.shape[0], self.num_tiles, self.room_shape[0], self.room_shape[1])


class RoomDiscriminator(nn.Module):
    def __init__(self, *, num_tiles: int, hidden: int, room_shape: Tuple[int, int]) -> None:
        super().__init__()
        in_dim = int(num_tiles) * int(room_shape[0]) * int(room_shape[1])
        self.net = nn.Sequential(
            nn.Linear(in_dim, int(hidden) * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(int(hidden) * 2, int(hidden)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(int(hidden), 1),
        )

    def forward(self, one_hot: torch.Tensor) -> torch.Tensor:
        return self.net(one_hot.reshape(one_hot.shape[0], -1)).squeeze(-1)


def grids_to_onehot(grids: Sequence[np.ndarray], *, num_tiles: int = 44) -> torch.Tensor:
    tokens = np.stack([np.clip(np.rint(grid), 0, num_tiles - 1).astype(np.int64) for grid in grids], axis=0)
    return F.one_hot(torch.as_tensor(tokens, dtype=torch.long), num_classes=int(num_tiles)).permute(0, 3, 1, 2).float()


def train_gan(
    grids: Sequence[np.ndarray],
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    z_dim: int,
    hidden: int,
    learning_rate: float,
    dry_run: bool,
) -> tuple[RoomGenerator, List[float], List[float]]:
    room_shape = tuple(np.asarray(grids[0]).shape)
    real = grids_to_onehot(grids).to(device)
    loader = DataLoader(TensorDataset(real.cpu()), batch_size=int(batch_size), shuffle=True, drop_last=False)
    generator = RoomGenerator(z_dim=z_dim, num_tiles=44, hidden=hidden, room_shape=room_shape).to(device)
    discriminator = RoomDiscriminator(num_tiles=44, hidden=hidden, room_shape=room_shape).to(device)
    opt_g = torch.optim.AdamW(generator.parameters(), lr=float(learning_rate), betas=(0.5, 0.999), weight_decay=0.0)
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=float(learning_rate), betas=(0.5, 0.999), weight_decay=0.0)
    g_losses: List[float] = []
    d_losses: List[float] = []
    max_epochs = 1 if dry_run else int(epochs)
    for _epoch in range(max_epochs):
        for (batch_real_cpu,) in loader:
            batch_real = batch_real_cpu.to(device)
            z = torch.randn(batch_real.shape[0], int(z_dim), device=device)
            fake_logits = generator(z)
            fake_soft = F.gumbel_softmax(fake_logits.permute(0, 2, 3, 1), tau=1.0, hard=True, dim=-1).permute(0, 3, 1, 2)

            real_logits = discriminator(batch_real)
            fake_detached_logits = discriminator(fake_soft.detach())
            d_loss = F.softplus(fake_detached_logits).mean() + F.softplus(-real_logits).mean()
            if not torch.isfinite(d_loss):
                raise RuntimeError("Non-finite discriminator loss.")
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()

            z = torch.randn(batch_real.shape[0], int(z_dim), device=device)
            fake_logits = generator(z)
            fake_soft = F.gumbel_softmax(fake_logits.permute(0, 2, 3, 1), tau=1.0, hard=True, dim=-1).permute(0, 3, 1, 2)
            g_loss = F.softplus(-discriminator(fake_soft)).mean()
            if not torch.isfinite(g_loss):
                raise RuntimeError("Non-finite generator loss.")
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()

            g_losses.append(float(g_loss.detach().cpu()))
            d_losses.append(float(d_loss.detach().cpu()))
            if dry_run:
                return generator, g_losses, d_losses
    return generator, g_losses, d_losses


def sample_generator(generator: RoomGenerator, *, num_samples: int, z_dim: int, device: torch.device) -> List[np.ndarray]:
    generator.eval()
    with torch.no_grad():
        z = torch.randn(int(num_samples), int(z_dim), device=device)
        logits = generator(z)
        grids = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int32)
    return [grid for grid in grids]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train/evaluate a minimal DCGAN Zelda baseline.")
    parser.add_argument("--data-dir", type=Path, default=Path("Data/The Legend of Zelda"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/baselines/dcgan"))
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--num-generate", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--z-dim", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-pcbs", action="store_true")
    args = parser.parse_args(argv)

    set_reproducible_seed(args.seed)
    reference = load_room_grids(args.data_dir, max_samples=(8 if args.dry_run else args.max_train_samples))
    device = torch.device(args.device)
    generator, g_losses, d_losses = train_gan(
        reference,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        z_dim=args.z_dim,
        hidden=args.hidden,
        learning_rate=args.learning_rate,
        dry_run=args.dry_run,
    )
    num_generate = 4 if args.dry_run else int(args.num_generate)
    generated = sample_generator(generator, num_samples=num_generate, z_dim=args.z_dim, device=device)
    report = evaluate_generated_grids(
        generated,
        reference,
        BaselineEvalConfig(name="dcgan_gumbel_softmax", seed=args.seed, run_pcbs=bool(args.run_pcbs and not args.dry_run)),
    )
    report["dcgan"] = {
        "dry_run": bool(args.dry_run),
        "g_losses": g_losses,
        "d_losses": d_losses,
        "final_g_loss": float(g_losses[-1]) if g_losses else None,
        "final_d_loss": float(d_losses[-1]) if d_losses else None,
        "note": "Generator training uses straight-through Gumbel-Softmax; discrete argmax is only used under torch.no_grad().",
    }
    output_path = write_json_report(args.output_dir / "dcgan_baseline_report.json", report)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
