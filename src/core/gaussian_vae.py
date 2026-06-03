"""
H-MOLQD Block II: Semantic Gaussian VAE
=======================================

Continuous latent baseline for the Zelda room autoencoding stage.

This module keeps the same semantic reconstruction objective as the VQ-VAE
path, but replaces the discrete codebook with a standard Gaussian posterior
parameterized by mean and log-variance maps.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.core.definitions import ROOM_HEIGHT, ROOM_WIDTH, TileID, normalize_room_shape
from src.core.vqvae import Decoder, Encoder, canonical_latent_shape

logger = logging.getLogger(__name__)


class SemanticGaussianVAE(nn.Module):
    """Semantic VAE with a continuous latent bottleneck."""

    RARE_TILES = [10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 30, 31, 32, 33, 40, 41, 42, 43]

    def __init__(
        self,
        num_classes: int = 44,
        num_tile_classes: Optional[int] = None,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        hidden_dims: Optional[List[int]] = None,
        num_res_blocks: int = 2,
        rare_tile_weight: float = 5.0,
        kl_weight: float = 1.0,
        use_coordconv: bool = True,
        mrf_penalty_weight: float = 0.05,
    ):
        super().__init__()

        if num_tile_classes is not None:
            num_classes = int(num_tile_classes)

        self.num_classes = int(num_classes)
        self.latent_dim = int(latent_dim)
        self.kl_weight = float(kl_weight)
        self.rare_tile_weight = float(rare_tile_weight)
        self.mrf_penalty_weight = float(max(0.0, mrf_penalty_weight))
        self.latent_spatial_shape = canonical_latent_shape((ROOM_HEIGHT, ROOM_WIDTH), channel_mult=(1, 2, 4))

        # Encode into twice the latent width so we can split mu/logvar.
        self.encoder = Encoder(
            in_channels=self.num_classes,
            hidden_channels=hidden_dim,
            latent_channels=self.latent_dim * 2,
            hidden_dims=hidden_dims,
            latent_dim=None,
            num_res_blocks=num_res_blocks,
            channel_mult=(1, 2, 4),
            use_coordconv=bool(use_coordconv),
        )
        self.decoder = Decoder(
            out_channels=self.num_classes,
            hidden_channels=hidden_dim,
            latent_channels=self.latent_dim,
            hidden_dims=hidden_dims,
            latent_dim=None,
            num_res_blocks=num_res_blocks,
            channel_mult=(4, 2, 1),
        )

        self.register_buffer("tile_weights", self._build_tile_weights())
        self.register_buffer("illegal_adjacency_matrix", self._build_illegal_adjacency_matrix())

    def _build_tile_weights(self) -> Tensor:
        weights = torch.ones(self.num_classes)
        for tile_id in self.RARE_TILES:
            if tile_id < self.num_classes:
                weights[tile_id] = self.rare_tile_weight
        return weights

    def _build_illegal_adjacency_matrix(self) -> Tensor:
        matrix = torch.zeros(self.num_classes, self.num_classes, dtype=torch.float32)

        door_ids = [
            int(TileID.DOOR_OPEN),
            int(TileID.DOOR_LOCKED),
            int(TileID.DOOR_BOMB),
            int(TileID.DOOR_PUZZLE),
            int(TileID.DOOR_BOSS),
            int(TileID.DOOR_SOFT),
        ]
        element_ids = [int(TileID.ELEMENT)]

        for a in element_ids:
            if a >= self.num_classes:
                continue
            for b in door_ids:
                if b >= self.num_classes:
                    continue
                matrix[a, b] = 1.0
                matrix[b, a] = 1.0

        return matrix

    def _illegal_adjacency_penalty(self, recon_logits: Tensor) -> Tensor:
        probs = F.softmax(recon_logits, dim=1)
        illegal = self.illegal_adjacency_matrix.to(dtype=probs.dtype, device=probs.device)

        shifts = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

        total = torch.tensor(0.0, device=probs.device, dtype=probs.dtype)
        padded = F.pad(probs, (1, 1, 1, 1), mode="constant", value=0.0)
        height = probs.shape[2]
        width = probs.shape[3]

        for dy, dx in shifts:
            y0 = 1 + dy
            x0 = 1 + dx
            neighbor = padded[:, :, y0:y0 + height, x0:x0 + width]
            pair_score = torch.einsum("bchw,cd,bdhw->bhw", probs, illegal, neighbor)
            total = total + pair_score.mean()

        return total / float(len(shifts))

    def _weighted_reconstruction_loss(self, recon: Tensor, target: Tensor) -> Tensor:
        if target.shape[1] == self.num_classes:
            target_idx = target.argmax(dim=1)
        else:
            target_idx = target.squeeze(1)

        weights = self.tile_weights[target_idx]
        loss = F.cross_entropy(recon, target_idx, reduction="none")
        return (loss * weights).mean()

    def _kl_divergence(self, mu: Tensor, logvar: Tensor) -> Tensor:
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        latent_volume = max(1, int(mu.shape[1] * mu.shape[2] * mu.shape[3]))
        return kl.sum(dim=(1, 2, 3)).mean() / float(latent_volume)

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Return posterior parameters (mu, logvar)."""
        z_params = self.encoder(x)
        mu, logvar = torch.chunk(z_params, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Sample a latent tensor using the reparameterization trick."""
        if not self.training:
            return mu
        logvar = logvar.clamp(min=-30.0, max=20.0)
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: Tensor, target_size: Optional[Tuple[int, int]] = None) -> Tensor:
        if target_size is None:
            target_size = (ROOM_HEIGHT, ROOM_WIDTH)
        else:
            target_size = normalize_room_shape(target_size)
        return self.decoder(z, target_size)

    def forward_with_losses(self, x: Tensor) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        input_size = (x.shape[2], x.shape[3])
        mu, logvar = self.encode(x)
        logvar = logvar.clamp(min=-30.0, max=20.0)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, target_size=input_size)

        recon_loss = self._weighted_reconstruction_loss(recon, x)
        kl_loss = self._kl_divergence(mu, logvar)
        illegal_penalty = self._illegal_adjacency_penalty(recon)
        total_loss = recon_loss + (self.kl_weight * kl_loss) + (self.mrf_penalty_weight * illegal_penalty)

        losses: Dict[str, Tensor] = {
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "kl_loss_weighted": self.kl_weight * kl_loss,
            "illegal_adjacency_penalty": illegal_penalty,
            "total_loss": total_loss,
        }
        return recon, z, losses

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        recon, _, losses = self.forward_with_losses(x)
        return recon, losses["kl_loss"], dict(losses)

    def compute_loss(self, x: Tensor) -> Dict[str, Tensor]:
        _, _, losses = self.forward_with_losses(x)
        return losses

    @torch.no_grad()
    def sample_prior(
        self,
        batch_size: int = 1,
        spatial_size: Optional[Tuple[int, int]] = None,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """Sample from the standard Gaussian prior and decode it."""
        if device is None:
            device = next(self.parameters()).device

        if spatial_size is None:
            latent_height, latent_width = self.latent_spatial_shape
        else:
            latent_height, latent_width = normalize_room_shape(spatial_size)

        latent = torch.randn(batch_size, self.latent_dim, latent_height, latent_width, device=device)
        return self.decode(latent)


class GaussianVAETrainer:
    """Training utilities for the continuous semantic baseline."""

    def __init__(
        self,
        model: SemanticGaussianVAE,
        lr: float = 1e-4,
        learning_rate: Optional[float] = None,
        weight_decay: float = 1e-5,
        grad_clip_norm: float = 1.0,
    ):
        if learning_rate is not None:
            lr = float(learning_rate)
        self.model = model
        self.grad_clip_norm = float(max(0.0, float(grad_clip_norm)))
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    def train_step(self, batch: Tensor, return_metrics: bool = False) -> float | Tuple[float, Dict[str, float]]:
        self.model.train()
        self.optimizer.zero_grad()

        losses = self.model.compute_loss(batch)
        loss = losses["total_loss"]
        loss.backward()

        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

        self.optimizer.step()

        metrics = {
            "loss": float(loss.item()),
            "recon_loss": float(losses["recon_loss"].item()),
            "kl_loss": float(losses["kl_loss"].item()),
            "kl_loss_weighted": float(losses["kl_loss_weighted"].item()),
            "illegal_adjacency_penalty": float(losses.get("illegal_adjacency_penalty", torch.tensor(0.0)).item()),
        }

        if return_metrics:
            return float(loss.item()), metrics
        return float(loss.item())

    @torch.no_grad()
    def eval_step(self, batch: Tensor) -> Dict[str, float]:
        self.model.eval()

        recon, _, losses = self.model(batch)
        pred = recon.argmax(dim=1)
        target = batch.argmax(dim=1) if batch.shape[1] > 1 else batch.squeeze(1)
        accuracy = (pred == target).float().mean()

        return {
            "loss": float(losses["total_loss"].item()),
            "recon_loss": float(losses["recon_loss"].item()),
            "kl_loss": float(losses["kl_loss"].item()),
            "kl_loss_weighted": float(losses["kl_loss_weighted"].item()),
            "illegal_adjacency_penalty": float(losses.get("illegal_adjacency_penalty", torch.tensor(0.0)).item()),
            "accuracy": float(accuracy.item()),
        }


def create_gaussian_vae(
    num_classes: int = 44,
    latent_dim: int = 64,
    kl_weight: float = 1.0,
    **kwargs: Any,
) -> SemanticGaussianVAE:
    """Create the continuous semantic VAE baseline."""
    return SemanticGaussianVAE(
        num_classes=num_classes,
        latent_dim=latent_dim,
        kl_weight=kl_weight,
        **kwargs,
    )
