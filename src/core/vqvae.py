"""
H-MOLQD Block II: Semantic VQ-VAE
=================================

Discrete Representation Learning for Zelda Dungeon Rooms.

This module implements a Vector Quantized Variational Autoencoder (VQ-VAE)
specialized for semantic dungeon grids. The model learns a discrete codebook
of room "vocabulary" that can be used by the diffusion model.

Mathematical Formulation:
-------------------------
Encoder: z_e = E(x) ∈ ℝ^{H'×W'×D}
Quantization: z_q = argmin_{e_k} ||z_e - e_k||_2
Decoder: x̂ = D(z_q)

Losses:
- Reconstruction: L_rec = ||x - x̂||_2^2 (weighted for rare tiles)
- Codebook: L_cb = ||sg[z_e] - e||_2^2
- Commitment: L_com = β||z_e - sg[e]||_2^2
- Total: L = L_rec + L_cb + βL_com

where sg[] is stop-gradient operator.

Architecture:
- Encoder: Conv2d -> ResBlocks -> Downsample -> Latent
- Vector Quantizer: Learned codebook with K entries
- Decoder: Upsample -> ResBlocks -> Conv2d -> Output

"""

import logging
import math
import threading
from typing import Dict, Iterable, List, Tuple, Optional, Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.utils.optimization import adamw_decay_param_groups

from src.core.definitions import ROOM_HEIGHT, ROOM_WIDTH, TileID, normalize_room_shape

logger = logging.getLogger(__name__)


# ============================================================================
# SHAPE HELPERS
# ============================================================================

def canonical_latent_shape(
    room_shape: Sequence[int] = (ROOM_HEIGHT, ROOM_WIDTH),
    *,
    channel_mult: Tuple[int, ...] = (1, 2, 4),
    downsample_factor: int = 2,
) -> Tuple[int, int]:
    """
    Compute the default VQ-VAE latent `(rows, cols)` for a room grid.

    The encoder downsamples once per level except the last using a stride-2
    convolution with padding=1, which yields `ceil(size / 2)` each time.
    Keeping this calculation in one place avoids scattering hardcoded
    `(ROOM_HEIGHT // 4, (ROOM_WIDTH + 3) // 4)` assumptions across the codebase.
    """
    latent_h, latent_w = normalize_room_shape(room_shape)
    stride = max(1, int(downsample_factor))
    num_downsamples = max(0, len(channel_mult) - 1)

    for _ in range(num_downsamples):
        latent_h = int(math.ceil(latent_h / stride))
        latent_w = int(math.ceil(latent_w / stride))

    return latent_h, latent_w


# ============================================================================
# VECTOR QUANTIZER
# ============================================================================

class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer with learnable codebook.
    
    Implements the VQ operation from VQ-VAE:
    z_q = e_k where k = argmin_j ||z_e - e_j||_2
    
    Uses straight-through estimator for backpropagation:
    Forward: z_q (discrete)
    Backward: gradient flows to z_e directly
    
    Args:
        num_embeddings: Size of codebook (K)
        embedding_dim: Dimension of embeddings (D)
        commitment_cost: Weight β for commitment loss
        decay: EMA decay for codebook updates (if using EMA)
        use_ema: Use exponential moving average for codebook
    """
    
    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        use_ema: bool = True,
        epsilon: float = 1e-5,
        dead_code_reset_interval: int = 100,
        dead_code_threshold: float = 0.05,
        dead_code_warmup_steps: int = 500,
        protect_active_codes_during_reset: bool = True,
        max_dead_code_resets_per_event: int = 16,
        ema_decay_warmup_steps: int = 100,
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.use_ema = use_ema
        self.epsilon = epsilon
        self.ema_decay_warmup_steps = int(max(0, ema_decay_warmup_steps))
        
        # Codebook embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / num_embeddings, 
            1.0 / num_embeddings
        )
        
        if use_ema:
            self.embedding.requires_grad_(False)
            # EMA cluster counts and sums
            self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
            self.register_buffer('ema_embedding_sum', self.embedding.weight.data.clone())
        
        # Statistics tracking
        self.register_buffer('codebook_usage', torch.zeros(num_embeddings))

        # --- Phase 1B: Dead code reset tracking ---
        self._reset_counter = 0
        self._reset_interval = int(max(1, dead_code_reset_interval))
        # The EMA cluster-size statistic converges to the expected count per
        # batch, not a cumulative lifetime count. Small room batches only
        # produce a few dozen assignments per step, so thresholds like 2.0 can
        # incorrectly classify healthy-but-infrequent codes as dead.
        self._dead_threshold = float(max(0.0, dead_code_threshold))
        self._dead_code_warmup_steps = int(max(0, dead_code_warmup_steps))
        self._protect_active_codes_during_reset = bool(protect_active_codes_during_reset)
        self._max_dead_code_resets_per_event = int(max(0, max_dead_code_resets_per_event))
        self._ema_update_counter = 0
        self._codebook_update_lock = threading.RLock()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_codebook_update_lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "_ema_update_counter" not in self.__dict__:
            self._ema_update_counter = 0
        if "ema_decay_warmup_steps" not in self.__dict__:
            self.ema_decay_warmup_steps = 100
        self._codebook_update_lock = threading.RLock()

    def get_extra_state(self) -> Dict[str, int]:
        """Persist EMA/dead-code schedule progression across checkpoints."""
        return {
            "reset_counter": int(self._reset_counter),
            "ema_update_counter": int(self._ema_update_counter),
        }

    def set_extra_state(self, state: object) -> None:
        if not isinstance(state, dict):
            return
        self._reset_counter = int(max(0, int(state.get("reset_counter", 0))))
        self._ema_update_counter = int(max(0, int(state.get("ema_update_counter", 0))))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Older checkpoints predate get_extra_state. Inject the current
        # counters so strict legacy loading remains backward compatible.
        extra_key = prefix + "_extra_state"
        if extra_key not in state_dict:
            state_dict[extra_key] = self.get_extra_state()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @staticmethod
    def _latent_mse(z_a: Tensor, z_b: Tensor) -> Tensor:
        """Average per-token squared error after summing embedding channels."""
        return (z_a - z_b).pow(2).sum(dim=-1).mean()

    def _current_ema_decay(self) -> float:
        if self.ema_decay_warmup_steps <= 0:
            return float(self.decay)
        step = float(max(1, self._ema_update_counter))
        warmup_decay = step / float(self.ema_decay_warmup_steps + step)
        return float(min(float(self.decay), warmup_decay))
    
    def forward(
        self,
        z_e: Tensor,
        return_info: bool = False,
        channel_first: Optional[bool] = None,
    ) -> Tuple[Tensor, Tensor, Tensor] | Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """
        Quantize continuous latents to discrete codebook entries.
        
        Args:
            z_e: Encoder output [B, D, H, W] or [B, H, W, D]
            channel_first: Explicitly declare layout for ambiguous rank-4
                tensors. When omitted, layout is inferred and ambiguous tensors
                with both channel positions equal to embedding_dim are rejected.
            
        Returns:
            If return_info=False (default):
                z_q: Quantized latents (same shape as input)
                vq_loss: Scalar quantization loss
                indices: Codebook indices [B, H, W]
            If return_info=True:
                z_q: Quantized latents (same shape as input)
                indices: Codebook indices [B, H, W]
                losses: Dict with 'vq_loss', 'commitment_loss', 'perplexity'
        """
        if z_e.dim() != 4:
            raise ValueError(f"VectorQuantizer expected rank-4 input [B,D,H,W] or [B,H,W,D], got {tuple(z_e.shape)}.")
        first_is_embedding = int(z_e.shape[1]) == int(self.embedding_dim)
        last_is_embedding = int(z_e.shape[-1]) == int(self.embedding_dim)
        if channel_first is None:
            if first_is_embedding and last_is_embedding:
                raise ValueError(
                    "Ambiguous VectorQuantizer input layout: both channel dimension and last dimension "
                    f"equal embedding_dim={self.embedding_dim}. Pass channel_first=True or False explicitly."
                )
            if not first_is_embedding and not last_is_embedding:
                raise ValueError(
                    "VectorQuantizer input must be [B,D,H,W] or [B,H,W,D] with D=embedding_dim; "
                    f"got shape={tuple(z_e.shape)} and embedding_dim={self.embedding_dim}."
                )
            channel_first = first_is_embedding
        else:
            channel_first = bool(channel_first)
            if channel_first and not first_is_embedding:
                raise ValueError(
                    f"channel_first=True requires z_e.shape[1] == embedding_dim={self.embedding_dim}, "
                    f"got shape={tuple(z_e.shape)}."
                )
            if not channel_first and not last_is_embedding:
                raise ValueError(
                    f"channel_first=False requires z_e.shape[-1] == embedding_dim={self.embedding_dim}, "
                    f"got shape={tuple(z_e.shape)}."
                )

        # Handle both channel-first and channel-last.
        if channel_first:
            # [B, D, H, W] -> [B, H, W, D]
            z_e = z_e.permute(0, 2, 3, 1).contiguous()
        
        B, H, W, D = z_e.shape
        
        # Flatten spatial dimensions
        z_flat = z_e.view(-1, D)  # [B*H*W, D]
        
        # Compute distances to codebook entries
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z·e
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        )  # [B*H*W, K]
        
        # Find nearest codebook entries
        indices = torch.argmin(distances, dim=1)  # [B*H*W]
        
        # Quantize
        z_q = self.embedding(indices)  # [B*H*W, D]
        z_q = z_q.view(B, H, W, D)
        
        # Compute losses
        losses = {}
        
        if self.use_ema:
            if self.training:
                # EMA codebook update
                self._ema_update(z_flat, indices)

            # EMA updates stay training-only, but validation still reports the
            # same commitment objective so train_loss and val_loss are comparable.
            commitment_loss = self._latent_mse(z_e, z_q.detach())
            losses['commitment_loss'] = self.commitment_cost * commitment_loss
            losses['vq_loss'] = losses['commitment_loss']
        else:
            # Standard VQ-VAE loss does not mutate codebook state, so both
            # terms are meaningful during training and validation.
            codebook_loss = self._latent_mse(z_q, z_e.detach())
            commitment_loss = self._latent_mse(z_e, z_q.detach())

            losses['codebook_loss'] = codebook_loss
            losses['commitment_loss'] = self.commitment_cost * commitment_loss
            losses['vq_loss'] = codebook_loss + losses['commitment_loss']
        
        # Straight-through estimator
        z_q = z_e + (z_q - z_e).detach()
        
        # Compute perplexity (measure of codebook usage)
        counts = torch.bincount(indices, minlength=self.num_embeddings).to(dtype=z_flat.dtype)
        avg_probs = counts / counts.sum().clamp_min(1.0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        losses['perplexity'] = perplexity
        
        # Update usage statistics in both train and eval so monitoring reflects
        # the most recent workload. Guard this with the same lock as EMA
        # updates because eval hooks can run concurrently with training.
        with self._codebook_update_lock, torch.no_grad():
            batch_usage = torch.bincount(
                indices, minlength=self.num_embeddings
            ).to(device=self.codebook_usage.device, dtype=self.codebook_usage.dtype)
            # EMA decay prevents stale early-training bias from blocking
            # diagnostics in long training or large inference runs.
            self.codebook_usage.mul_(0.99).add_(batch_usage, alpha=0.01)
            if self.training and not self.use_ema:
                self._reset_counter += 1
                if self._reset_counter % self._reset_interval == 0:
                    self._reset_dead_codes_standard(z_flat, indices)
        
        # Reshape indices
        indices = indices.view(B, H, W)
        
        # Convert back to channel-first if needed
        if channel_first:
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        if return_info:
            return z_q, indices, losses
        return z_q, losses['vq_loss'], indices
    
    def _ema_update(self, z_flat: Tensor, indices: Tensor):
        """Update codebook using exponential moving average (DDP-safe)."""
        with self._codebook_update_lock, torch.no_grad():
            cluster_size = torch.bincount(
                indices,
                minlength=self.num_embeddings,
            ).to(device=z_flat.device, dtype=z_flat.dtype)
            embedding_sum = torch.zeros(
                self.num_embeddings,
                self.embedding_dim,
                device=z_flat.device,
                dtype=z_flat.dtype,
            )
            embedding_sum.index_add_(0, indices, z_flat)
            
            # DDP synchronization: aggregate stats across all GPUs
            # before applying EMA so every replica sees the same update.
            try:
                import torch.distributed as dist
                if dist.is_initialized():
                    work_cluster = dist.all_reduce(cluster_size, op=dist.ReduceOp.SUM, async_op=True)
                    work_embedding = dist.all_reduce(embedding_sum, op=dist.ReduceOp.SUM, async_op=True)
                    work_cluster.wait()
                    work_embedding.wait()
            except (ImportError, RuntimeError):
                pass  # single-GPU fallback: no-op
            
            self._ema_update_counter += 1
            decay = self._current_ema_decay()

            self.ema_cluster_size = (
                decay * self.ema_cluster_size
                + (1 - decay) * cluster_size
            )
            
            self.ema_embedding_sum = (
                decay * self.ema_embedding_sum
                + (1 - decay) * embedding_sum
            )
            
            # Laplace smoothing
            n = torch.sum(self.ema_cluster_size)
            cluster_size_smoothed = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon) * n
            )
            cluster_size_smoothed = cluster_size_smoothed.clamp(min=self.epsilon)
            
            # Update embeddings
            self.embedding.weight.data.copy_(
                self.ema_embedding_sum / cluster_size_smoothed.unsqueeze(1)
            )
            
            # --- Phase 1B: Periodic dead code reset ---
            self._reset_counter += 1
            if self._reset_counter % self._reset_interval == 0:
                self._reset_dead_codes(z_flat, indices)

    def _reset_dead_codes(self, z_flat: Tensor, indices: Optional[Tensor] = None):
        """
        Reset dead codebook entries to random encoder outputs + noise.
        
        Dead codes (rarely or never selected) waste codebook capacity.
        Resetting them to actual encoder outputs ensures they capture
        useful patterns. (Dhariwal et al. 2020, Zeghidour et al. 2021)
        
        DDP-safe: replacement vectors are determined on rank-0 and
        broadcast to all replicas.
        """
        if not self.use_ema:
            self._reset_dead_codes_standard(z_flat, indices)
            return
        with self._codebook_update_lock, torch.no_grad():
            if self._reset_counter < self._dead_code_warmup_steps:
                return

            # Identify dead codes: those with very low EMA cluster size
            dead_mask = self.ema_cluster_size < self._dead_threshold
            ema_live_before = int((self.ema_cluster_size >= self._dead_threshold).sum().item())
            batch_active = 0
            if indices is not None and self._protect_active_codes_during_reset:
                batch_active_mask = torch.bincount(
                    indices.view(-1), minlength=self.num_embeddings
                ).to(device=self.ema_cluster_size.device) > 0
                try:
                    import torch.distributed as dist
                    if dist.is_initialized():
                        active_int = batch_active_mask.to(dtype=torch.int32)
                        dist.all_reduce(active_int, op=dist.ReduceOp.MAX)
                        batch_active_mask = active_int > 0
                except (ImportError, RuntimeError):
                    pass  # single-GPU fallback: no-op
                batch_active = int(batch_active_mask.sum().item())
                dead_mask = dead_mask & (~batch_active_mask)
            candidate_dead = int(dead_mask.sum().item())

            if candidate_dead > 0 and self._max_dead_code_resets_per_event > 0:
                candidate_indices = torch.nonzero(dead_mask, as_tuple=False).view(-1)
                if candidate_indices.numel() > self._max_dead_code_resets_per_event:
                    candidate_scores = self.ema_cluster_size[candidate_indices]
                    keep_idx = torch.topk(
                        candidate_scores,
                        k=self._max_dead_code_resets_per_event,
                        largest=False,
                    ).indices
                    selected = candidate_indices[keep_idx]
                    new_dead_mask = torch.zeros_like(dead_mask, dtype=torch.bool)
                    new_dead_mask[selected] = True
                    dead_mask = new_dead_mask
            num_dead = int(dead_mask.sum().item())
            
            if num_dead > 0 and z_flat.shape[0] > 0:
                # Replace dead codes with random encoder outputs + small noise
                random_indices = torch.randint(0, z_flat.shape[0], (int(num_dead),),
                                               device=z_flat.device)
                new_embeddings = z_flat[random_indices].detach()
                new_embeddings = new_embeddings + torch.randn_like(new_embeddings) * 0.01
                
                # DDP: broadcast from rank 0 so all replicas reset identically
                try:
                    import torch.distributed as dist
                    if dist.is_initialized():
                        dist.broadcast(new_embeddings, src=0)
                except (ImportError, RuntimeError):
                    pass  # single-GPU fallback: no-op
                
                self.embedding.weight.data[dead_mask] = new_embeddings
                
                # Keep EMA numerator and denominator consistent with the
                # injected code vector: embedding = sum / cluster_size.
                reset_cluster_size = float(max(1.0, self._dead_threshold))
                self.ema_cluster_size[dead_mask] = reset_cluster_size
                self.ema_embedding_sum[dead_mask] = new_embeddings * reset_cluster_size
                
                logger.debug(
                    f"VQ codebook: reset {num_dead}/{self.num_embeddings} dead codes "
                    f"(candidates={candidate_dead}/{self.num_embeddings}, "
                    f"ema_live_before={ema_live_before}/{self.num_embeddings}, "
                    f"batch_active={batch_active}/{self.num_embeddings}, "
                    f"threshold={self._dead_threshold:.4f}, "
                    f"warmup={self._dead_code_warmup_steps}, "
                    f"max_per_event={self._max_dead_code_resets_per_event})"
                )

    def _reset_dead_codes_standard(self, z_flat: Tensor, indices: Optional[Tensor] = None):
        """Reset dead learned-codebook entries when EMA statistics are disabled."""
        if (
            self._dead_threshold <= 0.0
            or self._max_dead_code_resets_per_event <= 0
            or z_flat.shape[0] <= 0
            or self._reset_counter < self._dead_code_warmup_steps
        ):
            return
        with self._codebook_update_lock, torch.no_grad():
            usage_total = self.codebook_usage.sum().clamp_min(1.0)
            usage_prob = self.codebook_usage / usage_total
            usage_threshold = float(self._dead_threshold) / float(max(1, self.num_embeddings))
            dead_mask = usage_prob < usage_threshold
            candidate_dead = int(dead_mask.sum().item())
            if candidate_dead <= 0:
                return
            if indices is not None and self._protect_active_codes_during_reset:
                batch_active_mask = torch.bincount(
                    indices.view(-1).to(device=self.codebook_usage.device),
                    minlength=self.num_embeddings,
                ).to(device=self.codebook_usage.device) > 0
                dead_mask = dead_mask & (~batch_active_mask)
            candidate_dead = int(dead_mask.sum().item())
            if candidate_dead <= 0:
                return
            candidate_indices = torch.nonzero(dead_mask, as_tuple=False).view(-1)
            if candidate_indices.numel() > self._max_dead_code_resets_per_event:
                candidate_scores = usage_prob[candidate_indices]
                keep_idx = torch.topk(
                    candidate_scores,
                    k=self._max_dead_code_resets_per_event,
                    largest=False,
                ).indices
                selected = candidate_indices[keep_idx]
                dead_mask = torch.zeros_like(dead_mask, dtype=torch.bool)
                dead_mask[selected] = True
            num_dead = int(dead_mask.sum().item())
            if num_dead <= 0:
                return
            random_indices = torch.randint(
                0,
                z_flat.shape[0],
                (num_dead,),
                device=z_flat.device,
            )
            new_embeddings = z_flat[random_indices].detach()
            new_embeddings = new_embeddings + torch.randn_like(new_embeddings) * 0.01
            weight_mask = dead_mask.to(device=self.embedding.weight.device)
            self.embedding.weight.data[weight_mask] = new_embeddings.to(
                device=self.embedding.weight.device,
                dtype=self.embedding.weight.dtype,
            )
            self.codebook_usage[dead_mask] = usage_total / float(max(1, self.num_embeddings))
            logger.debug(
                "VQ codebook: reset %d/%d non-EMA dead codes (candidates=%d, threshold=%.6f)",
                num_dead,
                self.num_embeddings,
                candidate_dead,
                usage_threshold,
            )
    
    def get_codebook_usage(self) -> Tensor:
        """Get normalized codebook usage statistics."""
        total = self.codebook_usage.sum()
        if total > 0:
            return self.codebook_usage / total
        return self.codebook_usage
    
    def encode_indices(self, indices: Tensor) -> Tensor:
        """Convert indices to embeddings."""
        return self.embedding(indices)


class FSQuantizer(nn.Module):
    """
    Finite Scalar Quantization ablation for VQ-VAE-style tokenizers.

    FSQ replaces a learned nearest-neighbor codebook with bounded scalar
    rounding per latent dimension. Optional 1x1 projections keep the public
    VQ-VAE latent interface unchanged while exposing implicit code indices for
    utilization diagnostics.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        levels: Optional[Iterable[int]] = None,
        num_dims: Optional[int] = None,
        saturation_penalty_weight: float = 1e-3,
    ) -> None:
        super().__init__()
        levels_list = [int(v) for v in (levels if levels is not None else [8, 5, 5, 5])]
        if not levels_list or any(v < 2 for v in levels_list):
            raise ValueError("FSQuantizer levels must be a non-empty sequence of integers >= 2.")
        if num_dims is not None:
            target_dims = int(max(1, num_dims))
            if len(levels_list) < target_dims:
                repeats = int(math.ceil(target_dims / len(levels_list)))
                levels_list = (levels_list * repeats)[:target_dims]
            elif len(levels_list) > target_dims:
                levels_list = levels_list[:target_dims]

        self.embedding_dim = int(max(1, embedding_dim))
        self.num_dims = int(len(levels_list))
        self.num_embeddings = int(math.prod(levels_list))
        self.use_ema = False
        self.saturation_penalty_weight = float(max(0.0, saturation_penalty_weight))

        self.pre_proj = (
            nn.Identity()
            if self.num_dims == self.embedding_dim
            else nn.Conv2d(self.embedding_dim, self.num_dims, kernel_size=1)
        )
        self.post_proj = (
            nn.Identity()
            if self.num_dims == self.embedding_dim
            else nn.Conv2d(self.num_dims, self.embedding_dim, kernel_size=1)
        )
        self.register_buffer("levels", torch.tensor(levels_list, dtype=torch.float32))
        self.register_buffer("basis", self._build_basis(levels_list))
        self.register_buffer("codebook_usage", torch.zeros(self.num_embeddings))

    @staticmethod
    def _build_basis(levels: List[int]) -> Tensor:
        basis: List[int] = []
        running = 1
        for level in levels:
            basis.append(running)
            running *= int(level)
        return torch.tensor(basis, dtype=torch.long)

    def _round_ste(self, bounded: Tensor) -> Tuple[Tensor, Tensor]:
        levels = self.levels.to(device=bounded.device, dtype=bounded.dtype).view(1, -1, 1, 1)
        scaled = (bounded + 1.0) * 0.5 * (levels - 1.0)
        rounded = torch.minimum(torch.round(scaled).clamp_min(0.0), levels - 1.0)
        quantized = (rounded / (levels - 1.0)) * 2.0 - 1.0
        return bounded + (quantized - bounded).detach(), rounded.to(dtype=torch.long)

    @staticmethod
    def _bound_ste(z: Tensor) -> Tensor:
        """Clamp FSQ inputs to the finite code range with an identity STE gradient."""
        clamped = z.clamp(-1.0, 1.0)
        return z + (clamped - z).detach()

    def _indices_from_digits(self, digits: Tensor) -> Tensor:
        basis = self.basis.to(device=digits.device).view(1, -1, 1, 1)
        return (digits * basis).sum(dim=1).to(dtype=torch.long)

    def forward(
        self,
        z_e: Tensor,
        return_info: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor] | Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        channel_first = bool(z_e.dim() == 4 and z_e.shape[1] == self.embedding_dim)
        if channel_first:
            z_in = z_e
        else:
            if z_e.dim() != 4 or z_e.shape[-1] != self.embedding_dim:
                raise ValueError(
                    f"FSQuantizer expected [B,D,H,W] or [B,H,W,D] with D={self.embedding_dim}, got {tuple(z_e.shape)}."
                )
            z_in = z_e.permute(0, 3, 1, 2).contiguous()

        z_low = self.pre_proj(z_in)
        bounded = self._bound_ste(z_low)
        z_q_low, digits = self._round_ste(bounded)
        z_q = self.post_proj(z_q_low)
        indices = self._indices_from_digits(digits)

        with torch.no_grad():
            batch_usage = torch.bincount(
                indices.reshape(-1),
                minlength=self.num_embeddings,
            ).to(device=self.codebook_usage.device, dtype=self.codebook_usage.dtype)
            self.codebook_usage.mul_(0.99).add_(batch_usage, alpha=0.01)

        encodings = F.one_hot(indices.reshape(-1), self.num_embeddings).float()
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        zero = torch.zeros((), device=z_q.device, dtype=z_q.dtype)
        saturation_loss = F.relu(z_low.abs() - 1.0).pow(2).mean()
        fsq_loss = self.saturation_penalty_weight * saturation_loss
        losses = {
            "vq_loss": fsq_loss,
            "commitment_loss": zero,
            "perplexity": perplexity,
            "fsq_loss": fsq_loss,
            "fsq_saturation_loss": saturation_loss,
        }

        if not channel_first:
            z_q = z_q.permute(0, 2, 3, 1).contiguous()

        if return_info:
            return z_q, indices, losses
        return z_q, losses["vq_loss"], indices

    def encode_indices(self, indices: Tensor) -> Tensor:
        basis_shape = [1] * indices.dim() + [self.num_dims]
        basis = self.basis.to(indices.device).view(basis_shape)
        levels = self.levels.to(indices.device).view(basis_shape)
        
        indices_expanded = indices.unsqueeze(-1)
        digits = (indices_expanded // basis) % levels
        quantized_low = (digits.float() / (levels.float() - 1.0)) * 2.0 - 1.0
        
        shape = quantized_low.shape
        flat = quantized_low.view(-1, self.num_dims, 1, 1)
        z_q_flat = self.post_proj(flat)
        return z_q_flat.view(*shape[:-1], self.embedding_dim)

    def get_codebook_usage(self) -> Tensor:
        total = self.codebook_usage.sum()
        if total > 0:
            return self.codebook_usage / total
        return self.codebook_usage


# ============================================================================
# COORDCONV
# ============================================================================

class CoordConv2d(nn.Module):
    """
    Conv2d with explicit normalized coordinate channels.

    Appends x/y channels in [-1, 1] before applying convolution so the model
    can learn absolute boundary-aware behaviors on fixed-size room grids.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels + 2,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _c, h, w = x.shape
        device = x.device
        dtype = x.dtype

        y_coords = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        x_coords = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)

        y_grid = y_coords.view(1, 1, h, 1).expand(b, 1, h, w)
        x_grid = x_coords.view(1, 1, 1, w).expand(b, 1, h, w)

        return self.conv(torch.cat([x, x_grid, y_grid], dim=1))


# ============================================================================
# RESIDUAL BLOCK
# ============================================================================

class ResidualBlock(nn.Module):
    """
    Residual block with GroupNorm and SiLU activation.
    
    Architecture:
    x -> GroupNorm -> SiLU -> Conv -> GroupNorm -> SiLU -> Conv -> + -> out
    └─────────────────── (skip connection) ────────────────┘
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 8,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.norm1 = nn.GroupNorm(
            num_groups=min(groups, in_channels), 
            num_channels=in_channels
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.norm2 = nn.GroupNorm(
            num_groups=min(groups, out_channels), 
            num_channels=out_channels
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
        
        self.activation = nn.SiLU()
    
    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        h = self.activation(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = self.activation(h)
        h = self.conv2(h)
        
        return h + self.skip(x)


# ============================================================================
# ENCODER
# ============================================================================

class Encoder(nn.Module):
    """
    Convolutional encoder for semantic grids.
    
    Architecture:
    Input [B, C, H, W] -> Conv -> ResBlocks -> Downsample -> ... -> Latent [B, D, H', W']
    
    For 11×16 rooms with 2 downsamples: output is [B, D, 2, 4] or similar
    """
    
    def __init__(
        self,
        in_channels: int = 44,
        hidden_channels: int = 128,
        latent_channels: int = 64,
        hidden_dims: Optional[List[int]] = None,
        latent_dim: Optional[int] = None,
        num_res_blocks: int = 2,
        channel_mult: Tuple[int, ...] = (1, 2, 4),
        downsample_factor: int = 2,
        use_coordconv: bool = True,
    ):
        super().__init__()

        if latent_dim is not None:
            latent_channels = int(latent_dim)

        if hidden_dims is not None and len(hidden_dims) > 0:
            block_channels = [int(max(1, c)) for c in hidden_dims]
            hidden_channels = int(block_channels[0])
        else:
            block_channels = [int(hidden_channels * mult) for mult in channel_mult]
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.latent_channels = latent_channels
        
        # Initial projection (CoordConv improves absolute spatial awareness).
        if bool(use_coordconv):
            self.conv_in = CoordConv2d(in_channels, block_channels[0], 3, padding=1)
        else:
            self.conv_in = nn.Conv2d(in_channels, block_channels[0], 3, padding=1)
        
        # Build encoder blocks
        self.down_blocks = nn.ModuleList()
        
        ch = block_channels[0]
        for i, out_ch in enumerate(block_channels):
            
            block = nn.ModuleList()
            
            # Residual blocks
            for _ in range(num_res_blocks):
                block.append(ResidualBlock(ch, out_ch))
                ch = out_ch
            
            # Downsample (except last level)
            if i < len(block_channels) - 1:
                block.append(
                    nn.Conv2d(ch, ch, 3, stride=downsample_factor, padding=1)
                )
            
            self.down_blocks.append(block)
        
        # Final layers
        self.norm_out = nn.GroupNorm(min(8, ch), ch)
        self.conv_out = nn.Conv2d(ch, latent_channels, 3, padding=1)
        
        self.activation = nn.SiLU()
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Latent tensor [B, D, H', W']
        """
        h = self.conv_in(x)
        
        for block in self.down_blocks:
            for layer in block:
                h = layer(h)
        
        h = self.norm_out(h)
        h = self.activation(h)
        h = self.conv_out(h)
        
        return h


# ============================================================================
# DECODER
# ============================================================================

class Decoder(nn.Module):
    """
    Transposed convolutional decoder for semantic grids.
    
    Architecture:
    Latent [B, D, H', W'] -> Conv -> Upsample -> ResBlocks -> ... -> Output [B, C, H, W]
    """
    
    def __init__(
        self,
        out_channels: int = 44,
        hidden_channels: int = 128,
        latent_channels: int = 64,
        hidden_dims: Optional[List[int]] = None,
        latent_dim: Optional[int] = None,
        num_res_blocks: int = 2,
        channel_mult: Tuple[int, ...] = (4, 2, 1),
        upsample_factor: int = 2,
    ):
        super().__init__()

        if latent_dim is not None:
            latent_channels = int(latent_dim)

        if hidden_dims is not None and len(hidden_dims) > 0:
            block_channels = [int(max(1, c)) for c in hidden_dims]
            hidden_channels = int(block_channels[-1])
        else:
            block_channels = [int(hidden_channels * mult) for mult in channel_mult]
        
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        
        # Initial projection
        ch = block_channels[0]
        self.conv_in = nn.Conv2d(latent_channels, ch, 3, padding=1)
        
        # Build decoder blocks
        self.up_blocks = nn.ModuleList()
        
        for i, out_ch in enumerate(block_channels):
            
            block = nn.ModuleList()
            
            # Residual blocks
            for _ in range(num_res_blocks):
                block.append(ResidualBlock(ch, out_ch))
                ch = out_ch
            
            # Upsample (except last level)
            if i < len(block_channels) - 1:
                is_final_upsample = i == len(block_channels) - 2
                if int(upsample_factor) == 2 and is_final_upsample:
                    block.append(
                        nn.ConvTranspose2d(
                            ch,
                            ch,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            output_padding=(1, 0),
                        )
                    )
                else:
                    block.append(
                        nn.ConvTranspose2d(
                            ch, ch, 4, stride=upsample_factor, padding=1
                        )
                    )
            
            self.up_blocks.append(block)
        
        # Final layers
        self.norm_out = nn.GroupNorm(min(8, ch), ch)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)
        
        self.activation = nn.SiLU()
    
    def forward(self, z: Tensor, target_size: Optional[Tuple[int, int]] = None) -> Tensor:
        """
        Decode latent to output.
        
        Args:
            z: Latent tensor [B, D, H', W']
            target_size: Optional (H, W) to resize output
            
        Returns:
            Output tensor [B, C, H, W]
        """
        h = self.conv_in(z)
        
        for block in self.up_blocks:
            for layer in block:
                h = layer(h)
        
        h = self.norm_out(h)
        h = self.activation(h)
        h = self.conv_out(h)
        
        # Resize to target if needed
        if target_size is not None:
            h = F.interpolate(h, size=target_size, mode='bilinear', align_corners=False)
        
        return h


# ============================================================================
# SEMANTIC VQ-VAE
# ============================================================================

class SemanticVQVAE(nn.Module):
    """
    Semantic Vector Quantized VAE for Zelda Dungeon Rooms.
    
    This model learns a discrete vocabulary of room patterns that can be
    used by the latent diffusion model for generation.
    
    Key Features:
    - Semantic-aware reconstruction loss (weighted for rare tiles)
    - Learnable codebook with EMA updates
    - Straight-through gradient estimation
    
    Architecture:
        Input (11×16×C) -> Encoder -> VQ -> Decoder -> Output (11×16×C)
    
    Args:
        num_classes: Number of semantic tile classes (default: 44)
        codebook_size: Number of codebook entries K (default: 512)
        latent_dim: Dimension of latent vectors D (default: 64)
        hidden_dim: Hidden channel dimension (default: 128)
        commitment_cost: VQ commitment loss weight β (default: 0.25)
        rare_tile_weight: Weight multiplier for rare tiles (default: 5.0)
    
    Usage:
        model = SemanticVQVAE(num_classes=44, codebook_size=512)
        
        # Forward pass
        x = room_tensor  # [B, C, H, W]
        recon, indices, losses = model(x)
        
        # Encode only
        z_q, indices = model.encode(x)
        
        # Decode only
        recon = model.decode(z_q)
    """
    
    # Tile IDs that are considered "rare" (keys, doors, items, etc.)
    RARE_TILES = [10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 30, 31, 32, 33, 40, 41, 42, 43]
    
    def __init__(
        self,
        num_classes: int = 44,
        num_tile_classes: Optional[int] = None,
        codebook_size: int = 512,
        num_embeddings: Optional[int] = None,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        hidden_dims: Optional[List[int]] = None,
        num_res_blocks: int = 2,
        commitment_cost: float = 0.25,
        rare_tile_weight: float = 5.0,
        use_codebook: bool = True,
        quantizer_type: str = "vq",
        fsq_levels: Optional[Iterable[int]] = None,
        fsq_num_dims: Optional[int] = None,
        use_ema: bool = True,
        use_coordconv: bool = True,
        mrf_penalty_weight: float = 0.05,
        dead_code_reset_interval: int = 100,
        dead_code_threshold: float = 0.05,
        dead_code_warmup_steps: int = 500,
        protect_active_codes_during_reset: bool = True,
        max_dead_code_resets_per_event: int = 16,
        ema_decay_warmup_steps: int = 100,
    ):
        super().__init__()

        if num_tile_classes is not None:
            num_classes = int(num_tile_classes)
        if num_embeddings is not None:
            codebook_size = int(num_embeddings)
        
        self.num_classes = num_classes
        self.use_codebook = bool(use_codebook)
        self.codebook_size = int(codebook_size) if self.use_codebook else 0
        self.latent_dim = latent_dim
        self.rare_tile_weight = rare_tile_weight
        self.mrf_penalty_weight = float(max(0.0, mrf_penalty_weight))
        self.quantizer_type = str(quantizer_type or "vq").strip().lower()
        
        # Encoder
        self.encoder = Encoder(
            in_channels=num_classes,
            hidden_channels=hidden_dim,
            latent_channels=latent_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            num_res_blocks=num_res_blocks,
            channel_mult=(1, 2, 4),
            use_coordconv=bool(use_coordconv),
        )
        
        # Vector Quantizer (optional for the no-codebook baseline)
        if self.use_codebook:
            if self.quantizer_type == "fsq":
                requested_codebook_size = int(codebook_size)
                self.quantizer = FSQuantizer(
                    embedding_dim=latent_dim,
                    levels=fsq_levels,
                    num_dims=fsq_num_dims,
                )
                self.codebook_size = int(self.quantizer.num_embeddings)
                if requested_codebook_size != self.codebook_size:
                    logger.warning(
                        "FSQ ignores learned VQ codebook_size=%d; using implicit product(levels)=%d.",
                        requested_codebook_size,
                        self.codebook_size,
                    )
            elif self.quantizer_type in {"vq", "vector", "vector_quantizer"}:
                self.quantizer = VectorQuantizer(
                    num_embeddings=codebook_size,
                    embedding_dim=latent_dim,
                    commitment_cost=commitment_cost,
                    use_ema=use_ema,
                    dead_code_reset_interval=dead_code_reset_interval,
                    dead_code_threshold=dead_code_threshold,
                    dead_code_warmup_steps=dead_code_warmup_steps,
                    protect_active_codes_during_reset=protect_active_codes_during_reset,
                    max_dead_code_resets_per_event=max_dead_code_resets_per_event,
                    ema_decay_warmup_steps=ema_decay_warmup_steps,
                )
            else:
                raise ValueError("quantizer_type must be 'vq' or 'fsq'.")
        else:
            self.quantizer = None
        
        # Decoder
        self.decoder = Decoder(
            out_channels=num_classes,
            hidden_channels=hidden_dim,
            latent_channels=latent_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            num_res_blocks=num_res_blocks,
            channel_mult=(4, 2, 1),
        )
        
        # Build rare tile mask for weighted loss
        self.register_buffer(
            'tile_weights',
            self._build_tile_weights()
        )
        self.register_buffer('illegal_adjacency_matrix', self._build_illegal_adjacency_matrix())
    
    def _build_tile_weights(self) -> Tensor:
        """Build weight tensor for semantic-aware reconstruction."""
        weights = torch.ones(self.num_classes)
        for tile_id in self.RARE_TILES:
            if tile_id < self.num_classes:
                weights[tile_id] = self.rare_tile_weight
        return weights

    def _build_illegal_adjacency_matrix(self) -> Tensor:
        """
        Build static illegal tile-pair mask for soft MRF-style penalties.

        Mask value 1 means adjacency is illegal and should be penalized.
        """
        m = torch.zeros(self.num_classes, self.num_classes, dtype=torch.float32)

        door_ids = [
            int(TileID.DOOR_OPEN),
            int(TileID.DOOR_LOCKED),
            int(TileID.DOOR_BOMB),
            int(TileID.DOOR_PUZZLE),
            int(TileID.DOOR_BOSS),
            int(TileID.DOOR_SOFT),
        ]
        element_ids = [
            int(TileID.ELEMENT),
        ]

        # Example hard rule from user request: Water/Element next to doors is illegal.
        for a in element_ids:
            if a >= self.num_classes:
                continue
            for b in door_ids:
                if b >= self.num_classes:
                    continue
                m[a, b] = 1.0
                m[b, a] = 1.0

        return m

    def _empty_indices(self, batch_size: int, height: int, width: int, device: torch.device) -> Tensor:
        return torch.zeros((batch_size, height, width), dtype=torch.long, device=device)

    def _illegal_adjacency_penalty(self, recon_logits: Tensor) -> Tensor:
        """
        Differentiable soft penalty over 3x3 neighborhood illegal adjacencies.
        """
        probs = F.softmax(recon_logits.float(), dim=1).to(dtype=recon_logits.dtype)
        illegal = self.illegal_adjacency_matrix.to(dtype=probs.dtype, device=probs.device)

        # 8-neighborhood shifts (3x3 window excluding center).
        shifts = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

        total = torch.tensor(0.0, device=probs.device, dtype=probs.dtype)
        padded = F.pad(probs, (1, 1, 1, 1), mode='constant', value=0.0)
        h = probs.shape[2]
        w = probs.shape[3]

        for dy, dx in shifts:
            y0 = 1 + dy
            x0 = 1 + dx
            neighbor = padded[:, :, y0:y0 + h, x0:x0 + w]
            # Expected illegal pair probability per pixel.
            pair_score = torch.einsum('bchw,cd,bdhw->bhw', probs, illegal, neighbor)
            total = total + pair_score.mean()

        return total / float(len(shifts))
    
    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode input to quantized latent.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            z_q: Quantized latent [B, D, H', W']
            indices: Codebook indices [B, H', W']
        """
        z_e = self.encoder(x)
        if self.use_codebook:
            z_q, _, indices = self.quantizer(z_e)
            return z_q, indices

        indices = self._empty_indices(z_e.shape[0], z_e.shape[2], z_e.shape[3], z_e.device)
        return z_e, indices

    def quantize(self, z_e: Tensor | Tuple[Tensor, Any]) -> Tuple[Tensor, Tensor, Tensor]:
        """Backward-compatible quantize helper returning (z_q, vq_loss, indices)."""
        if isinstance(z_e, (tuple, list)):
            z_e = z_e[0]
        if self.use_codebook:
            return self.quantizer(z_e)

        indices = self._empty_indices(z_e.shape[0], z_e.shape[2], z_e.shape[3], z_e.device)
        zero_loss = torch.zeros((), device=z_e.device, dtype=z_e.dtype)
        return z_e, zero_loss, indices
    
    def decode(
        self, 
        z_q: Tensor, 
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        """
        Decode quantized latent to output.
        
        Args:
            z_q: Quantized latent [B, D, H', W']
            target_size: Optional output size (H, W)
            
        Returns:
            Output logits [B, C, H, W]
        """
        if target_size is None:
            # Internal room tensors use row-major `(rows, cols)`.
            target_size = (ROOM_HEIGHT, ROOM_WIDTH)
        else:
            target_size = normalize_room_shape(target_size)
        return self.decoder(z_q, target_size)
    
    def decode_indices(
        self, 
        indices: Tensor,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        """
        Decode from codebook indices directly.
        
        Args:
            indices: Codebook indices [B, H', W']
            target_size: Optional output size (H, W)
            
        Returns:
            Output logits [B, C, H, W]
        """
        if not self.use_codebook or self.quantizer is None:
            raise RuntimeError("decode_indices is only available when use_codebook=True.")
        z_q = self.quantizer.encode_indices(indices)  # [B, H', W', D]
        z_q = z_q.permute(0, 3, 1, 2).contiguous()   # [B, D, H', W']
        return self.decode(z_q, target_size)
    
    def forward_with_losses(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """
        Forward pass: encode, quantize, decode.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            recon: Reconstructed logits [B, C, H, W]
            indices: Codebook indices [B, H', W']
            losses: Dict containing all loss components
        """
        input_size = (x.shape[2], x.shape[3])
        
        # Encode
        z_e = self.encoder(x)
        
        # Quantize or bypass codebook for the no-codebook baseline
        if self.use_codebook:
            z_q, indices, vq_losses = self.quantizer(z_e, return_info=True)
        else:
            z_q = z_e
            indices = self._empty_indices(z_e.shape[0], z_e.shape[2], z_e.shape[3], z_e.device)
            zero = torch.zeros((), device=z_e.device, dtype=z_e.dtype)
            vq_losses = {
                'vq_loss': zero,
                'commitment_loss': zero,
                'perplexity': zero,
            }
        
        # Decode
        recon = self.decoder(z_q, target_size=input_size)
        
        # Compute reconstruction loss
        losses = vq_losses.copy()
        
        # Weighted cross-entropy for semantic preservation
        recon_loss = self._weighted_reconstruction_loss(recon, x)
        losses['recon_loss'] = recon_loss

        illegal_penalty = self._illegal_adjacency_penalty(recon)
        losses['illegal_adjacency_penalty'] = illegal_penalty
        
        # Total loss
        losses['total_loss'] = recon_loss + losses['vq_loss'] + (self.mrf_penalty_weight * illegal_penalty)
        
        return recon, indices, losses

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """Backward-compatible forward returning (recon, vq_loss, losses)."""
        recon, indices, losses = self.forward_with_losses(x)
        losses = dict(losses)
        losses['indices'] = indices
        return recon, losses['vq_loss'], losses

    def compute_loss(self, x: Tensor) -> Dict[str, Tensor]:
        """Return detailed loss dictionary for training/evaluation code."""
        _, _, losses = self.forward_with_losses(x)
        return losses
    
    def _weighted_reconstruction_loss(
        self, 
        recon: Tensor, 
        target: Tensor,
    ) -> Tensor:
        """
        Compute weighted reconstruction loss.
        
        Applies higher weight to rare tiles (keys, items, etc.)
        to ensure they are preserved during reconstruction.
        
        Args:
            recon: Predicted logits [B, C, H, W]
            target: Target one-hot or class indices
            
        Returns:
            Weighted reconstruction loss
        """
        # Handle one-hot vs class index targets
        if target.shape[1] == self.num_classes:
            # One-hot encoded - convert to class indices
            target_idx = target.argmax(dim=1)  # [B, H, W]
        else:
            target_idx = target.squeeze(1)
        
        # Get per-class weights
        weights = self.tile_weights[target_idx]  # [B, H, W]
        
        # Cross-entropy loss with class weights
        loss = F.cross_entropy(
            recon, 
            target_idx,
            reduction='none'
        )  # [B, H, W]
        
        # Apply tile weights
        weighted_loss = (loss * weights).mean()
        
        return weighted_loss
    
    def get_codebook(self) -> Tensor:
        """Get the learned codebook embeddings."""
        if not self.use_codebook or self.quantizer is None:
            raise RuntimeError("get_codebook is only available when use_codebook=True.")
        if not hasattr(self.quantizer, "embedding"):
            raise RuntimeError("FSQ uses an implicit scalar codebook; get_codebook is not available.")
        return self.quantizer.embedding.weight.data

    def get_codebook_usage(self) -> Tensor:
        """Get codebook usage statistics."""
        if not self.use_codebook or self.quantizer is None:
            raise RuntimeError("get_codebook_usage is only available when use_codebook=True.")
        return self.quantizer.get_codebook_usage()


class SemanticVQVAE2(nn.Module):
    """
    Two-level hierarchical VQ-VAE for room grids.

    This follows the VQ-VAE-2 ablation idea: a coarse top codebook captures
    global room structure, while a bottom codebook captures local tile detail.
    For downstream compatibility with the existing latent-diffusion branch,
    encode() returns a single fused latent tensor with the same channel width as
    `SemanticVQVAE`.
    """

    RARE_TILES = SemanticVQVAE.RARE_TILES

    def __init__(
        self,
        num_classes: int = 44,
        num_tile_classes: Optional[int] = None,
        codebook_size: int = 512,
        top_codebook_size: Optional[int] = None,
        num_embeddings: Optional[int] = None,
        latent_dim: int = 64,
        top_latent_dim: Optional[int] = None,
        hidden_dim: int = 128,
        hidden_dims: Optional[List[int]] = None,
        num_res_blocks: int = 2,
        commitment_cost: float = 0.25,
        rare_tile_weight: float = 5.0,
        use_codebook: bool = True,
        quantizer_type: str = "vq",
        fsq_levels: Optional[Iterable[int]] = None,
        fsq_num_dims: Optional[int] = None,
        use_ema: bool = True,
        use_coordconv: bool = True,
        mrf_penalty_weight: float = 0.05,
        dead_code_reset_interval: int = 100,
        dead_code_threshold: float = 0.05,
        dead_code_warmup_steps: int = 500,
        protect_active_codes_during_reset: bool = True,
        max_dead_code_resets_per_event: int = 16,
    ):
        super().__init__()

        if num_tile_classes is not None:
            num_classes = int(num_tile_classes)
        if num_embeddings is not None:
            codebook_size = int(num_embeddings)
        if top_codebook_size is None:
            top_codebook_size = max(8, int(codebook_size) // 2)
        if top_latent_dim is None:
            top_latent_dim = int(latent_dim)

        self.num_classes = int(num_classes)
        self.use_codebook = bool(use_codebook)
        self.codebook_size = int(codebook_size) if self.use_codebook else 0
        self.top_codebook_size = int(top_codebook_size) if self.use_codebook else 0
        self.latent_dim = int(latent_dim)
        self.top_latent_dim = int(top_latent_dim)
        self.rare_tile_weight = float(rare_tile_weight)
        self.mrf_penalty_weight = float(max(0.0, mrf_penalty_weight))
        self.quantizer_type = str(quantizer_type or "vq").strip().lower()

        self.encoder = Encoder(
            in_channels=self.num_classes,
            hidden_channels=hidden_dim,
            latent_channels=self.latent_dim,
            hidden_dims=hidden_dims,
            latent_dim=self.latent_dim,
            num_res_blocks=num_res_blocks,
            channel_mult=(1, 2, 4),
            use_coordconv=bool(use_coordconv),
        )
        self.top_encoder = nn.Sequential(
            ResidualBlock(self.latent_dim, self.latent_dim),
            nn.Conv2d(self.latent_dim, self.top_latent_dim, kernel_size=3, stride=2, padding=1),
        )
        self.top_context_proj = nn.Conv2d(self.top_latent_dim, self.latent_dim, kernel_size=1)
        self.bottom_condition_proj = nn.Conv2d(self.latent_dim * 2, self.latent_dim, kernel_size=1)
        self.fusion = nn.Sequential(
            nn.Conv2d(self.latent_dim * 2, self.latent_dim, kernel_size=1),
            ResidualBlock(self.latent_dim, self.latent_dim),
        )

        if self.use_codebook:
            self.top_quantizer = VectorQuantizer(
                num_embeddings=int(top_codebook_size),
                embedding_dim=self.top_latent_dim,
                commitment_cost=commitment_cost,
                use_ema=use_ema,
                dead_code_reset_interval=dead_code_reset_interval,
                dead_code_threshold=dead_code_threshold,
                dead_code_warmup_steps=dead_code_warmup_steps,
                protect_active_codes_during_reset=protect_active_codes_during_reset,
                max_dead_code_resets_per_event=max_dead_code_resets_per_event,
            )
            self.bottom_quantizer = VectorQuantizer(
                num_embeddings=int(codebook_size),
                embedding_dim=self.latent_dim,
                commitment_cost=commitment_cost,
                use_ema=use_ema,
                dead_code_reset_interval=dead_code_reset_interval,
                dead_code_threshold=dead_code_threshold,
                dead_code_warmup_steps=dead_code_warmup_steps,
                protect_active_codes_during_reset=protect_active_codes_during_reset,
                max_dead_code_resets_per_event=max_dead_code_resets_per_event,
            )
            # Compatibility for code paths that expect `model.quantizer`.
            self.quantizer = self.bottom_quantizer
        else:
            self.top_quantizer = None
            self.bottom_quantizer = None
            self.quantizer = None

        self.decoder = Decoder(
            out_channels=self.num_classes,
            hidden_channels=hidden_dim,
            latent_channels=self.latent_dim,
            hidden_dims=hidden_dims,
            latent_dim=self.latent_dim,
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
        helper = SemanticVQVAE(
            num_classes=self.num_classes,
            codebook_size=max(8, self.codebook_size or 8),
            latent_dim=self.latent_dim,
            use_codebook=False,
        )
        return helper._build_illegal_adjacency_matrix()

    def _empty_indices(self, batch_size: int, height: int, width: int, device: torch.device) -> Tensor:
        return torch.zeros((batch_size, height, width), dtype=torch.long, device=device)

    def _illegal_adjacency_penalty(self, recon_logits: Tensor) -> Tensor:
        probs = F.softmax(recon_logits.float(), dim=1).to(dtype=recon_logits.dtype)
        illegal = self.illegal_adjacency_matrix.to(dtype=probs.dtype, device=probs.device)
        shifts = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
        total = torch.tensor(0.0, device=probs.device, dtype=probs.dtype)
        padded = F.pad(probs, (1, 1, 1, 1), mode="constant", value=0.0)
        h = probs.shape[2]
        w = probs.shape[3]
        for dy, dx in shifts:
            y0 = 1 + dy
            x0 = 1 + dx
            neighbor = padded[:, :, y0:y0 + h, x0:x0 + w]
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

    def _encode_hierarchy(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        bottom_e = self.encoder(x)
        top_e = self.top_encoder(bottom_e)

        if self.use_codebook:
            top_z, top_indices, top_losses = self.top_quantizer(top_e, return_info=True)
        else:
            top_z = top_e
            top_indices = self._empty_indices(top_e.shape[0], top_e.shape[2], top_e.shape[3], top_e.device)
            zero = torch.zeros((), device=top_e.device, dtype=top_e.dtype)
            top_losses = {"vq_loss": zero, "commitment_loss": zero, "perplexity": zero}

        top_context = self.top_context_proj(top_z)
        top_context = F.interpolate(top_context, size=bottom_e.shape[-2:], mode="nearest")
        bottom_e_conditioned = self.bottom_condition_proj(torch.cat([bottom_e, top_context], dim=1))

        if self.use_codebook:
            bottom_z, bottom_indices, bottom_losses = self.bottom_quantizer(bottom_e_conditioned, return_info=True)
        else:
            bottom_z = bottom_e_conditioned
            bottom_indices = self._empty_indices(
                bottom_e_conditioned.shape[0],
                bottom_e_conditioned.shape[2],
                bottom_e_conditioned.shape[3],
                bottom_e_conditioned.device,
            )
            zero = torch.zeros((), device=bottom_e_conditioned.device, dtype=bottom_e_conditioned.dtype)
            bottom_losses = {"vq_loss": zero, "commitment_loss": zero, "perplexity": zero}

        fused = self.fusion(torch.cat([bottom_z, top_context], dim=1))
        losses = {
            "top_vq_loss": top_losses["vq_loss"],
            "bottom_vq_loss": bottom_losses["vq_loss"],
            "vq_loss": top_losses["vq_loss"] + bottom_losses["vq_loss"],
            "top_commitment_loss": top_losses.get("commitment_loss", torch.zeros_like(top_losses["vq_loss"])),
            "bottom_commitment_loss": bottom_losses.get("commitment_loss", torch.zeros_like(bottom_losses["vq_loss"])),
            "top_perplexity": top_losses["perplexity"],
            "bottom_perplexity": bottom_losses["perplexity"],
            "perplexity": 0.5 * (top_losses["perplexity"] + bottom_losses["perplexity"]),
        }
        return fused, bottom_indices, top_indices, losses

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z_q, bottom_indices, _top_indices, _losses = self._encode_hierarchy(x)
        return z_q, bottom_indices

    def quantize(self, z_e: Tensor | Tuple[Tensor, Any]) -> Tuple[Tensor, Tensor, Tensor]:
        if isinstance(z_e, (tuple, list)):
            z_e = z_e[0]
        if self.use_codebook and self.bottom_quantizer is not None:
            return self.bottom_quantizer(z_e)
        indices = self._empty_indices(z_e.shape[0], z_e.shape[2], z_e.shape[3], z_e.device)
        zero_loss = torch.zeros((), device=z_e.device, dtype=z_e.dtype)
        return z_e, zero_loss, indices

    def decode(self, z_q: Tensor, target_size: Optional[Tuple[int, int]] = None) -> Tensor:
        if target_size is None:
            target_size = (ROOM_HEIGHT, ROOM_WIDTH)
        else:
            target_size = normalize_room_shape(target_size)
        return self.decoder(z_q, target_size)

    def decode_indices(self, indices: Tensor, target_size: Optional[Tuple[int, int]] = None) -> Tensor:
        if not self.use_codebook or self.bottom_quantizer is None:
            raise RuntimeError("decode_indices is only available when use_codebook=True.")
        z_q = self.bottom_quantizer.encode_indices(indices).permute(0, 3, 1, 2).contiguous()
        return self.decode(z_q, target_size)

    def forward_with_losses(self, x: Tensor) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        input_size = (x.shape[2], x.shape[3])
        z_q, bottom_indices, top_indices, losses = self._encode_hierarchy(x)
        recon = self.decoder(z_q, target_size=input_size)
        recon_loss = self._weighted_reconstruction_loss(recon, x)
        illegal_penalty = self._illegal_adjacency_penalty(recon)
        losses = dict(losses)
        losses["recon_loss"] = recon_loss
        losses["illegal_adjacency_penalty"] = illegal_penalty
        losses["top_indices"] = top_indices
        losses["total_loss"] = recon_loss + losses["vq_loss"] + (self.mrf_penalty_weight * illegal_penalty)
        return recon, bottom_indices, losses

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        recon, indices, losses = self.forward_with_losses(x)
        losses = dict(losses)
        losses["indices"] = indices
        return recon, losses["vq_loss"], losses

    def compute_loss(self, x: Tensor) -> Dict[str, Tensor]:
        _, _, losses = self.forward_with_losses(x)
        return losses

    def get_codebook(self) -> Tensor:
        if not self.use_codebook or self.bottom_quantizer is None:
            raise RuntimeError("get_codebook is only available when use_codebook=True.")
        return self.bottom_quantizer.embedding.weight.data

    def get_codebook_usage(self) -> Tensor:
        if not self.use_codebook or self.bottom_quantizer is None:
            raise RuntimeError("get_codebook_usage is only available when use_codebook=True.")
        return self.bottom_quantizer.get_codebook_usage()

    def get_hierarchical_codebook_usage(self) -> Dict[str, Tensor]:
        if not self.use_codebook or self.bottom_quantizer is None or self.top_quantizer is None:
            return {}
        return {
            "bottom": self.bottom_quantizer.get_codebook_usage(),
            "top": self.top_quantizer.get_codebook_usage(),
        }


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class VQVAETrainer:
    """
    Training utilities for Semantic VQ-VAE.
    
    Usage:
        model = SemanticVQVAE()
        trainer = VQVAETrainer(model)
        
        for batch in dataloader:
            loss, metrics = trainer.train_step(batch)
    """
    
    def __init__(
        self,
        model: SemanticVQVAE,
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
            adamw_decay_param_groups(
                model.named_parameters(),
                weight_decay=float(weight_decay),
                base_name="vqvae",
            ),
            lr=lr,
            weight_decay=0.0,
        )
    
    def train_step(
        self, 
        batch: Tensor,
        return_metrics: bool = False,
    ) -> float | Tuple[float, Dict[str, float]]:
        """
        Single training step.
        
        Args:
            batch: Input batch [B, C, H, W]
            
        Returns:
            loss: Total loss tensor
            metrics: Dict of metric values
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward
        losses = self.model.compute_loss(batch)
        
        # Backward
        loss = losses['total_loss']
        loss.backward()
        
        # Gradient clipping
        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        
        self.optimizer.step()
        
        # Compute metrics
        metrics = {
            'loss': loss.item(),
            'recon_loss': losses['recon_loss'].item(),
            'vq_loss': losses['vq_loss'].item(),
            'illegal_adjacency_penalty': losses.get('illegal_adjacency_penalty', torch.tensor(0.0)).item(),
            'perplexity': losses['perplexity'].item(),
        }
        
        loss_value = float(loss.item())
        if return_metrics:
            return loss_value, metrics
        return loss_value
    
    @torch.no_grad()
    def eval_step(self, batch: Tensor) -> Dict[str, float]:
        """Evaluation step."""
        self.model.eval()
        
        recon, _, losses = self.model.forward_with_losses(batch)
        
        # Compute accuracy
        pred = recon.argmax(dim=1)
        target = batch.argmax(dim=1) if batch.shape[1] > 1 else batch.squeeze(1)
        accuracy = (pred == target).float().mean()
        
        return {
            'loss': losses['total_loss'].item(),
            'recon_loss': losses['recon_loss'].item(),
            'illegal_adjacency_penalty': losses.get('illegal_adjacency_penalty', torch.tensor(0.0)).item(),
            'accuracy': accuracy.item(),
            'perplexity': losses['perplexity'].item(),
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_vqvae(
    num_classes: int = 44,
    codebook_size: int = 512,
    latent_dim: int = 64,
    use_codebook: bool = True,
    architecture: str = "vqvae",
    **kwargs,
) -> SemanticVQVAE | SemanticVQVAE2:
    """
    Create a Semantic VQ-VAE model.
    
    Args:
        num_classes: Number of semantic tile classes
        codebook_size: Number of codebook entries
        latent_dim: Latent vector dimension
        **kwargs: Additional arguments for SemanticVQVAE
        
    Returns:
        SemanticVQVAE model
    """
    model_variant = kwargs.pop("model_variant", None)
    top_codebook_size = kwargs.pop("top_codebook_size", None)
    top_latent_dim = kwargs.pop("top_latent_dim", None)
    arch_source = model_variant if model_variant is not None else architecture
    arch = str(arch_source or "vqvae").strip().lower().replace("-", "")
    if arch in {"fsq", "semanticfsq", "fsqvae", "fsqvqvae"}:
        return SemanticVQVAE(
            num_classes=num_classes,
            codebook_size=codebook_size,
            latent_dim=latent_dim,
            use_codebook=use_codebook,
            quantizer_type="fsq",
            **kwargs,
        )
    if arch in {"vqvae2", "vqvaeii", "hierarchical", "hierarchicalvqvae"}:
        return SemanticVQVAE2(
            num_classes=num_classes,
            codebook_size=codebook_size,
            latent_dim=latent_dim,
            use_codebook=use_codebook,
            top_codebook_size=top_codebook_size,
            top_latent_dim=top_latent_dim,
            **kwargs,
        )
    if arch not in {"vqvae", "semanticvqvae", "single"}:
        raise ValueError(f"Unsupported VQ-VAE architecture {architecture!r}. Expected 'vqvae', 'vqvae2', or 'fsq'.")
    return SemanticVQVAE(
        num_classes=num_classes,
        codebook_size=codebook_size,
        latent_dim=latent_dim,
        use_codebook=use_codebook,
        **kwargs,
    )
