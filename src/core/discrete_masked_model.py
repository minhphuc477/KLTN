"""
Graph-conditioned discrete masked room model.

This module provides a MaskGIT-style parallel masked-token generator for Zelda
rooms. It is intentionally implemented as a parallel alternative to the
existing VQ-VAE + latent diffusion path rather than a hard replacement.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.core.definitions import (
    ROOM_HEIGHT,
    ROOM_TOPOLOGY_CHANNEL_COUNT,
    ROOM_TOPOLOGY_CHANNELS,
    ROOM_WIDTH,
    SEMANTIC_PALETTE,
    TileID,
)

logger = logging.getLogger(__name__)


class ModelContextContractError(Exception):
    """Deterministic graph-context contract failure that retries cannot repair."""

    retryable = False


class _DisabledTransformerDecoder(nn.Module):
    """No-parameter placeholder for concat mode where cross-decoder is disabled."""

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        raise RuntimeError("TransformerDecoder is only available when context_attention_mode='cross_decoder'.")


class MaskedTokenTransformerBackbone(nn.Module):
    """Bidirectional MaskGIT-style backbone for masked token prediction."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        context_dim: int,
        num_steps: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        room_topology_channels: int,
        context_attention_mode: str = "concat_encoder",
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        mode = str(context_attention_mode).strip().lower()
        if mode in {"concat", "encoder", "original"}:
            mode = "concat_encoder"
        elif mode in {"cross", "decoder", "cross_attention"}:
            mode = "cross_decoder"
        if mode not in {"concat_encoder", "cross_decoder"}:
            raise ValueError(
                "context_attention_mode must be 'concat_encoder' or 'cross_decoder', "
                f"got {context_attention_mode!r}."
            )
        self.context_attention_mode = mode
        num_heads = int(max(1, num_heads))
        if self.hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} must be divisible by num_heads={num_heads} "
                "for the masked-token transformer backbone."
            )
        self.context_proj = nn.Linear(int(context_dim), self.hidden_dim)
        self.step_embedding = nn.Embedding(int(max(1, num_steps)), self.hidden_dim)
        self.room_topology_proj = nn.Conv2d(int(room_topology_channels), self.hidden_dim, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=max(self.hidden_dim * 4, self.hidden_dim),
            dropout=float(max(0.0, min(1.0, dropout))),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Original MaskGIT path: concatenate context and room tokens before
        # encoder self-attention. Keep this as the default checkpoint-compatible
        # baseline; cross_decoder is an explicit ablation.
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(max(1, num_layers)))
        if self.context_attention_mode == "cross_decoder":
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.hidden_dim,
                nhead=num_heads,
                dim_feedforward=max(self.hidden_dim * 4, self.hidden_dim),
                dropout=float(max(0.0, min(1.0, dropout))),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=int(max(1, num_layers)))
        else:
            self.decoder = _DisabledTransformerDecoder()
        self.norm = nn.LayerNorm(self.hidden_dim)

    def _context_tokens(self, context: Tensor) -> Tensor:
        if context.dim() == 2:
            context = context.unsqueeze(1)
        if context.dim() != 3:
            raise ModelContextContractError(
                f"context must be [B,C] or [B,N,C], got {tuple(context.shape)}."
            )
        return self.context_proj(context)

    def _context_key_padding_mask(
        self,
        context_tokens: Tensor,
        graph_data: Optional[Dict[str, Tensor]],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if not isinstance(graph_data, dict):
            return context_tokens, None
        node_mask = graph_data.get("node_mask")
        if not isinstance(node_mask, torch.Tensor):
            return context_tokens, None

        batch_size, seq_len, _ = context_tokens.shape
        valid = node_mask.to(device=context_tokens.device, dtype=torch.bool)
        if valid.dim() == 1:
            valid = valid.unsqueeze(0)
        if valid.dim() != 2:
            raise ModelContextContractError(
                f"node_mask must have shape [B,N] or [N], got {tuple(valid.shape)}."
            )
        if int(valid.shape[0]) == 1 and batch_size > 1:
            valid = valid.expand(batch_size, -1)
        if int(valid.shape[0]) != batch_size:
            raise ModelContextContractError(
                f"node_mask batch size {int(valid.shape[0])} does not match context batch size {batch_size}."
            )

        if bool(graph_data.get("has_room_anchor", False)) and int(valid.shape[1]) + 1 == seq_len:
            anchor = torch.ones(batch_size, 1, device=valid.device, dtype=torch.bool)
            valid = torch.cat([anchor, valid], dim=1)
        elif int(valid.shape[1]) != seq_len:
            raise ModelContextContractError(
                "node_mask/context token contract mismatch: "
                f"node_mask has {int(valid.shape[1])} entries but context has {seq_len} tokens. "
                "Refusing to pad or truncate graph context silently; fix the condition encoder "
                "or set has_room_anchor=True only for a single prepended room-anchor token."
            )

        valid_rows = valid.any(dim=1)
        if not bool(valid_rows.all()):
            context_tokens = context_tokens.clone()
            context_tokens[~valid_rows] = 0.0
            valid = valid.clone()
            valid[~valid_rows, 0] = True
        return context_tokens, ~valid

    @staticmethod
    def attention_complexity_metrics(
        *,
        context_tokens: int,
        room_height: int = ROOM_HEIGHT,
        room_width: int = ROOM_WIDTH,
        mode: str = "concat_encoder",
    ) -> Dict[str, float]:
        """Return approximate attention-pair counts for ablation comparison."""
        room_tokens = int(room_height) * int(room_width)
        ctx_tokens = int(max(0, context_tokens))
        normalized = str(mode).strip().lower()
        if normalized in {"concat", "encoder", "original"}:
            normalized = "concat_encoder"
        elif normalized in {"cross", "decoder", "cross_attention"}:
            normalized = "cross_decoder"
        if normalized == "cross_decoder":
            self_attention_pairs = room_tokens * room_tokens
            cross_attention_pairs = room_tokens * ctx_tokens
        else:
            self_attention_pairs = (room_tokens + ctx_tokens) * (room_tokens + ctx_tokens)
            cross_attention_pairs = 0
        total = self_attention_pairs + cross_attention_pairs
        baseline_total = (room_tokens + ctx_tokens) * (room_tokens + ctx_tokens)
        return {
            "room_tokens": float(room_tokens),
            "context_tokens": float(ctx_tokens),
            "self_attention_pairs": float(self_attention_pairs),
            "cross_attention_pairs": float(cross_attention_pairs),
            "total_attention_pairs": float(total),
            "baseline_concat_attention_pairs": float(baseline_total),
            "relative_to_concat": float(total / max(1, baseline_total)),
        }

    def _topology_bias(self, graph_data: Optional[Dict[str, Tensor]], *, batch_size: int, device: torch.device) -> Optional[Tensor]:
        if not isinstance(graph_data, dict):
            return None
        topo = graph_data.get("room_topology_map")
        if not isinstance(topo, torch.Tensor):
            return None
        topo = topo.to(device=device, dtype=self.room_topology_proj.weight.dtype)
        if topo.dim() == 3:
            topo = topo.unsqueeze(0)
        if int(topo.shape[0]) == 1 and batch_size > 1:
            topo = topo.expand(batch_size, -1, -1, -1)
        if tuple(topo.shape[0:1]) != (batch_size,) or tuple(topo.shape[-2:]) != (ROOM_HEIGHT, ROOM_WIDTH):
            raise ValueError(
                f"room_topology_map must match [B,C,{ROOM_HEIGHT},{ROOM_WIDTH}], got {tuple(topo.shape)}."
            )
        return self.room_topology_proj(topo)

    def forward(
        self,
        x: Tensor,
        step: Tensor,
        context: Tensor,
        *,
        graph_data: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        if x.dim() != 4:
            raise ValueError(f"x must be [B,C,H,W], got {tuple(x.shape)}.")
        batch_size, channels, height, width = x.shape
        if channels != self.hidden_dim:
            raise ValueError(f"x has {channels} channels, expected hidden_dim={self.hidden_dim}.")

        topo_bias = self._topology_bias(graph_data, batch_size=batch_size, device=x.device)
        if topo_bias is not None:
            x = x + topo_bias.to(dtype=x.dtype)

        step = step.to(device=x.device, dtype=torch.long).clamp(min=0, max=self.step_embedding.num_embeddings - 1)
        x = x + self.step_embedding(step)[:, :, None, None].to(dtype=x.dtype)
        room_tokens = x.flatten(2).transpose(1, 2)
        context_tokens = self._context_tokens(context).to(device=x.device, dtype=x.dtype)
        if self.context_attention_mode == "cross_decoder":
            if self.decoder is None:
                raise RuntimeError("cross_decoder mode requires a TransformerDecoder instance.")
            context_tokens, memory_key_padding_mask = self._context_key_padding_mask(context_tokens, graph_data)
            encoded_room_tokens = self.encoder(room_tokens)
            encoded_room = self.decoder(
                encoded_room_tokens,
                context_tokens,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        else:
            context_tokens, context_key_padding_mask = self._context_key_padding_mask(context_tokens, graph_data)
            sequence = torch.cat([context_tokens, room_tokens], dim=1)
            sequence_key_padding_mask = None
            if context_key_padding_mask is not None:
                room_key_padding_mask = torch.zeros(
                    batch_size,
                    room_tokens.shape[1],
                    device=context_key_padding_mask.device,
                    dtype=torch.bool,
                )
                sequence_key_padding_mask = torch.cat([context_key_padding_mask, room_key_padding_mask], dim=1)
            encoded = self.encoder(sequence, src_key_padding_mask=sequence_key_padding_mask)
            encoded_room = encoded[:, context_tokens.shape[1]:]
        encoded_room = self.norm(encoded_room)
        return encoded_room.transpose(1, 2).reshape(batch_size, self.hidden_dim, height, width)


class DiscreteMaskedRoomModel(nn.Module):
    """
    MaskGIT-style room generator with graph-conditioned cross-attention.

    Training:
    - randomly mask a subset of room tokens
    - predict original tile IDs on masked positions via cross entropy

    Inference:
    - start from all [MASK] tokens (or a partially fixed canvas)
    - iteratively fill the most confident unknown positions
    """

    def __init__(
        self,
        *,
        num_classes: int = 44,
        hidden_dim: int = 48,
        model_channels: int = 64,
        context_dim: int = 256,
        num_steps: int = 8,
        attention_mode: str = "softmax",
        topology_conditioning_mode: str = "additive",
        hedgehog_feature_dim: int = 32,
        graph_auto_linear_attention_nodes: int = 128,
        spatial_graph_gate_init: float = -2.0,
        spatial_topology_gate_init: float = -2.0,
        unet_channel_mult: Sequence[int] = (1, 2),
        unet_num_res_blocks: int = 1,
        unet_attention_resolutions: Sequence[int] = (0, 1),
        unet_num_heads: int = 4,
        unet_dropout: float = 0.1,
        room_topology_channels: int = ROOM_TOPOLOGY_CHANNEL_COUNT,
        context_attention_mode: str = "concat_encoder",
        mask_token_id: Optional[int] = None,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.context_dim = int(context_dim)
        self.default_num_steps = int(max(1, num_steps))
        mode = str(context_attention_mode).strip().lower()
        if mode in {"concat", "encoder", "original"}:
            mode = "concat_encoder"
        elif mode in {"cross", "decoder", "cross_attention"}:
            mode = "cross_decoder"
        if mode not in {"concat_encoder", "cross_decoder"}:
            raise ValueError(
                "context_attention_mode must be 'concat_encoder' or 'cross_decoder', "
                f"got {context_attention_mode!r}."
            )
        self.context_attention_mode = mode
        self.mask_token_id = int(self.num_classes if mask_token_id is None else mask_token_id)
        self.vocab_size = int(max(self.mask_token_id + 1, self.num_classes + 1))

        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.hidden_dim, ROOM_HEIGHT, ROOM_WIDTH)
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

        ignored_legacy_args = {
            "model_channels": (model_channels, 64),
            "attention_mode": (attention_mode, "softmax"),
            "topology_conditioning_mode": (topology_conditioning_mode, "additive"),
            "hedgehog_feature_dim": (hedgehog_feature_dim, 32),
            "graph_auto_linear_attention_nodes": (graph_auto_linear_attention_nodes, 128),
            "spatial_graph_gate_init": (spatial_graph_gate_init, -2.0),
            "spatial_topology_gate_init": (spatial_topology_gate_init, -2.0),
            "unet_attention_resolutions": (tuple(unet_attention_resolutions), (0, 1)),
        }
        changed_legacy_args = [
            name for name, (value, default) in ignored_legacy_args.items()
            if value != default
        ]
        if changed_legacy_args:
            raise ValueError(
                "DiscreteMaskedRoomModel uses MaskedTokenTransformerBackbone; "
                "the following legacy U-Net controls are not valid masked-room "
                f"ablations: {', '.join(changed_legacy_args)}."
            )
        self.backbone = MaskedTokenTransformerBackbone(
            hidden_dim=self.hidden_dim,
            context_dim=context_dim,
            num_steps=self.default_num_steps,
            num_layers=max(1, int(unet_num_res_blocks) * max(1, len(tuple(unet_channel_mult)))),
            num_heads=int(unet_num_heads),
            dropout=float(unet_dropout),
            room_topology_channels=room_topology_channels,
            context_attention_mode=self.context_attention_mode,
        )
        self.classifier = nn.Conv2d(self.hidden_dim, self.num_classes, kernel_size=1)

        # --- Edge-Aware Logit Bias ---
        # Pre-compute which class IDs are door-like vs wall-like for fast
        # boundary masking at inference and training time.
        _door_ids = [
            int(TileID.DOOR_OPEN),
            int(TileID.DOOR_LOCKED),
            int(TileID.DOOR_BOMB),
            int(TileID.DOOR_PUZZLE),
            int(TileID.DOOR_BOSS),
            int(TileID.DOOR_SOFT),
        ]
        _wall_id = int(TileID.WALL)
        # Boolean mask over class dimension: True for door-class indices.
        _door_class_mask = torch.zeros(self.num_classes, dtype=torch.bool)
        for _d in _door_ids:
            if _d < self.num_classes:
                _door_class_mask[_d] = True
        _wall_class_mask = torch.zeros(self.num_classes, dtype=torch.bool)
        if _wall_id < self.num_classes:
            _wall_class_mask[_wall_id] = True
        _semantic_door_masks = {}
        for _name, _tile in {
            "open": int(TileID.DOOR_OPEN),
            "key": int(TileID.DOOR_LOCKED),
            "bomb": int(TileID.DOOR_BOMB),
            "puzzle": int(TileID.DOOR_PUZZLE),
            "boss": int(TileID.DOOR_BOSS),
            "soft": int(TileID.DOOR_SOFT),
        }.items():
            _mask = torch.zeros(self.num_classes, dtype=torch.bool)
            if _tile < self.num_classes:
                _mask[_tile] = True
            _semantic_door_masks[_name] = _mask
        self.register_buffer('_door_class_mask', _door_class_mask)
        self.register_buffer('_wall_class_mask', _wall_class_mask)
        for _name, _mask in _semantic_door_masks.items():
            self.register_buffer(f'_door_{_name}_class_mask', _mask)

        # Topology channel indices for the four cardinal door channels.
        self._topo_door_ch = {
            'N': int(ROOM_TOPOLOGY_CHANNELS.get('door_n', -1)),
            'S': int(ROOM_TOPOLOGY_CHANNELS.get('door_s', -1)),
            'W': int(ROOM_TOPOLOGY_CHANNELS.get('door_w', -1)),
            'E': int(ROOM_TOPOLOGY_CHANNELS.get('door_e', -1)),
        }
        self._topo_gate_family_ch = {
            family: {
                direction: int(ROOM_TOPOLOGY_CHANNELS.get(f'{family}_{direction.lower()}', -1))
                for direction in ('N', 'S', 'E', 'W')
            }
            for family in (
                'gate_key',
                'gate_boss',
                'gate_bomb',
                'gate_soft',
                'gate_switch',
                'gate_item',
                'gate_secret',
                'gate_hazard',
            )
        }
        self._topo_gate_family_to_door_mask = {
            'gate_key': '_door_key_class_mask',
            'gate_boss': '_door_boss_class_mask',
            'gate_bomb': '_door_bomb_class_mask',
            'gate_soft': '_door_soft_class_mask',
            'gate_switch': '_door_puzzle_class_mask',
            'gate_item': '_door_puzzle_class_mask',
            'gate_secret': '_door_soft_class_mask',
            'gate_hazard': '_door_soft_class_mask',
        }

    def attention_complexity_metrics(self, context_tokens: int) -> Dict[str, float]:
        return self.backbone.attention_complexity_metrics(
            context_tokens=int(context_tokens),
            room_height=ROOM_HEIGHT,
            room_width=ROOM_WIDTH,
            mode=self.context_attention_mode,
        )

    @staticmethod
    def _build_generator(*, device: torch.device, seed: Optional[int]) -> Optional[torch.Generator]:
        if seed is None:
            return None
        generator_device = device.type if device.type == "cuda" else "cpu"
        return torch.Generator(device=generator_device).manual_seed(int(seed))

    @staticmethod
    def _remaining_mask_ratio(progress: float, *, schedule_mode: str) -> float:
        clipped = float(max(0.0, min(1.0, progress)))
        mode = str(schedule_mode or "cosine").strip().lower()
        if mode == "linear":
            return max(0.0, 1.0 - clipped)
        return max(0.0, math.cos(0.5 * math.pi * clipped))

    def _step_from_mask_ratio(self, mask_ratio: Tensor) -> Tensor:
        """Map corruption level to the same reverse-refinement step semantics used at inference."""
        max_step = max(0, int(self.default_num_steps) - 1)
        if max_step == 0:
            return torch.zeros_like(mask_ratio, dtype=torch.long)
        return torch.round(mask_ratio.clamp(0.0, 1.0) * float(max_step)).to(dtype=torch.long).clamp(0, max_step)

    @staticmethod
    def _sample_gumbel(
        shape: Sequence[int],
        *,
        device: torch.device,
        generator: Optional[torch.Generator],
    ) -> Tensor:
        uniform = torch.rand(shape, device=device, generator=generator).clamp_(1e-6, 1.0 - 1e-6)
        return -torch.log(-torch.log(uniform))

    def _apply_edge_aware_logit_bias(
        self,
        logits: Tensor,
        graph_data: Optional[Dict[str, Tensor]],
        *,
        bias_strength: float = 4.0,
        door_threshold: float = 0.5,
    ) -> Tensor:
        """
        Apply topology-conditioned boundary logit bias (edge-aware logits).

        For each room boundary row/column the topology map's door channels are
        consulted:
        - If a door channel is active at a boundary cell → boost all DOOR_*
          class logits and suppress the WALL class.
        - If no door channel is active at a boundary cell → boost the WALL
          class and suppress all DOOR_* class logits.

        This prevents the MaskGIT from generating walls across open doorways or
        open passages through solid boundary walls, which is the most common
        structural violation observed during unconstrained sampling.

        Args:
            logits: Raw classifier logits ``[B, C, H, W]``.
            graph_data: Optional graph conditioning dict.  Must contain
                ``room_topology_map`` under that key.
            bias_strength: Additive logit delta applied to preferred classes.
            door_threshold: Minimum topology-channel activation to treat a
                boundary cell as containing a door.

        Returns:
            Biased logits with the same shape as ``logits``.
        """
        if graph_data is None:
            return logits
        topo = graph_data.get('room_topology_map')
        if not isinstance(topo, torch.Tensor):
            return logits

        B, C, H, W = logits.shape
        topo = topo.to(logits.device)
        if topo.dim() == 3:
            topo = topo.unsqueeze(0)
        if topo.dim() != 4:
            raise ModelContextContractError(
                "room_topology_map must have shape [C,H,W] or [B,C,H,W], "
                f"got {tuple(topo.shape)}."
            )
        if int(topo.shape[0]) == 1 and B > 1:
            topo = topo.expand(B, -1, -1, -1)
        if int(topo.shape[0]) != B or topo.shape[-2:] != (H, W):
            raise ModelContextContractError(
                "room_topology_map/logit contract mismatch: "
                f"topology={tuple(topo.shape)}, logits={tuple(logits.shape)}. "
                "Refusing to drop topology conditioning silently."
            )

        bias = torch.zeros_like(logits)  # [B, C, H, W]
        door_mask = self._door_class_mask.to(logits.device)  # [C]
        wall_mask = self._wall_class_mask.to(logits.device)  # [C]

        # --- boundary slice definitions and their door channels ---
        boundary_specs = [
            ('N', slice(0, 1),        slice(None)),
            ('S', slice(H - 1, H),    slice(None)),
            ('W', slice(None),        slice(0, 1)),
            ('E', slice(None),        slice(W - 1, W)),
        ]

        for direction, row_sl, col_sl in boundary_specs:
            ch_idx = self._topo_door_ch[direction]
            if ch_idx < 0 or ch_idx >= int(topo.shape[1]):
                # Channel not present in this topology map - skip.
                continue
            door_act = topo[:, ch_idx, row_sl, col_sl]

            # door_act: [B, 1, boundary_len] or [B, boundary_len, 1]
            door_present = (door_act >= door_threshold)  # bool [B, h', w']

            # [B, 1, h', w'] -> broadcast over C
            door_present_4d = door_present.unsqueeze(1)   # [B, 1, h', w']

            # Boost door logits where door is present; boost wall otherwise.
            # Accumulate a fresh delta instead of rewriting the slice so corner
            # cells receive both row and column boundary evidence.
            bias_slice = bias[:, :, row_sl, col_sl]
            delta = torch.where(
                door_present_4d,
                torch.where(
                    door_mask.view(1, -1, 1, 1),
                    torch.full_like(bias_slice, bias_strength),
                    torch.where(
                        wall_mask.view(1, -1, 1, 1),
                        torch.full_like(bias_slice, -bias_strength),
                        torch.zeros_like(bias_slice),
                    ),
                ),
                torch.where(
                    wall_mask.view(1, -1, 1, 1),
                    torch.full_like(bias_slice, bias_strength),
                    torch.where(
                        door_mask.view(1, -1, 1, 1),
                        torch.full_like(bias_slice, -bias_strength),
                        torch.zeros_like(bias_slice),
                    ),
                ),
            )
            for family, direction_channels in self._topo_gate_family_ch.items():
                semantic_ch_idx = int(direction_channels.get(direction, -1))
                if semantic_ch_idx < 0 or semantic_ch_idx >= int(topo.shape[1]):
                    continue
                semantic_act = topo[:, semantic_ch_idx, row_sl, col_sl] >= door_threshold
                if not bool(semantic_act.any()):
                    continue
                target_mask_name = self._topo_gate_family_to_door_mask.get(family)
                target_mask = getattr(self, str(target_mask_name), None)
                if not isinstance(target_mask, torch.Tensor):
                    continue
                target_mask = target_mask.to(device=logits.device, dtype=torch.bool).view(1, -1, 1, 1)
                semantic_present_4d = semantic_act.unsqueeze(1)
                semantic_delta = torch.where(
                    semantic_present_4d,
                    torch.where(
                        target_mask,
                        torch.full_like(bias_slice, bias_strength),
                        torch.where(
                            door_mask.view(1, -1, 1, 1),
                            torch.full_like(bias_slice, -0.5 * bias_strength),
                            torch.zeros_like(bias_slice),
                        ),
                    ),
                    torch.zeros_like(bias_slice),
                )
                delta = delta + semantic_delta
            bias[:, :, row_sl, col_sl] = bias_slice + delta

        return logits + bias

    def _apply_fixed_token_logits(
        self,
        logits: Tensor,
        *,
        fixed_tokens: Tensor,
        fixed_mask: Tensor,
    ) -> Tensor:
        if not bool(fixed_mask.any()):
            return logits
        forced = torch.full_like(logits, -1e4)
        forced.scatter_(
            1,
            fixed_tokens.unsqueeze(1).clamp(min=0, max=self.num_classes - 1),
            torch.full_like(fixed_tokens.unsqueeze(1), 1e4, dtype=logits.dtype),
        )
        return torch.where(fixed_mask.unsqueeze(1), forced, logits)

    def _sample_predictions(
        self,
        probs: Tensor,
        *,
        stochastic: bool,
        generator: Optional[torch.Generator],
    ) -> Tuple[Tensor, Tensor]:
        if not bool(stochastic):
            confidence, prediction = probs.max(dim=1)
            return prediction, confidence

        batch_size, num_classes, height, width = probs.shape
        flat = probs.permute(0, 2, 3, 1).reshape(-1, num_classes)
        prediction = torch.multinomial(flat, num_samples=1, replacement=True, generator=generator)
        prediction = prediction.view(batch_size, height, width)
        confidence = probs.gather(1, prediction.unsqueeze(1)).squeeze(1)
        return prediction, confidence

    def _iterative_fill(
        self,
        *,
        tokens: Tensor,
        committed: Tensor,
        context: Tensor,
        graph_data: Optional[Dict[str, Tensor]],
        fixed_tokens: Tensor,
        fixed_mask: Tensor,
        total_steps: int,
        temperature: float,
        schedule_mode: str,
        stochastic: bool,
        generator: Optional[torch.Generator],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, float]]:
        batch_size = int(tokens.shape[0])
        device = tokens.device
        available_counts = ((~fixed_mask) & (~committed)).sum(dim=(1, 2)).clamp_min(1)
        total_editable_counts = (~fixed_mask).sum(dim=(1, 2)).clamp_min(1)

        logits = torch.zeros(batch_size, self.num_classes, ROOM_HEIGHT, ROOM_WIDTH, device=device)
        hidden = self._embed_tokens(tokens)
        steps_executed = 0
        committed_per_step: list[Tensor] = []
        for i in range(int(max(1, total_steps))):
            steps_executed += 1
            unresolved_before_step = (~committed).sum(dim=(1, 2))
            step = self._step_from_mask_ratio(
                unresolved_before_step.to(dtype=torch.float32)
                / total_editable_counts.to(dtype=torch.float32)
            )
            logits, hidden = self.forward(tokens, step, context, graph_data=graph_data, return_hidden=True)
            logits = self._apply_edge_aware_logit_bias(
                logits,
                graph_data,
                bias_strength=float(getattr(self, '_edge_bias_strength', 4.0)),
            )
            logits = self._apply_fixed_token_logits(logits, fixed_tokens=fixed_tokens, fixed_mask=fixed_mask)

            progress = float(i + 1) / float(max(1, total_steps))
            anneal = max(0.25, 1.0 - progress)
            current_temperature = max(1e-6, float(temperature) * anneal) if stochastic else max(1e-6, float(temperature))
            probs = F.softmax((logits / current_temperature).float(), dim=1).to(dtype=logits.dtype)
            prediction, confidence = self._sample_predictions(
                probs,
                stochastic=stochastic,
                generator=generator,
            )
            ranking_scores = confidence
            if bool(stochastic):
                ranking_scores = torch.log(ranking_scores.clamp_min(1e-9)) + (
                    current_temperature
                    * anneal
                    * self._sample_gumbel(
                        ranking_scores.shape,
                        device=device,
                        generator=generator,
                    )
                )

            unresolved = ~committed
            if not bool(unresolved.any()):
                break

            tokens_candidate = tokens.clone()
            tokens_candidate[unresolved] = prediction[unresolved]

            current_unresolved = unresolved.flatten(1).sum(dim=1)
            if i == int(total_steps) - 1:
                chosen_mask = unresolved
            else:
                remaining_ratio = self._remaining_mask_ratio(progress, schedule_mode=schedule_mode)
                target_remaining = torch.floor(
                    available_counts.to(dtype=torch.float32) * float(remaining_ratio)
                ).to(dtype=torch.long)
                keep_counts = (current_unresolved - target_remaining).clamp_min(1)
                keep_counts = torch.minimum(keep_counts, current_unresolved).clamp_min(0)
                flat_scores = ranking_scores.flatten(1).masked_fill(~unresolved.flatten(1), -torch.inf)
                order = torch.argsort(flat_scores, dim=1, descending=True)
                ranks = torch.empty_like(order)
                rank_values = torch.arange(order.shape[1], device=device).expand_as(order)
                ranks.scatter_(1, order, rank_values)
                chosen_mask = (
                    unresolved.flatten(1)
                    & (ranks < keep_counts.unsqueeze(1))
                    & (current_unresolved.unsqueeze(1) > 0)
                ).view_as(unresolved)

            new_commit = committed | chosen_mask
            committed_per_step.append(chosen_mask.flatten(1).sum(dim=1).to(dtype=torch.float32))

            tokens = torch.where(new_commit, tokens_candidate, tokens)
            committed = new_commit
            tokens[fixed_mask] = fixed_tokens[fixed_mask]

        committed_counts = (
            torch.stack(committed_per_step, dim=0).sum(dim=0)
            if committed_per_step
            else torch.zeros(batch_size, device=device, dtype=torch.float32)
        )
        fill_metrics = {
            "steps_executed": float(steps_executed),
            "mean_tokens_committed": float(committed_counts.mean().item()),
            "mean_tokens_committed_per_step": float(
                committed_counts.mean().item() / max(1, steps_executed)
            ),
            "mean_unresolved_tokens": float((~committed).flatten(1).sum(dim=1).float().mean().item()),
        }
        return tokens, logits, hidden, committed, fill_metrics

    @staticmethod
    def _normalize_fixed_layout(
        *,
        fixed_tokens: Optional[Tensor],
        fixed_mask: Optional[Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        if fixed_tokens is None and fixed_mask is None:
            return None, None

        if fixed_tokens is None:
            fixed_tokens = torch.zeros(batch_size, ROOM_HEIGHT, ROOM_WIDTH, device=device, dtype=torch.long)
        else:
            fixed_tokens = fixed_tokens.to(device=device, dtype=torch.long)
        if fixed_mask is None:
            fixed_mask = torch.zeros(batch_size, ROOM_HEIGHT, ROOM_WIDTH, device=device, dtype=torch.bool)
        else:
            fixed_mask = fixed_mask.to(device=device, dtype=torch.bool)

        if fixed_tokens.dim() == 2:
            fixed_tokens = fixed_tokens.unsqueeze(0)
        if fixed_mask.dim() == 2:
            fixed_mask = fixed_mask.unsqueeze(0)

        if int(fixed_tokens.shape[0]) == 1 and batch_size > 1:
            fixed_tokens = fixed_tokens.expand(batch_size, -1, -1).clone()
        if int(fixed_mask.shape[0]) == 1 and batch_size > 1:
            fixed_mask = fixed_mask.expand(batch_size, -1, -1).clone()

        if tuple(fixed_tokens.shape) != (batch_size, ROOM_HEIGHT, ROOM_WIDTH):
            raise ValueError(
                f"fixed_tokens must have shape [B,H,W]=({batch_size},{ROOM_HEIGHT},{ROOM_WIDTH}), "
                f"got {tuple(fixed_tokens.shape)}"
            )
        if tuple(fixed_mask.shape) != (batch_size, ROOM_HEIGHT, ROOM_WIDTH):
            raise ValueError(
                f"fixed_mask must have shape [B,H,W]=({batch_size},{ROOM_HEIGHT},{ROOM_WIDTH}), "
                f"got {tuple(fixed_mask.shape)}"
            )
        return fixed_tokens, fixed_mask

    @staticmethod
    def build_fixed_mask_from_topology_map(
        target_tokens: Tensor,
        room_topology_map: Optional[Tensor],
        *,
        num_classes: int = 44,
        semantic_anchor_threshold: float = 0.5,
    ) -> Tuple[Tensor, Tensor]:
        """Build the same hard semantic anchors used by runtime generation.

        VGLC room grids do not necessarily materialize mission-graph entities
        such as keys. Copying the observed tile at a ``role_key`` anchor can
        therefore freeze FLOOR during training while runtime freezes
        KEY_SMALL. Fixed token identities are derived from topology channels;
        observed targets are retained only for door cells represented in the
        room corpus.
        """
        tokens = target_tokens.long()
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
        B, H, W = tokens.shape
        fixed_tokens = torch.zeros_like(tokens)
        fixed_mask = torch.zeros(B, H, W, device=tokens.device, dtype=torch.bool)
        if room_topology_map is None:
            return fixed_tokens, fixed_mask

        topo = room_topology_map.to(tokens.device)
        if topo.dim() == 3:
            topo = topo.unsqueeze(0)
        if tuple(topo.shape[-2:]) != (H, W):
            raise ValueError(
                f"room_topology_map spatial shape must be {(H, W)}, got {tuple(topo.shape[-2:])}"
            )
        if int(topo.shape[0]) == 1 and B > 1:
            topo = topo.expand(B, -1, -1, -1)
        if int(topo.shape[0]) != B:
            raise ValueError(f"room_topology_map batch size {int(topo.shape[0])} does not match tokens batch {B}")

        threshold = float(semantic_anchor_threshold)

        def channel_mask(name: str) -> Tensor:
            channel = int(ROOM_TOPOLOGY_CHANNELS[name])
            if channel >= int(topo.shape[1]):
                return torch.zeros(B, H, W, device=tokens.device, dtype=torch.bool)
            return topo[:, channel] > threshold

        # Every room has local traversal start/goal anchors. Runtime fixes
        # these as floor unless a semantic START/GOAL role overrides them.
        local_anchors = channel_mask("start") | channel_mask("goal")
        fixed_mask |= local_anchors
        fixed_tokens[local_anchors] = int(SEMANTIC_PALETTE["FLOOR"])

        role_tokens = (
            ("role_start", "START", ("START",)),
            ("role_goal", "TRIFORCE", ("TRIFORCE",)),
            ("role_key", "KEY_SMALL", ("KEY_SMALL", "KEY_BOSS")),
            ("role_item", "KEY_ITEM", ("KEY_ITEM", "ITEM_MINOR", "STAIR")),
            ("role_boss", "BOSS", ("BOSS",)),
            ("role_puzzle", "PUZZLE", ("PUZZLE",)),
        )
        for channel_name, default_tile_name, compatible_tile_names in role_tokens:
            mask = channel_mask(channel_name)
            fixed_mask |= mask
            default_tile = int(SEMANTIC_PALETTE[default_tile_name])
            compatible_ids = torch.tensor(
                [int(SEMANTIC_PALETTE[name]) for name in compatible_tile_names],
                device=tokens.device,
                dtype=tokens.dtype,
            )
            observed_is_compatible = (tokens.unsqueeze(-1) == compatible_ids).any(dim=-1)
            assigned = torch.where(
                observed_is_compatible,
                tokens.clamp(0, num_classes - 1),
                torch.full_like(tokens, default_tile),
            )
            fixed_tokens[mask] = assigned[mask]

        door_mask = torch.zeros(B, H, W, device=tokens.device, dtype=torch.bool)
        for direction in ("n", "s", "e", "w"):
            door_mask |= channel_mask(f"door_{direction}")
        fixed_mask |= door_mask
        fixed_tokens[door_mask] = tokens[door_mask].clamp(0, num_classes - 1)
        return fixed_tokens, fixed_mask

    def _extract_context_topology(
        self,
        context: Tensor,
        graph_data: Optional[Dict[str, Tensor]],
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        if graph_data is None or context.dim() != 3:
            return None, None
        edge_index = graph_data.get("edge_index")
        if not isinstance(edge_index, torch.Tensor):
            return None, None

        batch_size = int(context.shape[0])
        adjusted = edge_index.to(context.device)
        if adjusted.dim() == 2:
            adjusted = adjusted.unsqueeze(0)
            if batch_size > 1:
                adjusted = adjusted.expand(batch_size, -1, -1)
        elif adjusted.dim() == 3 and int(adjusted.shape[0]) == 1 and batch_size > 1:
            adjusted = adjusted.expand(batch_size, -1, -1)
        if adjusted.dim() != 3 or int(adjusted.shape[0]) != batch_size or int(adjusted.shape[1]) != 2:
            raise ValueError(
                f"edge_index shape {tuple(adjusted.shape)} must match [B,2,E] for batch size {batch_size}."
            )
        node_mask = graph_data.get("node_mask")
        if isinstance(node_mask, torch.Tensor):
            node_mask = node_mask.to(context.device)
            if node_mask.dim() == 1:
                node_mask = node_mask.unsqueeze(0)
                if batch_size > 1:
                    node_mask = node_mask.expand(batch_size, -1)
            if node_mask.dim() != 2 or int(node_mask.shape[0]) != batch_size:
                raise ValueError(
                    f"node_mask shape {tuple(node_mask.shape)} must match context batch size {batch_size}."
                )
        else:
            node_mask = None

        has_room_anchor = bool(graph_data.get("has_room_anchor", False))
        if has_room_anchor:
            adjusted = adjusted + 1
            if node_mask is not None:
                anchor = torch.ones(node_mask.shape[0], 1, device=node_mask.device, dtype=node_mask.dtype)
                node_mask = torch.cat([anchor, node_mask], dim=1)

        return adjusted, node_mask

    def _extract_spatial_graph_context(
        self,
        context: Tensor,
        graph_data: Optional[Dict[str, Tensor]],
    ) -> Optional[Dict[str, Tensor]]:
        if not isinstance(graph_data, dict):
            return None
        node_features = graph_data.get("node_features")
        if not isinstance(node_features, torch.Tensor):
            return None

        batch_size = int(context.shape[0])
        node_features = node_features.to(context.device)
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)
            if batch_size > 1:
                node_features = node_features.expand(batch_size, -1, -1)
        if int(node_features.shape[0]) != batch_size:
            raise ValueError(
                f"node_features batch size {int(node_features.shape[0])} does not match context batch size {batch_size}."
            )
        num_nodes = int(node_features.shape[1])
        spatial: Dict[str, Tensor] = {}
        has_room_anchor = bool(graph_data.get("has_room_anchor", False))

        if context.dim() == 3:
            needed = num_nodes + (1 if has_room_anchor else 0)
            if int(context.shape[1]) < needed:
                raise ValueError(
                    f"context sequence length {int(context.shape[1])} is too short for {num_nodes} graph nodes."
                )
            spatial["graph_nodes"] = context[:, 1:1 + num_nodes, :] if has_room_anchor else context[:, :num_nodes, :]

        node_mask = graph_data.get("node_mask")
        if isinstance(node_mask, torch.Tensor):
            node_mask = node_mask.to(context.device)
            if node_mask.dim() == 1:
                node_mask = node_mask.unsqueeze(0)
                if batch_size > 1:
                    node_mask = node_mask.expand(batch_size, -1)
            if tuple(node_mask.shape) != (batch_size, num_nodes):
                raise ValueError(
                    f"node_mask shape {tuple(node_mask.shape)} must match [B,N]=({batch_size},{num_nodes})."
                )
            spatial["node_mask"] = node_mask

        for key in ("edge_index", "tpe", "node_positions", "current_node_distance", "room_topology_map"):
            value = graph_data.get(key)
            if not isinstance(value, torch.Tensor):
                continue
            value = value.to(context.device)
            if key == "edge_index" and value.dim() == 2:
                value = value.unsqueeze(0)
                if batch_size > 1:
                    value = value.expand(batch_size, -1, -1)
            if key == "edge_index" and value.dim() == 3 and int(value.shape[0]) == 1 and batch_size > 1:
                value = value.expand(batch_size, -1, -1)
            if key in {"tpe", "node_positions", "current_node_distance"} and value.dim() == 2:
                value = value.unsqueeze(0)
                if batch_size > 1:
                    value = value.expand(batch_size, -1, -1)
            if key == "room_topology_map" and value.dim() == 3:
                value = value.unsqueeze(0)
                if batch_size > 1:
                    value = value.expand(batch_size, -1, -1, -1)
            if key == "edge_index" and (value.dim() != 3 or int(value.shape[1]) != 2):
                raise ValueError(
                    f"edge_index shape {tuple(value.shape)} must match [B,2,E] for batch size {batch_size}."
                )
            if int(value.shape[0]) != batch_size:
                raise ValueError(
                    f"{key} batch size {int(value.shape[0])} does not match context batch size {batch_size}."
                )
            spatial["node_tpe" if key == "tpe" else key] = value
        return spatial or None

    def _embed_tokens(self, tokens: Tensor) -> Tensor:
        x = self.token_embedding(tokens.long())  # [B,H,W,C]
        x = x.permute(0, 3, 1, 2).contiguous()
        return x + self.position_embedding

    def forward(
        self,
        tokens: Tensor,
        step: Tensor,
        context: Tensor,
        *,
        graph_data: Optional[Dict[str, Tensor]] = None,
        return_hidden: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        if tokens.dim() != 3:
            raise ValueError(f"tokens must be [B,H,W], got {tuple(tokens.shape)}")
        x = self._embed_tokens(tokens)
        hidden = self.backbone(
            x,
            step,
            context,
            graph_data=graph_data,
        )
        logits = self.classifier(hidden)
        if return_hidden:
            return logits, hidden
        return logits

    def training_loss(
        self,
        target_tokens: Tensor,
        context: Tensor,
        *,
        graph_data: Optional[Dict[str, Tensor]] = None,
        fixed_tokens: Optional[Tensor] = None,
        fixed_mask: Optional[Tensor] = None,
        min_mask_ratio: float = 0.0,
        max_mask_ratio: float = 1.0,
        topology_focus_map: Optional[Tensor] = None,
        topology_alignment_weight: float = 0.0,
        return_aux: bool = False,
    ) -> Tuple[Tensor, Dict[str, float]] | Tuple[Tensor, Dict[str, float], Dict[str, Tensor]]:
        target = target_tokens.long()
        if target.dim() == 2:
            target = target.unsqueeze(0)
        batch_size = int(target.shape[0])
        device = target.device
        fixed_tokens, fixed_mask = self._normalize_fixed_layout(
            fixed_tokens=fixed_tokens,
            fixed_mask=fixed_mask,
            batch_size=batch_size,
            device=device,
        )
        if fixed_mask is None:
            fixed_mask = torch.zeros_like(target, dtype=torch.bool)
        available = ~fixed_mask
        min_mask_ratio = float(min_mask_ratio)
        max_mask_ratio = float(max_mask_ratio)
        if not 0.0 <= min_mask_ratio <= 1.0 or not 0.0 <= max_mask_ratio <= 1.0:
            raise ValueError(
                "min_mask_ratio and max_mask_ratio must both be in [0, 1], "
                f"got {min_mask_ratio} and {max_mask_ratio}."
            )
        if min_mask_ratio > max_mask_ratio:
            raise ValueError(
                f"min_mask_ratio must be <= max_mask_ratio, got {min_mask_ratio} > {max_mask_ratio}."
            )

        mask_ratio = torch.empty(batch_size, device=device).uniform_(min_mask_ratio, max_mask_ratio)
        random_mask = torch.rand_like(target.float()) < mask_ratio[:, None, None]
        train_mask = random_mask & available

        # Ensure every sample masks at least one non-fixed token.
        for i in range(batch_size):
            if not bool(train_mask[i].any()):
                candidates = torch.nonzero(available[i], as_tuple=False)
                if int(candidates.shape[0]) > 0:
                    idx = candidates[torch.randint(0, int(candidates.shape[0]), (1,), device=device).item()]
                    train_mask[i, int(idx[0]), int(idx[1])] = True

        masked_tokens = target.clone()
        masked_tokens[train_mask] = int(self.mask_token_id)
        if fixed_mask is not None and fixed_tokens is not None:
            masked_tokens[fixed_mask] = fixed_tokens[fixed_mask]

        step = self._step_from_mask_ratio(mask_ratio)
        logits = self.forward(masked_tokens, step, context, graph_data=graph_data)
        ignore_target = target.masked_fill(~train_mask, -100)
        loss_map = F.cross_entropy(logits, ignore_target, ignore_index=-100, reduction="none")
        denom = train_mask.float().sum().clamp(min=1.0)
        base_loss = F.cross_entropy(logits, ignore_target, ignore_index=-100, reduction="sum") / denom
        topology_focus_loss = torch.zeros((), device=device, dtype=base_loss.dtype)
        topology_focus_fraction = torch.zeros((), device=device, dtype=base_loss.dtype)
        topology_focus_map_t = topology_focus_map
        if topology_focus_map_t is not None:
            topology_focus_map_t = topology_focus_map_t.to(device=device, dtype=loss_map.dtype)
            if topology_focus_map_t.dim() == 2:
                topology_focus_map_t = topology_focus_map_t.unsqueeze(0)
            if tuple(topology_focus_map_t.shape) != (batch_size, ROOM_HEIGHT, ROOM_WIDTH):
                raise ValueError(
                    f"topology_focus_map must have shape [B,H,W]=({batch_size},{ROOM_HEIGHT},{ROOM_WIDTH}), "
                    f"got {tuple(topology_focus_map_t.shape)}"
                )
            masked_focus = topology_focus_map_t * train_mask.float()
            focus_denom = masked_focus.sum()
            if float(focus_denom.item()) > 0.0:
                topology_focus_loss = (loss_map * masked_focus).sum() / focus_denom.clamp(min=1.0)
                topology_focus_fraction = masked_focus.gt(0).float().mean()
        loss = base_loss + float(max(0.0, topology_alignment_weight)) * topology_focus_loss
        metrics = {
            "loss": float(loss.item()),
            "base_loss": float(base_loss.item()),
            "mask_ratio": float(mask_ratio.mean().item()),
            "step_mean": float(step.float().mean().item()),
            "masked_fraction": float(train_mask.float().mean().item()),
            "topology_focus_loss": float(topology_focus_loss.item()),
            "topology_focus_fraction": float(topology_focus_fraction.item()),
        }
        if return_aux:
            return loss, metrics, {
                "logits": logits,
                "train_mask": train_mask,
                "masked_tokens": masked_tokens,
                "step": step,
            }
        return loss, metrics

    @torch.no_grad()
    def sample(
        self,
        *,
        context: Tensor,
        graph_data: Optional[Dict[str, Tensor]] = None,
        fixed_tokens: Optional[Tensor] = None,
        fixed_mask: Optional[Tensor] = None,
        num_steps: Optional[int] = None,
        temperature: float = 1.0,
        schedule_mode: str = "cosine",
        stochastic: bool = True,
        corrector_steps: int = 0,
        corrector_mask_ratio: float = 0.0,
        seed: Optional[int] = None,
        return_sampling_metrics: bool = False,
        edge_bias_strength: float = 4.0,
    ) -> Any:
        batch_size = int(context.shape[0])
        device = context.device
        generator = self._build_generator(device=device, seed=seed)

        steps = int(max(1, num_steps or self.default_num_steps))
        schedule_mode = str(schedule_mode or "cosine").strip().lower()
        if schedule_mode not in {"cosine", "linear"}:
            raise ValueError(f"schedule_mode must be 'cosine' or 'linear', got {schedule_mode!r}.")
        corrector_steps = int(max(0, corrector_steps))
        corrector_mask_ratio = float(max(0.0, min(1.0, corrector_mask_ratio)))
        fixed_tokens, fixed_mask = self._normalize_fixed_layout(
            fixed_tokens=fixed_tokens,
            fixed_mask=fixed_mask,
            batch_size=batch_size,
            device=device,
        )
        if fixed_tokens is None or fixed_mask is None:
            fixed_tokens = torch.zeros(batch_size, ROOM_HEIGHT, ROOM_WIDTH, device=device, dtype=torch.long)
            fixed_mask = torch.zeros(batch_size, ROOM_HEIGHT, ROOM_WIDTH, device=device, dtype=torch.bool)

        tokens = torch.full(
            (batch_size, ROOM_HEIGHT, ROOM_WIDTH),
            fill_value=int(self.mask_token_id),
            device=device,
            dtype=torch.long,
        )
        tokens[fixed_mask] = fixed_tokens[fixed_mask]
        committed = fixed_mask.clone()

        initial_editable = (~fixed_mask).flatten(1).sum(dim=1).to(dtype=torch.float32)
        # Store bias strength for use inside _iterative_fill via self.
        self._edge_bias_strength = float(max(0.0, edge_bias_strength))
        tokens, logits, hidden, committed, initial_fill_metrics = self._iterative_fill(
            tokens=tokens,
            committed=committed,
            context=context,
            graph_data=graph_data,
            fixed_tokens=fixed_tokens,
            fixed_mask=fixed_mask,
            total_steps=steps,
            temperature=float(temperature),
            schedule_mode=schedule_mode,
            stochastic=bool(stochastic),
            generator=generator,
        )

        corrector_rounds_executed = 0
        corrector_fill_steps = 0.0
        corrector_tokens_committed = 0.0
        if corrector_steps > 0 and corrector_mask_ratio > 0.0:
            editable_base = ~fixed_mask
            refinement_steps = max(1, min(3, steps // 2 if steps > 1 else 1))
            for _ in range(corrector_steps):
                logits = self.forward(
                    tokens,
                    torch.zeros(batch_size, device=device, dtype=torch.long),
                    context,
                    graph_data=graph_data,
                )
                logits = self._apply_edge_aware_logit_bias(
                    logits,
                    graph_data,
                    bias_strength=self._edge_bias_strength,
                )
                logits = self._apply_fixed_token_logits(logits, fixed_tokens=fixed_tokens, fixed_mask=fixed_mask)
                current_probs = F.softmax(
                    (logits / max(1e-6, float(temperature))).float(),
                    dim=1,
                ).to(dtype=logits.dtype)
                safe_tokens = tokens.clamp(min=0, max=self.num_classes - 1)
                current_confidence = current_probs.gather(1, safe_tokens.unsqueeze(1)).squeeze(1)
                candidates = committed & editable_base
                candidate_counts = candidates.flatten(1).sum(dim=1)
                remask_counts = torch.ceil(candidate_counts.to(dtype=torch.float32) * float(corrector_mask_ratio)).to(
                    dtype=torch.long
                )
                remask_counts = torch.minimum(remask_counts.clamp_min(1), candidate_counts).clamp_min(0)
                flat_scores = current_confidence.flatten(1).masked_fill(~candidates.flatten(1), torch.inf)
                order = torch.argsort(flat_scores, dim=1, descending=False)
                ranks = torch.empty_like(order)
                rank_values = torch.arange(order.shape[1], device=device).expand_as(order)
                ranks.scatter_(1, order, rank_values)
                remask = (
                    candidates.flatten(1)
                    & (ranks < remask_counts.unsqueeze(1))
                    & (candidate_counts.unsqueeze(1) > 0)
                ).view_as(committed)
                if not bool(remask.any()):
                    break
                corrector_rounds_executed += 1
                tokens = tokens.clone()
                tokens[remask] = int(self.mask_token_id)
                committed = committed.clone()
                committed[remask] = False
                tokens, logits, hidden, committed, correction_metrics = self._iterative_fill(
                    tokens=tokens,
                    committed=committed,
                    context=context,
                    graph_data=graph_data,
                    fixed_tokens=fixed_tokens,
                    fixed_mask=fixed_mask,
                    total_steps=refinement_steps,
                    temperature=float(temperature),
                    schedule_mode=schedule_mode,
                    stochastic=bool(stochastic),
                    generator=generator,
                )
                corrector_fill_steps += float(correction_metrics["steps_executed"])
                corrector_tokens_committed += float(correction_metrics["mean_tokens_committed"])

        sampling_metrics = {
            "masked_refinement_steps_requested": float(steps),
            "masked_refinement_steps_executed": float(initial_fill_metrics["steps_executed"]),
            "masked_corrector_rounds_requested": float(corrector_steps),
            "masked_corrector_rounds_executed": float(corrector_rounds_executed),
            "masked_corrector_refinement_steps_executed": float(corrector_fill_steps),
            "masked_initial_editable_tokens": float(initial_editable.mean().item()),
            "masked_initial_tokens_committed": float(initial_fill_metrics["mean_tokens_committed"]),
            "masked_corrector_tokens_committed": float(corrector_tokens_committed),
            "masked_mean_tokens_committed_per_step": float(
                initial_fill_metrics["mean_tokens_committed_per_step"]
            ),
            "masked_final_unresolved_tokens": float((~committed).flatten(1).sum(dim=1).float().mean().item()),
            "masked_schedule_is_cosine": float(schedule_mode == "cosine"),
            "masked_sampling_is_stochastic": float(bool(stochastic)),
        }
        if return_sampling_metrics:
            return tokens, logits, hidden, sampling_metrics
        return tokens, logits, hidden


def create_discrete_masked_model(
    *,
    num_classes: int = 44,
    hidden_dim: int = 48,
    model_channels: int = 64,
    context_dim: int = 256,
    num_steps: int = 8,
    attention_mode: str = "softmax",
    topology_conditioning_mode: str = "additive",
    hedgehog_feature_dim: int = 32,
    graph_auto_linear_attention_nodes: int = 128,
    spatial_graph_gate_init: float = -2.0,
    spatial_topology_gate_init: float = -2.0,
    unet_channel_mult: Sequence[int] = (1, 2),
    unet_num_res_blocks: int = 1,
    unet_attention_resolutions: Sequence[int] = (0, 1),
    unet_num_heads: int = 4,
    unet_dropout: float = 0.1,
    room_topology_channels: int = ROOM_TOPOLOGY_CHANNEL_COUNT,
    context_attention_mode: str = "concat_encoder",
) -> DiscreteMaskedRoomModel:
    """Factory for the graph-conditioned discrete masked room model."""
    return DiscreteMaskedRoomModel(
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        model_channels=model_channels,
        context_dim=context_dim,
        num_steps=num_steps,
        attention_mode=attention_mode,
        topology_conditioning_mode=topology_conditioning_mode,
        hedgehog_feature_dim=hedgehog_feature_dim,
        graph_auto_linear_attention_nodes=graph_auto_linear_attention_nodes,
        spatial_graph_gate_init=spatial_graph_gate_init,
        spatial_topology_gate_init=spatial_topology_gate_init,
        unet_channel_mult=unet_channel_mult,
        unet_num_res_blocks=unet_num_res_blocks,
        unet_attention_resolutions=unet_attention_resolutions,
        unet_num_heads=unet_num_heads,
        unet_dropout=unet_dropout,
        room_topology_channels=room_topology_channels,
        context_attention_mode=context_attention_mode,
    )


__all__ = [
    "DiscreteMaskedRoomModel",
    "create_discrete_masked_model",
]
