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
)
from src.core.latent_diffusion import UNetDenoiser

logger = logging.getLogger(__name__)


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
        mask_token_id: Optional[int] = None,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.context_dim = int(context_dim)
        self.default_num_steps = int(max(1, num_steps))
        self.mask_token_id = int(self.num_classes if mask_token_id is None else mask_token_id)
        self.vocab_size = int(max(self.mask_token_id + 1, self.num_classes + 1))

        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.hidden_dim, ROOM_HEIGHT, ROOM_WIDTH)
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

        self.denoiser = UNetDenoiser(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            model_channels=model_channels,
            context_dim=context_dim,
            channel_mult=tuple(int(v) for v in unet_channel_mult),
            num_res_blocks=int(unet_num_res_blocks),
            attention_resolutions=tuple(int(v) for v in unet_attention_resolutions),
            num_heads=int(unet_num_heads),
            dropout=float(unet_dropout),
            attention_mode=attention_mode,
            hedgehog_feature_dim=hedgehog_feature_dim,
            topology_map_channels=room_topology_channels,
            topology_conditioning_mode=topology_conditioning_mode,
            auto_linear_attention_nodes=int(graph_auto_linear_attention_nodes),
            graph_gate_init=float(spatial_graph_gate_init),
            topology_gate_init=float(spatial_topology_gate_init),
        )
        self.classifier = nn.Conv2d(self.hidden_dim, self.num_classes, kernel_size=1)

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

    @staticmethod
    def _sample_gumbel(
        shape: Sequence[int],
        *,
        device: torch.device,
        generator: Optional[torch.Generator],
    ) -> Tensor:
        uniform = torch.rand(shape, device=device, generator=generator).clamp_(1e-6, 1.0 - 1e-6)
        return -torch.log(-torch.log(uniform))

    def _apply_fixed_token_logits(
        self,
        logits: Tensor,
        *,
        fixed_tokens: Tensor,
        fixed_mask: Tensor,
    ) -> Tensor:
        if not bool(fixed_mask.any()):
            return logits
        constrained = logits.clone()
        constrained = constrained.masked_fill(fixed_mask.unsqueeze(1), -1e4)
        constrained.scatter_(
            1,
            fixed_tokens.unsqueeze(1).clamp(min=0, max=self.num_classes - 1),
            torch.where(
                fixed_mask.unsqueeze(1),
                torch.full_like(fixed_tokens.unsqueeze(1), 1e4, dtype=constrained.dtype),
                torch.zeros_like(fixed_tokens.unsqueeze(1), dtype=constrained.dtype),
            ),
        )
        return constrained

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
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_size = int(tokens.shape[0])
        device = tokens.device
        available_counts = (~fixed_mask).sum(dim=(1, 2))

        logits = torch.zeros(batch_size, self.num_classes, ROOM_HEIGHT, ROOM_WIDTH, device=device)
        hidden = self._embed_tokens(tokens)
        for i in range(int(max(1, total_steps))):
            step = torch.full(
                (batch_size,),
                fill_value=max(0, int(total_steps) - 1 - i),
                device=device,
                dtype=torch.long,
            )
            logits, hidden = self.forward(tokens, step, context, graph_data=graph_data, return_hidden=True)
            logits = self._apply_fixed_token_logits(logits, fixed_tokens=fixed_tokens, fixed_mask=fixed_mask)

            progress = float(i + 1) / float(max(1, total_steps))
            anneal = max(0.25, 1.0 - progress)
            current_temperature = max(1e-6, float(temperature) * anneal) if stochastic else max(1e-6, float(temperature))
            probs = F.softmax(logits / current_temperature, dim=1)
            prediction, confidence = self._sample_predictions(
                probs,
                stochastic=stochastic,
                generator=generator,
            )
            ranking_scores = confidence
            if bool(stochastic):
                ranking_scores = ranking_scores + (
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

            new_commit = committed.clone()
            for b in range(batch_size):
                unresolved_idx = torch.nonzero(unresolved[b], as_tuple=False)
                if int(unresolved_idx.shape[0]) == 0:
                    continue
                if i == int(total_steps) - 1:
                    chosen = unresolved_idx
                else:
                    remaining_ratio = self._remaining_mask_ratio(progress, schedule_mode=schedule_mode)
                    target_remaining = int(math.floor(int(available_counts[b].item()) * remaining_ratio))
                    current_unresolved = int(unresolved_idx.shape[0])
                    keep_b = max(1, min(current_unresolved, current_unresolved - target_remaining))
                    scores = ranking_scores[b][unresolved[b]]
                    top_idx = torch.topk(scores, k=keep_b, largest=True).indices
                    chosen = unresolved_idx[top_idx]
                new_commit[b, chosen[:, 0], chosen[:, 1]] = True

            tokens = torch.where(new_commit, tokens_candidate, tokens)
            committed = new_commit
            tokens[fixed_mask] = fixed_tokens[fixed_mask]

        return tokens, logits, hidden, committed

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
        """
        Extract hard-known tokens from a topology map.

        For training we keep doors and explicit start/goal hints fixed so the
        model learns to paint the rest of the room around those facts.
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

        semantic_anchor_channels = (
            ROOM_TOPOLOGY_CHANNELS["start"],
            ROOM_TOPOLOGY_CHANNELS["goal"],
            ROOM_TOPOLOGY_CHANNELS["door_n"],
            ROOM_TOPOLOGY_CHANNELS["door_s"],
            ROOM_TOPOLOGY_CHANNELS["door_e"],
            ROOM_TOPOLOGY_CHANNELS["door_w"],
            ROOM_TOPOLOGY_CHANNELS["role_start"],
            ROOM_TOPOLOGY_CHANNELS["role_goal"],
            ROOM_TOPOLOGY_CHANNELS["role_enemy"],
            ROOM_TOPOLOGY_CHANNELS["role_key"],
            ROOM_TOPOLOGY_CHANNELS["role_item"],
            ROOM_TOPOLOGY_CHANNELS["role_boss"],
            ROOM_TOPOLOGY_CHANNELS["role_puzzle"],
        )
        keep = torch.zeros(B, H, W, device=tokens.device, dtype=torch.bool)
        threshold = float(semantic_anchor_threshold)
        for channel in semantic_anchor_channels:
            if int(channel) >= int(topo.shape[1]):
                continue
            keep |= topo[:, int(channel)] > threshold
        fixed_mask |= keep
        fixed_tokens[fixed_mask] = tokens[fixed_mask].clamp(0, num_classes - 1)
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
        context_edge_index, context_node_mask = self._extract_context_topology(context, graph_data)
        spatial_graph_data = self._extract_spatial_graph_context(context, graph_data)
        hidden = self.denoiser(
            x,
            step,
            context,
            context_edge_index=context_edge_index,
            context_node_mask=context_node_mask,
            spatial_graph_data=spatial_graph_data,
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
        min_mask_ratio: float = 0.15,
        max_mask_ratio: float = 0.90,
        topology_focus_map: Optional[Tensor] = None,
        topology_alignment_weight: float = 0.0,
    ) -> Tuple[Tensor, Dict[str, float]]:
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

        step = torch.round(mask_ratio * float(self.default_num_steps - 1)).long().clamp(min=0)
        logits = self.forward(masked_tokens, step, context, graph_data=graph_data)
        loss_map = F.cross_entropy(logits, target, reduction="none")
        denom = train_mask.float().sum().clamp(min=1.0)
        base_loss = (loss_map * train_mask.float()).sum() / denom
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
            "masked_fraction": float(train_mask.float().mean().item()),
            "topology_focus_loss": float(topology_focus_loss.item()),
            "topology_focus_fraction": float(topology_focus_fraction.item()),
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
    ) -> Tuple[Tensor, Tensor, Tensor]:
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

        tokens, logits, hidden, committed = self._iterative_fill(
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
                logits = self._apply_fixed_token_logits(logits, fixed_tokens=fixed_tokens, fixed_mask=fixed_mask)
                current_probs = F.softmax(logits / max(1e-6, float(temperature)), dim=1)
                safe_tokens = tokens.clamp(min=0, max=self.num_classes - 1)
                current_confidence = current_probs.gather(1, safe_tokens.unsqueeze(1)).squeeze(1)
                remask = torch.zeros_like(committed)
                for b in range(batch_size):
                    candidates = torch.nonzero(committed[b] & editable_base[b], as_tuple=False)
                    if int(candidates.shape[0]) == 0:
                        continue
                    remask_count = int(math.ceil(int(candidates.shape[0]) * corrector_mask_ratio))
                    remask_count = max(1, min(remask_count, int(candidates.shape[0])))
                    scores = current_confidence[b][committed[b] & editable_base[b]]
                    worst_idx = torch.topk(scores, k=remask_count, largest=False).indices
                    chosen = candidates[worst_idx]
                    remask[b, chosen[:, 0], chosen[:, 1]] = True
                if not bool(remask.any()):
                    break
                tokens = tokens.clone()
                tokens[remask] = int(self.mask_token_id)
                committed = committed.clone()
                committed[remask] = False
                tokens, logits, hidden, committed = self._iterative_fill(
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
    )


__all__ = [
    "DiscreteMaskedRoomModel",
    "create_discrete_masked_model",
]
