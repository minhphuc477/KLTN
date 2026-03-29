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

from src.core.definitions import ROOM_HEIGHT, ROOM_WIDTH
from src.core.latent_diffusion import UNetDenoiser
from src.pipeline.room_topology_conditioning import ROOM_TOPOLOGY_CHANNEL_COUNT

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
        hidden_dim: int = 64,
        model_channels: int = 128,
        context_dim: int = 256,
        num_steps: int = 8,
        attention_mode: str = "softmax",
        hedgehog_feature_dim: int = 32,
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
            attention_mode=attention_mode,
            hedgehog_feature_dim=hedgehog_feature_dim,
            topology_map_channels=room_topology_channels,
        )
        self.classifier = nn.Conv2d(self.hidden_dim, self.num_classes, kernel_size=1)

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

        keep = (
            (topo[:, 1] > 0.5)   # start
            | (topo[:, 2] > 0.5) # goal
            | (topo[:, 3] > 0.5) # door_n
            | (topo[:, 4] > 0.5) # door_s
            | (topo[:, 5] > 0.5) # door_e
            | (topo[:, 6] > 0.5) # door_w
        )
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

        for key in ("edge_index", "tpe", "node_positions", "room_topology_map"):
            value = graph_data.get(key)
            if not isinstance(value, torch.Tensor):
                continue
            value = value.to(context.device)
            if key in {"tpe", "node_positions"} and value.dim() == 2:
                value = value.unsqueeze(0)
                if batch_size > 1:
                    value = value.expand(batch_size, -1, -1)
            if key == "room_topology_map" and value.dim() == 3:
                value = value.unsqueeze(0)
                if batch_size > 1:
                    value = value.expand(batch_size, -1, -1, -1)
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

        mask_ratio = torch.empty(batch_size, device=device).uniform_(float(min_mask_ratio), float(max_mask_ratio))
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
        loss = (loss_map * train_mask.float()).sum() / denom
        metrics = {
            "loss": float(loss.item()),
            "mask_ratio": float(mask_ratio.mean().item()),
            "masked_fraction": float(train_mask.float().mean().item()),
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
        seed: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size = int(context.shape[0])
        device = context.device
        if seed is not None:
            torch.manual_seed(int(seed))

        steps = int(max(1, num_steps or self.default_num_steps))
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

        logits = torch.zeros(batch_size, self.num_classes, ROOM_HEIGHT, ROOM_WIDTH, device=device)
        hidden = self._embed_tokens(tokens)
        for i in range(steps):
            step = torch.full((batch_size,), fill_value=max(0, steps - 1 - i), device=device, dtype=torch.long)
            logits, hidden = self.forward(tokens, step, context, graph_data=graph_data, return_hidden=True)
            if bool(fixed_mask.any()):
                logits = logits.clone()
                logits = logits.masked_fill(fixed_mask.unsqueeze(1), -1e4)
                logits.scatter_(
                    1,
                    fixed_tokens.unsqueeze(1).clamp(min=0, max=self.num_classes - 1),
                    torch.where(
                        fixed_mask.unsqueeze(1),
                        torch.full_like(fixed_tokens.unsqueeze(1), 1e4, dtype=logits.dtype),
                        torch.zeros_like(fixed_tokens.unsqueeze(1), dtype=logits.dtype),
                    ),
                )

            probs = F.softmax(logits / max(float(temperature), 1e-6), dim=1)
            conf, pred = probs.max(dim=1)
            unresolved = ~committed
            if not bool(unresolved.any()):
                break

            remaining_fraction = 1.0 - float(i + 1) / float(steps)
            target_remaining = math.floor(float(unresolved[0].numel()) * max(0.0, remaining_fraction))
            keep_now = max(1, int(unresolved.sum(dim=(1, 2)).max().item()) - target_remaining)

            tokens_candidate = tokens.clone()
            tokens_candidate[unresolved] = pred[unresolved]

            new_commit = committed.clone()
            for b in range(batch_size):
                unresolved_idx = torch.nonzero(unresolved[b], as_tuple=False)
                if int(unresolved_idx.shape[0]) == 0:
                    continue
                if i == steps - 1:
                    chosen = unresolved_idx
                else:
                    keep_b = min(keep_now, int(unresolved_idx.shape[0]))
                    scores = conf[b][unresolved[b]]
                    top_idx = torch.topk(scores, k=max(1, keep_b), largest=True).indices
                    chosen = unresolved_idx[top_idx]
                new_commit[b, chosen[:, 0], chosen[:, 1]] = True

            tokens = torch.where(new_commit, tokens_candidate, tokens)
            committed = new_commit
            tokens[fixed_mask] = fixed_tokens[fixed_mask]

        return tokens, logits, hidden


def create_discrete_masked_model(
    *,
    num_classes: int = 44,
    hidden_dim: int = 64,
    model_channels: int = 128,
    context_dim: int = 256,
    num_steps: int = 8,
    attention_mode: str = "softmax",
    hedgehog_feature_dim: int = 32,
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
        hedgehog_feature_dim=hedgehog_feature_dim,
        room_topology_channels=room_topology_channels,
    )


__all__ = [
    "DiscreteMaskedRoomModel",
    "create_discrete_masked_model",
]
