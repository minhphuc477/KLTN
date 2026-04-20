"""
Auxiliary learned supervision for ordered puzzle-stage semantics.

This module turns the repo's shared `puzzle_stage_condition` metadata into
trainable targets and provides a lightweight CNN head over room tile logits.
The head is intentionally small: it is a structural regularizer for the
generator, not a separate large model.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.pipeline.room_topology_conditioning import (
    PUZZLE_STAGE_GATE_FAMILY_IDS,
    PUZZLE_STAGE_KIND_IDS,
)

DEFAULT_PUZZLE_STAGE_MAX_SEQUENCE_LENGTH = 6
DEFAULT_PUZZLE_STAGE_SEMANTICS_HIDDEN_DIM = 96

_PAD_STAGE_KIND = "__pad__"
_PAD_STAGE_KIND_ID = len(PUZZLE_STAGE_KIND_IDS)


def _normalize_stage_kind(kind: Any) -> int:
    key = str(kind or "").strip().lower()
    return int(PUZZLE_STAGE_KIND_IDS.get(key, _PAD_STAGE_KIND_ID))


def _normalize_gate_family(gate_family: Any) -> int:
    key = str(gate_family or "generic").strip().lower()
    return int(PUZZLE_STAGE_GATE_FAMILY_IDS.get(key, PUZZLE_STAGE_GATE_FAMILY_IDS["generic"]))


def build_puzzle_stage_semantic_targets(
    puzzle_stage_conditions: Optional[Iterable[Optional[Mapping[str, Any]]]],
    *,
    max_sequence_length: int = DEFAULT_PUZZLE_STAGE_MAX_SEQUENCE_LENGTH,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Convert `puzzle_stage_condition` payloads into dense tensor targets.
    """
    rows: List[Mapping[str, Any]] = [
        cond if isinstance(cond, Mapping) else {}
        for cond in (puzzle_stage_conditions or [])
    ]
    batch_size = len(rows)
    max_sequence_length = int(max(1, max_sequence_length))

    gate_targets = torch.zeros(batch_size, dtype=torch.long, device=device)
    sequence_required = torch.zeros(batch_size, dtype=torch.float32, device=device)
    stage_count_targets = torch.zeros(batch_size, dtype=torch.long, device=device)
    slot_targets = torch.full(
        (batch_size, max_sequence_length),
        fill_value=int(_PAD_STAGE_KIND_ID),
        dtype=torch.long,
        device=device,
    )
    slot_mask = torch.zeros(batch_size, max_sequence_length, dtype=torch.float32, device=device)

    for batch_index, payload in enumerate(rows):
        gate_targets[batch_index] = int(_normalize_gate_family(payload.get("gate_family", "generic")))
        sequence_required[batch_index] = 1.0 if bool(payload.get("sequence_required", False)) else 0.0

        raw_sequence = payload.get("stage_sequence")
        sequence_items = list(raw_sequence) if isinstance(raw_sequence, (list, tuple)) else []
        stage_count = min(len(sequence_items), max_sequence_length)
        stage_count_targets[batch_index] = int(stage_count)

        for slot_index, stage in enumerate(sequence_items[:max_sequence_length]):
            if isinstance(stage, Mapping):
                slot_targets[batch_index, slot_index] = int(_normalize_stage_kind(stage.get("kind")))
                slot_mask[batch_index, slot_index] = 1.0

    return {
        "gate_family": gate_targets,
        "sequence_required": sequence_required,
        "stage_count": stage_count_targets,
        "stage_slots": slot_targets,
        "stage_slot_mask": slot_mask,
    }


class PuzzleStageSemanticsHead(nn.Module):
    """
    Lightweight CNN head that predicts ordered puzzle semantics from room logits.
    """

    def __init__(
        self,
        *,
        num_tile_classes: int,
        hidden_dim: int = DEFAULT_PUZZLE_STAGE_SEMANTICS_HIDDEN_DIM,
        max_sequence_length: int = DEFAULT_PUZZLE_STAGE_MAX_SEQUENCE_LENGTH,
    ) -> None:
        super().__init__()
        self.num_tile_classes = int(max(2, num_tile_classes))
        self.hidden_dim = int(max(16, hidden_dim))
        self.max_sequence_length = int(max(1, max_sequence_length))
        self.num_gate_families = int(len(PUZZLE_STAGE_GATE_FAMILY_IDS))
        self.num_stage_kinds = int(len(PUZZLE_STAGE_KIND_IDS) + 1)  # + pad

        self.backbone = nn.Sequential(
            nn.Conv2d(self.num_tile_classes, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.gate_head = nn.Linear(self.hidden_dim, self.num_gate_families)
        self.sequence_head = nn.Linear(self.hidden_dim, 1)
        self.count_head = nn.Linear(self.hidden_dim, self.max_sequence_length + 1)
        self.slot_head = nn.Linear(self.hidden_dim, self.max_sequence_length * self.num_stage_kinds)

    def forward(self, tile_logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not isinstance(tile_logits, torch.Tensor) or tile_logits.dim() != 4:
            raise ValueError(f"tile_logits must be [B,C,H,W], got {type(tile_logits).__name__}")
        if int(tile_logits.shape[1]) != self.num_tile_classes:
            raise ValueError(
                f"tile_logits channel mismatch: expected {self.num_tile_classes}, got {int(tile_logits.shape[1])}."
            )
        features = self.proj(self.backbone(tile_logits))
        slot_logits = self.slot_head(features).view(
            int(tile_logits.shape[0]),
            self.max_sequence_length,
            self.num_stage_kinds,
        )
        return {
            "gate_logits": self.gate_head(features),
            "sequence_logits": self.sequence_head(features).squeeze(-1),
            "count_logits": self.count_head(features),
            "slot_logits": slot_logits,
        }

    def compute_loss(
        self,
        tile_logits: torch.Tensor,
        puzzle_stage_conditions: Optional[Iterable[Optional[Mapping[str, Any]]]],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        outputs = self.forward(tile_logits)
        targets = build_puzzle_stage_semantic_targets(
            puzzle_stage_conditions,
            max_sequence_length=self.max_sequence_length,
            device=tile_logits.device,
        )

        gate_loss = F.cross_entropy(outputs["gate_logits"], targets["gate_family"])
        sequence_loss = F.binary_cross_entropy_with_logits(
            outputs["sequence_logits"],
            targets["sequence_required"],
        )
        count_loss = F.cross_entropy(outputs["count_logits"], targets["stage_count"])

        flat_slot_logits = outputs["slot_logits"].reshape(-1, self.num_stage_kinds)
        flat_slot_targets = targets["stage_slots"].reshape(-1)
        flat_slot_mask = targets["stage_slot_mask"].reshape(-1)
        slot_loss_map = F.cross_entropy(flat_slot_logits, flat_slot_targets, reduction="none")
        slot_denom = flat_slot_mask.sum().clamp(min=1.0)
        slot_loss = (slot_loss_map * flat_slot_mask).sum() / slot_denom

        total_loss = gate_loss + sequence_loss + count_loss + slot_loss

        gate_acc = (outputs["gate_logits"].argmax(dim=-1) == targets["gate_family"]).float().mean()
        sequence_pred = (torch.sigmoid(outputs["sequence_logits"]) >= 0.5).float()
        sequence_acc = (sequence_pred == targets["sequence_required"]).float().mean()
        count_acc = (outputs["count_logits"].argmax(dim=-1) == targets["stage_count"]).float().mean()
        slot_pred = outputs["slot_logits"].argmax(dim=-1)
        valid_slot_mask = targets["stage_slot_mask"] > 0
        if bool(valid_slot_mask.any()):
            slot_acc = (slot_pred[valid_slot_mask] == targets["stage_slots"][valid_slot_mask]).float().mean()
        else:
            slot_acc = torch.ones((), device=tile_logits.device, dtype=tile_logits.dtype)

        metrics = {
            "puzzle_stage_semantic_loss": float(total_loss.detach().item()),
            "puzzle_stage_gate_loss": float(gate_loss.detach().item()),
            "puzzle_stage_sequence_loss": float(sequence_loss.detach().item()),
            "puzzle_stage_count_loss": float(count_loss.detach().item()),
            "puzzle_stage_slot_loss": float(slot_loss.detach().item()),
            "puzzle_stage_gate_acc": float(gate_acc.detach().item()),
            "puzzle_stage_sequence_acc": float(sequence_acc.detach().item()),
            "puzzle_stage_count_acc": float(count_acc.detach().item()),
            "puzzle_stage_slot_acc": float(slot_acc.detach().item()),
        }
        return total_loss, metrics
