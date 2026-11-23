"""Base layout and spatial helpers for mission graph production rules."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from src.generation.grammar.graph_types import (
    MissionGraph,
)

logger = logging.getLogger(__name__)

__all__ = ["_layout_bounds", "_bounded_free_position"]


def _layout_bounds(context: Dict[str, Any]) -> Tuple[int, int, int, int]:
    """Return inclusive row/col bounds for grammar placement."""
    raw = context.get("layout_bounds")
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        return tuple(int(v) for v in raw)  # type: ignore[return-value]
    return (
        int(context.get("min_layout_row", -32)),
        int(context.get("max_layout_row", 32)),
        int(context.get("min_layout_col", -32)),
        int(context.get("max_layout_col", 32)),
    )


def _bounded_free_position(
    graph: MissionGraph,
    anchor_pos: Tuple[int, ...],
    offsets: List[Tuple[int, int]],
    rng: Any,
    context: Dict[str, Any],
) -> Tuple[int, int, int]:
    """Choose a relative, in-bounds, unoccupied position near an anchor."""
    floor = int(anchor_pos[2]) if len(anchor_pos) > 2 else 0
    min_r, max_r, min_c, max_c = _layout_bounds(context)
    occupied = {
        (int(node.position[0]), int(node.position[1]), int(node.position[2]) if len(node.position) > 2 else 0)
        for node in graph.nodes.values()
    }
    shuffled = list(offsets)
    rng.shuffle(shuffled)
    for dr, dc in shuffled:
        row = max(min_r, min(max_r, int(anchor_pos[0]) + int(dr)))
        col = max(min_c, min(max_c, int(anchor_pos[1]) + int(dc)))
        candidate = (row, col, floor)
        if candidate not in occupied:
            return candidate

    # Deterministic expanding fallback: still relative to the anchor and bounded.
    for radius in range(1, 8):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                row = max(min_r, min(max_r, int(anchor_pos[0]) + dr))
                col = max(min_c, min(max_c, int(anchor_pos[1]) + dc))
                candidate = (row, col, floor)
                if candidate not in occupied:
                    return candidate
    return (
        max(min_r, min(max_r, int(anchor_pos[0]))),
        max(min_c, min(max_c, int(anchor_pos[1]))),
        floor,
    )
