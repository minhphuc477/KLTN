"""Shared graph/spatial parsing helpers for dungeon pipeline orchestration."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Set

import networkx as nx
import numpy as np

from src.core.definitions import (
    ROOM_HEIGHT,
    ROOM_WIDTH,
    ROOM_SHAPE,
    ROOM_TRANSPOSED_SHAPE,
    SEMANTIC_PALETTE,
    parse_node_label_tokens,
)


def parse_label_tokens(label: Any) -> Set[str]:
    """Split node labels like 'e,k' into normalized tokens."""
    if label is None:
        return set()
    return set(str(t).strip().lower() for t in parse_node_label_tokens(str(label)) if str(t).strip())


def normalize_node_id(value: Any) -> Optional[Any]:
    """Normalize externally supplied node IDs while preserving heterogeneous hashable IDs."""
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    try:
        hash(value)
    except TypeError:
        return None
    return value


def stable_node_sort_key(node: Any) -> Tuple[int, Any]:
    """Deterministic sort key that remains valid for mixed node-ID types."""
    normalized = normalize_node_id(node)
    if isinstance(normalized, (int, np.integer)):
        return (0, int(normalized))
    if isinstance(normalized, str):
        return (1, normalized)
    return (2, str(normalized))


def canonical_node_order(graph: nx.Graph) -> List[Any]:
    """Return the checkpoint-compatible node order used by graph encoders."""
    if graph is None:
        return []
    return sorted(graph.nodes(), key=stable_node_sort_key)


def coerce_bool(value: Any) -> bool:
    """Robust bool parser for graph attributes."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return value != 0
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y", "on"}:
            return True
        if v in {"false", "0", "no", "n", "off", ""}:
            return False
    return bool(value)


def coerce_difficulty(value: Any) -> float:
    """Convert numeric/string difficulty values into [0, 1]."""
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(max(0.0, min(1.0, float(value))))
    if isinstance(value, str):
        key = value.strip().upper()
        mapping = {
            "SAFE": 0.2,
            "EASY": 0.3,
            "MODERATE": 0.5,
            "MEDIUM": 0.5,
            "HARD": 0.8,
            "EXTREME": 1.0,
        }
        return mapping.get(key, 0.5)
    return 0.5


def parse_room_coord(value: Any) -> Optional[Tuple[int, int]]:
    """Parse room-local coordinates from tuple/list/dict/string."""
    if value is None:
        return None
    if isinstance(value, dict):
        row = value.get("row", value.get("r"))
        col = value.get("col", value.get("c"))
        if isinstance(row, (int, np.integer, float)) and isinstance(col, (int, np.integer, float)):
            return int(row), int(col)
        return None
    if isinstance(value, (tuple, list, np.ndarray)) and len(value) >= 2:
        row, col = value[0], value[1]
        if isinstance(row, (int, np.integer, float)) and isinstance(col, (int, np.integer, float)):
            return int(row), int(col)
        return None
    if isinstance(value, str):
        parts = value.replace("(", "").replace(")", "").split(",")
        if len(parts) >= 2:
            try:
                return int(float(parts[0].strip())), int(float(parts[1].strip()))
            except ValueError:
                return None
    return None


def clamp_room_coord(coord: Tuple[int, int]) -> Tuple[int, int]:
    """Clamp local coordinates into room bounds."""
    r, c = coord
    r = max(0, min(ROOM_HEIGHT - 1, int(r)))
    c = max(0, min(ROOM_WIDTH - 1, int(c)))
    return (r, c)


def get_node_grid_position(graph: nx.Graph, node_id: Any) -> Optional[Tuple[int, int]]:
    """Extract room-grid position for a node from graph metadata."""
    if node_id not in graph:
        return None
    attrs = graph.nodes[node_id]
    for key in ("position", "pos", "grid_pos", "coord", "coords"):
        pos = parse_room_coord(attrs.get(key))
        if pos is not None:
            return pos
    return None


def infer_direction(graph: nx.Graph, source_node: Any, target_node: Any) -> Optional[str]:
    """Infer cardinal direction of source_node relative to target_node."""
    source_pos = get_node_grid_position(graph, source_node)
    target_pos = get_node_grid_position(graph, target_node)
    if source_pos is None or target_pos is None:
        return None

    dr = source_pos[0] - target_pos[0]
    dc = source_pos[1] - target_pos[1]
    if abs(dr) + abs(dc) != 1:
        return None
    if dr == -1:
        return "N"
    if dr == 1:
        return "S"
    if dc == -1:
        return "W"
    if dc == 1:
        return "E"
    return None


def first_free_position(start_pos: Tuple[int, int], occupied: set) -> Tuple[int, int]:
    """Resolve position collisions by scanning downward in the same column."""
    row, col = start_pos
    while (row, col) in occupied:
        row += 1
    return (row, col)


def fit_room_grid(room_grid: np.ndarray) -> np.ndarray:
    """Ensure room grid has exact ROOM_HEIGHT x ROOM_WIDTH shape."""
    if room_grid.shape == ROOM_SHAPE:
        return room_grid.astype(np.int32, copy=False)
    if room_grid.shape == ROOM_TRANSPOSED_SHAPE:
        return room_grid.transpose().astype(np.int32, copy=False)

    void_id = int(SEMANTIC_PALETTE.get("VOID", 0))
    fitted = np.full((ROOM_HEIGHT, ROOM_WIDTH), void_id, dtype=np.int32)
    h = min(ROOM_HEIGHT, room_grid.shape[0])
    w = min(ROOM_WIDTH, room_grid.shape[1])
    fitted[:h, :w] = room_grid[:h, :w].astype(np.int32, copy=False)
    return fitted


def carve_room_connection(
    global_grid: np.ndarray,
    src_pos: Tuple[int, int],
    dst_pos: Tuple[int, int],
    edge_data: Optional[Dict[str, Any]] = None,
    has_reverse_edge: bool = False,
) -> None:
    """Compatibility wrapper around the shared bbox-based connection carver."""
    dr = dst_pos[0] - src_pos[0]
    dc = dst_pos[1] - src_pos[1]
    if abs(dr) + abs(dc) != 1:
        return

    from src.pipeline.room_stitching import carve_room_connection_between_bboxes

    src_bbox = (
        int(src_pos[1] * ROOM_WIDTH),
        int(src_pos[0] * ROOM_HEIGHT),
        int((src_pos[1] + 1) * ROOM_WIDTH - 1),
        int((src_pos[0] + 1) * ROOM_HEIGHT - 1),
    )
    dst_bbox = (
        int(dst_pos[1] * ROOM_WIDTH),
        int(dst_pos[0] * ROOM_HEIGHT),
        int((dst_pos[1] + 1) * ROOM_WIDTH - 1),
        int((dst_pos[0] + 1) * ROOM_HEIGHT - 1),
    )
    carve_room_connection_between_bboxes(
        global_grid,
        src_bbox,
        dst_bbox,
        edge_data=edge_data,
        has_reverse_edge=has_reverse_edge,
        fill_tile=int(SEMANTIC_PALETTE.get("VOID", 0)),
    )
