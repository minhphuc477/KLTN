"""Shared graph/spatial parsing helpers for dungeon pipeline orchestration."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Set

import networkx as nx
import numpy as np

from src.core.definitions import ROOM_HEIGHT, ROOM_WIDTH, SEMANTIC_PALETTE, parse_edge_type_tokens, parse_node_label_tokens


def parse_label_tokens(label: Any) -> Set[str]:
    """Split node labels like 'e,k' into normalized tokens."""
    if label is None:
        return set()
    return set(str(t).strip().lower() for t in parse_node_label_tokens(str(label)) if str(t).strip())


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


def get_node_grid_position(graph: nx.Graph, node_id: int) -> Optional[Tuple[int, int]]:
    """Extract room-grid position for a node from graph metadata."""
    if node_id not in graph:
        return None
    attrs = graph.nodes[node_id]
    for key in ("position", "pos", "grid_pos", "coord", "coords"):
        pos = parse_room_coord(attrs.get(key))
        if pos is not None:
            return pos
    return None


def infer_direction(graph: nx.Graph, source_node: int, target_node: int) -> Optional[str]:
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
    if room_grid.shape == (ROOM_HEIGHT, ROOM_WIDTH):
        return room_grid.astype(np.int32, copy=False)

    fitted = np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.int32)
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
    """Carve boundary connectors for adjacent rooms, preserving edge semantics when possible."""
    dr = dst_pos[0] - src_pos[0]
    dc = dst_pos[1] - src_pos[1]
    if abs(dr) + abs(dc) != 1:
        return

    floor_id = int(SEMANTIC_PALETTE.get("FLOOR", 1))
    wall_id = int(SEMANTIC_PALETTE.get("WALL", 2))

    data = edge_data or {}
    label = str(data.get("label", "") or "")
    edge_type = str(data.get("edge_type", data.get("type", "")) or "")
    edge_tokens = set(parse_edge_type_tokens(label=label, edge_type=edge_type))

    # Default connector semantics.
    src_tile = floor_id
    dst_tile = floor_id

    # Encode gate semantics into boundary tiles.
    if {"key_locked", "locked"}.intersection(edge_tokens):
        src_tile = int(SEMANTIC_PALETTE.get("DOOR_LOCKED", floor_id))
        dst_tile = int(SEMANTIC_PALETTE.get("DOOR_LOCKED", floor_id))
    elif "boss_locked" in edge_tokens:
        src_tile = int(SEMANTIC_PALETTE.get("DOOR_BOSS", floor_id))
        dst_tile = int(SEMANTIC_PALETTE.get("DOOR_BOSS", floor_id))
    elif "bombable" in edge_tokens:
        src_tile = int(SEMANTIC_PALETTE.get("DOOR_BOMB", floor_id))
        dst_tile = int(SEMANTIC_PALETTE.get("DOOR_BOMB", floor_id))
    elif {"item_gate", "item_locked", "switch", "switch_locked", "on_off_gate", "state_block"}.intersection(edge_tokens):
        src_tile = int(SEMANTIC_PALETTE.get("DOOR_PUZZLE", floor_id))
        dst_tile = int(SEMANTIC_PALETTE.get("DOOR_PUZZLE", floor_id))

    # If there is no reverse edge (or explicit one-way token), mark source as soft door
    # and keep destination as a normal doorway. This preserves directional intent in tiles
    # while keeping traversal possible for current grid-only validators.
    if (not has_reverse_edge) or {"soft_locked", "one_way", "shutter"}.intersection(edge_tokens):
        src_tile = int(SEMANTIC_PALETTE.get("DOOR_SOFT", src_tile))
        if dst_tile == wall_id:
            dst_tile = floor_id

    if dr != 0:
        src_row = (src_pos[0] + (1 if dr > 0 else 0)) * ROOM_HEIGHT - (1 if dr > 0 else 0)
        dst_row = src_row + (1 if dr > 0 else -1)
        center_c = src_pos[1] * ROOM_WIDTH + ROOM_WIDTH // 2
        for col in range(center_c - 2, center_c + 3):
            if 0 <= src_row < global_grid.shape[0] and 0 <= col < global_grid.shape[1]:
                global_grid[src_row, col] = src_tile
            if 0 <= dst_row < global_grid.shape[0] and 0 <= col < global_grid.shape[1]:
                global_grid[dst_row, col] = dst_tile
        return

    src_col = (src_pos[1] + (1 if dc > 0 else 0)) * ROOM_WIDTH - (1 if dc > 0 else 0)
    dst_col = src_col + (1 if dc > 0 else -1)
    center_r = src_pos[0] * ROOM_HEIGHT + ROOM_HEIGHT // 2
    for row in range(center_r - 2, center_r + 3):
        if 0 <= row < global_grid.shape[0] and 0 <= src_col < global_grid.shape[1]:
            global_grid[row, src_col] = src_tile
        if 0 <= row < global_grid.shape[0] and 0 <= dst_col < global_grid.shape[1]:
            global_grid[row, dst_col] = dst_tile
