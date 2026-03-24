import sys
from pathlib import Path
from typing import Any, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.zelda_data.zelda_core import ZeldaDungeonAdapter
from src.core.definitions import ROOM_HEIGHT, ROOM_WIDTH, TileID, parse_edge_type_tokens


def _boundary_connection_exists(global_grid: np.ndarray, src_pos: Tuple[int, int], dst_pos: Tuple[int, int]) -> bool:
    dr = int(dst_pos[0] - src_pos[0])
    dc = int(dst_pos[1] - src_pos[1])
    if abs(dr) + abs(dc) != 1:
        return False

    blocked = {
        int(TileID.VOID),
        int(TileID.WALL),
        int(TileID.BLOCK),
        int(TileID.ELEMENT),
    }

    def _is_passable(val: Any) -> bool:
        try:
            return int(val) not in blocked
        except Exception:
            return False

    if dr != 0:
        src_row = (src_pos[0] + (1 if dr > 0 else 0)) * ROOM_HEIGHT - (1 if dr > 0 else 0)
        dst_row = src_row + (1 if dr > 0 else -1)
        c0 = src_pos[1] * ROOM_WIDTH
        c1 = c0 + ROOM_WIDTH
        for col in range(c0, c1):
            if 0 <= src_row < global_grid.shape[0] and 0 <= dst_row < global_grid.shape[0] and 0 <= col < global_grid.shape[1]:
                if _is_passable(global_grid[src_row, col]) and _is_passable(global_grid[dst_row, col]):
                    return True
        return False

    src_col = (src_pos[1] + (1 if dc > 0 else 0)) * ROOM_WIDTH - (1 if dc > 0 else 0)
    dst_col = src_col + (1 if dc > 0 else -1)
    r0 = src_pos[0] * ROOM_HEIGHT
    r1 = r0 + ROOM_HEIGHT
    for row in range(r0, r1):
        if 0 <= row < global_grid.shape[0] and 0 <= src_col < global_grid.shape[1] and 0 <= dst_col < global_grid.shape[1]:
            if _is_passable(global_grid[row, src_col]) and _is_passable(global_grid[row, dst_col]):
                return True
    return False


def _boundary_has_directional_marker(global_grid: np.ndarray, src_pos: Tuple[int, int], dst_pos: Tuple[int, int]) -> bool:
    dr = int(dst_pos[0] - src_pos[0])
    dc = int(dst_pos[1] - src_pos[1])
    if abs(dr) + abs(dc) != 1:
        return False

    directional_marker_ids = {
        int(TileID.DOOR_SOFT),
        int(TileID.DOOR_LOCKED),
        int(TileID.DOOR_BOMB),
        int(TileID.DOOR_PUZZLE),
        int(TileID.DOOR_BOSS),
    }

    if dr != 0:
        src_row = (src_pos[0] + (1 if dr > 0 else 0)) * ROOM_HEIGHT - (1 if dr > 0 else 0)
        c0 = src_pos[1] * ROOM_WIDTH
        c1 = c0 + ROOM_WIDTH
        for col in range(c0, c1):
            if 0 <= src_row < global_grid.shape[0] and 0 <= col < global_grid.shape[1]:
                if int(global_grid[src_row, col]) in directional_marker_ids:
                    return True
        return False

    src_col = (src_pos[1] + (1 if dc > 0 else 0)) * ROOM_WIDTH - (1 if dc > 0 else 0)
    r0 = src_pos[0] * ROOM_HEIGHT
    r1 = r0 + ROOM_HEIGHT
    for row in range(r0, r1):
        if 0 <= row < global_grid.shape[0] and 0 <= src_col < global_grid.shape[1]:
            if int(global_grid[row, src_col]) in directional_marker_ids:
                return True
    return False

adapter = ZeldaDungeonAdapter("Data/The Legend of Zelda")
dungeon = adapter.load_dungeon(1, variant=1)
stitched = adapter.stitch_dungeon(dungeon)

placement = dict(stitched.node_to_room or {})
grid = np.asarray(stitched.global_grid, dtype=np.int32)

directional_tokens = {
    "soft_locked", "one_way", "state_block", "switch", "switch_locked", "on_off_gate",
    "item_gate", "item_locked", "boss_locked", "key_locked", "locked", "multi_lock", "hazard", "shutter"
}

directed_candidates = 0
directed_realized = 0
directed_leaks = 0
for u, v, attrs in stitched.graph.edges(data=True):
    if u not in placement or v not in placement or u == v:
        continue
    if abs(placement[u][0] - placement[v][0]) + abs(placement[u][1] - placement[v][1]) != 1:
        continue
    label = str(attrs.get("label", "") or "")
    edge_type = str(attrs.get("edge_type", attrs.get("type", "")) or "")
    tokens = set(parse_edge_type_tokens(label=label, edge_type=edge_type))
    reverse_exists = bool(stitched.graph.has_edge(v, u))
    is_directional = (not reverse_exists) or bool(tokens.intersection(directional_tokens))
    if not is_directional:
        continue
    directed_candidates += 1
    opened = _boundary_connection_exists(grid, placement[u], placement[v])
    if opened:
        directed_realized += 1
        if not reverse_exists:
            has_marker = _boundary_has_directional_marker(grid, placement[u], placement[v])
            if not has_marker:
                directed_leaks += 1

if directed_candidates == 0:
    print({"directed_candidates": 0, "directed_edge_preservation_score": 0.0})
else:
    realization = directed_realized / directed_candidates
    leak = directed_leaks / max(1, directed_realized)
    score = realization * (1.0 - leak)
    print({
        "directed_candidates": directed_candidates,
        "directed_edge_realization_rate": realization,
        "directed_directionality_leak_rate": leak,
        "directed_edge_preservation_score": score,
    })
