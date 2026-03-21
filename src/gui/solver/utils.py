"""Utility helpers shared by GUI solver orchestration code."""

from __future__ import annotations

import logging
import os
import pickle
from collections import deque
from typing import Any, Optional

logger = logging.getLogger(__name__)


def safe_unpickle(path: str) -> dict:
    """Safely load a pickle produced by our own processes and validate shape."""
    try:
        if not path or not os.path.exists(path):
            return {"success": False, "message": "output file missing"}
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, dict):
            logger.warning("safe_unpickle: unexpected payload type %s for %s", type(obj), path)
            return {"success": False, "message": "invalid output format"}
        return obj
    except Exception as e:
        logger.exception("safe_unpickle failed for %s: %s", path, e)
        return {"success": False, "message": "unpickle error"}


def convert_diagonal_to_4dir(path: Optional[list], grid: Optional[Any] = None):
    """Convert a path with diagonal moves to a purely 4-directional path."""
    if not path or len(path) < 2:
        return path

    try:
        from src.simulation.validator import BLOCKING_IDS, WATER_IDS
        obstacle_ids = BLOCKING_IDS | WATER_IDS
    except ImportError:
        obstacle_ids = {0, 2, 40}

    def is_walkable(pos):
        if grid is None:
            return True
        r, c = pos
        if r < 0 or r >= grid.shape[0] or c < 0 or c >= grid.shape[1]:
            return False
        tile_id = int(grid[r, c])
        return tile_id not in obstacle_ids

    def find_short_orth_path(start_pos, end_pos):
        if start_pos == end_pos:
            return [start_pos]
        if grid is None:
            return None

        min_r = max(0, min(start_pos[0], end_pos[0]) - 1)
        max_r = min(grid.shape[0] - 1, max(start_pos[0], end_pos[0]) + 1)
        min_c = max(0, min(start_pos[1], end_pos[1]) - 1)
        max_c = min(grid.shape[1] - 1, max(start_pos[1], end_pos[1]) + 1)

        q = deque([(start_pos, [start_pos])])
        visited = {start_pos}
        while q:
            pos, p = q.popleft()
            if len(p) > 6:
                continue
            if pos == end_pos:
                return p
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = pos[0] + dr, pos[1] + dc
                npos = (nr, nc)
                if not (min_r <= nr <= max_r and min_c <= nc <= max_c):
                    continue
                if npos in visited:
                    continue
                if not is_walkable(npos):
                    continue
                visited.add(npos)
                q.append((npos, p + [npos]))
        return None

    converted = [path[0]]
    for i in range(len(path) - 1):
        curr = path[i]
        next_pos = path[i + 1]
        dr = next_pos[0] - curr[0]
        dc = next_pos[1] - curr[1]

        if dr != 0 and dc != 0:
            vert_first = (curr[0] + dr, curr[1])
            horz_first = (curr[0], curr[1] + dc)

            if is_walkable(vert_first):
                intermediate = vert_first
            elif is_walkable(horz_first):
                intermediate = horz_first
            else:
                detour = find_short_orth_path(curr, next_pos)
                if detour and len(detour) >= 2:
                    converted.extend(detour[1:])
                else:
                    converted.append(next_pos)
                    logger.warning(
                        "Diagonal conversion: no safe orth split at %s->%s (vert=%s, horz=%s); keeping direct step",
                        curr,
                        next_pos,
                        vert_first,
                        horz_first,
                    )
                continue

            converted.append(intermediate)
            converted.append(next_pos)
        else:
            converted.append(next_pos)

    return converted
