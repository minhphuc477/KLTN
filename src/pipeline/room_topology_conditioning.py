"""Build explicit per-room spatial topology priors from mission-graph metadata."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from src.core.definitions import (
    DOOR_POSITIONS,
    ROOM_HEIGHT,
    ROOM_TOPOLOGY_CHANNELS,
    ROOM_TOPOLOGY_CHANNEL_COUNT,
    ROOM_TOPOLOGY_GATE_FAMILY_TOKENS,
    ROOM_WIDTH,
    SEMANTIC_PALETTE,
)
from src.simulation.validator import GameState, ZeldaLogicEnv

_DIRECTION_TO_DOOR_CHANNEL = {
    "N": ROOM_TOPOLOGY_CHANNELS["door_n"],
    "S": ROOM_TOPOLOGY_CHANNELS["door_s"],
    "E": ROOM_TOPOLOGY_CHANNELS["door_e"],
    "W": ROOM_TOPOLOGY_CHANNELS["door_w"],
}
_DIRECTION_TO_GATED_CHANNEL = {
    "N": ROOM_TOPOLOGY_CHANNELS["gated_n"],
    "S": ROOM_TOPOLOGY_CHANNELS["gated_s"],
    "E": ROOM_TOPOLOGY_CHANNELS["gated_e"],
    "W": ROOM_TOPOLOGY_CHANNELS["gated_w"],
}
_ROLE_TO_CHANNEL = {
    "is_start": ROOM_TOPOLOGY_CHANNELS["role_start"],
    "has_enemy": ROOM_TOPOLOGY_CHANNELS["role_enemy"],
    "has_key": ROOM_TOPOLOGY_CHANNELS["role_key"],
    "has_item": ROOM_TOPOLOGY_CHANNELS["role_item"],
    "has_goal": ROOM_TOPOLOGY_CHANNELS["role_goal"],
    "has_boss": ROOM_TOPOLOGY_CHANNELS["role_boss"],
    "has_puzzle": ROOM_TOPOLOGY_CHANNELS["role_puzzle"],
}

_GATED_EDGE_TYPES = set().union(*ROOM_TOPOLOGY_GATE_FAMILY_TOKENS.values())

_DEFAULT_WALKABLE_IDS = {
    int(SEMANTIC_PALETTE["FLOOR"]),
    int(SEMANTIC_PALETTE["DOOR_OPEN"]),
    int(SEMANTIC_PALETTE["DOOR_SOFT"]),
    int(SEMANTIC_PALETTE["START"]),
    int(SEMANTIC_PALETTE["TRIFORCE"]),
    int(SEMANTIC_PALETTE["KEY_SMALL"]),
    int(SEMANTIC_PALETTE["KEY_BOSS"]),
    int(SEMANTIC_PALETTE["KEY_ITEM"]),
    int(SEMANTIC_PALETTE["ITEM_MINOR"]),
    int(SEMANTIC_PALETTE["ELEMENT_FLOOR"]),
    int(SEMANTIC_PALETTE["STAIR"]),
    int(SEMANTIC_PALETTE["ENEMY"]),
    int(SEMANTIC_PALETTE["BOSS"]),
    int(SEMANTIC_PALETTE["PUZZLE"]),
}

_VALIDATOR_COMPLEX_EDGE_TYPES = {
    "key_locked",
    "locked",
    "boss_locked",
    "item_locked",
    "item_gate",
    "bombable",
    "soft_locked",
    "one_way",
    "shutter",
    "switch",
    "switch_locked",
    "state_block",
    "on_off_gate",
}

TOPOLOGY_ANCHOR_POLICY_VERSION = "2026-04-09.semantic_anchor_v7_stateful_puzzle_edge_semantics"
DEFAULT_SEMANTIC_ROLE_PRIOR_STRENGTH = 0.15
DEFAULT_SEMANTIC_ANCHOR_THRESHOLD = 0.5
DEFAULT_SEMANTIC_PUZZLE_OFFSET = 2
DEFAULT_VALIDATOR_PLAN_MAX_STATES = 512


def build_topology_anchor_policy_metadata(
    *,
    semantic_role_prior_strength: float = DEFAULT_SEMANTIC_ROLE_PRIOR_STRENGTH,
    semantic_anchor_threshold: float = DEFAULT_SEMANTIC_ANCHOR_THRESHOLD,
    semantic_puzzle_offset: int = DEFAULT_SEMANTIC_PUZZLE_OFFSET,
    fast_sampler_teacher_fallback_enabled: Optional[bool] = None,
    topology_supervision_mode: Optional[str] = None,
    semantic_constrained_decoding_enabled: Optional[bool] = None,
    semantic_marker_logit_bias: Optional[float] = None,
    semantic_marker_suppression_bias: Optional[float] = None,
    deterministic_graph_marker_overlay_enabled: Optional[bool] = None,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "version": TOPOLOGY_ANCHOR_POLICY_VERSION,
        "semantic_role_prior_strength": float(semantic_role_prior_strength),
        "semantic_anchor_threshold": float(semantic_anchor_threshold),
        "semantic_puzzle_offset": int(semantic_puzzle_offset),
    }
    if fast_sampler_teacher_fallback_enabled is not None:
        metadata["fast_sampler_teacher_fallback_enabled"] = bool(fast_sampler_teacher_fallback_enabled)
    if topology_supervision_mode is not None:
        metadata["topology_supervision_mode"] = str(topology_supervision_mode)
    if semantic_constrained_decoding_enabled is not None:
        metadata["semantic_constrained_decoding_enabled"] = bool(semantic_constrained_decoding_enabled)
    if semantic_marker_logit_bias is not None:
        metadata["semantic_marker_logit_bias"] = float(semantic_marker_logit_bias)
    if semantic_marker_suppression_bias is not None:
        metadata["semantic_marker_suppression_bias"] = float(semantic_marker_suppression_bias)
    if deterministic_graph_marker_overlay_enabled is not None:
        metadata["deterministic_graph_marker_overlay_enabled"] = bool(
            deterministic_graph_marker_overlay_enabled
        )
    return metadata


def _clamp_point(point: Tuple[int, int], shape: Tuple[int, int]) -> Tuple[int, int]:
    h, w = shape
    return (
        max(0, min(h - 1, int(point[0]))),
        max(0, min(w - 1, int(point[1]))),
    )


def _door_center(direction: str) -> Tuple[int, int]:
    spec = DOOR_POSITIONS[str(direction)]
    if direction in {"N", "S"}:
        col = (int(spec["col_start"]) + int(spec["col_end"])) // 2
        return (int(spec["row"]), col)
    row = (int(spec["row_start"]) + int(spec["row_end"])) // 2
    return (row, int(spec["col"]))


def _interior_point(point: Tuple[int, int], shape: Tuple[int, int]) -> Tuple[int, int]:
    h, w = shape
    return (
        max(1, min(h - 2, int(point[0]))),
        max(1, min(w - 2, int(point[1]))),
    )


def _centroid_point(
    points: Sequence[Tuple[int, int]],
    *,
    shape: Tuple[int, int],
    fallback: Tuple[int, int],
) -> Tuple[int, int]:
    if not points:
        return _interior_point(fallback, shape)
    rows = [float(p[0]) for p in points]
    cols = [float(p[1]) for p in points]
    return _interior_point((int(round(sum(rows) / len(rows))), int(round(sum(cols) / len(cols)))), shape)


def _interpolate_point(
    start: Tuple[int, int],
    end: Tuple[int, int],
    *,
    alpha: float,
    shape: Tuple[int, int],
) -> Tuple[int, int]:
    alpha = float(max(0.0, min(1.0, alpha)))
    row = (1.0 - alpha) * float(start[0]) + alpha * float(end[0])
    col = (1.0 - alpha) * float(start[1]) + alpha * float(end[1])
    return _interior_point((int(round(row)), int(round(col))), shape)


def _perpendicular_offset_point(
    source: Tuple[int, int],
    destination: Tuple[int, int],
    *,
    shape: Tuple[int, int],
    magnitude: int = 2,
) -> Tuple[int, int]:
    h, w = shape
    center = (h // 2, w // 2)
    d_row = int(destination[0]) - int(source[0])
    d_col = int(destination[1]) - int(source[1])
    if abs(d_col) >= abs(d_row):
        offset = (int(np.sign(d_col)) or 1, 0)
    else:
        offset = (0, -(int(np.sign(d_row)) or 1))
    return _interior_point(
        (center[0] + offset[0] * int(max(1, magnitude)), center[1] + offset[1] * int(max(1, magnitude))),
        shape,
    )


def build_room_semantic_anchor_points(
    *,
    room_shape: Tuple[int, int] = (ROOM_HEIGHT, ROOM_WIDTH),
    start: Optional[Tuple[int, int]] = None,
    goal: Optional[Tuple[int, int]] = None,
    required_doors: Optional[Mapping[str, bool]] = None,
    incoming_dirs: Optional[Set[str]] = None,
    outgoing_dirs: Optional[Set[str]] = None,
    room_role_flags: Optional[Mapping[str, bool]] = None,
    semantic_puzzle_offset: int = DEFAULT_SEMANTIC_PUZZLE_OFFSET,
) -> Dict[str, Tuple[int, int]]:
    """
    Build deterministic in-room semantic anchors shared by conditioning and placement.

    These anchors are intentionally simple and stable. They preserve coarse
    room-wide role information for backward compatibility while giving the model
    a stronger spatial hint for mission-critical semantics.
    """
    h, w = int(room_shape[0]), int(room_shape[1])
    shape = (h, w)
    center = _interior_point((h // 2, w // 2), shape)

    required = {str(direction): bool(enabled) for direction, enabled in dict(required_doors or {}).items()}
    incoming = {str(direction) for direction in set(incoming_dirs or set())}
    outgoing = {str(direction) for direction in set(outgoing_dirs or set())}
    role_flags = {str(key): bool(value) for key, value in dict(room_role_flags or {}).items()}

    anchors: Dict[str, Tuple[int, int]] = {}
    all_door_points: List[Tuple[int, int]] = []
    incoming_points: List[Tuple[int, int]] = []
    outgoing_points: List[Tuple[int, int]] = []

    for direction in ("N", "S", "E", "W"):
        if not required.get(direction, False):
            continue
        point = _clamp_point(_door_center(direction), shape)
        anchors[f"door:{direction}"] = point
        all_door_points.append(point)
        if direction in incoming:
            incoming_points.append(point)
        if direction in outgoing:
            outgoing_points.append(point)

    source_anchor = (
        _interior_point(start, shape)
        if start is not None
        else _centroid_point(incoming_points or all_door_points, shape=shape, fallback=center)
    )
    destination_anchor = (
        _interior_point(goal, shape)
        if goal is not None
        else _centroid_point(outgoing_points or all_door_points, shape=shape, fallback=center)
    )

    if start is not None or role_flags.get("is_start", False):
        anchors["start"] = source_anchor
    if goal is not None or role_flags.get("has_goal", False):
        anchors["goal"] = destination_anchor

    if role_flags.get("has_enemy", False):
        anchors["enemy"] = _interpolate_point(source_anchor, center, alpha=0.55, shape=shape)
    if role_flags.get("has_key", False):
        anchors["key"] = _interpolate_point(source_anchor, center, alpha=0.72, shape=shape)
    if role_flags.get("has_item", False):
        anchors["item"] = _interpolate_point(center, destination_anchor, alpha=0.38, shape=shape)
    if role_flags.get("has_boss", False):
        anchors["boss"] = _interpolate_point(center, destination_anchor, alpha=0.62, shape=shape)
    if role_flags.get("has_puzzle", False):
        anchors["puzzle"] = _perpendicular_offset_point(
            source_anchor,
            destination_anchor,
            shape=shape,
            magnitude=int(max(0, semantic_puzzle_offset)),
        )

    return anchors


def _paint_line(canvas: np.ndarray, start: Tuple[int, int], end: Tuple[int, int], value: float = 1.0) -> None:
    r0, c0 = start
    r1, c1 = end
    r, c = int(r0), int(c0)
    while r != int(r1):
        canvas[r, c] = value
        r += 1 if r1 > r else -1
    while c != int(c1):
        canvas[r, c] = value
        c += 1 if c1 > c else -1
    canvas[int(r1), int(c1)] = value


def _paint_door_strip(channel: np.ndarray, direction: str, value: float = 1.0) -> None:
    spec = DOOR_POSITIONS[str(direction)]
    if direction in {"N", "S"}:
        channel[int(spec["row"]), int(spec["col_start"]): int(spec["col_end"]) + 1] = value
    else:
        channel[int(spec["row_start"]): int(spec["row_end"]) + 1, int(spec["col"])] = value


def _paint_typed_gated_channels(
    topo: np.ndarray,
    *,
    direction: str,
    tokens: Set[str],
) -> None:
    direction_suffix = str(direction).strip().lower()
    for family_name, family_tokens in ROOM_TOPOLOGY_GATE_FAMILY_TOKENS.items():
        if not (tokens & family_tokens):
            continue
        channel_name = f"{family_name}_{direction_suffix}"
        channel_idx = ROOM_TOPOLOGY_CHANNELS.get(channel_name)
        if channel_idx is None:
            continue
        _paint_door_strip(topo[int(channel_idx)], direction)


def _is_walkable(
    room_grid: np.ndarray,
    point: Tuple[int, int],
    walkable_ids: Set[int],
) -> bool:
    r, c = point
    return 0 <= r < room_grid.shape[0] and 0 <= c < room_grid.shape[1] and int(room_grid[r, c]) in walkable_ids


def nearest_walkable_point(
    room_grid: np.ndarray,
    point: Tuple[int, int],
    *,
    walkable_ids: Optional[Iterable[int]] = None,
) -> Optional[Tuple[int, int]]:
    """Snap a point to the nearest walkable tile in a room grid."""
    walkable = {int(v) for v in (walkable_ids or _DEFAULT_WALKABLE_IDS)}
    start = _clamp_point(point, room_grid.shape[:2])
    if _is_walkable(room_grid, start, walkable):
        return start

    h, w = room_grid.shape[:2]
    visited = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r, c + 1), (r + 1, c), (r, c - 1), (r - 1, c)):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in visited:
                continue
            if _is_walkable(room_grid, (nr, nc), walkable):
                return (nr, nc)
            visited.add((nr, nc))
            queue.append((nr, nc))
    return None


def shortest_path_trace(
    room_grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    *,
    walkable_ids: Optional[Iterable[int]] = None,
) -> Optional[Sequence[Tuple[int, int]]]:
    """Compute a shortest 4-neighbour path on a room grid."""
    walkable = {int(v) for v in (walkable_ids or _DEFAULT_WALKABLE_IDS)}
    src = nearest_walkable_point(room_grid, start, walkable_ids=walkable)
    dst = nearest_walkable_point(room_grid, goal, walkable_ids=walkable)
    if src is None or dst is None:
        return None
    if src == dst:
        return [src]

    h, w = room_grid.shape[:2]
    queue = deque([src])
    parent = {src: None}
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r, c + 1), (r + 1, c), (r, c - 1), (r - 1, c)):
            nxt = (nr, nc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if nxt in parent:
                continue
            if not _is_walkable(room_grid, nxt, walkable):
                continue
            parent[nxt] = (r, c)
            if nxt == dst:
                path = [dst]
                cur = dst
                while parent[cur] is not None:
                    cur = parent[cur]
                    path.append(cur)
                path.reverse()
                return path
            queue.append(nxt)
    return None


def build_traversability_trace_mask(
    room_grid: np.ndarray,
    *,
    anchors: Mapping[str, Tuple[int, int]],
    anchor_pairs: Sequence[Tuple[str, str]],
    walkable_ids: Optional[Iterable[int]] = None,
) -> np.ndarray:
    """
    Build a binary traversability mask from actual shortest paths between anchors.

    This is intended for dataset-side supervision, where the real room geometry is
    available and we can supervise the traversability channel using true traces
    instead of a heuristic scaffold.
    """
    h, w = room_grid.shape[:2]
    trace = np.zeros((h, w), dtype=np.float32)
    for src_name, dst_name in anchor_pairs:
        src = anchors.get(str(src_name))
        dst = anchors.get(str(dst_name))
        if src is None or dst is None:
            continue
        path = shortest_path_trace(
            room_grid,
            src,
            dst,
            walkable_ids=walkable_ids,
        )
        if not path:
            continue
        for r, c in path:
            trace[int(r), int(c)] = 1.0
    return trace


def build_anchor_pairs_from_room_semantics(
    *,
    incoming_dirs: Set[str],
    outgoing_dirs: Set[str],
    required_doors: Mapping[str, bool],
    content_anchors: Mapping[str, Tuple[int, int]],
) -> List[Tuple[str, str]]:
    """Build heuristic anchor pairs for simple room-trace supervision."""
    pairs: List[Tuple[str, str]] = []
    incoming = [f"door:{direction}" for direction in ("N", "S", "E", "W") if direction in incoming_dirs and required_doors.get(direction, False)]
    outgoing = [f"door:{direction}" for direction in ("N", "S", "E", "W") if direction in outgoing_dirs and required_doors.get(direction, False)]
    all_doors = [f"door:{direction}" for direction in ("N", "S", "E", "W") if required_doors.get(direction, False)]

    def _add_pair(src: str, dst: str) -> None:
        if src == dst:
            return
        pair = (src, dst)
        if pair not in pairs:
            pairs.append(pair)

    goal_anchor = "goal" if "goal" in content_anchors else None
    start_anchor = "start" if "start" in content_anchors else None
    intermediate = next((name for name in ("key", "item", "boss", "puzzle", "enemy") if name in content_anchors), None)

    if start_anchor:
        if intermediate:
            _add_pair(start_anchor, intermediate)
            for dst in outgoing or [name for name in all_doors if name != start_anchor]:
                _add_pair(intermediate, dst)
        elif goal_anchor:
            _add_pair(start_anchor, goal_anchor)
        else:
            for dst in outgoing or [name for name in all_doors if name != start_anchor]:
                _add_pair(start_anchor, dst)

    if goal_anchor:
        if intermediate and not start_anchor:
            for src in incoming or [name for name in all_doors if name != goal_anchor]:
                _add_pair(src, intermediate)
            _add_pair(intermediate, goal_anchor)
        else:
            for src in incoming or [name for name in all_doors if name != goal_anchor]:
                _add_pair(src, goal_anchor)

    if intermediate and not start_anchor and not goal_anchor:
        for src in incoming or all_doors[:1]:
            _add_pair(src, intermediate)
        for dst in outgoing or [name for name in all_doors if name != intermediate]:
            _add_pair(intermediate, dst)

    if not pairs:
        if incoming and outgoing:
            for src in incoming:
                for dst in outgoing:
                    _add_pair(src, dst)
        elif len(all_doors) >= 2:
            for idx, src in enumerate(all_doors):
                for dst in all_doors[idx + 1:]:
                    _add_pair(src, dst)

    return pairs


def _sequence_sources(
    incoming_dirs: Set[str],
    required_doors: Mapping[str, bool],
    anchors: Mapping[str, Tuple[int, int]],
) -> List[str]:
    sources = [f"door:{direction}" for direction in ("N", "S", "E", "W") if direction in incoming_dirs and required_doors.get(direction, False) and f"door:{direction}" in anchors]
    if sources:
        return sources
    if "start" in anchors:
        return ["start"]
    fallback_doors = [f"door:{direction}" for direction in ("N", "S", "E", "W") if required_doors.get(direction, False) and f"door:{direction}" in anchors]
    return fallback_doors[:1]


def _sequence_destinations(
    outgoing_dirs: Set[str],
    required_doors: Mapping[str, bool],
    anchors: Mapping[str, Tuple[int, int]],
) -> List[str]:
    if "goal" in anchors:
        return ["goal"]
    destinations = [f"door:{direction}" for direction in ("N", "S", "E", "W") if direction in outgoing_dirs and required_doors.get(direction, False) and f"door:{direction}" in anchors]
    if destinations:
        return destinations
    return [f"door:{direction}" for direction in ("N", "S", "E", "W") if required_doors.get(direction, False) and f"door:{direction}" in anchors]


def _anchor_direction(anchor_name: str) -> Optional[str]:
    if anchor_name.startswith("door:") and len(anchor_name) == 6:
        return anchor_name[-1]
    return None


def _required_prerequisites(
    *,
    destination: str,
    anchors: Mapping[str, Tuple[int, int]],
    edge_constraint_tokens: Mapping[str, Set[str]],
    room_role_flags: Mapping[str, bool],
) -> List[str]:
    prereqs: List[str] = []
    direction = _anchor_direction(destination)
    tokens = edge_constraint_tokens.get(direction or "", set())

    if tokens & {"key_locked", "locked", "boss_locked"} and "key" in anchors:
        prereqs.append("key")
    if tokens & {"item_locked", "item_gate", "bombable"} and "item" in anchors:
        prereqs.append("item")
    if tokens & {"soft_locked", "one_way", "shutter", "switch", "switch_locked", "state_block", "on_off_gate"}:
        for candidate in ("enemy", "boss", "puzzle"):
            if candidate in anchors:
                prereqs.append(candidate)
                break

    if destination == "goal":
        for candidate in ("key", "item", "boss", "puzzle"):
            if candidate in anchors and room_role_flags.get(
                {
                    "key": "has_key",
                    "item": "has_item",
                    "boss": "has_boss",
                    "puzzle": "has_puzzle",
                }[candidate],
                False,
            ):
                prereqs.append(candidate)
                break

    # Preserve order but remove duplicates.
    deduped: List[str] = []
    for name in prereqs:
        if name not in deduped:
            deduped.append(name)
    return deduped


def _build_validator_sequences(
    *,
    anchors: Mapping[str, Tuple[int, int]],
    incoming_dirs: Set[str],
    outgoing_dirs: Set[str],
    required_doors: Mapping[str, bool],
    edge_constraint_tokens: Mapping[str, Set[str]],
    room_role_flags: Mapping[str, bool],
) -> List[List[str]]:
    sequences: List[List[str]] = []
    for source in _sequence_sources(incoming_dirs, required_doors, anchors):
        for destination in _sequence_destinations(outgoing_dirs, required_doors, anchors):
            sequence = [source]
            for prereq in _required_prerequisites(
                destination=destination,
                anchors=anchors,
                edge_constraint_tokens=edge_constraint_tokens,
                room_role_flags=room_role_flags,
            ):
                if prereq in anchors and prereq not in sequence and prereq != destination:
                    sequence.append(prereq)
            if destination not in sequence:
                sequence.append(destination)
            if len(sequence) >= 2 and sequence not in sequences:
                sequences.append(sequence)
    return sequences


def _room_requires_validator_plan(
    *,
    edge_constraint_tokens: Mapping[str, Set[str]],
    room_role_flags: Mapping[str, bool],
    anchors: Mapping[str, Tuple[int, int]],
) -> bool:
    if any(tokens & _VALIDATOR_COMPLEX_EDGE_TYPES for tokens in edge_constraint_tokens.values()):
        return True
    if any(bool(room_role_flags.get(key, False)) for key in ("has_key", "has_item", "has_enemy", "has_boss", "has_puzzle")):
        return True
    return any(name in anchors for name in ("key", "item", "enemy", "boss", "puzzle"))


def _state_key(state: GameState) -> Tuple[Tuple[int, int], int, int, bool, bool, frozenset, frozenset, frozenset, frozenset, int]:
    return (
        tuple(state.position),
        int(state.keys),
        int(state.bomb_count),
        bool(state.has_boss_key),
        bool(state.has_item),
        frozenset(state.opened_doors),
        frozenset(state.collected_items),
        frozenset(state.pushed_blocks),
        frozenset(state.defeated_enemies),
        int(getattr(state, "current_floor", 0)),
    )


def _room_local_state_search(
    env: ZeldaLogicEnv,
    start_state: GameState,
    goal_pos: Tuple[int, int],
    *,
    max_states: int = DEFAULT_VALIDATOR_PLAN_MAX_STATES,
) -> Optional[Tuple[List[Tuple[int, int]], GameState]]:
    """Plan a room-local path using validator transition rules."""
    goal = (int(goal_pos[0]), int(goal_pos[1]))
    state_budget = max(1, int(max_states))
    start_copy = start_state.copy()
    start_key = _state_key(start_copy)
    queue = deque([start_copy])
    parents: Dict[Tuple, Optional[Tuple]] = {start_key: None}
    states: Dict[Tuple, GameState] = {start_key: start_copy}

    while queue:
        current = queue.popleft()
        current_key = _state_key(current)
        if current.position == goal:
            path = [current.position]
            prev_key = parents[current_key]
            while prev_key is not None:
                path.append(states[prev_key].position)
                prev_key = parents[prev_key]
            path.reverse()
            return path, current

        r, c = current.position
        neighbor_positions = [(r, c + 1), (r + 1, c), (r, c - 1), (r - 1, c)]
        neighbor_positions.sort(
            key=lambda pos: abs(int(pos[0]) - goal[0]) + abs(int(pos[1]) - goal[1])
        )
        for nr, nc in neighbor_positions:
            if not (0 <= nr < env.height and 0 <= nc < env.width):
                continue
            can_move, next_state = env.try_move_pure(current, (nr, nc), int(env.grid[nr, nc]))
            if not can_move:
                continue
            next_key = _state_key(next_state)
            if next_key in states:
                continue
            if len(states) >= state_budget:
                continue
            states[next_key] = next_state
            parents[next_key] = current_key
            queue.append(next_state)
    return None


def _initial_state_for_sequence(
    room_grid: np.ndarray,
    start_pos: Tuple[int, int],
    sequence: Sequence[str],
    edge_constraint_tokens: Mapping[str, Set[str]],
) -> GameState:
    direction = _anchor_direction(sequence[-1]) if sequence else None
    final_tokens = edge_constraint_tokens.get(direction or "", set())
    needs_local_key = "key" in sequence
    needs_local_item = "item" in sequence

    return GameState(
        position=(int(start_pos[0]), int(start_pos[1])),
        keys=0 if needs_local_key else (1 if final_tokens & {"key_locked", "locked"} else 0),
        bomb_count=0 if needs_local_item else (1 if "bombable" in final_tokens else 0),
        has_boss_key=(not needs_local_key and "boss_locked" in final_tokens),
        has_item=(not needs_local_item and bool(final_tokens & {"item_locked", "item_gate"})),
    )


def build_validator_room_plan_trace_mask(
    room_grid: np.ndarray,
    *,
    anchors: Mapping[str, Tuple[int, int]],
    incoming_dirs: Set[str],
    outgoing_dirs: Set[str],
    required_doors: Mapping[str, bool],
    edge_constraint_tokens: Optional[Mapping[str, Set[str]]] = None,
    room_role_flags: Optional[Mapping[str, bool]] = None,
    validator_plan_max_states: int = DEFAULT_VALIDATOR_PLAN_MAX_STATES,
) -> np.ndarray:
    """
    Build a traversability mask from validator-aware ordered subgoal plans.

    Complex rooms route through a small room-local planner that reuses the
    validator's inventory/state transition rules, so key/item/combat/shutter
    rooms preserve the gameplay sequence rather than just the shortest corridor.
    """
    h, w = room_grid.shape[:2]
    trace = np.zeros((h, w), dtype=np.float32)
    normalized_tokens = {
        str(direction): {str(token).strip().lower() for token in tokens}
        for direction, tokens in dict(edge_constraint_tokens or {}).items()
    }
    role_flags = {str(key): bool(value) for key, value in dict(room_role_flags or {}).items()}

    sequences = _build_validator_sequences(
        anchors=anchors,
        incoming_dirs=set(incoming_dirs),
        outgoing_dirs=set(outgoing_dirs),
        required_doors=dict(required_doors),
        edge_constraint_tokens=normalized_tokens,
        room_role_flags=role_flags,
    )
    if not sequences:
        return trace

    for sequence in sequences:
        start_anchor = anchors.get(sequence[0])
        if start_anchor is None:
            continue
        env = ZeldaLogicEnv(room_grid, render_mode=False)
        state = _initial_state_for_sequence(room_grid, start_anchor, sequence, normalized_tokens)
        sequence_ok = True

        for anchor_name in sequence[1:]:
            goal_anchor = anchors.get(anchor_name)
            if goal_anchor is None:
                sequence_ok = False
                break
            result = _room_local_state_search(
                env,
                state,
                goal_anchor,
                max_states=int(validator_plan_max_states),
            )
            if result is None:
                sequence_ok = False
                break
            path, state = result
            for r, c in path:
                trace[int(r), int(c)] = 1.0

        if sequence_ok and np.any(trace > 0):
            continue

    return trace


def build_room_traversability_prior(
    room_grid: np.ndarray,
    *,
    anchors: Mapping[str, Tuple[int, int]],
    incoming_dirs: Set[str],
    outgoing_dirs: Set[str],
    required_doors: Mapping[str, bool],
    edge_constraint_tokens: Optional[Mapping[str, Set[str]]] = None,
    room_role_flags: Optional[Mapping[str, bool]] = None,
    validator_plan_max_states: int = DEFAULT_VALIDATOR_PLAN_MAX_STATES,
) -> np.ndarray:
    """Hybrid router: simple rooms use shortest traces, complex rooms use validator plans."""
    normalized_tokens = {
        str(direction): {str(token).strip().lower() for token in tokens}
        for direction, tokens in dict(edge_constraint_tokens or {}).items()
    }
    role_flags = {str(key): bool(value) for key, value in dict(room_role_flags or {}).items()}

    if _room_requires_validator_plan(
        edge_constraint_tokens=normalized_tokens,
        room_role_flags=role_flags,
        anchors=anchors,
    ):
        validator_trace = build_validator_room_plan_trace_mask(
            room_grid,
            anchors=anchors,
            incoming_dirs=incoming_dirs,
            outgoing_dirs=outgoing_dirs,
            required_doors=required_doors,
            edge_constraint_tokens=normalized_tokens,
            room_role_flags=role_flags,
            validator_plan_max_states=int(validator_plan_max_states),
        )
        if np.any(validator_trace > 0):
            return validator_trace

    return build_traversability_trace_mask(
        room_grid,
        anchors=anchors,
        anchor_pairs=build_anchor_pairs_from_room_semantics(
            incoming_dirs=incoming_dirs,
            outgoing_dirs=outgoing_dirs,
            required_doors=required_doors,
            content_anchors=anchors,
        ),
    )


def build_semantic_room_plan_trace(
    room_grid: np.ndarray,
    *,
    required_doors: Mapping[str, bool],
    incoming_dirs: Set[str],
    outgoing_dirs: Set[str],
    edge_constraint_tokens: Optional[Mapping[str, Set[str]]] = None,
    room_role_flags: Optional[Mapping[str, bool]] = None,
    start: Optional[Tuple[int, int]] = None,
    goal: Optional[Tuple[int, int]] = None,
    validator_plan_max_states: int = DEFAULT_VALIDATOR_PLAN_MAX_STATES,
) -> np.ndarray:
    """
    Build a room traversability prior directly from a concrete room grid.

    This is the shared inference/training entry point when actual room tiles are
    available. It derives semantic anchors from doors/items/enemies and routes
    through the hybrid planner.
    """
    role_flags = {str(key): bool(value) for key, value in dict(room_role_flags or {}).items()}
    heuristic_anchors = build_room_semantic_anchor_points(
        room_shape=room_grid.shape[:2],
        start=start,
        goal=goal,
        required_doors=required_doors,
        incoming_dirs=incoming_dirs,
        outgoing_dirs=outgoing_dirs,
        room_role_flags=role_flags,
    )
    anchors: Dict[str, Tuple[int, int]] = {}

    for direction, enabled in dict(required_doors).items():
        if not enabled:
            continue
        snapped = nearest_walkable_point(
            room_grid,
            heuristic_anchors.get(f"door:{direction}", _door_center(str(direction))),
        )
        if snapped is not None:
            anchors[f"door:{direction}"] = snapped

    tile_anchor_specs = {
        "start": ({int(SEMANTIC_PALETTE["START"])}, bool(start is not None) or role_flags.get("is_start", False)),
        "goal": ({int(SEMANTIC_PALETTE["TRIFORCE"])}, bool(goal is not None) or role_flags.get("has_goal", False)),
        "key": ({int(SEMANTIC_PALETTE["KEY_SMALL"]), int(SEMANTIC_PALETTE["KEY_BOSS"])}, role_flags.get("has_key", False)),
        "item": ({int(SEMANTIC_PALETTE["KEY_ITEM"]), int(SEMANTIC_PALETTE["ITEM_MINOR"])}, role_flags.get("has_item", False)),
        "enemy": ({int(SEMANTIC_PALETTE["ENEMY"])}, role_flags.get("has_enemy", False)),
        "boss": ({int(SEMANTIC_PALETTE["BOSS"])}, role_flags.get("has_boss", False)),
        "puzzle": ({int(SEMANTIC_PALETTE["PUZZLE"])}, role_flags.get("has_puzzle", False)),
    }

    def _first_tile(tile_ids: Set[int]) -> Optional[Tuple[int, int]]:
        for tile_id in tile_ids:
            hits = np.argwhere(room_grid == int(tile_id))
            if hits.size > 0:
                return (int(hits[0][0]), int(hits[0][1]))
        return None

    for anchor_name, (tile_ids, enabled) in tile_anchor_specs.items():
        if not enabled:
            continue
        explicit_point = heuristic_anchors.get(anchor_name) or {"start": start, "goal": goal}.get(anchor_name)
        snapped = nearest_walkable_point(
            room_grid,
            explicit_point or _first_tile(tile_ids) or (room_grid.shape[0] // 2, room_grid.shape[1] // 2),
        )
        if snapped is not None:
            anchors[anchor_name] = snapped

    return build_room_traversability_prior(
        room_grid,
        anchors=anchors,
        incoming_dirs=incoming_dirs,
        outgoing_dirs=outgoing_dirs,
        required_doors=required_doors,
        edge_constraint_tokens=edge_constraint_tokens,
        room_role_flags=role_flags,
        validator_plan_max_states=int(validator_plan_max_states),
    )


def _build_synthetic_topology_trace_grid(
    *,
    room_shape: Tuple[int, int],
    semantic_anchors: Mapping[str, Tuple[int, int]],
    required_doors: Mapping[str, bool],
    room_role_flags: Mapping[str, bool],
) -> np.ndarray:
    """Build a simple synthetic room grid for validator-aware topology tracing."""
    h, w = int(room_shape[0]), int(room_shape[1])
    wall = int(SEMANTIC_PALETTE["WALL"])
    floor = int(SEMANTIC_PALETTE["FLOOR"])
    grid = np.full((h, w), wall, dtype=np.int32)
    grid[1:h - 1, 1:w - 1] = floor

    for direction, enabled in dict(required_doors or {}).items():
        if not bool(enabled):
            continue
        spec = DOOR_POSITIONS[str(direction)]
        if direction in {"N", "S"}:
            row = int(spec["row"])
            grid[row, int(spec["col_start"]): int(spec["col_end"]) + 1] = int(SEMANTIC_PALETTE["DOOR_OPEN"])
        else:
            col = int(spec["col"])
            grid[int(spec["row_start"]): int(spec["row_end"]) + 1, col] = int(SEMANTIC_PALETTE["DOOR_OPEN"])

    anchor_tiles: Dict[str, int] = {
        "start": int(SEMANTIC_PALETTE["START"]),
        "goal": int(SEMANTIC_PALETTE["TRIFORCE"]),
        "key": int(SEMANTIC_PALETTE["KEY_SMALL"]),
        "item": int(SEMANTIC_PALETTE["KEY_ITEM"]),
        "enemy": int(SEMANTIC_PALETTE["ENEMY"]),
        "boss": int(SEMANTIC_PALETTE["BOSS"]),
        "puzzle": int(SEMANTIC_PALETTE["PUZZLE"]),
    }
    for anchor_name, tile_id in anchor_tiles.items():
        point = semantic_anchors.get(anchor_name)
        if point is None:
            continue
        r, c = _clamp_point(point, (h, w))
        grid[int(r), int(c)] = int(tile_id)

    return grid


def build_room_topology_condition_map(
    *,
    room_shape: Tuple[int, int] = (ROOM_HEIGHT, ROOM_WIDTH),
    start: Optional[Tuple[int, int]] = None,
    goal: Optional[Tuple[int, int]] = None,
    required_doors: Optional[Mapping[str, bool]] = None,
    incoming_dirs: Optional[Set[str]] = None,
    outgoing_dirs: Optional[Set[str]] = None,
    edge_constraint_tokens: Optional[Mapping[str, Set[str]]] = None,
    room_role_flags: Optional[Mapping[str, bool]] = None,
    traversability_trace: Optional[np.ndarray] = None,
    semantic_role_prior_strength: float = DEFAULT_SEMANTIC_ROLE_PRIOR_STRENGTH,
    semantic_puzzle_offset: int = DEFAULT_SEMANTIC_PUZZLE_OFFSET,
) -> np.ndarray:
    """
    Build a dense [C, H, W] topology prior for a single room.

    The map encodes:
    - explicit start/goal hints
    - required doorway locations and whether they are gated
    - a simple traversability scaffold between important anchors
    - room-role semantics with light room-wide priors plus exact anchor hints
    """
    h, w = int(room_shape[0]), int(room_shape[1])
    topo = np.zeros((ROOM_TOPOLOGY_CHANNEL_COUNT, h, w), dtype=np.float32)
    role_flags = {str(key): bool(value) for key, value in dict(room_role_flags or {}).items()}
    trace_source: Optional[np.ndarray] = None
    semantic_anchors = build_room_semantic_anchor_points(
        room_shape=(h, w),
        start=start,
        goal=goal,
        required_doors=required_doors,
        incoming_dirs=set(incoming_dirs or set()),
        outgoing_dirs=set(outgoing_dirs or set()),
        room_role_flags=role_flags,
        semantic_puzzle_offset=int(max(0, semantic_puzzle_offset)),
    )
    role_prior_strength = float(max(0.0, min(1.0, semantic_role_prior_strength)))

    if role_flags:
        for key, enabled in role_flags.items():
            channel = _ROLE_TO_CHANNEL.get(str(key))
            if channel is not None and bool(enabled):
                topo[channel, :, :] = np.maximum(topo[channel, :, :], role_prior_strength)
                anchor_key = {
                    "is_start": "start",
                    "has_enemy": "enemy",
                    "has_key": "key",
                    "has_item": "item",
                    "has_goal": "goal",
                    "has_boss": "boss",
                    "has_puzzle": "puzzle",
                }.get(str(key))
                if anchor_key is not None and anchor_key in semantic_anchors:
                    r, c = semantic_anchors[anchor_key]
                    topo[channel, int(r), int(c)] = 1.0

    center = (h // 2, w // 2)
    use_trace = isinstance(traversability_trace, np.ndarray) and traversability_trace.shape == (h, w) and bool(np.any(traversability_trace > 0))
    edge_constraint_tokens = {
        str(direction): {str(token).strip().lower() for token in tokens}
        for direction, tokens in dict(edge_constraint_tokens or {}).items()
    }

    if not use_trace and _room_requires_validator_plan(
        edge_constraint_tokens=edge_constraint_tokens,
        room_role_flags=role_flags,
        anchors=semantic_anchors,
    ):
        synthetic_grid = _build_synthetic_topology_trace_grid(
            room_shape=(h, w),
            semantic_anchors=semantic_anchors,
            required_doors=dict(required_doors or {}),
            room_role_flags=role_flags,
        )
        synthetic_trace = build_validator_room_plan_trace_mask(
            synthetic_grid,
            anchors=semantic_anchors,
            incoming_dirs=set(incoming_dirs or set()),
            outgoing_dirs=set(outgoing_dirs or set()),
            required_doors=dict(required_doors or {}),
            edge_constraint_tokens=edge_constraint_tokens,
            room_role_flags=role_flags,
        )
        if bool(np.any(synthetic_trace > 0)):
            trace_source = synthetic_trace.astype(np.float32, copy=False)
            use_trace = True

    if use_trace:
        if trace_source is None:
            trace_source = traversability_trace.astype(np.float32, copy=False)
        topo[ROOM_TOPOLOGY_CHANNELS["traversability"], :, :] = np.maximum(
            topo[ROOM_TOPOLOGY_CHANNELS["traversability"], :, :],
            trace_source,
        )
    else:
        topo[ROOM_TOPOLOGY_CHANNELS["traversability"], center[0], center[1]] = 1.0

    if start is not None:
        sr, sc = _clamp_point(start, (h, w))
        topo[ROOM_TOPOLOGY_CHANNELS["start"], sr, sc] = 1.0
        topo[ROOM_TOPOLOGY_CHANNELS["traversability"], sr, sc] = 1.0
        if not use_trace:
            _paint_line(topo[ROOM_TOPOLOGY_CHANNELS["traversability"]], center, (sr, sc))

    if goal is not None:
        gr, gc = _clamp_point(goal, (h, w))
        topo[ROOM_TOPOLOGY_CHANNELS["goal"], gr, gc] = 1.0
        topo[ROOM_TOPOLOGY_CHANNELS["traversability"], gr, gc] = 1.0
        if not use_trace:
            _paint_line(topo[ROOM_TOPOLOGY_CHANNELS["traversability"]], center, (gr, gc))

    if start is not None and goal is not None and not use_trace:
        _paint_line(
            topo[ROOM_TOPOLOGY_CHANNELS["traversability"]],
            _clamp_point(start, (h, w)),
            _clamp_point(goal, (h, w)),
        )

    required_doors = dict(required_doors or {})
    for direction in ("N", "S", "E", "W"):
        if not bool(required_doors.get(direction, False)):
            continue
        _paint_door_strip(topo[_DIRECTION_TO_DOOR_CHANNEL[direction]], direction)
        door_center = _door_center(direction)
        topo[ROOM_TOPOLOGY_CHANNELS["traversability"], door_center[0], door_center[1]] = 1.0
        if not use_trace:
            _paint_line(topo[ROOM_TOPOLOGY_CHANNELS["traversability"]], center, door_center)
        tokens = edge_constraint_tokens.get(direction, set())
        if tokens & _GATED_EDGE_TYPES:
            _paint_door_strip(topo[_DIRECTION_TO_GATED_CHANNEL[direction]], direction)
            _paint_typed_gated_channels(
                topo,
                direction=direction,
                tokens=tokens,
            )

    if not use_trace:
        for anchor_name, point in semantic_anchors.items():
            if anchor_name.startswith("door:"):
                continue
            topo[ROOM_TOPOLOGY_CHANNELS["traversability"], int(point[0]), int(point[1])] = 1.0
        for src_name, dst_name in build_anchor_pairs_from_room_semantics(
            incoming_dirs=set(incoming_dirs or set()),
            outgoing_dirs=set(outgoing_dirs or set()),
            required_doors=required_doors,
            content_anchors=semantic_anchors,
        ):
            src = semantic_anchors.get(str(src_name))
            dst = semantic_anchors.get(str(dst_name))
            if src is None or dst is None:
                continue
            _paint_line(topo[ROOM_TOPOLOGY_CHANNELS["traversability"]], src, dst)

    return topo
