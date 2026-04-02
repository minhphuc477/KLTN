"""
Adapters for comparing generated Zelda topologies against PCG Benchmark.

This module bridges the repo's mission-graph representation to the external
`pcg_benchmark` Zelda task, whose content space is a 2D grid with tiles:

0 wall, 1 empty, 2 player, 3 key, 4 door, 5 enemy

and control space:

{"player_key": int, "key_door": int}

The mapping is intentionally explicit and lossy. It preserves high-level
topology, primary progression anchors, and enemy pressure, while collapsing the
 richer mission-graph semantics into the simpler benchmark domain.
"""

from __future__ import annotations

import importlib
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from src.core.definitions import parse_node_label_tokens
from src.pipeline.spatial_utils import get_node_grid_position, stable_node_sort_key
from src.utils.stable_seed import stable_seed_offset


PCG_ZELDA_TILE_WALL = 0
PCG_ZELDA_TILE_EMPTY = 1
PCG_ZELDA_TILE_PLAYER = 2
PCG_ZELDA_TILE_KEY = 3
PCG_ZELDA_TILE_DOOR = 4
PCG_ZELDA_TILE_ENEMY = 5


@dataclass(frozen=True)
class PCGBenchmarkZeldaVariant:
    name: str
    width: int
    height: int
    enemies: int
    solution_length: int

    @property
    def enemy_tolerance(self) -> int:
        return max(int(self.enemies * 0.25), 1)

    @property
    def enemy_quality_min(self) -> int:
        return max(0, int(self.enemies - self.enemy_tolerance))

    @property
    def enemy_quality_max(self) -> int:
        return int(self.enemies + self.enemy_tolerance)

    @property
    def control_tolerance(self) -> int:
        return max(int((self.solution_length / 2.0) * 0.25), 1)

    @property
    def control_min(self) -> int:
        return int(self.solution_length / 2.0 + self.control_tolerance)

    @property
    def control_max(self) -> int:
        return int(self.width * self.height / 4)


PCG_BENCHMARK_ZELDA_VARIANTS: Dict[str, PCGBenchmarkZeldaVariant] = {
    "zelda-v0": PCGBenchmarkZeldaVariant(name="zelda-v0", width=11, height=7, enemies=3, solution_length=18),
    "zelda-enemies-v0": PCGBenchmarkZeldaVariant(
        name="zelda-enemies-v0", width=11, height=7, enemies=12, solution_length=18
    ),
    "zelda-large-v0": PCGBenchmarkZeldaVariant(
        name="zelda-large-v0", width=18, height=12, enemies=8, solution_length=30
    ),
}


@dataclass
class PCGBenchmarkZeldaMapping:
    problem_name: str
    content: np.ndarray
    graph_control: Dict[str, int]
    content_control: Dict[str, int]
    metadata: Dict[str, Any]


def _node_tokens(attrs: Dict[str, Any]) -> set[str]:
    tokens: List[str] = []
    label = str(attrs.get("label", "") or "")
    if label:
        tokens.extend(parse_node_label_tokens(label))
    node_type = str(attrs.get("type", "") or "")
    if node_type:
        tokens.append(node_type)
    out = set()
    for token in tokens:
        key = str(token).strip().lower()
        if key:
            out.add(key)
    if bool(attrs.get("is_start")) or bool(attrs.get("is_entry")):
        out.update({"start", "s"})
    if bool(attrs.get("is_goal")) or bool(attrs.get("is_triforce")):
        out.update({"goal", "triforce", "t"})
    if bool(attrs.get("has_key")):
        out.update({"key", "k"})
    if bool(attrs.get("has_enemy")):
        out.update({"enemy", "e"})
    return out


def _ordered_nodes(graph: nx.Graph) -> List[Any]:
    return sorted(graph.nodes(), key=stable_node_sort_key)


def _select_start_node(graph: nx.Graph) -> Optional[Any]:
    candidates = [node for node, attrs in graph.nodes(data=True) if {"start", "s"}.intersection(_node_tokens(dict(attrs)))]
    if candidates:
        return min(candidates, key=stable_node_sort_key)
    return None


def _select_goal_node(graph: nx.Graph, start: Any) -> Optional[Any]:
    candidates = [node for node, attrs in graph.nodes(data=True) if {"goal", "triforce", "t"}.intersection(_node_tokens(dict(attrs)))]
    if candidates:
        return min(candidates, key=stable_node_sort_key)
    return None


def _select_key_node(graph: nx.Graph, start: Any, goal: Any) -> Optional[Any]:
    U = graph.to_undirected()
    candidates = [node for node, attrs in graph.nodes(data=True) if {"key", "k", "boss_key", "key_small"}.intersection(_node_tokens(dict(attrs)))]
    if candidates:
        scored: List[Tuple[int, int, Any]] = []
        for node in candidates:
            try:
                d1 = int(nx.shortest_path_length(U, start, node))
                d2 = int(nx.shortest_path_length(U, node, goal))
            except Exception:
                continue
            scored.append((d1 + d2, d1, node))
        if scored:
            scored.sort(key=lambda item: (item[0], item[1], stable_node_sort_key(item[2])))
            return scored[0][2]
    return None


def _invalid_mapping(
    variant: PCGBenchmarkZeldaVariant,
    *,
    reason: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> PCGBenchmarkZeldaMapping:
    content = np.full((int(variant.height), int(variant.width)), PCG_ZELDA_TILE_WALL, dtype=np.int32)
    payload = {
        "semantic_valid": False,
        "semantic_error": str(reason),
        "enemy_target": int(variant.enemies),
        "enemy_quality_min": int(variant.enemy_quality_min),
        "enemy_quality_max": int(variant.enemy_quality_max),
        "solution_length_target": int(variant.solution_length),
        "graph_control_raw": {"player_key": 0, "key_door": 0},
        "graph_control_aligned": {"player_key": 0, "key_door": 0},
        "content_control_initial": {"player_key": 0, "key_door": 0},
        "content_control_final": {"player_key": 0, "key_door": 0},
        "mapper_mode": "invalid_semantics",
        "control_fallback_applied": False,
    }
    if metadata:
        payload.update(metadata)
    return PCGBenchmarkZeldaMapping(
        problem_name=variant.name,
        content=content,
        graph_control={"player_key": 0, "key_door": 0},
        content_control={"player_key": 0, "key_door": 0},
        metadata=payload,
    )


def _count_enemy_signal(graph: nx.Graph) -> int:
    total = 0
    for _, attrs in graph.nodes(data=True):
        attrs_dict = dict(attrs)
        tokens = _node_tokens(attrs_dict)
        hint = attrs_dict.get("enemy_count_hint", attrs_dict.get("enemy_count", 0))
        try:
            hint_value = max(0, int(hint))
        except Exception:
            hint_value = 0
        if {"enemy", "e", "boss"}.intersection(tokens):
            total += max(1, hint_value)
        else:
            total += hint_value
    return int(total)


def select_pcg_benchmark_zelda_problem(
    graph: nx.Graph,
    *,
    prefer_problem: Optional[str] = None,
) -> PCGBenchmarkZeldaVariant:
    if prefer_problem:
        key = str(prefer_problem).strip()
        if key not in PCG_BENCHMARK_ZELDA_VARIANTS:
            raise ValueError(
                f"Unsupported PCG benchmark Zelda problem '{prefer_problem}'. "
                f"Valid: {sorted(PCG_BENCHMARK_ZELDA_VARIANTS.keys())}"
            )
        return PCG_BENCHMARK_ZELDA_VARIANTS[key]

    node_count = int(graph.number_of_nodes())
    enemy_signal = _count_enemy_signal(graph)
    if node_count > 24:
        return PCG_BENCHMARK_ZELDA_VARIANTS["zelda-large-v0"]
    if enemy_signal >= 10:
        return PCG_BENCHMARK_ZELDA_VARIANTS["zelda-enemies-v0"]
    return PCG_BENCHMARK_ZELDA_VARIANTS["zelda-v0"]


def _graph_positions(graph: nx.Graph, *, seed: int) -> Dict[Any, Tuple[float, float]]:
    raw: Dict[Any, Tuple[float, float]] = {}
    for node in _ordered_nodes(graph):
        pos = get_node_grid_position(graph, node)
        if pos is not None:
            raw[node] = (float(pos[0]), float(pos[1]))
    if len(raw) == graph.number_of_nodes() and len(set(raw.values())) >= max(2, graph.number_of_nodes() // 3):
        return raw

    layout = nx.spring_layout(graph.to_undirected(), seed=int(seed))
    if not layout:
        ordered = _ordered_nodes(graph)
        return {node: (0.0, float(i)) for i, node in enumerate(ordered)}
    return {node: (float(layout[node][1]), float(layout[node][0])) for node in graph.nodes()}


def _nearest_free(
    target: Tuple[int, int],
    *,
    width: int,
    height: int,
    occupied: set[Tuple[int, int]],
) -> Tuple[int, int]:
    r0, c0 = target
    r0 = int(np.clip(r0, 0, height - 1))
    c0 = int(np.clip(c0, 0, width - 1))
    if (r0, c0) not in occupied:
        return (r0, c0)
    for radius in range(1, max(width, height) + 1):
        candidates: List[Tuple[int, int]] = []
        for dr in range(-radius, radius + 1):
            rem = radius - abs(dr)
            for dc in (-rem, rem):
                rr = r0 + dr
                cc = c0 + dc
                if 0 <= rr < height and 0 <= cc < width:
                    candidates.append((rr, cc))
        candidates = sorted(set(candidates))
        for cell in candidates:
            if cell not in occupied:
                return cell
    return (r0, c0)


def _embed_positions(
    positions: Dict[Any, Tuple[float, float]],
    *,
    width: int,
    height: int,
    locked: Optional[Dict[Any, Tuple[int, int]]] = None,
    occupied: Optional[Iterable[Tuple[int, int]]] = None,
) -> Dict[Any, Tuple[int, int]]:
    locked_positions = dict(locked or {})
    if not positions and not locked_positions:
        return {}
    if not positions:
        return {node: (int(cell[0]), int(cell[1])) for node, cell in locked_positions.items()}
    rows = np.asarray([p[0] for p in positions.values()], dtype=np.float64)
    cols = np.asarray([p[1] for p in positions.values()], dtype=np.float64)
    row_span = float(np.max(rows) - np.min(rows))
    col_span = float(np.max(cols) - np.min(cols))
    embedded: Dict[Any, Tuple[int, int]] = dict(locked_positions)
    occupied_cells: set[Tuple[int, int]] = set((int(r), int(c)) for r, c in (occupied or []))
    occupied_cells.update((int(r), int(c)) for r, c in locked_positions.values())
    for node in sorted(positions.keys(), key=stable_node_sort_key):
        if node in locked_positions:
            continue
        row, col = positions[node]
        nr = 0.0 if row_span <= 1e-9 else (float(row) - float(np.min(rows))) / row_span
        nc = 0.0 if col_span <= 1e-9 else (float(col) - float(np.min(cols))) / col_span
        target_r = int(round(nr * max(0, height - 1)))
        target_c = int(round(nc * max(0, width - 1)))
        cell = _nearest_free((target_r, target_c), width=width, height=height, occupied=occupied_cells)
        occupied_cells.add(cell)
        embedded[node] = cell
    return embedded


def _ordered_pair(u: Any, v: Any) -> Tuple[Any, Any]:
    if stable_node_sort_key(u) <= stable_node_sort_key(v):
        return u, v
    return v, u


def _carve_corridor(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    *,
    horizontal_first: bool,
) -> None:
    r, c = int(start[0]), int(start[1])
    tr, tc = int(goal[0]), int(goal[1])
    grid[r, c] = PCG_ZELDA_TILE_EMPTY

    def _step_row() -> None:
        nonlocal r
        while r != tr:
            r += 1 if tr > r else -1
            grid[r, c] = PCG_ZELDA_TILE_EMPTY

    def _step_col() -> None:
        nonlocal c
        while c != tc:
            c += 1 if tc > c else -1
            grid[r, c] = PCG_ZELDA_TILE_EMPTY

    if horizontal_first:
        _step_col()
        _step_row()
    else:
        _step_row()
        _step_col()


def _serpentine_budget_path(width: int, height: int) -> List[Tuple[int, int]]:
    path: List[Tuple[int, int]] = []
    walk_rows = [row for row in range(0, int(height), 2)]
    forward = True
    for idx, row in enumerate(walk_rows):
        cols = range(0, int(width)) if forward else range(int(width) - 1, -1, -1)
        for col in cols:
            path.append((int(row), int(col)))
        if idx < len(walk_rows) - 1:
            end_col = int(width - 1) if forward else 0
            path.append((int(row + 1), end_col))
        forward = not forward
    return path


def _tile_locations(content: np.ndarray, tile_value: int) -> List[Tuple[int, int]]:
    ys, xs = np.where(np.asarray(content, dtype=np.int32) == int(tile_value))
    return [(int(y), int(x)) for y, x in zip(ys.tolist(), xs.tolist())]


def _grid_distance(content: np.ndarray, start_tile: int, goal_tile: int) -> int:
    starts = _tile_locations(content, start_tile)
    goals = set(_tile_locations(content, goal_tile))
    if not starts or not goals:
        return 0
    passable = {
        PCG_ZELDA_TILE_EMPTY,
        PCG_ZELDA_TILE_PLAYER,
        PCG_ZELDA_TILE_KEY,
        PCG_ZELDA_TILE_DOOR,
        PCG_ZELDA_TILE_ENEMY,
    }
    q = deque()
    visited = set()
    for start in starts:
        q.append((start[0], start[1], 0))
        visited.add(start)
    height, width = content.shape
    while q:
        r, c, d = q.popleft()
        if (r, c) in goals:
            return int(d)
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if (nr, nc) in visited:
                continue
            if int(content[nr, nc]) not in passable:
                continue
            visited.add((nr, nc))
            q.append((nr, nc, d + 1))
    return 0


def _graph_path_lengths(graph: nx.Graph, start: Any, key: Any, goal: Any) -> Dict[str, int]:
    U = graph.to_undirected()
    out = {"player_key": 0, "key_door": 0}
    try:
        out["player_key"] = int(max(0, nx.shortest_path_length(U, start, key)))
    except Exception:
        out["player_key"] = 0
    try:
        out["key_door"] = int(max(0, nx.shortest_path_length(U, key, goal)))
    except Exception:
        out["key_door"] = 0
    return out


def _main_progression_nodes(graph: nx.Graph, start: Any, key: Any, goal: Any) -> set[Any]:
    U = graph.to_undirected()
    nodes: set[Any] = {start, key, goal}
    try:
        nodes.update(nx.shortest_path(U, start, key))
    except Exception:
        pass
    try:
        nodes.update(nx.shortest_path(U, key, goal))
    except Exception:
        pass
    return nodes


def _benchmark_aligned_graph_control(raw_control: Dict[str, int], variant: PCGBenchmarkZeldaVariant) -> Dict[str, int]:
    raw_player_key = int(max(0, raw_control.get("player_key", 0)))
    raw_key_door = int(max(0, raw_control.get("key_door", 0)))
    raw_total = raw_player_key + raw_key_door
    lower = int(variant.control_min)
    upper = int(max(lower, variant.control_max))
    base_total = int(max(variant.solution_length, lower * 2))
    extra_capacity = int(max(0, upper * 2 - base_total))
    extra_total = int(min(extra_capacity, max(0, raw_total - 2)))
    target_total = int(base_total + extra_total)
    if raw_total <= 0:
        ratio = 0.5
    else:
        ratio = float(raw_player_key) / float(raw_total)
    player_key = int(round(target_total * ratio))
    key_door = int(target_total - player_key)
    player_key = int(np.clip(player_key, lower, upper))
    key_door = int(np.clip(key_door, lower, upper))
    while player_key + key_door < target_total:
        if player_key <= key_door and player_key < upper:
            player_key += 1
            continue
        if key_door < upper:
            key_door += 1
            continue
        if player_key < upper:
            player_key += 1
            continue
        break
    return {
        "player_key": int(player_key),
        "key_door": int(key_door),
    }


def _reserve_progression_path(
    variant: PCGBenchmarkZeldaVariant,
    control_targets: Dict[str, int],
) -> Dict[str, Any]:
    full_path = _serpentine_budget_path(int(variant.width), int(variant.height))
    player_key = int(max(1, control_targets.get("player_key", variant.control_min)))
    key_door = int(max(1, control_targets.get("key_door", variant.control_min)))
    total_budget = int(player_key + key_door)
    usable_budget = int(min(total_budget, max(1, len(full_path) - 1)))
    max_start = int(max(0, len(full_path) - (usable_budget + 1)))
    start_idx = int(max_start // 2)
    if total_budget > usable_budget:
        scale = float(usable_budget) / float(total_budget)
        player_key = max(1, int(round(player_key * scale)))
        key_door = max(1, usable_budget - player_key)
        if player_key + key_door < usable_budget:
            key_door += usable_budget - (player_key + key_door)
    key_idx = int(start_idx + player_key)
    goal_idx = int(key_idx + key_door)
    corridor = list(full_path[start_idx : goal_idx + 1])
    return {
        "corridor": corridor,
        "player_key": int(key_idx - start_idx),
        "key_door": int(goal_idx - key_idx),
        "start_cell": corridor[0],
        "key_cell": corridor[int(key_idx - start_idx)],
        "goal_cell": corridor[-1],
    }


def _build_control_strict_zelda_layout(
    variant: PCGBenchmarkZeldaVariant,
    reserved: Dict[str, Any],
    *,
    enemy_target: int,
    seed: int,
) -> np.ndarray:
    """
    Build a benchmark-faithful fallback layout that preserves the requested
    player->key and key->door budgets exactly.

    This intentionally sacrifices high-fidelity topology embedding in favor of
    strict control-space preservation when the richer free-routing mapper
    introduces unintended shortcuts on large variants.
    """
    content = np.full((int(variant.height), int(variant.width)), PCG_ZELDA_TILE_WALL, dtype=np.int32)
    corridor = [tuple(cell) for cell in reserved["corridor"]]
    _carve_path(content, corridor)

    start_cell = tuple(reserved["start_cell"])
    key_cell = tuple(reserved["key_cell"])
    goal_cell = tuple(reserved["goal_cell"])
    content[int(start_cell[0]), int(start_cell[1])] = PCG_ZELDA_TILE_PLAYER
    content[int(key_cell[0]), int(key_cell[1])] = PCG_ZELDA_TILE_KEY
    content[int(goal_cell[0]), int(goal_cell[1])] = PCG_ZELDA_TILE_DOOR

    eligible_enemy_cells = [cell for cell in corridor if cell not in {start_cell, key_cell, goal_cell}]
    rng = np.random.default_rng(int(seed))
    rng.shuffle(eligible_enemy_cells)
    for cell in eligible_enemy_cells[: int(max(0, enemy_target))]:
        content[int(cell[0]), int(cell[1])] = PCG_ZELDA_TILE_ENEMY

    return content


def _control_error(content_control: Dict[str, int], target_control: Dict[str, int]) -> int:
    return int(
        abs(int(content_control.get("player_key", 0)) - int(target_control.get("player_key", 0)))
        + abs(int(content_control.get("key_door", 0)) - int(target_control.get("key_door", 0)))
    )


def _route_corridor(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    *,
    blocked: Optional[Iterable[Tuple[int, int]]] = None,
) -> List[Tuple[int, int]]:
    start_cell = (int(start[0]), int(start[1]))
    goal_cell = (int(goal[0]), int(goal[1]))
    if start_cell == goal_cell:
        return [start_cell]
    blocked_cells = set((int(r), int(c)) for r, c in (blocked or []))
    blocked_cells.discard(start_cell)
    blocked_cells.discard(goal_cell)
    height, width = int(grid.shape[0]), int(grid.shape[1])
    q = deque([start_cell])
    parents: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start_cell: None}
    while q:
        cell = q.popleft()
        if cell == goal_cell:
            break
        r, c = cell
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            nxt = (int(nr), int(nc))
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if nxt in parents:
                continue
            if nxt in blocked_cells:
                continue
            parents[nxt] = cell
            q.append(nxt)
    if goal_cell not in parents:
        return []
    path: List[Tuple[int, int]] = []
    cursor: Optional[Tuple[int, int]] = goal_cell
    while cursor is not None:
        path.append(cursor)
        cursor = parents.get(cursor)
    path.reverse()
    return path


def _carve_path(grid: np.ndarray, path: Sequence[Tuple[int, int]]) -> None:
    for r, c in path:
        grid[int(r), int(c)] = PCG_ZELDA_TILE_EMPTY


def map_graph_to_pcg_benchmark_zelda(
    graph: nx.Graph,
    *,
    problem_name: Optional[str] = None,
    enemy_target: Optional[int] = None,
    seed: int = 42,
) -> PCGBenchmarkZeldaMapping:
    variant = select_pcg_benchmark_zelda_problem(graph, prefer_problem=problem_name)
    content = np.full((int(variant.height), int(variant.width)), PCG_ZELDA_TILE_WALL, dtype=np.int32)
    if graph.number_of_nodes() <= 0:
        return _invalid_mapping(
            variant,
            reason="empty_graph",
            metadata={"empty_graph": True},
        )

    start = _select_start_node(graph)
    if start is None:
        return _invalid_mapping(
            variant,
            reason="missing_explicit_start",
            metadata={"empty_graph": False},
        )
    goal = _select_goal_node(graph, start)
    if goal is None:
        return _invalid_mapping(
            variant,
            reason="missing_explicit_goal",
            metadata={"start_node": start},
        )
    key = _select_key_node(graph, start, goal)
    if key is None:
        return _invalid_mapping(
            variant,
            reason="missing_explicit_key",
            metadata={"start_node": start, "goal_node": goal},
        )
    if len({start, key, goal}) != 3:
        return _invalid_mapping(
            variant,
            reason="non_distinct_progression_nodes",
            metadata={"start_node": start, "key_node": key, "goal_node": goal},
        )
    progression_nodes = _main_progression_nodes(graph, start, key, goal)
    absorbed_nodes = set(progression_nodes) - {start, key, goal}
    raw_graph_control = _graph_path_lengths(graph, start, key, goal)
    if int(raw_graph_control.get("player_key", 0)) <= 0 or int(raw_graph_control.get("key_door", 0)) <= 0:
        return _invalid_mapping(
            variant,
            reason="invalid_progression_paths",
            metadata={
                "start_node": start,
                "key_node": key,
                "goal_node": goal,
                "graph_control_raw": dict(raw_graph_control),
            },
        )
    graph_control = _benchmark_aligned_graph_control(raw_graph_control, variant)
    reserved = _reserve_progression_path(variant, graph_control)
    reserved_corridor = [tuple(cell) for cell in reserved["corridor"]]
    reserved_cells = set(reserved_corridor)
    locked_positions = {
        start: tuple(reserved["start_cell"]),
        key: tuple(reserved["key_cell"]),
        goal: tuple(reserved["goal_cell"]),
    }

    raw_positions = _graph_positions(graph, seed=seed + stable_seed_offset((variant.name, "layout"), modulo=100000))
    mapped_positions = {node: pos for node, pos in raw_positions.items() if node not in absorbed_nodes}
    positions = _embed_positions(
        mapped_positions,
        width=int(variant.width),
        height=int(variant.height),
        locked=locked_positions,
        occupied=reserved_cells,
    )

    _carve_path(content, reserved_corridor)
    for node, cell in positions.items():
        if node in locked_positions:
            continue
        content[int(cell[0]), int(cell[1])] = PCG_ZELDA_TILE_EMPTY

    seen_pairs = set()
    protected_corridor = set(reserved_corridor[1:-1])
    for u, v in graph.to_undirected().edges():
        pair = _ordered_pair(u, v)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        if u in absorbed_nodes or v in absorbed_nodes:
            continue
        if u not in positions or v not in positions:
            continue
        if {u, v}.issubset({start, key, goal}):
            continue
        route = _route_corridor(content, positions[u], positions[v], blocked=protected_corridor)
        if len(route) <= 1:
            horizontal_first = bool(stable_seed_offset((pair[0], pair[1], seed), modulo=2))
            _carve_corridor(content, positions[u], positions[v], horizontal_first=horizontal_first)
        else:
            _carve_path(content, route)

    if start in positions:
        content[positions[start][0], positions[start][1]] = PCG_ZELDA_TILE_PLAYER
    if key in positions:
        content[positions[key][0], positions[key][1]] = PCG_ZELDA_TILE_KEY
    if goal in positions:
        content[positions[goal][0], positions[goal][1]] = PCG_ZELDA_TILE_DOOR

    enemy_nodes: List[Any] = []
    for node, attrs in graph.nodes(data=True):
        tokens = _node_tokens(dict(attrs))
        if node in {start, key, goal}:
            continue
        if {"enemy", "e", "boss", "miniboss", "mini_boss"}.intersection(tokens):
            enemy_nodes.append(node)
    requested_enemy_count = int(max(0, enemy_target if enemy_target is not None else variant.enemies))
    target_enemy_count = int(min(requested_enemy_count, variant.enemies))
    placed_enemy_cells: set[Tuple[int, int]] = set()
    player_cells = _tile_locations(content, PCG_ZELDA_TILE_PLAYER)
    player_cell = player_cells[0] if player_cells else (0, 0)
    enemy_nodes = sorted(
        enemy_nodes,
        key=lambda node: (
            -abs(positions.get(node, player_cell)[0] - player_cell[0]) - abs(positions.get(node, player_cell)[1] - player_cell[1]),
            stable_node_sort_key(node),
        ),
    )
    for node in enemy_nodes:
        if len(placed_enemy_cells) >= target_enemy_count:
            break
        cell = positions.get(node)
        if cell is None or cell in placed_enemy_cells:
            continue
        if int(content[cell[0], cell[1]]) in {PCG_ZELDA_TILE_PLAYER, PCG_ZELDA_TILE_KEY, PCG_ZELDA_TILE_DOOR}:
            continue
        content[cell[0], cell[1]] = PCG_ZELDA_TILE_ENEMY
        placed_enemy_cells.add(cell)

    if len(placed_enemy_cells) < target_enemy_count:
        free_cells = [
            (r, c)
            for r in range(content.shape[0])
            for c in range(content.shape[1])
            if int(content[r, c]) == PCG_ZELDA_TILE_EMPTY
        ]
        free_cells.sort(
            key=lambda cell: (
                -abs(cell[0] - player_cell[0]) - abs(cell[1] - player_cell[1]),
                cell[0],
                cell[1],
            )
        )
        for cell in free_cells:
            if len(placed_enemy_cells) >= target_enemy_count:
                break
            if cell in placed_enemy_cells:
                continue
            content[cell[0], cell[1]] = PCG_ZELDA_TILE_ENEMY
            placed_enemy_cells.add(cell)

    content_control = {
        "player_key": int(_grid_distance(content, PCG_ZELDA_TILE_PLAYER, PCG_ZELDA_TILE_KEY)),
        "key_door": int(_grid_distance(content, PCG_ZELDA_TILE_KEY, PCG_ZELDA_TILE_DOOR)),
    }
    initial_content_control = dict(content_control)
    mapper_mode = "free_routed"
    control_fallback_applied = False
    if variant.name == "zelda-large-v0":
        current_error = _control_error(content_control, graph_control)
        current_solution_length = int(content_control["player_key"] + content_control["key_door"])
        should_fallback = (
            current_error > int(variant.control_tolerance)
            or current_solution_length < int(variant.solution_length)
        )
        if should_fallback:
            fallback_content = _build_control_strict_zelda_layout(
                variant,
                reserved,
                enemy_target=target_enemy_count,
                seed=seed + stable_seed_offset((variant.name, "strict_fallback"), modulo=100000),
            )
            fallback_control = {
                "player_key": int(_grid_distance(fallback_content, PCG_ZELDA_TILE_PLAYER, PCG_ZELDA_TILE_KEY)),
                "key_door": int(_grid_distance(fallback_content, PCG_ZELDA_TILE_KEY, PCG_ZELDA_TILE_DOOR)),
            }
            fallback_error = _control_error(fallback_control, graph_control)
            fallback_solution_length = int(fallback_control["player_key"] + fallback_control["key_door"])
            if (
                fallback_error < current_error
                or (
                    fallback_error == current_error
                    and fallback_solution_length >= current_solution_length
                )
            ):
                content = fallback_content
                content_control = fallback_control
                mapper_mode = "corridor_fallback"
                control_fallback_applied = True

    metadata = {
        "semantic_valid": True,
        "semantic_error": "",
        "start_node": start,
        "key_node": key,
        "goal_node": goal,
        "node_positions": {str(node): [int(pos[0]), int(pos[1])] for node, pos in positions.items()},
        "enemy_target": int(target_enemy_count),
        "enemy_count": int(len(_tile_locations(content, PCG_ZELDA_TILE_ENEMY))),
        "enemy_quality_min": int(variant.enemy_quality_min),
        "enemy_quality_max": int(variant.enemy_quality_max),
        "solution_length_target": int(variant.solution_length),
        "graph_control_raw": dict(raw_graph_control),
        "graph_control_aligned": dict(graph_control),
        "content_control_initial": dict(initial_content_control),
        "content_control_final": dict(content_control),
        "mapper_mode": str(mapper_mode),
        "control_fallback_applied": bool(control_fallback_applied),
        "reserved_budget_player_key": int(reserved["player_key"]),
        "reserved_budget_key_door": int(reserved["key_door"]),
        "absorbed_main_path_nodes": [str(node) for node in sorted(absorbed_nodes, key=stable_node_sort_key)],
        "num_nodes": int(graph.number_of_nodes()),
        "num_edges": int(graph.number_of_edges()),
    }
    return PCGBenchmarkZeldaMapping(
        problem_name=variant.name,
        content=content,
        graph_control=graph_control,
        content_control=content_control,
        metadata=metadata,
    )


def import_pcg_benchmark(*, repo_path: Optional[Path] = None):
    repo_root = None if repo_path is None else Path(repo_path)
    if repo_root is not None:
        if (repo_root / "pcg_benchmark").exists():
            repo_root_str = str(repo_root.resolve())
            if repo_root_str not in sys.path:
                sys.path.insert(0, repo_root_str)
    try:
        return importlib.import_module("pcg_benchmark")
    except ImportError as exc:
        raise ImportError(
            "pcg_benchmark is not importable. Install it with "
            "`pip install git+https://github.com/amidos2006/pcg_benchmark.git` "
            "or provide --pcg-benchmark-repo pointing at a local clone."
        ) from exc


def evaluate_graphs_with_pcg_benchmark_zelda(
    graphs: Sequence[nx.Graph],
    *,
    problem_name: str,
    control_mode: str = "graph",
    repo_path: Optional[Path] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    if problem_name not in PCG_BENCHMARK_ZELDA_VARIANTS:
        raise ValueError(
            f"Unsupported PCG benchmark Zelda problem '{problem_name}'. "
            f"Valid: {sorted(PCG_BENCHMARK_ZELDA_VARIANTS.keys())}"
        )
    control_mode_key = str(control_mode).strip().lower()
    if control_mode_key not in {"graph", "content"}:
        raise ValueError("control_mode must be either 'graph' or 'content'.")

    pcb = import_pcg_benchmark(repo_path=repo_path)
    env = pcb.make(problem_name)

    mappings = [
        map_graph_to_pcg_benchmark_zelda(
            graph,
            problem_name=problem_name,
            enemy_target=PCG_BENCHMARK_ZELDA_VARIANTS[problem_name].enemies,
            seed=seed + i,
        )
        for i, graph in enumerate(graphs)
    ]
    contents = [mapping.content.tolist() for mapping in mappings]
    controls = [
        dict(mapping.graph_control if control_mode_key == "graph" else mapping.content_control)
        for mapping in mappings
    ]

    quality, diversity, controlability, details, infos = env.evaluate(contents, controls)
    rows: List[Dict[str, Any]] = []
    quality_arr = list(details.get("quality", [])) if isinstance(details, dict) else []
    diversity_arr = list(details.get("diversity", [])) if isinstance(details, dict) else []
    control_arr = list(details.get("controlability", [])) if isinstance(details, dict) else []

    for idx, mapping in enumerate(mappings):
        info = infos[idx] if idx < len(infos) else {}
        quality_value = float(quality_arr[idx]) if idx < len(quality_arr) else float(quality)
        diversity_value = float(diversity_arr[idx]) if idx < len(diversity_arr) else 0.0
        control_value = float(control_arr[idx]) if idx < len(control_arr) else float(controlability)
        control_player_key = float(controls[idx].get("player_key", 0.0))
        control_key_door = float(controls[idx].get("key_door", 0.0))
        info_player_key = float(info.get("player_key", 0.0))
        info_key_door = float(info.get("key_door", 0.0))
        enemy_count = float(info.get("enemies", 0.0))
        solution_length = float(info_player_key + info_key_door)
        solution_target = float(mapping.metadata.get("solution_length_target", 0.0))
        enemy_quality_min = float(mapping.metadata.get("enemy_quality_min", 0.0))
        enemy_quality_max = float(mapping.metadata.get("enemy_quality_max", 0.0))
        raw_graph_control = dict(mapping.metadata.get("graph_control_raw", {}))
        rows.append(
            {
                "index": int(idx),
                "problem_name": str(problem_name),
                "mapper_mode": str(mapping.metadata.get("mapper_mode", "free_routed")),
                "semantic_valid": float(bool(mapping.metadata.get("semantic_valid", True))),
                "semantic_error": str(mapping.metadata.get("semantic_error", "")),
                "control_fallback_applied": float(bool(mapping.metadata.get("control_fallback_applied", False))),
                "quality": quality_value,
                "diversity": diversity_value,
                "controlability": control_value,
                "quality_pass": float(quality_value >= 1.0),
                "diversity_pass": float(diversity_value >= 1.0),
                "controlability_pass": float(control_value >= 1.0),
                "regions": float(info.get("regions", 0.0)),
                "players": float(info.get("players", 0.0)),
                "keys": float(info.get("keys", 0.0)),
                "doors": float(info.get("doors", 0.0)),
                "enemies": enemy_count,
                "player_key_info": info_player_key,
                "key_door_info": info_key_door,
                "player_key_control": control_player_key,
                "key_door_control": control_key_door,
                "player_key_abs_error": float(abs(info_player_key - control_player_key)),
                "key_door_abs_error": float(abs(info_key_door - control_key_door)),
                "player_key_abs_error_initial": float(
                    abs(
                        float(dict(mapping.metadata.get("content_control_initial", {})).get("player_key", 0.0))
                        - control_player_key
                    )
                ),
                "key_door_abs_error_initial": float(
                    abs(
                        float(dict(mapping.metadata.get("content_control_initial", {})).get("key_door", 0.0))
                        - control_key_door
                    )
                ),
                "initial_solution_length": float(
                    float(dict(mapping.metadata.get("content_control_initial", {})).get("player_key", 0.0))
                    + float(dict(mapping.metadata.get("content_control_initial", {})).get("key_door", 0.0))
                ),
                "solution_length": solution_length,
                "solution_length_target": solution_target,
                "solution_length_pass": float(solution_length >= solution_target),
                "enemy_quality_min": enemy_quality_min,
                "enemy_quality_max": enemy_quality_max,
                "enemy_band_pass": float(enemy_quality_min <= enemy_count <= enemy_quality_max),
                "graph_player_key": float(mapping.graph_control.get("player_key", 0.0)),
                "graph_key_door": float(mapping.graph_control.get("key_door", 0.0)),
                "graph_player_key_raw": float(raw_graph_control.get("player_key", 0.0)),
                "graph_key_door_raw": float(raw_graph_control.get("key_door", 0.0)),
                "content_player_key_initial": float(
                    dict(mapping.metadata.get("content_control_initial", {})).get("player_key", 0.0)
                ),
                "content_key_door_initial": float(
                    dict(mapping.metadata.get("content_control_initial", {})).get("key_door", 0.0)
                ),
                "content_player_key": float(mapping.content_control.get("player_key", 0.0)),
                "content_key_door": float(mapping.content_control.get("key_door", 0.0)),
                "enemy_target": float(mapping.metadata.get("enemy_target", 0.0)),
                "mapped_enemy_count": float(mapping.metadata.get("enemy_count", 0.0)),
            }
        )

    return {
        "problem_name": problem_name,
        "control_mode": control_mode_key,
        "quality_mean": float(quality),
        "diversity_mean": float(diversity),
        "controlability_mean": float(controlability),
        "rows": rows,
        "mappings": mappings,
    }


__all__ = [
    "PCGBenchmarkZeldaVariant",
    "PCGBenchmarkZeldaMapping",
    "PCG_BENCHMARK_ZELDA_VARIANTS",
    "select_pcg_benchmark_zelda_problem",
    "map_graph_to_pcg_benchmark_zelda",
    "import_pcg_benchmark",
    "evaluate_graphs_with_pcg_benchmark_zelda",
]
