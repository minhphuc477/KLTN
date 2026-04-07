"""Shared graph-aware room stitching helpers for generation pipelines."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import networkx as nx
import numpy as np

from src.core.definitions import SEMANTIC_PALETTE, TileID, parse_edge_type_tokens
from src.pipeline.spatial_utils import (
    first_free_position,
    get_node_grid_position,
    stable_node_sort_key,
)

logger = logging.getLogger(__name__)

NodeSortKey = Callable[[Any], Tuple[int, Any]]
NodePositionGetter = Callable[[nx.Graph, Any], Optional[Tuple[int, int]]]
DiagnosticCallback = Callable[[str], None]
ConnectorTileResolver = Callable[[Optional[Dict[str, Any]], bool], Tuple[int, int]]


@dataclass
class StitchedRoomLayout:
    """Shared stitched-room output for grid consumers and bbox-based systems."""

    dungeon_grid: np.ndarray
    slot_positions: Dict[Any, Tuple[int, int]]
    room_offsets: Dict[Any, Tuple[int, int]]
    layout_map: Dict[Any, Tuple[int, int, int, int]]

def extract_room_grid(room_like: Any) -> np.ndarray:
    """Coerce a room-like object or raw array into a 2D int32 grid."""
    grid = getattr(room_like, "room_grid", room_like)
    arr = np.asarray(grid, dtype=np.int32)
    if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] <= 0:
        raise ValueError(f"Room grid must be a non-empty 2D array, got shape {arr.shape}.")
    return arr


def solve_component_strict_adjacency(
    comp_nodes: List[Any],
    adjacency: Dict[Any, set],
    explicit_pos: Dict[Any, Tuple[int, int]],
    *,
    sort_key: NodeSortKey = stable_node_sort_key,
) -> Dict[Any, Tuple[int, int]]:
    """Backtracking solver for a single connected component strict embedding."""
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    placement: Dict[Any, Tuple[int, int]] = {}
    occupied: set = set()

    root = comp_nodes[0]
    root_pos = explicit_pos.get(root, (0, 0))
    placement[root] = root_pos
    occupied.add(root_pos)
    search_budget = 50000

    def _neighbors_of(pos: Tuple[int, int]) -> set:
        r, c = pos
        return {(r + dr, c + dc) for dr, dc in offsets}

    def _is_adjacent(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1

    def _candidate_positions(node: Any) -> List[Tuple[int, int]]:
        if node in explicit_pos:
            return [explicit_pos[node]]

        placed_neighbors = [n for n in adjacency[node] if n in placement]
        if placed_neighbors:
            common = None
            for pn in placed_neighbors:
                neigh_cells = _neighbors_of(placement[pn])
                common = neigh_cells if common is None else (common & neigh_cells)
            candidates = sorted(common or set())
        else:
            frontier = set()
            for pos in placement.values():
                frontier |= _neighbors_of(pos)
            candidates = sorted(frontier)

        filtered = [cand for cand in candidates if cand not in occupied]
        if not filtered:
            return []

        def _score(cell: Tuple[int, int]) -> Tuple[float, int, int]:
            neigh_score = 0.0
            for pn in adjacency[node]:
                if pn in placement:
                    neigh_score += abs(cell[0] - placement[pn][0]) + abs(cell[1] - placement[pn][1])
            pref = explicit_pos.get(node)
            pref_penalty = 0.0
            if pref is not None:
                pref_penalty = abs(cell[0] - pref[0]) + abs(cell[1] - pref[1])
            return (neigh_score + (0.25 * pref_penalty), cell[0], cell[1])

        filtered.sort(key=_score)
        return filtered

    def _node_priority(node: Any) -> Tuple[int, int, Any]:
        placed_neighbors = sum(1 for n in adjacency[node] if n in placement)
        return (-placed_neighbors, -len(adjacency[node]), sort_key(node))

    def _check_partial(node: Any, pos: Tuple[int, int]) -> bool:
        if pos in occupied:
            return False
        for pn in adjacency[node]:
            if pn in placement and not _is_adjacent(pos, placement[pn]):
                return False
        return True

    def _dfs() -> bool:
        nonlocal search_budget
        if len(placement) == len(comp_nodes):
            return True
        if search_budget <= 0:
            return False
        search_budget -= 1

        unplaced = [n for n in comp_nodes if n not in placement]
        unplaced.sort(key=_node_priority)
        node = unplaced[0]

        for cand in _candidate_positions(node):
            if not _check_partial(node, cand):
                continue
            placement[node] = cand
            occupied.add(cand)
            if _dfs():
                return True
            occupied.remove(cand)
            placement.pop(node, None)
        return False

    if not _dfs():
        raise ValueError(
            "Failed strict adjacency placement for component with nodes "
            f"{comp_nodes}. Consider simplifying topology or providing explicit positions."
        )

    seen_edges = set()
    for u in comp_nodes:
        for v in adjacency[u]:
            edge_key = tuple(sorted((u, v), key=sort_key))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            if abs(placement[u][0] - placement[v][0]) + abs(placement[u][1] - placement[v][1]) != 1:
                raise ValueError(f"Strict adjacency invariant failed for edge {u!r}<->{v!r}")

    return placement


def compute_relaxed_room_placement(
    graph: nx.Graph,
    room_ids: List[Any],
    *,
    sort_key: NodeSortKey = stable_node_sort_key,
    node_position_getter: NodePositionGetter = get_node_grid_position,
    first_free_position_fn: Callable[[Tuple[int, int], set], Tuple[int, int]] = first_free_position,
) -> Dict[Any, Tuple[int, int]]:
    """Deterministic non-overlapping fallback placement when strict embedding fails."""
    nodes = [n for n in room_ids if n in graph]
    if not nodes:
        return {}

    ordered_nodes = sorted(nodes, key=sort_key)
    placement: Dict[Any, Tuple[int, int]] = {}
    occupied: set = set()
    row_cursor = 0

    for node_id in ordered_nodes:
        pos = node_position_getter(graph, node_id)
        start = (int(pos[0]), int(pos[1])) if pos is not None else (row_cursor, 0)
        resolved = first_free_position_fn(start, occupied)
        placement[node_id] = resolved
        occupied.add(resolved)
        row_cursor = max(row_cursor, resolved[0] + 1)

    return placement


def compute_strict_room_placement(
    graph: nx.Graph,
    room_ids: List[Any],
    *,
    sort_key: NodeSortKey = stable_node_sort_key,
    node_position_getter: NodePositionGetter = get_node_grid_position,
    first_free_position_fn: Callable[[Tuple[int, int], set], Tuple[int, int]] = first_free_position,
) -> Dict[Any, Tuple[int, int]]:
    """Compute a strict room placement where every graph edge is Manhattan-adjacent."""
    nodes = [n for n in room_ids if n in graph]
    if not nodes:
        return {}

    undirected = graph.to_undirected()
    node_set = set(nodes)
    adjacency: Dict[Any, set] = {n: set(undirected.neighbors(n)) & node_set for n in nodes}
    explicit_pos: Dict[Any, Tuple[int, int]] = {}
    for node_id in nodes:
        pos = node_position_getter(graph, node_id)
        if pos is not None:
            explicit_pos[node_id] = (int(pos[0]), int(pos[1]))

    max_degree = max((len(adjacency[node_id]) for node_id in nodes), default=0)
    if max_degree > 4:
        logger.warning(
            "Strict adjacency placement impossible: node degree exceeds 4 (max_degree=%s). Using relaxed placement.",
            max_degree,
        )
        return compute_relaxed_room_placement(
            graph,
            room_ids,
            sort_key=sort_key,
            node_position_getter=node_position_getter,
            first_free_position_fn=first_free_position_fn,
        )

    components = [sorted(comp, key=sort_key) for comp in nx.connected_components(undirected.subgraph(nodes))]
    components.sort(key=lambda comp: sort_key(comp[0]) if comp else (99, ""))

    placement: Dict[Any, Tuple[int, int]] = {}
    row_cursor = 0
    for comp in components:
        try:
            comp_positions = solve_component_strict_adjacency(
                comp,
                adjacency,
                explicit_pos,
                sort_key=sort_key,
            )
        except ValueError as exc:
            logger.warning(
                "Strict adjacency solver failed for component %s (%s). Using relaxed placement.",
                comp,
                exc,
            )
            return compute_relaxed_room_placement(
                graph,
                room_ids,
                sort_key=sort_key,
                node_position_getter=node_position_getter,
                first_free_position_fn=first_free_position_fn,
            )

        min_r = min(r for r, _ in comp_positions.values())
        max_r = max(r for r, _ in comp_positions.values())
        min_c = min(c for _, c in comp_positions.values())
        translated = {
            node_id: (r - min_r + row_cursor, c - min_c)
            for node_id, (r, c) in comp_positions.items()
        }
        placement.update(translated)
        row_cursor += (max_r - min_r + 2)

    return placement


def compute_graph_aware_room_slots(
    graph: nx.Graph,
    room_ids: List[Any],
    *,
    sort_key: NodeSortKey = stable_node_sort_key,
    node_position_getter: NodePositionGetter = get_node_grid_position,
    first_free_position_fn: Callable[[Tuple[int, int], set], Tuple[int, int]] = first_free_position,
) -> Dict[Any, Tuple[int, int]]:
    """Compute one normalized slot position per room id using a shared policy."""
    placement = compute_strict_room_placement(
        graph,
        room_ids,
        sort_key=sort_key,
        node_position_getter=node_position_getter,
        first_free_position_fn=first_free_position_fn,
    )

    occupied = set(placement.values())
    next_row = max((r for r, _ in occupied), default=-1) + 1
    for room_id in sorted(room_ids, key=sort_key):
        if room_id in placement:
            continue
        pos = node_position_getter(graph, room_id) if room_id in graph else None
        start = (int(pos[0]), int(pos[1])) if pos is not None else (next_row, 0)
        resolved = first_free_position_fn(start, occupied)
        placement[room_id] = resolved
        occupied.add(resolved)
        next_row = max(next_row, resolved[0] + 1)

    if not placement:
        return {}

    min_r = min(r for r, _ in placement.values())
    min_c = min(c for _, c in placement.values())
    return {
        room_id: (r - min_r, c - min_c)
        for room_id, (r, c) in placement.items()
    }


def _connector_tiles(
    edge_data: Optional[Dict[str, Any]],
    has_reverse_edge: bool,
) -> Tuple[int, int]:
    floor_id = int(SEMANTIC_PALETTE.get("FLOOR", 1))
    wall_id = int(SEMANTIC_PALETTE.get("WALL", 2))

    data = edge_data or {}
    label = str(data.get("label", "") or "")
    edge_type = str(data.get("edge_type", data.get("type", "")) or "")
    edge_tokens = set(parse_edge_type_tokens(label=label, edge_type=edge_type))

    src_tile = floor_id
    dst_tile = floor_id

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

    if (not has_reverse_edge) or {"soft_locked", "one_way", "shutter"}.intersection(edge_tokens):
        src_tile = int(SEMANTIC_PALETTE.get("DOOR_SOFT", src_tile))
        if dst_tile == wall_id:
            dst_tile = floor_id

    return src_tile, dst_tile


def _bbox_center_row_col(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x_min, y_min, x_max, y_max = bbox
    return ((y_min + y_max) // 2, (x_min + x_max) // 2)


def _wall_off_corridor_path(
    global_grid: np.ndarray,
    corridor_cells: List[Tuple[int, int]],
    *,
    fill_tile: int,
) -> None:
    """Wrap relaxed-placement corridor floor with walls so it reads as a hallway."""
    if not corridor_cells:
        return

    wall_id = int(TileID.WALL)
    corridor_set = {(int(r), int(c)) for r, c in corridor_cells}
    h, w = global_grid.shape

    for row, col in corridor_set:
        for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            next_row = row + d_row
            next_col = col + d_col
            if not (0 <= next_row < h and 0 <= next_col < w):
                continue
            if (next_row, next_col) in corridor_set:
                continue
            if int(global_grid[next_row, next_col]) != int(fill_tile):
                continue
            global_grid[next_row, next_col] = wall_id


def carve_room_connection_between_bboxes(
    global_grid: np.ndarray,
    src_bbox: Tuple[int, int, int, int],
    dst_bbox: Tuple[int, int, int, int],
    *,
    edge_data: Optional[Dict[str, Any]] = None,
    has_reverse_edge: bool = False,
    fill_tile: int = 0,
    diagnostic_callback: Optional[DiagnosticCallback] = None,
    connector_tile_resolver: Optional[ConnectorTileResolver] = None,
) -> None:
    """Carve a connection between two stitched room bounding boxes."""
    floor_id = int(SEMANTIC_PALETTE.get("FLOOR", 1))
    tile_resolver = connector_tile_resolver or _connector_tiles
    src_tile, dst_tile = tile_resolver(edge_data, has_reverse_edge)

    src_x_min, src_y_min, src_x_max, src_y_max = src_bbox
    dst_x_min, dst_y_min, dst_x_max, dst_y_max = dst_bbox

    def _stroke(center: int, low: int, high: int) -> range:
        if low > high:
            low, high = high, low
        half_span = min(2, max(0, (high - low) // 2))
        start = max(low, center - half_span)
        stop = min(high, center + half_span)
        return range(start, stop + 1)

    if src_x_max + 1 == dst_x_min or dst_x_max + 1 == src_x_min:
        row_low = max(src_y_min + 1, dst_y_min + 1)
        row_high = min(src_y_max - 1, dst_y_max - 1)
        if row_low > row_high:
            row_low = max(src_y_min, dst_y_min)
            row_high = min(src_y_max, dst_y_max)
        if row_low <= row_high:
            center = (row_low + row_high) // 2
            for row in _stroke(center, row_low, row_high):
                global_grid[row, src_x_max if src_x_max < dst_x_min else src_x_min] = src_tile
                global_grid[row, dst_x_min if src_x_max < dst_x_min else dst_x_max] = dst_tile
            return

    if src_y_max + 1 == dst_y_min or dst_y_max + 1 == src_y_min:
        col_low = max(src_x_min + 1, dst_x_min + 1)
        col_high = min(src_x_max - 1, dst_x_max - 1)
        if col_low > col_high:
            col_low = max(src_x_min, dst_x_min)
            col_high = min(src_x_max, dst_x_max)
        if col_low <= col_high:
            center = (col_low + col_high) // 2
            for col in _stroke(center, col_low, col_high):
                global_grid[src_y_max if src_y_max < dst_y_min else src_y_min, col] = src_tile
                global_grid[dst_y_min if src_y_max < dst_y_min else dst_y_max, col] = dst_tile
            return

    src_r, src_c = _bbox_center_row_col(src_bbox)
    dst_r, dst_c = _bbox_center_row_col(dst_bbox)
    horizontal = abs(dst_c - src_c) >= abs(dst_r - src_r)

    if horizontal:
        src_anchor = (src_r, src_x_max if src_c <= dst_c else src_x_min)
        dst_anchor = (dst_r, dst_x_min if src_c <= dst_c else dst_x_max)
        start = (src_anchor[0], src_anchor[1] + (1 if src_c <= dst_c else -1))
        goal = (dst_anchor[0], dst_anchor[1] - (1 if src_c <= dst_c else -1))
    else:
        src_anchor = (src_y_max if src_r <= dst_r else src_y_min, src_c)
        dst_anchor = (dst_y_min if src_r <= dst_r else dst_y_max, dst_c)
        start = (src_anchor[0] + (1 if src_r <= dst_r else -1), src_anchor[1])
        goal = (dst_anchor[0] - (1 if src_r <= dst_r else -1), dst_anchor[1])

    H, W = global_grid.shape
    if not (0 <= start[0] < H and 0 <= start[1] < W and 0 <= goal[0] < H and 0 <= goal[1] < W):
        return

    global_grid[src_anchor[0], src_anchor[1]] = src_tile
    global_grid[dst_anchor[0], dst_anchor[1]] = dst_tile

    def neighbors(cell: Tuple[int, int]):
        r, c = cell
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                yield (nr, nc)

    def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    obstacle = global_grid != int(fill_tile)
    obstacle[start[0], start[1]] = False
    obstacle[goal[0], goal[1]] = False

    open_set: List[Tuple[int, int, Tuple[int, int]]] = []
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    gscore: Dict[Tuple[int, int], int] = {start: 0}
    closed = set()
    found = False
    expansions = 0
    max_expansions = int(max(1024, 4 * H * W))

    heappush(open_set, (heuristic(start, goal), 0, start))
    while open_set and expansions < max_expansions:
        _, cost, current = heappop(open_set)
        if cost > gscore.get(current, 10**12):
            continue
        if current in closed:
            continue
        closed.add(current)
        expansions += 1
        if current == goal:
            found = True
            break
        for nb in neighbors(current):
            if nb in closed or obstacle[nb[0], nb[1]]:
                continue
            tentative = cost + 1
            if nb not in gscore or tentative < gscore[nb]:
                gscore[nb] = tentative
                heappush(open_set, (tentative + heuristic(nb, goal), tentative, nb))
                came_from[nb] = current

    if found:
        current = goal
        path = []
        while current is not None:
            path.append(current)
            current = came_from.get(current)
        path.reverse()
        carved_cells: List[Tuple[int, int]] = []
        for row, col in path:
            if global_grid[row, col] == int(fill_tile):
                global_grid[row, col] = floor_id
                carved_cells.append((int(row), int(col)))
        _wall_off_corridor_path(
            global_grid,
            carved_cells,
            fill_tile=int(fill_tile),
        )
        return

    if expansions >= max_expansions and diagnostic_callback is not None:
        diagnostic_callback("corridor_astar_iteration_cap")
    if expansions >= max_expansions:
        logger.warning(
            "Corridor A* reached expansion cap (%d) between %s and %s; using fallback carve.",
            int(max_expansions),
            str(src_bbox),
            str(dst_bbox),
        )

    sr, sc = start
    tr, tc = goal
    step_c = 1 if tc >= sc else -1
    carved_cells: List[Tuple[int, int]] = []
    for col in range(sc, tc + step_c, step_c):
        if 0 <= sr < H and 0 <= col < W and global_grid[sr, col] == int(fill_tile):
            global_grid[sr, col] = floor_id
            carved_cells.append((int(sr), int(col)))
    step_r = 1 if tr >= sr else -1
    for row in range(sr, tr + step_r, step_r):
        if 0 <= row < H and 0 <= tc < W and global_grid[row, tc] == int(fill_tile):
            global_grid[row, tc] = floor_id
            carved_cells.append((int(row), int(tc)))
    _wall_off_corridor_path(
        global_grid,
        carved_cells,
        fill_tile=int(fill_tile),
    )


def build_stitched_room_layout(
    rooms: Mapping[Any, Any],
    graph: nx.Graph,
    *,
    fill_tile: int = 0,
    sort_key: NodeSortKey = stable_node_sort_key,
    node_position_getter: NodePositionGetter = get_node_grid_position,
    first_free_position_fn: Callable[[Tuple[int, int], set], Tuple[int, int]] = first_free_position,
    enforce_room_dimensions: Optional[Tuple[int, int]] = None,
    carve_connections: bool = True,
    diagnostic_callback: Optional[DiagnosticCallback] = None,
) -> StitchedRoomLayout:
    """Build a stitched dungeon grid and bbox layout map from room grids."""
    if not rooms:
        return StitchedRoomLayout(
            dungeon_grid=np.zeros((0, 0), dtype=np.int32),
            slot_positions={},
            room_offsets={},
            layout_map={},
        )

    room_grids: Dict[Any, np.ndarray] = {}
    for room_id, room_like in rooms.items():
        grid = extract_room_grid(room_like)
        if enforce_room_dimensions is not None and tuple(grid.shape) != tuple(enforce_room_dimensions):
            raise ValueError(
                "CRITICAL: Room dimension mismatch before stitching for "
                f"room {room_id}. Expected {enforce_room_dimensions[0]}x{enforce_room_dimensions[1]}, got {grid.shape[0]}x{grid.shape[1]}."
            )
        room_grids[room_id] = grid

    slot_positions = compute_graph_aware_room_slots(
        graph,
        list(room_grids.keys()),
        sort_key=sort_key,
        node_position_getter=node_position_getter,
        first_free_position_fn=first_free_position_fn,
    )

    stitched = build_room_canvas_from_slots(
        room_grids=room_grids,
        slot_positions=slot_positions,
        fill_tile=int(fill_tile),
    )

    if carve_connections:
        for u, v in graph.edges():
            if u in stitched.layout_map and v in stitched.layout_map:
                carve_room_connection_between_bboxes(
                    stitched.dungeon_grid,
                    stitched.layout_map[u],
                    stitched.layout_map[v],
                    edge_data=graph.get_edge_data(u, v, default={}) or {},
                    has_reverse_edge=bool(graph.has_edge(v, u)),
                    fill_tile=int(fill_tile),
                    diagnostic_callback=diagnostic_callback,
                )

    return stitched


def build_room_canvas_from_slots(
    room_grids: Mapping[Any, np.ndarray],
    slot_positions: Mapping[Any, Tuple[int, int]],
    *,
    fill_tile: int = 0,
) -> StitchedRoomLayout:
    """Build a stitched canvas from precomputed slot positions."""
    if not room_grids or not slot_positions:
        return StitchedRoomLayout(
            dungeon_grid=np.zeros((0, 0), dtype=np.int32),
            slot_positions=dict(slot_positions),
            room_offsets={},
            layout_map={},
        )

    rows = sorted({row for row, _ in slot_positions.values()})
    cols = sorted({col for _, col in slot_positions.values()})
    row_to_idx = {row: idx for idx, row in enumerate(rows)}
    col_to_idx = {col: idx for idx, col in enumerate(cols)}

    row_heights = [0] * len(rows)
    col_widths = [0] * len(cols)
    for room_id, (slot_row, slot_col) in slot_positions.items():
        grid = room_grids[room_id]
        row_heights[row_to_idx[slot_row]] = max(row_heights[row_to_idx[slot_row]], int(grid.shape[0]))
        col_widths[col_to_idx[slot_col]] = max(col_widths[col_to_idx[slot_col]], int(grid.shape[1]))

    y_offsets = [0] * len(rows)
    x_offsets = [0] * len(cols)
    for idx in range(1, len(rows)):
        y_offsets[idx] = y_offsets[idx - 1] + row_heights[idx - 1]
    for idx in range(1, len(cols)):
        x_offsets[idx] = x_offsets[idx - 1] + col_widths[idx - 1]

    dungeon_grid = np.full((int(sum(row_heights)), int(sum(col_widths))), int(fill_tile), dtype=np.int32)
    room_offsets: Dict[Any, Tuple[int, int]] = {}
    layout_map: Dict[Any, Tuple[int, int, int, int]] = {}

    for room_id, grid in room_grids.items():
        slot_row, slot_col = slot_positions[room_id]
        y0 = y_offsets[row_to_idx[slot_row]]
        x0 = x_offsets[col_to_idx[slot_col]]
        h, w = int(grid.shape[0]), int(grid.shape[1])
        dungeon_grid[y0:y0 + h, x0:x0 + w] = grid
        room_offsets[room_id] = (y0, x0)
        layout_map[room_id] = (x0, y0, x0 + w - 1, y0 + h - 1)

    return StitchedRoomLayout(
        dungeon_grid=dungeon_grid,
        slot_positions=dict(slot_positions),
        room_offsets=room_offsets,
        layout_map=layout_map,
    )
