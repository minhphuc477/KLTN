"""Topology helper utilities extracted from gui_runner for readability and reuse."""

from __future__ import annotations

from collections import deque
import copy
import heapq
import time
from typing import Any, Optional, Tuple


def _grid_dims(grid: Any) -> Optional[Tuple[int, int]]:
    """Return (h, w) for numpy arrays or nested Python lists."""
    try:
        h, w = grid.shape
        return int(h), int(w)
    except Exception:
        try:
            h = len(grid)
            w = len(grid[0]) if h > 0 else 0
            return int(h), int(w)
        except Exception:
            return None


def _grid_get(grid: Any, r: int, c: int) -> Any:
    """Index grid cell for numpy arrays and nested lists."""
    try:
        return grid[r, c]
    except Exception:
        return grid[r][c]


def room_for_global_position(
    pos: Optional[Tuple[int, int]],
    room_positions: dict,
    room_h: int = 16,
    room_w: int = 11,
) -> Optional[Tuple[int, int]]:
    """Map a global tile coordinate to a room-grid coordinate."""
    if pos is None or not room_positions:
        return None
    pr, pc = int(pos[0]), int(pos[1])
    for room_pos, offsets in room_positions.items():
        if not offsets:
            continue
        r_off, c_off = offsets
        if r_off <= pr < r_off + room_h and c_off <= pc < c_off + room_w:
            return room_pos
    return None


def node_has_small_key(attrs: dict) -> bool:
    """Best-effort small-key detection from graph node attributes/labels."""
    if not isinstance(attrs, dict):
        return False
    if bool(attrs.get("is_boss_key") or attrs.get("has_boss_key")):
        return False
    if bool(attrs.get("is_key") or attrs.get("has_key")):
        return True
    label = str(attrs.get("label", ""))
    tokens = [tok.strip() for tok in label.replace("\n", ",").split(",") if tok.strip()]
    for tok in tokens:
        if tok == "k" or tok.lower() == "key":
            return True
    return False


def node_has_critical_content(graph: Any, node_id: Any) -> bool:
    """Whether a graph node should be preserved during dead-end pruning."""
    try:
        attrs = graph.nodes[node_id] if graph is not None and node_id in graph else {}
    except Exception:
        attrs = {}
    if not isinstance(attrs, dict):
        return False

    if attrs.get("is_start") or attrs.get("is_triforce") or attrs.get("is_boss"):
        return True
    if attrs.get("has_item") or attrs.get("is_boss_key") or attrs.get("has_boss_key"):
        return True
    if node_has_small_key(attrs):
        return True

    label = str(attrs.get("label", ""))
    tokens = {tok.strip() for tok in label.replace("\n", ",").split(",") if tok.strip()}
    critical_tokens = {
        "s",
        "t",
        "b",
        "k",
        "K",
        "I",
        "start",
        "triforce",
        "boss",
        "key",
        "boss_key",
        "item",
    }
    return len(tokens.intersection(critical_tokens)) > 0


def build_room_adjacency_from_graph(graph: Any, room_to_node: dict, node_to_room: dict) -> dict:
    """Build undirected room adjacency from graph edges via node-room mapping."""
    adjacency = {room_pos: set() for room_pos in room_to_node.keys()}
    if graph is None:
        return adjacency
    try:
        for u, v in graph.edges():
            room_u = node_to_room.get(u)
            room_v = node_to_room.get(v)
            if room_u in adjacency and room_v in adjacency and room_u != room_v:
                adjacency[room_u].add(room_v)
                adjacency[room_v].add(room_u)
    except Exception:
        return adjacency
    return adjacency


def topology_has_path(graph: Any, start_node: Any, goal_node: Any) -> bool:
    """Check start-goal connectivity in topology graph."""
    try:
        import networkx as nx

        return bool(nx.has_path(graph.to_undirected(), start_node, goal_node))
    except Exception:
        return True


def min_locked_between(graph: Any, start_node: Any, goal_node: Any) -> float:
    """Compute the minimum number of locked/key-locked edges between two nodes."""
    if graph is None:
        return float("inf")

    def edge_locked_cost(u, v) -> int:
        data = graph.get_edge_data(u, v) or graph.get_edge_data(v, u) or {}
        label = str(data.get("label", "")).strip()
        etype = str(data.get("edge_type", "")).strip().lower()
        if etype in {"locked", "key_locked"}:
            return 1
        if label == "k":
            return 1
        return 0

    dist = {start_node: 0}
    pq = [(0, start_node)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == goal_node:
            return d
        if d != dist.get(u, 10**9):
            continue
        neighbors = set(graph.successors(u)) | set(graph.predecessors(u))
        for v in neighbors:
            nd = d + edge_locked_cost(u, v)
            if nd < dist.get(v, 10**9):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return float("inf")


def walkable_grid_reachable(
    grid: Any,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    blocked_tiles: set,
    action_deltas: dict,
) -> bool:
    """Coarse walkability connectivity check on plain grids."""
    if grid is None or start is None or goal is None:
        return True

    dims = _grid_dims(grid)
    if dims is None:
        return True
    h, w = dims

    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in action_deltas.values():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in seen:
                continue
            if int(_grid_get(grid, nr, nc)) in blocked_tiles:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return False


def capture_precheck_snapshot(current: Any, reason: str = "") -> dict:
    """Capture topology state for prune undo."""
    room_to_node = dict(getattr(current, "room_to_node", {}) or {})
    node_to_room = dict(getattr(current, "node_to_room", {}) or {v: k for k, v in room_to_node.items()})
    snapshot = {
        "timestamp": time.time(),
        "reason": reason,
        "room_to_node": room_to_node,
        "node_to_room": node_to_room,
        "rooms": None,
        "graph": None,
    }
    rooms = getattr(current, "rooms", None)
    if isinstance(rooms, dict):
        try:
            snapshot["rooms"] = copy.deepcopy(rooms)
        except Exception:
            snapshot["rooms"] = dict(rooms)
    graph = getattr(current, "graph", None)
    if graph is not None:
        try:
            snapshot["graph"] = copy.deepcopy(graph)
        except Exception:
            if hasattr(graph, "copy"):
                try:
                    snapshot["graph"] = graph.copy()
                except Exception:
                    snapshot["graph"] = graph
            else:
                snapshot["graph"] = graph
    return snapshot


def update_env_topology_view(env: Any, current: Any) -> None:
    """Synchronize topology attributes from current map into active env."""
    if env is None:
        return
    env.graph = getattr(current, "graph", None)
    env.room_to_node = getattr(current, "room_to_node", None)
    env.room_positions = getattr(current, "room_positions", None)
    env.node_to_room = getattr(current, "node_to_room", None)


def room_entry_point(
    grid: Any,
    room_positions: dict,
    room_pos: Any,
    walkable: set,
    *,
    stair_tile: int = 42,
    room_h: int = 16,
    room_w: int = 11,
) -> Optional[Tuple[int, int]]:
    """Find a passable position in a room (prefer stairs, then center, then any walkable)."""
    if room_pos not in room_positions:
        return None
    dims = _grid_dims(grid)
    if dims is None:
        return None
    h, w = dims

    ry, rx = room_positions[room_pos]

    for dy in range(room_h):
        for dx in range(room_w):
            y, x = ry + dy, rx + dx
            if 0 <= y < h and 0 <= x < w and int(_grid_get(grid, y, x)) == int(stair_tile):
                return (y, x)

    cy, cx = ry + room_h // 2, rx + room_w // 2
    if 0 <= cy < h and 0 <= cx < w and int(_grid_get(grid, cy, cx)) in walkable:
        return (cy, cx)

    for dy in range(room_h):
        for dx in range(room_w):
            y, x = ry + dy, rx + dx
            if 0 <= y < h and 0 <= x < w and int(_grid_get(grid, y, x)) in walkable:
                return (y, x)
    return None


def local_bfs_4dir(
    grid: Any,
    walkable: set,
    from_pos: Tuple[int, int],
    to_pos: Tuple[int, int],
    *,
    max_steps: int = 5000,
) -> Optional[list]:
    """4-direction BFS between two tiles over walkable tile ids."""
    if from_pos == to_pos:
        return [from_pos]

    dims = _grid_dims(grid)
    if dims is None:
        return None
    h, w = dims

    visited = {from_pos}
    queue = deque([(from_pos, [from_pos])])

    while queue and len(visited) < max_steps:
        pos, path = queue.popleft()
        y, x = pos

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in visited:
                tile = int(_grid_get(grid, ny, nx))
                if tile in walkable:
                    if (ny, nx) == to_pos:
                        return path + [(ny, nx)]
                    visited.add((ny, nx))
                    queue.append(((ny, nx), path + [(ny, nx)]))
    return None
