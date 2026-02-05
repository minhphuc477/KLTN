"""Simple grid solvers (baseline A*); JPS placeholder for later benchmarking."""
import heapq
from typing import List, Tuple, Optional

Grid = List[List[int]]  # 0 = free, 1 = blocked


def neighbors_4(grid: Grid, pos: Tuple[int, int]):
    r, c = pos
    h = len(grid); w = len(grid[0])
    for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
        nr, nc = r+dr, c+dc
        if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == 0:
            yield (nr, nc)


def neighbors_8(grid: Grid, pos: Tuple[int, int], allow_corner_cutting: bool = False):
    """8-neighbor (including diagonals).

    When allow_corner_cutting is False (default), diagonal moves are allowed only if
    both adjacent orthogonals are free to prevent corner-cutting.
    """
    r, c = pos
    h = len(grid); w = len(grid[0])
    for dr, dc in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)):
        nr, nc = r+dr, c+dc
        if not (0 <= nr < h and 0 <= nc < w):
            continue
        if grid[nr][nc] != 0:
            continue
        # If diagonal, enforce corner-cut prevention unless explicitly allowed
        if abs(dr) == 1 and abs(dc) == 1 and not allow_corner_cutting:
            if grid[r+dr][c] != 0 or grid[r][c+dc] != 0:
                continue
        yield (nr, nc)


import math

def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


def octile(a: Tuple[int,int], b: Tuple[int,int]) -> float:
    dx = abs(a[0]-b[0])
    dy = abs(a[1]-b[1])
    D = 1.0
    D2 = math.sqrt(2)
    return D * (dx + dy) + (D2 - 2*D) * min(dx, dy) if False else D * (dx + dy) + (D2 - 2*D) * min(dx, dy)


def euclidean(a: Tuple[int,int], b: Tuple[int,int]) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])


def astar(grid: Grid, start: Tuple[int,int], goal: Tuple[int,int], allow_diagonal: bool = False, allow_corner_cutting: bool = False):
    """Return path, nodes_expanded. Supports optional diagonal movement & corner policy.

    When allow_diagonal=True, diagonal moves cost sqrt(2) (Euclidean edge cost) and
    heuristic uses Euclidean distance; corner-cutting is controlled by allow_corner_cutting.
    """
    open_heap = []
    h0 = euclidean(start, goal) if allow_diagonal else manhattan(start, goal)
    heapq.heappush(open_heap, (h0, 0, start, [start]))
    closed = set()
    nodes_expanded = 0

    while open_heap:
        f, g, node, path = heapq.heappop(open_heap)
        if node in closed:
            continue
        nodes_expanded += 1
        if node == goal:
            return path, nodes_expanded
        closed.add(node)
        if allow_diagonal:
            neighs = list(neighbors_8(grid, node, allow_corner_cutting=allow_corner_cutting))
        else:
            neighs = list(neighbors_4(grid, node))
        for nb in neighs:
            if nb in closed:
                continue
            cost = math.hypot(nb[0]-node[0], nb[1]-node[1])
            h = euclidean(nb, goal) if allow_diagonal else manhattan(nb, goal)
            heapq.heappush(open_heap, (g+cost+h, g+cost, nb, path+[nb]))
    return None, nodes_expanded


def _bresenham_line(a: Tuple[int,int], b: Tuple[int,int]):
    """Return list of grid points on a straight line between a and b (inclusive) using Bresenham."""
    (x0, y0) = a
    (x1, y1) = b
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    points = [(x, y)]
    if dx > dy:
        err = dx // 2
        while x != x1:
            x += sx
            err -= dy
            if err < 0:
                y += sy
                err += dx
            points.append((x, y))
    else:
        err = dy // 2
        while y != y1:
            y += sy
            err -= dx
            if err < 0:
                x += sx
                err += dy
            points.append((x, y))
    return points


def jps(grid: Grid, start: Tuple[int,int], goal: Tuple[int,int], allow_diagonal: bool = False, trace: bool = False, allow_corner_cutting: bool = False):
    """Jump Point Search with optional diagonal support, trace, and corner policy.

    Parameters:
      - allow_diagonal: enable 8-direction movement
      - trace: if True, returns (path, nodes_expanded, trace_dict)
      - allow_corner_cutting: when False (default) prevents diagonal corner cuts

    This follows canonical forced-neighbor detection based on parent direction.
    """
    if start == goal:
        return [start], 0

    h = len(grid); w = len(grid[0])

    def in_bounds(p):
        r, c = p
        return 0 <= r < h and 0 <= c < w

    def is_free(p):
        r, c = p
        return in_bounds(p) and grid[r][c] == 0

    # neighbor deltas
    if allow_diagonal:
        DELTAS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    else:
        DELTAS = [(-1,0),(1,0),(0,-1),(0,1)]

    def count_open_neighbors(p):
        r, c = p
        cnt = 0
        for dr, dc in DELTAS:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == 0:
                cnt += 1
        return cnt

    def _is_blocked(p):
        r, c = p
        return not (0 <= r < h and 0 <= c < w and grid[r][c] == 0)

    def jump(from_p, dx, dy, goal, depth=0):
        # depth: recursion guard for diagonal->orthogonal recursive checks
        MAX_DEPTH = 3
        r, c = from_p
        nr, nc = r + dx, c + dy
        # safety counter to prevent infinite loops in pathological cases
        steps = 0
        max_steps = max(h, w) * max(h, w) * 2
        while True:
            steps += 1
            if steps > max_steps:
                # reached safety limit: abandon this jump (treat as no jump)
                return None
            # Bounds and obstacle
            if _is_blocked((nr, nc)):
                return None
            cur = (nr, nc)
            if cur == goal:
                # If moving diagonally and corner-cutting is disallowed, ensure the move is legal
                if dx != 0 and dy != 0 and not allow_corner_cutting:
                    if _is_blocked((nr - dx, nc)) or _is_blocked((nr, nc - dy)):
                        return None
                return cur

            # Canonical forced-neighbor detection based on parent direction
            # Orthogonal moves
            if dx == 0 or dy == 0:
                # If dx == 0 -> moving horizontally (col changes), if dy == 0 -> moving vertically (row changes)
                if dx == 0:  # moving horizontally
                    for sd in (-1, 1):
                        lateral = (nr + sd, nc)
                        forward_lateral = (nr + sd, nc + dy)
                        if not _is_blocked(lateral) and _is_blocked(forward_lateral):
                            return cur
                else:  # moving vertically
                    for sd in (-1, 1):
                        lateral = (nr, nc + sd)
                        forward_lateral = (nr + dx, nc + sd)
                        if not _is_blocked(lateral) and _is_blocked(forward_lateral):
                            return cur
                # continue until branching (degree != 2) also signals jump point
                if count_open_neighbors(cur) != 2:
                    return cur

            # Diagonal moves
            else:
                # Prevent corner cutting unless allowed
                if not allow_corner_cutting:
                    if _is_blocked((nr - dx, nc)) or _is_blocked((nr, nc - dy)):
                        # cannot move diagonally across a corner
                        return None
                # Forced neighbors for diagonal: canonical check by testing orthogonal jumps
                # To avoid deep recursion, bound recursion depth
                if depth < MAX_DEPTH:
                    # If jumping along orthogonal directions from cur yields a jump, cur is a jump point
                    j1 = jump((nr, nc), dx, 0, goal, depth + 1)
                    j2 = jump((nr, nc), 0, dy, goal, depth + 1)
                    if j1 is not None or j2 is not None:
                        return cur

            nr += dx; nc += dy


    def pruned_successors(node, parent):
        if parent is None:
            # start: all free neighbors
            r, c = node
            res = []
            for dr, dc in DELTAS:
                nb = (r+dr, c+dc)
                if is_free(nb):
                    res.append(nb)
            return res

        dx = node[0] - parent[0]
        dy = node[1] - parent[1]
        succ = []
        # natural forward
        forward = (node[0] + dx, node[1] + dy)
        if is_free(forward):
            succ.append(forward)
        # forced neighbors: check for openings that create forced moves
        r, c = node
        if allow_diagonal:
            # Canonical diagonal pruning: include natural forward and orthogonal neighbors
            forward = (r + dx, c + dy)
            if is_free(forward):
                succ.append(forward)
            # include orthogonal steps if free (may lead to further jumps)
            orth1 = (r + dx, c)
            orth2 = (r, c + dy)
            if is_free(orth1):
                succ.append(orth1)
            if is_free(orth2):
                succ.append(orth2)
            # Forced neighbors: if moving to surrounding cells reveals a jump point, include them
            # (Use jump checks with depth guard in jump())
            try:
                if is_free(orth1) and jump((r, c), dx, 0, goal) is not None:
                    if orth1 not in succ:
                        succ.append(orth1)
                if is_free(orth2) and jump((r, c), 0, dy, goal) is not None:
                    if orth2 not in succ:
                        succ.append(orth2)
            except Exception:
                # conservative fallback: do nothing
                pass
        else:
            # orthogonal forced neighbors heuristic
            if dx == 0:
                for sd in ((0,-1),(0,1)):
                    s = (r + sd[0], c + sd[1])
                    if is_free(s) and not is_free((r + dx, c + sd[1])):
                        succ.append(s)
            elif dy == 0:
                for sd in ((-1,0),(1,0)):
                    s = (r + sd[0], c + sd[1])
                    if is_free(s) and not is_free((r + sd[0], c + dy)):
                        succ.append(s)
        return succ

    open_heap = []
    heuristic = octile if allow_diagonal else manhattan
    heapq.heappush(open_heap, (heuristic(start, goal), 0, start, [start]))
    closed = set()
    nodes_expanded = 0

    # Tracing containers
    trace_expanded = []
    trace_jumps = []
    trace_segments = []

    while open_heap:
        f, g, node, path = heapq.heappop(open_heap)
        if node in closed:
            continue
        nodes_expanded += 1
        trace_expanded.append(node)
        if node == goal:
            # expand jumps into full-step path
            full = []
            for a,b in zip(path, path[1:]):
                segment = _bresenham_line(a, b)
                if full and segment[0] == full[-1]:
                    full.extend(segment[1:])
                else:
                    full.extend(segment)
            if trace:
                return full, nodes_expanded, {'expanded': trace_expanded, 'jumps': trace_jumps, 'segments': trace_segments}
            return full, nodes_expanded
        closed.add(node)
        parent = path[-2] if len(path) >= 2 else None
        for succ in pruned_successors(node, parent):
            dx = succ[0] - node[0]
            dy = succ[1] - node[1]
            jp = jump(node, dx, dy, goal)
            if jp is None:
                # fallback: include successor if move is legal
                if is_free(succ):
                    dr = succ[0] - node[0]
                    dc = succ[1] - node[1]
                    # prevent illegal diagonal fallback when corner-cutting is disallowed
                    if abs(dr) == 1 and abs(dc) == 1 and not allow_corner_cutting:
                        if _is_blocked((node[0] + dr, node[1])) or _is_blocked((node[0], node[1] + dc)):
                            continue
                    g2 = g + euclidean(node, succ)
                    heapq.heappush(open_heap, (g2 + heuristic(succ, goal), g2, succ, path + [succ]))
                continue
            # record jump segment
            trace_jumps.append(jp)
            trace_segments.append((node, jp))
            g2 = g + euclidean(node, jp)
            heapq.heappush(open_heap, (g2 + heuristic(jp, goal), g2, jp, path + [jp]))
    # JPS did not find a path; fall back to A* (guarantees a result when one exists)
    try:
        if trace:
            path, nodes = astar(grid, start, goal, allow_diagonal=allow_diagonal)
            return path, nodes, {'expanded': trace_expanded, 'jumps': trace_jumps, 'segments': trace_segments}
        path, nodes = astar(grid, start, goal, allow_diagonal=allow_diagonal)
        return path, nodes
    except Exception:
        if trace:
            return None, nodes_expanded, {'expanded': trace_expanded, 'jumps': trace_jumps, 'segments': trace_segments}
        return None, nodes_expanded
