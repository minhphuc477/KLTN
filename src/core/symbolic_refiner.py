"""
H-MOLQD Block VI: Symbolic Refiner with Wave Function Collapse
==============================================================

Neuro-symbolic repair module for fixing unsolvable dungeons.

When the External Validator detects that a generated dungeon is unsolvable,
the Symbolic Refiner performs targeted repairs using Wave Function Collapse
(WFC) constrained regeneration.

Pipeline:
---------
1. PathAnalyzer: Identify where/why A* pathfinding fails
2. EntropyReset: Create mask over invalid regions (reset to high entropy)
3. WaveFunctionCollapse: Regenerate masked regions with connectivity constraints
4. ConstraintPropagation: Ensure local consistency

Mathematical Formulation:
-------------------------
WFC State: For each cell c, maintain distribution P(c = t) over tile types t

Collapse Step:
    c* = argmin_c H(P(c))  where H is entropy
    t* ~ P(c*)             sample tile type
    P(c* = t*) = 1         collapse

Propagation Step:
    For neighbors n of collapsed cell:
    P(n = t) proportional_to P(n = t) * sum_{t'} A(t, t') * P(c = t')
    where A(t, t') = adjacency compatibility

Convergence:
    Repeat until all cells collapsed or contradiction

"""

import logging
from typing import Dict, List, Tuple, Optional, Set, Any, Callable
from dataclasses import dataclass, field
from enum import IntEnum
import heapq
from collections import defaultdict, deque

import numpy as np
from src.core.definitions import SEMANTIC_PALETTE, parse_edge_type_tokens, parse_node_label_tokens

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

logger = logging.getLogger(__name__)


# ============================================================================
# TILE DEFINITIONS
# ============================================================================

class TileType(IntEnum):
    """Tile types for WFC -- aligned with canonical TileID from definitions.py."""
    VOID = int(SEMANTIC_PALETTE["VOID"])
    FLOOR = int(SEMANTIC_PALETTE["FLOOR"])
    WALL = int(SEMANTIC_PALETTE["WALL"])
    BLOCK = int(SEMANTIC_PALETTE["BLOCK"])
    DOOR_OPEN = int(SEMANTIC_PALETTE["DOOR_OPEN"])
    DOOR_LOCKED = int(SEMANTIC_PALETTE["DOOR_LOCKED"])
    DOOR_BOMB = int(SEMANTIC_PALETTE["DOOR_BOMB"])
    DOOR_PUZZLE = int(SEMANTIC_PALETTE["DOOR_PUZZLE"])
    DOOR_BOSS = int(SEMANTIC_PALETTE["DOOR_BOSS"])
    DOOR_SOFT = int(SEMANTIC_PALETTE["DOOR_SOFT"])
    ENEMY = int(SEMANTIC_PALETTE["ENEMY"])
    START = int(SEMANTIC_PALETTE["START"])
    TRIFORCE = int(SEMANTIC_PALETTE["TRIFORCE"])
    BOSS = int(SEMANTIC_PALETTE["BOSS"])
    KEY_SMALL = int(SEMANTIC_PALETTE["KEY_SMALL"])
    KEY_BOSS = int(SEMANTIC_PALETTE["KEY_BOSS"])
    KEY_ITEM = int(SEMANTIC_PALETTE["KEY_ITEM"])
    ITEM_MINOR = int(SEMANTIC_PALETTE["ITEM_MINOR"])
    ELEMENT = int(SEMANTIC_PALETTE["ELEMENT"])
    ELEMENT_FLOOR = int(SEMANTIC_PALETTE["ELEMENT_FLOOR"])
    STAIR = int(SEMANTIC_PALETTE["STAIR"])
    PUZZLE = int(SEMANTIC_PALETTE["PUZZLE"])


# Walkable tile set (canonical, derives from definitions.py TileID)
_WALKABLE_TILES: Set[int] = {
    TileType.FLOOR.value, TileType.DOOR_OPEN.value, TileType.DOOR_LOCKED.value,
    TileType.DOOR_BOMB.value, TileType.DOOR_PUZZLE.value, TileType.DOOR_BOSS.value,
    TileType.DOOR_SOFT.value, TileType.KEY_SMALL.value, TileType.KEY_BOSS.value,
    TileType.KEY_ITEM.value, TileType.ITEM_MINOR.value, TileType.ENEMY.value,
    TileType.BOSS.value, TileType.START.value, TileType.TRIFORCE.value,
    TileType.STAIR.value, TileType.ELEMENT_FLOOR.value, TileType.PUZZLE.value,
}

# Door tile IDs (all door types)
_DOOR_TILES: Set[int] = {
    TileType.DOOR_OPEN.value, TileType.DOOR_LOCKED.value, TileType.DOOR_BOMB.value,
    TileType.DOOR_PUZZLE.value, TileType.DOOR_BOSS.value, TileType.DOOR_SOFT.value,
}

_ENTITY_TILES: Set[int] = {
    TileType.KEY_SMALL.value,
    TileType.KEY_BOSS.value,
    TileType.KEY_ITEM.value,
    TileType.ITEM_MINOR.value,
    TileType.ENEMY.value,
    TileType.START.value,
    TileType.TRIFORCE.value,
    TileType.BOSS.value,
    TileType.STAIR.value,
    TileType.PUZZLE.value,
}

# Entropy reset repairs local geometry. It must not mutate topology-owned
# transitions or the global route endpoints, even when a failure mask is
# dilated across them. Graph-owned room entities are reintroduced later by
# the generation pipeline, but doors/start/goal are connectivity contracts.
_IMMUTABLE_REPAIR_TILES: Set[int] = _DOOR_TILES | {
    TileType.START.value,
    TileType.TRIFORCE.value,
    TileType.STAIR.value,
}

_SELF_ADJACENT_TILES: Set[int] = {
    TileType.VOID.value,
    TileType.FLOOR.value,
    TileType.WALL.value,
    TileType.BLOCK.value,
    TileType.ELEMENT.value,
    TileType.ELEMENT_FLOOR.value,
}

def _symmetrize_adjacency(
    adjacency: Dict[int, Set[int]],
    *,
    self_adjacent_tiles: Optional[Set[int]] = None,
) -> Dict[int, Set[int]]:
    """
    Make compatibility bidirectional for this orientation-free tile vocabulary.

    Zelda room tiles are not direction-specific (e.g. there is no "wall-facing-east"
    variant in the semantic palette), so a local adjacency relation should hold in
    both directions. Thresholding learned frequencies or hand-written tables can
    otherwise introduce order-dependent contradictions inside the greedy WFC loop.
    """
    allowed_self = set(_SELF_ADJACENT_TILES if self_adjacent_tiles is None else self_adjacent_tiles)
    symmetric: Dict[int, Set[int]] = {}
    for src, neighbors in dict(adjacency).items():
        src_i = int(src)
        values = {int(dst) for dst in neighbors}
        if src_i in allowed_self:
            values.add(src_i)
        else:
            values.discard(src_i)
        symmetric[src_i] = values

    for src, neighbors in list(symmetric.items()):
        for dst in set(neighbors):
            dst_i = int(dst)
            symmetric.setdefault(dst_i, {dst_i} if dst_i in allowed_self else set()).add(int(src))

    return {int(src): set(sorted(int(dst) for dst in neighbors)) for src, neighbors in symmetric.items()}


def _build_default_adjacency() -> Dict[int, Set[int]]:
    """
    Build a permissive fallback prior for WFC repair.

    The repair stage is a backstop for neural samples, not a style authoring tool.
    Its fallback constraints therefore bias toward preserving valid walkable/door
    geometry instead of rejecting anything that does not match a tiny hand-written
    motif table.
    """
    walkable_support = {
        TileType.FLOOR.value,
        TileType.ELEMENT_FLOOR.value,
    } | _DOOR_TILES
    floor_like = walkable_support | _ENTITY_TILES | {
        TileType.WALL.value,
        TileType.BLOCK.value,
        TileType.ELEMENT.value,
    }
    solid_like = floor_like | {
        TileType.VOID.value,
    }

    adjacency: Dict[int, Set[int]] = {
        TileType.FLOOR.value: set(floor_like),
        TileType.WALL.value: set(solid_like),
        TileType.VOID.value: {
            TileType.VOID.value,
            TileType.WALL.value,
            TileType.BLOCK.value,
        },
        TileType.BLOCK.value: set(solid_like),
        TileType.ELEMENT.value: {
            TileType.WALL.value,
            TileType.BLOCK.value,
            TileType.ELEMENT.value,
            TileType.FLOOR.value,
            TileType.ELEMENT_FLOOR.value,
        },
        TileType.ELEMENT_FLOOR.value: set(floor_like),
    }

    for door_tile in _DOOR_TILES:
        adjacency[door_tile] = set(floor_like)

    for entity_tile in _ENTITY_TILES:
        adjacency[entity_tile] = set(floor_like)

    return _symmetrize_adjacency(adjacency)


# Default adjacency rules for Zelda dungeons -- uses canonical TileID values.
# Every tile must appear as a key; tiles not listed default to empty adjacency
# (WFC will reject them during constraint propagation).
DEFAULT_ADJACENCY: Dict[int, Set[int]] = _build_default_adjacency()

CANONICAL_WALKABLE_IDS: Set[int] = {
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


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class FailurePoint:
    """Location where pathfinding failed."""
    position: Tuple[Any, Any]       # Grid failures use (row, col); graph failures may store node pairs
    failure_type: str               # 'blocked', 'missing_key', 'disconnected'
    required_item: Optional[str]    # Required item to proceed
    blocking_tiles: List[Tuple[int, int]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Extra info (e.g., room_id)


def _normalize_grid_coord(
    coord: Any,
    grid_shape: Tuple[int, int],
    *,
    field_name: str,
) -> Tuple[int, int]:
    """Normalize public room coordinates to bounded (row, col) tuples."""
    if not isinstance(coord, (tuple, list, np.ndarray)) or len(coord) < 2:
        raise ValueError(f"{field_name} must be a 2-item (row, col) coordinate, got {coord!r}.")
    try:
        row = int(coord[0])
        col = int(coord[1])
    except (TypeError, ValueError, OverflowError) as e:
        raise ValueError(
            f"{field_name} must contain integer-compatible row/col values."
        ) from e
    h, w = int(grid_shape[0]), int(grid_shape[1])
    row = max(0, min(h - 1, row))
    col = max(0, min(w - 1, col))
    return (row, col)


@dataclass
class RepairPlan:
    """Plan for repairing dungeon."""
    failure_points: List[FailurePoint]
    mask: np.ndarray                # Boolean mask of regions to regenerate
    constraints: Dict[str, Any]     # Constraints for regeneration
    priority: float                 # Repair urgency


@dataclass
class WFCState:
    """State of Wave Function Collapse."""
    grid: np.ndarray                        # H x W x num_tiles (probability)
    collapsed: np.ndarray                   # H x W (boolean)
    tile_types: List[int]                   # Available tile types
    adjacency: Dict[int, Set[int]]          # Compatibility rules
    
    def entropy_at(self, row: int, col: int) -> float:
        """Compute entropy at a row-major grid cell."""
        probs = self.grid[row, col]
        # Filter out zeros for log
        probs = probs[probs > 0]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs + 1e-10))

    def entropy_grid(self) -> np.ndarray:
        """Vectorized entropy for every grid cell."""
        probs = np.asarray(self.grid, dtype=np.float64)
        positive = probs > 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy = -np.sum(np.where(positive, probs * np.log2(probs + 1e-10), 0.0), axis=-1)
        return np.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)

    def entropy(self, x: int, y: int) -> float:
        """Deprecated compatibility wrapper; use entropy_at(row, col)."""
        return self.entropy_at(row=y, col=x)
    
    def is_collapsed(self, x: int, y: int) -> bool:
        """Check if cell is collapsed. NOTE: x=column, y=row."""
        return self.collapsed[y, x]
    
    def get_options(self, x: int, y: int) -> List[int]:
        """Get possible tile types at cell. NOTE: x=column, y=row."""
        probs = self.grid[y, x]
        return [t for t, p in zip(self.tile_types, probs) if p > 0]


# ============================================================================
# PATH ANALYZER
# ============================================================================

class PathAnalyzer:
    """
    Analyze pathfinding failures in dungeons.
    
    Identifies:
    - Where the path is blocked
    - What items are needed
    - Which regions need repair
    """
    
    def __init__(self, walkable_tiles: Optional[Set[int]] = None):
        """
        Args:
            walkable_tiles: Set of walkable tile IDs
        """
        self.walkable_tiles = walkable_tiles or _WALKABLE_TILES
        self.walkable_tiles = set(int(v) for v in self.walkable_tiles) | CANONICAL_WALKABLE_IDS

    @staticmethod
    def _node_label_tokens(node_data: Dict[str, Any]) -> Set[str]:
        tokens: Set[str] = set()
        for key in ("label", "type", "node_type", "content"):
            value = node_data.get(key)
            if value is not None:
                raw = str(value).strip()
                tokens.add(raw)
                tokens.add(raw.lower())
                tokens.update(parse_node_label_tokens(raw))
        return tokens

    @classmethod
    def _node_has_boss_key(cls, node_data: Dict[str, Any]) -> bool:
        tokens = cls._node_label_tokens(node_data)
        return (
            "K" in tokens
            or "boss_key" in tokens
            or "big_key" in tokens
            or bool(node_data.get("has_boss_key", False))
            or bool(node_data.get("is_boss_key", False))
        )

    @classmethod
    def _node_has_small_key(cls, node_data: Dict[str, Any]) -> bool:
        tokens = cls._node_label_tokens(node_data)
        return (
            not cls._node_has_boss_key(node_data)
            and (
                "k" in tokens
                or "key" in tokens
                or "small_key" in tokens
                or bool(node_data.get("has_key", False))
            )
        )

    @staticmethod
    def _edge_type_tokens(edge_data: Dict[str, Any]) -> Set[str]:
        return set(
            parse_edge_type_tokens(
                label=str(edge_data.get("label", "") or ""),
                edge_type=str(edge_data.get("edge_type", edge_data.get("type", "")) or ""),
            )
        )
    
    def analyze_grid(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        cost_map: Optional[np.ndarray] = None,
    ) -> List[FailurePoint]:
        """
        Analyze pathfinding failures in a room grid.
        
        Args:
            grid: H x W tile grid
            start: Start position as (row, col)
            goal: Goal position as (row, col)
            
        Returns:
            List of failure points
        """
        start = _normalize_grid_coord(start, grid.shape[:2], field_name="start")
        goal = _normalize_grid_coord(goal, grid.shape[:2], field_name="goal")
        failures = []
        
        # Try A* pathfinding
        path = self._astar(grid, start, goal, cost_map=cost_map)
        
        if path is not None:
            # Path exists, no failures
            return []
        
        # Find reachable region from start
        reachable = self._flood_fill(grid, start)
        
        # Find reachable region from goal
        goal_reachable = self._flood_fill(grid, goal)
        
        # Check if start and goal are disconnected
        if goal not in reachable:
            # Find boundary between regions
            boundary = self._find_boundary(grid, reachable, goal_reachable)
            
            if boundary:
                # Blocked path
                failures.append(FailurePoint(
                    position=boundary[0],
                    failure_type='disconnected',
                    required_item=None,
                    blocking_tiles=boundary,
                ))
        
        return failures
    
    def analyze_graph(
        self,
        graph: 'nx.DiGraph',
        start_node: Any,
        goal_node: Any,
    ) -> List[FailurePoint]:
        """
        Analyze pathfinding failures in dungeon graph.
        
        Args:
            graph: Dungeon connectivity graph
            start_node: Starting room node
            goal_node: Goal room node
            
        Returns:
            List of failure points
        """
        if not HAS_NETWORKX:
            logger.warning("NetworkX not available, cannot analyze graph")
            return []
        
        failures = []
        
        # Simple connectivity check
        if not nx.has_path(graph.to_undirected(), start_node, goal_node):
            # Find disconnection point
            failures.append(FailurePoint(
                position=(start_node, goal_node),
                failure_type='disconnected',
                required_item=None,
            ))
            return failures
        
        if self._inventory_path_exists(graph, start_node, goal_node):
            return []

        return self._inventory_failure_points(graph, start_node)

    def _collect_node_inventory(self, graph: 'nx.DiGraph', node: Any, keys: int, has_boss_key: bool) -> Tuple[int, bool]:
        node_data = dict(graph.nodes[node])
        if self._node_has_small_key(node_data):
            keys += 1
        if self._node_has_boss_key(node_data):
            has_boss_key = True
        return keys, has_boss_key

    def _edge_requirements(self, edge_data: Dict[str, Any]) -> Tuple[bool, bool]:
        edge_types = self._edge_type_tokens(edge_data)
        needs_key = bool(edge_types.intersection({'key_locked', 'locked', 'k'}))
        needs_boss_key = bool(edge_types.intersection({'boss_locked', 'K'}))
        return needs_key, needs_boss_key

    def _inventory_path_exists(self, graph: 'nx.DiGraph', start_node: Any, goal_node: Any) -> bool:
        start_keys, start_boss = self._collect_node_inventory(graph, start_node, 0, False)
        max_small_keys = max(0, sum(1 for node in graph.nodes if self._node_has_small_key(dict(graph.nodes[node]))))
        start_state = (start_node, min(start_keys, max_small_keys), bool(start_boss))
        queue: deque[Tuple[Any, int, bool]] = deque([start_state])
        visited: Set[Tuple[Any, int, bool]] = {start_state}

        while queue:
            node, keys, has_boss_key = queue.popleft()
            if node == goal_node:
                return True

            for next_node in graph.successors(node):
                edge_data = dict(graph.edges[node, next_node])
                needs_key, needs_boss_key = self._edge_requirements(edge_data)
                if needs_boss_key and not has_boss_key:
                    continue
                next_keys = keys
                if needs_key:
                    if next_keys <= 0:
                        continue
                    next_keys -= 1
                next_keys, next_boss = self._collect_node_inventory(graph, next_node, next_keys, has_boss_key)
                next_state = (next_node, min(next_keys, max_small_keys), bool(next_boss))
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append(next_state)

        return False

    def _inventory_failure_points(self, graph: 'nx.DiGraph', start_node: Any) -> List[FailurePoint]:
        failures: List[FailurePoint] = []
        start_keys, start_boss = self._collect_node_inventory(graph, start_node, 0, False)
        seen_failures: Set[Tuple[Any, Any, str]] = set()
        max_small_keys = max(0, sum(1 for node in graph.nodes if self._node_has_small_key(dict(graph.nodes[node]))))
        start_state = (start_node, min(start_keys, max_small_keys), bool(start_boss))
        queue: deque[Tuple[Any, int, bool]] = deque([start_state])
        visited: Set[Tuple[Any, int, bool]] = {start_state}

        while queue:
            node, keys, has_boss_key = queue.popleft()

            for next_node in graph.successors(node):
                edge_data = dict(graph.edges[node, next_node])
                needs_key, needs_boss_key = self._edge_requirements(edge_data)
                if needs_boss_key and not has_boss_key:
                    key = (node, next_node, "missing_boss_key")
                    if key not in seen_failures:
                        failures.append(FailurePoint((node, next_node), "missing_boss_key", "boss_key"))
                        seen_failures.add(key)
                    continue
                next_keys = keys
                if needs_key:
                    if next_keys <= 0:
                        key = (node, next_node, "missing_key")
                        if key not in seen_failures:
                            failures.append(FailurePoint((node, next_node), "missing_key", "key"))
                            seen_failures.add(key)
                        continue
                    next_keys -= 1
                next_keys, next_boss = self._collect_node_inventory(graph, next_node, next_keys, has_boss_key)
                next_state = (next_node, min(next_keys, max_small_keys), bool(next_boss))
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append(next_state)

        if not failures:
            failures.append(FailurePoint((start_node, None), "no_path", None))
        return failures
    
    def _astar(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        cost_map: Optional[np.ndarray] = None,
    ) -> Optional[List[Tuple[int, int]]]:
        """A* pathfinding using (row, col) coordinates with parent-pointer reconstruction."""
        h, w = grid.shape[:2]
        costs = self._normalize_cost_map(cost_map, grid.shape[:2])
        
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        def neighbors(r: int, c: int):
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] in self.walkable_tiles:
                    yield (nr, nc)
        
        # A* with parent-pointer reconstruction (O(n) memory, no path-in-heap)
        open_set = [(heuristic(start, goal), 0, start)]
        g_score = {start: 0}
        parent = {start: None}
        
        while open_set:
            _, g, current = heapq.heappop(open_set)
            
            if current == goal:
                # Reconstruct path from parent pointers
                path = []
                node = goal
                while node is not None:
                    path.append(node)
                    node = parent[node]
                return path[::-1]
            
            if g > g_score.get(current, float('inf')):
                continue
            
            for next_pos in neighbors(*current):
                step_cost = float(costs[next_pos[0], next_pos[1]]) if costs is not None else 1.0
                new_g = g + max(1.0, step_cost)
                if new_g < g_score.get(next_pos, float('inf')):
                    g_score[next_pos] = new_g
                    parent[next_pos] = current
                    new_f = new_g + heuristic(next_pos, goal)
                    heapq.heappush(open_set, (new_f, new_g, next_pos))
        
        return None

    @staticmethod
    def _normalize_cost_map(
        cost_map: Optional[np.ndarray],
        shape: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """Validate and sanitize an optional row/col cost map."""
        if not isinstance(cost_map, np.ndarray):
            return None
        if tuple(cost_map.shape[:2]) != tuple(shape):
            logger.warning(
                "Ignoring symbolic repair cost_map with shape %s; expected %s.",
                tuple(cost_map.shape),
                tuple(shape),
            )
            return None
        costs = np.asarray(cost_map, dtype=np.float32)
        if costs.ndim != 2:
            logger.warning("Ignoring symbolic repair cost_map with rank %d; expected 2.", costs.ndim)
            return None
        return np.nan_to_num(costs, nan=1.0, posinf=1e6, neginf=1.0).clip(1.0, 1e6)
    
    def _flood_fill(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
    ) -> Set[Tuple[int, int]]:
        """Flood fill to find reachable region using (row, col) coordinates."""
        h, w = grid.shape[:2]
        reachable = set()
        queue = deque([start])
        
        while queue:
            r, c = queue.popleft()
            if (r, c) in reachable:
                continue
            if not (0 <= r < h and 0 <= c < w):
                continue
            if grid[r, c] not in self.walkable_tiles:
                continue
            
            reachable.add((r, c))
            queue.extend([(r, c + 1), (r + 1, c), (r, c - 1), (r - 1, c)])
        
        return reachable
    
    def _find_boundary(
        self,
        grid: np.ndarray,
        region_a: Set[Tuple[int, int]],
        region_b: Set[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """Find boundary tiles between two regions using (row, col) coordinates."""
        h, w = grid.shape[:2]
        boundary = []
        
        for r, c in region_a:
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if (nr, nc) not in region_a and (nr, nc) not in region_b:
                        boundary.append((nr, nc))
        
        return boundary


# ============================================================================
# ENTROPY RESET
# ============================================================================

class EntropyReset:
    """
    Reset regions to high entropy for WFC regeneration.
    
    Identifies invalid regions and creates a mask for
    targeted regeneration while preserving valid structure.
    """
    
    def __init__(self, margin: int = 2):
        """
        Args:
            margin: Extra cells around failure points to reset
        """
        self.margin = margin
    
    def create_mask(
        self,
        grid_shape: Tuple[int, int],
        failure_points: List[FailurePoint],
    ) -> np.ndarray:
        """
        Create mask of regions to regenerate.
        
        Args:
            grid_shape: (height, width) of grid
            failure_points: Failure points from PathAnalyzer
            
        Returns:
            Boolean mask (True = reset, False = keep)
        """
        h, w = grid_shape
        mask = np.zeros((h, w), dtype=bool)
        
        for fp in failure_points:
            # Mark failure position
            if isinstance(fp.position, tuple) and len(fp.position) == 2:
                row, col = fp.position
                if isinstance(row, int) and isinstance(col, int):
                    self._mark_region(mask, row, col)
            
            # Mark blocking tiles
            for block_row, block_col in fp.blocking_tiles:
                self._mark_region(mask, block_row, block_col)
        
        return mask
    
    def _mark_region(self, mask: np.ndarray, center_row: int, center_col: int):
        """Mark region around a grid center point using (row, col) coordinates."""
        h, w = mask.shape
        
        for dr in range(-self.margin, self.margin + 1):
            for dc in range(-self.margin, self.margin + 1):
                row, col = center_row + dr, center_col + dc
                if 0 <= row < h and 0 <= col < w:
                    mask[row, col] = True
    
    def expand_mask(
        self,
        mask: np.ndarray,
        iterations: int = 1,
    ) -> np.ndarray:
        """Expand mask by morphological dilation."""
        if iterations <= 0:
            return mask.copy()

        from scipy.ndimage import binary_dilation

        structure = np.array(
            [
                [False, True, False],
                [True, True, True],
                [False, True, False],
            ],
            dtype=bool,
        )
        return binary_dilation(mask, structure=structure, iterations=int(iterations)).astype(bool)


# ============================================================================
# LEARNED TILE STATISTICS (Phase 3B)
# ============================================================================

class LearnedTileStatistics:
    """
    Data-driven adjacency and weight learning from training rooms.
    
    Phase 3B: Instead of relying solely on hand-crafted DEFAULT_ADJACENCY
    and fixed tile weights, this class accumulates co-occurrence statistics
    from real VGLC dungeon rooms and derives:
    
    1. Adjacency rules: Which tile pairs actually appear next to each other
       (with configurable frequency threshold to filter noise).
    2. Tile weights: Relative frequency of each tile type, used as the
       initial probability distribution in WFC.
    
    This lets the WFC generate rooms whose tile distributions and local
    patterns match the training data, improving visual coherence.
    
    Usage:
        stats = LearnedTileStatistics()
        
        # Accumulate from training data
        for room_grid in training_rooms:
            stats.observe(room_grid)
        
        # Use learned rules
        adjacency = stats.get_adjacency_rules(threshold=0.01)
        weights = stats.get_tile_weights()
        
        wfc = WaveFunctionCollapse(
            tile_types=list(weights.keys()),
            adjacency=adjacency,
            tile_weights=weights,
        )
    """
    
    def __init__(self):
        # Pair counts: (tile_a, tile_b) -> count of adjacencies
        self._pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        # Tile frequency: tile -> total occurrences
        self._tile_counts: Dict[int, int] = defaultdict(int)
        # Total adjacency observations
        self._total_pairs: int = 0
        # Total tile observations
        self._total_tiles: int = 0
    
    def observe(self, grid: np.ndarray) -> None:
        """
        Accumulate statistics from a room grid.
        
        Args:
            grid: H x W integer tile grid
        """
        h, w = grid.shape[:2]
        
        for y in range(h):
            for x in range(w):
                tile = int(grid[y, x])
                self._tile_counts[tile] += 1
                self._total_tiles += 1
                
                # Check 4-directional neighbors
                for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ny, nx_ = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx_ < w:
                        neighbor = int(grid[ny, nx_])
                        self._pair_counts[(tile, neighbor)] += 1
                        self._total_pairs += 1
    
    def observe_batch(self, grids: List[np.ndarray]) -> None:
        """Accumulate statistics from a batch of room grids."""
        for grid in grids:
            self.observe(grid)

    @property
    def total_tiles(self) -> int:
        """Total number of observed tiles."""
        return self._total_tiles
    
    def get_adjacency_rules(
        self,
        threshold: float = 0.01,
    ) -> Dict[int, Set[int]]:
        """
        Derive adjacency rules from observed co-occurrences.
        
        A tile pair (a, b) is considered compatible if:
            count(a, b) / count(a, *) >= threshold
        
        Args:
            threshold: Minimum relative frequency to allow adjacency.
                       Lower = more permissive, higher = stricter.
        
        Returns:
            {tile_id: {compatible_neighbor_ids}}
        """
        if self._total_pairs == 0:
            logger.warning("No observations yet, returning DEFAULT_ADJACENCY")
            return DEFAULT_ADJACENCY.copy()
        
        # Compute per-tile totals
        tile_totals: Dict[int, int] = defaultdict(int)
        for (a, _b), count in self._pair_counts.items():
            tile_totals[a] += count
        
        # Build adjacency
        adjacency: Dict[int, Set[int]] = defaultdict(set)
        for (a, b), count in self._pair_counts.items():
            if tile_totals[a] > 0:
                freq = count / tile_totals[a]
                if freq >= threshold:
                    adjacency[a].add(b)
        
        # Ensure every observed tile has at least self-adjacency
        for tile in self._tile_counts:
            if tile not in adjacency:
                adjacency[tile] = {tile}
        
        return _symmetrize_adjacency(dict(adjacency))
    
    def get_tile_weights(self) -> Dict[int, float]:
        """
        Derive tile weights (relative frequencies) from observations.
        
        Returns:
            {tile_id: weight} where weights sum to 1.0
        """
        if self._total_tiles == 0:
            logger.warning("No observations yet, returning uniform weights")
            return {}
        
        weights = {}
        for tile, count in self._tile_counts.items():
            weights[tile] = count / self._total_tiles
        
        return weights
    
    def get_statistics_summary(self) -> Dict[str, Any]:
        """Return summary of accumulated statistics."""
        return {
            'total_tiles_observed': self._total_tiles,
            'total_pairs_observed': self._total_pairs,
            'unique_tiles': len(self._tile_counts),
            'unique_pairs': len(self._pair_counts),
            'tile_distribution': self.get_tile_weights(),
        }
    
    def merge(self, other: 'LearnedTileStatistics') -> None:
        """Merge statistics from another LearnedTileStatistics instance."""
        for key, count in other._pair_counts.items():
            self._pair_counts[key] += count
        for key, count in other._tile_counts.items():
            self._tile_counts[key] += count
        self._total_pairs += other._total_pairs
        self._total_tiles += other._total_tiles


# ============================================================================
# WAVE FUNCTION COLLAPSE
# ============================================================================

class WaveFunctionCollapse:
    """
    Wave Function Collapse for constrained tile generation.
    
    Generates valid tile patterns by iteratively:
    1. Selecting cell with lowest entropy
    2. Collapsing to a specific tile
    3. Propagating constraints to neighbors
    
    Phase 3B: Supports learned tile weights for data-driven initial
    probability distributions (from LearnedTileStatistics).
    """
    
    def __init__(
        self,
        tile_types: List[int],
        adjacency: Optional[Dict[int, Set[int]]] = None,
        tile_weights: Optional[Dict[int, float]] = None,
        max_iterations: int = 10000,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Args:
            tile_types: List of available tile type IDs
            adjacency: Compatibility rules {tile: {compatible_neighbors}}
            tile_weights: Per-tile initial weights (Phase 3B). If None, uniform.
            max_iterations: Maximum collapse iterations
        """
        self.tile_types = tile_types
        self.adjacency = adjacency or DEFAULT_ADJACENCY
        self.max_iterations = max_iterations
        self.rng = rng if rng is not None else np.random.default_rng()
        
        # Phase 3B: Learned tile weights for non-uniform initialization
        if tile_weights is not None:
            # Build weight array in tile_types order
            total = sum(tile_weights.get(t, 1e-6) for t in tile_types)
            self.initial_probs = np.array([
                tile_weights.get(t, 1e-6) / total for t in tile_types
            ])
        else:
            self.initial_probs = np.ones(len(tile_types)) / len(tile_types)
    
    def initialize_state(
        self,
        height: int,
        width: int,
        initial_grid: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ) -> WFCState:
        """
        Initialize WFC state.
        
        Args:
            height: Grid height
            width: Grid width
            initial_grid: Optional initial tile grid
            mask: Optional mask (True = uncollapsed, False = keep from initial)
            
        Returns:
            Initial WFC state
        """
        # Phase 3B: Use learned tile weights instead of uniform
        grid = np.tile(self.initial_probs, (height, width, 1))  # [H, W, T]
        collapsed = np.zeros((height, width), dtype=bool)
        
        # Apply initial grid where mask is False
        if initial_grid is not None and mask is not None:
            for y in range(height):
                for x in range(width):
                    if not mask[y, x]:
                        # Keep initial tile
                        tile = initial_grid[y, x]
                        if tile in self.tile_types:
                            tile_idx = self.tile_types.index(tile)
                            grid[y, x] = 0.0
                            grid[y, x, tile_idx] = 1.0
                            collapsed[y, x] = True
        
        return WFCState(
            grid=grid,
            collapsed=collapsed,
            tile_types=self.tile_types,
            adjacency=self.adjacency,
        )
    
    def collapse(self, state: WFCState) -> Tuple[np.ndarray, bool]:
        """
        Run WFC to completion.
        
        Args:
            state: Initial WFC state
            
        Returns:
            (result_grid, success) - Collapsed grid and success flag
        """
        h, w = state.collapsed.shape
        
        for _iteration in range(self.max_iterations):
            # Find cell with lowest entropy (that isn't collapsed)
            entropy = state.entropy_grid()
            entropy = np.where(state.collapsed, np.inf, entropy)
            if not np.isfinite(entropy).any():
                min_cell = None
            else:
                flat_idx = int(np.argmin(entropy))
                y, x = np.unravel_index(flat_idx, entropy.shape)
                min_cell = (int(x), int(y))
            
            if min_cell is None:
                # All cells collapsed
                break
            
            options = state.get_options(*min_cell)
            if not options:
                logger.warning(f"WFC contradiction at {min_cell}")
                return self._extract_grid(state), False
            
            # Collapse the cell
            x, y = min_cell
            success = self._collapse_cell(state, x, y)
            
            if not success:
                return self._extract_grid(state), False
            
            # Propagate constraints - returns False on contradiction.
            if not self._propagate(state, x, y):
                logger.warning(f"WFC contradiction detected during propagation after collapsing {min_cell}")
                return self._extract_grid(state), False
        
        return self._extract_grid(state), True
    
    def _collapse_cell(self, state: WFCState, x: int, y: int) -> bool:
        """Collapse a single cell to a tile type."""
        probs = state.grid[y, x].copy()
        
        # Get valid options
        options = [t for t, p in zip(state.tile_types, probs) if p > 0]
        
        if not options:
            return False
        
        # Sample tile weighted by probability
        valid_probs = probs[probs > 0]
        valid_probs /= valid_probs.sum()
        
        tile_idx = self.rng.choice(
            len(options),
            p=valid_probs,
        )
        tile = options[tile_idx]
        
        # Collapse
        state.grid[y, x] = 0.0
        full_idx = state.tile_types.index(tile)
        state.grid[y, x, full_idx] = 1.0
        state.collapsed[y, x] = True
        
        return True
    
    def _propagate(self, state: WFCState, x: int, y: int) -> bool:
        """Propagate constraints from a collapsed/changed cell using a worklist (AC-3 style).

        Starts with the 4 neighbours of (x, y) and continues until no further
        domain reductions are possible.  Returns ``True`` if the grid remains
        consistent, ``False`` if any cell's domain was reduced to zero (contradiction).

        Parameters
        ----------
        state:
            Current WFC state (modified in-place).
        x, y:
            Grid coordinates of the cell that was just collapsed or changed.
        """
        h, w = state.collapsed.shape

        # Worklist holds (cx, cy) cells whose domain was just reduced and whose
        # neighbours may need to be re-evaluated.
        worklist = deque()

        # Seed the worklist with the 4 neighbours of the starting cell.
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx_, ny_ = x + dx, y + dy
            if 0 <= nx_ < w and 0 <= ny_ < h and not state.is_collapsed(nx_, ny_):
                worklist.append((nx_, ny_))

        # Track which cells are already queued to avoid redundant processing.
        queued: Set[Tuple[int, int]] = set(worklist)

        while worklist:
            cx, cy = worklist.popleft()
            queued.discard((cx, cy))

            if state.is_collapsed(cx, cy):
                continue

            # Build the union of all tiles that at least ONE neighbour permits here.
            allowed: Optional[Set[int]] = None

            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx_, ny_ = cx + dx, cy + dy
                if not (0 <= nx_ < w and 0 <= ny_ < h):
                    continue

                # Determine the set of tiles that the neighbour at (nx_, ny_) is
                # compatible with.  For each tile currently possible at that
                # neighbour, gather the tiles it allows adjacent.
                neighbour_allows: Set[int] = set()
                for i, t in enumerate(state.tile_types):
                    if state.grid[ny_, nx_, i] > 0:
                        neighbour_allows |= self.adjacency.get(t, set())

                if allowed is None:
                    allowed = neighbour_allows
                else:
                    allowed &= neighbour_allows

            if allowed is None:
                # No neighbours at all (edge cell with all collapsed neighbours) -
                # keep all current options.
                continue

            # Restrict this cell's domain to the intersection of neighbour permissions.
            changed = False
            for i, t in enumerate(state.tile_types):
                if state.grid[cy, cx, i] > 0 and t not in allowed:
                    state.grid[cy, cx, i] = 0.0
                    changed = True

            if not changed:
                continue

            # Renormalise.
            total = state.grid[cy, cx].sum()
            if total <= 0:
                # Contradiction - this cell has no remaining valid tile.
                return False
            state.grid[cy, cx] /= total

            # Cascade: add this cell's uncollapsed neighbours to the worklist.
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx_, ny_ = cx + dx, cy + dy
                key = (nx_, ny_)
                if (0 <= nx_ < w and 0 <= ny_ < h
                        and not state.is_collapsed(nx_, ny_)
                        and key not in queued):
                    worklist.append(key)
                    queued.add(key)

        return True  # No contradiction found
    
    def _extract_grid(self, state: WFCState) -> np.ndarray:
        """Extract final tile grid from state."""
        h, w = state.collapsed.shape
        grid = np.zeros((h, w), dtype=int)
        
        for y in range(h):
            for x in range(w):
                tile_idx = np.argmax(state.grid[y, x])
                grid[y, x] = state.tile_types[tile_idx]
        
        return grid


# ============================================================================
# CONSTRAINT PROPAGATION
# ============================================================================

class ConstraintPropagator:
    """
    Arc consistency constraint propagation.
    
    Ensures local consistency after WFC to fix edge cases.
    """
    
    def __init__(
        self,
        adjacency: Optional[Dict[int, Set[int]]] = None,
    ):
        self.adjacency = adjacency or DEFAULT_ADJACENCY
    
    def enforce_connectivity(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        walkable: Set[int],
        required_floor_mask: Optional[np.ndarray] = None,
        cost_map: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Ensure start-goal connectivity.
        
        Creates a path if none exists. Coordinates use (row, col).
        """
        start = _normalize_grid_coord(start, grid.shape[:2], field_name="start")
        goal = _normalize_grid_coord(goal, grid.shape[:2], field_name="goal")
        
        result = grid.copy()
        costs = PathAnalyzer._normalize_cost_map(cost_map, result.shape)

        if isinstance(required_floor_mask, np.ndarray) and required_floor_mask.shape == result.shape:
            floor_id = int(SEMANTIC_PALETTE.get("FLOOR", TileType.FLOOR.value))
            constrained = required_floor_mask.astype(bool, copy=False)
            needs_floor = constrained & ~np.isin(result, list(walkable))
            result[needs_floor] = floor_id

        # Check existing path
        path = self._find_path(result, start, goal, walkable)

        if path is not None:
            return result

        # Create a path using a continuous cost field. When LogicNet or the
        # caller provides repair costs those dominate; otherwise derive a
        # soft reachability field from existing walkable structure so the
        # rescue path follows room context instead of drawing a fixed L shape.
        if costs is None:
            costs = self._derive_soft_carve_cost_map(result, start, goal, walkable)
        carve_path = self._cost_guided_carve(result, start, goal, walkable, costs)
        if not carve_path:
            logger.debug("Cost-guided connectivity carve failed; leaving grid unchanged.")
            return result

        for r, c in carve_path:
            if result[r, c] not in walkable:
                result[r, c] = TileType.FLOOR.value

        if isinstance(required_floor_mask, np.ndarray) and required_floor_mask.shape == result.shape:
            floor_id = int(SEMANTIC_PALETTE.get("FLOOR", TileType.FLOOR.value))
            constrained = required_floor_mask.astype(bool, copy=False)
            needs_floor = constrained & ~np.isin(result, list(walkable))
            result[needs_floor] = floor_id
        return result

    def _derive_soft_carve_cost_map(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        walkable: Set[int],
    ) -> np.ndarray:
        """
        Build a lightweight continuous carve prior from the current room.

        This approximates the useful behavior of a soft pathfinder without
        requiring a neural module inside symbolic repair: existing walkable
        cells are cheap, walls near existing walkable structure are cheaper
        than isolated walls, and start/goal are explicitly anchored.
        """
        h, w = grid.shape[:2]
        walkable_mask = np.isin(grid, list(walkable))
        sr, sc = _normalize_grid_coord(start, grid.shape[:2], field_name="start")
        gr, gc = _normalize_grid_coord(goal, grid.shape[:2], field_name="goal")
        walkable_mask[sr, sc] = True
        walkable_mask[gr, gc] = True

        distance = np.full((h, w), np.inf, dtype=np.float32)
        queue: deque[Tuple[int, int]] = deque()
        for r, c in np.argwhere(walkable_mask):
            rr, cc = int(r), int(c)
            distance[rr, cc] = 0.0
            queue.append((rr, cc))

        while queue:
            r, c = queue.popleft()
            next_dist = float(distance[r, c]) + 1.0
            for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if next_dist < float(distance[nr, nc]):
                    distance[nr, nc] = next_dist
                    queue.append((nr, nc))

        finite = np.isfinite(distance)
        if np.any(finite):
            max_dist = max(float(distance[finite].max()), 1.0)
            distance = np.where(finite, distance / max_dist, 1.0).astype(np.float32)
        else:
            distance = np.ones((h, w), dtype=np.float32)

        costs = np.where(walkable_mask, 0.20, 1.00 + 0.75 * distance).astype(np.float32)
        if h > 2 and w > 2:
            costs[0, :] += 0.35
            costs[-1, :] += 0.35
            costs[:, 0] += 0.35
            costs[:, -1] += 0.35
            costs[sr, sc] = 0.05
            costs[gr, gc] = 0.05
        return costs

    def _l_shape_path(
        self,
        r0: int,
        c0: int,
        r1: int,
        c1: int,
    ) -> List[Tuple[int, int]]:
        """Historical columns-then-rows carving fallback."""
        path: List[Tuple[int, int]] = []
        r, c = int(r0), int(c0)
        path.append((r, c))
        while c != c1:
            c += 1 if c1 > c else -1
            path.append((r, c))
        while r != r1:
            r += 1 if r1 > r else -1
            path.append((r, c))
        return path

    def _cost_guided_carve(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        walkable: Set[int],
        cost_map: Optional[np.ndarray],
    ) -> List[Tuple[int, int]]:
        """
        Dijkstra over all cells using repair costs.

        Unlike _find_path, this may traverse walls because its purpose is to
        choose which wall cells should be carved. Existing walkable cells get a
        small discount; non-walkable cells follow the provided LogicNet cost.
        """
        if cost_map is None:
            return []
        h, w = grid.shape[:2]
        start = _normalize_grid_coord(start, grid.shape[:2], field_name="start")
        goal = _normalize_grid_coord(goal, grid.shape[:2], field_name="goal")
        dist: Dict[Tuple[int, int], float] = {start: 0.0}
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        queue: List[Tuple[float, Tuple[int, int]]] = [(0.0, start)]

        while queue:
            cur_dist, (r, c) = heapq.heappop(queue)
            if cur_dist > dist.get((r, c), float("inf")):
                continue
            if (r, c) == goal:
                path: List[Tuple[int, int]] = []
                node: Optional[Tuple[int, int]] = goal
                while node is not None:
                    path.append(node)
                    node = parent[node]
                return list(reversed(path))

            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                step_cost = float(cost_map[nr, nc])
                if int(grid[nr, nc]) in walkable:
                    step_cost *= 0.25
                new_dist = cur_dist + max(1e-6, step_cost)
                if new_dist < dist.get((nr, nc), float("inf")):
                    dist[(nr, nc)] = new_dist
                    parent[(nr, nc)] = (r, c)
                    heapq.heappush(queue, (new_dist, (nr, nc)))
        return []

    def _find_path(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        walkable: Set[int],
    ) -> Optional[List[Tuple[int, int]]]:
        """BFS pathfinding using (row, col) coordinates."""
        h, w = grid.shape
        
        queue = deque([start])
        visited = {start}
        parent = {start: None}
        
        while queue:
            r, c = queue.popleft()
            
            if (r, c) == goal:
                # Reconstruct path
                path = []
                current = goal
                while current is not None:
                    path.append(current)
                    current = parent[current]
                return list(reversed(path))
            
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in visited:
                    continue
                if grid[nr, nc] not in walkable:
                    continue
                    
                visited.add((nr, nc))
                parent[(nr, nc)] = (r, c)
                queue.append((nr, nc))
        
        return None


# ============================================================================
# SYMBOLIC REFINER (Main Interface)
# ============================================================================

class SymbolicRefiner:
    """
    H-MOLQD Block VI: Symbolic Refiner.
    
    Neuro-symbolic repair module that fixes unsolvable dungeons
    using Wave Function Collapse regeneration.
    
    Pipeline:
    1. Analyze failures with PathAnalyzer
    2. Create reset mask with EntropyReset
    3. Regenerate with WaveFunctionCollapse
    4. Enforce constraints with ConstraintPropagator
    
    Usage:
        refiner = SymbolicRefiner()
        
        # Repair a room grid
        fixed_grid = refiner.repair_room(
            grid=room_grid,
            start=(0, 5),
            goal=(15, 5),
        )
        
        # Repair a dungeon graph
        fixed_dungeon = refiner.repair_dungeon(
            dungeon=dungeon,
            validator=external_validator,
        )
    """
    
    def __init__(
        self,
        tile_types: Optional[List[int]] = None,
        adjacency: Optional[Dict[int, Set[int]]] = None,
        tile_weights: Optional[Dict[int, float]] = None,
        learned_stats: Optional[LearnedTileStatistics] = None,
        max_repair_attempts: int = 5,
        margin: int = 2,
        adjacency_threshold: float = 0.01,
        seed: Optional[int] = None,
    ):
        """
        Args:
            tile_types: Available tile types for WFC
            adjacency: Tile adjacency rules (overrides learned)
            tile_weights: Per-tile WFC weights (overrides learned)
            learned_stats: LearnedTileStatistics for data-driven rules (Phase 3B)
            max_repair_attempts: Maximum repair iterations
            margin: Extra cells around failures to reset
            adjacency_threshold: Threshold for learned adjacency rules
        """
        # Default tile types
        if tile_types is None:
            tile_types = [t.value for t in TileType]
        
        self.tile_types = tile_types
        self.max_repair_attempts = max_repair_attempts
        self.learned_stats = learned_stats
        self.adjacency_threshold = float(adjacency_threshold)
        self._rng_seed = None if seed is None else int(seed)
        self._rng = np.random.default_rng(self._rng_seed)
        self._adjacency_override = (
            {int(src): set(int(dst) for dst in neighbors) for src, neighbors in adjacency.items()}
            if adjacency is not None
            else None
        )
        self._tile_weights_override = dict(tile_weights) if tile_weights is not None else None
        
        # Components
        self.path_analyzer = PathAnalyzer()
        self.entropy_reset = EntropyReset(margin=margin)
        self.adjacency = DEFAULT_ADJACENCY
        self.wfc = WaveFunctionCollapse(tile_types=tile_types)
        self.constraint_propagator = ConstraintPropagator()
        self.last_repair_diagnostics: Dict[str, Any] = {}
        self.refresh_learned_rules()

    def set_seed(self, seed: Optional[int]) -> None:
        """Reset WFC sampling RNG for reproducible repair runs."""
        self._rng_seed = None if seed is None else int(seed)
        self._rng = np.random.default_rng(self._rng_seed)
        if hasattr(self, "wfc"):
            self.wfc.rng = self._rng

    def _resolve_effective_rules(self) -> Tuple[Dict[int, Set[int]], Optional[Dict[int, float]]]:
        """Resolve the active adjacency/weight configuration for WFC repair."""
        effective_adjacency = self._adjacency_override
        effective_weights = self._tile_weights_override

        total_tiles = self.learned_stats.total_tiles if self.learned_stats is not None else 0
        if self.learned_stats is not None and total_tiles > 0:
            if effective_adjacency is None:
                effective_adjacency = self.learned_stats.get_adjacency_rules(
                    threshold=self.adjacency_threshold
                )
                logger.info(
                    "Using learned adjacency rules: %d tile types from %d observations",
                    len(effective_adjacency),
                    total_tiles,
                )
            if effective_weights is None:
                effective_weights = self.learned_stats.get_tile_weights()
                logger.info("Using learned tile weights from training data")

        if effective_adjacency is None:
            effective_adjacency = DEFAULT_ADJACENCY

        return effective_adjacency, effective_weights

    def refresh_learned_rules(self) -> None:
        """
        Rebuild WFC components from the current rule source.

        This allows a refiner created with empty learned statistics to start with
        safe defaults and later switch over once observations have been populated.
        """
        effective_adjacency, effective_weights = self._resolve_effective_rules()
        self.adjacency = effective_adjacency
        self.wfc = WaveFunctionCollapse(
            tile_types=self.tile_types,
            adjacency=effective_adjacency,
            tile_weights=effective_weights,
            rng=self._rng,
        )
        self.constraint_propagator = ConstraintPropagator(
            adjacency=effective_adjacency
        )

    def repair_room_with_feedback(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        feedback_callback: Optional[Callable[[np.ndarray, np.ndarray, Tuple[int, int], Tuple[int, int], int], np.ndarray]] = None,
        max_feedback_rounds: int = 2,
        required_floor_mask: Optional[np.ndarray] = None,
        cost_map: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """
        Repair a room and optionally call a neural feedback callback on WFC dead-ends.

        The callback receives (current_grid, dead_end_mask, start, goal, attempt)
        and must return a patched room grid of identical shape.
        """
        if seed is not None:
            self.set_seed(seed)
        self.refresh_learned_rules()
        current_grid = grid.copy()
        immutable_source_grid = np.asarray(grid).copy()
        immutable_source_mask = np.isin(
            immutable_source_grid,
            np.fromiter(_IMMUTABLE_REPAIR_TILES, dtype=np.int32),
        )
        start = _normalize_grid_coord(start, current_grid.shape[:2], field_name="start")
        goal = _normalize_grid_coord(goal, current_grid.shape[:2], field_name="goal")
        feedback_used = 0
        floor_mask = None
        if isinstance(required_floor_mask, np.ndarray) and required_floor_mask.shape == current_grid.shape:
            floor_mask = required_floor_mask.astype(bool, copy=False)
        repair_costs = PathAnalyzer._normalize_cost_map(cost_map, current_grid.shape)
        floor_id = int(SEMANTIC_PALETTE["FLOOR"])

        def _apply_required_floor_constraints(grid_in: np.ndarray) -> np.ndarray:
            constrained_grid = np.asarray(grid_in).copy()
            if floor_mask is not None:
                preserved_walkable_ids = np.array(
                    sorted(
                        CANONICAL_WALKABLE_IDS
                        | {
                            int(TileType.KEY_SMALL.value),
                            int(TileType.ITEM_MINOR.value),
                            int(TileType.ENEMY.value),
                        }
                    ),
                    dtype=np.int32,
                )
                non_floorable = np.isin(constrained_grid, preserved_walkable_ids)
                force_floor = floor_mask & (~non_floorable)
                constrained_grid[force_floor] = floor_id

            # WFC respects the reset mask, but an external neural feedback
            # callback receives the full grid and may accidentally write beyond
            # it. Restore topology-owned cells after every transformation so a
            # callback, connectivity carve, or later repair iteration cannot
            # erase original doors, stairs, start, or goal anchors.
            constrained_grid[immutable_source_mask] = immutable_source_grid[immutable_source_mask]
            return constrained_grid

        current_grid = _apply_required_floor_constraints(current_grid)
        diagnostics: Dict[str, Any] = {
            "attempts": 0,
            "wfc_failures": 0,
            "feedback_rounds": 0,
            "feedback_applied": 0,
            "last_dead_end_mask_pixels": 0,
            "required_floor_pixels": int(np.sum(floor_mask)) if floor_mask is not None else 0,
            "cost_guidance_used": bool(repair_costs is not None),
            "final_failure_count": 0,
        }

        for attempt in range(self.max_repair_attempts):
            diagnostics["attempts"] = int(attempt + 1)

            # Analyze failures
            failures = self.path_analyzer.analyze_grid(current_grid, start, goal, cost_map=repair_costs)

            if not failures:
                if floor_mask is not None:
                    walkable = {
                        TileType.FLOOR.value,
                        TileType.KEY_SMALL.value,
                        TileType.ITEM_MINOR.value,
                    } | CANONICAL_WALKABLE_IDS
                    current_grid = self.constraint_propagator.enforce_connectivity(
                        current_grid,
                        start,
                        goal,
                        walkable,
                        required_floor_mask=floor_mask,
                        cost_map=repair_costs,
                    )
                logger.info(f"Room repaired successfully in {attempt + 1} attempts")
                diagnostics["final_failure_count"] = 0
                self.last_repair_diagnostics = diagnostics
                return current_grid, True, diagnostics

            logger.debug(f"Repair attempt {attempt + 1}: {len(failures)} failure points")

            # Create localized reset mask around failure points.
            mask = self.entropy_reset.create_mask(grid.shape[:2], failures)
            mask = self.entropy_reset.expand_mask(mask, iterations=1)
            immutable_mask = np.isin(
                current_grid,
                np.fromiter(_IMMUTABLE_REPAIR_TILES, dtype=np.int32),
            )
            mask = mask & (~immutable_mask) & (~immutable_source_mask)
            if floor_mask is not None:
                mask = mask & (~floor_mask)
            current_grid = _apply_required_floor_constraints(current_grid)

            # Run local WFC repair for masked region.
            state = self.wfc.initialize_state(
                height=grid.shape[0],
                width=grid.shape[1],
                initial_grid=current_grid,
                mask=mask,
            )
            current_grid, wfc_success = self.wfc.collapse(state)
            current_grid = _apply_required_floor_constraints(current_grid)

            if not wfc_success:
                diagnostics["wfc_failures"] = int(diagnostics["wfc_failures"] + 1)
                logger.warning(f"WFC failed on attempt {attempt + 1}")

                # WFC-guided feedback loop: ask upstream neural module to inpaint
                # only the contradiction region, then continue repair iterations.
                if feedback_callback is not None and feedback_used < int(max(0, max_feedback_rounds)):
                    try:
                        feedback_grid = feedback_callback(
                            current_grid.copy(),
                            mask.copy(),
                            start,
                            goal,
                            int(attempt),
                        )
                        if isinstance(feedback_grid, np.ndarray) and feedback_grid.shape == current_grid.shape:
                            current_grid = _apply_required_floor_constraints(
                                feedback_grid.astype(current_grid.dtype, copy=False)
                            )
                            feedback_used += 1
                            diagnostics["feedback_rounds"] = int(feedback_used)
                            diagnostics["feedback_applied"] = int(diagnostics["feedback_applied"] + 1)
                            diagnostics["last_dead_end_mask_pixels"] = int(np.sum(mask))
                            continue
                    except (AttributeError, RuntimeError, ValueError, TypeError) as e:
                        logger.warning("Feedback callback failed: %s", e)
                continue

            walkable = {
                TileType.FLOOR.value,
                TileType.KEY_SMALL.value,
                TileType.ITEM_MINOR.value,
            } | CANONICAL_WALKABLE_IDS
            current_grid = self.constraint_propagator.enforce_connectivity(
                current_grid,
                start,
                goal,
                walkable,
                required_floor_mask=floor_mask,
                cost_map=repair_costs,
            )
            current_grid = _apply_required_floor_constraints(current_grid)

        # Final check.
        failures = self.path_analyzer.analyze_grid(current_grid, start, goal, cost_map=repair_costs)
        diagnostics["final_failure_count"] = int(len(failures))
        self.last_repair_diagnostics = diagnostics
        return current_grid, len(failures) == 0, diagnostics
    
    def repair_room(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        *,
        required_floor_mask: Optional[np.ndarray] = None,
        cost_map: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, bool]:
        """
        Repair a room grid to ensure solvability.
        
        Args:
            grid: H x W tile grid
            start: Start position as (row, col)
            goal: Goal position as (row, col)
            
        Returns:
            (repaired_grid, success)
        """
        repaired, success, diagnostics = self.repair_room_with_feedback(
            grid=grid,
            start=start,
            goal=goal,
            feedback_callback=None,
            max_feedback_rounds=0,
            required_floor_mask=required_floor_mask,
            cost_map=cost_map,
            seed=seed,
        )
        self.last_repair_diagnostics = diagnostics
        return repaired, success
    
    def repair_dungeon(
        self,
        dungeon: Any,
        validator: Optional[Any] = None,
    ) -> Tuple[Any, bool]:
        """
        Repair a full dungeon.
        
        Args:
            dungeon: Dungeon object with rooms and graph
            validator: Optional ExternalValidator for solvability check
            
        Returns:
            (repaired_dungeon, success)
        """
        # Import here to avoid circular dependency
        try:
            from src.evaluation.validator import ExternalValidator
            if validator is None:
                validator = ExternalValidator()
        except ImportError:
            logger.warning("ExternalValidator not available")
            return dungeon, False
        
        # Check if already solvable
        result = validator.validate(dungeon)
        if result.is_solvable:
            return dungeon, True
        
        # Try to repair each unsolvable room
        if hasattr(dungeon, 'rooms'):
            for _room_id, room in enumerate(dungeon.rooms):
                if hasattr(room, 'grid'):
                    # Find start/goal for this room
                    h, w = room.grid.shape[:2]
                    start = (0, w // 2)
                    goal = (h - 1, w // 2)
                    
                    repaired_grid, room_success = self.repair_room(
                        room.grid, start, goal
                    )
                    
                    if room_success:
                        room.grid = repaired_grid
        
        # Revalidate
        result = validator.validate(dungeon)
        return dungeon, result.is_solvable
    
    def analyze_failures(
        self,
        dungeon: Any,
    ) -> List[FailurePoint]:
        """
        Analyze failure points in a dungeon.
        
        Args:
            dungeon: Dungeon to analyze
            
        Returns:
            List of failure points
        """
        all_failures = []
        
        # Analyze graph-level failures
        if hasattr(dungeon, 'graph'):
            start = None
            goal = None
            
            for node, data in dungeon.graph.nodes(data=True):
                label = data.get('label', '')
                if 's' in label.split(','):
                    start = node
                if 't' in label.split(','):
                    goal = node
            
            if start and goal:
                failures = self.path_analyzer.analyze_graph(
                    dungeon.graph, start, goal
                )
                all_failures.extend(failures)
        
        # Analyze room-level failures
        if hasattr(dungeon, 'rooms'):
            for room_id, room in enumerate(dungeon.rooms):
                if hasattr(room, 'grid'):
                    h, w = room.grid.shape[:2]
                    start = (0, w // 2)
                    goal = (h - 1, w // 2)
                    
                    failures = self.path_analyzer.analyze_grid(
                        room.grid, start, goal
                    )
                    for f in failures:
                        f.metadata = {'room_id': room_id}
                    all_failures.extend(failures)
        
        return all_failures


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_symbolic_refiner(
    tile_types: Optional[List[int]] = None,
    max_repair_attempts: int = 5,
    learned_stats: Optional[LearnedTileStatistics] = None,
    seed: Optional[int] = None,
) -> SymbolicRefiner:
    """
    Create a SymbolicRefiner instance.
    
    Args:
        tile_types: Available tile types
        max_repair_attempts: Maximum repair iterations
        learned_stats: LearnedTileStatistics for data-driven WFC (Phase 3B)
        
    Returns:
        SymbolicRefiner instance
    """
    return SymbolicRefiner(
        tile_types=tile_types,
        max_repair_attempts=max_repair_attempts,
        learned_stats=learned_stats,
        seed=seed,
    )


def quick_repair(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, bool]:
    """
    Quick repair function for a single room.
    
    Args:
        grid: H x W tile grid
        start: Start position as (row, col)
        goal: Goal position as (row, col)
        
    Returns:
        (repaired_grid, success)
    """
    refiner = create_symbolic_refiner(seed=seed)
    return refiner.repair_room(grid, start, goal, seed=seed)
