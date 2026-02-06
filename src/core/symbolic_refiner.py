"""
H-MOLQD Block VII: Symbolic Refiner with Wave Function Collapse
===============================================================

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
    P(n = t) ∝ P(n = t) × Σ_{t'} A(t, t') × P(c = t')
    where A(t, t') = adjacency compatibility

Convergence:
    Repeat until all cells collapsed or contradiction

"""

import logging
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import heapq
from collections import defaultdict

import numpy as np

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

logger = logging.getLogger(__name__)


# ============================================================================
# TILE DEFINITIONS
# ============================================================================

class TileType(Enum):
    """Tile types for WFC."""
    VOID = 0        # Impassable
    FLOOR = 1       # Walkable
    WALL = 2        # Solid
    DOOR_N = 10     # North door
    DOOR_S = 11     # South door
    DOOR_E = 12     # East door
    DOOR_W = 13     # West door
    KEY = 30        # Small key
    CHEST = 40      # Chest
    ENEMY = 50      # Enemy spawn


# Default adjacency rules for Zelda dungeons
DEFAULT_ADJACENCY: Dict[int, Set[int]] = {
    TileType.FLOOR.value: {
        TileType.FLOOR.value, TileType.WALL.value, 
        TileType.KEY.value, TileType.CHEST.value, TileType.ENEMY.value,
        TileType.DOOR_N.value, TileType.DOOR_S.value,
        TileType.DOOR_E.value, TileType.DOOR_W.value,
    },
    TileType.WALL.value: {
        TileType.WALL.value, TileType.FLOOR.value, TileType.VOID.value,
    },
    TileType.VOID.value: {
        TileType.VOID.value, TileType.WALL.value,
    },
    TileType.DOOR_N.value: {TileType.FLOOR.value, TileType.WALL.value},
    TileType.DOOR_S.value: {TileType.FLOOR.value, TileType.WALL.value},
    TileType.DOOR_E.value: {TileType.FLOOR.value, TileType.WALL.value},
    TileType.DOOR_W.value: {TileType.FLOOR.value, TileType.WALL.value},
    TileType.KEY.value: {TileType.FLOOR.value},
    TileType.CHEST.value: {TileType.FLOOR.value},
    TileType.ENEMY.value: {TileType.FLOOR.value},
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class FailurePoint:
    """Location where pathfinding failed."""
    position: Tuple[int, int]       # (x, y) or (room_id, position)
    failure_type: str               # 'blocked', 'missing_key', 'disconnected'
    required_item: Optional[str]    # Required item to proceed
    blocking_tiles: List[Tuple[int, int]] = field(default_factory=list)


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
    
    def entropy(self, x: int, y: int) -> float:
        """Compute entropy at cell (x, y)."""
        probs = self.grid[y, x]
        # Filter out zeros for log
        probs = probs[probs > 0]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    def is_collapsed(self, x: int, y: int) -> bool:
        """Check if cell is collapsed."""
        return self.collapsed[y, x]
    
    def get_options(self, x: int, y: int) -> List[int]:
        """Get possible tile types at cell."""
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
        self.walkable_tiles = walkable_tiles or {
            TileType.FLOOR.value,
            TileType.DOOR_N.value, TileType.DOOR_S.value,
            TileType.DOOR_E.value, TileType.DOOR_W.value,
            TileType.KEY.value, TileType.CHEST.value,
        }
    
    def analyze_grid(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> List[FailurePoint]:
        """
        Analyze pathfinding failures in a room grid.
        
        Args:
            grid: H x W tile grid
            start: Start position
            goal: Goal position
            
        Returns:
            List of failure points
        """
        failures = []
        
        # Try A* pathfinding
        path = self._astar(grid, start, goal)
        
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
        
        # Check for key-lock requirements
        try:
            path = nx.shortest_path(graph, start_node, goal_node)
        except nx.NetworkXNoPath:
            failures.append(FailurePoint(
                position=(start_node, goal_node),
                failure_type='no_path',
                required_item=None,
            ))
            return failures
        
        # Analyze path for missing keys
        keys_available = 0
        for i, node in enumerate(path):
            # Count keys at node
            node_data = graph.nodes[node]
            if 'k' in node_data.get('label', '').split(','):
                keys_available += 1
            
            # Check edges for locks
            if i < len(path) - 1:
                next_node = path[i + 1]
                edge_data = graph.edges[node, next_node]
                edge_type = edge_data.get('edge_type', edge_data.get('label', ''))
                
                if edge_type in ('key_locked', 'k'):
                    if keys_available <= 0:
                        failures.append(FailurePoint(
                            position=(node, next_node),
                            failure_type='missing_key',
                            required_item='key',
                        ))
                    else:
                        keys_available -= 1
        
        return failures
    
    def _astar(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        """Simple A* pathfinding."""
        h, w = grid.shape[:2]
        
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        def neighbors(x, y):
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx_, ny_ = x + dx, y + dy
                if 0 <= nx_ < w and 0 <= ny_ < h:
                    if grid[ny_, nx_] in self.walkable_tiles:
                        yield (nx_, ny_)
        
        # A* search
        open_set = [(heuristic(start, goal), 0, start, [start])]
        closed = set()
        
        while open_set:
            _, g, current, path = heapq.heappop(open_set)
            
            if current == goal:
                return path
            
            if current in closed:
                continue
            closed.add(current)
            
            for next_pos in neighbors(*current):
                if next_pos not in closed:
                    new_g = g + 1
                    new_f = new_g + heuristic(next_pos, goal)
                    heapq.heappush(open_set, (new_f, new_g, next_pos, path + [next_pos]))
        
        return None
    
    def _flood_fill(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
    ) -> Set[Tuple[int, int]]:
        """Flood fill to find reachable region."""
        h, w = grid.shape[:2]
        reachable = set()
        queue = [start]
        
        while queue:
            x, y = queue.pop()
            if (x, y) in reachable:
                continue
            if not (0 <= x < w and 0 <= y < h):
                continue
            if grid[y, x] not in self.walkable_tiles:
                continue
            
            reachable.add((x, y))
            queue.extend([(x+1, y), (x-1, y), (x, y+1), (x, y-1)])
        
        return reachable
    
    def _find_boundary(
        self,
        grid: np.ndarray,
        region_a: Set[Tuple[int, int]],
        region_b: Set[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """Find boundary tiles between two regions."""
        h, w = grid.shape[:2]
        boundary = []
        
        for x, y in region_a:
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx_, ny_ = x + dx, y + dy
                if 0 <= nx_ < w and 0 <= ny_ < h:
                    if (nx_, ny_) not in region_a and (nx_, ny_) not in region_b:
                        boundary.append((nx_, ny_))
        
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
                x, y = fp.position
                if isinstance(x, int) and isinstance(y, int):
                    self._mark_region(mask, x, y)
            
            # Mark blocking tiles
            for bx, by in fp.blocking_tiles:
                self._mark_region(mask, bx, by)
        
        return mask
    
    def _mark_region(self, mask: np.ndarray, cx: int, cy: int):
        """Mark region around center point."""
        h, w = mask.shape
        
        for dy in range(-self.margin, self.margin + 1):
            for dx in range(-self.margin, self.margin + 1):
                x, y = cx + dx, cy + dy
                if 0 <= x < w and 0 <= y < h:
                    mask[y, x] = True
    
    def expand_mask(
        self,
        mask: np.ndarray,
        iterations: int = 1,
    ) -> np.ndarray:
        """Expand mask by morphological dilation."""
        expanded = mask.copy()
        h, w = mask.shape
        
        for _ in range(iterations):
            new_mask = expanded.copy()
            for y in range(h):
                for x in range(w):
                    if expanded[y, x]:
                        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            ny, nx = y + dy, x + dx
                            if 0 <= nx < w and 0 <= ny < h:
                                new_mask[ny, nx] = True
            expanded = new_mask
        
        return expanded


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
    """
    
    def __init__(
        self,
        tile_types: List[int],
        adjacency: Optional[Dict[int, Set[int]]] = None,
        max_iterations: int = 10000,
    ):
        """
        Args:
            tile_types: List of available tile type IDs
            adjacency: Compatibility rules {tile: {compatible_neighbors}}
            max_iterations: Maximum collapse iterations
        """
        self.tile_types = tile_types
        self.adjacency = adjacency or DEFAULT_ADJACENCY
        self.max_iterations = max_iterations
    
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
        num_tiles = len(self.tile_types)
        
        # Initialize uniform distribution
        grid = np.ones((height, width, num_tiles)) / num_tiles
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
        
        for iteration in range(self.max_iterations):
            # Find cell with lowest entropy (that isn't collapsed)
            min_entropy = float('inf')
            min_cell = None
            
            for y in range(h):
                for x in range(w):
                    if not state.is_collapsed(x, y):
                        entropy = state.entropy(x, y)
                        if entropy < min_entropy:
                            min_entropy = entropy
                            min_cell = (x, y)
            
            if min_cell is None:
                # All cells collapsed
                break
            
            if min_entropy == 0:
                # Contradiction - no valid options
                logger.warning(f"WFC contradiction at {min_cell}")
                return self._extract_grid(state), False
            
            # Collapse the cell
            x, y = min_cell
            success = self._collapse_cell(state, x, y)
            
            if not success:
                return self._extract_grid(state), False
            
            # Propagate constraints
            self._propagate(state, x, y)
        
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
        
        tile_idx = np.random.choice(
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
    
    def _propagate(self, state: WFCState, x: int, y: int):
        """Propagate constraints from collapsed cell."""
        h, w = state.collapsed.shape
        
        # Get collapsed tile
        tile_idx = np.argmax(state.grid[y, x])
        tile = state.tile_types[tile_idx]
        
        # Get compatible neighbors
        compatible = self.adjacency.get(tile, set())
        
        # Update neighbors
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx_, ny_ = x + dx, y + dy
            
            if not (0 <= nx_ < w and 0 <= ny_ < h):
                continue
            if state.is_collapsed(nx_, ny_):
                continue
            
            # Restrict to compatible tiles
            for i, t in enumerate(state.tile_types):
                if t not in compatible:
                    state.grid[ny_, nx_, i] = 0.0
            
            # Renormalize
            total = state.grid[ny_, nx_].sum()
            if total > 0:
                state.grid[ny_, nx_] /= total
    
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
    ) -> np.ndarray:
        """
        Ensure start-goal connectivity.
        
        Creates a path if none exists.
        """
        h, w = grid.shape
        
        # Check existing path
        path = self._find_path(grid, start, goal, walkable)
        
        if path is not None:
            return grid
        
        # Create a simple path
        result = grid.copy()
        
        x0, y0 = start
        x1, y1 = goal
        
        # Horizontal then vertical path
        x, y = x0, y0
        
        while x != x1:
            if result[y, x] not in walkable:
                result[y, x] = TileType.FLOOR.value
            x += 1 if x1 > x else -1
        
        while y != y1:
            if result[y, x] not in walkable:
                result[y, x] = TileType.FLOOR.value
            y += 1 if y1 > y else -1
        
        return result
    
    def _find_path(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        walkable: Set[int],
    ) -> Optional[List[Tuple[int, int]]]:
        """BFS pathfinding."""
        h, w = grid.shape
        
        queue = [start]
        visited = {start}
        parent = {start: None}
        
        while queue:
            x, y = queue.pop(0)
            
            if (x, y) == goal:
                # Reconstruct path
                path = []
                current = goal
                while current is not None:
                    path.append(current)
                    current = parent[current]
                return list(reversed(path))
            
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx_, ny_ = x + dx, y + dy
                
                if not (0 <= nx_ < w and 0 <= ny_ < h):
                    continue
                if (nx_, ny_) in visited:
                    continue
                if grid[ny_, nx_] not in walkable:
                    continue
                
                visited.add((nx_, ny_))
                parent[(nx_, ny_)] = (x, y)
                queue.append((nx_, ny_))
        
        return None


# ============================================================================
# SYMBOLIC REFINER (Main Interface)
# ============================================================================

class SymbolicRefiner:
    """
    H-MOLQD Block VII: Symbolic Refiner.
    
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
            start=(5, 0),
            goal=(5, 15),
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
        max_repair_attempts: int = 5,
        margin: int = 2,
    ):
        """
        Args:
            tile_types: Available tile types for WFC
            adjacency: Tile adjacency rules
            max_repair_attempts: Maximum repair iterations
            margin: Extra cells around failures to reset
        """
        # Default tile types
        if tile_types is None:
            tile_types = [t.value for t in TileType]
        
        self.tile_types = tile_types
        self.adjacency = adjacency or DEFAULT_ADJACENCY
        self.max_repair_attempts = max_repair_attempts
        
        # Components
        self.path_analyzer = PathAnalyzer()
        self.entropy_reset = EntropyReset(margin=margin)
        self.wfc = WaveFunctionCollapse(
            tile_types=tile_types,
            adjacency=adjacency,
        )
        self.constraint_propagator = ConstraintPropagator(adjacency=adjacency)
    
    def repair_room(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Tuple[np.ndarray, bool]:
        """
        Repair a room grid to ensure solvability.
        
        Args:
            grid: H x W tile grid
            start: Start position
            goal: Goal position
            
        Returns:
            (repaired_grid, success)
        """
        current_grid = grid.copy()
        
        for attempt in range(self.max_repair_attempts):
            # Analyze failures
            failures = self.path_analyzer.analyze_grid(
                current_grid, start, goal
            )
            
            if not failures:
                # No failures = solvable
                logger.info(f"Room repaired successfully in {attempt + 1} attempts")
                return current_grid, True
            
            logger.debug(f"Repair attempt {attempt + 1}: {len(failures)} failure points")
            
            # Create mask
            mask = self.entropy_reset.create_mask(
                grid.shape[:2], failures
            )
            
            # Expand mask slightly
            mask = self.entropy_reset.expand_mask(mask, iterations=1)
            
            # Initialize WFC state
            state = self.wfc.initialize_state(
                height=grid.shape[0],
                width=grid.shape[1],
                initial_grid=current_grid,
                mask=mask,
            )
            
            # Run WFC
            current_grid, wfc_success = self.wfc.collapse(state)
            
            if not wfc_success:
                logger.warning(f"WFC failed on attempt {attempt + 1}")
                continue
            
            # Enforce connectivity constraint
            walkable = {
                TileType.FLOOR.value,
                TileType.KEY.value,
                TileType.CHEST.value,
            }
            current_grid = self.constraint_propagator.enforce_connectivity(
                current_grid, start, goal, walkable
            )
        
        # Final check
        failures = self.path_analyzer.analyze_grid(current_grid, start, goal)
        return current_grid, len(failures) == 0
    
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
            for room_id, room in enumerate(dungeon.rooms):
                if hasattr(room, 'grid'):
                    # Find start/goal for this room
                    h, w = room.grid.shape[:2]
                    start = (w // 2, 0)
                    goal = (w // 2, h - 1)
                    
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
                    start = (w // 2, 0)
                    goal = (w // 2, h - 1)
                    
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
) -> SymbolicRefiner:
    """
    Create a SymbolicRefiner instance.
    
    Args:
        tile_types: Available tile types
        max_repair_attempts: Maximum repair iterations
        
    Returns:
        SymbolicRefiner instance
    """
    return SymbolicRefiner(
        tile_types=tile_types,
        max_repair_attempts=max_repair_attempts,
    )


def quick_repair(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Tuple[np.ndarray, bool]:
    """
    Quick repair function for a single room.
    
    Args:
        grid: H x W tile grid
        start: Start position
        goal: Goal position
        
    Returns:
        (repaired_grid, success)
    """
    refiner = create_symbolic_refiner()
    return refiner.repair_room(grid, start, goal)
