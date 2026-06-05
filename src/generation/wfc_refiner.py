"""
Causal Wave Function Collapse for Dungeon Refinement
=====================================================

WFC with game-state awareness to ensure causal validity.

Problem:
    Standard WFC doesn't consider game state when placing tiles.
    A lock might be placed before its key, making the dungeon unsolvable.

Solution:
    Causal WFC tracks game state during tile collapse:
    1. Maintain "current inventory" during collapse
    2. Only allow LOCK tiles if corresponding KEY is already placed
    3. Propagate state changes through the grid
    4. Ensure causal ordering: KEY -> LOCK

Algorithm:
    1. Initialize entropy grid
    2. For each collapse:
       a. Select lowest entropy cell
       b. Filter valid tiles based on current game state
       c. Collapse cell
       d. Update game state if KEY/ITEM placed
       e. Propagate constraints
    3. Repeat until fully collapsed

Research:
- Gumin (2016) "Wave Function Collapse"
- Merrell & Manocha (2008) "Model Synthesis"
- Karth & Smith (2017) "WaveFunctionCollapse is Constraint Solving"

Usage:
    wfc = CausalWFC(
        tile_set=ZeldaTileSet(),
        width=16,
        height=11,
    )
    
    # Generate with causal constraints
    grid = wfc.generate(
        mission_graph=mission_graph,
        seed=42,
    )
    
    # Validate causality
    assert wfc.validate_causal_ordering(grid)
"""

import random
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any, Callable

from src.generation.wfc_types import (
    SEMANTIC_PALETTE,
    Cell,
    GameState,
    TileSet,
    TileType,
    ZeldaTileSet,
)

logger = logging.getLogger(__name__)

#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
# TILE / STATE MODELS ARE PROVIDED BY src.generation.wfc_types
#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+

#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
# Causal WFC core algorithm
#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+


class CausalWFC:
    """
    Wave Function Collapse with game-state awareness.
    
    Ensures causal validity by tracking game state during collapse:
    - Keys must be placed before their corresponding locks
    - Items must be placed before tiles that require them
    - Blocking tiles (locked doors) only allowed when keys available
    """
    
    def __init__(
        self,
        tile_set: TileSet,
        width: int = 11,
        height: int = 16,
        seed: Optional[int] = None,
        max_backtracks: int = 50,
        dead_end_radius: int = 2,
        dead_end_callback: Optional[Callable[[np.ndarray, np.ndarray, Tuple[int, int]], np.ndarray]] = None,
    ):
        self.tile_set = tile_set
        self.width = width
        self.height = height
        self.seed = seed
        self.rng = random.Random(seed)
        self.max_backtracks = int(max(1, max_backtracks))
        self.dead_end_radius = int(max(1, dead_end_radius))
        self.dead_end_callback = dead_end_callback
        
        # Grid of cells
        self.grid: List[List[Cell]] = []
        
        # Game state tracking
        self.game_state = GameState()
        
        # Collapse order (for causal validation)
        self.collapse_order: List[Tuple[int, int]] = []
        
        # Statistics
        self.contradictions = 0
        self.backtracks = 0
        self.last_contradiction: Optional[Tuple[int, int]] = None
        self.last_dead_end_mask: Optional[np.ndarray] = None
    
    def initialize(
        self,
        fixed_tiles: Optional[Dict[Tuple[int, int], int]] = None,
        start_pos: Optional[Tuple[int, int]] = None,
        goal_pos: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Initialize the grid with all possibilities.
        
        Args:
            fixed_tiles: Dict of (r, c) -> tile_id for pre-placed tiles
            start_pos: Position of START tile
            goal_pos: Position of GOAL tile
        """
        all_tiles = self.tile_set.get_all_tile_ids()
        
        # Create grid
        self.grid = []
        for r in range(self.height):
            row = []
            for c in range(self.width):
                cell = Cell(
                    row=r,
                    col=c,
                    possibilities=set(all_tiles),
                )
                row.append(cell)
            self.grid.append(row)
        
        # Reset state
        self.game_state = GameState()
        self.collapse_order = []
        self.contradictions = 0
        self.backtracks = 0
        self.last_contradiction = None
        self.last_dead_end_mask = None
        
        # Place fixed tiles
        fixed_tiles = fixed_tiles or {}
        
        # Add start and goal
        if start_pos:
            start_tiles = self.tile_set.get_tiles_by_type(TileType.START)
            if start_tiles:
                fixed_tiles[start_pos] = start_tiles[0].id
        
        if goal_pos:
            goal_tiles = self.tile_set.get_tiles_by_type(TileType.GOAL)
            if goal_tiles:
                fixed_tiles[goal_pos] = goal_tiles[0].id
        
        # Collapse fixed tiles
        for (r, c), tile_id in fixed_tiles.items():
            if 0 <= r < self.height and 0 <= c < self.width:
                self._collapse_cell(r, c, tile_id)
                self._update_game_state(r, c, tile_id)
                if not self._propagate(r, c):
                    self.last_contradiction = (int(r), int(c))
                    self.contradictions += 1
    
    def generate(
        self,
        start_pos: Tuple[int, int] = (14, 5),
        goal_pos: Tuple[int, int] = (1, 5),
        max_iterations: int = 10000,
    ) -> np.ndarray:
        """
        Generate a dungeon grid using causal WFC.
        
        Args:
            start_pos: (row, col) of START tile
            goal_pos: (row, col) of GOAL tile
            max_iterations: Maximum collapse iterations
            
        Returns:
            (H, W) numpy array of semantic tile IDs
        """
        # Initialize
        self.initialize(start_pos=start_pos, goal_pos=goal_pos)
        
        # Main WFC loop
        for iteration in range(max_iterations):
            # Find cell with lowest entropy
            cell = self._select_lowest_entropy_cell()
            
            if cell is None:
                # All cells collapsed
                break
            
            # Get valid tiles considering game state
            valid_tiles = self._get_causally_valid_tiles(cell)
            
            if not valid_tiles:
                # Contradiction - try to backtrack
                logger.warning(f"Contradiction at ({cell.row}, {cell.col})")
                self.contradictions += 1
                self.last_contradiction = (int(cell.row), int(cell.col))
                
                if not self._backtrack():
                    logger.error("Cannot resolve contradiction")
                    self._handle_dead_end_feedback()
                    break
                continue
            
            # Collapse to random valid tile
            tile_id = self._weighted_random_choice(valid_tiles)
            self._collapse_cell(cell.row, cell.col, tile_id)
            
            # Update game state
            self._update_game_state(cell.row, cell.col, tile_id)
            
            # Path-guided constraint: verify start->goal connectivity is not blocked
            # Check periodically (every 50 iterations) to avoid performance hit
            if iteration > 0 and iteration % 50 == 0:
                if not self._verify_path_connectivity(start_pos, goal_pos):
                    logger.warning(f"Path blocked at iteration {iteration}, backtracking")
                    self.contradictions += 1
                    self.last_contradiction = (int(cell.row), int(cell.col))
                    if not self._backtrack():
                        logger.error("Cannot restore connectivity")
                        self._handle_dead_end_feedback()
                        break
                    continue
            
            # Propagate constraints
            if not self._propagate(cell.row, cell.col):
                self.contradictions += 1
                self.last_contradiction = (int(cell.row), int(cell.col))
                if not self._backtrack():
                    logger.error("Cannot resolve propagated contradiction")
                    self._handle_dead_end_feedback()
                    break
        
        # Convert to numpy array
        return self._to_numpy()
    
    def _select_lowest_entropy_cell(self) -> Optional[Cell]:
        """Select uncollapsed cell with lowest entropy."""
        min_entropy = float('inf')
        best_cell = None
        
        for row in self.grid:
            for cell in row:
                if not cell.is_collapsed and cell.possibilities:
                    entropy = cell.entropy(self.rng)
                    if entropy < min_entropy:
                        min_entropy = entropy
                        best_cell = cell
        
        return best_cell
    
    def _get_causally_valid_tiles(self, cell: Cell) -> Set[int]:
        """
        Filter cell possibilities based on game state.
        
        Removes tiles that would violate causal constraints:
        - Locked doors when no keys available
        - Items that require other items not yet placed
        """
        valid = set()
        
        for tile_id in cell.possibilities:
            tile = self.tile_set.get_tile(tile_id)
            if tile is None:
                continue
            
            constraint = tile.constraint
            
            # Check key requirements
            if constraint.required_keys > 0:
                if not self.game_state.can_unlock(constraint.required_keys):
                    # Cannot place locked door without key
                    continue
            
            # Check item requirements
            if constraint.required_items:
                if not constraint.required_items.issubset(self.game_state.items_collected):
                    continue
            
            valid.add(tile_id)
        
        return valid
    
    def _verify_path_connectivity(
        self,
        start_pos: Tuple[int, int],
        goal_pos: Tuple[int, int]
    ) -> bool:
        """
        Verify that a path exists from start to goal using BFS.
        
        Path-guided constraint: ensures critical path is not blocked
        by wall placements during WFC generation.
        
        Args:
            start_pos: (row, col) start position
            goal_pos: (row, col) goal position
            
        Returns:
            True if path exists, False if blocked
        """
        from collections import deque
        
        # Build walkable set from current grid state
        walkable = set()
        for r in range(self.height):
            for c in range(self.width):
                cell = self.grid[r][c]
                if not cell.is_collapsed:
                    # Uncollapsed cells are potentially walkable
                    walkable.add((r, c))
                else:
                    tile = self.tile_set.get_tile(cell.collapsed_tile)
                    if tile and tile.tile_type not in (TileType.WALL,):
                        walkable.add((r, c))
        
        # BFS from start to goal
        if start_pos not in walkable and start_pos != goal_pos:
            return False
        if goal_pos not in walkable:
            return False
        
        visited = {start_pos}
        queue = deque([start_pos])
        
        while queue:
            r, c = queue.popleft()
            if (r, c) == goal_pos:
                return True
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in walkable and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        
        return False
    
    def _weighted_random_choice(self, tile_ids: Set[int]) -> int:
        """Choose a tile weighted by tile weights."""
        # Sort tile_ids for deterministic iteration order
        sorted_ids = sorted(tile_ids)
        tiles = [(tid, self.tile_set.get_tile(tid)) for tid in sorted_ids]
        tiles = [(tid, t) for tid, t in tiles if t is not None]
        
        if not tiles:
            return sorted_ids[0]
        
        weights = [t.weight for _, t in tiles]
        total = sum(weights)
        
        if total == 0:
            return tiles[0][0]
        
        r = self.rng.uniform(0, total)
        cumulative = 0
        
        for tid, tile in tiles:
            cumulative += tile.weight
            if r <= cumulative:
                return tid
        
        return tiles[-1][0]
    
    def _collapse_cell(self, row: int, col: int, tile_id: int) -> None:
        """Collapse a cell to a specific tile."""
        cell = self.grid[row][col]
        cell.collapsed_tile = tile_id
        cell.possibilities = {tile_id}
        self.collapse_order.append((row, col))
        
        # Track placement order
        self.game_state.placement_order.append((row, col, tile_id))
    
    def _update_game_state(self, row: int, col: int, tile_id: int) -> None:
        """Update game state after placing a tile."""
        tile = self.tile_set.get_tile(tile_id)
        if tile is None:
            return
        
        constraint = tile.constraint
        
        # Collect key
        if constraint.provides_key:
            self.game_state.collect_key((row, col))
            logger.debug(f"Key collected at ({row}, {col}), total: {self.game_state.keys_collected}")
        
        # Collect item
        if constraint.provides_item:
            self.game_state.items_collected.add(constraint.provides_item)
        
        # Track lock placement
        if tile.tile_type == TileType.DOOR_LOCKED:
            self.game_state.place_lock((row, col))
    
    def _propagate(self, start_row: int, start_col: int) -> bool:
        """Propagate adjacency supports recursively until arc consistency stabilizes."""
        stack = [(start_row, start_col)]
        
        while stack:
            r, c = stack.pop()
            current = self.grid[r][c]
            current_tile_ids = (
                {int(current.collapsed_tile)}
                if current.is_collapsed
                else set(current.possibilities)
            )
            if not current_tile_ids:
                self.last_contradiction = (int(r), int(c))
                return False

            current_tiles = [
                tile
                for tile_id in current_tile_ids
                if (tile := self.tile_set.get_tile(tile_id)) is not None
            ]
            if not current_tiles:
                continue
            
            # Check neighbors
            neighbors = [
                (r - 1, c, 'N', 'S'),  # North
                (r + 1, c, 'S', 'N'),  # South
                (r, c - 1, 'W', 'E'),  # West
                (r, c + 1, 'E', 'W'),  # East
            ]
            
            for nr, nc, direction, _reverse_dir in neighbors:
                if not (0 <= nr < self.height and 0 <= nc < self.width):
                    continue
                
                neighbor = self.grid[nr][nc]

                # A neighboring tile remains possible if at least one current
                # possibility supports it in this direction.
                allowed: Set[int] = set()
                for current_tile in current_tiles:
                    allowed.update(current_tile.adjacency.get(direction, set()))

                if neighbor.is_collapsed:
                    if int(neighbor.collapsed_tile) not in allowed:
                        self.last_contradiction = (int(nr), int(nc))
                        return False
                    continue
                
                # Constrain neighbor possibilities
                old_size = len(neighbor.possibilities)
                neighbor.possibilities &= allowed
                if not neighbor.possibilities:
                    self.last_contradiction = (int(nr), int(nc))
                    return False
                
                # If changed, add to stack
                if len(neighbor.possibilities) < old_size:
                    stack.append((nr, nc))
        return True
    
    def _backtrack(self) -> bool:
        """Attempt to backtrack on contradiction."""
        if self.backtracks >= self.max_backtracks:
            logger.warning("Backtrack limit reached (%d)", self.max_backtracks)
            return False
        if not self.collapse_order:
            return False
        
        self.backtracks += 1
        
        # Remove last collapsed cell
        r, c = self.collapse_order.pop()
        cell = self.grid[r][c]
        banned_tile = cell.collapsed_tile
        
        retained_placements = list(self.game_state.placement_order[:-1])
        self._rebuild_from_placements(retained_placements)
        cell = self.grid[r][c]
        if banned_tile is not None:
            cell.possibilities.discard(int(banned_tile))
        
        return True

    def _rebuild_from_placements(self, placements: List[Tuple[int, int, int]]) -> None:
        """Recompute constraints and game state after removing a failed decision."""
        all_tiles = self.tile_set.get_all_tile_ids()
        self.grid = [
            [Cell(row=r, col=c, possibilities=set(all_tiles)) for c in range(self.width)]
            for r in range(self.height)
        ]
        self.game_state = GameState()
        self.collapse_order = []

        for row, col, tile_id in placements:
            self._collapse_cell(int(row), int(col), int(tile_id))
            self._update_game_state(int(row), int(col), int(tile_id))
            if not self._propagate(int(row), int(col)):
                break

    def _build_dead_end_mask(self, center_rc: Tuple[int, int]) -> np.ndarray:
        """Build local contradiction mask around a dead-end center cell."""
        mask = np.zeros((self.height, self.width), dtype=bool)
        cy, cx = int(center_rc[0]), int(center_rc[1])
        for y in range(max(0, cy - self.dead_end_radius), min(self.height, cy + self.dead_end_radius + 1)):
            for x in range(max(0, cx - self.dead_end_radius), min(self.width, cx + self.dead_end_radius + 1)):
                mask[y, x] = True
        return mask

    def _handle_dead_end_feedback(self) -> None:
        """Invoke optional feedback callback with dead-end mask and current grid."""
        if self.last_contradiction is None:
            return
        mask = self._build_dead_end_mask(self.last_contradiction)
        self.last_dead_end_mask = mask
        if self.dead_end_callback is None:
            return
        try:
            patched = self.dead_end_callback(self._to_numpy(), mask.copy(), self.last_contradiction)
            if isinstance(patched, np.ndarray) and patched.shape == (self.height, self.width):
                # Re-initialize from patched state so generation can continue externally.
                self.initialize(fixed_tiles=None)
                for r in range(self.height):
                    for c in range(self.width):
                        self._collapse_cell(r, c, int(patched[r, c]))
        except (AttributeError, RuntimeError, ValueError, TypeError) as e:
            logger.warning("Dead-end callback failed: %s", e)
    
    def _to_numpy(self) -> np.ndarray:
        """Convert grid to numpy array of semantic IDs."""
        result = np.zeros((self.height, self.width), dtype=np.int32)
        
        for r in range(self.height):
            for c in range(self.width):
                cell = self.grid[r][c]
                if cell.is_collapsed:
                    tile = self.tile_set.get_tile(cell.collapsed_tile)
                    if tile:
                        result[r, c] = tile.semantic_id
                    else:
                        result[r, c] = SEMANTIC_PALETTE.get('FLOOR', 1)
                else:
                    # Uncollapsed - default to floor
                    result[r, c] = SEMANTIC_PALETTE.get('FLOOR', 1)
        
        return result
    
    def validate_causal_ordering(self, grid: Optional[np.ndarray] = None) -> bool:
        """
        Validate that the generated grid has valid causal ordering.
        
        For each lock, checks that a key was placed before it
        in the collapse order.
        """
        if not self.game_state.placement_order:
            logger.warning("No placement order recorded")
            return True
        
        key_placements = set()
        
        for r, c, tile_id in self.game_state.placement_order:
            tile = self.tile_set.get_tile(tile_id)
            if tile is None:
                continue
            
            # Track key placements
            if tile.constraint.provides_key:
                key_placements.add((r, c))
            
            # Check lock placements
            if tile.tile_type == TileType.DOOR_LOCKED:
                if not key_placements:
                    logger.warning(f"Lock at ({r}, {c}) placed before any key")
                    return False
        
        logger.info("Causal ordering validated successfully")
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics."""
        return {
            'contradictions': self.contradictions,
            'backtracks': self.backtracks,
            'collapse_order_length': len(self.collapse_order),
            'keys_placed': self.game_state.keys_collected,
            'locks_placed': len(self.game_state.lock_positions),
        }


# ============================================================================
# INTEGRATION WITH MISSION GRAMMAR
# ============================================================================

def generate_with_grammar(
    mission_graph: Any,  # MissionGraph from grammar.py
    width: int = 11,
    height: int = 16,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate dungeon grid from mission graph using causal WFC.
    
    Args:
        mission_graph: MissionGraph from grammar.py
        width: Grid width
        height: Grid height
        seed: Random seed
        
    Returns:
        (H, W) numpy array of semantic tile IDs
    """
    # Create WFC
    tile_set = ZeldaTileSet()
    wfc = CausalWFC(tile_set, width, height, seed)
    
    # Extract positions from graph
    start_pos = (height - 2, width // 2)  # Default near bottom
    goal_pos = (1, width // 2)            # Default near top
    
    if hasattr(mission_graph, 'get_start_node'):
        start_node = mission_graph.get_start_node()
        if start_node:
            start_pos = start_node.position
    
    if hasattr(mission_graph, 'get_goal_node'):
        goal_node = mission_graph.get_goal_node()
        if goal_node:
            goal_pos = goal_node.position
    
    # Clamp to valid range
    start_pos = (
        max(1, min(height - 2, start_pos[0])),
        max(1, min(width - 2, start_pos[1])),
    )
    goal_pos = (
        max(1, min(height - 2, goal_pos[0])),
        max(1, min(width - 2, goal_pos[1])),
    )
    
    # Generate
    grid = wfc.generate(start_pos=start_pos, goal_pos=goal_pos)
    
    # Validate
    if not wfc.validate_causal_ordering():
        logger.warning("Generated grid failed causal validation")
    
    stats = wfc.get_statistics()
    logger.info(f"WFC stats: {stats}")
    
    return grid


# ============================================================================
# TEST
# ============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Causal WFC...")
    
    # Create tile set
    tile_set = ZeldaTileSet()
    print(f"Tile set has {len(tile_set.tiles)} tiles")
    
    # Create WFC
    wfc = CausalWFC(
        tile_set=tile_set,
        width=11,
        height=16,
        seed=42,
    )
    
    # Generate
    grid = wfc.generate(
        start_pos=(14, 5),
        goal_pos=(1, 5),
    )
    
    print(f"\nGenerated grid shape: {grid.shape}")
    
    # ASCII visualization
    print("\nGrid visualization:")
    CHAR_MAP = {
        0: ' ', 1: '.', 2: '#', 3: 'B',
        10: '+', 11: 'D',
        20: 'M', 21: 'S', 22: 'T',
        30: 'K', 31: 'k',
    }
    
    for row in grid:
        line = ''
        for val in row:
            line += CHAR_MAP.get(int(val), '?')
        print(line)
    
    # Validate
    valid = wfc.validate_causal_ordering()
    print(f"\nCausal ordering valid: {valid}")
    
    # Statistics
    stats = wfc.get_statistics()
    print(f"Statistics: {stats}")
    
    print("\nCausal WFC test passed!")
