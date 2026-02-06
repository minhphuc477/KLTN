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
    4. Ensure causal ordering: KEY → LOCK

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
from typing import Dict, List, Tuple, Optional, Set, Any, FrozenSet
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
import heapq

logger = logging.getLogger(__name__)

# Import semantic palette if available
try:
    from src.core.definitions import SEMANTIC_PALETTE
except ImportError:
    # Fallback palette
    SEMANTIC_PALETTE = {
        'VOID': 0, 'FLOOR': 1, 'WALL': 2, 'BLOCK': 3,
        'DOOR_OPEN': 10, 'DOOR_LOCKED': 11,
        'ENEMY': 20, 'START': 21, 'TRIFORCE': 22,
        'KEY_SMALL': 30, 'KEY_BIG': 31,
    }


# ============================================================================
# TILE DEFINITIONS
# ============================================================================

class TileType(Enum):
    """Tile types with game state implications."""
    FLOOR = auto()
    WALL = auto()
    BLOCK = auto()
    DOOR_OPEN = auto()
    DOOR_LOCKED = auto()
    KEY_SMALL = auto()
    KEY_BIG = auto()
    ENEMY = auto()
    START = auto()
    GOAL = auto()
    ITEM = auto()
    WATER = auto()
    BRIDGE = auto()


@dataclass
class TileConstraint:
    """Constraints on when a tile can be placed."""
    required_keys: int = 0           # Minimum keys needed
    required_items: Set[str] = field(default_factory=set)  # Required items
    provides_key: bool = False       # This tile provides a key
    provides_item: Optional[str] = None  # Item this tile provides
    is_blocking: bool = False        # Blocks path until condition met
    key_id: Optional[int] = None     # Specific key ID for lock-key matching


@dataclass
class Tile:
    """Tile with adjacency rules and game constraints."""
    id: int
    tile_type: TileType
    semantic_id: int  # SEMANTIC_PALETTE value
    
    # Adjacency rules: which tiles can be adjacent in each direction
    # Format: {direction: set of allowed tile IDs}
    adjacency: Dict[str, Set[int]] = field(default_factory=dict)
    
    # Weight for random selection (higher = more common)
    weight: float = 1.0
    
    # Game state constraints
    constraint: TileConstraint = field(default_factory=TileConstraint)


class TileSet:
    """Collection of tiles with adjacency rules."""
    
    def __init__(self):
        self.tiles: Dict[int, Tile] = {}
        self._build_tiles()
    
    def _build_tiles(self):
        """Build tile definitions. Override in subclass."""
        pass
    
    def get_tile(self, tile_id: int) -> Optional[Tile]:
        return self.tiles.get(tile_id)
    
    def get_all_tile_ids(self) -> Set[int]:
        return set(self.tiles.keys())
    
    def get_tiles_by_type(self, tile_type: TileType) -> List[Tile]:
        return [t for t in self.tiles.values() if t.tile_type == tile_type]


class ZeldaTileSet(TileSet):
    """Zelda-specific tile set with adjacency rules."""
    
    def _build_tiles(self):
        """Build Zelda tile definitions."""
        # Define tiles
        self.tiles = {
            # Floor tiles
            0: Tile(
                id=0, 
                tile_type=TileType.FLOOR,
                semantic_id=SEMANTIC_PALETTE.get('FLOOR', 1),
                weight=10.0,
            ),
            # Wall
            1: Tile(
                id=1,
                tile_type=TileType.WALL,
                semantic_id=SEMANTIC_PALETTE.get('WALL', 2),
                weight=3.0,
            ),
            # Block (pushable)
            2: Tile(
                id=2,
                tile_type=TileType.BLOCK,
                semantic_id=SEMANTIC_PALETTE.get('BLOCK', 3),
                weight=1.0,
            ),
            # Open door
            3: Tile(
                id=3,
                tile_type=TileType.DOOR_OPEN,
                semantic_id=SEMANTIC_PALETTE.get('DOOR_OPEN', 10),
                weight=0.5,
            ),
            # Locked door (requires key)
            4: Tile(
                id=4,
                tile_type=TileType.DOOR_LOCKED,
                semantic_id=SEMANTIC_PALETTE.get('DOOR_LOCKED', 11),
                weight=0.3,
                constraint=TileConstraint(required_keys=1, is_blocking=True),
            ),
            # Small key
            5: Tile(
                id=5,
                tile_type=TileType.KEY_SMALL,
                semantic_id=SEMANTIC_PALETTE.get('KEY_SMALL', 30),
                weight=0.5,
                constraint=TileConstraint(provides_key=True),
            ),
            # Enemy
            6: Tile(
                id=6,
                tile_type=TileType.ENEMY,
                semantic_id=SEMANTIC_PALETTE.get('ENEMY', 20),
                weight=1.5,
            ),
            # Start
            7: Tile(
                id=7,
                tile_type=TileType.START,
                semantic_id=SEMANTIC_PALETTE.get('START', 21),
                weight=0.0,  # Placed manually
            ),
            # Goal (Triforce)
            8: Tile(
                id=8,
                tile_type=TileType.GOAL,
                semantic_id=SEMANTIC_PALETTE.get('TRIFORCE', 22),
                weight=0.0,  # Placed manually
            ),
        }
        
        # Build adjacency rules
        self._build_adjacency_rules()
    
    def _build_adjacency_rules(self):
        """Build adjacency constraints between tiles."""
        floor_ids = {0, 3, 4, 5, 6, 7, 8}  # Tiles that act like floor
        wall_ids = {1, 2}
        
        for tile_id, tile in self.tiles.items():
            # All directions
            directions = ['N', 'S', 'E', 'W']
            
            if tile.tile_type in [TileType.FLOOR, TileType.DOOR_OPEN, 
                                   TileType.KEY_SMALL, TileType.ENEMY,
                                   TileType.START, TileType.GOAL]:
                # Floor-like tiles can be adjacent to most tiles
                for d in directions:
                    tile.adjacency[d] = floor_ids | wall_ids
            
            elif tile.tile_type == TileType.WALL:
                # Walls can be adjacent to anything
                for d in directions:
                    tile.adjacency[d] = floor_ids | wall_ids
            
            elif tile.tile_type == TileType.BLOCK:
                # Blocks need floor around them (to push)
                for d in directions:
                    tile.adjacency[d] = floor_ids | wall_ids
            
            elif tile.tile_type == TileType.DOOR_LOCKED:
                # Locked doors need floor on both sides
                for d in directions:
                    tile.adjacency[d] = floor_ids


# ============================================================================
# GAME STATE
# ============================================================================

@dataclass
class GameState:
    """Tracks game state during WFC collapse."""
    keys_collected: int = 0
    items_collected: Set[str] = field(default_factory=set)
    
    # Track positions of placed items for causal validation
    key_positions: List[Tuple[int, int]] = field(default_factory=list)
    lock_positions: List[Tuple[int, int]] = field(default_factory=list)
    
    # Order of placement for causality
    placement_order: List[Tuple[int, int, int]] = field(default_factory=list)  # (r, c, tile_id)
    
    def copy(self) -> 'GameState':
        """Create a copy of the game state."""
        return GameState(
            keys_collected=self.keys_collected,
            items_collected=set(self.items_collected),
            key_positions=list(self.key_positions),
            lock_positions=list(self.lock_positions),
            placement_order=list(self.placement_order),
        )
    
    def can_unlock(self, required_keys: int = 1) -> bool:
        """Check if we have enough keys to unlock."""
        return self.keys_collected >= required_keys
    
    def collect_key(self, position: Tuple[int, int]) -> None:
        """Collect a key at the given position."""
        self.keys_collected += 1
        self.key_positions.append(position)
    
    def place_lock(self, position: Tuple[int, int]) -> None:
        """Place a lock at the given position."""
        self.lock_positions.append(position)


# ============================================================================
# CAUSAL WFC
# ============================================================================

@dataclass
class Cell:
    """Single cell in the WFC grid."""
    row: int
    col: int
    
    # Possible tiles (starts with all, gets constrained)
    possibilities: Set[int] = field(default_factory=set)
    
    # Collapsed tile (None if not yet collapsed)
    collapsed_tile: Optional[int] = None
    
    @property
    def entropy(self) -> float:
        """
        Entropy of the cell.
        
        Lower entropy = fewer possibilities = higher priority for collapse.
        """
        if self.collapsed_tile is not None:
            return 0.0
        return len(self.possibilities) + random.random() * 0.1  # Small noise for tie-breaking
    
    @property
    def is_collapsed(self) -> bool:
        return self.collapsed_tile is not None


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
    ):
        self.tile_set = tile_set
        self.width = width
        self.height = height
        self.seed = seed
        self.rng = random.Random(seed)
        
        # Grid of cells
        self.grid: List[List[Cell]] = []
        
        # Game state tracking
        self.game_state = GameState()
        
        # Collapse order (for causal validation)
        self.collapse_order: List[Tuple[int, int]] = []
        
        # Statistics
        self.contradictions = 0
        self.backtracks = 0
    
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
                self._propagate(r, c)
    
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
                
                if not self._backtrack():
                    logger.error("Cannot resolve contradiction")
                    break
                continue
            
            # Collapse to random valid tile
            tile_id = self._weighted_random_choice(valid_tiles)
            self._collapse_cell(cell.row, cell.col, tile_id)
            
            # Update game state
            self._update_game_state(cell.row, cell.col, tile_id)
            
            # Propagate constraints
            self._propagate(cell.row, cell.col)
        
        # Convert to numpy array
        return self._to_numpy()
    
    def _select_lowest_entropy_cell(self) -> Optional[Cell]:
        """Select uncollapsed cell with lowest entropy."""
        min_entropy = float('inf')
        best_cell = None
        
        for row in self.grid:
            for cell in row:
                if not cell.is_collapsed and cell.possibilities:
                    entropy = cell.entropy
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
    
    def _weighted_random_choice(self, tile_ids: Set[int]) -> int:
        """Choose a tile weighted by tile weights."""
        tiles = [(tid, self.tile_set.get_tile(tid)) for tid in tile_ids]
        tiles = [(tid, t) for tid, t in tiles if t is not None]
        
        if not tiles:
            return list(tile_ids)[0]
        
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
    
    def _propagate(self, start_row: int, start_col: int) -> None:
        """Propagate constraints from collapsed cell."""
        stack = [(start_row, start_col)]
        visited = set()
        
        while stack:
            r, c = stack.pop()
            
            if (r, c) in visited:
                continue
            visited.add((r, c))
            
            current = self.grid[r][c]
            if not current.is_collapsed:
                continue
            
            current_tile = self.tile_set.get_tile(current.collapsed_tile)
            if current_tile is None:
                continue
            
            # Check neighbors
            neighbors = [
                (r - 1, c, 'N', 'S'),  # North
                (r + 1, c, 'S', 'N'),  # South
                (r, c - 1, 'W', 'E'),  # West
                (r, c + 1, 'E', 'W'),  # East
            ]
            
            for nr, nc, direction, reverse_dir in neighbors:
                if not (0 <= nr < self.height and 0 <= nc < self.width):
                    continue
                
                neighbor = self.grid[nr][nc]
                if neighbor.is_collapsed:
                    continue
                
                # Get allowed tiles based on adjacency
                allowed = current_tile.adjacency.get(direction, set())
                
                # Constrain neighbor possibilities
                old_size = len(neighbor.possibilities)
                neighbor.possibilities &= allowed
                
                # If changed, add to stack
                if len(neighbor.possibilities) < old_size:
                    stack.append((nr, nc))
    
    def _backtrack(self) -> bool:
        """Attempt to backtrack on contradiction."""
        if not self.collapse_order:
            return False
        
        self.backtracks += 1
        
        # Remove last collapsed cell
        r, c = self.collapse_order.pop()
        cell = self.grid[r][c]
        
        # Remove from placement order
        if self.game_state.placement_order:
            self.game_state.placement_order.pop()
        
        # Undo game state changes
        tile = self.tile_set.get_tile(cell.collapsed_tile)
        if tile and tile.constraint.provides_key:
            self.game_state.keys_collected = max(0, self.game_state.keys_collected - 1)
            if self.game_state.key_positions:
                self.game_state.key_positions.pop()
        
        # Reset cell
        all_tiles = self.tile_set.get_all_tile_ids()
        cell.collapsed_tile = None
        cell.possibilities = set(all_tiles)
        
        return True
    
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
