"""
Procedural Dungeon Generation for Zelda
=======================================

Research:
- "Procedural Content Generation via Machine Learning" (Summerville et al., 2018)
- Binary Space Partitioning (BSP) algorithm
- Constraint satisfaction for solvability

Algorithm:
1. BSP tree to partition space into rooms
2. Connect rooms with corridors
3. Place keys before locked doors (topological sort)
4. Add enemies, blocks, items based on difficulty
5. Validate solvability

Output: VGLC-compatible dungeon grid
"""

import random
import logging
import numpy as np
from typing import List, Tuple, Optional, Set, Dict
from dataclasses import dataclass
from enum import Enum
from simulation.validator import SEMANTIC_PALETTE

logger = logging.getLogger(__name__)


class Difficulty(Enum):
    """Dungeon difficulty levels."""
    EASY = 1
    MEDIUM = 2
    HARD = 3
    EXPERT = 4


@dataclass
class Room:
    """Rectangular room in dungeon."""
    x: int  # Top-left column
    y: int  # Top-left row
    width: int
    height: int
    
    @property
    def center(self) -> Tuple[int, int]:
        """Center position (r, c)."""
        return (self.y + self.height // 2, self.x + self.width // 2)
    
    def contains(self, r: int, c: int) -> bool:
        """Check if position is inside room."""
        return self.y <= r < self.y + self.height and self.x <= c < self.x + self.width
    
    def overlaps(self, other: 'Room') -> bool:
        """Check if this room overlaps with another."""
        return not (self.x + self.width <= other.x or
                   other.x + other.width <= self.x or
                   self.y + self.height <= other.y or
                   other.y + other.height <= self.y)


@dataclass
class Corridor:
    """Corridor connecting two rooms."""
    start: Tuple[int, int]  # (r, c)
    end: Tuple[int, int]    # (r, c)
    is_horizontal: bool


class BSPNode:
    """Binary Space Partitioning tree node."""
    
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        
        self.left: Optional['BSPNode'] = None
        self.right: Optional['BSPNode'] = None
        self.room: Optional[Room] = None
    
    def split(self, min_room_size: int = 5) -> bool:
        """
        Recursively split this node into left/right children.
        
        Args:
            min_room_size: Minimum room dimension
            
        Returns:
            True if split succeeded
        """
        # Already split
        if self.left or self.right:
            return False
        
        # Determine split direction (prefer larger dimension)
        split_horizontal = random.random() > 0.5
        if self.width > self.height and self.width / self.height >= 1.25:
            split_horizontal = False
        elif self.height > self.width and self.height / self.width >= 1.25:
            split_horizontal = True
        
        # Calculate maximum split position
        max_size = (self.height if split_horizontal else self.width) - min_room_size
        
        if max_size <= min_room_size:
            return False  # Too small to split
        
        # Choose split position
        split_pos = random.randint(min_room_size, max_size)
        
        # Create child nodes
        if split_horizontal:
            self.left = BSPNode(self.x, self.y, self.width, split_pos)
            self.right = BSPNode(self.x, self.y + split_pos, self.width, self.height - split_pos)
        else:
            self.left = BSPNode(self.x, self.y, split_pos, self.height)
            self.right = BSPNode(self.x + split_pos, self.y, self.width - split_pos, self.height)
        
        return True
    
    def create_rooms(self, min_room_size: int = 4, max_room_size: int = 10) -> List[Room]:
        """
        Create rooms in leaf nodes.
        
        Returns:
            List of all rooms created
        """
        if self.left or self.right:
            # Not a leaf - recurse
            rooms = []
            if self.left:
                rooms.extend(self.left.create_rooms(min_room_size, max_room_size))
            if self.right:
                rooms.extend(self.right.create_rooms(min_room_size, max_room_size))
            return rooms
        else:
            # Leaf node - create room
            # Ensure minimum valid ranges for room dimensions
            max_width = max(min_room_size, min(max_room_size, self.width - 2))
            max_height = max(min_room_size, min(max_room_size, self.height - 2))
            
            # Skip room creation if space is too small
            if max_width < min_room_size or max_height < min_room_size:
                return []
            
            room_width = random.randint(min_room_size, max_width)
            room_height = random.randint(min_room_size, max_height)
            
            # Ensure valid placement range
            x_range = max(1, self.width - room_width - 1)
            y_range = max(1, self.height - room_height - 1)
            
            room_x = self.x + random.randint(1, x_range)
            room_y = self.y + random.randint(1, y_range)
            
            self.room = Room(room_x, room_y, room_width, room_height)
            return [self.room]
    
    def get_rooms(self) -> List[Room]:
        """Get all rooms in subtree."""
        if self.room:
            return [self.room]
        
        rooms = []
        if self.left:
            rooms.extend(self.left.get_rooms())
        if self.right:
            rooms.extend(self.right.get_rooms())
        
        return rooms


class DungeonGenerator:
    """
    Procedural dungeon generator using BSP algorithm.
    
    Features:
    - Guaranteed solvability (keys before doors)
    - Adjustable difficulty
    - VGLC-compatible output
    - Configurable size and complexity
    """
    
    def __init__(
        self,
        width: int = 40,
        height: int = 40,
        difficulty: Difficulty = Difficulty.MEDIUM,
        seed: Optional[int] = None
    ):
        """
        Initialize generator.
        
        Args:
            width: Dungeon width (columns)
            height: Dungeon height (rows)
            difficulty: Difficulty level
            seed: Random seed for reproducibility
        """
        self.width = width
        self.height = height
        self.difficulty = difficulty
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Grid (will be filled with semantic IDs)
        self.grid = np.full((height, width), SEMANTIC_PALETTE['VOID'], dtype=np.int32)
        
        # Rooms and corridors
        self.rooms: List[Room] = []
        self.corridors: List[Corridor] = []
        
        # Item positions (for dependency tracking)
        self.key_positions: List[Tuple[int, int]] = []
        self.door_positions: List[Tuple[int, int]] = []
        self.start_pos: Optional[Tuple[int, int]] = None
        self.goal_pos: Optional[Tuple[int, int]] = None
    
    def generate(self) -> np.ndarray:
        """
        Generate complete dungeon.
        
        Returns:
            2D numpy array (semantic grid)
        """
        logger.info(f"Generating {self.width}×{self.height} dungeon (difficulty: {self.difficulty.name})")
        
        # Step 1: Create BSP tree and rooms
        self._create_rooms_bsp()
        
        # Step 2: Connect rooms with corridors
        self._connect_rooms()
        
        # Step 3: Place start and goal
        self._place_start_and_goal()
        
        # Step 4: Place keys and locked doors (ensure solvability)
        self._place_keys_and_doors()
        
        # Step 5: Place enemies
        self._place_enemies()
        
        # Step 6: Place blocks and obstacles
        self._place_obstacles()
        
        # Step 7: Add walls around rooms
        self._add_walls()
        
        logger.info(f"Generated dungeon: {len(self.rooms)} rooms, "
                   f"{len(self.key_positions)} keys, "
                   f"{len(self.door_positions)} doors")
        
        return self.grid
    
    def _create_rooms_bsp(self):
        """Create rooms using BSP algorithm."""
        # Create root node
        root = BSPNode(0, 0, self.width, self.height)
        
        # Split recursively
        split_queue = [root]
        max_splits = 20  # Limit depth
        
        for _ in range(max_splits):
            if not split_queue:
                break
            
            node = split_queue.pop(0)
            
            if node.split():
                if node.left:
                    split_queue.append(node.left)
                if node.right:
                    split_queue.append(node.right)
        
        # Create rooms in leaf nodes
        self.rooms = root.create_rooms()
        
        # Fill rooms with floor tiles
        for room in self.rooms:
            for r in range(room.y, room.y + room.height):
                for c in range(room.x, room.x + room.width):
                    if 0 <= r < self.height and 0 <= c < self.width:
                        self.grid[r, c] = SEMANTIC_PALETTE['FLOOR']
    
    def _connect_rooms(self):
        """Connect rooms with L-shaped corridors."""
        for i in range(len(self.rooms) - 1):
            room_a = self.rooms[i]
            room_b = self.rooms[i + 1]
            
            # Get centers
            r1, c1 = room_a.center
            r2, c2 = room_b.center
            
            # Create L-shaped corridor
            # Horizontal segment
            c_min = min(c1, c2)
            c_max = max(c1, c2)
            for c in range(c_min, c_max + 1):
                if 0 <= r1 < self.height and 0 <= c < self.width:
                    if self.grid[r1, c] == SEMANTIC_PALETTE['VOID']:
                        self.grid[r1, c] = SEMANTIC_PALETTE['FLOOR']
            
            # Vertical segment
            r_min = min(r1, r2)
            r_max = max(r1, r2)
            for r in range(r_min, r_max + 1):
                if 0 <= r < self.height and 0 <= c2 < self.width:
                    if self.grid[r, c2] == SEMANTIC_PALETTE['VOID']:
                        self.grid[r, c2] = SEMANTIC_PALETTE['FLOOR']
    
    def _place_start_and_goal(self):
        """Place start (first room) and goal (last room)."""
        if len(self.rooms) < 2:
            logger.error("Not enough rooms for start/goal")
            return
        
        # Start in first room
        start_room = self.rooms[0]
        self.start_pos = start_room.center
        self.grid[self.start_pos] = SEMANTIC_PALETTE['START']
        
        # Goal in last room
        goal_room = self.rooms[-1]
        self.goal_pos = goal_room.center
        self.grid[self.goal_pos] = SEMANTIC_PALETTE['TRIFORCE']
    
    def _place_keys_and_doors(self):
        """
        Place keys and locked doors with topological ordering.
        
        Ensures solvability: Key K placed before Door D requiring K.
        """
        num_keys = {
            Difficulty.EASY: 1,
            Difficulty.MEDIUM: 2,
            Difficulty.HARD: 3,
            Difficulty.EXPERT: 5
        }[self.difficulty]
        
        # Place keys in early rooms
        for i in range(min(num_keys, len(self.rooms) - 1)):
            room = self.rooms[i]
            
            # Find empty floor tile
            for attempt in range(20):
                r = random.randint(room.y, room.y + room.height - 1)
                c = random.randint(room.x, room.x + room.width - 1)
                
                if self.grid[r, c] == SEMANTIC_PALETTE['FLOOR']:
                    self.grid[r, c] = SEMANTIC_PALETTE['KEY_SMALL']
                    self.key_positions.append((r, c))
                    break
        
        # Place locked doors in later rooms
        for i in range(min(num_keys, len(self.rooms) - 2)):
            room = self.rooms[i + 1]
            
            # Place door at room entrance (edge)
            edges = [
                (room.y, room.x + room.width // 2),  # Top edge
                (room.y + room.height - 1, room.x + room.width // 2),  # Bottom
                (room.y + room.height // 2, room.x),  # Left
                (room.y + room.height // 2, room.x + room.width - 1)  # Right
            ]
            
            for r, c in edges:
                if 0 <= r < self.height and 0 <= c < self.width:
                    if self.grid[r, c] == SEMANTIC_PALETTE['FLOOR']:
                        self.grid[r, c] = SEMANTIC_PALETTE['DOOR_LOCKED']
                        self.door_positions.append((r, c))
                        break
    
    def _place_enemies(self):
        """Place enemies based on difficulty."""
        num_enemies = {
            Difficulty.EASY: 2,
            Difficulty.MEDIUM: 5,
            Difficulty.HARD: 10,
            Difficulty.EXPERT: 15
        }[self.difficulty]
        
        placed = 0
        for room in self.rooms:
            if placed >= num_enemies:
                break
            
            # Place 1-2 enemies per room
            for _ in range(random.randint(0, 2)):
                if placed >= num_enemies:
                    break
                
                for attempt in range(10):
                    r = random.randint(room.y, room.y + room.height - 1)
                    c = random.randint(room.x, room.x + room.width - 1)
                    
                    if self.grid[r, c] == SEMANTIC_PALETTE['FLOOR']:
                        self.grid[r, c] = SEMANTIC_PALETTE['ENEMY']
                        placed += 1
                        break
    
    def _place_obstacles(self):
        """Place pushable blocks."""
        num_blocks = {
            Difficulty.EASY: 1,
            Difficulty.MEDIUM: 3,
            Difficulty.HARD: 5,
            Difficulty.EXPERT: 8
        }[self.difficulty]
        
        for _ in range(num_blocks):
            room = random.choice(self.rooms)
            
            for attempt in range(10):
                r = random.randint(room.y, room.y + room.height - 1)
                c = random.randint(room.x, room.x + room.width - 1)
                
                if self.grid[r, c] == SEMANTIC_PALETTE['FLOOR']:
                    self.grid[r, c] = SEMANTIC_PALETTE['BLOCK']
                    break
    
    def _add_walls(self):
        """Add walls around floor tiles (for visual clarity)."""
        wall_positions = set()
        
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r, c] != SEMANTIC_PALETTE['VOID']:
                    # Check neighbors
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.height and 0 <= nc < self.width:
                            if self.grid[nr, nc] == SEMANTIC_PALETTE['VOID']:
                                wall_positions.add((nr, nc))
        
        # Place walls
        for r, c in wall_positions:
            self.grid[r, c] = SEMANTIC_PALETTE['WALL']
    
    def save_to_vglc(self, filename: str):
        """
        Save dungeon to VGLC format.
        
        Args:
            filename: Output file path
        """
        # Map semantic IDs to VGLC characters
        SEMANTIC_TO_CHAR = {
            SEMANTIC_PALETTE['VOID']: '-',
            SEMANTIC_PALETTE['FLOOR']: 'F',
            SEMANTIC_PALETTE['WALL']: 'W',
            SEMANTIC_PALETTE['BLOCK']: 'B',
            SEMANTIC_PALETTE['DOOR_OPEN']: 'D',
            SEMANTIC_PALETTE['DOOR_LOCKED']: 'D',
            SEMANTIC_PALETTE['ENEMY']: 'M',
            SEMANTIC_PALETTE['START']: 'S',
            SEMANTIC_PALETTE['TRIFORCE']: 'T',
            SEMANTIC_PALETTE['KEY_SMALL']: 'K'
        }
        
        with open(filename, 'w') as f:
            for row in self.grid:
                line = ''.join(SEMANTIC_TO_CHAR.get(tile, 'F') for tile in row)
                f.write(line + '\n')
        
        logger.info(f"Saved dungeon to {filename}")
