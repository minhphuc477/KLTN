"""
DUNGEON STITCHER
================
Combines individual rooms into a global dungeon map for cross-room pathfinding.

This module:
1. Stitches rooms based on graph connectivity
2. Connects doors between adjacent rooms
3. Creates a unified semantic grid for full dungeon validation

Author: KLTN Thesis Project
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Import semantic palette
try:
    from data.adapter import SEMANTIC_PALETTE, ID_TO_NAME, RoomData, DungeonData
except ImportError:
    try:
        from adapter import SEMANTIC_PALETTE, ID_TO_NAME, RoomData, DungeonData
    except ImportError:
        # Fallback definitions
        SEMANTIC_PALETTE = {
            'VOID': 0, 'FLOOR': 1, 'WALL': 2, 'BLOCK': 3,
            'DOOR_OPEN': 10, 'DOOR_LOCKED': 11, 'DOOR_BOMB': 12,
            'DOOR_PUZZLE': 13, 'DOOR_BOSS': 14, 'DOOR_SOFT': 15,
            'ENEMY': 20, 'START': 21, 'TRIFORCE': 22, 'BOSS': 23,
            'KEY_SMALL': 30, 'KEY_BOSS': 31, 'KEY_ITEM': 32, 'ITEM_MINOR': 33,
            'ELEMENT': 40, 'ELEMENT_FLOOR': 41, 'STAIR': 42, 'PUZZLE': 43,
        }
        ID_TO_NAME = {v: k for k, v in SEMANTIC_PALETTE.items()}
        
        # Fallback for RoomData and DungeonData classes
        from dataclasses import field
        
        @dataclass
        class RoomData:
            """Fallback RoomData class."""
            room_id: str
            grid: np.ndarray                    # Semantic grid [H, W]
            contents: List[str] = field(default_factory=list)  # Items in room
            doors: Dict[str, Dict] = field(default_factory=dict)  # Door info by direction
            position: Tuple[int, int] = (0, 0)  # Position in dungeon layout
        
        @dataclass
        class DungeonData:
            """Fallback DungeonData class."""
            dungeon_id: str
            rooms: Dict[str, RoomData]
            graph: nx.DiGraph


@dataclass
class StitchedDungeon:
    """Result of stitching a dungeon."""
    dungeon_id: str
    global_grid: np.ndarray                    # Full stitched semantic grid
    room_positions: Dict[str, Tuple[int, int]] # room_id -> (row_offset, col_offset) in global grid
    start_pos: Optional[Tuple[int, int]]       # Global START position
    goal_pos: Optional[Tuple[int, int]]        # Global TRIFORCE position
    room_bounds: Dict[str, Tuple[int, int, int, int]]  # room_id -> (r1, c1, r2, c2)


class DungeonStitcher:
    """
    Stitches individual rooms into a global dungeon map.
    
    Uses graph topology to determine room layout and connects doors.
    """
    
    # Standard room dimensions
    ROOM_HEIGHT = 16
    ROOM_WIDTH = 11
    PADDING = 1  # Padding between rooms
    
    def __init__(self, dungeon_data: 'DungeonData' = None):
        """
        Initialize the stitcher.
        
        Args:
            dungeon_data: Optional DungeonData object to stitch
        """
        self.dungeon_data = dungeon_data
        self.layout_grid = None  # 2D grid of room IDs
        self.global_grid = None
        
    def compute_layout(self, graph: nx.DiGraph) -> Dict[int, Tuple[int, int]]:
        """
        ADVANCED LAYOUT INFERENCE using graph topology + BFS.
        
        This algorithm computes a TRUE spatial layout by analyzing:
        1. Graph connectivity patterns
        2. Edge directionality hints (if available)
        3. BFS traversal order
        4. Collision avoidance
        
        REPLACES naive ID-based heuristics entirely.
        
        Args:
            graph: Room connectivity graph
            
        Returns:
            Dict mapping room_id -> (row, col) in layout
        """
        if graph.number_of_nodes() == 0:
            return {}

        positions: Dict[int, Tuple[int, int]] = {}
        visited = set()
        queue: List[int] = []

        # STEP 1: Use explicit position hints if available (from adapter.py)
        if self.dungeon_data is not None and self.dungeon_data.rooms:
            hinted = {}
            for room_key, room in self.dungeon_data.rooms.items():
                try:
                    node_id = int(room_key)
                except ValueError:
                    continue
                hinted[node_id] = room.position

            # If we have meaningful hints (not all zeros), use them as anchors
            if hinted and not all(pos == (0, 0) for pos in hinted.values()):
                min_r = min(p[0] for p in hinted.values())
                min_c = min(p[1] for p in hinted.values())
                normalized = {nid: (pos[0] - min_r, pos[1] - min_c) for nid, pos in hinted.items()}
                positions.update(normalized)
                visited.update(normalized.keys())
                queue.extend(normalized.keys())
        
        # STEP 2: Find start node (node with is_start=True)
        start_node = None
        for node, attrs in graph.nodes(data=True):
            if attrs.get('is_start', False):
                start_node = node
                break
        
        if start_node is None:
            start_node = min(graph.nodes())

        if start_node not in positions:
            positions[start_node] = (0, 0)
            visited.add(start_node)
            queue.append(start_node)
        elif start_node not in queue:
            queue.append(start_node)
        
        # STEP 3: BFS with intelligent direction assignment
        # We use edge attributes and neighbor analysis to infer spatial relationships
        
        while queue:
            current = queue.pop(0)
            curr_pos = positions[current]
            
            neighbors = list(graph.neighbors(current))
            
            # Sort neighbors by some heuristic (e.g., node ID) for consistency
            neighbors.sort()
            
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                
                # STEP 3A: Try to infer direction from graph structure
                # Look at current node's other neighbors to guess spatial arrangement
                direction = self._infer_direction_from_topology(
                    graph, current, neighbor, positions, visited
                )
                
                # STEP 3B: Find a valid position in that direction
                new_pos = self._find_valid_position(
                    curr_pos, direction, positions
                )
                
                if new_pos:
                    positions[neighbor] = new_pos
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # STEP 4: Normalize positions to start from (0, 0)
        if positions:
            min_r = min(p[0] for p in positions.values())
            min_c = min(p[1] for p in positions.values())
            positions = {k: (v[0] - min_r, v[1] - min_c) for k, v in positions.items()}
        
        return positions
    
    def _infer_direction_from_topology(self, graph: nx.DiGraph, current: int, 
                                       neighbor: int, positions: Dict[int, Tuple[int, int]],
                                       visited: set) -> str:
        """
        Infer the spatial direction to place neighbor relative to current.
        
        Uses graph topology analysis:
        - Count neighbors in each direction
        - Prefer directions that maintain grid-like structure
        - Avoid directions that would cause collisions
        
        Returns: 'north', 'south', 'east', or 'west'
        """
        curr_pos = positions[current]
        
        # Analyze which directions are already occupied by current's neighbors
        occupied_directions = set()
        for other_neighbor in graph.neighbors(current):
            if other_neighbor in positions and other_neighbor != neighbor:
                other_pos = positions[other_neighbor]
                dr = other_pos[0] - curr_pos[0]
                dc = other_pos[1] - curr_pos[1]
                
                if dr < 0:
                    occupied_directions.add('north')
                elif dr > 0:
                    occupied_directions.add('south')
                if dc < 0:
                    occupied_directions.add('west')
                elif dc > 0:
                    occupied_directions.add('east')
        
        # Prefer directions that are not occupied
        available_directions = ['east', 'south', 'west', 'north']
        for d in available_directions:
            if d not in occupied_directions:
                return d
        
        # Fallback: return east (default expansion direction)
        return 'east'
    
    def _find_valid_position(self, curr_pos: Tuple[int, int], direction: str,
                            positions: Dict[int, Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """
        Find a valid (unoccupied) position in the given direction.
        
        Tries the immediate neighbor first, then expands outward.
        """
        direction_map = {
            'north': (-1, 0),
            'south': (1, 0),
            'east': (0, 1),
            'west': (0, -1),
        }
        
        dr, dc = direction_map.get(direction, (0, 1))
        
        # Try immediate neighbor first
        for distance in range(1, 10):  # Max distance of 10
            new_pos = (curr_pos[0] + dr * distance, curr_pos[1] + dc * distance)
            if new_pos not in positions.values():
                return new_pos
        
        # If all positions in that direction are occupied, try adjacent directions
        adjacent_dirs = {
            'north': ['east', 'west'],
            'south': ['east', 'west'],
            'east': ['north', 'south'],
            'west': ['north', 'south'],
        }
        
        for alt_dir in adjacent_dirs.get(direction, ['east']):
            alt_dr, alt_dc = direction_map[alt_dir]
            for distance in range(1, 5):
                new_pos = (curr_pos[0] + alt_dr * distance, curr_pos[1] + alt_dc * distance)
                if new_pos not in positions.values():
                    return new_pos
        
        # Last resort: find any unoccupied position
        for r_offset in range(-5, 6):
            for c_offset in range(-5, 6):
                new_pos = (curr_pos[0] + r_offset, curr_pos[1] + c_offset)
                if new_pos not in positions.values():
                    return new_pos
        
        return None
    
    def stitch(self, dungeon_data: 'DungeonData' = None) -> StitchedDungeon:
        """
        Stitch rooms into a global dungeon map.
        
        FIX: Use actual room dimensions instead of fixed constants to prevent overlap.
        
        Args:
            dungeon_data: DungeonData to stitch (uses self.dungeon_data if not provided)
            
        Returns:
            StitchedDungeon object with global grid
        """
        if dungeon_data is not None:
            self.dungeon_data = dungeon_data
        
        if self.dungeon_data is None:
            raise ValueError("No dungeon data provided")
        
        # Compute layout from graph
        layout = self.compute_layout(self.dungeon_data.graph)
        self.layout = layout  # Save for debugging
        
        if not layout:
            # No layout - return empty result
            return StitchedDungeon(
                dungeon_id=self.dungeon_data.dungeon_id,
                global_grid=np.zeros((1, 1), dtype=np.int64),
                room_positions={},
                start_pos=None,
                goal_pos=None,
                room_bounds={}
            )
        
        # FIX: Calculate positions using ACTUAL room dimensions
        # First, find the max room dimensions in each row and column
        default_h, default_w = self._default_room_shape()
        rooms_by_layout = {}
        for room_id, (layout_row, layout_col) in layout.items():
            room_key = str(room_id)
            if room_key in self.dungeon_data.rooms:
                room = self.dungeon_data.rooms[room_key]
                grid = room.grid
            else:
                grid = self._create_ghost_room(room_id, self.dungeon_data.graph, default_h, default_w)
                ghost_room = RoomData(
                    room_id=room_key,
                    grid=grid,
                    contents=self.dungeon_data.graph.nodes[room_id].get('contents', []) if room_id in self.dungeon_data.graph.nodes else [],
                    doors={},
                    position=(layout_row, layout_col)
                )
                self.dungeon_data.rooms[room_key] = ghost_room
                room = ghost_room

            rooms_by_layout[(layout_row, layout_col)] = {
                'room_id': room_id,
                'room_key': room_key,
                'grid': grid,
                'height': grid.shape[0],
                'width': grid.shape[1],
            }

        # Calculate max height per row and max width per column
        max_row = max(p[0] for p in layout.values())
        max_col = max(p[1] for p in layout.values())
        
        row_heights = {}
        col_widths = {}
        
        for (layout_row, layout_col), room_info in rooms_by_layout.items():
            if layout_row not in row_heights or room_info['height'] > row_heights[layout_row]:
                row_heights[layout_row] = room_info['height']
            if layout_col not in col_widths or room_info['width'] > col_widths[layout_col]:
                col_widths[layout_col] = room_info['width']
        
        # Fill in missing rows/columns with default sizes
        for r in range(max_row + 1):
            if r not in row_heights:
                row_heights[r] = default_h
        for c in range(max_col + 1):
            if c not in col_widths:
                col_widths[c] = default_w
        
        # Calculate cumulative offsets
        row_offsets = {0: 0}
        for r in range(1, max_row + 1):
            row_offsets[r] = row_offsets[r-1] + row_heights[r-1] + self.PADDING
        
        col_offsets = {0: 0}
        for c in range(1, max_col + 1):
            col_offsets[c] = col_offsets[c-1] + col_widths[c-1] + self.PADDING
        
        # Calculate grid dimensions
        grid_height = row_offsets[max_row] + row_heights[max_row]
        grid_width = col_offsets[max_col] + col_widths[max_col]
        
        # Initialize global grid with VOID
        global_grid = np.full((grid_height, grid_width), 
                              SEMANTIC_PALETTE['VOID'], dtype=np.int64)
        
        room_positions = {}
        room_bounds = {}
        start_pos = None
        goal_pos = None
        
        # Place each room using calculated offsets
        for room_id, (layout_row, layout_col) in layout.items():
            room_key = str(room_id)
            
            if room_key not in self.dungeon_data.rooms:
                continue
            
            room = self.dungeon_data.rooms[room_key]
            room_grid = room.grid
            
            # Use calculated offsets instead of fixed multiplier
            r_offset = row_offsets[layout_row]
            c_offset = col_offsets[layout_col]
            
            # Get actual room dimensions
            rh, rw = room_grid.shape
            
            # Place room in global grid
            r_end = min(r_offset + rh, grid_height)
            c_end = min(c_offset + rw, grid_width)
            
            actual_rh = r_end - r_offset
            actual_rw = c_end - c_offset
            
            global_grid[r_offset:r_end, c_offset:c_end] = room_grid[:actual_rh, :actual_rw]
            
            room_positions[room_key] = (r_offset, c_offset)
            room_bounds[room_key] = (r_offset, c_offset, r_end, c_end)
            
            # Find START and TRIFORCE positions
            for r in range(actual_rh):
                for c in range(actual_rw):
                    tile = room_grid[r, c]
                    global_r = r_offset + r
                    global_c = c_offset + c
                    
                    if tile == SEMANTIC_PALETTE['START']:
                        start_pos = (global_r, global_c)
                    elif tile == SEMANTIC_PALETTE['TRIFORCE']:
                        goal_pos = (global_r, global_c)
        
        # Connect doors between adjacent rooms
        self._connect_doors(global_grid, layout, room_bounds)
        
        self.global_grid = global_grid
        
        return StitchedDungeon(
            dungeon_id=self.dungeon_data.dungeon_id,
            global_grid=global_grid,
            room_positions=room_positions,
            start_pos=start_pos,
            goal_pos=goal_pos,
            room_bounds=room_bounds
        )

    def _default_room_shape(self) -> Tuple[int, int]:
        """Return a reasonable room size based on existing data or defaults."""
        if self.dungeon_data and self.dungeon_data.rooms:
            shapes = [room.grid.shape for room in self.dungeon_data.rooms.values() if getattr(room, 'grid', None) is not None]
            if shapes:
                heights = [s[0] for s in shapes]
                widths = [s[1] for s in shapes]
                return int(np.median(heights)), int(np.median(widths))
        return self.ROOM_HEIGHT, self.ROOM_WIDTH

    def _create_ghost_room(self, node_id: int, graph: nx.DiGraph,
                           height: int, width: int) -> np.ndarray:
        """Synthesize a generic traversable room when map data is missing."""
        room = np.full((height, width), SEMANTIC_PALETTE['FLOOR'], dtype=np.int64)

        room[0, :] = SEMANTIC_PALETTE['WALL']
        room[-1, :] = SEMANTIC_PALETTE['WALL']
        room[:, 0] = SEMANTIC_PALETTE['WALL']
        room[:, -1] = SEMANTIC_PALETTE['WALL']

        center_r, center_c = height // 2, width // 2
        node_attrs = graph.nodes[node_id] if node_id in graph.nodes else {}

        if node_attrs.get('is_start'):
            room[center_r, center_c] = SEMANTIC_PALETTE['START']
        elif node_attrs.get('has_triforce'):
            room[center_r, center_c] = SEMANTIC_PALETTE['TRIFORCE']

        # Place keys if the node implies them
        if node_attrs.get('has_key') and width > 2:
            room[center_r, max(1, center_c - 1)] = SEMANTIC_PALETTE['KEY_SMALL']
        if node_attrs.get('has_boss_key') and width > 3:
            room[center_r, min(width - 2, center_c + 1)] = SEMANTIC_PALETTE['KEY_BOSS']

        # Provide doorway anchors in all four directions
        if center_c < width - 1:
            room[center_r, width - 1] = SEMANTIC_PALETTE['DOOR_OPEN']
        if center_c > 0:
            room[center_r, 0] = SEMANTIC_PALETTE['DOOR_OPEN']
        if center_r > 0:
            room[0, center_c] = SEMANTIC_PALETTE['DOOR_OPEN']
        if center_r < height - 1:
            room[height - 1, center_c] = SEMANTIC_PALETTE['DOOR_OPEN']

        return room
    
    def _connect_doors(self, global_grid: np.ndarray, 
                       layout: Dict[int, Tuple[int, int]],
                       room_bounds: Dict[str, Tuple[int, int, int, int]]):
        """
        Connect doors between adjacent rooms by:
        1. Punching doorways through walls at room boundaries
        2. Adding floor tiles in gaps between rooms
        
        **CRITICAL FIX: Only connect rooms that are BOTH:**
        - Adjacent in the layout grid
        - Connected in the dungeon graph
        
        This is necessary because VGLC room data doesn't have explicit
        door markers at boundaries that connect to adjacent rooms.
        """
        # For each pair of adjacent rooms in layout
        for room_id_a, (row_a, col_a) in layout.items():
            for room_id_b, (row_b, col_b) in layout.items():
                if room_id_a >= room_id_b:
                    continue
                
                key_a = str(room_id_a)
                key_b = str(room_id_b)
                
                if key_a not in room_bounds or key_b not in room_bounds:
                    continue
                
                bounds_a = room_bounds[key_a]
                bounds_b = room_bounds[key_b]
                
                # **CRITICAL FIX: Check if rooms are connected in graph**
                graph = self.dungeon_data.graph
                is_connected = (
                    graph.has_edge(room_id_a, room_id_b) or 
                    graph.has_edge(room_id_b, room_id_a)
                )
                
                if not is_connected:
                    # Don't connect rooms that aren't linked in the graph
                    continue
                
                # Check if rooms are adjacent in layout grid
                if row_a == row_b and abs(col_a - col_b) == 1:
                    # Horizontally adjacent - connect east/west
                    self._punch_horizontal_doorway(global_grid, bounds_a, bounds_b, col_a < col_b)
                elif col_a == col_b and abs(row_a - row_b) == 1:
                    # Vertically adjacent - connect north/south
                    self._punch_vertical_doorway(global_grid, bounds_a, bounds_b, row_a < row_b)
    
    def _punch_horizontal_doorway(self, global_grid: np.ndarray,
                                  bounds_a: Tuple[int, int, int, int],
                                  bounds_b: Tuple[int, int, int, int],
                                  a_is_left: bool):
        """
        Create a doorway between horizontally adjacent rooms.
        
        Punches through walls and fills gaps with floor tiles.
        """
        r1_a, c1_a, r2_a, c2_a = bounds_a
        r1_b, c1_b, r2_b, c2_b = bounds_b
        
        # Find overlapping row range
        r_start = max(r1_a, r1_b)
        r_end = min(r2_a, r2_b)
        
        if r_start >= r_end:
            return
        
        # Find middle of overlapping region for doorway
        door_r = (r_start + r_end) // 2
        door_width = 3  # Make a 3-tile wide doorway
        
        if a_is_left:
            # Punch through right wall of A and left wall of B
            # Room A ends at c2_a-1, Room B starts at c1_b
            for dr in range(-door_width//2, door_width//2 + 1):
                r = door_r + dr
                if r < r_start + 1 or r >= r_end - 1:
                    continue
                
                # Clear the wall at A's right edge
                if c2_a > 0:
                    global_grid[r, c2_a - 1] = SEMANTIC_PALETTE['FLOOR']
                
                # Fill the gap
                for c in range(c2_a, c1_b + 1):
                    if 0 <= c < global_grid.shape[1]:
                        global_grid[r, c] = SEMANTIC_PALETTE['FLOOR']
                
                # Clear the wall at B's left edge
                if c1_b < global_grid.shape[1]:
                    global_grid[r, c1_b] = SEMANTIC_PALETTE['FLOOR']
        else:
            # Punch through left wall of A and right wall of B
            for dr in range(-door_width//2, door_width//2 + 1):
                r = door_r + dr
                if r < r_start + 1 or r >= r_end - 1:
                    continue
                
                if c1_a < global_grid.shape[1]:
                    global_grid[r, c1_a] = SEMANTIC_PALETTE['FLOOR']
                
                for c in range(c2_b, c1_a + 1):
                    if 0 <= c < global_grid.shape[1]:
                        global_grid[r, c] = SEMANTIC_PALETTE['FLOOR']
                
                if c2_b > 0:
                    global_grid[r, c2_b - 1] = SEMANTIC_PALETTE['FLOOR']
    
    def _punch_vertical_doorway(self, global_grid: np.ndarray,
                                bounds_a: Tuple[int, int, int, int],
                                bounds_b: Tuple[int, int, int, int],
                                a_is_top: bool):
        """
        Create a doorway between vertically adjacent rooms.
        
        Punches through walls and fills gaps with floor tiles.
        """
        r1_a, c1_a, r2_a, c2_a = bounds_a
        r1_b, c1_b, r2_b, c2_b = bounds_b
        
        # Find overlapping column range
        c_start = max(c1_a, c1_b)
        c_end = min(c2_a, c2_b)
        
        if c_start >= c_end:
            return
        
        # Find middle of overlapping region for doorway
        door_c = (c_start + c_end) // 2
        door_width = 3  # Make a 3-tile wide doorway
        
        if a_is_top:
            # Punch through bottom wall of A and top wall of B
            for dc in range(-door_width//2, door_width//2 + 1):
                c = door_c + dc
                if c < c_start + 1 or c >= c_end - 1:
                    continue
                
                # Clear the wall at A's bottom edge
                if r2_a > 0:
                    global_grid[r2_a - 1, c] = SEMANTIC_PALETTE['FLOOR']
                
                # Fill the gap
                for r in range(r2_a, r1_b + 1):
                    if 0 <= r < global_grid.shape[0]:
                        global_grid[r, c] = SEMANTIC_PALETTE['FLOOR']
                
                # Clear the wall at B's top edge
                if r1_b < global_grid.shape[0]:
                    global_grid[r1_b, c] = SEMANTIC_PALETTE['FLOOR']
        else:
            # Punch through top wall of A and bottom wall of B
            for dc in range(-door_width//2, door_width//2 + 1):
                c = door_c + dc
                if c < c_start + 1 or c >= c_end - 1:
                    continue
                
                if r1_a < global_grid.shape[0]:
                    global_grid[r1_a, c] = SEMANTIC_PALETTE['FLOOR']
                
                for r in range(r2_b, r1_a + 1):
                    if 0 <= r < global_grid.shape[0]:
                        global_grid[r, c] = SEMANTIC_PALETTE['FLOOR']
                
                if r2_b > 0:
                    global_grid[r2_b - 1, c] = SEMANTIC_PALETTE['FLOOR']
    
    def _connect_horizontal(self, global_grid: np.ndarray,
                           bounds_a: Tuple[int, int, int, int],
                           bounds_b: Tuple[int, int, int, int],
                           a_is_left: bool):
        """DEPRECATED: Use _punch_horizontal_doorway instead."""
        self._punch_horizontal_doorway(global_grid, bounds_a, bounds_b, a_is_left)
    
    def _connect_vertical(self, global_grid: np.ndarray,
                         bounds_a: Tuple[int, int, int, int],
                         bounds_b: Tuple[int, int, int, int],
                         a_is_top: bool):
        """DEPRECATED: Use _punch_vertical_doorway instead."""
        self._punch_vertical_doorway(global_grid, bounds_a, bounds_b, a_is_top)
    
    @staticmethod
    def visualize_stitched(stitched: StitchedDungeon) -> str:
        """Create ASCII visualization of stitched dungeon."""
        symbol_map = {
            SEMANTIC_PALETTE['VOID']: ' ',
            SEMANTIC_PALETTE['FLOOR']: '.',
            SEMANTIC_PALETTE['WALL']: '#',
            SEMANTIC_PALETTE['BLOCK']: 'B',
            SEMANTIC_PALETTE['DOOR_OPEN']: 'O',
            SEMANTIC_PALETTE['DOOR_LOCKED']: 'L',
            SEMANTIC_PALETTE['DOOR_BOMB']: 'X',
            SEMANTIC_PALETTE['DOOR_PUZZLE']: 'P',
            SEMANTIC_PALETTE['DOOR_BOSS']: 'K',
            SEMANTIC_PALETTE['ENEMY']: 'E',
            SEMANTIC_PALETTE['START']: 'S',
            SEMANTIC_PALETTE['TRIFORCE']: 'T',
            SEMANTIC_PALETTE['KEY_SMALL']: 'k',
            SEMANTIC_PALETTE['KEY_BOSS']: 'K',
            SEMANTIC_PALETTE['ELEMENT']: '~',
            SEMANTIC_PALETTE['STAIR']: 's',
        }
        
        lines = []
        for row in stitched.global_grid:
            line = ''.join(symbol_map.get(cell, '?') for cell in row)
            lines.append(line)
        
        result = '\n'.join(lines)
        result += f'\n\nSTART: {stitched.start_pos}, GOAL: {stitched.goal_pos}'
        result += f'\nDimensions: {stitched.global_grid.shape}'
        
        return result


# ==========================================
# TESTING
# ==========================================

if __name__ == "__main__":
    import sys
    import os
    
    # Add parent dir to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from data.adapter import IntelligentDataAdapter
    from pathlib import Path
    
    # Load test dungeon
    data_root = Path(__file__).parent.parent / "Data" / "The Legend of Zelda"
    
    print(f"Loading data from: {data_root}")
    
    adapter = IntelligentDataAdapter(str(data_root))
    
    # Process dungeon 1
    map_file = data_root / "Processed" / "tloz1_1.txt"
    graph_file = data_root / "Graph Processed" / "LoZ_1.dot"
    
    if map_file.exists() and graph_file.exists():
        print("Processing dungeon 1...")
        dungeon = adapter.process_dungeon(str(map_file), str(graph_file), "zelda_1")
        
        print(f"Rooms: {len(dungeon.rooms)}")
        print(f"Graph nodes: {dungeon.graph.number_of_nodes()}")
        
        # Stitch
        stitcher = DungeonStitcher(dungeon)
        stitched = stitcher.stitch()
        
        print(f"\nStitched dungeon:")
        print(f"  Grid shape: {stitched.global_grid.shape}")
        print(f"  START: {stitched.start_pos}")
        print(f"  TRIFORCE: {stitched.goal_pos}")
        print(f"  Rooms: {list(stitched.room_positions.keys())}")
        
        # Visualize (truncated)
        viz = DungeonStitcher.visualize_stitched(stitched)
        lines = viz.split('\n')
        print("\nVisualization (first 30 lines):")
        for line in lines[:30]:
            print(line[:80])
    else:
        print(f"Test files not found!")
        print(f"  Map: {map_file.exists()}")
        print(f"  Graph: {graph_file.exists()}")
