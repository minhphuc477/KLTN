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
        Compute 2D positions for rooms based on graph connectivity.
        
        Uses BFS from start node to assign positions.
        
        Args:
            graph: Room connectivity graph
            
        Returns:
            Dict mapping room_id -> (row, col) in layout
        """
        if graph.number_of_nodes() == 0:
            return {}
        
        # Find start node (node with is_start=True)
        start_node = None
        for node, attrs in graph.nodes(data=True):
            if attrs.get('is_start', False):
                start_node = node
                break
        
        # Fallback to node 0 or first node
        if start_node is None:
            start_node = min(graph.nodes())
        
        # BFS to assign positions
        positions = {start_node: (0, 0)}
        visited = {start_node}
        queue = [start_node]
        
        # Direction offsets: (dr, dc)
        # We'll infer directions based on edge traversal order
        direction_cycle = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # E, S, W, N
        
        while queue:
            current = queue.pop(0)
            curr_pos = positions[current]
            
            neighbors = list(graph.neighbors(current))
            dir_idx = 0
            
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                
                # Assign position based on direction
                dr, dc = direction_cycle[dir_idx % 4]
                new_pos = (curr_pos[0] + dr, curr_pos[1] + dc)
                
                # Check for collision and adjust if needed
                while new_pos in positions.values():
                    dir_idx += 1
                    dr, dc = direction_cycle[dir_idx % 4]
                    new_pos = (curr_pos[0] + dr, curr_pos[1] + dc)
                
                positions[neighbor] = new_pos
                visited.add(neighbor)
                queue.append(neighbor)
                dir_idx += 1
        
        # Normalize positions to start from (0, 0)
        if positions:
            min_r = min(p[0] for p in positions.values())
            min_c = min(p[1] for p in positions.values())
            positions = {k: (v[0] - min_r, v[1] - min_c) for k, v in positions.items()}
        
        return positions
    
    def stitch(self, dungeon_data: 'DungeonData' = None) -> StitchedDungeon:
        """
        Stitch rooms into a global dungeon map.
        
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
        
        # Calculate global grid dimensions
        max_row = max(p[0] for p in layout.values())
        max_col = max(p[1] for p in layout.values())
        
        grid_height = (max_row + 1) * (self.ROOM_HEIGHT + self.PADDING)
        grid_width = (max_col + 1) * (self.ROOM_WIDTH + self.PADDING)
        
        # Initialize global grid with VOID
        global_grid = np.full((grid_height, grid_width), 
                              SEMANTIC_PALETTE['VOID'], dtype=np.int64)
        
        room_positions = {}
        room_bounds = {}
        start_pos = None
        goal_pos = None
        
        # Place each room
        for room_id, (layout_row, layout_col) in layout.items():
            room_key = str(room_id)
            
            if room_key not in self.dungeon_data.rooms:
                continue
            
            room = self.dungeon_data.rooms[room_key]
            room_grid = room.grid
            
            # Calculate position in global grid
            r_offset = layout_row * (self.ROOM_HEIGHT + self.PADDING)
            c_offset = layout_col * (self.ROOM_WIDTH + self.PADDING)
            
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
    
    def _connect_doors(self, global_grid: np.ndarray, 
                       layout: Dict[int, Tuple[int, int]],
                       room_bounds: Dict[str, Tuple[int, int, int, int]]):
        """
        Connect doors between adjacent rooms by adding floor tiles.
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
                
                # Check if rooms are adjacent
                if row_a == row_b and abs(col_a - col_b) == 1:
                    # Horizontally adjacent - connect east/west doors
                    self._connect_horizontal(global_grid, bounds_a, bounds_b, col_a < col_b)
                elif col_a == col_b and abs(row_a - row_b) == 1:
                    # Vertically adjacent - connect north/south doors
                    self._connect_vertical(global_grid, bounds_a, bounds_b, row_a < row_b)
    
    def _connect_horizontal(self, global_grid: np.ndarray,
                           bounds_a: Tuple[int, int, int, int],
                           bounds_b: Tuple[int, int, int, int],
                           a_is_left: bool):
        """Connect horizontally adjacent rooms."""
        r1_a, c1_a, r2_a, c2_a = bounds_a
        r1_b, c1_b, r2_b, c2_b = bounds_b
        
        if a_is_left:
            # Find doors on right edge of A and left edge of B
            connect_col_start = c2_a
            connect_col_end = c1_b
        else:
            connect_col_start = c2_b
            connect_col_end = c1_a
        
        # Find the row range to connect
        connect_r1 = max(r1_a, r1_b)
        connect_r2 = min(r2_a, r2_b)
        
        # Connect with floor tiles
        if connect_col_start < connect_col_end:
            for c in range(connect_col_start, connect_col_end):
                for r in range(connect_r1 + 2, connect_r2 - 2):  # Avoid corners
                    if global_grid[r, c] == SEMANTIC_PALETTE['VOID']:
                        global_grid[r, c] = SEMANTIC_PALETTE['FLOOR']
    
    def _connect_vertical(self, global_grid: np.ndarray,
                         bounds_a: Tuple[int, int, int, int],
                         bounds_b: Tuple[int, int, int, int],
                         a_is_top: bool):
        """Connect vertically adjacent rooms."""
        r1_a, c1_a, r2_a, c2_a = bounds_a
        r1_b, c1_b, r2_b, c2_b = bounds_b
        
        if a_is_top:
            connect_row_start = r2_a
            connect_row_end = r1_b
        else:
            connect_row_start = r2_b
            connect_row_end = r1_a
        
        # Find the col range to connect
        connect_c1 = max(c1_a, c1_b)
        connect_c2 = min(c2_a, c2_b)
        
        # Connect with floor tiles
        if connect_row_start < connect_row_end:
            for r in range(connect_row_start, connect_row_end):
                for c in range(connect_c1 + 2, connect_c2 - 2):  # Avoid corners
                    if global_grid[r, c] == SEMANTIC_PALETTE['VOID']:
                        global_grid[r, c] = SEMANTIC_PALETTE['FLOOR']
    
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
