"""
VGLC-FIRST ADAPTER: Use VGLC positions as ground truth.

Key insight: VGLC is the PHYSICAL dungeon layout.
Graph provides SEMANTIC information (START, TRIFORCE, etc.).

Strategy:
1. Extract all VGLC rooms with their (row, col) positions
2. Build connectivity from VGLC doors (physical adjacency)
3. Identify special rooms: STAIR=START, find TRIFORCE position
4. Convert char grids to semantic grids
5. The stitcher will use VGLC positions directly
"""
import numpy as np
import networkx as nx
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

# Semantic IDs
SEMANTIC_PALETTE = {
    'VOID': 0, 'FLOOR': 1, 'WALL': 2, 'BLOCK': 3,
    'DOOR_OPEN': 10, 'DOOR_LOCKED': 11, 'DOOR_BOMB': 12,
    'ENEMY': 20, 'START': 21, 'TRIFORCE': 22, 'BOSS': 23,
    'KEY_SMALL': 30, 'ITEM_MINOR': 33,
    'ELEMENT': 40, 'ELEMENT_FLOOR': 41, 'STAIR': 42,
}

ID_TO_NAME = {v: k for k, v in SEMANTIC_PALETTE.items()}

CHAR_TO_SEMANTIC = {
    '-': SEMANTIC_PALETTE['VOID'],
    'F': SEMANTIC_PALETTE['FLOOR'],
    '.': SEMANTIC_PALETTE['FLOOR'],
    'W': SEMANTIC_PALETTE['WALL'],
    'B': SEMANTIC_PALETTE['BLOCK'],
    'M': SEMANTIC_PALETTE['ENEMY'],
    'P': SEMANTIC_PALETTE['ELEMENT'],
    'O': SEMANTIC_PALETTE['ELEMENT_FLOOR'],
    'I': SEMANTIC_PALETTE['ELEMENT'],
    'S': SEMANTIC_PALETTE['STAIR'],  # STAIR = START position
    'D': SEMANTIC_PALETTE['DOOR_OPEN'],  # Will be refined later
}

@dataclass
class RoomData:
    room_id: str
    grid: np.ndarray  # Semantic grid
    contents: List[str] = field(default_factory=list)
    doors: Dict[str, Dict] = field(default_factory=dict)
    position: Tuple[int, int] = (0, 0)
    
@dataclass
class DungeonData:
    dungeon_id: str
    rooms: Dict[str, RoomData]
    graph: nx.DiGraph  # VGLC-based connectivity
    layout: np.ndarray
    tpe_vectors: np.ndarray
    p_matrix: np.ndarray
    node_features: np.ndarray


class VGLCFirstAdapter:
    """
    VGLC-First adapter that uses physical room positions as ground truth.
    """
    
    ROOM_WIDTH = 11
    ROOM_HEIGHT = 16
    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
    
    def process_dungeon(self, map_file: str, graph_file: str, dungeon_id: str) -> DungeonData:
        """
        Process dungeon using VGLC positions as ground truth.
        """
        # Step 1: Extract VGLC rooms
        vglc_rooms = self._extract_vglc_rooms(map_file)
        print(f"Extracted {len(vglc_rooms)} VGLC rooms")
        
        # Step 2: Build VGLC connectivity graph (from doors)
        vglc_graph = self._build_vglc_graph(vglc_rooms)
        
        # Step 3: Parse the .dot graph for semantic info
        dot_graph = self._parse_dot_graph(graph_file)
        
        # Step 4: Match graph nodes to VGLC rooms for semantic injection
        semantic_mapping = self._match_semantics(vglc_rooms, dot_graph)
        
        # Step 5: Convert to processed rooms
        processed_rooms: Dict[str, RoomData] = {}
        
        for pos, info in vglc_rooms.items():
            room_id = f"{pos[0]}_{pos[1]}"
            char_grid = info['grid']
            
            # Convert to semantic grid
            semantic_grid = self._char_to_semantic(char_grid)
            
            # Inject special markers from graph matching
            if pos in semantic_mapping:
                sem_info = semantic_mapping[pos]
                if sem_info.get('is_start'):
                    # Already has STAIR which we treat as START
                    pass
                if sem_info.get('has_triforce'):
                    # Inject TRIFORCE marker (replace center floor)
                    self._inject_triforce(semantic_grid)
            
            room = RoomData(
                room_id=room_id,
                grid=semantic_grid,
                contents=[],
                doors=info['doors'],
                position=pos
            )
            processed_rooms[room_id] = room
        
        # Create dungeon data
        return DungeonData(
            dungeon_id=dungeon_id,
            rooms=processed_rooms,
            graph=vglc_graph,
            layout=np.zeros((1, 1)),
            tpe_vectors=np.zeros((len(processed_rooms), 1)),
            p_matrix=np.zeros((len(processed_rooms), len(processed_rooms))),
            node_features=np.zeros((len(processed_rooms), 1))
        )
    
    def _extract_vglc_rooms(self, filepath: str) -> Dict[Tuple[int, int], dict]:
        """Extract rooms from VGLC file."""
        with open(filepath) as f:
            lines = [line.rstrip('\n') for line in f]
        
        rooms = {}
        for slot_r in range(10):  # Up to 10 rows
            for slot_c in range(10):  # Up to 10 cols
                start_row = slot_r * self.ROOM_HEIGHT
                start_col = slot_c * self.ROOM_WIDTH
                
                if start_row >= len(lines):
                    continue
                
                room_chars = []
                for r in range(start_row, min(start_row + self.ROOM_HEIGHT, len(lines))):
                    row_chars = []
                    line = lines[r] if r < len(lines) else ''
                    for c in range(start_col, start_col + self.ROOM_WIDTH):
                        char = line[c] if c < len(line) else '-'
                        row_chars.append(char)
                    room_chars.append(row_chars)
                
                room_grid = np.array(room_chars)
                
                # Check if valid room (not all void)
                if np.sum(room_grid == '-') < room_grid.size * 0.7:
                    # Analyze doors
                    has_north = 'D' in ''.join(room_grid[0:2, :].flatten())
                    has_south = 'D' in ''.join(room_grid[-2:, :].flatten())
                    has_west = 'D' in ''.join(room_grid[:, 0:2].flatten())
                    has_east = 'D' in ''.join(room_grid[:, -2:].flatten())
                    
                    rooms[(slot_r, slot_c)] = {
                        'grid': room_grid,
                        'has_stair': np.any(room_grid == 'S'),
                        'has_monster': np.any(room_grid == 'M'),
                        'doors': {'N': has_north, 'S': has_south, 'W': has_west, 'E': has_east}
                    }
        
        return rooms
    
    def _build_vglc_graph(self, vglc_rooms: dict) -> nx.DiGraph:
        """Build connectivity graph from VGLC doors."""
        G = nx.DiGraph()
        
        for pos, info in vglc_rooms.items():
            room_id = f"{pos[0]}_{pos[1]}"
            
            # Add node with attributes
            G.add_node(room_id, 
                      position=pos,
                      has_stair=info['has_stair'],
                      has_monster=info['has_monster'])
            
            # Add edges based on doors
            r, c = pos
            doors = info['doors']
            
            if doors['N'] and (r-1, c) in vglc_rooms:
                neighbor_id = f"{r-1}_{c}"
                G.add_edge(room_id, neighbor_id, direction='N')
                G.add_edge(neighbor_id, room_id, direction='S')
            
            if doors['S'] and (r+1, c) in vglc_rooms:
                neighbor_id = f"{r+1}_{c}"
                G.add_edge(room_id, neighbor_id, direction='S')
                G.add_edge(neighbor_id, room_id, direction='N')
            
            if doors['W'] and (r, c-1) in vglc_rooms:
                neighbor_id = f"{r}_{c-1}"
                G.add_edge(room_id, neighbor_id, direction='W')
                G.add_edge(neighbor_id, room_id, direction='E')
            
            if doors['E'] and (r, c+1) in vglc_rooms:
                neighbor_id = f"{r}_{c+1}"
                G.add_edge(room_id, neighbor_id, direction='E')
                G.add_edge(neighbor_id, room_id, direction='W')
        
        return G
    
    def _parse_dot_graph(self, filepath: str) -> nx.DiGraph:
        """Parse .dot graph file."""
        return nx.drawing.nx_pydot.read_dot(filepath)
    
    def _match_semantics(self, vglc_rooms: dict, dot_graph: nx.DiGraph) -> Dict[Tuple[int, int], dict]:
        """
        Match semantic info from .dot graph to VGLC rooms.
        
        Strategy: Use content clues from both sources.
        """
        mapping = {}
        
        # Find START: VGLC room with STAIR
        start_vglc = None
        for pos, info in vglc_rooms.items():
            if info['has_stair']:
                start_vglc = pos
                mapping[pos] = {'is_start': True}
                break
        
        # Find TRIFORCE: In Zelda dungeons, typically the room farthest from start
        # or a room accessible only after boss
        if start_vglc:
            # BFS to find farthest room
            G = self._build_vglc_graph(vglc_rooms)
            start_id = f"{start_vglc[0]}_{start_vglc[1]}"
            
            distances = nx.single_source_shortest_path_length(G, start_id)
            farthest = max(distances.items(), key=lambda x: x[1])
            farthest_id = farthest[0]
            farthest_pos = tuple(map(int, farthest_id.split('_')))
            
            # The triforce room is often after the boss room
            # Look for the second-farthest room as potential boss
            # and farthest as triforce (if it only has 1 door)
            farthest_info = vglc_rooms.get(farthest_pos)
            if farthest_info:
                door_count = sum(farthest_info['doors'].values())
                if door_count == 1:  # Dead-end room = likely triforce
                    mapping[farthest_pos] = mapping.get(farthest_pos, {})
                    mapping[farthest_pos]['has_triforce'] = True
        
        return mapping
    
    def _char_to_semantic(self, char_grid: np.ndarray) -> np.ndarray:
        """Convert character grid to semantic ID grid."""
        h, w = char_grid.shape
        semantic = np.zeros((h, w), dtype=np.int64)
        
        for r in range(h):
            for c in range(w):
                char = char_grid[r, c]
                semantic[r, c] = CHAR_TO_SEMANTIC.get(char, SEMANTIC_PALETTE['VOID'])
        
        return semantic
    
    def _inject_triforce(self, grid: np.ndarray):
        """Inject TRIFORCE marker into grid center."""
        h, w = grid.shape
        center_r, center_c = h // 2, w // 2
        
        # Find nearest floor tile to center
        for dr in range(max(h, w)):
            for r in range(center_r - dr, center_r + dr + 1):
                for c in range(center_c - dr, center_c + dr + 1):
                    if 0 <= r < h and 0 <= c < w:
                        if grid[r, c] == SEMANTIC_PALETTE['FLOOR']:
                            grid[r, c] = SEMANTIC_PALETTE['TRIFORCE']
                            return


# Test
if __name__ == '__main__':
    adapter = VGLCFirstAdapter('Data')
    
    map_file = 'Data/The Legend of Zelda/Processed/tloz1_1.txt'
    graph_file = 'Data/The Legend of Zelda/Graph Processed/LoZ_1.dot'
    
    dungeon = adapter.process_dungeon(map_file, graph_file, 'tloz1_1')
    
    print(f"\nDungeon: {dungeon.dungeon_id}")
    print(f"Rooms: {len(dungeon.rooms)}")
    print(f"Graph nodes: {dungeon.graph.number_of_nodes()}")
    print(f"Graph edges: {dungeon.graph.number_of_edges()}")
    
    # Find START
    for rid, room in dungeon.rooms.items():
        if SEMANTIC_PALETTE['STAIR'] in room.grid:
            print(f"\nSTART room: {rid} at position {room.position}")
            # Show START location
            start_locs = np.where(room.grid == SEMANTIC_PALETTE['STAIR'])
            print(f"  STAIR tiles: {list(zip(start_locs[0], start_locs[1]))}")
    
    # Find TRIFORCE
    for rid, room in dungeon.rooms.items():
        if SEMANTIC_PALETTE['TRIFORCE'] in room.grid:
            print(f"\nTRIFORCE room: {rid} at position {room.position}")
            tri_locs = np.where(room.grid == SEMANTIC_PALETTE['TRIFORCE'])
            print(f"  TRIFORCE tiles: {list(zip(tri_locs[0], tri_locs[1]))}")
