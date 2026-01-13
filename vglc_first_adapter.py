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
        Creates a ghost TRIFORCE room connected to BOSS if needed.
        """
        # Step 1: Extract VGLC rooms
        vglc_rooms = self._extract_vglc_rooms(map_file)
        print(f"Extracted {len(vglc_rooms)} VGLC rooms")
        
        # Step 2: Build VGLC connectivity graph (from doors)
        vglc_graph = self._build_vglc_graph(vglc_rooms)
        
        # Step 3: Parse the .dot graph for semantic info
        dot_graph = self._parse_dot_graph(graph_file)
        
        # Step 4: Match semantics (START, BOSS, TRIFORCE)
        semantic_mapping = self._match_semantics(vglc_rooms, dot_graph)
        
        # Step 5: Convert to processed rooms
        processed_rooms: Dict[str, RoomData] = {}
        
        boss_room_id = None
        
        for pos, info in vglc_rooms.items():
            room_id = f"{pos[0]}_{pos[1]}"
            char_grid = info['grid']
            
            # Convert to semantic grid
            semantic_grid = self._char_to_semantic(char_grid)
            
            # Inject markers from semantic matching
            if pos in semantic_mapping:
                sem_info = semantic_mapping[pos]
                if sem_info.get('is_boss'):
                    boss_room_id = room_id
                    # Mark as boss room (inject BOSS marker)
                    self._inject_boss(semantic_grid)
                if sem_info.get('has_triforce'):
                    self._inject_triforce(semantic_grid)
            
            room = RoomData(
                room_id=room_id,
                grid=semantic_grid,
                contents=[],
                doors=info['doors'],
                position=pos
            )
            processed_rooms[room_id] = room
        
        # Step 6: Create ghost TRIFORCE room connected to BOSS
        # In original Zelda, TRIFORCE appears after boss is defeated
        if boss_room_id:
            # Check if there's already a TRIFORCE room
            has_triforce = any(SEMANTIC_PALETTE['TRIFORCE'] in r.grid for r in processed_rooms.values())
            
            if not has_triforce:
                # Create ghost triforce room
                boss_pos = processed_rooms[boss_room_id].position
                ghost_pos = (boss_pos[0] + 1, boss_pos[1])  # Below boss
                ghost_id = f"{ghost_pos[0]}_{ghost_pos[1]}_triforce"
                
                # Create simple triforce room
                ghost_grid = self._create_triforce_room()
                
                ghost_room = RoomData(
                    room_id=ghost_id,
                    grid=ghost_grid,
                    contents=['triforce'],
                    doors={'N': True, 'S': False, 'W': False, 'E': False},
                    position=ghost_pos
                )
                processed_rooms[ghost_id] = ghost_room
                
                # Add edge in graph
                vglc_graph.add_node(ghost_id, position=ghost_pos, has_triforce=True)
                vglc_graph.add_edge(boss_room_id, ghost_id, direction='S')
                vglc_graph.add_edge(ghost_id, boss_room_id, direction='N')
                
                print(f"Created ghost TRIFORCE room {ghost_id} connected to BOSS {boss_room_id}")
        
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
        
        Strategy: Use content clues from VGLC layout.
        - START: Room with STAIR (S character)
        - BOSS: Dead-end room with many monsters + blocks
        - TRIFORCE: Dead-end room farthest from start (after boss)
        """
        mapping = {}
        
        # Find START: VGLC room with STAIR that has at least one door
        # (isolated stair rooms are secret rooms, not dungeon entrances)
        start_vglc = None
        stair_rooms = [(pos, info) for pos, info in vglc_rooms.items() if info['has_stair']]
        
        # Prefer rooms with doors
        for pos, info in stair_rooms:
            door_count = sum(info['doors'].values())
            if door_count > 0:
                start_vglc = pos
                mapping[pos] = {'is_start': True}
                break
        
        # Fallback to any stair room if none have doors
        if not start_vglc and stair_rooms:
            start_vglc = stair_rooms[0][0]
            mapping[start_vglc] = {'is_start': True}
        
        # If no STAIR room, use the room with most doors as entrance
        if not start_vglc:
            max_doors = 0
            for pos, info in vglc_rooms.items():
                door_count = sum(info['doors'].values())
                if door_count > max_doors:
                    max_doors = door_count
                    start_vglc = pos
            
            if start_vglc:
                mapping[start_vglc] = {'is_start': True}
        
        if not start_vglc:
            return mapping
        
        # Build VGLC graph for distance calculations
        G = self._build_vglc_graph(vglc_rooms)
        start_id = f"{start_vglc[0]}_{start_vglc[1]}"
        distances = nx.single_source_shortest_path_length(G, start_id)
        
        # Find BOSS room: Dead-end with many monsters/blocks (combat arena)
        boss_candidates = []
        for pos, info in vglc_rooms.items():
            door_count = sum(info['doors'].values())
            grid = info['grid']
            monster_count = np.sum(grid == 'M')
            block_count = np.sum(grid == 'B')
            
            # Boss room criteria: dead-end, monsters, complex block pattern
            if door_count == 1 and (monster_count >= 4 or block_count >= 20):
                room_id = f"{pos[0]}_{pos[1]}"
                dist = distances.get(room_id, 0)
                boss_candidates.append((pos, dist, monster_count, block_count))
        
        boss_pos = None
        if boss_candidates:
            # Pick the dead-end with most monsters/blocks at good distance
            boss_candidates.sort(key=lambda x: (x[2] + x[3], x[1]), reverse=True)
            boss_pos = boss_candidates[0][0]
            mapping[boss_pos] = mapping.get(boss_pos, {})
            mapping[boss_pos]['is_boss'] = True
        
        # Find TRIFORCE: Farthest dead-end room (that's not boss)
        max_dist = -1
        triforce_pos = None
        for pos, info in vglc_rooms.items():
            if pos == boss_pos or pos == start_vglc:
                continue
            
            door_count = sum(info['doors'].values())
            if door_count == 1:  # Dead-end
                room_id = f"{pos[0]}_{pos[1]}"
                dist = distances.get(room_id, 0)
                if dist > max_dist:
                    max_dist = dist
                    triforce_pos = pos
        
        if triforce_pos:
            mapping[triforce_pos] = mapping.get(triforce_pos, {})
            mapping[triforce_pos]['has_triforce'] = True
        
        return mapping
    
    def _char_to_semantic(self, char_grid: np.ndarray) -> np.ndarray:
        """Convert character grid to semantic ID grid."""
        h, w = char_grid.shape
        semantic = np.zeros((h, w), dtype=np.int64)
        
        # Check if this is a corridor room (has doors but mostly void interior)
        has_doors = np.any(char_grid == 'D')
        void_count = np.sum(char_grid == '-')
        interior_size = (h - 4) * (w - 4)  # Exclude 2-tile borders
        is_corridor = has_doors and void_count > interior_size * 0.5
        
        for r in range(h):
            for c in range(w):
                char = char_grid[r, c]
                
                # Convert void to floor in corridor rooms
                if char == '-' and is_corridor:
                    # Only convert interior void to floor
                    if 2 <= r < h-2 and 2 <= c < w-2:
                        semantic[r, c] = SEMANTIC_PALETTE['FLOOR']
                    else:
                        semantic[r, c] = SEMANTIC_PALETTE['VOID']
                else:
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
    
    def _inject_boss(self, grid: np.ndarray):
        """Mark boss room by converting some enemies to BOSS."""
        # Find enemy tiles and convert one to BOSS
        enemy_locs = np.where(grid == SEMANTIC_PALETTE['ENEMY'])
        if len(enemy_locs[0]) > 0:
            # Convert center enemy to BOSS marker
            idx = len(enemy_locs[0]) // 2
            grid[enemy_locs[0][idx], enemy_locs[1][idx]] = SEMANTIC_PALETTE['BOSS']
    
    def _create_triforce_room(self) -> np.ndarray:
        """Create a simple triforce room grid."""
        # Standard room: 16x11
        grid = np.full((self.ROOM_HEIGHT, self.ROOM_WIDTH), SEMANTIC_PALETTE['FLOOR'], dtype=np.int64)
        
        # Add walls
        grid[0, :] = SEMANTIC_PALETTE['WALL']
        grid[-1, :] = SEMANTIC_PALETTE['WALL']
        grid[:, 0] = SEMANTIC_PALETTE['WALL']
        grid[:, -1] = SEMANTIC_PALETTE['WALL']
        grid[1, :] = SEMANTIC_PALETTE['WALL']
        grid[-2, :] = SEMANTIC_PALETTE['WALL']
        grid[:, 1] = SEMANTIC_PALETTE['WALL']
        grid[:, -2] = SEMANTIC_PALETTE['WALL']
        
        # Add door at top
        grid[0, 4:7] = SEMANTIC_PALETTE['DOOR_OPEN']
        grid[1, 4:7] = SEMANTIC_PALETTE['FLOOR']
        
        # Add TRIFORCE in center
        grid[8, 5] = SEMANTIC_PALETTE['TRIFORCE']
        
        return grid


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
