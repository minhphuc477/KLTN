"""
BLOCK I: INTELLIGENT DATA ADAPTER
=================================
Offline Pre-computation, Augmentation & Ground Truth Engine for Zelda AI Validation.

This module handles:
1. DEFENSIVE SLR MAPPER - Safe ASCII to Semantic ID conversion with door logic
2. RELATIONAL LAPLACIAN ENGINE - Positional encoding from graph structure  
3. MULTI-HOT NODE EMBEDDER - Room feature extraction
4. P-MATRIX BUILDER - Dependency graph encoding
5. OFFLINE AUGMENTATION - Data augmentation with graph synchronization

Author: KLTN Thesis Project
"""

import os
import re
import json
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from pathlib import Path

# ==========================================
# SEMANTIC PALETTE (CRITICAL CONSTANTS)
# ==========================================
# These IDs MUST be consistent across all modules

SEMANTIC_PALETTE = {
    # Environment
    'VOID': 0,           # Empty space (-)
    'FLOOR': 1,          # Walkable floor (F, .)
    'WALL': 2,           # Solid wall (W)
    'BLOCK': 3,          # Pushable/decorative block (B)
    
    # Doors (Context-aware - determined by graph edge type)
    'DOOR_OPEN': 10,     # Open passage (D with no lock)
    'DOOR_LOCKED': 11,   # Key-locked door (D with 'k' edge)
    'DOOR_BOMB': 12,     # Bombable wall (D with 'b' edge)
    'DOOR_PUZZLE': 13,   # Puzzle/switch door (D with 'S' edge)
    'DOOR_BOSS': 14,     # Boss key door (D with 'K' edge)
    'DOOR_SOFT': 15,     # Soft-locked (one-way) door (D with 'l' edge)
    
    # Entities
    'ENEMY': 20,         # Monster/enemy (M, e)
    'START': 21,         # Starting position (S, s)
    'TRIFORCE': 22,      # Goal/Triforce piece (t, T, G)
    'BOSS': 23,          # Boss enemy (b - in node label)
    
    # Items
    'KEY_SMALL': 30,     # Small key (k - in node label)
    'KEY_BOSS': 31,      # Boss key (K - in node label)
    'KEY_ITEM': 32,      # Key item like bow/raft (I - in node label)
    'ITEM_MINOR': 33,    # Minor collectible (i - in node label)
    
    # Special
    'ELEMENT': 40,       # Hazard element like lava/water (P)
    'ELEMENT_FLOOR': 41, # Element with floor underneath (O)
    'STAIR': 42,         # Stairs/ladder (S in room)
    'PUZZLE': 43,        # Puzzle element (p - in node label)
}

# Reverse lookup for debugging
ID_TO_NAME = {v: k for k, v in SEMANTIC_PALETTE.items()}

# Character mapping from VGLC format
CHAR_TO_SEMANTIC = {
    '-': SEMANTIC_PALETTE['VOID'],
    'F': SEMANTIC_PALETTE['FLOOR'],
    '.': SEMANTIC_PALETTE['FLOOR'],
    'W': SEMANTIC_PALETTE['WALL'],
    'B': SEMANTIC_PALETTE['BLOCK'],
    'M': SEMANTIC_PALETTE['ENEMY'],
    'P': SEMANTIC_PALETTE['ELEMENT'],
    'O': SEMANTIC_PALETTE['ELEMENT_FLOOR'],
    'I': SEMANTIC_PALETTE['ELEMENT'],  # Element + Block in VGLC
    'S': SEMANTIC_PALETTE['STAIR'],     # Stair in room context
    # 'D' is handled specially by Defensive Router
}


class GridBasedRoomExtractor:
    """
    PRECISE Room Extractor for VGLC Zelda Maps (Forensically Verified).
    
    VGLC TEXT FORMAT (verified from tloz1_1.txt analysis):
    ======================================================
    - Grid is divided into fixed 11-col x 16-row SLOTS
    - Each slot is either a ROOM or a GAP (all dashes)
    - Room dimensions: 11 columns x 16 rows (including walls)
    - Room structure: WW (2 wall) + 7 interior + WW (2 wall) = 11 cols
    - Gap: 11 dashes in each row
    
    Example tloz1_1.txt (66 cols x 96 rows = 6x6 grid):
      - 6 column slots (0-10, 11-21, 22-32, 33-43, 44-54, 55-65)
      - 6 row slots (0-15, 16-31, 32-47, 48-63, 64-79, 80-95)
      - At row 16: slots 0,2,3,5 are ROOMS, slots 1,4 are GAPS
    
    Adjacent rooms in VGLC format ARE adjacent in the dungeon map.
    The stitcher should place them with shared walls.
    """

    # VERIFIED CONSTANTS
    SLOT_WIDTH = 11   # Characters per column slot
    SLOT_HEIGHT = 16  # Rows per row slot
    GAP_MARKER = '-'  # Void/gap character
    WALL_MARKER = 'W' # Wall character
    
    def __init__(self):
        pass

    def _load_grid(self, filepath: str) -> np.ndarray:
        """Load VGLC text file into numpy character array."""
        with open(filepath, 'r') as f:
            lines = [line.rstrip('\n') for line in f]
        
        if not lines:
            return np.zeros((0, 0), dtype='<U1')

        width = max(len(line) for line in lines) if lines else 0
        padded = [list(line.ljust(width, self.GAP_MARKER)) for line in lines]
        return np.array(padded)

    def _is_room_slot(self, slot_grid: np.ndarray) -> bool:
        """
        Check if a slot contains a room (not a gap).
        
        A gap has all dashes. A room has walls and floor.
        """
        if slot_grid.size == 0:
            return False
        
        # Gap detection: mostly dashes
        dash_count = np.sum(slot_grid == self.GAP_MARKER)
        total = slot_grid.size
        if dash_count > total * 0.7:
            return False
        
        # Room validation: must have walls and floor
        wall_count = np.sum(slot_grid == self.WALL_MARKER)
        floor_count = np.sum(slot_grid == 'F')
        
        # A valid room has significant walls (perimeter) and floor (interior)
        return wall_count >= 20 and floor_count >= 5

    def extract(self, filepath: str) -> List[Tuple[Tuple[int, int], np.ndarray]]:
        """
        Extract all rooms from VGLC file using fixed slot grid.
        
        Returns:
            List of ((row_idx, col_idx), room_grid) tuples where:
            - row_idx, col_idx are the room's position in the dungeon grid
            - room_grid is a 16x11 numpy char array
        """
        grid = self._load_grid(filepath)
        
        if grid.size == 0:
            return []
        
        h, w = grid.shape
        
        # Calculate number of slots
        num_row_slots = h // self.SLOT_HEIGHT
        num_col_slots = w // self.SLOT_WIDTH
        
        rooms = []
        
        for row_slot in range(num_row_slots):
            row_start = row_slot * self.SLOT_HEIGHT
            row_end = row_start + self.SLOT_HEIGHT
            
            for col_slot in range(num_col_slots):
                col_start = col_slot * self.SLOT_WIDTH
                col_end = col_start + self.SLOT_WIDTH
                
                # Extract slot
                slot_grid = grid[row_start:row_end, col_start:col_end]
                
                # Ensure proper size
                if slot_grid.shape[0] < self.SLOT_HEIGHT:
                    pad = np.full((self.SLOT_HEIGHT - slot_grid.shape[0], slot_grid.shape[1]), self.GAP_MARKER)
                    slot_grid = np.vstack([slot_grid, pad])
                if slot_grid.shape[1] < self.SLOT_WIDTH:
                    pad = np.full((slot_grid.shape[0], self.SLOT_WIDTH - slot_grid.shape[1]), self.GAP_MARKER)
                    slot_grid = np.hstack([slot_grid, pad])
                
                # Check if this slot contains a room
                if self._is_room_slot(slot_grid):
                    rooms.append(((row_slot, col_slot), slot_grid.copy()))
        
        return rooms
    
    def extract_with_ids(self, filepath: str) -> List[Tuple[int, np.ndarray]]:
        """
        Extract rooms with integer spatial IDs for backward compatibility.
        
        Returns: List of (spatial_id, room_grid) where spatial_id = row*100 + col
        """
        raw_rooms = self.extract(filepath)
        return [(r_idx * 100 + c_idx, grid) for ((r_idx, c_idx), grid) in raw_rooms]

# Room dimensions (VGLC Zelda dungeon format)
# CRITICAL: VGLC uses 11-character wide rooms, NOT 16!
ROOM_HEIGHT = 16  # Full room height including walls
ROOM_WIDTH = 11   # Full room width including walls (WW + 7 + WW)
FULL_ROOM_HEIGHT = 16  # Same as ROOM_HEIGHT for VGLC
FULL_ROOM_WIDTH = 11   # Same as ROOM_WIDTH for VGLC

# Edge type mapping from graph labels
EDGE_TYPE_MAP = {
    '': 'open',
    'k': 'locked',      # Small key required
    'K': 'boss_locked', # Boss key required
    'b': 'bombable',    # Bomb required
    'l': 'soft_locked', # One-way (can't return)
    'S': 'switch',      # Switch/puzzle required
    'I': 'item_locked', # Key item required
}

# Node content mapping from graph labels
NODE_CONTENT_MAP = {
    'e': 'enemy',
    's': 'start',
    't': 'triforce',
    'b': 'boss',
    'k': 'key',
    'K': 'boss_key',
    'I': 'key_item',
    'i': 'item',
    'p': 'puzzle',
}


@dataclass
class RoomData:
    """Represents a single room's data after processing."""
    room_id: str
    grid: np.ndarray                    # Semantic grid [H, W]
    contents: List[str] = field(default_factory=list)  # Items in room
    doors: Dict[str, Dict] = field(default_factory=dict)  # Door info by direction
    position: Tuple[int, int] = (0, 0)  # Position in dungeon layout


@dataclass  
class DungeonData:
    """Represents a complete dungeon's processed data."""
    dungeon_id: str
    rooms: Dict[str, RoomData]          # room_id -> RoomData
    graph: nx.DiGraph                    # Connectivity graph
    layout: np.ndarray                   # 2D layout of room positions
    tpe_vectors: np.ndarray             # Topological positional encoding
    p_matrix: np.ndarray                # Dependency matrix
    node_features: np.ndarray           # Node feature vectors


class IntelligentDataAdapter:
    """
    Main adapter class for processing Zelda dungeon data.
    
    Handles conversion from raw VGLC format to semantic tensors with
    proper door logic inference from graph structure.
    """
    
    def __init__(self, data_root: str, output_path: str = None):
        """
        Initialize the adapter.
        
        Args:
            data_root: Path to the raw data folder
            output_path: Path to save processed data (optional)
        """
        self.data_root = Path(data_root)
        self.output_path = Path(output_path) if output_path else None
        self.processed_dungeons: Dict[str, DungeonData] = {}
        
        # Load zelda.json if available for metadata
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict:
        """Load zelda.json metadata file if present."""
        meta_path = self.data_root / "zelda.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                return json.load(f)
        return {}
    
    # ==========================================
    # MODULE 1: MAP PARSER
    # ==========================================
    
    def parse_vglc_map(self, filepath: str) -> Tuple[List[List[np.ndarray]], Dict[Tuple[int,int], int]]:
        """
        Parse a VGLC format map file into room grids.
        
        The VGLC format stores multiple rooms in a 2D arrangement where
        rooms are separated by VOID (-) regions.
        
        Args:
            filepath: Path to the .txt map file
            
        Returns:
            rooms_grid: 2D list of room numpy arrays
            room_positions: Dict mapping (row, col) -> room_id
        """
        with open(filepath, 'r') as f:
            lines = [line.rstrip('\n') for line in f.readlines()]
        
        if not lines:
            return [], {}
            
        # Determine the full map dimensions
        max_width = max(len(line) for line in lines)
        height = len(lines)
        
        # Pad lines to consistent width
        padded_lines = [line.ljust(max_width, '-') for line in lines]
        
        # Convert to numpy array of characters
        full_map = np.array([list(line) for line in padded_lines])
        
        # Detect room boundaries (rooms are 11x22 including walls, or 16x11 interior)
        # Standard VGLC room: 11 rows (with 2-row wall top/bottom) x 22 cols (with 2-col wall left/right)
        # Actually looking at the data: rooms appear to be ~16 rows x 11 cols of content
        
        rooms = []
        room_positions = {}
        room_id = 0
        
        # Scan for rooms by looking for WW patterns (wall indicators)
        # Each room in VGLC Zelda is typically 11 wide x 16 tall (including walls)
        room_w = 11  # Characters per room column (including 'WW' borders)
        room_h = 16  # Lines per room row (including borders)
        
        # Alternative: detect rooms by finding connected 'W' regions
        # For now, use fixed grid approach based on observed data structure
        
        # The map appears to have rooms arranged in a grid
        # Let's detect room count by checking for wall patterns
        
        rows_of_rooms = height // room_h
        cols_per_row = []
        
        for row_idx in range(rows_of_rooms):
            start_line = row_idx * room_h
            # Check how many rooms in this row by looking at first interior line
            sample_line = padded_lines[start_line + 2] if start_line + 2 < height else ''
            
            # Count room columns by finding 'WW' patterns
            room_starts = []
            i = 0
            while i < len(sample_line):
                if sample_line[i:i+2] == 'WW':
                    room_starts.append(i)
                    i += room_w + 2  # Skip to next potential room
                else:
                    i += 1
            
            cols_per_row.append(len(room_starts))
        
        # Extract each room
        for row_idx in range(rows_of_rooms):
            room_row = []
            start_line = row_idx * room_h
            
            # Find rooms in this row
            sample_line = padded_lines[start_line] if start_line < height else ''
            
            col_idx = 0
            char_idx = 0
            
            while char_idx < len(sample_line):
                # Look for room start (WW pattern)
                if char_idx + room_w <= len(sample_line):
                    # Check if this is a valid room position
                    check_char = sample_line[char_idx:char_idx+2]
                    
                    if check_char == 'WW' or (char_idx + 11 <= len(sample_line) and 
                                               'W' in sample_line[char_idx:char_idx+11]):
                        # Extract room
                        room_chars = []
                        for line_offset in range(min(room_h, height - start_line)):
                            line = padded_lines[start_line + line_offset]
                            room_line = line[char_idx:char_idx + room_w]
                            if len(room_line) < room_w:
                                room_line = room_line.ljust(room_w, '-')
                            room_chars.append(list(room_line))
                        
                        if room_chars:
                            room_array = np.array(room_chars)
                            # Check if this is an actual room (not all void)
                            if not np.all(room_array == '-'):
                                room_positions[(row_idx, col_idx)] = room_id
                                room_row.append(room_array)
                                room_id += 1
                            else:
                                room_row.append(None)
                        
                        col_idx += 1
                        char_idx += room_w + 2  # Account for gap between rooms
                    else:
                        char_idx += 1
                else:
                    break
                    
            if room_row:
                rooms.append(room_row)
        
        return rooms, room_positions
    
    def extract_rooms_simple(self, filepath: str) -> List[Tuple[int, np.ndarray]]:
        """
        Simpler room extraction that finds rooms by detecting wall boundaries.
        
        Returns:
            List of (room_id, room_grid) tuples
        """
        with open(filepath, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        lines = [line for line in lines if line.strip()]  # Remove empty lines
        
        if not lines:
            return []
        
        # Find consistent line width
        max_width = max(len(line) for line in lines)
        lines = [line.ljust(max_width, '-') for line in lines]
        
        # Convert to 2D array
        full_map = np.array([list(line) for line in lines])
        h, w = full_map.shape
        
        rooms = []
        room_id = 0
        
        # Scan for room patterns (WW...WW with content inside)
        # Room detection: find isolated rectangular regions bounded by W
        
        visited = np.zeros((h, w), dtype=bool)
        
        for i in range(h):
            for j in range(w):
                if visited[i, j]:
                    continue
                    
                # Look for potential room start (top-left corner of walls)
                if full_map[i, j] == 'W':
                    # Try to find room bounds
                    room_grid, bounds = self._extract_single_room(full_map, i, j, visited)
                    if room_grid is not None and room_grid.size > 50:  # Minimum size check
                        rooms.append((room_id, room_grid))
                        room_id += 1
                else:
                    visited[i, j] = True
        
        return rooms

    def extract_rooms_strict(self, filepath: str) -> List[Tuple[int, np.ndarray]]:
        """Extract rooms using the slot-based GridBasedRoomExtractor."""
        extractor = GridBasedRoomExtractor()
        return extractor.extract_with_ids(filepath)
    
    def _extract_single_room(self, full_map: np.ndarray, start_i: int, start_j: int, 
                             visited: np.ndarray) -> Tuple[Optional[np.ndarray], Tuple]:
        """Extract a single room starting from given position."""
        h, w = full_map.shape
        
        # Standard room dimensions in VGLC Zelda
        std_heights = [11, 16]  # Common room heights
        std_widths = [11, 16, 22]   # Common room widths
        
        # Try to find room bounds
        # Look for continuous W region to define top border
        
        # Find right edge of top wall
        j_end = start_j
        while j_end < w and full_map[start_i, j_end] == 'W':
            j_end += 1
        
        room_width = j_end - start_j
        if room_width < 8:  # Too narrow
            visited[start_i, start_j] = True
            return None, ()
        
        # Find bottom edge
        i_end = start_i
        while i_end < h and full_map[i_end, start_j] == 'W':
            i_end += 1
        
        # Now scan down to find where room ends
        i_end = start_i
        for i in range(start_i, min(start_i + 20, h)):
            # Check if this row is still part of the room
            row_slice = full_map[i, start_j:j_end]
            if np.all(row_slice == '-'):
                break
            i_end = i + 1
        
        room_height = i_end - start_i
        if room_height < 8:  # Too short
            visited[start_i, start_j] = True
            return None, ()
        
        # Extract room
        room_grid = full_map[start_i:i_end, start_j:j_end].copy()
        
        # Mark as visited
        visited[start_i:i_end, start_j:j_end] = True
        
        return room_grid, (start_i, start_j, i_end, j_end)
    
    # ==========================================
    # MODULE 2: DEFENSIVE LOGIC-INFUSED MAPPER
    # ==========================================
    
    def defensive_mapper(self, char_grid: np.ndarray, room_id: int,
                        graph_attrs: Dict[str, Dict] = None,
                        node_attrs: Dict[str, Any] = None) -> np.ndarray:
        """
        Convert character grid to semantic IDs with door logic AND graph-guided injection.
        
        The "Defensive Router" ensures:
        1. Doors at invalid positions become walls
        2. Door types are inferred from graph edge attributes
        3. Corner positions are protected from invalid door placement
        4. START position injected when is_start=True (Graph-Guided Logic)
        5. TRIFORCE position injected when has_triforce=True
        6. Items (KEY, BOSS_KEY) injected based on node contents
        
        Args:
            char_grid: 2D numpy array of characters
            room_id: ID of the current room (for graph lookup)
            graph_attrs: Dict of {direction: {'type': edge_type, 'neighbor': neighbor_id}}
            node_attrs: Dict of node attributes from graph {'is_start', 'has_triforce', 'has_key', etc.}
        
        Returns:
            Semantic ID grid [H, W]
        """
        if graph_attrs is None:
            graph_attrs = {}
        if node_attrs is None:
            node_attrs = {}
            
        h, w = char_grid.shape
        semantic_grid = np.zeros((h, w), dtype=np.int64)
        
        # Track positions for injection
        floor_positions = []  # Candidate positions for injecting items
        stair_positions = []  # 'S' positions that might be START or STAIR
        
        # First pass: basic character mapping
        for r in range(h):
            for c in range(w):
                char = char_grid[r, c]
                
                # Handle door logic specially
                if char == 'D':
                    semantic_id = self._route_door(r, c, h, w, graph_attrs)
                elif char == 'S':
                    # 'S' is ambiguous: could be STAIR or START
                    # We'll mark it and resolve after based on node_attrs
                    stair_positions.append((r, c))
                    semantic_id = SEMANTIC_PALETTE['STAIR']  # Default to STAIR
                else:
                    # Standard character mapping
                    semantic_id = CHAR_TO_SEMANTIC.get(char, SEMANTIC_PALETTE['FLOOR'])
                
                semantic_grid[r, c] = semantic_id
                
                # Track walkable floor positions (for item injection)
                if semantic_id == SEMANTIC_PALETTE['FLOOR']:
                    floor_positions.append((r, c))
        
        # Second pass: Graph-Guided Logic Injection
        # If this room is the START room, convert one 'S' to START or inject START
        if node_attrs.get('is_start', False):
            if stair_positions:
                # Convert the first stair to START position
                sr, sc = stair_positions[0]
                semantic_grid[sr, sc] = SEMANTIC_PALETTE['START']
            elif floor_positions:
                # No 'S' found, inject START at a floor position
                # Prefer center of room
                center_r, center_c = h // 2, w // 2
                best_pos = min(floor_positions, 
                              key=lambda p: abs(p[0]-center_r) + abs(p[1]-center_c))
                semantic_grid[best_pos[0], best_pos[1]] = SEMANTIC_PALETTE['START']
        
        # If this room has TRIFORCE, inject it
        if node_attrs.get('has_triforce', False):
            if floor_positions:
                # Inject TRIFORCE at a floor position (prefer center-ish)
                center_r, center_c = h // 2, w // 2
                # Remove positions already used for START
                available = [p for p in floor_positions 
                            if semantic_grid[p[0], p[1]] != SEMANTIC_PALETTE['START']]
                if available:
                    best_pos = min(available, 
                                  key=lambda p: abs(p[0]-center_r) + abs(p[1]-center_c))
                    semantic_grid[best_pos[0], best_pos[1]] = SEMANTIC_PALETTE['TRIFORCE']
        
        # Inject keys based on node contents
        if node_attrs.get('has_key', False):
            available = [p for p in floor_positions 
                        if semantic_grid[p[0], p[1]] == SEMANTIC_PALETTE['FLOOR']]
            if available:
                # Place key at a random-ish position (use room_id for determinism)
                idx = room_id % len(available)
                kr, kc = available[idx]
                semantic_grid[kr, kc] = SEMANTIC_PALETTE['KEY_SMALL']
        
        if node_attrs.get('has_boss_key', False):
            available = [p for p in floor_positions 
                        if semantic_grid[p[0], p[1]] == SEMANTIC_PALETTE['FLOOR']]
            if available:
                idx = (room_id + 1) % len(available)
                kr, kc = available[idx]
                semantic_grid[kr, kc] = SEMANTIC_PALETTE['KEY_BOSS']
        
        return semantic_grid
    
    def _route_door(self, r: int, c: int, h: int, w: int, 
                   graph_attrs: Dict[str, Dict]) -> int:
        """
        Route door character to appropriate semantic ID based on position and graph.
        
        Defensive Router Rules:
        - North doors: Only valid at row 0, columns 2-13 (avoid corners)
        - South doors: Only valid at row h-1, columns 2-13
        - West doors: Only valid at column 0, rows 2-8 
        - East doors: Only valid at column w-1, rows 2-8
        - Invalid positions become WALL
        """
        direction = None
        
        # Determine direction based on position
        # Avoid corner "dead zones" (0,0), (0,w-1), (h-1,0), (h-1,w-1)
        margin = 2  # Minimum distance from corner
        
        if r <= 1:  # Top row region
            if margin < c < w - margin:
                direction = 'north'
        elif r >= h - 2:  # Bottom row region
            if margin < c < w - margin:
                direction = 'south'
        
        if direction is None:  # Check horizontal doors
            if c <= 1:  # Left column region
                if margin < r < h - margin:
                    direction = 'west'
            elif c >= w - 2:  # Right column region
                if margin < r < h - margin:
                    direction = 'east'
        
        if direction is None:
            # Door at invalid position -> treat as wall
            return SEMANTIC_PALETTE['WALL']
        
        # Look up edge type from graph
        edge_info = graph_attrs.get(direction, {})
        edge_type = edge_info.get('type', 'open')
        
        # Map edge type to door semantic ID
        door_type_map = {
            'open': SEMANTIC_PALETTE['DOOR_OPEN'],
            'locked': SEMANTIC_PALETTE['DOOR_LOCKED'],
            'boss_locked': SEMANTIC_PALETTE['DOOR_BOSS'],
            'bombable': SEMANTIC_PALETTE['DOOR_BOMB'],
            'soft_locked': SEMANTIC_PALETTE['DOOR_SOFT'],
            'switch': SEMANTIC_PALETTE['DOOR_PUZZLE'],
            'item_locked': SEMANTIC_PALETTE['DOOR_PUZZLE'],
        }
        
        return door_type_map.get(edge_type, SEMANTIC_PALETTE['DOOR_OPEN'])
    
    # ==========================================
    # MODULE 3: GRAPH PARSER
    # ==========================================
    
    def parse_dot_graph(self, filepath: str) -> nx.DiGraph:
        """
        Parse a .dot graph file into NetworkX DiGraph.
        
        Node labels contain room content (e.g., "e,k" = enemy + key)
        Edge labels contain door types (e.g., "k" = key locked)
        
        Args:
            filepath: Path to .dot file
            
        Returns:
            NetworkX DiGraph with parsed attributes
        """
        G = nx.DiGraph()
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Parse nodes: 0 [label="e,k"]
        # FIX: Use line-start anchor to avoid matching edge labels like "7 -> 8 [label=""]"
        # which would overwrite node attributes.
        # Node definitions appear at line start (after newline or start of file)
        node_pattern = r'^(\d+)\s*\[label="([^"]*)"\]'
        for match in re.finditer(node_pattern, content, re.MULTILINE):
            node_id = int(match.group(1))
            label = match.group(2)
            
            # Parse label into content list
            contents = [c.strip() for c in label.split(',') if c.strip()]
            content_types = []
            for c in contents:
                if c in NODE_CONTENT_MAP:
                    content_types.append(NODE_CONTENT_MAP[c])
            
            G.add_node(node_id, 
                      label=label,
                      contents=content_types,
                      is_start='start' in content_types,
                      has_enemy='enemy' in content_types,
                      has_key='key' in content_types,
                      has_boss_key='boss_key' in content_types,
                      has_triforce='triforce' in content_types,
                      has_boss='boss' in content_types)
        
        # Parse edges: 7 -> 8 [label="k"]
        edge_pattern = r'(\d+)\s*->\s*(\d+)\s*\[label="([^"]*)"\]'
        for match in re.finditer(edge_pattern, content):
            src = int(match.group(1))
            dst = int(match.group(2))
            label = match.group(3)
            
            edge_type = EDGE_TYPE_MAP.get(label, 'open')
            
            G.add_edge(src, dst, 
                      label=label,
                      type=edge_type,
                      requires_key=edge_type == 'locked',
                      requires_bomb=edge_type == 'bombable',
                      requires_boss_key=edge_type == 'boss_locked')
        
        return G
    
    # ==========================================
    # MODULE 4: RELATIONAL LAPLACIAN ENGINE
    # ==========================================
    
    def compute_laplacian_pe(self, G: nx.Graph, k_dim: int = 8) -> Tuple[np.ndarray, Dict[int, int]]:
        """
        Compute Positional Encoding using Graph Laplacian eigenvectors.
        
        This creates topology-aware position vectors for each node based on
        the graph's spectral properties.
        
        Args:
            G: NetworkX graph (will be treated as undirected)
            k_dim: Number of eigenvector dimensions to use
            
        Returns:
            tpe: Topological Positional Encoding array [N, k_dim]
            node_to_idx: Mapping from node ID to array index
        """
        # Convert to undirected for Laplacian
        G_undirected = G.to_undirected() if G.is_directed() else G
        
        nodes = sorted(G_undirected.nodes())
        n = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        if n == 0:
            return np.zeros((0, k_dim)), {}
        
        # Build weighted adjacency matrix
        adj = np.zeros((n, n))
        
        for u, v, data in G_undirected.edges(data=True):
            idx_u, idx_v = node_to_idx[u], node_to_idx[v]
            
            # Weight based on edge difficulty
            edge_type = data.get('type', 'open')
            # Harder edges (locked, bombable) get lower weight (higher resistance)
            weight = 0.5 if edge_type in ['locked', 'bombable', 'boss_locked', 'soft_locked'] else 1.0
            
            adj[idx_u, idx_v] = weight
            adj[idx_v, idx_u] = weight
        
        # Degree matrix
        degrees = np.sum(adj, axis=1)
        D = np.diag(degrees)
        
        # Laplacian: L = D - A
        L = D - adj
        
        try:
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            
            # Take k smallest eigenvectors (skip the first one which is constant)
            # The first eigenvalue is ~0 (Fiedler), skip it
            start_idx = 1
            end_idx = min(start_idx + k_dim, n)
            
            tpe = eigenvectors[:, start_idx:end_idx]
            
            # Pad if not enough dimensions
            if tpe.shape[1] < k_dim:
                padding = np.zeros((n, k_dim - tpe.shape[1]))
                tpe = np.hstack([tpe, padding])
                
        except np.linalg.LinAlgError:
            # Fallback: zeros
            tpe = np.zeros((n, k_dim))
        
        return tpe.astype(np.float32), node_to_idx
    
    # ==========================================
    # MODULE 5: MULTI-HOT NODE EMBEDDER
    # ==========================================
    
    def extract_node_features(self, G: nx.DiGraph, node_order: Dict[int, int]) -> np.ndarray:
        """
        Extract multi-hot feature vectors for each node.
        
        Feature vector (5 dimensions):
        [is_start, has_enemy, has_key, has_boss_key, has_triforce]
        
        Args:
            G: NetworkX graph with node attributes
            node_order: Mapping from node ID to feature array index
            
        Returns:
            Feature array [N, 5]
        """
        n = len(node_order)
        features = np.zeros((n, 5), dtype=np.float32)
        
        for node_id, idx in node_order.items():
            if node_id not in G.nodes:
                continue
                
            attrs = G.nodes[node_id]
            
            features[idx, 0] = 1.0 if attrs.get('is_start', False) else 0.0
            features[idx, 1] = 1.0 if attrs.get('has_enemy', False) else 0.0
            features[idx, 2] = 1.0 if attrs.get('has_key', False) else 0.0
            features[idx, 3] = 1.0 if attrs.get('has_boss_key', False) else 0.0
            features[idx, 4] = 1.0 if attrs.get('has_triforce', False) else 0.0
        
        return features
    
    # ==========================================
    # MODULE 6: P-MATRIX BUILDER
    # ==========================================
    
    def build_p_matrix(self, G: nx.DiGraph, node_order: Dict[int, int]) -> np.ndarray:
        """
        Build the dependency (prerequisite) matrix.
        
        P-Matrix shape: [N, N, 3]
        Channels:
        - 0: Small key dependency (edge requires key)
        - 1: Bomb dependency (edge requires bomb)
        - 2: Boss key dependency (edge requires boss key)
        
        Args:
            G: NetworkX graph
            node_order: Mapping from node ID to matrix index
            
        Returns:
            P-Matrix array [N, N, 3]
        """
        n = len(node_order)
        p_matrix = np.zeros((n, n, 3), dtype=np.float32)
        
        for u, v, data in G.edges(data=True):
            if u not in node_order or v not in node_order:
                continue
                
            i, j = node_order[u], node_order[v]
            edge_type = data.get('type', 'open')
            
            # Set dependency flags
            if edge_type == 'locked':
                p_matrix[i, j, 0] = 1.0
                p_matrix[j, i, 0] = 1.0  # Symmetric
            elif edge_type == 'bombable':
                p_matrix[i, j, 1] = 1.0
                p_matrix[j, i, 1] = 1.0
            elif edge_type == 'boss_locked':
                p_matrix[i, j, 2] = 1.0
                p_matrix[j, i, 2] = 1.0
        
        return p_matrix
    
    # ==========================================
    # MODULE 7: AUGMENTATION
    # ==========================================
    
    def augment_grid(self, grid: np.ndarray, mode: str) -> np.ndarray:
        """Augment grid with specified transformation."""
        if mode == 'orig':
            return grid.copy()
        elif mode == 'flip_h':
            return np.fliplr(grid)
        elif mode == 'flip_v':
            return np.flipud(grid)
        elif mode == 'flip_both':
            return np.flipud(np.fliplr(grid))
        return grid.copy()
    
    def augment_graph_attrs(self, attrs: Dict[str, Dict], mode: str) -> Dict[str, Dict]:
        """Synchronize graph attributes with grid augmentation."""
        if mode == 'orig':
            return attrs.copy()
        
        new_attrs = {}
        
        if mode == 'flip_h':
            swap_map = {'east': 'west', 'west': 'east', 'north': 'north', 'south': 'south'}
        elif mode == 'flip_v':
            swap_map = {'north': 'south', 'south': 'north', 'east': 'east', 'west': 'west'}
        elif mode == 'flip_both':
            swap_map = {'east': 'west', 'west': 'east', 'north': 'south', 'south': 'north'}
        else:
            return attrs.copy()
        
        for direction, info in attrs.items():
            new_direction = swap_map.get(direction, direction)
            new_attrs[new_direction] = info.copy()
        
        return new_attrs
    
    # ==========================================
    # MAIN PROCESSING PIPELINE
    # ==========================================
    
    def process_dungeon(self, map_file: str, graph_file: str, dungeon_id: str) -> DungeonData:
        """
        Process a complete dungeon using GRAPH-FIRST BEST PRACTICE approach.
        
        The .dot graph is the AUTHORITATIVE source for:
        - Room connectivity (edges)
        - Room contents (node labels: s=start, t=triforce, k=key, etc.)
        - Door types (edge labels: k=key-locked, b=bombable, l=soft-locked, etc.)
        
        Algorithm:
        1. Parse graph to get logical structure
        2. Extract VGLC rooms with their physical positions
        3. Build CONTENT-BASED MAPPING using landmarks (START, TRIFORCE, etc.)
        4. Propagate mapping using GRAPH ADJACENCY + VGLC ADJACENCY consistency
        5. Create GHOST ROOMS for graph nodes without VGLC matches
        6. Stitch based on GRAPH CONNECTIVITY (not VGLC positions)
        
        Args:
            map_file: Path to VGLC map .txt file
            graph_file: Path to .dot graph file
            dungeon_id: Identifier for this dungeon
            
        Returns:
            DungeonData object with all processed tensors
        """
        # STEP 1: Parse the authoritative graph structure
        graph = self.parse_dot_graph(graph_file)
        
        # STEP 2: Extract VGLC rooms with positions
        rooms_data = self.extract_rooms_strict(map_file)  # Returns [(spatial_id, char_grid), ...]
        
        # Build position -> room data mapping
        vglc_rooms: Dict[Tuple[int, int], np.ndarray] = {}
        for spatial_id, char_grid in rooms_data:
            pos = (spatial_id // 100, spatial_id % 100)
            vglc_rooms[pos] = char_grid
        
        # STEP 3: Content-based landmark mapping
        node_to_vglc = self._build_graph_to_vglc_mapping(graph, vglc_rooms)
        
        # STEP 4: Process each graph node (using mapping or ghost rooms)
        processed_rooms: Dict[str, RoomData] = {}
        layout_positions: Dict[int, Tuple[int, int]] = {}
        
        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            
            if node_id in node_to_vglc:
                # Has VGLC room - use it
                vglc_pos = node_to_vglc[node_id]
                char_grid = vglc_rooms[vglc_pos]
                layout_positions[node_id] = vglc_pos
            else:
                # No VGLC room - create ghost room from graph data
                char_grid = self._create_ghost_room_from_graph(node_id, graph)
                # Assign a position near connected nodes
                layout_positions[node_id] = self._find_ghost_position(
                    node_id, graph, layout_positions
                )
            
            # Get edge attributes for this room
            graph_attrs = self._get_room_graph_attrs(graph, node_id, layout_positions, char_grid)
            
            # Get node attributes
            node_attrs = {
                'is_start': node_data.get('is_start', False),
                'has_triforce': node_data.get('has_triforce', False),
                'has_key': node_data.get('has_key', False),
                'has_boss_key': node_data.get('has_boss_key', False),
                'has_enemy': node_data.get('has_enemy', False),
                'has_boss': node_data.get('has_boss', False),
                'contents': node_data.get('contents', []),
            }
            
            # Convert to semantic grid
            semantic_grid = self.defensive_mapper(char_grid, node_id, graph_attrs, node_attrs)
            
            processed_rooms[str(node_id)] = RoomData(
                room_id=str(node_id),
                grid=semantic_grid,
                contents=node_attrs.get('contents', []),
                doors=graph_attrs,
                position=layout_positions[node_id]
            )
        
        # STEP 5: Ensure START and TRIFORCE exist
        self._ensure_landmarks(processed_rooms, graph)
        
        # STEP 6: Compute TPE, node features, P-matrix
        tpe_vectors, node_order = self.compute_laplacian_pe(graph)
        node_features = self.extract_node_features(graph, node_order)
        p_matrix = self.build_p_matrix(graph, node_order)
        
        # Build layout matrix
        if layout_positions:
            max_r = max(p[0] for p in layout_positions.values()) + 1
            max_c = max(p[1] for p in layout_positions.values()) + 1
            layout = np.full((max_r, max_c), -1, dtype=int)
            for node_id, (r, c) in layout_positions.items():
                if 0 <= r < layout.shape[0] and 0 <= c < layout.shape[1]:
                    layout[r, c] = node_id
        else:
            layout = np.zeros((0, 0), dtype=int)
        
        return DungeonData(
            dungeon_id=dungeon_id,
            rooms=processed_rooms,
            graph=graph,
            layout=layout,
            tpe_vectors=tpe_vectors,
            p_matrix=p_matrix,
            node_features=node_features
        )
    
    def _build_graph_to_vglc_mapping(self, graph: nx.DiGraph, 
                                      vglc_rooms: Dict[Tuple[int, int], np.ndarray]) -> Dict[int, Tuple[int, int]]:
        """
        Build a mapping from graph node IDs to VGLC room positions.
        
        BEST PRACTICE: Content-based matching with adjacency propagation.
        
        Algorithm:
        1. Find landmark nodes (START, TRIFORCE, BOSS) and match to VGLC rooms
        2. BFS from landmarks: for each graph edge, find consistent VGLC adjacency
        3. Handle remaining nodes with content-based scoring
        
        Returns:
            Dict mapping graph node ID -> VGLC (row, col) position
        """
        mapping: Dict[int, Tuple[int, int]] = {}
        used_positions: Set[Tuple[int, int]] = set()
        
        # PHASE 1: Landmark matching
        # Find graph nodes with special content
        start_node = None
        triforce_node = None
        boss_node = None
        
        for node_id, attrs in graph.nodes(data=True):
            if attrs.get('is_start', False):
                start_node = node_id
            if attrs.get('has_triforce', False):
                triforce_node = node_id
            if attrs.get('has_boss', False):
                boss_node = node_id
        
        # Find VGLC rooms with special tiles
        start_rooms = []
        triforce_rooms = []
        boss_rooms = []  # Rooms with 'B' blocks often near boss (heuristic)
        
        for pos, grid in vglc_rooms.items():
            if np.any(grid == 'S'):
                start_rooms.append(pos)
            if np.any(np.isin(grid, ['T', 't', 'G'])):
                triforce_rooms.append(pos)
            # Boss rooms often have distinct patterns - use 'B' block count as hint
            b_count = np.sum(grid == 'B')
            if b_count >= 4:  # Multiple blocks suggest boss arena
                boss_rooms.append(pos)
        
        # Match landmarks
        if start_node is not None and start_rooms:
            mapping[start_node] = start_rooms[0]
            used_positions.add(start_rooms[0])
        
        if triforce_node is not None and triforce_rooms:
            mapping[triforce_node] = triforce_rooms[0]
            used_positions.add(triforce_rooms[0])
        
        # PHASE 2: BFS propagation from landmarks
        # For each mapped node, try to map its graph neighbors to adjacent VGLC positions
        queue = list(mapping.keys())
        visited = set(mapping.keys())
        
        while queue:
            current_node = queue.pop(0)
            current_pos = mapping[current_node]
            
            # Get all graph neighbors
            neighbors = set(graph.successors(current_node)) | set(graph.predecessors(current_node))
            
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                
                # Find adjacent VGLC positions that aren't used
                adjacent_positions = self._get_adjacent_vglc_positions(current_pos, vglc_rooms, used_positions)
                
                if adjacent_positions:
                    # Score each candidate by content similarity
                    best_pos = None
                    best_score = -1
                    
                    neighbor_attrs = graph.nodes.get(neighbor, {})
                    
                    for pos in adjacent_positions:
                        grid = vglc_rooms[pos]
                        score = self._content_match_score(neighbor_attrs, grid)
                        if score > best_score:
                            best_score = score
                            best_pos = pos
                    
                    if best_pos:
                        mapping[neighbor] = best_pos
                        used_positions.add(best_pos)
                        visited.add(neighbor)
                        queue.append(neighbor)
        
        # PHASE 3: Handle remaining unmapped nodes
        # Use content-based scoring for any remaining graph nodes
        remaining_nodes = set(graph.nodes()) - visited
        remaining_positions = set(vglc_rooms.keys()) - used_positions
        
        for node_id in sorted(remaining_nodes):
            node_attrs = graph.nodes.get(node_id, {})
            best_pos = None
            best_score = -1
            
            for pos in remaining_positions:
                grid = vglc_rooms[pos]
                score = self._content_match_score(node_attrs, grid)
                if score > best_score:
                    best_score = score
                    best_pos = pos
            
            if best_pos is not None:
                mapping[node_id] = best_pos
                used_positions.add(best_pos)
                remaining_positions.discard(best_pos)
        
        return mapping
    
    def _get_adjacent_vglc_positions(self, pos: Tuple[int, int], 
                                      vglc_rooms: Dict[Tuple[int, int], np.ndarray],
                                      used: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Get VGLC positions adjacent to 'pos' that exist and aren't used."""
        r, c = pos
        candidates = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
        return [p for p in candidates if p in vglc_rooms and p not in used]
    
    def _content_match_score(self, node_attrs: Dict, char_grid: np.ndarray) -> float:
        """
        Score how well a graph node matches a VGLC room based on content.
        
        Higher score = better match.
        """
        score = 0.0
        
        # Start room match
        if node_attrs.get('is_start', False):
            if np.any(char_grid == 'S'):
                score += 100  # Strong signal
        
        # Triforce/Goal match
        if node_attrs.get('has_triforce', False):
            if np.any(np.isin(char_grid, ['T', 't', 'G'])):
                score += 100
        
        # Enemy presence (weak signal)
        if node_attrs.get('has_enemy', False):
            if np.any(char_grid == 'M'):
                score += 5
        
        # Door count - more doors = more connected room
        door_count = np.sum(char_grid == 'D')
        edge_count = len(node_attrs.get('contents', []))
        score += min(door_count, 10)  # Cap contribution
        
        # Baseline score for any valid room
        if np.sum(char_grid == 'F') > 10:  # Has floor
            score += 1
        
        return score
    
    def _create_ghost_room_from_graph(self, node_id: int, graph: nx.DiGraph) -> np.ndarray:
        """
        Create a synthetic room for a graph node without VGLC data.
        
        Uses graph attributes to determine room contents.
        """
        # Standard room template
        grid = np.full((16, 11), 'W', dtype='<U1')
        
        # Interior floor
        grid[2:-2, 2:-2] = 'F'
        
        # Add content based on graph attributes
        node_attrs = graph.nodes.get(node_id, {})
        center_r, center_c = 8, 5
        
        if node_attrs.get('is_start', False):
            grid[center_r, center_c] = 'S'
        elif node_attrs.get('has_triforce', False):
            grid[center_r, center_c] = 'T'
        elif node_attrs.get('has_boss', False):
            grid[center_r, center_c] = 'M'  # Boss as enemy
        elif node_attrs.get('has_enemy', False):
            grid[center_r - 1, center_c] = 'M'
        
        # Add doors based on graph edges
        successors = list(graph.successors(node_id))
        predecessors = list(graph.predecessors(node_id))
        edge_count = len(set(successors + predecessors))
        
        # Place doors on edges (up to 4)
        door_positions = [
            (0, 5),    # North
            (15, 5),   # South  
            (8, 0),    # West
            (8, 10),   # East
        ]
        
        for i in range(min(edge_count, 4)):
            dr, dc = door_positions[i]
            grid[dr, dc] = 'D'
            # Also make adjacent tiles doors for 3-wide door
            if i < 2:  # North/South
                grid[dr, dc-1] = 'D'
                grid[dr, dc+1] = 'D'
            else:  # East/West
                grid[dr-1, dc] = 'D'
                grid[dr+1, dc] = 'D'
        
        return grid
    
    def _find_ghost_position(self, node_id: int, graph: nx.DiGraph,
                             existing_positions: Dict[int, Tuple[int, int]]) -> Tuple[int, int]:
        """
        Find a grid position for a ghost room based on its graph neighbors.
        """
        # Get neighbors that already have positions
        neighbors = set(graph.successors(node_id)) | set(graph.predecessors(node_id))
        positioned_neighbors = [n for n in neighbors if n in existing_positions]
        
        if not positioned_neighbors:
            # No positioned neighbors - find first free position
            used = set(existing_positions.values())
            for r in range(20):
                for c in range(20):
                    if (r, c) not in used:
                        return (r, c)
            return (0, 0)  # Fallback
        
        # Find position adjacent to a positioned neighbor
        used = set(existing_positions.values())
        for neighbor in positioned_neighbors:
            nr, nc = existing_positions[neighbor]
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                pos = (nr + dr, nc + dc)
                if pos not in used and pos[0] >= 0 and pos[1] >= 0:
                    return pos
        
        # Spiral out from first neighbor
        nr, nc = existing_positions[positioned_neighbors[0]]
        for radius in range(1, 10):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if abs(dr) == radius or abs(dc) == radius:
                        pos = (nr + dr, nc + dc)
                        if pos not in used and pos[0] >= 0 and pos[1] >= 0:
                            return pos
        
        return (0, 0)
    
    def _ensure_landmarks(self, processed_rooms: Dict[str, RoomData], graph: nx.DiGraph) -> None:
        """
        Ensure START and TRIFORCE tiles exist in the processed rooms.
        """
        has_start = False
        has_triforce = False
        start_room_id = None
        triforce_room_id = None
        
        # Check graph for landmark nodes
        for node_id, attrs in graph.nodes(data=True):
            if attrs.get('is_start', False):
                start_room_id = str(node_id)
            if attrs.get('has_triforce', False):
                triforce_room_id = str(node_id)
        
        # Check if landmarks exist in grids
        for room_id, room in processed_rooms.items():
            if np.any(room.grid == SEMANTIC_PALETTE['START']):
                has_start = True
            if np.any(room.grid == SEMANTIC_PALETTE['TRIFORCE']):
                has_triforce = True
        
        # Inject START if missing
        if not has_start and start_room_id and start_room_id in processed_rooms:
            grid = processed_rooms[start_room_id].grid
            self._inject_landmark(grid, SEMANTIC_PALETTE['START'])
        elif not has_start and processed_rooms:
            # Fallback: first room
            first_room = list(processed_rooms.values())[0]
            self._inject_landmark(first_room.grid, SEMANTIC_PALETTE['START'])
        
        # Inject TRIFORCE if missing
        if not has_triforce and triforce_room_id and triforce_room_id in processed_rooms:
            grid = processed_rooms[triforce_room_id].grid
            self._inject_landmark(grid, SEMANTIC_PALETTE['TRIFORCE'])
        elif not has_triforce and processed_rooms:
            # Fallback: last room
            last_room = list(processed_rooms.values())[-1]
            self._inject_landmark(last_room.grid, SEMANTIC_PALETTE['TRIFORCE'])
    
    def _inject_landmark(self, grid: np.ndarray, landmark_id: int) -> None:
        """Inject a landmark tile at the center of a room."""
        floor_mask = grid == SEMANTIC_PALETTE['FLOOR']
        if np.any(floor_mask):
            positions = np.argwhere(floor_mask)
            center = np.array(grid.shape) // 2
            distances = np.abs(positions - center).sum(axis=1)
            r, c = positions[int(np.argmin(distances))]
            grid[r, c] = landmark_id
    
    def _get_room_graph_attrs(self, graph: nx.DiGraph, room_id: int,
                              layout_positions: Optional[Dict[int, Tuple[int, int]]] = None,
                              char_grid: Optional[np.ndarray] = None) -> Dict[str, Dict]:
        """
        Extract edge attributes for a room's connections.
        
        Enhanced Version: If char_grid is provided, uses pixel-based direction detection
        to verify which directions actually have doors. This prevents the spatial-sequential
        mismatch bug where doors get placed in the wrong locations.
        
        Args:
            graph: NetworkX graph with room connectivity
            room_id: ID of the current room
            layout_positions: Optional explicit spatial layout mapping
            char_grid: Optional character grid to detect actual door positions
        
        Returns:
            Dict mapping direction -> {'type': edge_type, 'neighbor': neighbor_id}
        """
        attrs = {}
        
        if room_id not in graph.nodes:
            return attrs
        
        # Get all neighbors and their edge types
        neighbor_edges = {}
        for neighbor in graph.neighbors(room_id):
            edge_data = graph.get_edge_data(room_id, neighbor)
            if edge_data:
                direction = self._infer_direction(room_id, neighbor, layout_positions)
                neighbor_edges[neighbor] = {
                    'direction': direction,
                    'type': edge_data.get('type', 'open'),
                    'neighbor': neighbor
                }
        
        # PIXEL-BASED VALIDATION (if char_grid provided)
        # This ensures we only report doors that PHYSICALLY exist in the grid
        if char_grid is not None and char_grid.size > 0:
            actual_directions = self._infer_direction_from_grid(char_grid)
            
            # Filter: only keep edges where the direction matches an actual door
            # This prevents "ghost doors" caused by wrong ID-based direction inference
            for neighbor, edge_info in neighbor_edges.items():
                if edge_info['direction'] in actual_directions:
                    attrs[edge_info['direction']] = {
                        'type': edge_info['type'],
                        'neighbor': neighbor
                    }
                # If direction doesn't match actual grid, silently skip
                # (This can happen when graph topology conflicts with text layout)
        else:
            # Fallback: use ID-based inference (legacy behavior)
            for neighbor, edge_info in neighbor_edges.items():
                attrs[edge_info['direction']] = {
                    'type': edge_info['type'],
                    'neighbor': neighbor
                }
        
        return attrs
    
    def _infer_direction_from_grid(self, grid: np.ndarray) -> List[str]:
        """
        Determine which directions have door connections by examining the actual grid pixels.
        This is the CORRECT way - inspect the room borders to see where doors really are.
        
        Args:
            grid: Semantic grid of the room [H, W]
            
        Returns:
            List of directions where doors or openings exist ['north', 'south', 'east', 'west']
        """
        if grid.size == 0:
            return []
            
        h, w = grid.shape
        directions = []
        
        # Define door-like tiles (passable connections)
        door_tiles = {
            SEMANTIC_PALETTE['DOOR_OPEN'],
            SEMANTIC_PALETTE['DOOR_LOCKED'],
            SEMANTIC_PALETTE['DOOR_BOMB'],
            SEMANTIC_PALETTE['DOOR_PUZZLE'],
            SEMANTIC_PALETTE['DOOR_BOSS'],
            SEMANTIC_PALETTE['DOOR_SOFT'],
            SEMANTIC_PALETTE['FLOOR'],  # Sometimes doors are just floor at edges
        }
        
        # Check NORTH (top row, excluding corners)
        if h > 2 and w > 4:
            north_slice = grid[0, 2:w-2]
            if np.any(np.isin(north_slice, list(door_tiles))):
                directions.append('north')
        
        # Check SOUTH (bottom row, excluding corners)
        if h > 2 and w > 4:
            south_slice = grid[h-1, 2:w-2]
            if np.any(np.isin(south_slice, list(door_tiles))):
                directions.append('south')
        
        # Check WEST (left column, excluding corners)
        if h > 4 and w > 2:
            west_slice = grid[2:h-2, 0]
            if np.any(np.isin(west_slice, list(door_tiles))):
                directions.append('west')
        
        # Check EAST (right column, excluding corners)
        if h > 4 and w > 2:
            east_slice = grid[2:h-2, w-1]
            if np.any(np.isin(east_slice, list(door_tiles))):
                directions.append('east')
        
        return directions
    
    def _infer_direction(self, room_id: int, neighbor_id: int,
                        layout_positions: Optional[Dict[int, Tuple[int, int]]] = None) -> str:
        """
        Infer direction to neighbor based on spatial layout.
        
        PRIORITY ORDER:
        1. Use explicit layout positions if available (most accurate)
        2. Fallback to ID-based heuristic (legacy compatibility)
        
        This method is deprecated in favor of _infer_direction_from_grid for new code.
        """
        if layout_positions and room_id in layout_positions and neighbor_id in layout_positions:
            r0, c0 = layout_positions[room_id]
            r1, c1 = layout_positions[neighbor_id]
            if r1 < r0:
                return 'north'
            if r1 > r0:
                return 'south'
            if c1 < c0:
                return 'west'
            if c1 > c0:
                return 'east'

        # Fallback: ID-based heuristic (naive)
        diff = neighbor_id - room_id
        if diff == 1:
            return 'east'
        if diff == -1:
            return 'west'
        if diff > 1:
            return 'south'
        return 'north'
    
    def process_all_dungeons(self, processed_dir: str = None, graph_dir: str = None) -> Dict[str, DungeonData]:
        """
        Process all dungeons in the data folder.
        
        Args:
            processed_dir: Path to Processed/ folder with .txt files
            graph_dir: Path to Graph Processed/ folder with .dot files
            
        Returns:
            Dictionary of dungeon_id -> DungeonData
            
        Note:
            - tlozX_1.txt files are Quest 1 dungeons (use LoZ_X.dot)
            - tlozX_2.txt files are Quest 2 dungeons (use LoZ2_X.dot)
            - Each quest gets a unique dungeon_id to prevent overwrites
        """
        if processed_dir is None:
            processed_dir = self.data_root / "Processed"
        if graph_dir is None:
            graph_dir = self.data_root / "Graph Processed"
        
        processed_dir = Path(processed_dir)
        graph_dir = Path(graph_dir)
        
        results = {}
        
        # Find all map files
        map_files = sorted(processed_dir.glob("*.txt"))
        
        for map_file in map_files:
            if map_file.name == "README.txt":
                continue
                
            # Extract dungeon ID from filename (e.g., tloz1_1.txt -> dungeon 1, quest 1)
            # Pattern: tlozX_Y.txt where X is dungeon number, Y is quest (1 or 2)
            match = re.match(r'tloz(\d+)_(\d+)\.txt', map_file.name)
            if not match:
                continue
                
            dungeon_num = int(match.group(1))  # Dungeon number (1-9)
            quest_num = int(match.group(2))     # Quest number (1 or 2)
            
            # Find corresponding graph file based on quest number
            # Quest 1 uses LoZ_X.dot, Quest 2 uses LoZ2_X.dot
            if quest_num == 1:
                graph_file = graph_dir / f"LoZ_{dungeon_num}.dot"
            else:
                graph_file = graph_dir / f"LoZ2_{dungeon_num}.dot"
            
            if not graph_file.exists():
                print(f"Warning: No graph file found for {map_file.name} (expected {graph_file.name})")
                continue
            
            # Create unique dungeon_id that includes quest number
            # Format: zelda_<dungeon>_quest<quest>
            dungeon_id = f"zelda_{dungeon_num}_quest{quest_num}"
            
            try:
                dungeon_data = self.process_dungeon(
                    str(map_file), 
                    str(graph_file), 
                    dungeon_id
                )
                results[dungeon_id] = dungeon_data
                print(f"Processed {dungeon_id}: {len(dungeon_data.rooms)} rooms")
            except Exception as e:
                print(f"Error processing {dungeon_id}: {e}")
        
        self.processed_dungeons = results
        return results
    
    def save_processed_data(self, output_path: str = None):
        """Save all processed data to disk."""
        import pickle
        
        if output_path is None:
            output_path = self.output_path or (self.data_root / "processed_data.pkl")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        save_data = {}
        for dungeon_id, dungeon in self.processed_dungeons.items():
            save_data[dungeon_id] = {
                'rooms': {
                    rid: {
                        'grid': room.grid,
                        'contents': room.contents,
                        'doors': room.doors,
                        'position': room.position
                    }
                    for rid, room in dungeon.rooms.items()
                },
                'tpe_vectors': dungeon.tpe_vectors,
                'p_matrix': dungeon.p_matrix,
                'node_features': dungeon.node_features,
                'layout': dungeon.layout,
                'graph_edges': list(dungeon.graph.edges(data=True)),
                'graph_nodes': dict(dungeon.graph.nodes(data=True))
            }
        
        with open(output_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Saved processed data to {output_path}")


# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def visualize_semantic_grid(grid: np.ndarray, show_legend: bool = True) -> str:
    """
    Create ASCII visualization of semantic grid for debugging.
    """
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
    }
    
    lines = []
    for row in grid:
        line = ''.join(symbol_map.get(cell, '?') for cell in row)
        lines.append(line)
    
    result = '\n'.join(lines)
    
    if show_legend:
        result += '\n\nLegend: . floor, # wall, O open door, L locked door, X bomb wall'
        result += '\n        E enemy, S start, T triforce, k key, ~ hazard'
    
    return result


# ==========================================
# MAIN ENTRY POINT
# ==========================================

if __name__ == "__main__":
    import sys
    
    # Default data path
    data_root = Path(__file__).parent.parent / "Data" / "The Legend of Zelda"
    
    if len(sys.argv) > 1:
        data_root = Path(sys.argv[1])
    
    print(f"Processing Zelda data from: {data_root}")
    
    adapter = IntelligentDataAdapter(str(data_root))
    
    # Process all dungeons
    dungeons = adapter.process_all_dungeons()
    
    # Save results
    adapter.save_processed_data()
    
    # Print summary
    print("\n=== Processing Summary ===")
    for dungeon_id, dungeon in dungeons.items():
        print(f"{dungeon_id}:")
        print(f"  - Rooms: {len(dungeon.rooms)}")
        print(f"  - Graph nodes: {dungeon.graph.number_of_nodes()}")
        print(f"  - Graph edges: {dungeon.graph.number_of_edges()}")
        print(f"  - TPE shape: {dungeon.tpe_vectors.shape}")
        print(f"  - P-Matrix shape: {dungeon.p_matrix.shape}")
