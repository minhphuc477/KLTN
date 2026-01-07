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
from typing import Dict, List, Tuple, Optional, Any
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

# Room dimensions (standard Zelda dungeon room)
ROOM_HEIGHT = 11  # Interior height (excluding outer walls in some contexts)
ROOM_WIDTH = 16   # Interior width
FULL_ROOM_HEIGHT = 16  # Full room with walls
FULL_ROOM_WIDTH = 22   # Full room with walls

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
                        graph_attrs: Dict[str, Dict] = None) -> np.ndarray:
        """
        Convert character grid to semantic IDs with door logic injection.
        
        The "Defensive Router" ensures:
        1. Doors at invalid positions become walls
        2. Door types are inferred from graph edge attributes
        3. Corner positions are protected from invalid door placement
        
        Args:
            char_grid: 2D numpy array of characters
            room_id: ID of the current room (for graph lookup)
            graph_attrs: Dict of {direction: {'type': edge_type, 'neighbor': neighbor_id}}
        
        Returns:
            Semantic ID grid [H, W]
        """
        if graph_attrs is None:
            graph_attrs = {}
            
        h, w = char_grid.shape
        semantic_grid = np.zeros((h, w), dtype=np.int64)
        
        for r in range(h):
            for c in range(w):
                char = char_grid[r, c]
                
                # Handle door logic specially
                if char == 'D':
                    semantic_id = self._route_door(r, c, h, w, graph_attrs)
                else:
                    # Standard character mapping
                    semantic_id = CHAR_TO_SEMANTIC.get(char, SEMANTIC_PALETTE['FLOOR'])
                
                semantic_grid[r, c] = semantic_id
        
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
        node_pattern = r'(\d+)\s*\[label="([^"]*)"\]'
        for match in re.finditer(node_pattern, content):
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
        Process a complete dungeon from raw files.
        
        Args:
            map_file: Path to VGLC map .txt file
            graph_file: Path to .dot graph file
            dungeon_id: Identifier for this dungeon
            
        Returns:
            DungeonData object with all processed tensors
        """
        # Parse graph first to get room connectivity
        graph = self.parse_dot_graph(graph_file)
        
        # Extract rooms from map
        rooms_data = self.extract_rooms_simple(map_file)
        
        # Process each room
        processed_rooms = {}
        for room_id, char_grid in rooms_data:
            # Get graph attributes for this room (for door logic)
            graph_attrs = self._get_room_graph_attrs(graph, room_id)
            
            # Convert to semantic grid
            semantic_grid = self.defensive_mapper(char_grid, room_id, graph_attrs)
            
            # Get room contents from graph
            contents = []
            if room_id in graph.nodes:
                contents = graph.nodes[room_id].get('contents', [])
            
            processed_rooms[str(room_id)] = RoomData(
                room_id=str(room_id),
                grid=semantic_grid,
                contents=contents,
                doors=graph_attrs
            )
        
        # Compute graph-based features
        tpe_vectors, node_order = self.compute_laplacian_pe(graph)
        node_features = self.extract_node_features(graph, node_order)
        p_matrix = self.build_p_matrix(graph, node_order)
        
        # Create layout (placeholder - actual layout detection is complex)
        n_rooms = len(processed_rooms)
        layout = np.arange(n_rooms).reshape(-1, 1)  # Simple linear layout
        
        return DungeonData(
            dungeon_id=dungeon_id,
            rooms=processed_rooms,
            graph=graph,
            layout=layout,
            tpe_vectors=tpe_vectors,
            p_matrix=p_matrix,
            node_features=node_features
        )
    
    def _get_room_graph_attrs(self, graph: nx.DiGraph, room_id: int) -> Dict[str, Dict]:
        """
        Extract edge attributes for a room's connections.
        
        Note: In actual implementation, this would need layout information
        to map neighbor IDs to directions (north/south/east/west).
        """
        attrs = {}
        
        if room_id not in graph.nodes:
            return attrs
        
        # Get all neighbors and their edge types
        # This is a simplified version - actual implementation needs spatial layout
        for neighbor in graph.neighbors(room_id):
            edge_data = graph.get_edge_data(room_id, neighbor)
            if edge_data:
                # Placeholder direction mapping
                direction = self._infer_direction(room_id, neighbor)
                attrs[direction] = {
                    'type': edge_data.get('type', 'open'),
                    'neighbor': neighbor
                }
        
        return attrs
    
    def _infer_direction(self, room_id: int, neighbor_id: int) -> str:
        """
        Infer direction to neighbor based on IDs.
        
        This is a simplified heuristic - actual implementation would use
        spatial layout information.
        """
        # Heuristic based on typical dungeon layouts
        # Higher ID often means further right/down
        diff = neighbor_id - room_id
        
        if diff == 1:
            return 'east'
        elif diff == -1:
            return 'west'
        elif diff > 1:
            return 'south'
        else:
            return 'north'
    
    def process_all_dungeons(self, processed_dir: str = None, graph_dir: str = None) -> Dict[str, DungeonData]:
        """
        Process all dungeons in the data folder.
        
        Args:
            processed_dir: Path to Processed/ folder with .txt files
            graph_dir: Path to Graph Processed/ folder with .dot files
            
        Returns:
            Dictionary of dungeon_id -> DungeonData
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
                
            # Extract dungeon ID from filename (e.g., tloz1_1.txt -> tloz1)
            # Pattern: tlozX_Y.txt where X is dungeon, Y is sub-part
            match = re.match(r'(tloz\d+)_(\d+)\.txt', map_file.name)
            if not match:
                continue
                
            dungeon_base = match.group(1)  # tloz1
            dungeon_num = int(dungeon_base.replace('tloz', ''))
            
            # Find corresponding graph file
            graph_file = graph_dir / f"LoZ_{dungeon_num}.dot"
            if not graph_file.exists():
                graph_file = graph_dir / f"LoZ2_{dungeon_num}.dot"
            
            if not graph_file.exists():
                print(f"Warning: No graph file found for {map_file.name}")
                continue
            
            dungeon_id = f"zelda_{dungeon_num}"
            
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
