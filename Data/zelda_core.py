"""
ZELDA DUNGEON CORE - Clean Implementation
=========================================
Core logic for VGLC room extraction, graph alignment, and dungeon stitching.

This is the SINGLE SOURCE OF TRUTH for all dungeon processing.

VGLC Format (Zelda Dungeons):
- Each room is 16 rows × 11 columns
- Rooms are arranged in a grid with possible gaps (void regions)
- Characters: F=floor, W=wall, D=door, S=stair, B=block, M=monster, P=element

DOT Graph Format:
- Node labels: s=start, t=triforce, b=boss, e=enemy, k=key, I=item
- Edge labels: k=key_locked, b=bombable, l=soft_locked, empty=open

ML Features (from adapter.py integration):
- Topological Positional Encoding (TPE) via Laplacian eigenvectors
- Node feature vectors (multi-hot encoding)
- P-Matrix (dependency graph encoding)
- Grid-based room extraction

Author: KLTN Thesis Project (Cleaned + Consolidated)
"""

import os
import re
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path

# ==========================================
# SEMANTIC PALETTE (CRITICAL CONSTANTS)
# ==========================================
SEMANTIC_PALETTE = {
    'VOID': 0,
    'FLOOR': 1,
    'WALL': 2,
    'BLOCK': 3,
    'DOOR_OPEN': 10,
    'DOOR_LOCKED': 11,
    'DOOR_BOMB': 12,
    'DOOR_SOFT': 15,
    'ENEMY': 20,
    'START': 21,
    'TRIFORCE': 22,
    'BOSS': 23,
    'KEY': 30,
    'ITEM': 33,
    'ELEMENT': 40,
    'STAIR': 42,
}

# Character to semantic ID mapping
CHAR_TO_SEMANTIC = {
    '-': SEMANTIC_PALETTE['VOID'],
    'F': SEMANTIC_PALETTE['FLOOR'],
    '.': SEMANTIC_PALETTE['FLOOR'],
    'W': SEMANTIC_PALETTE['WALL'],
    'B': SEMANTIC_PALETTE['BLOCK'],
    'M': SEMANTIC_PALETTE['ENEMY'],
    'P': SEMANTIC_PALETTE['ELEMENT'],
    'O': SEMANTIC_PALETTE['FLOOR'],  # Element+Floor - walkable
    'I': SEMANTIC_PALETTE['BLOCK'],  # Element+Block - not walkable
    'D': SEMANTIC_PALETTE['DOOR_OPEN'],
    'S': SEMANTIC_PALETTE['STAIR'],
}

ID_TO_NAME = {v: k for k, v in SEMANTIC_PALETTE.items()}

# Room dimensions (VGLC Zelda standard)
ROOM_HEIGHT = 16
ROOM_WIDTH = 11

# Edge type mapping from graph labels (for ML features)
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


# ==========================================
# GRID-BASED ROOM EXTRACTOR (from adapter.py)
# ==========================================
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
    """

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
        """Check if a slot contains a room (not a gap)."""
        if slot_grid.size == 0:
            return False
        
        dash_count = np.sum(slot_grid == self.GAP_MARKER)
        total = slot_grid.size
        if dash_count > total * 0.7:
            return False
        
        wall_count = np.sum(slot_grid == self.WALL_MARKER)
        floor_count = np.sum(slot_grid == 'F')
        
        return wall_count >= 20 and floor_count >= 5

    def extract(self, filepath: str) -> List[Tuple[Tuple[int, int], np.ndarray]]:
        """
        Extract all rooms from VGLC file using fixed slot grid.
        
        Returns:
            List of ((row_idx, col_idx), room_grid) tuples
        """
        grid = self._load_grid(filepath)
        
        if grid.size == 0:
            return []
        
        h, w = grid.shape
        num_row_slots = h // self.SLOT_HEIGHT
        num_col_slots = w // self.SLOT_WIDTH
        
        rooms = []
        
        for row_slot in range(num_row_slots):
            row_start = row_slot * self.SLOT_HEIGHT
            row_end = row_start + self.SLOT_HEIGHT
            
            for col_slot in range(num_col_slots):
                col_start = col_slot * self.SLOT_WIDTH
                col_end = col_start + self.SLOT_WIDTH
                
                slot_grid = grid[row_start:row_end, col_start:col_end]
                
                if slot_grid.shape[0] < self.SLOT_HEIGHT:
                    pad = np.full((self.SLOT_HEIGHT - slot_grid.shape[0], slot_grid.shape[1]), self.GAP_MARKER)
                    slot_grid = np.vstack([slot_grid, pad])
                if slot_grid.shape[1] < self.SLOT_WIDTH:
                    pad = np.full((slot_grid.shape[0], self.SLOT_WIDTH - slot_grid.shape[1]), self.GAP_MARKER)
                    slot_grid = np.hstack([slot_grid, pad])
                
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


# ==========================================
# ML FEATURE EXTRACTION (from adapter.py)
# ==========================================
class MLFeatureExtractor:
    """
    Extract ML features from dungeon graph structure.
    
    Features:
    - Topological Positional Encoding (TPE) using Laplacian eigenvectors
    - Node feature vectors (multi-hot encoding)
    - P-Matrix (dependency graph encoding)
    """
    
    @staticmethod
    def compute_laplacian_pe(G: nx.Graph, k_dim: int = 8) -> Tuple[np.ndarray, Dict[int, int]]:
        """
        Compute Positional Encoding using Graph Laplacian eigenvectors.
        
        Creates topology-aware position vectors for each node based on
        the graph's spectral properties.
        
        Args:
            G: NetworkX graph (will be treated as undirected)
            k_dim: Number of eigenvector dimensions to use
            
        Returns:
            tpe: Topological Positional Encoding array [N, k_dim]
            node_to_idx: Mapping from node ID to array index
        """
        G_undirected = G.to_undirected() if G.is_directed() else G
        
        nodes = sorted(G_undirected.nodes())
        n = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        if n == 0:
            return np.zeros((0, k_dim)), {}
        
        adj = np.zeros((n, n))
        
        for u, v, data in G_undirected.edges(data=True):
            idx_u, idx_v = node_to_idx[u], node_to_idx[v]
            
            edge_type = data.get('edge_type', 'open')
            weight = 0.5 if edge_type in ['locked', 'bombable', 'boss_locked', 'soft_locked'] else 1.0
            
            adj[idx_u, idx_v] = weight
            adj[idx_v, idx_u] = weight
        
        degrees = np.sum(adj, axis=1)
        D = np.diag(degrees)
        L = D - adj
        
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            
            start_idx = 1
            end_idx = min(start_idx + k_dim, n)
            
            tpe = eigenvectors[:, start_idx:end_idx]
            
            if tpe.shape[1] < k_dim:
                padding = np.zeros((n, k_dim - tpe.shape[1]))
                tpe = np.hstack([tpe, padding])
                
        except np.linalg.LinAlgError:
            tpe = np.zeros((n, k_dim))
        
        return tpe.astype(np.float32), node_to_idx
    
    @staticmethod
    def extract_node_features(G: nx.DiGraph, node_order: Dict[int, int]) -> np.ndarray:
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
            features[idx, 3] = 1.0 if attrs.get('has_boss_key', False) or attrs.get('has_boss', False) else 0.0
            features[idx, 4] = 1.0 if attrs.get('has_triforce', False) or attrs.get('is_triforce', False) else 0.0
        
        return features
    
    @staticmethod
    def build_p_matrix(G: nx.DiGraph, node_order: Dict[int, int]) -> np.ndarray:
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
            edge_type = data.get('edge_type', 'open')
            label = data.get('label', '')
            
            # Determine edge type from label if edge_type not set
            if edge_type == 'open' and label:
                edge_type = EDGE_TYPE_MAP.get(label, 'open')
            
            if edge_type in ('locked', 'key_locked'):
                p_matrix[i, j, 0] = 1.0
                p_matrix[j, i, 0] = 1.0
            elif edge_type == 'bombable':
                p_matrix[i, j, 1] = 1.0
                p_matrix[j, i, 1] = 1.0
            elif edge_type == 'boss_locked':
                p_matrix[i, j, 2] = 1.0
                p_matrix[j, i, 2] = 1.0
        
        return p_matrix


# ==========================================
# DATA CLASSES
# ==========================================
@dataclass
class Room:
    """A single dungeon room."""
    position: Tuple[int, int]  # (row, col) in VGLC grid
    char_grid: np.ndarray      # Raw character grid (16×11)
    semantic_grid: np.ndarray  # Semantic ID grid (16×11)
    doors: Dict[str, bool]     # {N, S, E, W} -> has_door
    has_stair: bool
    has_triforce: bool = False
    has_boss: bool = False
    is_start: bool = False
    graph_node_id: Optional[int] = None


@dataclass 
class Dungeon:
    """A complete dungeon with rooms and connectivity."""
    dungeon_id: str
    rooms: Dict[Tuple[int, int], Room]  # position -> Room
    graph: nx.DiGraph                    # Connectivity graph from DOT
    start_pos: Optional[Tuple[int, int]] = None
    triforce_pos: Optional[Tuple[int, int]] = None
    boss_pos: Optional[Tuple[int, int]] = None


@dataclass
class StitchedDungeon:
    """Result of stitching rooms together."""
    dungeon_id: str
    global_grid: np.ndarray
    room_positions: Dict[Tuple[int, int], Tuple[int, int]]  # room_pos -> global_offset
    start_global: Optional[Tuple[int, int]]
    triforce_global: Optional[Tuple[int, int]]
    graph: Optional[nx.DiGraph] = None  # Store graph for stair connections
    room_to_node: Optional[Dict[Tuple[int, int], int]] = None  # Room position to graph node ID


# ==========================================
# COMPATIBILITY DATACLASSES (from adapter.py)
# ==========================================
@dataclass
class RoomData:
    """
    Represents a single room's data after processing.
    Compatible with adapter.py interface.
    """
    room_id: str
    grid: np.ndarray                    # Semantic grid [H, W]
    contents: List[str] = field(default_factory=list)  # Items in room
    doors: Dict[str, Dict] = field(default_factory=dict)  # Door info by direction
    position: Tuple[int, int] = (0, 0)  # Position in dungeon layout


@dataclass  
class DungeonData:
    """
    Represents a complete dungeon's processed data.
    Compatible with adapter.py interface.
    """
    dungeon_id: str
    rooms: Dict[str, RoomData]          # room_id -> RoomData
    graph: nx.DiGraph                    # Connectivity graph
    layout: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=int))
    tpe_vectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 8), dtype=np.float32))
    p_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 3), dtype=np.float32))
    node_features: np.ndarray = field(default_factory=lambda: np.zeros((0, 5), dtype=np.float32))


# ==========================================
# INVENTORY STATE FOR STATE-SPACE SEARCH
# ==========================================
@dataclass
class InventoryState:
    """
    Track player inventory state for state-space pathfinding.
    
    Keys are SINGLE-USE (consumed when opening a door).
    Bombs are SINGLE-USE but regenerate from enemy drops.
    Items are PERMANENT once collected.
    """
    keys_held: int = 0
    keys_collected: Set[int] = field(default_factory=set)  # Node IDs where keys collected
    doors_opened: Set[Tuple[int, int]] = field(default_factory=set)  # (from, to) edge IDs
    items_collected: Set[str] = field(default_factory=set)  # Item types collected
    
    def copy(self) -> 'InventoryState':
        """Create a copy of this state."""
        return InventoryState(
            keys_held=self.keys_held,
            keys_collected=self.keys_collected.copy(),
            doors_opened=self.doors_opened.copy(),
            items_collected=self.items_collected.copy()
        )
    
    def __hash__(self):
        """Hash for use in visited sets."""
        return hash((
            self.keys_held,
            frozenset(self.keys_collected),
            frozenset(self.doors_opened),
            frozenset(self.items_collected)
        ))
    
    def __eq__(self, other):
        if not isinstance(other, InventoryState):
            return False
        return (self.keys_held == other.keys_held and
                self.keys_collected == other.keys_collected and
                self.doors_opened == other.doors_opened and
                self.items_collected == other.items_collected)


class ValidationMode:
    """Validation modes for dungeon solvability checking."""
    STRICT = 'strict'       # Only normal doors (what's visible in tiles)
    REALISTIC = 'realistic' # Normal + soft-locked + stairs (no items needed)
    FULL = 'full'           # All edges with full inventory tracking


# ==========================================
# STATE-SPACE GRAPH SOLVER
# ==========================================
class StateSpaceGraphSolver:
    """
    State-space search pathfinder that tracks inventory.
    
    This solver properly handles:
    - Key collection from rooms with 'k' label
    - Key consumption for 'k' (key-locked) edges
    - Bombable walls 'b' (assumed infinite bombs)
    - Soft-locked doors 'l' (one-way, always passable forward)
    - Stairs/warps 's' (bidirectional teleports)
    """
    
    def __init__(self, graph: nx.DiGraph, mode: str = ValidationMode.FULL):
        """
        Initialize solver.
        
        Args:
            graph: NetworkX DiGraph from DOT file
            mode: ValidationMode (STRICT, REALISTIC, FULL)
        """
        self.graph = graph
        self.mode = mode
        
        # Build key room lookup (rooms that give keys)
        self.key_rooms = set()
        self.item_rooms = {}  # node_id -> item_type
        
        for node_id, data in graph.nodes(data=True):
            label = data.get('label', '')
            parts = [p.strip() for p in label.split(',')]
            
            if 'k' in parts:
                self.key_rooms.add(node_id)
            if 'I' in parts:
                self.item_rooms[node_id] = 'key_item'
            if 'i' in parts:
                self.item_rooms[node_id] = 'minor_item'
    
    def can_traverse_edge(self, from_node: int, to_node: int, 
                          state: InventoryState) -> Tuple[bool, InventoryState, str]:
        """
        Check if an edge can be traversed with current inventory.
        
        Args:
            from_node: Source node ID
            to_node: Destination node ID
            state: Current inventory state
            
        Returns:
            (can_traverse, new_state, edge_type)
        """
        edge_data = self.graph.get_edge_data(from_node, to_node)
        if not edge_data:
            return False, state, 'none'
        
        edge_label = edge_data.get('label', '')
        edge_type = edge_data.get('edge_type', 'open')
        edge_id = (from_node, to_node)
        
        new_state = state.copy()
        
        # STRICT mode: only normal doors
        if self.mode == ValidationMode.STRICT:
            if edge_label != '':
                return False, state, edge_type
            return True, new_state, 'open'
        
        # REALISTIC mode: normal + soft-locked + stairs
        if self.mode == ValidationMode.REALISTIC:
            if edge_label in ('', 'l', 's'):
                return True, new_state, edge_type
            return False, state, edge_type
        
        # FULL mode: all edges with inventory tracking
        if edge_label == '':
            # Normal door - always passable
            return True, new_state, 'open'
        
        elif edge_label == 'k':
            # Key-locked door
            if edge_id in state.doors_opened:
                return True, new_state, 'key_locked'  # Already opened
            
            if state.keys_held > 0:
                new_state.keys_held -= 1
                new_state.doors_opened.add(edge_id)
                # Also add reverse edge as opened (doors stay open)
                new_state.doors_opened.add((to_node, from_node))
                return True, new_state, 'key_locked'
            
            return False, state, 'key_locked'
        
        elif edge_label == 'b':
            # Bombable wall - assume infinite bombs for now
            if edge_id in state.doors_opened:
                return True, new_state, 'bombable'
            
            # Bomb it open (permanent)
            new_state.doors_opened.add(edge_id)
            new_state.doors_opened.add((to_node, from_node))
            return True, new_state, 'bombable'
        
        elif edge_label == 'l':
            # Soft-locked (one-way) - always passable forward
            return True, new_state, 'soft_locked'
        
        elif edge_label == 's':
            # Stair/warp - bidirectional teleport
            return True, new_state, 'stair'
        
        elif edge_label == 'I':
            # Item-locked - check if we have the required item
            if 'key_item' in state.items_collected:
                return True, new_state, 'item_locked'
            return False, state, 'item_locked'
        
        elif edge_label in ('S', 'S1'):
            # Switch-locked - assume puzzle solved
            return True, new_state, 'switch_locked'
        
        # Unknown edge type - allow traversal
        return True, new_state, edge_label
    
    def collect_room_items(self, node: int, state: InventoryState) -> InventoryState:
        """
        Collect items when entering a room.
        
        Args:
            node: Node ID of room being entered
            state: Current inventory state
            
        Returns:
            Updated inventory state
        """
        new_state = state.copy()
        
        # Collect key if room has one and not yet collected
        if node in self.key_rooms and node not in state.keys_collected:
            new_state.keys_held += 1
            new_state.keys_collected.add(node)
        
        # Collect item if room has one
        if node in self.item_rooms and node not in state.keys_collected:
            item_type = self.item_rooms[node]
            new_state.items_collected.add(item_type)
        
        return new_state
    
    def solve(self, start_node: int, goal_node: int) -> Dict:
        """
        Find path from start to goal using state-space BFS.
        
        Args:
            start_node: Starting node ID
            goal_node: Goal node ID
            
        Returns:
            Dict with solvable, path, inventory_final, edge_types
        """
        from collections import deque
        
        # Initial state: collect items in start room
        initial_state = InventoryState()
        initial_state = self.collect_room_items(start_node, initial_state)
        
        # State: (node, inventory_hash)
        # Track: (node, inventory) -> (path, edge_types)
        visited = {}
        
        # Queue: (node, inventory, path, edge_types)
        queue = deque([(start_node, initial_state, [start_node], [])])
        visited[(start_node, hash(initial_state))] = True
        
        keys_available_max = 0
        keys_used_total = 0
        
        while queue:
            current_node, current_state, path, edge_types = queue.popleft()
            
            # Track max keys
            keys_available_max = max(keys_available_max, 
                                     current_state.keys_held + len(current_state.keys_collected))
            
            # Check if we reached the goal
            if current_node == goal_node:
                keys_used = len([e for e in edge_types if e == 'key_locked'])
                return {
                    'solvable': True,
                    'path': path,
                    'path_length': len(path) - 1,
                    'rooms_traversed': len(path),
                    'edge_types': edge_types,
                    'keys_available': len(current_state.keys_collected),
                    'keys_used': keys_used,
                    'final_inventory': current_state
                }
            
            # Explore neighbors
            for neighbor in self.graph.neighbors(current_node):
                can_traverse, new_state, edge_type = self.can_traverse_edge(
                    current_node, neighbor, current_state
                )
                
                if not can_traverse:
                    continue
                
                # Collect items in the new room
                new_state = self.collect_room_items(neighbor, new_state)
                
                # Check if this state was visited
                state_key = (neighbor, hash(new_state))
                if state_key in visited:
                    continue
                
                visited[state_key] = True
                queue.append((
                    neighbor, 
                    new_state, 
                    path + [neighbor],
                    edge_types + [edge_type]
                ))
        
        # No path found
        return {
            'solvable': False,
            'reason': f'No path from {start_node} to {goal_node} with current inventory constraints',
            'mode': self.mode,
            'keys_found': keys_available_max
        }


# ==========================================
# VGLC PARSER (Auto-Alignment)
# ==========================================
class VGLCParser:
    """
    Parse VGLC text files into room grids.
    
    Uses AUTO-ALIGNMENT to handle variable padding:
    1. Find the first column containing actual content (not void)
    2. Slice rooms from that offset
    """
    
    def __init__(self):
        pass
    
    def parse(self, filepath: str) -> Dict[Tuple[int, int], Room]:
        """
        Parse VGLC file into rooms.
        
        Args:
            filepath: Path to .txt VGLC file
            
        Returns:
            Dict mapping (row, col) -> Room
        """
        with open(filepath, 'r') as f:
            lines = [line.rstrip('\n') for line in f]
        
        if not lines:
            return {}
        
        # Pad all lines to same width
        max_width = max(len(line) for line in lines)
        lines = [line.ljust(max_width, '-') for line in lines]
        
        # Calculate grid dimensions
        num_rows = len(lines) // ROOM_HEIGHT
        num_cols = max_width // ROOM_WIDTH
        
        rooms = {}
        
        for row in range(num_rows):
            for col in range(num_cols):
                # Extract room slice
                room_chars = []
                is_void = True
                has_stair = False
                
                for r in range(ROOM_HEIGHT):
                    y = row * ROOM_HEIGHT + r
                    x_start = col * ROOM_WIDTH
                    x_end = x_start + ROOM_WIDTH
                    
                    if y < len(lines):
                        line_slice = lines[y][x_start:x_end]
                        if len(line_slice) < ROOM_WIDTH:
                            line_slice = line_slice.ljust(ROOM_WIDTH, '-')
                    else:
                        line_slice = '-' * ROOM_WIDTH
                    
                    room_chars.append(list(line_slice))
                    
                    # Check for content
                    if any(c not in '-' for c in line_slice):
                        is_void = False
                    if 'S' in line_slice:
                        has_stair = True
                
                if is_void:
                    continue
                
                # Create numpy array
                char_grid = np.array(room_chars, dtype='<U1')
                
                # Detect doors
                doors = self._detect_doors(char_grid)
                
                # Convert to semantic grid
                semantic_grid = self._to_semantic(char_grid, doors)
                
                rooms[(row, col)] = Room(
                    position=(row, col),
                    char_grid=char_grid,
                    semantic_grid=semantic_grid,
                    doors=doors,
                    has_stair=has_stair,
                )
        
        return rooms
    
    def _detect_doors(self, char_grid: np.ndarray) -> Dict[str, bool]:
        """Detect door presence in each direction."""
        # Door positions in standard VGLC room:
        # North: row 1, center columns
        # South: row 14, center columns
        # West: col 1, center rows (rows 7-8)
        # East: col 9, center rows (rows 7-8)
        
        doors = {}
        
        # North door (row 1)
        north_row = ''.join(char_grid[1, :]) if char_grid.shape[0] > 1 else ''
        doors['N'] = 'D' in north_row
        
        # South door (row 14)
        south_row = ''.join(char_grid[14, :]) if char_grid.shape[0] > 14 else ''
        doors['S'] = 'D' in south_row
        
        # West door (col 1, rows 7-8)
        west_col = ''.join(char_grid[7:9, 1]) if char_grid.shape[1] > 1 else ''
        doors['W'] = 'D' in west_col
        
        # East door (col 9, rows 7-8)
        east_col = ''.join(char_grid[7:9, 9]) if char_grid.shape[1] > 9 else ''
        doors['E'] = 'D' in east_col
        
        return doors
    
    def _to_semantic(self, char_grid: np.ndarray, doors: Dict[str, bool]) -> np.ndarray:
        """Convert character grid to semantic ID grid."""
        semantic = np.zeros(char_grid.shape, dtype=np.int32)
        
        for r in range(char_grid.shape[0]):
            for c in range(char_grid.shape[1]):
                char = char_grid[r, c]
                semantic[r, c] = CHAR_TO_SEMANTIC.get(char, SEMANTIC_PALETTE['VOID'])
        
        # Handle corridor rooms (rooms with doors but void interior)
        # These need void converted to floor for pathfinding
        has_any_door = any(doors.values())
        interior = semantic[2:14, 2:9]  # Interior region
        void_count = np.sum(interior == SEMANTIC_PALETTE['VOID'])
        total_interior = interior.size
        
        if has_any_door and void_count > total_interior * 0.5:
            # This is a corridor room - convert void to floor
            mask = semantic == SEMANTIC_PALETTE['VOID']
            # Only convert interior void (not outer walls)
            for r in range(2, 14):
                for c in range(2, 9):
                    if semantic[r, c] == SEMANTIC_PALETTE['VOID']:
                        semantic[r, c] = SEMANTIC_PALETTE['FLOOR']
        
        return semantic


# ==========================================
# DOT GRAPH PARSER
# ==========================================
class DOTParser:
    """Parse DOT graph files."""
    
    def parse(self, filepath: str) -> nx.DiGraph:
        """
        Parse DOT file into NetworkX graph.
        
        Args:
            filepath: Path to .dot file
            
        Returns:
            NetworkX DiGraph with node/edge attributes
        """
        with open(filepath, 'r') as f:
            content = f.read()
        
        graph = nx.DiGraph()
        
        # Parse nodes: 0 [label="e"]
        node_pattern = r'^(\d+)\s*\[label="([^"]*)"\]'
        for match in re.finditer(node_pattern, content, re.MULTILINE):
            node_id = int(match.group(1))
            label = match.group(2).strip()
            
            # Parse label contents
            parts = [p.strip() for p in label.split(',')]
            
            graph.add_node(node_id, 
                          label=label,
                          is_start='s' in parts,
                          is_triforce='t' in parts,
                          is_boss='b' in parts,
                          has_key='k' in parts,
                          has_item='I' in parts or 'i' in parts,
                          has_enemy='e' in parts,
                          has_puzzle='p' in parts)
        
        # Parse edges: 7 -> 8 [label="k"]
        edge_pattern = r'(\d+)\s*->\s*(\d+)\s*\[label="([^"]*)"\]'
        for match in re.finditer(edge_pattern, content):
            src = int(match.group(1))
            dst = int(match.group(2))
            label = match.group(3)
            
            edge_type = 'open'
            if label == 'k':
                edge_type = 'key_locked'
            elif label == 'b':
                edge_type = 'bombable'
            elif label == 'l':
                edge_type = 'soft_locked'
            elif label == 'I':
                edge_type = 'item_locked'
            
            graph.add_edge(src, dst, label=label, edge_type=edge_type)
        
        return graph


# ==========================================
# ROOM-TO-GRAPH MATCHER
# ==========================================
class RoomGraphMatcher:
    """
    Match VGLC rooms to DOT graph nodes.
    
    Strategy:
    1. Find START node in graph (label='s')
    2. Find START room in VGLC (has STAIR 'S')
    3. Use BFS to match remaining nodes based on adjacency
    """
    
    def match(self, rooms: Dict[Tuple[int, int], Room], 
              graph: nx.DiGraph) -> Dungeon:
        """
        Match rooms to graph nodes and return complete Dungeon.
        
        Args:
            rooms: Dict of VGLC rooms by position
            graph: DOT graph
            
        Returns:
            Dungeon with rooms annotated with graph info
        """
        dungeon = Dungeon(
            dungeon_id="",
            rooms=rooms,
            graph=graph
        )
        
        # Find special nodes in graph
        start_node = None
        triforce_node = None
        boss_node = None
        
        for node, attrs in graph.nodes(data=True):
            if attrs.get('is_start'):
                start_node = node
            if attrs.get('is_triforce'):
                triforce_node = node
            if attrs.get('is_boss'):
                boss_node = node
        
        # Find START room (has STAIR WITH doors - not isolated secret rooms)
        start_room_pos = None
        stair_rooms_with_doors = []
        
        for pos, room in rooms.items():
            if room.has_stair:
                door_count = sum(room.doors.values())
                if door_count > 0:
                    stair_rooms_with_doors.append((pos, door_count))
        
        # Prefer STAIR room with doors (actual entrance)
        if stair_rooms_with_doors:
            # Use the one with most doors
            stair_rooms_with_doors.sort(key=lambda x: x[1], reverse=True)
            start_room_pos = stair_rooms_with_doors[0][0]
        
        # Fallback: use room with most doors as hub (NOT isolated stairs)
        if start_room_pos is None:
            max_doors = 0
            for pos, room in rooms.items():
                door_count = sum(room.doors.values())
                if door_count > max_doors:
                    max_doors = door_count
                    start_room_pos = pos
        
        if start_room_pos:
            rooms[start_room_pos].is_start = True
            dungeon.start_pos = start_room_pos
        
        # Build room adjacency and match rooms to graph nodes using BFS
        room_adjacency = self._build_room_adjacency(rooms)
        
        # Match rooms to graph nodes using parallel BFS
        room_to_node, node_to_room = self._match_rooms_to_nodes_bfs(
            rooms, room_adjacency, graph, start_room_pos, start_node
        )
        
        # Store mapping in rooms
        for room_pos, node_id in room_to_node.items():
            rooms[room_pos].graph_node_id = node_id
        
        # Find TRIFORCE room using the mapping or graph distance heuristic
        triforce_room_pos = None
        graph_path_length = 0
        
        # Calculate expected path length from graph
        if triforce_node is not None and start_node is not None:
            try:
                graph_path = nx.shortest_path(graph.to_undirected(), start_node, triforce_node)
                graph_path_length = len(graph_path) - 1  # Number of edges
            except nx.NetworkXNoPath:
                graph_path_length = len(rooms) // 2  # Fallback: estimate as half the dungeon
        
        # Try to use node mapping first
        if triforce_node is not None:
            triforce_room_pos = node_to_room.get(triforce_node)
        
        # If mapping failed, find room at approximately the graph path distance
        if triforce_room_pos is None and graph_path_length > 0:
            triforce_room_pos = self._find_room_at_distance(rooms, room_adjacency, start_room_pos, graph_path_length)
        
        # Last fallback: use dead-end farthest from start
        if triforce_room_pos is None and start_room_pos:
            triforce_room_pos = self._find_farthest_dead_end(rooms, start_room_pos)
        
        if triforce_room_pos:
            rooms[triforce_room_pos].has_triforce = True
            dungeon.triforce_pos = triforce_room_pos
        
        return dungeon
    
    def _match_rooms_to_nodes_bfs(self, rooms: Dict[Tuple[int, int], Room],
                                   room_adjacency: Dict[Tuple[int, int], List[Tuple[int, int]]],
                                   graph: nx.DiGraph,
                                   start_room: Optional[Tuple[int, int]],
                                   start_node: Optional[int]) -> Tuple[Dict, Dict]:
        """
        Match rooms to graph nodes using parallel BFS from start.
        
        Returns:
            (room_to_node, node_to_room) mappings
        """
        room_to_node = {}
        node_to_room = {}
        
        if start_room is None or start_node is None:
            return room_to_node, node_to_room
        
        # Initialize with start
        room_to_node[start_room] = start_node
        node_to_room[start_node] = start_room
        
        # BFS queues for both graph and rooms
        from collections import deque
        
        room_queue = deque([start_room])
        visited_rooms = {start_room}
        visited_nodes = {start_node}
        
        while room_queue:
            current_room = room_queue.popleft()
            current_node = room_to_node.get(current_room)
            
            if current_node is None:
                continue
            
            # Get neighbors in both spaces
            room_neighbors = [r for r in room_adjacency.get(current_room, []) if r not in visited_rooms]
            graph_neighbors = [n for n in set(graph.successors(current_node)) | set(graph.predecessors(current_node)) 
                             if n not in visited_nodes]
            
            # Match neighbors by door count similarity
            for room_neighbor in room_neighbors:
                if not graph_neighbors:
                    break
                
                room_door_count = sum(rooms[room_neighbor].doors.values())
                
                # Find best matching graph neighbor
                best_node = None
                best_score = -float('inf')
                
                for node in graph_neighbors:
                    node_degree = graph.in_degree(node) + graph.out_degree(node)
                    # Score: prefer nodes with similar connectivity
                    score = -abs(room_door_count * 2 - node_degree)  # *2 because edges are bidirectional
                    if score > best_score:
                        best_score = score
                        best_node = node
                
                if best_node is not None:
                    room_to_node[room_neighbor] = best_node
                    node_to_room[best_node] = room_neighbor
                    visited_rooms.add(room_neighbor)
                    visited_nodes.add(best_node)
                    graph_neighbors.remove(best_node)
                    room_queue.append(room_neighbor)
        
        # FALLBACK: Match remaining unmapped rooms to unmapped nodes
        # This handles disconnected components
        unmapped_rooms = [pos for pos in rooms.keys() if pos not in room_to_node]
        unmapped_nodes = [n for n in graph.nodes() if n not in node_to_room]
        
        if unmapped_rooms and unmapped_nodes:
            # Match by similarity (door count vs node degree)
            for room_pos in unmapped_rooms:
                if not unmapped_nodes:
                    break
                
                room_door_count = sum(rooms[room_pos].doors.values())
                
                # Find best matching unmapped node
                best_node = None
                best_score = -float('inf')
                
                for node in unmapped_nodes:
                    node_degree = graph.in_degree(node) + graph.out_degree(node)
                    score = -abs(room_door_count * 2 - node_degree)
                    
                    # Bonus for special node types
                    node_data = graph.nodes[node]
                    if node_data.get('is_triforce') and rooms[room_pos].has_triforce:
                        score += 100
                    if node_data.get('is_boss') and rooms[room_pos].has_boss:
                        score += 100
                    
                    if score > best_score:
                        best_score = score
                        best_node = node
                
                if best_node is not None:
                    room_to_node[room_pos] = best_node
                    node_to_room[best_node] = room_pos
                    unmapped_nodes.remove(best_node)
        
        return room_to_node, node_to_room
    
    def _find_room_at_distance(self, rooms: Dict[Tuple[int, int], Room],
                                room_adjacency: Dict[Tuple[int, int], List[Tuple[int, int]]],
                                start_pos: Tuple[int, int],
                                target_distance: int) -> Optional[Tuple[int, int]]:
        """Find a room at approximately the target distance from start.
        
        Prioritizes dead-ends (1 door) at or near the target distance.
        """
        from collections import deque
        
        distances = {start_pos: 0}
        queue = deque([start_pos])
        
        while queue:
            pos = queue.popleft()
            for neighbor in room_adjacency.get(pos, []):
                if neighbor not in distances:
                    distances[neighbor] = distances[pos] + 1
                    queue.append(neighbor)
        
        # Find candidate rooms prioritized by:
        # 1. Distance from target (closer is better)
        # 2. Dead-end rooms (1 door) preferred
        # 3. Rooms farther from start preferred if tied
        
        candidates = []
        for room_pos, dist in distances.items():
            if room_pos == start_pos:
                continue
            door_count = sum(rooms[room_pos].doors.values())
            is_dead_end = door_count == 1
            distance_diff = abs(dist - target_distance)
            # Score: lower is better (distance_diff, then non-dead-end penalty, then negative distance)
            score = (distance_diff, 0 if is_dead_end else 1, -dist)
            candidates.append((score, room_pos))
        
        if candidates:
            candidates.sort()
            return candidates[0][1]
        
        return None
    
    def _trace_path_to_position(self, rooms: Dict[Tuple[int, int], Room],
                                 start_pos: Tuple[int, int],
                                 graph: nx.DiGraph,
                                 path: List[int]) -> Optional[Tuple[int, int]]:
        """
        Trace a graph path through VGLC rooms.
        
        This is approximate - we follow doors in the direction
        suggested by the graph path length.
        """
        # For now, just return the farthest room
        # A more sophisticated approach would track room-to-node mapping
        return None
    
    def _find_farthest_dead_end(self, rooms: Dict[Tuple[int, int], Room],
                                 start_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Find the dead-end room (1 door) farthest from start."""
        # Build room adjacency
        adjacency = self._build_room_adjacency(rooms)
        
        # BFS to find distances
        from collections import deque
        distances = {start_pos: 0}
        queue = deque([start_pos])
        
        while queue:
            pos = queue.popleft()
            for neighbor in adjacency.get(pos, []):
                if neighbor not in distances:
                    distances[neighbor] = distances[pos] + 1
                    queue.append(neighbor)
        
        # Find farthest dead-end (1 door)
        farthest = None
        max_dist = -1
        
        for pos, room in rooms.items():
            door_count = sum(room.doors.values())
            dist = distances.get(pos, 0)
            
            if door_count == 1 and dist > max_dist:
                max_dist = dist
                farthest = pos
        
        # Fallback: just farthest room
        if farthest is None:
            for pos, dist in distances.items():
                if dist > max_dist:
                    max_dist = dist
                    farthest = pos
        
        return farthest
    
    def _build_room_adjacency(self, rooms: Dict[Tuple[int, int], Room]) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        """Build adjacency list from room door connections."""
        adjacency = {pos: [] for pos in rooms}
        
        for pos, room in rooms.items():
            row, col = pos
            
            # Check each door direction
            if room.doors.get('N') and (row - 1, col) in rooms:
                if rooms[(row - 1, col)].doors.get('S'):
                    adjacency[pos].append((row - 1, col))
            
            if room.doors.get('S') and (row + 1, col) in rooms:
                if rooms[(row + 1, col)].doors.get('N'):
                    adjacency[pos].append((row + 1, col))
            
            if room.doors.get('W') and (row, col - 1) in rooms:
                if rooms[(row, col - 1)].doors.get('E'):
                    adjacency[pos].append((row, col - 1))
            
            if room.doors.get('E') and (row, col + 1) in rooms:
                if rooms[(row, col + 1)].doors.get('W'):
                    adjacency[pos].append((row, col + 1))
        
        return adjacency


# ==========================================
# DUNGEON STITCHER
# ==========================================
class DungeonStitcher:
    """
    Stitch rooms into a global grid.
    
    Uses VGLC positions directly (graph-constrained).
    Connects doors by punching through shared walls.
    """
    
    def stitch(self, dungeon: Dungeon, compact: bool = True) -> StitchedDungeon:
        """
        Stitch dungeon rooms into global grid.
        
        Args:
            dungeon: Dungeon with rooms
            compact: If True, remove empty rows/columns of rooms
            
        Returns:
            StitchedDungeon with global grid
        """
        if not dungeon.rooms:
            return StitchedDungeon(
                dungeon_id=dungeon.dungeon_id,
                global_grid=np.zeros((1, 1), dtype=np.int32),
                room_positions={},
                start_global=None,
                triforce_global=None,
                graph=dungeon.graph,
                room_to_node={}
            )
        
        # For compact mode: remap positions to eliminate gaps
        if compact:
            rooms_remapped, pos_remap = self._compact_rooms(dungeon.rooms)
        else:
            rooms_remapped = dungeon.rooms
            pos_remap = {pos: pos for pos in dungeon.rooms.keys()}
        
        # Calculate grid bounds using remapped positions
        max_row = max(pos[0] for pos in rooms_remapped.keys())
        max_col = max(pos[1] for pos in rooms_remapped.keys())
        
        # Global grid size (no padding - walls overlap)
        global_height = (max_row + 1) * ROOM_HEIGHT
        global_width = (max_col + 1) * ROOM_WIDTH
        
        # Initialize with void
        global_grid = np.zeros((global_height, global_width), dtype=np.int32)
        
        room_positions = {}
        
        # Place each room using remapped positions
        for pos, room in rooms_remapped.items():
            row, col = pos
            r_offset = row * ROOM_HEIGHT
            c_offset = col * ROOM_WIDTH
            
            room_positions[pos] = (r_offset, c_offset)
            
            # Copy semantic grid
            h, w = room.semantic_grid.shape
            global_grid[r_offset:r_offset+h, c_offset:c_offset+w] = room.semantic_grid
        
        # Connect doors by punching through walls
        self._connect_doors(global_grid, rooms_remapped)
        
        # Mark special positions
        start_global = None
        triforce_global = None
        
        # Remap start/triforce positions
        start_pos_remapped = pos_remap.get(dungeon.start_pos) if dungeon.start_pos else None
        triforce_pos_remapped = pos_remap.get(dungeon.triforce_pos) if dungeon.triforce_pos else None
        
        if start_pos_remapped:
            r_off, c_off = room_positions[start_pos_remapped]
            # Find floor tile in start room for actual start position
            start_global = self._find_floor_near_door(global_grid, r_off, c_off)
            # Mark the start position in the grid
            if start_global:
                global_grid[start_global[0], start_global[1]] = SEMANTIC_PALETTE['START']
        
        if triforce_pos_remapped:
            r_off, c_off = room_positions[triforce_pos_remapped]
            # Place triforce marker at center
            center_r = r_off + ROOM_HEIGHT // 2
            center_c = c_off + ROOM_WIDTH // 2
            global_grid[center_r, center_c] = SEMANTIC_PALETTE['TRIFORCE']
            triforce_global = (center_r, center_c)
        
        return StitchedDungeon(
            dungeon_id=dungeon.dungeon_id,
            global_grid=global_grid,
            room_positions=room_positions,
            start_global=start_global,
            triforce_global=triforce_global,
            graph=dungeon.graph,
            room_to_node={pos_remap[old_pos]: room.graph_node_id 
                          for old_pos, room in dungeon.rooms.items()
                          if old_pos in pos_remap and room.graph_node_id is not None}
        )
    
    def _compact_rooms(self, rooms: Dict[Tuple[int, int], Room]) -> Tuple[Dict[Tuple[int, int], Room], Dict[Tuple[int, int], Tuple[int, int]]]:
        """
        Remap room positions to eliminate empty rows/columns.
        
        Returns:
            (remapped_rooms, original_to_new_pos_map)
        """
        # Get all occupied rows and columns
        occupied_rows = sorted(set(pos[0] for pos in rooms.keys()))
        occupied_cols = sorted(set(pos[1] for pos in rooms.keys()))
        
        # Create mapping from old indices to compact indices
        row_remap = {old_r: new_r for new_r, old_r in enumerate(occupied_rows)}
        col_remap = {old_c: new_c for new_c, old_c in enumerate(occupied_cols)}
        
        # Remap all rooms
        remapped_rooms = {}
        pos_map = {}
        
        for pos, room in rooms.items():
            old_row, old_col = pos
            new_pos = (row_remap[old_row], col_remap[old_col])
            
            # Create new room with remapped position
            new_room = Room(
                position=new_pos,
                char_grid=room.char_grid,
                semantic_grid=room.semantic_grid,
                doors=room.doors,
                has_stair=room.has_stair,
                has_triforce=room.has_triforce,
                has_boss=room.has_boss,
                is_start=room.is_start,
                graph_node_id=room.graph_node_id
            )
            remapped_rooms[new_pos] = new_room
            pos_map[pos] = new_pos
        
        return remapped_rooms, pos_map
    
    def _connect_doors(self, grid: np.ndarray, rooms: Dict[Tuple[int, int], Room]):
        """
        Punch through walls to connect door pairs.
        
        For each pair of adjacent rooms with matching doors,
        ensure BOTH sides of the boundary are passable.
        Also ensures internal room connectivity by clearing floor paths.
        """
        for pos, room in rooms.items():
            row, col = pos
            r_base = row * ROOM_HEIGHT
            c_base = col * ROOM_WIDTH
            
            # North door - connect to room above
            if room.doors.get('N'):
                north_pos = (row - 1, col)
                if north_pos in rooms and rooms[north_pos].doors.get('S'):
                    # Punch through wall row at boundary (this room's top row)
                    wall_row = r_base  # Row 0 of this room
                    for c in range(c_base + 3, c_base + 8):  # Door columns
                        if 0 <= c < grid.shape[1]:
                            grid[wall_row, c] = SEMANTIC_PALETTE['FLOOR']
                            # Also punch the row above (bottom row of north room)
                            if wall_row > 0:
                                grid[wall_row - 1, c] = SEMANTIC_PALETTE['FLOOR']
                    # Ensure path from door to room interior
                    for r in range(r_base + 1, r_base + 4):
                        for c in range(c_base + 4, c_base + 7):
                            if grid[r, c] == SEMANTIC_PALETTE['WALL']:
                                grid[r, c] = SEMANTIC_PALETTE['FLOOR']
            
            # South door - connect to room below
            if room.doors.get('S'):
                south_pos = (row + 1, col)
                if south_pos in rooms and rooms[south_pos].doors.get('N'):
                    wall_row = r_base + ROOM_HEIGHT - 1  # Last row of this room
                    for c in range(c_base + 3, c_base + 8):
                        if 0 <= c < grid.shape[1]:
                            grid[wall_row, c] = SEMANTIC_PALETTE['FLOOR']
                            # Also punch the row below (top row of south room)
                            if wall_row + 1 < grid.shape[0]:
                                grid[wall_row + 1, c] = SEMANTIC_PALETTE['FLOOR']
                    # Ensure path from door to room interior
                    for r in range(r_base + ROOM_HEIGHT - 4, r_base + ROOM_HEIGHT - 1):
                        for c in range(c_base + 4, c_base + 7):
                            if grid[r, c] == SEMANTIC_PALETTE['WALL']:
                                grid[r, c] = SEMANTIC_PALETTE['FLOOR']
            
            # West door - connect to room left
            if room.doors.get('W'):
                west_pos = (row, col - 1)
                if west_pos in rooms and rooms[west_pos].doors.get('E'):
                    wall_col = c_base  # Col 0 of this room
                    for r in range(r_base + 5, r_base + 11):  # Door rows (expanded)
                        if 0 <= r < grid.shape[0]:
                            grid[r, wall_col] = SEMANTIC_PALETTE['FLOOR']
                            # Also punch the column to the left (right edge of west room)
                            if wall_col > 0:
                                grid[r, wall_col - 1] = SEMANTIC_PALETTE['FLOOR']
                    # Ensure path from door to room interior
                    for r in range(r_base + 6, r_base + 10):
                        for c in range(c_base + 1, c_base + 4):
                            if grid[r, c] == SEMANTIC_PALETTE['WALL']:
                                grid[r, c] = SEMANTIC_PALETTE['FLOOR']
            
            # East door - connect to room right
            if room.doors.get('E'):
                east_pos = (row, col + 1)
                if east_pos in rooms and rooms[east_pos].doors.get('W'):
                    wall_col = c_base + ROOM_WIDTH - 1  # Last col of this room
                    for r in range(r_base + 5, r_base + 11):  # Door rows (expanded)
                        if 0 <= r < grid.shape[0]:
                            grid[r, wall_col] = SEMANTIC_PALETTE['FLOOR']
                            # Also punch the column to the right (left edge of east room)
                            if wall_col + 1 < grid.shape[1]:
                                grid[r, wall_col + 1] = SEMANTIC_PALETTE['FLOOR']
                    # Ensure path from door to room interior
                    for r in range(r_base + 6, r_base + 10):
                        for c in range(c_base + ROOM_WIDTH - 4, c_base + ROOM_WIDTH - 1):
                            if grid[r, c] == SEMANTIC_PALETTE['WALL']:
                                grid[r, c] = SEMANTIC_PALETTE['FLOOR']
        
        # Second pass: ensure each room has internal floor connectivity
        self._ensure_room_connectivity(grid, rooms)
    
    def _ensure_room_connectivity(self, grid: np.ndarray, rooms: Dict[Tuple[int, int], Room]):
        """Ensure each room has connected floor tiles from center to all doors."""
        for pos, room in rooms.items():
            row, col = pos
            r_base = row * ROOM_HEIGHT
            c_base = col * ROOM_WIDTH
            
            center_r = r_base + ROOM_HEIGHT // 2
            center_c = c_base + ROOM_WIDTH // 2
            
            # Ensure center is walkable
            if grid[center_r, center_c] == SEMANTIC_PALETTE['WALL']:
                grid[center_r, center_c] = SEMANTIC_PALETTE['FLOOR']
            
            # Create paths from center to each door
            if room.doors.get('N'):
                for r in range(r_base + 1, center_r + 1):
                    if grid[r, center_c] == SEMANTIC_PALETTE['WALL']:
                        grid[r, center_c] = SEMANTIC_PALETTE['FLOOR']
            
            if room.doors.get('S'):
                for r in range(center_r, r_base + ROOM_HEIGHT - 1):
                    if grid[r, center_c] == SEMANTIC_PALETTE['WALL']:
                        grid[r, center_c] = SEMANTIC_PALETTE['FLOOR']
            
            if room.doors.get('W'):
                for c in range(c_base + 1, center_c + 1):
                    if grid[center_r, c] == SEMANTIC_PALETTE['WALL']:
                        grid[center_r, c] = SEMANTIC_PALETTE['FLOOR']
            
            if room.doors.get('E'):
                for c in range(center_c, c_base + ROOM_WIDTH - 1):
                    if grid[center_r, c] == SEMANTIC_PALETTE['WALL']:
                        grid[center_r, c] = SEMANTIC_PALETTE['FLOOR']
    
    def _find_floor_near_door(self, grid: np.ndarray, 
                               r_off: int, c_off: int) -> Tuple[int, int]:
        """Find a walkable tile in the room for starting position."""
        # First try to find actual START tile in this room
        room_slice = grid[r_off:r_off+ROOM_HEIGHT, c_off:c_off+ROOM_WIDTH]
        start_positions = np.where(room_slice == SEMANTIC_PALETTE['START'])
        
        if len(start_positions[0]) > 0:
            # Return first START tile position
            return (r_off + start_positions[0][0], c_off + start_positions[1][0])
        
        # Otherwise check center first
        center_r = r_off + ROOM_HEIGHT // 2
        center_c = c_off + ROOM_WIDTH // 2
        
        if grid[center_r, center_c] == SEMANTIC_PALETTE['FLOOR']:
            return (center_r, center_c)
        
        # Search outward from center for any walkable tile
        WALKABLE = {
            SEMANTIC_PALETTE['FLOOR'],
            SEMANTIC_PALETTE['DOOR_OPEN'],
            SEMANTIC_PALETTE['STAIR']
        }
        
        for dr in range(-5, 6):
            for dc in range(-4, 5):
                r, c = center_r + dr, center_c + dc
                if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                    if grid[r, c] in WALKABLE:
                        return (r, c)
        
        # Fallback to center
        return (center_r, center_c)


# ==========================================
# MAIN ADAPTER CLASS
# ==========================================
class ZeldaDungeonAdapter:
    """
    Main adapter for processing Zelda dungeon data.
    
    Usage:
        adapter = ZeldaDungeonAdapter(data_root)
        dungeon = adapter.load_dungeon(dungeon_num)
        stitched = adapter.stitch_dungeon(dungeon)
    """
    
    def __init__(self, data_root: str):
        """
        Initialize adapter.
        
        Args:
            data_root: Path to "The Legend of Zelda" folder
        """
        self.data_root = Path(data_root)
        self.vglc_parser = VGLCParser()
        self.dot_parser = DOTParser()
        self.matcher = RoomGraphMatcher()
        self.stitcher = DungeonStitcher()
    
    def load_dungeon(self, dungeon_num: int, variant: int = 1) -> Dungeon:
        """
        Load a dungeon by number.
        
        Args:
            dungeon_num: Dungeon number (1-9)
            variant: Variant number (1 for Quest 1, 2 for Quest 2)
            
        Returns:
            Dungeon object
        """
        # Paths
        vglc_path = self.data_root / "Processed" / f"tloz{dungeon_num}_{variant}.txt"
        
        # Use correct DOT file for variant:
        # - Variant 1 (Quest 1): LoZ_{num}.dot
        # - Variant 2 (Quest 2): LoZ2_{num}.dot
        if variant == 2:
            dot_path = self.data_root / "Graph Processed" / f"LoZ2_{dungeon_num}.dot"
        else:
            dot_path = self.data_root / "Graph Processed" / f"LoZ_{dungeon_num}.dot"
        
        if not vglc_path.exists():
            raise FileNotFoundError(f"VGLC file not found: {vglc_path}")
        if not dot_path.exists():
            raise FileNotFoundError(f"DOT file not found: {dot_path}")
        
        # Parse VGLC rooms
        rooms = self.vglc_parser.parse(str(vglc_path))
        
        # Parse DOT graph
        graph = self.dot_parser.parse(str(dot_path))
        
        # Match rooms to graph
        dungeon = self.matcher.match(rooms, graph)
        dungeon.dungeon_id = f"D{dungeon_num}"
        
        return dungeon
    
    def stitch_dungeon(self, dungeon: Dungeon) -> StitchedDungeon:
        """
        Stitch dungeon into global grid.
        
        Args:
            dungeon: Dungeon to stitch
            
        Returns:
            StitchedDungeon
        """
        return self.stitcher.stitch(dungeon)


# ==========================================
# SOLVER / VALIDATOR
# ==========================================
class DungeonSolver:
    """
    Validate dungeon solvability using state-space pathfinding.
    
    Supports multiple validation modes:
    - STRICT: Only normal doors (what's visible in tiles)
    - REALISTIC: Normal + soft-locked + stairs (no items needed)
    - FULL: All edges with full inventory tracking (keys, bombs)
    """
    
    # Walkable tile IDs
    WALKABLE = {
        SEMANTIC_PALETTE['FLOOR'],
        SEMANTIC_PALETTE['DOOR_OPEN'],
        SEMANTIC_PALETTE['TRIFORCE'],
        SEMANTIC_PALETTE['STAIR'],
        SEMANTIC_PALETTE['KEY'],
        SEMANTIC_PALETTE['ITEM'],
        SEMANTIC_PALETTE['ELEMENT'],  # Can walk on elements
        SEMANTIC_PALETTE['START'],    # Can walk on START tile
    }
    
    def solve(self, stitched: StitchedDungeon, mode: str = ValidationMode.FULL) -> Dict:
        """
        Check if dungeon is solvable (START -> TRIFORCE path exists).
        Uses state-space graph search with inventory tracking.
        
        Args:
            stitched: Stitched dungeon to solve
            mode: ValidationMode (STRICT, REALISTIC, FULL)
        
        Returns:
            Dict with 'solvable', 'path_length', 'rooms_traversed', 
                  'edge_types', 'keys_available', 'keys_used'
        """
        if stitched.start_global is None:
            return {'solvable': False, 'reason': 'No START position'}
        
        if stitched.triforce_global is None:
            return {'solvable': False, 'reason': 'No TRIFORCE position'}
        
        # If we have graph and room mappings, use state-space solver
        if stitched.graph and stitched.room_to_node:
            return self._solve_with_state_space(stitched, mode)
        else:
            # Fallback to grid-based BFS
            return self._solve_with_grid(stitched)
    
    def _solve_with_state_space(self, stitched: StitchedDungeon, mode: str) -> Dict:
        """
        Check solvability using state-space search with inventory tracking.
        This properly handles keys, bombs, soft-locks, and stairs.
        """
        # Find which room contains start and triforce
        start_room = None
        triforce_room = None
        
        for room_pos, (r_off, c_off) in stitched.room_positions.items():
            r_end = r_off + ROOM_HEIGHT
            c_end = c_off + ROOM_WIDTH
            
            if stitched.start_global and r_off <= stitched.start_global[0] < r_end and c_off <= stitched.start_global[1] < c_end:
                start_room = room_pos
            
            if stitched.triforce_global and r_off <= stitched.triforce_global[0] < r_end and c_off <= stitched.triforce_global[1] < c_end:
                triforce_room = room_pos
        
        if not start_room or not triforce_room:
            return {'solvable': False, 'reason': 'Could not locate start/triforce rooms'}
        
        # Get graph node IDs
        start_node = stitched.room_to_node.get(start_room)
        triforce_node = stitched.room_to_node.get(triforce_room)
        
        if start_node is None:
            return {'solvable': False, 'reason': f'Start room {start_room} not mapped to graph node'}
        
        if triforce_node is None:
            return {'solvable': False, 'reason': f'Triforce room {triforce_room} not mapped to graph node'}
        
        # Use state-space solver
        solver = StateSpaceGraphSolver(stitched.graph, mode=mode)
        result = solver.solve(start_node, triforce_node)
        
        # Add room mapping info
        if result.get('solvable'):
            result['start_room'] = start_room
            result['triforce_room'] = triforce_room
            result['mode'] = mode
        
        return result
    
    def _solve_with_graph(self, stitched: StitchedDungeon) -> Dict:
        """Legacy: Check solvability using simple graph connectivity (ignores edge types)."""
        # Find which room contains start and triforce
        start_room = None
        triforce_room = None
        
        for room_pos, (r_off, c_off) in stitched.room_positions.items():
            r_end = r_off + ROOM_HEIGHT
            c_end = c_off + ROOM_WIDTH
            
            if stitched.start_global and r_off <= stitched.start_global[0] < r_end and c_off <= stitched.start_global[1] < c_end:
                start_room = room_pos
            
            if stitched.triforce_global and r_off <= stitched.triforce_global[0] < r_end and c_off <= stitched.triforce_global[1] < c_end:
                triforce_room = room_pos
        
        if not start_room or not triforce_room:
            return {'solvable': False, 'reason': 'Could not locate start/triforce rooms'}
        
        # Get graph node IDs
        start_node = stitched.room_to_node.get(start_room)
        triforce_node = stitched.room_to_node.get(triforce_room)
        
        if start_node is None:
            return {'solvable': False, 'reason': f'Start room {start_room} not mapped to graph node'}
        
        if triforce_node is None:
            return {'solvable': False, 'reason': f'Triforce room {triforce_room} not mapped to graph node'}
        
        # Check graph reachability
        try:
            path = nx.shortest_path(stitched.graph, start_node, triforce_node)
            return {
                'solvable': True,
                'path_length': len(path) - 1,
                'rooms_traversed': len(path)
            }
        except nx.NetworkXNoPath:
            return {
                'solvable': False,
                'reason': f'No graph path from node {start_node} to {triforce_node}'
            }
    
    def _solve_with_grid(self, stitched: StitchedDungeon) -> Dict:
        """Fallback: check solvability using grid BFS (no stairs)."""
        # BFS pathfinding
        from collections import deque
        
        grid = stitched.global_grid
        start = stitched.start_global
        goal = stitched.triforce_global
        
        visited = {start}
        queue = deque([(start, 0)])  # (position, distance)
        
        while queue:
            pos, dist = queue.popleft()
            
            if pos == goal:
                # Count rooms traversed
                rooms_hit = set()
                for room_pos, (r_off, c_off) in stitched.room_positions.items():
                    r_end = r_off + ROOM_HEIGHT
                    c_end = c_off + ROOM_WIDTH
                    if r_off <= pos[0] < r_end and c_off <= pos[1] < c_end:
                        rooms_hit.add(room_pos)
                
                return {
                    'solvable': True,
                    'path_length': dist,
                    'rooms_traversed': len(rooms_hit) if rooms_hit else 1
                }
            
            # Check 4 directions
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = pos[0] + dr, pos[1] + dc
                
                if (nr, nc) in visited:
                    continue
                
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    tile = grid[nr, nc]
                    if tile in self.WALKABLE or tile == SEMANTIC_PALETTE['TRIFORCE']:
                        visited.add((nr, nc))
                        queue.append(((nr, nc), dist + 1))
        
        return {
            'solvable': False,
            'reason': 'No path found',
            'reachable_tiles': len(visited)
        }


# ==========================================
# CONVENIENCE FUNCTION
# ==========================================
def test_all_dungeons(data_root: str, include_variants: bool = True) -> Dict[str, Dict]:
    """
    Test solvability of all dungeons.
    
    Args:
        data_root: Path to "The Legend of Zelda" folder
        include_variants: If True, test both variants (18 total). If False, only variant 1 (9 total)
        
    Returns:
        Dict mapping dungeon_id -> result
    """
    adapter = ZeldaDungeonAdapter(data_root)
    solver = DungeonSolver()
    results = {}
    
    variants = [1, 2] if include_variants else [1]
    
    for d in range(1, 10):
        for v in variants:
            dungeon_key = f"D{d}-{v}" if include_variants else f"D{d}"
            try:
                dungeon = adapter.load_dungeon(d, variant=v)
                stitched = adapter.stitch_dungeon(dungeon)
                result = solver.solve(stitched)
                results[dungeon_key] = result
                
                status = "✓ SOLVABLE" if result['solvable'] else "✗ NOT SOLVABLE"
                print(f"{dungeon_key}: {status}")
                if result['solvable']:
                    print(f"    Path: {result['path_length']} steps, {result['rooms_traversed']} rooms")
                else:
                    print(f"    Reason: {result.get('reason', 'Unknown')}")
            except Exception as e:
                results[dungeon_key] = {'solvable': False, 'error': str(e)}
                print(f"{dungeon_key}: ✗ ERROR - {e}")
    
    # Summary
    solvable_count = sum(1 for r in results.values() if r.get('solvable'))
    total = len(results)
    print(f"\nSUMMARY: {solvable_count}/{total} solvable ({100*solvable_count/total:.1f}%)")
    
    return results


# ==========================================
# UTILITY FUNCTIONS
# ==========================================
def visualize_semantic_grid(grid: np.ndarray, show_legend: bool = True) -> str:
    """
    Create ASCII visualization of semantic grid for debugging.
    
    Args:
        grid: Semantic ID grid
        show_legend: Whether to include legend in output
        
    Returns:
        ASCII string representation of the grid
    """
    symbol_map = {
        SEMANTIC_PALETTE['VOID']: ' ',
        SEMANTIC_PALETTE['FLOOR']: '.',
        SEMANTIC_PALETTE['WALL']: '#',
        SEMANTIC_PALETTE['BLOCK']: 'B',
        SEMANTIC_PALETTE['DOOR_OPEN']: 'O',
        SEMANTIC_PALETTE['DOOR_LOCKED']: 'L',
        SEMANTIC_PALETTE['DOOR_BOMB']: 'X',
        SEMANTIC_PALETTE['ENEMY']: 'E',
        SEMANTIC_PALETTE['START']: 'S',
        SEMANTIC_PALETTE['TRIFORCE']: 'T',
        SEMANTIC_PALETTE['KEY']: 'k',
        SEMANTIC_PALETTE['ITEM']: 'i',
        SEMANTIC_PALETTE['ELEMENT']: '~',
        SEMANTIC_PALETTE['STAIR']: '^',
        SEMANTIC_PALETTE['BOSS']: 'B',
    }
    
    lines = []
    for row in grid:
        line = ''.join(symbol_map.get(int(cell), '?') for cell in row)
        lines.append(line)
    
    result = '\n'.join(lines)
    
    if show_legend:
        result += '\n\nLegend: . floor, # wall, O open door, L locked door, X bomb wall'
        result += '\n        E enemy, S start, T triforce, k key, ~ hazard, ^ stair'
    
    return result


def convert_room_to_roomdata(room: Room) -> RoomData:
    """Convert Room dataclass to RoomData for adapter.py compatibility."""
    doors_dict = {}
    direction_map = {'N': 'north', 'S': 'south', 'E': 'east', 'W': 'west'}
    for d, has_door in room.doors.items():
        if has_door:
            doors_dict[direction_map.get(d, d)] = {'type': 'open'}
    
    contents = []
    if room.is_start:
        contents.append('start')
    if room.has_triforce:
        contents.append('triforce')
    if room.has_boss:
        contents.append('boss')
    
    return RoomData(
        room_id=str(room.graph_node_id) if room.graph_node_id else f"{room.position[0]}_{room.position[1]}",
        grid=room.semantic_grid,
        contents=contents,
        doors=doors_dict,
        position=room.position
    )


def convert_dungeon_to_dungeondata(dungeon: Dungeon) -> DungeonData:
    """Convert Dungeon to DungeonData for adapter.py compatibility."""
    rooms_dict = {}
    for pos, room in dungeon.rooms.items():
        room_data = convert_room_to_roomdata(room)
        rooms_dict[room_data.room_id] = room_data
    
    # Compute ML features
    ml_extractor = MLFeatureExtractor()
    tpe_vectors, node_order = ml_extractor.compute_laplacian_pe(dungeon.graph)
    node_features = ml_extractor.extract_node_features(dungeon.graph, node_order)
    p_matrix = ml_extractor.build_p_matrix(dungeon.graph, node_order)
    
    # Build layout matrix
    positions = [pos for pos in dungeon.rooms.keys()]
    if positions:
        max_r = max(p[0] for p in positions) + 1
        max_c = max(p[1] for p in positions) + 1
        layout = np.full((max_r, max_c), -1, dtype=int)
        for pos, room in dungeon.rooms.items():
            if room.graph_node_id is not None:
                layout[pos[0], pos[1]] = room.graph_node_id
    else:
        layout = np.zeros((0, 0), dtype=int)
    
    return DungeonData(
        dungeon_id=dungeon.dungeon_id,
        rooms=rooms_dict,
        graph=dungeon.graph,
        layout=layout,
        tpe_vectors=tpe_vectors,
        p_matrix=p_matrix,
        node_features=node_features
    )


if __name__ == "__main__":
    # Test with default path - all 18 variants
    DATA_ROOT = r"C:\Users\MPhuc\Desktop\KLTN\Data\The Legend of Zelda"
    print("Testing ALL 18 dungeon variants (9 dungeons × 2 variants)...")
    print("="*60)
    test_all_dungeons(DATA_ROOT, include_variants=True)
