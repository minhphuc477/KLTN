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

"""

import os
import re
import logging
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__) 

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
    'DOOR_PUZZLE': 13,
    'DOOR_BOSS': 14,
    'DOOR_SOFT': 15,
    'ENEMY': 20,
    'START': 21,
    'TRIFORCE': 22,
    'BOSS': 23,
    'KEY_SMALL': 30,
    'KEY_BOSS': 31,
    'KEY_ITEM': 32,
    'ITEM_MINOR': 33,
    'KEY': 30,  # Alias for KEY_SMALL
    'ITEM': 33,  # Alias for ITEM_MINOR
    'ELEMENT': 40,
    'ELEMENT_FLOOR': 41,
    'STAIR': 42,
    'PUZZLE': 43,
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
    'k': 'key_locked',      # Small key required (use canonical 'key_locked')
    'K': 'boss_locked',     # Boss key required
    'b': 'bombable',        # Bomb required
    'l': 'soft_locked',     # One-way (can't return)
    'S': 'switch_locked',   # Switch/puzzle required (canonical 'switch_locked' to match traversal semantics)
    'I': 'item_locked',     # Key item required
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
        # Accept both 'F' and '.' as floor markers in VGLC
        floor_count = np.sum((slot_grid == 'F') | (slot_grid == '.'))

        return bool(wall_count >= 20 and floor_count >= 5)

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
        Uses canonicalized edge_type when available and falls back to edge label.

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
        edge_type = edge_data.get('edge_type') if edge_data.get('edge_type') else EDGE_TYPE_MAP.get(edge_label, '')
        edge_id = (from_node, to_node)

        new_state = state.copy()

        # STRICT mode: only normal doors
        if self.mode == ValidationMode.STRICT:
            if edge_type and edge_type != 'open':
                return False, state, edge_type
            if edge_label != '':
                return False, state, edge_label
            return True, new_state, 'open'

        # REALISTIC mode: normal + soft-locked + stairs
        if self.mode == ValidationMode.REALISTIC:
            if edge_type in ('open', 'soft_locked', 'stair') or edge_label in ('', 'l', 's'):
                return True, new_state, edge_type or edge_label
            return False, state, edge_type or edge_label

        # FULL mode: prefer edge_type canonical checks (supports backward compatibility via label fallback)
        et = edge_type or edge_label

        if et in ('', 'open'):
            # Normal door - always passable
            return True, new_state, 'open'

        if et in ('key_locked', 'k'):
            # Key-locked door
            if edge_id in state.doors_opened:
                return True, new_state, 'key_locked'  # Already opened

            if state.keys_held > 0:
                new_state.keys_held -= 1
                new_state.doors_opened.add(edge_id)
                new_state.doors_opened.add((to_node, from_node))
                return True, new_state, 'key_locked'

            return False, state, 'key_locked'

        if et in ('bombable', 'b'):
            # Bombable wall - assume infinite bombs for now
            if edge_id in state.doors_opened:
                return True, new_state, 'bombable'

            new_state.doors_opened.add(edge_id)
            new_state.doors_opened.add((to_node, from_node))
            return True, new_state, 'bombable'

        if et in ('soft_locked', 'l'):
            # Soft-locked (one-way) - always passable forward
            return True, new_state, 'soft_locked'

        if et in ('stair', 's'):
            # Stair/warp - bidirectional teleport
            return True, new_state, 'stair'

        if et in ('item_locked', 'I'):
            # Item-locked - check if we have the required item
            if 'key_item' in state.items_collected:
                return True, new_state, 'item_locked'
            return False, state, 'item_locked'

        if et in ('switch_locked', 'S', 'S1'):
            # Switch-locked - assume puzzle solved
            return True, new_state, 'switch_locked'

        # Unknown edge type - allow traversal but return canonical label if available
        return True, new_state, edge_type or edge_label
    
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
        
        # Collect item if room has one and it's not already collected
        if node in self.item_rooms:
            item_type = self.item_rooms[node]
            if item_type not in state.items_collected:
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
                    'keys_available': current_state.keys_held,
                    'keys_collected': len(current_state.keys_collected),
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
        
        # Parse nodes: handle arbitrary attributes and whitespace; robustly extract label if present
        node_pattern = r'^\s*(\d+)\s*\[([^\]]*)\]'
        for match in re.finditer(node_pattern, content, re.MULTILINE):
            node_id = int(match.group(1))
            attrs = match.group(2)
            # Robust label extraction: accept quoted or unquoted labels, allow commas inside unquoted label
            label = ''
            q = re.search(r'label\s*=\s*"', attrs)
            if q:
                start = q.end()
                end = attrs.find('"', start)
                label = attrs[start:end] if end != -1 else attrs[start:]
            else:
                q2 = re.search(r'label\s*=\s*', attrs)
                if q2:
                    start = q2.end()
                    m2 = re.search(r',\s*\w+\s*=', attrs[start:])
                    if m2:
                        end = start + m2.start()
                    else:
                        end = len(attrs)
                    label = attrs[start:end].strip()
            parts = [p.strip() for p in label.split(',')] if label else []
            
            graph.add_node(node_id, 
                          label=label,
                          is_start='s' in parts,
                          is_triforce='t' in parts,
                          is_boss='b' in parts,
                          has_key='k' in parts,
                          has_item='I' in parts or 'i' in parts,
                          has_enemy='e' in parts,
                          has_puzzle='p' in parts) 
        
        # Parse edges: 7 -> 8 [label="k"] or simple edges like '1 -> 2;'
        # Support labeled and unlabeled edges (unlabeled -> 'open')
        edge_pattern = r'(\d+)\s*->\s*(\d+)(?:\s*\[([^\]]*)\])?'
        for match in re.finditer(edge_pattern, content, re.MULTILINE):
            src = int(match.group(1))
            dst = int(match.group(2))
            attrs = match.group(3) or ''
            lab_m = None  # replaced by robust extraction below
            # Robust label extraction: accept quoted or unquoted labels, allow commas inside unquoted label
            label = ''
            q = re.search(r'label\s*=\s*"', attrs)
            if q:
                start = q.end()
                end = attrs.find('"', start)
                label = attrs[start:end] if end != -1 else attrs[start:]
            else:
                q2 = re.search(r'label\s*=\s*', attrs)
                if q2:
                    start = q2.end()
                    m2 = re.search(r',\s*\w+\s*=', attrs[start:])
                    if m2:
                        end = start + m2.start()
                    else:
                        end = len(attrs)
                    label = attrs[start:end].strip()

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
            if getattr(room, 'has_stair', False):
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
        
        # Normalize graph to canonical labels & types
        self._normalize_graph(graph)

        # Build room adjacency and match rooms to graph nodes using BFS
        room_adjacency = self._build_room_adjacency(rooms)
        
        # Match rooms to graph nodes using parallel BFS (deterministic assignment)
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

        This implementation is deterministic and attempts a local optimal assignment
        at each BFS wave. When available, it uses the Hungarian algorithm (scipy)
        for small bipartite assignments; otherwise it falls back to a stable
        greedy assignment with deterministic tie-breaking.
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

            # Get neighbors in both spaces (deterministic order)
            room_neighbors = [r for r in room_adjacency.get(current_room, []) if r not in visited_rooms]
            graph_neighbors = [n for n in list(graph.successors(current_node)) + list(graph.predecessors(current_node))
                               if n not in visited_nodes]

            if not room_neighbors or not graph_neighbors:
                continue

            # Deterministic ordering: sort rooms by position, nodes by structural signature
            R = sorted(room_neighbors)
            N = sorted(graph_neighbors, key=lambda x: self._node_signature(graph, x))

            # Build cost matrix: lower cost = better match
            cost_matrix = []
            for r in R:
                row = []
                r_deg = sum(rooms[r].doors.values())
                r_trif = getattr(rooms[r], 'has_triforce', False)
                r_boss = getattr(rooms[r], 'has_boss', False)
                for n in N:
                    node_data = graph.nodes[n]
                    n_deg = graph.in_degree(n) + graph.out_degree(n)
                    base = abs((r_deg * 2) - n_deg)
                    # Bonus for matching special nodes
                    if node_data.get('is_triforce') and r_trif:
                        base -= 100
                    if node_data.get('is_boss') and r_boss:
                        base -= 100
                    # adjacency overlap: prefer assignments where already-mapped neighbor rooms map to neighbors of n
                    overlap = 0
                    for rn in room_adjacency.get(r, []):
                        mapped = room_to_node.get(rn)
                        if mapped is not None and (graph.has_edge(mapped, n) or graph.has_edge(n, mapped)):
                            overlap += 1
                    base -= overlap * 1.0
                    row.append(float(base))
                cost_matrix.append(row)

            assigned_pairs: List[Tuple[Tuple[int, int], int]] = []
            try:
                # Use Hungarian when sizes are small to get a global optimum for the local wave
                if len(R) <= 10 and len(N) <= 10:
                    from scipy.optimize import linear_sum_assignment
                    import numpy as _np
                    cm = _np.array(cost_matrix, dtype=float)
                    row_ind, col_ind = linear_sum_assignment(cm)
                    for i, j in zip(row_ind, col_ind):
                        if i < len(R) and j < len(N):
                            assigned_pairs.append((R[i], N[j]))
                else:
                    raise RuntimeError('skip hungarian')
            except Exception as e:
                # Deterministic greedy assignment: sort all pairs by (cost, node id) and pick non-conflicting pairs
                pairs = []
                for i, r in enumerate(R):
                    for j, n in enumerate(N):
                        pairs.append((cost_matrix[i][j], i, n))
                pairs.sort(key=lambda x: (x[0], self._node_signature(graph, x[2])))  # deterministic tie-break using node signature
                used_r = set()
                used_n = set()
                for cost, i, n in pairs:
                    if i in used_r or n in used_n:
                        continue
                    used_r.add(i)
                    used_n.add(n)
                    assigned_pairs.append((R[i], n))

            # Apply assignments
            for r, n in assigned_pairs:
                if r in visited_rooms or n in visited_nodes:
                    continue
                room_to_node[r] = n
                node_to_room[n] = r
                visited_rooms.add(r)
                visited_nodes.add(n)
                try:
                    graph_neighbors.remove(n)
                except ValueError:
                    pass
                room_queue.append(r)

        # FALLBACK: Global assignment for remaining unmapped rooms/nodes (deterministic)
        unmapped_rooms = sorted([pos for pos in rooms.keys() if pos not in room_to_node])
        unmapped_nodes = sorted([n for n in graph.nodes() if n not in node_to_room], key=lambda x: self._node_signature(graph, x))

        if unmapped_rooms and unmapped_nodes:
            R = unmapped_rooms
            N = unmapped_nodes
            cm = []
            for r in R:
                row = []
                r_deg = sum(rooms[r].doors.values())
                r_trif = getattr(rooms[r], 'has_triforce', False)
                r_boss = getattr(rooms[r], 'has_boss', False)
                for n in N:
                    node_data = graph.nodes[n]
                    n_deg = graph.in_degree(n) + graph.out_degree(n)
                    score = abs((r_deg * 2) - n_deg)
                    if node_data.get('is_triforce') and r_trif:
                        score -= 100
                    if node_data.get('is_boss') and r_boss:
                        score -= 100
                    row.append(float(score))
                cm.append(row)

            try:
                from scipy.optimize import linear_sum_assignment
                import numpy as _np
                matrix = _np.array(cm, dtype=float)
                row_ind, col_ind = linear_sum_assignment(matrix)
                for i, j in zip(row_ind, col_ind):
                    if i < len(R) and j < len(N):
                        room_to_node[R[i]] = N[j]
                        node_to_room[N[j]] = R[i]
            except Exception:
                pairs = []
                for i, r in enumerate(R):
                    for j, n in enumerate(N):
                        pairs.append((cm[i][j], r, n))
                pairs.sort(key=lambda x: (x[0], self._node_signature(graph, x[2])))
                used_n = set()
                for cost, r, n in pairs:
                    if n in used_n:
                        continue
                    room_to_node[r] = n
                    node_to_room[n] = r
                    used_n.add(n)

        # Local refinement: try pairwise swaps to improve adjacency consistency
        # This helps fix small symmetric mismatches that greedy assignment may create
        def _try_improve_swaps(max_iters: int = 100):
            cur_cons = self._validate_mapping(rooms, room_adjacency, graph, room_to_node)
            rooms_list = sorted(list(room_to_node.keys()))
            it = 0
            improved = True
            while improved and it < max_iters:
                improved = False
                it += 1
                for i in range(len(rooms_list)):
                    for j in range(i + 1, len(rooms_list)):
                        r1 = rooms_list[i]; r2 = rooms_list[j]
                        n1 = room_to_node[r1]; n2 = room_to_node[r2]
                        # swap
                        room_to_node[r1], room_to_node[r2] = n2, n1
                        node_to_room[n1], node_to_room[n2] = r2, r1
                        new_cons = self._validate_mapping(rooms, room_adjacency, graph, room_to_node)
                        if new_cons > cur_cons + 1e-9:
                            cur_cons = new_cons
                            improved = True
                            break
                        # revert
                        room_to_node[r1], room_to_node[r2] = n1, n2
                        node_to_room[n1], node_to_room[n2] = r1, r2
                    if improved:
                        break
            return cur_cons

        try:
            # attempt local swaps before final validation
            try:
                consistency_after = _try_improve_swaps()
            except Exception:
                consistency_after = None
            # Validate mapping quality and log a warning if low consistency
            consistency = self._validate_mapping(rooms, room_adjacency, graph, room_to_node)
            if consistency < 0.2:
                logger.warning('Low room-node mapping consistency: %.2f', consistency)
        except Exception:
            logger.debug('Mapping validation failed to run', exc_info=True)

        return room_to_node, node_to_room

    def _normalize_graph(self, graph: nx.DiGraph) -> None:
        """Normalize graph labels and edge types so downstream logic can be deterministic."""
        def _canonical_edge_type(val: str):
            if not val:
                return None
            v = str(val).strip()
            # Check exact mapping
            if v in EDGE_TYPE_MAP:
                return EDGE_TYPE_MAP[v]
            # Case-insensitive fallback
            vl = v.lower()
            if vl in EDGE_TYPE_MAP:
                return EDGE_TYPE_MAP[vl]
            return v

        for u, v, data in graph.edges(data=True):
            label_raw = data.get('label', '')
            label = '' if label_raw is None else str(label_raw).strip()
            data['label'] = label

            # Normalize any existing edge_type or derive from label; prefer explicit edge_type
            edge_type_raw = data.get('edge_type') or ''
            edge_type_can = _canonical_edge_type(edge_type_raw) if edge_type_raw else None
            if not edge_type_can:
                edge_type_can = _canonical_edge_type(label) or 'open'
            data['edge_type'] = edge_type_can

        for n, data in graph.nodes(data=True):
            label = (data.get('label') or data.get('name') or '')
            s = str(label).strip().lower()
            # Canonical flags
            if s == 's' or 'start' in s or data.get('is_start'):
                data['is_start'] = True
            if s == 't' or 'triforce' in s or data.get('is_triforce'):
                data['is_triforce'] = True
            if s == 'b' or 'boss' in s or data.get('is_boss'):
                data['is_boss'] = True

    def _validate_mapping(self, rooms: Dict[Tuple[int, int], Room],
                          room_adjacency: Dict[Tuple[int, int], List[Tuple[int, int]]],
                          graph: nx.DiGraph,
                          room_to_node: Dict[Tuple[int, int], int]) -> float:
        """Return fraction of room adjacencies that are consistent with graph edges."""
        consistent = 0
        total = 0
        for r, n in room_to_node.items():
            for rn in room_adjacency.get(r, []):
                total += 1
                nn = room_to_node.get(rn)
                if nn is not None and (graph.has_edge(n, nn) or graph.has_edge(nn, n)):
                    consistent += 1
        return 1.0 if total == 0 else (consistent / total)

    def _node_signature(self, graph: nx.DiGraph, n: int):
        """Return a deterministic, relabel-invariant signature for a node."""
        data = graph.nodes[n]
        deg = graph.in_degree(n) + graph.out_degree(n)
        trif = bool(data.get('is_triforce'))
        boss = bool(data.get('is_boss'))
        # Use sorted neighbor degrees as a compact structural signature
        neigh_degs = sorted([graph.in_degree(nb) + graph.out_degree(nb) for nb in graph.neighbors(n)])
        return (deg, trif, boss, tuple(neigh_degs))

    def infer_missing_mappings(self, rooms: Dict[Tuple[int, int], Room],
                               graph: nx.DiGraph,
                               room_positions: Optional[Dict[Tuple[int, int], Tuple[int, int]]] = None,
                               room_to_node: Optional[Dict[Tuple[int, int], int]] = None,
                               confidence_threshold: float = 0.0
                               ) -> Tuple[Dict[Tuple[int, int], int], Dict[int, Tuple[int, int]], Dict[int, float]]:
        """Infer candidate mappings for unmatched graph nodes.

        Returns (proposed_room_to_node, proposed_node_to_room, confidences)
        - proposed maps only include nodes/rooms that were previously unmapped
        - confidences: node_id -> confidence (0.0-1.0)

        Improvements implemented:
        - More robust label parsing (many formats)
        - Spatial distance based scoring when `room_positions` available
        - Component-aware scoring using seeded anchors
        - Global assignment using Hungarian algorithm (scipy) when available; falls back to greedy
        """
        import re
        from math import hypot

        # Prepare mappings
        existing_room_to_node = dict(room_to_node or {})
        existing_node_to_room = {v: k for k, v in existing_room_to_node.items()}

        all_nodes = list(graph.nodes())
        unmatched_nodes = [n for n in all_nodes if n not in existing_node_to_room]
        unmatched_rooms = [r for r in rooms.keys() if r not in existing_room_to_node]

        proposed_room_to_node: Dict[Tuple[int, int], int] = {}
        proposed_node_to_room: Dict[int, Tuple[int, int]] = {}
        confidences: Dict[int, float] = {}

        # Quick exit
        if not unmatched_nodes or not unmatched_rooms:
            return proposed_room_to_node, proposed_node_to_room, confidences

        # 1) Try to use strong anchors (start/triforce/boss) even if room_to_node is empty
        if not existing_room_to_node:
            # Search graph special nodes
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

            # Map special nodes to rooms by heuristics (stairs/triforce/boss detection)
            # Find candidate start room
            start_room_pos = None
            stair_rooms_with_doors = []
            for pos, room in rooms.items():
                if getattr(room, 'has_stair', False):
                    door_count = sum(room.doors.values())
                    if door_count > 0:
                        stair_rooms_with_doors.append((pos, door_count))
            if stair_rooms_with_doors:
                stair_rooms_with_doors.sort(key=lambda x: x[1], reverse=True)
                start_room_pos = stair_rooms_with_doors[0][0]

            if start_node is not None and start_room_pos is not None:
                existing_room_to_node[start_room_pos] = start_node
                existing_node_to_room[start_node] = start_room_pos
                proposed_room_to_node[start_room_pos] = start_node
                proposed_node_to_room[start_node] = start_room_pos
                confidences[start_node] = 0.98

        # 2) BFS propagation from existing anchors using existing _match_rooms_to_nodes_bfs
        anchors = list(existing_room_to_node.items())
        used_nodes = set(existing_node_to_room.keys())
        used_rooms = set(existing_room_to_node.keys())
        for room_anchor, node_anchor in anchors:
            r2n, n2r = self._match_rooms_to_nodes_bfs(rooms, self._build_room_adjacency(rooms), graph, room_anchor, node_anchor)
            for rpos, nid in r2n.items():
                if rpos in existing_room_to_node or rpos in proposed_room_to_node:
                    continue
                if nid in used_nodes or nid in proposed_node_to_room:
                    continue
                proposed_room_to_node[rpos] = nid
                proposed_node_to_room[nid] = rpos
                confidences[nid] = 0.9
                used_nodes.add(nid)
                used_rooms.add(rpos)

        # Refresh unmatched lists after BFS pass
        unmatched_nodes = [n for n in unmatched_nodes if n not in proposed_node_to_room]
        unmatched_rooms = [r for r in unmatched_rooms if r not in proposed_room_to_node]

        # 3) Label-based hints (robust regex)
        # Accept formats: '3,4', '(3,4)', '3_4', 'r3c4', 'r:3,c:4', '3 4'
        coord_re = re.compile(r"\(?\s*(\d+)\s*[,_x\\/\s:-]\s*(\d+)\s*\)?")
        for node in list(unmatched_nodes):
            attrs = graph.nodes[node]
            label = attrs.get('label') or attrs.get('name') or ''
            m = coord_re.search(str(label))
            if m:
                r = int(m.group(1)); c = int(m.group(2))
                if (r, c) in unmatched_rooms and node not in proposed_node_to_room:
                    proposed_node_to_room[node] = (r, c)
                    proposed_room_to_node[(r, c)] = node
                    confidences[node] = 0.98
                    unmatched_nodes.remove(node)
                    unmatched_rooms.remove((r, c))

        # 4) Component-aware building (map graph comps -> room comps via known anchors)
        graph_comp_of = {}
        for comp in nx.weakly_connected_components(graph):
            for n in comp:
                graph_comp_of[n] = id(comp)

        room_adj = self._build_room_adjacency(rooms)
        room_graph = nx.Graph()
        room_graph.add_nodes_from(rooms.keys())
        for k, vs in room_adj.items():
            for v in vs:
                room_graph.add_edge(k, v)
        room_comp_of = {}
        for comp in nx.connected_components(room_graph):
            for r in comp:
                room_comp_of[r] = id(comp)

        # Build a mapping from graph component -> candidate room components based on existing anchors
        comp_room_candidates: Dict[int, Set[int]] = {}
        for rpos, nid in dict(existing_room_to_node).items():
            gc = graph_comp_of.get(nid)
            rc = room_comp_of.get(rpos)
            if gc is not None and rc is not None:
                comp_room_candidates.setdefault(gc, set()).add(rc)

        # Spatial centers (normalized) for rooms using room_positions if available
        centers = {}
        if room_positions:
            xs = [off[1] for off in room_positions.values() if off]
            ys = [off[0] for off in room_positions.values() if off]
            if xs and ys:
                minx, maxx = min(xs), max(xs)
                miny, maxy = min(ys), max(ys)
                span = max(maxx - minx, maxy - miny, 1)
                for r in unmatched_rooms:
                    off = room_positions.get(r)
                    if off:
                        centers[r] = ((off[1] - minx) / span, (off[0] - miny) / span)

        # Optional: use seeded spectral match to propose additional mappings before building score matrix
        if existing_room_to_node:
            try:
                spectral_props, spectral_confs = self.seeded_spectral_match(rooms, graph, room_positions, seeds=existing_room_to_node, k_dim=8)
                for nid, rpos in spectral_props.items():
                    if nid in unmatched_nodes and rpos in unmatched_rooms:
                        proposed_node_to_room[nid] = rpos
                        proposed_room_to_node[rpos] = nid
                        confidences[nid] = max(confidences.get(nid, 0.0), spectral_confs.get(nid, 0.1))
                unmatched_nodes = [n for n in unmatched_nodes if n not in proposed_node_to_room]
                unmatched_rooms = [r for r in unmatched_rooms if r not in proposed_room_to_node]
            except Exception as e:
                logger.exception("seeded_spectral_match failed during infer_missing_mappings: %s", e) 

        # 5) Build score matrix for remaining unmatched nodes/rooms
        deg = {n: (graph.in_degree(n) + graph.out_degree(n)) for n in unmatched_nodes}
        room_degs = {r: sum(rooms[r].doors.values()) for r in unmatched_rooms}
        score_matrix: Dict[Tuple[int, Tuple[int, int]], float] = {}
        max_deg = max(list(deg.values()) + [1])
        for n in unmatched_nodes:
            for r in unmatched_rooms:
                node_deg = deg.get(n, 0)
                room_deg = room_degs.get(r, 0)
                # degree similarity (normalized)
                deg_score = 1.0 - (abs(node_deg - (room_deg * 2)) / float(max_deg + room_deg + 1.0))
                # spatial score based on normalized distance
                spat_score = 0.0
                if r in centers:
                    # use hypothetical node position inference (no node positions typically)
                    spat_score = 0.5
                # component bonus: if node's graph component has known room components prefer those rooms
                comp_bonus = 0.0
                gc = graph_comp_of.get(n)
                rc = room_comp_of.get(r)
                if gc is not None and gc in comp_room_candidates and rc in comp_room_candidates[gc]:
                    comp_bonus = 0.2
                score = 0.7 * deg_score + 0.25 * spat_score + comp_bonus
                score_matrix[(n, r)] = max(0.0, min(1.0, score))

        # Try to use global assignment (Hungarian) for best overall matching
        assigned_pairs: List[Tuple[int, Tuple[int, int]]] = []
        try:
            from scipy.optimize import linear_sum_assignment
            # Build matrix (rows=nodes, cols=rooms)
            nodes_idx = {n: i for i, n in enumerate(unmatched_nodes)}
            rooms_idx = {r: j for j, r in enumerate(unmatched_rooms)}
            import numpy as np
            cost = np.zeros((len(unmatched_nodes), len(unmatched_rooms)), dtype=np.float32)
            for (n, r), s in score_matrix.items():
                cost[nodes_idx[n], rooms_idx[r]] = -float(s)  # negative because linear_sum_assignment minimizes
            row_ind, col_ind = linear_sum_assignment(cost)
            for i, j in zip(row_ind, col_ind):
                n = unmatched_nodes[i]
                r = unmatched_rooms[j]
                s = score_matrix.get((n, r), 0.0)
                if s > 0:
                    assigned_pairs.append((n, r))
        except Exception as e:
            logger.exception("Hungarian assignment failed; falling back to greedy assignment: %s", e)
            # fallback greedy matching
            local_scores = dict(score_matrix)
            remaining_nodes = list(unmatched_nodes)
            remaining_rooms = list(unmatched_rooms)
            while local_scores and remaining_nodes and remaining_rooms:
                best = max(local_scores.items(), key=lambda kv: kv[1])[0]
                best_score = local_scores[best]
                n, r = best
                if best_score <= 0:
                    break
                assigned_pairs.append((n, r))
                remaining_nodes = [x for x in remaining_nodes if x != n]
                remaining_rooms = [x for x in remaining_rooms if x != r]
                for key in list(local_scores.keys()):
                    if key[0] == n or key[1] == r:
                        del local_scores[key]

        # Apply assigned pairs and set confidences normalized
        for n, r in assigned_pairs:
            proposed_node_to_room[n] = r
            proposed_room_to_node[r] = n
            confidences[n] = float(max(0.1, score_matrix.get((n, r), 0.1)))

        # Local refinement: try swap moves to improve edge-consistency
        try:
            room_adj = self._build_room_adjacency(rooms)
            refined = self._local_refine_assignments(proposed_node_to_room, graph, room_adj, score_matrix, iterations=200)
            if refined:
                # reassign
                proposed_node_to_room = refined
                proposed_room_to_node = {r: n for n, r in refined.items()}
                # update confidences
                for n, r in refined.items():
                    confidences[n] = max(confidences.get(n, 0.1), float(score_matrix.get((n, r), 0.1)))
        except Exception as e:
            logger.exception("Local refinement of assignments failed: %s", e) 

        # Final safety: filter by confidence_threshold if requested
        if confidence_threshold > 0:
            low_conf = [node for node, conf in list(confidences.items()) if conf < confidence_threshold]
            for node in low_conf:
                room = proposed_node_to_room.pop(node, None)
                if room:
                    proposed_room_to_node.pop(room, None)
                confidences.pop(node, None)

        # Logging (helpful for diagnostics)
        if proposed_node_to_room:
            logger.info('infer_missing_mappings: proposed %d matches; sample confidences: %s', len(proposed_node_to_room), {n: round(c, 2) for n, c in list(confidences.items())[:5]})

        return proposed_room_to_node, proposed_node_to_room, confidences

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

    def _room_signature(self, room: Room) -> Tuple[int, int, int, int]:
        """Return simple door signature (N,S,E,W) as ints and door count.

        Useful for comparing structural compatibility of rooms.
        """
        return (1 if room.doors.get('N') else 0,
                1 if room.doors.get('S') else 0,
                1 if room.doors.get('W') else 0,
                1 if room.doors.get('E') else 0)

    def _edge_consistency_score(self, n2r: Dict[int, Tuple[int, int]], graph: nx.DiGraph, room_adj: Dict[Tuple[int, int], List[Tuple[int, int]]]) -> int:
        """Compute how many graph edges are consistent with adjacent room pairs.

        Score is number of directed edges (u->v) where assigned rooms r_u and r_v are adjacent in room_adj.
        """
        score = 0
        for u, v in graph.edges():
            ru = n2r.get(u)
            rv = n2r.get(v)
            if ru is None or rv is None:
                continue
            if rv in room_adj.get(ru, []):
                score += 1
        return score

    def _local_refine_assignments(self, n2r: Dict[int, Tuple[int, int]], graph: nx.DiGraph, room_adj: Dict[Tuple[int, int], List[Tuple[int, int]]], score_matrix: Dict[Tuple[int, Tuple[int, int]], float], iterations: int = 100) -> Optional[Dict[int, Tuple[int, int]]]:
        """Try local pairwise swaps to increase combined score (assignment score + edge consistency).

        Deterministic improvement pass: consider all pairs and perform swap if it improves total objective. Repeat until no improvement or max iterations.
        """
        # Make mutable copy
        n2r = dict(n2r)

        def objective(mapping: Dict[int, Tuple[int, int]]) -> float:
            assign_score = sum(float(score_matrix.get((n, r), 0.0)) for n, r in mapping.items())
            edge_score = self._edge_consistency_score(mapping, graph, room_adj)
            return assign_score + 0.5 * edge_score  # weight edge consistency

        best_obj = objective(n2r)
        improved = True
        it = 0
        nodes = list(n2r.keys())
        while improved and it < iterations:
            improved = False
            it += 1
            # Examine all pairs
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    a = nodes[i]
                    b = nodes[j]
                    ra = n2r[a]
                    rb = n2r[b]
                    # Skip identical
                    if ra == rb:
                        continue
                    # Propose swap
                    n2r[a], n2r[b] = rb, ra
                    new_obj = objective(n2r)
                    if new_obj > best_obj + 1e-6:
                        best_obj = new_obj
                        improved = True
                        # keep swap
                    else:
                        # revert
                        n2r[a], n2r[b] = ra, rb
            if not improved:
                break
        return n2r if best_obj > 0 else n2r

    def seeded_spectral_match(self, rooms: Dict[Tuple[int, int], Room], graph: nx.DiGraph, room_positions: Optional[Dict[Tuple[int, int], Tuple[int, int]]] = None, seeds: Optional[Dict[Tuple[int, int], int]] = None, k_dim: int = 8) -> Tuple[Dict[int, Tuple[int, int]], Dict[int, float]]:
        """Perform seeded spectral matching between graph nodes and rooms.

        Steps:
        - Compute spectral embeddings for graph and room adjacency
        - Use seeded correspondences to compute an orthogonal Procrustes alignment
        - Match remaining nodes by nearest neighbor in embedding space and refine with Hungarian

        Returns: (node_to_room_proposal, confidences)
        """
        import numpy as np
        try:
            from scipy.linalg import orthogonal_procrustes
        except Exception as e:
            logger.debug("scipy.linalg.orthogonal_procrustes not available: %s", e)
            orthogonal_procrustes = None

        # Build room adjacency and small undirected graphs
        room_adj = self._build_room_adjacency(rooms)
        RG = nx.Graph()
        RG.add_nodes_from(rooms.keys())
        for r, nbrs in room_adj.items():
            for nb in nbrs:
                RG.add_edge(r, nb)

        # If graphs are empty, return nothing
        if len(graph) == 0 or len(RG) == 0:
            return {}, {}

        # Spectral embedding via Laplacian eigenvectors
        def laplacian_embedding(G, dim):
            import numpy as np
            G_u = G.to_undirected() if G.is_directed() else G
            nodes = sorted(G_u.nodes())
            n = len(nodes)
            node_to_idx = {node: i for i, node in enumerate(nodes)}
            adj = np.zeros((n, n), dtype=float)
            for u, v in G_u.edges():
                i, j = node_to_idx[u], node_to_idx[v]
                adj[i, j] = 1.0
                adj[j, i] = 1.0
            deg = np.sum(adj, axis=1)
            D = np.diag(deg)
            L = D - adj
            try:
                eigvals, eigvecs = np.linalg.eigh(L)
                # skip smallest (trivial) eigenvector
                start = 1
                end = min(start + dim, n)
                emb = eigvecs[:, start:end]
                if emb.shape[1] < dim:
                    pad = np.zeros((n, dim - emb.shape[1]))
                    emb = np.hstack([emb, pad])
            except Exception:
                emb = np.zeros((n, dim))
            return nodes, emb

        graph_nodes, g_emb = laplacian_embedding(graph, k_dim)
        room_nodes, r_emb = laplacian_embedding(RG, k_dim)

        # Build seed matrices
        seed_pairs = []
        if seeds:
            for rpos, nid in seeds.items():
                if nid in graph_nodes and rpos in room_nodes:
                    seed_pairs.append((nid, rpos))

        if not seed_pairs:
            # No seeds -> fallback to empty result
            return {}, {}

        # Build matrices X (graph) and Y (rooms) with rows corresponding to seeds
        g_idx = {n: i for i, n in enumerate(graph_nodes)}
        r_idx = {r: i for i, r in enumerate(room_nodes)}
        X = np.array([g_emb[g_idx[nid]] for nid, _ in seed_pairs])
        Y = np.array([r_emb[r_idx[rpos]] for _, rpos in seed_pairs])

        # Compute orthogonal transform R that maps X -> Y
        if orthogonal_procrustes is not None:
            try:
                R, scale = orthogonal_procrustes(X, Y)
            except Exception as e:
                logger.exception("orthogonal_procrustes failed in seeded_spectral_match: %s", e)
                R = None
        else:
            # Compute via SVD
            try:
                U, s, Vt = np.linalg.svd(X.T.dot(Y))
                R = U.dot(Vt)
            except Exception as e:
                logger.exception("SVD-based alignment failed in seeded_spectral_match: %s", e)
                R = None

        if R is None:
            return {}, {}

        # Transform full graph embeddings
        g_emb_aligned = g_emb.dot(R)

        # For each graph node compute nearest room embedding
        from scipy.optimize import linear_sum_assignment
        try:
            import numpy as np
            cost = np.zeros((len(graph_nodes), len(room_nodes)), dtype=float)
            for i, nid in enumerate(graph_nodes):
                for j, rpos in enumerate(room_nodes):
                    cost[i, j] = np.linalg.norm(g_emb_aligned[i] - r_emb[j])
            row_ind, col_ind = linear_sum_assignment(cost)
            proposals = {}
            confidences = {}
            for i, j in zip(row_ind, col_ind):
                nid = graph_nodes[i]
                rpos = room_nodes[j]
                proposals[nid] = rpos
                # Confidence inversely proportional to normalized distance
                maxd = cost.max() if cost.size else 1.0
                dist = cost[i, j]
                confidences[nid] = float(max(0.01, 1.0 - (dist / (maxd + 1e-6))))
            return proposals, confidences
        except Exception as e:
            logger.exception("Spectral matching assignment failed: %s", e)
            return {}, {}




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

    # --- Precheck & pruning utilities ---
    @staticmethod
    def precheck_dungeon(dungeon: Dungeon) -> Tuple[bool, Optional[str]]:
        """Run lightweight prechecks to determine if solving is worth attempting.

        Returns (ok, message). If ok==False, message explains failure reason.
        Checks include: start/triforce existence, graph connectivity, simple key vs locked-door lower bound.
        """
        if dungeon is None:
            return False, 'No dungeon data'

        # Check start/triforce presence
        if dungeon.start_pos is None:
            return False, 'PRECHECK_FAIL: Missing start position'
        if dungeon.triforce_pos is None:
            return False, 'PRECHECK_FAIL: Missing triforce position'

        # Graph connectivity (if graph exists)
        G = getattr(dungeon, 'graph', None)
        if G is None or len(G) == 0:
            return False, 'PRECHECK_FAIL: No topology graph available'

        # Attempt to find start and triforce nodes
        start_node = None
        triforce_node = None
        for n, attrs in G.nodes(data=True):
            if attrs.get('is_start'):
                start_node = n
            if attrs.get('is_triforce'):
                triforce_node = n
        # If nodes unknown, try use room_to_node mapping
        room_to_node = getattr(dungeon, 'room_to_node', {}) or {}
        if not start_node:
            start_node = room_to_node.get(dungeon.start_pos)
        if not triforce_node:
            triforce_node = room_to_node.get(dungeon.triforce_pos)

        # If either node is still missing, fall back to connectivity on undirected graph
        if start_node is None or triforce_node is None:
            # Ensure graph has at least two nodes
            if len(G.nodes()) < 2:
                return False, 'PRECHECK_FAIL: Topology too small'
        else:
            try:
                if not nx.has_path(G.to_undirected(), start_node, triforce_node):
                    return False, 'PRECHECK_FAIL: Start and triforce disconnected in topology'
            except Exception:
                # Fallback to optimistic pass
                pass

        # Locked door minimal key requirement: compute shortest path in terms of locked-door count
        def locked_cost(u, v, data):
            label = data.get('label', '')
            etype = data.get('edge_type') or EDGE_TYPE_MAP.get(label, 'open')
            return 1 if etype in ('locked', 'key_locked') else 0

        try:
            import heapq
            # Modified Dijkstra on G treating locked edges as cost=1
            def min_locked_between(s, t):
                dist = {s: 0}
                pq = [(0, s)]
                while pq:
                    d, u = heapq.heappop(pq)
                    if u == t:
                        return d
                    if d != dist.get(u, 1e9):
                        continue
                    for v in set(G.successors(u)) | set(G.predecessors(u)):
                        c = locked_cost(u, v, G.get_edge_data(u, v, {}))
                        nd = d + c
                        if nd < dist.get(v, 1e9):
                            dist[v] = nd
                            heapq.heappush(pq, (nd, v))
                return 1e9

            if start_node is not None and triforce_node is not None:
                min_locked = min_locked_between(start_node, triforce_node)
                # Count available small keys in rooms
                key_count = 0
                for pos, room in dungeon.rooms.items():
                    # Look for key tiles in semantic grid if available
                    if getattr(room, 'semantic_grid', None) is not None:
                        if (room.semantic_grid == SEMANTIC_PALETTE['KEY']).any():
                            key_count += 1
                if key_count < min_locked:
                    return False, f'PRECHECK_FAIL: Insufficient small keys (need {min_locked}, have {key_count})'
        except Exception:
            # Prior check best effort; ignore on failure
            pass

        return True, None

    @staticmethod
    def prune_dead_ends(rooms: Dict[Tuple[int, int], Room], preserve: Optional[Set[Tuple[int, int]]] = None) -> Tuple[Dict[Tuple[int, int], Room], List[Tuple[int, int]]]:
        """Iteratively remove leaf rooms (degree==1) that do not contain keys/triforce/start/boss.

        Returns (pruned_rooms, removed_positions)
        """
        preserve = set(preserve or [])
        pruned = dict(rooms)
        removed = []
        changed = True
        while changed:
            changed = False
            adj = RoomGraphMatcher()._build_room_adjacency(pruned)
            leaves = [pos for pos, nbrs in adj.items() if len(nbrs) <= 1 and pos not in preserve]
            for pos in leaves:
                room = pruned.get(pos)
                if room is None:
                    continue
                # Preserve if room contains key/triforce/boss/start
                has_key = False
                if getattr(room, 'semantic_grid', None) is not None:
                    has_key = ((room.semantic_grid == SEMANTIC_PALETTE['KEY']).any())
                if room.has_triforce or room.has_boss or room.is_start or has_key:
                    continue
                # Safe to remove
                pruned.pop(pos, None)
                removed.append(pos)
                changed = True
        return pruned, removed

    
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
    
    def process_all_dungeons(self, processed_dir: str = None, graph_dir: str = None) -> Dict[str, Dungeon]:
        """
        Process all dungeons in the data folder.
        
        Args:
            processed_dir: Path to Processed/ folder with .txt files
            graph_dir: Path to Graph Processed/ folder with .dot files
            
        Returns:
            Dictionary of dungeon_id -> Dungeon
            
        Note:
            - tlozX_1.txt files are Quest 1 dungeons (use LoZ_X.dot)
            - tlozX_2.txt files are Quest 2 dungeons (use LoZ2_X.dot)
            - Each quest gets a unique dungeon_id to prevent overwrites
        """
        import re
        
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
            
            # Create unique dungeon_id that includes quest number
            dungeon_id = f"zelda_{dungeon_num}_quest{quest_num}"
            
            try:
                dungeon = self.load_dungeon(dungeon_num, variant=quest_num)
                dungeon.dungeon_id = dungeon_id
                results[dungeon_id] = dungeon
                logger.info("Processed %s: %d rooms", dungeon_id, len(dungeon.rooms))
            except Exception as e:
                logger.exception("Error processing %s", dungeon_id)
        
        self.processed_dungeons = results
        return results

    def save_processed_data(self, output_path: str = None):
        """Save processed dungeons to disk (pickle).

        Args:
            output_path: Optional path to output pickle file. If not provided,
                         defaults to '<data_root>/processed_data.pkl'.
        Returns:
            The path to the saved file as string.
        """
        import pickle

        if output_path is None:
            output_path = self.data_root / "processed_data.pkl"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {}
        for dungeon_id, dungeon in getattr(self, 'processed_dungeons', {}).items():
            # Normalize rooms to a serializable dict
            rooms_out = {}
            for rid, room in dungeon.rooms.items():
                # Avoid boolean check on numpy arrays
                grid = getattr(room, 'grid', None)
                if grid is None:
                    grid = getattr(room, 'semantic_grid', None)
                rooms_out[str(rid)] = {
                    'grid': grid,
                    'contents': getattr(room, 'contents', []),
                    'doors': getattr(room, 'doors', {}),
                    'position': getattr(room, 'position', None)
                }

            save_data[dungeon_id] = {
                'rooms': rooms_out,
                'graph_edges': list(getattr(dungeon, 'graph', nx.DiGraph()).edges(data=True)),
                'graph_nodes': dict(getattr(dungeon, 'graph', nx.DiGraph()).nodes(data=True)),
                'layout': getattr(dungeon, 'layout', None),
                'tpe_vectors': getattr(dungeon, 'tpe_vectors', None),
                'p_matrix': getattr(dungeon, 'p_matrix', None),
                'node_features': getattr(dungeon, 'node_features', None)
            }

        with open(output_path, 'wb') as f:
            pickle.dump(save_data, f)

        logger.info("Saved processed data to %s", output_path)
        return str(output_path)


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
                
                status = "SOLVABLE" if result['solvable'] else "NOT SOLVABLE"
                logger.info("%s: %s", dungeon_key, status)
                if result['solvable']:
                    logger.debug("%s Path: %d steps, %d rooms", dungeon_key, result['path_length'], result['rooms_traversed'])
                else:
                    logger.debug("%s Reason: %s", dungeon_key, result.get('reason', 'Unknown'))
            except Exception as e:
                results[dungeon_key] = {'solvable': False, 'error': str(e)}
                logger.exception("%s: ERROR during processing", dungeon_key)
    
    # Summary
    solvable_count = sum(1 for r in results.values() if r.get('solvable'))
    total = len(results)
    logger.info("SUMMARY: %d/%d solvable (%.1f%%)", solvable_count, total, 100*solvable_count/total)
    
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
    import argparse
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=r"C:\Users\MPhuc\Desktop\KLTN\Data\The Legend of Zelda")
    p.add_argument("--no-variants", action="store_true", help="Only run variant 1")
    args = p.parse_args()
    logger.info("Testing dungeons at %s", args.data_root)
    test_all_dungeons(args.data_root, include_variants=not args.no_variants)
