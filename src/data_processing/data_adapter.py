"""
H-MOLQD Block I: Intelligent Data Adapter
==========================================

Data Ingestion & Topological Alignment for Zelda Dungeon Generation.

This module handles:
1. VGLC text file parsing (.txt) for dungeon layouts
2. Graphviz topology files (.dot) for dungeon graphs
3. Auto-phase alignment to correct padding/offset errors
4. Graph fingerprinting to map text chunks to graph nodes

Mathematical Formulation:
-------------------------
Given raw text T and graph G = (V, E):
1. Extract rooms R = {r_1, ..., r_n} from T via slot-based parsing
2. Parse G to obtain node attributes and edge types
3. Align R ↔ V via spatial heuristics and content matching
4. Output DungeonTensor D ∈ ℝ^{N×H×W×C} and aligned NetworkX graph

"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field

import numpy as np
import networkx as nx
from PIL import Image

# Import core definitions from KLTN project
from src.core.definitions import (
    SEMANTIC_PALETTE,
    CHAR_TO_SEMANTIC,
    ROOM_HEIGHT,
    ROOM_WIDTH,
    EDGE_TYPE_MAP,
    NODE_CONTENT_MAP,
    TileID,
)

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class RoomTensor:
    """
    A single room represented as a multi-channel tensor.
    
    Attributes:
        room_id: Unique identifier (typically row*100 + col)
        position: (row, col) grid position in dungeon layout
        semantic_grid: Semantic tile IDs [H, W]
        char_grid: Original character grid [H, W]
        graph_node_id: Matched graph node ID (if aligned)
        contents: List of items/entities in the room
        doors: Door information {direction: door_type}
        features: Additional computed features
    """
    room_id: int
    position: Tuple[int, int]
    semantic_grid: np.ndarray  # [ROOM_HEIGHT, ROOM_WIDTH] int
    char_grid: np.ndarray      # [ROOM_HEIGHT, ROOM_WIDTH] str
    graph_node_id: Optional[int] = None
    contents: List[str] = field(default_factory=list)
    doors: Dict[str, str] = field(default_factory=dict)
    features: Dict[str, Any] = field(default_factory=dict)
    
    def to_tensor(self, num_classes: int = 44) -> np.ndarray:
        """
        Convert to one-hot tensor [H, W, C].
        
        Args:
            num_classes: Number of semantic classes
            
        Returns:
            One-hot encoded tensor
        """
        one_hot = np.zeros(
            (self.semantic_grid.shape[0], self.semantic_grid.shape[1], num_classes),
            dtype=np.float32
        )
        for i in range(self.semantic_grid.shape[0]):
            for j in range(self.semantic_grid.shape[1]):
                tile_id = int(self.semantic_grid[i, j])
                if 0 <= tile_id < num_classes:
                    one_hot[i, j, tile_id] = 1.0
        return one_hot


@dataclass
class DungeonTensor:
    """
    Complete dungeon representation as structured tensors.
    
    Attributes:
        dungeon_id: Unique dungeon identifier
        rooms: Dict mapping position to RoomTensor
        graph: NetworkX DiGraph with aligned node attributes
        layout_grid: 2D array showing room positions
        global_semantic: Stitched semantic grid (if computed)
        tpe_vectors: Topological Positional Encoding [N, k_dim]
        p_matrix: Dependency matrix [N, N, 3]
        node_features: Node feature vectors [N, 5]
        node_to_room: Mapping from graph node to room position
    """
    dungeon_id: str
    rooms: Dict[Tuple[int, int], RoomTensor]
    graph: nx.DiGraph
    layout_grid: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=int))
    global_semantic: Optional[np.ndarray] = None
    tpe_vectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 8), dtype=np.float32))
    p_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 3), dtype=np.float32))
    node_features: np.ndarray = field(default_factory=lambda: np.zeros((0, 5), dtype=np.float32))
    node_to_room: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    
    @property
    def num_rooms(self) -> int:
        return len(self.rooms)
    
    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()
    
    def get_room_tensors(self, num_classes: int = 44) -> np.ndarray:
        """
        Get stacked room tensors [N, H, W, C].
        
        Args:
            num_classes: Number of semantic classes
            
        Returns:
            Stacked one-hot tensors for all rooms
        """
        if not self.rooms:
            return np.zeros((0, ROOM_HEIGHT, ROOM_WIDTH, num_classes), dtype=np.float32)
        
        tensors = []
        for pos in sorted(self.rooms.keys()):
            tensors.append(self.rooms[pos].to_tensor(num_classes))
        return np.stack(tensors, axis=0)


# ============================================================================
# VGLC PARSER
# ============================================================================

class VGLCParser:
    """
    Parser for Video Game Level Corpus (VGLC) text files.
    
    VGLC Zelda Format:
    - Grid divided into 11-col × 16-row slots
    - Each slot is either a room or a gap (all dashes)
    - Characters: F=floor, W=wall, D=door, S=stair, B=block, M=monster
    """
    
    SLOT_WIDTH = ROOM_WIDTH    # 11 characters per room
    SLOT_HEIGHT = ROOM_HEIGHT  # 16 rows per room
    GAP_CHAR = '-'
    
    def __init__(self, gap_threshold: float = 0.7):
        """
        Initialize parser.
        
        Args:
            gap_threshold: Fraction of dashes to consider slot as gap
        """
        self.gap_threshold = gap_threshold
        
    def load_grid(self, filepath: Union[str, Path]) -> np.ndarray:
        """
        Load VGLC text file into numpy character array.
        
        Args:
            filepath: Path to .txt file
            
        Returns:
            2D character array
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"VGLC file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [line.rstrip('\n') for line in f]
        
        if not lines:
            return np.zeros((0, 0), dtype='<U1')
        
        # Pad all lines to max width
        max_width = max(len(line) for line in lines)
        padded = [list(line.ljust(max_width, self.GAP_CHAR)) for line in lines]
        
        return np.array(padded, dtype='<U1')
    
    def is_room_slot(self, slot_grid: np.ndarray) -> bool:
        """
        Check if a slot contains a valid room (not a gap).
        
        Args:
            slot_grid: Character array for one slot
            
        Returns:
            True if slot contains a room
        """
        if slot_grid.size == 0:
            return False
        
        # Count gap characters
        gap_count = np.sum(slot_grid == self.GAP_CHAR)
        total = slot_grid.size
        
        if gap_count > total * self.gap_threshold:
            return False
        
        # Check for structural elements
        wall_count = np.sum(slot_grid == 'W')
        floor_count = np.sum((slot_grid == 'F') | (slot_grid == '.'))
        
        return wall_count >= 20 and floor_count >= 5
    
    def extract_rooms(self, filepath: Union[str, Path]) -> List[RoomTensor]:
        """
        Extract all rooms from VGLC file.
        
        Args:
            filepath: Path to .txt file
            
        Returns:
            List of RoomTensor objects
        """
        grid = self.load_grid(filepath)
        
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
                
                # Pad if needed
                if slot_grid.shape[0] < self.SLOT_HEIGHT:
                    pad_h = self.SLOT_HEIGHT - slot_grid.shape[0]
                    slot_grid = np.vstack([
                        slot_grid,
                        np.full((pad_h, slot_grid.shape[1]), self.GAP_CHAR)
                    ])
                if slot_grid.shape[1] < self.SLOT_WIDTH:
                    pad_w = self.SLOT_WIDTH - slot_grid.shape[1]
                    slot_grid = np.hstack([
                        slot_grid,
                        np.full((slot_grid.shape[0], pad_w), self.GAP_CHAR)
                    ])
                
                if self.is_room_slot(slot_grid):
                    # Convert to semantic grid
                    semantic_grid = self._chars_to_semantic(slot_grid)
                    
                    # Extract room features
                    contents = self._extract_contents(slot_grid)
                    doors = self._detect_doors(slot_grid)
                    
                    room = RoomTensor(
                        room_id=row_slot * 100 + col_slot,
                        position=(row_slot, col_slot),
                        semantic_grid=semantic_grid,
                        char_grid=slot_grid.copy(),
                        contents=contents,
                        doors=doors,
                    )
                    rooms.append(room)
        
        logger.info(f"Extracted {len(rooms)} rooms from {filepath}")
        return rooms
    
    def _chars_to_semantic(self, char_grid: np.ndarray) -> np.ndarray:
        """Convert character grid to semantic IDs."""
        semantic = np.zeros(char_grid.shape, dtype=np.int32)
        
        for i in range(char_grid.shape[0]):
            for j in range(char_grid.shape[1]):
                char = char_grid[i, j]
                semantic[i, j] = CHAR_TO_SEMANTIC.get(char, TileID.VOID)
        
        return semantic
    
    def _extract_contents(self, char_grid: np.ndarray) -> List[str]:
        """Extract notable contents from room."""
        contents = []
        
        unique_chars = set(char_grid.flatten())
        
        if 'M' in unique_chars:
            contents.append('enemy')
        if 'S' in unique_chars:
            contents.append('stair')
        if 'P' in unique_chars:
            contents.append('element')
        if 'B' in unique_chars:
            contents.append('block')
        
        return contents
    
    def _detect_doors(self, char_grid: np.ndarray) -> Dict[str, str]:
        """
        Detect doors on room boundaries.
        
        Door positions in 16×11 room:
        - North: row 0, cols 4-6
        - South: row 15, cols 4-6
        - East: col 10, rows 7-8
        - West: col 0, rows 7-8
        """
        doors = {}
        
        # North door
        north_cells = char_grid[0, 4:7]
        if 'D' in north_cells or 'F' in north_cells or '.' in north_cells:
            doors['N'] = 'open'
        
        # South door
        south_cells = char_grid[15, 4:7] if char_grid.shape[0] > 15 else []
        if len(south_cells) > 0 and ('D' in south_cells or 'F' in south_cells or '.' in south_cells):
            doors['S'] = 'open'
        
        # East door
        east_cells = char_grid[7:9, 10] if char_grid.shape[1] > 10 else []
        if len(east_cells) > 0 and ('D' in east_cells or 'F' in east_cells or '.' in east_cells):
            doors['E'] = 'open'
        
        # West door
        west_cells = char_grid[7:9, 0]
        if 'D' in west_cells or 'F' in west_cells or '.' in west_cells:
            doors['W'] = 'open'
        
        return doors


# ============================================================================
# GRAPHVIZ PARSER
# ============================================================================

class GraphvizParser:
    """
    Parser for Graphviz DOT files representing dungeon topology.
    
    DOT Format:
    - Nodes: integer IDs with labels indicating content
    - Edges: directed with labels indicating lock type
    - Node labels: s=start, t=triforce, b=boss, e=enemy, k=key, I=item
    - Edge labels: k=key_locked, b=bombable, l=soft_locked, empty=open
    """
    
    def __init__(self):
        self.node_pattern = re.compile(
            r'(\d+)\s*\[([^\]]*)\]'
        )
        self.edge_pattern = re.compile(
            r'(\d+)\s*->\s*(\d+)\s*(?:\[([^\]]*)\])?'
        )
        self.label_pattern = re.compile(r'label\s*=\s*"([^"]*)"')
    
    def parse(self, filepath: Union[str, Path]) -> nx.DiGraph:
        """
        Parse DOT file into NetworkX DiGraph.
        
        Args:
            filepath: Path to .dot file
            
        Returns:
            NetworkX DiGraph with node and edge attributes
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"DOT file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        G = nx.DiGraph()
        
        # Parse nodes
        for match in self.node_pattern.finditer(content):
            node_id = int(match.group(1))
            attrs_str = match.group(2)
            
            attrs = self._parse_node_attrs(attrs_str)
            G.add_node(node_id, **attrs)
        
        # Parse edges
        for match in self.edge_pattern.finditer(content):
            from_node = int(match.group(1))
            to_node = int(match.group(2))
            attrs_str = match.group(3) or ''
            
            attrs = self._parse_edge_attrs(attrs_str)
            G.add_edge(from_node, to_node, **attrs)
        
        # Ensure all edge endpoints exist as nodes
        for u, v in list(G.edges()):
            if u not in G.nodes:
                G.add_node(u, label='')
            if v not in G.nodes:
                G.add_node(v, label='')
        
        logger.info(f"Parsed graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges from {filepath}")
        return G
    
    def _parse_node_attrs(self, attrs_str: str) -> Dict[str, Any]:
        """Parse node attributes from DOT string."""
        attrs = {
            'label': '',
            'is_start': False,
            'has_enemy': False,
            'has_key': False,
            'has_boss_key': False,
            'has_triforce': False,
            'has_boss': False,
            'has_item': False,
            'contents': [],
        }
        
        # Extract label
        label_match = self.label_pattern.search(attrs_str)
        if label_match:
            label = label_match.group(1)
            attrs['label'] = label
            
            # Parse label parts
            parts = [p.strip() for p in label.split(',')]
            
            for part in parts:
                if part in NODE_CONTENT_MAP:
                    content = NODE_CONTENT_MAP[part]
                    attrs['contents'].append(content)
                    
                    if content == 'start':
                        attrs['is_start'] = True
                    elif content == 'enemy':
                        attrs['has_enemy'] = True
                    elif content == 'key':
                        attrs['has_key'] = True
                    elif content == 'boss_key':
                        attrs['has_boss_key'] = True
                    elif content == 'triforce':
                        attrs['has_triforce'] = True
                    elif content == 'boss':
                        attrs['has_boss'] = True
                    elif content in ('item', 'key_item'):
                        attrs['has_item'] = True
        
        return attrs
    
    def _parse_edge_attrs(self, attrs_str: str) -> Dict[str, Any]:
        """Parse edge attributes from DOT string."""
        attrs = {
            'label': '',
            'edge_type': 'open',
        }
        
        label_match = self.label_pattern.search(attrs_str)
        if label_match:
            label = label_match.group(1)
            attrs['label'] = label
            attrs['edge_type'] = EDGE_TYPE_MAP.get(label, 'open')
        
        return attrs


# ============================================================================
# PHASE ALIGNER
# ============================================================================

class PhaseAligner:
    """
    Auto-Phase Alignment for VGLC data.
    
    Corrects common issues:
    - Padding errors (extra whitespace)
    - Offset misalignment between rooms
    - Boundary detection issues
    """
    
    def __init__(self, tolerance: int = 2):
        """
        Initialize aligner.
        
        Args:
            tolerance: Maximum pixel offset to attempt correction
        """
        self.tolerance = tolerance
    
    def analyze_density(self, rooms: List[RoomTensor]) -> Dict[str, Any]:
        """
        Analyze structural density across rooms.
        
        Args:
            rooms: List of extracted rooms
            
        Returns:
            Density statistics
        """
        stats = {
            'num_rooms': len(rooms),
            'avg_wall_count': 0.0,
            'avg_floor_count': 0.0,
            'density_variance': 0.0,
            'potential_issues': [],
        }
        
        if not rooms:
            return stats
        
        wall_counts = []
        floor_counts = []
        
        for room in rooms:
            walls = np.sum(room.semantic_grid == TileID.WALL)
            floors = np.sum(room.semantic_grid == TileID.FLOOR)
            wall_counts.append(walls)
            floor_counts.append(floors)
        
        stats['avg_wall_count'] = np.mean(wall_counts)
        stats['avg_floor_count'] = np.mean(floor_counts)
        stats['density_variance'] = np.var(wall_counts)
        
        # Detect potential issues
        for i, room in enumerate(rooms):
            if wall_counts[i] < stats['avg_wall_count'] * 0.5:
                stats['potential_issues'].append({
                    'room_id': room.room_id,
                    'issue': 'low_wall_count',
                    'value': wall_counts[i],
                })
        
        return stats
    
    def correct_boundaries(self, rooms: List[RoomTensor]) -> List[RoomTensor]:
        """
        Attempt to correct misaligned room boundaries.
        
        Args:
            rooms: List of rooms to correct
            
        Returns:
            List of corrected rooms
        """
        corrected = []
        
        for room in rooms:
            # Check for boundary issues
            grid = room.semantic_grid
            
            # Check if walls are properly placed at boundaries
            top_walls = np.sum(grid[0, :] == TileID.WALL)
            bottom_walls = np.sum(grid[-1, :] == TileID.WALL)
            left_walls = np.sum(grid[:, 0] == TileID.WALL)
            right_walls = np.sum(grid[:, -1] == TileID.WALL)
            
            # If boundaries seem off, try to fix
            if top_walls < 5 or bottom_walls < 5:
                # Potentially misaligned - attempt correction
                grid = self._shift_grid(grid, direction='vertical')
            
            if left_walls < 5 or right_walls < 5:
                grid = self._shift_grid(grid, direction='horizontal')
            
            # Create corrected room
            corrected_room = RoomTensor(
                room_id=room.room_id,
                position=room.position,
                semantic_grid=grid,
                char_grid=room.char_grid,
                graph_node_id=room.graph_node_id,
                contents=room.contents,
                doors=room.doors,
                features=room.features,
            )
            corrected.append(corrected_room)
        
        return corrected
    
    def _shift_grid(self, grid: np.ndarray, direction: str) -> np.ndarray:
        """Attempt to shift grid to fix alignment."""
        # This is a placeholder for more sophisticated alignment
        # In practice, would use cross-correlation or template matching
        return grid


# ============================================================================
# GRAPH FINGERPRINTER
# ============================================================================

class GraphFingerprinter:
    """
    Maps extracted rooms to graph nodes via content matching.
    
    Fingerprinting Strategy:
    1. Match start node to room with start position
    2. Match triforce node to room with triforce
    3. Use spatial proximity and door connectivity
    4. Fall back to content similarity scoring
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize fingerprinter.
        
        Args:
            strict_mode: Require exact content matches
        """
        self.strict_mode = strict_mode
    
    def align(
        self,
        rooms: List[RoomTensor],
        graph: nx.DiGraph,
    ) -> Tuple[Dict[int, Tuple[int, int]], Dict[Tuple[int, int], int]]:
        """
        Align graph nodes to room positions.
        
        Args:
            rooms: List of extracted rooms
            graph: Parsed topology graph
            
        Returns:
            (node_to_room, room_to_node) mapping dicts
        """
        node_to_room: Dict[int, Tuple[int, int]] = {}
        room_to_node: Dict[Tuple[int, int], int] = {}
        
        # Build room position index
        room_by_pos = {room.position: room for room in rooms}
        
        # Step 1: Match special nodes (start, triforce, boss)
        for node_id, data in graph.nodes(data=True):
            if data.get('is_start'):
                # Find room with start indicator (usually top-left or center)
                # VGLC convention: start is often at a specific position
                start_room = self._find_start_room(rooms)
                if start_room:
                    node_to_room[node_id] = start_room.position
                    room_to_node[start_room.position] = node_id
            
            elif data.get('has_triforce'):
                triforce_room = self._find_triforce_room(rooms)
                if triforce_room:
                    node_to_room[node_id] = triforce_room.position
                    room_to_node[triforce_room.position] = node_id
        
        # Step 2: Use spatial layout heuristics
        # Graph traversal order often matches spatial layout
        unmatched_nodes = [n for n in graph.nodes() if n not in node_to_room]
        unmatched_rooms = [r for r in rooms if r.position not in room_to_node]
        
        # BFS from start to assign remaining
        if node_to_room:
            start_node = next(iter(node_to_room.keys()))
            self._bfs_match(
                graph, start_node, room_by_pos,
                node_to_room, room_to_node,
                set(node_to_room.keys()), set(room_to_node.keys())
            )
        
        # Step 3: Greedy matching for remaining
        for node_id in unmatched_nodes:
            if node_id in node_to_room:
                continue
            
            best_room = self._find_best_match(
                node_id, graph.nodes[node_id],
                [r for r in rooms if r.position not in room_to_node]
            )
            
            if best_room:
                node_to_room[node_id] = best_room.position
                room_to_node[best_room.position] = node_id
        
        logger.info(f"Aligned {len(node_to_room)} nodes to rooms")
        return node_to_room, room_to_node
    
    def _find_start_room(self, rooms: List[RoomTensor]) -> Optional[RoomTensor]:
        """Find the starting room."""
        # Check for explicit start marker in semantic grid
        for room in rooms:
            if TileID.START in room.semantic_grid:
                return room
        
        # Fall back to positional heuristic (often bottom-center)
        if rooms:
            # Sort by row (descending) then col (ascending to center)
            sorted_rooms = sorted(rooms, key=lambda r: (-r.position[0], abs(r.position[1] - 2)))
            return sorted_rooms[0]
        
        return None
    
    def _find_triforce_room(self, rooms: List[RoomTensor]) -> Optional[RoomTensor]:
        """Find the triforce room."""
        for room in rooms:
            if TileID.TRIFORCE in room.semantic_grid:
                return room
        return None
    
    def _bfs_match(
        self,
        graph: nx.DiGraph,
        start_node: int,
        room_by_pos: Dict[Tuple[int, int], RoomTensor],
        node_to_room: Dict[int, Tuple[int, int]],
        room_to_node: Dict[Tuple[int, int], int],
        matched_nodes: Set[int],
        matched_rooms: Set[Tuple[int, int]],
    ):
        """BFS to match nodes to spatially adjacent rooms."""
        from collections import deque
        
        queue = deque([start_node])
        visited = {start_node}
        
        while queue:
            node = queue.popleft()
            
            if node not in node_to_room:
                continue
            
            room_pos = node_to_room[node]
            
            # Check neighbors in graph
            for neighbor in graph.neighbors(node):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                
                if neighbor not in node_to_room:
                    # Try to find adjacent room
                    edge_data = graph.edges[node, neighbor]
                    direction = self._infer_direction(edge_data)
                    
                    if direction:
                        adj_pos = self._get_adjacent_pos(room_pos, direction)
                        if adj_pos in room_by_pos and adj_pos not in matched_rooms:
                            node_to_room[neighbor] = adj_pos
                            room_to_node[adj_pos] = neighbor
                            matched_nodes.add(neighbor)
                            matched_rooms.add(adj_pos)
                
                queue.append(neighbor)
    
    def _infer_direction(self, edge_data: Dict) -> Optional[str]:
        """Infer movement direction from edge data."""
        # This would need more sophisticated logic based on graph structure
        return None
    
    def _get_adjacent_pos(
        self, pos: Tuple[int, int], direction: str
    ) -> Tuple[int, int]:
        """Get adjacent room position."""
        row, col = pos
        if direction == 'N':
            return (row - 1, col)
        elif direction == 'S':
            return (row + 1, col)
        elif direction == 'E':
            return (row, col + 1)
        elif direction == 'W':
            return (row, col - 1)
        return pos
    
    def _find_best_match(
        self,
        node_id: int,
        node_data: Dict,
        available_rooms: List[RoomTensor],
    ) -> Optional[RoomTensor]:
        """Find best matching room for a node."""
        if not available_rooms:
            return None
        
        best_room = None
        best_score = -1
        
        for room in available_rooms:
            score = self._compute_match_score(node_data, room)
            if score > best_score:
                best_score = score
                best_room = room
        
        return best_room
    
    def _compute_match_score(
        self, node_data: Dict, room: RoomTensor
    ) -> float:
        """Compute content similarity score."""
        score = 0.0
        
        # Check for content matches
        if node_data.get('has_enemy') and 'enemy' in room.contents:
            score += 1.0
        if node_data.get('has_key') and TileID.KEY_SMALL in room.semantic_grid:
            score += 2.0
        if node_data.get('has_boss') and TileID.BOSS in room.semantic_grid:
            score += 3.0
        if 'stair' in room.contents:
            score += 0.5  # Stairs are common
        
        return score


# ============================================================================
# ML FEATURE EXTRACTOR
# ============================================================================

class MLFeatureExtractor:
    """
    Extract ML features from dungeon graph structure.
    
    Features:
    - Topological Positional Encoding (TPE) via Laplacian eigenvectors
    - Node feature vectors (multi-hot encoding)
    - P-Matrix (dependency graph encoding)
    """
    
    @staticmethod
    def compute_tpe(
        graph: nx.Graph,
        k_dim: int = 8,
    ) -> Tuple[np.ndarray, Dict[int, int]]:
        """
        Compute Topological Positional Encoding using graph Laplacian.
        
        Mathematical formulation:
        L = D - A (graph Laplacian)
        TPE = eigenvectors of L (skipping first trivial eigenvector)
        
        Args:
            graph: NetworkX graph
            k_dim: Number of eigenvector dimensions
            
        Returns:
            tpe: Positional encoding [N, k_dim]
            node_to_idx: Node ID to array index mapping
        """
        # Convert to undirected for Laplacian
        G = graph.to_undirected() if graph.is_directed() else graph
        
        nodes = sorted(G.nodes())
        n = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        if n == 0:
            return np.zeros((0, k_dim), dtype=np.float32), {}
        
        # Build adjacency matrix with edge weights
        adj = np.zeros((n, n), dtype=np.float32)
        
        for u, v, data in G.edges(data=True):
            i, j = node_to_idx[u], node_to_idx[v]
            
            # Weight by edge type (locked edges have lower weight)
            edge_type = data.get('edge_type', 'open')
            weight = 0.5 if edge_type in ('key_locked', 'bombable', 'boss_locked') else 1.0
            
            adj[i, j] = weight
            adj[j, i] = weight
        
        # Compute Laplacian: L = D - A
        degrees = np.sum(adj, axis=1)
        D = np.diag(degrees)
        L = D - adj
        
        # Compute eigenvectors
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            
            # Skip first eigenvector (constant), take next k_dim
            tpe = eigenvectors[:, 1:1+k_dim]
            
            # Pad if not enough eigenvectors
            if tpe.shape[1] < k_dim:
                padding = np.zeros((n, k_dim - tpe.shape[1]))
                tpe = np.hstack([tpe, padding])
                
        except np.linalg.LinAlgError:
            logger.warning("Laplacian eigendecomposition failed, using zeros")
            tpe = np.zeros((n, k_dim))
        
        return tpe.astype(np.float32), node_to_idx
    
    @staticmethod
    def extract_node_features(
        graph: nx.DiGraph,
        node_order: Dict[int, int],
    ) -> np.ndarray:
        """
        Extract multi-hot feature vectors for each node.
        
        Feature vector (5 dims):
        [is_start, has_enemy, has_key, has_boss_key, has_triforce]
        
        Args:
            graph: NetworkX graph
            node_order: Node ID to index mapping
            
        Returns:
            Feature array [N, 5]
        """
        n = len(node_order)
        features = np.zeros((n, 5), dtype=np.float32)
        
        for node_id, idx in node_order.items():
            if node_id not in graph.nodes:
                continue
            
            attrs = graph.nodes[node_id]
            
            features[idx, 0] = float(attrs.get('is_start', False))
            features[idx, 1] = float(attrs.get('has_enemy', False))
            features[idx, 2] = float(attrs.get('has_key', False))
            features[idx, 3] = float(attrs.get('has_boss_key', False) or attrs.get('has_boss', False))
            features[idx, 4] = float(attrs.get('has_triforce', False))
        
        return features
    
    @staticmethod
    def build_p_matrix(
        graph: nx.DiGraph,
        node_order: Dict[int, int],
    ) -> np.ndarray:
        """
        Build dependency (prerequisite) matrix.
        
        P-Matrix shape: [N, N, 3]
        Channels:
        - 0: Key dependency
        - 1: Bomb dependency
        - 2: Boss key dependency
        
        Args:
            graph: NetworkX graph
            node_order: Node ID to index mapping
            
        Returns:
            P-Matrix [N, N, 3]
        """
        n = len(node_order)
        p_matrix = np.zeros((n, n, 3), dtype=np.float32)
        
        for u, v, data in graph.edges(data=True):
            if u not in node_order or v not in node_order:
                continue
            
            i, j = node_order[u], node_order[v]
            edge_type = data.get('edge_type', 'open')
            
            if edge_type == 'key_locked':
                p_matrix[i, j, 0] = 1.0
                p_matrix[j, i, 0] = 1.0
            elif edge_type == 'bombable':
                p_matrix[i, j, 1] = 1.0
                p_matrix[j, i, 1] = 1.0
            elif edge_type == 'boss_locked':
                p_matrix[i, j, 2] = 1.0
                p_matrix[j, i, 2] = 1.0
        
        return p_matrix


# ============================================================================
# INTELLIGENT DATA ADAPTER (Main Interface)
# ============================================================================

class IntelligentDataAdapter:
    """
    Main interface for H-MOLQD Block I: Intelligent Data Adapter.
    
    Combines all components for end-to-end data processing:
    1. Parse VGLC text files and DOT graphs
    2. Auto-align rooms with phase correction
    3. Fingerprint and match rooms to graph nodes
    4. Extract ML features (TPE, P-matrix, node features)
    5. Output structured DungeonTensor
    
    Usage:
        adapter = IntelligentDataAdapter(data_dir='Data/The Legend of Zelda')
        dungeon = adapter.load_dungeon('tloz1_1')
        
        # Access tensors
        room_tensors = dungeon.get_room_tensors()  # [N, H, W, C]
        tpe = dungeon.tpe_vectors                   # [N, k_dim]
        p_matrix = dungeon.p_matrix                 # [N, N, 3]
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        tpe_dim: int = 8,
        auto_align: bool = True,
    ):
        """
        Initialize adapter.
        
        Args:
            data_dir: Root directory containing VGLC data
            tpe_dim: Dimension of topological positional encoding
            auto_align: Enable auto-phase alignment
        """
        self.data_dir = Path(data_dir)
        self.tpe_dim = tpe_dim
        self.auto_align = auto_align
        
        # Initialize components
        self.vglc_parser = VGLCParser()
        self.dot_parser = GraphvizParser()
        self.aligner = PhaseAligner()
        self.fingerprinter = GraphFingerprinter()
        self.feature_extractor = MLFeatureExtractor()
        
        # Data paths (KLTN project structure)
        self.text_dir = self.data_dir / 'Processed'
        self.graph_dir = self.data_dir / 'Graph Processed'
        self.image_dir = self.data_dir / 'Original'
    
    def load_dungeon(
        self,
        dungeon_id: str,
        load_images: bool = False,
    ) -> DungeonTensor:
        """
        Load and process a complete dungeon.
        
        Args:
            dungeon_id: Dungeon identifier (e.g., 'tloz1_1')
            load_images: Also load original screenshots
            
        Returns:
            DungeonTensor with all processed data
        """
        logger.info(f"Loading dungeon: {dungeon_id}")
        
        # Determine file paths
        text_path = self._find_text_file(dungeon_id)
        graph_path = self._find_graph_file(dungeon_id)
        
        # Parse VGLC rooms
        rooms = self.vglc_parser.extract_rooms(text_path)
        
        # Auto-align if enabled
        if self.auto_align:
            density_stats = self.aligner.analyze_density(rooms)
            if density_stats['potential_issues']:
                logger.info(f"Auto-aligning {len(density_stats['potential_issues'])} rooms")
                rooms = self.aligner.correct_boundaries(rooms)
        
        # Parse graph
        graph = self.dot_parser.parse(graph_path)
        
        # Fingerprint and align
        node_to_room, room_to_node = self.fingerprinter.align(rooms, graph)
        
        # Update rooms with graph node IDs
        for room in rooms:
            if room.position in room_to_node:
                room.graph_node_id = room_to_node[room.position]
        
        # Extract ML features
        tpe, node_order = self.feature_extractor.compute_tpe(graph, self.tpe_dim)
        node_features = self.feature_extractor.extract_node_features(graph, node_order)
        p_matrix = self.feature_extractor.build_p_matrix(graph, node_order)
        
        # Build layout grid
        layout_grid = self._build_layout_grid(rooms)
        
        # Create DungeonTensor
        rooms_dict = {room.position: room for room in rooms}
        
        dungeon = DungeonTensor(
            dungeon_id=dungeon_id,
            rooms=rooms_dict,
            graph=graph,
            layout_grid=layout_grid,
            tpe_vectors=tpe,
            p_matrix=p_matrix,
            node_features=node_features,
            node_to_room=node_to_room,
        )
        
        logger.info(f"Loaded dungeon with {dungeon.num_rooms} rooms, {dungeon.num_nodes} nodes")
        return dungeon
    
    def load_all_dungeons(self) -> List[DungeonTensor]:
        """Load all available dungeons."""
        dungeons = []
        
        # Find all text files
        if self.text_dir.exists():
            for txt_file in self.text_dir.glob('*.txt'):
                if txt_file.name == 'README.txt':
                    continue
                
                dungeon_id = txt_file.stem
                try:
                    dungeon = self.load_dungeon(dungeon_id)
                    dungeons.append(dungeon)
                except Exception as e:
                    logger.warning(f"Failed to load {dungeon_id}: {e}")
        
        return dungeons
    
    def _find_text_file(self, dungeon_id: str) -> Path:
        """Find VGLC text file for dungeon."""
        # Try exact match
        exact_path = self.text_dir / f"{dungeon_id}.txt"
        if exact_path.exists():
            return exact_path
        
        # Try variations
        for pattern in [f"{dungeon_id}*.txt", f"*{dungeon_id}*.txt"]:
            matches = list(self.text_dir.glob(pattern))
            if matches:
                return matches[0]
        
        raise FileNotFoundError(f"No text file found for dungeon: {dungeon_id}")
    
    def _find_graph_file(self, dungeon_id: str) -> Path:
        """Find DOT graph file for dungeon."""
        # Map VGLC naming to graph naming
        # tloz1_1 -> LoZ_1.dot
        name_map = {
            'tloz1_1': 'LoZ_1', 'tloz1_2': 'LoZ_1',
            'tloz2_1': 'LoZ_2', 'tloz2_2': 'LoZ_2',
            'tloz3_1': 'LoZ_3', 'tloz3_2': 'LoZ_3',
            'tloz4_1': 'LoZ_4', 'tloz4_2': 'LoZ_4',
            'tloz5_1': 'LoZ_5', 'tloz5_2': 'LoZ_5',
            'tloz6_1': 'LoZ_6', 'tloz6_2': 'LoZ_6',
            'tloz7_1': 'LoZ_7', 'tloz7_2': 'LoZ_7',
            'tloz8_1': 'LoZ_8', 'tloz8_2': 'LoZ_8',
            'tloz9_1': 'LoZ_9', 'tloz9_2': 'LoZ_9',
        }
        
        graph_name = name_map.get(dungeon_id, dungeon_id)
        graph_path = self.graph_dir / f"{graph_name}.dot"
        
        if graph_path.exists():
            return graph_path
        
        # Try direct match
        direct_path = self.graph_dir / f"{dungeon_id}.dot"
        if direct_path.exists():
            return direct_path
        
        raise FileNotFoundError(f"No graph file found for dungeon: {dungeon_id}")
    
    def _build_layout_grid(self, rooms: List[RoomTensor]) -> np.ndarray:
        """Build 2D layout grid showing room positions."""
        if not rooms:
            return np.zeros((0, 0), dtype=int)
        
        # Find bounds
        max_row = max(r.position[0] for r in rooms)
        max_col = max(r.position[1] for r in rooms)
        
        layout = np.zeros((max_row + 1, max_col + 1), dtype=int)
        
        for room in rooms:
            layout[room.position] = room.room_id
        
        return layout


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_zelda_dungeon(
    dungeon_id: str,
    data_dir: str = 'Data/The Legend of Zelda',
) -> DungeonTensor:
    """
    Convenience function to load a Zelda dungeon.
    
    Args:
        dungeon_id: Dungeon identifier (e.g., 'tloz1_1')
        data_dir: Path to VGLC data directory
        
    Returns:
        DungeonTensor with all processed data
    """
    adapter = IntelligentDataAdapter(data_dir)
    return adapter.load_dungeon(dungeon_id)


def batch_load_dungeons(
    data_dir: str = 'Data/The Legend of Zelda',
) -> List[DungeonTensor]:
    """
    Load all available Zelda dungeons.
    
    Args:
        data_dir: Path to VGLC data directory
        
    Returns:
        List of DungeonTensor objects
    """
    adapter = IntelligentDataAdapter(data_dir)
    return adapter.load_all_dungeons()
