"""
ZELDA DUNGEON CORE - Clean Implementation
=========================================
Core logic for VGLC room extraction, graph alignment, and dungeon stitching.

This is the SINGLE SOURCE OF TRUTH for all dungeon processing.

VGLC Format (Zelda Dungeons):
- Each room is 16 rows x 11 columns
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

# pyright: reportPrivateUsage=false

import logging
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__) 

# ==========================================
# CONSTANTS - Import from canonical source
# ==========================================
# Import from src/core/definitions.py (the SINGLE SOURCE OF TRUTH)
# This avoids duplication and ensures consistency across the codebase
from src.core.definitions import (
    SEMANTIC_PALETTE,
    ROOM_HEIGHT,
    ROOM_WIDTH,
    parse_node_label_tokens,
    normalize_node_label,
    parse_edge_type_tokens,
    select_primary_edge_type,
)
from src.zelda_data.matching.room_graph_matching import (
    refine_mapping_with_swaps,
    solve_assignment_with_fallback,
)
from src.zelda_data.matching.infer_missing_mappings import (
    apply_label_hints,
    assign_pairs_from_scores,
    build_component_context,
    build_score_matrix,
    compute_normalized_room_centers,
    propagate_from_anchors,
    seed_from_special_nodes,
)
from src.zelda_data.matching.spectral_refinement import (
    edge_consistency_score,
    local_refine_assignments,
    seeded_spectral_match as seeded_spectral_match_impl,
)
from src.zelda_data.matching.match_orchestration import (
    match_rooms_to_graph as match_rooms_to_graph_impl,
)
from src.zelda_data.matching.topology_utils import (
    build_room_adjacency,
    find_entrance_room,
    find_farthest_dead_end,
    find_room_at_distance,
    room_signature,
)
from src.zelda_data.stitching.stitch_orchestration import (
    build_stitched_room_layout_from_rooms,
    build_room_node_mappings,
    place_special_markers,
    project_output_metadata,
)
from src.zelda_data.stitching.graph_placement import (
    apply_door_types_from_graph,
    find_boundary_doors,
    place_entities_from_graph,
    place_items_from_graph,
)
from src.zelda_data.stitching.connectivity import (
    connect_doors,
    ensure_room_connectivity,
    find_floor_near_door,
)
from src.zelda_data.stitching.compaction import compact_rooms
from src.zelda_data.validation.precheck_pruning import (
    precheck_dungeon as precheck_dungeon_impl,
    prune_dead_ends as prune_dead_ends_impl,
)
from src.zelda_data.validation.dungeon_validation import (
    validate_dungeon as validate_dungeon_impl,
)
from src.zelda_data.adapter_io import (
    layout_from_graph as layout_from_graph_impl,
    load_dungeon as load_dungeon_impl,
    process_all_dungeons as process_all_dungeons_impl,
    save_processed_data as save_processed_data_impl,
)
from src.zelda_data.solver_validation import (
    solve as solve_dungeon_impl,
    solve_with_graph as solve_with_graph_impl,
    solve_with_grid as solve_with_grid_impl,
    solve_with_state_space as solve_with_state_space_impl,
)
from src.zelda_data.convenience import (
    test_all_dungeons as test_all_dungeons_impl,
    visualize_semantic_grid as visualize_semantic_grid_impl,
)
from src.zelda_data.conversion import (
    convert_dungeon_to_dungeondata as convert_dungeon_to_dungeondata_impl,
    convert_room_to_roomdata as convert_room_to_roomdata_impl,
)
from src.zelda_data.parsers.core_parsers import (
    DOTParser as DOTParserImpl,
    GridBasedRoomExtractor as GridBasedRoomExtractorImpl,
    VGLCParser as VGLCParserImpl,
)
from src.zelda_data.features.ml_features import MLFeatureExtractor as MLFeatureExtractorImpl
from src.zelda_data.layout.hybrid_layout import HybridLayoutEngine as HybridLayoutEngineImpl
from src.zelda_data.solver.state_space import StateSpaceGraphSolverCore
from src.zelda_data.reporting.virtual_node_reporting import log_virtual_node_report


# ==========================================
# GRID-BASED ROOM EXTRACTOR (from adapter.py)
# ==========================================
class GridBasedRoomExtractor:
    """Compatibility wrapper around extracted parser implementation."""

    SLOT_WIDTH = GridBasedRoomExtractorImpl.SLOT_WIDTH
    SLOT_HEIGHT = GridBasedRoomExtractorImpl.SLOT_HEIGHT
    GAP_MARKER = GridBasedRoomExtractorImpl.GAP_MARKER
    WALL_MARKER = GridBasedRoomExtractorImpl.WALL_MARKER

    def __init__(self):
        self._impl = GridBasedRoomExtractorImpl()

    def _load_grid(self, filepath: str) -> np.ndarray:
        return self._impl._load_grid(filepath)

    def _is_room_slot(self, slot_grid: np.ndarray) -> bool:
        return self._impl._is_room_slot(slot_grid)

    def extract(self, filepath: str) -> List[Tuple[Tuple[int, int], np.ndarray]]:
        return self._impl.extract(filepath)

    def extract_with_ids(self, filepath: str) -> List[Tuple[int, np.ndarray]]:
        return self._impl.extract_with_ids(filepath)


# ==========================================
# ML FEATURE EXTRACTION (from adapter.py)
# ==========================================
class MLFeatureExtractor:
    """Compatibility wrapper around extracted ML feature implementation."""

    @staticmethod
    def compute_laplacian_pe(G: nx.Graph, k_dim: int = 8) -> Tuple[np.ndarray, Dict[int, int]]:
        return MLFeatureExtractorImpl.compute_laplacian_pe(G, k_dim=k_dim)

    @staticmethod
    def extract_node_features(G: nx.DiGraph, node_order: Dict[int, int]) -> np.ndarray:
        return MLFeatureExtractorImpl.extract_node_features(G, node_order)

    @staticmethod
    def build_p_matrix(G: nx.DiGraph, node_order: Dict[int, int]) -> np.ndarray:
        return MLFeatureExtractorImpl.build_p_matrix(G, node_order)


# ==========================================
# DATA CLASSES
# ==========================================
@dataclass
class Room:
    """A single dungeon room."""
    position: Tuple[int, int]  # (row, col) in VGLC grid
    char_grid: np.ndarray      # Raw character grid [ROOM_HEIGHT, ROOM_WIDTH]
    semantic_grid: np.ndarray  # Semantic ID grid [ROOM_HEIGHT, ROOM_WIDTH]
    doors: Dict[str, bool]     # {N, S, E, W} -> has_door
    has_stair: bool
    has_triforce: bool = False
    has_boss: bool = False
    is_start: bool = False
    graph_node_id: Optional[int] = None
    node_label: Optional[str] = None  # Graph node label (e.g. 'e,k')


@dataclass 
class Dungeon:
    """A complete dungeon with rooms and connectivity."""
    dungeon_id: str
    rooms: Dict[Tuple[int, int], Room]  # position -> Room
    graph: nx.DiGraph                    # Connectivity graph from DOT
    start_pos: Optional[Tuple[int, int]] = None
    triforce_pos: Optional[Tuple[int, int]] = None
    boss_pos: Optional[Tuple[int, int]] = None

    @property
    def edges(self) -> List[Tuple[int, int, Dict]]:
        """Return list of (src, dst, data) edge tuples from the graph.

        Convenience accessor so callers can use ``len(dungeon.edges)``
        without reaching into the NetworkX object directly.
        """
        if self.graph is None:
            return []
        return list(self.graph.edges(data=True))


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
    node_to_room: Optional[Dict[int, Tuple[int, int]]] = None  # Graph node ID to room position (includes virtual nodes)
    missing_items: Optional[List[Dict]] = None  # Items that couldn't be placed (node_id, item_type, reason)
    # Optional room-level metadata (used by GUI topology tooling, precheck prune, and undo).
    rooms: Optional[Dict[Tuple[int, int], Room]] = None
    # Room coordinates (not global pixel coordinates) after compaction/remap.
    start_pos: Optional[Tuple[int, int]] = None
    triforce_pos: Optional[Tuple[int, int]] = None


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
    node_features: np.ndarray = field(default_factory=lambda: np.zeros((0, 6), dtype=np.float32))


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
        """Compatibility wrapper around extracted state-space solver core."""
        self._impl = StateSpaceGraphSolverCore(
            graph=graph,
            mode=mode,
            validation_mode_cls=ValidationMode,
            state_cls=InventoryState,
            parse_node_label_tokens_fn=parse_node_label_tokens,
            parse_edge_type_tokens_fn=parse_edge_type_tokens,
            select_primary_edge_type_fn=select_primary_edge_type,
        )
        self.graph = self._impl.graph
        self.mode = self._impl.mode
        self.key_rooms = self._impl.key_rooms
        self.item_rooms = self._impl.item_rooms
    
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
        return self._impl.can_traverse_edge(from_node=from_node, to_node=to_node, state=state)
    
    def collect_room_items(self, node: int, state: InventoryState) -> InventoryState:
        """
        Collect items when entering a room.
        
        Args:
            node: Node ID of room being entered
            state: Current inventory state
            
        Returns:
            Updated inventory state
        """
        return self._impl.collect_room_items(node=node, state=state)
    
    def solve(self, start_node: int, goal_node: int) -> Dict:
        """
        Find path from start to goal using state-space BFS.
        
        Args:
            start_node: Starting node ID
            goal_node: Goal node ID
            
        Returns:
            Dict with solvable, path, inventory_final, edge_types
        """
        return self._impl.solve(start_node=start_node, goal_node=goal_node)


# ==========================================
# VGLC PARSER (Auto-Alignment)
# ==========================================
class VGLCParser:
    """Compatibility wrapper around extracted parser implementation."""

    def __init__(self):
        self._impl = VGLCParserImpl(room_cls=Room)

    def parse(self, filepath: str) -> Dict[Tuple[int, int], Room]:
        return self._impl.parse(filepath)

    def _detect_doors(self, char_grid: np.ndarray) -> Dict[str, bool]:
        return self._impl._detect_doors(char_grid)

    def _to_semantic(self, char_grid: np.ndarray, doors: Dict[str, bool]) -> np.ndarray:
        return self._impl._to_semantic(char_grid, doors)


# ==========================================
# DOT GRAPH PARSER
# ==========================================
class DOTParser:
    """Parse DOT graph files."""

    def __init__(self):
        self._impl = DOTParserImpl()

    def parse(self, filepath: str) -> nx.DiGraph:
        return self._impl.parse(filepath)


# ==========================================
# HYBRID LAYOUT ENGINE (Spectral + SA)
# ==========================================
class HybridLayoutEngine(HybridLayoutEngineImpl):
    """Compatibility wrapper around extracted layout engine implementation."""


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
        return match_rooms_to_graph_impl(
            rooms=rooms,
            graph=graph,
            dungeon_cls=Dungeon,
            logger=logger,
            normalize_graph_fn=self._normalize_graph,
            build_room_adjacency_fn=self._build_room_adjacency,
            find_entrance_room_fn=self._find_entrance_room,
            match_rooms_to_nodes_bfs_fn=self._match_rooms_to_nodes_bfs,
            find_room_at_distance_fn=self._find_room_at_distance,
            find_farthest_dead_end_fn=self._find_farthest_dead_end,
        )
    
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
            # Skip nodes marked as start pointer (they are not physical rooms)
            graph_neighbors = [n for n in list(graph.successors(current_node)) + list(graph.predecessors(current_node))
                               if n not in visited_nodes and not graph.nodes[n].get('is_start_pointer', False)]

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

            assigned_pairs = solve_assignment_with_fallback(
                cost_matrix=cost_matrix,
                rooms_order=R,
                nodes_order=N,
                node_signature_fn=lambda node_id: self._node_signature(graph, node_id),
                logger=logger,
                failure_log_prefix='Local Hungarian assignment failed; using deterministic greedy fallback',
                max_hungarian_size=10,
            )

            # Apply assignments
            for r, n in assigned_pairs:
                if r in visited_rooms or n in visited_nodes:
                    continue
                room_to_node[r] = n
                node_to_room[n] = r
                visited_rooms.add(r)
                visited_nodes.add(n)
                if n in graph_neighbors:
                    graph_neighbors.remove(n)
                room_queue.append(r)

        # FALLBACK: Global assignment for remaining unmapped rooms/nodes (deterministic)
        unmapped_rooms = sorted([pos for pos in rooms.keys() if pos not in room_to_node])
        # Exclude start pointer nodes - they are virtual (not physical rooms)
        unmapped_nodes = sorted([n for n in graph.nodes() if n not in node_to_room
                                  and not graph.nodes[n].get('is_start_pointer', False)],
                                 key=lambda x: self._node_signature(graph, x))

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

            assigned_pairs = solve_assignment_with_fallback(
                cost_matrix=cm,
                rooms_order=R,
                nodes_order=N,
                node_signature_fn=lambda node_id: self._node_signature(graph, node_id),
                logger=logger,
                failure_log_prefix='Global Hungarian assignment failed; using deterministic greedy fallback',
                max_hungarian_size=None,
            )
            for r, n in assigned_pairs:
                room_to_node[r] = n
                node_to_room[n] = r

        # ============================================================
        # FIX 1: Handle Stair Edges (non-spatial warp connections)
        # ============================================================
        # After main BFS, match nodes connected via stair edges ('s' type)
        # to rooms containing STAIR tiles. Stair edges represent non-adjacent
        # teleport connections that BFS cannot discover via room adjacency.
        still_unmapped_nodes = [n for n in graph.nodes() if n not in node_to_room
                                and not graph.nodes[n].get('is_start_pointer', False)]
        stair_rooms = sorted([pos for pos in rooms.keys() 
                              if getattr(rooms[pos], 'has_stair', False)])
        
        for unmapped_n in list(still_unmapped_nodes):
            # Check if this node has a stair edge to any already-mapped node
            has_stair_edge = False
            _mapped_neighbor = None
            for neighbor in list(graph.successors(unmapped_n)) + list(graph.predecessors(unmapped_n)):
                edge_data = graph.get_edge_data(unmapped_n, neighbor) or graph.get_edge_data(neighbor, unmapped_n) or {}
                edge_type = edge_data.get('edge_type', '') or edge_data.get('label', '')
                if edge_type in ('stair', 's'):
                    has_stair_edge = True
                    if neighbor in node_to_room:
                        _mapped_neighbor = neighbor
                        break
            
            if has_stair_edge and stair_rooms:
                # Prefer stair room not yet used, or closest to the mapped neighbor's room
                best_stair_room = None
                for sr in stair_rooms:
                    if sr not in room_to_node:
                        best_stair_room = sr
                        break
                
                # If all stair rooms are already used, use the first one (multi-node per room)
                if best_stair_room is None and stair_rooms:
                    best_stair_room = stair_rooms[0]
                
                if best_stair_room is not None:
                    # Only add to room_to_node if not already there (avoid overwrite)
                    if best_stair_room not in room_to_node:
                        room_to_node[best_stair_room] = unmapped_n
                    node_to_room[unmapped_n] = best_stair_room
                    still_unmapped_nodes.remove(unmapped_n)
                    logger.debug("STAIR_EDGE_FIX: Matched node %d to stair room %s", unmapped_n, best_stair_room)

        # ============================================================
        # FIX 2 & 3: Multi-Node Per Room + Virtual Node Propagation
        # ============================================================
        # If there are more graph nodes than rooms, remaining unmapped nodes
        # are "virtual" nodes representing sub-areas within already-mapped rooms.
        # Propagate room assignments from their closest mapped neighbor.
        still_unmapped_nodes = [n for n in graph.nodes() if n not in node_to_room
                                and not graph.nodes[n].get('is_start_pointer', False)]
        still_unmapped_rooms = sorted([pos for pos in rooms.keys() if pos not in room_to_node])
        
        # FIRST: If there are both unmapped rooms and unmapped non-virtual nodes,
        # try to assign them 1:1 before treating any node as virtual.
        # This handles cases where BFS couldn't reach a room (e.g., behind a boss).
        if still_unmapped_rooms and still_unmapped_nodes:
            logger.debug(
                "RESIDUAL_MATCH: %d unmapped rooms %s, %d unmapped nodes %s - attempting 1:1 assignment",
                len(still_unmapped_rooms), still_unmapped_rooms,
                len(still_unmapped_nodes), still_unmapped_nodes
            )
            # Use neighbor-proximity: prefer assigning an unmapped node to a room
            # that is spatially adjacent to the room of one of the node's graph neighbors.
            for unmapped_n in list(still_unmapped_nodes):
                if not still_unmapped_rooms:
                    break
                # Find which rooms are adjacent to the rooms of this node's mapped neighbors
                best_room = None
                best_score = -999
                neighbors_in_graph = list(graph.successors(unmapped_n)) + list(graph.predecessors(unmapped_n))
                neighbor_rooms = set()
                for nb in neighbors_in_graph:
                    if nb in node_to_room:
                        neighbor_rooms.add(node_to_room[nb])
                
                for candidate_room in still_unmapped_rooms:
                    score = 0
                    # Prefer rooms adjacent to where this node's neighbors are mapped
                    for nr in neighbor_rooms:
                        if candidate_room in room_adjacency.get(nr, []):
                            score += 10
                    # Degree matching bonus
                    r_deg = sum(rooms[candidate_room].doors.values())
                    n_deg = graph.in_degree(unmapped_n) + graph.out_degree(unmapped_n)
                    score -= abs(r_deg * 2 - n_deg)
                    if score > best_score:
                        best_score = score
                        best_room = candidate_room
                
                if best_room is None:
                    best_room = still_unmapped_rooms[0]
                
                room_to_node[best_room] = unmapped_n
                node_to_room[unmapped_n] = best_room
                still_unmapped_rooms.remove(best_room)
                still_unmapped_nodes.remove(unmapped_n)
                logger.debug("RESIDUAL_MATCH: Assigned node %d to room %s (score=%d)",
                            unmapped_n, best_room, best_score)
        
        # Now handle any remaining unmapped nodes as virtual (more nodes than rooms)
        if still_unmapped_nodes:
            logger.debug("VIRTUAL_NODE_FIX: %d nodes remain unmapped, propagating from neighbors", 
                        len(still_unmapped_nodes))
            
            for unmapped_n in still_unmapped_nodes:
                # BFS from unmapped_node to find nearest node with a room mapping
                from collections import deque
                bfs_queue = deque([unmapped_n])
                bfs_visited = {unmapped_n}
                closest_mapped_neighbor = None
                
                while bfs_queue and closest_mapped_neighbor is None:
                    current = bfs_queue.popleft()
                    neighbors = list(graph.successors(current)) + list(graph.predecessors(current))
                    for nb in neighbors:
                        if nb in node_to_room:
                            closest_mapped_neighbor = nb
                            break
                        if nb not in bfs_visited:
                            bfs_visited.add(nb)
                            bfs_queue.append(nb)
                
                if closest_mapped_neighbor is not None:
                    # Share the room assignment with the closest mapped neighbor
                    shared_room = node_to_room[closest_mapped_neighbor]
                    node_to_room[unmapped_n] = shared_room
                    # Mark as virtual in node attributes for downstream processing
                    graph.nodes[unmapped_n]['is_virtual'] = True
                    graph.nodes[unmapped_n]['virtual_parent'] = closest_mapped_neighbor
                    logger.debug("VIRTUAL_NODE_FIX: Node %d shares room %s with neighbor %d (virtual)",
                                unmapped_n, shared_room, closest_mapped_neighbor)
                else:
                    # Keep truly orphan nodes virtual and unmapped.
                    # Assigning an arbitrary fallback room fabricates dataset structure
                    # and can cause spurious item placement/teleport behavior.
                    graph.nodes[unmapped_n]['is_virtual'] = True
                    graph.nodes[unmapped_n]['virtual_parent'] = None
                    graph.nodes[unmapped_n]['unmapped_virtual'] = True
                    logger.info(
                        "VIRTUAL_NODE_FIX: Node %d remains unmapped virtual (no mapped neighbor found)",
                        unmapped_n,
                    )

        try:
            # attempt local swaps before final validation
            try:
                _consistency_after = refine_mapping_with_swaps(
                    rooms=rooms,
                    room_adjacency=room_adjacency,
                    graph=graph,
                    room_to_node=room_to_node,
                    node_to_room=node_to_room,
                    validate_mapping_fn=self._validate_mapping,
                )
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                _consistency_after = None
            # Validate mapping quality and log a warning if low consistency
            consistency = self._validate_mapping(rooms, room_adjacency, graph, room_to_node)
            if consistency < 0.2:
                logger.warning('Low room-node mapping consistency: %.2f', consistency)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            logger.debug('Mapping validation failed to run', exc_info=True)

        return room_to_node, node_to_room

    def _normalize_graph(self, graph: nx.DiGraph) -> None:
        """Normalize graph labels and edge types so downstream logic can be deterministic."""
        for _, _, data in graph.edges(data=True):
            label_raw = data.get('label', '')
            label = '' if label_raw is None else str(label_raw).replace('\n', ',').strip()
            if label:
                label = ",".join([p.strip() for p in label.split(',') if p.strip()])
            data['label'] = label

            # Normalize to composite constraints + representative edge_type.
            edge_type_raw = data.get('edge_type') or ''
            constraints = parse_edge_type_tokens(label=label, edge_type=edge_type_raw)
            data['edge_constraints'] = constraints
            data['edge_type'] = select_primary_edge_type(constraints)

        for _, data in graph.nodes(data=True):
            label = (data.get('label') or data.get('name') or '')
            s = normalize_node_label(str(label))
            data['label'] = s
            parts = parse_node_label_tokens(s)
            parts_l = {p.lower() for p in parts}
            # Canonical flags - use parts-based detection to handle
            # composite labels like "e,k" correctly (avoid matching 
            # 'b' in 'boss' against the substring)
            has_start_pointer_token = 's' in parts
            has_start_room_token = ('S' in parts) or ('start' in parts_l)
            if has_start_pointer_token or has_start_room_token or data.get('is_start'):
                data['is_start'] = True
                # Preserve is_start_pointer if already set by DOTParser
                if data.get('is_start_pointer') is None:
                    # Check if this is a pure start pointer
                    data['is_start_pointer'] = all(
                        p in ('s', '') for p in parts
                    )
                elif has_start_room_token:
                    data['is_start_pointer'] = False
            if 't' in parts or 'triforce' in parts_l or data.get('is_triforce'):
                data['is_triforce'] = True
            if 'b' in parts or 'boss' in parts_l or data.get('is_boss'):
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
        seed_from_special_nodes(
            rooms=rooms,
            graph=graph,
            existing_room_to_node=existing_room_to_node,
            existing_node_to_room=existing_node_to_room,
            proposed_room_to_node=proposed_room_to_node,
            proposed_node_to_room=proposed_node_to_room,
            confidences=confidences,
        )

        # 2) BFS propagation from existing anchors using existing _match_rooms_to_nodes_bfs
        propagate_from_anchors(
            rooms=rooms,
            graph=graph,
            existing_room_to_node=existing_room_to_node,
            existing_node_to_room=existing_node_to_room,
            proposed_room_to_node=proposed_room_to_node,
            proposed_node_to_room=proposed_node_to_room,
            confidences=confidences,
            match_rooms_to_nodes_bfs_fn=self._match_rooms_to_nodes_bfs,
            build_room_adjacency_fn=self._build_room_adjacency,
        )

        # Refresh unmatched lists after BFS pass
        unmatched_nodes = [n for n in unmatched_nodes if n not in proposed_node_to_room]
        unmatched_rooms = [r for r in unmatched_rooms if r not in proposed_room_to_node]

        # 3) Label-based hints (robust regex)
        unmatched_nodes, unmatched_rooms = apply_label_hints(
            unmatched_nodes=unmatched_nodes,
            unmatched_rooms=unmatched_rooms,
            graph=graph,
            proposed_room_to_node=proposed_room_to_node,
            proposed_node_to_room=proposed_node_to_room,
            confidences=confidences,
        )

        # 4) Component-aware building (map graph comps -> room comps via known anchors)
        room_adj = self._build_room_adjacency(rooms)
        graph_comp_of, room_comp_of, comp_room_candidates = build_component_context(
            graph=graph,
            rooms=rooms,
            room_adjacency=room_adj,
            existing_room_to_node=existing_room_to_node,
        )

        # Spatial centers (normalized) for rooms using room_positions if available
        centers = compute_normalized_room_centers(
            unmatched_rooms=unmatched_rooms,
            room_positions=room_positions,
        )

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
            except (AttributeError, RuntimeError, ValueError, TypeError) as e:
                logger.exception("seeded_spectral_match failed during infer_missing_mappings: %s", e) 

        # 5) Build score matrix for remaining unmatched nodes/rooms
        score_matrix = build_score_matrix(
            unmatched_nodes=unmatched_nodes,
            unmatched_rooms=unmatched_rooms,
            graph=graph,
            rooms=rooms,
            centers=centers,
            graph_comp_of=graph_comp_of,
            room_comp_of=room_comp_of,
            comp_room_candidates=comp_room_candidates,
        )

        # Try to use global assignment (Hungarian) for best overall matching
        assigned_pairs = assign_pairs_from_scores(
            unmatched_nodes=unmatched_nodes,
            unmatched_rooms=unmatched_rooms,
            score_matrix=score_matrix,
            logger=logger,
        )

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
        except (AttributeError, RuntimeError, ValueError, TypeError) as e:
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
        return find_room_at_distance(
            rooms=rooms,
            room_adjacency=room_adjacency,
            start_pos=start_pos,
            target_distance=target_distance,
        )
    
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
    
    def _find_entrance_room(self, rooms: Dict[Tuple[int, int], Room]) -> Optional[Tuple[int, int]]:
        """
        Find the dungeon entrance room.
        
        The entrance is the room with a door leading OUTSIDE the dungeon 
        (i.e., to an empty grid slot). In Zelda dungeons, this is where 
        Link enters from the overworld.
        
        This is CRITICAL for correct graph matching - the entrance room
        should be matched to the graph's START node.
        
        Returns:
            Position of the entrance room, or None if not found.
        """
        return find_entrance_room(rooms=rooms, logger=logger)
    
    def _find_farthest_dead_end(self, rooms: Dict[Tuple[int, int], Room],
                                 start_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Find the dead-end room (1 door) farthest from start."""
        return find_farthest_dead_end(
            rooms=rooms,
            start_pos=start_pos,
            room_adjacency_fn=self._build_room_adjacency,
        )
    
    def _build_room_adjacency(self, rooms: Dict[Tuple[int, int], Room]) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        """Build adjacency list from room door connections."""
        return build_room_adjacency(rooms=rooms)

    def _room_signature(self, room: Room) -> Tuple[int, int, int, int]:
        """Return simple door signature (N,S,E,W) as ints and door count.

        Useful for comparing structural compatibility of rooms.
        """
        return room_signature(room=room)

    def _edge_consistency_score(self, n2r: Dict[int, Tuple[int, int]], graph: nx.DiGraph, room_adj: Dict[Tuple[int, int], List[Tuple[int, int]]]) -> int:
        """Compute how many graph edges are consistent with adjacent room pairs.

        Score is number of directed edges (u->v) where assigned rooms r_u and r_v are adjacent in room_adj.
        """
        return edge_consistency_score(n2r=n2r, graph=graph, room_adj=room_adj)

    def _local_refine_assignments(self, n2r: Dict[int, Tuple[int, int]], graph: nx.DiGraph, room_adj: Dict[Tuple[int, int], List[Tuple[int, int]]], score_matrix: Dict[Tuple[int, Tuple[int, int]], float], iterations: int = 100) -> Optional[Dict[int, Tuple[int, int]]]:
        """Try local pairwise swaps to increase combined score (assignment score + edge consistency).

        Deterministic improvement pass: consider all pairs and perform swap if it improves total objective. Repeat until no improvement or max iterations.
        """
        return local_refine_assignments(
            n2r=n2r,
            graph=graph,
            room_adj=room_adj,
            score_matrix=score_matrix,
            iterations=iterations,
        )

    def seeded_spectral_match(self, rooms: Dict[Tuple[int, int], Room], graph: nx.DiGraph, room_positions: Optional[Dict[Tuple[int, int], Tuple[int, int]]] = None, seeds: Optional[Dict[Tuple[int, int], int]] = None, k_dim: int = 8) -> Tuple[Dict[int, Tuple[int, int]], Dict[int, float]]:
        """Perform seeded spectral matching between graph nodes and rooms.

        Steps:
        - Compute spectral embeddings for graph and room adjacency
        - Use seeded correspondences to compute an orthogonal Procrustes alignment
        - Match remaining nodes by nearest neighbor in embedding space and refine with Hungarian

        Returns: (node_to_room_proposal, confidences)
        """
        return seeded_spectral_match_impl(
            rooms=rooms,
            graph=graph,
            build_room_adjacency_fn=self._build_room_adjacency,
            logger=logger,
            room_positions=room_positions,
            seeds=seeds,
            k_dim=k_dim,
        )




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
                room_to_node={},
                node_to_room={},
                rooms={},
                start_pos=None,
                triforce_pos=None,
            )
        
        # For compact mode: remap positions to eliminate gaps
        if compact:
            rooms_remapped, pos_remap = self._compact_rooms(dungeon.rooms)
        else:
            rooms_remapped = dungeon.rooms
            pos_remap = {pos: pos for pos in dungeon.rooms.keys()}

        stitched_layout = build_stitched_room_layout_from_rooms(
            rooms_remapped=rooms_remapped,
            room_height=ROOM_HEIGHT,
            room_width=ROOM_WIDTH,
        )
        global_grid = stitched_layout.dungeon_grid
        room_positions = stitched_layout.room_offsets
        
        # Connect doors by punching through walls
        self._connect_doors(global_grid, rooms_remapped)
        
        room_to_node, node_to_room = build_room_node_mappings(
            dungeon_rooms=dungeon.rooms,
            pos_remap=pos_remap,
            graph=dungeon.graph,
        )
        if dungeon.graph is not None:
            for node_id, room_pos in node_to_room.items():
                if node_id in room_to_node:
                    continue
                node_data = dungeon.graph.nodes.get(node_id, {})
                parent = node_data.get('virtual_parent')
                if parent is not None and parent in node_to_room:
                    logger.debug(
                        "STITCH: Virtual node %d shares room %s with parent %d",
                        node_id,
                        room_pos,
                        parent,
                    )
        
        # CRITICAL FIX: Place items (keys, items) based on graph node labels
        # VGLC data does NOT contain item placement - only the graph specifies this
        missing_items = self._place_items_from_graph(global_grid, dungeon.graph, room_to_node, room_positions)
        
        # Log summary of missing items for solver awareness (INFO level, not WARNING)
        # This is expected behavior for VGLC data with graph-room count mismatches
        if missing_items:
            logger.info(
                "STITCH_SUMMARY: %d items from graph nodes could not be placed (solver will proceed without them)",
                len(missing_items)
            )
        
        # CRITICAL FIX: Place entities (enemies, boss, triforce) based on graph node labels
        # VGLC rooms may not have ENEMY/BOSS tiles if the raw text files don't include monsters
        self._place_entities_from_graph(global_grid, dungeon.graph, room_to_node, room_positions)
        
        # CRITICAL FIX: Convert door types based on graph edge labels
        # VGLC data only has generic 'D' doors - graph specifies k/b/l requirements
        self._apply_door_types_from_graph(global_grid, dungeon.graph, room_to_node, room_positions)
        
        # Remap start/triforce positions
        start_pos_remapped = pos_remap.get(dungeon.start_pos) if dungeon.start_pos else None
        triforce_pos_remapped = pos_remap.get(dungeon.triforce_pos) if dungeon.triforce_pos else None

        start_global, triforce_global = place_special_markers(
            global_grid=global_grid,
            room_positions=room_positions,
            start_pos_remapped=start_pos_remapped,
            triforce_pos_remapped=triforce_pos_remapped,
            find_floor_near_door_fn=self._find_floor_near_door,
            semantic_palette=SEMANTIC_PALETTE,
            room_height=ROOM_HEIGHT,
            room_width=ROOM_WIDTH,
        )

        (
            room_positions_out,
            room_to_node_out,
            node_to_room_out,
            start_pos_out,
            triforce_pos_out,
        ) = project_output_metadata(
            pos_remap=pos_remap,
            room_positions=room_positions,
            room_to_node=room_to_node,
            node_to_room=node_to_room,
            dungeon_start_pos=dungeon.start_pos,
            dungeon_triforce_pos=dungeon.triforce_pos,
            start_pos_remapped=start_pos_remapped,
            triforce_pos_remapped=triforce_pos_remapped,
        )
        
        return StitchedDungeon(
            dungeon_id=dungeon.dungeon_id,
            global_grid=global_grid,
            room_positions=room_positions_out,
            start_global=start_global,
            triforce_global=triforce_global,
            graph=dungeon.graph,
            room_to_node=room_to_node_out,
            node_to_room=node_to_room_out,
            missing_items=missing_items if missing_items else None,
            rooms=dungeon.rooms,
            start_pos=start_pos_out,
            triforce_pos=triforce_pos_out,
        )
    
    def _compact_rooms(self, rooms: Dict[Tuple[int, int], Room]) -> Tuple[Dict[Tuple[int, int], Room], Dict[Tuple[int, int], Tuple[int, int]]]:
        """
        Remap room positions to eliminate empty rows/columns.
        
        Returns:
            (remapped_rooms, original_to_new_pos_map)
        """
        return compact_rooms(
            rooms=rooms,
            clone_room_with_position_fn=lambda room, new_pos: Room(
                position=new_pos,
                char_grid=room.char_grid,
                semantic_grid=room.semantic_grid,
                doors=room.doors,
                has_stair=room.has_stair,
                has_triforce=room.has_triforce,
                has_boss=room.has_boss,
                is_start=room.is_start,
                graph_node_id=room.graph_node_id,
                node_label=room.node_label,
            ),
        )
    
    def _connect_doors(self, grid: np.ndarray, rooms: Dict[Tuple[int, int], Room]):
        """
        Punch through walls to connect door pairs.
        
        For each pair of adjacent rooms with matching doors,
        ensure BOTH sides of the boundary are passable.
        Also ensures internal room connectivity by clearing floor paths.
        """
        connect_doors(
            grid=grid,
            rooms=rooms,
            semantic_palette=SEMANTIC_PALETTE,
            room_height=ROOM_HEIGHT,
            room_width=ROOM_WIDTH,
        )
    
    def _ensure_room_connectivity(self, grid: np.ndarray, rooms: Dict[Tuple[int, int], Room]):
        """Ensure each room has connected floor tiles from center to all doors."""
        ensure_room_connectivity(
            grid=grid,
            rooms=rooms,
            semantic_palette=SEMANTIC_PALETTE,
            room_height=ROOM_HEIGHT,
            room_width=ROOM_WIDTH,
        )
    
    def _find_floor_near_door(self, grid: np.ndarray, 
                               r_off: int, c_off: int) -> Tuple[int, int]:
        """Find a walkable tile in the room for starting position.
        
        IMPORTANT: Prioritizes positions that are actually reachable from doors,
        not just any walkable tile. This handles rooms where water/elements 
        surround the center (e.g., D9-2).
        """
        return find_floor_near_door(
            grid=grid,
            r_off=r_off,
            c_off=c_off,
            semantic_palette=SEMANTIC_PALETTE,
            room_height=ROOM_HEIGHT,
            room_width=ROOM_WIDTH,
        )
    
    def _place_items_from_graph(self, grid: np.ndarray, graph: nx.DiGraph,
                                 room_to_node: Dict[Tuple[int, int], int],
                                 room_positions: Dict[Tuple[int, int], Tuple[int, int]]) -> List[Dict]:
        """
        Place items (keys, items) in the grid based on graph node labels.
        
        CRITICAL FIX: VGLC text files do NOT contain item placement data.
        The DOT graph specifies which rooms should have items via node labels:
        - 'k' = small key (KEY_SMALL)
        - 'K' = boss key (KEY_BOSS)
        - 'I' = major item/key item (KEY_ITEM) - enables crossing water, etc.
        - 'i' = minor item (ITEM_MINOR)
        
        Items are placed at a walkable floor position near the room center.
        If no valid floor exists, tries alternate positions (adjacent tiles, room corners).
        
        For unmapped nodes, attempts to place items in the nearest accessible room.
        
        Returns:
            List of dicts describing items that couldn't be placed (for solver awareness)
        """
        return place_items_from_graph(
            grid=grid,
            graph=graph,
            room_to_node=room_to_node,
            room_positions=room_positions,
            parse_node_label_tokens_fn=parse_node_label_tokens,
            semantic_palette=SEMANTIC_PALETTE,
            room_height=ROOM_HEIGHT,
            room_width=ROOM_WIDTH,
            logger=logger,
        )

    def _place_entities_from_graph(
        self,
        grid: np.ndarray,
        graph,
        room_to_node: Dict[Tuple[int, int], int],
        room_positions: Dict[Tuple[int, int], Tuple[int, int]]
    ):
        """
        Place entity tiles (ENEMY, BOSS) based on graph node labels.
        
        VGLC rooms may not have ENEMY/BOSS tiles if the raw text files don't
        include monsters. The graph specifies which rooms should have entities:
        - 'e' = enemy room -> at least 1 ENEMY tile
        - 'b' = boss room -> at least 1 BOSS tile
        
        Only places entities if the room doesn't already have them.
        """
        place_entities_from_graph(
            grid=grid,
            graph=graph,
            room_to_node=room_to_node,
            room_positions=room_positions,
            parse_node_label_tokens_fn=parse_node_label_tokens,
            semantic_palette=SEMANTIC_PALETTE,
            room_height=ROOM_HEIGHT,
            room_width=ROOM_WIDTH,
            logger=logger,
        )

    def _apply_door_types_from_graph(
        self, 
        grid: np.ndarray, 
        graph: nx.DiGraph, 
        room_to_node: Dict[Tuple[int, int], int],
        room_positions: Dict[Tuple[int, int], Tuple[int, int]]
    ):
        """
        Convert generic DOOR_OPEN/FLOOR tiles at room boundaries to specific door types
        based on graph edge labels.
        
        Edge labels:
        - 'k' = key-locked door (DOOR_LOCKED)
        - 'b' = bombable wall (DOOR_BOMB)
        - 'l' = soft/one-way door (DOOR_SOFT)
        - 'K' = boss key door (DOOR_BOSS)
        - empty = normal open door (DOOR_OPEN)
        
        IMPORTANT: Only the CENTER tile of each door is marked with the special type.
        This matches Zelda semantics where one key opens the entire door.
        """
        apply_door_types_from_graph(
            grid=grid,
            graph=graph,
            room_to_node=room_to_node,
            room_positions=room_positions,
            semantic_palette=SEMANTIC_PALETTE,
            room_height=ROOM_HEIGHT,
            room_width=ROOM_WIDTH,
            logger=logger,
        )
    
    def _find_boundary_doors(
        self,
        grid: np.ndarray,
        from_offset: Tuple[int, int],
        to_offset: Tuple[int, int],
        from_room: Tuple[int, int],
        to_room: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """
        Find door tiles at the boundary between two adjacent rooms.
        """
        return find_boundary_doors(
            grid=grid,
            from_offset=from_offset,
            to_offset=to_offset,
            from_room=from_room,
            to_room=to_room,
            room_height=ROOM_HEIGHT,
            room_width=ROOM_WIDTH,
        )


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
        return precheck_dungeon_impl(
            dungeon=dungeon,
            parse_edge_type_tokens_fn=parse_edge_type_tokens,
            parse_node_label_tokens_fn=parse_node_label_tokens,
            semantic_palette=SEMANTIC_PALETTE,
            logger=logger,
        )

    @staticmethod
    def prune_dead_ends(rooms: Dict[Tuple[int, int], Room], preserve: Optional[Set[Tuple[int, int]]] = None) -> Tuple[Dict[Tuple[int, int], Room], List[Tuple[int, int]]]:
        """Iteratively remove leaf rooms (degree==1) that do not contain keys/triforce/start/boss.

        Returns (pruned_rooms, removed_positions)
        """
        return prune_dead_ends_impl(
            rooms=rooms,
            parse_node_label_tokens_fn=parse_node_label_tokens,
            semantic_palette=SEMANTIC_PALETTE,
            preserve=preserve,
        )

    
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

    def _log_virtual_node_report(self, dungeon: Dungeon, context: str = "load") -> None:
        """Compatibility wrapper around extracted reporting helper."""
        log_virtual_node_report(dungeon, context=context, logger=logger)
    
    def validate_dungeon(self, dungeon: Dungeon, stitched: Optional[StitchedDungeon] = None) -> Dict:
        """
        Validate dungeon integrity and return detailed warnings/errors.
        
        Checks:
        - All graph nodes have corresponding rooms
        - All items from graph are placed in grid (if stitched provided)
        - Start and goal positions are valid and reachable
        - Room-to-node mapping completeness
        
        Args:
            dungeon: Dungeon object to validate
            stitched: Optional StitchedDungeon for grid-level checks
            
        Returns:
            Dict with:
            - 'valid': bool (True if no errors, warnings may still exist)
            - 'errors': List[str] (critical issues)
            - 'warnings': List[str] (non-critical issues)
            - 'stats': Dict with counts and statistics
        """
        return validate_dungeon_impl(
            dungeon=dungeon,
            stitched=stitched,
            parse_node_label_tokens_fn=parse_node_label_tokens,
            semantic_palette=SEMANTIC_PALETTE,
            solver_cls=DungeonSolver,
        )

    def load_dungeon(self, dungeon_num: int, variant: int = 1) -> Dungeon:
        """
        Load a dungeon by number.
        
        Args:
            dungeon_num: Dungeon number (1-9)
            variant: Variant number (1 for Quest 1, 2 for Quest 2)
            
        Returns:
            Dungeon object
        """
        return load_dungeon_impl(
            data_root=self.data_root,
            vglc_parser=self.vglc_parser,
            dot_parser=self.dot_parser,
            matcher=self.matcher,
            log_virtual_node_report_fn=lambda d, c: self._log_virtual_node_report(d, context=c),
            dungeon_num=dungeon_num,
            variant=variant,
        )
    
    def stitch_dungeon(self, dungeon: Dungeon) -> StitchedDungeon:
        """
        Stitch dungeon into global grid.
        
        Args:
            dungeon: Dungeon to stitch
            
        Returns:
            StitchedDungeon
        """
        return self.stitcher.stitch(dungeon)

    def layout_from_graph(self, dungeon_num: int, variant: int = 1,
                          **sa_kwargs) -> Dict[int, Tuple[int, int]]:
        """
        Compute a Spectral + SA grid layout from the DOT graph alone.

        This is useful when no text file is available (procedural graphs)
        or for evaluating layout quality independently.

        Returns
        -------
        positions : dict[int, (row, col)]
            Physical node -> grid cell.  The ``s`` start-pointer is excluded.
        """
        return layout_from_graph_impl(
            data_root=self.data_root,
            dot_parser=self.dot_parser,
            hybrid_layout_engine_cls=HybridLayoutEngine,
            dungeon_num=dungeon_num,
            variant=variant,
            **sa_kwargs,
        )
    
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
        results = process_all_dungeons_impl(
            data_root=self.data_root,
            load_dungeon_fn=lambda dungeon_num, quest_num: self.load_dungeon(dungeon_num, variant=quest_num),
            logger=logger,
            processed_dir=processed_dir,
            graph_dir=graph_dir,
        )
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
        return save_processed_data_impl(
            data_root=self.data_root,
            processed_dungeons=getattr(self, 'processed_dungeons', {}),
            output_path=output_path,
            logger=logger,
        )


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
        return solve_dungeon_impl(
            stitched=stitched,
            mode=mode,
            solve_with_state_space_fn=self._solve_with_state_space,
            solve_with_grid_fn=self._solve_with_grid,
        )
    
    def _solve_with_state_space(self, stitched: StitchedDungeon, mode: str) -> Dict:
        """
        Check solvability using state-space search with inventory tracking.
        This properly handles keys, bombs, soft-locks, and stairs.
        """
        return solve_with_state_space_impl(
            stitched=stitched,
            mode=mode,
            room_height=ROOM_HEIGHT,
            room_width=ROOM_WIDTH,
            state_space_solver_cls=StateSpaceGraphSolver,
        )
    
    def _solve_with_graph(self, stitched: StitchedDungeon) -> Dict:
        """Legacy: Check solvability using simple graph connectivity (ignores edge types)."""
        return solve_with_graph_impl(
            stitched=stitched,
            room_height=ROOM_HEIGHT,
            room_width=ROOM_WIDTH,
        )
    
    def _solve_with_grid(self, stitched: StitchedDungeon) -> Dict:
        """Fallback: check solvability using grid BFS (no stairs)."""
        return solve_with_grid_impl(
            stitched=stitched,
            walkable_tiles=self.WALKABLE,
            triforce_tile=SEMANTIC_PALETTE['TRIFORCE'],
            room_height=ROOM_HEIGHT,
            room_width=ROOM_WIDTH,
        )


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
    return test_all_dungeons_impl(
        data_root=data_root,
        include_variants=include_variants,
        adapter_cls=ZeldaDungeonAdapter,
        solver_cls=DungeonSolver,
        logger=logger,
    )


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
    return visualize_semantic_grid_impl(
        grid=grid,
        show_legend=show_legend,
        semantic_palette=SEMANTIC_PALETTE,
    )


def convert_room_to_roomdata(room: Room) -> RoomData:
    """Convert Room dataclass to RoomData for adapter.py compatibility."""
    return convert_room_to_roomdata_impl(
        room=room,
        roomdata_cls=RoomData,
    )


def convert_dungeon_to_dungeondata(dungeon: Dungeon) -> DungeonData:
    """Convert Dungeon to DungeonData for adapter.py compatibility."""
    return convert_dungeon_to_dungeondata_impl(
        dungeon=dungeon,
        convert_room_to_roomdata_fn=lambda room: convert_room_to_roomdata_impl(
            room=room,
            roomdata_cls=RoomData,
        ),
        ml_feature_extractor_cls=MLFeatureExtractor,
        dungeondata_cls=DungeonData,
    )


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    default_data_root = Path(__file__).resolve().parents[2] / "Data" / "The Legend of Zelda"
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(default_data_root))
    p.add_argument("--no-variants", action="store_true", help="Only run variant 1")
    args = p.parse_args()
    logger.info("Testing dungeons at %s", args.data_root)
    test_all_dungeons(args.data_root, include_variants=not args.no_variants)
