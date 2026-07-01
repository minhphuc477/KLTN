"""Mission graph data structures and tensor adapter hooks."""

from __future__ import annotations

import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any

logger = logging.getLogger(__name__)

def _require_torch_adapters() -> Any:
    """Return optional torch adapters or raise a targeted dependency error."""
    try:
        from src.generation import grammar_torch_adapters
    except ImportError as exc:
        if getattr(exc, "name", None) == "torch":
            raise RuntimeError(
                "PyTorch is required for MissionGraph tensor export. Install torch "
                "or use the symbolic grammar APIs that return Python data structures."
            ) from exc
        raise
    return grammar_torch_adapters


# ============================================================================
# LAYOUT CONSTANTS
# ============================================================================

# Layout algorithm parameters for node positioning
LAYOUT_LAYER_SPACING = 2      # Spacing between graph layers in grid units
LAYOUT_OFFSET_SPACING = 2     # Spacing between nodes in same layer
LAYOUT_BASE_OFFSET = 5        # Starting y-offset for node placement
LAYOUT_HUB_RADIUS = 3         # Radius for hub spoke placement
LAYOUT_HUB_BRANCH_SPACING = 2 # Spacing multiplier for hub branch extensions


# ============================================================================
# NODE TYPES
# ============================================================================

class NodeType(Enum):
    """Mission graph node types."""
    START = auto()
    GOAL = auto()
    KEY = auto()
    LOCK = auto()
    ENEMY = auto()
    PUZZLE = auto()
    ITEM = auto()
    EMPTY = auto()  # Connector room
    SWITCH = auto()  # State-changing switch (Thesis Upgrade #2)
    BIG_KEY = auto()  # Boss key (Thesis Upgrade #3)
    BOSS_DOOR = auto()  # Final barrier before goal (Thesis Upgrade #3)
    BOSS = auto()  # Boss encounter room
    STAIRS_UP = auto()  # Stairs to upper floor
    STAIRS_DOWN = auto()  # Stairs to lower floor
    SECRET = auto()  # Secret/hidden room
    TOKEN = auto()  # Collection token (for tri-force patterns)
    ARENA = auto()  # Combat arena room (alternative to flag)
    TREASURE = auto()  # Treasure/reward room
    PROTECTION_ITEM = auto()  # Protection items (fire tunic, etc.)
    # Wave 3: Pedagogical & Quality Control
    MINI_BOSS = auto()  # Mini-boss encounter guarding items
    SCENIC = auto()  # Empty scenic/rest room (pacing breaker)
    RESOURCE_FARM = auto()  # Spawns consumable resources
    TUTORIAL_PUZZLE = auto()  # Safe puzzle teaching mechanic
    COMBAT_PUZZLE = auto()  # Moderate puzzle with enemies
    COMPLEX_PUZZLE = auto()  # Hard puzzle combining mechanics


class EdgeType(Enum):
    """Mission graph edge types."""
    PATH = auto()       # Normal path
    LOCKED = auto()     # Requires key
    ONE_WAY = auto()    # One-directional
    HIDDEN = auto()     # Secret passage
    SHORTCUT = auto()   # Shortcut/loop (Thesis Upgrade #1)
    ON_OFF_GATE = auto()  # Switch-controlled (Thesis Upgrade #2)
    BOSS_LOCKED = auto()  # Boss door requiring big key (Thesis Upgrade #3)
    ITEM_GATE = auto()  # Requires specific item (BOMB, HOOKSHOT, etc.)
    STATE_BLOCK = auto()  # Blocked by global state (alias for switch-type mechanics)
    WARP = auto()  # Teleportation/warp connection
    STAIRS = auto()  # Vertical connection between floors
    VISUAL_LINK = auto()  # Visual connection (window, non-traversable)
    SHUTTER = auto()  # One-way in, conditional out (arena doors)
    HAZARD = auto()  # Risky path (lava, spikes) with damage
    MULTI_LOCK = auto()  # Requires multiple tokens/keys


# ============================================================================
# GRAPH DATA STRUCTURES
# ============================================================================

@dataclass
class MissionNode:
    """Node in the mission graph."""
    id: int
    node_type: NodeType
    position: Tuple[int, int, int] = (0, 0, 0)  # (row, col, floor/z) in dungeon layout
    
    # Key-lock binding
    key_id: Optional[int] = None  # For LOCK: which key opens this
    
    # Metadata
    difficulty: float = 0.5
    required_item: Optional[str] = None  # For ITEM_GATE: specific item required (e.g., "BOMB", "HOOKSHOT")
    item_type: Optional[str] = None  # For ITEM nodes: what item this provides
    switch_id: Optional[int] = None  # For SWITCH/STATE_BLOCK: which switch controls this
    is_hub: bool = False  # Marks this node as a central hub
    is_secret: bool = False  # Marks this as a secret/hidden room
    
    # Advanced rule extensions
    room_size: Tuple[int, int] = (1, 1)  # For big rooms: (width, height)
    sector_id: int = 0  # Thematic zone identifier
    sector_theme: Optional[str] = None  # Sector theme (FIRE, WATER, ICE, etc.)
    virtual_layer: int = 0  # Virtual layer (balcony, basement) at same x,y
    is_arena: bool = False  # Combat arena with shutters
    is_big_room: bool = False  # Merged into macro room
    token_id: Optional[str] = None  # For TOKEN nodes: unique token identifier
    
    # Wave 3: Pedagogical patterns
    difficulty_rating: str = "MODERATE"  # SAFE, MODERATE, HARD, EXTREME
    is_sanctuary: bool = False  # Pacing breaker (safe rest area)
    drops_resource: Optional[str] = None  # Resource type (BOMBS, ARROWS, HEARTS)
    is_tutorial: bool = False  # Tutorial/teaching room
    is_mini_boss: bool = False  # Mini-boss flag
    tension_value: float = 0.5  # 0=calm, 1=intense (for pacing)
    enemy_count_hint: int = 0  # Estimated number of enemies for room-level spawning
    key_count_hint: int = 0  # Estimated number of key items in this room
    
    def to_feature_vector(self) -> List[float]:
        """Convert to feature vector for GNN."""
        # One-hot encode node type
        type_vec = [0.0] * len(NodeType)
        type_vec[self.node_type.value - 1] = 1.0
        
        # Position encoding (3D)
        pos_vec = [
            self.position[0] / 10.0,  # row
            self.position[1] / 10.0,  # col
            self.position[2] / 5.0 if len(self.position) > 2 else 0.0  # floor
        ]
        
        # Additional features
        extra = [
            self.difficulty,
            1.0 if self.key_id is not None else 0.0,
            1.0 if self.required_item is not None else 0.0,
            1.0 if self.is_hub else 0.0,
            1.0 if self.is_secret else 0.0,
            # Advanced features
            self.room_size[0] / 2.0,  # Normalized width
            self.room_size[1] / 2.0,  # Normalized height
            self.sector_id / 10.0,  # Normalized sector ID
            self.virtual_layer / 3.0,  # Normalized layer
            1.0 if self.is_arena else 0.0,
            1.0 if self.is_big_room else 0.0,
        ]
        
        return type_vec + pos_vec + extra


@dataclass
class MissionEdge:
    """Edge in the mission graph."""
    source: int
    target: int
    edge_type: EdgeType = EdgeType.PATH
    key_required: Optional[int] = None  # Key ID if LOCKED
    item_required: Optional[str] = None  # Item name if ITEM_GATE (e.g., "BOMB")
    switch_id: Optional[int] = None  # Switch ID if STATE_BLOCK/ON_OFF_GATE
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional edge properties
    
    # Advanced rule extensions
    requires_key_count: int = 0  # Fungible keys (inventory-based)
    token_count: int = 0  # Number of tokens required (for collection challenges)
    token_id: Optional[str] = None  # Specific token ID if applicable
    is_window: bool = False  # Visual link (non-traversable)
    hazard_damage: int = 0  # Damage amount for hazard edges
    protection_item_id: Optional[str] = None  # Item that protects from hazard
    preferred_direction: Optional[str] = None  # "forward" or "backward" for ONE_WAY edges
    
    # Wave 3: Quality control patterns
    battery_id: Optional[int] = None  # Multi-switch battery identifier
    switches_required: List[int] = field(default_factory=list)  # Switch IDs for battery pattern
    path_savings: int = 0  # Steps saved by shortcut (metadata)


@dataclass
class MissionGraph:
    """Complete mission graph for a dungeon."""
    BIDIRECTIONAL_EDGE_TYPES: ClassVar[Set[EdgeType]] = {
        EdgeType.PATH,
        EdgeType.LOCKED,
        EdgeType.ITEM_GATE,
        EdgeType.BOSS_LOCKED,
        EdgeType.SHORTCUT,
        EdgeType.WARP,
        EdgeType.STAIRS,
        EdgeType.HIDDEN,
    }
    NON_TRAVERSABLE_EDGE_TYPES: ClassVar[Set[EdgeType]] = {
        EdgeType.VISUAL_LINK,
    }

    nodes: Dict[int, MissionNode] = field(default_factory=dict)
    edges: List[MissionEdge] = field(default_factory=list)
    generation_stats: Dict[str, Any] = field(default_factory=dict)
    
    # Quick lookup structures
    _adjacency: Dict[int, List[int]] = field(default_factory=lambda: defaultdict(list))
    _key_to_lock: Dict[int, int] = field(default_factory=dict)  # key_id -> lock_node_id

    def __post_init__(self) -> None:
        self._ensure_generation_stats_defaults()

    def _ensure_generation_stats_defaults(self) -> None:
        defaults: Dict[str, Any] = {
            "lock_key_repairs": 0,
            "progression_repairs": 0,
            "wave3_repairs": 0,
            "repair_rounds": 0,
            "total_repairs": 0,
            "repair_applied": False,
        }
        for key, value in defaults.items():
            self.generation_stats.setdefault(key, value)

    def ensure_generation_stats_defaults(self) -> None:
        """Public wrapper to initialize generation-stats keys."""
        self._ensure_generation_stats_defaults()

    def record_repair(self, repair_key: str, amount: int = 1) -> None:
        """
        Record a repair event for downstream benchmarking/analysis.
        """
        self._ensure_generation_stats_defaults()
        delta = int(max(0, amount))
        if delta <= 0:
            return
        self.generation_stats[repair_key] = int(self.generation_stats.get(repair_key, 0)) + delta
        self.generation_stats["total_repairs"] = int(self.generation_stats.get("total_repairs", 0)) + delta
        self.generation_stats["repair_applied"] = True
    
    def _normalize_node_resource_hints(self, node: MissionNode) -> None:
        """
        Fill count hints from node semantics when explicit hints are missing.

        Hints stay conservative and are used by downstream room/entity generation.
        Enemy defaults are tuned to VGLC Zelda stats (most encounters are low-count,
        with occasional arena spikes).
        """
        enemy_hint = int(max(0, getattr(node, "enemy_count_hint", 0) or 0))
        key_hint = int(max(0, getattr(node, "key_count_hint", 0) or 0))
        diff = float(getattr(node, "difficulty", 0.5) or 0.5)
        diff = max(0.0, min(1.0, diff))

        enemy_node_types = {
            NodeType.ENEMY,
            NodeType.BOSS,
            NodeType.MINI_BOSS,
            NodeType.ARENA,
            NodeType.COMBAT_PUZZLE,
        }
        key_node_types = {
            NodeType.KEY,
            NodeType.BIG_KEY,
        }

        if enemy_hint <= 0 and node.node_type in enemy_node_types:
            # VGLC-aligned defaults:
            # - regular combat rooms are usually low-intensity (1-2 enemies)
            # - arena rooms can spike to ~4
            # - boss encounters are usually 1-2 major enemies
            if node.node_type in {NodeType.BOSS, NodeType.MINI_BOSS}:
                enemy_hint = 1 + int(diff >= 0.75)
            elif node.node_type == NodeType.ARENA:
                enemy_hint = 2 + int(diff >= 0.50) + int(diff >= 0.85)
            elif node.node_type == NodeType.COMBAT_PUZZLE:
                enemy_hint = 1 + int(diff >= 0.70)
            else:
                enemy_hint = 1 + int(diff >= 0.60)

        if key_hint <= 0 and node.node_type in key_node_types:
            # Key progression is modeled as one token per key node by default.
            key_hint = 1

        # Keep hints bounded for stable downstream spawning/evaluation.
        node.enemy_count_hint = int(max(0, min(12, enemy_hint)))
        node.key_count_hint = int(max(0, min(4, key_hint)))

    def add_node(self, node: MissionNode) -> None:
        """Add a node to the graph."""
        self._normalize_node_resource_hints(node)
        self.nodes[node.id] = node
    
    def add_edge(
        self,
        source: int,
        target: int,
        edge_type: EdgeType = EdgeType.PATH,
        key_required: Optional[int] = None,
        item_required: Optional[str] = None,
        switch_id: Optional[int] = None,
    ) -> None:
        """Add an edge to the graph."""
        edge = MissionEdge(
            source=source,
            target=target,
            edge_type=edge_type,
            key_required=key_required,
            item_required=item_required,
            switch_id=switch_id,
        )
        self.edges.append(edge)
        self._adjacency[source].append(target)
        
        # Add reverse traversal for bidirectional edge semantics.
        if edge_type in self.BIDIRECTIONAL_EDGE_TYPES:
            self._adjacency[target].append(source)

    def rebuild_adjacency(self) -> None:
        """
        Rebuild adjacency from edge list and prune dangling edges.

        This keeps `edges` and `_adjacency` consistent after rule operations
        that directly rewrite `graph.edges`.
        """
        valid_nodes = set(self.nodes.keys())
        rebuilt_edges: List[MissionEdge] = []
        new_adj: Dict[int, List[int]] = defaultdict(list)

        # Ensure every node has an adjacency bucket.
        for node_id in valid_nodes:
            new_adj[node_id] = []

        for edge in self.edges:
            if edge.source not in valid_nodes or edge.target not in valid_nodes:
                continue
            rebuilt_edges.append(edge)
            new_adj[edge.source].append(edge.target)
            if edge.edge_type in self.BIDIRECTIONAL_EDGE_TYPES:
                new_adj[edge.target].append(edge.source)

        # De-duplicate while preserving order.
        for node_id, neighbors in list(new_adj.items()):
            seen: Set[int] = set()
            deduped: List[int] = []
            for neighbor in neighbors:
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                deduped.append(neighbor)
            new_adj[node_id] = deduped

        self.edges = rebuilt_edges
        self._adjacency = new_adj

    def sanitize(self) -> None:
        """
        Normalize graph internal structures after arbitrary rule rewrites.
        """
        self._ensure_generation_stats_defaults()
        self.rebuild_adjacency()

        # Drop stale key->lock references that point to removed nodes.
        valid_nodes = set(self.nodes.keys())
        self._key_to_lock = {
            key_id: lock_id
            for key_id, lock_id in self._key_to_lock.items()
            if key_id in valid_nodes and lock_id in valid_nodes
        }
    
    def get_node(self, node_id: int) -> Optional[MissionNode]:
        """Get node by ID."""
        return self.nodes.get(node_id)
    
    def get_neighbors(self, node_id: int) -> List[int]:
        """Get neighbor node IDs."""
        return self._adjacency.get(node_id, [])

    def get_out_degree(self, node_id: int) -> int:
        """Get directed out-degree from explicit forward edges."""
        return int(len(self.get_forward_adjacency_map().get(node_id, [])))

    def get_adjacency_map(self) -> Dict[Any, List[Any]]:
        """Return a shallow copy of adjacency for read-only traversal."""
        return {node_id: list(neighbors) for node_id, neighbors in self._adjacency.items()}

    def get_forward_adjacency_map(self) -> Dict[Any, List[Any]]:
        """
        Return forward-only traversable adjacency from the explicit edge list.

        This keeps mission ordering separate from the weak/bidirectional
        traversal map used for reachability and layout.
        """
        adjacency: Dict[Any, List[Any]] = {node_id: [] for node_id in self.nodes.keys()}
        for edge in self.edges:
            if edge.source not in self.nodes or edge.target not in self.nodes:
                continue
            if edge.edge_type in self.NON_TRAVERSABLE_EDGE_TYPES:
                continue
            adjacency.setdefault(edge.source, []).append(edge.target)

        for node_id, neighbors in list(adjacency.items()):
            seen: Set[Any] = set()
            deduped: List[Any] = []
            for neighbor in neighbors:
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                deduped.append(neighbor)
            adjacency[node_id] = deduped
        return adjacency

    def get_forward_successors(self, node_id: int, depth: int = 1) -> List[MissionNode]:
        """Get forward-only successors reachable within `depth` mission steps."""
        if node_id not in self.nodes:
            return []

        adjacency = self.get_forward_adjacency_map()
        successors: List[MissionNode] = []
        visited = {node_id}
        queue = deque([(node_id, 0)])

        while queue:
            current, current_depth = queue.popleft()
            if current_depth >= depth:
                continue
            for neighbor in adjacency.get(current, []):
                if neighbor in visited or neighbor not in self.nodes:
                    continue
                visited.add(neighbor)
                successors.append(self.nodes[neighbor])
                queue.append((neighbor, current_depth + 1))
        return successors
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[MissionNode]:
        """Get all nodes of a given type."""
        return [n for n in self.nodes.values() if n.node_type == node_type]
    
    def get_start_node(self) -> Optional[MissionNode]:
        """Get the START node."""
        starts = self.get_nodes_by_type(NodeType.START)
        return starts[0] if starts else None
    
    def get_goal_node(self) -> Optional[MissionNode]:
        """Get the GOAL node."""
        goals = self.get_nodes_by_type(NodeType.GOAL)
        return goals[0] if goals else None

    def _node_index_map(self) -> Tuple[List[int], Dict[int, int]]:
        """
        Build a stable dense node index for tensor exports.

        Rule rewrites may legitimately delete interior nodes, leaving sparse
        IDs such as [0, 2, 5]. Tensor exporters must remap these IDs onto a
        contiguous [0, N) index space so edge_index and adjacency stay aligned
        with node_features.
        """
        node_ids = sorted(self.nodes.keys())
        id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        return node_ids, id_to_idx
    
    def to_tensor(self) -> Tuple[Tensor, Tensor]:
        """
        Convert to PyTorch tensors for GNN.
        
        Returns:
            edge_index: [2, num_edges] edge connections
            node_features: [num_nodes, feature_dim] node features
        """
        adapters = _require_torch_adapters()
        return adapters.mission_graph_to_tensor(self)
    
    def to_adjacency_matrix(self) -> Tensor:
        """Convert to adjacency matrix."""
        adapters = _require_torch_adapters()
        return adapters.mission_graph_to_adjacency_matrix(self)
    
    def compute_tpe(self) -> Tensor:
        """
        Compute Topological Positional Encoding for nodes.
        
        TPE encodes:
        - Distance from start
        - Distance to goal
        - Node degree
        - Local clustering
        - Path centrality
        
        Returns:
            [num_nodes, 8] TPE features
        """
        adapters = _require_torch_adapters()
        return adapters.mission_graph_compute_tpe(self)
    
    def _build_reverse_adjacency(self) -> Dict[int, List[int]]:
        """Build reverse traversal adjacency that respects directed edge semantics."""
        reverse_adj: Dict[int, List[int]] = defaultdict(list)
        valid_nodes = set(self.nodes.keys())

        for node_id in valid_nodes:
            reverse_adj[node_id] = []

        for edge in self.edges:
            if edge.source not in valid_nodes or edge.target not in valid_nodes:
                continue
            reverse_adj[edge.target].append(edge.source)
            if edge.edge_type in self.BIDIRECTIONAL_EDGE_TYPES:
                reverse_adj[edge.source].append(edge.target)

        for node_id, neighbors in list(reverse_adj.items()):
            seen: Set[int] = set()
            deduped: List[int] = []
            for neighbor in neighbors:
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                deduped.append(neighbor)
            reverse_adj[node_id] = deduped

        return reverse_adj

    def _bfs_distances(
        self,
        start_id: int,
        *,
        adjacency: Optional[Dict[int, List[int]]] = None,
    ) -> Dict[int, int]:
        """Compute BFS distances from a node."""
        if start_id not in self.nodes:
            return {}

        adjacency_map = self._adjacency if adjacency is None else adjacency
        distances = {start_id: 0}
        queue = deque([start_id])
        
        while queue:
            current = queue.popleft()
            current_dist = distances[current]
            
            for neighbor in adjacency_map.get(current, []):
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
        
        return distances
    
    def get_shortest_path_length(self, node_a: int, node_b: int) -> int:
        """
        Get shortest path length between two nodes using BFS.
        
        Returns:
            Path length, or -1 if nodes are not connected
        """
        if node_a == node_b:
            return 0
        
        distances = self._bfs_distances(node_a)
        return distances.get(node_b, -1)

    def get_forward_shortest_path_length(self, node_a: int, node_b: int) -> int:
        """
        Get shortest directed path length using explicit forward mission edges.

        This is the safe choice for progression semantics such as "item before
        gate" or "tutorial before climax".
        """
        if node_a == node_b:
            return 0

        distances = self._bfs_distances(
            node_a,
            adjacency=self.get_forward_adjacency_map(),
        )
        return distances.get(node_b, -1)
    
    def get_node_degree(self, node_id: int) -> int:
        """Get the degree (number of connections) of a node."""
        return len(self._adjacency.get(node_id, []))
    
    def get_reachable_nodes(
        self,
        start_node: int,
        excluded_edges: Optional[Set[Tuple[int, int]]] = None,
        excluded_nodes: Optional[Set[int]] = None,
    ) -> Set[int]:
        """
        Get all nodes reachable from start_node via BFS.
        
        Args:
            start_node: Starting node ID
            excluded_edges: Set of (source, target) tuples to exclude
            excluded_nodes: Set of node IDs to exclude from traversal
            
        Returns:
            Set of reachable node IDs
        """
        if excluded_edges is None:
            excluded_edges = set()
        if excluded_nodes is None:
            excluded_nodes = set()
        
        reachable = {start_node}
        queue = deque([start_node])
        
        while queue:
            current = queue.popleft()
            
            for neighbor in self._adjacency.get(current, []):
                # Check exclusions
                if neighbor in excluded_nodes:
                    continue
                if (current, neighbor) in excluded_edges:
                    continue
                if neighbor in reachable:
                    continue
                
                reachable.add(neighbor)
                queue.append(neighbor)
        
        return reachable
    
    def get_manhattan_distance(self, node_a: int, node_b: int) -> int:
        """
        Get Manhattan distance between two nodes based on position.
        
        Args:
            node_a: First node ID
            node_b: Second node ID
            
        Returns:
            Manhattan distance in grid coordinates
        """
        if node_a not in self.nodes or node_b not in self.nodes:
            return 999
        
        pos_a = self.nodes[node_a].position
        pos_b = self.nodes[node_b].position
        
        # Handle both 2D and 3D positions
        dist = abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])
        if len(pos_a) > 2 and len(pos_b) > 2:
            dist += abs(pos_a[2] - pos_b[2])
        
        return dist
    
    def get_nodes_with_degree_less_than(self, max_degree: int) -> List[MissionNode]:
        """Get all nodes with degree less than max_degree."""
        return [
            node for node in self.nodes.values()
            if self.get_node_degree(node.id) < max_degree
        ]
    
    def detect_cycles(self, max_cycle_length: int = 20) -> List[List[int]]:
        """
        Detect all cycles in the graph using DFS.
        
        Args:
            max_cycle_length: Maximum cycle length to detect (performance optimization).
                            Prevents exploring very long cycles that could be slow.
        
        Returns:
            List of cycles, where each cycle is a list of node IDs
        """
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node: int, path: List[int]) -> None:
            # Early termination for long paths (performance optimization)
            if len(path) > max_cycle_length:
                return
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self._adjacency.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, path[:])
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    if len(cycle) >= 3:  # Meaningful cycles only
                        cycles.append(cycle)
            
            rec_stack.remove(node)
        
        for node_id in self.nodes:
            if node_id not in visited:
                dfs(node_id, [])
        
        return cycles
    
    def trace_branch(self, start_node: int, max_depth: int = 10) -> List[int]:
        """
        Trace a branch from a starting node using DFS.
        
        Args:
            start_node: Starting node ID
            max_depth: Maximum depth to trace
            
        Returns:
            List of node IDs in the branch
        """
        branch = []
        visited = {start_node}
        queue = deque([(start_node, 0)])
        
        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            
            branch.append(current)
            
            for neighbor in self._adjacency.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
        
        return branch
    
    def get_nodes_in_different_branches(self, hub_id: int) -> List[List[int]]:
        """
        Partition nodes into different branches from a hub.
        
        Args:
            hub_id: Hub node ID
            
        Returns:
            List of branches, where each branch is a list of node IDs
        """
        if hub_id not in self.nodes:
            return []
        
        neighbors = self._adjacency.get(hub_id, [])
        branches = []
        
        for neighbor in neighbors:
            # Trace from this neighbor, excluding hub
            branch = []
            visited = {hub_id, neighbor}
            queue = deque([neighbor])
            
            while queue:
                current = queue.popleft()
                branch.append(current)
                
                for next_node in self._adjacency.get(current, []):
                    if next_node not in visited:
                        visited.add(next_node)
                        queue.append(next_node)
            
            if branch:
                branches.append(branch)
        
        return branches
    
    def count_keys_available_before(self, node_id: int) -> int:
        """
        Count how many KEY nodes are reachable before reaching node_id.
        
        Args:
            node_id: Target node ID
            
        Returns:
            Number of keys available
        """
        start = self.get_start_node()
        if not start:
            return 0
        
        # BFS from start, excluding node_id
        reachable = self.get_reachable_nodes(start.id, excluded_nodes={node_id})
        
        # Count KEY nodes in reachable set
        key_count = 0
        for nid in reachable:
            if nid in self.nodes and self.nodes[nid].node_type == NodeType.KEY:
                key_count += 1
        
        return key_count
    
    # ========================================================================
    # Wave 3: Helper Methods for Pedagogical & Quality Control Rules
    # ========================================================================
    
    def get_successors(self, node_id: int, depth: int = 1) -> List[MissionNode]:
        """
        Get nodes reachable from this node within depth steps.
        
        Args:
            node_id: Starting node ID
            depth: Maximum depth to traverse
            
        Returns:
            List of successor nodes within depth
        """
        if node_id not in self.nodes:
            return []
        
        successors = []
        visited = {node_id}
        queue = deque([(node_id, 0)])
        
        while queue:
            current, current_depth = queue.popleft()
            
            if current_depth >= depth:
                continue
            
            for neighbor in self._adjacency.get(current, []):
                if neighbor not in visited:
                    # Adjacency can temporarily contain stale IDs during rewrite
                    # passes; skip neighbors that no longer exist in node table.
                    if neighbor not in self.nodes:
                        continue
                    visited.add(neighbor)
                    successors.append(self.nodes[neighbor])
                    queue.append((neighbor, current_depth + 1))
        
        return successors
    
    def detect_high_tension_chains(self, min_length: int = 3) -> List[List[int]]:
        """
        Find sequences of combat/trap rooms (high tension areas).
        
        Args:
            min_length: Minimum chain length to detect
            
        Returns:
            List of chains, each chain is a list of node IDs
        """
        high_tension_types = {
            NodeType.ENEMY, NodeType.PUZZLE, NodeType.BOSS,
            NodeType.MINI_BOSS, NodeType.ARENA, NodeType.BOSS_DOOR
        }
        
        chains = []
        visited = set()
        
        for node_id in self.nodes:
            if node_id in visited:
                continue
            
            node = self.nodes[node_id]
            if node.node_type not in high_tension_types:
                continue
            
            # Start a chain
            chain = [node_id]
            visited.add(node_id)
            current = node_id
            
            # Extend forward
            while True:
                neighbors = self._adjacency.get(current, [])
                next_nodes = [
                    n for n in neighbors
                    if n not in visited
                    and n in self.nodes
                    and self.nodes[n].node_type in high_tension_types
                ]
                
                if not next_nodes:
                    break
                
                next_node = next_nodes[0]
                chain.append(next_node)
                visited.add(next_node)
                current = next_node
            
            if len(chain) >= min_length:
                chains.append(chain)
        
        return chains
    
    def get_branches_from_hub(self, hub_id: int) -> List[List[int]]:
        """
        Get distinct branches emanating from a hub node.
        
        Args:
            hub_id: Hub node ID
            
        Returns:
            List of branches, each branch is a list of node IDs
        """
        return self.get_nodes_in_different_branches(hub_id)
    
    def calculate_path_savings(self, new_edge: Tuple[int, int]) -> int:
        """
        Calculate how many steps a new edge would save.
        
        Args:
            new_edge: (source, target) tuple for potential edge
            
        Returns:
            Number of steps saved (original path - 1)
        """
        source, target = new_edge
        
        if source not in self.nodes or target not in self.nodes:
            return 0
        
        # Get current path length
        current_length = self.get_shortest_path_length(source, target)
        
        if current_length <= 0:
            return 0
        
        # New edge would make it 1 hop
        savings = current_length - 1
        return max(0, savings)
    
    def is_graph_connected(self) -> bool:
        """
        Check weak physical connectivity across all mission nodes.

        Returns:
            True if every node belongs to one undirected component
        """
        if not self.nodes:
            return True

        adjacency: Dict[int, Set[int]] = {node_id: set() for node_id in self.nodes}
        for edge in self.edges:
            if edge.edge_type in self.NON_TRAVERSABLE_EDGE_TYPES:
                continue
            if edge.source not in adjacency or edge.target not in adjacency:
                continue
            adjacency[edge.source].add(edge.target)
            adjacency[edge.target].add(edge.source)

        start_id = next(iter(self.nodes))
        visited = {start_id}
        queue = deque([start_id])
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
        return len(visited) == len(self.nodes)
    
    def get_item_for_gate(self, edge: MissionEdge) -> Optional[str]:
        """
        Get which item is required for an item-gated edge.
        
        Args:
            edge: Edge to check
            
        Returns:
            Item name if edge is item-gated, None otherwise
        """
        if edge.edge_type == EdgeType.ITEM_GATE:
            return edge.item_required
        return None
