"""
Mission Grammar: Graph Rewriting for Dungeon Structure
=======================================================

Implements a formal grammar for generating mission graphs
that represent dungeon structure and progression.

Research:
- Dormans & Bakkes (2011) "Procedural Adventure Game Design"
- "Generating Missions and Spaces for Adaptable Play Experiences"
- Graph Grammar approaches for level design

Grammar Rules (Chomsky-like):
-----------------------------
Start Symbol: S -> Goal

Terminal Symbols:
    - GOAL: Triforce/Boss room
    - KEY: Key pickup
    - LOCK: Locked door
    - ENEMY: Combat encounter
    - PUZZLE: Puzzle room
    - ITEM: Item room (bow, bomb, etc.)
    - START: Entrance

Non-Terminal Symbols:
    - S: Start symbol
    - BRANCH: Branching path
    - SEGMENT: Linear segment
    - CHALLENGE: Combat or puzzle

Production Rules:
    S --> START, SEGMENT, GOAL
    SEGMENT --> CHALLENGE | CHALLENGE, SEGMENT | BRANCH
    BRANCH --> (SEGMENT) + (SEGMENT)  [parallel paths]
    CHALLENGE --> ENEMY | PUZZLE | LOCK_KEY
    LOCK_KEY --> KEY, LOCK  [key must precede lock]

Constraint: Lock-Key Ordering
    For each LOCK node, there must exist a KEY node
    that is reachable before the LOCK in the graph.

Usage:
    grammar = MissionGrammar(seed=42)
    
    # Generate mission graph
    graph = grammar.generate(
        difficulty=Difficulty.MEDIUM,
        num_rooms=8,
    )
    
    # Validate lock-key constraints
    assert grammar.validate_lock_key_ordering(graph)
    
    # Convert to edge_index for GNN
    edge_index, node_features = graph.to_tensor()
"""

import random
import logging
from typing import ClassVar, Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
import math

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

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
        EdgeType.SHORTCUT,
        EdgeType.WARP,
        EdgeType.STAIRS,
        EdgeType.HIDDEN,
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
        """Get directed out-degree from cached adjacency."""
        return int(len(self._adjacency.get(node_id, [])))

    def get_adjacency_map(self) -> Dict[int, List[int]]:
        """Return a shallow copy of adjacency for read-only traversal."""
        return {int(node_id): list(neighbors) for node_id, neighbors in self._adjacency.items()}
    
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
    
    def to_tensor(self) -> Tuple[Tensor, Tensor]:
        """
        Convert to PyTorch tensors for GNN.
        
        Returns:
            edge_index: [2, num_edges] edge connections
            node_features: [num_nodes, feature_dim] node features
        """
        # Build edge index
        sources = []
        targets = []
        for edge in self.edges:
            sources.append(edge.source)
            targets.append(edge.target)
        
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        
        # Build node features
        node_ids = sorted(self.nodes.keys())
        features = []
        for nid in node_ids:
            features.append(self.nodes[nid].to_feature_vector())
        
        node_features = torch.tensor(features, dtype=torch.float32)
        
        return edge_index, node_features
    
    def to_adjacency_matrix(self) -> Tensor:
        """Convert to adjacency matrix."""
        n = len(self.nodes)
        adj = torch.zeros(n, n)
        
        for edge in self.edges:
            adj[edge.source, edge.target] = 1.0
            if edge.edge_type in self.BIDIRECTIONAL_EDGE_TYPES:  # Bidirectional
                adj[edge.target, edge.source] = 1.0
        
        return adj
    
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
        n = len(self.nodes)
        node_ids = sorted(self.nodes.keys())
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        
        tpe = torch.zeros(n, 8)
        
        # Get start and goal
        start = self.get_start_node()
        goal = self.get_goal_node()
        
        start_id = start.id if start else 0
        goal_id = goal.id if goal else n - 1
        
        # Compute BFS distances from start
        dist_from_start = self._bfs_distances(start_id)
        
        # Compute BFS distances from goal (reverse)
        dist_to_goal = self._bfs_distances(goal_id)
        
        max_dist = max(max(dist_from_start.values(), default=1), 1)
        
        for nid in node_ids:
            idx = id_to_idx[nid]
            
            # Distance features (normalized)
            tpe[idx, 0] = dist_from_start.get(nid, max_dist) / max_dist
            tpe[idx, 1] = dist_to_goal.get(nid, max_dist) / max_dist
            
            # Degree
            degree = len(self._adjacency.get(nid, []))
            tpe[idx, 2] = min(degree / 4.0, 1.0)
            
            # Is on critical path
            if start and goal:
                on_path = (dist_from_start.get(nid, float('inf')) + 
                          dist_to_goal.get(nid, float('inf')) ==
                          dist_from_start.get(goal_id, float('inf')))
                tpe[idx, 3] = 1.0 if on_path else 0.0
            
            # Node type indicators
            node = self.nodes[nid]
            tpe[idx, 4] = 1.0 if node.node_type == NodeType.KEY else 0.0
            tpe[idx, 5] = 1.0 if node.node_type == NodeType.LOCK else 0.0
            tpe[idx, 6] = node.difficulty
            tpe[idx, 7] = 1.0 if node.key_id is not None else 0.0
        
        return tpe
    
    def _bfs_distances(self, start_id: int) -> Dict[int, int]:
        """Compute BFS distances from a node."""
        distances = {start_id: 0}
        queue = [start_id]
        
        while queue:
            current = queue.pop(0)
            current_dist = distances[current]
            
            for neighbor in self._adjacency.get(current, []):
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
        queue = [start_node]
        
        while queue:
            current = queue.pop(0)
            
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
        queue = [(start_node, 0)]
        
        while queue:
            current, depth = queue.pop(0)
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
            queue = [neighbor]
            
            while queue:
                current = queue.pop(0)
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
        queue = [(node_id, 0)]
        
        while queue:
            current, current_depth = queue.pop(0)
            
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
        Check if graph remains connected.
        
        Returns:
            True if all nodes reachable from any starting node
        """
        if not self.nodes:
            return True
        
        start_id = next(iter(self.nodes.keys()))
        reachable = self.get_reachable_nodes(start_id)
        
        return len(reachable) == len(self.nodes)
    
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


# ============================================================================
# GRAMMAR RULES
# ============================================================================

class ProductionRule:
    """Base class for grammar production rules."""
    
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
    
    def can_apply(
        self,
        graph: MissionGraph,
        context: Dict[str, Any],
    ) -> bool:
        """Check if this rule can be applied."""
        return True
    
    def apply(
        self,
        graph: MissionGraph,
        context: Dict[str, Any],
    ) -> MissionGraph:
        """Apply the rule and return modified graph."""
        raise NotImplementedError


class StartRule(ProductionRule):
    """S -> START, SEGMENT, GOAL"""
    
    def __init__(self):
        super().__init__("Start", weight=1.0)
    
    def apply(
        self,
        graph: MissionGraph,
        context: Dict[str, Any],
    ) -> MissionGraph:
        """Create initial graph with START and GOAL."""
        # Create START node
        start = MissionNode(
            id=0,
            node_type=NodeType.START,
            position=(0, 0, 0),  # 3D position (row, col, floor)
            difficulty=0.0,
        )
        graph.add_node(start)
        
        # Create GOAL node
        goal = MissionNode(
            id=1,
            node_type=NodeType.GOAL,
            position=(context.get('goal_row', 5), context.get('goal_col', 5), 0),
            difficulty=1.0,
        )
        graph.add_node(goal)
        
        # Connect with edge (to be filled in)
        graph.add_edge(0, 1, EdgeType.PATH)
        
        return graph


class InsertChallengeRule(ProductionRule):
    """Insert a challenge node between two connected nodes."""
    
    def __init__(self, challenge_type: NodeType = NodeType.ENEMY):
        super().__init__(f"InsertChallenge_{challenge_type.name}", weight=1.0)
        self.challenge_type = challenge_type
    
    def can_apply(
        self,
        graph: MissionGraph,
        context: Dict[str, Any],
    ) -> bool:
        """Can apply if there are traversable PATH edges to split."""
        return any(edge.edge_type == EdgeType.PATH for edge in graph.edges)
    
    def apply(
        self,
        graph: MissionGraph,
        context: Dict[str, Any],
    ) -> MissionGraph:
        """Insert challenge node on a random edge."""
        path_edges = [
            (idx, edge) for idx, edge in enumerate(graph.edges)
            if edge.edge_type == EdgeType.PATH
        ]
        if not path_edges:
            return graph
        
        # Select random traversable edge
        rng = context.get('rng') or random
        edge_idx, edge = rng.choice(path_edges)
        
        # Create new challenge node
        new_id = max(graph.nodes.keys()) + 1
        
        # Interpolate position
        src_pos = graph.nodes[edge.source].position
        tgt_pos = graph.nodes[edge.target].position
        new_pos = (
            (src_pos[0] + tgt_pos[0]) // 2,
            (src_pos[1] + tgt_pos[1]) // 2,
            src_pos[2] if len(src_pos) > 2 else 0,  # Same floor
        )
        
        challenge = MissionNode(
            id=new_id,
            node_type=self.challenge_type,
            position=new_pos,
            difficulty=context.get('difficulty', 0.5),
        )
        graph.add_node(challenge)
        
        # Remove old edge and add new ones
        graph.edges.pop(edge_idx)
        graph.add_edge(edge.source, new_id, EdgeType.PATH)
        graph.add_edge(new_id, edge.target, EdgeType.PATH)
        graph.sanitize()
        
        return graph


class InsertLockKeyRule(ProductionRule):
    """
    Insert a Lock-Key pair ensuring key precedes lock.
    
    KEY -> ... -> LOCK -> continuation
    
    The key MUST be reachable before the lock in the graph.
    """
    
    def __init__(self):
        super().__init__("InsertLockKey", weight=0.8)
    
    def can_apply(
        self,
        graph: MissionGraph,
        context: Dict[str, Any],
    ) -> bool:
        """Can apply if there's a path of at least 2 edges."""
        if len(graph.nodes) < 2:
            return False
        return any(edge.edge_type == EdgeType.PATH for edge in graph.edges)
    
    def apply(
        self,
        graph: MissionGraph,
        context: Dict[str, Any],
    ) -> MissionGraph:
        """Insert KEY before LOCK along a start->goal progression path."""
        if len(graph.edges) < 1 or len(graph.nodes) < 2:
            return graph
        
        rng = context.get('rng') or random
        graph.sanitize()

        # Prefer splitting edges on the current critical path (START -> GOAL).
        start = graph.get_start_node()
        goal = graph.get_goal_node()
        path_edges: List[Tuple[int, MissionEdge]] = []
        if start is not None and goal is not None:
            path_nodes = self._find_shortest_path_nodes(graph, start.id, goal.id)
            if len(path_nodes) >= 2:
                for i in range(len(path_nodes) - 1):
                    edge_idx = self._find_path_edge_index(graph, path_nodes[i], path_nodes[i + 1])
                    if edge_idx is not None:
                        path_edges.append((edge_idx, graph.edges[edge_idx]))

        if path_edges:
            key_edge_idx, key_edge = rng.choice(path_edges)
        else:
            fallback_path_edges = [
                (i, e) for i, e in enumerate(graph.edges)
                if e.edge_type == EdgeType.PATH
            ]
            if not fallback_path_edges:
                return graph
            key_edge_idx, key_edge = rng.choice(fallback_path_edges)
        
        # Create KEY node
        key_id = max(graph.nodes.keys()) + 1
        key_node = MissionNode(
            id=key_id,
            node_type=NodeType.KEY,
            position=self._interpolate_pos(graph, key_edge.source, key_edge.target, 0.3),
            difficulty=context.get('difficulty', 0.5) * 0.5,
            key_id=key_id,  # Self-referencing key ID
        )
        graph.add_node(key_node)
        
        # Insert KEY on the edge
        graph.edges.pop(key_edge_idx)
        graph.add_edge(key_edge.source, key_id, EdgeType.PATH)
        graph.add_edge(key_id, key_edge.target, EdgeType.PATH)
        graph.sanitize()
        
        # Now find an edge AFTER the key position for the LOCK
        lock_candidates: List[Tuple[int, MissionEdge]] = []
        if start is not None and goal is not None:
            updated_path = self._find_shortest_path_nodes(graph, start.id, goal.id)
            if len(updated_path) >= 2 and key_id in updated_path:
                key_idx_on_path = updated_path.index(key_id)
                for i in range(key_idx_on_path + 1, len(updated_path) - 1):
                    edge_idx = self._find_path_edge_index(graph, updated_path[i], updated_path[i + 1])
                    if edge_idx is not None:
                        edge = graph.edges[edge_idx]
                        # Keep lock insertion on normal traversable edges.
                        if edge.edge_type == EdgeType.PATH:
                            lock_candidates.append((edge_idx, edge))

        # Fallback when no strict "after-key" path edge is available.
        if not lock_candidates:
            lock_candidates = [
                (i, e) for i, e in enumerate(graph.edges)
                if e.edge_type == EdgeType.PATH and e.source != key_id and e.target != key_id
            ]
        
        if lock_candidates:
            lock_edge_idx, lock_edge = rng.choice(lock_candidates)
            
            # Create LOCK node
            lock_id = max(graph.nodes.keys()) + 1
            lock_node = MissionNode(
                id=lock_id,
                node_type=NodeType.LOCK,
                position=self._interpolate_pos(graph, lock_edge.source, lock_edge.target, 0.7),
                difficulty=context.get('difficulty', 0.5),
                key_id=key_id,  # Reference to required key
            )
            graph.add_node(lock_node)
            
            # Insert LOCK with locked edge type
            graph.edges = [e for i, e in enumerate(graph.edges) if i != lock_edge_idx]
            graph.add_edge(lock_edge.source, lock_id, EdgeType.PATH)
            graph.add_edge(lock_id, lock_edge.target, EdgeType.LOCKED, key_required=key_id)
            
            # Track key-lock relationship
            graph._key_to_lock[key_id] = lock_id

        graph.sanitize()
        return graph

    def _find_shortest_path_nodes(
        self,
        graph: MissionGraph,
        start_id: int,
        goal_id: int,
    ) -> List[int]:
        """Return one shortest path (node sequence) over adjacency."""
        if start_id == goal_id:
            return [start_id]

        visited = {start_id}
        queue: List[Tuple[int, List[int]]] = [(start_id, [start_id])]
        while queue:
            current, path = queue.pop(0)
            for neighbor in graph._adjacency.get(current, []):
                if neighbor in visited:
                    continue
                new_path = path + [neighbor]
                if neighbor == goal_id:
                    return new_path
                visited.add(neighbor)
                queue.append((neighbor, new_path))
        return []

    def _find_path_edge_index(
        self,
        graph: MissionGraph,
        node_a: int,
        node_b: int,
    ) -> Optional[int]:
        """
        Find a PATH edge index connecting two adjacent nodes, in either direction.
        """
        for idx, edge in enumerate(graph.edges):
            if edge.edge_type != EdgeType.PATH:
                continue
            if (
                (edge.source == node_a and edge.target == node_b) or
                (edge.source == node_b and edge.target == node_a)
            ):
                return idx
        return None
    
    def _interpolate_pos(
        self,
        graph: MissionGraph,
        src: int,
        tgt: int,
        t: float,
    ) -> Tuple[int, int, int]:
        """Interpolate position between two nodes."""
        src_pos = graph.nodes[src].position
        tgt_pos = graph.nodes[tgt].position
        z = src_pos[2] if len(src_pos) > 2 else 0
        return (
            int(src_pos[0] * (1 - t) + tgt_pos[0] * t),
            int(src_pos[1] * (1 - t) + tgt_pos[1] * t),
            z,
        )


class BranchRule(ProductionRule):
    """Create a branch (parallel paths)."""
    
    def __init__(self):
        super().__init__("Branch", weight=0.5)
    
    def can_apply(
        self,
        graph: MissionGraph,
        context: Dict[str, Any],
    ) -> bool:
        """Can apply if there's a node with degree < 4."""
        for node in graph.nodes.values():
            if len(graph._adjacency.get(node.id, [])) < 4:
                return True
        return False
    
    def apply(
        self,
        graph: MissionGraph,
        context: Dict[str, Any],
    ) -> MissionGraph:
        """Create a branch from a random node."""
        rng = context.get('rng') or random
        
        # Find nodes that can have more connections
        candidates = [
            n for n in graph.nodes.values()
            if len(graph._adjacency.get(n.id, [])) < 4
            and n.node_type not in [NodeType.START, NodeType.GOAL]
        ]
        
        if not candidates:
            return graph
        
        branch_node = rng.choice(candidates)
        
        # Create branch endpoint
        new_id = max(graph.nodes.keys()) + 1
        offset_r = rng.randint(-2, 2)
        offset_c = rng.randint(-2, 2)
        
        branch_pos = branch_node.position
        floor = branch_pos[2] if len(branch_pos) > 2 else 0
        
        new_node = MissionNode(
            id=new_id,
            node_type=rng.choice([NodeType.ITEM, NodeType.PUZZLE, NodeType.EMPTY]),
            position=(
                branch_node.position[0] + offset_r,
                branch_node.position[1] + offset_c,
                floor,
            ),
            difficulty=context.get('difficulty', 0.5) * rng.uniform(0.5, 1.0),
        )
        graph.add_node(new_node)
        graph.add_edge(branch_node.id, new_id, EdgeType.PATH)
        
        return graph


# ============================================================================
# MISSION GRAMMAR
# ============================================================================

class Difficulty(Enum):
    """Dungeon difficulty levels."""
    EASY = 1
    MEDIUM = 2
    HARD = 3
    EXPERT = 4


class MissionGrammar:
    """
    Grammar for generating mission graphs.
    
    Uses production rules to build a mission graph that represents
    dungeon structure, ensuring lock-key constraints are satisfied.
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.rng = random.Random(seed)
        
        # Define production rules
        # Core structural rules
        self.rules = [
            StartRule(),
            
            # Basic building blocks
            InsertChallengeRule(NodeType.ENEMY),
            InsertChallengeRule(NodeType.PUZZLE),
            InsertLockKeyRule(),
            BranchRule(),
            
            # Advanced topology rules (Thesis Upgrades #1-3)
            MergeRule(),  # Creates shortcuts/cycles
            InsertSwitchRule(),  # Dynamic state changes
            AddBossGauntlet(),  # Big key hierarchy
            
            # Item-based progression
            AddItemGateRule(),  # Specific item requirements
            
            # Structural complexity
            CreateHubRule(),  # Central intersection points
            AddStairsRule(),  # Multi-floor support
            
            # Optional/hidden content
            AddSecretRule(),  # Hidden rooms
            AddTeleportRule(),  # Warp connections
            
            # Cleanup
            PruneGraphRule(),  # Simplify overly complex graphs
            
            # ========================================================
            # ADVANCED RULES (Thesis-Grade Patterns)
            # Based on Dormans "Unexplored" & Brown "Boss Keys"
            # ========================================================
            AddFungibleLockRule(),  # Economy: Fungible key inventory
            FormBigRoomRule(),  # Geometry: Merge rooms into great halls
            AddValveRule(),  # Directionality: One-way edges in cycles
            AddForeshadowingRule(),  # Design: Visual links (windows)
            AddCollectionChallengeRule(),  # Design: Multi-token gates
            AddArenaRule(),  # Pacing: Combat shutters
            AddSectorRule(),  # Structure: Thematic zones
            AddEntangledBranchesRule(),  # Design: Cross-branch dependencies
            AddHazardGateRule(),  # Design: Risky paths with protection
            SplitRoomRule(),  # Geometry: Virtual room layers
            
            # ========================================================
            # WAVE 3: PEDAGOGICAL & QUALITY CONTROL RULES
            # Nintendo-grade level design patterns
            # ========================================================
            AddSkillChainRule(),  # Pedagogy: Tutorial sequences
            AddPacingBreakerRule(),  # Pedagogy: Negative space/sanctuaries
            AddResourceLoopRule(),  # Safety: Farming spots prevent soft-locks
            AddGatekeeperRule(),  # Quality: Mini-boss guardians
            AddMultiLockRule(),  # Quality: Multi-switch batteries
            AddItemShortcutRule(),  # Quality: Item-gated returns
            PruneDeadEndRule(),  # Quality: Garbage collection (run late)
        ]
    
    def validate_all_constraints(self, graph: MissionGraph) -> bool:
        """
        Run all validation checks on generated graph.
        
        Performs comprehensive validation including:
        - Lock-key ordering constraints
        - Skill chain progression (Wave 3)
        - Battery reachability (Wave 3)
        - Resource loop validity (Wave 3)
        
        Args:
            graph: MissionGraph to validate
        
        Returns:
            True if all validation checks pass, False otherwise
        """
        results = {
            'anchor_nodes': self.validate_anchor_nodes(graph),
            'lock_key_ordering': self.validate_lock_key_ordering(graph),
            'progression_constraints': self.validate_progression_constraints(graph),
            'skill_chains': validate_skill_chains(graph),
            'battery_reachability': validate_battery_reachability(graph),
            'resource_loops': validate_resource_loops(graph),
        }
        
        # Log any failures
        all_passed = True
        for check, passed in results.items():
            if not passed:
                logger.warning(f"Validation failed: {check}")
                all_passed = False
            else:
                logger.debug(f"Validation passed: {check}")
        
        if all_passed:
            logger.info("All validation checks passed")
        
        return all_passed
    
    def generate(
        self,
        difficulty: Difficulty = Difficulty.MEDIUM,
        num_rooms: int = 8,
        max_keys: int = 2,
        validate_all: bool = True,
    ) -> MissionGraph:
        """
        Generate a mission graph.
        
        Args:
            difficulty: Dungeon difficulty level
            num_rooms: Approximate number of rooms
            max_keys: Maximum number of key-lock pairs
            validate_all: If True, run comprehensive validation checks after generation
            
        Returns:
            Generated MissionGraph
        """
        graph = MissionGraph()
        
        context = {
            'rng': self.rng,
            'difficulty': difficulty.value / 4.0,
            'goal_row': num_rooms // 2,
            'goal_col': num_rooms // 2,
        }
        
        # Apply start rule
        graph = self.rules[0].apply(graph, context)
        graph.sanitize()
        
        # Track how many of each have been added
        num_keys_added = 0
        num_challenges_added = 0
        
        # Apply rules until we have enough nodes
        max_iterations = num_rooms * 3
        iteration = 0
        
        while len(graph.nodes) < num_rooms and iteration < max_iterations:
            iteration += 1
            graph.sanitize()
            
            # Select a rule using adaptive weights to stage structure/gating/polish.
            applicable_rules = []
            weights = []
            
            for rule in self.rules[1:]:  # Skip start rule
                if not rule.can_apply(graph, context):
                    continue
                
                # Limit key-lock pairs
                if isinstance(rule, InsertLockKeyRule):
                    if num_keys_added >= max_keys:
                        continue

                adaptive_weight = self._compute_adaptive_rule_weight(
                    rule=rule,
                    graph=graph,
                    iteration=iteration,
                    max_iterations=max_iterations,
                    num_rooms=num_rooms,
                    num_keys_added=num_keys_added,
                    max_keys=max_keys,
                    num_challenges_added=num_challenges_added,
                )
                if adaptive_weight <= 0.0:
                    continue

                applicable_rules.append(rule)
                weights.append(adaptive_weight)
            
            if not applicable_rules:
                break
            
            # Weighted random selection
            total_weight = sum(weights)
            r = self.rng.uniform(0, total_weight)
            cumulative = 0
            selected_rule = applicable_rules[0]
            
            for rule, weight in zip(applicable_rules, weights):
                cumulative += weight
                if r <= cumulative:
                    selected_rule = rule
                    break
            
            # Apply rule
            graph = selected_rule.apply(graph, context)
            graph.sanitize()
            
            # Track additions
            if isinstance(selected_rule, InsertLockKeyRule):
                num_keys_added += 1
            elif isinstance(selected_rule, InsertChallengeRule):
                num_challenges_added += 1

        # Repair/validation convergence loop.
        # Multiple rounds are needed because one repair can expose another issue.
        rounds = 4 if validate_all else 2
        for _ in range(rounds):
            graph = self._ensure_anchor_nodes(graph)
            graph.sanitize()
            repairs_before = int(graph.generation_stats.get("total_repairs", 0))
            round_had_repairs = False

            lock_ok = self.validate_lock_key_ordering(graph)
            if not lock_ok:
                logger.warning("Generated graph failed lock-key validation, fixing...")
                graph = self._fix_lock_key_ordering(graph)
                graph = self._ensure_anchor_nodes(graph)
                graph.sanitize()
                round_had_repairs = int(graph.generation_stats.get("total_repairs", 0)) > repairs_before

            if not validate_all:
                if round_had_repairs:
                    graph.generation_stats["repair_rounds"] = int(
                        graph.generation_stats.get("repair_rounds", 0)
                    ) + 1
                if self.validate_lock_key_ordering(graph):
                    break
                continue

            if self.validate_all_constraints(graph):
                if round_had_repairs:
                    graph.generation_stats["repair_rounds"] = int(
                        graph.generation_stats.get("repair_rounds", 0)
                    ) + 1
                break

            graph = self._repair_progression_constraints(graph)
            graph = self._repair_wave3_constraints(graph)
            graph = self._ensure_anchor_nodes(graph)
            graph.sanitize()
            round_had_repairs = int(graph.generation_stats.get("total_repairs", 0)) > repairs_before
            if round_had_repairs:
                graph.generation_stats["repair_rounds"] = int(
                    graph.generation_stats.get("repair_rounds", 0)
                ) + 1

            if self.validate_all_constraints(graph):
                break

        if validate_all and not self.validate_all_constraints(graph):
            logger.warning(
                "Graph validation failed on some checks even after repair. "
                "Graph is being returned but may have constraint violations. "
                "Consider regenerating with different seed or parameters."
            )
        elif not validate_all and not self.validate_lock_key_ordering(graph):
            logger.warning(
                "Lock-key repair could not fully satisfy constraints; "
                "invalid locks were downgraded to preserve solvability."
            )
        
        # Update positions for layout
        graph = self._layout_graph(graph)
        
        return graph

    def _compute_adaptive_rule_weight(
        self,
        rule: ProductionRule,
        graph: MissionGraph,
        iteration: int,
        max_iterations: int,
        num_rooms: int,
        num_keys_added: int,
        max_keys: int,
        num_challenges_added: int,
    ) -> float:
        """
        Compute dynamic rule weight to reduce late-stage repairs.

        The policy stages generation into:
        1) topology growth,
        2) progression/gating,
        3) cleanup/polish.
        """
        base_weight = max(0.0, float(getattr(rule, "weight", 1.0)))
        if base_weight <= 0.0:
            return 0.0

        rule_name = getattr(rule, "name", "")
        node_count = len(graph.nodes)
        progress_nodes = min(1.0, node_count / max(1, num_rooms))
        progress_iterations = min(1.0, iteration / max(1, max_iterations))
        progress = max(progress_nodes, progress_iterations * 0.8)

        if rule_name in {"PruneGraph", "PruneDeadEnd"}:
            if progress < 0.70:
                return 0.0
            base_weight *= 1.0 + (progress - 0.70) * 1.8

        if rule_name.startswith("InsertChallenge_"):
            target_challenges = max(2, int(num_rooms * 0.35))
            if num_challenges_added < target_challenges:
                base_weight *= 1.35
            if progress > 0.90:
                base_weight *= 0.75

        if rule_name == "InsertLockKey":
            if num_keys_added >= max_keys:
                return 0.0
            if node_count < 4:
                base_weight *= 0.30
            if progress < 0.25:
                base_weight *= 0.65
            elif progress <= 0.80:
                base_weight *= 1.20
            else:
                base_weight *= 0.70

        if rule_name in {"Branch", "MergeShortcut", "CreateHub", "AddStairs", "AddSector", "SplitRoom"}:
            if progress < 0.55:
                base_weight *= 1.20
            elif progress > 0.90:
                base_weight *= 0.75

        if rule_name in {"AddItemGate", "AddFungibleLock", "AddMultiLock", "AddGatekeeper", "AddHazardGate", "AddBossGauntlet"}:
            if progress < 0.40:
                base_weight *= 0.45
            elif progress > 0.90:
                base_weight *= 0.80
            else:
                base_weight *= 1.15

        if rule_name in {"AddSkillChain", "AddPacingBreaker", "AddResourceLoop", "AddItemShortcut", "AddForeshadowing"}:
            if progress < 0.55:
                base_weight *= 0.60
            else:
                base_weight *= 1.10

        return max(0.01, base_weight)

    def validate_anchor_nodes(self, graph: MissionGraph) -> bool:
        """Validate there is exactly one START and exactly one GOAL."""
        starts = graph.get_nodes_by_type(NodeType.START)
        goals = graph.get_nodes_by_type(NodeType.GOAL)
        return len(starts) == 1 and len(goals) == 1

    def _ensure_anchor_nodes(self, graph: MissionGraph) -> MissionGraph:
        """
        Enforce stable mission anchors.

        Some transformation rules can accidentally retag START/GOAL nodes.
        This pass restores a single START and GOAL while preserving topology.
        """
        graph.sanitize()
        if not graph.nodes:
            return graph

        # Ensure START exists and is unique.
        start_nodes = sorted(graph.get_nodes_by_type(NodeType.START), key=lambda n: n.id)
        if not start_nodes:
            preferred_start = graph.nodes.get(0)
            if preferred_start is None:
                first_id = min(graph.nodes.keys())
                preferred_start = graph.nodes[first_id]
            preferred_start.node_type = NodeType.START
            preferred_start.difficulty = 0.0
            preferred_start.key_id = None
            preferred_start.is_tutorial = False
            preferred_start.is_mini_boss = False
            preferred_start.is_sanctuary = False
            preferred_start.difficulty_rating = "SAFE"
            preferred_start.tension_value = 0.0
            start_nodes = [preferred_start]
        elif len(start_nodes) > 1:
            keep_start = start_nodes[0]
            for extra_start in start_nodes[1:]:
                extra_start.node_type = NodeType.EMPTY
                extra_start.is_tutorial = False
                extra_start.is_sanctuary = False
            start_nodes = [keep_start]

        # Ensure GOAL exists and is unique.
        goal_nodes = sorted(graph.get_nodes_by_type(NodeType.GOAL), key=lambda n: n.id)
        if not goal_nodes:
            preferred_goal = graph.nodes.get(1)
            if preferred_goal is None or preferred_goal.id == start_nodes[0].id:
                # Pick the highest-ID non-start node when possible.
                non_start_ids = [nid for nid in graph.nodes if nid != start_nodes[0].id]
                if non_start_ids:
                    preferred_goal = graph.nodes[max(non_start_ids)]
                else:
                    preferred_goal = start_nodes[0]
            preferred_goal.node_type = NodeType.GOAL
            preferred_goal.difficulty = 1.0
            preferred_goal.key_id = None
            preferred_goal.is_tutorial = False
            preferred_goal.is_mini_boss = False
            preferred_goal.is_sanctuary = False
            preferred_goal.difficulty_rating = "HARD"
            preferred_goal.tension_value = 1.0
            goal_nodes = [preferred_goal]
        elif len(goal_nodes) > 1:
            keep_goal = goal_nodes[0]
            for extra_goal in goal_nodes[1:]:
                extra_goal.node_type = NodeType.EMPTY
                extra_goal.is_tutorial = False
                extra_goal.is_sanctuary = False
            goal_nodes = [keep_goal]

        graph.sanitize()
        return graph

    def ensure_anchor_nodes(self, graph: MissionGraph) -> MissionGraph:
        """Public wrapper for anchor-node normalization."""
        return self._ensure_anchor_nodes(graph)
    
    def validate_lock_key_ordering(self, graph: MissionGraph) -> bool:
        """
        Validate that all keys can be reached before their locks.
        
        For each LOCK node, verifies that its required KEY is
        reachable from START without passing through the LOCK.
        """
        graph.sanitize()
        start = graph.get_start_node()
        if not start:
            return False
        
        lock_types = {NodeType.LOCK, NodeType.BOSS_DOOR}
        key_types = {NodeType.KEY, NodeType.BIG_KEY}
        locks = [n for n in graph.nodes.values() if n.node_type in lock_types]
        keys = [n for n in graph.nodes.values() if n.node_type in key_types]
        
        # Build key lookup (multiple providers can map to same key_id).
        key_by_id: Dict[int, List[MissionNode]] = defaultdict(list)
        for key_node in keys:
            if key_node.key_id is not None:
                key_by_id[key_node.key_id].append(key_node)
        
        for lock in locks:
            if lock.key_id is None:
                continue
            
            providers = key_by_id.get(lock.key_id, [])
            if not providers:
                logger.warning(f"Lock {lock.id} references non-existent key {lock.key_id}")
                return False
            
            # Check if key is reachable from start without passing through lock
            if not any(
                self._is_reachable_without(graph, start.id, key.id, exclude={lock.id})
                for key in providers
            ):
                logger.warning(
                    f"No provider key for lock {lock.id} is reachable before lock "
                    f"(key_id={lock.key_id})"
                )
                return False
        
        return True

    def validate_progression_constraints(self, graph: MissionGraph) -> bool:
        """
        Validate edge-level progression constraints (beyond lock-node ordering).

        Checks:
        - LOCKED/BOSS_LOCKED edges have a valid key provider reachable pre-gate.
        - ITEM_GATE edges have matching item providers reachable pre-gate.
        - MULTI_LOCK edges have enough reachable TOKEN nodes pre-gate.
        - Fungible key locks (requires_key_count) have enough reachable keys pre-gate.
        """
        graph.sanitize()
        start = graph.get_start_node()
        if not start:
            return False

        # Provider indexes.
        key_providers: Dict[int, List[int]] = defaultdict(list)
        for node in graph.nodes.values():
            if node.node_type in {NodeType.KEY, NodeType.BIG_KEY} and node.key_id is not None:
                key_providers[node.key_id].append(node.id)

        item_providers: Dict[str, List[int]] = defaultdict(list)
        for node in graph.nodes.values():
            if node.node_type == NodeType.ITEM and node.item_type:
                item_providers[str(node.item_type)].append(node.id)

        token_nodes = [n.id for n in graph.nodes.values() if n.node_type == NodeType.TOKEN]
        key_nodes = [
            n.id for n in graph.nodes.values()
            if n.node_type in {NodeType.KEY, NodeType.BIG_KEY}
        ]

        for edge in graph.edges:
            excluded_edge = {(edge.source, edge.target)}

            if edge.edge_type in {EdgeType.LOCKED, EdgeType.BOSS_LOCKED}:
                # Fungible small-key locks intentionally use requires_key_count
                # without a specific key_required ID.
                if edge.key_required is None:
                    if edge.edge_type == EdgeType.LOCKED and edge.requires_key_count > 0:
                        pass
                    else:
                        logger.warning(
                            f"{edge.edge_type.name} edge {edge.source}->{edge.target} missing key_required"
                        )
                        return False
                if edge.key_required is not None:
                    providers = key_providers.get(edge.key_required, [])
                    if not providers:
                        logger.warning(
                            f"No key provider for edge {edge.source}->{edge.target} "
                            f"(key_required={edge.key_required})"
                        )
                        return False
                    if not any(
                        self._is_reachable_without_edges(graph, start.id, provider, excluded_edge)
                        for provider in providers
                    ):
                        logger.warning(
                            f"Key for locked edge {edge.source}->{edge.target} is not reachable pre-gate"
                        )
                        return False

            if edge.requires_key_count > 0:
                reachable = graph.get_reachable_nodes(start.id, excluded_edges=excluded_edge)
                reachable_keys = sum(1 for key_node in key_nodes if key_node in reachable)
                if reachable_keys < edge.requires_key_count:
                    logger.warning(
                        f"Fungible key lock {edge.source}->{edge.target} requires "
                        f"{edge.requires_key_count} but only {reachable_keys} keys are reachable pre-gate"
                    )
                    return False

            if edge.edge_type == EdgeType.ITEM_GATE and edge.item_required:
                providers = item_providers.get(str(edge.item_required), [])
                if not providers:
                    logger.warning(
                        f"No item provider for ITEM_GATE {edge.source}->{edge.target} "
                        f"(item_required={edge.item_required})"
                    )
                    return False
                if not any(
                    self._is_reachable_without_edges(graph, start.id, provider, excluded_edge)
                    for provider in providers
                ):
                    logger.warning(
                        f"Item {edge.item_required} not reachable before ITEM_GATE "
                        f"{edge.source}->{edge.target}"
                    )
                    return False

            if edge.edge_type == EdgeType.MULTI_LOCK and edge.token_count > 0:
                reachable = graph.get_reachable_nodes(start.id, excluded_edges=excluded_edge)
                reachable_tokens = sum(1 for token_node in token_nodes if token_node in reachable)
                if reachable_tokens < edge.token_count:
                    logger.warning(
                        f"MULTI_LOCK {edge.source}->{edge.target} requires {edge.token_count} "
                        f"tokens but only {reachable_tokens} are reachable pre-gate"
                    )
                    return False

            if edge.edge_type == EdgeType.STATE_BLOCK and edge.switches_required:
                reachable = graph.get_reachable_nodes(start.id, excluded_edges=excluded_edge)
                missing_switches = [sid for sid in edge.switches_required if sid not in reachable]
                if missing_switches:
                    logger.warning(
                        f"STATE_BLOCK {edge.source}->{edge.target} has unreachable switches before gate: "
                        f"{missing_switches}"
                    )
                    return False

        return True
    
    def _is_reachable_without(
        self,
        graph: MissionGraph,
        start: int,
        target: int,
        exclude: Set[int],
    ) -> bool:
        """BFS reachability check excluding certain nodes."""
        if start == target:
            return True
        
        visited = {start}
        queue = [start]
        
        while queue:
            current = queue.pop(0)
            
            for neighbor in graph._adjacency.get(current, []):
                if neighbor in visited or neighbor in exclude:
                    continue
                
                if neighbor == target:
                    return True
                
                visited.add(neighbor)
                queue.append(neighbor)
        
        return False

    def _is_reachable_without_edges(
        self,
        graph: MissionGraph,
        start: int,
        target: int,
        exclude_edges: Set[Tuple[int, int]],
    ) -> bool:
        """BFS reachability check excluding specific directed edges."""
        if start == target:
            return True

        visited = {start}
        queue = [start]

        while queue:
            current = queue.pop(0)

            for neighbor in graph._adjacency.get(current, []):
                if (current, neighbor) in exclude_edges:
                    continue
                if neighbor in visited:
                    continue
                if neighbor == target:
                    return True

                visited.add(neighbor)
                queue.append(neighbor)

        return False
    
    def _fix_lock_key_ordering(self, graph: MissionGraph) -> MissionGraph:
        """
        Repair invalid lock/key setups by downgrading unsatisfied gates.

        Note: Position swaps do not change graph reachability. This repair
        therefore edits progression constraints directly to restore consistency.
        """
        graph.sanitize()
        start = graph.get_start_node()
        if not start:
            return graph

        key_providers: Dict[int, List[int]] = defaultdict(list)
        for node in graph.nodes.values():
            if node.node_type in {NodeType.KEY, NodeType.BIG_KEY} and node.key_id is not None:
                key_providers[node.key_id].append(node.id)

        demoted_lock_nodes = 0
        demoted_lock_edges = 0

        # Repair lock nodes first.
        for lock in [n for n in graph.nodes.values() if n.node_type in {NodeType.LOCK, NodeType.BOSS_DOOR}]:
            key_id = lock.key_id
            if key_id is None:
                continue

            providers = key_providers.get(key_id, [])
            reachable = any(
                self._is_reachable_without(graph, start.id, provider_id, {lock.id})
                for provider_id in providers
            )
            if providers and reachable:
                continue

            lock.node_type = NodeType.EMPTY
            lock.key_id = None
            demoted_lock_nodes += 1

            # Remove keyed requirement on outgoing edge(s) from this lock node.
            for edge in graph.edges:
                if edge.source != lock.id:
                    continue
                if edge.edge_type in {EdgeType.LOCKED, EdgeType.BOSS_LOCKED}:
                    edge.edge_type = EdgeType.PATH
                    edge.key_required = None
                    edge.requires_key_count = 0

        graph.sanitize()

        # Repair keyed edges that still violate pre-gate reachability.
        for edge in graph.edges:
            if edge.edge_type not in {EdgeType.LOCKED, EdgeType.BOSS_LOCKED}:
                continue

            # Keep fungible key locks; they are validated via requires_key_count.
            if edge.edge_type == EdgeType.LOCKED and edge.requires_key_count > 0 and edge.key_required is None:
                continue

            key_id = edge.key_required
            providers = key_providers.get(key_id, []) if key_id is not None else []
            reachable = any(
                self._is_reachable_without_edges(graph, start.id, provider_id, {(edge.source, edge.target)})
                for provider_id in providers
            )
            if providers and reachable:
                continue

            edge.edge_type = EdgeType.PATH
            edge.key_required = None
            edge.requires_key_count = 0
            demoted_lock_edges += 1

        graph.sanitize()
        if demoted_lock_nodes > 0 or demoted_lock_edges > 0:
            graph.record_repair(
                "lock_key_repairs",
                amount=int(demoted_lock_nodes + demoted_lock_edges),
            )
            logger.info(
                "Lock-key repair: demoted %d lock nodes and %d lock edges",
                demoted_lock_nodes,
                demoted_lock_edges,
            )
        return graph

    def fix_lock_key_ordering(self, graph: MissionGraph) -> MissionGraph:
        """Public wrapper for lock-key consistency repair."""
        return self._fix_lock_key_ordering(graph)

    def _repair_progression_constraints(self, graph: MissionGraph) -> MissionGraph:
        """
        Best-effort repair for progression constraints after generation.

        Strategy: preserve topology where possible; relax only unsatisfied gate
        requirements that would cause invalid progression.
        """
        graph.sanitize()
        start = graph.get_start_node()
        if not start:
            return graph

        key_providers: Dict[int, List[int]] = defaultdict(list)
        for node in graph.nodes.values():
            if node.node_type in {NodeType.KEY, NodeType.BIG_KEY} and node.key_id is not None:
                key_providers[node.key_id].append(node.id)

        item_providers: Dict[str, List[int]] = defaultdict(list)
        for node in graph.nodes.values():
            if node.node_type == NodeType.ITEM and node.item_type:
                item_providers[str(node.item_type)].append(node.id)

        token_nodes = [n.id for n in graph.nodes.values() if n.node_type == NodeType.TOKEN]
        key_nodes = [
            n.id for n in graph.nodes.values()
            if n.node_type in {NodeType.KEY, NodeType.BIG_KEY}
        ]

        relaxed_edges = 0

        for edge in graph.edges:
            excluded_edge = {(edge.source, edge.target)}

            # Malformed lock edges: neither specific key nor fungible key budget.
            if (
                edge.edge_type in {EdgeType.LOCKED, EdgeType.BOSS_LOCKED}
                and edge.key_required is None
                and edge.requires_key_count <= 0
            ):
                edge.edge_type = EdgeType.PATH
                edge.key_required = None
                edge.requires_key_count = 0
                relaxed_edges += 1
                continue

            # Key-specific lock handling.
            if edge.edge_type in {EdgeType.LOCKED, EdgeType.BOSS_LOCKED} and edge.key_required is not None:
                providers = key_providers.get(edge.key_required, [])
                reachable = any(
                    self._is_reachable_without_edges(graph, start.id, provider, excluded_edge)
                    for provider in providers
                )
                if not providers or not reachable:
                    edge.edge_type = EdgeType.PATH
                    edge.key_required = None
                    edge.requires_key_count = 0
                    relaxed_edges += 1

            # Fungible locks.
            if edge.requires_key_count > 0:
                reachable = graph.get_reachable_nodes(start.id, excluded_edges=excluded_edge)
                reachable_keys = sum(1 for key_node in key_nodes if key_node in reachable)
                if reachable_keys <= 0:
                    edge.edge_type = EdgeType.PATH
                    edge.key_required = None
                    edge.requires_key_count = 0
                    relaxed_edges += 1
                elif reachable_keys < edge.requires_key_count:
                    edge.requires_key_count = reachable_keys
                    relaxed_edges += 1

            # Item gates.
            if edge.edge_type == EdgeType.ITEM_GATE and edge.item_required:
                providers = item_providers.get(str(edge.item_required), [])
                reachable = any(
                    self._is_reachable_without_edges(graph, start.id, provider, excluded_edge)
                    for provider in providers
                )
                if not providers or not reachable:
                    edge.edge_type = EdgeType.PATH
                    edge.item_required = None
                    relaxed_edges += 1

            # Token locks.
            if edge.edge_type == EdgeType.MULTI_LOCK and edge.token_count > 0:
                reachable = graph.get_reachable_nodes(start.id, excluded_edges=excluded_edge)
                reachable_tokens = sum(1 for token_node in token_nodes if token_node in reachable)
                if reachable_tokens <= 0:
                    edge.edge_type = EdgeType.PATH
                    edge.token_count = 0
                    relaxed_edges += 1
                elif reachable_tokens < edge.token_count:
                    edge.token_count = reachable_tokens
                    relaxed_edges += 1

            # Switch gates / batteries.
            if edge.edge_type == EdgeType.STATE_BLOCK and edge.switches_required:
                reachable = graph.get_reachable_nodes(start.id, excluded_edges=excluded_edge)
                kept_switches = [sid for sid in edge.switches_required if sid in reachable]
                if not kept_switches:
                    edge.edge_type = EdgeType.PATH
                    edge.switches_required = []
                    edge.battery_id = None
                    relaxed_edges += 1
                elif len(kept_switches) != len(edge.switches_required):
                    edge.switches_required = kept_switches
                    relaxed_edges += 1

        graph.sanitize()
        if relaxed_edges > 0:
            graph.record_repair("progression_repairs", amount=int(relaxed_edges))
            logger.info("Progression repair relaxed %d edge constraints", relaxed_edges)
        return graph

    def repair_progression_constraints(self, graph: MissionGraph) -> MissionGraph:
        """Public wrapper for progression-constraint repair."""
        return self._repair_progression_constraints(graph)

    def _repair_wave3_constraints(self, graph: MissionGraph) -> MissionGraph:
        """
        Best-effort normalization for Wave 3 quality constraints.

        This pass only relaxes/normalizes metadata and does not remove critical
        structure unless required to regain consistency.
        """
        graph.sanitize()
        start = graph.get_start_node()
        changes = 0

        # Normalize tutorial progression by nearest pedagogical successors.
        tutorial_nodes = [n for n in graph.nodes.values() if n.is_tutorial]
        pedagogical_types = {NodeType.COMBAT_PUZZLE, NodeType.COMPLEX_PUZZLE}
        for tutorial in tutorial_nodes:
            successors = [
                n for n in graph.get_successors(tutorial.id, depth=3)
                if n.node_type in pedagogical_types
            ]
            if len(successors) < 2:
                continue
            successors.sort(key=lambda n: graph.get_shortest_path_length(tutorial.id, n.id))
            first, second = successors[0], successors[1]
            if first.difficulty > second.difficulty:
                first.difficulty, second.difficulty = second.difficulty, first.difficulty
                first.difficulty_rating = "MODERATE"
                second.difficulty_rating = "HARD"
                changes += 1

        # Repair battery edges by keeping only reachable switches pre-gate.
        if start is not None:
            for edge in graph.edges:
                if edge.battery_id is None:
                    continue
                reachable = graph.get_reachable_nodes(
                    start.id,
                    excluded_edges={(edge.source, edge.target)},
                )
                kept_switches = [sid for sid in edge.switches_required if sid in reachable]
                if not kept_switches:
                    edge.edge_type = EdgeType.PATH
                    edge.switches_required = []
                    edge.battery_id = None
                    changes += 1
                elif len(kept_switches) != len(edge.switches_required):
                    edge.switches_required = kept_switches
                    changes += 1

            # Resource farms should only remain if they are reachable pre-gate.
            farms = [n for n in graph.nodes.values() if n.node_type == NodeType.RESOURCE_FARM]
            for farm in farms:
                if not farm.drops_resource:
                    continue
                related_gates = [e for e in graph.edges if e.item_required == farm.drops_resource]
                reachable_for_all = True
                for gate in related_gates:
                    reachable = graph.get_reachable_nodes(
                        start.id,
                        excluded_edges={(gate.source, gate.target)},
                    )
                    if farm.id not in reachable:
                        reachable_for_all = False
                        break
                if not reachable_for_all:
                    farm.node_type = NodeType.EMPTY
                    farm.drops_resource = None
                    farm.difficulty_rating = "MODERATE"
                    farm.tension_value = 0.5
                    changes += 1

        graph.sanitize()
        if changes > 0:
            graph.record_repair("wave3_repairs", amount=int(changes))
            logger.info("Wave3 repair normalized %d quality constraints", changes)
        return graph
    
    def _layout_graph(self, graph: MissionGraph) -> MissionGraph:
        """Apply a simple layout algorithm to position nodes."""
        graph.sanitize()
        start = graph.get_start_node()
        if not start:
            return graph
        
        # BFS to assign layers
        layers = {start.id: 0}
        queue = [start.id]
        
        while queue:
            current = queue.pop(0)
            current_layer = layers[current]
            
            for neighbor in graph._adjacency.get(current, []):
                if neighbor not in layers:
                    layers[neighbor] = current_layer + 1
                    queue.append(neighbor)
        
        # Group by layer
        layer_nodes = defaultdict(list)
        for node_id, layer in layers.items():
            layer_nodes[layer].append(node_id)
        
        # Position nodes
        for layer, nodes in layer_nodes.items():
            for i, node_id in enumerate(nodes):
                # Skip nodes that were removed by rules (e.g., FormBigRoomRule, PruneGraphRule)
                if node_id not in graph.nodes:
                    continue
                
                offset = i - len(nodes) // 2
                current_pos = graph.nodes[node_id].position
                floor = current_pos[2] if len(current_pos) > 2 else 0
                # Use layout constants for consistent spacing
                graph.nodes[node_id].position = (
                    layer * LAYOUT_LAYER_SPACING,
                    offset * LAYOUT_OFFSET_SPACING + LAYOUT_BASE_OFFSET,
                    floor
                )
        
        return graph


# ============================================================================
# INTEGRATION WITH CONDITION ENCODER
# ============================================================================

def graph_to_gnn_input(
    graph: MissionGraph,
    current_node_idx: Optional[int] = None,
) -> Dict[str, Tensor]:
    """
    Convert MissionGraph to tensors for GNN conditioning.
    
    Args:
        graph: MissionGraph to convert
        current_node_idx: Current node being generated (for local context)
        
    Returns:
        Dict with:
            - edge_index: [2, E] edges
            - node_features: [N, D] features
            - tpe: [N, 8] topological encoding
            - current_node: int
    """
    edge_index, node_features = graph.to_tensor()
    tpe = graph.compute_tpe()
    
    return {
        'edge_index': edge_index,
        'node_features': node_features,
        'tpe': tpe,
        'current_node': current_node_idx or 0,
        'adjacency': graph.to_adjacency_matrix(),
    }


# ============================================================================
# TEST
# ============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Mission Grammar...")
    
    # Create grammar
    grammar = MissionGrammar(seed=42)
    
    # Generate mission graph
    graph = grammar.generate(
        difficulty=Difficulty.MEDIUM,
        num_rooms=8,
        max_keys=2,
    )
    
    print(f"\nGenerated graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    
    # Print nodes
    print("\nNodes:")
    for node in sorted(graph.nodes.values(), key=lambda n: n.id):
        key_info = f" (key_id={node.key_id})" if node.key_id else ""
        print(f"  {node.id}: {node.node_type.name} at {node.position}{key_info}")
    
    # Print edges
    print("\nEdges:")
    for edge in graph.edges:
        key_req = f" [requires key {edge.key_required}]" if edge.key_required else ""
        print(f"  {edge.source} -> {edge.target} ({edge.edge_type.name}){key_req}")
    
    # Validate
    valid = grammar.validate_lock_key_ordering(graph)
    print(f"\nLock-key ordering valid: {valid}")
    
    # Convert to tensors
    gnn_input = graph_to_gnn_input(graph, current_node_idx=0)
    print(f"\nGNN Input:")


# ============================================================================
# THESIS UPGRADES: Advanced Production Rules
# ============================================================================

class MergeRule(ProductionRule):
    """
    THESIS UPGRADE #1: Create shortcuts by merging two separate branches.
    
    Finds two non-adjacent nodes and connects them with a shortcut edge,
    creating cycles in the dungeon topology (loops for backtracking).
    
    Research: Dormans (2011) - Cyclic dungeon graphs improve player agency.
    """
    
    def __init__(self):
        super().__init__("MergeShortcut", weight=0.5)
    
    def can_apply(self, graph: MissionGraph, context: Dict[str, Any]) -> bool:
        """Check if we can find two nodes to merge."""
        # Need at least 4 nodes to make meaningful loop closures.
        if len(graph.nodes) < 4:
            return False

        start = graph.get_start_node()
        goal = graph.get_goal_node()
        protected_ids = {
            nid
            for nid in [start.id if start else None, goal.id if goal else None]
            if nid is not None
        }
        max_loop_span = max(3, int(round(0.35 * float(max(1, len(graph.nodes))))))

        # Check if any valid pairs exist.
        nodes = list(graph.nodes.keys())
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                if node1 in protected_ids or node2 in protected_ids:
                    continue
                # Check not already adjacent
                if node2 not in graph._adjacency.get(node1, []):
                    # Check both have degree < 3 (room for another connection)
                    if (len(graph._adjacency.get(node1, [])) < 3 and 
                        len(graph._adjacency.get(node2, [])) < 3):
                        dist = self._graph_distance(graph, node1, node2)
                        if 2 <= dist <= max_loop_span:
                            return True
        return False
    
    def apply(self, graph: MissionGraph, context: Dict[str, Any]) -> MissionGraph:
        """Add a loop-closure edge between two separate branches."""
        rng = context.get('rng') or random
        start = graph.get_start_node()
        goal = graph.get_goal_node()
        protected_ids = {
            nid
            for nid in [start.id if start else None, goal.id if goal else None]
            if nid is not None
        }
        max_loop_span = max(3, int(round(0.35 * float(max(1, len(graph.nodes))))))
        candidates = []
        nodes = list(graph.nodes.keys())
        
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                if node1 in protected_ids or node2 in protected_ids:
                    continue
                if node2 not in graph._adjacency.get(node1, []):
                    if (len(graph._adjacency.get(node1, [])) < 3 and 
                        len(graph._adjacency.get(node2, [])) < 3):
                        # Keep loop closures local-to-mid range to avoid
                        # collapsing the main progression path.
                        dist = self._graph_distance(graph, node1, node2)
                        if 2 <= dist <= max_loop_span:
                            candidates.append((node1, node2, dist))
        
        if not candidates:
            return graph
        
        # Prefer longer admissible loop closures.
        candidates.sort(key=lambda x: x[2], reverse=True)
        node1, node2, dist = candidates[0]
        
        # Add as PATH: Zelda-like loops are usually regular doors/corridors.
        graph.add_edge(node1, node2, EdgeType.PATH)
        if graph.edges:
            metadata = graph.edges[-1].metadata if isinstance(graph.edges[-1].metadata, dict) else {}
            metadata["loop_closure"] = True
            metadata["loop_span"] = int(dist)
            graph.edges[-1].metadata = metadata
        logger.info("MergeRule: Added loop closure %s -> %s (span=%s)", node1, node2, dist)
        return graph
    
    def _graph_distance(self, graph: MissionGraph, start: int, end: int) -> int:
        """BFS to find shortest path distance."""
        if start == end:
            return 0
        
        visited = {start}
        queue = [(start, 0)]
        
        while queue:
            current, dist = queue.pop(0)
            for neighbor in graph._adjacency.get(current, []):
                if neighbor == end:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        
        return 999  # Not connected


class InsertSwitchRule(ProductionRule):
    """
    THESIS UPGRADE #2: Add switch nodes that control ON/OFF gates.
    
    Creates dynamic topology where paths open after activating switches.
    Implements global state changes in dungeon progression.
    
    Research: Smith & Mateas (2011) - State-dependent level design.
    """
    
    def __init__(self):
        super().__init__("InsertSwitch", weight=0.4)
    
    def can_apply(self, graph: MissionGraph, context: Dict[str, Any]) -> bool:
        """Check if we have edges that could become gated."""
        # Need at least 4 nodes and some normal edges
        if len(graph.nodes) < 4:
            return False
        normal_edges = [e for e in graph.edges if e.edge_type == EdgeType.PATH]
        return len(normal_edges) > 0
    
    def apply(self, graph: MissionGraph, context: Dict[str, Any]) -> MissionGraph:
        """Insert switch + gated edge."""
        rng = context.get('rng') or random
        
        # Find suitable edge to gate
        normal_edges = [(i, e) for i, e in enumerate(graph.edges) 
                       if e.edge_type == EdgeType.PATH]
        if not normal_edges:
            return graph
        
        edge_idx, edge = rng.choice(normal_edges)
        
        # Change edge to ON_OFF_GATE
        graph.edges[edge_idx].edge_type = EdgeType.ON_OFF_GATE
        
        # Add switch node in a separate branch
        switch_id = max(graph.nodes.keys()) + 1
        switch_node = MissionNode(
            id=switch_id,
            node_type=NodeType.SWITCH,
            position=(rng.randint(0, 5), rng.randint(0, 5), 0),
            difficulty=context.get('difficulty', 0.5) * 0.6,
            switch_id=switch_id,  # Self-referencing switch ID
        )
        graph.add_node(switch_node)
        
        # Connect switch to graph (not near the gated edge)
        other_nodes = [n for n in graph.nodes.keys() 
                      if n not in [edge.source, edge.target, switch_id]]
        if other_nodes:
            anchor = rng.choice(other_nodes)
            graph.add_edge(anchor, switch_id, EdgeType.PATH)
        
        logger.info(f"InsertSwitchRule: Switch {switch_id} controls edge {edge.source}->{edge.target}")
        return graph


class AddBossGauntlet(ProductionRule):
    """
    THESIS UPGRADE #3: Enforce Big Key -> Boss Door -> Goal hierarchy.
    
    Ensures the final challenge requires backtracking for the Big Key,
    enforcing the classic Zelda dungeon structure.
    
    Research: Treanor et al. (2015) - Lock-and-key design patterns.
    """
    
    def __init__(self):
        super().__init__("AddBossGauntlet", weight=1.0)
    
    def can_apply(self, graph: MissionGraph, context: Dict[str, Any]) -> bool:
        """
        Apply at most once per mission graph and only when GOAL has an incoming edge.
        """
        has_goal = any(n.node_type == NodeType.GOAL for n in graph.nodes.values())
        if not has_goal:
            return False
        if any(n.node_type == NodeType.BOSS_DOOR for n in graph.nodes.values()):
            return False
        if any(e.edge_type == EdgeType.BOSS_LOCKED for e in graph.edges):
            return False

        goal_nodes = [n for n in graph.nodes.values() if n.node_type == NodeType.GOAL]
        goal_id = goal_nodes[0].id if goal_nodes else None
        if goal_id is None:
            return False
        return any(e.target == goal_id for e in graph.edges)
    
    def apply(self, graph: MissionGraph, context: Dict[str, Any]) -> MissionGraph:
        """Insert Boss Door before Goal, spawn Big Key far away."""
        rng = context.get('rng') or random
        graph.sanitize()
        
        # Find goal node
        goal_nodes = [n for n in graph.nodes.values() if n.node_type == NodeType.GOAL]
        if not goal_nodes:
            return graph
        
        goal = goal_nodes[0]
        
        # Find predecessor of goal
        preds = [src for src, tgt in [(e.source, e.target) for e in graph.edges] 
                if tgt == goal.id]
        if not preds:
            return graph
        
        pred = preds[0]
        
        # Create Boss Door
        boss_door_id = max(graph.nodes.keys()) + 1
        goal_pos = goal.position
        boss_door = MissionNode(
            id=boss_door_id,
            node_type=NodeType.BOSS_DOOR,
            position=(goal_pos[0] - 1, goal_pos[1], goal_pos[2] if len(goal_pos) > 2 else 0),
            difficulty=0.9,
            key_id=boss_door_id,  # Requires big key
        )
        graph.add_node(boss_door)
        
        # Rewire: pred -> boss_door -> goal
        edge_to_remove = next((i for i, e in enumerate(graph.edges) 
                              if e.source == pred and e.target == goal.id), None)
        if edge_to_remove is not None:
            graph.edges.pop(edge_to_remove)
        
        graph.add_edge(pred, boss_door_id, EdgeType.BOSS_LOCKED, key_required=boss_door_id)
        graph.add_edge(boss_door_id, goal.id, EdgeType.PATH)
        graph.sanitize()
        
        # Spawn Big Key in a node guaranteed reachable before the boss lock.
        start = graph.get_start_node()
        excluded_edge = {(pred, boss_door_id)}
        excluded_nodes = {boss_door_id, goal.id}
        pre_lock_reachable: Set[int] = set()
        if start is not None:
            pre_lock_reachable = graph.get_reachable_nodes(
                start.id,
                excluded_edges=excluded_edge,
                excluded_nodes=excluded_nodes,
            )

        def _reachable_without_edges(source: int, target: int) -> bool:
            if source == target:
                return True
            visited = {source}
            queue = [source]
            while queue:
                current = queue.pop(0)
                for neighbor in graph._adjacency.get(current, []):
                    if (current, neighbor) in excluded_edge:
                        continue
                    if neighbor in visited:
                        continue
                    if neighbor == target:
                        return True
                    visited.add(neighbor)
                    queue.append(neighbor)
            return False

        disallowed = {goal.id, boss_door_id}
        if start is not None:
            disallowed.add(start.id)

        candidates: List[int] = []
        for node_id in pre_lock_reachable:
            if node_id in disallowed:
                continue
            node = graph.nodes.get(node_id)
            if node is None:
                continue
            if node.node_type in {NodeType.GOAL, NodeType.BOSS_DOOR}:
                continue
            # Prefer placements that still allow returning to the boss approach.
            if _reachable_without_edges(node_id, pred):
                candidates.append(node_id)

        # Fallback to any pre-lock-reachable node if return path constraint is too strict.
        if not candidates:
            candidates = [nid for nid in pre_lock_reachable if nid not in disallowed and nid in graph.nodes]

        # Last-resort fallback: hang Big Key directly off pre-door predecessor.
        if not candidates:
            candidates = [pred]

        def _score(node_id: int) -> Tuple[int, int]:
            dist_from_start = -1
            if start is not None:
                dist_from_start = graph.get_shortest_path_length(start.id, node_id)
            dist_to_pred = graph.get_shortest_path_length(node_id, pred)
            return (max(0, dist_from_start), max(0, dist_to_pred))

        anchor_id = max(candidates, key=_score)
        anchor = graph.nodes[anchor_id]

        big_key_id = max(graph.nodes.keys()) + 1
        anchor_pos = anchor.position
        big_key = MissionNode(
            id=big_key_id,
            node_type=NodeType.BIG_KEY,
            position=(
                anchor_pos[0],
                anchor_pos[1] + 1,
                anchor_pos[2] if len(anchor_pos) > 2 else 0,
            ),
            difficulty=0.7,
            key_id=boss_door_id,  # Opens boss door
        )
        graph.add_node(big_key)
        graph.add_edge(anchor_id, big_key_id, EdgeType.PATH)

        # Register key-lock mapping
        graph._key_to_lock[boss_door_id] = boss_door_id

        distance_hint = graph.get_shortest_path_length(anchor_id, pred)
        logger.info(
            "AddBossGauntlet: Boss Door %s, Big Key %s anchored at %s (return_dist=%s)",
            boss_door_id,
            big_key_id,
            anchor_id,
            distance_hint,
        )
        
        return graph


class AddItemGateRule(ProductionRule):
    """
    Add item-based gates (requires specific items like BOMB, HOOKSHOT, etc.).
    
    Creates an ITEM node and an ITEM_GATE edge that requires that specific
    item to pass. Ensures the item is obtainable before the gate is encountered.
    
    Similar to key-lock mechanics but uses named items instead of key IDs.
    """
    
    def __init__(self):
        super().__init__("AddItemGate", weight=0.4)
    
    def can_apply(self, graph: MissionGraph, context: Dict[str, Any]) -> bool:
        """Can apply if we have edges that could become gated."""
        # Need at least 3 nodes and normal edges
        if len(graph.nodes) < 3:
            return False
        normal_edges = [e for e in graph.edges if e.edge_type == EdgeType.PATH]
        return len(normal_edges) > 1
    
    def apply(self, graph: MissionGraph, context: Dict[str, Any]) -> MissionGraph:
        """Insert ITEM and ITEM_GATE on the path."""
        rng = context.get('rng') or random
        
        # Choose item type
        item_types = ["BOMB", "HOOKSHOT", "BOW", "FIRE_ROD", "ICE_ROD"]
        item_name = rng.choice(item_types)
        
        # Find an edge to place the ITEM
        normal_edges = [(i, e) for i, e in enumerate(graph.edges) 
                       if e.edge_type == EdgeType.PATH]
        if len(normal_edges) < 2:
            return graph
        
        item_edge_idx, item_edge = rng.choice(normal_edges)
        
        # Create ITEM node
        item_id = max(graph.nodes.keys()) + 1
        item_node = MissionNode(
            id=item_id,
            node_type=NodeType.ITEM,
            position=self._interpolate_pos(graph, item_edge.source, item_edge.target, 0.4),
            difficulty=context.get('difficulty', 0.5) * 0.6,
            item_type=item_name,  # Store what item this provides
        )
        graph.add_node(item_node)
        
        # Insert ITEM on the edge
        graph.edges.pop(item_edge_idx)
        graph.add_edge(item_edge.source, item_id, EdgeType.PATH)
        graph.add_edge(item_id, item_edge.target, EdgeType.PATH)
        
        # Now find a LATER edge for the ITEM_GATE.
        # Candidate must keep the item reachable pre-gate to avoid immediate
        # progression repair.
        start = graph.get_start_node()
        gate_candidates: List[Tuple[int, MissionEdge]] = []
        for i, e in enumerate(graph.edges):
            if e.edge_type != EdgeType.PATH:
                continue
            if e.source == item_id:
                continue
            # Gate should appear downstream from the item branch.
            if graph.get_shortest_path_length(item_id, e.source) <= 0:
                continue
            if start is not None:
                reachable = graph.get_reachable_nodes(
                    start.id,
                    excluded_edges={(e.source, e.target)},
                )
                # Item provider and gate entrance must both be reachable before
                # the gate edge is traversable.
                if item_id not in reachable or e.source not in reachable:
                    continue
            gate_candidates.append((i, e))
        
        if gate_candidates:
            gate_edge_idx, gate_edge = rng.choice(gate_candidates)
            
            # Create ITEM_GATE node
            gate_id = max(graph.nodes.keys()) + 1
            gate_node = MissionNode(
                id=gate_id,
                node_type=NodeType.EMPTY,  # Just a connector with special edge
                position=self._interpolate_pos(graph, gate_edge.source, gate_edge.target, 0.6),
                difficulty=context.get('difficulty', 0.5) * 0.7,
                required_item=item_name,  # References the required item
            )
            graph.add_node(gate_node)
            
            # Replace edge with gated version
            graph.edges = [e for i, e in enumerate(graph.edges) if i != gate_edge_idx]
            graph.add_edge(gate_edge.source, gate_id, EdgeType.PATH)
            
            # Create gated edge
            gated_edge = MissionEdge(
                source=gate_id,
                target=gate_edge.target,
                edge_type=EdgeType.ITEM_GATE,
                item_required=item_name,
            )
            graph.edges.append(gated_edge)
            graph._adjacency[gate_id].append(gate_edge.target)
            
            logger.info(f"AddItemGateRule: Item {item_name} at {item_id}, gate at {gate_id}")
        else:
            logger.debug(
                "AddItemGateRule: No pre-gate-valid edge found after placing item %s at %s",
                item_name,
                item_id,
            )
        
        return graph
    
    def _interpolate_pos(
        self,
        graph: MissionGraph,
        src: int,
        tgt: int,
        t: float,
    ) -> Tuple[int, int, int]:
        """Interpolate position between two nodes."""
        src_pos = graph.nodes[src].position
        tgt_pos = graph.nodes[tgt].position
        z = src_pos[2] if len(src_pos) > 2 else 0
        return (
            int(src_pos[0] * (1 - t) + tgt_pos[0] * t),
            int(src_pos[1] * (1 - t) + tgt_pos[1] * t),
            z,
        )


class CreateHubRule(ProductionRule):
    """
    Create a central hub room with multiple branches.
    
    Selects a node with low degree and forces it to become a hub with
    4 connections by attaching multiple branches. Hubs create interesting
    choice points and central areas for backtracking.
    """
    
    def __init__(self):
        super().__init__("CreateHub", weight=0.3)
    
    def can_apply(self, graph: MissionGraph, context: Dict[str, Any]) -> bool:
        """Can apply if there's a node with degree <= 2."""
        candidates = graph.get_nodes_with_degree_less_than(3)
        # Exclude START and GOAL
        candidates = [
            n for n in candidates
            if n.node_type not in [NodeType.START, NodeType.GOAL]
        ]
        return len(candidates) > 0
    
    def apply(self, graph: MissionGraph, context: Dict[str, Any]) -> MissionGraph:
        """Convert a node into a hub with multiple branches."""
        rng = context.get('rng') or random
        
        # Find suitable hub candidate
        candidates = graph.get_nodes_with_degree_less_than(3)
        candidates = [
            n for n in candidates
            if n.node_type not in [NodeType.START, NodeType.GOAL, NodeType.BOSS_DOOR]
        ]
        
        if not candidates:
            return graph
        
        hub_node = rng.choice(candidates)
        hub_node.is_hub = True  # Mark as hub
        
        # Calculate how many branches to add (target degree 4)
        current_degree = graph.get_node_degree(hub_node.id)
        branches_to_add = max(0, 4 - current_degree)
        
        if branches_to_add == 0:
            return graph
        
        # Add branches
        hub_pos = hub_node.position
        floor = hub_pos[2] if len(hub_pos) > 2 else 0
        
        for i in range(branches_to_add):
            # Create branch with at least 2 nodes
            branch_start_id = max(graph.nodes.keys()) + 1
            angle = (2 * math.pi * i) / branches_to_add
            # Calculate hub spoke positions using circular layout
            offset_r = int(LAYOUT_HUB_RADIUS * math.cos(angle))
            offset_c = int(LAYOUT_HUB_RADIUS * math.sin(angle))
            
            branch_start = MissionNode(
                id=branch_start_id,
                node_type=rng.choice([NodeType.ENEMY, NodeType.PUZZLE, NodeType.EMPTY]),
                position=(hub_pos[0] + offset_r, hub_pos[1] + offset_c, floor),
                difficulty=context.get('difficulty', 0.5) * rng.uniform(0.6, 0.9),
            )
            graph.add_node(branch_start)
            graph.add_edge(hub_node.id, branch_start_id, EdgeType.PATH)
            
            # Add one more node to make branch meaningful (extended further from hub)
            branch_end_id = max(graph.nodes.keys()) + 1
            branch_end = MissionNode(
                id=branch_end_id,
                node_type=rng.choice([NodeType.ITEM, NodeType.KEY, NodeType.EMPTY]),
                position=(
                    hub_pos[0] + offset_r * LAYOUT_HUB_BRANCH_SPACING,
                    hub_pos[1] + offset_c * LAYOUT_HUB_BRANCH_SPACING,
                    floor
                ),
                difficulty=context.get('difficulty', 0.5) * rng.uniform(0.7, 1.0),
            )
            graph.add_node(branch_end)
            graph.add_edge(branch_start_id, branch_end_id, EdgeType.PATH)
        
        logger.info(f"CreateHubRule: Node {hub_node.id} -> hub with {branches_to_add} new branches")
        return graph


class AddStairsRule(ProductionRule):
    """
    Add vertical connections between floors using stairs.
    
    Creates STAIRS_DOWN and STAIRS_UP nodes that connect the same (x, y)
    position on different z levels. Enables multi-floor dungeon design.
    """
    
    def __init__(self):
        super().__init__("AddStairs", weight=0.25)
    
    def can_apply(self, graph: MissionGraph, context: Dict[str, Any]) -> bool:
        """Can apply if we have nodes on floor 0 (can add floor 1)."""
        # Need at least 3 nodes to justify adding a second floor
        if len(graph.nodes) < 3:
            return False
        
        # Check if we already have stairs (don't add too many)
        stairs_nodes = [
            n for n in graph.nodes.values()
            if n.node_type in [NodeType.STAIRS_UP, NodeType.STAIRS_DOWN]
        ]
        return len(stairs_nodes) < 2  # Max 2 pairs of stairs
    
    def apply(self, graph: MissionGraph, context: Dict[str, Any]) -> MissionGraph:
        """Add stairs connecting two floors."""
        rng = context.get('rng') or random
        
        # Find a node on floor 0 with low degree
        candidates = [
            n for n in graph.nodes.values()
            if len(n.position) > 2 and n.position[2] == 0
            and graph.get_node_degree(n.id) < 3
            and n.node_type not in [NodeType.START, NodeType.GOAL, NodeType.STAIRS_UP, NodeType.STAIRS_DOWN]
        ]
        
        if not candidates:
            return graph
        
        anchor_node = rng.choice(candidates)
        anchor_pos = anchor_node.position
        
        # Create STAIRS_DOWN node on floor 0
        stairs_down_id = max(graph.nodes.keys()) + 1
        stairs_down = MissionNode(
            id=stairs_down_id,
            node_type=NodeType.STAIRS_DOWN,
            position=(anchor_pos[0], anchor_pos[1], 0),
            difficulty=context.get('difficulty', 0.5) * 0.5,
        )
        graph.add_node(stairs_down)
        graph.add_edge(anchor_node.id, stairs_down_id, EdgeType.PATH)
        
        # Create STAIRS_UP node on floor 1 (same x, y but z=1)
        stairs_up_id = max(graph.nodes.keys()) + 1
        stairs_up = MissionNode(
            id=stairs_up_id,
            node_type=NodeType.STAIRS_UP,
            position=(anchor_pos[0], anchor_pos[1], 1),  # Different floor
            difficulty=context.get('difficulty', 0.5) * 0.5,
        )
        graph.add_node(stairs_up)
        
        # Connect stairs with special edge type
        stair_edge = MissionEdge(
            source=stairs_down_id,
            target=stairs_up_id,
            edge_type=EdgeType.STAIRS,
        )
        graph.edges.append(stair_edge)
        graph._adjacency[stairs_down_id].append(stairs_up_id)
        graph._adjacency[stairs_up_id].append(stairs_down_id)  # Bidirectional
        
        # Add a small room on floor 1
        room_id = max(graph.nodes.keys()) + 1
        room = MissionNode(
            id=room_id,
            node_type=rng.choice([NodeType.ITEM, NodeType.ENEMY, NodeType.PUZZLE]),
            position=(anchor_pos[0] + rng.randint(1, 2), anchor_pos[1], 1),
            difficulty=context.get('difficulty', 0.5) * 0.8,
        )
        graph.add_node(room)
        graph.add_edge(stairs_up_id, room_id, EdgeType.PATH)
        
        logger.info(f"AddStairsRule: Stairs at ({anchor_pos[0]}, {anchor_pos[1]}) connecting floors 0â†”1")
        return graph


class AddSecretRule(ProductionRule):
    """
    Add a secret/hidden room accessible via HIDDEN edge.
    
    Creates optional off-critical-path areas with rewards,
    accessible through hidden passages (bombable walls, fake walls, etc.).
    """
    
    def __init__(self):
        super().__init__("AddSecret", weight=0.35)
    
    def can_apply(self, graph: MissionGraph, context: Dict[str, Any]) -> bool:
        """Can apply if we have nodes that can have secrets attached."""
        # Need nodes with degree < 4
        candidates = graph.get_nodes_with_degree_less_than(4)
        return len(candidates) > 0
    
    def apply(self, graph: MissionGraph, context: Dict[str, Any]) -> MissionGraph:
        """Add a secret room with hidden connection."""
        rng = context.get('rng') or random
        
        # Find anchor point for secret
        candidates = graph.get_nodes_with_degree_less_than(4)
        candidates = [
            n for n in candidates
            if n.node_type not in [NodeType.START, NodeType.GOAL, NodeType.SECRET]
        ]
        
        if not candidates:
            return graph
        
        anchor_node = rng.choice(candidates)
        anchor_pos = anchor_node.position
        floor = anchor_pos[2] if len(anchor_pos) > 2 else 0
        
        # Create secret room
        secret_id = max(graph.nodes.keys()) + 1
        secret_node = MissionNode(
            id=secret_id,
            node_type=NodeType.SECRET,
            position=(anchor_pos[0] + rng.randint(-1, 1), 
                     anchor_pos[1] + rng.randint(2, 3),
                     floor),
            difficulty=context.get('difficulty', 0.5) * 0.6,
            is_secret=True,
        )
        graph.add_node(secret_node)
        
        # Connect with HIDDEN edge
        hidden_edge = MissionEdge(
            source=anchor_node.id,
            target=secret_id,
            edge_type=EdgeType.HIDDEN,
        )
        graph.edges.append(hidden_edge)
        graph._adjacency[anchor_node.id].append(secret_id)
        graph._adjacency[secret_id].append(anchor_node.id)  # Can return
        
        # Add reward in secret room (ITEM or extra KEY)
        reward_id = max(graph.nodes.keys()) + 1
        reward_node = MissionNode(
            id=reward_id,
            node_type=rng.choice([NodeType.ITEM, NodeType.ITEM, NodeType.KEY]),  # Bias toward items
            position=(secret_node.position[0] + 1, secret_node.position[1], floor),
            difficulty=context.get('difficulty', 0.5) * 0.4,
        )
        graph.add_node(reward_node)
        graph.add_edge(secret_id, reward_id, EdgeType.PATH)
        
        logger.info(f"AddSecretRule: Secret room {secret_id} hidden from {anchor_node.id}")
        return graph


class AddTeleportRule(ProductionRule):
    """
    Add a teleport/warp connection between distant nodes.
    
    Creates shortcuts via WARP edges that don't require spatial adjacency.
    Useful for late-game backtracking or connecting distant regions.
    """
    
    def __init__(self):
        super().__init__("AddTeleport", weight=0.2)
    
    def can_apply(self, graph: MissionGraph, context: Dict[str, Any]) -> bool:
        """Can apply if we have nodes that are far apart."""
        if len(graph.nodes) < 5:
            return False
        
        # Check if any nodes are topologically distant
        nodes = list(graph.nodes.keys())
        for i, node1 in enumerate(nodes[:len(nodes)//2]):
            for node2 in nodes[len(nodes)//2:]:
                dist = graph.get_shortest_path_length(node1, node2)
                if dist >= 4:  # Far enough to warrant teleport
                    return True
        return False
    
    def apply(self, graph: MissionGraph, context: Dict[str, Any]) -> MissionGraph:
        """Add teleport/warp between distant nodes."""
        rng = context.get('rng') or random
        
        # Find two nodes that are topologically far but could use a shortcut
        nodes = list(graph.nodes.keys())
        candidates = []
        
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                # Check topological distance
                dist = graph.get_shortest_path_length(node1, node2)
                if dist >= 4 and dist < 999:  # Far but connected
                    # Check both have degree < 3
                    if (graph.get_node_degree(node1) < 3 and 
                        graph.get_node_degree(node2) < 3):
                        candidates.append((node1, node2, dist))
        
        if not candidates:
            return graph
        
        # Prefer longest distances for most useful warps
        candidates.sort(key=lambda x: x[2], reverse=True)
        node1, node2, dist = candidates[0]
        
        # Add WARP edge (bidirectional)
        warp_edge_1 = MissionEdge(
            source=node1,
            target=node2,
            edge_type=EdgeType.WARP,
        )
        warp_edge_2 = MissionEdge(
            source=node2,
            target=node1,
            edge_type=EdgeType.WARP,
        )
        graph.edges.extend([warp_edge_1, warp_edge_2])
        graph._adjacency[node1].append(node2)
        graph._adjacency[node2].append(node1)
        
        logger.info(f"AddTeleportRule: Warp between {node1} â†” {node2} (saved {dist} hops)")
        return graph


class PruneGraphRule(ProductionRule):
    """
    Prune unnecessary nodes and simplify the graph.
    
    Detects chains of 3+ EMPTY nodes in sequence and merges or removes
    redundant ones. Simplifies overly complex branches while preserving
    interesting structure.
    """
    
    def __init__(self):
        super().__init__("PruneGraph", weight=0.15)
    
    def can_apply(self, graph: MissionGraph, context: Dict[str, Any]) -> bool:
        """Can apply if there are chains of EMPTY nodes."""
        # Look for sequences of EMPTY nodes
        empty_chains = self._find_empty_chains(graph)
        return len(empty_chains) > 0
    
    def apply(self, graph: MissionGraph, context: Dict[str, Any]) -> MissionGraph:
        """Simplify the graph by pruning empty chains."""
        rng = context.get('rng') or random
        
        empty_chains = self._find_empty_chains(graph)
        if not empty_chains:
            return graph
        
        # Prune the longest chain
        chain = max(empty_chains, key=len)
        if len(chain) < 3:
            return graph
        
        # Keep first and last, remove middle nodes
        keep_first = chain[0]
        keep_last = chain[-1]
        remove_nodes = chain[1:-1]
        
        # Reconnect: attach neighbors of removed nodes to keep_first and keep_last
        for node_id in remove_nodes:
            # Remove from graph
            if node_id in graph.nodes:
                del graph.nodes[node_id]
            
            # Remove edges involving this node
            graph.edges = [e for e in graph.edges 
                          if e.source != node_id and e.target != node_id]
            
            # Clean adjacency
            if node_id in graph._adjacency:
                del graph._adjacency[node_id]
            for adj_list in graph._adjacency.values():
                if node_id in adj_list:
                    adj_list.remove(node_id)
        
        # Ensure keep_first and keep_last are connected
        if keep_last not in graph._adjacency.get(keep_first, []):
            graph.add_edge(keep_first, keep_last, EdgeType.PATH)
        
        logger.info(f"PruneGraphRule: Pruned chain of {len(remove_nodes)} empty nodes")
        return graph
    
    def _find_empty_chains(self, graph: MissionGraph) -> List[List[int]]:
        """Find chains of EMPTY nodes connected in sequence."""
        chains = []
        visited = set()
        
        for node in graph.nodes.values():
            if node.node_type != NodeType.EMPTY:
                continue
            if node.id in visited:
                continue
            
            # Start a chain
            chain = [node.id]
            visited.add(node.id)
            
            # Extend forward
            current = node.id
            while True:
                neighbors = graph._adjacency.get(current, [])
                empty_neighbors = [
                    n for n in neighbors
                    if n not in visited
                    and n in graph.nodes
                    and graph.nodes[n].node_type == NodeType.EMPTY
                    and graph.get_node_degree(n) == 2  # Only linear connections
                ]
                
                if not empty_neighbors:
                    break
                
                next_node = empty_neighbors[0]
                chain.append(next_node)
                visited.add(next_node)
                current = next_node
            
            if len(chain) >= 3:
                chains.append(chain)
        
        return chains


# ============================================================================
# ADVANCED PRODUCTION RULES (Thesis-Grade Patterns)
# Based on Joris Dormans' "Unexplored" and Mark Brown's "Boss Keys" Analysis
# ============================================================================

class AddFungibleLockRule(ProductionRule):
    """
    ADVANCED RULE #1: Fungible Key Economy System
    
    Creates small keys that work as a CURRENCY (inventory count) rather than
    unique key-lock pairs. Player collects keys that increment a counter,
    and locked doors decrement the counter without requiring specific key IDs.
    
    Example: Zelda's small keys - any key opens any small key door.
    
    Research: Dormans (2011) - Resource management in procedural dungeons
    """
    
    def __init__(self):
        super().__init__("AddFungibleLock", weight=0.45)
    
    def can_apply(self, graph: MissionGraph, context: Dict[str, Any]) -> bool:
        """Can apply if we have edges that could become fungible locks."""
        if len(graph.nodes) < 4:
            return False
        # Check if we have PATH edges to convert
        normal_edges = [e for e in graph.edges if e.edge_type == EdgeType.PATH]
        return len(normal_edges) >= 2
    
    def apply(self, graph: MissionGraph, context: Dict[str, Any]) -> MissionGraph:
        """Add fungible key and lock using inventory count."""
        rng = context.get('rng') or random
        graph.sanitize()
        
        # Find an edge to place KEY
        normal_edges = [(i, e) for i, e in enumerate(graph.edges) 
                       if e.edge_type == EdgeType.PATH]
        if len(normal_edges) < 2:
            return graph
        
        key_edge_idx, key_edge = rng.choice(normal_edges)
        
        # Create KEY node
        key_id = max(graph.nodes.keys()) + 1
        key_node = MissionNode(
            id=key_id,
            node_type=NodeType.KEY,
            position=self._interpolate_pos(graph, key_edge.source, key_edge.target, 0.3),
            difficulty=context.get('difficulty', 0.5) * 0.5,
            # NO key_id - fungible keys don't have unique IDs
        )
        graph.add_node(key_node)
        
        # Insert KEY on edge
        graph.edges.pop(key_edge_idx)
        graph.add_edge(key_edge.source, key_id, EdgeType.PATH)
        graph.add_edge(key_id, key_edge.target, EdgeType.PATH)
        graph.sanitize()
        
        # Find a LATER edge for the lock.
        # Candidate must preserve pre-gate key reachability.
        start = graph.get_start_node()
        key_node_types = {NodeType.KEY, NodeType.BIG_KEY}
        lock_candidates: List[Tuple[int, MissionEdge]] = []
        for i, e in enumerate(graph.edges):
            if e.edge_type != EdgeType.PATH:
                continue
            if e.source == key_id:
                continue
            if graph.get_shortest_path_length(key_id, e.source) <= 0:
                continue
            if start is not None:
                reachable = graph.get_reachable_nodes(
                    start.id,
                    excluded_edges={(e.source, e.target)},
                )
                if key_id not in reachable or e.source not in reachable:
                    continue
                reachable_keys = sum(
                    1
                    for node_id in reachable
                    if (
                        node_id in graph.nodes
                        and graph.nodes[node_id].node_type in key_node_types
                    )
                )
                if reachable_keys < 1:
                    continue
            lock_candidates.append((i, e))
        
        if lock_candidates:
            lock_edge_idx, lock_edge = rng.choice(lock_candidates)
            
            # Convert edge to fungible lock (requires_key_count instead of key_id)
            graph.edges[lock_edge_idx].edge_type = EdgeType.LOCKED
            graph.edges[lock_edge_idx].requires_key_count = 1  # Requires any 1 key
            
            logger.info(f"AddFungibleLockRule: Fungible key at {key_id}, lock at edge {lock_edge.source}->{lock_edge.target}")
            return graph

        # No valid lock placement: rollback key insertion to avoid injecting
        # free progression resources without gating semantics.
        if key_id in graph.nodes:
            del graph.nodes[key_id]
        graph.edges = [
            e for e in graph.edges
            if not (
                (e.source == key_edge.source and e.target == key_id)
                or (e.source == key_id and e.target == key_edge.target)
            )
        ]
        graph.add_edge(key_edge.source, key_edge.target, EdgeType.PATH)
        graph.sanitize()
        logger.debug(
            "AddFungibleLockRule: No pre-gate-valid lock edge after key insertion; rolled back key %s",
            key_id,
        )
        
        return graph
    
    def _interpolate_pos(
        self,
        graph: MissionGraph,
        src: int,
        tgt: int,
        t: float,
    ) -> Tuple[int, int, int]:
        """Interpolate position between two nodes."""
        src_pos = graph.nodes[src].position
        tgt_pos = graph.nodes[tgt].position
        z = src_pos[2] if len(src_pos) > 2 else 0
        return (
            int(src_pos[0] * (1 - t) + tgt_pos[0] * t),
            int(src_pos[1] * (1 - t) + tgt_pos[1] * t),
            z,
        )


class FormBigRoomRule(ProductionRule):
    """
    ADVANCED RULE #2: Merge Nodes into Big Rooms (Great Halls)
    
    Merges two spatially adjacent nodes into a single larger room (2x1 or 2x2).
    Creates impressive "great hall" spaces that break the single-tile grid.
    
    Example: Zelda's large boss arenas, great halls in Cathedral-style dungeons.
    
    Research: Brown "Boss Keys" - Spatial variation in dungeon layout
    """
    
    def __init__(self):
        super().__init__("FormBigRoom", weight=0.3)
    
    def can_apply(self, graph: MissionGraph, context: Dict[str, Any]) -> bool:
        """Can apply if we have two adjacent connected nodes."""
        if len(graph.nodes) < 3:
            return False
        
        # Look for spatially adjacent nodes that are connected
        for edge in graph.edges:
            if edge.edge_type != EdgeType.PATH:
                continue
            node_a = graph.nodes.get(edge.source)
            node_b = graph.nodes.get(edge.target)
            if not node_a or not node_b:
                continue
            
            # Check spatial adjacency (Manhattan distance = 1 horizontally)
            dist = abs(node_a.position[0] - node_b.position[0]) + abs(node_a.position[1] - node_b.position[1])
            if dist <= 2:  # Adjacent or close
                # Not START/GOAL/BOSS_DOOR
                if (node_a.node_type not in [NodeType.START, NodeType.GOAL, NodeType.BOSS_DOOR] and
                    node_b.node_type not in [NodeType.START, NodeType.GOAL, NodeType.BOSS_DOOR]):
                    return True
        return False
    
    def apply(self, graph: MissionGraph, context: Dict[str, Any]) -> MissionGraph:
        """Merge two nodes into a big room."""
        rng = context.get('rng') or random
        
        # Find adjacent nodes to merge
        candidates = []
        for edge in graph.edges:
            if edge.edge_type != EdgeType.PATH:
                continue
            node_a = graph.nodes.get(edge.source)
            node_b = graph.nodes.get(edge.target)
            if not node_a or not node_b:
                continue
            
            dist = abs(node_a.position[0] - node_b.position[0]) + abs(node_a.position[1] - node_b.position[1])
            if dist <= 2:
                if (node_a.node_type not in [NodeType.START, NodeType.GOAL, NodeType.BOSS_DOOR] and
                    node_b.node_type not in [NodeType.START, NodeType.GOAL, NodeType.BOSS_DOOR]):
                    candidates.append((edge.source, edge.target))
        
        if not candidates:
            return graph
        
        node_a_id, node_b_id = rng.choice(candidates)
        node_a = graph.nodes[node_a_id]
        node_b = graph.nodes[node_b_id]
        
        # Merge node_b into node_a
        merged_pos = (
            min(node_a.position[0], node_b.position[0]),
            min(node_a.position[1], node_b.position[1]),
            node_a.position[2],
        )
        
        # Determine room size based on spatial relationship
        horizontal_dist = abs(node_a.position[1] - node_b.position[1])
        vertical_dist = abs(node_a.position[0] - node_b.position[0])
        
        if horizontal_dist >= vertical_dist:
            room_size = (2, 1)  # 2x1 horizontal
        else:
            room_size = (1, 2)  # 1x2 vertical
        
        # Update node_a to be the big room
        node_a.position = merged_pos
        node_a.room_size = room_size
        node_a.is_big_room = True
        
        # Transfer edges from node_b to node_a
        for i, edge in enumerate(graph.edges):
            if edge.source == node_b_id:
                graph.edges[i].source = node_a_id
            if edge.target == node_b_id:
                graph.edges[i].target = node_a_id
        
        # Remove node_b
        if node_b_id in graph.nodes:
            del graph.nodes[node_b_id]
        
        # Clean up adjacency
        if node_b_id in graph._adjacency:
            neighbors = graph._adjacency[node_b_id]
            del graph._adjacency[node_b_id]
            # Transfer to node_a
            for neighbor in neighbors:
                if neighbor != node_a_id and neighbor not in graph._adjacency.get(node_a_id, []):
                    graph._adjacency[node_a_id].append(neighbor)
        
        # Remove self-loops
        graph.edges = [e for e in graph.edges if e.source != e.target]
        
        logger.info(f"FormBigRoomRule: Merged nodes {node_a_id} and {node_b_id} into {room_size} big room")
        return graph


class AddValveRule(ProductionRule):
    """
    ADVANCED RULE #3: One-Way Valves in Cycles
    
    Detects cycles in the graph and converts one edge to ONE_WAY, creating
    a "valve" where you can't immediately backtrack but must loop around.
    
    Example: Zelda's ledges you can drop down but can't climb back up.
    
    Research: Dormans & Bakkes (2011) - Directed flow in cyclic graphs
    """
    
    def __init__(self):
        super().__init__("AddValve", weight=0.35)

    @staticmethod
    def _bfs_path(
        adjacency: Dict[int, List[int]],
        start_id: int,
        goal_id: int,
    ) -> Optional[List[int]]:
        """Shortest path over current directed adjacency."""
        if start_id == goal_id:
            return [start_id]
        visited = {start_id}
        queue: List[Tuple[int, List[int]]] = [(start_id, [start_id])]
        while queue:
            current, path = queue.pop(0)
            for neighbor in adjacency.get(current, []):
                if neighbor in visited:
                    continue
                new_path = path + [neighbor]
                if neighbor == goal_id:
                    return new_path
                visited.add(neighbor)
                queue.append((neighbor, new_path))
        return None

    def _critical_path_pairs(self, graph: MissionGraph) -> Set[Tuple[int, int]]:
        """
        Directed/undirected edge pairs on current START->GOAL path.

        Valves should prefer non-critical loops so directionality mechanics do
        not inflate primary progression path length.
        """
        start = graph.get_start_node()
        goal = graph.get_goal_node()
        if not start or not goal:
            return set()
        path = self._bfs_path(graph._adjacency, start.id, goal.id)
        if not path or len(path) < 2:
            return set()
        pairs: Set[Tuple[int, int]] = set()
        for i in range(len(path) - 1):
            a = int(path[i])
            b = int(path[i + 1])
            pairs.add((a, b))
            pairs.add((b, a))
        return pairs
    
    def can_apply(self, graph: MissionGraph, context: Dict[str, Any]) -> bool:
        """Can apply if there are cycles in the graph."""
        if len(graph.nodes) < 4:
            return False
        cycles = graph.detect_cycles()
        return len(cycles) > 0
    
    def apply(self, graph: MissionGraph, context: Dict[str, Any]) -> MissionGraph:
        """Convert one edge in a cycle to ONE_WAY."""
        rng = context.get('rng') or random
        
        cycles = graph.detect_cycles()
        if not cycles:
            return graph

        critical_pairs = self._critical_path_pairs(graph)
        start = graph.get_start_node()
        goal = graph.get_goal_node()
        protected_nodes = {n.id for n in [start, goal] if n is not None}

        cycle_order = list(range(len(cycles)))
        rng.shuffle(cycle_order)
        chosen: Optional[Tuple[int, MissionEdge]] = None

        for cycle_idx in cycle_order:
            cycle = cycles[cycle_idx]
            if len(cycle) < 3:
                continue

            # Make the ring explicit even when detect_cycles omits closing node.
            cycle_steps: List[Tuple[int, int]] = []
            for i in range(len(cycle) - 1):
                cycle_steps.append((int(cycle[i]), int(cycle[i + 1])))
            if int(cycle[0]) != int(cycle[-1]):
                cycle_steps.append((int(cycle[-1]), int(cycle[0])))

            all_candidates: List[Tuple[int, MissionEdge]] = []
            noncritical_candidates: List[Tuple[int, MissionEdge]] = []
            safe_candidates: List[Tuple[int, MissionEdge]] = []
            seen_edge_ids: Set[int] = set()

            for src, tgt in cycle_steps:
                for idx, edge in enumerate(graph.edges):
                    if idx in seen_edge_ids or edge.edge_type != EdgeType.PATH:
                        continue
                    if not (
                        (int(edge.source) == src and int(edge.target) == tgt)
                        or (int(edge.source) == tgt and int(edge.target) == src)
                    ):
                        continue
                    seen_edge_ids.add(idx)
                    candidate = (idx, edge)
                    all_candidates.append(candidate)
                    touches_protected = (edge.source in protected_nodes) or (edge.target in protected_nodes)
                    on_critical = (
                        (int(edge.source), int(edge.target)) in critical_pairs
                        or (int(edge.target), int(edge.source)) in critical_pairs
                    )
                    if not on_critical:
                        noncritical_candidates.append(candidate)
                        if not touches_protected:
                            safe_candidates.append(candidate)

            if safe_candidates:
                chosen = rng.choice(safe_candidates)
                break
            if noncritical_candidates:
                chosen = rng.choice(noncritical_candidates)
                break
            if all_candidates and chosen is None:
                # Fallback only when no safe/non-critical candidate exists.
                chosen = rng.choice(all_candidates)

        if chosen is None:
            return graph

        edge_idx, edge = chosen

        # Convert to ONE_WAY
        graph.edges[edge_idx].edge_type = EdgeType.ONE_WAY
        graph.edges[edge_idx].preferred_direction = "forward"
        if not isinstance(graph.edges[edge_idx].metadata, dict):
            graph.edges[edge_idx].metadata = {}
        graph.edges[edge_idx].metadata["valve_cycle"] = True
        
        # Remove backward adjacency (it's now one-way only)
        if edge.target in graph._adjacency:
            graph._adjacency[edge.target] = [
                n for n in graph._adjacency.get(edge.target, []) if int(n) != int(edge.source)
            ]
        
        logger.info(f"AddValveRule: Made edge {edge.source}->{edge.target} ONE_WAY in cycle")
        return graph


class AddForeshadowingRule(ProductionRule):
    """
    ADVANCED RULE #4: Visual Foreshadowing (Windows)
    
    Creates visual connections between physically close but topologically
    distant nodes. Player can see a reward through a window but must take
    a long path to reach it.
    
    Example: Zelda's window views of treasure through locked doors.
    
    Research: Brown "Boss Keys" - Environmental storytelling through layout
    """
    
    def __init__(self):
        super().__init__("AddForeshadowing", weight=0.25)
    
    def can_apply(self, graph: MissionGraph, context: Dict[str, Any]) -> bool:
        """Can apply if there are nodes close spatially but far topologically."""
        if len(graph.nodes) < 5:
            return False
        
        # Look for nodes with Manhattan distance â‰¤ 2 and path distance > 4
        nodes = list(graph.nodes.keys())
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                manhattan = graph.get_manhattan_distance(node1, node2)
                if manhattan <= 2:
                    path_dist = graph.get_shortest_path_length(node1, node2)
                    if path_dist > 4:
                        return True
        return False
    
    def apply(self, graph: MissionGraph, context: Dict[str, Any]) -> MissionGraph:
        """Add visual link between close but distant nodes."""
        rng = context.get('rng') or random
        
        # Find candidate pairs
        candidates = []
        nodes = list(graph.nodes.keys())
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                manhattan = graph.get_manhattan_distance(node1, node2)
                if manhattan <= 2 and manhattan > 0:
                    path_dist = graph.get_shortest_path_length(node1, node2)
                    if path_dist > 4:
                        candidates.append((node1, node2, path_dist))
        
        if not candidates:
            return graph
        
        # Prefer largest path distance for best foreshadowing
        candidates.sort(key=lambda x: x[2], reverse=True)
        node1, node2, dist = candidates[0]
        
        # Place reward at target node if not already interesting
        target_node = graph.nodes[node2]
        if target_node.node_type == NodeType.EMPTY:
            target_node.node_type = rng.choice([NodeType.TREASURE, NodeType.ITEM, NodeType.KEY])
        
        # Add VISUAL_LINK edge (non-traversable)
        visual_edge = MissionEdge(
            source=node1,
            target=node2,
            edge_type=EdgeType.VISUAL_LINK,
            is_window=True,
        )
        graph.edges.append(visual_edge)
        # Don't add to adjacency - it's not traversable!
        
        logger.info(f"AddForeshadowingRule: Visual link {node1}->{node2} (path={dist}, spatial=close)")
        return graph


class AddCollectionChallengeRule(ProductionRule):
    """
    ADVANCED RULE #5: Collect N Tokens Pattern (Tri-Force)
    
    Requires collecting multiple tokens scattered across different branches
    before progressing. Creates MULTI_LOCK edge requiring N tokens.
    
    Example: Zelda's Tri-Force pieces, Metroid's Chozo artifacts.
    
    Research: Treanor et al. (2015) - Collection mechanics in adventure games
    """
    
    def __init__(self):
        super().__init__("AddCollectionChallenge", weight=0.2)
    
    def can_apply(self, graph: MissionGraph, context: Dict[str, Any]) -> bool:
        """Can apply if we have a hub with multiple branches."""
        if len(graph.nodes) < 6:
            return False
        # Look for nodes with degree >= 3 (potential hubs)
        for node in graph.nodes.values():
            if graph.get_node_degree(node.id) >= 3:
                return True
        return False
    
    def apply(self, graph: MissionGraph, context: Dict[str, Any]) -> MissionGraph:
        """Add token collection challenge."""
        rng = context.get('rng') or random
        graph.sanitize()
        
        # Find hub nodes
        hubs = [n for n in graph.nodes.values() if graph.get_node_degree(n.id) >= 3]
        if not hubs:
            return graph
        
        hub = rng.choice(hubs)
        branches = graph.get_nodes_in_different_branches(hub.id)
        
        # Need at least 3 branches for tri-force pattern
        if len(branches) < 3:
            return graph
        
        num_tokens = min(3, len(branches))
        selected_branches = rng.sample(branches, num_tokens)
        
        # Place tokens in different branches
        token_ids = []
        for i, branch in enumerate(selected_branches):
            if not branch:
                continue
            
            # Place token at end of branch
            target_node_id = branch[-1] if branch else branch[0]
            
            # Create TOKEN node
            token_id = max(graph.nodes.keys()) + 1
            token_node = MissionNode(
                id=token_id,
                node_type=NodeType.TOKEN,
                position=graph.nodes[target_node_id].position,
                difficulty=context.get('difficulty', 0.5) * 0.6,
                token_id=f"TOKEN_{i}",
            )
            graph.add_node(token_node)
            graph.add_edge(target_node_id, token_id, EdgeType.PATH)
            token_ids.append(token_id)
        
        if len(token_ids) < 2:
            return graph  # Not enough tokens placed
        graph.sanitize()
        
        # Find an edge to convert to MULTI_LOCK (preferably near hub).
        # Candidate must keep all required tokens reachable before the gate.
        start = graph.get_start_node()
        normal_edges: List[Tuple[int, MissionEdge]] = []
        for i, e in enumerate(graph.edges):
            if e.edge_type != EdgeType.PATH:
                continue
            if graph.get_shortest_path_length(hub.id, e.source) <= 0:
                continue
            if start is not None:
                reachable = graph.get_reachable_nodes(
                    start.id,
                    excluded_edges={(e.source, e.target)},
                )
                if e.source not in reachable:
                    continue
                if not all(token_id in reachable for token_id in token_ids):
                    continue
            normal_edges.append((i, e))
        
        if normal_edges:
            lock_edge_idx, lock_edge = rng.choice(normal_edges)
            
            # Convert to MULTI_LOCK
            graph.edges[lock_edge_idx].edge_type = EdgeType.MULTI_LOCK
            graph.edges[lock_edge_idx].token_count = len(token_ids)
            
            logger.info(f"AddCollectionChallengeRule: {len(token_ids)} tokens required for MULTI_LOCK at {lock_edge.source}->{lock_edge.target}")
            return graph

        # No valid lock edge: rollback token-only inserts so this rule remains
        # semantically meaningful (collection + gate), not pure rewards.
        token_set = set(token_ids)
        for token_id in token_set:
            if token_id in graph.nodes:
                del graph.nodes[token_id]
        graph.edges = [
            e for e in graph.edges
            if e.source not in token_set and e.target not in token_set
        ]
        graph.sanitize()
        logger.debug(
            "AddCollectionChallengeRule: No pre-gate-valid MULTI_LOCK edge; rolled back %d tokens",
            len(token_ids),
        )
        
        return graph


class AddArenaRule(ProductionRule):
    """
    ADVANCED RULE #6: Combat Arenas with Shutters
    
    Creates trap rooms where doors close (SHUTTER edges) until enemies are
    cleared. One-way in, conditional exit based on combat.
    
    Example: Zelda's rooms where doors lock during combat.
    
    Research: Smith & Mateas (2011) - Dynamic challenge pacing
    """
    
    def __init__(self):
        super().__init__("AddArena", weight=0.3)
    
    def can_apply(self, graph: MissionGraph, context: Dict[str, Any]) -> bool:
        """Can apply if we have thoroughfare nodes (degree >= 2)."""
        if len(graph.nodes) < 4:
            return False
        for node in graph.nodes.values():
            if (graph.get_node_degree(node.id) >= 2 and
                node.node_type not in [NodeType.START, NodeType.GOAL, NodeType.BOSS_DOOR]):
                return True
        return False
    
    def apply(self, graph: MissionGraph, context: Dict[str, Any]) -> MissionGraph:
        """Convert node to combat arena."""
        rng = context.get('rng') or random
        
        # Find thoroughfare nodes
        candidates = [
            n for n in graph.nodes.values()
            if (graph.get_node_degree(n.id) >= 2 and
                n.node_type not in [NodeType.START, NodeType.GOAL, NodeType.BOSS_DOOR, NodeType.ARENA])
        ]
        
        if not candidates:
            return graph
        
        arena_node = rng.choice(candidates)
        
        # Mark as arena
        arena_node.node_type = NodeType.ARENA
        arena_node.is_arena = True
        
        # Convert incoming edges to SHUTTER type
        for i, edge in enumerate(graph.edges):
            if edge.target == arena_node.id and edge.edge_type == EdgeType.PATH:
                graph.edges[i].edge_type = EdgeType.SHUTTER
        
        logger.info(f"AddArenaRule: Node {arena_node.id} converted to combat arena with shutters")
        return graph


class AddSectorRule(ProductionRule):
    """
    ADVANCED RULE #7: Thematic Sectors (Wings/Zones)
    
    Groups nodes into thematic zones with consistent visual/mechanical themes.
    Creates sector-specific locks and coherent area progression.
    
    Example: Fire Temple, Water Temple, Forest Temple in Zelda OoT.
    
    Research: Dormans (2011) - Thematic coherence in procedural spaces
    """
    
    def __init__(self):
        super().__init__("AddSector", weight=0.25)
    
    def can_apply(self, graph: MissionGraph, context: Dict[str, Any]) -> bool:
        """Can apply if we have hub with branches."""
        if len(graph.nodes) < 6:
            return False
        # Need a hub or branch point
        for node in graph.nodes.values():
            if graph.get_node_degree(node.id) >= 2:
                return True
        return False
    
    def apply(self, graph: MissionGraph, context: Dict[str, Any]) -> MissionGraph:
        """Create thematic sector."""
        rng = context.get('rng') or random
        
        # Find branch points
        branch_points = [n for n in graph.nodes.values() if graph.get_node_degree(n.id) >= 2]
        if not branch_points:
            return graph
        
        branch_point = rng.choice(branch_points)
        
        # Generate chain of 5-8 nodes
        chain_length = rng.randint(5, 8)
        sector_id = max([n.sector_id for n in graph.nodes.values()], default=0) + 1
        sector_theme = rng.choice(["FIRE", "WATER", "ICE", "FOREST", "SHADOW", "SPIRIT"])
        
        # Start from branch point
        current_id = branch_point.id
        sector_nodes = [current_id]
        
        for i in range(chain_length):
            # Create new node in sector
            new_id = max(graph.nodes.keys()) + 1
            new_node = MissionNode(
                id=new_id,
                node_type=rng.choice([NodeType.ENEMY, NodeType.PUZZLE, NodeType.EMPTY]),
                position=(
                    branch_point.position[0] + i + 1,
                    branch_point.position[1] + rng.randint(-1, 1),
                    branch_point.position[2],
                ),
                difficulty=context.get('difficulty', 0.5) * rng.uniform(0.6, 0.9),
                sector_id=sector_id,
                sector_theme=sector_theme,
            )
            graph.add_node(new_node)
            graph.add_edge(current_id, new_id, EdgeType.PATH)
            sector_nodes.append(new_id)
            current_id = new_id
        
        # Tag all nodes in sector
        for node_id in sector_nodes:
            if node_id in graph.nodes:
                graph.nodes[node_id].sector_id = sector_id
                graph.nodes[node_id].sector_theme = sector_theme
        
        logger.info(f"AddSectorRule: Created {sector_theme} sector (ID={sector_id}) with {len(sector_nodes)} nodes")
        return graph


class AddEntangledBranchesRule(ProductionRule):
    """
    ADVANCED RULE #8: Cross-Branch Dependencies
    
    Switch in Branch A controls gate in Branch B. Requires exploring
    multiple paths and understanding cross-branch relationships.
    
    Example: Zelda's crystal switches affecting distant barriers.
    
    Research: Kreminski & Mateas (2020) - Emergent narrative through mechanics
    """
    
    def __init__(self):
        super().__init__("AddEntangledBranches", weight=0.3)
    
    def can_apply(self, graph: MissionGraph, context: Dict[str, Any]) -> bool:
        """Can apply if we have hub with at least 2 branches."""
        if len(graph.nodes) < 6:
            return False
        # Find hubs with degree >= 3
        for node in graph.nodes.values():
            if graph.get_node_degree(node.id) >= 3:
                branches = graph.get_nodes_in_different_branches(node.id)
                if len(branches) >= 2:
                    return True
        return False
    
    def apply(self, graph: MissionGraph, context: Dict[str, Any]) -> MissionGraph:
        """Create entangled branch dependencies."""
        rng = context.get('rng') or random
        
        # Find hub with multiple branches
        hubs = [
            n for n in graph.nodes.values()
            if graph.get_node_degree(n.id) >= 3
        ]
        
        if not hubs:
            return graph
        
        hub = rng.choice(hubs)
        branches = graph.get_nodes_in_different_branches(hub.id)
        
        if len(branches) < 2:
            return graph
        
        # Select two branches
        branch_a, branch_b = rng.sample(branches, 2)
        
        if not branch_a or not branch_b:
            return graph
        
        # Place SWITCH at end of branch A
        switch_anchor = branch_a[-1] if len(branch_a) > 1 else branch_a[0]
        switch_id = max(graph.nodes.keys()) + 1
        switch_node = MissionNode(
            id=switch_id,
            node_type=NodeType.SWITCH,
            position=graph.nodes[switch_anchor].position,
            difficulty=context.get('difficulty', 0.5) * 0.7,
            switch_id=switch_id,
        )
        graph.add_node(switch_node)
        graph.add_edge(switch_anchor, switch_id, EdgeType.PATH)
        
        # Place STATE_BLOCK in branch B guarding reward
        block_anchor = branch_b[-1] if len(branch_b) > 1 else branch_b[0]
        
        # Create reward node
        reward_id = max(graph.nodes.keys()) + 1
        reward_node = MissionNode(
            id=reward_id,
            node_type=rng.choice([NodeType.BIG_KEY, NodeType.ITEM, NodeType.TREASURE]),
            position=graph.nodes[block_anchor].position,
            difficulty=context.get('difficulty', 0.5) * 0.8,
        )
        graph.add_node(reward_node)
        
        # Add STATE_BLOCK edge
        block_edge = MissionEdge(
            source=block_anchor,
            target=reward_id,
            edge_type=EdgeType.STATE_BLOCK,
            switch_id=switch_id,
        )
        graph.edges.append(block_edge)
        graph._adjacency[block_anchor].append(reward_id)
        
        logger.info(f"AddEntangledBranchesRule: Switch {switch_id} (branch A) controls gate to {reward_id} (branch B)")
        return graph


class AddHazardGateRule(ProductionRule):
    """
    ADVANCED RULE #9: Soft Gates with Risk-Reward
    
    Creates traversable but costly paths (lava, spikes) with optional
    protection items that eliminate damage.
    
    Example: Zelda's lava rooms (cross with damage or get fire tunic first).
    
    Research: Adams & Dormans (2012) - Optional challenge paths
    """
    
    def __init__(self):
        super().__init__("AddHazardGate", weight=0.25)
    
    def can_apply(self, graph: MissionGraph, context: Dict[str, Any]) -> bool:
        """Can apply if we have normal edges."""
        if len(graph.nodes) < 4:
            return False
        normal_edges = [e for e in graph.edges if e.edge_type == EdgeType.PATH]
        return len(normal_edges) >= 2
    
    def apply(self, graph: MissionGraph, context: Dict[str, Any]) -> MissionGraph:
        """Add hazard path with optional protection."""
        rng = context.get('rng') or random
        
        # Find edge to make hazardous
        normal_edges = [
            (i, e) for i, e in enumerate(graph.edges)
            if e.edge_type == EdgeType.PATH
        ]
        
        if len(normal_edges) < 2:
            return graph
        
        hazard_edge_idx, hazard_edge = rng.choice(normal_edges)
        
        # Choose hazard type
        hazard_types = ["LAVA", "SPIKES", "POISON", "ICE"]
        hazard_type = rng.choice(hazard_types)
        protection_item = f"{hazard_type}_PROTECTION"  # e.g., LAVA_PROTECTION (fire tunic)
        
        # Convert edge to HAZARD
        graph.edges[hazard_edge_idx].edge_type = EdgeType.HAZARD
        graph.edges[hazard_edge_idx].hazard_damage = rng.randint(1, 3)
        graph.edges[hazard_edge_idx].protection_item_id = protection_item
        
        # Place protection item in a side branch (optional)
        # Find a node not on critical path
        side_nodes = [
            n for n in graph.nodes.values()
            if (graph.get_node_degree(n.id) <= 2 and
                n.node_type in [NodeType.EMPTY, NodeType.ENEMY, NodeType.PUZZLE])
        ]
        
        if side_nodes:
            side_node = rng.choice(side_nodes)
            
            # Create protection item node
            protection_id = max(graph.nodes.keys()) + 1
            protection_node = MissionNode(
                id=protection_id,
                node_type=NodeType.PROTECTION_ITEM,
                position=side_node.position,
                difficulty=context.get('difficulty', 0.5) * 0.5,
                item_type=protection_item,
            )
            graph.add_node(protection_node)
            graph.add_edge(side_node.id, protection_id, EdgeType.PATH)
            
            logger.info(f"AddHazardGateRule: {hazard_type} hazard at {hazard_edge.source}->{hazard_edge.target}, protection at {protection_id}")
        
        return graph


class SplitRoomRule(ProductionRule):
    """
    ADVANCED RULE #10: Virtual Room Layering (Balconies/Basements)
    
    Creates two logically distinct nodes at the same (x, y) coordinate but
    different virtual layers (balcony above, basement below).
    
    Example: Zelda's rooms with balconies you can see but not immediately reach.
    
    Research: Brown "Boss Keys" - Vertical layering in 2D dungeons
    """
    
    def __init__(self):
        super().__init__("SplitRoom", weight=0.15)
    
    def can_apply(self, graph: MissionGraph, context: Dict[str, Any]) -> bool:
        """Can apply if we have nodes with low degree."""
        if len(graph.nodes) < 3:
            return False
        candidates = graph.get_nodes_with_degree_less_than(3)
        return len(candidates) > 0
    
    def apply(self, graph: MissionGraph, context: Dict[str, Any]) -> MissionGraph:
        """Split node into two virtual layers."""
        rng = context.get('rng') or random
        
        # Find node to split
        candidates = graph.get_nodes_with_degree_less_than(3)
        candidates = [
            n for n in candidates
            if n.node_type not in [NodeType.START, NodeType.GOAL, NodeType.BOSS_DOOR]
            and n.virtual_layer == 0  # Don't split already split nodes
        ]
        
        if not candidates:
            return graph
        
        original_node = rng.choice(candidates)
        
        # Create virtual layer node at same position
        virtual_id = max(graph.nodes.keys()) + 1
        virtual_node = MissionNode(
            id=virtual_id,
            node_type=rng.choice([NodeType.TREASURE, NodeType.ITEM, NodeType.KEY]),
            position=original_node.position,  # SAME x, y, z
            difficulty=context.get('difficulty', 0.5) * 0.7,
            virtual_layer=1,  # Different virtual layer
        )
        graph.add_node(virtual_node)
        
        # Connect via ONE_WAY (fall from balcony) or STAIRS
        connection_type = rng.choice([EdgeType.ONE_WAY, EdgeType.STAIRS])
        
        if connection_type == EdgeType.ONE_WAY:
            # Fall down - one direction only
            fall_edge = MissionEdge(
                source=virtual_id,  # From balcony
                target=original_node.id,  # To ground
                edge_type=EdgeType.ONE_WAY,
                preferred_direction="down",
            )
            graph.edges.append(fall_edge)
            graph._adjacency[virtual_id].append(original_node.id)
        else:
            # Stairs - bidirectional
            graph.add_edge(original_node.id, virtual_id, EdgeType.STAIRS)
        
        logger.info(f"SplitRoomRule: Created virtual layer at node {original_node.id} (layer 1 = {virtual_id})")
        return graph


# ============================================================================
# END OF ADVANCED RULES
# ============================================================================


# ============================================================================
# WAVE 3: PEDAGOGICAL & QUALITY CONTROL RULES
# ============================================================================

class AddSkillChainRule(ProductionRule):
    """
    WAVE 3 RULE #1: Tutorial Sequences (Learn -> Practice -> Master)
    
    After player acquires an item, creates a 3-node pedagogical sequence:
    1. TUTORIAL_PUZZLE: Safe room teaching item use (no enemies)
    2. COMBAT_PUZZLE: Moderate challenge (item + enemies)
    3. COMPLEX_PUZZLE: Hard challenge (item + previous mechanics)
    
    Example: Get Bow -> Shoot target -> Kill enemies with bow -> Complex archery puzzle
    
    Research: Nintendo's "kishÅtenketsu" pedagogy (introduction-development-twist-conclusion)
    """
    
    def __init__(self):
        super().__init__("AddSkillChain", weight=0.15)
    
    def can_apply(self, graph: MissionGraph, context: Dict[str, Any]) -> bool:
        """Can apply if there's an ITEM node with at least 3 successors."""
        items = graph.get_nodes_by_type(NodeType.ITEM)
        for item in items:
            successors = self._eligible_successors(graph, item.id, depth=4)
            if len(successors) >= 3:
                return True
        return False
    
    def apply(self, graph: MissionGraph, context: Dict[str, Any]) -> MissionGraph:
        """Create skill chain after item acquisition."""
        rng = context.get('rng') or random
        
        # Find ITEM nodes with sufficient successors
        items = graph.get_nodes_by_type(NodeType.ITEM)
        candidates = []
        for item in items:
            successors = self._eligible_successors(graph, item.id, depth=4)
            if len(successors) >= 3:
                candidates.append((item, successors))
        
        if not candidates:
            return graph
        
        item_node, successors = rng.choice(candidates)
        
        # Select 3 successors to convert
        selected = sorted(
            successors,
            key=lambda node: graph.get_shortest_path_length(item_node.id, node.id)
        )[:3]
        
        # Convert to tutorial sequence
        for i, node in enumerate(selected):
            if i == 0:
                # SAFE tutorial
                node.node_type = NodeType.TUTORIAL_PUZZLE
                node.difficulty_rating = "SAFE"
                node.difficulty = 0.2
                node.is_tutorial = True
                node.tension_value = 0.1
            elif i == 1:
                # MODERATE combat
                node.node_type = NodeType.COMBAT_PUZZLE
                node.difficulty_rating = "MODERATE"
                node.difficulty = 0.5
                node.tension_value = 0.5
            else:
                # HARD complex
                node.node_type = NodeType.COMPLEX_PUZZLE
                node.difficulty_rating = "HARD"
                node.difficulty = 0.8
                node.tension_value = 0.7
        
        logger.info(f"AddSkillChainRule: Created tutorial chain after item {item_node.id} ({item_node.item_type})")
        return graph

    def _eligible_successors(
        self,
        graph: MissionGraph,
        item_id: int,
        depth: int = 4,
    ) -> List[MissionNode]:
        """Filter successors to progression-relevant rooms only."""
        blocked_types = {
            NodeType.START,
            NodeType.GOAL,
            NodeType.BOSS,
            NodeType.BOSS_DOOR,
            NodeType.BIG_KEY,
        }
        seen: Set[int] = set()
        filtered: List[MissionNode] = []
        for node in graph.get_successors(item_id, depth=depth):
            if node.id == item_id or node.id in seen:
                continue
            if node.node_type in blocked_types:
                continue
            seen.add(node.id)
            filtered.append(node)
        return filtered


class AddPacingBreakerRule(ProductionRule):
    """
    WAVE 3 RULE #2: Sanctuary/Negative Space (Pacing Breakers)
    
    Inserts empty scenic rooms after high-tension sequences to provide
    breathing room and prevent player exhaustion.
    
    Detects 3+ consecutive combat/puzzle rooms and adds SCENIC room afterward.
    
    Example: After gauntlet of 4 enemy rooms -> peaceful vista room with lore
    
    Research: Schell "The Art of Game Design" - Pacing through negative space
    """
    
    def __init__(self):
        super().__init__("AddPacingBreaker", weight=0.2)
    
    def can_apply(self, graph: MissionGraph, context: Dict[str, Any]) -> bool:
        """Can apply if there are high-tension chains."""
        chains = graph.detect_high_tension_chains(min_length=3)
        return len(chains) > 0
    
    def apply(self, graph: MissionGraph, context: Dict[str, Any]) -> MissionGraph:
        """Insert sanctuary room after tension chain."""
        rng = context.get('rng') or random
        
        chains = graph.detect_high_tension_chains(min_length=3)
        if not chains:
            return graph
        
        # Select longest chain
        chain = max(chains, key=len)
        chain_end = chain[-1]
        
        # Find edges leaving chain end
        outgoing_edges = [
            (i, e) for i, e in enumerate(graph.edges)
            if e.source == chain_end and e.edge_type == EdgeType.PATH
        ]
        
        if not outgoing_edges:
            return graph
        
        # Insert SCENIC node on first outgoing edge
        edge_idx, edge = outgoing_edges[0]
        
        # Create sanctuary room
        sanctuary_id = max(graph.nodes.keys()) + 1
        sanctuary_pos = graph.nodes[chain_end].position
        floor = sanctuary_pos[2] if len(sanctuary_pos) > 2 else 0
        
        sanctuary = MissionNode(
            id=sanctuary_id,
            node_type=NodeType.SCENIC,
            position=(sanctuary_pos[0] + 1, sanctuary_pos[1], floor),
            difficulty=0.0,
            difficulty_rating="SAFE",
            is_sanctuary=True,
            tension_value=0.0,
        )
        graph.add_node(sanctuary)
        
        # Rewire edge through sanctuary
        graph.edges.pop(edge_idx)
        graph.add_edge(chain_end, sanctuary_id, EdgeType.PATH)
        graph.add_edge(sanctuary_id, edge.target, EdgeType.PATH)
        
        # Update adjacency
        if edge.target in graph._adjacency.get(chain_end, []):
            graph._adjacency[chain_end].remove(edge.target)
        if chain_end in graph._adjacency.get(edge.target, []):
            graph._adjacency[edge.target].remove(chain_end)
        
        logger.info(f"AddPacingBreakerRule: Inserted sanctuary {sanctuary_id} after tension chain of {len(chain)} rooms")
        return graph


class AddResourceLoopRule(ProductionRule):
    """
    WAVE 3 RULE #3: Resource Farming Spots (Soft-Lock Prevention)
    
    Prevents soft-locks by creating resource farming areas near gates that
    consume resources (bomb walls, arrow switches).
    
    Finds resource gates (BOMB_WALL, etc.) and places RESOURCE_FARM in
    a neighboring loop/cycle for repeated farming.
    
    Example: Bomb wall blocks progress -> nearby room respawns bomb drops
    
    Research: Dormans "Engineering Emergence" - Balancing resource economy
    """
    
    def __init__(self):
        super().__init__("AddResourceLoop", weight=0.25)
    
    def can_apply(self, graph: MissionGraph, context: Dict[str, Any]) -> bool:
        """Can apply if there are item-gated edges."""
        item_gates = [e for e in graph.edges if e.edge_type == EdgeType.ITEM_GATE]
        return len(item_gates) > 0
    
    def apply(self, graph: MissionGraph, context: Dict[str, Any]) -> MissionGraph:
        """Create resource farming spot near gate."""
        rng = context.get('rng') or random
        
        # Find item gates
        item_gates = [e for e in graph.edges if e.edge_type == EdgeType.ITEM_GATE]
        if not item_gates:
            return graph
        
        gate_edge = rng.choice(item_gates)
        required_item = gate_edge.item_required
        
        if not required_item:
            return graph
        
        # Find neighbors of gate source (reachable before gate)
        gate_source = gate_edge.source
        neighbors = graph._adjacency.get(gate_source, [])
        
        # Filter out the gate target (don't place farm past the gate)
        protected_types = {NodeType.START, NodeType.GOAL, NodeType.BOSS_DOOR, NodeType.BOSS}
        neighbors = [
            n for n in neighbors
            if (
                n != gate_edge.target
                and n in graph.nodes
                and graph.nodes[n].node_type not in protected_types
            )
        ]
        
        if not neighbors:
            # Create new neighbor
            farm_id = max(graph.nodes.keys()) + 1
            farm_pos = graph.nodes[gate_source].position
            floor = farm_pos[2] if len(farm_pos) > 2 else 0
            
            farm_node = MissionNode(
                id=farm_id,
                node_type=NodeType.RESOURCE_FARM,
                position=(farm_pos[0] + rng.randint(-1, 1), farm_pos[1] + rng.randint(1, 2), floor),
                difficulty=0.3,
                difficulty_rating="SAFE",
                drops_resource=required_item,  # e.g., "BOMBS"
                tension_value=0.2,
            )
            graph.add_node(farm_node)
            graph.add_edge(gate_source, farm_id, EdgeType.PATH)
            
            # Try to create loop (cycle back)
            start = graph.get_start_node()
            if start and start.id != gate_source:
                # Connect farm back toward start (create loop)
                graph.add_edge(farm_id, start.id, EdgeType.SHORTCUT)
            
            logger.info(f"AddResourceLoopRule: Created {required_item} farm {farm_id} near gate {gate_source}->{gate_edge.target}")
        else:
            # Convert existing neighbor to farm
            farm_id = rng.choice(neighbors)
            farm_node = graph.nodes[farm_id]
            farm_node.node_type = NodeType.RESOURCE_FARM
            farm_node.drops_resource = required_item
            farm_node.difficulty_rating = "SAFE"
            farm_node.tension_value = 0.2
            
            logger.info(f"AddResourceLoopRule: Converted node {farm_id} to {required_item} farm near gate")
        
        return graph


class AddGatekeeperRule(ProductionRule):
    """
    WAVE 3 RULE #4: Mini-Boss Guardians (Quality Control)
    
    Guards dungeon items with mini-boss fights, creating memorable
    high-stakes encounters for major rewards.
    
    Finds ITEM nodes and converts their immediate predecessor to MINI_BOSS,
    with special BOSS_DOOR or SHUTTER edge.
    
    Example: Mini-boss fight -> Hookshot acquisition
    
    Research: Brown "Boss Keys" - Guardian encounters as validation tests
    """
    
    def __init__(self):
        super().__init__("AddGatekeeper", weight=0.3)
    
    def can_apply(self, graph: MissionGraph, context: Dict[str, Any]) -> bool:
        """Can apply if there are ITEM nodes with single predecessors."""
        protected_types = {NodeType.START, NodeType.GOAL, NodeType.BOSS_DOOR}
        items = graph.get_nodes_by_type(NodeType.ITEM)
        for item in items:
            # Count incoming edges
            predecessors = [e.source for e in graph.edges if e.target == item.id]
            if len(predecessors) == 1:
                pred_id = predecessors[0]
                pred_node = graph.nodes.get(pred_id)
                if pred_node and pred_node.node_type not in [NodeType.MINI_BOSS, NodeType.BOSS] and pred_node.node_type not in protected_types:
                    return True
        return False
    
    def apply(self, graph: MissionGraph, context: Dict[str, Any]) -> MissionGraph:
        """Add mini-boss guarding item."""
        rng = context.get('rng') or random
        protected_types = {NodeType.START, NodeType.GOAL, NodeType.BOSS_DOOR}
        
        # Find suitable items
        items = graph.get_nodes_by_type(NodeType.ITEM)
        candidates = []
        
        for item in items:
            predecessors = [e.source for e in graph.edges if e.target == item.id]
            if len(predecessors) == 1:
                pred_id = predecessors[0]
                pred_node = graph.nodes.get(pred_id)
                if pred_node and pred_node.node_type not in [NodeType.MINI_BOSS, NodeType.BOSS] and pred_node.node_type not in protected_types:
                    candidates.append((item, pred_id))
        
        if not candidates:
            return graph
        
        item_node, pred_id = rng.choice(candidates)
        pred_node = graph.nodes[pred_id]
        
        # Convert predecessor to MINI_BOSS
        pred_node.node_type = NodeType.MINI_BOSS
        pred_node.is_mini_boss = True
        pred_node.difficulty = 0.75
        pred_node.difficulty_rating = "HARD"
        pred_node.tension_value = 0.9
        pred_node.room_size = (2, 2)  # Larger room for boss fight
        
        # Convert edge to special type
        for i, edge in enumerate(graph.edges):
            if edge.source == pred_id and edge.target == item_node.id:
                graph.edges[i].edge_type = EdgeType.SHUTTER  # Boss door that opens after fight
                break
        
        logger.info(f"AddGatekeeperRule: Mini-boss {pred_id} now guards item {item_node.id} ({item_node.item_type})")
        return graph


class AddMultiLockRule(ProductionRule):
    """
    WAVE 3 RULE #5: Battery Pattern (Multi-Switch Doors)
    
    Single door requires activating N switches scattered across different
    branches. All switches must be activated to open the lock.
    
    Creates battery_id linking multiple switches to one lock.
    
    Example: 3 crystal switches in different wings -> central door opens
    
    Research: Kreminski & Mateas "Gardening Games" - Interconnected mechanics
    """
    
    def __init__(self):
        super().__init__("AddMultiLock", weight=0.15)
    
    def can_apply(self, graph: MissionGraph, context: Dict[str, Any]) -> bool:
        """Can apply if there's a hub with 3+ branches."""
        if len(graph.nodes) < 8:
            return False
        
        for node in graph.nodes.values():
            if graph.get_node_degree(node.id) >= 3:
                branches = graph.get_branches_from_hub(node.id)
                if len(branches) >= 3:
                    return True
        return False
    
    def apply(self, graph: MissionGraph, context: Dict[str, Any]) -> MissionGraph:
        """Create multi-switch battery pattern."""
        rng = context.get('rng') or random
        
        # Find hub with 3+ branches
        hubs = [n for n in graph.nodes.values() if graph.get_node_degree(n.id) >= 3]
        if not hubs:
            return graph
        
        hub = rng.choice(hubs)
        branches = graph.get_branches_from_hub(hub.id)
        
        if len(branches) < 3:
            return graph
        
        # Create battery ID
        battery_id = max([e.battery_id for e in graph.edges if e.battery_id is not None], default=0) + 1
        
        # Place 3 switches in different branches
        num_switches = 3
        selected_branches = rng.sample(branches, min(num_switches, len(branches)))
        switch_ids = []
        
        for i, branch in enumerate(selected_branches):
            if not branch:
                continue
            
            # Place switch at end of branch
            target_node_id = branch[-1] if len(branch) > 1 else branch[0]
            
            # Create SWITCH node
            switch_id = max(graph.nodes.keys()) + 1
            switch_node = MissionNode(
                id=switch_id,
                node_type=NodeType.SWITCH,
                position=graph.nodes[target_node_id].position,
                difficulty=context.get('difficulty', 0.5) * 0.6,
                difficulty_rating="MODERATE",
                switch_id=switch_id,
            )
            graph.add_node(switch_node)
            graph.add_edge(target_node_id, switch_id, EdgeType.PATH)
            switch_ids.append(switch_id)
        
        if len(switch_ids) < 2:
            return graph
        
        # Find edge to lock with battery
        # Prefer edges leaving the hub
        lock_candidates = [
            (i, e) for i, e in enumerate(graph.edges)
            if e.source == hub.id and e.edge_type == EdgeType.PATH
        ]
        
        if not lock_candidates:
            # Fallback: any PATH edge
            lock_candidates = [
                (i, e) for i, e in enumerate(graph.edges)
                if e.edge_type == EdgeType.PATH
            ]
        
        if lock_candidates:
            start = graph.get_start_node()
            viable_candidates: List[Tuple[int, MissionEdge]] = []

            if start is not None:
                for idx, edge in lock_candidates:
                    reachable = graph.get_reachable_nodes(
                        start.id,
                        excluded_edges={(edge.source, edge.target)},
                    )
                    if all(switch_id in reachable for switch_id in switch_ids):
                        viable_candidates.append((idx, edge))
            else:
                viable_candidates = lock_candidates

            if not viable_candidates:
                logger.warning(
                    "AddMultiLockRule: No viable lock edge keeps all switches reachable; "
                    "skipping lock conversion"
                )
                return graph

            lock_edge_idx, lock_edge = rng.choice(viable_candidates)

            # Convert to battery-locked edge
            graph.edges[lock_edge_idx].edge_type = EdgeType.STATE_BLOCK
            graph.edges[lock_edge_idx].battery_id = battery_id
            graph.edges[lock_edge_idx].switches_required = switch_ids

            logger.info(
                f"AddMultiLockRule: {len(switch_ids)} switches (battery {battery_id}) "
                f"control lock {lock_edge.source}->{lock_edge.target}"
            )
        
        return graph


class AddItemShortcutRule(ProductionRule):
    """
    WAVE 3 RULE #6: Item-Gated Shortcut (Item-Based Return)
    
    Creates shortcuts from item locations back toward start, gated by
    the specific item just acquired. Rewards exploration and backtracking.
    
    Example: Get Hookshot -> use it to shortcut back over gap to start area
    
    Research: Brown "Boss Keys" - Item-gated backtracking rewards
    """
    
    def __init__(self):
        super().__init__("AddItemShortcut", weight=0.2)
    
    def can_apply(self, graph: MissionGraph, context: Dict[str, Any]) -> bool:
        """Can apply if there are ITEM nodes far from start."""
        start = graph.get_start_node()
        if not start:
            return False
        
        items = graph.get_nodes_by_type(NodeType.ITEM)
        for item in items:
            dist = graph.get_shortest_path_length(start.id, item.id)
            if dist > 5:
                return True
        return False
    
    def apply(self, graph: MissionGraph, context: Dict[str, Any]) -> MissionGraph:
        """Create item-gated shortcut to start area."""
        rng = context.get('rng') or random
        
        start = graph.get_start_node()
        if not start:
            return graph
        
        # Find distant items
        items = graph.get_nodes_by_type(NodeType.ITEM)
        candidates = []
        
        for item in items:
            dist = graph.get_shortest_path_length(start.id, item.id)
            if dist > 5:
                candidates.append((item, dist))
        
        if not candidates:
            return graph
        
        # Prefer furthest item
        candidates.sort(key=lambda x: x[1], reverse=True)
        item_node, original_dist = candidates[0]
        
        # Find node in start area (within 2 hops of start)
        start_neighbors = graph.get_successors(start.id, depth=2)
        if not start_neighbors:
            return graph
        
        target_node = rng.choice(start_neighbors)
        
        # Calculate savings
        savings = graph.calculate_path_savings((item_node.id, target_node.id))
        
        if savings < 3:
            return graph  # Not worth it
        
        # Create shortcut edge gated by the item
        shortcut_edge = MissionEdge(
            source=item_node.id,
            target=target_node.id,
            edge_type=EdgeType.ITEM_GATE,
            item_required=item_node.item_type,
            preferred_direction="backward",
            path_savings=savings,
        )
        graph.edges.append(shortcut_edge)
        graph._adjacency[item_node.id].append(target_node.id)
        
        logger.info(f"AddItemShortcutRule: Shortcut {item_node.id}->{target_node.id} gated by {item_node.item_type} (saves {savings} hops)")
        return graph


class PruneDeadEndRule(ProductionRule):
    """
    WAVE 3 RULE #7: Dead-End Garbage Collection (Quality Control)
    
    Removes useless dead-end rooms that don't contain valuable content.
    Preserves graph connectivity and never prunes critical nodes.
    
    Example: Empty dead-end chain -> removed if no keys/items/secrets
    
    Research: Smith "Variations Forever" - Quality control via pruning
    """
    
    def __init__(self):
        super().__init__("PruneDeadEnd", weight=0.1)
    
    def can_apply(self, graph: MissionGraph, context: Dict[str, Any]) -> bool:
        """Can apply if there are degree-1 nodes without valuable content."""
        valuable_types = {
            NodeType.KEY, NodeType.ITEM, NodeType.BOSS, NodeType.MINI_BOSS,
            NodeType.SWITCH, NodeType.GOAL, NodeType.START, NodeType.BIG_KEY,
            NodeType.TREASURE, NodeType.TOKEN
        }
        
        for node in graph.nodes.values():
            degree = graph.get_node_degree(node.id)
            if degree == 1 and node.node_type not in valuable_types:
                if not node.is_hub and not node.is_secret:
                    return True
        return False
    
    def apply(self, graph: MissionGraph, context: Dict[str, Any]) -> MissionGraph:
        """Prune useless dead ends."""
        valuable_types = {
            NodeType.KEY, NodeType.ITEM, NodeType.BOSS, NodeType.MINI_BOSS,
            NodeType.SWITCH, NodeType.GOAL, NodeType.START, NodeType.BIG_KEY,
            NodeType.TREASURE, NodeType.TOKEN
        }
        
        # Find dead ends without value
        dead_ends = []
        for node in graph.nodes.values():
            degree = graph.get_node_degree(node.id)
            if degree == 1 and node.node_type not in valuable_types:
                if not node.is_hub and not node.is_secret:
                    dead_ends.append(node.id)
        
        if not dead_ends:
            return graph
        
        # Remove first dead-end that keeps remaining graph connected.
        for node_id in dead_ends:
            remaining_nodes = set(graph.nodes.keys()) - {node_id}
            if not remaining_nodes:
                continue

            traversal_start = next(iter(remaining_nodes))
            reachable = graph.get_reachable_nodes(
                traversal_start,
                excluded_nodes={node_id},
            )
            if len(reachable.intersection(remaining_nodes)) != len(remaining_nodes):
                logger.warning(f"PruneDeadEndRule: Would disconnect graph, skipping node {node_id}")
                continue

            # Safe to prune.
            del graph.nodes[node_id]
            graph.edges = [e for e in graph.edges if e.source != node_id and e.target != node_id]
            graph.sanitize()
            logger.info(f"PruneDeadEndRule: Pruned dead-end node {node_id}")
            return graph
        
        return graph


# ============================================================================
# VALIDATION METHODS FOR WAVE 3 RULES
# ============================================================================

def validate_skill_chains(graph: MissionGraph) -> bool:
    """
    Ensure tutorial sequences are properly ordered.
    
    Returns:
        True if all skill chains have proper difficulty progression
    """
    graph.sanitize()
    tutorial_nodes = [n for n in graph.nodes.values() if n.is_tutorial]
    pedagogical_types = {NodeType.COMBAT_PUZZLE, NodeType.COMPLEX_PUZZLE}
    
    for tutorial in tutorial_nodes:
        # Check nearest pedagogical successors only.
        successors = [
            n for n in graph.get_successors(tutorial.id, depth=3)
            if n.node_type in pedagogical_types
        ]
        if len(successors) < 2:
            continue

        successors.sort(key=lambda n: graph.get_shortest_path_length(tutorial.id, n.id))
        first, second = successors[0], successors[1]
        if first.difficulty > second.difficulty:
            logger.warning(f"Skill chain from {tutorial.id} has improper difficulty progression")
            return False
    
    return True


def validate_battery_reachability(graph: MissionGraph) -> bool:
    """
    Ensure all switches in battery are reachable before locked door.
    
    Returns:
        True if all battery patterns are valid
    """
    graph.sanitize()
    start = graph.get_start_node()
    if not start:
        return True
    
    # Find battery-locked edges
    battery_edges = [e for e in graph.edges if e.battery_id is not None]
    
    for edge in battery_edges:
        required_switches = edge.switches_required
        
        for switch_id in required_switches:
            # Check if switch is reachable before the locked edge
            reachable = graph.get_reachable_nodes(start.id, excluded_edges={(edge.source, edge.target)})
            if switch_id not in reachable:
                logger.warning(f"Battery switch {switch_id} not reachable before lock {edge.source}->{edge.target}")
                return False
    
    return True


def validate_resource_loops(graph: MissionGraph) -> bool:
    """
    Ensure resource farms are reachable before their gates.
    
    Returns:
        True if all resource farms are properly placed
    """
    graph.sanitize()
    start = graph.get_start_node()
    if not start:
        return True
    
    # Find resource farms
    farms = [n for n in graph.nodes.values() if n.node_type == NodeType.RESOURCE_FARM]
    
    for farm in farms:
        resource = farm.drops_resource
        if not resource:
            continue
        
        # Find gates requiring this resource
        gates = [e for e in graph.edges if e.item_required == resource]
        
        for gate in gates:
            # Verify farm is reachable before gate
            reachable = graph.get_reachable_nodes(start.id, excluded_edges={(gate.source, gate.target)})
            if farm.id not in reachable:
                logger.warning(f"Resource farm {farm.id} ({resource}) not reachable before gate {gate.source}->{gate.target}")
                return False
    
    return True


# ============================================================================
# END OF WAVE 3 RULES
# ============================================================================



# ============================================================================
# TEST
# ============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Mission Grammar...")
    
    # Create grammar
    grammar = MissionGrammar(seed=42)
    
    # Generate mission graph
    graph = grammar.generate(
        difficulty=Difficulty.MEDIUM,
        num_rooms=8,
        max_keys=2,
    )
    
    print(f"\nGenerated graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    
    # Print nodes
    print("\nNodes:")
    for node in sorted(graph.nodes.values(), key=lambda n: n.id):
        key_info = f" (key_id={node.key_id})" if node.key_id else ""
        print(f"  {node.id}: {node.node_type.name} at {node.position}{key_info}")
    
    # Print edges
    print("\nEdges:")
    for edge in graph.edges:
        key_req = f" [requires key {edge.key_required}]" if edge.key_required else ""
        print(f"  {edge.source} -> {edge.target} ({edge.edge_type.name}){key_req}")
    
    # Validate
    valid = grammar.validate_lock_key_ordering(graph)
    print(f"\nLock-key ordering valid: {valid}")
    
    # Convert to tensors
    gnn_input = graph_to_gnn_input(graph, current_node_idx=0)
    print(f"\nGNN Input:")
    print(f"  edge_index: {gnn_input['edge_index'].shape}")
    print(f"  node_features: {gnn_input['node_features'].shape}")
    print(f"  tpe: {gnn_input['tpe'].shape}")
    
    print("\nMission Grammar test passed!")

