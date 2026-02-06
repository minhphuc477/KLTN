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
Start Symbol: S → Goal

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
    S → START, SEGMENT, GOAL
    SEGMENT → CHALLENGE | CHALLENGE, SEGMENT | BRANCH
    BRANCH → (SEGMENT) + (SEGMENT)  [parallel paths]
    CHALLENGE → ENEMY | PUZZLE | LOCK_KEY
    LOCK_KEY → KEY, LOCK  [key must precede lock]

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
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
import math

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


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


class EdgeType(Enum):
    """Mission graph edge types."""
    PATH = auto()       # Normal path
    LOCKED = auto()     # Requires key
    ONE_WAY = auto()    # One-directional
    HIDDEN = auto()     # Secret passage


# ============================================================================
# GRAPH DATA STRUCTURES
# ============================================================================

@dataclass
class MissionNode:
    """Node in the mission graph."""
    id: int
    node_type: NodeType
    position: Tuple[int, int] = (0, 0)  # (row, col) in dungeon layout
    
    # Key-lock binding
    key_id: Optional[int] = None  # For LOCK: which key opens this
    
    # Metadata
    difficulty: float = 0.5
    required_item: Optional[str] = None
    
    def to_feature_vector(self) -> List[float]:
        """Convert to feature vector for GNN."""
        # One-hot encode node type
        type_vec = [0.0] * len(NodeType)
        type_vec[self.node_type.value - 1] = 1.0
        
        # Position encoding
        pos_vec = [self.position[0] / 10.0, self.position[1] / 10.0]
        
        # Additional features
        extra = [
            self.difficulty,
            1.0 if self.key_id is not None else 0.0,
        ]
        
        return type_vec + pos_vec + extra


@dataclass
class MissionEdge:
    """Edge in the mission graph."""
    source: int
    target: int
    edge_type: EdgeType = EdgeType.PATH
    key_required: Optional[int] = None  # Key ID if LOCKED


@dataclass
class MissionGraph:
    """Complete mission graph for a dungeon."""
    nodes: Dict[int, MissionNode] = field(default_factory=dict)
    edges: List[MissionEdge] = field(default_factory=list)
    
    # Quick lookup structures
    _adjacency: Dict[int, List[int]] = field(default_factory=lambda: defaultdict(list))
    _key_to_lock: Dict[int, int] = field(default_factory=dict)  # key_id -> lock_node_id
    
    def add_node(self, node: MissionNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node
    
    def add_edge(
        self,
        source: int,
        target: int,
        edge_type: EdgeType = EdgeType.PATH,
        key_required: Optional[int] = None,
    ) -> None:
        """Add an edge to the graph."""
        edge = MissionEdge(source, target, edge_type, key_required)
        self.edges.append(edge)
        self._adjacency[source].append(target)
        
        # Add reverse edge for bidirectional paths
        if edge_type == EdgeType.PATH:
            self._adjacency[target].append(source)
    
    def get_node(self, node_id: int) -> Optional[MissionNode]:
        """Get node by ID."""
        return self.nodes.get(node_id)
    
    def get_neighbors(self, node_id: int) -> List[int]:
        """Get neighbor node IDs."""
        return self._adjacency.get(node_id, [])
    
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
            if edge.edge_type == EdgeType.PATH:  # Bidirectional
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
    """S → START, SEGMENT, GOAL"""
    
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
            position=(0, 0),
            difficulty=0.0,
        )
        graph.add_node(start)
        
        # Create GOAL node
        goal = MissionNode(
            id=1,
            node_type=NodeType.GOAL,
            position=(context.get('goal_row', 5), context.get('goal_col', 5)),
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
        """Can apply if there are any edges."""
        return len(graph.edges) > 0
    
    def apply(
        self,
        graph: MissionGraph,
        context: Dict[str, Any],
    ) -> MissionGraph:
        """Insert challenge node on a random edge."""
        if not graph.edges:
            return graph
        
        # Select random edge
        edge_idx = context.get('rng', random).randint(0, len(graph.edges) - 1)
        edge = graph.edges[edge_idx]
        
        # Create new challenge node
        new_id = max(graph.nodes.keys()) + 1
        
        # Interpolate position
        src_pos = graph.nodes[edge.source].position
        tgt_pos = graph.nodes[edge.target].position
        new_pos = (
            (src_pos[0] + tgt_pos[0]) // 2,
            (src_pos[1] + tgt_pos[1]) // 2,
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
        graph.add_edge(new_id, edge.target, edge.edge_type, edge.key_required)
        
        # Update adjacency
        if edge.target in graph._adjacency[edge.source]:
            graph._adjacency[edge.source].remove(edge.target)
        if edge.source in graph._adjacency[edge.target]:
            graph._adjacency[edge.target].remove(edge.source)
        
        return graph


class InsertLockKeyRule(ProductionRule):
    """
    Insert a Lock-Key pair ensuring key precedes lock.
    
    KEY → ... → LOCK → continuation
    
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
        return len(graph.edges) >= 1 and len(graph.nodes) >= 2
    
    def apply(
        self,
        graph: MissionGraph,
        context: Dict[str, Any],
    ) -> MissionGraph:
        """Insert KEY before LOCK on the path to goal."""
        if len(graph.edges) < 1:
            return graph
        
        rng = context.get('rng', random)
        
        # Find an edge to split for KEY
        key_edge_idx = rng.randint(0, len(graph.edges) - 1)
        key_edge = graph.edges[key_edge_idx]
        
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
        
        # Now find an edge AFTER the key position for the LOCK
        # We need an edge that comes after key_edge.target in the graph
        lock_candidates = [
            (i, e) for i, e in enumerate(graph.edges)
            if e.source in [key_id, key_edge.target] or 
               any(e.source in graph._adjacency.get(n, []) 
                   for n in [key_id, key_edge.target])
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
        
        return graph
    
    def _interpolate_pos(
        self,
        graph: MissionGraph,
        src: int,
        tgt: int,
        t: float,
    ) -> Tuple[int, int]:
        """Interpolate position between two nodes."""
        src_pos = graph.nodes[src].position
        tgt_pos = graph.nodes[tgt].position
        return (
            int(src_pos[0] * (1 - t) + tgt_pos[0] * t),
            int(src_pos[1] * (1 - t) + tgt_pos[1] * t),
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
        rng = context.get('rng', random)
        
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
        
        new_node = MissionNode(
            id=new_id,
            node_type=rng.choice([NodeType.ITEM, NodeType.PUZZLE, NodeType.EMPTY]),
            position=(
                branch_node.position[0] + offset_r,
                branch_node.position[1] + offset_c,
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
        self.rules = [
            StartRule(),
            InsertChallengeRule(NodeType.ENEMY),
            InsertChallengeRule(NodeType.PUZZLE),
            InsertLockKeyRule(),
            BranchRule(),
        ]
    
    def generate(
        self,
        difficulty: Difficulty = Difficulty.MEDIUM,
        num_rooms: int = 8,
        max_keys: int = 2,
    ) -> MissionGraph:
        """
        Generate a mission graph.
        
        Args:
            difficulty: Dungeon difficulty level
            num_rooms: Approximate number of rooms
            max_keys: Maximum number of key-lock pairs
            
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
        
        # Track how many of each have been added
        num_keys_added = 0
        num_challenges_added = 0
        
        # Apply rules until we have enough nodes
        max_iterations = num_rooms * 3
        iteration = 0
        
        while len(graph.nodes) < num_rooms and iteration < max_iterations:
            iteration += 1
            
            # Select a rule based on weights and constraints
            applicable_rules = []
            weights = []
            
            for rule in self.rules[1:]:  # Skip start rule
                if not rule.can_apply(graph, context):
                    continue
                
                # Limit key-lock pairs
                if isinstance(rule, InsertLockKeyRule):
                    if num_keys_added >= max_keys:
                        continue
                
                applicable_rules.append(rule)
                weights.append(rule.weight)
            
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
            
            # Track additions
            if isinstance(selected_rule, InsertLockKeyRule):
                num_keys_added += 1
            elif isinstance(selected_rule, InsertChallengeRule):
                num_challenges_added += 1
        
        # Validate and fix if needed
        if not self.validate_lock_key_ordering(graph):
            logger.warning("Generated graph failed lock-key validation, fixing...")
            graph = self._fix_lock_key_ordering(graph)
        
        # Update positions for layout
        graph = self._layout_graph(graph)
        
        return graph
    
    def validate_lock_key_ordering(self, graph: MissionGraph) -> bool:
        """
        Validate that all keys can be reached before their locks.
        
        For each LOCK node, verifies that its required KEY is
        reachable from START without passing through the LOCK.
        """
        start = graph.get_start_node()
        if not start:
            return False
        
        locks = graph.get_nodes_by_type(NodeType.LOCK)
        keys = graph.get_nodes_by_type(NodeType.KEY)
        
        # Build key lookup
        key_by_id = {k.key_id: k for k in keys if k.key_id is not None}
        
        for lock in locks:
            if lock.key_id is None:
                continue
            
            key = key_by_id.get(lock.key_id)
            if not key:
                logger.warning(f"Lock {lock.id} references non-existent key {lock.key_id}")
                return False
            
            # Check if key is reachable from start without passing through lock
            if not self._is_reachable_without(graph, start.id, key.id, exclude={lock.id}):
                logger.warning(f"Key {key.id} not reachable before lock {lock.id}")
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
    
    def _fix_lock_key_ordering(self, graph: MissionGraph) -> MissionGraph:
        """Fix any lock-key ordering violations by swapping positions."""
        start = graph.get_start_node()
        if not start:
            return graph
        
        locks = graph.get_nodes_by_type(NodeType.LOCK)
        keys = graph.get_nodes_by_type(NodeType.KEY)
        key_by_id = {k.key_id: k for k in keys if k.key_id is not None}
        
        for lock in locks:
            if lock.key_id is None:
                continue
            
            key = key_by_id.get(lock.key_id)
            if not key:
                # Remove orphan lock
                lock.node_type = NodeType.EMPTY
                continue
            
            # If key not reachable before lock, swap positions
            if not self._is_reachable_without(graph, start.id, key.id, {lock.id}):
                # Swap positions
                key.position, lock.position = lock.position, key.position
                logger.info(f"Swapped key {key.id} and lock {lock.id} positions")
        
        return graph
    
    def _layout_graph(self, graph: MissionGraph) -> MissionGraph:
        """Apply a simple layout algorithm to position nodes."""
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
                offset = i - len(nodes) // 2
                graph.nodes[node_id].position = (layer * 2, offset * 2 + 5)
        
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
        print(f"  {edge.source} → {edge.target} ({edge.edge_type.name}){key_req}")
    
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
