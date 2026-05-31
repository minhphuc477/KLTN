"""Core mission graph production rules."""

from __future__ import annotations

import logging
import random
from collections import deque
from enum import Enum
from typing import Any, Dict, List, Optional

from .graph_types import EdgeType, MissionEdge, MissionGraph, MissionNode, NodeType

logger = logging.getLogger(__name__)

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

    @staticmethod
    def _small_key_supply(graph: MissionGraph) -> int:
        return sum(1 for node in graph.nodes.values() if node.node_type == NodeType.KEY)

    @staticmethod
    def _small_key_demand(graph: MissionGraph) -> int:
        demand = 0
        for edge in graph.edges:
            if edge.edge_type != EdgeType.LOCKED:
                continue
            if edge.requires_key_count > 0:
                demand += int(max(1, edge.requires_key_count))
            else:
                demand += 1
        return int(max(0, demand))

    def _allow_bonus_small_key(self, graph: MissionGraph) -> bool:
        """Only allow side-reward keys when the graph is actually key-starved."""
        supply = self._small_key_supply(graph)
        demand = self._small_key_demand(graph)
        # Avoid minting free keys on side branches unless we are still behind.
        return bool(supply < demand)

    def _reward_node_choices(
        self,
        graph: MissionGraph,
        *,
        include_empty: bool = False,
        include_treasure: bool = True,
        include_protection_item: bool = False,
    ) -> List[NodeType]:
        choices: List[NodeType] = [NodeType.ITEM]
        if include_treasure:
            choices.append(NodeType.TREASURE)
        if include_protection_item:
            choices.append(NodeType.PROTECTION_ITEM)
        if include_empty:
            choices.append(NodeType.EMPTY)
        if self._allow_bonus_small_key(graph):
            choices.append(NodeType.KEY)
        return choices


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
        """Return one shortest path (node sequence) over forward progression edges."""
        if start_id == goal_id:
            return [start_id]

        adjacency = graph.get_forward_adjacency_map()
        visited = {start_id}
        queue = deque([(start_id, [start_id])])
        while queue:
            current, path = queue.popleft()
            for neighbor in adjacency.get(current, []):
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
