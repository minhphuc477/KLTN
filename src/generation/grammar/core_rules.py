"""Core mission graph production rules."""

from __future__ import annotations

import logging
import random
from collections import deque
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

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

    @staticmethod
    def _layout_bounds(context: Optional[Dict[str, Any]] = None) -> Tuple[int, int, int, int]:
        """Return inclusive row/col bounds for collision-aware grammar placement."""
        ctx = context or {}
        raw = ctx.get("layout_bounds")
        if isinstance(raw, (list, tuple)) and len(raw) == 4:
            return tuple(int(v) for v in raw)  # type: ignore[return-value]
        return (
            int(ctx.get("min_layout_row", -32)),
            int(ctx.get("max_layout_row", 32)),
            int(ctx.get("min_layout_col", -32)),
            int(ctx.get("max_layout_col", 32)),
        )

    @staticmethod
    def _occupied_positions(graph: MissionGraph) -> set[Tuple[int, int, int]]:
        occupied: set[Tuple[int, int, int]] = set()
        for node in graph.nodes.values():
            pos = node.position
            occupied.add((int(pos[0]), int(pos[1]), int(pos[2]) if len(pos) > 2 else 0))
        return occupied

    def _nearest_free_position(
        self,
        graph: MissionGraph,
        ideal_pos: Tuple[float, float, float],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, int, int]:
        """Snap an ideal coordinate to the nearest bounded unoccupied grid cell."""
        min_r, max_r, min_c, max_c = self._layout_bounds(context)
        base_r = max(min_r, min(max_r, int(round(float(ideal_pos[0])))))
        base_c = max(min_c, min(max_c, int(round(float(ideal_pos[1])))))
        floor = int(round(float(ideal_pos[2])))
        occupied = self._occupied_positions(graph)
        base = (base_r, base_c, floor)
        if base not in occupied:
            return base

        max_radius = max(max_r - min_r, max_c - min_c, 1)
        for radius in range(1, max_radius + 1):
            candidates: List[Tuple[int, int, int]] = []
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if max(abs(dr), abs(dc)) != radius:
                        continue
                    row = base_r + dr
                    col = base_c + dc
                    if min_r <= row <= max_r and min_c <= col <= max_c:
                        candidates.append((row, col, floor))
            candidates.sort(key=lambda p: (abs(p[0] - base_r) + abs(p[1] - base_c), p[0], p[1]))
            for candidate in candidates:
                if candidate not in occupied:
                    return candidate
        return base

    def _interpolate_free_position(
        self,
        graph: MissionGraph,
        src: int,
        tgt: int,
        t: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, int, int]:
        """Interpolate between two nodes, then resolve collisions near the ideal point."""
        src_pos = graph.nodes[src].position
        tgt_pos = graph.nodes[tgt].position
        z = src_pos[2] if len(src_pos) > 2 else 0
        ideal = (
            float(src_pos[0]) * (1.0 - float(t)) + float(tgt_pos[0]) * float(t),
            float(src_pos[1]) * (1.0 - float(t)) + float(tgt_pos[1]) * float(t),
            float(z),
        )
        return self._nearest_free_position(graph, ideal, context)


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
        
        new_pos = self._interpolate_free_position(graph, edge.source, edge.target, 0.5, context)
        
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
            position=self._interpolate_pos(graph, key_edge.source, key_edge.target, 0.3, context),
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

        # Fallback to forward descendants of the key only. Using an arbitrary
        # PATH edge here can place a lock before its provider and violate the
        # rule's causal contract.
        if not lock_candidates:
            forward = graph.get_forward_adjacency_map()
            reachable_after_key = {key_id}
            queue = deque([key_id])
            while queue:
                current = queue.popleft()
                for neighbor in forward.get(current, []):
                    if neighbor not in reachable_after_key:
                        reachable_after_key.add(neighbor)
                        queue.append(neighbor)
            lock_candidates = [
                (i, e) for i, e in enumerate(graph.edges)
                if (
                    e.edge_type == EdgeType.PATH
                    and e.source in reachable_after_key
                    and e.source != key_id
                    and e.target != key_id
                )
            ]
        
        if lock_candidates:
            lock_edge_idx, lock_edge = rng.choice(lock_candidates)
            
            # Create LOCK node
            lock_id = max(graph.nodes.keys()) + 1
            lock_node = MissionNode(
                id=lock_id,
                node_type=NodeType.LOCK,
                position=self._interpolate_pos(graph, lock_edge.source, lock_edge.target, 0.7, context),
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
        Find a forward PATH edge index connecting two adjacent nodes.
        """
        for idx, edge in enumerate(graph.edges):
            if edge.edge_type != EdgeType.PATH:
                continue
            if edge.source == node_a and edge.target == node_b:
                return idx
        return None
    
    def _interpolate_pos(
        self,
        graph: MissionGraph,
        src: int,
        tgt: int,
        t: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, int, int]:
        """Interpolate position between two nodes without colliding with endpoints."""
        return self._interpolate_free_position(graph, src, tgt, t, context)


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
            position=self._nearest_free_position(
                graph,
                (
                    float(branch_node.position[0] + offset_r),
                    float(branch_node.position[1] + offset_c),
                    float(floor),
                ),
                context,
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
