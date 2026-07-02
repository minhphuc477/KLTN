"""Advanced and pedagogical mission graph production rules."""

from __future__ import annotations

import copy
import logging
import math
import random
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

from .core_rules import ProductionRule
from .graph_types import (
    LAYOUT_HUB_BRANCH_SPACING,
    LAYOUT_HUB_RADIUS,
    EdgeType,
    MissionEdge,
    MissionGraph,
    MissionNode,
    NodeType,
)
from src.generation.grammar_validators import (
    validate_battery_reachability,
    validate_resource_loops,
    validate_skill_chains,
)

logger = logging.getLogger(__name__)


def _layout_bounds(context: Dict[str, Any]) -> Tuple[int, int, int, int]:
    """Return inclusive row/col bounds for grammar placement."""
    raw = context.get("layout_bounds")
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        return tuple(int(v) for v in raw)  # type: ignore[return-value]
    return (
        int(context.get("min_layout_row", -32)),
        int(context.get("max_layout_row", 32)),
        int(context.get("min_layout_col", -32)),
        int(context.get("max_layout_col", 32)),
    )


def _bounded_free_position(
    graph: MissionGraph,
    anchor_pos: Tuple[int, ...],
    offsets: List[Tuple[int, int]],
    rng: Any,
    context: Dict[str, Any],
) -> Tuple[int, int, int]:
    """Choose a relative, in-bounds, unoccupied position near an anchor."""
    floor = int(anchor_pos[2]) if len(anchor_pos) > 2 else 0
    min_r, max_r, min_c, max_c = _layout_bounds(context)
    occupied = {
        (int(node.position[0]), int(node.position[1]), int(node.position[2]) if len(node.position) > 2 else 0)
        for node in graph.nodes.values()
    }
    shuffled = list(offsets)
    rng.shuffle(shuffled)
    for dr, dc in shuffled:
        row = max(min_r, min(max_r, int(anchor_pos[0]) + int(dr)))
        col = max(min_c, min(max_c, int(anchor_pos[1]) + int(dc)))
        candidate = (row, col, floor)
        if candidate not in occupied:
            return candidate

    # Deterministic expanding fallback: still relative to the anchor and bounded.
    for radius in range(1, 8):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                row = max(min_r, min(max_r, int(anchor_pos[0]) + dr))
                col = max(min_c, min(max_c, int(anchor_pos[1]) + dc))
                candidate = (row, col, floor)
                if candidate not in occupied:
                    return candidate
    return (
        max(min_r, min(max_r, int(anchor_pos[0]))),
        max(min_c, min(max_c, int(anchor_pos[1]))),
        floor,
    )


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
        queue = deque([(start, 0)])
        
        while queue:
            current, dist = queue.popleft()
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
        
        # Pick an existing reachable anchor before adding the switch so the
        # switch placement can be resolved relative to occupied geometry.
        other_nodes = [
            n for n in graph.nodes.keys()
            if n not in [edge.source, edge.target]
        ]
        anchor = rng.choice(other_nodes) if other_nodes else edge.source
        anchor_pos = graph.nodes[anchor].position
        switch_offsets = [
            (0, 1), (1, 0), (0, -1), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
            (0, 2), (2, 0), (0, -2), (-2, 0),
        ]

        # Change edge to ON_OFF_GATE
        graph.edges[edge_idx].edge_type = EdgeType.ON_OFF_GATE
        
        # Add switch node in a separate branch
        switch_id = max(graph.nodes.keys()) + 1
        switch_node = MissionNode(
            id=switch_id,
            node_type=NodeType.SWITCH,
            position=_bounded_free_position(graph, anchor_pos, switch_offsets, rng, context),
            difficulty=context.get('difficulty', 0.5) * 0.6,
            switch_id=switch_id,  # Self-referencing switch ID
        )
        graph.add_node(switch_node)
        
        # Connect switch to graph (not near the gated edge when possible)
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
        """Insert Boss Door -> Boss -> Goal, spawn Big Key on the pre-lock side."""
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
        goal_pos = goal.position
        # Create Boss Door and Boss chain directly so strict validation passes
        # without needing a later normalization pass.
        boss_door_id = max(graph.nodes.keys()) + 1
        boss_door = MissionNode(
            id=boss_door_id,
            node_type=NodeType.BOSS_DOOR,
            position=_bounded_free_position(
                graph,
                goal_pos,
                [(-2, 0), (-1, 0), (0, -1), (0, 1), (-2, -1), (-2, 1), (1, 0)],
                rng,
                context,
            ),
            difficulty=0.9,
            key_id=boss_door_id,  # Requires big key
        )
        graph.add_node(boss_door)

        boss_id = boss_door_id + 1
        boss = MissionNode(
            id=boss_id,
            node_type=NodeType.BOSS,
            position=_bounded_free_position(
                graph,
                goal_pos,
                [(-1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, 0)],
                rng,
                context,
            ),
            difficulty=0.95,
        )
        graph.add_node(boss)

        # Rewire every approach edge into a single boss-door chain and make the
        # goal a strict terminal leaf off the boss.
        graph.edges = [
            edge for edge in graph.edges
            if edge.target != goal.id and edge.source != goal.id
        ]

        for source in sorted(set(preds)):
            graph.add_edge(source, boss_door_id, EdgeType.BOSS_LOCKED, key_required=boss_door_id)
        graph.add_edge(boss_door_id, boss_id, EdgeType.PATH)
        graph.add_edge(boss_id, goal.id, EdgeType.PATH)
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
            queue = deque([source])
            while queue:
                current = queue.popleft()
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
            position=_bounded_free_position(
                graph,
                anchor_pos,
                [
                    (0, 1), (1, 0), (0, -1), (-1, 0),
                    (1, 1), (1, -1), (-1, 1), (-1, -1),
                    (0, 2), (2, 0), (0, -2), (-2, 0),
                ],
                rng,
                context,
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
            position=self._interpolate_pos(graph, item_edge.source, item_edge.target, 0.4, context),
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
            if graph.get_forward_shortest_path_length(item_id, e.source) <= 0:
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
                position=self._interpolate_pos(graph, gate_edge.source, gate_edge.target, 0.6, context),
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
            if item_id in graph.nodes:
                del graph.nodes[item_id]
            graph.edges = [
                existing_edge
                for existing_edge in graph.edges
                if existing_edge.source != item_id and existing_edge.target != item_id
            ]
            graph.add_edge(item_edge.source, item_edge.target, EdgeType.PATH)
            graph.sanitize()
            logger.debug(
                "AddItemGateRule: No pre-gate-valid edge found after placing item %s at %s; rolled back item-only insert",
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
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, int, int]:
        """Interpolate position between two nodes, then resolve collisions."""
        return self._interpolate_free_position(graph, src, tgt, t, context)


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
                position=_bounded_free_position(
                    graph,
                    hub_pos,
                    [(offset_r, offset_c), (offset_r + 1, offset_c), (offset_r, offset_c + 1)],
                    rng,
                    context,
                ),
                difficulty=context.get('difficulty', 0.5) * rng.uniform(0.6, 0.9),
            )
            graph.add_node(branch_start)
            graph.add_edge(hub_node.id, branch_start_id, EdgeType.PATH)
            
            # Add one more node to make branch meaningful (extended further from hub)
            branch_end_id = max(graph.nodes.keys()) + 1
            branch_end = MissionNode(
                id=branch_end_id,
                node_type=rng.choice(
                    self._reward_node_choices(
                        graph,
                        include_empty=True,
                        include_treasure=True,
                    )
                ),
                position=_bounded_free_position(
                    graph,
                    hub_pos,
                    [
                        (offset_r * LAYOUT_HUB_BRANCH_SPACING, offset_c * LAYOUT_HUB_BRANCH_SPACING),
                        (offset_r * LAYOUT_HUB_BRANCH_SPACING + 1, offset_c * LAYOUT_HUB_BRANCH_SPACING),
                        (offset_r * LAYOUT_HUB_BRANCH_SPACING, offset_c * LAYOUT_HUB_BRANCH_SPACING + 1),
                    ],
                    rng,
                    context,
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
            position=_bounded_free_position(
                graph,
                (anchor_pos[0], anchor_pos[1], 1),
                [(1, 0), (2, 0), (1, 1), (2, 1), (0, 1), (-1, 1)],
                rng,
                context,
            ),
            difficulty=context.get('difficulty', 0.5) * 0.8,
        )
        graph.add_node(room)
        graph.add_edge(stairs_up_id, room_id, EdgeType.PATH)
        
        logger.info(f"AddStairsRule: Stairs at ({anchor_pos[0]}, {anchor_pos[1]}) connecting floors 0<->1")
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
        secret_pos = _bounded_free_position(
            graph,
            anchor_pos,
            [(-1, 2), (0, 2), (1, 2), (-1, 3), (0, 3), (1, 3), (2, 1), (-2, 1)],
            rng,
            context,
        )
        
        # Create secret room
        secret_id = max(graph.nodes.keys()) + 1
        secret_node = MissionNode(
            id=secret_id,
            node_type=NodeType.SECRET,
            position=secret_pos,
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
        
        # Add reward in secret room. Bonus small keys are only allowed when the
        # graph is actually key-starved; otherwise prefer treasure/items.
        reward_pos = _bounded_free_position(
            graph,
            secret_node.position,
            [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1)],
            rng,
            context,
        )
        reward_id = max(graph.nodes.keys()) + 1
        reward_node = MissionNode(
            id=reward_id,
            node_type=rng.choice(
                self._reward_node_choices(
                    graph,
                    include_treasure=True,
                    include_protection_item=True,
                )
            ),
            position=reward_pos,
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
        for node1 in nodes[:len(nodes)//2]:
            for node2 in nodes[len(nodes)//2:]:
                dist = graph.get_shortest_path_length(node1, node2)
                if dist >= 4:  # Far enough to warrant teleport
                    return True
        return False
    
    def apply(self, graph: MissionGraph, context: Dict[str, Any]) -> MissionGraph:
        """Add teleport/warp between distant nodes."""
        
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
        
        logger.info(f"AddTeleportRule: Warp between {node1} <-> {node2} (saved {dist} hops)")
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
            position=self._interpolate_pos(graph, key_edge.source, key_edge.target, 0.3, context),
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
        key_node_types = {NodeType.KEY}
        lock_candidates: List[Tuple[int, MissionEdge]] = []
        for i, e in enumerate(graph.edges):
            if e.edge_type != EdgeType.PATH:
                continue
            if e.source == key_id:
                continue
            if graph.get_forward_shortest_path_length(key_id, e.source) <= 0:
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
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, int, int]:
        """Interpolate position between two nodes, then resolve collisions."""
        return self._interpolate_free_position(graph, src, tgt, t, context)


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
        return None

    def _critical_path_pairs(self, graph: MissionGraph) -> Set[Tuple[Any, Any]]:
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
        pairs: Set[Tuple[Any, Any]] = set()
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            pairs.add((a, b))
            pairs.add((b, a))
        return pairs

    def _has_free_return_path_after_valve(self, graph: MissionGraph, edge: MissionEdge) -> bool:
        """Ensure the one-way drop has an ungated loop back to its source."""
        allowed = {
            EdgeType.PATH,
            EdgeType.SHORTCUT,
            EdgeType.WARP,
            EdgeType.STAIRS,
            EdgeType.HIDDEN,
        }
        adjacency: Dict[int, List[int]] = defaultdict(list)
        for other in graph.edges:
            if other is edge or other.edge_type not in allowed:
                continue
            adjacency[other.source].append(other.target)
            if other.edge_type in graph.BIDIRECTIONAL_EDGE_TYPES:
                adjacency[other.target].append(other.source)
        return self._bfs_path(adjacency, edge.target, edge.source) is not None
    
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
            cycle_steps: List[Tuple[Any, Any]] = []
            for i in range(len(cycle) - 1):
                cycle_steps.append((cycle[i], cycle[i + 1]))
            if cycle[0] != cycle[-1]:
                cycle_steps.append((cycle[-1], cycle[0]))

            all_candidates: List[Tuple[int, MissionEdge]] = []
            noncritical_candidates: List[Tuple[int, MissionEdge]] = []
            safe_candidates: List[Tuple[int, MissionEdge]] = []
            seen_edge_ids: Set[int] = set()

            for src, tgt in cycle_steps:
                for idx, edge in enumerate(graph.edges):
                    if idx in seen_edge_ids or edge.edge_type != EdgeType.PATH:
                        continue
                    if not (
                        (edge.source == src and edge.target == tgt)
                        or (edge.source == tgt and edge.target == src)
                    ):
                        continue
                    if not self._has_free_return_path_after_valve(graph, edge):
                        continue
                    seen_edge_ids.add(idx)
                    candidate = (idx, edge)
                    all_candidates.append(candidate)
                    touches_protected = (edge.source in protected_nodes) or (edge.target in protected_nodes)
                    on_critical = (
                        (edge.source, edge.target) in critical_pairs
                        or (edge.target, edge.source) in critical_pairs
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
                n for n in graph._adjacency.get(edge.target, []) if n != edge.source
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
        
        # Look for nodes with Manhattan distance <= 2 and path distance > 4
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
            target_node.node_type = rng.choice(
                self._reward_node_choices(
                    graph,
                    include_treasure=True,
                    include_protection_item=False,
                )
            )
        
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
                position=_bounded_free_position(
                    graph,
                    graph.nodes[target_node_id].position,
                    [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)],
                    rng,
                    context,
                ),
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
            if graph.get_forward_shortest_path_length(hub.id, e.source) <= 0:
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
            position=_bounded_free_position(
                graph,
                graph.nodes[switch_anchor].position,
                [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)],
                rng,
                context,
            ),
            difficulty=context.get('difficulty', 0.5) * 0.7,
            switch_id=switch_id,
        )
        graph.add_node(switch_node)
        graph.add_edge(switch_anchor, switch_id, EdgeType.PATH)
        
        # Place STATE_BLOCK in branch B guarding reward
        block_anchor = branch_b[-1] if len(branch_b) > 1 else branch_b[0]
        
        # Create reward node
        has_big_key = any(node.node_type == NodeType.BIG_KEY for node in graph.nodes.values())
        has_boss_door = any(node.node_type == NodeType.BOSS_DOOR for node in graph.nodes.values())
        reward_choices = [NodeType.ITEM, NodeType.TREASURE, NodeType.PROTECTION_ITEM]
        if has_boss_door and not has_big_key:
            reward_choices.append(NodeType.BIG_KEY)
        reward_id = max(graph.nodes.keys()) + 1
        reward_node = MissionNode(
            id=reward_id,
            node_type=rng.choice(reward_choices),
            position=_bounded_free_position(
                graph,
                graph.nodes[block_anchor].position,
                [(0, 1), (1, 0), (0, -1), (-1, 0), (2, 0), (0, 2), (-2, 0), (0, -2)],
                rng,
                context,
            ),
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
        # The current final-map vocabulary has one generic protection-item
        # tile. Keep graph-only experiments expressive, but use one stable
        # semantic identity in the spatially compilable profile so the tile
        # oracle does not conflate several mutually exclusive protections.
        hazard_type = (
            "LAVA"
            if bool(context.get("spatial_compilable", False))
            else rng.choice(hazard_types)
        )
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


class SoftGateRule(AddHazardGateRule):
    """
    Compatibility alias for reviewer terminology: "soft gate".

    Behavior is identical to AddHazardGateRule and models optional
    risk-reward traversal with a mitigation item path.
    """

    def __init__(self):
        super().__init__()
        self.name = "SoftGate"


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
            node_type=rng.choice(
                self._reward_node_choices(
                    graph,
                    include_treasure=True,
                    include_protection_item=False,
                )
            ),
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
        super().__init__("AddSkillChain", weight=0.20)
    
    def can_apply(self, graph: MissionGraph, context: Dict[str, Any]) -> bool:
        """Can apply if an ITEM node can anchor a downstream pedagogical chain."""
        if self._has_stage_variety_cap(graph):
            return False
        if self._has_existing_skill_chain(graph):
            return False
        items = graph.get_nodes_by_type(NodeType.ITEM)
        for item in items:
            path_nodes = self._find_forward_progression_path(graph, item.id)
            successors = self._eligible_successors(graph, path_nodes)
            if len(successors) >= 3 or self._eligible_direct_edges(graph, item.id) or self._select_progression_anchor(graph, item.id) is not None:
                return True
        return False
    
    def apply(self, graph: MissionGraph, context: Dict[str, Any]) -> MissionGraph:
        """Create skill chain after item acquisition."""
        if self._has_stage_variety_cap(graph):
            return graph
        if self._has_existing_skill_chain(graph):
            return graph
        rng = context.get('rng') or random
        
        # Find ITEM nodes with sufficient successors
        items = graph.get_nodes_by_type(NodeType.ITEM)
        candidates: List[Tuple[MissionNode, List[int], List[MissionNode]]] = []
        for item in items:
            path_nodes = self._find_forward_progression_path(graph, item.id)
            successors = self._eligible_successors(graph, path_nodes)
            if len(successors) >= 3:
                candidates.append((item, path_nodes, successors))
        
        if candidates:
            item_node, _, successors = rng.choice(candidates)
            
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

        fallback_candidates: List[Tuple[MissionNode, List[Tuple[int, MissionEdge]], Optional[int]]] = []
        for item in items:
            direct_edges = self._eligible_direct_edges(graph, item.id)
            anchor_id = self._select_progression_anchor(graph, item.id)
            if direct_edges or anchor_id is not None:
                fallback_candidates.append((item, direct_edges, anchor_id))

        if not fallback_candidates:
            return graph

        item_node, direct_edges, anchor_id = rng.choice(fallback_candidates)
        edge_idx: Optional[int] = None
        edge_target: Optional[int] = None
        matching_anchor_edges = [
            (candidate_idx, candidate_edge)
            for candidate_idx, candidate_edge in direct_edges
            if anchor_id is not None and int(candidate_edge.target) == int(anchor_id)
        ]
        if matching_anchor_edges:
            edge_idx, edge = rng.choice(matching_anchor_edges)
            edge_target = int(edge.target)
        elif anchor_id is not None:
            edge_target = int(anchor_id)
            if direct_edges:
                edge_idx, _ = rng.choice(direct_edges)
        elif direct_edges:
            edge_idx, edge = rng.choice(direct_edges)
            edge_target = int(edge.target)
        if edge_target is None:
            return graph

        next_id = max(graph.nodes.keys(), default=-1) + 1
        base_pos = graph.nodes[item_node.id].position
        base_floor = base_pos[2] if len(base_pos) > 2 else 0
        base_x = base_pos[0]
        base_y = base_pos[1]

        created_nodes: List[MissionNode] = []
        for offset_idx, (node_type, offsets, node_kwargs) in enumerate(
            [
                (
                    NodeType.TUTORIAL_PUZZLE,
                    [(1, 0), (0, 1), (0, -1), (-1, 0)],
                    {
                        "difficulty": 0.2,
                        "difficulty_rating": "SAFE",
                        "is_tutorial": True,
                        "tension_value": 0.1,
                    },
                ),
                (
                    NodeType.COMBAT_PUZZLE,
                    [(2, 0), (1, 1), (1, -1), (0, 2), (0, -2)],
                    {
                        "difficulty": 0.5,
                        "difficulty_rating": "MODERATE",
                        "tension_value": 0.5,
                    },
                ),
                (
                    NodeType.COMPLEX_PUZZLE,
                    [(3, 0), (2, 1), (2, -1), (1, 2), (1, -2)],
                    {
                        "difficulty": 0.8,
                        "difficulty_rating": "HARD",
                        "tension_value": 0.7,
                    },
                ),
            ]
        ):
            node = MissionNode(
                id=next_id + offset_idx,
                node_type=node_type,
                position=_bounded_free_position(
                    graph,
                    (base_x, base_y, base_floor),
                    offsets,
                    rng,
                    context,
                ),
                **node_kwargs,
            )
            graph.add_node(node)
            created_nodes.append(node)

        if edge_idx is not None:
            graph.edges = [existing for i, existing in enumerate(graph.edges) if i != edge_idx]
        graph.add_edge(item_node.id, created_nodes[0].id, EdgeType.PATH)
        graph.add_edge(created_nodes[0].id, created_nodes[1].id, EdgeType.PATH)
        graph.add_edge(created_nodes[1].id, created_nodes[2].id, EdgeType.PATH)
        graph.add_edge(created_nodes[2].id, edge_target, EdgeType.PATH)
        graph.sanitize()

        logger.info(
            "AddSkillChainRule: Inserted pedagogical chain after item %s (%s) before node %s",
            item_node.id,
            item_node.item_type,
            edge_target,
        )
        return graph

    def _eligible_successors(
        self,
        graph: MissionGraph,
        path_nodes: List[int],
    ) -> List[MissionNode]:
        """Filter forward path nodes to progression-relevant rooms only."""
        blocked_types = {
            NodeType.START,
            NodeType.GOAL,
            NodeType.BOSS,
            NodeType.BOSS_DOOR,
            NodeType.BIG_KEY,
        }
        seen: Set[int] = set()
        filtered: List[MissionNode] = []
        for node_id in path_nodes[1:]:
            node = graph.get_node(node_id)
            if node is None or node.id in seen:
                continue
            if node.node_type in blocked_types:
                continue
            seen.add(node.id)
            filtered.append(node)
        return filtered

    def _find_forward_progression_path(
        self,
        graph: MissionGraph,
        item_id: int,
    ) -> List[int]:
        """Find a forward-only path from an item toward the nearest climax anchor."""
        forward_adj = graph.get_forward_adjacency_map()
        candidate_paths: List[Tuple[int, int, List[int]]] = []
        for priority, node_type in enumerate((NodeType.BOSS_DOOR, NodeType.BOSS, NodeType.GOAL)):
            for target in graph.get_nodes_by_type(node_type):
                path = self._find_forward_path_nodes(forward_adj, item_id, target.id)
                if path and len(path) >= 2:
                    candidate_paths.append((len(path), priority, path))
        if not candidate_paths:
            return []
        candidate_paths.sort(key=lambda entry: (entry[0], entry[1]))
        return list(candidate_paths[0][2])

    def _select_progression_anchor(
        self,
        graph: MissionGraph,
        item_id: int,
    ) -> Optional[int]:
        """
        Choose a downstream anchor that lets an item branch rejoin the climax route.
        """
        forward_path = self._find_forward_progression_path(graph, item_id)
        if len(forward_path) >= 2:
            return int(forward_path[1])

        start = graph.get_start_node()
        goal = graph.get_goal_node()
        if start is None or goal is None:
            return None
        critical_path = self._find_forward_path_nodes(
            graph.get_forward_adjacency_map(),
            start.id,
            goal.id,
        )
        if not critical_path:
            return None

        preferred_types = {NodeType.BOSS_DOOR, NodeType.BOSS, NodeType.GOAL}
        for node_id in critical_path:
            node = graph.get_node(node_id)
            if node is not None and node.node_type in preferred_types:
                return int(node_id)
        if len(critical_path) >= 2:
            return int(critical_path[-1])
        return None

    def _has_existing_skill_chain(self, graph: MissionGraph) -> bool:
        """Return True when the graph already has one valid item-to-climax skill chain."""
        stage_order = (
            NodeType.TUTORIAL_PUZZLE,
            NodeType.COMBAT_PUZZLE,
            NodeType.COMPLEX_PUZZLE,
        )
        for item in graph.get_nodes_by_type(NodeType.ITEM):
            path_nodes = self._find_forward_progression_path(graph, item.id)
            if len(path_nodes) < 2:
                continue
            prev_index = 0
            matched_count = 0
            for stage in stage_order:
                matched_index = None
                for idx in range(prev_index + 1, len(path_nodes)):
                    node = graph.get_node(int(path_nodes[idx]))
                    if node is not None and node.node_type == stage:
                        matched_index = idx
                        break
                if matched_index is None:
                    break
                prev_index = matched_index
                matched_count += 1
            if matched_count == len(stage_order):
                return True
        return False

    def _has_stage_variety_cap(self, graph: MissionGraph) -> bool:
        """Treat one complete tutorial/combat/complex set as enough pedagogy for one topology."""
        counts = {
            NodeType.TUTORIAL_PUZZLE: 0,
            NodeType.COMBAT_PUZZLE: 0,
            NodeType.COMPLEX_PUZZLE: 0,
        }
        for node in graph.nodes.values():
            if node.node_type in counts:
                counts[node.node_type] += 1
        return all(count >= 1 for count in counts.values())

    def _find_forward_path_nodes(
        self,
        adjacency: Dict[int, List[int]],
        start_id: int,
        goal_id: int,
    ) -> List[int]:
        """Return one forward-only shortest path over explicit edge direction."""
        if start_id == goal_id:
            return [start_id]

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

    def _eligible_direct_edges(
        self,
        graph: MissionGraph,
        item_id: int,
    ) -> List[Tuple[int, MissionEdge]]:
        """Fallback edges that can be rewritten into an explicit skill chain."""
        blocked_types = {
            NodeType.START,
            NodeType.GOAL,
            NodeType.BOSS,
            NodeType.BOSS_DOOR,
            NodeType.BIG_KEY,
        }
        candidates: List[Tuple[int, MissionEdge]] = []
        for edge_idx, edge in enumerate(graph.edges):
            if edge.source != item_id or edge.edge_type != EdgeType.PATH:
                continue
            target = graph.get_node(edge.target)
            if target is None or target.node_type in blocked_types:
                continue
            candidates.append((edge_idx, edge))
        return candidates


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
            farm_node = MissionNode(
                id=farm_id,
                node_type=NodeType.RESOURCE_FARM,
                position=_bounded_free_position(
                    graph,
                    farm_pos,
                    [(-1, 1), (0, 1), (1, 1), (-1, 2), (0, 2), (1, 2)],
                    rng,
                    context,
                ),
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
        original = copy.deepcopy(graph)
        
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
        
        for branch in selected_branches:
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
            return original
        
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
                return original

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
        item_node, _original_dist = candidates[0]
        
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
        """Return whether a truly empty connector leaf can be removed."""
        for node in graph.nodes.values():
            degree = graph.get_node_degree(node.id)
            if degree == 1 and node.node_type == NodeType.EMPTY:
                if not node.is_hub and not node.is_secret:
                    return True
        return False
    
    def apply(self, graph: MissionGraph, context: Dict[str, Any]) -> MissionGraph:
        """Prune one empty connector leaf while preserving authored content."""
        dead_ends = []
        for node in graph.nodes.values():
            degree = graph.get_node_degree(node.id)
            if degree == 1 and node.node_type == NodeType.EMPTY:
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
