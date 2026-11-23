"""Gate, lock, switch, and barrier mission graph production rules."""

from __future__ import annotations

import copy
import logging
import math
import random
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

from src.generation.grammar.core_rules import ProductionRule
from src.generation.grammar.graph_types import (
    LAYOUT_HUB_BRANCH_SPACING,
    LAYOUT_HUB_RADIUS,
    EdgeType,
    MissionEdge,
    MissionGraph,
    MissionNode,
    NodeType,
)
from src.generation.grammar.rules.base_helpers import (
    _bounded_free_position,
    _layout_bounds,
)

logger = logging.getLogger(__name__)

__all__ = ['InsertSwitchRule', 'AddItemGateRule', 'AddFungibleLockRule', 'AddValveRule', 'AddHazardGateRule', 'SoftGateRule', 'AddGatekeeperRule', 'AddMultiLockRule']

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
        graph.sanitize()

        logger.info(f"InsertSwitchRule: Switch {switch_id} controls edge {edge.source}->{edge.target}")
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
        original_edge = graph.edges.pop(item_edge_idx)
        graph.sanitize()
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
            graph.add_edge(gate_id, gate_edge.target, edge_type=EdgeType.ITEM_GATE, item_required=item_name)

            logger.info(f"AddItemGateRule: Item {item_name} at {item_id}, gate at {gate_id}")
        else:
            if item_id in graph.nodes:
                del graph.nodes[item_id]
            graph.edges = [
                existing_edge
                for existing_edge in graph.edges
                if existing_edge.source != item_id and existing_edge.target != item_id
            ]
            graph.edges.append(original_edge)
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
        graph.edges.insert(min(int(key_edge_idx), len(graph.edges)), key_edge)
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


class AddHazardGateRule(ProductionRule):
    """
    ADVANCED RULE #9: Soft Gates with Risk-Reward
    
    Creates protection-gated paths (lava, spikes) with a reachable mitigation
    item. The current exact state model has no health budget, so this rule must
    not claim optional damage traversal until that state variable is modeled.
    
    Example: Zelda's lava rooms (cross with damage or get fire tunic first).
    
    Research: Adams & Dormans (2012) - Optional challenge paths
    """

    def __init__(self):
        super().__init__("AddHazardGate", weight=0.25)

    def can_apply(self, graph: MissionGraph, context: Dict[str, Any]) -> bool:
        """Require both a candidate path and a legal provider anchor."""
        if len(graph.nodes) < 4:
            return False
        normal_edges = [e for e in graph.edges if e.edge_type == EdgeType.PATH]
        provider_anchors = [
            node
            for node in graph.nodes.values()
            if graph.get_node_degree(node.id) <= 2
            and node.node_type in {NodeType.EMPTY, NodeType.ENEMY, NodeType.PUZZLE}
        ]
        return len(normal_edges) >= 2 and bool(provider_anchors)

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

        # The provider must be reachable while the candidate hazard edge stays
        # closed. Otherwise the rule creates the same item-behind-own-gate
        # deadlock as a malformed lock/key pair.
        start = graph.get_start_node()
        reachable_pre_gate = (
            graph.get_reachable_nodes(
                start.id,
                excluded_edges={(hazard_edge.source, hazard_edge.target)},
            )
            if start is not None
            else set()
        )
        side_nodes = [
            n for n in graph.nodes.values()
            if n.id in reachable_pre_gate
            and graph.get_node_degree(n.id) <= 2
            and n.node_type in {NodeType.EMPTY, NodeType.ENEMY, NodeType.PUZZLE}
        ]
        if not side_nodes:
            return graph

        side_node = rng.choice(side_nodes)
        graph.edges[hazard_edge_idx].edge_type = EdgeType.HAZARD
        graph.edges[hazard_edge_idx].hazard_damage = rng.randint(1, 3)
        graph.edges[hazard_edge_idx].protection_item_id = protection_item

        protection_id = max(graph.nodes.keys()) + 1
        protection_node = MissionNode(
            id=protection_id,
            node_type=NodeType.PROTECTION_ITEM,
            position=_bounded_free_position(
                graph,
                side_node.position,
                [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1)],
                rng,
                context,
            ),
            difficulty=context.get('difficulty', 0.5) * 0.5,
            item_type=protection_item,
        )
        graph.add_node(protection_node)
        graph.add_edge(side_node.id, protection_id, EdgeType.PATH)

        logger.info(
            "AddHazardGateRule: %s hazard at %s->%s, protection at %s",
            hazard_type,
            hazard_edge.source,
            hazard_edge.target,
            protection_id,
        )

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
        protected_types = {NodeType.START, NodeType.GOAL, NodeType.BOSS_DOOR, NodeType.BOSS, NodeType.KEY, NodeType.BIG_KEY, NodeType.ITEM, NodeType.SWITCH, NodeType.TOKEN, NodeType.PROTECTION_ITEM}
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
        protected_types = {NodeType.START, NodeType.GOAL, NodeType.BOSS_DOOR, NodeType.BOSS, NodeType.KEY, NodeType.BIG_KEY, NodeType.ITEM, NodeType.SWITCH, NodeType.TOKEN, NodeType.PROTECTION_ITEM}

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
        original = graph

        # Find hub with 3+ branches
        hubs = [n for n in graph.nodes.values() if graph.get_node_degree(n.id) >= 3]
        if not hubs:
            return graph

        hub = rng.choice(hubs)
        branches = graph.get_branches_from_hub(hub.id)

        if len(branches) < 3:
            return graph

        hub_id = hub.id
        graph = copy.deepcopy(graph)
        hub = graph.nodes[hub_id]

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


