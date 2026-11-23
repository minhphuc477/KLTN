"""Pruning, shortcut, and secret discovery mission graph production rules."""

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

__all__ = ['AddSecretRule', 'PruneGraphRule', 'AddItemShortcutRule', 'PruneDeadEndRule']

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


