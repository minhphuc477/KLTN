"""Challenge, combat, gauntlet, and pacing mission graph production rules."""

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

__all__ = ['AddBossGauntlet', 'AddForeshadowingRule', 'AddCollectionChallengeRule', 'AddArenaRule', 'AddSkillChainRule', 'AddPacingBreakerRule', 'AddResourceLoopRule']

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

        approach_nodes = sorted(set(preds))
        pred = approach_nodes[0]
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

        for source in approach_nodes:
            graph.add_edge(source, boss_door_id, EdgeType.BOSS_LOCKED, key_required=boss_door_id)
        graph.add_edge(boss_door_id, boss_id, EdgeType.PATH)
        graph.add_edge(boss_id, goal.id, EdgeType.PATH)
        graph.sanitize()

        # Spawn Big Key in a node guaranteed reachable before the boss lock.
        start = graph.get_start_node()
        excluded_edges = {
            (source, boss_door_id)
            for source in approach_nodes
        }
        excluded_nodes = {boss_door_id, goal.id}
        pre_lock_reachable: Set[int] = set()
        if start is not None:
            pre_lock_reachable = graph.get_reachable_nodes(
                start.id,
                excluded_edges=excluded_edges,
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
                    if (current, neighbor) in excluded_edges:
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
            # Prefer placements that still allow returning to at least one
            # boss approach without crossing any boss-door edge.
            if any(
                _reachable_without_edges(node_id, approach)
                for approach in approach_nodes
            ):
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
            approach_distances = [
                graph.get_shortest_path_length(node_id, approach)
                for approach in approach_nodes
            ]
            reachable_distances = [distance for distance in approach_distances if distance >= 0]
            dist_to_approach = min(reachable_distances) if reachable_distances else -1
            return (max(0, dist_from_start), max(0, dist_to_approach))

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
            target_node_id = branch[-1]

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

        def _rollback_tokens() -> None:
            token_set = set(token_ids)
            for token_id in token_set:
                graph.nodes.pop(token_id, None)
            graph.edges = [
                edge
                for edge in graph.edges
                if edge.source not in token_set and edge.target not in token_set
            ]
            graph.sanitize()

        if len(token_ids) < 2:
            _rollback_tokens()
            logger.debug(
                "AddCollectionChallengeRule: Fewer than two usable branches; "
                "rolled back %d token insert(s)",
                len(token_ids),
            )
            return graph
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
        _rollback_tokens()
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
                n.node_type not in [NodeType.START, NodeType.GOAL, NodeType.BOSS_DOOR, NodeType.BOSS, NodeType.ARENA, NodeType.KEY, NodeType.BIG_KEY, NodeType.ITEM, NodeType.SWITCH, NodeType.TOKEN, NodeType.PROTECTION_ITEM])
        ]

        if not candidates:
            return graph

        arena_node = rng.choice(candidates)

        # Mark as arena
        arena_node.node_type = NodeType.ARENA
        arena_node.is_arena = True

        # Convert incoming edges to SHUTTER type
        incoming_sources = []
        for i, edge in enumerate(graph.edges):
            if edge.target == arena_node.id and edge.edge_type == EdgeType.PATH:
                graph.edges[i].edge_type = EdgeType.SHUTTER
                incoming_sources.append(edge.source)

        goal = graph.get_goal_node()
        start = graph.get_start_node()
        goal_reachable = goal is not None and graph.get_forward_shortest_path_length(arena_node.id, goal.id) > 0
        start_reachable = start is not None and graph.get_forward_shortest_path_length(arena_node.id, start.id) > 0

        if not (goal_reachable or start_reachable):
            # Trapped leaf node: add a return STAIRS edge
            if incoming_sources:
                graph.add_edge(arena_node.id, incoming_sources[0], edge_type=EdgeType.STAIRS)
            else:
                return graph  # Abort if somehow no incoming paths

        logger.info(f"AddArenaRule: Node {arena_node.id} converted to combat arena with shutters")
        return graph


class AddSkillChainRule(ProductionRule):
    """
    WAVE 3 RULE #1: Tutorial Sequences (Learn -> Practice -> Master)
    
    After player acquires an item, creates a 3-node pedagogical sequence:
    1. TUTORIAL_PUZZLE: Safe room teaching item use (no enemies)
    2. COMBAT_PUZZLE: Moderate challenge (item + enemies)
    3. COMPLEX_PUZZLE: Hard challenge (item + previous mechanics)
    
    Example: Get Bow -> Shoot target -> Kill enemies with bow -> Complex archery puzzle
    
    Research: Nintendo's "kishotenketsu" pedagogy (introduction-development-twist-conclusion)
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
            NodeType.KEY, NodeType.ITEM, NodeType.SWITCH, NodeType.TOKEN, NodeType.PROTECTION_ITEM,
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
            NodeType.KEY, NodeType.ITEM, NodeType.SWITCH, NodeType.TOKEN, NodeType.PROTECTION_ITEM,
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
        protected_types = {NodeType.START, NodeType.GOAL, NodeType.BOSS_DOOR, NodeType.BOSS, NodeType.KEY, NodeType.BIG_KEY, NodeType.ITEM, NodeType.SWITCH, NodeType.TOKEN, NodeType.PROTECTION_ITEM}
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

            # The farm is a pre-gate spur. Connecting it directly to START via
            # a bidirectional SHORTCUT would also create START -> farm and can
            # bypass every progression gate between START and gate_source.
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


