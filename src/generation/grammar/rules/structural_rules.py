"""Topological and structural layout mission graph production rules."""

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

__all__ = ['MergeRule', 'CreateHubRule', 'AddStairsRule', 'AddTeleportRule', 'FormBigRoomRule', 'AddSectorRule', 'AddEntangledBranchesRule', 'SplitRoomRule']

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
        graph.sanitize()
        return graph

    def _graph_distance(self, graph: MissionGraph, start: int, end: int) -> int:
        """BFS to find shortest path distance."""
        if start == end:
            return 0

        allowed_edges = {EdgeType.PATH, EdgeType.SHORTCUT, EdgeType.WARP, EdgeType.STAIRS, EdgeType.HIDDEN}
        safe_adj = {}
        for edge in graph.edges:
            if edge.edge_type in allowed_edges:
                safe_adj.setdefault(edge.source, []).append(edge.target)
                if edge.edge_type in graph.BIDIRECTIONAL_EDGE_TYPES:
                    safe_adj.setdefault(edge.target, []).append(edge.source)

        visited = {start}
        queue = deque([(start, 0)])

        while queue:
            current, dist = queue.popleft()
            for neighbor in safe_adj.get(current, []):
                if neighbor == end:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return 999  # Not connected


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


class AddTeleportRule(ProductionRule):
    """
    Add a teleport/warp connection between distant nodes.
    
    Creates shortcuts via WARP edges that don't require spatial adjacency.
    Useful for late-game backtracking or connecting distant regions.
    """

    def __init__(self):
        super().__init__("AddTeleport", weight=0.2)

    def _graph_distance(self, graph: MissionGraph, start: int, end: int) -> int:
        """BFS to find shortest path distance avoiding locks."""
        if start == end:
            return 0

        allowed_edges = {EdgeType.PATH, EdgeType.SHORTCUT, EdgeType.WARP, EdgeType.STAIRS, EdgeType.HIDDEN}
        safe_adj = {}
        for edge in graph.edges:
            if edge.edge_type in allowed_edges:
                safe_adj.setdefault(edge.source, []).append(edge.target)
                if edge.edge_type in graph.BIDIRECTIONAL_EDGE_TYPES:
                    safe_adj.setdefault(edge.target, []).append(edge.source)

        visited = {start}
        queue = deque([(start, 0)])

        while queue:
            current, dist = queue.popleft()
            for neighbor in safe_adj.get(current, []):
                if neighbor == end:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return 999999

    def can_apply(self, graph: MissionGraph, context: Dict[str, Any]) -> bool:
        """Can apply if we have nodes that are far apart."""
        if len(graph.nodes) < 5:
            return False

        # Check if any nodes are topologically distant
        nodes = list(graph.nodes.keys())
        for node1 in nodes[:len(nodes)//2]:
            for node2 in nodes[len(nodes)//2:]:
                dist = self._graph_distance(graph, node1, node2)
                if dist >= 4 and dist < 999:  # Far enough to warrant teleport
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
                dist = self._graph_distance(graph, node1, node2)
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

            floor_a = int(node_a.position[2]) if len(node_a.position) > 2 else 0
            floor_b = int(node_b.position[2]) if len(node_b.position) > 2 else 0
            if floor_a != floor_b:
                continue

            # Check spatial adjacency (Manhattan distance = 1 horizontally)
            dist = abs(node_a.position[0] - node_b.position[0]) + abs(node_a.position[1] - node_b.position[1])
            if dist <= 2:  # Adjacent or close
                # Not START/GOAL/BOSS_DOOR
                if (node_a.node_type not in [NodeType.START, NodeType.GOAL, NodeType.BOSS_DOOR, NodeType.BOSS, NodeType.KEY, NodeType.BIG_KEY, NodeType.ITEM, NodeType.SWITCH, NodeType.TOKEN, NodeType.PROTECTION_ITEM] and
                    node_b.node_type not in [NodeType.START, NodeType.GOAL, NodeType.BOSS_DOOR, NodeType.BOSS, NodeType.KEY, NodeType.BIG_KEY, NodeType.ITEM, NodeType.SWITCH, NodeType.TOKEN, NodeType.PROTECTION_ITEM]):
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

            floor_a = int(node_a.position[2]) if len(node_a.position) > 2 else 0
            floor_b = int(node_b.position[2]) if len(node_b.position) > 2 else 0
            if floor_a != floor_b:
                continue

            dist = abs(node_a.position[0] - node_b.position[0]) + abs(node_a.position[1] - node_b.position[1])
            if dist <= 2:
                if (node_a.node_type not in [NodeType.START, NodeType.GOAL, NodeType.BOSS_DOOR, NodeType.BOSS, NodeType.KEY, NodeType.BIG_KEY, NodeType.ITEM, NodeType.SWITCH, NodeType.TOKEN, NodeType.PROTECTION_ITEM] and
                    node_b.node_type not in [NodeType.START, NodeType.GOAL, NodeType.BOSS_DOOR, NodeType.BOSS, NodeType.KEY, NodeType.BIG_KEY, NodeType.ITEM, NodeType.SWITCH, NodeType.TOKEN, NodeType.PROTECTION_ITEM]):
                    candidates.append((edge.source, edge.target))

        if not candidates:
            return graph

        node_a_id, node_b_id = rng.choice(candidates)
        node_a = graph.nodes[node_a_id]
        node_b = graph.nodes[node_b_id]

        floor = int(node_a.position[2]) if len(node_a.position) > 2 else 0

        # Merge node_b into node_a
        merged_pos = (
            min(node_a.position[0], node_b.position[0]),
            min(node_a.position[1], node_b.position[1]),
            floor,
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
            # Transfer to node_a — use setdefault so node_a_id is auto-created if missing
            node_a_neighbors = graph._adjacency.setdefault(node_a_id, [])
            for neighbor in neighbors:
                if neighbor != node_a_id and neighbor not in node_a_neighbors:
                    node_a_neighbors.append(neighbor)

        for nid, neighbors in list(graph._adjacency.items()):
            if node_b_id in neighbors:
                neighbors.remove(node_b_id)
                if node_a_id not in neighbors and nid != node_a_id:
                    neighbors.append(node_a_id)

        # Remove self-loops
        graph.edges = [e for e in graph.edges if e.source != e.target]

        graph.sanitize()
        logger.info(f"FormBigRoomRule: Merged nodes {node_a_id} and {node_b_id} into {room_size} big room")
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
        added_nodes: List[int] = []

        for i in range(chain_length):
            # Create new node in sector
            new_id = max(graph.nodes.keys()) + 1
            anchor = graph.nodes[current_id].position
            position = _bounded_free_position(
                graph,
                anchor,
                [
                    (1, 0),
                    (0, 1),
                    (-1, 0),
                    (0, -1),
                    (1, 1),
                    (1, -1),
                    (-1, 1),
                    (-1, -1),
                ],
                rng,
                context,
            )
            occupied = {
                (
                    int(node.position[0]),
                    int(node.position[1]),
                    int(node.position[2]) if len(node.position) > 2 else 0,
                )
                for node in graph.nodes.values()
            }
            if position in occupied:
                added_set = set(added_nodes)
                graph.edges = [
                    edge
                    for edge in graph.edges
                    if edge.source not in added_set and edge.target not in added_set
                ]
                for added_id in added_nodes:
                    graph.nodes.pop(added_id, None)
                graph.sanitize()
                logger.debug(
                    "AddSectorRule: Insufficient free cells inside layout_bounds; rolled back sector."
                )
                return graph
            new_node = MissionNode(
                id=new_id,
                node_type=rng.choice([NodeType.ENEMY, NodeType.PUZZLE, NodeType.EMPTY]),
                position=position,
                difficulty=context.get('difficulty', 0.5) * rng.uniform(0.6, 0.9),
                sector_id=sector_id,
                sector_theme=sector_theme,
            )
            graph.add_node(new_node)
            graph.add_edge(current_id, new_id, EdgeType.PATH)
            added_nodes.append(new_id)
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
            switches_required=[switch_id],
        )
        graph.edges.append(block_edge)
        graph._adjacency[block_anchor].append(reward_id)

        # Bug 3 fix: Add return STAIRS edge and set key_id for BIG_KEY
        graph.add_edge(reward_id, block_anchor, edge_type=EdgeType.STAIRS)
        if reward_node.node_type == NodeType.BIG_KEY:
            boss_doors = graph.get_nodes_by_type(NodeType.BOSS_DOOR)
            if boss_doors:
                reward_node.key_id = boss_doors[0].id

        logger.info(f"AddEntangledBranchesRule: Switch {switch_id} (branch A) controls gate to {reward_id} (branch B)")
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

        # Always use STAIRS for virtual-layer connections.
        #
        # ONE_WAY edges are only valid inside pre-existing cycles (see
        # AddValveRule, which verifies a free return path before converting).
        # SplitRoomRule creates a *leaf* node — there is no cycle, so a
        # ONE_WAY drop would leave the balcony permanently unreachable
        # (in-degree 0), causing LogicGraphContractErrors downstream.
        #
        # STAIRS is bidirectional (player can climb up and jump down) and is
        # treated as non-spatial by the stitcher, so same-position vertical
        # connections work correctly without forcing physical 2D adjacency.
        graph.add_edge(original_node.id, virtual_id, EdgeType.STAIRS)

        logger.info(f"SplitRoomRule: Created virtual layer at node {original_node.id} (layer 1 = {virtual_id})")
        return graph


