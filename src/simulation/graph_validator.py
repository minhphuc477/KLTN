"""Graph-guided dungeon validation over mission topology and room grids."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from src.simulation.edge_logic import edge_type_from_data
from src.simulation.state import (
    WALKABLE_IDS,
    is_graph_goal_node as _is_graph_goal_node,
    is_graph_start_node as _is_graph_start_node,
)

logger = logging.getLogger(__name__)


class GraphGuidedValidator:
    """
    Validator that uses graph topology to determine dungeon solvability.
    
    Instead of pathfinding through a stitched grid (which fails when rooms are missing),
    this validator:
    1. Uses the graph to understand logical room connectivity
    2. Validates that paths exist WITHIN each room
    3. Verifies that connected rooms have traversable doorways
    4. Uses graph-based BFS to determine if START can reach TRIFORCE
    
    This approach handles the VGLC dataset limitation where some logical rooms
    are missing from the physical room data.
    """
    
    def __init__(self):
        """Initialize the graph-guided validator."""
        self.validation_cache = {}
    
    def _normalize_rooms(self, rooms: Dict) -> Dict[int, Any]:
        """
        Normalize room dictionary to use integer keys.
        
        Handles two input formats:
        - Dungeon objects: keys are tuples like (0, 0), (0, 1)
        - DungeonData objects: keys are strings like '0', '1'
        
        Args:
            rooms: Dictionary with either tuple or string keys
            
        Returns:
            Dictionary with integer keys and room data
        """
        normalized = {}
        used_room_ids: Set[int] = set()
        tuple_key_to_id: Dict[Tuple[Any, ...], int] = {}

        tuple_keys = [key for key in rooms.keys() if isinstance(key, tuple)]
        for key in rooms.keys():
            if isinstance(key, str):
                try:
                    used_room_ids.add(int(key))
                except ValueError:
                    continue
            elif isinstance(key, int):
                used_room_ids.add(int(key))

        next_tuple_room_id = (max(used_room_ids) + 1) if used_room_ids else 0
        for key in sorted(tuple_keys, key=repr):
            while next_tuple_room_id in used_room_ids:
                next_tuple_room_id += 1
            tuple_key_to_id[key] = next_tuple_room_id
            used_room_ids.add(next_tuple_room_id)
            next_tuple_room_id += 1

        for key, room_data in rooms.items():
            try:
                # Handle tuple keys (e.g., from Dungeon objects)
                if isinstance(key, tuple):
                    room_id = tuple_key_to_id[key]
                    logger.debug("Normalized tuple key %s to %s", key, room_id)
                # Handle string keys (e.g., from DungeonData)
                elif isinstance(key, str):
                    room_id = int(key)
                # Already an integer
                elif isinstance(key, int):
                    room_id = key
                else:
                    logger.warning("Unknown room key type: %s, key=%s", type(key), key)
                    continue
                    
                normalized[room_id] = room_data
            except (ValueError, TypeError) as e:
                logger.error("Failed to normalize room key %s: %s", key, e)
                continue
        
        return normalized
    
    def validate_dungeon_with_graph(self, dungeon_data, stitched_result=None) -> 'GraphValidationResult':
        """
        Validate a dungeon using its graph topology.
        
        Args:
            dungeon_data: DungeonData object with rooms and graph
            stitched_result: Optional StitchedDungeon (for visualization)
            
        Returns:
            GraphValidationResult with detailed analysis
        """
        del stitched_result
        
        graph = dungeon_data.graph
        rooms = dungeon_data.rooms
        
        # Normalize room keys to handle both Dungeon (tuple keys) and DungeonData (string keys)
        normalized_rooms = self._normalize_rooms(rooms)
        existing_room_ids = set(normalized_rooms.keys())
        
        logger.debug("Normalized %d rooms to %d integer IDs", len(rooms), len(existing_room_ids))
        
        # Step 1: Find START and goal nodes from graph.
        # Accept both the older explicit flags and the repo's current
        # START/GOAL typed mission-graph schema used by topology generation.
        start_node = None
        triforce_node = None
        
        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            if _is_graph_start_node(node_data):
                start_node = node_id
            if _is_graph_goal_node(node_data):
                triforce_node = node_id
        
        if start_node is None or triforce_node is None:
            return GraphValidationResult(
                is_solvable=False,
                graph_path=[],
                missing_rooms=[],
                room_validations={},
                error_message="No START or TRIFORCE node found in graph"
            )
        
        # Step 2: Find shortest path in graph from START to TRIFORCE
        try:
            graph_path = nx.shortest_path(graph, source=start_node, target=triforce_node)
        except nx.NetworkXNoPath:
            return GraphValidationResult(
                is_solvable=False,
                graph_path=[],
                missing_rooms=[],
                room_validations={},
                error_message="No path exists in graph from START to TRIFORCE"
            )
        
        # Step 3: Check which rooms in the path exist
        missing_rooms = [n for n in graph_path if n not in existing_room_ids]
        existing_in_path = [n for n in graph_path if n in existing_room_ids]
        
        # Step 4: Validate each existing room is internally traversable
        room_validations = {}
        for room_id in existing_in_path:
            room_data = normalized_rooms.get(room_id)
            if room_data is not None:
                room_grid = room_data.grid
                is_traversable, floor_count = self._validate_room_traversability(room_grid)
                room_validations[room_id] = {
                    'is_traversable': is_traversable,
                    'floor_count': floor_count,
                    'shape': room_grid.shape
                }
        
        # Step 5: Determine solvability based on graph analysis
        # A dungeon is "graph-solvable" if:
        # - All existing rooms in the path are internally traversable
        # - OR we can find an alternate path using only existing rooms
        
        all_existing_traversable = all(
            rv['is_traversable'] for rv in room_validations.values()
        )
        
        # Try to find a path using only existing rooms
        subgraph_path = self._find_path_in_existing_rooms(
            graph, start_node, triforce_node, existing_room_ids
        )
        if subgraph_path is not None:
            for room_id in subgraph_path:
                room_data = normalized_rooms.get(room_id)
                if room_id in room_validations or room_data is None:
                    continue
                room_grid = room_data.grid
                is_traversable, floor_count = self._validate_room_traversability(room_grid)
                room_validations[room_id] = {
                    'is_traversable': is_traversable,
                    'floor_count': floor_count,
                    'shape': room_grid.shape,
                }

        subgraph_traversable = bool(subgraph_path) and all(
            room_id in existing_room_ids
            and bool(room_validations.get(room_id, {}).get('is_traversable', False))
            for room_id in (subgraph_path or [])
        )
        is_solvable = bool(
            (len(missing_rooms) == 0 and all_existing_traversable)
            or subgraph_traversable
        )
        
        # Calculate graph-based metrics
        connectivity_score = len(existing_in_path) / len(graph_path) if graph_path else 0
        
        return GraphValidationResult(
            is_solvable=is_solvable,
            graph_path=graph_path,
            subgraph_path=subgraph_path or [],
            missing_rooms=missing_rooms,
            room_validations=room_validations,
            connectivity_score=connectivity_score,
            start_node=start_node,
            triforce_node=triforce_node,
            error_message="" if is_solvable else f"Path requires {len(missing_rooms)} missing rooms"
        )
    
    def _validate_room_traversability(self, room_grid: np.ndarray) -> Tuple[bool, int]:
        """
        Check if a room is internally traversable.
        
        A room is traversable if:
        - It has floor tiles
        - Floor tiles form a connected region
        """
        floor_mask = np.isin(room_grid, list(WALKABLE_IDS))
        floor_count = np.sum(floor_mask)
        
        if floor_count == 0:
            return False, 0
        
        # Check connectivity using flood fill
        visited = np.zeros_like(floor_mask, dtype=bool)
        positions = np.argwhere(floor_mask)
        
        if len(positions) == 0:
            return False, 0
        
        # Start flood fill from first floor position
        start = tuple(positions[0])
        stack = [start]
        connected_count = 0
        
        while stack:
            r, c = stack.pop()
            if visited[r, c]:
                continue
            if not floor_mask[r, c]:
                continue
            
            visited[r, c] = True
            connected_count += 1
            
            # Add neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < room_grid.shape[0] and 0 <= nc < room_grid.shape[1]:
                    if not visited[nr, nc] and floor_mask[nr, nc]:
                        stack.append((nr, nc))
        
        # Room is traversable if most floor tiles are connected
        is_traversable = connected_count >= 0.5 * floor_count
        return is_traversable, int(floor_count)
    
    def _find_path_in_existing_rooms(self, graph, start_node, end_node, 
                                     existing_room_ids: Set[int]) -> Optional[List[int]]:
        """
        Try to find a path that only uses existing rooms.
        
        Uses BFS on the subgraph of existing rooms.
        """
        # Create subgraph with only existing rooms
        existing_nodes = [n for n in graph.nodes() if n in existing_room_ids]
        
        if start_node not in existing_nodes or end_node not in existing_nodes:
            return None
        
        subgraph = graph.subgraph(existing_nodes).copy()
        
        try:
            path = nx.shortest_path(subgraph, source=start_node, target=end_node)
            return path
        except nx.NetworkXNoPath:
            return None
        except nx.NodeNotFound:
            return None
    
    def validate_with_edge_types(self, dungeon_data, inventory_start: Dict = None) -> 'GraphValidationResult':
        """
        Validate considering edge types (locked doors, bombable walls, etc.)
        
        This performs a state-space search on the GRAPH, where:
        - States = (current_node, keys_held, bombs_held, doors_opened)
        - Edges are traversable based on their type and current inventory
        
        Args:
            dungeon_data: DungeonData with rooms and graph
            inventory_start: Initial inventory (default: no keys, no bombs)
            
        Returns:
            GraphValidationResult with solution path
        """
        if inventory_start is None:
            inventory_start = {'keys': 0, 'bombs': 0, 'boss_key': False}
        
        graph = dungeon_data.graph
        rooms = dungeon_data.rooms
        
        # Normalize room keys to handle both Dungeon (tuple keys) and DungeonData (string keys)
        normalized_rooms = self._normalize_rooms(rooms)
        existing_room_ids = set(normalized_rooms.keys())
        
        logger.debug("Edge-type validation: normalized %d rooms to %d IDs", len(rooms), len(existing_room_ids))
        
        # Find START and goal nodes. Accept both explicit flags and the
        # topology generator's START/GOAL typed schema.
        start_node = None
        triforce_node = None
        key_nodes = []
        bomb_nodes = []
        boss_key_nodes = []
        item_nodes = []
        
        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            if _is_graph_start_node(node_data):
                start_node = node_id
            if _is_graph_goal_node(node_data):
                triforce_node = node_id
            if node_data.get('has_key', False):
                key_nodes.append(node_id)
            node_tokens = " ".join(
                str(node_data.get(key, ""))
                for key in ("type", "node_type", "label", "contents", "items")
            ).lower()
            if 'bomb' in node_tokens:
                bomb_nodes.append(node_id)
            if (
                bool(node_data.get("has_boss_key", False))
                or bool(node_data.get("is_boss_key", False))
                or "boss_key" in node_tokens
                or "big_key" in node_tokens
            ):
                boss_key_nodes.append(node_id)
            if (
                bool(node_data.get("has_item", False))
                or bool(node_data.get("is_item", False))
                or "key_item" in node_tokens
                or "item_minor" in node_tokens
            ):
                item_nodes.append(node_id)
        
        if start_node is None or triforce_node is None:
            return GraphValidationResult(
                is_solvable=False,
                graph_path=[],
                missing_rooms=[],
                room_validations={},
                error_message="No START or TRIFORCE in graph"
            )
        
        # State-space BFS on the graph. Consumable resources are decremented
        # only when opening a previously unopened edge; boss keys and traversal
        # items are persistent dungeon affordances.
        initial_state = (
            start_node,
            frozenset(),  # collected items
            frozenset(),  # opened doors (edge tuples)
            int(inventory_start.get('keys', 0)),
            int(inventory_start.get('bombs', 0)),
            bool(inventory_start.get('boss_key', False)),
            bool(inventory_start.get('item', False)),
        )
        
        queue = deque([initial_state])
        visited = {initial_state}
        parents: Dict[Any, Optional[Any]] = {initial_state: None}
        nodes_for_state: Dict[Any, Any] = {initial_state: start_node}
        
        while queue:
            state = queue.popleft()
            current_node, collected, opened, keys, bombs, boss_key, has_item = state
            
            # Check win
            if current_node == triforce_node:
                path = []
                key: Optional[Any] = state
                while key is not None:
                    path.append(nodes_for_state[key])
                    key = parents[key]
                path.reverse()
                return GraphValidationResult(
                    is_solvable=True,
                    graph_path=path,
                    subgraph_path=path,
                    missing_rooms=[n for n in path if n not in existing_room_ids],
                    room_validations={},
                    connectivity_score=1.0,
                    start_node=start_node,
                    triforce_node=triforce_node,
                    error_message=""
                )
            
            # Collect items at current node
            new_collected = collected
            new_keys = keys
            new_bombs = bombs
            new_boss_key = boss_key
            new_has_item = has_item
            if current_node not in collected and current_node in key_nodes:
                new_collected = collected | {current_node}
                new_keys = keys + 1
            if current_node not in collected and current_node in bomb_nodes:
                new_collected = new_collected | {current_node}
                node_data = graph.nodes[current_node]
                new_bombs += int(max(1, node_data.get("bomb_count", 1)))
            if current_node not in collected and current_node in boss_key_nodes:
                new_collected = new_collected | {current_node}
                new_boss_key = True
            if current_node not in collected and current_node in item_nodes:
                new_collected = new_collected | {current_node}
                new_has_item = True
            
            # Explore edges
            for neighbor in graph.neighbors(current_node):
                edge_data = graph.get_edge_data(current_node, neighbor) or {}
                edge_type = edge_type_from_data(edge_data)
                edge_key = tuple(sorted((current_node, neighbor), key=lambda value: (type(value).__name__, str(value))))
                
                can_traverse = False
                new_opened = opened
                use_key = False
                use_bomb = False
                
                if edge_type in {'open', '', 'path', 'stair'}:
                    can_traverse = True
                elif edge_type in {'locked', 'key_locked'}:
                    if new_keys > 0 or edge_key in opened:
                        can_traverse = True
                        if edge_key not in opened:
                            new_opened = opened | {edge_key}
                            use_key = True
                elif edge_type in {'soft_locked', 'one_way'}:
                    can_traverse = True  # One-way but passable
                elif edge_type in {'bomb', 'bombable'}:
                    if new_bombs > 0 or edge_key in opened:
                        can_traverse = True
                        if edge_key not in opened:
                            new_opened = opened | {edge_key}
                            use_bomb = True
                elif edge_type in {'boss', 'boss_locked'}:
                    can_traverse = bool(new_boss_key)
                elif edge_type in {'item_locked', 'item_gate'}:
                    can_traverse = bool(new_has_item)
                elif edge_type == 'switch':
                    can_traverse = True
                else:
                    # This graph-only validator does not model the resource or
                    # state needed by the remaining constraints. Fail closed.
                    can_traverse = False
                
                if can_traverse:
                    final_keys = new_keys - 1 if use_key else new_keys
                    final_keys = max(0, final_keys)
                    final_bombs = new_bombs - 1 if use_bomb else new_bombs
                    final_bombs = max(0, final_bombs)
                    
                    new_state = (
                        neighbor,
                        new_collected,
                        new_opened,
                        final_keys,
                        final_bombs,
                        new_boss_key,
                        new_has_item,
                    )
                    if new_state not in visited:
                        visited.add(new_state)
                        parents[new_state] = state
                        nodes_for_state[new_state] = neighbor
                        queue.append(new_state)
        
        # No path found
        return GraphValidationResult(
            is_solvable=False,
            graph_path=[],
            subgraph_path=[],
            missing_rooms=[],
            room_validations={},
            connectivity_score=0.0,
            start_node=start_node,
            triforce_node=triforce_node,
            error_message="No valid path considering locked doors and keys"
        )


@dataclass
class GraphValidationResult:
    """Result of graph-guided validation."""
    is_solvable: bool
    graph_path: List[int]  # Path through graph nodes
    subgraph_path: List[int] = field(default_factory=list)  # Path using only existing rooms
    missing_rooms: List[int] = field(default_factory=list)
    room_validations: Dict[int, Dict] = field(default_factory=dict)
    connectivity_score: float = 0.0
    start_node: Optional[int] = None
    triforce_node: Optional[int] = None
    error_message: str = ""
