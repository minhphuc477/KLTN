"""State-space graph solver core extracted from zelda_core."""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Dict, Set, Tuple

import networkx as nx


class StateSpaceGraphSolverCore:
    """Dependency-injected state-space solver for dungeon graph traversal."""

    def __init__(
        self,
        graph: nx.DiGraph,
        mode: str,
        validation_mode_cls: Any,
        state_cls: Any,
        parse_node_label_tokens_fn: Callable[[str], list],
        parse_edge_type_tokens_fn: Callable[..., list],
        select_primary_edge_type_fn: Callable[[list], str],
    ):
        self.graph = graph
        self.mode = mode
        self.validation_mode_cls = validation_mode_cls
        self.state_cls = state_cls
        self.parse_node_label_tokens_fn = parse_node_label_tokens_fn
        self.parse_edge_type_tokens_fn = parse_edge_type_tokens_fn
        self.select_primary_edge_type_fn = select_primary_edge_type_fn

        self.key_rooms: Set[int] = set()
        self.item_rooms: Dict[int, str] = {}

        for node_id, data in graph.nodes(data=True):
            label = data.get("label", "")
            parts = parse_node_label_tokens_fn(label)

            if "k" in parts:
                self.key_rooms.add(node_id)
            if "K" in parts:
                self.item_rooms[node_id] = "boss_key"
            elif "I" in parts:
                self.item_rooms[node_id] = "key_item"
            elif "i" in parts:
                self.item_rooms[node_id] = "minor_item"

    def can_traverse_edge(self, from_node: int, to_node: int, state: Any) -> Tuple[bool, Any, str]:
        """Check edge traversability and apply inventory/state effects."""
        edge_data = self.graph.get_edge_data(from_node, to_node)
        if not edge_data:
            return False, state, "none"

        edge_label = str(edge_data.get("label", "") or "")
        edge_type_raw = str(edge_data.get("edge_type", "") or "")
        edge_constraints = edge_data.get("edge_constraints")
        if isinstance(edge_constraints, (list, tuple, set)):
            constraints = [str(c).strip() for c in edge_constraints if str(c).strip()]
        else:
            constraints = self.parse_edge_type_tokens_fn(label=edge_label, edge_type=edge_type_raw)
        edge_type = self.select_primary_edge_type_fn(constraints)
        edge_id = (from_node, to_node)

        new_state = state.copy()

        if self.mode == self.validation_mode_cls.STRICT:
            if all(et == "open" for et in constraints):
                return True, new_state, "open"
            return False, state, edge_type

        if self.mode == self.validation_mode_cls.REALISTIC:
            allowed = {"open", "soft_locked", "stair", "switch"}
            if all(et in allowed for et in constraints):
                return True, new_state, edge_type
            return False, state, edge_type

        for et in constraints:
            if et in ("", "open", "soft_locked", "stair", "switch"):
                continue

            if et == "key_locked":
                if edge_id in new_state.doors_opened:
                    continue
                if new_state.keys_held <= 0:
                    return False, state, edge_type
                new_state.keys_held -= 1
                new_state.doors_opened.add(edge_id)
                new_state.doors_opened.add((to_node, from_node))
                continue

            if et == "bombable":
                if edge_id in new_state.doors_opened:
                    continue
                new_state.doors_opened.add(edge_id)
                new_state.doors_opened.add((to_node, from_node))
                continue

            if et == "boss_locked":
                if not new_state.items_collected or "boss_key" not in new_state.items_collected:
                    return False, state, edge_type
                continue

            if et == "item_locked":
                if "key_item" not in new_state.items_collected:
                    return False, state, edge_type
                continue

            continue

        return True, new_state, edge_type

    def collect_room_items(self, node: int, state: Any) -> Any:
        """Collect key/item pickups for entering a room node."""
        new_state = state.copy()

        if node in self.key_rooms and node not in state.keys_collected:
            new_state.keys_held += 1
            new_state.keys_collected.add(node)

        if node in self.item_rooms:
            item_type = self.item_rooms[node]
            if item_type not in state.items_collected:
                new_state.items_collected.add(item_type)

        return new_state

    def solve(self, start_node: int, goal_node: int) -> Dict:
        """Run BFS over (node, inventory_state) state-space."""
        initial_state = self.state_cls()
        initial_state = self.collect_room_items(start_node, initial_state)

        visited = {}
        queue = deque([(start_node, initial_state, [start_node], [])])
        visited[(start_node, hash(initial_state))] = True

        keys_available_max = 0

        while queue:
            current_node, current_state, path, edge_types = queue.popleft()

            keys_available_max = max(keys_available_max, current_state.keys_held + len(current_state.keys_collected))

            if current_node == goal_node:
                keys_used = len([edge for edge in edge_types if edge == "key_locked"])
                return {
                    "solvable": True,
                    "path": path,
                    "path_length": len(path) - 1,
                    "rooms_traversed": len(path),
                    "edge_types": edge_types,
                    "keys_available": current_state.keys_held,
                    "keys_collected": len(current_state.keys_collected),
                    "keys_used": keys_used,
                    "final_inventory": current_state,
                }

            for neighbor in self.graph.neighbors(current_node):
                can_traverse, new_state, edge_type = self.can_traverse_edge(current_node, neighbor, current_state)
                if not can_traverse:
                    continue

                new_state = self.collect_room_items(neighbor, new_state)

                state_key = (neighbor, hash(new_state))
                if state_key in visited:
                    continue

                visited[state_key] = True
                queue.append((neighbor, new_state, path + [neighbor], edge_types + [edge_type]))

        return {
            "solvable": False,
            "reason": f"No path from {start_node} to {goal_node} with current inventory constraints",
            "mode": self.mode,
            "keys_found": keys_available_max,
        }
