"""Inventory transitions shared by physical passages and graph transport."""

from __future__ import annotations

from typing import Any, Mapping

from src.core.definitions import SEMANTIC_PALETTE, parse_node_label_tokens
from src.simulation.edge_logic import edge_constraints_from_data, graph_edge_state_key
from src.simulation.state import GameState


def switch_provider_ids(node: Any, data: Mapping[str, Any]) -> set[Any]:
    """Use explicit provider roles, never a room number or an arbitrary puzzle."""
    tokens = {str(value).lower() for value in parse_node_label_tokens(str(data.get("label", "")))}
    kind = str(data.get("type", data.get("node_type", ""))).lower()
    if not (
        tokens & {"switch", "puzzle", "p"}
        or kind in {"switch", "puzzle", "tutorial_puzzle", "combat_puzzle", "complex_puzzle"}
        or data.get("has_puzzle", False)
        or data.get("has_switch", False)
    ):
        return set()
    ids = {node}
    if data.get("switch_id") is not None:
        ids.add(data["switch_id"])
    return ids


def key_provider_ids(node: Any, data: Mapping[str, Any], *, boss: bool = False) -> set[Any]:
    """Extract explicit small-key or boss-key provider identities from a graph node."""
    del node
    kind = str(data.get("type", data.get("node_type", "")) or "").strip().lower()
    raw_parts = {
        part.strip()
        for field in ("label", "symbol")
        for part in str(data.get(field, "") or "").replace("\n", ",").split(",")
        if part.strip()
    }
    parsed_tokens = (
        set(parse_node_label_tokens(str(data.get("label", "") or "")))
        | set(parse_node_label_tokens(str(data.get("symbol", "") or "")))
    )
    raw_tokens = raw_parts | parsed_tokens
    tokens = {str(value).strip().lower() for value in raw_tokens if str(value).strip()}
    tokens.update(
        part.strip().lower()
        for field in ("type", "node_type")
        for part in str(data.get(field, "") or "").split(",")
        if part.strip()
    )
    if boss:
        is_provider = bool(
            kind in {"big_key", "boss_key"}
            or "big_key" in tokens
            or "boss_key" in tokens
            or "K" in raw_tokens
            or bool(data.get("has_boss_key", False))
        )
    else:
        is_provider = bool(
            kind in {"key", "small_key"}
            or "key" in tokens
            or "small_key" in tokens
            or "k" in raw_tokens
            or (
                bool(data.get("has_key", False))
                and kind not in {"big_key", "boss_key"}
                and not bool(data.get("has_boss_key", False))
            )
        )
    if not is_provider:
        return set()
    ids: set[Any] = set()
    for field in ("key_id", "item_id"):
        value = data.get(field)
        if value is not None:
            ids.add(value)
    return ids


def _state_has_collected_key_id(
    env,
    state: GameState,
    key_required: Any,
    *,
    boss: bool,
    required_count: int = 1,
    current_edge_key: Any = None,
) -> bool:
    """Verify named key identity against collected room-mapped pickup tiles."""
    graph = getattr(env, "graph", None)
    room_to_node = getattr(env, "room_to_node", None)
    original_grid = getattr(env, "original_grid", None)
    get_room = getattr(env, "get_room_for_position", None) or getattr(env, "_get_room_for_position", None)
    collected_items = getattr(state, "collected_items", None)
    if (
        graph is None
        or not isinstance(room_to_node, Mapping)
        or original_grid is None
        or not callable(get_room)
        or not collected_items
    ):
        return False

    target_tile_id = int(SEMANTIC_PALETTE["KEY_BOSS"] if boss else SEMANTIC_PALETTE["KEY_SMALL"])
    height, width = original_grid.shape[:2]
    matched_pickups = 0
    for pos in collected_items:
        r, c = int(pos[0]), int(pos[1])
        if r < 0 or r >= height or c < 0 or c >= width:
            continue
        if int(original_grid[r, c]) != target_tile_id:
            continue
        room_pos = get_room((r, c))
        node_id = room_to_node.get(room_pos)
        if node_id is None or node_id not in graph.nodes:
            continue
        if key_required in key_provider_ids(node_id, graph.nodes[node_id], boss=boss):
            matched_pickups += 1

    if boss:
        return matched_pickups >= 1

    consumed_named = 0
    for opened_edge in getattr(state, "opened_graph_edges", ()):
        if opened_edge == current_edge_key:
            continue
        edata = None
        if isinstance(opened_edge, tuple) and len(opened_edge) == 2:
            u, v = opened_edge
            edata = graph.get_edge_data(u, v)
            if edata is None and bool(graph.is_directed()):
                edata = graph.get_edge_data(v, u)
        elif isinstance(opened_edge, tuple) and len(opened_edge) == 3:
            u, v, k = opened_edge
            edata = graph.get_edge_data(u, v, k)
            if edata is None and bool(graph.is_directed()):
                edata = graph.get_edge_data(v, u, k)
        if not isinstance(edata, Mapping):
            continue
        if "key_locked" in edge_constraints_from_data(edata) and edata.get("key_required") == key_required:
            cnt = edata.get("requires_key_count", edata.get("key_count", 1))
            if cnt is None or (not isinstance(cnt, bool) and cnt == 0):
                cnt = 1
            consumed_named += max(1, int(cnt))

    return (matched_pickups - consumed_named) >= max(1, int(required_count))


def required_switch_ids(graph):
    """Which controller identities affect a gate; None means any controller."""
    required = set()
    if graph is None:
        return required
    for _, _, data in graph.edges(data=True):
        if not set(edge_constraints_from_data(data)) & {'switch', 'switch_locked', 'puzzle'}:
            continue
        ids = set(data.get('switches_required', ()) or ()) - {None}
        if not ids and data.get('switch_id') is not None:
            ids.add(data['switch_id'])
        if not ids:
            return None
        required.update(ids)
    return required


def starting_bomb_count(graph, default: int) -> int:
    """Read the explicit finite inventory, refusing silently ignored fields."""
    inventory = graph.graph.get('initial_inventory') if graph is not None else None
    if inventory is None:
        return default
    if not isinstance(inventory, dict) or set(inventory) != {'bomb_count'}:
        raise ValueError('Initial inventory currently requires exactly bomb_count')
    count = inventory['bomb_count']
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError('Initial bomb_count must be a nonnegative integer')
    return count


def apply_graph_gate(env, state: GameState, source: Any, target: Any, *, edge_data_override=None) -> tuple[bool, GameState]:
    """Enforce every requirement atomically and charge a passage only once.

    A failed composite gate cannot consume resources. Named permanent
    requirements are checked even after a consumable part has been opened.
    Parallel edge alternatives require explicit identity and fail closed here.
    """
    graph = env.graph
    if graph is None or not graph.has_edge(source, target) or graph.is_multigraph():
        return False, state
    data = dict(graph.get_edge_data(source, target) or {}) if edge_data_override is None else dict(edge_data_override)
    constraints = edge_constraints_from_data(data)
    key_required = data.get("key_required")
    if key_required is not None and "boss_locked" not in constraints and "key_locked" not in constraints:
        return False, state
    edge_key = graph_edge_state_key(graph, source, target)
    opened = edge_key in state.opened_graph_edges
    candidate = state.copy()

    def has_item(raw):
        name = str(raw or "").strip().upper()
        inventory = {str(value).upper() for value in candidate.item_names}
        return candidate.has_item and (not name or "*" in inventory or name in inventory)

    consumable_gate = False
    for kind in constraints:
        kind = str(kind).strip().lower()
        if kind in {"open", "path", "shortcut", "stair", "one_way"}:
            continue
        if kind == "soft_locked":
            room = (env.node_to_room or {}).get(source)
            if room is None or not env._is_room_cleared(room, candidate):
                return False, state
        elif kind in {"switch", "switch_locked", "puzzle"}:
            required = set(data.get("switches_required", ()) or ()) - {None}
            if not required and data.get("switch_id") is not None:
                required.add(data["switch_id"])
            if not (required.issubset(candidate.switches_activated) if required else candidate.switches_activated):
                return False, state
        elif kind in {"item_locked", "item_gate"}:
            if not has_item(data.get("item_required")):
                return False, state
        elif kind in {"hazard", "hazard_protected"}:
            if data.get("protection_item_id") and not has_item(data["protection_item_id"]):
                return False, state
        elif kind == "boss_locked":
            if not candidate.has_boss_key:
                return False, state
            if key_required is not None and not _state_has_collected_key_id(
                env, candidate, key_required, boss=True
            ):
                return False, state
            consumable_gate = True
        elif kind == "key_locked":
            count = data.get("requires_key_count", 1)
            if count is None or (not isinstance(count, bool) and count == 0):
                count = 1
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                return False, state
            if not opened:
                if key_required is not None and not _state_has_collected_key_id(
                    env,
                    candidate,
                    key_required,
                    boss=False,
                    required_count=count,
                    current_edge_key=edge_key,
                ):
                    return False, state
                if candidate.keys < count:
                    return False, state
                candidate.keys -= count
            consumable_gate = True
        elif kind == "bombable":
            if not opened:
                if candidate.bomb_count < 1:
                    return False, state
                candidate.bomb_count -= 1
            consumable_gate = True
        else:
            return False, state
    if consumable_gate:
        candidate.opened_graph_edges.add(edge_key)
    return True, candidate


class PhysicalGraphBridge:
    """Bind recorded passage cells to directed mission-edge state transitions.

    Passage traversal stays attached to its source until the player reaches
    either endpoint's interior. Shared cells between different passages are
    ambiguous and are rejected, rather than permitting a corridor shortcut.
    """

    def __init__(self, env, records):
        self.env = env
        self.cells: dict[tuple[int, int], set[frozenset]] = {}
        self.portals: dict[tuple[tuple[int, int], tuple[int, int]], tuple[Any, Any]] = {}
        if isinstance(records, (list, tuple)):
            items = [((entry['source'], entry['target']), entry.get('record', entry)) for entry in records]
        else:
            items = records.items() if hasattr(records, 'items') else []
        for (source, target), record in items:
            pair = frozenset((source, target))
            if (len(pair) != 2 and record.get('mode') != 'graph_transition') or not env.graph.has_edge(source, target):
                raise ValueError("Passage record must name an existing non-self graph edge")
            if record.get('mode') == 'graph_transition':
                entry, exit_pos = tuple(record['source_position']), tuple(record['target_position'])
                if (self.node_at(entry) != source or self.node_at(exit_pos) != target
                        or record.get('transition_cost') != 1
                        or any(int(env.original_grid[pos]) != SEMANTIC_PALETTE['STAIR'] for pos in (entry, exit_pos))):
                    raise ValueError('Graph transition needs intact stair endpoints in its declared rooms and unit cost')
                self.portals[(entry, exit_pos)] = (source, target)
                if not env.graph.is_directed():
                    self.portals[(exit_pos, entry)] = (target, source)
                continue
            for row, col in record.get("cells", ()):
                self.cells.setdefault((int(row), int(col)), set()).add(pair)

    def node_at(self, pos):
        room = self.env.get_room_for_position(pos)
        return (self.env.room_to_node or {}).get(room)

    def transition(self, state, target, *, copy_on_noop: bool = True):
        source_pos = state.position
        edge = self.portals.get((source_pos, target))
        if edge is not None:
            if state.active_graph_edge is not None:
                return False, state, False
            allowed, candidate = apply_graph_gate(self.env, state, *edge)
            return allowed, candidate, False
        if sum(abs(a - b) for a, b in zip(source_pos, target)) != 1:
            return False, state, False
        memberships = self.cells.get(target, set())
        if len(memberships) > 1:
            return False, state, False
        current_node = self.node_at(source_pos)
        target_node = self.node_at(target)
        active = state.active_graph_edge
        if active is not None:
            pair = frozenset(active)
            if memberships:
                if pair not in memberships:
                    return False, state, False
                if current_node == active[1] and target_node != current_node:
                    allowed, candidate = apply_graph_gate(self.env, state, active[1], active[0])
                    if not allowed:
                        return False, state, False
                    candidate.active_graph_edge = (active[1], active[0])
                    return True, candidate, True
                return True, (state.copy() if copy_on_noop else state), True
            if target_node not in pair:
                return False, state, False
            candidate = state.copy()
            candidate.active_graph_edge = None
            return True, candidate, False
        if memberships:
            pair = next(iter(memberships))
            if current_node not in pair:
                return False, state, False
            other = next(node for node in pair if node != current_node)
            allowed, candidate = apply_graph_gate(self.env, state, current_node, other)
            if not allowed:
                return False, state, False
            candidate.active_graph_edge = (current_node, other)
            return True, candidate, True
        # Without a recorded passage, an accidental opening is never an edge.
        if current_node != target_node or current_node is None:
            return False, state, False
        return True, (state.copy() if copy_on_noop else state), False

    def destinations(self, position):
        return [target for source, target in self.portals if source == position]
