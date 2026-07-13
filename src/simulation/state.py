"""Canonical game state and movement semantics for Zelda search algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, FrozenSet, Mapping, Set, Tuple

from src.core.definitions import SEMANTIC_PALETTE


WALKABLE_IDS = {
    SEMANTIC_PALETTE["FLOOR"],
    SEMANTIC_PALETTE["DOOR_OPEN"],
    SEMANTIC_PALETTE["DOOR_SOFT"],
    SEMANTIC_PALETTE["START"],
    SEMANTIC_PALETTE["TRIFORCE"],
    SEMANTIC_PALETTE["KEY_SMALL"],
    SEMANTIC_PALETTE["KEY_BOSS"],
    SEMANTIC_PALETTE["KEY_ITEM"],
    SEMANTIC_PALETTE["ITEM_MINOR"],
    SEMANTIC_PALETTE["ELEMENT_FLOOR"],
    SEMANTIC_PALETTE["STAIR"],
    SEMANTIC_PALETTE["ENEMY"],
    SEMANTIC_PALETTE["BOSS"],
    SEMANTIC_PALETTE["PUZZLE"],
}

BLOCKING_IDS = {
    SEMANTIC_PALETTE["VOID"],
    SEMANTIC_PALETTE["WALL"],
}

TRANSITION_IDS = {
    SEMANTIC_PALETTE["STAIR"],
    SEMANTIC_PALETTE["DOOR_OPEN"],
    SEMANTIC_PALETTE["DOOR_SOFT"],
}

CONDITIONAL_IDS = {
    SEMANTIC_PALETTE["DOOR_LOCKED"],
    SEMANTIC_PALETTE["DOOR_BOMB"],
    SEMANTIC_PALETTE["DOOR_BOSS"],
    SEMANTIC_PALETTE["DOOR_PUZZLE"],
}

PUSHABLE_IDS = {SEMANTIC_PALETTE["BLOCK"]}
WATER_IDS = {SEMANTIC_PALETTE["ELEMENT"]}
BRIDGE_FILL_IDS = {SEMANTIC_PALETTE["ELEMENT"]}
PICKUP_IDS = {
    SEMANTIC_PALETTE["KEY_SMALL"],
    SEMANTIC_PALETTE["KEY_BOSS"],
    SEMANTIC_PALETTE["KEY_ITEM"],
    SEMANTIC_PALETTE["ITEM_MINOR"],
}

EDGE_TYPE_MAP = {
    "locked": "key_locked",
    "k": "key_locked",
    "key_locked": "key_locked",
    "bomb": "bombable",
    "b": "bombable",
    "bombable": "bombable",
    "boss": "boss_locked",
    "K": "boss_locked",
    "boss_locked": "boss_locked",
    "puzzle": "switch",
    "S": "switch",
    "S1": "switch",
    "switch": "switch",
    "I": "item_locked",
    "item_locked": "item_locked",
    "l": "soft_locked",
    "soft_locked": "soft_locked",
    "s": "stair",
    "stair": "stair",
    "open": "open",
    "": "open",
}


def graph_node_role_tokens(node_data: Mapping[str, Any]) -> Set[str]:
    """Normalize node-role hints from heterogeneous graph schemas."""
    tokens: Set[str] = set()
    for key in ("type", "label", "node_type", "stage"):
        raw = str(node_data.get(key, "") or "").strip().lower()
        if raw:
            tokens.add(raw)
    contents = node_data.get("contents", ())
    if isinstance(contents, (list, tuple, set, frozenset)):
        for item in contents:
            raw = str(item or "").strip().lower()
            if raw:
                tokens.add(raw)
    return tokens


def is_graph_start_node(node_data: Mapping[str, Any]) -> bool:
    return bool(node_data.get("is_start", False)) or "start" in graph_node_role_tokens(node_data)


def is_graph_goal_node(node_data: Mapping[str, Any]) -> bool:
    if bool(node_data.get("has_triforce", False)) or bool(node_data.get("has_goal", False)):
        return True
    return bool(graph_node_role_tokens(node_data) & {"goal", "triforce"})


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    UP_LEFT = 4
    UP_RIGHT = 5
    DOWN_LEFT = 6
    DOWN_RIGHT = 7


ACTION_DELTAS = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
    Action.UP_LEFT: (-1, -1),
    Action.UP_RIGHT: (-1, 1),
    Action.DOWN_LEFT: (1, -1),
    Action.DOWN_RIGHT: (1, 1),
}

CARDINAL_COST = 1.0
DIAGONAL_COST = 1.0


@dataclass
class GameState:
    """Complete inventory, world-geometry, and progression state."""

    position: Tuple[int, int]
    keys: int = 0
    bomb_count: int = 0
    has_boss_key: bool = False
    has_item: bool = False
    item_names: Set[str] = field(default_factory=set)
    opened_doors: Set[Tuple[int, int]] = field(default_factory=set)
    collected_items: Set[Tuple[int, int]] = field(default_factory=set)
    pushed_blocks: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = field(default_factory=set)
    filled_block_origins: Set[Tuple[int, int]] = field(default_factory=set)
    bridged_tiles: Set[Tuple[int, int]] = field(default_factory=set)
    defeated_enemies: Set[Tuple[int, int]] = field(default_factory=set)
    completed_puzzle_stages: Set[Tuple[str, int]] = field(default_factory=set)
    current_floor: int = 0
    opened_graph_edges: Set[Tuple[Any, Any]] = field(default_factory=set)

    @property
    def has_bomb(self) -> bool:
        return self.bomb_count > 0

    @has_bomb.setter
    def has_bomb(self, value: bool) -> None:
        if value and self.bomb_count <= 0:
            self.bomb_count = 1
        elif not value:
            self.bomb_count = 0

    def __hash__(self) -> int:
        return hash(game_state_key(self))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GameState) and game_state_key(self) == game_state_key(other)

    def copy(self) -> "GameState":
        return GameState(
            position=self.position,
            keys=self.keys,
            bomb_count=self.bomb_count,
            has_boss_key=self.has_boss_key,
            has_item=self.has_item,
            item_names=set(self.item_names),
            opened_doors=set(self.opened_doors),
            collected_items=set(self.collected_items),
            pushed_blocks=set(self.pushed_blocks),
            filled_block_origins=set(self.filled_block_origins),
            bridged_tiles=set(self.bridged_tiles),
            defeated_enemies=set(self.defeated_enemies),
            completed_puzzle_stages=set(self.completed_puzzle_stages),
            current_floor=self.current_floor,
            opened_graph_edges=set(self.opened_graph_edges),
        )


def game_state_key(state: GameState) -> Tuple[Any, ...]:
    """Immutable value key for search maps; safe under Python hash collisions."""
    return (
        state.position,
        state.keys,
        state.bomb_count,
        state.has_boss_key,
        state.has_item,
        frozenset(str(name).upper() for name in state.item_names),
        frozenset(state.opened_doors),
        frozenset(state.collected_items),
        frozenset(state.pushed_blocks),
        frozenset(state.filled_block_origins),
        frozenset(state.bridged_tiles),
        frozenset(state.defeated_enemies),
        frozenset(state.completed_puzzle_stages),
        state.current_floor,
        frozenset(state.opened_graph_edges),
    )


def has_pushed_block_at(state: GameState, pos: Tuple[int, int]) -> bool:
    return any(to_pos == pos for _, to_pos in state.pushed_blocks)


def was_block_vacated(state: GameState, pos: Tuple[int, int]) -> bool:
    return tuple(pos) in state.filled_block_origins or any(
        from_pos == pos for from_pos, _ in state.pushed_blocks
    )


def dynamic_geometry_key(state: GameState) -> Tuple[FrozenSet, FrozenSet, FrozenSet]:
    return (
        frozenset(state.pushed_blocks),
        frozenset(state.filled_block_origins),
        frozenset(state.bridged_tiles),
    )


def is_push_destination_available(state: GameState, pos: Tuple[int, int], static_tile: int) -> bool:
    return (
        not has_pushed_block_at(state, pos)
        and (
            int(static_tile) in WALKABLE_IDS
            or int(static_tile) in BRIDGE_FILL_IDS
            or was_block_vacated(state, pos)
            or tuple(pos) in state.bridged_tiles
        )
    )


def dominates(state_a: GameState, state_b: GameState) -> bool:
    """Return whether state A can safely replace state B in a Pareto frontier."""
    if state_a.position != state_b.position:
        return False
    if state_a.keys < state_b.keys or state_a.bomb_count < state_b.bomb_count:
        return False
    if not state_a.has_boss_key and state_b.has_boss_key:
        return False
    if not state_a.has_item and state_b.has_item:
        return False
    if not {
        str(name).upper() for name in state_a.item_names
    }.issuperset(str(name).upper() for name in state_b.item_names):
        return False
    if not state_a.opened_doors.issuperset(state_b.opened_doors):
        return False
    if not state_a.opened_graph_edges.issuperset(state_b.opened_graph_edges):
        return False
    # Collected consumables must match: leaving a pickup for later can be useful.
    if state_a.collected_items != state_b.collected_items:
        return False
    if not state_a.defeated_enemies.issuperset(state_b.defeated_enemies):
        return False
    if not state_a.completed_puzzle_stages.issuperset(state_b.completed_puzzle_stages):
        return False
    # Dynamic geometry is comparable only when its complete history matches.
    if state_a.pushed_blocks != state_b.pushed_blocks:
        return False
    if state_a.filled_block_origins != state_b.filled_block_origins:
        return False
    if state_a.bridged_tiles != state_b.bridged_tiles:
        return False
    return True


__all__ = [
    "ACTION_DELTAS",
    "BLOCKING_IDS",
    "BRIDGE_FILL_IDS",
    "CARDINAL_COST",
    "CONDITIONAL_IDS",
    "DIAGONAL_COST",
    "EDGE_TYPE_MAP",
    "PICKUP_IDS",
    "PUSHABLE_IDS",
    "TRANSITION_IDS",
    "WALKABLE_IDS",
    "WATER_IDS",
    "Action",
    "GameState",
    "dominates",
    "dynamic_geometry_key",
    "game_state_key",
    "graph_node_role_tokens",
    "has_pushed_block_at",
    "is_graph_goal_node",
    "is_graph_start_node",
    "is_push_destination_available",
    "was_block_vacated",
]
