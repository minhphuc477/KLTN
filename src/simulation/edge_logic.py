"""Edge-type and traversal helpers extracted from validator monolith."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from src.core.definitions import parse_edge_type_tokens, select_primary_edge_type


def edge_constraints_from_data(edge_data: Optional[Dict[str, Any]]) -> List[str]:
    """Return canonical edge constraints from edge attributes."""
    if not edge_data:
        return ["open"]
    return parse_edge_type_tokens(
        label=edge_data.get("label", ""),
        edge_type=edge_data.get("edge_type", edge_data.get("type", "")),
    )


def edge_type_from_data(edge_data: Optional[Dict[str, Any]]) -> str:
    """Return primary canonical edge type from edge attributes."""
    primary = select_primary_edge_type(edge_constraints_from_data(edge_data))
    if (
        primary == "hazard"
        and edge_data
        and edge_data.get("protection_item_id") is not None
        and str(edge_data.get("protection_item_id")).strip()
    ):
        # The tile-state validator has one generic permanent traversal item.
        # Named identity is checked by the abstract graph oracle before this
        # lossy graph-to-grid boundary.
        return "hazard_protected"
    traversal_aliases = {
        "item_gate": "item_locked",
        "switch_locked": "switch",
        "state_block": "switch",
        "on_off_gate": "switch",
        "shutter": "switch",
        "puzzle": "switch",
        "one_way": "soft_locked",
        "hazard": "hazard",
    }
    return traversal_aliases.get(primary, primary)


def combine_edge_types(type1: str, type2: str) -> str:
    """Combine two edge types and keep the more restrictive one."""
    priority = {
        "boss": 5,
        "bomb": 4,
        "locked": 3,
        "key_locked": 3,
        "puzzle": 2,
        "hazard_protected": 2,
        "hazard": 1,
        "open": 1,
        "": 1,
    }
    p1 = priority.get(type1, 1)
    p2 = priority.get(type2, 1)
    return type1 if p1 >= p2 else type2


def can_traverse_edge_type(
    edge_type: str,
    state: Any,
    *,
    strict_original_mode: bool,
    get_room_for_position: Callable[[Any], Any],
    is_room_cleared: Callable[[Any, Any], bool],
) -> bool:
    """Check whether current state can traverse a graph edge type."""
    normalized = str(edge_type or "").strip().lower()

    if normalized in ("open", "", "path", "stair"):
        return True
    if normalized in ("hazard", "hazard_protected"):
        return state.has_item
    if normalized in ("locked", "key_locked"):
        return state.keys > 0
    if normalized in ("bomb", "bombable"):
        return state.bomb_count > 0
    if normalized in ("boss", "boss_locked"):
        return state.has_boss_key
    if normalized in ("item_locked", "item_gate"):
        return state.has_item
    if normalized in ("soft_locked", "one_way"):
        if strict_original_mode:
            current_room = get_room_for_position(state.position)
            return is_room_cleared(current_room, state)
        return True
    if normalized in ("puzzle", "switch", "switch_locked", "state_block", "on_off_gate", "shutter"):
        if strict_original_mode:
            current_room = get_room_for_position(state.position)
            return is_room_cleared(current_room, state)
        return True
    return False
