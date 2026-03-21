"""Helpers to assemble solver input payloads from GUI state."""

from typing import Any, Dict, Optional


def get_solver_map_context(current_map: Any) -> Dict[str, Any]:
    """Extract grid and optional topology connectivity references from a map object."""
    if hasattr(current_map, "global_grid"):
        return {
            "grid_arr": current_map.global_grid,
            "graph": getattr(current_map, "graph", None),
            "room_to_node": getattr(current_map, "room_to_node", None),
            "room_positions": getattr(current_map, "room_positions", None),
            "node_to_room": getattr(current_map, "node_to_room", None),
        }
    return {
        "grid_arr": current_map,
        "graph": None,
        "room_to_node": None,
        "room_positions": None,
        "node_to_room": None,
    }


def build_priority_options(
    feature_flags: Dict[str, Any],
    ara_weight: float,
    search_representation: str,
) -> Dict[str, Any]:
    """Build solver priority/search options from feature flags and UI values."""
    return {
        "tie_break": feature_flags.get("priority_tie_break", False),
        "key_boost": feature_flags.get("priority_key_boost", False),
        "enable_ara": feature_flags.get("enable_ara", False),
        "ara_weight": float(ara_weight),
        "representation": str(search_representation),
        "allow_diagonals": True,
    }


def build_solver_request(
    current_map: Any,
    env: Any,
    feature_flags: Dict[str, Any],
    algorithm_idx: int,
    ara_weight: float,
    search_representation: str,
) -> Optional[Dict[str, Any]]:
    """Build canonical solver request payload, or None when start/goal are missing."""
    if env is None or not getattr(env, "start_pos", None) or not getattr(env, "goal_pos", None):
        return None

    ctx = get_solver_map_context(current_map)
    return {
        **ctx,
        "start": tuple(env.start_pos),
        "goal": tuple(env.goal_pos),
        "alg_idx": int(algorithm_idx),
        "flags": dict(feature_flags),
        "priority_options": build_priority_options(
            feature_flags=feature_flags,
            ara_weight=ara_weight,
            search_representation=search_representation,
        ),
    }
