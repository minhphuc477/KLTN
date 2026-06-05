"""Zelda-specific symbolic graph schema."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Set

import numpy as np

from src.core.definitions import parse_node_label_tokens
from src.core.domain.schema import NeuroSymbolicSchema, ROOM_ROLE_KEYS


def _parse_label_tokens(label: Any) -> Set[str]:
    if label is None:
        return set()
    return {str(token).strip().lower() for token in parse_node_label_tokens(str(label)) if str(token).strip()}


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value) != 0
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"true", "1", "yes", "y", "on"}:
            return True
        if key in {"false", "0", "no", "n", "off", ""}:
            return False
    return bool(value)


class ZeldaSchema(NeuroSymbolicSchema):
    """Room-role extraction rules for Zelda-style mission graphs."""

    name = "zelda"

    def room_role_flags(self, attrs: Mapping[str, Any]) -> Dict[str, bool]:
        values = dict(attrs or {})
        tokens = _parse_label_tokens(values.get("label"))
        raw_type = str(values.get("type", values.get("node_type", values.get("room_type", ""))) or "").strip().lower()
        role_tokens = set(tokens) | _parse_label_tokens(raw_type)
        difficulty_rating = str(values.get("difficulty_rating", "") or "").strip().upper()

        def hint(name: str, *aliases: str) -> bool:
            return _coerce_bool(values.get(name)) or any(_coerce_bool(values.get(alias)) for alias in aliases)

        flags = {
            "is_start": hint("is_start", "is_entry")
            or raw_type in {"start", "entry"}
            or "start" in role_tokens
            or "entry" in role_tokens,
            "has_enemy": hint("has_enemy") or "e" in role_tokens or "enemy" in role_tokens,
            "has_key": hint("has_key") or "k" in role_tokens or "key" in role_tokens,
            "has_item": hint("has_item", "has_macro_item", "has_minor_item")
            or "i" in role_tokens
            or "item" in role_tokens
            or "treasure" in role_tokens,
            "has_goal": hint("has_triforce", "is_triforce", "is_goal")
            or raw_type in {"goal", "triforce"}
            or "t" in role_tokens
            or "goal" in role_tokens
            or "triforce" in role_tokens,
            "has_boss": hint("has_boss", "is_boss") or "b" in role_tokens or "boss" in role_tokens,
            "has_puzzle": hint("has_puzzle")
            or "p" in role_tokens
            or "puzzle" in role_tokens
            or raw_type in {"switch", "puzzle", "tutorial_puzzle", "combat_puzzle", "complex_puzzle"}
            or "puzzle" in raw_type,
            "is_tutorial_puzzle": bool(hint("is_tutorial") or raw_type == "tutorial_puzzle" or difficulty_rating == "SAFE"),
            "is_combat_puzzle": bool(raw_type == "combat_puzzle"),
            "is_complex_puzzle": bool(raw_type == "complex_puzzle" or difficulty_rating in {"HARD", "EXTREME"}),
            "is_switch_puzzle": bool(raw_type == "switch"),
        }
        return {key: bool(flags.get(key, False)) for key in ROOM_ROLE_KEYS}
