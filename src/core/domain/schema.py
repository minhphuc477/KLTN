"""Generic domain-schema contract for symbolic graph semantics.

The neural pipeline consumes domain-independent tensors. Domain-specific
interpretation of graph metadata, such as whether a node is a boss or goal
room, belongs behind this interface so ablations and cross-domain experiments
can swap schemas without touching model code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping

ROOM_ROLE_KEYS = (
    "is_start",
    "has_enemy",
    "has_key",
    "has_item",
    "has_goal",
    "has_boss",
    "has_puzzle",
    "is_tutorial_puzzle",
    "is_combat_puzzle",
    "is_complex_puzzle",
    "is_switch_puzzle",
)


class NeuroSymbolicSchema(ABC):
    """Interface for converting domain graph metadata into pipeline semantics."""

    name: str = "generic"

    @abstractmethod
    def room_role_flags(self, attrs: Mapping[str, Any]) -> Dict[str, bool]:
        """Return canonical room-role booleans for one graph node."""

    def normalize_room_role_flags(self, attrs: Mapping[str, Any]) -> Dict[str, bool]:
        """Return all canonical keys with boolean values and no silent extras."""
        raw = dict(self.room_role_flags(attrs))
        return {key: bool(raw.get(key, False)) for key in ROOM_ROLE_KEYS}


def resolve_domain_schema(schema: Any = None) -> NeuroSymbolicSchema:
    """Resolve constructor input into a concrete schema instance."""
    from src.core.domain.zelda_schema import ZeldaSchema

    if schema is None:
        return ZeldaSchema()
    if isinstance(schema, NeuroSymbolicSchema):
        return schema
    if hasattr(schema, "room_role_flags"):
        return schema  # duck-typed external schema for experiments.
    if isinstance(schema, str):
        key = schema.strip().lower().replace("-", "_")
        if key in {"zelda", "zelda_v1", "default"}:
            return ZeldaSchema()
    if isinstance(schema, Mapping):
        key = str(schema.get("name", schema.get("domain", "zelda"))).strip().lower().replace("-", "_")
        if key in {"zelda", "zelda_v1", "default"}:
            return ZeldaSchema()
    raise ValueError(
        "domain_schema must be None, 'zelda', a NeuroSymbolicSchema, or an object with room_role_flags(attrs)."
    )
