"""Domain schema interfaces for neuro-symbolic graph semantics."""

from src.core.domain.schema import NeuroSymbolicSchema, ROOM_ROLE_KEYS, resolve_domain_schema
from src.core.domain.zelda_schema import ZeldaSchema

__all__ = [
    "NeuroSymbolicSchema",
    "ROOM_ROLE_KEYS",
    "ZeldaSchema",
    "resolve_domain_schema",
]
