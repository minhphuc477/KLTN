"""
Shared types for game-state search algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple


class SearchRepresentation(str, Enum):
    """Search space representation mode."""

    TILE = "tile"
    GRAPH = "graph"
    HYBRID = "hybrid"

    @classmethod
    def parse(cls, value: Any) -> "SearchRepresentation":
        raw = str(value or cls.HYBRID.value).strip().lower()
        if raw in (cls.TILE.value, cls.GRAPH.value, cls.HYBRID.value):
            return cls(raw)
        return cls.HYBRID


@dataclass
class GameStateSearchConfig:
    """Configuration for state-space search execution."""

    timeout: int = 100000
    tie_break: bool = False
    key_boost: bool = False
    enable_ara: bool = False
    ara_weight: float = 1.0
    # Zelda-style room traversal is 4-neighbor by default. Diagonal movement
    # must be enabled explicitly so all validators share the same physics.
    allow_diagonals: bool = False
    rules_profile: str = "vglc_strict"  # vglc_strict | extended | strict_original
    representation: SearchRepresentation = SearchRepresentation.HYBRID
    max_depth: int = 500
    use_iddfs: bool = True

    def to_priority_options(self) -> Dict[str, Any]:
        return {
            "tie_break": self.tie_break,
            "key_boost": self.key_boost,
            "enable_ara": self.enable_ara,
            "ara_weight": self.ara_weight,
            "allow_diagonals": self.allow_diagonals,
            "rules_profile": str(self.rules_profile or "vglc_strict"),
            # Representation is consumed by StateSpaceAStar.
            "representation": self.representation.value,
            "max_depth": int(self.max_depth),
            "use_iddfs": bool(self.use_iddfs),
        }


@dataclass
class GameStateSearchResult:
    """Normalized solver output."""

    success: bool
    path: List[Tuple[int, int]]
    states_explored: int
    algorithm: str
    metadata: Dict[str, Any] = field(default_factory=dict)
