"""
Shared types and tile/state models for causal WFC generation.

This module exists to keep `wfc_refiner.py` focused on algorithmic flow,
while preserving backward-compatible class names and behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple
import random

# Import semantic palette if available
try:
    from src.core.definitions import SEMANTIC_PALETTE
except ImportError:
    SEMANTIC_PALETTE = {
        'VOID': 0,
        'FLOOR': 1,
        'WALL': 2,
        'BLOCK': 3,
        'DOOR_OPEN': 10,
        'DOOR_LOCKED': 11,
        'ENEMY': 20,
        'START': 21,
        'TRIFORCE': 22,
        'KEY_SMALL': 30,
        'KEY_BIG': 31,
    }


class TileType(Enum):
    """Tile types with game state implications."""
    FLOOR = auto()
    WALL = auto()
    BLOCK = auto()
    DOOR_OPEN = auto()
    DOOR_LOCKED = auto()
    KEY_SMALL = auto()
    KEY_BIG = auto()
    ENEMY = auto()
    START = auto()
    GOAL = auto()
    ITEM = auto()
    WATER = auto()
    BRIDGE = auto()


@dataclass
class TileConstraint:
    """Constraints on when a tile can be placed."""
    required_keys: int = 0
    required_items: Set[str] = field(default_factory=set)
    provides_key: bool = False
    provides_item: Optional[str] = None
    is_blocking: bool = False
    key_id: Optional[int] = None


@dataclass
class Tile:
    """Tile with adjacency rules and game constraints."""
    id: int
    tile_type: TileType
    semantic_id: int
    adjacency: Dict[str, Set[int]] = field(default_factory=dict)
    weight: float = 1.0
    constraint: TileConstraint = field(default_factory=TileConstraint)


class TileSet:
    """Collection of tiles with adjacency rules."""

    def __init__(self):
        self.tiles: Dict[int, Tile] = {}
        self._build_tiles()

    def _build_tiles(self):
        """Build tile definitions. Override in subclass."""
        raise NotImplementedError("TileSet subclasses must implement _build_tiles")

    def get_tile(self, tile_id: int) -> Optional[Tile]:
        return self.tiles.get(tile_id)

    def get_all_tile_ids(self) -> Set[int]:
        return set(self.tiles.keys())

    def get_tiles_by_type(self, tile_type: TileType) -> List[Tile]:
        return [t for t in self.tiles.values() if t.tile_type == tile_type]


class ZeldaTileSet(TileSet):
    """Zelda-specific tile set with adjacency rules."""

    def _build_tiles(self):
        self.tiles = {
            0: Tile(
                id=0,
                tile_type=TileType.FLOOR,
                semantic_id=SEMANTIC_PALETTE.get('FLOOR', 1),
                weight=10.0,
            ),
            1: Tile(
                id=1,
                tile_type=TileType.WALL,
                semantic_id=SEMANTIC_PALETTE.get('WALL', 2),
                weight=3.0,
            ),
            2: Tile(
                id=2,
                tile_type=TileType.BLOCK,
                semantic_id=SEMANTIC_PALETTE.get('BLOCK', 3),
                weight=1.0,
            ),
            3: Tile(
                id=3,
                tile_type=TileType.DOOR_OPEN,
                semantic_id=SEMANTIC_PALETTE.get('DOOR_OPEN', 10),
                weight=0.5,
            ),
            4: Tile(
                id=4,
                tile_type=TileType.DOOR_LOCKED,
                semantic_id=SEMANTIC_PALETTE.get('DOOR_LOCKED', 11),
                weight=0.3,
                constraint=TileConstraint(required_keys=1, is_blocking=True),
            ),
            5: Tile(
                id=5,
                tile_type=TileType.KEY_SMALL,
                semantic_id=SEMANTIC_PALETTE.get('KEY_SMALL', 30),
                weight=0.5,
                constraint=TileConstraint(provides_key=True),
            ),
            6: Tile(
                id=6,
                tile_type=TileType.ENEMY,
                semantic_id=SEMANTIC_PALETTE.get('ENEMY', 20),
                weight=1.5,
            ),
            7: Tile(
                id=7,
                tile_type=TileType.START,
                semantic_id=SEMANTIC_PALETTE.get('START', 21),
                weight=0.0,
            ),
            8: Tile(
                id=8,
                tile_type=TileType.GOAL,
                semantic_id=SEMANTIC_PALETTE.get('TRIFORCE', 22),
                weight=0.0,
            ),
        }
        self._build_adjacency_rules()

    def _build_adjacency_rules(self):
        floor_ids = {0, 3, 4, 5, 6, 7, 8}
        wall_ids = {1, 2}

        for tile in self.tiles.values():
            directions = ['N', 'S', 'E', 'W']

            if tile.tile_type in [
                TileType.FLOOR,
                TileType.DOOR_OPEN,
                TileType.KEY_SMALL,
                TileType.ENEMY,
                TileType.START,
                TileType.GOAL,
            ]:
                for d in directions:
                    tile.adjacency[d] = floor_ids | wall_ids
            elif tile.tile_type == TileType.WALL:
                for d in directions:
                    tile.adjacency[d] = floor_ids | wall_ids
            elif tile.tile_type == TileType.BLOCK:
                for d in directions:
                    tile.adjacency[d] = floor_ids | wall_ids
            elif tile.tile_type == TileType.DOOR_LOCKED:
                for d in directions:
                    tile.adjacency[d] = floor_ids


@dataclass
class GameState:
    """Tracks game state during WFC collapse."""
    keys_collected: int = 0
    items_collected: Set[str] = field(default_factory=set)
    key_positions: List[Tuple[int, int]] = field(default_factory=list)
    lock_positions: List[Tuple[int, int]] = field(default_factory=list)
    placement_order: List[Tuple[int, int, int]] = field(default_factory=list)

    def copy(self) -> 'GameState':
        return GameState(
            keys_collected=self.keys_collected,
            items_collected=set(self.items_collected),
            key_positions=list(self.key_positions),
            lock_positions=list(self.lock_positions),
            placement_order=list(self.placement_order),
        )

    def can_unlock(self, required_keys: int = 1) -> bool:
        return self.keys_collected >= required_keys

    def collect_key(self, position: Tuple[int, int]) -> None:
        self.keys_collected += 1
        self.key_positions.append(position)

    def place_lock(self, position: Tuple[int, int]) -> None:
        self.lock_positions.append(position)


@dataclass
class Cell:
    """Single cell in the WFC grid."""
    row: int
    col: int
    possibilities: Set[int] = field(default_factory=set)
    collapsed_tile: Optional[int] = None

    def entropy(self, rng: Optional[random.Random] = None) -> float:
        if self.collapsed_tile is not None:
            return 0.0
        if rng is not None:
            noise = rng.random() * 0.1
        else:
            noise = ((self.row * 31 + self.col * 17) % 1000) / 10000.0
        return len(self.possibilities) + noise

    @property
    def is_collapsed(self) -> bool:
        return self.collapsed_tile is not None
