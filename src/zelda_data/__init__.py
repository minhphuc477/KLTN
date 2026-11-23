"""Canonical Zelda data package."""

from src.zelda_data.zelda_core import (
    Dungeon,
    DungeonSolver,
    GridBasedRoomExtractor,
    Room,
    RoomGraphMatcher,
    StitchedDungeon,
    ValidationMode,
    ZeldaDungeonAdapter,
)
from src.zelda_data.zelda_loader import (
    TILE_MAPPING,
    ZeldaDungeonDataset,
    ZeldaRoomDataset,
    apply_room_symmetry_transform,
    compute_room_tile_frequencies,
    create_dataloader,
)

__all__ = [
    "Dungeon",
    "DungeonSolver",
    "GridBasedRoomExtractor",
    "Room",
    "RoomGraphMatcher",
    "StitchedDungeon",
    "ValidationMode",
    "ZeldaDungeonAdapter",
    "TILE_MAPPING",
    "ZeldaDungeonDataset",
    "ZeldaRoomDataset",
    "apply_room_symmetry_transform",
    "compute_room_tile_frequencies",
    "create_dataloader",
]
