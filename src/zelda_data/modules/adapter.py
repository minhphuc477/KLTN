"""Adapter and conversion exports from the monolithic zelda_core module."""

from src.zelda_data.zelda_core import (
    DungeonData,
    RoomData,
    ZeldaDungeonAdapter,
    convert_dungeon_to_dungeondata,
    convert_room_to_roomdata,
)

__all__ = [
    "DungeonData",
    "RoomData",
    "ZeldaDungeonAdapter",
    "convert_dungeon_to_dungeondata",
    "convert_room_to_roomdata",
]
