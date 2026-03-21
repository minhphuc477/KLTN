"""Path analysis helpers for collecting item/door summaries along planned paths."""

from typing import Any


def scan_items_along_path(gui: Any, semantic_palette: dict, logger: Any, path=None) -> dict:
    """Scan path positions and summarize collectible items and door interactions."""
    if path is None:
        path = getattr(gui, "auto_path", [])

    if not path or len(path) == 0:
        gui.path_items_summary = {}
        gui.path_item_positions = {}
        return {}

    key_small = semantic_palette.get("KEY_SMALL", 30)
    key_boss = semantic_palette.get("KEY_BOSS", 31)
    key_item = semantic_palette.get("KEY_ITEM", 32)
    item_minor = semantic_palette.get("ITEM_MINOR", 33)
    door_locked = semantic_palette.get("DOOR_LOCKED", 11)
    door_bomb = semantic_palette.get("DOOR_BOMB", 12)
    door_boss = semantic_palette.get("DOOR_BOSS", 14)
    triforce = semantic_palette.get("TRIFORCE", 22)

    summary = {
        "keys": 0,
        "boss_keys": 0,
        "ladders": 0,
        "bombs": 0,
        "doors_locked": 0,
        "doors_bomb": 0,
        "doors_boss": 0,
        "triforce": 0,
    }
    positions = {
        "keys": [],
        "boss_keys": [],
        "ladders": [],
        "bombs": [],
        "doors_locked": [],
        "doors_bomb": [],
        "doors_boss": [],
        "triforce": [],
    }

    if not gui.env or not hasattr(gui.env, "grid"):
        gui.path_items_summary = summary
        gui.path_item_positions = positions
        return summary

    grid = gui.env.grid
    already_collected = getattr(gui, "collected_positions", set())

    for pos in path:
        r, c = pos
        if r < 0 or r >= grid.shape[0] or c < 0 or c >= grid.shape[1]:
            continue

        tile_id = int(grid[r, c])
        if pos in already_collected:
            continue

        if tile_id == key_small:
            summary["keys"] += 1
            positions["keys"].append(pos)
        elif tile_id == key_boss:
            summary["boss_keys"] += 1
            positions["boss_keys"].append(pos)
        elif tile_id == key_item:
            summary["ladders"] += 1
            positions["ladders"].append(pos)
        elif tile_id == item_minor:
            summary["bombs"] += 1
            positions["bombs"].append(pos)
        elif tile_id == door_locked:
            summary["doors_locked"] += 1
            positions["doors_locked"].append(pos)
        elif tile_id == door_bomb:
            summary["doors_bomb"] += 1
            positions["doors_bomb"].append(pos)
        elif tile_id == door_boss:
            summary["doors_boss"] += 1
            positions["doors_boss"].append(pos)
        elif tile_id == triforce:
            summary["triforce"] += 1
            positions["triforce"].append(pos)

    gui.path_items_summary = summary
    gui.path_item_positions = positions

    items_found = []
    if summary["keys"] > 0:
        items_found.append(f"{summary['keys']} key(s)")
    if summary["boss_keys"] > 0:
        items_found.append(f"{summary['boss_keys']} boss key(s)")
    if summary["ladders"] > 0:
        items_found.append(f"{summary['ladders']} ladder(s)")
    if summary["bombs"] > 0:
        items_found.append(f"{summary['bombs']} bomb(s)")
    if summary["doors_locked"] > 0:
        items_found.append(f"{summary['doors_locked']} locked door(s)")
    if summary["doors_bomb"] > 0:
        items_found.append(f"{summary['doors_bomb']} bomb door(s)")
    if summary["doors_boss"] > 0:
        items_found.append(f"{summary['doors_boss']} boss door(s)")
    if summary["triforce"] > 0:
        items_found.append(f"{summary['triforce']} triforce")

    if items_found:
        logger.info("PATH ITEMS: %s", ", ".join(items_found))
    else:
        logger.info("PATH ITEMS: No collectible items along path")

    return summary
