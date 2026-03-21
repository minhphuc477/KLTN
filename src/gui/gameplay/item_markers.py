"""Helpers for item marker scanning and direct pickup application."""

from typing import Any, Tuple


def scan_and_mark_items(gui: Any, semantic_palette: dict, logger: Any, item_marker_effect_cls: Any) -> None:
    """Scan current grid for collectible items and register markers/counts."""
    gui.item_markers.clear()
    gui.item_type_map.clear()
    gui.total_keys = 0
    gui.total_bombs = 0
    gui.total_boss_keys = 0

    if not gui.env:
        return

    h, w = gui.env.height, gui.env.width
    for r in range(h):
        for c in range(w):
            tile_id = gui.env.grid[r, c]
            pos = (r, c)
            if pos in getattr(gui.env.state, "collected_items", set()):
                continue

            if tile_id == semantic_palette["KEY_SMALL"]:
                gui.total_keys += 1
                marker = item_marker_effect_cls(pos, "key", "K")
                gui.item_markers[pos] = marker
                gui.item_type_map[pos] = "key"
                if gui.effects:
                    gui.effects.add_effect(marker)
            elif tile_id == semantic_palette.get("ITEM_BOMB", -1) or tile_id == semantic_palette.get("ITEM_MINOR", -1):
                gui.total_bombs += 1
                marker = item_marker_effect_cls(pos, "bomb", "B")
                gui.item_markers[pos] = marker
                gui.item_type_map[pos] = "bomb"
                if gui.effects:
                    gui.effects.add_effect(marker)
            elif tile_id == semantic_palette["KEY_BOSS"]:
                gui.total_boss_keys += 1
                marker = item_marker_effect_cls(pos, "boss_key", "BK")
                gui.item_markers[pos] = marker
                gui.item_type_map[pos] = "boss_key"
                if gui.effects:
                    gui.effects.add_effect(marker)
            elif tile_id == semantic_palette["TRIFORCE"]:
                marker = item_marker_effect_cls(pos, "triforce", "TRI")
                gui.item_markers[pos] = marker
                gui.item_type_map[pos] = "triforce"
                if gui.effects:
                    gui.effects.add_effect(marker)

    logger.debug(
        "_scan_and_mark_items: found %d keys, %d bombs, %d boss_keys, item_type_map has %d entries",
        gui.total_keys,
        gui.total_bombs,
        gui.total_boss_keys,
        len(gui.item_type_map),
    )


def apply_pickup_at(
    gui: Any,
    pos: Tuple[int, int],
    semantic_palette: dict,
    logger: Any,
    time_module: Any,
    item_collection_effect_cls: Any,
) -> bool:
    """Apply pickup side effects at position for teleports/external mutations."""
    if not gui.env:
        return False
    try:
        r, c = pos
        if r < 0 or c < 0 or r >= gui.env.height or c >= gui.env.width:
            return False
        if pos in getattr(gui.env.state, "collected_items", set()):
            return False
        tile_id = int(gui.env.grid[r, c])
        if tile_id not in (
            semantic_palette["KEY_SMALL"],
            semantic_palette.get("ITEM_BOMB", -1),
            semantic_palette["KEY_BOSS"],
            semantic_palette["TRIFORCE"],
        ):
            return False

        try:
            collected = set(getattr(gui.env.state, "collected_items", set()) or set())
            gui.env.state.collected_items = collected | {pos}
        except Exception:
            pass

        if tile_id == semantic_palette["KEY_SMALL"]:
            try:
                gui.env.state.keys = getattr(gui.env.state, "keys", 0) + 1
            except Exception:
                pass
            try:
                gui.env.grid[r, c] = semantic_palette["FLOOR"]
            except Exception:
                logger.warning("Failed to update grid for collected key")
            gui.collected_items.append((pos, "key", time_module.time()))
            gui.collected_positions.add(pos)
            gui.item_pickup_times["key"] = time_module.time()
            gui.keys_collected = getattr(gui, "keys_collected", 0) + 1
            if pos in gui.item_markers and gui.item_markers[pos].item_type == "key":
                del gui.item_markers[pos]
            if gui.effects:
                eff = item_collection_effect_cls(pos, "key", "KEY", f"Key collected at ({pos[0]}, {pos[1]})!")
                gui.effects.add_effect(eff)
                gui.collection_effects.append(eff)
            gui._show_toast(f"Key collected! Now have {gui.keys_collected}/{gui.total_keys}", duration=2.5, toast_type="success")
            gui.item_type_map[pos] = "key"
            return True

        if tile_id == semantic_palette.get("ITEM_BOMB", -1):
            try:
                gui.env.state.bomb_count += 4
            except Exception:
                pass
            try:
                gui.env.grid[r, c] = semantic_palette["FLOOR"]
            except Exception:
                logger.warning("Failed to update grid for collected bomb")
            gui.collected_items.append((pos, "bomb", time_module.time()))
            gui.collected_positions.add(pos)
            gui.item_pickup_times["bomb"] = time_module.time()
            gui.bombs_collected = getattr(gui, "bombs_collected", 0) + 1
            if pos in gui.item_markers and gui.item_markers[pos].item_type == "bomb":
                del gui.item_markers[pos]
            if gui.effects:
                eff = item_collection_effect_cls(pos, "bomb", "BOMB", f"Bomb collected at ({pos[0]}, {pos[1]})!")
                gui.effects.add_effect(eff)
                gui.collection_effects.append(eff)
            gui._show_toast("Bomb acquired! Can now blow up weak walls", duration=3.0, toast_type="success")
            gui.item_type_map[pos] = "bomb"
            return True

        if tile_id == semantic_palette["KEY_BOSS"]:
            try:
                gui.env.state.has_boss_key = True
            except Exception:
                pass
            try:
                gui.env.grid[r, c] = semantic_palette["FLOOR"]
            except Exception:
                logger.warning("Failed to update grid for collected boss key")
            gui.collected_items.append((pos, "boss_key", time_module.time()))
            gui.collected_positions.add(pos)
            gui.item_pickup_times["boss_key"] = time_module.time()
            gui.boss_keys_collected = getattr(gui, "boss_keys_collected", 0) + 1
            if pos in gui.item_markers and gui.item_markers[pos].item_type == "boss_key":
                del gui.item_markers[pos]
            if gui.effects:
                eff = item_collection_effect_cls(pos, "boss_key", "BOSS KEY", f"Boss Key collected at ({pos[0]}, {pos[1]})!")
                gui.effects.add_effect(eff)
                gui.collection_effects.append(eff)
            gui._show_toast("Boss Key acquired! Can now face the boss", duration=3.0, toast_type="success")
            gui.item_type_map[pos] = "boss_key"
            return True

        if tile_id == semantic_palette["TRIFORCE"]:
            try:
                gui.env.state.collected_items = getattr(gui.env.state, "collected_items", set()) | {pos}
            except Exception:
                pass
            if gui.effects:
                eff = item_collection_effect_cls(pos, "triforce", "TRI", f"Triforce at ({pos[0]}, {pos[1]})!")
                gui.effects.add_effect(eff)
            gui._show_toast("Triforce found!", duration=3.0, toast_type="success")
            gui.item_type_map[pos] = "triforce"
            return True
    except Exception as e:
        logger.warning(f"_apply_pickup_at failed: {e}")
    return False
