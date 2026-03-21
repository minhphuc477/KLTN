"""Inventory and item usage tracking helpers for Zelda GUI."""

from typing import Any


def update_inventory_and_hud(gui: Any, logger: Any) -> None:
    """Reconcile counters and update modern HUD, deferring to main thread when needed."""
    try:
        thread_name = None
        import threading

        thread_name = threading.current_thread().name
    except Exception:
        thread_name = "unknown"
    logger.debug(
        "_update_inventory_and_hud: entry (thread=%s, inventory_needs_refresh=%s)",
        thread_name,
        getattr(gui, "inventory_needs_refresh", False),
    )

    try:
        before_keys = getattr(gui, "keys_collected", None)
        env_keys = getattr(getattr(gui, "env", None), "state", None)
        env_keys_val = getattr(env_keys, "keys", None) if env_keys is not None else None
        logger.debug("Counters before sync: env.keys=%s, keys_collected=%s", env_keys_val, before_keys)
        gui._sync_inventory_counters()
        after_keys = getattr(gui, "keys_collected", None)
        logger.debug("Counters after sync: keys_collected=%s", after_keys)
    except Exception:
        logger.exception("_update_inventory_and_hud: failed while syncing counters")

    try:
        if getattr(gui.env, "state", None):
            if getattr(gui.env.state, "keys", 0) < 0:
                logger.warning("Inventory inconsistency: env.state.keys < 0")
    except Exception:
        pass

    try:
        import threading

        if threading.current_thread() is not threading.main_thread():
            gui.inventory_needs_refresh = True
            logger.debug("_update_inventory_and_hud: deferred to main thread (set inventory_needs_refresh=True)")
            return
    except Exception:
        pass

    try:
        if getattr(gui, "inventory_needs_refresh", False):
            logger.debug("_update_inventory_and_hud: clearing deferred flag (main thread)")
            try:
                gui.inventory_needs_refresh = False
            except Exception:
                pass

        if getattr(gui, "modern_hud", None):
            try:
                hud_before = getattr(gui.modern_hud, "last", None) if hasattr(gui.modern_hud, "last") else None
            except Exception:
                hud_before = None

            gui.modern_hud.update_game_state(
                keys=getattr(gui.env.state, "keys", 0),
                bombs=getattr(gui.env.state, "bomb_count", 0),
                has_boss_key=getattr(gui.env.state, "has_boss_key", False),
                position=getattr(gui.env.state, "position", (0, 0)),
                steps=getattr(gui, "step_count", 0),
                message=getattr(gui, "message", ""),
            )
            try:
                if hasattr(gui.modern_hud, "keys_collected"):
                    gui.modern_hud.keys_collected = getattr(gui, "keys_collected", 0)
                if hasattr(gui.modern_hud, "bombs_collected"):
                    gui.modern_hud.bombs_collected = getattr(gui, "bombs_collected", 0)
                if hasattr(gui.modern_hud, "boss_keys_collected"):
                    gui.modern_hud.boss_keys_collected = getattr(gui, "boss_keys_collected", 0)

                if hasattr(gui.modern_hud, "keys_used"):
                    gui.modern_hud.keys_used = getattr(gui, "keys_used", 0)
                if hasattr(gui.modern_hud, "bombs_used"):
                    gui.modern_hud.bombs_used = getattr(gui, "bombs_used", 0)
                if hasattr(gui.modern_hud, "boss_keys_used"):
                    gui.modern_hud.boss_keys_used = getattr(gui, "boss_keys_used", 0)

                if hasattr(gui.modern_hud, "inventory"):
                    try:
                        gui.modern_hud.inventory.keys_collected = getattr(gui, "keys_collected", 0)
                        gui.modern_hud.inventory.bombs_collected = getattr(gui, "bombs_collected", 0)
                        gui.modern_hud.inventory.boss_keys_collected = getattr(gui, "boss_keys_collected", 0)
                        gui.modern_hud.inventory.keys_used = getattr(gui, "keys_used", 0)
                        gui.modern_hud.inventory.bombs_used = getattr(gui, "bombs_used", 0)
                        gui.modern_hud.inventory.boss_keys_used = getattr(gui, "boss_keys_used", 0)
                    except Exception:
                        logger.exception("Failed setting nested inventory attributes")
            except Exception:
                logger.exception("Failed setting HUD count attributes")

            try:
                hud_after = getattr(gui.modern_hud, "last", None) if hasattr(gui.modern_hud, "last") else None
                logger.debug(
                    "HUD updated: before=%r after=%r env.keys=%s keys_collected=%s",
                    hud_before,
                    hud_after,
                    getattr(gui.env.state, "keys", None),
                    getattr(gui, "keys_collected", None),
                )
            except Exception:
                logger.exception("Failed to log HUD post-update state")
    except Exception as e:
        logger.warning(f"Failed to update modern HUD: {e}")


def remove_from_path_items(gui: Any, pos: tuple, item_type: str, logger: Any) -> None:
    """Remove a collected item from path item preview structures."""
    try:
        path_positions = getattr(gui, "path_item_positions", {})
        path_summary = getattr(gui, "path_items_summary", {})

        if item_type in path_positions and pos in path_positions[item_type]:
            path_positions[item_type].remove(pos)
            if item_type in path_summary and path_summary[item_type] > 0:
                path_summary[item_type] -= 1
            logger.debug("Removed %s at %s from path items preview", item_type, pos)
    except Exception as e:
        logger.warning("Failed to remove %s at %s from path items: %s", item_type, pos, e)


def track_item_collection(
    gui: Any,
    old_state: Any,
    new_state: Any,
    time_module: Any,
    logger: Any,
    pop_effect_cls: Any,
    item_collection_effect_cls: Any,
) -> None:
    """Detect collected items from state transitions and update UI state/effects."""
    logger.debug(
        "TRACK_COLLECTION: old_keys=%d, new_keys=%d, old_bomb=%s, new_bomb=%s, old_boss=%s, new_boss=%s",
        old_state.keys,
        new_state.keys,
        old_state.has_bomb,
        new_state.has_bomb,
        getattr(old_state, "has_boss_key", False),
        getattr(new_state, "has_boss_key", False),
    )

    if new_state.keys > old_state.keys:
        keys_collected = new_state.keys - old_state.keys
        pos = new_state.position
        timestamp = time_module.time()

        logger.info("KEY_COLLECTED: pos=%s, count=%d, total_collected_now=%d", pos, keys_collected, gui.keys_collected + keys_collected)
        gui.collected_items.append((pos, "key", timestamp))
        gui.collected_positions.add(pos)
        gui.keys_collected += keys_collected
        gui.item_pickup_times["key"] = timestamp
        try:
            gui.item_type_map[pos] = gui.item_type_map.get(pos, "key")
        except Exception:
            pass

        if pos in gui.item_markers and gui.item_markers[pos].item_type == "key":
            del gui.item_markers[pos]

        gui._remove_from_path_items(pos, "keys")

        if gui.effects:
            effect = item_collection_effect_cls(pos, "key", "KEY", f"Key collected at ({pos[0]}, {pos[1]})!")
            gui.effects.add_effect(effect)
            gui.collection_effects.append(effect)
            gui.effects.add_effect(pop_effect_cls(pos, (255, 215, 0)))

        gui._show_toast(f"Key collected! Now have {gui.keys_collected}/{gui.total_keys}", duration=2.5, toast_type="success")
        try:
            logger.info(
                "Detected key collection at %s (keys_collected=%s, env.keys=%s)",
                pos,
                gui.keys_collected,
                getattr(gui.env.state, "keys", None),
            )
        except Exception:
            pass
        try:
            gui.last_pickup_msg = f"Picked up key at {pos}"
        except Exception:
            pass
        try:
            gui._update_inventory_and_hud()
        except Exception:
            pass

    if new_state.has_bomb and not old_state.has_bomb:
        pos = new_state.position
        timestamp = time_module.time()

        gui.collected_items.append((pos, "bomb", timestamp))
        gui.collected_positions.add(pos)
        gui.bombs_collected += 1
        gui.item_pickup_times["bomb"] = timestamp
        try:
            gui.item_type_map[pos] = gui.item_type_map.get(pos, "bomb")
        except Exception:
            pass

        gui._remove_from_path_items(pos, "bombs")

        if pos in gui.item_markers and gui.item_markers[pos].item_type == "bomb":
            del gui.item_markers[pos]

        if gui.effects:
            effect = item_collection_effect_cls(pos, "bomb", "BOMB", f"Bomb collected at ({pos[0]}, {pos[1]})!")
            gui.effects.add_effect(effect)
            gui.collection_effects.append(effect)
            gui.effects.add_effect(pop_effect_cls(pos, (255, 107, 53)))

        gui._show_toast("Bomb acquired! Can now blow up weak walls", duration=3.0, toast_type="success")
        try:
            logger.info(
                "Detected bomb collection at %s (bombs_collected=%s, env.has_bomb=%s)",
                pos,
                gui.bombs_collected,
                getattr(gui.env.state, "has_bomb", None),
            )
        except Exception:
            pass
        try:
            gui._update_inventory_and_hud()
        except Exception:
            pass

    if new_state.has_boss_key and not old_state.has_boss_key:
        pos = new_state.position
        timestamp = time_module.time()

        gui.collected_items.append((pos, "boss_key", timestamp))
        gui.collected_positions.add(pos)
        gui.boss_keys_collected += 1

        if pos in gui.item_markers and gui.item_markers[pos].item_type == "boss_key":
            del gui.item_markers[pos]

        gui._remove_from_path_items(pos, "boss_keys")

        if gui.effects:
            effect = item_collection_effect_cls(pos, "boss_key", "BOSS KEY", f"Boss Key collected at ({pos[0]}, {pos[1]})!")
            gui.effects.add_effect(effect)
            gui.collection_effects.append(effect)
            gui.effects.add_effect(pop_effect_cls(pos, (176, 66, 255)))

        gui._show_toast("Boss Key acquired! Can now face the boss", duration=3.0, toast_type="success")
        try:
            gui._update_inventory_and_hud()
        except Exception:
            pass


def track_item_usage(gui: Any, old_state: Any, new_state: Any, time_module: Any, logger: Any, item_usage_effect_cls: Any) -> None:
    """Detect used items from state transitions and update counters/effects."""
    _ = time_module
    if not hasattr(gui, "used_items"):
        gui.used_items = []
    if not hasattr(gui, "usage_effects"):
        gui.usage_effects = []

    if new_state.keys < old_state.keys:
        keys_used = old_state.keys - new_state.keys
        pos = new_state.position
        timestamp = time_module.time()

        try:
            gui.used_items.append((pos, "key", pos, timestamp))
        except Exception:
            logger.exception("Failed appending to used_items")
        gui.keys_used = getattr(gui, "keys_used", 0) + keys_used
        try:
            logger.info("Key used at %s (keys_used=%s, env.keys=%s)", pos, gui.keys_used, getattr(gui.env.state, "keys", None))
            gui.last_use_msg = f"Used key at {pos}"
        except Exception:
            pass
        try:
            gui._update_inventory_and_hud()
        except Exception:
            pass

        if gui.effects:
            effect = item_usage_effect_cls(old_state.position, pos, "key")
            gui.effects.add_effect(effect)
            gui.usage_effects.append(effect)
        gui._show_toast(f"Key used! ({gui.keys_used} used)", duration=1.8, toast_type="info")

    if old_state.has_bomb and not new_state.has_bomb:
        pos = new_state.position
        timestamp = time_module.time()

        gui.used_items.append((pos, "bomb", pos, timestamp))
        gui.bombs_used = getattr(gui, "bombs_used", 0) + 1
        try:
            logger.info("Bomb used at %s (bombs_used=%s, env.has_bomb=%s)", pos, gui.bombs_used, getattr(gui.env.state, "has_bomb", None))
        except Exception:
            pass

        if gui.effects:
            effect = item_usage_effect_cls(old_state.position, pos, "bomb")
            gui.effects.add_effect(effect)
            gui.usage_effects.append(effect)
        gui._show_toast(f"Bomb used! ({gui.bombs_used} used)", duration=1.8, toast_type="info")
        try:
            gui._update_inventory_and_hud()
        except Exception:
            pass

    if old_state.has_boss_key and not new_state.has_boss_key:
        pos = new_state.position
        timestamp = time_module.time()

        gui.used_items.append((pos, "boss_key", pos, timestamp))
        gui.boss_keys_used = getattr(gui, "boss_keys_used", 0) + 1

        if gui.effects:
            effect = item_usage_effect_cls(old_state.position, pos, "boss_key")
            gui.effects.add_effect(effect)
            gui.usage_effects.append(effect)
        gui._show_toast(f"Boss key used! ({gui.boss_keys_used} used)", duration=2.5, toast_type="info")
        try:
            gui._update_inventory_and_hud()
        except Exception:
            pass


def sync_inventory_counters(gui: Any) -> None:
    """Reconcile collected and used counters from state and tracking buffers."""
    gui.keys_collected = getattr(gui, "keys_collected", 0)
    gui.bombs_collected = getattr(gui, "bombs_collected", 0)
    gui.boss_keys_collected = getattr(gui, "boss_keys_collected", 0)

    kc_list = 0
    bc_list = 0
    bkc_list = 0
    if gui.collected_items:
        for _pos, item_type, _ts in gui.collected_items:
            if item_type == "key":
                kc_list += 1
            elif item_type == "bomb":
                bc_list += 1
            elif item_type == "boss_key":
                bkc_list += 1

    kc_map = 0
    bc_map = 0
    bkc_map = 0
    try:
        collected_set = set(getattr(gui.env.state, "collected_items", set()) or set())
        for pos in collected_set:
            it = gui.item_type_map.get(pos)
            if it == "key":
                kc_map += 1
            elif it == "bomb":
                bc_map += 1
            elif it == "boss_key":
                bkc_map += 1
    except Exception:
        pass

    gui.keys_collected = max(kc_list, kc_map)
    gui.bombs_collected = max(bc_list, bc_map)
    gui.boss_keys_collected = max(bkc_list, bkc_map)

    gui.keys_used = getattr(gui, "keys_used", 0)
    gui.bombs_used = getattr(gui, "bombs_used", 0)
    gui.boss_keys_used = getattr(gui, "boss_keys_used", 0)
