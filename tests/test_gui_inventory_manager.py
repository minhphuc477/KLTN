from types import SimpleNamespace

from src.gui.overlay.inventory_manager import (
    remove_from_path_items,
    sync_inventory_counters,
    track_item_usage,
)


def test_remove_from_path_items_updates_summary():
    gui = SimpleNamespace(
        path_item_positions={"keys": [(1, 2)]},
        path_items_summary={"keys": 1},
    )
    logger = SimpleNamespace(debug=lambda *a, **k: None, warning=lambda *a, **k: None)

    remove_from_path_items(gui, (1, 2), "keys", logger)

    assert gui.path_item_positions["keys"] == []
    assert gui.path_items_summary["keys"] == 0


def test_sync_inventory_counters_uses_max_sources():
    env_state = SimpleNamespace(collected_items={(0, 0), (0, 1)})
    gui = SimpleNamespace(
        keys_collected=0,
        bombs_collected=0,
        boss_keys_collected=0,
        keys_used=0,
        bombs_used=0,
        boss_keys_used=0,
        collected_items=[((0, 0), "key", 1.0)],
        item_type_map={(0, 0): "key", (0, 1): "key"},
        env=SimpleNamespace(state=env_state),
    )

    sync_inventory_counters(gui)

    assert gui.keys_collected == 2


def test_track_item_usage_updates_key_counter():
    old_state = SimpleNamespace(keys=2, has_bomb=False, has_boss_key=False, position=(1, 1))
    new_state = SimpleNamespace(keys=1, has_bomb=False, has_boss_key=False, position=(1, 2))
    toasts = []
    gui = SimpleNamespace(
        used_items=[],
        usage_effects=[],
        effects=None,
        keys_used=0,
        bombs_used=0,
        boss_keys_used=0,
        env=SimpleNamespace(state=new_state),
        _show_toast=lambda msg, duration, toast_type: toasts.append((msg, duration, toast_type)),
        _update_inventory_and_hud=lambda: None,
    )
    logger = SimpleNamespace(info=lambda *a, **k: None, exception=lambda *a, **k: None)
    time_module = SimpleNamespace(time=lambda: 10.0)

    track_item_usage(gui, old_state, new_state, time_module, logger, item_usage_effect_cls=lambda a, b, c: (a, b, c))

    assert gui.keys_used == 1
    assert gui.used_items
    assert toasts

