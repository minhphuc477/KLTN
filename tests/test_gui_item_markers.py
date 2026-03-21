from types import SimpleNamespace

import numpy as np

from src.gui.overlay.item_markers import apply_pickup_at, scan_and_mark_items


class _Effects:
    def __init__(self):
        self.effects = []

    def add_effect(self, eff):
        self.effects.append(eff)


def test_scan_and_mark_items_counts_and_maps_types():
    semantic = {
        "KEY_SMALL": 30,
        "KEY_BOSS": 31,
        "ITEM_BOMB": 33,
        "ITEM_MINOR": 33,
        "TRIFORCE": 22,
    }
    grid = np.array([[30, 33], [31, 22]])
    env_state = SimpleNamespace(collected_items=set())
    gui = SimpleNamespace(
        item_markers={},
        item_type_map={},
        total_keys=0,
        total_bombs=0,
        total_boss_keys=0,
        env=SimpleNamespace(height=2, width=2, grid=grid, state=env_state),
        effects=_Effects(),
    )
    logger = SimpleNamespace(debug=lambda *a, **k: None)

    scan_and_mark_items(gui, semantic, logger, item_marker_effect_cls=lambda pos, typ, label: SimpleNamespace(pos=pos, item_type=typ, label=label))

    assert gui.total_keys == 1
    assert gui.total_bombs == 1
    assert gui.total_boss_keys == 1
    assert gui.item_type_map[(1, 1)] == "triforce"


def test_apply_pickup_at_key_mutates_state_and_grid():
    semantic = {
        "KEY_SMALL": 30,
        "ITEM_BOMB": 33,
        "KEY_BOSS": 31,
        "TRIFORCE": 22,
        "FLOOR": 0,
    }
    grid = np.array([[30]])
    env_state = SimpleNamespace(collected_items=set(), keys=0, bomb_count=0, has_boss_key=False)
    toasts = []
    gui = SimpleNamespace(
        env=SimpleNamespace(height=1, width=1, grid=grid, state=env_state),
        collected_items=[],
        collected_positions=set(),
        item_pickup_times={},
        keys_collected=0,
        bombs_collected=0,
        boss_keys_collected=0,
        total_keys=1,
        total_bombs=0,
        total_boss_keys=0,
        item_markers={},
        item_type_map={},
        effects=_Effects(),
        collection_effects=[],
        _show_toast=lambda *args, **kwargs: toasts.append((args, kwargs)),
    )
    logger = SimpleNamespace(warning=lambda *a, **k: None)
    time_module = SimpleNamespace(time=lambda: 5.0)

    ok = apply_pickup_at(gui, (0, 0), semantic, logger, time_module, item_collection_effect_cls=lambda *a, **k: (a, k))

    assert ok is True
    assert gui.env.state.keys == 1
    assert int(gui.env.grid[0, 0]) == 0
    assert (0, 0) in gui.env.state.collected_items
    assert gui.keys_collected == 1
    assert toasts

