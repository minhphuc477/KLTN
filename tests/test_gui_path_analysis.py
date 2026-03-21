from types import SimpleNamespace

import numpy as np

from src.gui.services.path_analysis import scan_items_along_path


def test_scan_items_along_path_counts_items_and_doors():
    semantic_palette = {
        "KEY_SMALL": 30,
        "KEY_BOSS": 31,
        "ITEM_MINOR": 33,
        "DOOR_LOCKED": 11,
        "TRIFORCE": 22,
    }
    grid = np.array(
        [
            [30, 11, 0],
            [31, 33, 22],
        ]
    )
    gui = SimpleNamespace(
        auto_path=[(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)],
        env=SimpleNamespace(grid=grid),
        collected_positions=set(),
        path_items_summary={},
        path_item_positions={},
    )
    logger = SimpleNamespace(info=lambda *a, **k: None)

    summary = scan_items_along_path(gui, semantic_palette, logger)

    assert summary["keys"] == 1
    assert summary["doors_locked"] == 1
    assert summary["boss_keys"] == 1
    assert summary["bombs"] == 1
    assert summary["triforce"] == 1

