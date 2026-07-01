from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.gui.gameplay.path_strategies import smart_grid_path


def test_completionist_and_novice_skip_quick_grid_path():
    """P-CBS personas must use their cognitive solver, not the quick grid path."""
    grid = np.zeros((3, 3), dtype=np.int32)
    grid[1, 1] = 21
    grid[1, 2] = 22
    logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)

    for algorithm_idx in (13, 14):
        gui = SimpleNamespace(
            maps=[grid],
            current_map_idx=0,
            env=SimpleNamespace(start_pos=(1, 1), goal_pos=(1, 2)),
            algorithm_idx=algorithm_idx,
            show_heatmap=False,
            feature_flags={},
            _algorithm_name=lambda idx: f"P-CBS {idx}",
        )

        success, path, teleports = smart_grid_path(
            gui,
            logger,
            convert_diagonal_to_4dir=lambda p, grid=None: p,
            semantic_palette={},
            np_module=np,
            path_cls=Path,
            os_module=SimpleNamespace(environ={}),
        )

        assert success is False
        assert path == []
        assert teleports == 0
