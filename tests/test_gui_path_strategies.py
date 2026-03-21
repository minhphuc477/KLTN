from types import SimpleNamespace

import numpy as np

from src.gui.services.path_strategies import graph_guided_path, hybrid_graph_grid_path, smart_grid_path


class _Logger:
    def debug(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


def test_smart_grid_path_bfs_returns_path_for_simple_grid():
    grid = np.ones((3, 3), dtype=int)
    gui = SimpleNamespace(
        maps=[grid],
        current_map_idx=0,
        env=SimpleNamespace(start_pos=(0, 0), goal_pos=(2, 2), state=SimpleNamespace()),
        show_heatmap=False,
        search_heatmap={},
        feature_flags={"ml_heuristic": False, "diagonal_movement": False, "use_jps": False},
        algorithm_idx=1,
        _ml_heuristic_ready=True,
        _ml_heuristic=None,
        _algorithm_name=lambda idx: f"alg-{idx}",
    )

    ok, path, teleports = smart_grid_path(
        gui=gui,
        logger=_Logger(),
        convert_diagonal_to_4dir=lambda p, grid=None: p,
        semantic_palette={"DOOR_LOCKED": 11},
        np_module=np,
        path_cls=__import__("pathlib").Path,
        os_module=__import__("os"),
    )

    assert ok is True
    assert teleports == 0
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)


def test_graph_guided_path_falls_back_to_direct_bfs_without_topology():
    grid = np.ones((2, 2), dtype=int)
    gui = SimpleNamespace(
        maps=[grid],
        current_map_idx=0,
        env=SimpleNamespace(start_pos=(0, 0), goal_pos=(1, 1)),
    )

    ok, path, teleports = graph_guided_path(gui)

    assert ok is True
    assert teleports == 0
    assert path[0] == (0, 0)
    assert path[-1] == (1, 1)


def test_hybrid_graph_grid_path_falls_back_to_direct_bfs_without_topology():
    grid = np.ones((2, 3), dtype=int)
    gui = SimpleNamespace(
        maps=[grid],
        current_map_idx=0,
        env=SimpleNamespace(start_pos=(0, 0), goal_pos=(1, 2)),
    )

    ok, path, teleports = hybrid_graph_grid_path(gui)

    assert ok is True
    assert teleports == 0
    assert path[0] == (0, 0)
    assert path[-1] == (1, 2)

