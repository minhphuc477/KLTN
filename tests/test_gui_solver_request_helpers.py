from src.gui.services.solver_request_helpers import (
    build_priority_options,
    build_solver_request,
    get_solver_map_context,
)


class _MapWithGrid:
    def __init__(self):
        self.global_grid = [[0, 1], [1, 0]]
        self.graph = {"dummy": True}
        self.room_to_node = {"a": 1}
        self.room_positions = {(0, 0): (0, 0)}
        self.node_to_room = {1: (0, 0)}


class _Env:
    def __init__(self, start_pos=(1, 1), goal_pos=(2, 2)):
        self.start_pos = start_pos
        self.goal_pos = goal_pos


def test_get_solver_map_context_with_global_grid():
    ctx = get_solver_map_context(_MapWithGrid())
    assert ctx["grid_arr"] == [[0, 1], [1, 0]]
    assert ctx["graph"] == {"dummy": True}
    assert ctx["room_to_node"] == {"a": 1}


def test_get_solver_map_context_plain_grid():
    grid = [[1, 2], [3, 4]]
    ctx = get_solver_map_context(grid)
    assert ctx["grid_arr"] == grid
    assert ctx["graph"] is None


def test_build_priority_options():
    flags = {
        "priority_tie_break": True,
        "priority_key_boost": False,
        "enable_ara": True,
    }
    opts = build_priority_options(flags, ara_weight=1.5, search_representation="hybrid")
    assert opts["tie_break"] is True
    assert opts["key_boost"] is False
    assert opts["enable_ara"] is True
    assert opts["ara_weight"] == 1.5
    assert opts["representation"] == "hybrid"
    assert opts["allow_diagonals"] is True


def test_build_solver_request_missing_start_goal():
    req = build_solver_request(
        current_map=_MapWithGrid(),
        env=_Env(start_pos=None, goal_pos=None),
        feature_flags={},
        algorithm_idx=0,
        ara_weight=1.0,
        search_representation="hybrid",
    )
    assert req is None


def test_build_solver_request_success():
    req = build_solver_request(
        current_map=_MapWithGrid(),
        env=_Env(start_pos=(0, 1), goal_pos=(2, 3)),
        feature_flags={"enable_ara": False},
        algorithm_idx=4,
        ara_weight=1.25,
        search_representation="tile",
    )
    assert req is not None
    assert req["start"] == (0, 1)
    assert req["goal"] == (2, 3)
    assert req["alg_idx"] == 4
    assert req["priority_options"]["representation"] == "tile"

