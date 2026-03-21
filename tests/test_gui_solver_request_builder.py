import gui_runner


def test_build_solver_request_returns_payload():
    gui = gui_runner.ZeldaGUI(maps=[gui_runner.create_test_map()])
    gui.algorithm_idx = 2
    gui.search_representation = "hybrid"
    gui.ara_weight = 1.25

    request = gui._build_solver_request(algorithm_idx=2)

    assert request is not None
    assert request["alg_idx"] == 2
    assert request["start"] == tuple(gui.env.start_pos)
    assert request["goal"] == tuple(gui.env.goal_pos)
    assert "grid_arr" in request
    assert "flags" in request
    assert request["priority_options"]["representation"] == "hybrid"


def test_build_solver_request_handles_missing_start_goal():
    gui = gui_runner.ZeldaGUI(maps=[gui_runner.create_test_map()])
    gui.env.start_pos = None

    request = gui._build_solver_request()

    assert request is None
