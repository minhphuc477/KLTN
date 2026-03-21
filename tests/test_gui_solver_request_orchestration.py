from src.gui.services.solver_request_orchestration import (
    build_solver_request,
    get_solver_map_context,
)


class DummyGui:
    def __init__(self):
        self.maps = ["map0"]
        self.current_map_idx = 0
        self.env = "env"
        self.feature_flags = {"a": True}
        self.algorithm_idx = 2
        self.ara_weight = 1.25
        self.search_representation = "hybrid"
        self.messages = []

    def _set_message(self, msg):
        self.messages.append(msg)


def test_get_solver_map_context_uses_current_map():
    gui = DummyGui()

    ctx = get_solver_map_context(gui=gui, get_solver_map_context_helper=lambda m: {"grid_arr": m})

    assert ctx == {"grid_arr": "map0"}


def test_build_solver_request_passes_expected_fields():
    gui = DummyGui()

    def helper(**kwargs):
        assert kwargs["current_map"] == "map0"
        assert kwargs["env"] == "env"
        assert kwargs["feature_flags"] == {"a": True}
        assert kwargs["algorithm_idx"] == 5
        assert kwargs["ara_weight"] == 1.25
        assert kwargs["search_representation"] == "hybrid"
        return {"ok": True}

    req = build_solver_request(gui=gui, build_solver_request_helper=helper, algorithm_idx=5)

    assert req == {"ok": True}


def test_build_solver_request_sets_missing_message():
    gui = DummyGui()

    req = build_solver_request(
        gui=gui,
        build_solver_request_helper=lambda **kwargs: None,
        algorithm_idx=None,
        on_missing_message="missing start",
    )

    assert req is None
    assert gui.messages[-1] == "missing start"

