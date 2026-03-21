from src.gui.controls.runtime_flags import load_runtime_flags


def test_load_runtime_flags_defaults(monkeypatch):
    monkeypatch.delenv("KLTN_LOG_LEVEL", raising=False)
    monkeypatch.delenv("KLTN_DEBUG_INPUT", raising=False)
    monkeypatch.delenv("KLTN_SYNC_SOLVER", raising=False)
    monkeypatch.delenv("KLTN_DEBUG_SOLVER_FLOW", raising=False)

    flags = load_runtime_flags()

    assert flags.log_level == ""
    assert flags.debug_input_active is False
    assert flags.debug_sync_solver is False
    assert flags.debug_solver_flow is False


def test_load_runtime_flags_custom(monkeypatch):
    monkeypatch.setenv("KLTN_LOG_LEVEL", "debug")
    monkeypatch.setenv("KLTN_DEBUG_INPUT", "1")
    monkeypatch.setenv("KLTN_SYNC_SOLVER", "1")
    monkeypatch.setenv("KLTN_DEBUG_SOLVER_FLOW", "1")

    flags = load_runtime_flags()

    assert flags.log_level == "DEBUG"
    assert flags.debug_input_active is True
    assert flags.debug_sync_solver is True
    assert flags.debug_solver_flow is True

