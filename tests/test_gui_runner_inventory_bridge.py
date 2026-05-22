from types import SimpleNamespace

import gui_runner


def test_gui_runner_inventory_wrappers_forward_time_module(monkeypatch):
    gui = gui_runner.ZeldaGUI.__new__(gui_runner.ZeldaGUI)
    old_state = SimpleNamespace(position=(1, 1))
    new_state = SimpleNamespace(position=(1, 2))
    calls = {}

    def fake_collection(**kwargs):
        calls["collection"] = kwargs

    def fake_usage(**kwargs):
        calls["usage"] = kwargs

    monkeypatch.setattr(gui_runner, "_track_item_collection_orchestration_helper", fake_collection)
    monkeypatch.setattr(gui_runner, "_track_item_usage_orchestration_helper", fake_usage)

    gui._track_item_collection(old_state, new_state)
    gui._track_item_usage(old_state, new_state)

    assert calls["collection"]["gui"] is gui
    assert calls["collection"]["old_state"] is old_state
    assert calls["collection"]["new_state"] is new_state
    assert calls["collection"]["time_module"] is gui_runner.time
    assert calls["usage"]["gui"] is gui
    assert calls["usage"]["old_state"] is old_state
    assert calls["usage"]["new_state"] is new_state
    assert calls["usage"]["time_module"] is gui_runner.time


def test_gui_runner_manual_step_wrapper_forwards_dependencies(monkeypatch):
    gui = gui_runner.ZeldaGUI.__new__(gui_runner.ZeldaGUI)
    calls = {}
    sentinel = object()

    def fake_manual_step(**kwargs):
        calls["manual_step"] = kwargs
        return sentinel

    monkeypatch.setattr(gui_runner, "_manual_step_orchestration_helper", fake_manual_step)

    result = gui._manual_step(gui_runner.Action.UP)

    assert result is sentinel
    assert calls["manual_step"]["gui"] is gui
    assert calls["manual_step"]["action"] is gui_runner.Action.UP
    assert calls["manual_step"]["action_deltas"] is gui_runner.ACTION_DELTAS
    assert calls["manual_step"]["pop_effect_cls"] is gui_runner.PopEffect
    assert calls["manual_step"]["flash_effect_cls"] is gui_runner.FlashEffect
    assert calls["manual_step"]["time_module"] is gui_runner.time
