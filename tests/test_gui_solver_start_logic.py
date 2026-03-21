from src.gui.services.solver_start_logic import (
    default_solver_timeout_for_algorithm,
    evaluate_solver_recovery_state,
    scale_timeout_by_grid_size,
    sync_solver_dropdown_settings,
)


class _Widget:
    def __init__(self, control_name, selected=0, options=None):
        self.control_name = control_name
        self.selected = selected
        self.options = options or []


def test_sync_solver_dropdown_settings_reads_widgets():
    widgets = [
        _Widget("algorithm", selected=3),
        _Widget("representation", selected=1),
        _Widget("ara_weight", selected=2, options=["1.0", "1.25", "1.5"]),
    ]
    alg, rep, weight = sync_solver_dropdown_settings(0, "hybrid", 1.0, widgets)
    assert alg == 3
    assert rep == "tile"
    assert weight == 1.5


def test_sync_solver_dropdown_settings_normalizes_invalid_values():
    widgets = [_Widget("representation", selected=99)]
    alg, rep, weight = sync_solver_dropdown_settings(2, "invalid", -5.0, widgets)
    assert alg == 2
    assert rep == "hybrid"
    assert weight == 1.0


def test_default_solver_timeout_for_algorithm():
    assert default_solver_timeout_for_algorithm(0) == 60.0
    assert default_solver_timeout_for_algorithm(1) == 180.0
    assert default_solver_timeout_for_algorithm(4) == 120.0
    assert default_solver_timeout_for_algorithm(99) == 240.0


def test_scale_timeout_by_grid_size():
    assert scale_timeout_by_grid_size(100.0, 0) == 100.0
    assert scale_timeout_by_grid_size(100.0, 16 * 11 * 8) == 100.0
    assert scale_timeout_by_grid_size(100.0, (16 * 11 * 8) * 2) == 200.0
    # capped at 3x
    assert scale_timeout_by_grid_size(100.0, (16 * 11 * 8) * 99) == 300.0


def test_evaluate_solver_recovery_state():
    assert evaluate_solver_recovery_state(False, False, False, 0.0, 10.0)[0] is True
    assert evaluate_solver_recovery_state(True, False, False, 0.0, 10.0)[0] is True
    assert evaluate_solver_recovery_state(True, True, False, 11.0, 10.0)[0] is True
    assert evaluate_solver_recovery_state(True, True, False, 5.0, 10.0)[0] is False

