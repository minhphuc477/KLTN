from src.gui.controls.control_panel_logic import (
    algorithm_label,
    apply_preset_feature_flags,
    difficulty_label,
    representation_from_dropdown,
    zoom_label,
    zoom_level_index_from_dropdown,
)


def test_zoom_dropdown_mapping():
    assert zoom_level_index_from_dropdown(0) == 0
    assert zoom_level_index_from_dropdown(2) == 1
    assert zoom_level_index_from_dropdown(5) == 4
    assert zoom_level_index_from_dropdown(99) is None


def test_labels_and_fallbacks():
    assert zoom_label(3) == "100%"
    assert difficulty_label(2) == "Hard"
    assert algorithm_label(0) == "A*"
    assert difficulty_label(99) == "Difficulty 99"
    assert algorithm_label(99) == "Algorithm 99"


def test_representation_dropdown_mapping():
    assert representation_from_dropdown(0) == "hybrid"
    assert representation_from_dropdown(1) == "tile"
    assert representation_from_dropdown(2) == "graph"
    assert representation_from_dropdown(77, current="hybrid") == "hybrid"


def test_apply_preset_feature_flags():
    flags = {
        "show_heatmap": False,
        "solver_comparison": True,
        "ml_heuristic": False,
        "parallel_search": False,
        "speedrun_mode": False,
    }

    apply_preset_feature_flags(flags, "Debugging")
    assert flags["show_heatmap"] is True
    assert flags["solver_comparison"] is False
    assert flags["ml_heuristic"] is False

    apply_preset_feature_flags(flags, "Fast Approx")
    assert flags["ml_heuristic"] is True
    assert flags["parallel_search"] is True

    apply_preset_feature_flags(flags, "Optimal")
    assert flags["ml_heuristic"] is False
    assert flags["parallel_search"] is False

    apply_preset_feature_flags(flags, "Speedrun")
    assert flags["speedrun_mode"] is True
    assert flags["ml_heuristic"] is True

