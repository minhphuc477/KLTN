"""Pure control-panel rules used by ZeldaGUI event handling."""

from src.gui.common.constants import (
    GUI_ALGORITHM_NAMES,
    GUI_DIFFICULTY_NAMES,
    GUI_ZOOM_LABELS,
)

ZOOM_DROPDOWN_TO_LEVEL_INDEX = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 4}
REPRESENTATION_BY_INDEX = {0: "hybrid", 1: "tile", 2: "graph"}


def zoom_level_index_from_dropdown(dropdown_index: int):
    """Map zoom dropdown selection to ZeldaGUI.ZOOM_LEVELS index."""
    return ZOOM_DROPDOWN_TO_LEVEL_INDEX.get(dropdown_index)


def zoom_label(dropdown_index: int) -> str:
    """Return user-facing zoom label for dropdown selection."""
    if 0 <= dropdown_index < len(GUI_ZOOM_LABELS):
        return GUI_ZOOM_LABELS[dropdown_index]
    return f"{dropdown_index}"


def difficulty_label(index: int) -> str:
    """Return user-facing difficulty name."""
    if 0 <= index < len(GUI_DIFFICULTY_NAMES):
        return GUI_DIFFICULTY_NAMES[index]
    return f"Difficulty {index}"


def algorithm_label(index: int) -> str:
    """Return user-facing solver name."""
    if 0 <= index < len(GUI_ALGORITHM_NAMES):
        return GUI_ALGORITHM_NAMES[index]
    return f"Algorithm {index}"


def representation_from_dropdown(selected_index: int, current: str = "hybrid") -> str:
    """Translate representation dropdown selection into internal mode."""
    return REPRESENTATION_BY_INDEX.get(selected_index, current)


def apply_preset_feature_flags(feature_flags: dict, preset_name: str) -> None:
    """Apply preset mutations to existing feature flag dict in place."""
    if preset_name == "Debugging":
        feature_flags["show_heatmap"] = True
        feature_flags["solver_comparison"] = False
        feature_flags["ml_heuristic"] = False
    elif preset_name == "Fast Approx":
        feature_flags["ml_heuristic"] = True
        feature_flags["parallel_search"] = True
    elif preset_name == "Optimal":
        feature_flags["ml_heuristic"] = False
        feature_flags["parallel_search"] = False
    elif preset_name == "Speedrun":
        feature_flags["speedrun_mode"] = True
        feature_flags["ml_heuristic"] = True

