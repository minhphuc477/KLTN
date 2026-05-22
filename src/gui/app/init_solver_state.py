"""Solver and execution state initialization helpers for ZeldaGUI."""

from __future__ import annotations

from typing import Any


def initialize_solver_execution_state(*, gui: Any, threading_module: Any) -> None:
    """Initialize solver, preview, and algorithm execution state."""
    gui.match_undo_stack = []
    gui.show_heatmap = False
    gui.search_heatmap = {}

    gui.env = None
    gui.solver = None
    gui.auto_path = []
    gui.auto_step_idx = 0
    gui.auto_mode = False
    gui.auto_step_timer = 0.0
    gui.auto_step_interval = 0.15

    gui.error_message = None
    gui.error_time = 0
    gui.status_message = "Ready"
    gui.show_help = False

    gui.solver_result = None
    gui.current_keys_held = 0
    gui.current_keys_used = 0
    gui.current_edge_types = []
    gui.door_unlock_times = {}

    gui.path_preview_dialog = None
    gui.path_preview_mode = False
    gui.preview_modal_enabled = False
    gui.preview_overlay_visible = False

    gui.show_topology = False
    gui.topology_export_path = None
    gui.show_topology_legend = False
    gui.topology_semantics = {
        "nodes": {
            "e": ["room", "enemy"],
            "S": ["room", "switch"],
            "b": ["room", "boss"],
            "k": ["room", "key"],
            "K": ["room", "boss key"],
            "I": ["room", "key item"],
            "p": ["room", "puzzle"],
            "s": ["room", "start"],
            "t": ["room", "triforce"],
        },
        "edges": {
            "S": ["door", "switch locked"],
            "b": ["door", "bombable"],
            "k": ["door", "key locked"],
            "K": ["door", "boss key locked"],
            "I": ["door", "key item locked"],
            "l": ["door", "soft locked"],
            "s": ["visible", "impassable"],
        },
    }

    gui.last_solver_metrics = None
    gui.solver_comparison_results = None
    gui.show_solver_comparison_overlay = False

    gui.solver_running = False
    gui.solver_proc = None
    gui.solver_done = True
    gui.solver_outfile = None
    gui.solver_gridfile = None
    gui.solver_thread = None
    gui._pending_solver_trigger = False
    gui._solver_lock = threading_module.Lock()

    gui.preview_proc = None
    gui.preview_outfile = None
    gui.preview_gridfile = None
    gui.preview_done = True
    gui.preview_result = None
    gui.preview_thread = None

    gui.presets = ["Debugging", "Fast Approx", "Optimal", "Speedrun"]
    gui.current_preset_idx = 0

    gui.dstar_solver = None
    gui.dstar_active = False

    gui.parallel_search_thread = None
    gui.parallel_search_done = False
    gui.parallel_search_result = None

    gui.ai_gen_thread = None
    gui.ai_gen_done = False
    gui.ai_gen_result = None

    gui._precheck_snapshot = None

    gui.agent_visual_pos = None
    gui.agent_target_pos = None

    gui.block_push_animations = []
    gui.block_push_duration = 200

    gui.speed_levels = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    gui.speed_index = 2
    gui.speed_multiplier = gui.speed_levels[gui.speed_index]

    gui.step_count = 0
    gui.item_pickup_times = {}
