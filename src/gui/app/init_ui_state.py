"""UI/control state initialization helpers for ZeldaGUI."""

from __future__ import annotations

from typing import Any


def initialize_ui_control_state(*, gui: Any, pygame: Any, widgets_available: bool, os_module: Any, time_module: Any) -> None:
    """Initialize item tracking, control panel, feature flags, and selection state."""
    gui.total_keys = 0
    gui.total_bombs = 0
    gui.total_boss_keys = 0
    gui.keys_collected = 0
    gui.bombs_collected = 0
    gui.boss_keys_collected = 0

    gui.toast_notifications = []
    gui.debug_overlay_enabled = False
    gui.debug_click_log = []
    gui.advanced_gui = os_module.environ.get("KLTN_ADVANCED_GUI", "0") == "1"

    gui.keys_held = {
        pygame.K_UP: False,
        pygame.K_DOWN: False,
        pygame.K_LEFT: False,
        pygame.K_RIGHT: False,
    }
    gui.move_timer = 0.0
    gui.move_delay = 0.15

    gui.show_minimap = True
    gui.minimap_size = 150
    gui.minimap_clickable = True
    gui.ai_mission_graph_editor_enabled = False
    gui.ai_mission_graph_draft = None
    gui.ai_mission_graph_layout = {}
    gui.ai_mission_graph_seed = None
    gui.ai_mission_graph_boss_node = None
    gui.ai_mission_graph_locked_edges = []
    gui.ai_mission_graph_pending_lock_source = None
    gui.ai_constraint_boss_norm = None
    gui.ai_constraint_lock_norm = None
    gui.ai_constraint_key_norm = None
    checkpoint_override = str(os_module.environ.get("KLTN_CHECKPOINT_PATH", "")).strip()
    gui.ai_checkpoint_path = checkpoint_override or None

    gui.collected_items = []
    gui.collected_positions = set()
    gui.item_type_map = {}
    gui.used_items = []
    gui.item_markers = {}
    gui.collection_effects = []
    gui.usage_effects = []

    gui.path_items_summary = {}
    gui.path_item_positions = {}

    gui.toast_notifications = []

    gui.control_panel_enabled = widgets_available
    gui.widget_manager = None
    gui.control_panel_width = 360
    gui.control_panel_width_current = float(gui.control_panel_width)
    gui.control_panel_collapsed = False
    gui.control_panel_rect = None
    gui.collapse_button_rect = None

    gui.control_panel_animating = False
    gui.control_panel_anim_start = 0.0
    gui.control_panel_anim_from = float(gui.control_panel_width)
    gui.control_panel_anim_to = float(gui.control_panel_width)
    gui.control_panel_anim_duration = 0.22
    gui.control_panel_target_collapsed = False
    gui.control_panel_x = None
    gui.control_panel_y = None
    gui.dragging_panel = False
    gui.drag_panel_offset = (0, 0)
    gui.resizing_panel = False
    gui.resize_edge = None

    gui.control_panel_scroll = 0
    gui.control_panel_scroll_step = 20
    gui.control_panel_can_scroll = False
    gui.control_panel_scroll_max = 0
    gui.control_panel_scroll_track_rect = None
    gui.control_panel_scroll_thumb_rect = None
    gui.control_panel_scroll_dragging = False
    gui.control_panel_scroll_drag_offset = 0
    gui.control_panel_content_height = 0
    gui.debug_control_panel = False

    gui.inventory_needs_refresh = False
    gui.control_panel_scroll_velocity = 0.0
    gui.control_panel_scroll_damping = 6.0
    gui.control_panel_ignore_click_until = 0.0

    gui.min_panel_width = 250
    gui.max_panel_width = 500
    gui.min_panel_height = 300

    gui.feature_flags = {
        "solver_comparison": False,
        "parallel_search": False,
        "multi_goal": False,
        "ml_heuristic": False,
        "dstar_lite": False,
        "show_heatmap": False,
        "show_topology_legend": False,
        "show_minimap": True,
        "show_path": True,
        "show_topology": False,
        "diagonal_movement": False,
        "speedrun_mode": False,
        "strict_original_mode": False,
        "dynamic_difficulty": False,
        "force_grid": False,
        "enable_prechecks": False,
        "auto_prune_on_precheck": False,
        "priority_tie_break": False,
        "priority_key_boost": False,
        "enable_ara": False,
        "use_jps": False,
        "show_jps_overlay": False,
        "show_map_elites": False,
    }
    gui.force_grid_algorithm = False

    gui.current_floor = 1
    gui.zoom_level_idx = 3
    gui.difficulty_idx = 1
    gui.algorithm_idx = 0
    gui.search_representation = "hybrid"
    gui.ara_weight = 1.0

    gui.message = "Press SPACE to auto-solve, Arrow keys to move"
    gui.message_time = time_module.time()
    gui.message_duration = 3.0

    gui.preview_on_next_solver_result = False
    gui.auto_start_preview = os_module.environ.get("KLTN_AUTO_START_PREVIEW", "0") == "1"
    gui.auto_start_solver = os_module.environ.get("KLTN_AUTO_START_SOLVER", "1") != "0"
