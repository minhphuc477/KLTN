"""
GUI Runner for ZAVE (Zelda AI Validation Environment)
====================================================

Interactive visual interface for validating Zelda dungeon maps.

Features:
- Real-time visualization of map and agent
- Manual play mode (arrow keys)
- Auto-solve mode (A* pathfinding)
- Map loading from processed data
- Smooth delta-time animations
- Heatmap overlay for A* search visualization
- Modern semi-transparent HUD

Controls:
- Arrow Keys: Move Link
- SPACE: Run A* solver (auto-solve)
- R: Reset map
- N: Next map (if multiple loaded)
- P: Previous map
- H: Toggle heatmap overlay
- ESC: Quit


"""

import sys
import os
import time
import math
import copy
import logging
import threading
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Any

from src.gui.runtime.flags import load_runtime_flags
from src.gui.app.main_loop_utils import (
    compute_solver_timeout_seconds,
    find_path_tile_violations,
    resolve_test_mode_max_frames,
    run_auto_step_tick,
    run_continuous_movement_tick,
    should_attempt_focus_fallback,
)
from src.gui.app.event_loop_handlers import (
    clear_stale_preview_overlay,
    handle_global_keydown_shortcuts,
    handle_mouse_button_down_preamble,
    handle_mouse_button_up_event,
    handle_mouse_motion_diagnostics,
    handle_preview_overlay_events,
    handle_window_focus_event,
    poll_pygame_events,
    run_input_focus_fallback,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
runtime_flags = load_runtime_flags()
# Allow debug mode via env var KLTN_LOG_LEVEL=DEBUG for interactive troubleshooting
if runtime_flags.log_level == 'DEBUG':
    logger.setLevel(logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)

# Allow targeted input diagnostics via env var KLTN_DEBUG_INPUT=1
DEBUG_INPUT_ACTIVE = runtime_flags.debug_input_active
if DEBUG_INPUT_ACTIVE:
    logger.info('INPUT_DIAG: KLTN_DEBUG_INPUT is active (diagnostic input dumps enabled)')

# DEBUG: Synchronous solver mode to bypass multiprocessing issues
# Set KLTN_SYNC_SOLVER=1 only for debugging (will block UI during solving)
# ASYNC mode by default for responsive UI during long solves
DEBUG_SYNC_SOLVER = runtime_flags.debug_sync_solver
if DEBUG_SYNC_SOLVER:
    logger.info('Solver running in SYNC mode - UI will freeze during solving (direct execution, no pickle files)')
else:
    logger.info('Solver running in ASYNC mode - animated solving enabled')

# DEBUG: Verbose solver flow logging
DEBUG_SOLVER_FLOW = runtime_flags.debug_solver_flow
if DEBUG_SOLVER_FLOW:
    logger.setLevel(logging.DEBUG)
    logger.warning('DEBUG: KLTN_DEBUG_SOLVER_FLOW=1 - Verbose solver logging enabled')

# Import simulation components (use new canonical paths)
from src.simulation.validator import (
    ZeldaLogicEnv, 
    ZeldaValidator, 
    StateSpaceAStar,
    SanityChecker,
    create_test_map,
    SEMANTIC_PALETTE,
    Action,
    GameState,
    ACTION_DELTAS,
    PUSHABLE_IDS,
    WALKABLE_IDS
)

# Local matcher/adapters for topology repair and precheck pruning (use canonical path)
from src.data.zelda_core import RoomGraphMatcher, ZeldaDungeonAdapter
from src.gui.components.constants import (
    GUI_ALGORITHM_NAMES,
    GUI_DIFFICULTY_NAMES,
    GUI_PRESETS,
    GUI_ZOOM_LABELS,
)
from src.gui.control_panel.logic import (
    algorithm_label,
)
from src.gui.control_panel.interactions import (
    control_panel_hit_rect as _control_panel_hit_rect_helper,
    handle_outside_control_panel_click as _handle_outside_control_panel_click_helper,
    refresh_control_panel_layout_if_needed as _refresh_control_panel_layout_if_needed_helper,
    retry_control_panel_click_after_auto_scroll as _retry_control_panel_click_after_auto_scroll_helper,
    should_swallow_control_panel_click as _should_swallow_control_panel_click_helper,
    translate_control_panel_click as _translate_control_panel_click_helper,
)
from src.gui.control_panel.updates import (
    apply_algorithm_dropdown_update as _apply_algorithm_dropdown_update_helper,
    apply_checkbox_widget_update as _apply_checkbox_widget_update_helper,
    apply_control_panel_widget_updates as _apply_control_panel_widget_updates_helper,
    apply_dropdown_widget_update as _apply_dropdown_widget_update_helper,
)
from src.gui.solver.start_logic import (
    default_solver_timeout_for_algorithm,
    evaluate_solver_recovery_state,
    scale_timeout_by_grid_size,
    sync_solver_dropdown_settings,
)
from src.gui.solver.request_helpers import (
    build_solver_request as _build_solver_request_helper,
    get_solver_map_context as _get_solver_map_context_helper,
)
from src.gui.solver.launching import (
    create_solver_temp_files as _create_solver_temp_files_helper,
    launch_solver_process as _launch_solver_process_helper,
    solver_thread_fallback_worker as _solver_thread_fallback_worker_helper,
    start_solver_thread_fallback as _start_solver_thread_fallback_helper,
)
from src.gui.solver.scheduling import schedule_solver as _schedule_solver_helper
from src.gui.gameplay.preview_startup import start_preview_for_current_map as _start_preview_for_current_map_helper
from src.gui.gameplay.auto_solve_execution import (
    execute_auto_solve as _execute_auto_solve_helper,
    execute_auto_solve_from_preview as _execute_auto_solve_from_preview_helper,
)
from src.gui.solver.recovery import (
    compute_solver_timeout_seconds as _compute_solver_timeout_seconds_helper,
    force_solver_recovery_state as _force_solver_recovery_state_helper,
    log_active_solver_state as _log_active_solver_state_helper,
    prepare_active_solver_for_new_start as _prepare_active_solver_for_new_start_helper,
    terminate_hung_solver_process as _terminate_hung_solver_process_helper,
)
from src.gui.solver.prestart_cleanup import (
    cleanup_preview_before_solver_start as _cleanup_preview_before_solver_start_helper,
    reset_solver_visual_state_before_start as _reset_solver_visual_state_before_start_helper,
)
from src.gui.solver.core_state import (
    clear_solver_state as _clear_solver_state_helper,
    sync_solver_dropdown_settings as _sync_solver_dropdown_settings_helper,
)
from src.gui.solver.worker_bootstrap import launch_solver_worker as _launch_solver_worker_helper
from src.gui.solver.start_flow import start_auto_solve as _start_auto_solve_helper
from src.gui.solver.sync_execution import run_solver_sync as _run_solver_sync_helper
from src.gui.solver.request_orchestration import (
    build_solver_request as _build_solver_request_orchestration_helper,
    get_solver_map_context as _get_solver_map_context_orchestration_helper,
)
from src.gui.runtime.watchdog_monitor import watchdog_loop as _watchdog_loop_helper
from src.gui.runtime.route_io import (
    export_route as _export_route_helper,
    load_route as _load_route_helper,
)
from src.gui.gameplay.path_controls import (
    reset_map as _reset_map_helper,
    show_path_preview as _show_path_preview_helper,
    clear_path as _clear_path_helper,
)
from src.gui.runtime.temp_file_management import (
    open_temp_folder as _open_temp_folder_orchestration_helper,
    collect_temp_file_candidates as _collect_temp_file_candidates_orchestration_helper,
    delete_temp_files as _delete_temp_files_orchestration_helper,
)
from src.gui.topology.export import export_topology as _export_topology_helper
from src.gui.runtime.toast_messages import (
    set_message as _set_message_helper,
    show_toast as _show_toast_helper,
    update_toasts as _update_toasts_helper,
    render_toasts as _render_toasts_helper,
)
from src.gui.map.minimap import (
    render_minimap as _render_minimap_helper,
    handle_minimap_click as _handle_minimap_click_helper,
)
from src.gui.map.navigation import (
    next_map as _next_map_helper,
    prev_map as _prev_map_helper,
    clamp_view_offset as _clamp_view_offset_helper,
    center_on_player as _center_on_player_helper,
)
from src.gui.gameplay.block_push_controls import (
    start_block_push_animation as _start_block_push_animation_helper,
    update_block_push_animations as _update_block_push_animations_helper,
    render_block_push_animations as _render_block_push_animations_helper,
    get_animating_block_positions as _get_animating_block_positions_helper,
    check_and_start_block_push as _check_and_start_block_push_helper,
)
from src.gui.rendering.help_overlay import render_help_overlay as _render_help_overlay_helper
from src.gui.rendering.helpers import (
    render_topology_overlay as _render_topology_overlay_helper,
    render_solver_comparison_overlay as _render_solver_comparison_overlay_helper,
)
from src.gui.topology.helpers import (
    room_for_global_position as _room_for_global_position_helper,
    node_has_small_key as _node_has_small_key_helper,
    node_has_critical_content as _node_has_critical_content_helper,
    capture_precheck_snapshot as _capture_precheck_snapshot_helper,
    update_env_topology_view as _update_env_topology_view_helper,
    build_room_adjacency_from_graph as _build_room_adjacency_from_graph_helper,
    topology_has_path as _topology_has_path_helper,
    min_locked_between as _min_locked_between_helper,
    walkable_grid_reachable as _walkable_grid_reachable_helper,
)
from src.gui.topology.precheck import (
    prune_dead_end_topology as _prune_dead_end_topology_flow_helper,
    run_prechecks_and_optional_prune as _run_prechecks_and_optional_prune_flow_helper,
    undo_prune as _undo_prune_flow_helper,
)
from src.gui.rendering.status_display import (
    render_error_banner as _render_error_banner_helper,
    render_solver_status_banner as _render_solver_status_banner_helper,
    render_status_bar as _render_status_bar_helper,
    show_error as _show_error_helper,
    show_message as _show_message_helper,
    show_warning as _show_warning_helper,
)
from src.gui.rendering.bottom_panel import (
    render_unified_bottom_panel as _render_unified_bottom_panel_helper,
    render_message_section as _render_message_section_helper,
    render_progress_bar as _render_progress_bar_helper,
    render_inventory_section as _render_inventory_section_helper,
    render_metrics_section as _render_metrics_section_helper,
    render_controls_section as _render_controls_section_helper,
    render_status_section as _render_status_section_helper,
)
from src.gui.rendering.debug_overlay import render_debug_overlay as _render_debug_overlay_helper
from src.gui.rendering.widget_tooltips import (
    render_tooltips as _render_tooltips_helper,
    draw_tooltip as _draw_tooltip_helper,
)
from src.gui.solver.metrics_tooltips import format_cbs_metrics_tooltip as _format_cbs_metrics_tooltip_helper
from src.gui.map.viewport import (
    center_view as _center_view_helper,
    auto_fit_zoom as _auto_fit_zoom_helper,
    change_zoom as _change_zoom_helper,
)
from src.gui.runtime.display_lifecycle import (
    safe_set_mode as _safe_set_mode_helper,
    attempt_display_reinit as _attempt_display_reinit_helper,
    ensure_display_alive as _ensure_display_alive_helper,
)
from src.gui.runtime.display_diagnostics import (
    handle_watchdog_screenshot as _handle_watchdog_screenshot_helper,
    report_ui_state as _report_ui_state_helper,
)
from src.gui.runtime.window_focus import (
    force_focus as _force_focus_helper,
    toggle_fullscreen as _toggle_fullscreen_helper,
)
from src.gui.control_panel.animation import (
    start_toggle_panel_animation as _start_toggle_panel_animation_helper,
    update_control_panel_animation as _update_control_panel_animation_helper,
)
from src.gui.control_panel.scroll import update_control_panel_scroll as _update_control_panel_scroll_helper
from src.gui.control_panel.view import (
    dump_control_panel_widget_state as _dump_control_panel_widget_state_helper,
    render_control_panel as _render_control_panel_helper,
    reposition_widgets as _reposition_widgets_helper,
    update_control_panel_positions as _update_control_panel_positions_helper,
)
from src.gui.gameplay.inventory_manager import (
    update_inventory_and_hud as _update_inventory_and_hud_helper,
    remove_from_path_items as _remove_from_path_items_helper,
    track_item_collection as _track_item_collection_helper,
    track_item_usage as _track_item_usage_helper,
    sync_inventory_counters as _sync_inventory_counters_helper,
)
from src.gui.gameplay.path_analysis import scan_items_along_path as _scan_items_along_path_helper
from src.gui.rendering.inventory_display import (
    get_path_items_display_text as _get_path_items_display_text_helper,
    render_item_legend as _render_item_legend_helper,
)
from src.gui.gameplay.item_markers import (
    scan_and_mark_items as _scan_and_mark_items_helper,
    apply_pickup_at as _apply_pickup_at_helper,
)
from src.gui.gameplay.path_strategies import (
    smart_grid_path as _smart_grid_path_helper,
    graph_guided_path as _graph_guided_path_helper,
    hybrid_graph_grid_path as _hybrid_graph_grid_path_helper,
)
from src.gui.gameplay.auto_step_controller import (
    stop_auto as _stop_auto_helper,
    auto_step as _auto_step_helper,
)
from src.gui.gameplay.manual_step_controller import manual_step as _manual_step_flow_helper
from src.gui.rendering.path_guaranteed_renderer import (
    render_path_guaranteed as _render_path_guaranteed_flow_helper,
)
from src.gui.gameplay.map_elites_controls import (
    start_map_elites as _start_map_elites_flow_helper,
    map_elites_worker as _map_elites_worker_flow_helper,
)
from src.gui.rendering.map_overlays import (
    log_draw_ranges as _log_draw_ranges_overlay_helper,
    render_empty_range_warning as _render_empty_range_warning_overlay_helper,
    render_jps_overlay as _render_jps_overlay_helper,
    render_map_elites_overlay as _render_map_elites_overlay_helper,
)
from src.gui.rendering.sidebar_sections import (
    render_sidebar_header_inventory_solver as _render_sidebar_header_inventory_solver_helper,
    render_sidebar_status_message_metrics_controls as _render_sidebar_status_message_metrics_controls_helper,
)
from src.gui.topology.match_controls import (
    match_missing_nodes as _match_missing_nodes_helper,
    undo_last_match as _undo_last_match_helper,
    apply_tentative_matches as _apply_tentative_matches_helper,
)
from src.gui.solver.comparison_runner import (
    run_solver_comparison as _run_solver_comparison_helper,
    set_last_solver_metrics as _set_last_solver_metrics_helper,
)
from src.gui.solver.utils import (
    safe_unpickle as _safe_unpickle_helper,
    convert_diagonal_to_4dir as _convert_diagonal_to_4dir_helper,
)
from src.gui.solver.process_worker import (
    _solve_in_subprocess as _solve_in_subprocess_helper,
    _run_solver_and_dump as _run_solver_and_dump_helper,
    _run_preview_and_dump as _run_preview_and_dump_helper,
)
from src.gui.ai.generation_controls import (
    start_ai_dungeon_generation as _start_ai_dungeon_generation_helper,
)
from src.gui.ai.generation_worker import (
    run_ai_generation_worker as _run_ai_generation_worker_helper,
)
from src.gui.map.loading import (
    load_current_map as _load_current_map_helper,
    load_visual_assets as _load_visual_assets_helper,
    load_visual_map as _load_visual_map_helper,
    place_items_from_graph as _place_items_from_graph_helper,
)
from src.gui.runtime.temp_file_tools import (
    delete_files as _delete_files_helper,
    find_temp_files as _find_temp_files_helper,
    list_existing_paths as _list_existing_paths_helper,
    open_folder as _open_folder_helper,
)
from src.gui.components.fallbacks import get_visualization_fallbacks, get_widget_fallbacks
from src.gui.runtime.toast_notification import ToastNotification

# Try to import Pygame
# NOTE: Importing pygame does NOT create a window - windows are only created
# when pygame.display.set_mode() is called. The ZeldaGUI class is only
# instantiated in main(), which is protected by if __name__ == "__main__".
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    pygame = None  # type: ignore[assignment]
    PYGAME_AVAILABLE = False
    logger.warning("Pygame not installed. Run 'pip install pygame' for GUI support.")

# Try to import new visualization system
try:
    from src.visualization.renderer import ZeldaRenderer, ThemeConfig, Vector2
    from src.visualization.effects import (
        EffectManager, PopEffect, FlashEffect, RippleEffect,
        ItemCollectionEffect, ItemUsageEffect, ItemMarkerEffect
    )
    from src.visualization.hud import ModernHUD, HUDTheme
    from src.visualization.path_preview import PathPreviewDialog
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

    _visual_fallbacks = get_visualization_fallbacks(
        pygame_available=PYGAME_AVAILABLE,
        pygame_module=pygame,
    )
    ZeldaRenderer = _visual_fallbacks["ZeldaRenderer"]
    ThemeConfig = _visual_fallbacks["ThemeConfig"]
    Vector2 = _visual_fallbacks["Vector2"]
    EffectManager = _visual_fallbacks["EffectManager"]
    PopEffect = _visual_fallbacks["PopEffect"]
    FlashEffect = _visual_fallbacks["FlashEffect"]
    RippleEffect = _visual_fallbacks["RippleEffect"]
    ItemCollectionEffect = _visual_fallbacks["ItemCollectionEffect"]
    ItemUsageEffect = _visual_fallbacks["ItemUsageEffect"]
    ItemMarkerEffect = _visual_fallbacks["ItemMarkerEffect"]
    ModernHUD = _visual_fallbacks["ModernHUD"]
    HUDTheme = _visual_fallbacks["HUDTheme"]
    PathPreviewDialog = _visual_fallbacks["PathPreviewDialog"]

    logger.warning("New visualization system not available; using no-op fallbacks for GUI components.")

# Try to import GUI widgets
try:
    from src.gui.components.widgets import (
        CheckboxWidget, DropdownWidget, ButtonWidget,
        WidgetManager, WidgetTheme
    )
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False

    _widget_fallbacks = get_widget_fallbacks()
    CheckboxWidget = _widget_fallbacks["CheckboxWidget"]
    DropdownWidget = _widget_fallbacks["DropdownWidget"]
    ButtonWidget = _widget_fallbacks["ButtonWidget"]
    WidgetManager = _widget_fallbacks["WidgetManager"]
    WidgetTheme = _widget_fallbacks["WidgetTheme"]

    logger.warning("GUI widgets not available Î“Ã‡Ã¶ using no-op widget manager.")

# --- Subprocess-based solver helper ---
# This helper runs inside a separate process to avoid blocking the main thread
# with heavy CPU-bound pathfinding work (which would starve the GUI due to the GIL).
import pickle
import tempfile
import multiprocessing


def _safe_unpickle(path: str) -> dict:
    """Safely load a pickle produced by our own processes and validate shape.

    Returns a dict with at least a 'success' key. Any error returns a failure dict.
    """
    return _safe_unpickle_helper(path)


def _convert_diagonal_to_4dir(path, grid=None):
    """Convert a path with diagonal moves to 4-directional movement.
    
    Each diagonal move (e.g., NE) is split into two orthogonal moves.
    This preserves pathfinding speed while showing standard grid-based animation.
    
    CRITICAL FIX: When grid is provided, we validate intermediate positions
    to avoid routing through water/walls. We try vertical-first, then
    horizontal-first, and pick whichever doesn't go through obstacles.
    
    Args:
        path: List of (row, col) tuples
        grid: Optional numpy array of tile IDs - used to validate intermediate positions
    
    Returns:
        List of (row, col) tuples with only orthogonal (4-dir) moves
    """
    return _convert_diagonal_to_4dir_helper(path, grid=grid)

def _solve_in_subprocess(grid, start_pos, goal_pos, algorithm_idx, feature_flags, priority_options,
                         graph=None, room_to_node=None, room_positions=None, node_to_room=None):
    """Compute a path for a grid in a separate process and return a picklable dict.

    Arguments: 
        grid: may be an ndarray-like or nested lists
        graph: Optional NetworkX DiGraph for room connectivity (enables stair traversal)
        room_to_node: Optional mapping of room positions to graph nodes
        room_positions: Optional mapping of room positions to pixel offsets
        node_to_room: Optional mapping of graph nodes to room positions (includes virtual nodes)
    
    The function re-creates a ZeldaLogicEnv locally inside the child process and runs 
    the same solver logic used on the main thread.
    """
    return _solve_in_subprocess_helper(
        grid,
        start_pos,
        goal_pos,
        algorithm_idx,
        feature_flags,
        priority_options,
        graph=graph,
        room_to_node=room_to_node,
        room_positions=room_positions,
        node_to_room=node_to_room,
    )


def _run_solver_and_dump(grid_or_path, start_pos, goal_pos, algorithm_idx, feature_flags, priority_options, out_path,
                         graph=None, room_to_node=None, room_positions=None, node_to_room=None):
    """Top-level helper to run solver and pickle the result to disk.

    This must be module-level so it is picklable by multiprocessing on Windows.
    `grid_or_path` may be a nested list (legacy) or a filesystem path to a .npy file.
    
    Args:
        graph: Optional NetworkX DiGraph for room connectivity
        room_to_node: Optional mapping of room positions to graph nodes
        room_positions: Optional mapping of room positions to pixel offsets
        node_to_room: Optional mapping of graph nodes to room positions (includes virtual nodes)
    """
    return _run_solver_and_dump_helper(
        grid_or_path,
        start_pos,
        goal_pos,
        algorithm_idx,
        feature_flags,
        priority_options,
        out_path,
        graph=graph,
        room_to_node=room_to_node,
        room_positions=room_positions,
        node_to_room=node_to_room,
    )


def _run_preview_and_dump(grid_or_path, start_pos, goal_pos, algorithm_idx, feature_flags, priority_options, out_path,
                          graph=None, room_to_node=None, room_positions=None, node_to_room=None):
    """Lightweight preview runner that writes a short preview result quickly.

    Runs in a separate process to avoid blocking the GUI. Attempts a fast StateSpaceAStar
    with a small timeout or returns failure quickly.
    """
    return _run_preview_and_dump_helper(
        grid_or_path,
        start_pos,
        goal_pos,
        algorithm_idx,
        feature_flags,
        priority_options,
        out_path,
        graph=graph,
        room_to_node=room_to_node,
        room_positions=room_positions,
        node_to_room=node_to_room,
    )



class ZeldaGUI:
    """
    Interactive GUI for Zelda dungeon validation.
    
    Features:
    - Resizable window (drag corners/edges)
    - Zoom in/out with +/- keys or mouse wheel
    - Pan with middle mouse or WASD when zoomed
    - Fullscreen toggle with F11
    - Smooth delta-time based animations
    - Heatmap overlay for A* search (toggle with H)
    """
    
    # Zoom levels available
    ZOOM_LEVELS = [16, 24, 32, 48, 64]
    DEFAULT_ZOOM_IDX = 2  # 32px default
    
    # Minimum window size
    MIN_WIDTH = 400
    MIN_HEIGHT = 300
    
    def __init__(self, maps: list = None, map_names: list = None):
        """
        Initialize GUI.
        
        Args:
            maps: List of semantic grids to visualize
            map_names: List of names for each map
        """
        if not PYGAME_AVAILABLE:
            raise ImportError("Pygame is required for GUI. Install with: pip install pygame")
        # Type-narrowing for static analysis: ensure `pygame` is available below
        assert pygame is not None
        
        self.maps: List[Any] = maps if maps else [create_test_map()]
        self.map_names = map_names if map_names else [f"Map {i+1}" for i in range(len(self.maps))]
        self.current_map_idx = 0

        # Repository-local export directories (avoid writing outside the repo due to cwd changes).
        self.repo_root = Path(__file__).resolve().parent
        self.exports_root = self.repo_root / 'exports'
        self.route_export_dir = self.exports_root / 'routes'
        self.topology_export_dir = self.exports_root / 'topology'
        self.artifacts_dir = str(self.exports_root / 'artifacts')
        try:
            self.route_export_dir.mkdir(parents=True, exist_ok=True)
            self.topology_export_dir.mkdir(parents=True, exist_ok=True)
            Path(self.artifacts_dir).mkdir(parents=True, exist_ok=True)
        except Exception:
            logger.exception('Failed to create export directories under repo root')
        
        # Attempt to enable Windows DPI awareness *before* initializing Pygame so mouse coords match pixels
        try:
            import ctypes
            # Prefer the per-monitor v2 context if available (Windows 10+)
            try:
                DPI_AWARE_CONTEXT_PER_MONITOR_AWARE_V2 = -4
                ctypes.windll.user32.SetProcessDpiAwarenessContext(DPI_AWARE_CONTEXT_PER_MONITOR_AWARE_V2)
                logger.debug('SetProcessDpiAwarenessContext(PER_MONITOR_AWARE_V2) succeeded')
            except Exception:
                try:
                    # Try SetProcessDpiAwareness (Windows 8.1+ via shcore)
                    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
                    logger.debug('SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE) succeeded')
                except Exception:
                    try:
                        # Fallback to legacy API
                        ctypes.windll.user32.SetProcessDPIAware()
                        logger.debug('SetProcessDPIAware() succeeded')
                    except Exception:
                        logger.debug('Could not set process DPI awareness')
        except Exception:
            logger.debug('DPI awareness calls not supported on this platform')

        # Initialize Pygame
        try:
            pygame.init()
        except Exception:
            logger.exception("Failed to initialize Pygame")
            raise

        # Wrap pygame.mouse.set_cursor to be tolerant on platforms where system cursors are unsupported
        try:
            _orig_set_cursor = pygame.mouse.set_cursor
            def _wrapped_set_cursor(cursor):
                try:
                    _orig_set_cursor(cursor)
                except Exception:
                    logger.debug('set_cursor failed or unsupported in this environment', exc_info=True)
            pygame.mouse.set_cursor = _wrapped_set_cursor
        except Exception:
            # If we can't patch, just ignore - cursor changes may still work
            logger.debug('Could not wrap pygame.mouse.set_cursor; continuing')
        
        # Display settings
        self.zoom_idx = self.DEFAULT_ZOOM_IDX
        self.TILE_SIZE = self.ZOOM_LEVELS[self.zoom_idx]
        self.HUD_HEIGHT = 10  # Minimal bottom margin (status/message moved to sidebar)
        self.SIDEBAR_WIDTH = 220  # Wider for dungeon names
        
        # Get screen info for smart sizing
        display_info = pygame.display.Info()
        max_screen_w = display_info.current_w - 100
        max_screen_h = display_info.current_h - 100
        
        # Calculate initial window size (fit largest map)
        # Handle both raw grids and StitchedDungeon objects
        def _grid_shape(m: Any):
            g = getattr(m, 'global_grid', m)
            return getattr(g, 'shape')[0], getattr(g, 'shape')[1]
        max_map_h = max(_grid_shape(m)[0] for m in self.maps)
        max_map_w = max(_grid_shape(m)[1] for m in self.maps)
        
        # Smart sizing: fit map with some padding, but don't exceed screen
        ideal_w = max_map_w * self.TILE_SIZE + self.SIDEBAR_WIDTH
        ideal_h = max_map_h * self.TILE_SIZE + self.HUD_HEIGHT
        
        self.screen_w = min(ideal_w, max_screen_w)
        self.screen_h = min(ideal_h, max_screen_h)
        
        # Ensure minimum size
        self.screen_w = max(self.screen_w, self.MIN_WIDTH)
        self.screen_h = max(self.screen_h, self.MIN_HEIGHT)
        
        # Create resizable window
        self.screen = pygame.display.set_mode(
            (self.screen_w, self.screen_h), 
            pygame.RESIZABLE
        )
        pygame.display.set_caption("ZAVE: Zelda AI Validation Environment")
        # Remember previous window size so we can restore it after exiting fullscreen
        self._prev_window_size = (self.screen_w, self.screen_h)

        # Ensure mouse events are not grabbed and cursor is visible on startup
        try:
            pygame.event.set_grab(False)
        except Exception:
            logger.debug('Could not clear event grab at startup')
        try:
            pygame.mouse.set_visible(True)
        except Exception:
            logger.debug('Could not ensure mouse cursor visible at startup')

        # Try to raise and focus the window on Windows so clicks are accepted
        try:
            if os.name == 'nt':
                try:
                    # Prefer ctypes to avoid adding pywin32 dependency
                    import ctypes
                    user32 = ctypes.windll.user32
                    hwnd = pygame.display.get_wm_info().get('window')
                    if hwnd:
                        logger.debug('Attempting to bring window to foreground (hwnd=%s)', hwnd)
                        SW_SHOW = 5
                        user32.ShowWindow(hwnd, SW_SHOW)
                        user32.SetForegroundWindow(hwnd)
                        pygame.event.pump()
                        pygame.mouse.set_visible(True)
                        pygame.event.set_grab(False)
                        logger.debug('Set focus to window via Win32 API')
                except Exception:
                    logger.debug('Windows focus helper failed', exc_info=True)
        except Exception:
            logger.debug('Focus bring-to-front helper encountered an error', exc_info=True)

        # Track last ungrab attempt to avoid spamming
        self._last_ungrab_attempt = 0.0
        
        # Display health & recovery settings
        self._display_check_interval = float(os.environ.get('KLTN_DISPLAY_CHECK_INTERVAL', '1.0'))
        self._display_check_last = time.time()
        self._display_recovery_attempts = 0
        self._display_recovery_attempts_limit = int(os.environ.get('KLTN_DISPLAY_RECOVER_LIMIT', '3'))

        # View offset for panning
        self.view_offset_x = 0
        self.view_offset_y = 0
        self.dragging = False
        self.drag_start = (0, 0)
        
        # Fullscreen state
        self.fullscreen = False
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 14, bold=True)
        self.big_font = pygame.font.SysFont('Arial', 20, bold=True)
        self.small_font = pygame.font.SysFont('Arial', 12)
        
        # Debug helpers for control panel visualisation and hit-padding
        # Can be enabled via env KLTN_DEBUG_CONTROL_PANEL=1 or toggled at runtime (F8)
        self.debug_control_panel = os.environ.get('KLTN_DEBUG_CONTROL_PANEL', '0') == '1'
        self.debug_panel_click_padding = int(os.environ.get('KLTN_DEBUG_PANEL_PADDING', '40')) if self.debug_control_panel else 0

        # Delta-time tracking for smooth animations
        self.last_frame_time = time.time()
        self.delta_time = 0.0
        # Display health check timing (throttled to avoid per-frame work)
        self._display_check_last = 0.0
        self._display_check_interval = float(os.environ.get('KLTN_DISPLAY_CHECK_INTERVAL', '1.0'))

        # Start a watchdog thread to detect UI freezes and dump stack/screenshot for debugging
        # Disabled by default - enable via KLTN_ENABLE_WATCHDOG=1 for troubleshooting
        try:
            import faulthandler
            self._watchdog_enabled = os.environ.get('KLTN_ENABLE_WATCHDOG', '0') == '1'
            self._watchdog_threshold = float(os.environ.get('KLTN_WATCHDOG_THRESHOLD', '1.25'))
            self._watchdog_last_dump = 0.0
            self._watchdog_dump_limit = int(os.environ.get('KLTN_WATCHDOG_DUMP_LIMIT', '3'))
            self._watchdog_dumps = 0
            # Watchdog thread handle (declared up-front for type-checkers)
            self._watchdog_thread: Optional[threading.Thread] = None
            # Path requested by watchdog for the main thread to save a screenshot (thread-safe)
            self._watchdog_request_screenshot = None
            if self._watchdog_enabled:
                def _watchdog_start():
                    try:
                        t = threading.Thread(target=self._watchdog_loop, daemon=True)
                        t.start()
                        self._watchdog_thread = t
                        logger.debug('Watchdog thread started (threshold=%s s)', self._watchdog_threshold)
                    except Exception:
                        logger.exception('Failed to start watchdog thread')
                _watchdog_start()
        except Exception:
            # If faulthandler or threading not available, skip watchdog
            self._watchdog_enabled = False

        # Track consecutive empty render frames and threshold beyond which we force a display reinit
        self._consecutive_empty_frames = 0
        try:
            self._empty_frame_recovery_threshold = int(os.environ.get('KLTN_EMPTY_FRAME_RECOVERY', '8'))
        except Exception:
            self._empty_frame_recovery_threshold = 8
        
        # New visualization system
        if VISUALIZATION_AVAILABLE:
            self.renderer = ZeldaRenderer(self.TILE_SIZE)
            self.effects = EffectManager()
            self.modern_hud = ModernHUD()
        else:
            # Instantiate no-op fallbacks so attribute access is safe and
            # downstream code does not need to guard every call.
            self.renderer = ZeldaRenderer(self.TILE_SIZE)
            self.effects = EffectManager()
            self.modern_hud = ModernHUD()

        # State for match/undo stack
        self.match_undo_stack = []

        # Heatmap state for A* visualization
        self.show_heatmap = False
        self.search_heatmap = {}  # position -> visit count
        
        # Load assets (fallback for when new system unavailable)
        self._load_assets()
        
        # Initialize environment
        self.env = None
        self.solver = None
        self.auto_path = []
        # ===== DEBUG TEST PATH =====
        # Set KLTN_DEBUG_TEST_PATH=1 to enable red debug path overlay
        if os.environ.get('KLTN_DEBUG_TEST_PATH') == '1':
            self._test_path = [(5, 5), (5, 6), (5, 7), (5, 8), (6, 8), (7, 8), (8, 8), (8, 9), (8, 10)]
            print(f"[DEBUG_INIT] _test_path ENABLED with {len(self._test_path)} points for visual testing")
        else:
            self._test_path = None
        # ===========================
        self.auto_step_idx = 0
        self.auto_mode = False
        self.auto_step_timer = 0.0  # Timer for controlling animation speed
        self.auto_step_interval = 0.15  # Base interval between steps (seconds)
        self.message = "Press SPACE to auto-solve, Arrow keys to move"
        self.message_time = time.time()  # Track when message was set
        self.message_duration = 3.0  # How long to show messages (seconds)
        self.error_message = None
        self.error_time = 0
        self.status_message = "Ready"
        self.show_help = False  # Toggle help overlay
        
        # State-space solver tracking (inventory/edge info)
        self.solver_result = None  # Stores keys_available, keys_used, edge_types etc.
        self.current_keys_held = 0  # Keys currently held during auto-solve
        self.current_keys_used = 0  # Keys used so far during auto-solve
        self.current_edge_types = []  # Edge types traversed so far
        self.door_unlock_times = {}  # Track when doors are unlocked for visual feedback
        
        # Path preview dialog (Feature 5)
        self.path_preview_dialog = None  # PathPreviewDialog instance when showing preview
        self.path_preview_mode = False  # True when showing path preview
        # If True, show a blocking modal dialog. If False, show non-modal overlay + sidebar summary.
        # Default: False to avoid blocking the map view (user prefers sidebar preview).
        self.preview_modal_enabled = False
        # When True the map will show the path overlay and a small sidebar preview box (non-modal)
        self.preview_overlay_visible = False
        # If True, automatically start animation after solver completes (skip preview confirmation)
        # Default: True for immediate animation on SPACE press
        # Set KLTN_AUTO_START_SOLVER=0 to require confirmation
        self.auto_start_solver = os.environ.get('KLTN_AUTO_START_SOLVER', '1') != '0'
        # One-shot flag: when True, next solver result must show preview (never auto-start).
        self.preview_on_next_solver_result = False

        # Topology overlay and DOT export
        self.show_topology = False
        self.topology_export_path = None
        # Topology legend & semantics (for overlays/tooltips)
        self.show_topology_legend = False
        self.topology_semantics = {
            "nodes": {
                "e": ["room", "enemy"],
                "S": ["room", "switch"],
                "b": ["room", "boss"],
                "k": ["room", "key"],
                "K": ["room", "boss key"],
                "I": ["room", "key item"],
                "p": ["room", "puzzle"],
                "s": ["room", "start"],
                "t": ["room", "triforce"]
            },
            "edges": {
                "S": ["door", "switch locked"],
                "b": ["door", "bombable"],
                "k": ["door", "key locked"],
                "K": ["door", "boss key locked"],
                "I": ["door", "key item locked"],
                "l": ["door", "soft locked"],
                "s": ["visible", "impassable"]
            }
        }

        # Solver metrics and comparison results
        self.last_solver_metrics = None  # dict: {name,nodes,time_ms,path_len}
        self.solver_comparison_results = None  # list of dicts
        self.show_solver_comparison_overlay = False

        # === CRITICAL: Solver subprocess state (must be initialized!) ===
        # These variables track the background solver process and must exist before
        # any solver-related code runs (including _schedule_solver, _start_auto_solve)
        self.solver_running = False      # True while solver subprocess is active
        self.solver_proc = None          # multiprocessing.Process handle
        self.solver_done = True          # True when no solver pending (initially done)
        self.solver_outfile = None       # Temp file for solver pickle output
        self.solver_gridfile = None      # Temp file for grid numpy array
        self.solver_thread = None        # Thread fallback handle when process spawn fails
        self._pending_solver_trigger = False  # Flag to trigger solver on next frame (for algorithm changes)
        # Lock to make solver scheduling atomic and thread-safe
        self._solver_lock = threading.Lock()
        
        # Preview subprocess state (separate from main solver)
        self.preview_proc = None         # multiprocessing.Process handle for preview
        self.preview_outfile = None      # Temp file for preview pickle output
        self.preview_gridfile = None     # Temp file for preview grid
        self.preview_done = True         # True when no preview pending
        self.preview_result = None       # Cached result from preview worker
        self.preview_thread = None       # Threading fallback for preview

        # Presets
        self.presets = ['Debugging', 'Fast Approx', 'Optimal', 'Speedrun']
        self.current_preset_idx = 0

        # D* Lite integration
        self.dstar_solver = None
        self.dstar_active = False

        # Parallel search state
        self.parallel_search_thread = None
        self.parallel_search_done = False
        self.parallel_search_result = None

        # Precheck/prune undo snapshot (single-level undo for auto-prune).
        self._precheck_snapshot = None
        
        # Smooth agent animation state
        self.agent_visual_pos = None  # Vector2 for smooth movement
        self.agent_target_pos = None  # Grid position target
        
        # === BLOCK PUSH ANIMATION SYSTEM ===
        # List of active block push animations
        # Each entry: {'from_pos': (r,c), 'to_pos': (r,c), 'start_time': float, 'duration': int}
        self.block_push_animations = []
        self.block_push_duration = 200  # milliseconds for block slide animation
        
        # Speed control system
        self.speed_levels = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
        self.speed_index = 2  # Start at 1.0x
        self.speed_multiplier = self.speed_levels[self.speed_index]
        
        # Game metrics
        self.step_count = 0  # Total steps taken
        self.item_pickup_times = {}  # Track when items were picked up for animation
        
        # Item totals for "X/Y collected" display
        self.total_keys = 0  # Total keys in dungeon
        self.total_bombs = 0  # Total bomb items
        self.total_boss_keys = 0  # Total boss keys
        self.keys_collected = 0  # Keys collected so far
        self.bombs_collected = 0  # Bombs collected
        self.boss_keys_collected = 0  # Boss keys collected
        
        # Toast notification system
        self.toast_notifications = []  # List of ToastNotification objects
        # Debug overlay & logging
        self.debug_overlay_enabled = False
        self.debug_click_log = []  # List of (pos, time, handled_widget_name)
        
        # Continuous movement (hold key to move)
        self.keys_held = {pygame.K_UP: False, pygame.K_DOWN: False, pygame.K_LEFT: False, pygame.K_RIGHT: False}
        self.move_timer = 0.0  # Timer for continuous movement
        self.move_delay = 0.15  # Delay between moves (seconds)
        
        # Minimap settings
        self.show_minimap = True  # Toggle minimap display
        self.minimap_size = 150  # Pixel size of minimap
        self.minimap_clickable = True  # Allow clicking minimap to navigate
        
        # === NEW: Item tracking for enhanced visualization ===
        self.collected_items = []  # List of (pos, item_type, timestamp)
        self.collected_positions = set()  # Set of (row, col) for O(1) lookup during rendering
        self.item_type_map = {}  # pos -> item_type (key, bomb, boss_key, triforce)
        self.used_items = []       # List of (pos, item_type, target_pos, timestamp)
        self.item_markers = {}     # Dict: position -> ItemMarkerEffect
        self.collection_effects = []  # Active collection effects
        self.usage_effects = []    # Active usage effects
        
        # === PATH ITEMS PREVIEW - Track items along auto-solve path ===
        self.path_items_summary = {}  # {item_type: count} - items along path
        self.path_item_positions = {}  # {item_type: [(row, col), ...]} - positions of items on path
        
        # === Toast Notification System ===
        self.toast_notifications = []  # List of ToastNotification objects
        
        # === NEW: GUI Control Panel ===
        self.control_panel_enabled = WIDGETS_AVAILABLE
        self.widget_manager = None
        self.control_panel_width = 360  # Logical expanded width (increased)
        self.control_panel_width_current = float(self.control_panel_width)  # Animated visual width
        self.control_panel_collapsed = False  # Track collapsed state
        self.control_panel_rect = None
        self.collapse_button_rect = None  # Rectangle for collapse button

        # Animation state for smooth collapse/expand
        self.control_panel_animating = False
        self.control_panel_anim_start = 0.0
        self.control_panel_anim_from = float(self.control_panel_width)
        self.control_panel_anim_to = float(self.control_panel_width)
        self.control_panel_anim_duration = 0.22
        self.control_panel_target_collapsed = False
        self.control_panel_x = None  # Custom X position (None = default right side)
        self.control_panel_y = None  # Custom Y position (None = default below minimap)
        self.dragging_panel = False
        self.drag_panel_offset = (0, 0)
        self.resizing_panel = False
        self.resize_edge = None  # 'left', 'right', 'top', 'bottom'

        # Control panel scroll state (for small screens)
        self.control_panel_scroll = 0
        self.control_panel_scroll_step = 20
        self.control_panel_can_scroll = False
        self.control_panel_scroll_max = 0
        self.control_panel_scroll_track_rect = None
        self.control_panel_scroll_thumb_rect = None
        self.control_panel_scroll_dragging = False
        self.control_panel_scroll_drag_offset = 0
        self.control_panel_content_height = 0
        # Debug toggle to draw layout markers and print metrics
        self.debug_control_panel = False

        # Inventory refresh flag (used when updates originate from worker threads)
        self.inventory_needs_refresh = False

        # Scroll inertia/momentum
        self.control_panel_scroll_velocity = 0.0  # pixels per second
        self.control_panel_scroll_damping = 6.0   # damping factor (higher = faster stop)
        # Ignore clicks during active scroll or shortly after to avoid accidental toggles
        self.control_panel_ignore_click_until = 0.0

        self.min_panel_width = 250
        self.max_panel_width = 500
        self.min_panel_height = 300
        
        # Feature toggles (controlled by checkboxes)
        self.feature_flags = {
            'solver_comparison': False,
            'parallel_search': False,
            'multi_goal': False,
            'ml_heuristic': False,
            'dstar_lite': False,
            'show_heatmap': False,
            'show_topology_legend': False,
            'show_minimap': True,
            'show_path': True,  # Show solver path overlay (always visible when path exists)
            'show_topology': False,  # Show topology graph overlay
            'diagonal_movement': False,
            'speedrun_mode': False,
            'strict_original_mode': False,
            'dynamic_difficulty': False,
            'force_grid': False,
            'enable_prechecks': False,
            'auto_prune_on_precheck': False,
            'priority_tie_break': False,
            'priority_key_boost': False,
            'enable_ara': False,
            'use_jps': False,
            'show_jps_overlay': False,
            # MAP-Elites visualization toggle - when enabled the last MAP-Elites
            # heatmap generated by the evaluator will be rendered as an overlay
            'show_map_elites': False,
        }
        # Toggle to force using selected grid algorithm even when graph info exists
        self.force_grid_algorithm = False
        
        # Dropdown selections
        self.current_floor = 1
        self.zoom_level_idx = 3  # 100%
        self.difficulty_idx = 1  # Medium
        self.algorithm_idx = 0   # A*
        self.search_representation = 'hybrid'  # hybrid | tile | graph
        self.ara_weight = 1.0
        
        self._load_current_map()
        self._center_view()  # Center the map in view
        
        # Initialize control panel after map loaded
        if self.control_panel_enabled:
            self._init_control_panel()

        # Draw an initial frame to ensure window contents are painted promptly
        try:
            self._render()
            pygame.display.flip()
        except Exception:
            pass

    
    def _load_assets(self):
        """Load tile images - using colored squares for reliability."""
        self.images = {}
        
        # Color definitions for tile rendering
        color_map = {
            SEMANTIC_PALETTE['VOID']: (20, 20, 20),
            SEMANTIC_PALETTE['FLOOR']: (200, 180, 140),
            SEMANTIC_PALETTE['WALL']: (60, 60, 140),
            SEMANTIC_PALETTE['BLOCK']: (139, 90, 43),
            SEMANTIC_PALETTE['DOOR_OPEN']: (100, 80, 60),
            SEMANTIC_PALETTE['DOOR_LOCKED']: (139, 69, 19),
            SEMANTIC_PALETTE['DOOR_BOMB']: (80, 80, 80),
            SEMANTIC_PALETTE['DOOR_BOSS']: (180, 40, 40),
            SEMANTIC_PALETTE['DOOR_PUZZLE']: (140, 80, 180),
            SEMANTIC_PALETTE['DOOR_SOFT']: (100, 100, 60),
            SEMANTIC_PALETTE['ENEMY']: (200, 50, 50),
            SEMANTIC_PALETTE['START']: (80, 180, 80),
            SEMANTIC_PALETTE['TRIFORCE']: (255, 215, 0),
            SEMANTIC_PALETTE['BOSS']: (150, 20, 20),
            SEMANTIC_PALETTE['KEY_SMALL']: (255, 200, 50),
            SEMANTIC_PALETTE['KEY_BOSS']: (200, 100, 50),
            SEMANTIC_PALETTE['KEY_ITEM']: (100, 200, 255),
            SEMANTIC_PALETTE['ITEM_MINOR']: (200, 200, 200),
            SEMANTIC_PALETTE['ELEMENT']: (50, 80, 180),
            SEMANTIC_PALETTE['ELEMENT_FLOOR']: (80, 100, 160),
            SEMANTIC_PALETTE['STAIR']: (120, 100, 80),
            SEMANTIC_PALETTE['PUZZLE']: (180, 100, 180),
        }
        
        # Create colored square tiles for each semantic ID
        for tile_id, color in color_map.items():
            surf = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE))
            surf.fill(color)
            
            # Add visual indicators for special tiles
            if tile_id == SEMANTIC_PALETTE['DOOR_LOCKED']:
                # Draw keyhole indicator
                pygame.draw.circle(surf, (255, 200, 50), 
                                 (self.TILE_SIZE//2, self.TILE_SIZE//2 - 4), 4)
                pygame.draw.rect(surf, (255, 200, 50),
                               (self.TILE_SIZE//2 - 2, self.TILE_SIZE//2, 4, 8))
            elif tile_id == SEMANTIC_PALETTE['DOOR_BOMB']:
                # Draw crack pattern
                pygame.draw.line(surf, (40, 40, 40), (8, 8), (24, 24), 2)
                pygame.draw.line(surf, (40, 40, 40), (24, 8), (8, 24), 2)
            elif tile_id == SEMANTIC_PALETTE['KEY_SMALL']:
                # Draw key with glow effect for better visibility
                # Outer glow (yellow)
                pygame.draw.circle(surf, (255, 255, 100), (16, 10), 9)
                # Key head (circle)
                pygame.draw.circle(surf, (255, 215, 0), (16, 10), 6)
                # Key shaft
                pygame.draw.rect(surf, (255, 215, 0), (14, 10, 4, 16))
                # Key teeth
                pygame.draw.rect(surf, (255, 215, 0), (14, 22, 2, 3))
                pygame.draw.rect(surf, (255, 215, 0), (16, 24, 2, 2))
                # Inner shine
                pygame.draw.circle(surf, (255, 255, 200), (17, 9), 2)
                pygame.draw.circle(surf, (255, 255, 100, 150), (16, 10), 9)
                # Key head (circle)
                pygame.draw.circle(surf, (255, 215, 0), (16, 10), 6)
                # Key shaft
                pygame.draw.rect(surf, (255, 215, 0), (14, 10, 4, 16))
                # Key teeth
                pygame.draw.rect(surf, (255, 215, 0), (14, 22, 2, 3))
                pygame.draw.rect(surf, (255, 215, 0), (16, 24, 2, 2))
                # Inner shine
                pygame.draw.circle(surf, (255, 255, 200), (17, 9), 2)
            elif tile_id == SEMANTIC_PALETTE['TRIFORCE']:
                # Draw golden triangle
                points = [(16, 4), (4, 28), (28, 28)]
                pygame.draw.polygon(surf, (255, 255, 200), points)
                pygame.draw.polygon(surf, (200, 180, 0), points, 2)
            elif tile_id == SEMANTIC_PALETTE['ENEMY']:
                # Draw enemy indicator (red circle with eyes)
                pygame.draw.circle(surf, (255, 100, 100), (16, 16), 10)
                pygame.draw.circle(surf, (0, 0, 0), (12, 12), 3)
                pygame.draw.circle(surf, (0, 0, 0), (20, 12), 3)
            elif tile_id == SEMANTIC_PALETTE['START']:
                # Draw stair pattern
                pygame.draw.rect(surf, (60, 140, 60), (4, 4, 24, 24))
                for i in range(4):
                    pygame.draw.line(surf, (40, 100, 40), (8, 8+i*6), (24, 8+i*6), 2)
            elif tile_id == SEMANTIC_PALETTE['STAIR']:
                # Draw stair steps
                for i in range(4):
                    pygame.draw.rect(surf, (100, 80, 60), (4+i*4, 20-i*4, 20-i*4, 4))
            elif tile_id == SEMANTIC_PALETTE['WALL']:
                # Add brick pattern to walls
                pygame.draw.rect(surf, (50, 50, 120), (2, 2, 28, 28), 2)
                pygame.draw.line(surf, (70, 70, 150), (0, 16), (32, 16), 1)
                pygame.draw.line(surf, (70, 70, 150), (16, 0), (16, 32), 1)
            elif tile_id == SEMANTIC_PALETTE['BLOCK']:
                # Add block texture
                pygame.draw.rect(surf, (100, 60, 30), (2, 2, 28, 28), 2)
            elif tile_id == SEMANTIC_PALETTE['DOOR_OPEN']:
                # Draw open doorway
                pygame.draw.rect(surf, (40, 30, 20), (8, 0, 16, 32))
            elif tile_id == SEMANTIC_PALETTE['ELEMENT']:
                # Water/lava pattern
                for i in range(4):
                    pygame.draw.arc(surf, (80, 120, 200), (i*8, 8, 16, 16), 0, 3.14, 2)
                    pygame.draw.arc(surf, (80, 120, 200), (i*8, 16, 16, 16), 3.14, 6.28, 2)
            
            # Convert surface to display format with alpha for robust blitting
            try:
                self.images[tile_id] = surf.convert_alpha()
            except Exception:
                # Fallback to raw surface if convert_alpha fails
                self.images[tile_id] = surf
        
        # Create Link sprite
        self.link_img = self._create_link_sprite()

        # Create a small stair sprite (glowing marker) for visual emphasis
        try:
            # Force stair sprite to full tile size and use a bright, high-contrast overlay
            sprite_size = self.TILE_SIZE
            self.stair_sprite = pygame.Surface((sprite_size, sprite_size), pygame.SRCALPHA)
            self.stair_sprite.fill((0, 0, 0, 0))

            # Full-tile translucent fill (warm gold)
            pygame.draw.rect(self.stair_sprite, (255, 220, 100, 180), (0, 0, sprite_size, sprite_size))
            # Strong border for clear visibility
            pygame.draw.rect(self.stair_sprite, (255, 200, 50), (1, 1, sprite_size-2, sprite_size-2), 4)

            # Center triangle to indicate stair direction
            pts = [(sprite_size//2, sprite_size//6), (sprite_size//6, sprite_size*5//6), (sprite_size*5//6, sprite_size*5//6)]
            pygame.draw.polygon(self.stair_sprite, (255, 245, 180), pts)
            pygame.draw.polygon(self.stair_sprite, (255, 200, 50), pts, 2)

            # Slight inner highlight circle
            pygame.draw.circle(self.stair_sprite, (255, 255, 220, 64), (sprite_size//2, sprite_size//2), max(6, sprite_size//6))

            # Convert stair sprite for robust blitting
            try:
                self.stair_sprite = self.stair_sprite.convert_alpha()
            except Exception:
                pass
            self.stair_anim_phase = 0.0
        except Exception:
            self.stair_sprite = None
            self.stair_anim_phase = 0.0
    

    def _create_link_sprite(self):
        """Create a detailed Link sprite using pygame drawing."""
        link_img = pygame.Surface((self.TILE_SIZE - 4, self.TILE_SIZE - 4), pygame.SRCALPHA)
        
        # Transparent background
        link_img.fill((0, 0, 0, 0))
        
        # Body colors
        green = (0, 168, 0)
        skin = (252, 216, 168)
        brown = (136, 112, 0)
        dark_green = (0, 120, 0)
        
        # Draw Link's body (green tunic)
        pygame.draw.rect(link_img, green, (8, 12, 12, 12))  # Torso
        pygame.draw.rect(link_img, dark_green, (6, 18, 4, 8))  # Left arm
        pygame.draw.rect(link_img, dark_green, (18, 18, 4, 8))  # Right arm
        
        # Draw head
        pygame.draw.rect(link_img, skin, (8, 2, 12, 10))  # Face
        pygame.draw.circle(link_img, (0, 0, 0), (11, 6), 2)  # Left eye
        pygame.draw.circle(link_img, (0, 0, 0), (17, 6), 2)  # Right eye
        
        # Draw hair/cap (brown)
        pygame.draw.rect(link_img, brown, (6, 0, 16, 4))  # Hair top
        pygame.draw.rect(link_img, brown, (4, 2, 4, 6))  # Hair left
        pygame.draw.rect(link_img, brown, (20, 2, 4, 6))  # Hair right
        
        # Draw shield (brown rectangle on left side)
        pygame.draw.rect(link_img, brown, (2, 14, 6, 10))
        pygame.draw.rect(link_img, (200, 150, 50), (3, 15, 4, 8))  # Shield front
        
        # Draw sword (on right side)
        pygame.draw.rect(link_img, (180, 180, 180), (22, 12, 4, 14))  # Blade
        pygame.draw.rect(link_img, brown, (22, 10, 4, 4))  # Hilt
        
        try:
            return link_img.convert_alpha()
        except Exception:
            return link_img
    
    def _init_control_panel(self):
        """Initialize the GUI control panel with widgets."""
        if not WIDGETS_AVAILABLE:
            return
        
        self.widget_manager = WidgetManager()
        self._update_control_panel_positions()
    
    def _update_control_panel_positions(self):
        """Update control panel and widget positions (called on resize)."""
        _update_control_panel_positions_helper(
            self,
            pygame,
            logger,
            widgets_available=WIDGETS_AVAILABLE,
            checkbox_widget_cls=CheckboxWidget,
            dropdown_widget_cls=DropdownWidget,
            button_widget_cls=ButtonWidget,
            zoom_labels=GUI_ZOOM_LABELS,
            difficulty_names=GUI_DIFFICULTY_NAMES,
            algorithm_names=GUI_ALGORITHM_NAMES,
        )
    
    def _reposition_widgets(self, panel_x: int, panel_y: int):
        """Reposition existing widgets when panel is dragged (without rebuilding)."""
        _reposition_widgets_helper(
            self,
            panel_x,
            panel_y,
            checkbox_widget_cls=CheckboxWidget,
            dropdown_widget_cls=DropdownWidget,
            button_widget_cls=ButtonWidget,
        )

    def _dump_control_panel_widget_state(self, mouse_pos: tuple):
        """Debug helper: log each widget rects and whether mouse/sc_pos hit them.

        This is defensive and avoids using any variables that may not be available in
        other layout helper contexts.
        """
        _dump_control_panel_widget_state_helper(
            self,
            mouse_pos,
            logger=logger,
            debug_input_active=DEBUG_INPUT_ACTIVE,
        )
        
    
    def _update_inventory_and_hud(self):
        """Reconcile counters and update the modern HUD (if present).

        This centralizes synchronization so any pickup/usage path calls the same routine.
        If called from a non-main thread, set a flag so the main thread performs the UI update
        (pygame surfaces & rendering should be touched only from the main thread).
        """
        _update_inventory_and_hud_helper(self, logger)

    def _remove_from_path_items(self, pos, item_type):
        """Remove a collected item from path_item_positions and update summary.
        
        Args:
            pos: (row, col) position of collected item
            item_type: 'keys', 'boss_keys', 'ladders', 'bombs', etc.
        """
        _remove_from_path_items_helper(self, pos, item_type, logger)

    def _track_item_collection(self, old_state, new_state):
        """Detect when items are collected by comparing states."""
        _track_item_collection_helper(self, old_state, new_state, time, logger, PopEffect, ItemCollectionEffect)
    
    def _track_item_usage(self, old_state, new_state):
        """Detect when items are used (doors opened, walls bombed)."""
        _track_item_usage_helper(self, old_state, new_state, time, logger, ItemUsageEffect)
    
    def _scan_and_mark_items(self):
        """Scan the map for all items and create markers.
        
        This populates item_type_map with all item positions so that
        _sync_inventory_counters() can correctly count collected items.
        """
        _scan_and_mark_items_helper(self, SEMANTIC_PALETTE, logger, ItemMarkerEffect)

    def _apply_pickup_at(self, pos: Tuple[int, int]) -> bool:
        """Apply pickup logic at a position for teleport landings or external mutations.

        This mutates self.env.state to include the collected item and updates
        visual markers/effects and pickup timers. Returns True if an item was
        collected at the position.
        """
        return _apply_pickup_at_helper(self, pos, SEMANTIC_PALETTE, logger, time, ItemCollectionEffect)
    
    def _render_item_legend(self, surface):
        """Render legend showing item counts and path items preview."""
        _render_item_legend_helper(self, surface, pygame)

    def _sync_inventory_counters(self):
        """Reconcile counters from collected_items and env.state to ensure UI accuracy.

        Uses multiple sources for robustness:
        1. self.collected_items list (primary - actively maintained by _track_item_collection)
        2. self.env.state.collected_items + item_type_map (backup)
        
        This ensures real-time updates work correctly during auto-solve.
        """
        _sync_inventory_counters_helper(self)

    def _scan_items_along_path(self, path=None):
        """Scan a path and identify all collectible items along it.
        
        This function analyzes the path positions and finds:
        - KEY_SMALL (30): Regular keys
        - KEY_BOSS (31): Boss keys  
        - KEY_ITEM (32): Ladder/special item
        - ITEM_MINOR (33): Bombs and other minor items
        - DOOR_LOCKED (11): Where keys will be used
        - DOOR_BOMB (12): Where bombs will be used
        - DOOR_BOSS (14): Where boss key will be used
        
        Results stored in:
        - self.path_items_summary: {item_type: count}
        - self.path_item_positions: {item_type: [(row, col), ...]}
        
        Returns:
            dict: Summary of items found along path
        """
        return _scan_items_along_path_helper(self, SEMANTIC_PALETTE, logger, path=path)

    def _get_path_items_display_text(self):
        """Generate a display string summarizing items along the path.
        
        Returns:
            str: Human-readable summary like "Path: 3 keys, 2 doors, 1 boss key"
        """
        return _get_path_items_display_text_helper(self)
    
    def _render_error_banner(self, surface):
        """Render error message banner at top of screen with fade effect."""
        _render_error_banner_helper(self, surface, pygame, time)
    
    def _render_solver_status_banner(self, surface):
        """Render solver status banner showing current algorithm and progress."""
        _render_solver_status_banner_helper(self, surface, pygame, math, time, logger)
    
    def _render_status_bar(self, surface):
        """Render status bar at bottom of screen."""
        _render_status_bar_helper(self, surface, pygame)
    
    def _render_control_panel(self, surface):
        """Render the control panel with all GUI widgets and metrics."""
        _render_control_panel_helper(
            self,
            surface,
            pygame=pygame,
            logger=logger,
            dropdown_widget_cls=DropdownWidget,
        )
    def _render_tooltips(self, surface, mouse_pos):
        """Render tooltips for widgets under mouse cursor."""
        _render_tooltips_helper(self, surface, mouse_pos, ButtonWidget, pygame)
    
    def _draw_tooltip(self, surface, pos, text):
        """Draw a tooltip box at the specified position."""
        _draw_tooltip_helper(self, surface, pos, text, pygame)
    
    def _handle_control_panel_click(self, pos, button, event_type='down'):
        """Handle mouse clicks on control panel widgets."""
        if not self.control_panel_enabled or not self.widget_manager:
            return False
        
        if event_type == 'down':
            panel_hit_rect = self._control_panel_hit_rect()
            if self._should_swallow_control_panel_click(panel_hit_rect, pos):
                return True
            sc_pos = self._translate_control_panel_click(pos, panel_hit_rect)

            outside_result = self._handle_outside_control_panel_click(panel_hit_rect, pos, button)
            if outside_result is not None:
                return outside_result


            # Debug: log transformation and scroll state
            logger.debug('Control panel click: pos=%s sc_pos=%s scroll=%s header_h=%s', pos, sc_pos, getattr(self, 'control_panel_scroll', 0), 45)

            any_contains = self._refresh_control_panel_layout_if_needed(sc_pos)

            handled = self.widget_manager.handle_mouse_down(sc_pos, button)

            logger.debug('Control panel click handled=%s at pos=%s sc_pos=%s any_contains=%s', handled, pos, sc_pos, any_contains)
            if not handled:
                if DEBUG_INPUT_ACTIVE:
                    try:
                        self._dump_control_panel_widget_state(pos)
                    except Exception:
                        logger.exception('Failed to dump widget hit tests after unhandled click')
                handled = self._retry_control_panel_click_after_auto_scroll(pos, sc_pos, button, handled)

            if handled:
                logger.debug('Control panel click handled by widget manager at pos=%r (button=%r)', pos, button)
                self._apply_control_panel_widget_updates()

            return handled
        elif event_type == 'up':
            # Translate pos like we do for mouse-down when scrolled
            if getattr(self, 'control_panel_can_scroll', False) and getattr(self, 'control_panel_rect', None) and self.control_panel_rect.collidepoint(pos):
                sc_pos = (pos[0], pos[1] + getattr(self, 'control_panel_scroll', 0))
            else:
                sc_pos = pos
            return self.widget_manager.handle_mouse_up(sc_pos, button)
        return False

    def _control_panel_hit_rect(self):
        return _control_panel_hit_rect_helper(
            panel_rect=getattr(self, 'control_panel_rect', None),
            debug_control_panel=getattr(self, 'debug_control_panel', False),
            debug_panel_click_padding=getattr(self, 'debug_panel_click_padding', 0),
            rect_factory=pygame.Rect,
        )

    def _should_swallow_control_panel_click(self, panel_hit_rect, pos) -> bool:
        return _should_swallow_control_panel_click_helper(
            dragging=getattr(self, 'control_panel_scroll_dragging', False),
            ignore_click_until=getattr(self, 'control_panel_ignore_click_until', 0.0),
            panel_hit_rect=panel_hit_rect,
            pos=pos,
            logger=logger,
        )

    def _translate_control_panel_click(self, pos, panel_hit_rect):
        return _translate_control_panel_click_helper(
            pos=pos,
            panel_hit_rect=panel_hit_rect,
            panel_rect=getattr(self, 'control_panel_rect', None),
            can_scroll=getattr(self, 'control_panel_can_scroll', False),
            control_panel_scroll=getattr(self, 'control_panel_scroll', 0),
        )

    def _handle_outside_control_panel_click(self, panel_hit_rect, pos, button):
        return _handle_outside_control_panel_click_helper(
            panel_hit_rect=panel_hit_rect,
            pos=pos,
            button=button,
            widget_manager=self.widget_manager,
            dropdown_type=DropdownWidget,
            logger=logger,
        )

    def _refresh_control_panel_layout_if_needed(self, sc_pos) -> bool:
        return _refresh_control_panel_layout_if_needed_helper(
            widget_manager=self.widget_manager,
            sc_pos=sc_pos,
            debug_input_active=DEBUG_INPUT_ACTIVE,
            panel_rect=getattr(self, 'control_panel_rect', None),
            reposition_widgets=self._reposition_widgets,
            logger=logger,
        )

    def _retry_control_panel_click_after_auto_scroll(self, pos, sc_pos, button, handled):
        handled, new_scroll, ignore_until = _retry_control_panel_click_after_auto_scroll_helper(
            pos=pos,
            sc_pos=sc_pos,
            button=button,
            handled=handled,
            panel_rect=getattr(self, 'control_panel_rect', None),
            widget_manager=self.widget_manager,
            can_scroll=getattr(self, 'control_panel_can_scroll', False),
            control_panel_scroll=getattr(self, 'control_panel_scroll', 0),
            control_panel_scroll_max=getattr(self, 'control_panel_scroll_max', 0),
            logger=logger,
        )
        self.control_panel_scroll = new_scroll
        if ignore_until:
            self.control_panel_ignore_click_until = ignore_until
        return handled

    def _apply_control_panel_widget_updates(self):
        """Apply checkbox and dropdown state after a handled control-panel click."""
        _apply_control_panel_widget_updates_helper(
            gui=self,
            widget_manager=self.widget_manager,
            checkbox_type=CheckboxWidget,
            logger=logger,
        )

    def _apply_checkbox_widget_update(self, widget):
        _apply_checkbox_widget_update_helper(gui=self, widget=widget, logger=logger)

    def _apply_dropdown_widget_update(self, widget):
        _apply_dropdown_widget_update_helper(gui=self, widget=widget, logger=logger)

    def _apply_algorithm_dropdown_update(self, widget):
        _apply_algorithm_dropdown_update_helper(gui=self, widget=widget, logger=logger)
    
    # Button callbacks
    def _stop_auto_solve(self):
        """Stop auto-solve and clear visual state."""
        self.auto_mode = False
        self.auto_path = []  # Clear path display
        self.auto_step_idx = 0
        self.block_push_animations = []  # Clear block push animations
        self.message = "Auto-solve stopped"
    
    def _generate_dungeon(self):
        """Generate a new random dungeon using the procedural generator."""
        try:
            from src.generation.dungeon_generator import DungeonGenerator, Difficulty
            import random
            
            # Generate random seed for reproducibility display
            seed = random.randint(0, 999999)
            
            # Create generator with medium difficulty, reasonable size
            generator = DungeonGenerator(
                width=40,
                height=40,
                difficulty=Difficulty.MEDIUM,
                seed=seed
            )
            
            # Generate the dungeon grid
            grid = generator.generate()
            
            # Add the generated dungeon to the map list
            dungeon_name = f"Generated #{seed}"
            self.maps.append(grid)
            self.map_names.append(dungeon_name)
            
            # Switch to the new map
            self.current_map_idx = len(self.maps) - 1
            self._load_current_map()
            self._center_view()
            
            # Clear any existing effects and reset state
            if self.effects:
                self.effects.clear()
            self.step_count = 0
            self.auto_path = []
            self.auto_mode = False
            
            self._set_message(f"Generated dungeon (seed: {seed}, {len(generator.rooms)} rooms)")
            logger.info(f"Generated dungeon: seed={seed}, rooms={len(generator.rooms)}, keys={len(generator.key_positions)}")
            
        except ImportError as e:
            logger.warning(f"Dungeon generator not available: {e}")
            self._set_message("Dungeon generator module not found")
        except Exception as e:
            logger.exception(f"Failed to generate dungeon: {e}")
            self._set_message(f"Generation failed: {str(e)}")

    def _generate_ai_dungeon(self):
        """Non-blocking wrapper to spawn background worker and return immediately."""
        _start_ai_dungeon_generation_helper(self, threading)


    def _generate_ai_dungeon_worker(self):
        """Background worker entry point for AI generation pipeline."""
        _run_ai_generation_worker_helper(self, logger)

    def _reset_map(self):
        """Reset the current map."""
        _reset_map_helper(self)
    
    def _show_path_preview(self):
        """
        Show path preview for the currently available route.

        Behavior:
        - If a path already exists, open preview immediately.
        - If solver is running, request preview on completion.
        - If no path exists and solver is idle, start solver and force preview when it finishes.
        """
        _show_path_preview_helper(self, PathPreviewDialog, logger)
    
    def _clear_path(self):
        """Clear the current path."""
        _clear_path_helper(self)

    def _open_temp_folder(self):
        """Open OS temp folder where solver/preview artifacts are stored."""
        _open_temp_folder_orchestration_helper(self, tempfile, _open_folder_helper)

    def _collect_temp_file_candidates(self):
        """Collect active and stale GUI temp files used by solver/preview flows."""
        return _collect_temp_file_candidates_orchestration_helper(
            self,
            tempfile,
            _list_existing_paths_helper,
            _find_temp_files_helper,
        )

    def _delete_temp_files(self):
        """Delete stale temp files and optionally active tracked files when safe."""
        _delete_temp_files_orchestration_helper(
            self,
            os,
            logger,
            self._collect_temp_file_candidates,
            _list_existing_paths_helper,
            _delete_files_helper,
        )
    
    def _export_route(self):
        """Export the current route to JSON file."""
        _export_route_helper(self)
    
    def _load_route(self):
        """Load a saved route from JSON file."""
        _load_route_helper(self)

    def load_visual_assets(self, templates_dir: str = None, link_sprite_path: str = None):
        """Optional: override GUI assets with extracted visual tiles/sprites.

        Usage (copy-paste into startup code):
            gui = ZeldaGUI(maps)
            gui.load_visual_assets('data/tileset.png', 'data/link_sprite.png')

        Behaviour:
        - If `templates_dir` is a folder of tile images, create pygame surfaces from them
          and assign to `self.images` keyed by semantic id (best-effort).
        - If `link_sprite_path` is provided, attempt to cut a Link sprite and replace `self.link_img`.
        """
        return _load_visual_assets_helper(
            self,
            templates_dir=templates_dir,
            link_sprite_path=link_sprite_path,
            pygame=pygame,
            os_module=os,
            logger=logger,
            semantic_palette=SEMANTIC_PALETTE,
        )

    def load_visual_map(self, image_path: str, templates_dir: str | None = None):
        """Public API: create a GUI map from a screenshot and switch to it.

        - `image_path` can be a full screenshot (HUD allowed).
        - `templates_dir` is passed to the visual extractor (tileset or folder).

        This method is intentionally permissive and returns a bool for success
        so automated tests can call it without a file dialog.
        """
        return _load_visual_map_helper(
            self,
            image_path=image_path,
            templates_dir=templates_dir,
        )

    def _place_items_from_graph(self, grid: np.ndarray, graph, room_positions: dict, room_to_node: dict):
        """Place items (keys, boss keys, etc.) from graph node attributes into the grid.
        
        The VGLC data adapter stores items as graph node attributes (has_key=True, etc.)
        but doesn't place them in the semantic grid. This function materializes those
        items into the grid so the inventory system can track them.
        
        Args:
            grid: Numpy array of semantic tile IDs (modified in place)
            graph: NetworkX graph with node attributes
            room_positions: Dict mapping room position -> (row_offset, col_offset) in global grid
            room_to_node: Dict mapping room position -> graph node ID
        """
        _place_items_from_graph_helper(
            self,
            grid=grid,
            graph=graph,
            room_positions=room_positions,
            room_to_node=room_to_node,
            logger=logger,
            semantic_palette=SEMANTIC_PALETTE,
        )

    def _load_current_map(self):
        """Load and initialize the current map."""
        _load_current_map_helper(
            self,
            os_module=os,
            logger=logger,
            zelda_logic_env_cls=ZeldaLogicEnv,
            sanity_checker_cls=SanityChecker,
            semantic_palette=SEMANTIC_PALETTE,
        )
    
    def _center_view(self):
        """Center the current map in the view."""
        _center_view_helper(self)
    
    def _auto_fit_zoom(self):
        """Automatically set zoom level to fit the entire map in view."""
        _auto_fit_zoom_helper(self)
    
    def _change_zoom(self, delta: int, center: tuple | None = None):
        """Change zoom level by delta steps.

        If `center` is provided (screen coordinates), the view will be adjusted so
        that the map tile under the `center` pixel remains under the cursor after
        the zoom. If `center` is None, the view is centered as before.
        """
        _change_zoom_helper(self, delta, center)
    
    def _safe_set_mode(self, size, flags=0, allow_fallback=True):
        """Robust wrapper around pygame.display.set_mode.

        Attempts set_mode and, on failure or invalid surface (size 0), performs
        a display reinit and retries. If all attempts fail and allow_fallback is
        True, falls back to a windowed 800x600 surface to avoid leaving the
        application with a null/zero-sized display.
        Returns the created screen surface (or None on fatal failure).
        """
        return _safe_set_mode_helper(size, pygame, logger, flags=flags, allow_fallback=allow_fallback)

    def _attempt_display_reinit(self):
        """Attempt to fully reinitialize the SDL display and restore mode."""
        return _attempt_display_reinit_helper(self, pygame, logger)

    def _handle_watchdog_screenshot(self) -> bool:
        """Save the requested watchdog screenshot on the main thread and clear the request.

        Returns True if a screenshot was saved, False otherwise. Always clears the
        request to avoid repeated attempts.
        """
        return _handle_watchdog_screenshot_helper(self, pygame, logger, os)

    def report_ui_state(self) -> dict:
        """Return diagnostic information about GUI state for troubleshooting (callable from REPL)."""
        return _report_ui_state_helper(self, logger)

    def _ensure_display_alive(self, force=False):
        """Check display health and attempt recovery if needed.

        If the display surface is None or has zero size, try to restore it.
        This method is intentionally conservative and returns False only when
        no recovery was possible.
        """
        return _ensure_display_alive_helper(self, pygame, logger, force=force)

    def _force_focus(self) -> bool:
        """Try to force the window to the foreground on Windows.

        Uses a conservative Win32 sequence (AttachThreadInput + SetForegroundWindow + temporary TOPMOST) to
        work around Windows' foreground activation blocking. Returns True on success.
        No-op on non-Windows platforms.
        """
        return _force_focus_helper(self, pygame, logger, os)

    def _toggle_fullscreen(self):
        """Toggle fullscreen mode with robust handling.

        Uses `pygame.display.Info()` to obtain a valid fullscreen size and
        preserves the previous windowed size for restore. Ensures event pump
        and asset/layout reinitialization to avoid dark screens or unresponsiveness.
        """
        return _toggle_fullscreen_helper(self, pygame, logger, os, __import__('platform'))

    # ------------------ Control Panel Animation ------------------
    def _start_toggle_panel_animation(self, target_collapsed: bool):
        """Begin animated transition to collapsed or expanded state."""
        _start_toggle_panel_animation_helper(self, target_collapsed, time)

    def _update_control_panel_animation(self):
        """Update animation state; should be called each frame."""
        _update_control_panel_animation_helper(self, time)

    def _update_control_panel_scroll(self):
        """Per-frame update that applies inertia (momentum) and clamps scroll."""
        _update_control_panel_scroll_helper(self, time)

    def run(self, max_frames: Optional[int] = None):
        """Main game loop with delta-time support.

        When running under tests (env var KLTN_TEST_MODE or under pytest), a small
        default max_frames is used to avoid infinite loops. Callers can override
        with the optional max_frames parameter.
        """
        max_frames = resolve_test_mode_max_frames(max_frames, os.environ)

        # Heartbeat logging variables for responsiveness debugging
        heartbeat_last = time.time()
        heartbeat_interval = 0.5  # seconds (more frequent for debugging)

        running = True
        frame_count = 0
        
        while running:
            # Calculate delta time for smooth animations
            current_time = time.time()
            self.delta_time = current_time - self.last_frame_time
            self.last_frame_time = current_time
            
            _events = poll_pygame_events(pygame, time, logger)

            run_input_focus_fallback(
                self,
                pygame,
                time,
                logger,
                should_attempt_focus_fallback,
            )
            for event in _events:
                clear_stale_preview_overlay(self, logger)

                # Handle window focus events (improves input responsiveness on Windows)
                if handle_window_focus_event(self, event, pygame, logger):
                    continue

                if handle_global_keydown_shortcuts(
                    self,
                    event,
                    pygame,
                    time,
                    logger,
                    CheckboxWidget,
                ):
                    continue

                if handle_preview_overlay_events(self, event, pygame):
                    continue

                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.VIDEORESIZE:
                    # Handle window resize
                    self.screen_w = max(event.w, self.MIN_WIDTH)
                    self.screen_h = max(event.h, self.MIN_HEIGHT)
                    if not self.fullscreen:
                        # Use safe wrapper to avoid producing a zero-sized or invalid surface
                        screen = self._safe_set_mode((self.screen_w, self.screen_h), pygame.RESIZABLE)
                        if not screen:
                            logger.warning('VIDEORESIZE: _safe_set_mode failed; attempting display reinit')
                            # Attempt a display reinit when set_mode fails
                            try:
                                self._attempt_display_reinit()
                            except Exception:
                                logger.exception('VIDEORESIZE: display reinit failed')
                        else:
                            self.screen = screen
                            try:
                                self.screen_w, self.screen_h = self.screen.get_size()
                            except Exception:
                                pass
                        # Refresh assets/layout and force an immediate present
                        try:
                            self._load_assets()
                            self._render()
                            try:
                                pygame.display.flip()
                            except Exception:
                                logger.exception('Flip failed after VIDEORESIZE')
                        except Exception:
                            logger.exception('Failed to refresh UI after VIDEORESIZE')
                    # Update control panel widget positions
                    if self.control_panel_enabled:
                        self._update_control_panel_positions()
                
                elif event.type == pygame.MOUSEWHEEL:
                    mouse_pos = pygame.mouse.get_pos()
                    # If mouse is over control panel and scrolling is enabled, apply momentum to panel
                    panel_rect = getattr(self, 'control_panel_rect', None)
                    padding = getattr(self, 'debug_panel_click_padding', 0) if getattr(self, 'debug_control_panel', False) else 0
                    panel_hit_rect = (pygame.Rect(panel_rect.x - padding, panel_rect.y, panel_rect.width + padding, panel_rect.height) if panel_rect and padding else panel_rect)
                    if self.control_panel_enabled and getattr(self, 'control_panel_can_scroll', False) and panel_hit_rect and panel_hit_rect.collidepoint(mouse_pos) and not self.control_panel_collapsed:
                        # Use wheel to add velocity (pixels per second)
                        wheel_power = getattr(self, 'control_panel_scroll_step', 20) * 12
                        # Negative event.y means scroll down? We want positive y to scroll up (decrease coord)
                        self.control_panel_scroll_velocity += -event.y * wheel_power
                        # Clamp velocity to reasonable bounds
                        max_v = 2000
                        self.control_panel_scroll_velocity = max(-max_v, min(max_v, self.control_panel_scroll_velocity))
                        # Ignore immediate clicks while momentum is active
                        self.control_panel_ignore_click_until = time.time() + 0.12
                    else:
                        # Zoom with mouse wheel when not over panel
                        # Only perform mouse-centered zoom when the mouse is over the main map area
                        sidebar_x = self.screen_w - self.SIDEBAR_WIDTH
                        if mouse_pos[0] < sidebar_x:
                            self._change_zoom(event.y, center=mouse_pos)
                        else:
                            # Falling back to center zoom if wheel over sidebar
                            self._change_zoom(event.y)
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos, consumed = handle_mouse_button_down_preamble(
                        self,
                        event,
                        pygame,
                        time,
                        logger,
                        DEBUG_INPUT_ACTIVE,
                    )
                    if consumed:
                        continue
                    
                    # Handle collapse button click first (animated)
                    if self.control_panel_enabled and self.collapse_button_rect and self.collapse_button_rect.collidepoint(mouse_pos):
                        # Ignore input if animation already running
                        if not getattr(self, 'control_panel_animating', False):
                            target_collapsed = not self.control_panel_collapsed
                            self._start_toggle_panel_animation(target_collapsed)
                        continue
                    
                    # Check if starting to drag panel (click on title bar area)
                    if self.control_panel_enabled and self.control_panel_rect and not self.control_panel_collapsed:
                        # Check if clicking on scrollbar thumb to start drag
                        if event.button == 1 and getattr(self, 'control_panel_scroll_thumb_rect', None) and self.control_panel_scroll_thumb_rect.collidepoint(mouse_pos):
                            self.control_panel_scroll_dragging = True
                            self.control_panel_scroll_drag_offset = mouse_pos[1] - self.control_panel_scroll_thumb_rect.y
                            continue
                        # Clicking on track -> page to that location
                        if event.button == 1 and getattr(self, 'control_panel_scroll_track_rect', None) and self.control_panel_scroll_track_rect.collidepoint(mouse_pos):
                            tr = self.control_panel_scroll_track_rect
                            rel = mouse_pos[1] - tr.y
                            max_move = tr.height - getattr(self, 'control_panel_scroll_thumb_rect', pygame.Rect(0,0,0,20)).height
                            ratio = max(0.0, min(1.0, rel / tr.height))
                            self.control_panel_scroll = int(ratio * getattr(self, 'control_panel_scroll_max', 0))
                            continue

                        title_bar_height = 45
                        title_bar_rect = pygame.Rect(
                            self.control_panel_rect.x,
                            self.control_panel_rect.y,
                            self.control_panel_rect.width,
                            title_bar_height
                        )
                        if title_bar_rect.collidepoint(mouse_pos) and not self.collapse_button_rect.collidepoint(mouse_pos):
                            self.dragging_panel = True
                            self.drag_panel_offset = (mouse_pos[0] - self.control_panel_rect.x, mouse_pos[1] - self.control_panel_rect.y)
                            continue
                        
                        # Check if starting to resize panel (near edges)
                        edge_threshold = 8
                        mx, my = mouse_pos
                        rect = self.control_panel_rect
                        
                        if abs(mx - rect.left) < edge_threshold and rect.top <= my <= rect.bottom:
                            self.resizing_panel = True
                            self.resize_edge = 'left'
                            continue
                        elif abs(mx - rect.right) < edge_threshold and rect.top <= my <= rect.bottom:
                            self.resizing_panel = True
                            self.resize_edge = 'right'
                            continue
                        elif abs(my - rect.bottom) < edge_threshold and rect.left <= mx <= rect.right:
                            self.resizing_panel = True
                            self.resize_edge = 'bottom'
                            continue
                    
                    # Handle control panel clicks
                    if self.control_panel_enabled and self._handle_control_panel_click(mouse_pos, event.button, 'down'):
                        continue  # Control panel handled the click
                    
                    if event.button == 1:  # Left click - check minimap and start map drag if on map
                        if self._handle_minimap_click(mouse_pos):
                            pass  # Minimap click handled
                        else:
                            # Start map drag with left button when clicking on the main map area (not on sidebar or panel)
                            sidebar_x = self.screen_w - self.SIDEBAR_WIDTH
                            if mouse_pos[0] < sidebar_x and not (self.control_panel_enabled and getattr(self, 'control_panel_rect', None) and self.control_panel_rect.collidepoint(mouse_pos)):
                                self.dragging = True
                                self.dragging_button = 1
                                self.drag_start = event.pos
                    elif event.button == 2:  # Middle mouse
                        self.dragging = True
                        self.dragging_button = 2
                        self.drag_start = event.pos
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    if handle_mouse_button_up_event(self, event, pygame, time, logger):
                        continue
                
                elif event.type == pygame.MOUSEMOTION:
                    mouse_pos = handle_mouse_motion_diagnostics(self, event, pygame, time, logger)
                    
                    # Handle panel dragging
                    if self.dragging_panel:
                        self.control_panel_x = mouse_pos[0] - self.drag_panel_offset[0]
                        self.control_panel_y = mouse_pos[1] - self.drag_panel_offset[1]
                        # Clamp to screen bounds
                        self.control_panel_x = max(0, min(self.control_panel_x, self.screen_w - self.control_panel_width))
                        self.control_panel_y = max(0, min(self.control_panel_y, self.screen_h - 100))
                        # Update widget positions to follow panel
                        self._reposition_widgets(self.control_panel_x, self.control_panel_y)
                    
                    # Handle panel resizing
                    elif self.resizing_panel and self.control_panel_rect:
                        if self.resize_edge == 'left':
                            old_right = self.control_panel_rect.right
                            new_x = mouse_pos[0]
                            new_width = old_right - new_x
                            if self.min_panel_width <= new_width <= self.max_panel_width:
                                self.control_panel_width = new_width
                                self.control_panel_x = new_x
                        elif self.resize_edge == 'right':
                            new_width = mouse_pos[0] - self.control_panel_rect.x
                            if self.min_panel_width <= new_width <= self.max_panel_width:
                                self.control_panel_width = new_width
                        elif self.resize_edge == 'bottom':
                            new_height = mouse_pos[1] - self.control_panel_rect.y
                            if self.min_panel_height <= new_height <= self.screen_h - self.control_panel_rect.y - 20:
                                pass  # Height is auto-calculated, just update visual feedback
                    
                    # Handle scrollbar thumb dragging
                    elif getattr(self, 'control_panel_scroll_dragging', False) and getattr(self, 'control_panel_can_scroll', False) and getattr(self, 'control_panel_scroll_track_rect', None):
                        track_rect = self.control_panel_scroll_track_rect
                        thumb_rect = getattr(self, 'control_panel_scroll_thumb_rect', None)
                        if thumb_rect is None:
                            continue
                        # Compute local mouse position inside track
                        rel_y = mouse_pos[1] - track_rect.y
                        max_move = track_rect.height - thumb_rect.height
                        new_thumb_top = max(0, min(rel_y - getattr(self, 'control_panel_scroll_drag_offset', 0), max_move))
                        if max_move > 0:
                            ratio = new_thumb_top / max_move
                            self.control_panel_scroll = int(ratio * getattr(self, 'control_panel_scroll_max', 0))
                            # Clamp
                            self.control_panel_scroll = max(0, min(self.control_panel_scroll, getattr(self, 'control_panel_scroll_max', 0)))
                    
                    # Handle map dragging
                    elif self.dragging:
                        dx = self.drag_start[0] - event.pos[0]
                        dy = self.drag_start[1] - event.pos[1]
                        self.view_offset_x += dx
                        self.view_offset_y += dy
                        self.drag_start = event.pos
                        self._clamp_view_offset()
                    # Update cursor for resize edges (when not dragging)
                    elif self.control_panel_enabled and self.control_panel_rect and not self.control_panel_collapsed:
                        edge_threshold = 8
                        mx, my = mouse_pos
                        rect = self.control_panel_rect
                        
                        if abs(mx - rect.left) < edge_threshold and rect.top <= my <= rect.bottom:
                            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_SIZEWE)
                        elif abs(mx - rect.right) < edge_threshold and rect.top <= my <= rect.bottom:
                            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_SIZEWE)
                        elif abs(my - rect.bottom) < edge_threshold and rect.left <= mx <= rect.right:
                            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_SIZENS)
                        else:
                            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)

                elif event.type == pygame.KEYUP:
                    # Ensure we stop continuous movement when keys are released
                    try:
                        if event.key in getattr(self, 'keys_held', {}):
                            self.keys_held[event.key] = False
                    except Exception:
                        logger.debug('Failed to handle KEYUP for %r', getattr(event, 'key', None))

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.fullscreen:
                            self._toggle_fullscreen()
                        else:
                            running = False
                    
                    elif event.key == pygame.K_F11:
                        self._toggle_fullscreen()
                    
                    elif event.key == pygame.K_h:
                        # Toggle heatmap overlay (H key)
                        if not self.show_help:  # Don't toggle if help shown
                            self.show_heatmap = not self.show_heatmap
                            self.feature_flags['show_heatmap'] = self.show_heatmap
                            if self.renderer:
                                self.renderer.show_heatmap = self.show_heatmap
                            # Update checkbox widget if available
                            if self.widget_manager:
                                for widget in self.widget_manager.widgets:
                                    if isinstance(widget, CheckboxWidget) and hasattr(widget, 'flag_name') and widget.flag_name == 'show_heatmap':
                                        widget.checked = self.show_heatmap
                            self.message = f"Heatmap: {'ON' if self.show_heatmap else 'OFF'}"
                    
                    elif event.key == pygame.K_F1:
                        self.show_help = not self.show_help
                    
                    elif event.key == pygame.K_TAB:
                        # Toggle control panel with Tab key
                        if self.control_panel_enabled:
                            # Animate toggle for Tab key as well
                            if not getattr(self, 'control_panel_animating', False):
                                target_collapsed = not self.control_panel_collapsed
                                self._start_toggle_panel_animation(target_collapsed)

                    elif event.key == pygame.K_F7:
                        # Diagnostic hotkey: dump widget hit-test state at current mouse
                        try:
                            pos = pygame.mouse.get_pos()
                            logger.info('DIAG DUMP (F7): mouse_pos=%s control_panel_rect=%s scroll=%s', pos, getattr(self,'control_panel_rect',None), getattr(self,'control_panel_scroll',0))
                            try:
                                self._dump_control_panel_widget_state(pos)
                            except Exception:
                                logger.exception('F7: _dump_control_panel_widget_state failed')
                        except Exception:
                            logger.exception('F7 diagnostic failed')

                    elif event.key == pygame.K_F8:
                        # Toggle debug overlay for control panel and hit-padding
                        try:
                            self.debug_control_panel = not getattr(self, 'debug_control_panel', False)
                            self.debug_panel_click_padding = int(os.environ.get('KLTN_DEBUG_PANEL_PADDING', '40')) if self.debug_control_panel else 0
                            self._show_toast(f"Debug control panel {'ON' if self.debug_control_panel else 'OFF'}", 1.6, 'info')
                            logger.info('Toggled debug_control_panel=%s padding=%s', self.debug_control_panel, self.debug_panel_click_padding)
                        except Exception:
                            logger.exception('Failed to toggle debug control panel')
                    
                    elif event.key == pygame.K_m:
                        # Toggle minimap
                        self.show_minimap = not self.show_minimap
                        self.feature_flags['show_minimap'] = self.show_minimap
                        # Update checkbox widget if available
                        if self.widget_manager:
                            for widget in self.widget_manager.widgets:
                                if isinstance(widget, CheckboxWidget) and hasattr(widget, 'flag_name') and widget.flag_name == 'show_minimap':
                                    widget.checked = self.show_minimap
                        self.message = f"Minimap: {'ON' if self.show_minimap else 'OFF'}"
                    
                    elif event.key == pygame.K_RIGHTBRACKET or event.key == pygame.K_PERIOD:
                        # Increase speed
                        self.speed_index = min(len(self.speed_levels) - 1, self.speed_index + 1)
                        self.speed_multiplier = self.speed_levels[self.speed_index]
                        self.message = f"Speed: {self.speed_multiplier}x"
                    
                    elif event.key == pygame.K_LEFTBRACKET or event.key == pygame.K_COMMA:
                        # Decrease speed
                        self.speed_index = max(0, self.speed_index - 1)
                        self.speed_multiplier = self.speed_levels[self.speed_index]
                        self.message = f"Speed: {self.speed_multiplier}x"
                    
                    elif event.key == pygame.K_SPACE:
                        self._start_auto_solve()
                    
                    elif event.key == pygame.K_r:
                        self._load_current_map()
                        self._center_view()
                        # Clear effects
                        if self.effects:
                            self.effects.clear()
                        # Reset step count
                        self.step_count = 0
                        self.message = "Map Reset"
                    
                    elif event.key == pygame.K_n:
                        self._next_map()
                    
                    elif event.key == pygame.K_p:
                        self._prev_map()

                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        self._change_zoom(1)

                    elif event.key == pygame.K_MINUS:
                        self._change_zoom(-1)

                    elif event.key == pygame.K_0:
                        # Reset zoom to default
                        self.zoom_idx = self.DEFAULT_ZOOM_IDX
                        self.TILE_SIZE = self.ZOOM_LEVELS[self.zoom_idx]
                        self._load_assets()
                        self._center_view()
                        self.message = "Zoom reset to default"

                    elif event.key == pygame.K_f:
                        # Auto-fit zoom to show entire map
                        self._auto_fit_zoom()
                        self.message = f"Auto-fit: {self.TILE_SIZE}px"

                    elif event.key == pygame.K_c:
                        # Center view on player
                        self._center_on_player()

                    elif event.key == pygame.K_l:
                        ok = self.load_visual_map(os.path.join(os.getcwd(), 'screenshot.png'))
                        if not ok:
                            self.message = "Failed to load ./screenshot.png"

                    # Track key holds for continuous movement
                    elif event.key in self.keys_held and not self.auto_mode:
                        self.keys_held[event.key] = True
                        self.move_timer = 0.0  # Reset timer for immediate first move
                        
                    elif not self.auto_mode:
                        # Manual movement - check for diagonal combos first
                        keys = pygame.key.get_pressed()
                        action = None

                        # Check diagonal combinations (two arrow keys pressed)
                        if keys[pygame.K_UP] and keys[pygame.K_LEFT]:
                            action = Action.UP_LEFT
                        elif keys[pygame.K_UP] and keys[pygame.K_RIGHT]:
                            action = Action.UP_RIGHT
                        elif keys[pygame.K_DOWN] and keys[pygame.K_LEFT]:
                            action = Action.DOWN_LEFT
                        elif keys[pygame.K_DOWN] and keys[pygame.K_RIGHT]:
                            action = Action.DOWN_RIGHT
                        # Cardinal directions (single key)
                        elif keys[pygame.K_UP]:
                            action = Action.UP
                        elif keys[pygame.K_DOWN]:
                            action = Action.DOWN
                        elif keys[pygame.K_LEFT]:
                            action = Action.LEFT
                        elif keys[pygame.K_RIGHT]:
                            action = Action.RIGHT

                        if action is not None:
                            self._manual_step(action)
                            self._center_on_player()

            # Auto-solve stepping with timer-based animation.
            run_auto_step_tick(self, logger, frame_count)
            
            # Update widget manager with mouse position
            if self.widget_manager:
                mouse_pos = pygame.mouse.get_pos()
                self.widget_manager.update(mouse_pos, self.delta_time)
            
            # Handle continuous movement (hold key to move) with diagonal support.
            run_continuous_movement_tick(self, pygame, Action)
            
            # Update toast notifications
            self._update_toasts()

            # Periodic heartbeat to confirm main loop alive
            try:
                now = time.time()
                if now - heartbeat_last > heartbeat_interval:
                    heartbeat_last = now
                    logger.debug("GUI heartbeat - frame=%d auto_mode=%s solver_running=%s", frame_count, getattr(self,'auto_mode',False), getattr(self,'solver_running',False))
            except Exception:
                pass

            # Check if algorithm change triggered a pending solver (deferred to avoid blocking event handler)
            if getattr(self, '_pending_solver_trigger', False):
                self._pending_solver_trigger = False
                alg_name = self._algorithm_name(self.algorithm_idx)
                logger.info('â‰¡Æ’Ã¶Ã¤ Processing pending solver trigger: Starting %s solver...', alg_name)
                self._start_auto_solve()

            # Update animated control panel state (if active)
            self._update_control_panel_animation()

            # Update control panel scroll inertia (momentum)
            self._update_control_panel_scroll()

            # If parallel search ran in background, handle result on main thread
            if getattr(self, 'parallel_search_done', False) and getattr(self, 'parallel_search_result', None):
                best = self.parallel_search_result
                # Convert alg index to name
                alg_names = ['A*','BFS','Dijkstra','Greedy']
                name = alg_names[best['alg']] if best['alg'] < len(alg_names) else f"Alg{best['alg']}"
                self._set_message(f"Parallel best: {name} ({best['nodes']} nodes, {best['time_ms']:.0f}ms)")
                self.parallel_search_done = False
                self.parallel_search_result = None
                # Use found path
                _handle_found_path = None
                try:
                    # Reuse the same handling as in _start_auto_solve: set auto_path and show preview
                    self.auto_path = best['path']
                    self.preview_overlay_visible = True
                    logger.debug('Parallel search: setting preview_overlay_visible=True (parallel best path)')
                    self.path_preview_dialog = PathPreviewDialog(path=self.auto_path, env=self.env, solver_result={}, speed_multiplier=self.speed_multiplier)
                    self._set_message('Parallel result ready (sidebar preview)')
                except Exception as e:
                    logger.warning(f"Failed to display parallel search preview: {e}")

            # If a preview process finished, read its output and apply it
            if getattr(self, 'preview_proc', None) and not getattr(self, 'preview_done', False):
                p = getattr(self, 'preview_proc')
                if not p.is_alive():
                    out = getattr(self, 'preview_outfile', None)
                    res = None
                    try:
                        if out:
                            res = _safe_unpickle(out)
                    except Exception as e:
                        logger.exception('Failed to read preview output: %s', e)
                    finally:
                        try:
                            p.join(timeout=0.1)
                        except Exception:
                            pass
                        try:
                            if out and os.path.exists(out):
                                os.remove(out)
                        except Exception:
                            pass
                        try:
                            gf = getattr(self, 'preview_gridfile', None)
                            if gf and os.path.exists(gf):
                                os.remove(gf)
                        except Exception:
                            pass
                        self.preview_proc = None
                        self.preview_outfile = None
                        self.preview_gridfile = None
                        self.preview_done = True

                    if res:
                        try:
                            if res.get('success') and res.get('path'):
                                self.auto_path = res.get('path')
                                self.preview_overlay_visible = True
                                logger.debug('Preview result: setting preview_overlay_visible=True (preview has path)')
                                try:
                                    # Use (x or {}) pattern to handle None values
                                    solver_result_preview = (res.get('solver_result') or {}) if res else {}
                                    self.path_preview_dialog = PathPreviewDialog(path=self.auto_path, env=self.env, solver_result=solver_result_preview, speed_multiplier=self.speed_multiplier)
                                except Exception:
                                    self.path_preview_dialog = None
                                self._set_message('Preview ready (sidebar)')
                            else:
                                msg = res.get('message') or 'Preview finished with no path'
                                self._set_message(msg)
                        except Exception as e:
                            logger.exception('Failed to apply preview output on main thread: %s', e)
                    else:
                        self._set_message('Preview finished (no output)')
                    self.preview_done = True
                if getattr(self, 'solver_running', False):
                    # small ping in status message to reassure user
                    self.status_message = 'Solving...'
                else:
                    self.status_message = 'Ready'

            # If a solver subprocess (or thread fallback) finished, read its output and apply result on the main thread
            if not getattr(self, 'solver_done', False):
                proc = getattr(self, 'solver_proc', None)
                solver_thread = getattr(self, 'solver_thread', None)
                proc_alive = False
                thread_alive = False
                solver_starting = getattr(self, 'solver_starting', False)
                
                # CRITICAL: Check for solver timeout (scaled by algorithm and map size).
                active_alg = int(getattr(self, 'solver_algorithm_idx', getattr(self, 'algorithm_idx', 0)))
                grid_cells = None
                try:
                    current_map = self.maps[self.current_map_idx]
                    grid_ref = current_map.global_grid if hasattr(current_map, 'global_grid') else current_map
                    grid_cells = int(np.asarray(grid_ref).size)
                except Exception:
                    pass
                solver_timeout = compute_solver_timeout_seconds(
                    active_alg,
                    grid_cell_count=grid_cells,
                    env_getter=os.environ.get,
                )
                solver_start_time = getattr(self, 'solver_start_time', None)
                timed_out = False
                # Timeout handling is only meaningful for process mode.
                if proc and solver_start_time and (time.time() - solver_start_time) > solver_timeout:
                    timed_out = True
                    logger.error('SOLVER: TIMEOUT after %.1fs - forcefully terminating', solver_timeout)
                    if proc:
                        try:
                            # Give subprocess a brief chance to finish normal shutdown/write output.
                            proc.join(timeout=0.2)
                            if proc.is_alive():
                                proc.terminate()
                                proc.join(timeout=0.5)
                        except Exception as e:
                            logger.exception('SOLVER: Failed to terminate timed-out process: %s', e)
                    proc_alive = False
                
                if not timed_out:
                    try:
                        proc_alive = proc.is_alive() if proc else False
                    except Exception as e:
                        logger.exception('SOLVER: proc.is_alive() raised exception: %s', e)
                        proc_alive = False
                    try:
                        thread_alive = solver_thread.is_alive() if solver_thread else False
                    except Exception as e:
                        logger.exception('SOLVER: solver_thread.is_alive() raised exception: %s', e)
                        thread_alive = False

                # Startup grace: avoid treating proc None as completion while spawn thread is still starting
                startup_grace = float(os.environ.get('KLTN_SOLVER_STARTUP_GRACE', '1.5'))
                solver_age = (time.time() - solver_start_time) if solver_start_time else 0.0
                out = getattr(self, 'solver_outfile', None)
                out_exists = os.path.exists(out) if out else False
                if solver_starting and proc is None and not thread_alive and not out_exists and not timed_out and solver_age < startup_grace:
                    logger.debug('SOLVER: Waiting for process start (age=%.2fs < %.2fs grace)', solver_age, startup_grace)
                elif thread_alive:
                    logger.debug('SOLVER: Waiting for thread fallback completion (age=%.2fs)', solver_age)
                elif proc is None or not proc_alive:
                    # CRITICAL: Wrap ENTIRE completion block in try/finally to guarantee solver_running cleanup
                    try:
                        proc_exitcode = None
                        if proc is not None:
                            proc_exitcode = getattr(proc, 'exitcode', None)
                            logger.info('SOLVER: Subprocess done, proc.is_alive()=False, exitcode=%s', proc.exitcode)
                        else:
                            logger.info('SOLVER: No subprocess handle (thread fallback or spawn failure)')

                        # Try to load the outfile
                        out = getattr(self, 'solver_outfile', None)
                        logger.info('SOLVER: Reading result from %s, exists=%s', out, os.path.exists(out) if out else 'N/A')
                        res = None
                        try:
                            if out:
                                res = _safe_unpickle(out)
                                path_len = len(res.get('path', []) or []) if res else 0
                                solver_result_safe = (res.get('solver_result') or {}) if res else {}
                                logger.info('SOLVER: Result loaded, path_len=%d, success=%s, keys=%s',
                                            path_len,
                                            res.get('success') if res else None,
                                            solver_result_safe.get('keys_used', 'N/A'))
                            else:
                                logger.warning('SOLVER: Output file missing or path is None: %s', out)
                        except Exception as e:
                            logger.exception('SOLVER: Failed to read solver output: %s', e)

                        # Apply results on main thread
                        if res:
                            try:
                                if res.get('success') and res.get('path'):
                                    self.auto_path = res.get('path')
                                    # Use (x or {}) pattern to handle None values
                                    solver_result = (res.get('solver_result') or {}) if res else {}
                                    
                                    # CRITICAL: Verify path doesn't go through water
                                    water_violations = find_path_tile_violations(
                                        self.auto_path,
                                        self.env.grid,
                                        blocked_tile_ids={40},  # ELEMENT (water)
                                    )
                                    
                                    if water_violations:
                                        print(f"\n{'='*60}")
                                        print(f"ERROR: PATH GOES THROUGH WATER!")
                                        print(f"Found {len(water_violations)} water tiles in path:")
                                        for step, r, c, tid in water_violations[:5]:
                                            print(f"  Step {step}: position ({r}, {c}) = tile ID {tid} (WATER)")
                                        print(f"{'='*60}\n")
                                        logger.error(f"PATH ERROR: {len(water_violations)} water tiles in path!")
                                    else:
                                        print(f"\n{'='*60}")
                                        print(f"PATH VERIFIED: No water tiles")
                                        print(f"Path length: {len(self.auto_path)} steps")
                                        print(f"{'='*60}\n")
                                    
                                    # Print path sample to verify water avoidance
                                    print(f"\n{'='*60}")
                                    print(f"PATH LOADED: {len(self.auto_path)} steps")
                                    if len(self.auto_path) > 10:
                                        print(f"First 10 steps: {self.auto_path[:10]}")
                                    print(f"{'='*60}\n")
                                    
                                    logger.info('SOLVER: Path applied! auto_path len=%d, first=%s, last=%s',
                                                len(self.auto_path),
                                                self.auto_path[0] if self.auto_path else None,
                                                self.auto_path[-1] if self.auto_path else None)
                                    
                                    # Auto-start mode unless caller explicitly requested preview.
                                    force_preview = bool(getattr(self, 'preview_on_next_solver_result', False))
                                    if getattr(self, 'auto_start_solver', False) and not force_preview:
                                        logger.info('SOLVER: auto_start_solver=True, starting animation immediately')
                                        self._execute_auto_solve(self.auto_path, solver_result, teleports=0)
                                        self._set_message(f'Auto-solve started! Path: {len(self.auto_path)} steps')
                                        logger.info('SOLVER: Animation started, auto_mode=%s, auto_step_idx=%s',
                                                    getattr(self, 'auto_mode', None),
                                                    getattr(self, 'auto_step_idx', None))
                                    else:
                                        # Show preview for user confirmation.
                                        logger.info(
                                            'SOLVER: showing preview dialog (auto_start_solver=%s, force_preview=%s)',
                                            getattr(self, 'auto_start_solver', False),
                                            force_preview,
                                        )
                                        self.path_preview_dialog = PathPreviewDialog(
                                            path=self.auto_path,
                                            env=self.env,
                                            solver_result=solver_result,
                                            speed_multiplier=self.speed_multiplier,
                                        )
                                        if getattr(self, 'preview_modal_enabled', False):
                                            self.path_preview_mode = True
                                            self.preview_overlay_visible = False
                                        else:
                                            self.path_preview_mode = False
                                            self.preview_overlay_visible = True
                                        self._set_message('Solver finished (press ENTER to start or ESC to dismiss)')
                                    self.preview_on_next_solver_result = False
                                else:
                                    msg = res.get('message') or 'Solver finished with no path'
                                    if timed_out:
                                        msg = f'Solver timed out after {int(solver_timeout)}s'
                                        logger.info('SOLVER: %s', msg)
                                    elif msg == 'output file missing' and proc_exitcode is not None and proc_exitcode < 0:
                                        msg = f'Solver terminated (exitcode={proc_exitcode}) before writing output'
                                        logger.info('SOLVER: %s', msg)
                                    else:
                                        logger.warning('SOLVER: No valid path in result: %s', msg)
                                    self._set_message(msg)
                                    self.preview_on_next_solver_result = False
                            except Exception as e:
                                logger.exception('SOLVER: Failed to apply result on main thread: %s', e)
                                self._set_message('Solver error (see logs)')
                                self.preview_on_next_solver_result = False
                        else:
                            if timed_out:
                                logger.error('SOLVER: No result - subprocess timed out')
                                self._set_message(f'Solver timed out after {int(solver_timeout)}s')
                            elif proc_exitcode is not None and proc_exitcode < 0:
                                logger.info(
                                    'SOLVER: Subprocess terminated before output (exitcode=%s)',
                                    proc_exitcode,
                                )
                                self._set_message(f'Solver terminated (exitcode={proc_exitcode})')
                            else:
                                logger.warning('SOLVER: No result loaded (res is None), subprocess may have crashed')
                                self._set_message('Solver finished (no output)')
                            self.preview_on_next_solver_result = False
                    finally:
                        # CRITICAL: ALWAYS clean up process and files in finally block
                        # This MUST run even if result loading/application crashes
                        logger.info('SOLVER: Entering cleanup finally block')
                        try:
                            if proc:
                                proc.join(timeout=0.1)
                        except Exception as e:
                            logger.exception('SOLVER: proc.join() failed: %s', e)
                        try:
                            out = getattr(self, 'solver_outfile', None)
                            if out and os.path.exists(out):
                                os.remove(out)
                        except Exception as e:
                            logger.exception('SOLVER: Failed to remove output file: %s', e)
                        try:
                            gf = getattr(self, 'solver_gridfile', None)
                            if gf and os.path.exists(gf):
                                os.remove(gf)
                        except Exception as e:
                            logger.exception('SOLVER: Failed to remove grid file: %s', e)
                        
                        # CRITICAL: Clear all solver state atomically using centralized helper
                        self._clear_solver_state(reason="solver completed/failed")

            # If AI generation worker finished, apply result on main thread
            if getattr(self, 'ai_gen_done', False):
                res = getattr(self, 'ai_gen_result', None)
                self.ai_gen_done = False
                self.ai_gen_result = None
                if res and res.get('success') and res.get('grid') is not None:
                    self.maps.append(res['grid'])
                    self.map_names.append(res.get('name', 'AI Generated'))
                    self.current_map_idx = len(self.maps) - 1
                    self._load_current_map()
                    self._center_view()
                    self._set_message('AI generation complete', 3.0)
                else:
                    self._set_message('AI generation failed', 3.0)

            # Render
            self._render()

            # Present frame to the display (ensure visual updates after resize/fullscreen changes)
            try:
                pygame.display.flip()
            except Exception:
                logger.exception('pygame.display.flip() failed; attempting pygame.display.update() and fallback')
                try:
                    pygame.display.update()
                except Exception:
                    logger.exception('pygame.display.update() also failed')
                    # Try a reinit if flip/update both fail and display seems unhealthy
                    try:
                        if not self._ensure_display_alive():
                            logger.warning('Display not healthy after flip/update; attempted recovery')
                    except Exception:
                        logger.exception('Attempted display recovery after flip/update failures')

            # If watchdog requested a screenshot, perform it on the main thread (thread-safe)
            try:
                # Let a dedicated helper perform the watchdog screenshot save and clear the request
                try:
                    self._handle_watchdog_screenshot()
                except Exception:
                    logger.exception('Error during watchdog screenshot handling')
            except Exception:
                # Be defensive: avoid crashing the main loop due to watchdog handling
                logger.exception('Error handling watchdog screenshot request')

            # Periodic display health check (throttled)
            try:
                now = time.time()
                if now - getattr(self, '_display_check_last', 0.0) >= getattr(self, '_display_check_interval', 1.0):
                    self._display_check_last = now
                    ok = self._ensure_display_alive()
                    if not ok:
                        # If recovery failed, show a persistent message to the user
                        self._set_message('Display recovery attempted; see logs', 6.0)
            except Exception:
                logger.exception('Error during display health check')

            # Increment frame counter and check test-mode limit
            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                logger.debug("Exiting run loop due to max_frames=%r", max_frames)
                running = False

            # Cap framerate - use higher FPS during auto_mode for smoother animations
            self.clock.tick(60 if self.auto_mode else 30)
        
        pygame.quit()

    def _next_map(self):
        """Move to the next map and stop auto-solve if running."""
        _next_map_helper(self, logger)

    def _prev_map(self):
        """Move to the previous map and stop auto-solve if running."""
        _prev_map_helper(self, logger)
    
    def _clamp_view_offset(self):
        """Clamp view offset to valid range.

        When the dungeon/map is smaller than the viewport, allow negative offsets so
        the user can pan the small map freely inside the window (showing empty
        margins) while still preventing arbitrary unrestricted panning.
        """
        _clamp_view_offset_helper(self)
    
    def _center_on_player(self):
        """Center the view on the player position."""
        _center_on_player_helper(self)
    
    def _start_preview_for_current_map(self):
        _start_preview_for_current_map_helper(
            gui=self,
            logger=logger,
            pygame_module=pygame,
            multiprocessing_module=multiprocessing,
            threading_module=threading,
            time_module=time,
            run_preview_and_dump=_run_preview_and_dump,
        )

    def _clear_solver_state(self, reason="cleanup"):
        """Helper to centralize solver state cleanup and ensure consistency.
        
        Args:
            reason: Description of why solver is being cleared (for logging)
        """
        _clear_solver_state_helper(gui=self, reason=reason, logger=logger)

    def _sync_solver_dropdown_settings(self):
        """Refresh algorithm/representation/ARA values from dropdown widgets."""
        return _sync_solver_dropdown_settings_helper(gui=self, sync_fn=sync_solver_dropdown_settings)

    def _algorithm_name(self, algorithm_idx):
        """Return canonical display label for a solver index."""
        return algorithm_label(algorithm_idx)

    def _start_auto_solve(self):
        """Start auto-solve mode using state-space solver with inventory tracking.

        This schedules the heavy solver in a background process/thread using
        the existing `_schedule_solver()` helper. Non-blocking and safe to call
        from the main loop or event handlers.
        """
        
        _start_auto_solve_helper(gui=self, logger=logger, debug_sync_solver=DEBUG_SYNC_SOLVER)

    def _prepare_active_solver_for_new_start(self) -> bool:
        """Return True when a new solver run may proceed, False to block startup."""
        return _prepare_active_solver_for_new_start_helper(
            gui=self,
            logger=logger,
            time_module=time,
            evaluate_solver_recovery_state=evaluate_solver_recovery_state,
            compute_timeout_seconds=self._compute_solver_timeout_seconds,
            terminate_hung_process=self._terminate_hung_solver_process,
            force_recovery_state=self._force_solver_recovery_state,
            log_active_state=self._log_active_solver_state,
        )

    def _log_active_solver_state(self):
        _log_active_solver_state_helper(gui=self, logger=logger, os_module=os, time_module=time)

    def _compute_solver_timeout_seconds(self, active_alg: int) -> float:
        return _compute_solver_timeout_seconds_helper(
            gui=self,
            active_alg=active_alg,
            default_solver_timeout_for_algorithm=default_solver_timeout_for_algorithm,
            scale_timeout_by_grid_size=scale_timeout_by_grid_size,
            np_module=np,
            os_module=os,
        )

    def _terminate_hung_solver_process(self, proc):
        _terminate_hung_solver_process_helper(proc=proc, logger=logger)

    def _force_solver_recovery_state(self, recovery_reason: str):
        _force_solver_recovery_state_helper(gui=self, recovery_reason=recovery_reason, logger=logger)

    def _cleanup_preview_before_solver_start(self):
        """Stop preview workers/files so new solve starts from a clean state."""
        _cleanup_preview_before_solver_start_helper(gui=self, logger=logger, os_module=os)

    def _reset_solver_visual_state_before_start(self):
        """Clear solver/visual state from previous runs before scheduling a new solve."""
        _reset_solver_visual_state_before_start_helper(gui=self)

    def _get_solver_map_context(self):
        """Return current grid and optional topology context needed by solver backends."""
        return _get_solver_map_context_orchestration_helper(
            gui=self,
            get_solver_map_context_helper=_get_solver_map_context_helper,
        )

    def _build_solver_request(self, algorithm_idx=None, on_missing_message='Start/goal not defined for this map'):
        """Build a canonical solver request payload from current GUI state."""
        return _build_solver_request_orchestration_helper(
            gui=self,
            build_solver_request_helper=_build_solver_request_helper,
            algorithm_idx=algorithm_idx,
            on_missing_message=on_missing_message,
        )

    def _run_solver_sync(self, algorithm_idx=None):
        """DEBUG: Run solver synchronously in main thread to bypass multiprocessing issues.
        
        This blocks the UI but helps diagnose whether the issue is in multiprocessing
        or in the solver/animation logic itself.
        """
        _run_solver_sync_helper(
            gui=self,
            logger=logger,
            solve_in_subprocess=_solve_in_subprocess,
            algorithm_idx=algorithm_idx,
        )

    def _watchdog_loop(self):
        """Background watchdog that writes stack traces and a screenshot when the main loop stalls.

        Controlled by environment vars:
        - KLTN_ENABLE_WATCHDOG (default '1') enable watchdog
        - KLTN_WATCHDOG_THRESHOLD (seconds, default 1.25)
        - KLTN_WATCHDOG_DUMP_LIMIT (how many dumps to write, default 3)
        - KLTN_WATCHDOG_TERMINATE_SOLVER (if '1' will terminate solver proc when dumping)
        """
        _watchdog_loop_helper(
            gui=self,
            logger=logger,
            os_module=os,
            time_module=time,
            tempfile_module=tempfile,
        )
        return
        

    
    def _schedule_solver(self, algorithm_idx=None):
        """Start solver in background worker process/thread.
        
        Args:
            algorithm_idx: Algorithm index to use (if None, read from self.algorithm_idx)
        """
        return _schedule_solver_helper(
            gui=self,
            algorithm_idx=algorithm_idx,
            logger=logger,
            time_module=time,
            threading_module=threading,
        )

    def _create_solver_temp_files(self, grid_arr):
        """Create output and optional grid temp files for solver worker launch."""
        return _create_solver_temp_files_helper(grid_arr)

    def _launch_solver_worker(self, **kwargs):
        """Launch solver process, with thread-based fallback on process failure."""
        _launch_solver_worker_helper(
            gui=self,
            kwargs=kwargs,
            logger=logger,
            launch_solver_process=self._launch_solver_process,
            start_solver_thread_fallback=self._start_solver_thread_fallback,
            multiprocessing_module=multiprocessing,
        )

    def _launch_solver_process(self, **kwargs):
        _launch_solver_process_helper(
            gui=self,
            launch_kwargs=kwargs,
            run_solver_and_dump=_run_solver_and_dump,
            multiprocessing_module=multiprocessing,
            logger=logger,
        )

    def _solver_thread_fallback_worker(self, **kwargs):
        _solver_thread_fallback_worker_helper(
            gui=self,
            launch_kwargs=kwargs,
            solve_in_subprocess=_solve_in_subprocess,
            logger=logger,
        )

    def _start_solver_thread_fallback(self, **kwargs):
        _start_solver_thread_fallback_helper(
            gui=self,
            launch_kwargs=kwargs,
            threading_module=threading,
            worker_target=self._solver_thread_fallback_worker,
            logger=logger,
        )

    def _execute_auto_solve(self, path, solver_result, teleports=0):
        """
        Execute auto-solve immediately without preview (fallback).
        
        Args:
            path: Planned path
            solver_result: Solver metadata (may include CBS metrics)
            teleports: Number of teleport/warp moves
        """
        _execute_auto_solve_helper(
            gui=self,
            path=path,
            solver_result=solver_result,
            teleports=teleports,
            logger=logger,
        )
    
    def _execute_auto_solve_from_preview(self):
        """
        Start auto-solve after user confirms path preview.
        """
        _execute_auto_solve_from_preview_helper(gui=self, logger=logger)
    
    def _smart_grid_path(self):
        """
        Smart pathfinding that prioritizes walking and only warps via STAIRs.
        Returns (success, path, teleport_count).
        """
        return _smart_grid_path_helper(
            gui=self,
            logger=logger,
            convert_diagonal_to_4dir=_convert_diagonal_to_4dir,
            semantic_palette=SEMANTIC_PALETTE,
            np_module=np,
            path_cls=Path,
            os_module=os,
        )

    def _graph_guided_path(self):
        """Fallback: follow graph path with teleportation when needed."""
        return _graph_guided_path_helper(gui=self)

    def _hybrid_graph_grid_path(self):
        """
        Hybrid pathfinding: use graph to find room sequence, 
        then BFS within each room and teleport between disconnected clusters.
        """
        return _hybrid_graph_grid_path_helper(gui=self)

    def _stop_auto(self, reason: str = None):
        """Stop auto-solve mode with consistent logging and cleanup."""
        return _stop_auto_helper(gui=self, reason=reason, logger=logger)

    def _auto_step(self):
        """Execute one step of auto-solve with comprehensive error handling."""
        import traceback

        return _auto_step_helper(
            gui=self,
            logger=logger,
            game_state_cls=GameState,
            action_enum=Action,
            ripple_effect_cls=RippleEffect,
            flash_effect_cls=FlashEffect,
            traceback_module=traceback,
        )
    
    def _show_error(self, message: str):
        """Display error message to user with visual feedback."""
        return _show_error_helper(self, message, logger, time)
    
    def _show_message(self, message: str, duration: float = 3.0):
        """Display informational message to user."""
        return _show_message_helper(self, message, duration, logger, time)

    # --- Topology helpers ---
    def _export_topology(self):
        """Export current map topology to a DOT file (if available)."""
        return _export_topology_helper(self)


    def _render_topology_overlay(self, surface):
        """Draw room nodes and edges on the map area with high-visibility styling."""
        current = self.maps[self.current_map_idx]
        _render_topology_overlay_helper(
            surface=surface,
            current=current,
            tile_size=self.TILE_SIZE,
            view_offset_x=self.view_offset_x,
            view_offset_y=self.view_offset_y,
            pygame=pygame,
        )

    def _match_missing_nodes(self):
        """Attempt to infer and stage mapping proposals for unmatched nodes.

        Uses RoomGraphMatcher.infer_missing_mappings to generate proposals with confidences.
        High-confidence proposals (>= configured threshold) are applied automatically.
        Lower confidence proposals are kept as 'tentative' in `current.match_proposals` for manual apply.
        """
        return _match_missing_nodes_helper(gui=self, matcher_cls=RoomGraphMatcher, logger=logger)

    def _undo_last_match(self):
        """Undo last applied match snapshot, if any."""
        return _undo_last_match_helper(gui=self, logger=logger)

    def _room_for_global_position(self, pos: Optional[Tuple[int, int]], room_positions: dict) -> Optional[Tuple[int, int]]:
        """Map a global tile coordinate to a room-grid coordinate."""
        return _room_for_global_position_helper(pos, room_positions)

    @staticmethod
    def _node_has_small_key(attrs: dict) -> bool:
        """Best-effort small-key detection from graph node attributes/labels."""
        return _node_has_small_key_helper(attrs)

    def _node_has_critical_content(self, graph, node_id: Any) -> bool:
        """Whether a node should be preserved during dead-end pruning."""
        return _node_has_critical_content_helper(graph, node_id)

    def _capture_precheck_snapshot(self, current: Any, reason: str = "") -> None:
        """Capture current topology state so Undo Prune can restore it."""
        self._precheck_snapshot = _capture_precheck_snapshot_helper(current, reason=reason)

    def _update_env_topology_view(self, current: Any) -> None:
        """Synchronize current map topology attributes into the active env object."""
        _update_env_topology_view_helper(getattr(self, 'env', None), current)

    def _build_room_adjacency_from_graph(self, graph: Any, room_to_node: dict, node_to_room: dict) -> dict:
        """Build undirected room adjacency from graph edges via node-room mapping."""
        return _build_room_adjacency_from_graph_helper(graph, room_to_node, node_to_room)

    def _prune_dead_end_topology(self, current: Any, preserve_rooms: set) -> List[Tuple[int, int]]:
        """Prune dead-end rooms from topology mapping when room objects are unavailable."""
        return _prune_dead_end_topology_flow_helper(
            gui=self,
            current=current,
            preserve_rooms=preserve_rooms,
            logger=logger,
            build_room_adjacency_fn=self._build_room_adjacency_from_graph,
            node_has_critical_content_fn=self._node_has_critical_content,
        )

    def _run_prechecks_and_optional_prune(self) -> Tuple[bool, Optional[str]]:
        """Run lightweight prechecks and optional dead-end pruning before solve."""
        current = self.maps[self.current_map_idx]
        return _run_prechecks_and_optional_prune_flow_helper(
            gui=self,
            current=current,
            logger=logger,
            np_module=np,
            semantic_palette=SEMANTIC_PALETTE,
            action_deltas=ACTION_DELTAS,
            topology_has_path_fn=_topology_has_path_helper,
            min_locked_between_fn=_min_locked_between_helper,
            walkable_grid_reachable_fn=_walkable_grid_reachable_helper,
            node_has_small_key_fn=self._node_has_small_key,
            room_for_global_position_fn=self._room_for_global_position,
            zelda_dungeon_adapter=ZeldaDungeonAdapter,
            capture_snapshot_fn=self._capture_precheck_snapshot,
            update_env_topology_view_fn=self._update_env_topology_view,
            prune_dead_end_topology_fn=self._prune_dead_end_topology,
        )

    def _undo_prune(self):
        """Undo the last applied prune snapshot, if any."""
        current = self.maps[self.current_map_idx]
        return _undo_prune_flow_helper(
            gui=self,
            current=current,
            logger=logger,
            update_env_topology_view_fn=self._update_env_topology_view,
        )

    def _apply_tentative_matches(self):
        """Apply staged tentative matches above the configured threshold."""
        return _apply_tentative_matches_helper(gui=self, logger=logger)

    # --- Solver comparison helpers ---
    def _set_last_solver_metrics(self, name, nodes, time_ms, path_len):
        return _set_last_solver_metrics_helper(
            gui=self,
            name=name,
            nodes=nodes,
            time_ms=time_ms,
            path_len=path_len,
        )

    def _run_solver_comparison(self):
        """Start an asynchronous solver comparison worker to avoid blocking the GUI."""
        return _run_solver_comparison_helper(
            gui=self,
            logger=logger,
            time_module=time,
            game_state_cls=GameState,
            solve_in_subprocess=_solve_in_subprocess,
            threading_module=threading,
        )

    def _start_map_elites(self, n_samples: int = 200, resolution: int = 20):
        """Start a background MAP-Elites evaluation on the currently loaded maps.

        Runs on a background thread so the GUI stays responsive. Results are stored
        in `self.map_elites_result` and a toast is shown when complete.
        """
        return _start_map_elites_flow_helper(
            gui=self,
            n_samples=n_samples,
            resolution=resolution,
            threading_module=threading,
        )

    def _map_elites_worker(self, maps, n_samples: int, resolution: int):
        """Background worker implementing MAP-Elites on a set of pre-loaded maps.

        This function uses the lightweight `src.simulation.map_elites` helper and the
        built-in `DungeonSolver` for validation.
        """
        return _map_elites_worker_flow_helper(
            gui=self,
            maps=maps,
            n_samples=n_samples,
            resolution=resolution,
            logger=logger,
            os_module=os,
        )

    def _render_solver_comparison_overlay(self, surface):
        """Render a small sidebar table with solver comparison results."""
        _render_solver_comparison_overlay_helper(
            surface=surface,
            results=getattr(self, 'solver_comparison_results', None),
            screen_w=self.screen_w,
            sidebar_width=self.SIDEBAR_WIDTH,
            pygame=pygame,
        )
    
    def _set_message(self, message: str, duration: float = 3.0):
        """Set status message with timestamp for auto-hide."""
        _set_message_helper(self, message, duration, time)
    
    def _show_toast(self, message: str, duration: float = 3.0, toast_type: str = 'info'):
        """Show a floating toast notification."""
        _show_toast_helper(self, message, duration, toast_type, ToastNotification)
    
    def _format_cbs_metrics_tooltip(self, cbs_metrics: dict) -> str:
        """Format CBS metrics for detailed tooltip display."""
        return _format_cbs_metrics_tooltip_helper(cbs_metrics)
    
    def _update_toasts(self):
        """Update and remove expired toasts."""
        _update_toasts_helper(self)
    
    def _render_toasts(self, surface):
        """Render all active toast notifications."""
        _render_toasts_helper(self, surface)
    
    # ========================================
    # BLOCK PUSH ANIMATION SYSTEM
    # ========================================
    
    def _start_block_push_animation(self, block_from: Tuple[int, int], block_to: Tuple[int, int]):
        """Start animating a block being pushed from one position to another.
        
        Args:
            block_from: Original block position (row, col)
            block_to: Destination position (row, col)
        """
        _start_block_push_animation_helper(self, block_from, block_to, pygame, logger)
    
    def _update_block_push_animations(self):
        """Update all active block push animations and complete finished ones."""
        _update_block_push_animations_helper(self, pygame, SEMANTIC_PALETTE, PopEffect, logger)
    
    def _render_block_push_animations(self, surface):
        """Render blocks that are currently being pushed with smooth interpolation.
        
        Args:
            surface: The pygame surface to draw on (map_surface)
        """
        _render_block_push_animations_helper(self, surface, pygame, SEMANTIC_PALETTE)
    
    def _get_animating_block_positions(self) -> set:
        """Get set of block positions currently being animated (to skip normal rendering)."""
        return _get_animating_block_positions_helper(self)
    
    def _check_and_start_block_push(self, player_pos: Tuple[int, int], target_pos: Tuple[int, int], 
                                     action: Action) -> bool:
        """Check if moving to target_pos would push a block and start animation if so.
        
        Args:
            player_pos: Current player position (row, col)
            target_pos: Position player is trying to move to (row, col)
            action: The movement action being taken
            
        Returns:
            True if a block push was initiated, False otherwise
        """
        _ = action
        return _check_and_start_block_push_helper(self, player_pos, target_pos, WALKABLE_IDS, PUSHABLE_IDS)

    def _show_warning(self, message: str):
        """Display warning message to user."""
        _show_warning_helper(self, message, logger)
    
    def _manual_step(self, action: Action):
        """Execute manual step."""
        return _manual_step_flow_helper(
            gui=self,
            action=action,
            action_deltas=ACTION_DELTAS,
            pop_effect_cls=PopEffect,
            flash_effect_cls=FlashEffect,
            time_module=time,
        )
    
    def _render_path_GUARANTEED(self, surface):
        """GUARANTEED path rendering - draws path no matter what.
        
        This method provides bulletproof path visualization that works
        regardless of auto_mode, preview state, or feature flags.
        Call this AFTER tiles are drawn but BEFORE HUD elements.
        """
        return _render_path_guaranteed_flow_helper(
            gui=self,
            surface=surface,
            pygame=pygame,
            math_module=math,
            time_module=time,
            logger=logger,
        )

    def _render(self):
        """Render the current state using new visualization system or fallback."""
        # Clear screen
        self.screen.fill((25, 25, 35))
        
        h, w = self.env.height, self.env.width
        # Compute view area and ensure valid integer sizes (avoid zero/negative surfaces)
        view_w = max(1, int(self.screen_w - self.SIDEBAR_WIDTH))
        view_h = max(1, int(self.screen_h - self.HUD_HEIGHT))
        
        # Create map surface for the main view area (use convert for faster blits)
        try:
            map_surface = pygame.Surface((view_w, view_h)).convert()
        except Exception:
            # Fallback to plain surface if convert is unsupported
            map_surface = pygame.Surface((view_w, view_h))
        map_surface.fill((20, 20, 30))


        
        tiles_drawn = 0
        
        # Apply speed multiplier to animation updates
        effective_dt = self.delta_time * self.speed_multiplier
        
        # Update new renderer if available
        if self.renderer:
            self.renderer.update(effective_dt)
        if self.effects:
            self.effects.update(effective_dt)
        
        # Update block push animations
        self._update_block_push_animations()
        
        # If a background thread requested an inventory refresh, perform it here (main thread)
        if getattr(self, 'inventory_needs_refresh', False):
            try:
                logger.debug("Processing deferred inventory refresh on main thread")
                self._update_inventory_and_hud()
            except Exception:
                pass
            finally:
                self.inventory_needs_refresh = False

        # Update modern HUD with current game state every frame (real-time)
        if self.modern_hud and self.env:
            self.modern_hud.update_game_state(
                keys=self.env.state.keys,
                bombs=1 if self.env.state.has_bomb else 0,
                has_boss_key=self.env.state.has_boss_key,
                position=self.env.state.position,
                steps=self.step_count,
                message=self.message
            )
            if hasattr(self.modern_hud, 'inventory'):
                self.modern_hud.inventory.keys_collected = self.keys_collected
                self.modern_hud.inventory.bombs_collected = self.bombs_collected
                self.modern_hud.inventory.boss_keys_collected = self.boss_keys_collected
                self.modern_hud.inventory.keys_used = getattr(self, 'keys_used', 0)
                self.modern_hud.inventory.bombs_used = getattr(self, 'bombs_used', 0)
                self.modern_hud.inventory.boss_keys_used = getattr(self, 'boss_keys_used', 0)
            # Backwards compatibility: also set direct attributes if present
            if hasattr(self.modern_hud, 'keys_collected'):
                self.modern_hud.keys_collected = self.keys_collected
                self.modern_hud.bombs_collected = self.bombs_collected
                self.modern_hud.boss_keys_collected = self.boss_keys_collected
            if hasattr(self.modern_hud, 'keys_used'):
                self.modern_hud.keys_used = getattr(self, 'keys_used', 0)
            if hasattr(self.modern_hud, 'bombs_used'):
                self.modern_hud.bombs_used = getattr(self, 'bombs_used', 0)
            if hasattr(self.modern_hud, 'boss_keys_used'):
                self.modern_hud.boss_keys_used = getattr(self, 'boss_keys_used', 0)
        
        # Draw grid (only visible tiles for performance)
        start_c = max(0, int(self.view_offset_x) // self.TILE_SIZE)
        start_r = max(0, int(self.view_offset_y) // self.TILE_SIZE)
        end_c = min(w, start_c + (view_w // self.TILE_SIZE) + 2)
        end_r = min(h, start_r + (view_h // self.TILE_SIZE) + 2)
        _log_draw_ranges_overlay_helper(
            gui=self,
            start_r=start_r,
            end_r=end_r,
            start_c=start_c,
            end_c=end_c,
            h=h,
            w=w,
            time_module=time,
            logger=logger,
        )
        _render_empty_range_warning_overlay_helper(
            gui=self,
            start_r=start_r,
            end_r=end_r,
            start_c=start_c,
            end_c=end_c,
            pygame=pygame,
        )
        
        # Pre-fetch collected items for efficient lookup during rendering
        # Combine env state collected_items with GUI's collected_positions for robustness
        env_collected = getattr(self.env.state, 'collected_items', set()) or set()
        gui_collected = getattr(self, 'collected_positions', set()) or set()
        collected_items = env_collected | gui_collected  # Union of both sets
        # Define collectible tile IDs that should be hidden if in collected_items
        COLLECTIBLE_TILE_IDS = (
            SEMANTIC_PALETTE.get('KEY_SMALL', -1),
            SEMANTIC_PALETTE.get('KEY_BOSS', -1),
            SEMANTIC_PALETTE.get('ITEM_BOMB', -1),
            SEMANTIC_PALETTE.get('KEY_ITEM', -1),
            SEMANTIC_PALETTE.get('ITEM_MINOR', -1),
        )
        
        # Get positions of blocks currently being animated (to skip their normal rendering)
        animating_block_positions = self._get_animating_block_positions()
        
        # Use new renderer for map tiles if available
        if self.renderer:
            for r in range(start_r, end_r):
                for c in range(start_c, end_c):
                    tile_id = self.env.grid[r, c]
                    screen_x = c * self.TILE_SIZE - self.view_offset_x
                    screen_y = r * self.TILE_SIZE - self.view_offset_y
                    
                    # FALLBACK: If position is in collected_items and it's a collectible tile,
                    # render as FLOOR instead (defensive in case grid wasn't updated)
                    if (r, c) in collected_items and tile_id in COLLECTIBLE_TILE_IDS:
                        tile_id = SEMANTIC_PALETTE['FLOOR']
                    
                    # Skip blocks being animated - render FLOOR underneath instead
                    if (r, c) in animating_block_positions and tile_id == SEMANTIC_PALETTE['BLOCK']:
                        tile_id = SEMANTIC_PALETTE['FLOOR']
                    
                    # Use sprite manager (with procedural fallback)
                    tile_surface = self.renderer.sprite_manager.get_tile(tile_id, self.TILE_SIZE)
                    map_surface.blit(tile_surface, (screen_x, screen_y))
                    tiles_drawn += 1
                    # Draw stair sprite overlay if tile is stair
                    if tile_id == SEMANTIC_PALETTE['STAIR'] and getattr(self, 'stair_sprite', None):
                        try:
                            alpha = int(140 + 90 * math.sin(time.time() * 3.0))
                            s = self.stair_sprite.copy()
                            s.set_alpha(max(20, alpha))
                            sx = screen_x + (self.TILE_SIZE - s.get_width()) // 2
                            sy = screen_y + (self.TILE_SIZE - s.get_height()) // 2
                            map_surface.blit(s, (sx, sy))
                        except Exception:
                            pass
        else:
            # Fallback rendering
            for r in range(start_r, end_r):
                for c in range(start_c, end_c):
                    tile_id = self.env.grid[r, c]
                    
                    # FALLBACK: If position is in collected_items and it's a collectible tile,
                    # render as FLOOR instead (defensive in case grid wasn't updated)
                    if (r, c) in collected_items and tile_id in COLLECTIBLE_TILE_IDS:
                        tile_id = SEMANTIC_PALETTE['FLOOR']
                    
                    # Skip blocks being animated - render FLOOR underneath instead
                    if (r, c) in animating_block_positions and tile_id == SEMANTIC_PALETTE['BLOCK']:
                        tile_id = SEMANTIC_PALETTE['FLOOR']
                    
                    img = self.images.get(tile_id, self.images.get(SEMANTIC_PALETTE['FLOOR']))
                    screen_x = c * self.TILE_SIZE - self.view_offset_x
                    screen_y = r * self.TILE_SIZE - self.view_offset_y
                    map_surface.blit(img, (screen_x, screen_y))
                    tiles_drawn += 1
                    # Draw stair sprite overlay for fallback tiles
                    if tile_id == SEMANTIC_PALETTE['STAIR'] and getattr(self, 'stair_sprite', None):
                        try:
                            alpha = int(140 + 90 * math.sin(time.time() * 3.0))
                            s = self.stair_sprite.copy()
                            s.set_alpha(max(20, alpha))
                            sx = screen_x + (self.TILE_SIZE - s.get_width()) // 2
                            sy = screen_y + (self.TILE_SIZE - s.get_height()) // 2
                            map_surface.blit(s, (sx, sy))
                        except Exception:
                            pass
        
        # === RENDER ANIMATED BLOCKS ===
        # Draw blocks that are currently being pushed with smooth interpolation
        try:
            self._render_block_push_animations(map_surface)
        except Exception as e:
            logger.warning('Failed to render block push animations: %s', e)
        
        # Draw heatmap overlay if enabled and we have search data
        if self.show_heatmap and self.search_heatmap:
            max_visits = max(self.search_heatmap.values()) if self.search_heatmap else 1
            for (r, c), visits in self.search_heatmap.items():
                if start_r <= r < end_r and start_c <= c < end_c:
                    # Normalize intensity 0.0 - 1.0
                    intensity = visits / max_visits
                    # Blue (cold) to Red (hot) gradient
                    red = int(255 * intensity)
                    blue = int(255 * (1 - intensity))
                    heat_color = (red, 0, blue, 100)
                    
                    heat_surf = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
                    heat_surf.fill(heat_color)
                    screen_x = c * self.TILE_SIZE - self.view_offset_x
                    screen_y = r * self.TILE_SIZE - self.view_offset_y
                    map_surface.blit(heat_surf, (screen_x, screen_y))

        _render_jps_overlay_helper(
            gui=self,
            map_surface=map_surface,
            start_r=start_r,
            end_r=end_r,
            start_c=start_c,
            end_c=end_c,
            pygame=pygame,
        )
        _render_map_elites_overlay_helper(
            gui=self,
            map_surface=map_surface,
            pygame=pygame,
        )
        
        # Draw solution path whenever a path exists
        show_path = self.auto_path and len(self.auto_path) > 0
        # DEBUG: Log path rendering decision (throttled to avoid spam)
        if not hasattr(self, '_path_render_log_counter'):
            self._path_render_log_counter = 0
        self._path_render_log_counter += 1
        if self._path_render_log_counter % 120 == 1:  # Log every 120 frames (~2 seconds at 60fps)
            logger.debug('DEBUG_RENDER: show_path=%s, auto_path=%s, len=%d, auto_mode=%s, preview_visible=%s',
                         show_path,
                         bool(self.auto_path),
                         len(self.auto_path) if self.auto_path else 0,
                         self.auto_mode,
                         getattr(self, 'preview_overlay_visible', False))
            if self.auto_path and len(self.auto_path) > 0:
                logger.debug('DEBUG_RENDER: Path first=%s, last=%s, step_idx=%d, view_offset=(%d,%d)',
                             self.auto_path[0], self.auto_path[-1],
                             getattr(self, 'auto_step_idx', 0),
                             getattr(self, 'view_offset_x', 0),
                             getattr(self, 'view_offset_y', 0))
        if show_path:
            logger.debug(f"Drawing path overlay: {len(self.auto_path)} points, auto_mode={self.auto_mode}, step_idx={self.auto_step_idx}")
            # FIRST: Draw the FULL planned path as a line (cyan/light blue, behind visited tiles)
            if len(self.auto_path) > 1:
                for i in range(len(self.auto_path) - 1):
                    r1, c1 = self.auto_path[i]
                    r2, c2 = self.auto_path[i + 1]
                    # Convert to screen coordinates (center of each tile)
                    # Note: positions are (row, col) where row=y, col=x
                    x1 = int(c1 * self.TILE_SIZE - self.view_offset_x + self.TILE_SIZE // 2)
                    y1 = int(r1 * self.TILE_SIZE - self.view_offset_y + self.TILE_SIZE // 2)
                    x2 = int(c2 * self.TILE_SIZE - self.view_offset_x + self.TILE_SIZE // 2)
                    y2 = int(r2 * self.TILE_SIZE - self.view_offset_y + self.TILE_SIZE // 2)
                    # Draw future path (cyan) vs visited path (green)
                    if i >= self.auto_step_idx:
                        # Future path - bright cyan with outline for visibility
                        pygame.draw.line(map_surface, (0, 0, 0), (x1, y1), (x2, y2), 5)  # Black outline
                        pygame.draw.line(map_surface, (0, 255, 255), (x1, y1), (x2, y2), 3)  # Cyan fill
                    else:
                        # Visited path - bright green with outline
                        pygame.draw.line(map_surface, (0, 0, 0), (x1, y1), (x2, y2), 6)  # Black outline
                        pygame.draw.line(map_surface, (0, 255, 0), (x1, y1), (x2, y2), 4)  # Green fill
            
            # THIRD: Draw start and end markers for clear visibility
            if len(self.auto_path) >= 1:
                # Start marker (green circle)
                start_r, start_c = self.auto_path[0]
                start_x = int(start_c * self.TILE_SIZE - self.view_offset_x + self.TILE_SIZE // 2)
                start_y = int(start_r * self.TILE_SIZE - self.view_offset_y + self.TILE_SIZE // 2)
                pygame.draw.circle(map_surface, (0, 0, 0), (start_x, start_y), 10)  # Black outline
                pygame.draw.circle(map_surface, (0, 255, 100), (start_x, start_y), 8)  # Green fill
                
                # End/goal marker (red/gold circle)
                end_r, end_c = self.auto_path[-1]
                end_x = int(end_c * self.TILE_SIZE - self.view_offset_x + self.TILE_SIZE // 2)
                end_y = int(end_r * self.TILE_SIZE - self.view_offset_y + self.TILE_SIZE // 2)
                pygame.draw.circle(map_surface, (0, 0, 0), (end_x, end_y), 10)  # Black outline
                pygame.draw.circle(map_surface, (255, 215, 0), (end_x, end_y), 8)  # Gold fill
            
            # FOURTH: Draw tile highlights for visited positions (when animating)
            if self.auto_mode and self.auto_step_idx > 0:
                for i, pos in enumerate(self.auto_path[:self.auto_step_idx + 1]):
                    pr, pc = pos
                    path_surf = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
                    
                    # Check if this position is a recently unlocked door
                    current_time = time.time()
                    is_recent_unlock = pos in self.door_unlock_times and (current_time - self.door_unlock_times[pos]) < 2.0
                    
                    if is_recent_unlock:
                        # Flash effect for recently unlocked doors (yellow/gold)
                        flash_alpha = (math.sin(current_time * 8) + 1) / 2  # 0 to 1
                        alpha = int(150 + 105 * flash_alpha)
                        path_surf.fill((255, 215, 0, alpha))  # Gold
                    else:
                        # Use green with slight gradient based on progress
                        alpha = 40 + int(20 * (i / max(1, len(self.auto_path))))
                        path_surf.fill((0, 255, 0, alpha))
                    
                    screen_x = pc * self.TILE_SIZE - self.view_offset_x
                    screen_y = pr * self.TILE_SIZE - self.view_offset_y
                    map_surface.blit(path_surf, (screen_x, screen_y))
        
        # === GUARANTEED PATH RENDERING ===
        # This ensures the path is ALWAYS visible when auto_path has data,
        # regardless of auto_mode, preview state, or feature flags.
        try:
            self._render_path_GUARANTEED(map_surface)
        except Exception as e:
            logger.warning('_render_path_GUARANTEED failed: %s', e)
        
        # Draw Link (use smooth animation if renderer available)
        if self.renderer and self.renderer.agent_visual_pos:
            # Smooth animated position
            visual_pos = self.renderer.agent_visual_pos
            link_x = int(visual_pos.x * self.TILE_SIZE - self.view_offset_x + 2)
            link_y = int(visual_pos.y * self.TILE_SIZE - self.view_offset_y + 2)
        else:
            # Direct grid position
            pr, pc = self.env.state.position
            link_x = pc * self.TILE_SIZE - self.view_offset_x + 2
            link_y = pr * self.TILE_SIZE - self.view_offset_y + 2
        map_surface.blit(self.link_img, (link_x, link_y))
        
        # Render visual effects on map surface
        if self.effects:
            self.effects.render(map_surface, self.TILE_SIZE, (self.view_offset_x, self.view_offset_y))
        
        # If nothing was drawn (e.g., view region outside map or sizes miscalculated), attempt an auto-fix and show diagnostics
        if tiles_drawn == 0:
            try:
                # Try auto-fit + center once to recover from bad offsets
                if not getattr(self, '_auto_recenter_done', False):
                    logger.info('No tiles drawn Î“Ã‡Ã¶ attempting auto-fit zoom + center')
                    try:
                        self._auto_fit_zoom()
                        self._center_view()
                    except Exception:
                        pass
                    self._auto_recenter_done = True

                diag_font = pygame.font.SysFont('Arial', 18, bold=True)
                diag_text = diag_font.render('No map tiles visible - check zoom/offset', True, (255, 100, 100))
                tx = max(10, (view_w - diag_text.get_width()) // 2)
                ty = max(10, (view_h - diag_text.get_height()) // 2)
                # Draw a semi-opaque box behind message for visibility
                box = pygame.Surface((diag_text.get_width() + 20, diag_text.get_height() + 18), pygame.SRCALPHA)
                box.fill((30, 10, 10, 200))
                map_surface.blit(box, (tx - 10, ty - 9))
                map_surface.blit(diag_text, (tx, ty))

                # Additional diagnostic lines useful for debugging
                small = pygame.font.SysFont('Arial', 12)
                try:
                    map_w = self.env.width if self.env is not None else 0
                    map_h = self.env.height if self.env is not None else 0
                except Exception:
                    map_w = map_h = 0
                diag2 = small.render(f'Tile: {self.TILE_SIZE}px  ViewOffset: ({self.view_offset_x},{self.view_offset_y})', True, (220, 220, 220))
                diag3 = small.render(f'Map: {map_w}x{map_h}  View: {view_w}x{view_h}', True, (200, 200, 200))
                map_surface.blit(diag2, (10, ty + diag_text.get_height() + 8))
                map_surface.blit(diag3, (10, ty + diag_text.get_height() + 24))
            except Exception:
                pass

            # Track consecutive empty frames and try to recover display if persistent
            try:
                self._consecutive_empty_frames = getattr(self, '_consecutive_empty_frames', 0) + 1
                if self._consecutive_empty_frames >= getattr(self, '_empty_frame_recovery_threshold', 8):
                    logger.warning('Detected %d consecutive empty frames Î“Ã‡Ã¶ attempting display reinit', self._consecutive_empty_frames)
                    try:
                        recovered = self._attempt_display_reinit()
                        if recovered:
                            self._show_toast('Recovered display after blank frames', 3.0, 'success')
                            logger.info('Recovered display after empty-frame sequence')
                        else:
                            self._show_toast('Display recovery failed', 4.0, 'error')
                    except Exception:
                        logger.exception('Error during forced display reinit')
                    finally:
                        self._consecutive_empty_frames = 0
            except Exception:
                logger.exception('Failed handling consecutive empty frames counter')
        else:
            # Reset counter when frames are healthy
            try:
                self._consecutive_empty_frames = 0
            except Exception:
                pass

        # Blit map surface to screen
        self.screen.blit(map_surface, (0, 0))

        # Debug overlay removed - was causing yellow/magenta square in corner

        # Draw translucent overlays that may capture clicks so users can see what's on top
        try:
            # Preview overlay (non-modal) indicator
            if getattr(self, 'preview_overlay_visible', False):
                try:
                    logger.debug('Rendering preview overlay (will capture clicks)')
                    ov = pygame.Surface((view_w, view_h), pygame.SRCALPHA)
                    ov.fill((40, 30, 40, 130))
                    self.screen.blit(ov, (0, 0))
                    label = self.big_font.render('PATH PREVIEW (overlay) - captures clicks', True, (255, 220, 120))
                    self.screen.blit(label, (20, view_h//2 - 20))
                except Exception:
                    pass
            # Solver comparison modal
            if getattr(self, 'show_solver_comparison_overlay', False):
                try:
                    logger.debug('Rendering solver comparison modal (captures clicks)')
                    ov2 = pygame.Surface((view_w, view_h), pygame.SRCALPHA)
                    ov2.fill((20, 20, 20, 180))
                    self.screen.blit(ov2, (0, 0))
                    label2 = self.big_font.render('SOLVER COMPARISON - modal', True, (200, 200, 255))
                    self.screen.blit(label2, (20, view_h//2 - 20))
                except Exception:
                    pass
        except Exception:
            pass
        
        # Get current position for display (use actual grid position)
        pr, pc = self.env.state.position
        
        # Draw sidebar background
        sidebar_x = self.screen_w - self.SIDEBAR_WIDTH
        pygame.draw.rect(self.screen, (35, 35, 50), (sidebar_x, 0, self.SIDEBAR_WIDTH, self.screen_h))
        pygame.draw.line(self.screen, (60, 60, 80), (sidebar_x, 0), (sidebar_x, self.screen_h), 2)
        
        # Sidebar content
        y_pos = 10
        y_pos = _render_sidebar_header_inventory_solver_helper(
            gui=self,
            screen=self.screen,
            sidebar_x=sidebar_x,
            y_pos=y_pos,
            map_w=w,
            map_h=h,
            time_module=time,
            math_module=math,
            pygame=pygame,
            logger=logger,
        )

        y_pos = _render_sidebar_status_message_metrics_controls_helper(
            gui=self,
            screen=self.screen,
            sidebar_x=sidebar_x,
            y_pos=y_pos,
            player_row=pr,
            player_col=pc,
            pygame=pygame,
            time_module=time,
            math_module=math,
            semantic_palette=SEMANTIC_PALETTE,
        )
        
        # Render minimap if enabled
        if self.show_minimap:
            self._render_minimap()
        
        # Help overlay
        if self.show_help:
            self._render_help_overlay()
        
        # Path preview dialog (Feature 5) - render on top of everything
        if self.path_preview_mode and self.path_preview_dialog:
            # Render path overlay on map
            try:
                self.path_preview_dialog.render_path_overlay(
                    self.screen,
                    self.TILE_SIZE,
                    self.view_offset_x,
                    self.view_offset_y,
                    self.SIDEBAR_WIDTH,
                    self.HUD_HEIGHT
                )
            except Exception as e:
                logger.warning(f"Failed to render path overlay: {e}")
            
            # Render dialog box
            try:
                self.path_preview_dialog.render(self.screen)
            except Exception as e:
                logger.warning(f"Failed to render path preview dialog: {e}")
        elif getattr(self, 'preview_overlay_visible', False) and getattr(self, 'path_preview_dialog', None):
            # Non-modal overlay: render only the path overlay (no blocking dialog)
            try:
                self.path_preview_dialog.render_path_overlay(
                    self.screen,
                    self.TILE_SIZE,
                    self.view_offset_x,
                    self.view_offset_y,
                    self.SIDEBAR_WIDTH,
                    self.HUD_HEIGHT
                )
            except Exception as e:
                logger.warning(f"Failed to render path overlay (non-modal): {e}")

            # Minimal sidebar preview box with start/dismiss buttons
            try:
                sidebar_x = self.screen_w - self.SIDEBAR_WIDTH
                box_h = 80
                box_y = 120  # fixed area near top of sidebar (below header area)
                box_rect = pygame.Rect(sidebar_x + 10, box_y, self.SIDEBAR_WIDTH - 20, box_h)
                pygame.draw.rect(self.screen, (40, 40, 60), box_rect)
                pygame.draw.rect(self.screen, (100, 150, 255), box_rect, 2)

                # Text details
                font = pygame.font.SysFont('Arial', 14, bold=True)
                small = pygame.font.SysFont('Arial', 12)
                path_len = len(self.auto_path) if getattr(self, 'auto_path', None) else 0
                text1 = font.render(f"Preview: {path_len} steps", True, (200, 200, 255))
                self.screen.blit(text1, (box_rect.x + 8, box_rect.y + 8))

                # Keys info (if available)
                keys_used = getattr(self, 'solver_result', {}).get('keys_used', 0) if getattr(self, 'solver_result', None) else 0
                keys_avail = getattr(self, 'solver_result', {}).get('keys_available', 0) if getattr(self, 'solver_result', None) else 0
                keys_text = f"Keys: {keys_used} / {keys_avail}" if keys_avail > 0 else "Keys: None"
                self.screen.blit(small.render(keys_text, True, (200, 200, 200)), (box_rect.x + 8, box_rect.y + 34))

                # Start & Dismiss buttons
                start_rect = pygame.Rect(box_rect.x + 8, box_rect.y + 48, 140, 24)
                dismiss_rect = pygame.Rect(box_rect.x + 156, box_rect.y + 48, 60, 24)
                pygame.draw.rect(self.screen, (40, 140, 40), start_rect)
                pygame.draw.rect(self.screen, (140, 40, 40), dismiss_rect)
                pygame.draw.rect(self.screen, (100, 255, 100), start_rect, 1)
                pygame.draw.rect(self.screen, (255, 100, 100), dismiss_rect, 1)
                self.sidebar_start_button_rect = start_rect
                self.sidebar_dismiss_button_rect = dismiss_rect

                start_text = small.render("Start Auto-Solve", True, (255, 255, 255))
                dismiss_text = small.render("Dismiss", True, (255, 255, 255))
                self.screen.blit(start_text, (start_rect.x + 8, start_rect.y + 4))
                self.screen.blit(dismiss_text, (dismiss_rect.x + 6, dismiss_rect.y + 4))

            except Exception as e:
                logger.warning(f"Failed to render sidebar preview box: {e}")
        else:
            # Ensure stale sidebar button rects are cleared
            self.sidebar_start_button_rect = None
            self.sidebar_dismiss_button_rect = None

        # Render topology overlay (if enabled via checkbox or feature_flags)
        # Sync feature_flags to instance variable
        if self.feature_flags.get('show_topology', False):
            self.show_topology = True
        if getattr(self, 'show_topology', False):
            try:
                logger.debug("Rendering topology overlay")
                self._render_topology_overlay(self.screen)
            except Exception as e:
                logger.warning(f"Topology overlay failed: {e}")

        # Render solver comparison overlay (if available)
        if getattr(self, 'show_solver_comparison_overlay', False):
            try:
                self._render_solver_comparison_overlay(self.screen)
            except Exception as e:
                logger.warning(f"Solver comparison overlay failed: {e}")
        
        # Render control panel
        if self.control_panel_enabled:
            self._render_control_panel(self.screen)

        # Render developer debug overlay (toggle with F12)
        if getattr(self, 'debug_overlay_enabled', False):
            try:
                self._render_debug_overlay(self.screen)
            except Exception as e:
                logger.warning(f"Debug overlay render failed: {e}")
        
        # Render item legend
        if self.auto_mode:
            self._render_item_legend(self.screen)
        
        # Render error banner (on top of everything)
        self._render_error_banner(self.screen)
        
        # Render solver status banner (shows algorithm being used)
        self._render_solver_status_banner(self.screen)
        
        # Render toast notifications (on top of everything)
        self._render_toasts(self.screen)
        
        # NOTE: pygame.display.flip() is called by the main run() loop after _render()
        # Do NOT call flip() here to avoid double-buffer swap issues

    def _render_debug_overlay(self, surface):
        """Render debug overlay with mouse coords, widget rects, and recent clicks.
        Toggle with F12. Shift-F11 clears click log.
        """
        _render_debug_overlay_helper(self, surface, pygame, time)

    def _render_unified_bottom_panel(self):
        """Render unified bottom HUD panel - STATUS and MESSAGE only (inventory moved to sidebar)."""
        _render_unified_bottom_panel_helper(self, pygame)
    
    def _render_message_section(self, x: int, y: int, width: int, height: int):
        """Render message/status section in bottom panel."""
        _render_message_section_helper(self, x, y, width, height)
    
    def _render_progress_bar(self, surface, x: int, y: int, width: int, height: int, 
                             filled: int, total: int, color_filled: tuple, color_empty: tuple):
        """Render a segmented progress bar with filled/empty indicators."""
        _render_progress_bar_helper(
            surface,
            x,
            y,
            width,
            height,
            filled,
            total,
            color_filled,
            color_empty,
            pygame,
        )
    
    def _render_inventory_section(self, x: int, y: int, width: int, height: int):
        """Render inventory section with progress bars and icons."""
        _render_inventory_section_helper(self, x, y, width, height, pygame, time, logger)
    
    def _render_metrics_section(self, x: int, y: int, width: int, height: int):
        """Render metrics section (steps, speed, zoom, env)."""
        _render_metrics_section_helper(self, x, y, width, height)
    
    def _render_controls_section(self, x: int, y: int, width: int, height: int):
        """Render controls section in two-column layout."""
        _render_controls_section_helper(self, x, y, width, height)
    
    def _render_status_section(self, x: int, y: int, width: int, height: int):
        """Render status section with game state information."""
        _render_status_section_helper(self, x, y, width, height)
    
    def _render_minimap(self):
        """Render small dungeon overview map in bottom-right corner."""
        _render_minimap_helper(self, pygame)
    
    def _handle_minimap_click(self, mouse_pos: Tuple[int, int]) -> bool:
        """Handle mouse click on minimap to jump to that location."""
        return _handle_minimap_click_helper(self, mouse_pos)
    
    def _render_help_overlay(self):
        """Render help overlay."""
        _render_help_overlay_helper(self, pygame)


def load_maps_from_adapter():
    """Load processed maps from data adapter using new zelda_core - ALL 18 variants."""
    try:
        from src.data.zelda_core import ZeldaDungeonAdapter, DungeonSolver
        from pathlib import Path
        
        data_root = Path(__file__).parent / "Data" / "The Legend of Zelda"
        
        if not data_root.exists():
            print(f"Data folder not found: {data_root}")
            return None, None
        
        adapter = ZeldaDungeonAdapter(str(data_root))
        solver = DungeonSolver()
        
        maps = []  # Store full StitchedDungeon objects
        map_names = []  # Track dungeon names
        print("Loading all 18 dungeon variants (9 dungeons x 2 variants)...")
        
        for dungeon_num in range(1, 10):
            for variant in [1, 2]:
                try:
                    dungeon = adapter.load_dungeon(dungeon_num, variant=variant)
                    stitched = adapter.stitch_dungeon(dungeon)

                    # Store the full stitched dungeon (includes graph and room mappings)
                    maps.append(stitched)

                    # Store name
                    quest_name = "Quest 1" if variant == 1 else "Quest 2"
                    map_names.append(f"Dungeon {dungeon_num} ({quest_name})")

                    # Fast startup: do not block on expensive solvability checks here.
                    print(f"  D{dungeon_num}-{variant}: Loaded - {stitched.global_grid.shape}")
                except Exception as e:
                    print(f"  D{dungeon_num}-{variant}: Error - {e}")

        # If requested, perform precalculation asynchronously so startup is not blocked
        if os.environ.get('KLTN_PRECALC_SOLVES', '0') == '1':
            try:
                import threading
                def _precalc_worker():
                    print('Starting background precalc solves for loaded maps...')
                    for idx, m in enumerate(maps):
                        try:
                            r = solver.solve(m)
                            status = '[OK]' if r.get('solvable') else '[X]'
                            print(f"  [precalc] Map {idx+1}: {status}")
                        except Exception as e:
                            print(f"  [precalc] Map {idx+1}: Error - {e}")
                threading.Thread(target=_precalc_worker, daemon=True).start()
            except Exception:
                print('Precalc worker failed to start')
        
        return maps if maps else None, map_names if map_names else None
        
    except Exception as e:
        print(f"Error loading maps: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """Main entry point."""
    print("=== ZAVE GUI Runner ===\n")
    
    if not PYGAME_AVAILABLE:
        print("Pygame is not installed. Please run: pip install pygame")
        return
    
    # Try to load processed maps
    maps, map_names = load_maps_from_adapter()
    
    if maps:
        print(f"Loaded {len(maps)} maps from data adapter")
    else:
        print("Using test map")
        maps = [create_test_map()]
        map_names = ["Test Map"]
    
    # Start GUI
    gui = ZeldaGUI(maps, map_names)
    gui.run()


if __name__ == "__main__":
    # Required for multiprocessing on Windows (freeze_support)
    multiprocessing.freeze_support()
    main()



