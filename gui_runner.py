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
from src.gui.app.run_completion_handlers import (
    handle_ai_generation_completion,
    handle_parallel_search_completion,
    handle_preview_process_completion,
    handle_solver_process_completion,
)
from src.gui.app.frame_loop_handlers import (
    advance_frame_and_check_limit,
    handle_pending_solver_trigger,
    handle_watchdog_screenshot_request,
    render_and_present_frame,
    run_periodic_display_health_check,
    tick_frame_clock,
    update_heartbeat,
)
from src.gui.app.event_loop_handlers import (
    clear_stale_preview_overlay,
    handle_global_keydown_shortcuts,
    handle_keydown_event,
    handle_keyup_event,
    handle_mouse_button_down_preamble,
    handle_mouse_button_down_event,
    handle_mouse_button_up_event,
    handle_mouse_motion_diagnostics,
    handle_mouse_motion_event,
    handle_mousewheel_event,
    handle_preview_overlay_events,
    handle_videoresize_event,
    handle_window_focus_event,
    poll_pygame_events,
    run_input_focus_fallback,
)
from src.gui.app.init_bootstrap import (
    configure_windows_dpi_awareness,
    ensure_repo_export_dirs,
    initialize_pygame_runtime,
)
from src.gui.app.init_display_setup import initialize_display_window
from src.gui.app.init_runtime_watchdog import initialize_runtime_timing_state
from src.gui.app.init_solver_state import initialize_solver_execution_state
from src.gui.app.init_ui_state import initialize_ui_control_state
from src.gui.app.init_visualization import (
    initialize_debug_test_path,
    initialize_visualization_components,
)
from src.gui.app.init_final_boot import finalize_initial_map_boot
from src.gui.app.asset_boot_orchestration import (
    create_link_sprite as _create_link_sprite_orchestration_helper,
    init_control_panel as _init_control_panel_orchestration_helper,
    load_assets as _load_assets_orchestration_helper,
)
from src.gui.app.gui_startup import run_gui_main as _run_gui_main_helper
from src.gui.app.map_adapter_loader import load_maps_from_adapter as _load_maps_from_adapter_helper
from src.gui.app.run_loop_pipeline import run_main_loop as _run_main_loop_helper
from src.gui.app.entrypoint_orchestration import (
    load_maps_from_adapter as _load_maps_from_adapter_orchestration_helper,
    run_main_entry as _run_main_entry_orchestration_helper,
)
from src.gui.app.runtime_loop_orchestration import run as _run_orchestration_helper

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
from src.gui.control_panel.click_dispatch import (
    handle_control_panel_click as _handle_control_panel_click_dispatch_helper,
)
from src.gui.control_panel.click_render_orchestration import (
    apply_algorithm_dropdown_update as _apply_algorithm_dropdown_update_orchestration_helper,
    apply_checkbox_widget_update as _apply_checkbox_widget_update_orchestration_helper,
    apply_control_panel_widget_updates as _apply_control_panel_widget_updates_orchestration_helper,
    apply_dropdown_widget_update as _apply_dropdown_widget_update_orchestration_helper,
    control_panel_hit_rect as _control_panel_hit_rect_orchestration_helper,
    draw_tooltip as _draw_tooltip_orchestration_helper,
    handle_control_panel_click as _handle_control_panel_click_orchestration_helper,
    handle_outside_control_panel_click as _handle_outside_control_panel_click_orchestration_helper,
    refresh_control_panel_layout_if_needed as _refresh_control_panel_layout_if_needed_orchestration_helper,
    render_control_panel as _render_control_panel_orchestration_helper,
    render_tooltips as _render_tooltips_orchestration_helper,
    retry_control_panel_click_after_auto_scroll as _retry_control_panel_click_after_auto_scroll_orchestration_helper,
    should_swallow_control_panel_click as _should_swallow_control_panel_click_orchestration_helper,
    translate_control_panel_click as _translate_control_panel_click_orchestration_helper,
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
from src.gui.solver.session_orchestration import (
    cleanup_preview_before_solver_start as _cleanup_preview_before_solver_start_orchestration_helper,
    clear_solver_state as _clear_solver_state_orchestration_helper,
    compute_solver_timeout_seconds as _compute_solver_timeout_seconds_orchestration_helper,
    create_solver_temp_files as _create_solver_temp_files_orchestration_helper,
    force_solver_recovery_state as _force_solver_recovery_state_orchestration_helper,
    log_active_solver_state as _log_active_solver_state_orchestration_helper,
    prepare_active_solver_for_new_start as _prepare_active_solver_for_new_start_orchestration_helper,
    reset_solver_visual_state_before_start as _reset_solver_visual_state_before_start_orchestration_helper,
    run_solver_sync as _run_solver_sync_orchestration_helper,
    start_auto_solve as _start_auto_solve_orchestration_helper,
    sync_solver_dropdown_settings as _sync_solver_dropdown_settings_orchestration_helper,
    terminate_hung_solver_process as _terminate_hung_solver_process_orchestration_helper,
)
from src.gui.solver.launch_orchestration import (
    launch_solver_process as _launch_solver_process_orchestration_helper,
    launch_solver_worker as _launch_solver_worker_orchestration_helper,
    schedule_solver as _schedule_solver_orchestration_helper,
    solver_thread_fallback_worker as _solver_thread_fallback_worker_orchestration_helper,
    start_preview_for_current_map as _start_preview_for_current_map_orchestration_helper,
    start_solver_thread_fallback as _start_solver_thread_fallback_orchestration_helper,
)
from src.gui.runtime.watchdog_monitor import watchdog_loop as _watchdog_loop_helper
from src.gui.runtime.route_io import (
    export_route as _export_route_helper,
    load_route as _load_route_helper,
)
from src.gui.runtime.route_orchestration import (
    export_route as _export_route_orchestration_helper,
    load_route as _load_route_orchestration_helper,
)
from src.gui.gameplay.path_controls import (
    reset_map as _reset_map_helper,
    show_path_preview as _show_path_preview_helper,
    clear_path as _clear_path_helper,
)
from src.gui.gameplay.control_actions_orchestration import (
    clear_path as _clear_path_orchestration_helper,
    reset_map as _reset_map_orchestration_helper,
    run_ai_dungeon_generation_worker as _run_ai_dungeon_generation_worker_orchestration_helper,
    show_path_preview as _show_path_preview_orchestration_helper,
    start_ai_dungeon_generation as _start_ai_dungeon_generation_orchestration_helper,
)
from src.gui.gameplay.dungeon_generation_controls import (
    generate_dungeon as _generate_dungeon_flow_helper,
    stop_auto_solve as _stop_auto_solve_flow_helper,
)
from src.gui.gameplay.dungeon_generation_orchestration import (
    generate_dungeon as _generate_dungeon_orchestration_helper,
    stop_auto_solve as _stop_auto_solve_orchestration_helper,
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
from src.gui.rendering.status_toast_orchestration import (
    format_cbs_metrics_tooltip as _format_cbs_metrics_tooltip_orchestration_helper,
    render_error_banner as _render_error_banner_orchestration_helper,
    render_solver_status_banner as _render_solver_status_banner_orchestration_helper,
    render_status_bar as _render_status_bar_orchestration_helper,
    render_toasts as _render_toasts_orchestration_helper,
    set_message as _set_message_orchestration_helper,
    show_error as _show_error_orchestration_helper,
    show_message as _show_message_orchestration_helper,
    show_toast as _show_toast_orchestration_helper,
    show_warning as _show_warning_orchestration_helper,
    update_toasts as _update_toasts_orchestration_helper,
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
from src.gui.map.navigation_orchestration import (
    auto_fit_zoom as _auto_fit_zoom_orchestration_helper,
    center_on_player as _center_on_player_orchestration_helper,
    center_view as _center_view_orchestration_helper,
    change_zoom as _change_zoom_orchestration_helper,
    clamp_view_offset as _clamp_view_offset_orchestration_helper,
    handle_minimap_click as _handle_minimap_click_orchestration_helper,
    load_current_map as _load_current_map_orchestration_helper,
    next_map as _next_map_orchestration_helper,
    prev_map as _prev_map_orchestration_helper,
    render_minimap as _render_minimap_orchestration_helper,
)
from src.gui.gameplay.block_push_controls import (
    start_block_push_animation as _start_block_push_animation_helper,
    update_block_push_animations as _update_block_push_animations_helper,
    render_block_push_animations as _render_block_push_animations_helper,
    get_animating_block_positions as _get_animating_block_positions_helper,
    check_and_start_block_push as _check_and_start_block_push_helper,
)
from src.gui.rendering.help_overlay import render_help_overlay as _render_help_overlay_helper
from src.gui.rendering.panel_overlay_orchestration import (
    render_controls_section as _render_controls_section_orchestration_helper,
    render_debug_overlay as _render_debug_overlay_orchestration_helper,
    render_help_overlay as _render_help_overlay_orchestration_helper,
    render_inventory_section as _render_inventory_section_orchestration_helper,
    render_message_section as _render_message_section_orchestration_helper,
    render_metrics_section as _render_metrics_section_orchestration_helper,
    render_progress_bar as _render_progress_bar_orchestration_helper,
    render_status_section as _render_status_section_orchestration_helper,
    render_unified_bottom_panel as _render_unified_bottom_panel_orchestration_helper,
)
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
from src.gui.topology.helper_orchestration import (
    build_room_adjacency_from_graph as _build_room_adjacency_from_graph_orchestration_helper,
    export_topology as _export_topology_orchestration_helper,
    node_has_critical_content as _node_has_critical_content_orchestration_helper,
    node_has_small_key as _node_has_small_key_orchestration_helper,
    room_for_global_position as _room_for_global_position_orchestration_helper,
)
from src.gui.topology.precheck import (
    prune_dead_end_topology as _prune_dead_end_topology_flow_helper,
    run_prechecks_and_optional_prune as _run_prechecks_and_optional_prune_flow_helper,
    undo_prune as _undo_prune_flow_helper,
)
from src.gui.topology.orchestration import (
    capture_precheck_snapshot as _capture_precheck_snapshot_orchestration_helper,
    prune_dead_end_topology as _prune_dead_end_topology_orchestration_helper,
    render_topology_overlay as _render_topology_overlay_orchestration_helper,
    run_prechecks_and_optional_prune as _run_prechecks_and_optional_prune_orchestration_helper,
    undo_prune as _undo_prune_orchestration_helper,
    update_env_topology_view as _update_env_topology_view_orchestration_helper,
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
from src.gui.runtime.display_orchestration import (
    attempt_display_reinit as _attempt_display_reinit_orchestration_helper,
    ensure_display_alive as _ensure_display_alive_orchestration_helper,
    force_focus as _force_focus_orchestration_helper,
    handle_watchdog_screenshot as _handle_watchdog_screenshot_orchestration_helper,
    report_ui_state as _report_ui_state_orchestration_helper,
    safe_set_mode as _safe_set_mode_orchestration_helper,
    toggle_fullscreen as _toggle_fullscreen_orchestration_helper,
    watchdog_loop as _watchdog_loop_orchestration_helper,
)
from src.gui.control_panel.animation import (
    start_toggle_panel_animation as _start_toggle_panel_animation_helper,
    update_control_panel_animation as _update_control_panel_animation_helper,
)
from src.gui.control_panel.animation_orchestration import (
    start_toggle_panel_animation as _start_toggle_panel_animation_orchestration_helper,
    update_control_panel_animation as _update_control_panel_animation_orchestration_helper,
    update_control_panel_scroll as _update_control_panel_scroll_orchestration_helper,
)
from src.gui.control_panel.scroll import update_control_panel_scroll as _update_control_panel_scroll_helper
from src.gui.control_panel.view import (
    dump_control_panel_widget_state as _dump_control_panel_widget_state_helper,
    render_control_panel as _render_control_panel_helper,
    reposition_widgets as _reposition_widgets_helper,
    update_control_panel_positions as _update_control_panel_positions_helper,
)
from src.gui.control_panel.layout_orchestration import (
    dump_control_panel_widget_state as _dump_control_panel_widget_state_orchestration_helper,
    reposition_widgets as _reposition_widgets_orchestration_helper,
    update_control_panel_positions as _update_control_panel_positions_orchestration_helper,
)
from src.gui.gameplay.inventory_manager import (
    update_inventory_and_hud as _update_inventory_and_hud_helper,
    remove_from_path_items as _remove_from_path_items_helper,
    track_item_collection as _track_item_collection_helper,
    track_item_usage as _track_item_usage_helper,
    sync_inventory_counters as _sync_inventory_counters_helper,
)
from src.gui.gameplay.inventory_orchestration import (
    apply_pickup_at as _apply_pickup_at_orchestration_helper,
    get_path_items_display_text as _get_path_items_display_text_orchestration_helper,
    remove_from_path_items as _remove_from_path_items_orchestration_helper,
    render_item_legend as _render_item_legend_orchestration_helper,
    scan_and_mark_items as _scan_and_mark_items_orchestration_helper,
    scan_items_along_path as _scan_items_along_path_orchestration_helper,
    sync_inventory_counters as _sync_inventory_counters_orchestration_helper,
    track_item_collection as _track_item_collection_orchestration_helper,
    track_item_usage as _track_item_usage_orchestration_helper,
    update_inventory_and_hud as _update_inventory_and_hud_orchestration_helper,
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
from src.gui.gameplay.action_orchestration import (
    auto_step as _auto_step_orchestration_helper,
    check_and_start_block_push as _check_and_start_block_push_orchestration_helper,
    execute_auto_solve as _execute_auto_solve_orchestration_helper,
    execute_auto_solve_from_preview as _execute_auto_solve_from_preview_orchestration_helper,
    get_animating_block_positions as _get_animating_block_positions_orchestration_helper,
    graph_guided_path as _graph_guided_path_orchestration_helper,
    hybrid_graph_grid_path as _hybrid_graph_grid_path_orchestration_helper,
    manual_step as _manual_step_orchestration_helper,
    render_block_push_animations as _render_block_push_animations_orchestration_helper,
    smart_grid_path as _smart_grid_path_orchestration_helper,
    start_block_push_animation as _start_block_push_animation_orchestration_helper,
    stop_auto as _stop_auto_orchestration_helper,
    update_block_push_animations as _update_block_push_animations_orchestration_helper,
)
from src.gui.gameplay.manual_step_controller import manual_step as _manual_step_flow_helper
from src.gui.rendering.path_guaranteed_renderer import (
    render_path_guaranteed as _render_path_guaranteed_flow_helper,
)
from src.gui.rendering.map_render_pipeline import (
    collect_item_render_state as _collect_item_render_state_helper,
    compute_visible_bounds as _compute_visible_bounds_helper,
    create_map_surface as _create_map_surface_helper,
    render_heatmap_overlay as _render_heatmap_overlay_helper,
    render_visible_tiles as _render_visible_tiles_helper,
)
from src.gui.rendering.path_overlay_pipeline import (
    render_planned_path_overlay as _render_planned_path_overlay_helper,
)
from src.gui.rendering.overlay_ui_pipeline import (
    render_preview_layer as _render_preview_layer_helper,
    render_translucent_event_overlays as _render_translucent_event_overlays_helper,
)
from src.gui.rendering.render_diagnostics_pipeline import (
    handle_empty_frame_recovery as _handle_empty_frame_recovery_helper,
)
from src.gui.rendering.frame_state_pipeline import (
    render_player_and_effects as _render_player_and_effects_helper,
    update_frame_render_state as _update_frame_render_state_helper,
)
from src.gui.rendering.post_map_ui_pipeline import (
    draw_sidebar_shell as _draw_sidebar_shell_helper,
    render_post_map_layers as _render_post_map_layers_helper,
    render_sidebar_content as _render_sidebar_content_helper,
    render_top_ui_layers as _render_top_ui_layers_helper,
)
from src.gui.rendering.render_frame_pipeline import render_frame as _render_frame_helper
from src.gui.rendering.frame_orchestration import (
    render_frame as _render_frame_orchestration_helper,
    render_path_guaranteed as _render_path_guaranteed_orchestration_helper,
)
from src.gui.rendering.tile_asset_builder import (
    build_stair_marker_sprite as _build_stair_marker_sprite_helper,
    build_tile_images as _build_tile_images_helper,
    default_tile_color_map as _default_tile_color_map_helper,
)
from src.gui.rendering.link_sprite_builder import build_link_sprite as _build_link_sprite_helper
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
from src.gui.topology.match_orchestration import (
    apply_tentative_matches as _apply_tentative_matches_orchestration_helper,
    match_missing_nodes as _match_missing_nodes_orchestration_helper,
    undo_last_match as _undo_last_match_orchestration_helper,
)
from src.gui.solver.comparison_runner import (
    run_solver_comparison as _run_solver_comparison_helper,
    set_last_solver_metrics as _set_last_solver_metrics_helper,
)
from src.gui.solver.comparison_orchestration import (
    map_elites_worker as _map_elites_worker_orchestration_helper,
    render_solver_comparison_overlay as _render_solver_comparison_overlay_orchestration_helper,
    run_solver_comparison as _run_solver_comparison_orchestration_helper,
    set_last_solver_metrics as _set_last_solver_metrics_orchestration_helper,
    start_map_elites as _start_map_elites_orchestration_helper,
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
from src.gui.solver.process_api_orchestration import (
    convert_diagonal_to_4dir as _convert_diagonal_to_4dir_orchestration_helper,
    run_preview_and_dump as _run_preview_and_dump_orchestration_helper,
    run_solver_and_dump as _run_solver_and_dump_orchestration_helper,
    safe_unpickle as _safe_unpickle_orchestration_helper,
    solve_in_subprocess as _solve_in_subprocess_orchestration_helper,
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
from src.gui.map.asset_orchestration import (
    load_visual_assets as _load_visual_assets_orchestration_helper,
    load_visual_map as _load_visual_map_orchestration_helper,
    place_items_from_graph as _place_items_from_graph_orchestration_helper,
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
    return _safe_unpickle_orchestration_helper(
        path=path,
        safe_unpickle_helper=_safe_unpickle_helper,
    )


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
    return _convert_diagonal_to_4dir_orchestration_helper(
        path=path,
        grid=grid,
        convert_diagonal_to_4dir_helper=_convert_diagonal_to_4dir_helper,
    )

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
    return _solve_in_subprocess_orchestration_helper(
        grid=grid,
        start_pos=start_pos,
        goal_pos=goal_pos,
        algorithm_idx=algorithm_idx,
        feature_flags=feature_flags,
        priority_options=priority_options,
        graph=graph,
        room_to_node=room_to_node,
        room_positions=room_positions,
        node_to_room=node_to_room,
        solve_in_subprocess_helper=_solve_in_subprocess_helper,
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
    return _run_solver_and_dump_orchestration_helper(
        grid_or_path=grid_or_path,
        start_pos=start_pos,
        goal_pos=goal_pos,
        algorithm_idx=algorithm_idx,
        feature_flags=feature_flags,
        priority_options=priority_options,
        out_path=out_path,
        graph=graph,
        room_to_node=room_to_node,
        room_positions=room_positions,
        node_to_room=node_to_room,
        run_solver_and_dump_helper=_run_solver_and_dump_helper,
    )


def _run_preview_and_dump(grid_or_path, start_pos, goal_pos, algorithm_idx, feature_flags, priority_options, out_path,
                          graph=None, room_to_node=None, room_positions=None, node_to_room=None):
    """Lightweight preview runner that writes a short preview result quickly.

    Runs in a separate process to avoid blocking the GUI. Attempts a fast StateSpaceAStar
    with a small timeout or returns failure quickly.
    """
    return _run_preview_and_dump_orchestration_helper(
        grid_or_path=grid_or_path,
        start_pos=start_pos,
        goal_pos=goal_pos,
        algorithm_idx=algorithm_idx,
        feature_flags=feature_flags,
        priority_options=priority_options,
        out_path=out_path,
        graph=graph,
        room_to_node=room_to_node,
        room_positions=room_positions,
        node_to_room=node_to_room,
        run_preview_and_dump_helper=_run_preview_and_dump_helper,
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

        ensure_repo_export_dirs(gui=self, path_cls=Path, logger=logger)
        configure_windows_dpi_awareness(logger=logger)
        initialize_pygame_runtime(pygame=pygame, logger=logger)
        initialize_display_window(gui=self, pygame=pygame, os_module=os, logger=logger)
        initialize_runtime_timing_state(
            gui=self,
            pygame=pygame,
            os_module=os,
            time_module=time,
            threading_module=threading,
            logger=logger,
        )
        
        initialize_visualization_components(
            gui=self,
            visualization_available=VISUALIZATION_AVAILABLE,
            renderer_cls=ZeldaRenderer,
            effects_cls=EffectManager,
            hud_cls=ModernHUD,
        )
        # Load assets (fallback for when new system unavailable)
        self._load_assets()

        initialize_solver_execution_state(gui=self, threading_module=threading)

        initialize_debug_test_path(gui=self, os_module=os)

        initialize_ui_control_state(
            gui=self,
            pygame=pygame,
            widgets_available=WIDGETS_AVAILABLE,
            os_module=os,
            time_module=time,
        )
        
        finalize_initial_map_boot(gui=self, pygame=pygame, logger=logger)

    
    def _load_assets(self):
        """Load tile images - using colored squares for reliability."""
        _load_assets_orchestration_helper(
            gui=self,
            semantic_palette=SEMANTIC_PALETTE,
            pygame=pygame,
            default_tile_color_map_helper=_default_tile_color_map_helper,
            build_tile_images_helper=_build_tile_images_helper,
            build_stair_marker_sprite_helper=_build_stair_marker_sprite_helper,
        )
    

    def _create_link_sprite(self):
        """Create a detailed Link sprite using pygame drawing."""
        return _create_link_sprite_orchestration_helper(
            tile_size=self.TILE_SIZE,
            pygame=pygame,
            build_link_sprite_helper=_build_link_sprite_helper,
        )
    
    def _init_control_panel(self):
        """Initialize the GUI control panel with widgets."""
        _init_control_panel_orchestration_helper(
            gui=self,
            widgets_available=WIDGETS_AVAILABLE,
            widget_manager_cls=WidgetManager,
        )
    
    def _update_control_panel_positions(self):
        """Update control panel and widget positions (called on resize)."""
        _update_control_panel_positions_orchestration_helper(
            gui=self,
            pygame=pygame,
            logger=logger,
            widgets_available=WIDGETS_AVAILABLE,
            checkbox_widget_cls=CheckboxWidget,
            dropdown_widget_cls=DropdownWidget,
            button_widget_cls=ButtonWidget,
            zoom_labels=GUI_ZOOM_LABELS,
            difficulty_names=GUI_DIFFICULTY_NAMES,
            algorithm_names=GUI_ALGORITHM_NAMES,
            update_control_panel_positions_helper=_update_control_panel_positions_helper,
        )

    def _reposition_widgets(self, panel_x: int, panel_y: int):
        """Reposition existing widgets when panel is dragged (without rebuilding)."""
        _reposition_widgets_orchestration_helper(
            gui=self,
            panel_x=panel_x,
            panel_y=panel_y,
            checkbox_widget_cls=CheckboxWidget,
            dropdown_widget_cls=DropdownWidget,
            button_widget_cls=ButtonWidget,
            reposition_widgets_helper=_reposition_widgets_helper,
        )

    def _dump_control_panel_widget_state(self, mouse_pos: tuple):
        """Debug helper: log each widget rects and whether mouse/sc_pos hit them.

        This is defensive and avoids using any variables that may not be available in
        other layout helper contexts.
        """
        _dump_control_panel_widget_state_orchestration_helper(
            gui=self,
            mouse_pos=mouse_pos,
            logger=logger,
            debug_input_active=DEBUG_INPUT_ACTIVE,
            dump_control_panel_widget_state_helper=_dump_control_panel_widget_state_helper,
        )
        
    
    def _update_inventory_and_hud(self):
        """Reconcile counters and update the modern HUD (if present).

        This centralizes synchronization so any pickup/usage path calls the same routine.
        If called from a non-main thread, set a flag so the main thread performs the UI update
        (pygame surfaces & rendering should be touched only from the main thread).
        """
        _update_inventory_and_hud_orchestration_helper(
            gui=self,
            logger=logger,
            update_inventory_and_hud_helper=_update_inventory_and_hud_helper,
        )

    def _remove_from_path_items(self, pos, item_type):
        """Remove a collected item from path_item_positions and update summary.
        
        Args:
            pos: (row, col) position of collected item
            item_type: 'keys', 'boss_keys', 'ladders', 'bombs', etc.
        """
        _remove_from_path_items_orchestration_helper(
            gui=self,
            pos=pos,
            item_type=item_type,
            logger=logger,
            remove_from_path_items_helper=_remove_from_path_items_helper,
        )

    def _track_item_collection(self, old_state, new_state):
        """Detect when items are collected by comparing states."""
        _track_item_collection_orchestration_helper(
            gui=self,
            old_state=old_state,
            new_state=new_state,
            time_module=time,
            logger=logger,
            pop_effect_cls=PopEffect,
            item_collection_effect_cls=ItemCollectionEffect,
            track_item_collection_helper=_track_item_collection_helper,
        )
    
    def _track_item_usage(self, old_state, new_state):
        """Detect when items are used (doors opened, walls bombed)."""
        _track_item_usage_orchestration_helper(
            gui=self,
            old_state=old_state,
            new_state=new_state,
            time_module=time,
            logger=logger,
            item_usage_effect_cls=ItemUsageEffect,
            track_item_usage_helper=_track_item_usage_helper,
        )
    
    def _scan_and_mark_items(self):
        """Scan the map for all items and create markers.
        
        This populates item_type_map with all item positions so that
        _sync_inventory_counters() can correctly count collected items.
        """
        _scan_and_mark_items_orchestration_helper(
            gui=self,
            semantic_palette=SEMANTIC_PALETTE,
            logger=logger,
            item_marker_effect_cls=ItemMarkerEffect,
            scan_and_mark_items_helper=_scan_and_mark_items_helper,
        )

    def _apply_pickup_at(self, pos: Tuple[int, int]) -> bool:
        """Apply pickup logic at a position for teleport landings or external mutations.

        This mutates self.env.state to include the collected item and updates
        visual markers/effects and pickup timers. Returns True if an item was
        collected at the position.
        """
        return _apply_pickup_at_orchestration_helper(
            gui=self,
            pos=pos,
            semantic_palette=SEMANTIC_PALETTE,
            logger=logger,
            time_module=time,
            item_collection_effect_cls=ItemCollectionEffect,
            apply_pickup_at_helper=_apply_pickup_at_helper,
        )
    
    def _render_item_legend(self, surface):
        """Render legend showing item counts and path items preview."""
        _render_item_legend_orchestration_helper(
            gui=self,
            surface=surface,
            pygame=pygame,
            render_item_legend_helper=_render_item_legend_helper,
        )

    def _sync_inventory_counters(self):
        """Reconcile counters from collected_items and env.state to ensure UI accuracy.

        Uses multiple sources for robustness:
        1. self.collected_items list (primary - actively maintained by _track_item_collection)
        2. self.env.state.collected_items + item_type_map (backup)
        
        This ensures real-time updates work correctly during auto-solve.
        """
        _sync_inventory_counters_orchestration_helper(
            gui=self,
            sync_inventory_counters_helper=_sync_inventory_counters_helper,
        )

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
        return _scan_items_along_path_orchestration_helper(
            gui=self,
            semantic_palette=SEMANTIC_PALETTE,
            logger=logger,
            path=path,
            scan_items_along_path_helper=_scan_items_along_path_helper,
        )

    def _get_path_items_display_text(self):
        """Generate a display string summarizing items along the path.
        
        Returns:
            str: Human-readable summary like "Path: 3 keys, 2 doors, 1 boss key"
        """
        return _get_path_items_display_text_orchestration_helper(
            gui=self,
            get_path_items_display_text_helper=_get_path_items_display_text_helper,
        )
    
    def _render_error_banner(self, surface):
        """Render error message banner at top of screen with fade effect."""
        _render_error_banner_orchestration_helper(
            gui=self,
            surface=surface,
            pygame=pygame,
            time_module=time,
            render_error_banner_helper=_render_error_banner_helper,
        )
    
    def _render_solver_status_banner(self, surface):
        """Render solver status banner showing current algorithm and progress."""
        _render_solver_status_banner_orchestration_helper(
            gui=self,
            surface=surface,
            pygame=pygame,
            math_module=math,
            time_module=time,
            logger=logger,
            render_solver_status_banner_helper=_render_solver_status_banner_helper,
        )
    
    def _render_status_bar(self, surface):
        """Render status bar at bottom of screen."""
        _render_status_bar_orchestration_helper(
            gui=self,
            surface=surface,
            pygame=pygame,
            render_status_bar_helper=_render_status_bar_helper,
        )
    
    def _render_control_panel(self, surface):
        """Render the control panel with all GUI widgets and metrics."""
        _render_control_panel_orchestration_helper(
            gui=self,
            surface=surface,
            pygame=pygame,
            logger=logger,
            dropdown_widget_cls=DropdownWidget,
            render_control_panel_helper=_render_control_panel_helper,
        )

    def _render_tooltips(self, surface, mouse_pos):
        """Render tooltips for widgets under mouse cursor."""
        _render_tooltips_orchestration_helper(
            gui=self,
            surface=surface,
            mouse_pos=mouse_pos,
            button_widget_cls=ButtonWidget,
            pygame=pygame,
            render_tooltips_helper=_render_tooltips_helper,
        )
    
    def _draw_tooltip(self, surface, pos, text):
        """Draw a tooltip box at the specified position."""
        _draw_tooltip_orchestration_helper(
            gui=self,
            surface=surface,
            pos=pos,
            text=text,
            pygame=pygame,
            draw_tooltip_helper=_draw_tooltip_helper,
        )
    
    def _handle_control_panel_click(self, pos, button, event_type='down'):
        """Handle mouse clicks on control panel widgets."""
        return _handle_control_panel_click_orchestration_helper(
            gui=self,
            pos=pos,
            button=button,
            event_type=event_type,
            logger=logger,
            debug_input_active=DEBUG_INPUT_ACTIVE,
            dispatch_helper=_handle_control_panel_click_dispatch_helper,
        )

    def _control_panel_hit_rect(self):
        return _control_panel_hit_rect_orchestration_helper(
            gui=self,
            pygame=pygame,
            control_panel_hit_rect_helper=_control_panel_hit_rect_helper,
        )

    def _should_swallow_control_panel_click(self, panel_hit_rect, pos) -> bool:
        return _should_swallow_control_panel_click_orchestration_helper(
            gui=self,
            panel_hit_rect=panel_hit_rect,
            pos=pos,
            logger=logger,
            should_swallow_control_panel_click_helper=_should_swallow_control_panel_click_helper,
        )

    def _translate_control_panel_click(self, pos, panel_hit_rect):
        return _translate_control_panel_click_orchestration_helper(
            gui=self,
            pos=pos,
            panel_hit_rect=panel_hit_rect,
            translate_control_panel_click_helper=_translate_control_panel_click_helper,
        )

    def _handle_outside_control_panel_click(self, panel_hit_rect, pos, button):
        return _handle_outside_control_panel_click_orchestration_helper(
            gui=self,
            panel_hit_rect=panel_hit_rect,
            pos=pos,
            button=button,
            dropdown_widget_cls=DropdownWidget,
            logger=logger,
            handle_outside_control_panel_click_helper=_handle_outside_control_panel_click_helper,
        )

    def _refresh_control_panel_layout_if_needed(self, sc_pos) -> bool:
        return _refresh_control_panel_layout_if_needed_orchestration_helper(
            gui=self,
            sc_pos=sc_pos,
            debug_input_active=DEBUG_INPUT_ACTIVE,
            logger=logger,
            refresh_control_panel_layout_if_needed_helper=_refresh_control_panel_layout_if_needed_helper,
        )

    def _retry_control_panel_click_after_auto_scroll(self, pos, sc_pos, button, handled):
        return _retry_control_panel_click_after_auto_scroll_orchestration_helper(
            gui=self,
            pos=pos,
            sc_pos=sc_pos,
            button=button,
            handled=handled,
            logger=logger,
            retry_control_panel_click_after_auto_scroll_helper=_retry_control_panel_click_after_auto_scroll_helper,
        )

    def _apply_control_panel_widget_updates(self):
        """Apply checkbox and dropdown state after a handled control-panel click."""
        _apply_control_panel_widget_updates_orchestration_helper(
            gui=self,
            checkbox_widget_cls=CheckboxWidget,
            logger=logger,
            apply_control_panel_widget_updates_helper=_apply_control_panel_widget_updates_helper,
        )

    def _apply_checkbox_widget_update(self, widget):
        _apply_checkbox_widget_update_orchestration_helper(
            gui=self,
            widget=widget,
            logger=logger,
            apply_checkbox_widget_update_helper=_apply_checkbox_widget_update_helper,
        )

    def _apply_dropdown_widget_update(self, widget):
        _apply_dropdown_widget_update_orchestration_helper(
            gui=self,
            widget=widget,
            logger=logger,
            apply_dropdown_widget_update_helper=_apply_dropdown_widget_update_helper,
        )

    def _apply_algorithm_dropdown_update(self, widget):
        _apply_algorithm_dropdown_update_orchestration_helper(
            gui=self,
            widget=widget,
            logger=logger,
            apply_algorithm_dropdown_update_helper=_apply_algorithm_dropdown_update_helper,
        )
    
    # Button callbacks
    def _stop_auto_solve(self):
        """Stop auto-solve and clear visual state."""
        _stop_auto_solve_orchestration_helper(
            gui=self,
            stop_auto_solve_flow_helper=_stop_auto_solve_flow_helper,
        )
    
    def _generate_dungeon(self):
        """Generate a new random dungeon using the procedural generator."""
        _generate_dungeon_orchestration_helper(
            gui=self,
            logger=logger,
            generate_dungeon_flow_helper=_generate_dungeon_flow_helper,
        )

    def _generate_ai_dungeon(self):
        """Non-blocking wrapper to spawn background worker and return immediately."""
        _start_ai_dungeon_generation_orchestration_helper(
            gui=self,
            threading_module=threading,
            start_ai_dungeon_generation_helper=_start_ai_dungeon_generation_helper,
        )


    def _generate_ai_dungeon_worker(self):
        """Background worker entry point for AI generation pipeline."""
        _run_ai_dungeon_generation_worker_orchestration_helper(
            gui=self,
            logger=logger,
            run_ai_generation_worker_helper=_run_ai_generation_worker_helper,
        )

    def _reset_map(self):
        """Reset the current map."""
        _reset_map_orchestration_helper(gui=self, reset_map_helper=_reset_map_helper)
    
    def _show_path_preview(self):
        """
        Show path preview for the currently available route.

        Behavior:
        - If a path already exists, open preview immediately.
        - If solver is running, request preview on completion.
        - If no path exists and solver is idle, start solver and force preview when it finishes.
        """
        _show_path_preview_orchestration_helper(
            gui=self,
            path_preview_dialog_cls=PathPreviewDialog,
            logger=logger,
            show_path_preview_helper=_show_path_preview_helper,
        )
    
    def _clear_path(self):
        """Clear the current path."""
        _clear_path_orchestration_helper(gui=self, clear_path_helper=_clear_path_helper)

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
        _export_route_orchestration_helper(gui=self, export_route_helper=_export_route_helper)
    
    def _load_route(self):
        """Load a saved route from JSON file."""
        _load_route_orchestration_helper(gui=self, load_route_helper=_load_route_helper)

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
        return _load_visual_assets_orchestration_helper(
            gui=self,
            templates_dir=templates_dir,
            link_sprite_path=link_sprite_path,
            pygame=pygame,
            os_module=os,
            logger=logger,
            semantic_palette=SEMANTIC_PALETTE,
            load_visual_assets_helper=_load_visual_assets_helper,
        )

    def load_visual_map(self, image_path: str, templates_dir: str | None = None):
        """Public API: create a GUI map from a screenshot and switch to it.

        - `image_path` can be a full screenshot (HUD allowed).
        - `templates_dir` is passed to the visual extractor (tileset or folder).

        This method is intentionally permissive and returns a bool for success
        so automated tests can call it without a file dialog.
        """
        return _load_visual_map_orchestration_helper(
            gui=self,
            image_path=image_path,
            templates_dir=templates_dir,
            load_visual_map_helper=_load_visual_map_helper,
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
        _place_items_from_graph_orchestration_helper(
            gui=self,
            grid=grid,
            graph=graph,
            room_positions=room_positions,
            room_to_node=room_to_node,
            logger=logger,
            semantic_palette=SEMANTIC_PALETTE,
            place_items_from_graph_helper=_place_items_from_graph_helper,
        )

    def _load_current_map(self):
        """Load and initialize the current map."""
        _load_current_map_orchestration_helper(
            gui=self,
            os_module=os,
            logger=logger,
            zelda_logic_env_cls=ZeldaLogicEnv,
            sanity_checker_cls=SanityChecker,
            semantic_palette=SEMANTIC_PALETTE,
            load_current_map_helper=_load_current_map_helper,
        )
    
    def _center_view(self):
        """Center the current map in the view."""
        _center_view_orchestration_helper(gui=self, center_view_helper=_center_view_helper)
    
    def _auto_fit_zoom(self):
        """Automatically set zoom level to fit the entire map in view."""
        _auto_fit_zoom_orchestration_helper(gui=self, auto_fit_zoom_helper=_auto_fit_zoom_helper)
    
    def _change_zoom(self, delta: int, center: tuple | None = None):
        """Change zoom level by delta steps.

        If `center` is provided (screen coordinates), the view will be adjusted so
        that the map tile under the `center` pixel remains under the cursor after
        the zoom. If `center` is None, the view is centered as before.
        """
        _change_zoom_orchestration_helper(
            gui=self,
            delta=delta,
            center=center,
            change_zoom_helper=_change_zoom_helper,
        )
    
    def _safe_set_mode(self, size, flags=0, allow_fallback=True):
        """Robust wrapper around pygame.display.set_mode.

        Attempts set_mode and, on failure or invalid surface (size 0), performs
        a display reinit and retries. If all attempts fail and allow_fallback is
        True, falls back to a windowed 800x600 surface to avoid leaving the
        application with a null/zero-sized display.
        Returns the created screen surface (or None on fatal failure).
        """
        return _safe_set_mode_orchestration_helper(
            size=size,
            pygame=pygame,
            logger=logger,
            safe_set_mode_helper=_safe_set_mode_helper,
            flags=flags,
            allow_fallback=allow_fallback,
        )

    def _attempt_display_reinit(self):
        """Attempt to fully reinitialize the SDL display and restore mode."""
        return _attempt_display_reinit_orchestration_helper(
            gui=self,
            pygame=pygame,
            logger=logger,
            attempt_display_reinit_helper=_attempt_display_reinit_helper,
        )

    def _handle_watchdog_screenshot(self) -> bool:
        """Save the requested watchdog screenshot on the main thread and clear the request.

        Returns True if a screenshot was saved, False otherwise. Always clears the
        request to avoid repeated attempts.
        """
        return _handle_watchdog_screenshot_orchestration_helper(
            gui=self,
            pygame=pygame,
            logger=logger,
            os_module=os,
            handle_watchdog_screenshot_helper=_handle_watchdog_screenshot_helper,
        )

    def report_ui_state(self) -> dict:
        """Return diagnostic information about GUI state for troubleshooting (callable from REPL)."""
        return _report_ui_state_orchestration_helper(
            gui=self,
            logger=logger,
            report_ui_state_helper=_report_ui_state_helper,
        )

    def _ensure_display_alive(self, force=False):
        """Check display health and attempt recovery if needed.

        If the display surface is None or has zero size, try to restore it.
        This method is intentionally conservative and returns False only when
        no recovery was possible.
        """
        return _ensure_display_alive_orchestration_helper(
            gui=self,
            pygame=pygame,
            logger=logger,
            ensure_display_alive_helper=_ensure_display_alive_helper,
            force=force,
        )

    def _force_focus(self) -> bool:
        """Try to force the window to the foreground on Windows.

        Uses a conservative Win32 sequence (AttachThreadInput + SetForegroundWindow + temporary TOPMOST) to
        work around Windows' foreground activation blocking. Returns True on success.
        No-op on non-Windows platforms.
        """
        return _force_focus_orchestration_helper(gui=self, force_focus_helper=_force_focus_helper)

    def _toggle_fullscreen(self):
        """Toggle fullscreen mode with robust handling.

        Uses `pygame.display.Info()` to obtain a valid fullscreen size and
        preserves the previous windowed size for restore. Ensures event pump
        and asset/layout reinitialization to avoid dark screens or unresponsiveness.
        """
        return _toggle_fullscreen_orchestration_helper(
            gui=self,
            pygame=pygame,
            logger=logger,
            os_module=os,
            platform_module=__import__('platform'),
            toggle_fullscreen_helper=_toggle_fullscreen_helper,
        )

    # ------------------ Control Panel Animation ------------------
    def _start_toggle_panel_animation(self, target_collapsed: bool):
        """Begin animated transition to collapsed or expanded state."""
        _start_toggle_panel_animation_orchestration_helper(
            gui=self,
            target_collapsed=target_collapsed,
            time_module=time,
            start_toggle_panel_animation_helper=_start_toggle_panel_animation_helper,
        )

    def _update_control_panel_animation(self):
        """Update animation state; should be called each frame."""
        _update_control_panel_animation_orchestration_helper(
            gui=self,
            time_module=time,
            update_control_panel_animation_helper=_update_control_panel_animation_helper,
        )

    def _update_control_panel_scroll(self):
        """Per-frame update that applies inertia (momentum) and clamps scroll."""
        _update_control_panel_scroll_orchestration_helper(
            gui=self,
            time_module=time,
            update_control_panel_scroll_helper=_update_control_panel_scroll_helper,
        )

    def run(self, max_frames: Optional[int] = None):
        """Main game loop with delta-time support.

        When running under tests (env var KLTN_TEST_MODE or under pytest), a small
        default max_frames is used to avoid infinite loops. Callers can override
        with the optional max_frames parameter.
        """
        _run_orchestration_helper(
            gui=self,
            max_frames=max_frames,
            env=os.environ,
            resolve_test_mode_max_frames_fn=resolve_test_mode_max_frames,
            run_main_loop_helper=_run_main_loop_helper,
            pygame=pygame,
            os_module=os,
            logger=logger,
            time_module=time,
            np_module=np,
            action_enum=Action,
            checkbox_widget_cls=CheckboxWidget,
            path_preview_dialog_cls=PathPreviewDialog,
            safe_unpickle_fn=_safe_unpickle,
            should_attempt_focus_fallback_fn=should_attempt_focus_fallback,
            poll_pygame_events_fn=poll_pygame_events,
            run_input_focus_fallback_fn=run_input_focus_fallback,
            clear_stale_preview_overlay_fn=clear_stale_preview_overlay,
            handle_window_focus_event_fn=handle_window_focus_event,
            handle_global_keydown_shortcuts_fn=handle_global_keydown_shortcuts,
            handle_preview_overlay_events_fn=handle_preview_overlay_events,
            handle_videoresize_event_fn=handle_videoresize_event,
            handle_mousewheel_event_fn=handle_mousewheel_event,
            handle_mouse_button_down_preamble_fn=handle_mouse_button_down_preamble,
            handle_mouse_button_down_event_fn=handle_mouse_button_down_event,
            handle_mouse_button_up_event_fn=handle_mouse_button_up_event,
            handle_mouse_motion_diagnostics_fn=handle_mouse_motion_diagnostics,
            handle_mouse_motion_event_fn=handle_mouse_motion_event,
            handle_keyup_event_fn=handle_keyup_event,
            handle_keydown_event_fn=handle_keydown_event,
            run_auto_step_tick_fn=run_auto_step_tick,
            run_continuous_movement_tick_fn=run_continuous_movement_tick,
            update_heartbeat_fn=update_heartbeat,
            handle_pending_solver_trigger_fn=handle_pending_solver_trigger,
            handle_parallel_search_completion_fn=handle_parallel_search_completion,
            handle_preview_process_completion_fn=handle_preview_process_completion,
            handle_solver_process_completion_fn=handle_solver_process_completion,
            handle_ai_generation_completion_fn=handle_ai_generation_completion,
            render_and_present_frame_fn=render_and_present_frame,
            handle_watchdog_screenshot_request_fn=handle_watchdog_screenshot_request,
            run_periodic_display_health_check_fn=run_periodic_display_health_check,
            advance_frame_and_check_limit_fn=advance_frame_and_check_limit,
            tick_frame_clock_fn=tick_frame_clock,
            compute_solver_timeout_seconds_fn=compute_solver_timeout_seconds,
            find_path_tile_violations_fn=find_path_tile_violations,
            debug_input_active=DEBUG_INPUT_ACTIVE,
        )

    def _next_map(self):
        """Move to the next map and stop auto-solve if running."""
        _next_map_orchestration_helper(gui=self, logger=logger, next_map_helper=_next_map_helper)

    def _prev_map(self):
        """Move to the previous map and stop auto-solve if running."""
        _prev_map_orchestration_helper(gui=self, logger=logger, prev_map_helper=_prev_map_helper)
    
    def _clamp_view_offset(self):
        """Clamp view offset to valid range.

        When the dungeon/map is smaller than the viewport, allow negative offsets so
        the user can pan the small map freely inside the window (showing empty
        margins) while still preventing arbitrary unrestricted panning.
        """
        _clamp_view_offset_orchestration_helper(gui=self, clamp_view_offset_helper=_clamp_view_offset_helper)
    
    def _center_on_player(self):
        """Center the view on the player position."""
        _center_on_player_orchestration_helper(gui=self, center_on_player_helper=_center_on_player_helper)
    
    def _start_preview_for_current_map(self):
        _start_preview_for_current_map_orchestration_helper(
            gui=self,
            logger=logger,
            pygame_module=pygame,
            multiprocessing_module=multiprocessing,
            threading_module=threading,
            time_module=time,
            run_preview_and_dump=_run_preview_and_dump,
            start_preview_for_current_map_helper=_start_preview_for_current_map_helper,
        )

    def _clear_solver_state(self, reason="cleanup"):
        """Helper to centralize solver state cleanup and ensure consistency.
        
        Args:
            reason: Description of why solver is being cleared (for logging)
        """
        _clear_solver_state_orchestration_helper(
            gui=self,
            reason=reason,
            logger=logger,
            clear_solver_state_helper=_clear_solver_state_helper,
        )

    def _sync_solver_dropdown_settings(self):
        """Refresh algorithm/representation/ARA values from dropdown widgets."""
        return _sync_solver_dropdown_settings_orchestration_helper(
            gui=self,
            sync_fn=sync_solver_dropdown_settings,
            sync_solver_dropdown_settings_helper=_sync_solver_dropdown_settings_helper,
        )

    def _algorithm_name(self, algorithm_idx):
        """Return canonical display label for a solver index."""
        return algorithm_label(algorithm_idx)

    def _start_auto_solve(self):
        """Start auto-solve mode using state-space solver with inventory tracking.

        This schedules the heavy solver in a background process/thread using
        the existing `_schedule_solver()` helper. Non-blocking and safe to call
        from the main loop or event handlers.
        """
        
        _start_auto_solve_orchestration_helper(
            gui=self,
            logger=logger,
            debug_sync_solver=DEBUG_SYNC_SOLVER,
            start_auto_solve_helper=_start_auto_solve_helper,
        )

    def _prepare_active_solver_for_new_start(self) -> bool:
        """Return True when a new solver run may proceed, False to block startup."""
        return _prepare_active_solver_for_new_start_orchestration_helper(
            gui=self,
            logger=logger,
            time_module=time,
            evaluate_solver_recovery_state=evaluate_solver_recovery_state,
            compute_timeout_seconds=self._compute_solver_timeout_seconds,
            terminate_hung_process=self._terminate_hung_solver_process,
            force_recovery_state=self._force_solver_recovery_state,
            log_active_state=self._log_active_solver_state,
            prepare_active_solver_for_new_start_helper=_prepare_active_solver_for_new_start_helper,
        )

    def _log_active_solver_state(self):
        _log_active_solver_state_orchestration_helper(
            gui=self,
            logger=logger,
            os_module=os,
            time_module=time,
            log_active_solver_state_helper=_log_active_solver_state_helper,
        )

    def _compute_solver_timeout_seconds(self, active_alg: int) -> float:
        return _compute_solver_timeout_seconds_orchestration_helper(
            gui=self,
            active_alg=active_alg,
            default_solver_timeout_for_algorithm=default_solver_timeout_for_algorithm,
            scale_timeout_by_grid_size=scale_timeout_by_grid_size,
            np_module=np,
            os_module=os,
            compute_solver_timeout_seconds_helper=_compute_solver_timeout_seconds_helper,
        )

    def _terminate_hung_solver_process(self, proc):
        _terminate_hung_solver_process_orchestration_helper(
            proc=proc,
            logger=logger,
            terminate_hung_solver_process_helper=_terminate_hung_solver_process_helper,
        )

    def _force_solver_recovery_state(self, recovery_reason: str):
        _force_solver_recovery_state_orchestration_helper(
            gui=self,
            recovery_reason=recovery_reason,
            logger=logger,
            force_solver_recovery_state_helper=_force_solver_recovery_state_helper,
        )

    def _cleanup_preview_before_solver_start(self):
        """Stop preview workers/files so new solve starts from a clean state."""
        _cleanup_preview_before_solver_start_orchestration_helper(
            gui=self,
            logger=logger,
            os_module=os,
            cleanup_preview_before_solver_start_helper=_cleanup_preview_before_solver_start_helper,
        )

    def _reset_solver_visual_state_before_start(self):
        """Clear solver/visual state from previous runs before scheduling a new solve."""
        _reset_solver_visual_state_before_start_orchestration_helper(
            gui=self,
            reset_solver_visual_state_before_start_helper=_reset_solver_visual_state_before_start_helper,
        )

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
        _run_solver_sync_orchestration_helper(
            gui=self,
            logger=logger,
            solve_in_subprocess=_solve_in_subprocess,
            algorithm_idx=algorithm_idx,
            run_solver_sync_helper=_run_solver_sync_helper,
        )

    def _watchdog_loop(self):
        """Background watchdog that writes stack traces and a screenshot when the main loop stalls.

        Controlled by environment vars:
        - KLTN_ENABLE_WATCHDOG (default '1') enable watchdog
        - KLTN_WATCHDOG_THRESHOLD (seconds, default 1.25)
        - KLTN_WATCHDOG_DUMP_LIMIT (how many dumps to write, default 3)
        - KLTN_WATCHDOG_TERMINATE_SOLVER (if '1' will terminate solver proc when dumping)
        """
        _watchdog_loop_orchestration_helper(
            gui=self,
            logger=logger,
            os_module=os,
            time_module=time,
            tempfile_module=tempfile,
            watchdog_loop_helper=_watchdog_loop_helper,
        )
        

    
    def _schedule_solver(self, algorithm_idx=None):
        """Start solver in background worker process/thread.
        
        Args:
            algorithm_idx: Algorithm index to use (if None, read from self.algorithm_idx)
        """
        return _schedule_solver_orchestration_helper(
            gui=self,
            algorithm_idx=algorithm_idx,
            logger=logger,
            time_module=time,
            threading_module=threading,
            schedule_solver_helper=_schedule_solver_helper,
        )

    def _create_solver_temp_files(self, grid_arr):
        """Create output and optional grid temp files for solver worker launch."""
        return _create_solver_temp_files_orchestration_helper(
            grid_arr=grid_arr,
            create_solver_temp_files_helper=_create_solver_temp_files_helper,
        )

    def _launch_solver_worker(self, **kwargs):
        """Launch solver process, with thread-based fallback on process failure."""
        _launch_solver_worker_orchestration_helper(
            gui=self,
            kwargs=kwargs,
            logger=logger,
            launch_solver_process=self._launch_solver_process,
            start_solver_thread_fallback=self._start_solver_thread_fallback,
            multiprocessing_module=multiprocessing,
            launch_solver_worker_helper=_launch_solver_worker_helper,
        )

    def _launch_solver_process(self, **kwargs):
        _launch_solver_process_orchestration_helper(
            gui=self,
            launch_kwargs=kwargs,
            run_solver_and_dump=_run_solver_and_dump,
            multiprocessing_module=multiprocessing,
            logger=logger,
            launch_solver_process_helper=_launch_solver_process_helper,
        )

    def _solver_thread_fallback_worker(self, **kwargs):
        _solver_thread_fallback_worker_orchestration_helper(
            gui=self,
            launch_kwargs=kwargs,
            solve_in_subprocess=_solve_in_subprocess,
            logger=logger,
            solver_thread_fallback_worker_helper=_solver_thread_fallback_worker_helper,
        )

    def _start_solver_thread_fallback(self, **kwargs):
        _start_solver_thread_fallback_orchestration_helper(
            gui=self,
            launch_kwargs=kwargs,
            threading_module=threading,
            worker_target=self._solver_thread_fallback_worker,
            logger=logger,
            start_solver_thread_fallback_helper=_start_solver_thread_fallback_helper,
        )

    def _execute_auto_solve(self, path, solver_result, teleports=0):
        """
        Execute auto-solve immediately without preview (fallback).
        
        Args:
            path: Planned path
            solver_result: Solver metadata (may include CBS metrics)
            teleports: Number of teleport/warp moves
        """
        _execute_auto_solve_orchestration_helper(
            gui=self,
            path=path,
            solver_result=solver_result,
            teleports=teleports,
            logger=logger,
            execute_auto_solve_helper=_execute_auto_solve_helper,
        )
    
    def _execute_auto_solve_from_preview(self):
        """
        Start auto-solve after user confirms path preview.
        """
        _execute_auto_solve_from_preview_orchestration_helper(
            gui=self,
            logger=logger,
            execute_auto_solve_from_preview_helper=_execute_auto_solve_from_preview_helper,
        )
    
    def _smart_grid_path(self):
        """
        Smart pathfinding that prioritizes walking and only warps via STAIRs.
        Returns (success, path, teleport_count).
        """
        return _smart_grid_path_orchestration_helper(
            gui=self,
            logger=logger,
            convert_diagonal_to_4dir=_convert_diagonal_to_4dir,
            semantic_palette=SEMANTIC_PALETTE,
            np_module=np,
            path_cls=Path,
            os_module=os,
            smart_grid_path_helper=_smart_grid_path_helper,
        )

    def _graph_guided_path(self):
        """Fallback: follow graph path with teleportation when needed."""
        return _graph_guided_path_orchestration_helper(
            gui=self,
            graph_guided_path_helper=_graph_guided_path_helper,
        )

    def _hybrid_graph_grid_path(self):
        """
        Hybrid pathfinding: use graph to find room sequence, 
        then BFS within each room and teleport between disconnected clusters.
        """
        return _hybrid_graph_grid_path_orchestration_helper(
            gui=self,
            hybrid_graph_grid_path_helper=_hybrid_graph_grid_path_helper,
        )

    def _stop_auto(self, reason: str = None):
        """Stop auto-solve mode with consistent logging and cleanup."""
        return _stop_auto_orchestration_helper(
            gui=self,
            reason=reason,
            logger=logger,
            stop_auto_helper=_stop_auto_helper,
        )

    def _auto_step(self):
        """Execute one step of auto-solve with comprehensive error handling."""
        import traceback

        return _auto_step_orchestration_helper(
            gui=self,
            logger=logger,
            game_state_cls=GameState,
            action_enum=Action,
            ripple_effect_cls=RippleEffect,
            flash_effect_cls=FlashEffect,
            traceback_module=traceback,
            auto_step_helper=_auto_step_helper,
        )
    
    def _show_error(self, message: str):
        """Display error message to user with visual feedback."""
        return _show_error_orchestration_helper(
            gui=self,
            message=message,
            logger=logger,
            time_module=time,
            show_error_helper=_show_error_helper,
        )
    
    def _show_message(self, message: str, duration: float = 3.0):
        """Display informational message to user."""
        return _show_message_orchestration_helper(
            gui=self,
            message=message,
            duration=duration,
            logger=logger,
            time_module=time,
            show_message_helper=_show_message_helper,
        )

    # --- Topology helpers ---
    def _export_topology(self):
        """Export current map topology to a DOT file (if available)."""
        return _export_topology_orchestration_helper(
            gui=self,
            export_topology_helper=_export_topology_helper,
        )


    def _render_topology_overlay(self, surface):
        """Draw room nodes and edges on the map area with high-visibility styling."""
        _render_topology_overlay_orchestration_helper(
            gui=self,
            surface=surface,
            pygame=pygame,
            render_topology_overlay_helper=_render_topology_overlay_helper,
        )

    def _match_missing_nodes(self):
        """Attempt to infer and stage mapping proposals for unmatched nodes.

        Uses RoomGraphMatcher.infer_missing_mappings to generate proposals with confidences.
        High-confidence proposals (>= configured threshold) are applied automatically.
        Lower confidence proposals are kept as 'tentative' in `current.match_proposals` for manual apply.
        """
        return _match_missing_nodes_orchestration_helper(
            gui=self,
            matcher_cls=RoomGraphMatcher,
            logger=logger,
            match_missing_nodes_helper=_match_missing_nodes_helper,
        )

    def _undo_last_match(self):
        """Undo last applied match snapshot, if any."""
        return _undo_last_match_orchestration_helper(
            gui=self,
            logger=logger,
            undo_last_match_helper=_undo_last_match_helper,
        )

    def _room_for_global_position(self, pos: Optional[Tuple[int, int]], room_positions: dict) -> Optional[Tuple[int, int]]:
        """Map a global tile coordinate to a room-grid coordinate."""
        return _room_for_global_position_orchestration_helper(
            pos=pos,
            room_positions=room_positions,
            room_for_global_position_helper=_room_for_global_position_helper,
        )

    @staticmethod
    def _node_has_small_key(attrs: dict) -> bool:
        """Best-effort small-key detection from graph node attributes/labels."""
        return _node_has_small_key_orchestration_helper(
            attrs=attrs,
            node_has_small_key_helper=_node_has_small_key_helper,
        )

    def _node_has_critical_content(self, graph, node_id: Any) -> bool:
        """Whether a node should be preserved during dead-end pruning."""
        return _node_has_critical_content_orchestration_helper(
            graph=graph,
            node_id=node_id,
            node_has_critical_content_helper=_node_has_critical_content_helper,
        )

    def _capture_precheck_snapshot(self, current: Any, reason: str = "") -> None:
        """Capture current topology state so Undo Prune can restore it."""
        _capture_precheck_snapshot_orchestration_helper(
            gui=self,
            current=current,
            reason=reason,
            capture_precheck_snapshot_helper=_capture_precheck_snapshot_helper,
        )

    def _update_env_topology_view(self, current: Any) -> None:
        """Synchronize current map topology attributes into the active env object."""
        _update_env_topology_view_orchestration_helper(
            gui=self,
            current=current,
            update_env_topology_view_helper=_update_env_topology_view_helper,
        )

    def _build_room_adjacency_from_graph(self, graph: Any, room_to_node: dict, node_to_room: dict) -> dict:
        """Build undirected room adjacency from graph edges via node-room mapping."""
        return _build_room_adjacency_from_graph_orchestration_helper(
            graph=graph,
            room_to_node=room_to_node,
            node_to_room=node_to_room,
            build_room_adjacency_from_graph_helper=_build_room_adjacency_from_graph_helper,
        )

    def _prune_dead_end_topology(self, current: Any, preserve_rooms: set) -> List[Tuple[int, int]]:
        """Prune dead-end rooms from topology mapping when room objects are unavailable."""
        return _prune_dead_end_topology_orchestration_helper(
            gui=self,
            current=current,
            preserve_rooms=preserve_rooms,
            logger=logger,
            prune_dead_end_topology_flow_helper=_prune_dead_end_topology_flow_helper,
            build_room_adjacency_fn=self._build_room_adjacency_from_graph,
            node_has_critical_content_fn=self._node_has_critical_content,
        )

    def _run_prechecks_and_optional_prune(self) -> Tuple[bool, Optional[str]]:
        """Run lightweight prechecks and optional dead-end pruning before solve."""
        return _run_prechecks_and_optional_prune_orchestration_helper(
            gui=self,
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
            run_prechecks_and_optional_prune_flow_helper=_run_prechecks_and_optional_prune_flow_helper,
        )

    def _undo_prune(self):
        """Undo the last applied prune snapshot, if any."""
        return _undo_prune_orchestration_helper(
            gui=self,
            logger=logger,
            update_env_topology_view_fn=self._update_env_topology_view,
            undo_prune_flow_helper=_undo_prune_flow_helper,
        )

    def _apply_tentative_matches(self):
        """Apply staged tentative matches above the configured threshold."""
        return _apply_tentative_matches_orchestration_helper(
            gui=self,
            logger=logger,
            apply_tentative_matches_helper=_apply_tentative_matches_helper,
        )

    # --- Solver comparison helpers ---
    def _set_last_solver_metrics(self, name, nodes, time_ms, path_len):
        return _set_last_solver_metrics_orchestration_helper(
            gui=self,
            name=name,
            nodes=nodes,
            time_ms=time_ms,
            path_len=path_len,
            set_last_solver_metrics_helper=_set_last_solver_metrics_helper,
        )

    def _run_solver_comparison(self):
        """Start an asynchronous solver comparison worker to avoid blocking the GUI."""
        return _run_solver_comparison_orchestration_helper(
            gui=self,
            logger=logger,
            time_module=time,
            game_state_cls=GameState,
            solve_in_subprocess=_solve_in_subprocess,
            threading_module=threading,
            run_solver_comparison_helper=_run_solver_comparison_helper,
        )

    def _start_map_elites(self, n_samples: int = 200, resolution: int = 20):
        """Start a background MAP-Elites evaluation on the currently loaded maps.

        Runs on a background thread so the GUI stays responsive. Results are stored
        in `self.map_elites_result` and a toast is shown when complete.
        """
        return _start_map_elites_orchestration_helper(
            gui=self,
            n_samples=n_samples,
            resolution=resolution,
            threading_module=threading,
            start_map_elites_flow_helper=_start_map_elites_flow_helper,
        )

    def _map_elites_worker(self, maps, n_samples: int, resolution: int):
        """Background worker implementing MAP-Elites on a set of pre-loaded maps.

        This function uses the lightweight `src.simulation.map_elites` helper and the
        built-in `DungeonSolver` for validation.
        """
        return _map_elites_worker_orchestration_helper(
            gui=self,
            maps=maps,
            n_samples=n_samples,
            resolution=resolution,
            logger=logger,
            os_module=os,
            map_elites_worker_flow_helper=_map_elites_worker_flow_helper,
        )

    def _render_solver_comparison_overlay(self, surface):
        """Render a small sidebar table with solver comparison results."""
        _render_solver_comparison_overlay_orchestration_helper(
            gui=self,
            surface=surface,
            pygame=pygame,
            render_solver_comparison_overlay_helper=_render_solver_comparison_overlay_helper,
        )
    
    def _set_message(self, message: str, duration: float = 3.0):
        """Set status message with timestamp for auto-hide."""
        _set_message_orchestration_helper(
            gui=self,
            message=message,
            duration=duration,
            time_module=time,
            set_message_helper=_set_message_helper,
        )
    
    def _show_toast(self, message: str, duration: float = 3.0, toast_type: str = 'info'):
        """Show a floating toast notification."""
        _show_toast_orchestration_helper(
            gui=self,
            message=message,
            duration=duration,
            toast_type=toast_type,
            toast_cls=ToastNotification,
            show_toast_helper=_show_toast_helper,
        )
    
    def _format_cbs_metrics_tooltip(self, cbs_metrics: dict) -> str:
        """Format CBS metrics for detailed tooltip display."""
        return _format_cbs_metrics_tooltip_orchestration_helper(
            cbs_metrics=cbs_metrics,
            format_cbs_metrics_tooltip_helper=_format_cbs_metrics_tooltip_helper,
        )
    
    def _update_toasts(self):
        """Update and remove expired toasts."""
        _update_toasts_orchestration_helper(gui=self, update_toasts_helper=_update_toasts_helper)
    
    def _render_toasts(self, surface):
        """Render all active toast notifications."""
        _render_toasts_orchestration_helper(
            gui=self,
            surface=surface,
            render_toasts_helper=_render_toasts_helper,
        )
    
    # ========================================
    # BLOCK PUSH ANIMATION SYSTEM
    # ========================================
    
    def _start_block_push_animation(self, block_from: Tuple[int, int], block_to: Tuple[int, int]):
        """Start animating a block being pushed from one position to another.
        
        Args:
            block_from: Original block position (row, col)
            block_to: Destination position (row, col)
        """
        _start_block_push_animation_orchestration_helper(
            gui=self,
            block_from=block_from,
            block_to=block_to,
            pygame=pygame,
            logger=logger,
            start_block_push_animation_helper=_start_block_push_animation_helper,
        )
    
    def _update_block_push_animations(self):
        """Update all active block push animations and complete finished ones."""
        _update_block_push_animations_orchestration_helper(
            gui=self,
            pygame=pygame,
            semantic_palette=SEMANTIC_PALETTE,
            pop_effect_cls=PopEffect,
            logger=logger,
            update_block_push_animations_helper=_update_block_push_animations_helper,
        )
    
    def _render_block_push_animations(self, surface):
        """Render blocks that are currently being pushed with smooth interpolation.
        
        Args:
            surface: The pygame surface to draw on (map_surface)
        """
        _render_block_push_animations_orchestration_helper(
            gui=self,
            surface=surface,
            pygame=pygame,
            semantic_palette=SEMANTIC_PALETTE,
            render_block_push_animations_helper=_render_block_push_animations_helper,
        )
    
    def _get_animating_block_positions(self) -> set:
        """Get set of block positions currently being animated (to skip normal rendering)."""
        return _get_animating_block_positions_orchestration_helper(
            gui=self,
            get_animating_block_positions_helper=_get_animating_block_positions_helper,
        )
    
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
        return _check_and_start_block_push_orchestration_helper(
            gui=self,
            player_pos=player_pos,
            target_pos=target_pos,
            walkable_ids=WALKABLE_IDS,
            pushable_ids=PUSHABLE_IDS,
            check_and_start_block_push_helper=_check_and_start_block_push_helper,
        )

    def _show_warning(self, message: str):
        """Display warning message to user."""
        _show_warning_orchestration_helper(
            gui=self,
            message=message,
            logger=logger,
            show_warning_helper=_show_warning_helper,
        )
    
    def _manual_step(self, action: Action):
        """Execute manual step."""
        return _manual_step_orchestration_helper(
            gui=self,
            action=action,
            action_deltas=ACTION_DELTAS,
            pop_effect_cls=PopEffect,
            flash_effect_cls=FlashEffect,
            time_module=time,
            manual_step_helper=_manual_step_flow_helper,
        )
    
    def _render_path_GUARANTEED(self, surface):
        """GUARANTEED path rendering - draws path no matter what.
        
        This method provides bulletproof path visualization that works
        regardless of auto_mode, preview state, or feature flags.
        Call this AFTER tiles are drawn but BEFORE HUD elements.
        """
        return _render_path_guaranteed_orchestration_helper(
            gui=self,
            surface=surface,
            pygame=pygame,
            math_module=math,
            time_module=time,
            logger=logger,
            render_path_guaranteed_flow_helper=_render_path_guaranteed_flow_helper,
        )

    def _render(self):
        """Render the current state using new visualization system or fallback."""
        _render_frame_orchestration_helper(
            gui=self,
            pygame=pygame,
            logger=logger,
            time_module=time,
            math_module=math,
            semantic_palette=SEMANTIC_PALETTE,
            create_map_surface_fn=_create_map_surface_helper,
            update_frame_render_state_fn=_update_frame_render_state_helper,
            compute_visible_bounds_fn=_compute_visible_bounds_helper,
            log_draw_ranges_fn=_log_draw_ranges_overlay_helper,
            render_empty_range_warning_fn=_render_empty_range_warning_overlay_helper,
            collect_item_render_state_fn=_collect_item_render_state_helper,
            render_visible_tiles_fn=_render_visible_tiles_helper,
            render_block_push_animations_fn=self._render_block_push_animations,
            render_heatmap_overlay_fn=_render_heatmap_overlay_helper,
            render_jps_overlay_fn=_render_jps_overlay_helper,
            render_map_elites_overlay_fn=_render_map_elites_overlay_helper,
            render_planned_path_overlay_fn=_render_planned_path_overlay_helper,
            render_path_guaranteed_fn=self._render_path_GUARANTEED,
            render_player_and_effects_fn=_render_player_and_effects_helper,
            handle_empty_frame_recovery_fn=_handle_empty_frame_recovery_helper,
            render_translucent_event_overlays_fn=_render_translucent_event_overlays_helper,
            draw_sidebar_shell_fn=_draw_sidebar_shell_helper,
            render_sidebar_content_fn=_render_sidebar_content_helper,
            render_post_map_layers_fn=_render_post_map_layers_helper,
            render_top_ui_layers_fn=_render_top_ui_layers_helper,
            render_sidebar_header_fn=_render_sidebar_header_inventory_solver_helper,
            render_sidebar_status_fn=_render_sidebar_status_message_metrics_controls_helper,
            render_preview_layer_fn=_render_preview_layer_helper,
            render_frame_helper=_render_frame_helper,
        )

    def _render_debug_overlay(self, surface):
        """Render debug overlay with mouse coords, widget rects, and recent clicks.
        Toggle with F12. Shift-F11 clears click log.
        """
        _render_debug_overlay_orchestration_helper(
            gui=self,
            surface=surface,
            pygame=pygame,
            time_module=time,
            render_debug_overlay_helper=_render_debug_overlay_helper,
        )

    def _render_unified_bottom_panel(self):
        """Render unified bottom HUD panel - STATUS and MESSAGE only (inventory moved to sidebar)."""
        _render_unified_bottom_panel_orchestration_helper(
            gui=self,
            pygame=pygame,
            render_unified_bottom_panel_helper=_render_unified_bottom_panel_helper,
        )
    
    def _render_message_section(self, x: int, y: int, width: int, height: int):
        """Render message/status section in bottom panel."""
        _render_message_section_orchestration_helper(
            gui=self,
            x=x,
            y=y,
            width=width,
            height=height,
            render_message_section_helper=_render_message_section_helper,
        )
    
    def _render_progress_bar(self, surface, x: int, y: int, width: int, height: int, 
                             filled: int, total: int, color_filled: tuple, color_empty: tuple):
        """Render a segmented progress bar with filled/empty indicators."""
        _render_progress_bar_orchestration_helper(
            surface=surface,
            x=x,
            y=y,
            width=width,
            height=height,
            filled=filled,
            total=total,
            color_filled=color_filled,
            color_empty=color_empty,
            pygame=pygame,
            render_progress_bar_helper=_render_progress_bar_helper,
        )
    
    def _render_inventory_section(self, x: int, y: int, width: int, height: int):
        """Render inventory section with progress bars and icons."""
        _render_inventory_section_orchestration_helper(
            gui=self,
            x=x,
            y=y,
            width=width,
            height=height,
            pygame=pygame,
            time_module=time,
            logger=logger,
            render_inventory_section_helper=_render_inventory_section_helper,
        )
    
    def _render_metrics_section(self, x: int, y: int, width: int, height: int):
        """Render metrics section (steps, speed, zoom, env)."""
        _render_metrics_section_orchestration_helper(
            gui=self,
            x=x,
            y=y,
            width=width,
            height=height,
            render_metrics_section_helper=_render_metrics_section_helper,
        )
    
    def _render_controls_section(self, x: int, y: int, width: int, height: int):
        """Render controls section in two-column layout."""
        _render_controls_section_orchestration_helper(
            gui=self,
            x=x,
            y=y,
            width=width,
            height=height,
            render_controls_section_helper=_render_controls_section_helper,
        )
    
    def _render_status_section(self, x: int, y: int, width: int, height: int):
        """Render status section with game state information."""
        _render_status_section_orchestration_helper(
            gui=self,
            x=x,
            y=y,
            width=width,
            height=height,
            render_status_section_helper=_render_status_section_helper,
        )
    
    def _render_minimap(self):
        """Render small dungeon overview map in bottom-right corner."""
        _render_minimap_orchestration_helper(
            gui=self,
            pygame=pygame,
            render_minimap_helper=_render_minimap_helper,
        )
    
    def _handle_minimap_click(self, mouse_pos: Tuple[int, int]) -> bool:
        """Handle mouse click on minimap to jump to that location."""
        return _handle_minimap_click_orchestration_helper(
            gui=self,
            mouse_pos=mouse_pos,
            handle_minimap_click_helper=_handle_minimap_click_helper,
        )
    
    def _render_help_overlay(self):
        """Render help overlay."""
        _render_help_overlay_orchestration_helper(
            gui=self,
            pygame=pygame,
            render_help_overlay_helper=_render_help_overlay_helper,
        )


def load_maps_from_adapter():
    """Load processed maps from data adapter using new zelda_core - ALL 18 variants."""
    return _load_maps_from_adapter_orchestration_helper(
        os_module=os,
        file_path=__file__,
        print_fn=print,
        load_maps_from_adapter_helper=_load_maps_from_adapter_helper,
    )


def main():
    """Main entry point."""
    _run_main_entry_orchestration_helper(
        pygame_available=PYGAME_AVAILABLE,
        load_maps_fn=load_maps_from_adapter,
        create_test_map_fn=create_test_map,
        gui_cls=ZeldaGUI,
        print_fn=print,
        run_gui_main_helper=_run_gui_main_helper,
    )


if __name__ == "__main__":
    # Required for multiprocessing on Windows (freeze_support)
    multiprocessing.freeze_support()
    main()



