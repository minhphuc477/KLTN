"""Frame and guaranteed-path rendering orchestration bridges for ZeldaGUI."""

from __future__ import annotations

from src.gui.rendering.path_guaranteed_renderer import (
    render_path_guaranteed as _render_path_guaranteed_flow_helper,
)
from src.gui.rendering.render_frame_pipeline import (
    render_frame as _render_frame_helper,
)


def render_path_guaranteed(*, gui, surface, pygame, math_module, time_module, logger):
    return _render_path_guaranteed_flow_helper(
        gui=gui,
        surface=surface,
        pygame=pygame,
        math_module=math_module,
        time_module=time_module,
        logger=logger,
    )


def render_frame(
    *,
    gui,
    pygame,
    logger,
    time_module,
    math_module,
    semantic_palette,
    create_map_surface_fn,
    update_frame_render_state_fn,
    compute_visible_bounds_fn,
    log_draw_ranges_fn,
    render_empty_range_warning_fn,
    collect_item_render_state_fn,
    render_visible_tiles_fn,
    render_block_push_animations_fn,
    render_heatmap_overlay_fn,
    render_jps_overlay_fn,
    render_map_elites_overlay_fn,
    render_planned_path_overlay_fn,
    render_path_guaranteed_fn,
    render_player_and_effects_fn,
    handle_empty_frame_recovery_fn,
    render_translucent_event_overlays_fn,
    draw_sidebar_shell_fn,
    render_sidebar_content_fn,
    render_post_map_layers_fn,
    render_top_ui_layers_fn,
    render_sidebar_header_fn,
    render_sidebar_status_fn,
    render_preview_layer_fn,
):
    _render_frame_helper(
        gui=gui,
        pygame=pygame,
        logger=logger,
        time_module=time_module,
        math_module=math_module,
        semantic_palette=semantic_palette,
        create_map_surface_fn=create_map_surface_fn,
        update_frame_render_state_fn=update_frame_render_state_fn,
        compute_visible_bounds_fn=compute_visible_bounds_fn,
        log_draw_ranges_fn=log_draw_ranges_fn,
        render_empty_range_warning_fn=render_empty_range_warning_fn,
        collect_item_render_state_fn=collect_item_render_state_fn,
        render_visible_tiles_fn=render_visible_tiles_fn,
        render_block_push_animations_fn=render_block_push_animations_fn,
        render_heatmap_overlay_fn=render_heatmap_overlay_fn,
        render_jps_overlay_fn=render_jps_overlay_fn,
        render_map_elites_overlay_fn=render_map_elites_overlay_fn,
        render_planned_path_overlay_fn=render_planned_path_overlay_fn,
        render_path_guaranteed_fn=render_path_guaranteed_fn,
        render_player_and_effects_fn=render_player_and_effects_fn,
        handle_empty_frame_recovery_fn=handle_empty_frame_recovery_fn,
        render_translucent_event_overlays_fn=render_translucent_event_overlays_fn,
        draw_sidebar_shell_fn=draw_sidebar_shell_fn,
        render_sidebar_content_fn=render_sidebar_content_fn,
        render_post_map_layers_fn=render_post_map_layers_fn,
        render_top_ui_layers_fn=render_top_ui_layers_fn,
        render_sidebar_header_fn=render_sidebar_header_fn,
        render_sidebar_status_fn=render_sidebar_status_fn,
        render_preview_layer_fn=render_preview_layer_fn,
    )
