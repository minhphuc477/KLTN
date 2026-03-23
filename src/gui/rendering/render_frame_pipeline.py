"""Top-level frame rendering orchestration for ZeldaGUI."""

from __future__ import annotations


def render_frame(
    gui,
    *,
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
    """Render one GUI frame preserving legacy draw order and diagnostics behavior."""
    if getattr(gui, "env", None) is None:
        logger.debug("render_frame skipped: gui.env is not initialized")
        return

    gui.screen.fill((25, 25, 35))

    map_h, map_w = gui.env.height, gui.env.width
    map_surface, view_w, view_h = create_map_surface_fn(gui=gui, pygame=pygame)

    update_frame_render_state_fn(gui=gui, logger=logger)

    start_r, end_r, start_c, end_c = compute_visible_bounds_fn(
        gui=gui,
        view_w=view_w,
        view_h=view_h,
        map_h=map_h,
        map_w=map_w,
    )

    log_draw_ranges_fn(
        gui=gui,
        start_r=start_r,
        end_r=end_r,
        start_c=start_c,
        end_c=end_c,
        h=map_h,
        w=map_w,
        time_module=time_module,
        logger=logger,
    )
    render_empty_range_warning_fn(
        gui=gui,
        start_r=start_r,
        end_r=end_r,
        start_c=start_c,
        end_c=end_c,
        pygame=pygame,
    )

    collected_items, collectible_tile_ids = collect_item_render_state_fn(
        gui=gui,
        semantic_palette=semantic_palette,
    )
    animating_block_positions = gui._get_animating_block_positions()
    tiles_drawn = render_visible_tiles_fn(
        gui=gui,
        map_surface=map_surface,
        start_r=start_r,
        end_r=end_r,
        start_c=start_c,
        end_c=end_c,
        collected_items=collected_items,
        collectible_tile_ids=collectible_tile_ids,
        animating_block_positions=animating_block_positions,
        semantic_palette=semantic_palette,
        math_module=math_module,
        time_module=time_module,
    )

    try:
        render_block_push_animations_fn(map_surface)
    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
        logger.warning("Failed to render block push animations: %s", exc)

    render_heatmap_overlay_fn(
        gui=gui,
        map_surface=map_surface,
        start_r=start_r,
        end_r=end_r,
        start_c=start_c,
        end_c=end_c,
        pygame=pygame,
    )

    render_jps_overlay_fn(
        gui=gui,
        map_surface=map_surface,
        start_r=start_r,
        end_r=end_r,
        start_c=start_c,
        end_c=end_c,
        pygame=pygame,
    )
    render_map_elites_overlay_fn(
        gui=gui,
        map_surface=map_surface,
        pygame=pygame,
    )

    render_planned_path_overlay_fn(
        gui=gui,
        map_surface=map_surface,
        pygame=pygame,
        math_module=math_module,
        time_module=time_module,
        logger=logger,
    )

    try:
        render_path_guaranteed_fn(map_surface)
    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
        logger.warning("_render_path_GUARANTEED failed: %s", exc)

    render_player_and_effects_fn(gui=gui, map_surface=map_surface)

    handle_empty_frame_recovery_fn(
        gui=gui,
        map_surface=map_surface,
        view_w=view_w,
        view_h=view_h,
        tiles_drawn=tiles_drawn,
        pygame=pygame,
        logger=logger,
    )

    gui.screen.blit(map_surface, (0, 0))

    render_translucent_event_overlays_fn(
        gui=gui,
        view_w=view_w,
        view_h=view_h,
        pygame=pygame,
        logger=logger,
    )

    pr, pc = gui.env.state.position
    sidebar_x = draw_sidebar_shell_fn(gui=gui, pygame=pygame)
    render_sidebar_content_fn(
        gui=gui,
        sidebar_x=sidebar_x,
        map_w=map_w,
        map_h=map_h,
        player_row=pr,
        player_col=pc,
        pygame=pygame,
        time_module=time_module,
        math_module=math_module,
        semantic_palette=semantic_palette,
        logger=logger,
        render_sidebar_header_fn=render_sidebar_header_fn,
        render_sidebar_status_fn=render_sidebar_status_fn,
    )
    render_post_map_layers_fn(
        gui=gui,
        pygame=pygame,
        logger=logger,
        render_preview_layer_fn=render_preview_layer_fn,
    )
    render_top_ui_layers_fn(gui=gui, logger=logger)
