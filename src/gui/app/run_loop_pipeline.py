"""Run-loop orchestration helpers for ZeldaGUI."""

from __future__ import annotations


def run_main_loop(
    gui,
    *,
    max_frames,
    pygame,
    os_module,
    logger,
    time_module,
    np_module,
    action_enum,
    checkbox_widget_cls,
    path_preview_dialog_cls,
    safe_unpickle_fn,
    should_attempt_focus_fallback_fn,
    poll_pygame_events_fn,
    run_input_focus_fallback_fn,
    clear_stale_preview_overlay_fn,
    handle_window_focus_event_fn,
    handle_global_keydown_shortcuts_fn,
    handle_preview_overlay_events_fn,
    handle_videoresize_event_fn,
    handle_mousewheel_event_fn,
    handle_mouse_button_down_preamble_fn,
    handle_mouse_button_down_event_fn,
    handle_mouse_button_up_event_fn,
    handle_mouse_motion_diagnostics_fn,
    handle_mouse_motion_event_fn,
    handle_keyup_event_fn,
    handle_keydown_event_fn,
    run_auto_step_tick_fn,
    run_continuous_movement_tick_fn,
    update_heartbeat_fn,
    handle_pending_solver_trigger_fn,
    handle_parallel_search_completion_fn,
    handle_preview_process_completion_fn,
    handle_solver_process_completion_fn,
    handle_ai_generation_completion_fn,
    render_and_present_frame_fn,
    handle_watchdog_screenshot_request_fn,
    run_periodic_display_health_check_fn,
    advance_frame_and_check_limit_fn,
    tick_frame_clock_fn,
    compute_solver_timeout_seconds_fn,
    find_path_tile_violations_fn,
    debug_input_active,
):
    """Execute the GUI main loop while preserving existing behavior and ordering."""
    heartbeat_last = time_module.time()
    heartbeat_interval = 0.5

    running = True
    frame_count = 0

    while running:
        current_time = time_module.time()
        gui.delta_time = current_time - gui.last_frame_time
        gui.last_frame_time = current_time

        events = poll_pygame_events_fn(pygame, time_module, logger)

        run_input_focus_fallback_fn(
            gui,
            pygame,
            time_module,
            logger,
            should_attempt_focus_fallback_fn,
        )
        for event in events:
            clear_stale_preview_overlay_fn(gui, logger)

            if handle_window_focus_event_fn(gui, event, pygame, logger):
                continue

            if handle_global_keydown_shortcuts_fn(
                gui,
                event,
                pygame,
                time_module,
                logger,
                checkbox_widget_cls,
            ):
                continue

            if handle_preview_overlay_events_fn(gui, event, pygame):
                continue

            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.VIDEORESIZE:
                handle_videoresize_event_fn(gui, event, pygame, logger)

            elif event.type == pygame.MOUSEWHEEL:
                handle_mousewheel_event_fn(gui, event, pygame, time_module)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos, consumed = handle_mouse_button_down_preamble_fn(
                    gui,
                    event,
                    pygame,
                    time_module,
                    logger,
                    debug_input_active,
                )
                if consumed:
                    continue
                if handle_mouse_button_down_event_fn(gui, event, mouse_pos, pygame):
                    continue

            elif event.type == pygame.MOUSEBUTTONUP:
                if handle_mouse_button_up_event_fn(gui, event, pygame, time_module, logger):
                    continue

            elif event.type == pygame.MOUSEMOTION:
                mouse_pos = handle_mouse_motion_diagnostics_fn(gui, event, pygame, time_module, logger)
                handle_mouse_motion_event_fn(gui, event, mouse_pos, pygame)

            elif event.type == pygame.KEYUP:
                handle_keyup_event_fn(gui, event, logger)

            elif event.type == pygame.KEYDOWN:
                running = handle_keydown_event_fn(
                    gui,
                    event,
                    pygame,
                    os_module,
                    logger,
                    checkbox_widget_cls,
                    action_enum,
                    running,
                )

        run_auto_step_tick_fn(gui, logger, frame_count)

        if gui.widget_manager:
            mouse_pos = pygame.mouse.get_pos()
            gui.widget_manager.update(mouse_pos, gui.delta_time)

        run_continuous_movement_tick_fn(gui, pygame, action_enum)

        gui._update_toasts()

        gui.frame_count = frame_count
        heartbeat_last = update_heartbeat_fn(gui, logger, time_module, heartbeat_last, heartbeat_interval)

        handle_pending_solver_trigger_fn(gui, logger)

        gui._update_control_panel_animation()
        gui._update_control_panel_scroll()

        handle_parallel_search_completion_fn(gui, logger, path_preview_dialog_cls)

        handle_preview_process_completion_fn(
            gui,
            os_module,
            logger,
            safe_unpickle_fn,
            path_preview_dialog_cls,
        )

        handle_solver_process_completion_fn(
            gui,
            os_module,
            time_module,
            np_module,
            logger,
            compute_solver_timeout_seconds_fn,
            safe_unpickle_fn,
            find_path_tile_violations_fn,
            path_preview_dialog_cls,
        )

        handle_ai_generation_completion_fn(gui)

        render_and_present_frame_fn(gui, pygame, logger)

        handle_watchdog_screenshot_request_fn(gui, logger)

        run_periodic_display_health_check_fn(gui, time_module, logger)

        frame_count, keep_running = advance_frame_and_check_limit_fn(frame_count, max_frames, logger)
        if not keep_running:
            running = False

        tick_frame_clock_fn(gui)

    pygame.quit()
