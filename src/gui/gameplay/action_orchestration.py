"""Gameplay action and block-push orchestration bridges for ZeldaGUI."""

from __future__ import annotations


def execute_auto_solve(*, gui, path, solver_result, teleports, logger, execute_auto_solve_helper):
    execute_auto_solve_helper(
        gui=gui,
        path=path,
        solver_result=solver_result,
        teleports=teleports,
        logger=logger,
    )


def execute_auto_solve_from_preview(*, gui, logger, execute_auto_solve_from_preview_helper):
    execute_auto_solve_from_preview_helper(gui=gui, logger=logger)


def smart_grid_path(
    *,
    gui,
    logger,
    convert_diagonal_to_4dir,
    semantic_palette,
    np_module,
    path_cls,
    os_module,
    smart_grid_path_helper,
):
    return smart_grid_path_helper(
        gui=gui,
        logger=logger,
        convert_diagonal_to_4dir=convert_diagonal_to_4dir,
        semantic_palette=semantic_palette,
        np_module=np_module,
        path_cls=path_cls,
        os_module=os_module,
    )


def graph_guided_path(*, gui, graph_guided_path_helper):
    return graph_guided_path_helper(gui=gui)


def hybrid_graph_grid_path(*, gui, hybrid_graph_grid_path_helper):
    return hybrid_graph_grid_path_helper(gui=gui)


def stop_auto(*, gui, reason, logger, stop_auto_helper):
    return stop_auto_helper(gui=gui, reason=reason, logger=logger)


def auto_step(
    *,
    gui,
    logger,
    game_state_cls,
    action_enum,
    ripple_effect_cls,
    flash_effect_cls,
    traceback_module,
    auto_step_helper,
):
    return auto_step_helper(
        gui=gui,
        logger=logger,
        game_state_cls=game_state_cls,
        action_enum=action_enum,
        ripple_effect_cls=ripple_effect_cls,
        flash_effect_cls=flash_effect_cls,
        traceback_module=traceback_module,
    )


def manual_step(*, gui, action, action_deltas, pop_effect_cls, flash_effect_cls, time_module, manual_step_helper):
    return manual_step_helper(
        gui=gui,
        action=action,
        action_deltas=action_deltas,
        pop_effect_cls=pop_effect_cls,
        flash_effect_cls=flash_effect_cls,
        time_module=time_module,
    )


def start_block_push_animation(*, gui, block_from, block_to, pygame, logger, start_block_push_animation_helper):
    start_block_push_animation_helper(gui, block_from, block_to, pygame, logger)


def update_block_push_animations(*, gui, pygame, semantic_palette, pop_effect_cls, logger, update_block_push_animations_helper):
    update_block_push_animations_helper(gui, pygame, semantic_palette, pop_effect_cls, logger)


def render_block_push_animations(*, gui, surface, pygame, semantic_palette, render_block_push_animations_helper):
    render_block_push_animations_helper(gui, surface, pygame, semantic_palette)


def get_animating_block_positions(*, gui, get_animating_block_positions_helper):
    return get_animating_block_positions_helper(gui)


def check_and_start_block_push(*, gui, player_pos, target_pos, walkable_ids, pushable_ids, check_and_start_block_push_helper):
    return check_and_start_block_push_helper(gui, player_pos, target_pos, walkable_ids, pushable_ids)
