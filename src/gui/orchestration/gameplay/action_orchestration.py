"""Gameplay action and block-push orchestration bridges for ZeldaGUI."""

from __future__ import annotations

from src.gui.gameplay.auto_solve_execution import (
    execute_auto_solve as _execute_auto_solve,
    execute_auto_solve_from_preview as _execute_auto_solve_from_preview,
)
from src.gui.gameplay.auto_step_controller import auto_step as _auto_step, stop_auto as _stop_auto
from src.gui.gameplay.block_push_controls import (
    check_and_start_block_push as _check_and_start_block_push,
    get_animating_block_positions as _get_animating_block_positions,
    render_block_push_animations as _render_block_push_animations,
    start_block_push_animation as _start_block_push_animation,
    update_block_push_animations as _update_block_push_animations,
)
from src.gui.gameplay.manual_step_controller import manual_step as _manual_step
from src.gui.gameplay.path_strategies import (
    graph_guided_path as _graph_guided_path,
    hybrid_graph_grid_path as _hybrid_graph_grid_path,
    smart_grid_path as _smart_grid_path,
)


def execute_auto_solve(*, gui, path, solver_result, teleports, logger):
    _execute_auto_solve(
        gui=gui,
        path=path,
        solver_result=solver_result,
        teleports=teleports,
        logger=logger,
    )


def execute_auto_solve_from_preview(*, gui, logger):
    _execute_auto_solve_from_preview(gui=gui, logger=logger)


def smart_grid_path(
    *,
    gui,
    logger,
    convert_diagonal_to_4dir,
    semantic_palette,
    np_module,
    path_cls,
    os_module,
):
    return _smart_grid_path(
        gui=gui,
        logger=logger,
        convert_diagonal_to_4dir=convert_diagonal_to_4dir,
        semantic_palette=semantic_palette,
        np_module=np_module,
        path_cls=path_cls,
        os_module=os_module,
    )


def graph_guided_path(*, gui):
    return _graph_guided_path(gui=gui)


def hybrid_graph_grid_path(*, gui):
    return _hybrid_graph_grid_path(gui=gui)


def stop_auto(*, gui, reason, logger):
    return _stop_auto(gui=gui, reason=reason, logger=logger)


def auto_step(
    *,
    gui,
    logger,
    game_state_cls,
    action_enum,
    ripple_effect_cls,
    flash_effect_cls,
    traceback_module,
):
    return _auto_step(
        gui=gui,
        logger=logger,
        game_state_cls=game_state_cls,
        action_enum=action_enum,
        ripple_effect_cls=ripple_effect_cls,
        flash_effect_cls=flash_effect_cls,
        traceback_module=traceback_module,
    )


def manual_step(*, gui, action, action_deltas, pop_effect_cls, flash_effect_cls, time_module):
    return _manual_step(
        gui=gui,
        action=action,
        action_deltas=action_deltas,
        pop_effect_cls=pop_effect_cls,
        flash_effect_cls=flash_effect_cls,
        time_module=time_module,
    )


def start_block_push_animation(*, gui, block_from, block_to, pygame, logger):
    _start_block_push_animation(gui, block_from, block_to, pygame, logger)


def update_block_push_animations(*, gui, pygame, semantic_palette, pop_effect_cls, logger):
    _update_block_push_animations(gui, pygame, semantic_palette, pop_effect_cls, logger)


def render_block_push_animations(*, gui, surface, pygame, semantic_palette):
    _render_block_push_animations(gui, surface, pygame, semantic_palette)


def get_animating_block_positions(*, gui):
    return _get_animating_block_positions(gui)


def check_and_start_block_push(*, gui, player_pos, target_pos, walkable_ids, pushable_ids):
    return _check_and_start_block_push(gui, player_pos, target_pos, walkable_ids, pushable_ids)
