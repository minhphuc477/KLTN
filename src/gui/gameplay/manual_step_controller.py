"""Helpers for manual step execution flow in GUI runner."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def manual_step(
    *,
    gui: Any,
    action: Any,
    action_deltas: dict,
    pop_effect_cls: Any,
    flash_effect_cls: Any,
    time_module: Any,
) -> None:
    """Execute one manual input step and update UI/effects/state."""
    if gui.env.done:
        return

    old_pos = gui.env.state.position
    old_keys = gui.env.state.keys
    old_bombs = gui.env.state.bomb_count
    old_boss_key = gui.env.state.has_boss_key
    old_state = SimpleNamespace(
        position=old_pos,
        keys=old_keys,
        bomb_count=old_bombs,
        has_boss_key=old_boss_key,
        has_bomb=bool(old_bombs > 0),
    )

    dr, dc = action_deltas.get(action, (0, 0))
    target_pos = (old_pos[0] + dr, old_pos[1] + dc)

    if 0 <= target_pos[0] < gui.env.height and 0 <= target_pos[1] < gui.env.width:
        gui._check_and_start_block_push(old_pos, target_pos, action)

    _state, _reward, done, info = gui.env.step(int(action))
    new_pos = gui.env.state.position

    gui.step_count += 1

    # Use the canonical tracking pipeline (same as auto-step) so that
    # collected_items, collected_positions, item_type_map, and effects
    # are all updated consistently.
    try:
        gui._track_item_collection(old_state, gui.env.state)
    except (AttributeError, RuntimeError, ValueError, TypeError):
        pass

    try:
        gui._track_item_usage(old_state, gui.env.state)
    except (AttributeError, RuntimeError, ValueError, TypeError):
        pass

    if gui.modern_hud:
        gui.modern_hud.update_game_state(
            keys=gui.env.state.keys,
            bombs=gui.env.state.bomb_count,
            has_boss_key=gui.env.state.has_boss_key,
            position=new_pos,
            steps=gui.step_count,
            message=getattr(gui, "message", ""),
        )
        if hasattr(gui.modern_hud, "keys_used"):
            gui.modern_hud.keys_used = getattr(gui, "keys_used", 0)
        if hasattr(gui.modern_hud, "bombs_used"):
            gui.modern_hud.bombs_used = getattr(gui, "bombs_used", 0)
        if hasattr(gui.modern_hud, "boss_keys_used"):
            gui.modern_hud.boss_keys_used = getattr(gui, "boss_keys_used", 0)

    if gui.renderer and new_pos != old_pos:
        gui.renderer.set_agent_position(new_pos[0], new_pos[1], immediate=False)
        if gui.effects:
            gui.effects.add_effect(pop_effect_cls(new_pos, (100, 255, 100)))

    if done:
        if gui.env.won:
            gui.message = "YOU WIN!"
            if gui.effects:
                goal_pos = gui.env.goal_pos
                gui.effects.add_effect(flash_effect_cls(goal_pos, (255, 215, 0), 0.5))
        else:
            gui.message = f"Game Over: {info.get('msg', '')}"
    else:
        msg = info.get("msg", "")
        if msg:
            gui.message = msg
