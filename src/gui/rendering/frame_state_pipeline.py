"""Per-frame state update and map-layer actor rendering helpers."""

from __future__ import annotations

from typing import Any


def update_frame_render_state(*, gui: Any, logger: Any) -> None:
    """Update renderer/effects state and synchronize HUD/inventory counters."""
    effective_dt = gui.delta_time * gui.speed_multiplier
    if gui.renderer:
        gui.renderer.update(effective_dt)
    if gui.effects:
        gui.effects.update(effective_dt)

    gui._update_block_push_animations()

    if getattr(gui, "inventory_needs_refresh", False):
        try:
            logger.debug("Processing deferred inventory refresh on main thread")
            gui._update_inventory_and_hud()
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            pass
        finally:
            gui.inventory_needs_refresh = False

    if not gui.modern_hud or not gui.env:
        return

    gui.modern_hud.update_game_state(
        keys=gui.env.state.keys,
        bombs=1 if gui.env.state.has_bomb else 0,
        has_boss_key=gui.env.state.has_boss_key,
        position=gui.env.state.position,
        steps=gui.step_count,
        message=gui.message,
    )

    if hasattr(gui.modern_hud, "inventory"):
        gui.modern_hud.inventory.keys_collected = gui.keys_collected
        gui.modern_hud.inventory.bombs_collected = gui.bombs_collected
        gui.modern_hud.inventory.boss_keys_collected = gui.boss_keys_collected
        gui.modern_hud.inventory.keys_used = getattr(gui, "keys_used", 0)
        gui.modern_hud.inventory.bombs_used = getattr(gui, "bombs_used", 0)
        gui.modern_hud.inventory.boss_keys_used = getattr(gui, "boss_keys_used", 0)

    if hasattr(gui.modern_hud, "keys_collected"):
        gui.modern_hud.keys_collected = gui.keys_collected
        gui.modern_hud.bombs_collected = gui.bombs_collected
        gui.modern_hud.boss_keys_collected = gui.boss_keys_collected
    if hasattr(gui.modern_hud, "keys_used"):
        gui.modern_hud.keys_used = getattr(gui, "keys_used", 0)
    if hasattr(gui.modern_hud, "bombs_used"):
        gui.modern_hud.bombs_used = getattr(gui, "bombs_used", 0)
    if hasattr(gui.modern_hud, "boss_keys_used"):
        gui.modern_hud.boss_keys_used = getattr(gui, "boss_keys_used", 0)


def render_player_and_effects(*, gui: Any, map_surface: Any) -> None:
    """Render player sprite and runtime effects onto map surface."""
    if gui.renderer and gui.renderer.agent_visual_pos:
        visual_pos = gui.renderer.agent_visual_pos
        link_x = int(visual_pos.x * gui.TILE_SIZE - gui.view_offset_x + 2)
        link_y = int(visual_pos.y * gui.TILE_SIZE - gui.view_offset_y + 2)
    else:
        player_row, player_col = gui.env.state.position
        link_x = player_col * gui.TILE_SIZE - gui.view_offset_x + 2
        link_y = player_row * gui.TILE_SIZE - gui.view_offset_y + 2

    map_surface.blit(gui.link_img, (link_x, link_y))

    if gui.effects:
        gui.effects.render(map_surface, gui.TILE_SIZE, (gui.view_offset_x, gui.view_offset_y))

