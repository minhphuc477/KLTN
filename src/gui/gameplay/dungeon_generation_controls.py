"""Procedural dungeon generation and stop controls for ZeldaGUI."""

from __future__ import annotations

from typing import Any


def stop_auto_solve(gui: Any) -> None:
    """Stop auto-solve and clear transient animation/path state."""
    gui.auto_mode = False
    gui.auto_path = []
    gui.auto_step_idx = 0
    gui.block_push_animations = []
    gui.message = "Auto-solve stopped"


def generate_dungeon(gui: Any, logger: Any) -> None:
    """Generate a procedural dungeon and switch the GUI to the new map."""
    try:
        from src.generation.dungeon_generator import DungeonGenerator, Difficulty
        import random

        seed = random.randint(0, 999999)

        generator = DungeonGenerator(
            width=40,
            height=40,
            difficulty=Difficulty.MEDIUM,
            seed=seed,
        )

        grid = generator.generate()

        dungeon_name = f"Generated #{seed}"
        gui.maps.append(grid)
        gui.map_names.append(dungeon_name)

        gui.current_map_idx = len(gui.maps) - 1
        gui._load_current_map()
        gui._center_view()

        if gui.effects:
            gui.effects.clear()
        gui.step_count = 0
        gui.auto_path = []
        gui.auto_mode = False

        gui._set_message(f"Generated dungeon (seed: {seed}, {len(generator.rooms)} rooms)")
        logger.info(
            "Generated dungeon: seed=%s, rooms=%s, keys=%s",
            seed,
            len(generator.rooms),
            len(generator.key_positions),
        )

    except ImportError as exc:
        logger.warning("Dungeon generator not available: %s", exc)
        gui._set_message("Dungeon generator module not found")
    except (AttributeError, RuntimeError, ValueError, TypeError):
        logger.exception("Failed to generate dungeon: %s", exc)
        gui._set_message(f"Generation failed: {str(exc)}")
