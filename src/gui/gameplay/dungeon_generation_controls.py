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
    """Generate a procedural dungeon with user-specified configuration.

    Opens native dialogs to collect width, height, num_rooms, difficulty,
    max_keys, and seed. Falls back to defaults when dialogs are unavailable.
    """
    try:
        from src.generation.dungeon_generator import DungeonGenerator, Difficulty
        from src.gui.gameplay.generation_config_dialog import ask_generation_config

        # Ask user for specs via native dialogs
        config = ask_generation_config(gui, logger_obj=logger)
        if config is None:
            return  # User cancelled

        difficulty_map = {
            "EASY": Difficulty.EASY,
            "MEDIUM": Difficulty.MEDIUM,
            "HARD": Difficulty.HARD,
            "EXPERT": Difficulty.EXPERT,
        }
        difficulty = difficulty_map.get(config.get("difficulty", "MEDIUM"), Difficulty.MEDIUM)
        seed = config.get("seed", 0)
        width = config.get("width", 40)
        height = config.get("height", 40)

        generator = DungeonGenerator(
            width=width,
            height=height,
            difficulty=difficulty,
            seed=seed,
        )

        grid = generator.generate()

        dungeon_name = f"Generated #{seed} ({width}\u00d7{height})"
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

        gui._set_message(
            f"Generated dungeon (seed: {seed}, {len(generator.rooms)} rooms, "
            f"{width}\u00d7{height}, {config.get('difficulty', 'MEDIUM')})"
        )
        logger.info(
            "Generated dungeon: seed=%s, rooms=%s, keys=%s, size=%dx%d, difficulty=%s",
            seed,
            len(generator.rooms),
            len(generator.key_positions),
            width,
            height,
            config.get("difficulty", "MEDIUM"),
        )

    except ImportError as exc:
        logger.warning("Dungeon generator not available: %s", exc)
        gui._set_message("Dungeon generator module not found")
    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
        logger.exception("Failed to generate dungeon: %s", exc)
        gui._set_message(f"Generation failed: {str(exc)}")
