"""Generation configuration dialog for ZeldaGUI.

Uses native tkinter dialogs to collect generation parameters before
creating a new dungeon. This avoids the complexity of building text-input
widgets inside pygame while giving the user full control over specs.

Supported Parameters:
- Grid width/height (BSP generator)
- Number of rooms/nodes (MissionGrammar)
- Difficulty level
- Random seed (optional)
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Default generation presets
DEFAULT_CONFIG = {
    "width": 40,
    "height": 40,
    "num_rooms": 8,
    "difficulty": "MEDIUM",
    "seed": None,  # None = random
    "max_keys": 2,
}

DIFFICULTY_CHOICES = ["EASY", "MEDIUM", "HARD", "EXPERT"]


def ask_generation_config(gui: Any, logger_obj: Any = None) -> Optional[Dict[str, Any]]:
    """Open native dialog boxes to collect dungeon generation parameters.

    Returns:
        A dict with keys ``width``, ``height``, ``num_rooms``, ``difficulty``,
        ``seed``, ``max_keys`` — or ``None`` if the user cancelled.
    """
    log = logger_obj or logger

    try:
        import tkinter as tk
        from tkinter import simpledialog, messagebox
    except (ImportError, RuntimeError) as exc:
        log.warning("tkinter not available for generation config: %s", exc)
        gui._set_message("Config dialog unavailable; using defaults")
        return dict(DEFAULT_CONFIG)

    tk_error = getattr(tk, "TclError", RuntimeError)
    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except (AttributeError, RuntimeError, ValueError, TypeError):
            pass

        # Width and height are now auto-calculated based on num_rooms

        # --- Number of Rooms ---
        num_rooms = simpledialog.askinteger(
            "Number of Rooms",
            "Enter number of rooms/nodes:\n\n"
            "This controls BSP splits and graph complexity.\n"
            "Recommended: 4-20",
            initialvalue=DEFAULT_CONFIG["num_rooms"],
            minvalue=2,
            maxvalue=50,
            parent=root,
        )
        if num_rooms is None:
            gui._set_message("Generation cancelled")
            return None

        # --- Max Keys ---
        max_keys = simpledialog.askinteger(
            "Max Keys",
            "Number of key-lock pairs:\n\nMore keys = more complex progression.",
            initialvalue=DEFAULT_CONFIG["max_keys"],
            minvalue=0,
            maxvalue=10,
            parent=root,
        )
        if max_keys is None:
            gui._set_message("Generation cancelled")
            return None

        # --- Difficulty ---
        difficulty_idx = simpledialog.askinteger(
            "Difficulty",
            "Select difficulty level:\n\n"
            "1 = EASY\n2 = MEDIUM\n3 = HARD\n4 = EXPERT",
            initialvalue=2,
            minvalue=1,
            maxvalue=4,
            parent=root,
        )
        if difficulty_idx is None:
            gui._set_message("Generation cancelled")
            return None
        difficulty = DIFFICULTY_CHOICES[max(0, min(3, difficulty_idx - 1))]

        # --- Seed (optional) ---
        seed_str = simpledialog.askstring(
            "Random Seed (optional)",
            "Enter a seed for reproducibility,\nor leave blank for random:",
            initialvalue="",
            parent=root,
        )
        seed = None
        if seed_str is not None and seed_str.strip():
            try:
                seed = int(seed_str.strip())
            except ValueError:
                # Use hash of string as seed
                seed = hash(seed_str.strip()) % 1_000_000

        # Auto-size the grid based on room count so the user doesn't have to guess
        auto_grid_size = max(30, int((num_rooms ** 0.5) * 16))

        config = {
            "width": auto_grid_size,
            "height": auto_grid_size,
            "num_rooms": num_rooms,
            "difficulty": difficulty,
            "seed": seed if seed is not None else random.randint(0, 999_999),
            "max_keys": max_keys,
        }

        log.info("Generation config: %s", config)
        return config

    except (tk_error, AttributeError, RuntimeError, ValueError, TypeError, OSError) as exc:
        log.warning("Generation config dialog failed: %s", exc)
        gui._set_message(f"Config dialog error: {exc}")
        return None
    finally:
        if root is not None:
            try:
                root.destroy()
            except (AttributeError, RuntimeError, ValueError, TypeError):
                pass
