"""Native dialog for configuring GUI AI level generation."""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_AI_CONFIG: Dict[str, Any] = {
    "num_rooms": 12,
    "difficulty": "HARD",
    "max_keys": 3,
    "seed": None,
    "diffusion_steps": 50,
    "use_fast_sampler": False,
}

DIFFICULTY_CHOICES = ["EASY", "MEDIUM", "HARD", "EXPERT"]


def _initial_config(gui: Any) -> Dict[str, Any]:
    config = dict(DEFAULT_AI_CONFIG)
    existing = getattr(gui, "ai_generation_config", None)
    if isinstance(existing, dict):
        config.update({k: v for k, v in existing.items() if k in config})

    for attr_name, key in (
        ("ai_num_rooms", "num_rooms"),
        ("ai_difficulty", "difficulty"),
        ("ai_max_keys", "max_keys"),
        ("ai_seed", "seed"),
        ("ai_diffusion_steps", "diffusion_steps"),
        ("ai_use_fast_sampler", "use_fast_sampler"),
    ):
        if hasattr(gui, attr_name):
            value = getattr(gui, attr_name)
            if value is not None:
                config[key] = value
    return config


def ask_ai_generation_config(gui: Any, logger_obj: Any = None) -> Optional[Dict[str, Any]]:
    """Collect canonical AI generation settings from native input dialogs."""
    log = logger_obj or logger
    initial = _initial_config(gui)

    try:
        import tkinter as tk
        from tkinter import simpledialog
    except (ImportError, RuntimeError) as exc:
        log.warning("tkinter not available for AI generation config: %s", exc)
        gui._set_message("AI config dialog unavailable; using current settings", 2.0)
        fallback = dict(initial)
        if fallback.get("seed") is None:
            fallback["seed"] = random.randint(0, 999_999)
        fallback["use_fast_sampler"] = False
        return fallback

    tk_error = getattr(tk, "TclError", RuntimeError)
    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except (AttributeError, RuntimeError, ValueError, TypeError):
            pass

        num_rooms = simpledialog.askinteger(
            "AI Mission Nodes",
            "Number of mission rooms/nodes.\nRecommended for presentation: 10-18.",
            initialvalue=int(initial.get("num_rooms") or DEFAULT_AI_CONFIG["num_rooms"]),
            minvalue=5,
            maxvalue=24,
            parent=root,
        )
        if num_rooms is None:
            gui._set_message("AI generation cancelled", 1.5)
            return None

        max_keys = simpledialog.askinteger(
            "AI Key-Lock Pairs",
            "Maximum key-lock pairs.",
            initialvalue=int(initial.get("max_keys") or DEFAULT_AI_CONFIG["max_keys"]),
            minvalue=0,
            maxvalue=8,
            parent=root,
        )
        if max_keys is None:
            gui._set_message("AI generation cancelled", 1.5)
            return None

        difficulty_idx = simpledialog.askinteger(
            "AI Difficulty",
            "1 = EASY\n2 = MEDIUM\n3 = HARD\n4 = EXPERT",
            initialvalue=max(
                1,
                min(
                    4,
                    DIFFICULTY_CHOICES.index(str(initial.get("difficulty", "HARD")).upper()) + 1
                    if str(initial.get("difficulty", "HARD")).upper() in DIFFICULTY_CHOICES
                    else 3,
                ),
            ),
            minvalue=1,
            maxvalue=4,
            parent=root,
        )
        if difficulty_idx is None:
            gui._set_message("AI generation cancelled", 1.5)
            return None
        difficulty = DIFFICULTY_CHOICES[int(difficulty_idx) - 1]

        diffusion_steps = simpledialog.askinteger(
            "AI Diffusion Steps",
            "Canonical diffusion steps.\nHigher is slower but usually cleaner.",
            initialvalue=int(initial.get("diffusion_steps") or DEFAULT_AI_CONFIG["diffusion_steps"]),
            minvalue=8,
            maxvalue=100,
            parent=root,
        )
        if diffusion_steps is None:
            gui._set_message("AI generation cancelled", 1.5)
            return None

        seed_initial = "" if initial.get("seed") is None else str(initial.get("seed"))
        seed_str = simpledialog.askstring(
            "AI Seed",
            "Seed for reproducibility, or leave blank for a new random map.",
            initialvalue=seed_initial,
            parent=root,
        )
        if seed_str is None:
            gui._set_message("AI generation cancelled", 1.5)
            return None
        seed = None
        if seed_str.strip():
            try:
                seed = int(seed_str.strip())
            except ValueError:
                seed = hash(seed_str.strip()) % 1_000_000

        config = {
            "num_rooms": int(num_rooms),
            "difficulty": difficulty,
            "max_keys": int(max_keys),
            "seed": seed,
            "diffusion_steps": int(diffusion_steps),
            "use_fast_sampler": False,
        }
        log.info("AI generation config: %s", config)
        return config
    except (tk_error, AttributeError, RuntimeError, ValueError, TypeError, OSError) as exc:
        log.warning("AI generation config dialog failed: %s", exc)
        gui._set_message(f"AI config dialog error: {exc}", 3.0)
        return None
    finally:
        if root is not None:
            try:
                root.destroy()
            except (AttributeError, RuntimeError, ValueError, TypeError):
                pass
