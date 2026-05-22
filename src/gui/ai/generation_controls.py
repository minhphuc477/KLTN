"""Control helpers for AI dungeon generation flow."""

from pathlib import Path
import logging


logger = logging.getLogger(__name__)


def _sync_ai_generation_config_to_gui(gui, config):
    """Persist dialog/config values onto the GUI object for the worker."""
    if not isinstance(config, dict):
        return
    gui.ai_generation_config = dict(config)
    gui.ai_num_rooms = int(config.get("num_rooms", getattr(gui, "ai_num_rooms", 12)) or 12)
    gui.ai_difficulty = str(config.get("difficulty", getattr(gui, "ai_difficulty", "HARD")) or "HARD").upper()
    gui.ai_max_keys = int(config.get("max_keys", getattr(gui, "ai_max_keys", 3)) or 0)
    gui.ai_seed = config.get("seed", None)
    gui.ai_diffusion_steps = int(
        config.get("diffusion_steps", getattr(gui, "ai_diffusion_steps", 50)) or 50
    )
    gui.ai_use_fast_sampler = False


def _configure_ai_generation_if_needed(gui, logger_obj=None) -> bool:
    """Prompt for AI generation settings when enabled."""
    log = logger_obj or logger
    if not bool(getattr(gui, "ai_generation_prompt_enabled", False)):
        return True

    import os

    if str(os.environ.get("KLTN_AI_SKIP_CONFIG_DIALOG", "")).strip().lower() in {"1", "true", "yes", "on"}:
        return True

    try:
        from src.gui.ai.generation_config_dialog import ask_ai_generation_config
    except (ImportError, RuntimeError, ValueError, TypeError) as exc:
        log.warning("AI generation config dialog unavailable: %s", exc)
        return True

    config = ask_ai_generation_config(gui, logger_obj=log)
    if config is None:
        return False
    previous_fast = bool(getattr(gui, "ai_use_fast_sampler", False))
    _sync_ai_generation_config_to_gui(gui, config)
    if previous_fast != bool(getattr(gui, "ai_use_fast_sampler", False)):
        gui._ai_generation_pipeline_cache = None
    gui._set_message(
        "AI config: "
        f"{gui.ai_num_rooms} rooms, {gui.ai_difficulty}, "
        f"{gui.ai_diffusion_steps} diffusion steps",
        2.5,
    )
    return True


def _select_discovered_best_checkpoint(gui, logger_obj=None) -> bool:
    """Select the repo's preferred GUI checkpoint when discovery is enabled."""
    log = logger_obj or logger
    if not bool(getattr(gui, "prefer_ai_checkpoint_discovery", False)):
        return False
    try:
        from src.gui.ai.generation_pipeline import discover_best_output_checkpoint
    except (ImportError, RuntimeError, ValueError, TypeError):
        return False
    discovered = discover_best_output_checkpoint()
    if discovered is None or not Path(discovered).exists():
        return False
    gui.ai_checkpoint_path = str(Path(discovered).resolve())
    log.info("Using discovered best AI checkpoint: %s", gui.ai_checkpoint_path)
    return True


def start_ai_dungeon_generation(gui, threading_module, logger_obj=None):
    """Start AI generation in a background thread if not already running."""
    active_thread = getattr(gui, "ai_gen_thread", None)
    if active_thread and getattr(active_thread, "is_alive", lambda: False)():
        gui._set_message("AI generation already running", 1.5)
        gui.status_message = "AI generation running..."
        return
    if getattr(gui, "ai_gen_done", False) and getattr(gui, "ai_gen_result", None) is not None:
        gui._set_message("AI generation result pending", 1.5)
        gui.status_message = "AI generation result pending"
        return

    if not _configure_ai_generation_if_needed(gui, logger_obj=logger_obj):
        return
    if not str(getattr(gui, "ai_checkpoint_path", "") or "").strip():
        _select_discovered_best_checkpoint(gui, logger_obj or logger)

    gui.ai_gen_result = None
    gui.ai_gen_done = False
    thread = threading_module.Thread(target=gui._generate_ai_dungeon_worker, daemon=True)
    gui.ai_gen_thread = thread
    gui.status_message = "AI generation starting..."
    try:
        thread.start()
    except (RuntimeError, ValueError, TypeError) as exc:
        gui.ai_gen_thread = None
        gui.ai_gen_done = False
        gui.ai_gen_result = None
        gui.status_message = "AI generation failed to start"
        gui._set_message(f"AI generation failed to start: {exc}", 3.0)
        return

    gui.status_message = "AI generation running..."
    gui._set_message("AI generation started (background)")


def select_ai_model_checkpoint(gui, logger):
    """Select an AI checkpoint path for future generation runs."""
    def _select_existing_checkpoint(checkpoint_value) -> bool:
        checkpoint_text = str(checkpoint_value or "").strip()
        if not checkpoint_text:
            return False
        checkpoint_path = Path(checkpoint_text).expanduser().resolve()
        if not checkpoint_path.exists():
            return False
        gui.ai_checkpoint_path = str(checkpoint_path)
        gui._ai_generation_pipeline_cache = None
        gui._set_message(f"AI model selected: {checkpoint_path.name}", 2.5)
        logger.info("AI model checkpoint selected: %s", checkpoint_path)
        return True

    try:
        import tkinter as tk
        from tkinter import filedialog
    except (ImportError, RuntimeError, ValueError, TypeError) as exc:
        if _select_existing_checkpoint(getattr(gui, "ai_checkpoint_path", None)):
            return
        logger.warning("Model file picker unavailable: %s", exc)
        gui._set_message("Model picker unavailable; set KLTN_CHECKPOINT_PATH", 3.0)
        return

    tk_error = getattr(tk, "TclError", RuntimeError)
    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except (AttributeError, RuntimeError, ValueError, TypeError):
            pass

        initial_dir = Path.cwd()
        current = str(getattr(gui, "ai_checkpoint_path", "") or "").strip()
        if current:
            current_parent = Path(current).expanduser().parent
            if current_parent.exists():
                initial_dir = current_parent

        selected = filedialog.askopenfilename(
            title="Load AI checkpoint",
            initialdir=str(initial_dir),
            filetypes=[
                ("PyTorch checkpoints", "*.pth *.pt *.ckpt"),
                ("All files", "*.*"),
            ],
        )
    except (tk_error, AttributeError, RuntimeError, ValueError, TypeError, OSError) as exc:
        logger.warning("Model file picker failed: %s", exc)
        if _select_existing_checkpoint(getattr(gui, "ai_checkpoint_path", None)):
            return
        gui._set_message(f"Model picker failed: {exc}", 3.0)
        return
    finally:
        if root is not None:
            try:
                root.destroy()
            except (AttributeError, RuntimeError, ValueError, TypeError):
                pass

    if not selected:
        gui._set_message("Model load cancelled", 1.5)
        return

    checkpoint_path = Path(selected).expanduser().resolve()
    if not checkpoint_path.exists():
        gui._set_message(f"Model not found: {checkpoint_path}", 3.0)
        logger.warning("Selected AI model checkpoint does not exist: %s", checkpoint_path)
        return

    gui.ai_checkpoint_path = str(checkpoint_path)
    gui._ai_generation_pipeline_cache = None
    gui._set_message(f"AI model selected: {checkpoint_path.name}", 2.5)
    logger.info("AI model checkpoint selected: %s", checkpoint_path)


def generate_level(gui, threading_module, logger):
    """Generate a level, using a selected AI checkpoint when available."""
    checkpoint_path = str(getattr(gui, "ai_checkpoint_path", "") or "").strip()
    if not checkpoint_path and _select_discovered_best_checkpoint(gui, logger):
        checkpoint_path = str(getattr(gui, "ai_checkpoint_path", "") or "").strip()

    if checkpoint_path:
        resolved_checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not resolved_checkpoint.exists():
            gui._set_message(f"Loaded AI model not found: {resolved_checkpoint}", 3.0)
            logger.warning("Loaded AI model checkpoint not found: %s", resolved_checkpoint)
            return
        gui.ai_checkpoint_path = str(resolved_checkpoint)
        start_ai_dungeon_generation(gui, threading_module, logger_obj=logger)
        return

    logger.info("No AI checkpoint selected; using procedural level generator.")
    gui._generate_dungeon()
