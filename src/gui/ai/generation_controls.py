"""Control helpers for AI dungeon generation flow."""

from pathlib import Path


def start_ai_dungeon_generation(gui, threading_module):
    """Start AI generation in a background thread if not already running."""
    active_thread = getattr(gui, "ai_gen_thread", None)
    if active_thread and getattr(active_thread, "is_alive", lambda: False)():
        gui._set_message("AI generation already running", 1.5)
        return
    if getattr(gui, "ai_gen_done", False) and getattr(gui, "ai_gen_result", None) is not None:
        gui._set_message("AI generation result pending", 1.5)
        return

    gui.ai_gen_result = None
    gui.ai_gen_done = False
    thread = threading_module.Thread(target=gui._generate_ai_dungeon_worker, daemon=True)
    gui.ai_gen_thread = thread
    thread.start()
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
    gui._set_message(f"AI model selected: {checkpoint_path.name}", 2.5)
    logger.info("AI model checkpoint selected: %s", checkpoint_path)


def generate_level(gui, threading_module, logger):
    """Generate a level, using a selected AI checkpoint when available."""
    checkpoint_path = str(getattr(gui, "ai_checkpoint_path", "") or "").strip()
    if checkpoint_path:
        resolved_checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not resolved_checkpoint.exists():
            gui._set_message(f"Loaded AI model not found: {resolved_checkpoint}", 3.0)
            logger.warning("Loaded AI model checkpoint not found: %s", resolved_checkpoint)
            return
        gui.ai_checkpoint_path = str(resolved_checkpoint)
        start_ai_dungeon_generation(gui, threading_module)
        return

    logger.info("No AI checkpoint selected; using procedural level generator.")
    gui._generate_dungeon()
