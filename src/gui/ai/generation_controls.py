"""Control helpers for AI dungeon generation flow."""


def start_ai_dungeon_generation(gui, threading_module):
    """Start AI generation in a background thread if not already running."""
    active_thread = getattr(gui, "ai_gen_thread", None)
    if active_thread and getattr(active_thread, "is_alive", lambda: False)():
        gui._set_message("AI generation already running", 1.5)
        return

    gui.ai_gen_result = None
    gui.ai_gen_done = False
    thread = threading_module.Thread(target=gui._generate_ai_dungeon_worker, daemon=True)
    gui.ai_gen_thread = thread
    thread.start()
    gui._set_message("AI generation started (background)")
