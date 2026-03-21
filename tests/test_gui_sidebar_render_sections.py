from types import SimpleNamespace

from src.gui.controls.sidebar_render_sections import (
    render_sidebar_header_inventory_solver,
    render_sidebar_status_message_metrics_controls,
)


class _Text:
    def __init__(self, text):
        self.text = text


class _Font:
    def render(self, text, _aa, _color):
        return _Text(text)


class _Screen:
    def __init__(self):
        self.blits = []

    def blit(self, obj, pos):
        self.blits.append((obj, pos))


class _Draw:
    def __init__(self):
        self.lines = []
        self.rects = []

    def line(self, surface, color, p1, p2, width=1):
        self.lines.append((surface, color, p1, p2, width))

    def rect(self, surface, color, rect):
        self.rects.append((surface, color, rect))


class _Pygame:
    def __init__(self):
        self.draw = _Draw()


class _Time:
    @staticmethod
    def time():
        return 10.0


class _Logger:
    def __init__(self):
        self.debug_calls = []

    def debug(self, *args, **kwargs):
        self.debug_calls.append((args, kwargs))


def test_render_sidebar_header_inventory_solver_runs_and_logs():
    screen = _Screen()
    logger = _Logger()
    sync_state = {"called": False}

    def _sync_inventory_counters():
        sync_state["called"] = True

    gui = SimpleNamespace(
        big_font=_Font(),
        font=_Font(),
        small_font=_Font(),
        current_map_idx=0,
        map_names=["Dungeon A"],
        maps=[[[0]]],
        screen_w=1024,
        item_pickup_times={"key": 9.5, "bomb": 9.5, "boss_key": 9.5},
        _sync_inventory_counters=_sync_inventory_counters,
        _inv_render_frame_count=59,
        keys_collected=1,
        total_keys=2,
        bombs_collected=1,
        total_bombs=2,
        boss_keys_collected=0,
        total_boss_keys=1,
        collected_items=["key", "bomb"],
        env=SimpleNamespace(state=SimpleNamespace(keys=1, has_bomb=True, has_boss_key=False)),
        last_pickup_msg="Picked key",
        last_use_msg="Used bomb",
        solver_result={"keys_available": 3, "keys_used": 1, "edge_types": ["open", "open", "key_locked"]},
    )

    y = render_sidebar_header_inventory_solver(
        gui=gui,
        screen=screen,
        sidebar_x=0,
        y_pos=10,
        map_w=8,
        map_h=6,
        time_module=_Time,
        math_module=__import__("math"),
        pygame=_Pygame(),
        logger=logger,
    )

    assert y > 10
    assert sync_state["called"]
    assert screen.blits
    assert logger.debug_calls


def test_render_sidebar_status_message_metrics_controls_draws_hud_and_victory():
    screen = _Screen()
    pygame = _Pygame()

    gui = SimpleNamespace(
        font=_Font(),
        small_font=_Font(),
        big_font=_Font(),
        screen_w=1024,
        screen_h=768,
        SIDEBAR_WIDTH=260,
        HUD_HEIGHT=100,
        map_names=["Dungeon A"],
        current_map_idx=0,
        env=SimpleNamespace(
            state=SimpleNamespace(position=(2, 3)),
            won=True,
            _find_all_positions=lambda _tile: [(1, 1), (2, 2)],
        ),
        auto_mode=False,
        dstar_active=True,
        dstar_solver=SimpleNamespace(replans_count=2),
        feature_flags={"dstar_lite": True},
        stair_sprite=object(),
        message="This is a long status message to wrap in the sidebar area",
        message_time=9.8,
        message_duration=3.0,
        step_count=42,
        speed_multiplier=1.0,
        TILE_SIZE=16,
        clock=SimpleNamespace(get_fps=lambda: 30.0),
    )

    y = render_sidebar_status_message_metrics_controls(
        gui=gui,
        screen=screen,
        sidebar_x=0,
        y_pos=100,
        player_row=5,
        player_col=6,
        pygame=pygame,
        time_module=_Time,
        math_module=__import__("math"),
        semantic_palette={"STAIR": 9},
    )

    assert y > 100
    assert pygame.draw.rects
    rendered_texts = [obj.text for (obj, _pos) in screen.blits if hasattr(obj, "text")]
    assert "*** VICTORY! ***" in rendered_texts

