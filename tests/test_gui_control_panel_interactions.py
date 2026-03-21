from src.gui.controls.control_panel_interactions import (
    control_panel_hit_rect,
    handle_outside_control_panel_click,
    refresh_control_panel_layout_if_needed,
    retry_control_panel_click_after_auto_scroll,
    should_swallow_control_panel_click,
    translate_control_panel_click,
)


class DummyRect:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.bottom = y + height

    def collidepoint(self, pos):
        px, py = pos
        return self.x <= px <= (self.x + self.width) and self.y <= py <= (self.y + self.height)


class DummyWidget:
    def __init__(self, rect, full_rect=None, is_open=False):
        self.rect = rect
        self.full_rect = full_rect if full_rect is not None else rect
        self.is_open = is_open


class DummyDropdown(DummyWidget):
    pass


class DummyWidgetManager:
    def __init__(self, widgets, mouse_down_result=True):
        self.widgets = widgets
        self.mouse_down_result = mouse_down_result
        self.last_mouse_down = None

    def handle_mouse_down(self, pos, button):
        self.last_mouse_down = (pos, button)
        return self.mouse_down_result


def test_control_panel_hit_rect_padding():
    rect = DummyRect(10, 20, 100, 50)
    out = control_panel_hit_rect(
        panel_rect=rect,
        debug_control_panel=True,
        debug_panel_click_padding=15,
        rect_factory=DummyRect,
    )
    assert out.x == -5
    assert out.y == 20
    assert out.width == 115
    assert out.height == 50


def test_should_swallow_control_panel_click_when_dragging_inside():
    rect = DummyRect(0, 0, 200, 100)
    swallowed = should_swallow_control_panel_click(
        dragging=True,
        ignore_click_until=0.0,
        panel_hit_rect=rect,
        pos=(10, 10),
    )
    assert swallowed is True


def test_translate_control_panel_click_applies_scroll_below_header():
    panel = DummyRect(100, 50, 220, 500)
    translated = translate_control_panel_click(
        pos=(120, 120),
        panel_hit_rect=panel,
        panel_rect=panel,
        can_scroll=True,
        control_panel_scroll=80,
    )
    assert translated == (120, 200)


def test_handle_outside_control_panel_click_routes_to_open_dropdown():
    panel = DummyRect(100, 50, 220, 500)
    widgets = [DummyDropdown(DummyRect(130, 80, 90, 30), is_open=True)]
    wm = DummyWidgetManager(widgets=widgets, mouse_down_result=True)

    result = handle_outside_control_panel_click(
        panel_hit_rect=panel,
        pos=(10, 10),
        button=1,
        widget_manager=wm,
        dropdown_type=DummyDropdown,
    )

    assert result is True
    assert wm.last_mouse_down == ((10, 10), 1)


def test_refresh_control_panel_layout_if_needed_repositions_and_hits():
    panel = DummyRect(100, 50, 220, 500)
    target_pos = (140, 200)
    widget = DummyWidget(rect=DummyRect(0, 0, 20, 20))
    wm = DummyWidgetManager([widget])

    def reposition_widgets(x, y):
        assert (x, y) == (100, 50)
        widget.rect = DummyRect(130, 180, 40, 40)
        widget.full_rect = widget.rect

    hit = refresh_control_panel_layout_if_needed(
        widget_manager=wm,
        sc_pos=target_pos,
        debug_input_active=False,
        panel_rect=panel,
        reposition_widgets=reposition_widgets,
    )

    assert hit is True


def test_retry_control_panel_click_after_auto_scroll_updates_scroll_and_dispatches():
    panel = DummyRect(100, 50, 220, 500)
    clipped_widget = DummyWidget(rect=DummyRect(110, 560, 40, 30))
    wm = DummyWidgetManager([clipped_widget], mouse_down_result=True)

    handled, new_scroll, ignore_until = retry_control_panel_click_after_auto_scroll(
        pos=(120, 100),
        sc_pos=(120, 100),
        button=1,
        handled=False,
        panel_rect=panel,
        widget_manager=wm,
        can_scroll=True,
        control_panel_scroll=0,
        control_panel_scroll_max=800,
    )

    assert handled is True
    assert new_scroll > 0
    assert ignore_until > 0
    assert wm.last_mouse_down[1] == 1

