from src.gui.overlay.toast_messages import render_toasts, set_message, show_toast, update_toasts


class DummyTime:
    @staticmethod
    def time():
        return 123.0


class DummyToast:
    def __init__(self, message, duration, toast_type):
        self.message = message
        self.duration = duration
        self.toast_type = toast_type
        self.expired = False
        self.render_calls = []

    def is_expired(self):
        return self.expired

    def render(self, surface, x, y):
        self.render_calls.append((surface, x, y))


class DummyGui:
    def __init__(self):
        self.message = ""
        self.message_time = 0.0
        self.message_duration = 0.0
        self.toast_notifications = []
        self.screen_w = 800
        self.screen_h = 600


def test_set_message_updates_fields_with_time():
    gui = DummyGui()
    set_message(gui, "hello", 4.5, DummyTime)
    assert gui.message == "hello"
    assert gui.message_time == 123.0
    assert gui.message_duration == 4.5


def test_show_update_render_toasts_flow():
    gui = DummyGui()
    show_toast(gui, "a", 2.0, "info", DummyToast)
    show_toast(gui, "b", 3.0, "warn", DummyToast)

    assert len(gui.toast_notifications) == 2
    gui.toast_notifications[0].expired = True
    update_toasts(gui)
    assert len(gui.toast_notifications) == 1

    surface = object()
    render_toasts(gui, surface)
    assert gui.toast_notifications[0].render_calls[0][0] is surface

