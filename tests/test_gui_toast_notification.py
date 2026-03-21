from src.gui.overlay.toast_notification import ToastNotification


class _FakeText:
    def __init__(self, w=40, h=16):
        self._w = w
        self._h = h
        self.alpha = None

    def get_width(self):
        return self._w

    def get_height(self):
        return self._h

    def copy(self):
        return _FakeText(self._w, self._h)

    def set_alpha(self, alpha):
        self.alpha = alpha


class _FakeFont:
    def render(self, _msg, _aa, _color):
        return _FakeText()


class _FakeSurface:
    def __init__(self, *_args, **_kwargs):
        self.blits = []

    def blit(self, obj, pos):
        self.blits.append((obj, pos))


class _FakePygame:
    SRCALPHA = 1

    class font:
        @staticmethod
        def SysFont(_name, _size):
            return _FakeFont()

    @staticmethod
    def Surface(_size, _flags=None):
        return _FakeSurface()

    @staticmethod
    def Rect(x, y, w, h):
        return (x, y, w, h)

    class draw:
        @staticmethod
        def rect(_surface, _color, _rect, _width=0, border_radius=0):
            _ = border_radius
            return None


class _FakeTime:
    def __init__(self, values):
        self._values = list(values)

    def time(self):
        return self._values.pop(0) if self._values else 0.0


def test_toast_alpha_fades_in_and_out():
    tm = _FakeTime([0.0, 0.1, 2.7])
    toast = ToastNotification("ok", duration=3.0, time_module=tm)

    fade_in = toast.get_alpha()
    fade_out = toast.get_alpha()

    assert 0 <= fade_in < 240
    assert 0 <= fade_out < 240


def test_toast_render_blits_to_surface():
    tm = _FakeTime([1.0, 1.2])
    toast = ToastNotification("hello", duration=3.0, time_module=tm)
    target = _FakeSurface()

    toast.render(target, center_x=100, y=20, pygame_module=_FakePygame)

    assert len(target.blits) == 1


def test_toast_expiration_check():
    tm = _FakeTime([5.0, 9.1])
    toast = ToastNotification("expire", duration=4.0, time_module=tm)

    assert toast.is_expired() is True

