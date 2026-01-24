from gui_runner import ZeldaGUI


def test_next_map_stops_auto_mode():
    runner = type('R', (), {})()
    # minimal attributes required by _next_map
    runner.auto_mode = True
    runner.current_map_idx = 0
    runner.maps = [1, 2]
    # stub methods
    runner._load_current_map = lambda: setattr(runner, 'loaded', True)
    runner._center_view = lambda: setattr(runner, 'centered', True)
    runner.effects = None
    runner.step_count = 5

    # bind the methods
    runner._stop_auto = ZeldaGUI._stop_auto.__get__(runner)
    runner._next_map = ZeldaGUI._next_map.__get__(runner)

    runner._next_map()
    assert runner.auto_mode is False
    assert runner.current_map_idx == 1
    assert getattr(runner, 'loaded', False) is True
    assert getattr(runner, 'centered', False) is True
    assert runner.step_count == 0


def test_prev_map_stops_auto_mode():
    runner = type('R', (), {})()
    runner.auto_mode = True
    runner.current_map_idx = 0
    runner.maps = [1, 2]
    runner._load_current_map = lambda: setattr(runner, 'loaded', True)
    runner._center_view = lambda: setattr(runner, 'centered', True)
    runner.effects = None
    runner.step_count = 5

    runner._stop_auto = ZeldaGUI._stop_auto.__get__(runner)
    runner._prev_map = ZeldaGUI._prev_map.__get__(runner)

    runner._prev_map()
    assert runner.auto_mode is False
    assert runner.current_map_idx == 1  # wrap-around
    assert getattr(runner, 'loaded', False) is True
    assert getattr(runner, 'centered', False) is True
    assert runner.step_count == 0