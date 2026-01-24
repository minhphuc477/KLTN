import threading
import time
import numpy as np
from simulation.validator import ZeldaLogicEnv, SEMANTIC_PALETTE
from gui_runner import ZeldaGUI


class DummyHUD:
    def __init__(self):
        self.last = None
        self.keys_collected = 0

    def update_game_state(self, **kwargs):
        self.last = kwargs


def make_simple_env():
    # Grid layout (1x4): S K . G
    S = SEMANTIC_PALETTE['START']
    K = SEMANTIC_PALETTE['KEY_SMALL']
    G = SEMANTIC_PALETTE['TRIFORCE']
    F = SEMANTIC_PALETTE['FLOOR']

    grid = np.array([[S, K, F, G]], dtype=np.int64)
    env = ZeldaLogicEnv(grid)
    env.start_pos = (0, 0)
    env.state.position = env.start_pos
    env.goal_pos = (0, 3)
    return env


def test_auto_solve_picks_up_key_and_updates_hud():
    env = make_simple_env()

    runner = type('R', (), {})()
    runner.env = env
    runner.item_markers = {}
    runner.item_type_map = {}
    runner.collected_items = []
    runner.item_pickup_times = {}
    runner.keys_collected = 0
    runner.bombs_collected = 0
    runner.boss_keys_collected = 0
    runner.keys_used = 0
    runner.bombs_used = 0
    runner.boss_keys_used = 0
    runner.used_items = []
    runner.usage_effects = []

    # Attach required methods
    runner._sync_inventory_counters = ZeldaGUI._sync_inventory_counters.__get__(runner)
    runner._update_inventory_and_hud = ZeldaGUI._update_inventory_and_hud.__get__(runner)
    runner._apply_pickup_at = ZeldaGUI._apply_pickup_at.__get__(runner)
    runner._track_item_collection = ZeldaGUI._track_item_collection.__get__(runner)
    runner._track_item_usage = ZeldaGUI._track_item_usage.__get__(runner)
    runner._show_toast = lambda *a, **k: None
    runner.effects = None
    runner.total_keys = 1

    runner.modern_hud = DummyHUD()
    # Minimal stubs used by _auto_step
    runner.feature_flags = {}
    runner.dstar_active = False
    runner.dstar_solver = None
    runner._show_error = lambda *a, **k: None
    runner._set_message = lambda msg: setattr(runner, 'message', msg)
    runner.renderer = None
    runner.effects = None

    # Setup auto-solve state
    runner.auto_mode = True
    runner.auto_path = [(0,0),(0,1),(0,2),(0,3)]
    runner.auto_step_idx = 0
    runner.step_count = 0

    # Run auto steps until path complete
    while getattr(runner, 'auto_mode', True) and runner.auto_step_idx < len(runner.auto_path) - 1:
        runner._auto_step = ZeldaGUI._auto_step.__get__(runner)
        runner._auto_step()
        # HUD should reflect env.keys immediately after step
        if runner.modern_hud.last is None:
            continue
        assert runner.modern_hud.last['keys'] == getattr(runner.env.state, 'keys', 0)
        # If key picked up, counts should match
        if getattr(runner.env.state, 'keys', 0) > 0:
            assert runner.keys_collected >= 1
            assert runner.modern_hud.last['keys'] == runner.env.state.keys
            break

    # Ensure we did pick up the key
    assert getattr(runner.env.state, 'keys', 0) == 1
    assert runner.keys_collected == 1


def test_auto_solve_handles_deferred_worker_pickup_midrun():
    # Simulate scenario where background thread picks up an item while auto-solve is running
    env = make_simple_env()

    runner = type('R', (), {})()
    runner.env = env
    runner.item_markers = {}
    runner.item_type_map = {}
    runner.collected_items = []
    runner.item_pickup_times = {}
    runner.keys_collected = 0
    runner.bombs_collected = 0
    runner.boss_keys_collected = 0
    runner.keys_used = 0
    runner.bombs_used = 0
    runner.boss_keys_used = 0
    runner.used_items = []
    runner.usage_effects = []

    runner._sync_inventory_counters = ZeldaGUI._sync_inventory_counters.__get__(runner)
    runner._update_inventory_and_hud = ZeldaGUI._update_inventory_and_hud.__get__(runner)
    runner._apply_pickup_at = ZeldaGUI._apply_pickup_at.__get__(runner)
    runner._track_item_collection = ZeldaGUI._track_item_collection.__get__(runner)
    runner._track_item_usage = ZeldaGUI._track_item_usage.__get__(runner)
    runner._show_toast = lambda *a, **k: None
    runner.effects = None
    runner.total_keys = 1

    runner.modern_hud = DummyHUD()
    # Minimal stubs used by _auto_step
    runner.feature_flags = {}
    runner.dstar_active = False
    runner.dstar_solver = None
    runner._show_error = lambda *a, **k: None
    runner._set_message = lambda msg: setattr(runner, 'message', msg)
    runner.renderer = None
    runner.effects = None

    runner.auto_mode = True
    # Path is long enough that worker thread can fire while auto-solve is in progress
    runner.auto_path = [(0,0),(0,1),(0,2),(0,3)]
    runner.auto_step_idx = 0
    runner.step_count = 0

    # Worker will wait a tiny bit then pick up the key directly (simulate external thread)
    def worker():
        time.sleep(0.02)
        # Direct pickup at (0,1) from worker; may have been picked by auto-steppers already
        try:
            picked = runner._apply_pickup_at((0,1))
        except Exception:
            picked = False
        # Not asserting here to avoid thread exceptions; just request HUD refresh if pickup occurred
        if picked:
            try:
                runner._update_inventory_and_hud()
            except Exception:
                pass

    t = threading.Thread(target=worker, name='worker-thread')
    t.start()

    # Run a few auto steps; the auto-step should process the deferred flag when it sees it
    runner._auto_step = ZeldaGUI._auto_step.__get__(runner)
    for _ in range(5):
        runner._auto_step()
        # If deferred refresh was processed, flag should be cleared and HUD updated
        if hasattr(runner, 'inventory_needs_refresh') and not runner.inventory_needs_refresh:
            break
        time.sleep(0.01)

    t.join(timeout=1.0)
    assert not t.is_alive()

    # After processing, HUD should reflect the pickup
    assert getattr(runner, 'inventory_needs_refresh', False) is False
    assert runner.modern_hud.last is not None
    assert runner.modern_hud.last['keys'] == getattr(runner.env.state, 'keys', 0)
    assert runner.keys_collected == 1
