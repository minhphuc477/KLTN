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


def make_small_env():
    # Grid layout (3x4): S K L T
    S = SEMANTIC_PALETTE['START']
    K = SEMANTIC_PALETTE['KEY_SMALL']
    L = SEMANTIC_PALETTE['DOOR_LOCKED']
    T = SEMANTIC_PALETTE['TRIFORCE']
    F = SEMANTIC_PALETTE['FLOOR']

    grid = np.array([
        [F, F, F, F],
        [S, K, L, T],
        [F, F, F, F]
    ], dtype=np.int64)
    env = ZeldaLogicEnv(grid)
    env.start_pos = (1, 0)
    env.state.position = env.start_pos
    env.goal_pos = (1, 3)
    return env


def test_deferred_inventory_refresh_from_worker_thread():
    env = make_small_env()

    # Build runner-like object (lightweight)
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

    # Attach helper methods from ZeldaGUI
    runner._sync_inventory_counters = ZeldaGUI._sync_inventory_counters.__get__(runner)
    runner._update_inventory_and_hud = ZeldaGUI._update_inventory_and_hud.__get__(runner)
    runner._apply_pickup_at = ZeldaGUI._apply_pickup_at.__get__(runner)
    runner._track_item_collection = ZeldaGUI._track_item_collection.__get__(runner)
    runner._track_item_usage = ZeldaGUI._track_item_usage.__get__(runner)
    runner._show_toast = lambda *a, **k: None
    runner.effects = None
    runner.total_keys = 1

    runner.modern_hud = DummyHUD()

    # Background thread will apply pickup and call _update_inventory_and_hud (should defer)
    def worker():
        key_pos = (1, 1)
        picked = runner._apply_pickup_at(key_pos)
        assert picked is True
        # This call is intentionally from a background thread
        runner._update_inventory_and_hud()

    t = threading.Thread(target=worker, name='worker-thread')
    t.start()
    t.join(timeout=2.0)
    assert not t.is_alive()

    # After worker ran, the deferred flag should be set
    assert getattr(runner, 'inventory_needs_refresh', False) is True

    # Now simulate main thread processing of deferred refresh
    runner._update_inventory_and_hud()
    # Flag should be cleared after main thread handles the update
    assert getattr(runner, 'inventory_needs_refresh', False) is False
    # HUD should reflect updated keys value
    assert runner.modern_hud.last is not None
    assert runner.modern_hud.last['keys'] == getattr(runner.env.state, 'keys', 0)
    # And collected counts mirror onto HUD
    if hasattr(runner.modern_hud, 'keys_collected'):
        assert runner.modern_hud.keys_collected == runner.keys_collected
