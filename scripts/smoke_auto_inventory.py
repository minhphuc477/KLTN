"""Smoke script: run a short auto-solve session and simulate a worker-thread pickup.

This script logs `_update_inventory_and_hud` activity and the HUD state so you can
observe whether inventory updates are applied immediately during auto-solve or
are deferred and processed on the next main-thread update.

Usage:
  python scripts/smoke_auto_inventory.py

"""
import logging
import time
import threading
import numpy as np
import os, sys
# Ensure project root is on sys.path so local packages (simulation, src, etc.) can be imported when running script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simulation.validator import ZeldaLogicEnv, SEMANTIC_PALETTE

# Prefer importing the full GUI if available; otherwise run a headless runner
try:
    from gui_runner import ZeldaGUI
    PYGAME_AVAILABLE = True
except Exception:
    from gui_runner import ZeldaGUI  # try anyway; tests will exercise headless paths
    PYGAME_AVAILABLE = True

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


def run_headless():
    env = make_simple_env()

    # Lightweight runner object that borrows ZeldaGUI behavior
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

    # Attach required methods and minimal stubs
    runner._sync_inventory_counters = ZeldaGUI._sync_inventory_counters.__get__(runner)
    runner._update_inventory_and_hud = ZeldaGUI._update_inventory_and_hud.__get__(runner)
    runner._apply_pickup_at = ZeldaGUI._apply_pickup_at.__get__(runner)
    runner._track_item_collection = ZeldaGUI._track_item_collection.__get__(runner)
    runner._track_item_usage = ZeldaGUI._track_item_usage.__get__(runner)
    runner._show_toast = lambda *a, **k: None
    runner.total_keys = 1
    runner.modern_hud = type('H', (), {'last': None, 'keys_collected': 0, 'update_game_state': lambda self, **k: setattr(self, 'last', k)})()
    runner.feature_flags = {}
    runner.dstar_active = False
    runner.dstar_solver = None
    runner._show_error = lambda *a, **k: None
    runner._set_message = lambda msg: setattr(runner, 'message', msg)
    runner.renderer = None
    runner.effects = None

    # Auto solve prep
    runner.auto_mode = True
    runner.auto_path = [(0,0),(0,1),(0,2),(0,3)]
    runner.auto_step_idx = 0
    runner.step_count = 0

    # Worker thread simulating a concurrent pickup
    def worker():
        time.sleep(0.02)
        logger.info('[worker] attempting pickup at (0,1)')
        picked = runner._apply_pickup_at((0,1))
        logger.info('[worker] pickup result=%s (keys=%s keys_collected=%s)', picked, getattr(runner.env.state, 'keys', None), getattr(runner, 'keys_collected', None))
        runner._update_inventory_and_hud()

    t = threading.Thread(target=worker, name='worker-thread')
    t.start()

    # Run auto steps and log HUD/counters
    for i in range(20):
        logger.debug('[main] auto_step loop iter=%d auto_step_idx=%d inventory_needs_refresh=%s', i, getattr(runner, 'auto_step_idx', None), getattr(runner, 'inventory_needs_refresh', False))
        ZeldaGUI._auto_step.__get__(runner)()
        # Immediately log hud and counters
        logger.debug('[main] after auto_step: env.keys=%s keys_collected=%s hud.last=%r inventory_needs_refresh=%s',
                     getattr(runner.env.state, 'keys', None), getattr(runner, 'keys_collected', None), getattr(runner.modern_hud, 'last', None), getattr(runner, 'inventory_needs_refresh', False))
        if not getattr(runner, 'auto_mode', False):
            break
        time.sleep(0.01)

    t.join(timeout=1.0)
    logger.info('Finished headless run: env.keys=%s keys_collected=%s hud.last=%r', getattr(runner.env.state, 'keys', None), getattr(runner, 'keys_collected', None), getattr(runner.modern_hud, 'last', None))


if __name__ == '__main__':
    print('Starting smoke_auto_inventory (headless)')
    run_headless()
    print('Done')
