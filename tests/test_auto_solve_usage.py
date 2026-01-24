import numpy as np
from simulation.validator import ZeldaLogicEnv, SEMANTIC_PALETTE
from gui_runner import ZeldaGUI


class DummyHUDWithInventory:
    def __init__(self):
        self.last = None
        self.keys_collected = 0
        self.keys_used = 0
        class Inv:
            def __init__(self):
                self.keys_collected = 0
                self.keys_used = 0
        self.inventory = Inv()

    def update_game_state(self, **kwargs):
        self.last = kwargs


def make_env_with_door():
    # Grid layout (1x4): S K L G (start, key, locked door, goal)
    S = SEMANTIC_PALETTE['START']
    K = SEMANTIC_PALETTE['KEY_SMALL']
    L = SEMANTIC_PALETTE['DOOR_LOCKED']
    G = SEMANTIC_PALETTE['TRIFORCE']
    F = SEMANTIC_PALETTE['FLOOR']

    grid = np.array([[S, K, L, G]], dtype=np.int64)
    env = ZeldaLogicEnv(grid)
    env.start_pos = (0, 0)
    env.state.position = env.start_pos
    env.goal_pos = (0, 3)
    return env


def test_auto_solve_uses_key_and_updates_hud_usage_counters():
    env = make_env_with_door()

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
    runner.total_keys = 1

    runner.modern_hud = DummyHUDWithInventory()
    runner.feature_flags = {}
    runner.dstar_active = False
    runner.dstar_solver = None
    runner._show_error = lambda *a, **k: None
    runner._set_message = lambda msg: setattr(runner, 'message', msg)
    runner.renderer = None
    runner.effects = None

    runner.auto_mode = True
    runner.auto_path = [(0,0),(0,1),(0,2),(0,3)]
    runner.auto_step_idx = 0
    runner.step_count = 0

    runner._auto_step = ZeldaGUI._auto_step.__get__(runner)

    # Step until through the door (should pick up key at (0,1) and use at (0,2))
    for _ in range(4):
        runner._auto_step()

    assert runner.keys_collected == 1
    assert runner.keys_used == 1
    # HUD inventory should show used count
    assert runner.modern_hud.keys_used == runner.keys_used
    assert runner.modern_hud.inventory.keys_used == runner.keys_used
