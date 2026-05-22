"""
Comprehensive demo dungeon maps for GUI showcase.

These maps demonstrate ALL entity types, door mechanics, and puzzle
elements available in the ZAVE system, providing rich test beds for
every solver algorithm (A*, BFS, Dijkstra, Greedy, D* Lite, DFS/IDDFS,
Bidirectional A*, and all P-CBS variants).
"""

import numpy as np
from src.core.definitions import SEMANTIC_PALETTE


def create_showcase_map() -> np.ndarray:
    """Create a comprehensive 4-room dungeon showcasing ALL entity types.

    Layout (32 rows x 22 cols):
      ┌───────────┬───────────┐
      │  Room A    │  Room B   │  (rows  0-15)
      │  (Start)   │  (Keys)   │
      ├───────────┼───────────┤
      │  Room C    │  Room D   │  (rows 16-31)
      │  (Puzzle)  │  (Boss)   │
      └───────────┴───────────┘

    Entity showcase:
      FLOOR (1), WALL (2), BLOCK (3),
      DOOR_OPEN (10), DOOR_LOCKED (11), DOOR_BOMB (12),
      DOOR_PUZZLE (13), DOOR_BOSS (14), DOOR_SOFT (15),
      ENEMY (20), START (21), TRIFORCE (22), BOSS (23),
      KEY_SMALL (30), KEY_BOSS (31), KEY_ITEM (32), ITEM_MINOR (33),
      ELEMENT (40), ELEMENT_FLOOR (41), STAIR (42), PUZZLE (43)

    Solution path:
      Start(A) -> KEY_SMALL(A) -> DOOR_LOCKED -> Room B
      -> KEY_BOSS(B) -> DOOR_OPEN south -> Room D
      -> defeat BOSS(D) -> DOOR_BOSS -> TRIFORCE
    """
    P = SEMANTIC_PALETTE
    # Short aliases
    V  = P['VOID']
    F  = P['FLOOR']
    W  = P['WALL']
    B  = P['BLOCK']
    DO = P['DOOR_OPEN']
    DL = P['DOOR_LOCKED']
    DB = P['DOOR_BOMB']
    DP = P['DOOR_PUZZLE']
    DK = P['DOOR_BOSS']
    DS = P['DOOR_SOFT']
    EN = P['ENEMY']
    ST = P['START']
    TR = P['TRIFORCE']
    BO = P['BOSS']
    KS = P['KEY_SMALL']
    KB = P['KEY_BOSS']
    KI = P['KEY_ITEM']
    IM = P['ITEM_MINOR']
    EL = P['ELEMENT']
    EF = P['ELEMENT_FLOOR']
    SR = P['STAIR']
    PZ = P['PUZZLE']

    grid = np.array([
        # ===== Row 0-15: Room A (top-left, 11 cols) | Room B (top-right, 11 cols) =====
        [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
        [W, ST, F, F, F, F, F, F, F, F, W, W, F, F, F, F, F, F, F, F, F, W],
        [W, F, F, EN, F, F, F, F, F, F, W, W, F, F, F, F, F, EN, F, F, F, W],
        [W, F, F, F, F, B, F, F, F, F, W, W, F, F, F, F, F, F, F, F, F, W],
        [W, F, F, F, F, F, F, KS, F, F, W, W, F, F, EL, EL, EL, F, F, F, F, W],
        [W, F, F, F, F, F, F, F, F, F, W, W, F, F, EL, EF, EL, F, F, F, F, W],
        [W, F, F, F, F, F, F, F, F, F, W, W, F, F, EL, EL, EL, F, F, KB, F, W],
        [W, F, F, F, F, F, F, F, F, F, DL, DL, F, F, F, F, F, F, F, F, F, W],
        [W, F, F, F, IM, F, F, F, F, F, W, W, F, F, F, F, F, F, F, F, F, W],
        [W, F, F, F, F, F, F, F, SR, F, W, W, F, F, F, B, F, F, F, F, F, W],
        [W, F, F, F, F, F, PZ, F, F, F, W, W, F, F, F, F, F, F, F, IM, F, W],
        [W, F, F, F, F, F, F, F, F, F, W, W, F, F, F, F, F, F, F, F, F, W],
        [W, F, F, EN, F, F, F, F, F, F, W, W, F, F, EN, F, F, F, F, F, F, W],
        [W, F, F, F, F, KI, F, F, F, F, W, W, F, F, F, F, F, F, F, F, F, W],
        [W, F, F, F, F, F, F, F, F, F, W, W, F, F, F, F, F, F, F, F, F, W],
        [W, W, W, W, W, DB, W, W, W, W, W, W, W, W, W, W, DO, W, W, W, W, W],
        # ===== Row 16-31: Room C (bottom-left) | Room D (bottom-right) =====
        [W, W, W, W, W, DB, W, W, W, W, W, W, W, W, W, W, DO, W, W, W, W, W],
        [W, F, F, F, F, F, F, F, F, F, W, W, F, F, F, F, F, F, F, F, F, W],
        [W, F, F, PZ, F, F, F, F, F, F, W, W, F, F, F, F, F, F, F, F, F, W],
        [W, F, EL, EL, EL, F, F, F, F, F, W, W, F, F, F, EN, F, F, EN, F, F, W],
        [W, F, EL, EF, EL, F, F, F, F, F, W, W, F, F, F, F, F, F, F, F, F, W],
        [W, F, EL, EL, EL, F, F, F, F, F, DP, DP, F, F, F, F, F, F, F, F, F, W],
        [W, F, F, F, F, F, F, KS, F, F, W, W, F, F, F, F, F, F, F, F, F, W],
        [W, F, F, F, F, F, F, F, F, F, W, W, F, F, F, F, F, F, F, F, F, W],
        [W, F, EN, F, EN, F, F, F, F, F, W, W, F, F, F, W, W, DK, W, W, F, W],
        [W, F, F, F, F, F, F, F, F, F, W, W, F, F, F, W, F, F, F, W, F, W],
        [W, F, F, F, F, F, F, F, SR, F, W, W, F, F, F, W, F, TR, F, W, F, W],
        [W, F, F, F, F, F, F, F, F, F, W, W, F, F, F, W, F, F, F, W, F, W],
        [W, F, F, F, F, F, F, F, F, F, W, W, F, F, F, W, W, W, W, W, F, W],
        [W, F, F, F, IM, F, F, F, F, F, W, W, F, F, F, F, BO, F, F, F, F, W],
        [W, F, F, F, F, F, F, F, F, F, W, W, F, F, F, F, F, F, F, F, F, W],
        [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
    ], dtype=np.int64)

    return grid


def create_enemy_gauntlet_map() -> np.ndarray:
    """A single large room with many enemies, demonstrating combat pathfinding.

    Layout (16 rows x 22 cols): wide room with enemies forming a gauntlet.
    The solver must navigate through or around enemies to reach the goal.
    Shows how different algorithms handle enemy-dense environments.
    """
    P = SEMANTIC_PALETTE
    F  = P['FLOOR']
    W  = P['WALL']
    EN = P['ENEMY']
    ST = P['START']
    TR = P['TRIFORCE']
    BO = P['BOSS']
    IM = P['ITEM_MINOR']
    KS = P['KEY_SMALL']
    B  = P['BLOCK']

    grid = np.array([
        [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
        [W, ST, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, W],
        [W, F, F, F, F, EN, F, F, F, F, F, F, F, F, F, F, EN, F, F, F, F, W],
        [W, F, F, EN, F, F, F, B, F, F, EN, F, F, B, F, F, F, EN, F, F, F, W],
        [W, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, W],
        [W, F, EN, F, F, F, EN, F, F, F, F, F, F, F, EN, F, F, F, EN, F, F, W],
        [W, F, F, F, F, F, F, F, KS, F, F, F, F, KS, F, F, F, F, F, F, F, W],
        [W, F, F, F, EN, F, F, F, F, F, EN, EN, F, F, F, F, F, EN, F, F, F, W],
        [W, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, W],
        [W, F, EN, F, F, F, EN, F, F, F, IM, IM, F, F, F, EN, F, F, EN, F, F, W],
        [W, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, W],
        [W, F, F, EN, F, F, F, EN, F, F, F, F, F, F, EN, F, F, EN, F, F, F, W],
        [W, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, W],
        [W, F, F, F, F, F, F, F, F, EN, F, F, EN, F, F, F, F, F, BO, F, F, W],
        [W, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, TR, W],
        [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
    ], dtype=np.int64)

    return grid


def create_water_maze_map() -> np.ndarray:
    """Water/element puzzle map requiring the KEY_ITEM (ladder) to cross.

    Layout (16 rows x 22 cols): The player must find the ladder item
    to cross water barriers that block the path to the goal. Demonstrates
    ELEMENT, ELEMENT_FLOOR, and KEY_ITEM mechanics.
    """
    P = SEMANTIC_PALETTE
    F  = P['FLOOR']
    W  = P['WALL']
    EL = P['ELEMENT']
    EF = P['ELEMENT_FLOOR']
    ST = P['START']
    TR = P['TRIFORCE']
    KI = P['KEY_ITEM']
    KS = P['KEY_SMALL']
    DL = P['DOOR_LOCKED']
    IM = P['ITEM_MINOR']
    EN = P['ENEMY']

    grid = np.array([
        [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
        [W, ST, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, W],
        [W, F, F, F, F, EN, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, W],
        [W, F, F, F, F, F, F, F, KS, F, F, W, W, DL, W, W, F, F, F, F, F, W],
        [W, F, F, F, F, F, F, F, F, F, F, W, F, F, F, W, F, F, F, F, F, W],
        [W, EL, EL, EL, EL, EL, EL, EL, EL, F, F, W, F, KI, F, W, F, F, F, F, F, W],
        [W, EL, EF, EF, EF, EL, EL, EF, EL, F, F, W, F, F, F, W, F, F, F, F, F, W],
        [W, EL, EL, EL, EL, EL, EL, EL, EL, F, F, W, W, W, W, W, F, F, F, F, F, W],
        [W, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, W],
        [W, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, W],
        [W, EL, EL, EL, EL, EL, EL, EL, EL, EL, EL, EL, EL, EL, EL, EL, EL, EL, EL, EL, EL, W],
        [W, EL, EF, EL, EL, EF, EL, EL, EF, EL, EL, EF, EL, EL, EF, EL, EL, EF, EL, EL, EL, W],
        [W, EL, EL, EL, EL, EL, EL, EL, EL, EL, EL, EL, EL, EL, EL, EL, EL, EL, EL, EL, EL, W],
        [W, F, F, F, F, F, IM, F, F, F, F, F, F, F, F, F, IM, F, F, F, F, W],
        [W, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, TR, W],
        [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
    ], dtype=np.int64)

    return grid


def create_block_puzzle_map() -> np.ndarray:
    """Block pushing puzzle demonstrating BLOCK mechanics.

    Layout (16 rows x 16 cols): Multiple pushable blocks must be moved
    to clear paths and reach keys/doors. Shows how solvers handle
    state-space expansion with block positions.
    """
    P = SEMANTIC_PALETTE
    F  = P['FLOOR']
    W  = P['WALL']
    B  = P['BLOCK']
    ST = P['START']
    TR = P['TRIFORCE']
    KS = P['KEY_SMALL']
    DL = P['DOOR_LOCKED']
    IM = P['ITEM_MINOR']

    grid = np.array([
        [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
        [W, ST, F, F, F, W, F, F, F, F, F, W, F, F, F, W],
        [W, F, F, B, F, W, F, B, F, F, F, W, F, F, F, W],
        [W, F, F, F, F, W, F, F, F, B, F, W, F, F, F, W],
        [W, F, B, F, F, F, F, F, F, F, F, F, F, B, F, W],
        [W, F, F, F, KS, W, F, F, F, F, F, W, F, F, F, W],
        [W, W, W, DL, W, W, F, F, B, F, F, W, F, F, F, W],
        [W, F, F, F, F, F, F, F, F, F, F, W, F, F, F, W],
        [W, F, F, F, F, F, F, B, F, F, F, F, F, F, F, W],
        [W, F, B, F, F, F, F, F, F, F, F, F, F, B, F, W],
        [W, F, F, F, F, W, F, F, F, F, F, W, F, F, F, W],
        [W, F, F, F, F, W, F, F, B, F, F, W, F, F, F, W],
        [W, F, F, F, F, W, F, F, F, F, F, W, F, F, F, W],
        [W, F, F, F, IM, W, F, F, F, IM, F, W, F, F, F, W],
        [W, F, F, F, F, W, F, F, F, F, F, W, F, F, TR, W],
        [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
    ], dtype=np.int64)

    return grid


def create_all_doors_map() -> np.ndarray:
    """Map with EVERY door type to demonstrate lock/key mechanics.

    Layout (16 rows x 22 cols): Corridor-style dungeon where each
    section is gated by a different door type. Excellent for showing
    how inventory-aware solvers (State-Space A*, P-CBS) differ from
    basic pathfinders (BFS, Dijkstra).
    """
    P = SEMANTIC_PALETTE
    F  = P['FLOOR']
    W  = P['WALL']
    DO = P['DOOR_OPEN']
    DL = P['DOOR_LOCKED']
    DB = P['DOOR_BOMB']
    DP = P['DOOR_PUZZLE']
    DK = P['DOOR_BOSS']
    DS = P['DOOR_SOFT']
    ST = P['START']
    TR = P['TRIFORCE']
    KS = P['KEY_SMALL']
    KB = P['KEY_BOSS']
    IM = P['ITEM_MINOR']
    EN = P['ENEMY']
    PZ = P['PUZZLE']
    BO = P['BOSS']

    grid = np.array([
        [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
        [W, ST, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, W],
        [W, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, W],
        [W, F, F, KS, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, W],
        [W, W, W, W, DL, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
        [W, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, W],
        [W, F, F, F, F, F, EN, F, F, F, F, F, F, F, PZ, F, F, F, F, F, F, W],
        [W, W, W, W, W, W, W, W, DB, W, W, W, W, W, W, W, DP, W, W, W, W, W],
        [W, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, W],
        [W, F, F, F, F, F, IM, F, F, F, F, F, F, F, F, F, F, KB, F, F, F, W],
        [W, W, W, W, W, W, W, W, W, W, DS, W, W, W, W, W, W, W, W, W, W, W],
        [W, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, W],
        [W, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, W],
        [W, W, W, W, W, W, W, W, W, W, W, W, W, W, DK, W, W, W, W, W, W, W],
        [W, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, BO, TR, W],
        [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
    ], dtype=np.int64)

    return grid


def get_all_demo_maps():
    """Return all demo maps with names, ready for GUI consumption.

    Returns:
        tuple: (maps_list, names_list) where each map is a numpy array
               and each name is a display string.
    """
    maps = [
        create_showcase_map(),
        create_all_doors_map(),
        create_enemy_gauntlet_map(),
        create_water_maze_map(),
        create_block_puzzle_map(),
    ]
    names = [
        "Demo: Full Showcase (4-Room Dungeon)",
        "Demo: All Door Types",
        "Demo: Enemy Gauntlet",
        "Demo: Water Maze Puzzle",
        "Demo: Block Pushing Puzzle",
    ]
    return maps, names
