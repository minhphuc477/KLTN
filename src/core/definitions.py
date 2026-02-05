"""
KLTN ZELDA DEFINITIONS
======================
Central constants and type definitions for the entire project.

This file is the SINGLE SOURCE OF TRUTH for:
- Semantic Palette (tile IDs)
- Character mappings
- Room dimensions
- Edge type mappings
- Node content mappings

Import from here instead of duplicating constants across modules.

"""

from typing import Dict, Set
from enum import IntEnum

# ==========================================
# SEMANTIC PALETTE (CRITICAL CONSTANTS)
# ==========================================
# These IDs MUST be consistent across all modules
# Block I produces these numbers; Block VI reads them

class TileID(IntEnum):
    """Semantic tile IDs for dungeon grid representation."""
    # Environment
    VOID = 0            # Empty space (outside map)
    FLOOR = 1           # Walkable floor
    WALL = 2            # Solid wall (impassable)
    BLOCK = 3           # Pushable/decorative block
    
    # Doors (determined by graph edge type)
    DOOR_OPEN = 10      # Open passage (no lock)
    DOOR_LOCKED = 11    # Key-locked door (requires small key)
    DOOR_BOMB = 12      # Bombable wall (requires bomb)
    DOOR_PUZZLE = 13    # Puzzle/switch door
    DOOR_BOSS = 14      # Boss key door
    DOOR_SOFT = 15      # Soft-locked (one-way) door
    
    # Entities
    ENEMY = 20          # Monster/enemy
    START = 21          # Starting position
    TRIFORCE = 22       # Goal/Triforce piece
    BOSS = 23           # Boss enemy
    
    # Items
    KEY_SMALL = 30      # Small key (consumable)
    KEY_BOSS = 31       # Boss key (permanent)
    KEY_ITEM = 32       # Key item like bow/raft
    ITEM_MINOR = 33     # Minor collectible
    
    # Special
    ELEMENT = 40        # Hazard element (lava/water)
    ELEMENT_FLOOR = 41  # Element with floor underneath
    STAIR = 42          # Stairs/ladder/warp
    PUZZLE = 43         # Puzzle element


# Dict version for backward compatibility
SEMANTIC_PALETTE: Dict[str, int] = {
    # Environment
    'VOID': TileID.VOID,
    'FLOOR': TileID.FLOOR,
    'WALL': TileID.WALL,
    'BLOCK': TileID.BLOCK,
    
    # Doors
    'DOOR_OPEN': TileID.DOOR_OPEN,
    'DOOR_LOCKED': TileID.DOOR_LOCKED,
    'DOOR_BOMB': TileID.DOOR_BOMB,
    'DOOR_PUZZLE': TileID.DOOR_PUZZLE,
    'DOOR_BOSS': TileID.DOOR_BOSS,
    'DOOR_SOFT': TileID.DOOR_SOFT,
    
    # Entities
    'ENEMY': TileID.ENEMY,
    'START': TileID.START,
    'TRIFORCE': TileID.TRIFORCE,
    'BOSS': TileID.BOSS,
    
    # Items
    'KEY_SMALL': TileID.KEY_SMALL,
    'KEY': TileID.KEY_SMALL,  # Alias
    'KEY_BOSS': TileID.KEY_BOSS,
    'KEY_ITEM': TileID.KEY_ITEM,
    'ITEM_MINOR': TileID.ITEM_MINOR,
    'ITEM': TileID.ITEM_MINOR,  # Alias
    
    # Special
    'ELEMENT': TileID.ELEMENT,
    'ELEMENT_FLOOR': TileID.ELEMENT_FLOOR,
    'STAIR': TileID.STAIR,
    'PUZZLE': TileID.PUZZLE,
}

# Reverse lookup for debugging
ID_TO_NAME: Dict[int, str] = {v: k for k, v in SEMANTIC_PALETTE.items()}

# ==========================================
# CHARACTER MAPPINGS (VGLC Format)
# ==========================================

CHAR_TO_SEMANTIC: Dict[str, int] = {
    '-': TileID.VOID,
    'F': TileID.FLOOR,
    '.': TileID.FLOOR,
    'W': TileID.WALL,
    'B': TileID.BLOCK,
    'M': TileID.ENEMY,
    'P': TileID.ELEMENT,
    'O': TileID.ELEMENT_FLOOR,  # Element+Floor (walkable)
    'I': TileID.ELEMENT,        # Element+Block (not walkable)
    'S': TileID.STAIR,
    'D': TileID.DOOR_OPEN,      # Default door (type determined by graph)
}

# Characters that represent walkable tiles
WALKABLE_CHARS: Set[str] = {'F', '.', 'O', 'D', 'S'}

# Characters that represent walls/obstacles
WALL_CHARS: Set[str] = {'W', 'B', 'I', 'P'}

# ==========================================
# ROOM DIMENSIONS (VGLC Zelda Standard)
# ==========================================

# Standard VGLC Zelda room dimensions
ROOM_HEIGHT: int = 16  # Rows per room (including walls)
ROOM_WIDTH: int = 11   # Columns per room (including walls)

# Interior dimensions (without walls)
ROOM_INTERIOR_HEIGHT: int = 14
ROOM_INTERIOR_WIDTH: int = 9

# Grid slot dimensions (same as room for VGLC)
SLOT_HEIGHT: int = 16
SLOT_WIDTH: int = 11

# ==========================================
# GRAPH EDGE TYPES (DOT Format)
# ==========================================

EDGE_TYPE_MAP: Dict[str, str] = {
    '': 'open',              # Normal open passage
    'k': 'key_locked',       # Small key required
    'K': 'boss_locked',      # Boss key required
    'b': 'bombable',         # Bomb required
    'l': 'soft_locked',      # One-way (can't return)
    's': 'stair',            # Stair/warp connection
    'S': 'switch',           # Switch/puzzle required
    'S1': 'switch',          # Switch variant
    'I': 'item_locked',      # Key item required
}

# Edge types that consume resources
RESOURCE_EDGES: Set[str] = {'key_locked', 'boss_locked', 'bombable'}

# Edge types that are one-way
ONEWAY_EDGES: Set[str] = {'soft_locked'}

# Edge types that are warps (non-adjacent)
WARP_EDGES: Set[str] = {'stair'}

# ==========================================
# GRAPH NODE CONTENT (DOT Format)
# ==========================================

NODE_CONTENT_MAP: Dict[str, str] = {
    'e': 'enemy',
    's': 'start',
    't': 'triforce',
    'b': 'boss',
    'k': 'key',
    'K': 'boss_key',
    'I': 'key_item',
    'i': 'item',
    'p': 'puzzle',
}

# ==========================================
# DOOR POSITIONS IN ROOM
# ==========================================

# Where doors can appear in a 16x11 room grid (row, col ranges)
DOOR_POSITIONS: Dict[str, Dict[str, int]] = {
    'N': {'row': 0, 'col_start': 4, 'col_end': 7},      # North door
    'S': {'row': 15, 'col_start': 4, 'col_end': 7},     # South door
    'E': {'col': 10, 'row_start': 7, 'row_end': 9},     # East door
    'W': {'col': 0, 'row_start': 7, 'row_end': 9},      # West door
}

# Direction opposites
DIRECTION_OPPOSITE: Dict[str, str] = {
    'N': 'S',
    'S': 'N',
    'E': 'W',
    'W': 'E',
}

# Direction to coordinate offset (row_delta, col_delta)
DIRECTION_OFFSET: Dict[str, tuple] = {
    'N': (-1, 0),
    'S': (1, 0),
    'E': (0, 1),
    'W': (0, -1),
}

# ==========================================
# VALIDATION MODES
# ==========================================

class ValidationMode:
    """Validation modes for dungeon solvability checking."""
    STRICT = 'strict'        # Only normal doors (what's visible in tiles)
    REALISTIC = 'realistic'  # Normal + soft-locked + stairs (no items needed)
    FULL = 'full'            # All edges with full inventory tracking


# ==========================================
# ML FEATURE DIMENSIONS
# ==========================================

# Topological Positional Encoding dimensions
TPE_DIM: int = 8

# Node feature vector dimensions
# [is_start, has_enemy, has_key, has_boss_key, has_triforce]
NODE_FEATURE_DIM: int = 5

# P-Matrix channels
# [small_key_dependency, bomb_dependency, boss_key_dependency]
P_MATRIX_CHANNELS: int = 3


# ==========================================
# EXPORTS
# ==========================================

__all__ = [
    # Enums
    'TileID',
    'ValidationMode',
    
    # Palettes
    'SEMANTIC_PALETTE',
    'ID_TO_NAME',
    'CHAR_TO_SEMANTIC',
    
    # Character sets
    'WALKABLE_CHARS',
    'WALL_CHARS',
    
    # Dimensions
    'ROOM_HEIGHT',
    'ROOM_WIDTH',
    'ROOM_INTERIOR_HEIGHT',
    'ROOM_INTERIOR_WIDTH',
    'SLOT_HEIGHT',
    'SLOT_WIDTH',
    
    # Graph mappings
    'EDGE_TYPE_MAP',
    'NODE_CONTENT_MAP',
    'RESOURCE_EDGES',
    'ONEWAY_EDGES',
    'WARP_EDGES',
    
    # Door positions
    'DOOR_POSITIONS',
    'DIRECTION_OPPOSITE',
    'DIRECTION_OFFSET',
    
    # ML dimensions
    'TPE_DIM',
    'NODE_FEATURE_DIM',
    'P_MATRIX_CHANNELS',
]
