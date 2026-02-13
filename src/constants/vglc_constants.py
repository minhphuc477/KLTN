"""
VGLC Constants - Compliance Module
====================================

This module provides VGLC (Video Game Level Corpus) dataset-compliant
constants for Zelda dungeon generation. ALL dimensions and grammar rules
are derived from ground-truth analysis of the VGLC dataset.

**CRITICAL**: This file documents the canonical VGLC specification.
Do NOT modify values without verifying against actual VGLC dataset files.

Sources:
- LoZ_1.dot, LoZ_2.dot (DOT graph files)
- tloz1_1.txt, tloz1_2.txt (grid text files)
- src/data/zelda_core.py:GridBasedRoomExtractor

"""

from typing import Dict, Set, Tuple
from enum import IntEnum

# ==========================================
# ROOM DIMENSIONS (CRITICAL: NON-SQUARE!)
# ==========================================

# Standard VGLC Zelda room dimensions
# These are GROUND TRUTH from the dataset
ROOM_WIDTH_TILES: int = 11   # Horizontal (columns)
ROOM_HEIGHT_TILES: int = 16  # Vertical (rows)

# Tile dimensions in pixels (VGLC standard)
TILE_SIZE_PX: int = 16       # Each tile is 16×16 pixels

# Room dimensions in pixels
ROOM_WIDTH_PX: int = ROOM_WIDTH_TILES * TILE_SIZE_PX    # 176 pixels
ROOM_HEIGHT_PX: int = ROOM_HEIGHT_TILES * TILE_SIZE_PX  # 256 pixels

# Aspect ratio (important for layout algorithms)
ROOM_ASPECT_RATIO: float = ROOM_WIDTH_TILES / ROOM_HEIGHT_TILES  # 0.6875 (11:16)

# Grid shape for numpy arrays (row-major: height first)
# Use this for creating room arrays: np.zeros(ROOM_SHAPE)
ROOM_SHAPE: Tuple[int, int] = (ROOM_HEIGHT_TILES, ROOM_WIDTH_TILES)  # (16, 11)
ROOM_SHAPE_PX: Tuple[int, int] = (ROOM_HEIGHT_PX, ROOM_WIDTH_PX)     # (256, 176)

# Interior dimensions (without walls)
ROOM_INTERIOR_HEIGHT: int = ROOM_HEIGHT_TILES - 2  # 14 (excluding top/bottom walls)
ROOM_INTERIOR_WIDTH: int = ROOM_WIDTH_TILES - 2    # 9 (excluding left/right walls)
ROOM_INTERIOR_SHAPE: Tuple[int, int] = (ROOM_INTERIOR_HEIGHT, ROOM_INTERIOR_WIDTH)

# ==========================================
# VGLC NODE TYPES (DOT Format)
# ==========================================

# Node label grammar from VGLC .dot files
# Labels can be COMPOSITE (e.g., "e,k,p" = enemy + key + puzzle)

NODE_TYPE_MAP: Dict[str, str] = {
    's': 'start_pointer',     # VIRTUAL - pointer to physical start (DO NOT PLACE ON GRID)
    'S': 'start',              # Physical start room (explicit start marker)
    't': 'triforce',          # Goal node (must be leaf in graph)
    'b': 'boss',              # Boss encounter room
    'e': 'enemy',             # Enemy/combat room
    'k': 'key',               # Small key pickup
    'I': 'macro_item',        # Key item (bow, raft, ladder, etc.)
    'i': 'minor_item',        # Consumable/compass/map
    'p': 'puzzle',            # Puzzle room
}

# Virtual nodes that should NOT be placed on the physical grid
# These are meta-nodes used for graph structure only
VIRTUAL_NODE_TYPES: Set[str] = {'s'}  # Start pointer

# Physical node types that can be placed
PHYSICAL_NODE_TYPES: Set[str] = {'S', 't', 'b', 'e', 'k', 'I', 'i', 'p'}

# Leaf node types (should have degree 1 in graph)
LEAF_NODE_TYPES: Set[str] = {'t'}  # Triforce must be leaf

# ==========================================
# VGLC EDGE TYPES (DOT Format)
# ==========================================

# Edge label grammar from VGLC .dot files
# Edge labels define the CONNECTION TYPE between rooms

EDGE_TYPE_MAP: Dict[str, str] = {
    '': 'open',               # Open passage (no lock, visible)
    'k': 'key_locked',        # Small key required (consumes key)
    'b': 'bombable',          # Bombable wall (secret passage, looks like wall)
    'l': 'soft_lock',         # Soft-locked (shutters, usually one-way after event)
    's': 'stairs_warp',       # Stairs/warp (NOT physically adjacent in grid)
}

# Edge types that consume items
RESOURCE_EDGE_TYPES: Set[str] = {'key_locked', 'bombable'}

# Edge types that are one-way
ONEWAY_EDGE_TYPES: Set[str] = {'soft_lock'}

# Edge types that are non-adjacent (teleport)
WARP_EDGE_TYPES: Set[str] = {'stairs_warp'}

# ==========================================
# TOPOLOGY CONSTRAINTS (Ground Truth)
# ==========================================

# Boss-Goal Subgraph Pattern (strict VGLC pattern)
# Canonical structure: [Pre-Boss] --[k/b]--> [Boss:b] --[l/open]--> [Goal:t]

# Minimum path length from start to goal
MIN_PATH_LENGTH_START_TO_GOAL: int = 3  # At least 3 rooms (start -> middle -> goal)

# Goal node constraints
GOAL_NODE_MAX_DEGREE: int = 1           # Goal MUST be leaf (degree 1)
GOAL_CONNECTS_TO_BOSS: bool = True      # Goal MUST connect to Boss
GOAL_NO_ENEMIES: bool = True            # Goal room has no enemies (safe zone)

# Boss node constraints
BOSS_REQUIRED_FOR_GOAL: bool = True     # Boss must exist if Goal exists
BOSS_GUARDS_GOAL: bool = True           # Boss is always before Goal

# Lock-key constraints
KEY_BEFORE_LOCK: bool = True            # Keys must be reachable before locks
MAX_KEYS_PER_DUNGEON: int = 9           # VGLC dungeons have ≤9 small keys

# ==========================================
# LAYOUT CONSTRAINTS
# ==========================================

# Grid layout boundaries (max dungeon size in rooms)
MAX_DUNGEON_ROWS: int = 8               # VGLC dungeons are typically 8×8 or smaller
MAX_DUNGEON_COLS: int = 8

# Minimum rooms for valid dungeon
MIN_ROOMS_DUNGEON: int = 3              # Start, middle, goal

# Room adjacency rules
ALLOW_DIAGONAL_ADJACENCY: bool = False  # Only N/S/E/W adjacency (not diagonal)
REQUIRE_DOOR_ALIGNMENT: bool = True     # Doors must align between adjacent rooms

# ==========================================
# VALIDATION THRESHOLDS
# ==========================================

# Solvability validation
MIN_SOLVABILITY_SCORE: float = 1.0      # Must be fully solvable

# Graph connectivity
REQUIRE_CONNECTED: bool = True          # All rooms must be reachable from start
ALLOW_CYCLES: bool = True               # Zelda has cycles (but not through goal)
ALLOW_BACKTRACKING: bool = True         # Allowed to revisit rooms

# ==========================================
# COMPOSITE NODE LABEL PARSING
# ==========================================

# Delimiter for composite node labels
NODE_LABEL_DELIMITER: str = ','         # e.g., "e,k,p" splits on comma

def parse_composite_node_label(label: str) -> Set[str]:
    """
    Parse a composite node label into a set of node types.
    
    Handles both VGLC codes (e.g., "e", "b") and full type names (e.g., "enemy", "start").
    
    Args:
        label: Raw label from DOT file or type name (e.g., "e,k,p" or "start" or "enemy")
        
    Returns:
        Set of node type names (e.g., {'enemy', 'key', 'puzzle'})
        
    Examples:
        >>> parse_composite_node_label("e,k,p")
        {'enemy', 'key', 'puzzle'}
        
        >>> parse_composite_node_label("t")
        {'triforce'}
        
        >>> parse_composite_node_label("start")  # Full name
        {'start'}
        
        >>> parse_composite_node_label("")
        set()
    """
    if not label:
        return set()
    
    # Split on delimiter and strip whitespace
    label_parts = [code.strip() for code in label.split(NODE_LABEL_DELIMITER)]
    
    result = set()
    for code in label_parts:
        if not code:
            continue
        # Check if it's a VGLC code
        if code in NODE_TYPE_MAP:
            result.add(NODE_TYPE_MAP[code])
        # Check if it's already a full type name
        elif code in NODE_TYPE_MAP.values():
            result.add(code)
        # Otherwise, keep as-is (might be custom type)
        else:
            result.add(code)
    
    return result


def parse_edge_label(label: str) -> str:
    """
    Parse an edge label into a connection type.
    
    Args:
        label: Raw label from DOT file (e.g., "k", "b", "")
        
    Returns:
        Edge type name (e.g., "key_locked", "open")
        
    Examples:
        >>> parse_edge_label("k")
        'key_locked'
        
        >>> parse_edge_label("")
        'open'
    """
    return EDGE_TYPE_MAP.get(label.strip() if label else '', 'open')


# ==========================================
# EXPORTS
# ==========================================

__all__ = [
    # Dimensions
    'ROOM_WIDTH_TILES',
    'ROOM_HEIGHT_TILES',
    'TILE_SIZE_PX',
    'ROOM_WIDTH_PX',
    'ROOM_HEIGHT_PX',
    'ROOM_ASPECT_RATIO',
    'ROOM_SHAPE',
    'ROOM_SHAPE_PX',
    'ROOM_INTERIOR_HEIGHT',
    'ROOM_INTERIOR_WIDTH',
    'ROOM_INTERIOR_SHAPE',
    
    # Node types
    'NODE_TYPE_MAP',
    'VIRTUAL_NODE_TYPES',
    'PHYSICAL_NODE_TYPES',
    'LEAF_NODE_TYPES',
    
    # Edge types
    'EDGE_TYPE_MAP',
    'RESOURCE_EDGE_TYPES',
    'ONEWAY_EDGE_TYPES',
    'WARP_EDGE_TYPES',
    
    # Constraints
    'MIN_PATH_LENGTH_START_TO_GOAL',
    'GOAL_NODE_MAX_DEGREE',
    'GOAL_CONNECTS_TO_BOSS',
    'GOAL_NO_ENEMIES',
    'BOSS_REQUIRED_FOR_GOAL',
    'BOSS_GUARDS_GOAL',
    'KEY_BEFORE_LOCK',
    'MAX_KEYS_PER_DUNGEON',
    
    # Layout
    'MAX_DUNGEON_ROWS',
    'MAX_DUNGEON_COLS',
    'MIN_ROOMS_DUNGEON',
    'ALLOW_DIAGONAL_ADJACENCY',
    'REQUIRE_DOOR_ALIGNMENT',
    
    # Validation
    'MIN_SOLVABILITY_SCORE',
    'REQUIRE_CONNECTED',
    'ALLOW_CYCLES',
    'ALLOW_BACKTRACKING',
    
    # Parsing
    'NODE_LABEL_DELIMITER',
    'parse_composite_node_label',
    'parse_edge_label',
]
