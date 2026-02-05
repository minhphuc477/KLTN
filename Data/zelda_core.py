"""
DEPRECATED: Backward-compatibility shim.

This module re-exports from src.data.zelda_core for backward compatibility.
New code should import directly from src.data.zelda_core.

Example:
    # OLD (deprecated):
    from Data.zelda_core import ZeldaDungeonAdapter
    
    # NEW (preferred):
    from src.data.zelda_core import ZeldaDungeonAdapter
"""

import warnings
warnings.warn(
    "Importing from 'Data.zelda_core' is deprecated. "
    "Use 'from src.data.zelda_core import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

from src.data.zelda_core import *

# Also re-export from src.core.definitions for common constants
from src.core.definitions import (
    SEMANTIC_PALETTE,
    CHAR_TO_SEMANTIC,
    ID_TO_NAME,
    ROOM_HEIGHT,
    ROOM_WIDTH,
    EDGE_TYPE_MAP,
    NODE_CONTENT_MAP,
)
