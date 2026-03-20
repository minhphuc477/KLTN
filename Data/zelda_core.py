"""
DEPRECATED: Backward-compatibility shim for legacy imports.

This module re-exports `src.data.zelda_core` so there is a single source of
truth for dungeon parsing/solving logic.
"""

import warnings

warnings.warn(
    "Importing from 'Data.zelda_core' is deprecated. "
    "Use 'from src.data.zelda_core import ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the canonical implementation.
from src.data.zelda_core import *  # noqa: F401,F403

