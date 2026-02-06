"""
DEPRECATED: Data package backward-compatibility shim.

The Data/ folder now only contains:
- Raw VGLC data files (The Legend of Zelda/)
- Assets (Assets/)

Core logic has moved to src/data/.
"""

import warnings
warnings.warn(
    "Importing from 'Data' is deprecated. "
    "Use 'from src.data import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

from .zelda_core import *
