"""
DEPRECATED: Backward-compatibility shim for simulation module.

This module re-exports from src.simulation for backward compatibility.
New code should import directly from src.simulation.

Example:
    # OLD (deprecated):
    from simulation.validator import StateSpaceAStar
    
    # NEW (preferred):
    from src.simulation import StateSpaceAStar
"""

import warnings
warnings.warn(
    "Importing from 'simulation' is deprecated. "
    "Use 'from src.simulation import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from src.simulation for backward compatibility
from src.simulation import *
from src.simulation import __all__
