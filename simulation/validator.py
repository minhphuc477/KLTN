"""
DEPRECATED: Backward-compatibility shim.

This module re-exports from src.simulation.validator for backward compatibility.
New code should import directly from src.simulation.validator.
"""

import warnings
warnings.warn(
    "Importing from 'simulation.validator' is deprecated. "
    "Use 'from src.simulation.validator import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

from src.simulation.validator import *
