"""
DEPRECATED: Backward-compatibility shim.
Use 'from src.simulation.dstar_lite import ...' instead.
"""
import warnings
warnings.warn("Importing from 'simulation.dstar_lite' is deprecated.", DeprecationWarning, stacklevel=2)
from src.simulation.dstar_lite import *
