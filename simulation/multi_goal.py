"""
DEPRECATED: Backward-compatibility shim.
Use 'from src.simulation.multi_goal import ...' instead.
"""
import warnings
warnings.warn("Importing from 'simulation.multi_goal' is deprecated.", DeprecationWarning, stacklevel=2)
from src.simulation.multi_goal import *
