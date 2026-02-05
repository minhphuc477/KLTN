"""
DEPRECATED: Backward-compatibility shim.
Use 'from src.simulation.parallel_astar import ...' instead.
"""
import warnings
warnings.warn("Importing from 'simulation.parallel_astar' is deprecated.", DeprecationWarning, stacklevel=2)
from src.simulation.parallel_astar import *
