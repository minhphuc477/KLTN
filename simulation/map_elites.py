"""
DEPRECATED: Backward-compatibility shim.
Use 'from src.simulation.map_elites import ...' instead.
"""
import warnings
warnings.warn("Importing from 'simulation.map_elites' is deprecated.", DeprecationWarning, stacklevel=2)
from src.simulation.map_elites import *
