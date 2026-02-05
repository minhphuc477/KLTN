"""
DEPRECATED: Backward-compatibility shim.
Use 'from src.simulation.solver_comparison import ...' instead.
"""
import warnings
warnings.warn("Importing from 'simulation.solver_comparison' is deprecated.", DeprecationWarning, stacklevel=2)
from src.simulation.solver_comparison import *
