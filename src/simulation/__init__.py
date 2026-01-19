"""
KLTN Simulation Module
======================
Validation and simulation components for Zelda dungeon analysis.

This module contains:
- validator: External validation suite (ZeldaValidator, ZeldaLogicEnv, etc.)
"""

# Re-export from the parent simulation module for convenience
try:
    from simulation.validator import (
        ZeldaValidator,
        ZeldaLogicEnv,
        SanityChecker,
        MetricsEngine,
        DiversityEvaluator,
        ValidationResult,
        BatchValidationResult,
    )
except ImportError:
    pass  # Allow import even if parent module not available

__all__ = [
    'ZeldaValidator',
    'ZeldaLogicEnv',
    'SanityChecker',
    'MetricsEngine',
    'DiversityEvaluator',
    'ValidationResult',
    'BatchValidationResult',
]
