"""
H-MOLQD Evaluation Module
=========================

Block VI: External Validator & MAP-Elites
- Agent simulation for solvability verification
- MAP-Elites quality diversity archive
- Expressive range analysis
"""

from .validator import (
    ExternalValidator,
    AgentSimulator,
    SolvabilityChecker,
    PathVerifier,
)
from .map_elites import (
    MAPElites,
    FeatureExtractor,
    LinearityLeniencyExtractor,
    DensityDifficultyExtractor,
    EliteArchive,
    Elite,
    DiversityMetrics,
    ArchiveStats,
)

__all__ = [
    # Validator
    'ExternalValidator',
    'AgentSimulator',
    'SolvabilityChecker',
    'PathVerifier',
    # MAP-Elites
    'MAPElites',
    'FeatureExtractor',
    'LinearityLeniencyExtractor',
    'DensityDifficultyExtractor',
    'EliteArchive',
    'Elite',
    'DiversityMetrics',
    'ArchiveStats',
]
