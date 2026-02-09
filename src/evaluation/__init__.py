"""
H-MOLQD Evaluation Module
=========================

Block VI: External Validator & MAP-Elites
- Agent simulation for solvability verification
- MAP-Elites quality diversity archive
- Expressive range analysis
- CBS-based fitness for QD optimization
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
    CVTEliteArchive,
    CombinedFeatureExtractor,
    CBSFeatureExtractor,
    FullFeatureExtractor,
    create_map_elites,
)
from .cbs_fitness import (
    compute_cbs_fitness,
    cbs_loss_term,
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
    'CVTEliteArchive',
    'CombinedFeatureExtractor',
    'CBSFeatureExtractor',
    'FullFeatureExtractor',
    'create_map_elites',
    # CBS Fitness
    'compute_cbs_fitness',
    'cbs_loss_term',
]
