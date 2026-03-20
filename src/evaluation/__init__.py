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
from .benchmark_suite import (
    GraphDescriptor,
    BenchmarkSummary,
    extract_graph_descriptor,
    load_vglc_reference_graphs,
    load_vglc_reference_rooms,
    audit_block0_dataset,
    generate_block_i_graphs,
    run_wfc_robustness_probe,
    calibrate_rule_weights_to_vglc,
    run_block_i_benchmark,
    run_block_i_benchmark_from_scratch,
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
    # Benchmark suite
    'GraphDescriptor',
    'BenchmarkSummary',
    'extract_graph_descriptor',
    'load_vglc_reference_graphs',
    'load_vglc_reference_rooms',
    'audit_block0_dataset',
    'generate_block_i_graphs',
    'run_wfc_robustness_probe',
    'calibrate_rule_weights_to_vglc',
    'run_block_i_benchmark',
    'run_block_i_benchmark_from_scratch',
]
