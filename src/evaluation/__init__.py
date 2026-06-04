"""
H-MOLQD Evaluation Module
=========================

Block VII: External Validator & MAP-Elites
- Agent simulation for solvability verification
- MAP-Elites quality diversity archive
- Expressive range analysis
- CBS-based fitness for QD optimization
"""

from importlib import import_module

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
from .pcbs_validation import (
    PreparedValidationDungeon,
    prepare_dungeon_grid_for_validation,
    evaluate_astar_vs_pcbs,
    build_ieee_markdown_table,
)
from .tile_distribution import (
    TilePatternDistributionResult,
    compare_tile_pattern_distributions,
    iter_tile_patterns,
    tile_pattern_counts,
)
from .perturb_and_map import (
    PerturbAndMAPReachabilityResult,
    perturb_and_map_reachability,
)
from .pcbs_rl_ablation import (
    BeliefStateQAgent,
    RLAblationMetrics,
    compute_cross_persona_agreement,
    compute_persona_divergence_from_paths,
    run_pcbs_rl_alignment_ablation,
    train_belief_state_q_agent,
)

_BENCHMARK_EXPORTS = [
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
    'PCGBenchmarkZeldaVariant',
    'PCGBenchmarkZeldaMapping',
    'PCG_BENCHMARK_ZELDA_VARIANTS',
    'select_pcg_benchmark_zelda_problem',
    'map_graph_to_pcg_benchmark_zelda',
    'import_pcg_benchmark',
    'evaluate_graphs_with_pcg_benchmark_zelda',
]

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
    # Validation handoff / P-CBS evaluation
    'PreparedValidationDungeon',
    'prepare_dungeon_grid_for_validation',
    'evaluate_astar_vs_pcbs',
    'build_ieee_markdown_table',
    # Discrete tile distribution realism metrics
    'TilePatternDistributionResult',
    'compare_tile_pattern_distributions',
    'iter_tile_patterns',
    'tile_pattern_counts',
    # Perturb-and-MAP hard-solver ablation
    'PerturbAndMAPReachabilityResult',
    'perturb_and_map_reachability',
    # P-CBS release-paper RL ablation
    'BeliefStateQAgent',
    'RLAblationMetrics',
    'compute_cross_persona_agreement',
    'compute_persona_divergence_from_paths',
    'run_pcbs_rl_alignment_ablation',
    'train_belief_state_q_agent',
    # Benchmark suite (lazy-loaded)
    *_BENCHMARK_EXPORTS,
]


def __getattr__(name: str):
    """Lazily expose benchmark_suite symbols to avoid import cycles."""
    if name in _BENCHMARK_EXPORTS:
        benchmark_mod = import_module('src.evaluation.benchmark_suite')
        if hasattr(benchmark_mod, name):
            globals()[name] = getattr(benchmark_mod, name)
            return globals()[name]
        alignment_mod = import_module('src.evaluation.pcg_benchmark_alignment')
        globals()[name] = getattr(alignment_mod, name)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
