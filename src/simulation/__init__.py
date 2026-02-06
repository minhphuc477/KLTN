"""
KLTN Simulation Module
======================
Validation and simulation components for Zelda dungeon analysis.

This module contains:
- validator: External validation suite (ZeldaValidator, ZeldaLogicEnv, etc.)
- cognitive_bounded_search: Human-like CBS solver with cognitive limitations
- dstar_lite: D* Lite incremental replanning
- multi_goal: Multi-goal pathfinding (TSP-style)
- parallel_astar: Parallel A* search
- solver_comparison: Solver benchmarking
- map_elites: Quality-Diversity evaluation
"""

# Re-export from local validator module
from .validator import (
    ZeldaValidator,
    ZeldaLogicEnv,
    SanityChecker,
    MetricsEngine,
    DiversityEvaluator,
    ValidationResult,
    BatchValidationResult,
    StateSpaceAStar,
    GameState,
    ACTION_DELTAS,
    SEMANTIC_PALETTE,
    WALKABLE_IDS,
    BLOCKING_IDS,
    WATER_IDS,
)

# Re-export Cognitive Bounded Search (CBS) components
from .cognitive_bounded_search import (
    CognitiveBoundedSearch,
    CBSMetrics,
    BeliefMap,
    VisionSystem,
    WorkingMemory,
    MemoryItem,
    MemoryItemType,
    TileObservation,
    TileKnowledge,
    CognitiveState,
    AgentPersona,
    PersonaConfig,
    # Heuristics
    DecisionHeuristic,
    CuriosityHeuristic,
    RecencyHeuristic,
    SafetyHeuristic,
    GoalSeekingHeuristic,
    ItemSeekingHeuristic,
    # Convenience functions
    solve_with_cbs,
    compare_personas,
)

# Re-export advanced solvers
from .dstar_lite import DStarLiteSolver
from .multi_goal import MultiGoalPathfinder
from .parallel_astar import ParallelAStarSolver
from .solver_comparison import SolverComparison
from .map_elites import MAPElitesEvaluator, run_map_elites_on_maps, plot_heatmap

__all__ = [
    # Core validator components
    'ZeldaValidator',
    'ZeldaLogicEnv',
    'SanityChecker',
    'MetricsEngine',
    'DiversityEvaluator',
    'ValidationResult',
    'BatchValidationResult',
    'StateSpaceAStar',
    'GameState',
    'ACTION_DELTAS',
    'SEMANTIC_PALETTE',
    'WALKABLE_IDS',
    'BLOCKING_IDS',
    'WATER_IDS',
    # Cognitive Bounded Search (CBS)
    'CognitiveBoundedSearch',
    'CBSMetrics',
    'BeliefMap',
    'VisionSystem',
    'WorkingMemory',
    'MemoryItem',
    'MemoryItemType',
    'TileObservation',
    'TileKnowledge',
    'CognitiveState',
    'AgentPersona',
    'PersonaConfig',
    'DecisionHeuristic',
    'CuriosityHeuristic',
    'RecencyHeuristic',
    'SafetyHeuristic',
    'GoalSeekingHeuristic',
    'ItemSeekingHeuristic',
    'solve_with_cbs',
    'compare_personas',
    # Advanced solvers
    'DStarLiteSolver',
    'MultiGoalPathfinder',
    'ParallelAStarSolver',
    'SolverComparison',
    'MAPElitesEvaluator',
    'run_map_elites_on_maps',
    'plot_heatmap',
]
