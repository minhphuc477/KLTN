"""Evolutionary topology generation package with legacy-compatible exports."""

from .converters import mission_graph_to_networkx, networkx_to_mission_graph
from .evaluator import TensionCurveEvaluator
from .executor import GraphGrammarExecutor
from .generator import EvolutionaryTopologyGenerator
from .individual import Individual
from .utils import print_graph_summary, visualize_evolution_stats
from ._shared import (
    CVTEliteArchive,
    DEFAULT_REALISM_TUNING,
    DEFAULT_REPLAY_PAYLOAD_MAX_BYTES,
    DEFAULT_ZELDA_TRANSITIONS,
)

__all__ = [
    'GraphGrammarExecutor',
    'mission_graph_to_networkx',
    'networkx_to_mission_graph',
    'TensionCurveEvaluator',
    'Individual',
    'EvolutionaryTopologyGenerator',
    'visualize_evolution_stats',
    'print_graph_summary',
    'DEFAULT_ZELDA_TRANSITIONS',
    'DEFAULT_REALISM_TUNING',
    'DEFAULT_REPLAY_PAYLOAD_MAX_BYTES',
    'CVTEliteArchive',
]
