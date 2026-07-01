"""
Feature 4: Structural Experience Proxies
========================================
Compute graph-derived proxies for frustration, explorability, flow, and pacing.

These values are not direct measurements of enjoyment or engagement. They must
be calibrated against player traces or user studies before being interpreted as
human-experience evidence.

Solution:
    - Frustration Score: Backtracking, dead ends, unclear goals
    - Explorability: Discovery potential, secret rooms, rewards
    - Flow Score: Challenge-skill balance (Csikszentmihalyi)
    - Pacing Score: Tension curve alignment
    
Integration Point: MAP-Elites evaluation, after full dungeon generation
"""

import networkx as nx
from typing import Dict, List, Set

from src.evaluation.fun_analyzers import (
    ExplorabilityAnalyzer,
    FlowAnalyzer,
    FrustrationAnalyzer,
    PacingAnalyzer,
)
from src.evaluation.fun_types import (
    ExplorabilityMetrics,
    FlowMetrics,
    FrustrationMetrics,
    FunMetrics,
    PacingMetrics,
)

__all__ = [
    "FrustrationMetrics",
    "ExplorabilityMetrics",
    "FlowMetrics",
    "PacingMetrics",
    "FunMetrics",
    "FrustrationAnalyzer",
    "ExplorabilityAnalyzer",
    "FlowAnalyzer",
    "PacingAnalyzer",
    "FunMetricsEvaluator",
]


# ============================================================================
# MASTER FUN EVALUATOR
# ============================================================================

class FunMetricsEvaluator:
    """
    Combine structural experience proxies under the legacy ``FunMetrics`` API.
    
    Usage:
        evaluator = FunMetricsEvaluator()
        fun_metrics = evaluator.evaluate(
            mission_graph=graph,
            solution_path=path,
            room_contents=contents,
            critical_path=critical_set
        )
        
        print(f"Structural proxy score: {fun_metrics.overall_fun_score:.2f}")
    """
    
    def __init__(self):
        self.frustration_analyzer = FrustrationAnalyzer()
        self.explorability_analyzer = ExplorabilityAnalyzer()
        self.flow_analyzer = FlowAnalyzer()
        self.pacing_analyzer = PacingAnalyzer()
    
    def evaluate(
        self,
        mission_graph: nx.Graph,
        solution_path: List[int],
        room_contents: Dict[int, Dict],
        critical_path: Set[int]
    ) -> FunMetrics:
        """
        Evaluate graph- and path-derived experience proxies.
        
        Returns:
            FunMetrics with all sub-metrics and overall score
        """
        # Compute sub-metrics
        frustration = self.frustration_analyzer.analyze(
            mission_graph, solution_path, room_contents
        )
        
        explorability = self.explorability_analyzer.analyze(
            mission_graph, critical_path, room_contents
        )
        
        flow = self.flow_analyzer.analyze(
            mission_graph, solution_path, room_contents
        )
        
        pacing = self.pacing_analyzer.analyze(
            mission_graph, solution_path, room_contents
        )
        
        # Legacy aggregate proxy. Do not interpret as observed human enjoyment
        # without an external calibration study.
        overall_fun_score = (
            0.3 * explorability.discovery_potential +
            0.3 * flow.flow_score +
            0.2 * pacing.pacing_score +
            0.2 * (1.0 - frustration.total_frustration)
        )
        
        return FunMetrics(
            frustration=frustration,
            explorability=explorability,
            flow=flow,
            pacing=pacing,
            overall_fun_score=overall_fun_score
        )


# ============================================================================
# INTEGRATION EXAMPLE
# ============================================================================
# Example wiring lives with runtime integrations (e.g., MAP-Elites / GUI modules).
