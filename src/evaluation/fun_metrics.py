"""
Feature 4: Fun Metrics
======================
Quantify player experience: frustration, explorability, flow, pacing.

Problem:
    Current metrics (solvability, difficulty) don't capture "fun".
    Need objective measures of player engagement and experience quality.

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
    Master evaluator combining all fun metrics.
    
    Usage:
        evaluator = FunMetricsEvaluator()
        fun_metrics = evaluator.evaluate(
            mission_graph=graph,
            solution_path=path,
            room_contents=contents,
            critical_path=critical_set
        )
        
        print(f"Fun Score: {fun_metrics.overall_fun_score:.2f}")
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
        Comprehensive fun evaluation.
        
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
        
        # Overall fun score
        # Fun = high explorability + high flow - high frustration
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
