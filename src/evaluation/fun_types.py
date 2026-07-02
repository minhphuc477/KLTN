"""Data structures for player-experience (fun) metrics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FrustrationMetrics:
    """Metrics quantifying player frustration."""

    backtracking_ratio: float
    dead_end_count: int
    unclear_goal_score: float
    empty_room_ratio: float
    total_frustration: float
    depth_backtracking_score: float = 0.0


@dataclass
class ExplorabilityMetrics:
    """Metrics quantifying exploration potential."""

    optional_room_ratio: float
    secret_count: int
    reward_density: float
    discovery_potential: float


@dataclass
class FlowMetrics:
    """Metrics for challenge-skill balance."""

    difficulty_progression: float
    skill_utilization: float
    challenge_balance: float
    flow_score: float


@dataclass
class PacingMetrics:
    """Metrics for tension curve and pacing."""

    tension_variance: float
    peak_placement: float
    rest_areas: int
    pacing_score: float


@dataclass
class FunMetrics:
    """Comprehensive fun/engagement metrics."""

    frustration: FrustrationMetrics
    explorability: ExplorabilityMetrics
    flow: FlowMetrics
    pacing: PacingMetrics
    overall_fun_score: float
