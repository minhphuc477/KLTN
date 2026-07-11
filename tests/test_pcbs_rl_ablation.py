import math

import numpy as np

from src.core.definitions import SEMANTIC_PALETTE
from src.evaluation.pcbs_rl_ablation import (
    compute_cross_persona_agreement,
    compute_persona_divergence_from_paths,
    run_pcbs_rl_alignment_ablation,
    train_belief_state_q_agent,
)


def _tiny_level() -> np.ndarray:
    grid = np.full((7, 7), SEMANTIC_PALETTE["WALL"], dtype=np.int64)
    grid[1:6, 1:6] = SEMANTIC_PALETTE["FLOOR"]
    grid[1, 1] = SEMANTIC_PALETTE["START"]
    grid[5, 5] = SEMANTIC_PALETTE["TRIFORCE"]
    grid[3, 3] = SEMANTIC_PALETTE["ENEMY"]
    return grid


def test_belief_state_q_agent_returns_finite_release_metrics():
    metrics = train_belief_state_q_agent(
        _tiny_level(),
        reward_variant="goal",
        memory_capacity=4,
        episodes=4,
        max_steps=40,
        seed=123,
    )

    payload = metrics.to_dict()
    assert payload["reward_variant"] == "goal"
    assert payload["memory_capacity"] == 4
    assert 0.0 <= payload["success_rate"] <= 1.0
    assert payload["total_steps"] >= 1
    assert math.isfinite(payload["confusion_index"])
    assert math.isfinite(payload["navigation_entropy"])
    assert math.isfinite(payload["cognitive_load"])
    assert math.isfinite(payload["linearity_ratio"])
    assert payload["combat_engagements"] >= 0
    assert payload["pickups_collected"] >= 0


def test_pcbs_rl_alignment_ablation_matches_rl_to_persona_profile():
    result = run_pcbs_rl_alignment_ablation(
        _tiny_level(),
        personas=("speedrunner", "explorer"),
        reward_variants=("goal", "combat"),
        memory_capacities=(4,),
        episodes=3,
        timeout_pcbs=100,
        seed=7,
    )

    assert set(result["persona_metrics"]) == {"speedrunner", "explorer"}
    assert len(result["rl_results"]) == 2
    for rl_row in result["rl_results"]:
        assert rl_row["closest_persona"] in {"speedrunner", "explorer"}
        assert 0.0 <= rl_row["alignment_score"] <= 1.0
    assert {row["reward_variant"] for row in result["rl_results"]} == {"goal", "combat"}
    assert 0.0 <= result["cross_persona_agreement_rate"] <= 1.0
    assert result["persona_divergence"] >= 0.0


def test_persona_divergence_and_agreement_helpers_are_stable():
    divergence = compute_persona_divergence_from_paths(
        {
            "a": [(0, 0), (0, 1), (0, 1)],
            "b": [(0, 0), (1, 0), (1, 0)],
        }
    )
    agreement = compute_cross_persona_agreement(
        {
            "a": {"success": True},
            "b": {"success": False},
            "c": {"success": False},
        }
    )

    assert divergence > 0.0
    assert agreement == 2 / 3
