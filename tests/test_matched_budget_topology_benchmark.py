import pandas as pd

from scripts.run_matched_budget_topology_benchmark import _build_summary_row


def test_build_summary_row_includes_topology_semantics_metrics():
    sub = pd.DataFrame(
        [
            {
                "fitness": 0.8,
                "feasible_search": 1.0,
                "feasible_operational": 1.0,
                "overall_completeness": 0.95,
                "constraint_valid": 1.0,
                "linearity": 0.45,
                "leniency": 0.70,
                "progression_complexity": 0.62,
                "topology_complexity": 0.51,
                "path_length": 7.0,
                "num_nodes": 10.0,
                "key_gate_count": 2.0,
                "key_before_lock_rate": 1.0,
                "switch_gate_count": 1.0,
                "switch_before_gate_rate": 1.0,
                "battery_gate_count": 1.0,
                "battery_satisfaction_rate": 1.0,
                "path_redundancy": 0.25,
                "articulation_count": 2.0,
                "articulation_ratio": 0.20,
                "branch_count": 3.0,
                "branch_utility_rate": 0.67,
                "secret_component_count": 1.0,
                "secret_content_discoverability_rate": 1.0,
                "repair_applied": 0.0,
                "generation_constraint_rejections": 0.0,
                "candidate_repairs_applied": 0.0,
                "novelty_vs_reference": 0.18,
                "graph_edit_distance": 0.22,
                "generation_time_sec": 1.5,
                "evaluations_used": 256.0,
            }
        ]
    )
    payload = {
        "completeness": {
            "key_before_lock_rate": 1.0,
            "switch_before_gate_rate": 1.0,
            "battery_satisfaction_rate": 1.0,
        },
        "expressive_range": {
            "mean_path_redundancy": 0.25,
            "mean_articulation_ratio": 0.20,
            "mean_branch_utility_rate": 0.67,
            "mean_secret_content_discoverability_rate": 1.0,
            "coverage_linearity_leniency": 0.11,
            "coverage_progression_topology": 0.12,
            "coverage_redundancy_articulation": 0.13,
            "coverage_branch_secret": 0.14,
        },
        "generated_descriptor_means": {
            "key_gate_count": 2.0,
            "switch_gate_count": 1.0,
            "battery_gate_count": 1.0,
            "articulation_count": 2.0,
            "branch_count": 3.0,
            "secret_component_count": 1.0,
        },
        "reference_comparison": {
            "fidelity_js_divergence": 0.05,
            "expressive_overlap_reference": 0.42,
        },
    }

    row = _build_summary_row(method="FULL", sub=sub, payload=payload)

    assert row["method"] == "FULL"
    assert row["key_gate_count"] == 2.0
    assert row["key_before_lock_rate"] == 1.0
    assert row["switch_before_gate_rate"] == 1.0
    assert row["battery_satisfaction_rate"] == 1.0
    assert row["path_redundancy"] == 0.25
    assert row["articulation_ratio"] == 0.20
    assert row["branch_utility_rate"] == 0.67
    assert row["secret_content_discoverability_rate"] == 1.0
    assert row["coverage_redundancy_articulation"] == 0.13
    assert row["coverage_branch_secret"] == 0.14
