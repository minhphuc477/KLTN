from __future__ import annotations

from types import SimpleNamespace

from scripts.run_conditioning_logicnet_repair_ablation import (
    build_experiment_matrix,
    summarize_rows,
    write_plan,
)


def test_conditioning_logicnet_repair_matrix_is_complete():
    variants = build_experiment_matrix()
    names = {variant.name for variant in variants}

    assert len(variants) == 12
    assert "full__repair_on__logic_on" in names
    assert "no_graph_tokens__repair_off__logic_off" in names
    assert "no_stage_tokens__repair_on__logic_off" in names


def test_conditioning_logicnet_summary_separates_pre_post_validity():
    rows = [
        {
            "conditioning": "full",
            "repair_enabled": True,
            "logic_enabled": True,
            "pre_oracle_solved": False,
            "post_oracle_solved": True,
            "pre_pcbs_solved": False,
            "post_pcbs_solved": True,
            "repair_count": 2,
            "repair_time_sec": 0.25,
            "total_tiles_repaired": 12,
            "generation_time_sec": 3.0,
            "post_readability_score": 0.8,
            "post_bounded_rationality_index": 0.2,
            "logicnet_dungeon_solvability": 0.9,
        }
    ]

    summary = summarize_rows(rows)

    assert summary[0]["pre_oracle_valid_rate"] == 0.0
    assert summary[0]["post_oracle_valid_rate"] == 1.0
    assert summary[0]["repair_count_mean"] == 2.0


def test_conditioning_logicnet_plan_is_plan_only(tmp_path):
    args = SimpleNamespace(execute=False, seeds="1,2")
    variants = build_experiment_matrix()[:2]

    write_plan(tmp_path, variants, args, {"vqvae_checkpoint": None})

    assert (tmp_path / "conditioning_logicnet_repair_plan.json").exists()
    assert (tmp_path / "conditioning_logicnet_repair_plan.md").exists()
