from __future__ import annotations

import json
from types import SimpleNamespace

from scripts.run_conditioning_logicnet_repair_ablation import (
    build_experiment_matrix,
    build_logic_delta_rows,
    pipeline_cache_key,
    summarize_rows,
    validate_execute_checkpoints,
    write_plan,
)


def test_conditioning_logicnet_repair_matrix_is_complete():
    variants = build_experiment_matrix()
    names = {variant.name for variant in variants}

    assert len(variants) == 12
    assert "full__repair_on__logic_on" in names
    assert "no_graph_tokens__repair_off__logic_off" in names
    assert "no_stage_tokens__repair_on__logic_off" in names


def test_pipeline_cache_key_reuses_runtime_only_logic_and_repair_variants():
    variants = {variant.name: variant for variant in build_experiment_matrix()}
    checkpoints = {
        "vqvae_checkpoint": "vqvae.pth",
        "diffusion_checkpoint": "diffusion.pth",
        "logic_net_checkpoint": "logic.pth",
    }

    full_off = pipeline_cache_key(
        variants["full__repair_off__logic_off"],
        checkpoints,
        device="cuda",
    )
    full_on = pipeline_cache_key(
        variants["full__repair_on__logic_on"],
        checkpoints,
        device="cuda",
    )
    no_stage = pipeline_cache_key(
        variants["no_stage_tokens__repair_on__logic_on"],
        checkpoints,
        device="cuda",
    )
    no_graph = pipeline_cache_key(
        variants["no_graph_tokens__repair_on__logic_on"],
        checkpoints,
        device="cuda",
    )

    assert full_off == full_on
    assert full_off != no_stage
    assert full_off != no_graph


def test_conditioning_logicnet_summary_separates_pre_post_validity():
    rows = [
        {
            "conditioning": "full",
            "repair_enabled": True,
            "logic_enabled": True,
            "raw_oracle_solved": False,
            "pre_oracle_solved": False,
            "post_oracle_solved": True,
            "raw_pcbs_solved": False,
            "pre_pcbs_solved": False,
            "post_pcbs_solved": True,
            "raw_invalid_tile_ids": 4,
            "raw_to_cleaned_tiles_changed": 5,
            "raw_to_final_tiles_changed": 17,
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
    assert summary[0]["raw_oracle_valid_rate"] == 0.0
    assert summary[0]["post_oracle_valid_rate"] == 1.0
    assert summary[0]["raw_solvability_rate_without_fix"] == 0.0
    assert summary[0]["pre_repair_solvability_rate"] == 0.0
    assert summary[0]["post_solvability_rate"] == 1.0
    assert summary[0]["wfc_destroyed_raw_solution_rate"] == 0.0
    assert summary[0]["repair_recovered_raw_failure_rate"] == 1.0
    assert summary[0]["raw_invalid_tile_ids_mean"] == 4.0
    assert summary[0]["raw_to_cleaned_tiles_changed_mean"] == 5.0
    assert summary[0]["raw_to_final_tiles_changed_mean"] == 17.0
    assert summary[0]["repair_count_mean"] == 2.0


def test_conditioning_logicnet_delta_rows_are_paired_by_seed_condition_and_repair():
    rows = [
        {
            "variant": "full__repair_on__logic_off",
            "conditioning": "full",
            "repair_enabled": True,
            "logic_enabled": False,
            "seed": 7,
            "raw_oracle_solved": False,
            "pre_oracle_solved": False,
            "post_oracle_solved": True,
            "post_readability_score": 0.50,
            "logicnet_dungeon_solvability": 0.40,
        },
        {
            "variant": "full__repair_on__logic_on",
            "conditioning": "full",
            "repair_enabled": True,
            "logic_enabled": True,
            "seed": 7,
            "raw_oracle_solved": False,
            "pre_oracle_solved": True,
            "post_oracle_solved": True,
            "post_readability_score": 0.75,
            "logicnet_dungeon_solvability": 0.90,
        },
    ]

    deltas = build_logic_delta_rows(rows)

    assert len(deltas) == 1
    assert deltas[0]["raw_oracle_solved_delta_on_minus_off"] == 0.0
    assert deltas[0]["pre_oracle_solved_delta_on_minus_off"] == 1.0
    assert deltas[0]["post_readability_score_delta_on_minus_off"] == 0.25
    assert deltas[0]["logicnet_dungeon_solvability_delta_on_minus_off"] == 0.5


def test_conditioning_logicnet_execute_requires_checkpoints_unless_explicitly_allowed():
    variants = build_experiment_matrix()

    try:
        validate_execute_checkpoints(
            {"vqvae_checkpoint": None, "diffusion_checkpoint": None, "logic_net_checkpoint": None},
            variants,
        )
    except FileNotFoundError as exc:
        assert "Missing" in str(exc)
        assert "vqvae_checkpoint" in str(exc)
    else:
        raise AssertionError("Missing checkpoints should fail execute validation")

    validate_execute_checkpoints(
        {"vqvae_checkpoint": None, "diffusion_checkpoint": None, "logic_net_checkpoint": None},
        variants,
        allow_random_fallback=True,
    )


def test_conditioning_logicnet_plan_is_plan_only(tmp_path):
    args = SimpleNamespace(execute=False, seeds="1,2")
    variants = build_experiment_matrix()[:2]

    write_plan(tmp_path, variants, args, {"vqvae_checkpoint": None})

    assert (tmp_path / "conditioning_logicnet_repair_plan.json").exists()
    assert (tmp_path / "conditioning_logicnet_repair_plan.md").exists()
    payload = json.loads((tmp_path / "conditioning_logicnet_repair_plan.json").read_text(encoding="utf-8"))
    assert payload["pipeline_initialization_upper_bound"] == 1
    assert payload["pipeline_cache_fields"] == [
        "conditioning",
        "device",
        "vqvae_checkpoint",
        "diffusion_checkpoint",
        "logic_net_checkpoint",
    ]
