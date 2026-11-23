from __future__ import annotations

import copy

from scripts.check_training_hyperparameters import build_training_hyperparameter_report
from src.config_system import merge_config


def _issues_by_id(report: dict) -> set[str]:
    return {str(issue["check_id"]) for issue in report["issues"]}


def test_default_training_hyperparameter_check_has_no_errors() -> None:
    config = merge_config(yaml_path=None, cli_overrides=None)
    report = build_training_hyperparameter_report(
        config,
        data_probe={"probed": True, "train_internal_total_samples": 459},
    )

    assert report["summary"]["status"] == "pass"
    assert report["summary"]["issue_counts"]["ERROR"] == 0
    assert all(plan["train_batches_per_epoch"] > 0 for plan in report["batch_plans"])


def test_training_hyperparameter_check_catches_condition_head_mismatch() -> None:
    config = merge_config(yaml_path=None, cli_overrides=None)
    bad_config = copy.deepcopy(config)
    bad_config["diffusion"]["context_dim"] = 250

    report = build_training_hyperparameter_report(bad_config)

    assert report["summary"]["status"] == "fail"
    assert "condition_output_heads_not_divisible" in _issues_by_id(report)


def test_training_hyperparameter_check_catches_zero_batches_with_drop_last() -> None:
    config = merge_config(yaml_path=None, cli_overrides=None)
    bad_config = copy.deepcopy(config)
    bad_config["dataset"]["batch_size"] = 8
    bad_config["dataset"]["drop_last"] = True
    bad_config["dataset"]["min_samples_per_epoch"] = 4

    report = build_training_hyperparameter_report(
        bad_config,
        data_probe={"probed": True, "train_internal_total_samples": 5},
    )

    assert report["summary"]["status"] == "fail"
    assert "zero_train_batches" in _issues_by_id(report)


def test_glmdm_yaml_presets_pass_training_hyperparameter_check() -> None:
    preset_paths = (
        "configs/glmdm_canonical.yaml",
        "configs/glmdm_g_l_local_logic.yaml",
        "configs/glmdm_g_graph.yaml",
        "configs/glmdm_n_local.yaml",
        "configs/zelda_hmolqd_masked_small.yaml",
    )
    for preset in preset_paths:
        config = merge_config(yaml_path=preset, cli_overrides=None)
        report = build_training_hyperparameter_report(config)
        assert report["summary"]["issue_counts"]["ERROR"] == 0, preset


def test_training_hyperparameter_check_catches_masked_heads_and_rrwp_mismatch() -> None:
    config = merge_config(yaml_path=None, cli_overrides=None)
    bad_config = copy.deepcopy(config)
    bad_config["masked_room"]["hidden_dim"] = 50
    bad_config["masked_room"]["unet_num_heads"] = 4
    bad_config["diffusion"]["condition_gnn_type"] = "gcn"
    bad_config["diffusion"]["condition_use_rrwp_edge_features"] = True

    report = build_training_hyperparameter_report(bad_config)
    issue_ids = _issues_by_id(report)

    assert report["summary"]["status"] == "fail"
    assert "masked_room_hidden_heads_not_divisible" in issue_ids
    assert "rrwp_incompatible_gnn_type" in issue_ids
