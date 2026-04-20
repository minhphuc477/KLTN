from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

import main
import src.train as legacy_train
from src.core.gaussian_vae import create_gaussian_vae
from src.core.vqvae import create_vqvae
from src.config_system import (
    cli_overrides_from_namespace,
    find_resolved_config_path,
    load_resolved_config_for_artifact,
    merge_config,
)
from src.pipeline import (
    generation_runtime_kwargs_from_resolved_config,
    pipeline_kwargs_from_resolved_config,
    topology_generation_kwargs_from_resolved_config,
)
from src.train_diffusion import (
    DiffusionTrainer,
    DiffusionTrainingConfig,
    _resolve_vqvae_architecture,
    build_diffusion_training_config_from_args,
    diffusion_training_kwargs_from_resolved_config,
)
from src.core.definitions import ROOM_TOPOLOGY_CHANNEL_COUNT
from src.train_lcm import build_fast_sampler_training_config_from_args, fast_sampler_training_kwargs_from_resolved_config
from src.train_masked_room import masked_room_training_kwargs_from_resolved_config
from src.train_vqvae import build_vqvae_training_args_from_args, vqvae_training_kwargs_from_resolved_config


def _write_yaml(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def test_yaml_and_cli_merge_order_prefers_cli(tmp_path: Path):
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(
        cfg_path,
        {
            "training": {"stage": "diffusion"},
            "runtime": {"output_dir": str(tmp_path / "yaml_out")},
            "dataset": {"batch_size": 2},
            "diffusion": {"epochs": 5},
        },
    )

    parser = main._build_root_parser()
    args = parser.parse_args(
        [
            "train",
            "--config",
            str(cfg_path),
            "--stage",
            "masked_room",
            "--batch-size",
            "8",
            "--diffusion-epochs",
            "7",
            "--output-dir",
            str(tmp_path / "cli_out"),
        ]
    )

    resolved = merge_config(yaml_path=args.config, cli_overrides=cli_overrides_from_namespace(args))

    assert resolved["training"]["stage"] == "masked_room"
    assert resolved["dataset"]["batch_size"] == 8
    assert resolved["diffusion"]["epochs"] == 7
    assert Path(resolved["runtime"]["output_dir"]) == (tmp_path / "cli_out")


def test_validation_rejects_fixed_schema_breakage(tmp_path: Path):
    cfg_path = tmp_path / "bad.yaml"
    _write_yaml(
        cfg_path,
        {
            "dataset": {"tpe_dim": 7},
        },
    )

    with pytest.raises(ValueError, match="dataset.tpe_dim"):
        merge_config(yaml_path=str(cfg_path), cli_overrides=None)


def test_diffusion_unet_list_fields_merge_from_yaml_and_cli(tmp_path: Path):
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(
        cfg_path,
        {
            "diffusion": {
                "unet_channel_mult": [1, 3, 4],
                "unet_attention_resolutions": [1, 2],
                "unet_num_heads": 8,
            },
        },
    )

    parser = main._build_root_parser()
    args = parser.parse_args(
        [
            "train",
            "--config",
            str(cfg_path),
            "--diffusion-unet-channel-mult",
            "1",
            "2",
            "4",
            "--diffusion-unet-attention-resolutions",
            "0",
            "2",
        ]
    )

    resolved = merge_config(yaml_path=args.config, cli_overrides=cli_overrides_from_namespace(args))

    assert resolved["diffusion"]["unet_channel_mult"] == [1, 2, 4]
    assert resolved["diffusion"]["unet_attention_resolutions"] == [0, 2]


def test_diffusion_config_accepts_gps_condition_backbone_and_spade_topology_mode(tmp_path: Path):
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(
        cfg_path,
        {
            "diffusion": {
                "condition_gnn_type": "gps",
                "topology_conditioning_mode": "spade",
            },
        },
    )

    resolved = merge_config(yaml_path=str(cfg_path), cli_overrides=None)

    assert resolved["diffusion"]["condition_gnn_type"] == "gps"
    assert resolved["diffusion"]["topology_conditioning_mode"] == "spade"


def test_default_config_uses_small_data_recommended_room_model_profile():
    resolved = merge_config(yaml_path=None, cli_overrides=None)

    assert resolved["runtime"]["auto_resume"] is True
    assert resolved["runtime"]["checkpoint_storage_budget_gb"] is None
    assert resolved["runtime"]["checkpoint_storage_warning_fraction"] == pytest.approx(0.8)
    assert resolved["runtime"]["checkpoint_storage_cleanup_enabled"] is True
    assert resolved["runtime"]["checkpoint_storage_cleanup_target_fraction"] == pytest.approx(0.6)
    assert resolved["dataset"]["topology_supervision_mode"] == "runtime_aligned"
    assert resolved["vqvae"]["keep_last"] == 2
    assert resolved["vqvae"]["validation_fraction"] == pytest.approx(0.1)
    assert resolved["vqvae"]["validation_max_batches"] == 16
    assert resolved["vqvae"]["best_checkpoint_metric"] == "val_loss"
    assert resolved["diffusion"]["keep_last"] == 2
    assert resolved["fast_sampler"]["keep_last"] == 2
    assert resolved["fast_sampler"]["decode_alignment_weight"] == pytest.approx(0.25)
    assert resolved["fast_sampler"]["topology_alignment_weight"] == pytest.approx(0.25)
    assert resolved["fast_sampler"]["topology_marker_weight"] == pytest.approx(2.0)
    assert resolved["fast_sampler"]["topology_trace_weight"] == pytest.approx(0.75)
    assert resolved["fast_sampler"]["topology_focus_dilation"] == 1
    assert resolved["fast_sampler"]["validation_fraction"] == pytest.approx(0.1)
    assert resolved["fast_sampler"]["validation_max_batches"] == 16
    assert resolved["fast_sampler"]["best_checkpoint_metric"] == "val_decode_ce_loss"
    assert resolved["masked_room"]["keep_last"] == 2
    assert resolved["diffusion"]["model_channels"] == 96
    assert resolved["diffusion"]["condition_hidden_dim"] == 192
    assert resolved["diffusion"]["condition_num_gnn_layers"] == 2
    assert resolved["diffusion"]["condition_gnn_type"] == "gps"
    assert resolved["diffusion"]["condition_use_reference_room_maps"] is True
    assert resolved["masked_room"]["model_channels"] == 64
    assert resolved["masked_room"]["hidden_dim"] == 48
    assert resolved["masked_room"]["condition_hidden_dim"] == 192
    assert resolved["masked_room"]["condition_num_gnn_layers"] == 2
    assert resolved["masked_room"]["condition_use_reference_room_maps"] is True
    assert resolved["masked_room"]["unet_channel_mult"] == [1, 2]
    assert resolved["masked_room"]["unet_num_res_blocks"] == 1
    assert resolved["masked_room"]["unet_attention_resolutions"] == [0, 1]
    assert resolved["masked_room"]["unet_num_heads"] == 4
    assert resolved["masked_room"]["min_mask_ratio"] == pytest.approx(0.12)
    assert resolved["masked_room"]["max_mask_ratio"] == pytest.approx(0.85)
    assert resolved["fast_sampler"]["topology_alignment_weight"] == pytest.approx(0.25)
    assert resolved["fast_sampler"]["topology_marker_weight"] == pytest.approx(2.0)
    assert resolved["fast_sampler"]["topology_trace_weight"] == pytest.approx(0.75)
    assert resolved["fast_sampler"]["topology_focus_dilation"] == 1
    assert resolved["masked_room"]["topology_alignment_weight"] == pytest.approx(0.25)
    assert resolved["masked_room"]["topology_marker_weight"] == pytest.approx(2.0)
    assert resolved["masked_room"]["topology_trace_weight"] == pytest.approx(0.75)
    assert resolved["masked_room"]["topology_focus_dilation"] == 1
    assert resolved["masked_room"]["validation_fraction"] == pytest.approx(0.1)
    assert resolved["masked_room"]["validation_max_batches"] == 16
    assert resolved["masked_room"]["best_checkpoint_metric"] == "val_loss"
    assert resolved["masked_room"]["topology_alignment_weight"] == pytest.approx(0.25)
    assert resolved["masked_room"]["topology_marker_weight"] == pytest.approx(2.0)
    assert resolved["masked_room"]["topology_trace_weight"] == pytest.approx(0.75)
    assert resolved["masked_room"]["topology_focus_dilation"] == 1
    assert resolved["dataset"]["topology_supervision_mode"] == "runtime_aligned"
    assert resolved["generation"]["guidance_scale"] == pytest.approx(3.0)
    assert resolved["generation"]["logic_guidance_scale"] == pytest.approx(0.0)
    assert resolved["generation"]["num_diffusion_steps"] == 50
    assert resolved["generation"]["apply_repair"] is True
    assert resolved["generation"]["enable_map_elites"] is False
    assert resolved["generation"]["symbolic_max_repair_attempts"] == 5
    assert resolved["generation"]["symbolic_repair_margin"] == 2
    assert resolved["generation"]["symbolic_adjacency_threshold"] == pytest.approx(0.01)
    assert resolved["generation"]["semantic_role_prior_strength"] == pytest.approx(0.15)
    assert resolved["generation"]["semantic_anchor_threshold"] == pytest.approx(0.5)
    assert resolved["generation"]["semantic_puzzle_offset"] == 2
    assert resolved["generation"]["semantic_constrained_decoding_enabled"] is True
    assert resolved["generation"]["semantic_marker_logit_bias"] == pytest.approx(10000.0)
    assert resolved["generation"]["semantic_marker_suppression_bias"] == pytest.approx(100.0)
    assert resolved["generation"]["puzzle_room_scaffold_enabled"] is True
    assert resolved["generation"]["puzzle_room_structure_enabled"] is True
    assert resolved["generation"]["puzzle_room_scaffold_min_structure_tiles"] == 10
    assert resolved["generation"]["puzzle_room_archetype_mode"] == "auto"
    assert resolved["generation"]["puzzle_room_branch_density"] == pytest.approx(0.75)
    assert resolved["generation"]["puzzle_room_block_budget"] == 28
    assert resolved["generation"]["puzzle_room_preserve_route_margin"] == 0
    assert resolved["generation"]["puzzle_room_switch_pocket_depth"] == 3
    assert resolved["generation"]["puzzle_room_resource_bypass_offset"] == 2
    assert resolved["generation"]["puzzle_room_key_pocket_depth"] == 3
    assert resolved["generation"]["puzzle_room_item_slot_depth"] == 3
    assert resolved["generation"]["puzzle_room_toggle_corridor_offset"] == 2
    assert resolved["generation"]["puzzle_room_novelty_enabled"] is True
    assert resolved["generation"]["puzzle_room_candidate_count"] == 4
    assert resolved["generation"]["puzzle_room_novelty_weight"] == pytest.approx(0.45)
    assert resolved["generation"]["puzzle_room_min_quality_gain"] == pytest.approx(0.5)
    assert resolved["generation"]["validator_plan_max_states"] == 512
    assert resolved["generation"]["deterministic_graph_marker_overlay_enabled"] is True
    assert resolved["generation"]["fast_sampler_teacher_fallback_enabled"] is True
    assert resolved["generation"]["masked_room_teacher_fallback_enabled"] is True
    assert resolved["generation"]["masked_room_sampling_temperature"] == pytest.approx(1.0)
    assert resolved["generation"]["masked_room_sampling_schedule"] == "cosine"
    assert resolved["generation"]["masked_room_sampling_stochastic"] is True
    assert resolved["generation"]["masked_room_corrector_steps"] == 1
    assert resolved["generation"]["masked_room_corrector_mask_ratio"] == pytest.approx(0.1)
    assert resolved["diffusion"]["validation_num_samples"] == 8
    assert resolved["diffusion"]["validation_num_diffusion_samples"] == 64


def test_canonical_yaml_uses_downsized_masked_room_profile():
    resolved = merge_config(yaml_path="configs/zelda_hmolqd.yaml", cli_overrides=None)

    assert resolved["diffusion"]["room_topology_channels"] == ROOM_TOPOLOGY_CHANNEL_COUNT
    assert resolved["masked_room"]["room_topology_channels"] == ROOM_TOPOLOGY_CHANNEL_COUNT
    assert resolved["masked_room"]["model_channels"] == 64
    assert resolved["masked_room"]["hidden_dim"] == 48
    assert resolved["masked_room"]["unet_channel_mult"] == [1, 2]
    assert resolved["masked_room"]["unet_num_res_blocks"] == 1
    assert resolved["masked_room"]["unet_attention_resolutions"] == [0, 1]
    assert resolved["masked_room"]["unet_num_heads"] == 4
    assert resolved["masked_room"]["min_mask_ratio"] == pytest.approx(0.12)
    assert resolved["masked_room"]["max_mask_ratio"] == pytest.approx(0.85)
    assert resolved["generation"]["guidance_scale"] == pytest.approx(3.0)
    assert resolved["generation"]["logic_guidance_scale"] == pytest.approx(0.0)
    assert resolved["generation"]["symbolic_max_repair_attempts"] == 5
    assert resolved["generation"]["symbolic_repair_margin"] == 2
    assert resolved["generation"]["symbolic_adjacency_threshold"] == pytest.approx(0.01)
    assert resolved["generation"]["semantic_role_prior_strength"] == pytest.approx(0.15)
    assert resolved["generation"]["semantic_anchor_threshold"] == pytest.approx(0.5)
    assert resolved["generation"]["semantic_puzzle_offset"] == 2
    assert resolved["generation"]["semantic_constrained_decoding_enabled"] is True
    assert resolved["generation"]["semantic_marker_logit_bias"] == pytest.approx(10000.0)
    assert resolved["generation"]["semantic_marker_suppression_bias"] == pytest.approx(100.0)
    assert resolved["generation"]["puzzle_room_scaffold_enabled"] is True
    assert resolved["generation"]["puzzle_room_structure_enabled"] is True
    assert resolved["generation"]["puzzle_room_scaffold_min_structure_tiles"] == 10
    assert resolved["generation"]["puzzle_room_archetype_mode"] == "auto"
    assert resolved["generation"]["puzzle_room_branch_density"] == pytest.approx(0.75)
    assert resolved["generation"]["puzzle_room_block_budget"] == 28
    assert resolved["generation"]["puzzle_room_preserve_route_margin"] == 0
    assert resolved["generation"]["puzzle_room_switch_pocket_depth"] == 3
    assert resolved["generation"]["puzzle_room_resource_bypass_offset"] == 2
    assert resolved["generation"]["puzzle_room_key_pocket_depth"] == 3
    assert resolved["generation"]["puzzle_room_item_slot_depth"] == 3
    assert resolved["generation"]["puzzle_room_toggle_corridor_offset"] == 2
    assert resolved["generation"]["puzzle_room_novelty_enabled"] is True
    assert resolved["generation"]["puzzle_room_candidate_count"] == 4
    assert resolved["generation"]["puzzle_room_novelty_weight"] == pytest.approx(0.45)
    assert resolved["generation"]["puzzle_room_min_quality_gain"] == pytest.approx(0.5)
    assert resolved["generation"]["validator_plan_max_states"] == 512
    assert resolved["generation"]["deterministic_graph_marker_overlay_enabled"] is True
    assert resolved["generation"]["fast_sampler_teacher_fallback_enabled"] is True
    assert resolved["generation"]["masked_room_teacher_fallback_enabled"] is True
    assert resolved["generation"]["masked_room_sampling_temperature"] == pytest.approx(1.0)
    assert resolved["generation"]["masked_room_sampling_schedule"] == "cosine"
    assert resolved["generation"]["masked_room_sampling_stochastic"] is True
    assert resolved["generation"]["masked_room_corrector_steps"] == 1
    assert resolved["generation"]["masked_room_corrector_mask_ratio"] == pytest.approx(0.1)


def test_stage_helpers_forward_checkpoint_retention_and_resume_defaults():
    resolved = merge_config(yaml_path=None, cli_overrides=None)

    diffusion_kwargs = diffusion_training_kwargs_from_resolved_config(resolved)
    fast_sampler_kwargs = fast_sampler_training_kwargs_from_resolved_config(resolved)
    masked_room_kwargs = masked_room_training_kwargs_from_resolved_config(resolved)
    vqvae_kwargs = vqvae_training_kwargs_from_resolved_config(resolved)

    assert diffusion_kwargs["keep_last"] == 2
    assert diffusion_kwargs["auto_resume"] is True
    assert diffusion_kwargs["resume_checkpoint"] is None
    assert diffusion_kwargs["checkpoint_storage_budget_gb"] is None
    assert fast_sampler_kwargs["keep_last"] == 2
    assert fast_sampler_kwargs["auto_resume"] is True
    assert fast_sampler_kwargs["resume_checkpoint"] is None
    assert fast_sampler_kwargs["checkpoint_storage_budget_gb"] is None
    assert fast_sampler_kwargs["decode_alignment_weight"] == pytest.approx(0.25)
    assert fast_sampler_kwargs["topology_alignment_weight"] == pytest.approx(0.25)
    assert fast_sampler_kwargs["topology_marker_weight"] == pytest.approx(2.0)
    assert fast_sampler_kwargs["topology_trace_weight"] == pytest.approx(0.75)
    assert fast_sampler_kwargs["topology_focus_dilation"] == 1
    assert fast_sampler_kwargs["validation_fraction"] == pytest.approx(0.1)
    assert fast_sampler_kwargs["validation_max_batches"] == 16
    assert fast_sampler_kwargs["best_checkpoint_metric"] == "val_decode_ce_loss"
    assert masked_room_kwargs["keep_last"] == 2
    assert masked_room_kwargs["auto_resume"] is True
    assert masked_room_kwargs["resume_checkpoint"] is None
    assert masked_room_kwargs["checkpoint_storage_budget_gb"] is None
    assert masked_room_kwargs["topology_alignment_weight"] == pytest.approx(0.25)
    assert masked_room_kwargs["topology_marker_weight"] == pytest.approx(2.0)
    assert masked_room_kwargs["topology_trace_weight"] == pytest.approx(0.75)
    assert masked_room_kwargs["topology_focus_dilation"] == 1
    assert masked_room_kwargs["validation_fraction"] == pytest.approx(0.1)
    assert masked_room_kwargs["validation_max_batches"] == 16
    assert masked_room_kwargs["best_checkpoint_metric"] == "val_loss"
    assert vqvae_kwargs["keep_last"] == 2
    assert vqvae_kwargs["validation_fraction"] == pytest.approx(0.1)
    assert vqvae_kwargs["validation_max_batches"] == 16
    assert vqvae_kwargs["best_checkpoint_metric"] == "val_loss"
    assert vqvae_kwargs["auto_resume"] is True
    assert vqvae_kwargs["checkpoint_storage_budget_gb"] is None
    assert vqvae_kwargs["resume"] is None
    assert diffusion_kwargs["vqvae_hidden_dim"] == resolved["vqvae"]["hidden_dim"]
    assert diffusion_kwargs["vqvae_codebook_size"] == resolved["vqvae"]["codebook_size"]
    assert diffusion_kwargs["vqvae_use_coordconv"] == resolved["vqvae"]["use_coordconv"]
    assert diffusion_kwargs["vqvae_mrf_penalty_weight"] == pytest.approx(resolved["vqvae"]["mrf_penalty_weight"])
    assert diffusion_kwargs["validation_num_samples"] == 8
    assert diffusion_kwargs["validation_num_diffusion_samples"] == 64

    generation_kwargs = generation_runtime_kwargs_from_resolved_config(resolved)
    pipeline_kwargs = pipeline_kwargs_from_resolved_config(resolved)
    assert generation_kwargs["default_guidance_scale"] == pytest.approx(3.0)
    assert generation_kwargs["default_logic_guidance_scale"] == pytest.approx(0.0)
    assert generation_kwargs["default_num_diffusion_steps"] == 50
    assert generation_kwargs["symbolic_max_repair_attempts"] == 5
    assert generation_kwargs["symbolic_repair_margin"] == 2
    assert generation_kwargs["symbolic_adjacency_threshold"] == pytest.approx(0.01)
    assert generation_kwargs["default_start_goal_coords"] == ((1, 5), (14, 5))
    assert generation_kwargs["default_semantic_role_prior_strength"] == pytest.approx(0.15)
    assert generation_kwargs["default_semantic_anchor_threshold"] == pytest.approx(0.5)
    assert generation_kwargs["default_semantic_puzzle_offset"] == 2
    assert generation_kwargs["default_semantic_constrained_decoding_enabled"] is True
    assert generation_kwargs["default_semantic_marker_logit_bias"] == pytest.approx(10000.0)
    assert generation_kwargs["default_semantic_marker_suppression_bias"] == pytest.approx(100.0)
    assert generation_kwargs["default_puzzle_room_scaffold_enabled"] is True
    assert generation_kwargs["default_puzzle_room_structure_enabled"] is True
    assert generation_kwargs["default_puzzle_room_scaffold_min_structure_tiles"] == 10
    assert generation_kwargs["default_puzzle_room_archetype_mode"] == "auto"
    assert generation_kwargs["default_puzzle_room_branch_density"] == pytest.approx(0.75)
    assert generation_kwargs["default_puzzle_room_block_budget"] == 28
    assert generation_kwargs["default_puzzle_room_preserve_route_margin"] == 0
    assert generation_kwargs["default_puzzle_room_switch_pocket_depth"] == 3
    assert generation_kwargs["default_puzzle_room_resource_bypass_offset"] == 2
    assert generation_kwargs["default_puzzle_room_key_pocket_depth"] == 3
    assert generation_kwargs["default_puzzle_room_item_slot_depth"] == 3
    assert generation_kwargs["default_puzzle_room_toggle_corridor_offset"] == 2
    assert generation_kwargs["default_puzzle_room_novelty_enabled"] is True
    assert generation_kwargs["default_puzzle_room_candidate_count"] == 4
    assert generation_kwargs["default_puzzle_room_novelty_weight"] == pytest.approx(0.45)
    assert generation_kwargs["default_puzzle_room_min_quality_gain"] == pytest.approx(0.5)
    assert generation_kwargs["default_validator_plan_max_states"] == 512
    assert generation_kwargs["default_deterministic_graph_marker_overlay_enabled"] is True
    assert generation_kwargs["default_fast_sampler_teacher_fallback_enabled"] is True
    assert generation_kwargs["default_masked_room_teacher_fallback_enabled"] is True
    assert generation_kwargs["default_masked_room_sampling_temperature"] == pytest.approx(1.0)
    assert generation_kwargs["default_masked_room_sampling_schedule"] == "cosine"
    assert generation_kwargs["default_masked_room_sampling_stochastic"] is True
    assert generation_kwargs["default_masked_room_corrector_steps"] == 1
    assert generation_kwargs["default_masked_room_corrector_mask_ratio"] == pytest.approx(0.1)
    assert pipeline_kwargs["topology_default_target_curve"] == resolved["topology"]["default_target_curve"]
    assert pipeline_kwargs["topology_num_rooms"] == resolved["topology"]["num_rooms"]
    assert pipeline_kwargs["topology_population_size"] == resolved["topology"]["population_size"]
    assert "target_curve" not in pipeline_kwargs
    assert "num_rooms" not in pipeline_kwargs
    assert pipeline_kwargs["condition_gnn_type"] == resolved["diffusion"]["condition_gnn_type"]
    assert pipeline_kwargs["fast_sampling_steps"] == resolved["fast_sampler"]["num_inference_steps"]
    assert pipeline_kwargs["diffusion_fallback_config"]["model_channels"] == resolved["diffusion"]["model_channels"]
    assert pipeline_kwargs["condition_encoder_fallback_config"]["context_dim"] == resolved["diffusion"]["context_dim"]
    assert pipeline_kwargs["masked_room_fallback_config"]["model_channels"] == resolved["masked_room"]["model_channels"]


def test_generation_runtime_kwargs_remain_backward_compatible_when_newer_generation_fields_are_missing():
    resolved = merge_config(yaml_path=None, cli_overrides=None)
    generation = resolved["generation"]
    for key in (
        "semantic_role_prior_strength",
        "symbolic_max_repair_attempts",
        "symbolic_repair_margin",
        "symbolic_adjacency_threshold",
        "semantic_anchor_threshold",
        "semantic_puzzle_offset",
        "semantic_constrained_decoding_enabled",
        "semantic_marker_logit_bias",
        "semantic_marker_suppression_bias",
        "puzzle_room_scaffold_enabled",
        "puzzle_room_structure_enabled",
        "puzzle_room_scaffold_min_structure_tiles",
        "puzzle_room_archetype_mode",
        "puzzle_room_branch_density",
        "puzzle_room_block_budget",
        "puzzle_room_preserve_route_margin",
        "puzzle_room_switch_pocket_depth",
        "puzzle_room_resource_bypass_offset",
        "puzzle_room_key_pocket_depth",
        "puzzle_room_item_slot_depth",
        "puzzle_room_toggle_corridor_offset",
        "puzzle_room_novelty_enabled",
        "puzzle_room_candidate_count",
        "puzzle_room_novelty_weight",
        "puzzle_room_min_quality_gain",
        "validator_plan_max_states",
        "deterministic_graph_marker_overlay_enabled",
        "fast_sampler_teacher_fallback_enabled",
        "masked_room_teacher_fallback_enabled",
        "masked_room_sampling_temperature",
        "masked_room_sampling_schedule",
        "masked_room_sampling_stochastic",
        "masked_room_corrector_steps",
        "masked_room_corrector_mask_ratio",
    ):
        generation.pop(key, None)

    generation_kwargs = generation_runtime_kwargs_from_resolved_config(resolved)
    pipeline_kwargs = pipeline_kwargs_from_resolved_config(resolved)

    assert generation_kwargs["default_semantic_role_prior_strength"] == pytest.approx(0.15)
    assert generation_kwargs["symbolic_max_repair_attempts"] == 5
    assert generation_kwargs["symbolic_repair_margin"] == 2
    assert generation_kwargs["symbolic_adjacency_threshold"] == pytest.approx(0.01)
    assert generation_kwargs["default_semantic_anchor_threshold"] == pytest.approx(0.5)
    assert generation_kwargs["default_semantic_puzzle_offset"] == 2
    assert generation_kwargs["default_semantic_constrained_decoding_enabled"] is True
    assert generation_kwargs["default_semantic_marker_logit_bias"] == pytest.approx(10000.0)
    assert generation_kwargs["default_semantic_marker_suppression_bias"] == pytest.approx(100.0)
    assert generation_kwargs["default_puzzle_room_scaffold_enabled"] is True
    assert generation_kwargs["default_puzzle_room_structure_enabled"] is True
    assert generation_kwargs["default_puzzle_room_scaffold_min_structure_tiles"] == 10
    assert generation_kwargs["default_puzzle_room_archetype_mode"] == "auto"
    assert generation_kwargs["default_puzzle_room_branch_density"] == pytest.approx(0.75)
    assert generation_kwargs["default_puzzle_room_block_budget"] == 28
    assert generation_kwargs["default_puzzle_room_preserve_route_margin"] == 0
    assert generation_kwargs["default_puzzle_room_switch_pocket_depth"] == 3
    assert generation_kwargs["default_puzzle_room_resource_bypass_offset"] == 2
    assert generation_kwargs["default_puzzle_room_key_pocket_depth"] == 3
    assert generation_kwargs["default_puzzle_room_item_slot_depth"] == 3
    assert generation_kwargs["default_puzzle_room_toggle_corridor_offset"] == 2
    assert generation_kwargs["default_puzzle_room_novelty_enabled"] is True
    assert generation_kwargs["default_puzzle_room_candidate_count"] == 4
    assert generation_kwargs["default_puzzle_room_novelty_weight"] == pytest.approx(0.45)
    assert generation_kwargs["default_puzzle_room_min_quality_gain"] == pytest.approx(0.5)
    assert generation_kwargs["default_validator_plan_max_states"] == 512
    assert generation_kwargs["default_deterministic_graph_marker_overlay_enabled"] is True
    assert generation_kwargs["default_fast_sampler_teacher_fallback_enabled"] is True
    assert generation_kwargs["default_masked_room_teacher_fallback_enabled"] is True
    assert generation_kwargs["default_masked_room_sampling_temperature"] == pytest.approx(1.0)
    assert generation_kwargs["default_masked_room_sampling_schedule"] == "cosine"
    assert generation_kwargs["default_masked_room_sampling_stochastic"] is True
    assert generation_kwargs["default_masked_room_corrector_steps"] == 1
    assert generation_kwargs["default_masked_room_corrector_mask_ratio"] == pytest.approx(0.1)
    assert pipeline_kwargs["default_semantic_role_prior_strength"] == pytest.approx(0.15)
    assert pipeline_kwargs["symbolic_max_repair_attempts"] == 5
    assert pipeline_kwargs["symbolic_repair_margin"] == 2
    assert pipeline_kwargs["symbolic_adjacency_threshold"] == pytest.approx(0.01)
    assert pipeline_kwargs["default_semantic_anchor_threshold"] == pytest.approx(0.5)
    assert pipeline_kwargs["default_semantic_puzzle_offset"] == 2
    assert pipeline_kwargs["default_semantic_constrained_decoding_enabled"] is True
    assert pipeline_kwargs["default_semantic_marker_logit_bias"] == pytest.approx(10000.0)
    assert pipeline_kwargs["default_semantic_marker_suppression_bias"] == pytest.approx(100.0)
    assert pipeline_kwargs["default_puzzle_room_scaffold_enabled"] is True
    assert pipeline_kwargs["default_puzzle_room_structure_enabled"] is True
    assert pipeline_kwargs["default_puzzle_room_scaffold_min_structure_tiles"] == 10
    assert pipeline_kwargs["default_puzzle_room_archetype_mode"] == "auto"
    assert pipeline_kwargs["default_puzzle_room_branch_density"] == pytest.approx(0.75)
    assert pipeline_kwargs["default_puzzle_room_block_budget"] == 28
    assert pipeline_kwargs["default_puzzle_room_preserve_route_margin"] == 0
    assert pipeline_kwargs["default_puzzle_room_switch_pocket_depth"] == 3
    assert pipeline_kwargs["default_puzzle_room_resource_bypass_offset"] == 2
    assert pipeline_kwargs["default_puzzle_room_key_pocket_depth"] == 3
    assert pipeline_kwargs["default_puzzle_room_item_slot_depth"] == 3
    assert pipeline_kwargs["default_puzzle_room_toggle_corridor_offset"] == 2
    assert pipeline_kwargs["default_puzzle_room_novelty_enabled"] is True
    assert pipeline_kwargs["default_puzzle_room_candidate_count"] == 4
    assert pipeline_kwargs["default_puzzle_room_novelty_weight"] == pytest.approx(0.45)
    assert pipeline_kwargs["default_puzzle_room_min_quality_gain"] == pytest.approx(0.5)
    assert pipeline_kwargs["default_validator_plan_max_states"] == 512
    assert pipeline_kwargs["default_deterministic_graph_marker_overlay_enabled"] is True
    assert pipeline_kwargs["default_fast_sampler_teacher_fallback_enabled"] is True
    assert pipeline_kwargs["default_masked_room_teacher_fallback_enabled"] is True
    assert pipeline_kwargs["default_masked_room_sampling_temperature"] == pytest.approx(1.0)
    assert pipeline_kwargs["default_masked_room_sampling_schedule"] == "cosine"
    assert pipeline_kwargs["default_masked_room_sampling_stochastic"] is True
    assert pipeline_kwargs["default_masked_room_corrector_steps"] == 1
    assert pipeline_kwargs["default_masked_room_corrector_mask_ratio"] == pytest.approx(0.1)


def test_root_parser_accepts_topology_comparison_subcommands(tmp_path: Path):
    parser = main._build_root_parser()

    compare_args = parser.parse_args(
        [
            "topology-compare-manual",
            "--run-dir",
            str(tmp_path / "run"),
            "--output-dir",
            str(tmp_path / "compare_out"),
            "--variants",
            "diffusion_cfg3_logic0_steps50,masked_room_full",
            "--reuse-existing-variants",
            "--semantic-role-prior-strength",
            "0.25",
            "--symbolic-max-repair-attempts",
            "7",
            "--symbolic-repair-margin",
            "3",
            "--symbolic-adjacency-threshold",
            "0.02",
            "--semantic-puzzle-offset",
            "3",
            "--no-semantic-constrained-decoding-enabled",
            "--semantic-marker-logit-bias",
            "9.5",
            "--semantic-marker-suppression-bias",
            "1.5",
            "--no-puzzle-room-scaffold-enabled",
            "--no-puzzle-room-structure-enabled",
            "--puzzle-room-scaffold-min-structure-tiles",
            "6",
            "--puzzle-room-archetype-mode",
            "hub",
            "--puzzle-room-branch-density",
            "0.5",
            "--puzzle-room-block-budget",
            "18",
            "--puzzle-room-preserve-route-margin",
            "2",
            "--puzzle-room-switch-pocket-depth",
            "4",
            "--puzzle-room-resource-bypass-offset",
            "3",
            "--puzzle-room-key-pocket-depth",
            "5",
            "--puzzle-room-item-slot-depth",
            "4",
            "--puzzle-room-toggle-corridor-offset",
            "3",
            "--no-puzzle-room-novelty-enabled",
            "--puzzle-room-candidate-count",
            "5",
            "--puzzle-room-novelty-weight",
            "1.1",
            "--puzzle-room-min-quality-gain",
            "1.4",
            "--validator-plan-max-states",
            "384",
            "--no-deterministic-graph-marker-overlay-enabled",
            "--no-fast-sampler-teacher-fallback-enabled",
            "--no-masked-room-teacher-fallback-enabled",
        ]
    )
    fixed_args = parser.parse_args(
        [
            "topology-audit-fixed-graph",
            "--run-dir",
            str(tmp_path / "run"),
            "--output-dir",
            str(tmp_path / "audit_out"),
            "--seeds",
            "1",
            "2",
            "--include-puzzle-ablations",
            "--fast-sampler-teacher-fallback-enabled",
        ]
    )

    assert compare_args.command == "topology-compare-manual"
    assert fixed_args.command == "topology-audit-fixed-graph"
    assert fixed_args.seeds == [1, 2]
    assert compare_args.variants == "diffusion_cfg3_logic0_steps50,masked_room_full"
    assert compare_args.reuse_existing_variants is True
    assert compare_args.semantic_role_prior_strength == pytest.approx(0.25)
    assert compare_args.symbolic_max_repair_attempts == 7
    assert compare_args.symbolic_repair_margin == 3
    assert compare_args.symbolic_adjacency_threshold == pytest.approx(0.02)
    assert compare_args.semantic_puzzle_offset == 3
    assert compare_args.semantic_constrained_decoding_enabled is False
    assert compare_args.semantic_marker_logit_bias == pytest.approx(9.5)
    assert compare_args.semantic_marker_suppression_bias == pytest.approx(1.5)
    assert compare_args.puzzle_room_scaffold_enabled is False
    assert compare_args.puzzle_room_structure_enabled is False
    assert compare_args.puzzle_room_scaffold_min_structure_tiles == 6
    assert compare_args.puzzle_room_archetype_mode == "hub"
    assert compare_args.puzzle_room_branch_density == pytest.approx(0.5)
    assert compare_args.puzzle_room_block_budget == 18
    assert compare_args.puzzle_room_preserve_route_margin == 2
    assert compare_args.puzzle_room_switch_pocket_depth == 4
    assert compare_args.puzzle_room_resource_bypass_offset == 3
    assert compare_args.puzzle_room_key_pocket_depth == 5
    assert compare_args.puzzle_room_item_slot_depth == 4
    assert compare_args.puzzle_room_toggle_corridor_offset == 3
    assert compare_args.puzzle_room_novelty_enabled is False
    assert compare_args.puzzle_room_candidate_count == 5
    assert compare_args.puzzle_room_novelty_weight == pytest.approx(1.1)
    assert compare_args.puzzle_room_min_quality_gain == pytest.approx(1.4)
    assert compare_args.validator_plan_max_states == 384
    assert compare_args.deterministic_graph_marker_overlay_enabled is False
    assert compare_args.fast_sampler_teacher_fallback_enabled is False
    assert compare_args.masked_room_teacher_fallback_enabled is False
    assert fixed_args.fast_sampler_teacher_fallback_enabled is True
    assert fixed_args.include_puzzle_ablations is True
    assert fixed_args.mission_graph is None


def test_train_parser_accepts_distributed_cli_flags(tmp_path: Path):
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(
        cfg_path,
        {
            "training": {"stage": "diffusion"},
        },
    )

    parser = main._build_root_parser()
    args = parser.parse_args(
        [
            "train",
            "--config",
            str(cfg_path),
            "--distributed-enabled",
            "--distributed-backend",
            "nccl",
            "--nproc-per-node",
            "2",
            "--master-port",
            "29555",
            "--cuda-visible-devices",
            "0,1",
        ]
    )

    resolved = merge_config(yaml_path=args.config, cli_overrides=cli_overrides_from_namespace(args))

    assert resolved["distributed"]["enabled"] is True
    assert resolved["distributed"]["backend"] == "nccl"
    assert resolved["distributed"]["nproc_per_node"] == 2
    assert resolved["distributed"]["master_port"] == 29555
    assert resolved["distributed"]["cuda_visible_devices"] == "0,1"


def test_validation_rejects_out_of_bounds_generation_room_coords(tmp_path: Path):
    cfg_path = tmp_path / "bad_generation.yaml"
    _write_yaml(
        cfg_path,
        {
            "generation": {
                "default_goal_coord": [99, 5],
            },
        },
    )

    with pytest.raises(ValueError, match="generation.default_goal_coord"):
        merge_config(yaml_path=str(cfg_path), cli_overrides=None)


def test_validation_rejects_diffusion_vqvae_latent_dim_mismatch(tmp_path: Path):
    cfg_path = tmp_path / "bad_latent.yaml"
    _write_yaml(
        cfg_path,
        {
            "vqvae": {"latent_dim": 64},
            "diffusion": {"latent_dim": 96},
        },
    )

    with pytest.raises(ValueError, match="diffusion.latent_dim must match vqvae.latent_dim"):
        merge_config(yaml_path=str(cfg_path), cli_overrides=None)


def test_validation_rejects_invalid_room_topology_channel_contract(tmp_path: Path):
    cfg_path = tmp_path / "bad_topology_channels.yaml"
    _write_yaml(
        cfg_path,
        {
            "diffusion": {"room_topology_channels": 49},
        },
    )

    with pytest.raises(ValueError, match="diffusion.room_topology_channels"):
        merge_config(yaml_path=str(cfg_path), cli_overrides=None)


def test_validation_rejects_fast_sampler_steps_above_diffusion_timesteps(tmp_path: Path):
    cfg_path = tmp_path / "bad_fast_sampler.yaml"
    _write_yaml(
        cfg_path,
        {
            "diffusion": {"num_timesteps": 12},
            "fast_sampler": {"num_inference_steps": 16},
        },
    )

    with pytest.raises(ValueError, match="fast_sampler.num_inference_steps"):
        merge_config(yaml_path=str(cfg_path), cli_overrides=None)


def test_resolved_config_lookup_finds_output_snapshot_from_checkpoint_path(tmp_path: Path):
    output_dir = tmp_path / "run_out"
    checkpoint_path = output_dir / "checkpoints" / "diffusion" / "best_model.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"")

    resolved = merge_config(yaml_path=None, cli_overrides={"runtime": {"output_dir": str(output_dir)}})
    with open(output_dir / "resolved_config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved, handle, sort_keys=False)

    discovered = find_resolved_config_path(checkpoint_path)
    assert discovered == output_dir / "resolved_config.yaml"

    loaded = load_resolved_config_for_artifact(checkpoint_path)
    assert loaded is not None
    assert loaded["runtime"]["output_dir"] == str(output_dir)
    assert loaded["generation"]["default_start_coord"] == [1, 5]


def test_run_training_from_args_writes_snapshots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "outputs"
    _write_yaml(
        cfg_path,
        {
            "training": {"stage": "masked_room"},
            "runtime": {"output_dir": str(out_dir)},
            "masked_room": {"epochs": 2},
        },
    )

    seen = {}

    def _fake_masked_room(config: dict) -> None:
        seen["stage"] = config["training"]["stage"]
        seen["output_dir"] = config["runtime"]["output_dir"]

    monkeypatch.setattr(main, "_run_masked_room_stage_from_config", _fake_masked_room)

    parser = main._build_root_parser()
    args = parser.parse_args(["train", "--config", str(cfg_path), "--stage", "masked_room"])
    main.run_training_from_args(args)

    assert seen["stage"] == "masked_room"
    assert Path(seen["output_dir"]) == out_dir
    assert (out_dir / "resolved_config.yaml").exists()
    assert (out_dir / "resolved_config.json").exists()
    assert (out_dir / "run_metadata.json").exists()
    assert (out_dir / "training.log").exists()

    metadata = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert "seed" in metadata
    assert "command" in metadata
    assert metadata["dataset_schema_profile"] == "zelda_v1"
    assert "num_classes=44" in metadata["dataset_schema_lock"]
    assert "room_shape=16x11" in metadata["dataset_schema_lock"]

    resolved = yaml.safe_load((out_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    assert resolved["dataset"]["schema_profile"] == "zelda_v1"


def test_run_diffusion_stage_uses_canonical_vqvae_checkpoint_when_present(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    resolved = merge_config(
        yaml_path=None,
        cli_overrides={"runtime": {"output_dir": str(tmp_path / "out")}},
    )
    checkpoint_dir = Path(resolved["vqvae"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    canonical_ckpt = checkpoint_dir / "vqvae_pretrained.pth"
    canonical_ckpt.write_bytes(b"stub")

    captured = {}

    def _fake_train_diffusion(cfg):
        captured["vqvae_checkpoint"] = cfg.vqvae_checkpoint

    monkeypatch.setattr(main, "train_diffusion", _fake_train_diffusion)
    main._run_diffusion_stage_from_config(resolved, None)

    assert Path(captured["vqvae_checkpoint"]) == canonical_ckpt


def test_diffusion_helper_preserves_yaml_only_methodology_knobs(tmp_path: Path):
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(
        cfg_path,
        {
            "training": {"stage": "diffusion"},
            "runtime": {"verbose": True},
            "dataset": {"batch_size": 2},
            "diffusion": {
                "cfg_schedule_mode": "cosine_decay",
                "cfg_schedule_power": 2.5,
                "guidance_active_fraction": 0.55,
                "min_snr_gamma": 2.5,
                "topology_conditioning_mode": "spade",
                "condition_gnn_type": "gps",
                "condition_use_reference_room_maps": True,
                "condition_reference_tile_vocab_size": 44,
                "condition_reference_embedding_dim": 24,
                "condition_reference_hidden_dim": 48,
                "logic_topology_trace_weight": 0.6,
                "logic_topology_anchor_weight": 0.3,
            },
        },
    )

    resolved = merge_config(yaml_path=str(cfg_path), cli_overrides=None)
    kwargs = diffusion_training_kwargs_from_resolved_config(resolved)

    assert kwargs["batch_size"] == 2
    assert kwargs["cfg_schedule_mode"] == "cosine_decay"
    assert kwargs["cfg_schedule_power"] == pytest.approx(2.5)
    assert kwargs["guidance_active_fraction"] == pytest.approx(0.55)
    assert kwargs["min_snr_gamma"] == pytest.approx(2.5)
    assert kwargs["topology_conditioning_mode"] == "spade"
    assert kwargs["condition_gnn_type"] == "gps"
    assert kwargs["condition_use_reference_room_maps"] is True
    assert kwargs["condition_reference_tile_vocab_size"] == 44
    assert kwargs["condition_reference_embedding_dim"] == 24
    assert kwargs["condition_reference_hidden_dim"] == 48
    assert kwargs["logic_topology_trace_weight"] == pytest.approx(0.6)
    assert kwargs["logic_topology_anchor_weight"] == pytest.approx(0.3)
    assert kwargs["seed"] == 42


def test_diffusion_vqvae_architecture_resolves_from_checkpoint_metadata(tmp_path: Path):
    ckpt = tmp_path / "vqvae_pretrained.pth"
    ckpt.write_bytes(b"stub")
    meta = {
        "format_version": "1.0",
        "model_type": "vqvae",
        "architecture": {
            "num_classes": 44,
            "latent_dim": 64,
            "hidden_dim": 96,
            "codebook_size": 256,
            "use_coordconv": False,
            "mrf_penalty_weight": 0.125,
        },
    }
    ckpt.with_suffix(".pth.meta.json").write_text(json.dumps(meta), encoding="utf-8")

    resolved = _resolve_vqvae_architecture(
        str(ckpt),
        num_classes=44,
        latent_dim=64,
        hidden_dim=128,
        codebook_size=512,
        use_coordconv=True,
        mrf_penalty_weight=0.05,
    )

    assert resolved["hidden_dim"] == 96
    assert resolved["codebook_size"] == 256
    assert resolved["use_coordconv"] is False
    assert resolved["mrf_penalty_weight"] == pytest.approx(0.125)


def test_diffusion_trainer_updates_config_to_loaded_vqvae_architecture(tmp_path: Path):
    ckpt = tmp_path / "vqvae_pretrained.pth"
    model = create_vqvae(
        num_classes=44,
        latent_dim=64,
        hidden_dim=96,
        codebook_size=512,
        use_coordconv=True,
        mrf_penalty_weight=0.05,
    )
    torch.save({"model_state_dict": model.state_dict()}, ckpt)
    meta = {
        "format_version": "1.0",
        "model_type": "vqvae",
        "architecture": {
            "num_classes": 44,
            "latent_dim": 64,
            "hidden_dim": 96,
            "codebook_size": 512,
            "use_coordconv": True,
            "mrf_penalty_weight": 0.05,
        },
    }
    ckpt.with_suffix(".pth.meta.json").write_text(json.dumps(meta), encoding="utf-8")

    trainer = object.__new__(DiffusionTrainer)
    trainer.config = DiffusionTrainingConfig(
        vqvae_checkpoint=str(ckpt),
        num_classes=44,
        latent_dim=64,
        vqvae_hidden_dim=96,
        vqvae_codebook_size=256,
        vqvae_use_coordconv=False,
        vqvae_mrf_penalty_weight=0.125,
    )

    vqvae = trainer._create_vqvae()

    assert vqvae.codebook_size == 512
    assert trainer.config.vqvae_codebook_size == 512
    assert trainer.config.vqvae_use_coordconv is True
    assert trainer.config.vqvae_mrf_penalty_weight == pytest.approx(0.05)


def test_diffusion_trainer_loads_gaussian_latent_autoencoder_from_metadata(tmp_path: Path):
    ckpt = tmp_path / "gaussian_vae_pretrained.pth"
    model = create_gaussian_vae(
        num_classes=44,
        latent_dim=64,
        hidden_dim=96,
        kl_weight=1.0,
        rare_tile_weight=5.0,
        use_coordconv=True,
        mrf_penalty_weight=0.05,
    )
    torch.save({"model_state_dict": model.state_dict()}, ckpt)
    meta = {
        "format_version": "1.0",
        "model_type": "gaussian_vae",
        "architecture": {
            "num_classes": 44,
            "latent_dim": 64,
            "hidden_dim": 96,
            "use_coordconv": True,
            "mrf_penalty_weight": 0.05,
            "kl_weight": 1.0,
            "rare_tile_weight": 5.0,
        },
    }
    ckpt.with_suffix(".pth.meta.json").write_text(json.dumps(meta), encoding="utf-8")

    trainer = object.__new__(DiffusionTrainer)
    trainer.config = DiffusionTrainingConfig(
        vqvae_checkpoint=str(ckpt),
        num_classes=44,
        latent_dim=64,
        vqvae_hidden_dim=96,
        vqvae_codebook_size=256,
        vqvae_use_coordconv=False,
        vqvae_mrf_penalty_weight=0.125,
    )

    latent_model = trainer._create_vqvae()

    assert getattr(latent_model, "model_type", None) == "gaussian_vae"
    assert trainer.config.latent_autoencoder_model_type == "gaussian_vae"
    assert trainer.config.vqvae_hidden_dim == 96
    assert trainer.config.vqvae_use_coordconv is True
    assert trainer.config.vqvae_mrf_penalty_weight == pytest.approx(0.05)

    sample = torch.zeros(1, 44, 16, 11)
    z, indices = latent_model.encode(sample)
    decoded = latent_model.decode(z, target_size=(16, 11))

    assert z.shape[0] == 1
    assert z.shape[1] == 64
    assert indices.shape[0] == 1
    assert decoded.shape[:2] == (1, 44)


def test_diffusion_validation_preview_export_writes_artifacts(tmp_path: Path):
    trainer = object.__new__(DiffusionTrainer)
    trainer.config = SimpleNamespace(
        checkpoint_dir=str(tmp_path / "checkpoints"),
        latent_autoencoder_model_type="gaussian_vae",
    )
    trainer.epoch = 2

    tile_logits = torch.zeros(1, 44, 16, 11)
    trainer._save_validation_preview(tile_logits)

    preview_root = tmp_path / "visual_previews" / "gaussian_vae" / "epoch_0003"
    txt_path = preview_root / "sample_0000.txt"
    png_path = preview_root / "sample_0000.png"

    assert txt_path.exists()
    assert txt_path.read_text(encoding="utf-8").strip()

    import importlib.util

    if importlib.util.find_spec("PIL") is None:
        assert not png_path.exists()
    else:
        assert png_path.exists()


def test_build_diffusion_training_config_from_args_preserves_yaml_only_methodology_knobs(tmp_path: Path):
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(
        cfg_path,
        {
            "training": {"stage": "diffusion"},
            "runtime": {"verbose": True},
            "dataset": {"batch_size": 2},
            "diffusion": {
                "cfg_schedule_mode": "cosine_decay",
                "cfg_schedule_power": 2.5,
                "guidance_active_fraction": 0.55,
                "min_snr_gamma": 2.5,
                "topology_conditioning_mode": "spade",
                "condition_gnn_type": "gps",
                "condition_use_reference_room_maps": True,
                "condition_reference_tile_vocab_size": 44,
                "condition_reference_embedding_dim": 24,
                "condition_reference_hidden_dim": 48,
                "logic_topology_trace_weight": 0.6,
                "logic_topology_anchor_weight": 0.3,
            },
        },
    )

    args = SimpleNamespace(
        config=str(cfg_path),
        data_dir=None,
        batch_size=None,
        room_level=None,
        epochs=None,
        lr=None,
        model_channels=None,
        context_dim=None,
        unet_channel_mult=None,
        unet_num_res_blocks=None,
        unet_attention_resolutions=None,
        unet_num_heads=None,
        unet_dropout=None,
        alpha_logic=None,
        logic_loss_mode=None,
        graph_conditioning_mode=None,
        condition_gnn_type=None,
        condition_use_reference_room_maps=None,
        condition_reference_tile_vocab_size=None,
        condition_reference_embedding_dim=None,
        condition_reference_hidden_dim=None,
        vqvae_hidden_dim=None,
        vqvae_codebook_size=None,
        vqvae_use_coordconv=None,
        vqvae_mrf_penalty_weight=None,
        topology_refinement_mode=None,
        attention_mode=None,
        topology_conditioning_mode=None,
        hedgehog_feature_dim=None,
        graph_auto_linear_attention_nodes=None,
        spatial_graph_gate_init=None,
        spatial_topology_gate_init=None,
        use_teacher_forced_neighbor_latents=None,
        use_current_node_distance_features=None,
        current_node_distance_max=None,
        logic_topology_trace_weight=None,
        logic_topology_anchor_weight=None,
        guidance_scale=None,
        checkpoint_dir=None,
        vqvae_checkpoint=None,
        device=None,
        distributed_enabled=None,
        distributed_backend=None,
        quick=None,
        verbose=None,
    )
    config = build_diffusion_training_config_from_args(args)

    assert config.batch_size == 2
    assert config.cfg_schedule_mode == "cosine_decay"
    assert config.cfg_schedule_power == pytest.approx(2.5)
    assert config.guidance_active_fraction == pytest.approx(0.55)
    assert config.min_snr_gamma == pytest.approx(2.5)
    assert config.topology_conditioning_mode == "spade"
    assert config.condition_gnn_type == "gps"
    assert config.condition_use_reference_room_maps is True
    assert config.condition_reference_tile_vocab_size == 44
    assert config.condition_reference_embedding_dim == 24
    assert config.condition_reference_hidden_dim == 48
    assert config.logic_topology_trace_weight == pytest.approx(0.6)
    assert config.logic_topology_anchor_weight == pytest.approx(0.3)
    assert config.seed == 42
    assert args.verbose is True


def test_masked_room_stage_derives_num_classes_and_latent_dim_from_shared_schema(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(
        cfg_path,
        {
            "training": {"stage": "masked_room"},
            "vqvae": {"latent_dim": 96},
            "diffusion": {"latent_dim": 96},
            "masked_room": {
                "epochs": 2,
                "validation_fraction": 0.2,
                "validation_max_batches": 5,
                "best_checkpoint_metric": "val_topology_focus_loss",
            },
        },
    )

    resolved = merge_config(yaml_path=str(cfg_path), cli_overrides=None)
    kwargs = masked_room_training_kwargs_from_resolved_config(resolved)
    assert kwargs["num_classes"] == 44
    assert kwargs["latent_dim"] == 96
    assert kwargs["validation_fraction"] == pytest.approx(0.2)
    assert kwargs["validation_max_batches"] == 5
    assert kwargs["best_checkpoint_metric"] == "val_topology_focus_loss"
    assert kwargs["seed"] == 42

    captured = {}

    def _fake_train_masked_room(config):
        captured["config"] = config

    monkeypatch.setattr(main, "train_masked_room", _fake_train_masked_room)
    main._run_masked_room_stage_from_config(resolved)

    assert captured["config"].num_classes == 44
    assert captured["config"].latent_dim == 96
    assert captured["config"].best_checkpoint_metric == "val_topology_focus_loss"


def test_masked_room_helper_preserves_yaml_only_unet_and_mask_schedule_knobs(tmp_path: Path):
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(
        cfg_path,
        {
            "training": {"stage": "masked_room"},
            "masked_room": {
                "attention_mode": "linear_hedgehog",
                "topology_conditioning_mode": "spade",
                "condition_use_reference_room_maps": True,
                "condition_reference_tile_vocab_size": 44,
                "condition_reference_embedding_dim": 20,
                "condition_reference_hidden_dim": 40,
                "hedgehog_feature_dim": 48,
                "graph_auto_linear_attention_nodes": 96,
                "spatial_graph_gate_init": -1.25,
                "spatial_topology_gate_init": -0.75,
                "unet_channel_mult": [1, 2],
                "unet_num_res_blocks": 3,
                "unet_attention_resolutions": [0, 1],
                "unet_num_heads": 4,
                "unet_dropout": 0.2,
                "min_mask_ratio": 0.2,
                "max_mask_ratio": 0.65,
                "validation_fraction": 0.2,
                "validation_max_batches": 7,
                "best_checkpoint_metric": "val_topology_focus_loss",
            },
        },
    )

    resolved = merge_config(yaml_path=str(cfg_path), cli_overrides=None)
    kwargs = masked_room_training_kwargs_from_resolved_config(resolved)

    assert kwargs["attention_mode"] == "linear_hedgehog"
    assert kwargs["topology_conditioning_mode"] == "spade"
    assert kwargs["condition_use_reference_room_maps"] is True
    assert kwargs["condition_reference_tile_vocab_size"] == 44
    assert kwargs["condition_reference_embedding_dim"] == 20
    assert kwargs["condition_reference_hidden_dim"] == 40
    assert kwargs["hedgehog_feature_dim"] == 48
    assert kwargs["graph_auto_linear_attention_nodes"] == 96
    assert kwargs["spatial_graph_gate_init"] == pytest.approx(-1.25)
    assert kwargs["spatial_topology_gate_init"] == pytest.approx(-0.75)
    assert kwargs["unet_channel_mult"] == (1, 2)
    assert kwargs["unet_num_res_blocks"] == 3
    assert kwargs["unet_attention_resolutions"] == (0, 1)
    assert kwargs["unet_num_heads"] == 4
    assert kwargs["unet_dropout"] == pytest.approx(0.2)
    assert kwargs["min_mask_ratio"] == pytest.approx(0.2)
    assert kwargs["max_mask_ratio"] == pytest.approx(0.65)
    assert kwargs["validation_fraction"] == pytest.approx(0.2)
    assert kwargs["validation_max_batches"] == 7
    assert kwargs["best_checkpoint_metric"] == "val_topology_focus_loss"


def test_reference_room_vocab_size_must_match_dataset_schema_when_enabled(tmp_path: Path):
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(
        cfg_path,
        {
            "diffusion": {
                "condition_use_reference_room_maps": True,
                "condition_reference_tile_vocab_size": 43,
            },
        },
    )

    with pytest.raises(ValueError, match="condition_reference_tile_vocab_size"):
        merge_config(yaml_path=str(cfg_path), cli_overrides=None)


def test_topology_helper_preserves_yaml_generation_knobs(tmp_path: Path):
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(
        cfg_path,
        {
            "topology": {
                "default_target_curve": [0.1, 0.3, 0.55, 0.85],
                "num_rooms": 11,
                "population_size": 24,
                "generations": 12,
                "mutation_rate": 0.22,
                "crossover_rate": 0.6,
                "genome_length": 17,
                "rule_space": "full",
                "transition_mix": 0.5,
                "search_strategy": "cvt_emitter",
                "qd_archive_cells": 160,
                "qd_init_random_fraction": 0.25,
                "qd_emitter_mutation_rate": 0.27,
                "max_lock_key_rules": 2,
                "enable_rule_credit_assignment": True,
                "enforce_generation_constraints": True,
                "allow_candidate_repairs": True,
            },
        },
    )

    resolved = merge_config(yaml_path=str(cfg_path), cli_overrides=None)
    kwargs = topology_generation_kwargs_from_resolved_config(resolved)

    assert kwargs["target_curve"] == [0.1, 0.3, 0.55, 0.85]
    assert kwargs["num_rooms"] == 11
    assert kwargs["population_size"] == 24
    assert kwargs["generations"] == 12
    assert kwargs["mutation_rate"] == pytest.approx(0.22)
    assert kwargs["crossover_rate"] == pytest.approx(0.6)
    assert kwargs["genome_length"] == 17
    assert kwargs["rule_space"] == "full"
    assert kwargs["transition_mix"] == pytest.approx(0.5)
    assert kwargs["search_strategy"] == "cvt_emitter"
    assert kwargs["qd_archive_cells"] == 160
    assert kwargs["qd_init_random_fraction"] == pytest.approx(0.25)
    assert kwargs["qd_emitter_mutation_rate"] == pytest.approx(0.27)
    assert kwargs["max_lock_key_rules"] == 2
    assert kwargs["enable_rule_credit_assignment"] is True
    assert kwargs["enforce_generation_constraints"] is True
    assert kwargs["allow_candidate_repairs"] is True


def test_fast_sampler_stage_inherits_shared_yaml_runtime_and_dataset_settings(tmp_path: Path):
    cfg_path = tmp_path / "config.yaml"
    base_ckpt = tmp_path / "teacher.pth"
    base_ckpt.write_bytes(b"stub")
    _write_yaml(
        cfg_path,
        {
            "runtime": {"verbose": True, "device": "cpu", "seed": 321},
            "dataset": {
                "batch_size": 3,
                "num_workers": 2,
                "use_vglc": False,
            },
            "fast_sampler": {
                "base_diffusion_checkpoint": str(base_ckpt),
                "epochs": 7,
                "num_inference_steps": 6,
                "lora_rank": 4,
                "lora_alpha": 12.0,
                "decode_alignment_weight": 0.4,
                "topology_alignment_weight": 0.3,
                "topology_marker_weight": 2.5,
                "topology_trace_weight": 0.9,
                "topology_focus_dilation": 2,
                "validation_fraction": 0.2,
                "validation_max_batches": 5,
                "best_checkpoint_metric": "val_topology_decode_ce_loss",
            },
        },
    )

    resolved = merge_config(yaml_path=str(cfg_path), cli_overrides=None)
    kwargs = fast_sampler_training_kwargs_from_resolved_config(resolved)

    assert kwargs["base_diffusion_checkpoint"] == str(base_ckpt)
    assert kwargs["batch_size"] == 3
    assert kwargs["num_workers"] == 2
    assert kwargs["epochs"] == 7
    assert kwargs["num_inference_steps"] == 6
    assert kwargs["lora_rank"] == 4
    assert kwargs["lora_alpha"] == pytest.approx(12.0)
    assert kwargs["decode_alignment_weight"] == pytest.approx(0.4)
    assert kwargs["topology_alignment_weight"] == pytest.approx(0.3)
    assert kwargs["topology_marker_weight"] == pytest.approx(2.5)
    assert kwargs["topology_trace_weight"] == pytest.approx(0.9)
    assert kwargs["topology_focus_dilation"] == 2
    assert kwargs["validation_fraction"] == pytest.approx(0.2)
    assert kwargs["validation_max_batches"] == 5
    assert kwargs["best_checkpoint_metric"] == "val_topology_decode_ce_loss"
    assert kwargs["device"] == "cpu"
    assert kwargs["seed"] == 321
    assert kwargs["use_vglc"] is False

    args = build_fast_sampler_training_config_from_args(SimpleNamespace(config=str(cfg_path), verbose=None))

    assert args.base_diffusion_checkpoint == str(base_ckpt)
    assert args.batch_size == 3
    assert args.num_workers == 2
    assert args.epochs == 7
    assert args.num_inference_steps == 6
    assert args.lora_rank == 4
    assert args.lora_alpha == pytest.approx(12.0)
    assert args.decode_alignment_weight == pytest.approx(0.4)
    assert args.topology_alignment_weight == pytest.approx(0.3)
    assert args.topology_marker_weight == pytest.approx(2.5)
    assert args.topology_trace_weight == pytest.approx(0.9)
    assert args.topology_focus_dilation == 2
    assert args.validation_fraction == pytest.approx(0.2)
    assert args.validation_max_batches == 5
    assert args.best_checkpoint_metric == "val_topology_decode_ce_loss"
    assert args.device == "cpu"
    assert args.seed == 321
    assert args.use_vglc is False

    captured = {}

    def _fake_train_fast_sampler(config):
        captured["config"] = config

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(main, "train_fast_sampler", _fake_train_fast_sampler)
    try:
        main._run_fast_sampler_stage_from_config(resolved)
    finally:
        monkeypatch.undo()

    assert captured["config"].use_vglc is False


def test_vqvae_stage_inherits_shared_yaml_runtime_and_dataset_settings(tmp_path: Path):
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(
        cfg_path,
        {
            "runtime": {"verbose": True, "device": "cpu", "seed": 123, "quick": True},
            "dataset": {
                "batch_size": 2,
                "min_samples_per_epoch": 96,
                "normalize": False,
                "room_level": True,
            },
            "vqvae": {
                "epochs": 12,
                "latent_dim": 96,
                "hidden_dim": 160,
                "use_coordconv": False,
                "mrf_penalty_weight": 0.2,
            },
            "diffusion": {
                "latent_dim": 96,
            },
        },
    )

    resolved = merge_config(yaml_path=str(cfg_path), cli_overrides=None)
    kwargs = vqvae_training_kwargs_from_resolved_config(resolved)

    assert kwargs["batch_size"] == 2
    assert kwargs["min_samples_per_epoch"] == 96
    assert kwargs["epochs"] == 12
    assert kwargs["latent_dim"] == 96
    assert kwargs["hidden_dim"] == 160
    assert kwargs["use_coordconv"] is False
    assert kwargs["mrf_penalty_weight"] == pytest.approx(0.2)
    assert kwargs["validation_fraction"] == pytest.approx(0.1)
    assert kwargs["validation_max_batches"] == 16
    assert kwargs["best_checkpoint_metric"] == "val_loss"
    assert kwargs["device"] == "cpu"
    assert kwargs["seed"] == 123
    assert kwargs["verbose"] is True
    assert kwargs["normalize"] is False
    assert kwargs["room_level"] is True
    assert kwargs["quick"] is True

    args = build_vqvae_training_args_from_args(SimpleNamespace(config=str(cfg_path)))

    assert args.batch_size == 2
    assert args.min_samples_per_epoch == 96
    assert args.epochs == 12
    assert args.latent_dim == 96
    assert args.hidden_dim == 160
    assert args.use_coordconv is False
    assert args.mrf_penalty_weight == pytest.approx(0.2)
    assert args.validation_fraction == pytest.approx(0.1)
    assert args.validation_max_batches == 16
    assert args.best_checkpoint_metric == "val_loss"
    assert args.device == "cpu"
    assert args.seed == 123
    assert args.verbose is True
    assert args.normalize is False
    assert args.room_level is True
    assert args.quick is True

    captured = {}

    def _fake_train_vqvae(namespace):
        captured["args"] = namespace

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(main, "train_vqvae", _fake_train_vqvae)
    try:
        main._run_vqvae_stage_from_config(resolved)
    finally:
        monkeypatch.undo()

    assert captured["args"].normalize is False
    assert captured["args"].room_level is True
    assert captured["args"].quick is True


def test_src_train_is_a_compatibility_wrapper(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def _fake_main(argv=None):
        captured["argv"] = argv

    monkeypatch.setattr(legacy_train.root_main, "main", _fake_main)
    legacy_train.main(["train", "--config", "configs/zelda_hmolqd.yaml", "--stage", "diffusion"])

    assert captured["argv"] == ["train", "--config", "configs/zelda_hmolqd.yaml", "--stage", "diffusion"]
