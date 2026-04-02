from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import main
import src.train as legacy_train
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
    _resolve_vqvae_architecture,
    build_diffusion_training_config_from_args,
    diffusion_training_kwargs_from_resolved_config,
)
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
    assert resolved["vqvae"]["keep_last"] == 2
    assert resolved["diffusion"]["keep_last"] == 2
    assert resolved["fast_sampler"]["keep_last"] == 2
    assert resolved["masked_room"]["keep_last"] == 2
    assert resolved["diffusion"]["model_channels"] == 96
    assert resolved["diffusion"]["condition_hidden_dim"] == 192
    assert resolved["diffusion"]["condition_num_gnn_layers"] == 2
    assert resolved["diffusion"]["condition_gnn_type"] == "gps"
    assert resolved["diffusion"]["condition_use_reference_room_maps"] is True
    assert resolved["masked_room"]["model_channels"] == 96
    assert resolved["masked_room"]["condition_hidden_dim"] == 192
    assert resolved["masked_room"]["condition_num_gnn_layers"] == 2
    assert resolved["masked_room"]["condition_use_reference_room_maps"] is True
    assert resolved["generation"]["guidance_scale"] == pytest.approx(7.5)
    assert resolved["generation"]["logic_guidance_scale"] == pytest.approx(1.0)
    assert resolved["generation"]["num_diffusion_steps"] == 50
    assert resolved["generation"]["apply_repair"] is True
    assert resolved["generation"]["enable_map_elites"] is False


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
    assert masked_room_kwargs["keep_last"] == 2
    assert masked_room_kwargs["auto_resume"] is True
    assert masked_room_kwargs["resume_checkpoint"] is None
    assert masked_room_kwargs["checkpoint_storage_budget_gb"] is None
    assert vqvae_kwargs["keep_last"] == 2
    assert vqvae_kwargs["auto_resume"] is True
    assert vqvae_kwargs["checkpoint_storage_budget_gb"] is None
    assert vqvae_kwargs["resume"] is None
    assert diffusion_kwargs["vqvae_hidden_dim"] == resolved["vqvae"]["hidden_dim"]
    assert diffusion_kwargs["vqvae_codebook_size"] == resolved["vqvae"]["codebook_size"]
    assert diffusion_kwargs["vqvae_use_coordconv"] == resolved["vqvae"]["use_coordconv"]
    assert diffusion_kwargs["vqvae_mrf_penalty_weight"] == pytest.approx(resolved["vqvae"]["mrf_penalty_weight"])

    generation_kwargs = generation_runtime_kwargs_from_resolved_config(resolved)
    pipeline_kwargs = pipeline_kwargs_from_resolved_config(resolved)
    assert generation_kwargs["default_guidance_scale"] == pytest.approx(7.5)
    assert generation_kwargs["default_logic_guidance_scale"] == pytest.approx(1.0)
    assert generation_kwargs["default_num_diffusion_steps"] == 50
    assert generation_kwargs["default_start_goal_coords"] == ((1, 5), (14, 5))
    assert pipeline_kwargs["condition_gnn_type"] == resolved["diffusion"]["condition_gnn_type"]
    assert pipeline_kwargs["fast_sampling_steps"] == resolved["fast_sampler"]["num_inference_steps"]
    assert pipeline_kwargs["diffusion_fallback_config"]["model_channels"] == resolved["diffusion"]["model_channels"]
    assert pipeline_kwargs["condition_encoder_fallback_config"]["context_dim"] == resolved["diffusion"]["context_dim"]
    assert pipeline_kwargs["masked_room_fallback_config"]["model_channels"] == resolved["masked_room"]["model_channels"]


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
            "masked_room": {"epochs": 2},
        },
    )

    resolved = merge_config(yaml_path=str(cfg_path), cli_overrides=None)
    kwargs = masked_room_training_kwargs_from_resolved_config(resolved)
    assert kwargs["num_classes"] == 44
    assert kwargs["latent_dim"] == 96
    assert kwargs["seed"] == 42

    captured = {}

    def _fake_train_masked_room(config):
        captured["config"] = config

    monkeypatch.setattr(main, "train_masked_room", _fake_train_masked_room)
    main._run_masked_room_stage_from_config(resolved)

    assert captured["config"].num_classes == 44
    assert captured["config"].latent_dim == 96


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
