from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import main
import src.train as legacy_train
from src.config_system import cli_overrides_from_namespace, merge_config
from src.train_diffusion import build_diffusion_training_config_from_args, diffusion_training_kwargs_from_resolved_config
from src.train_masked_room import masked_room_training_kwargs_from_resolved_config


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
    assert kwargs["logic_topology_trace_weight"] == pytest.approx(0.6)
    assert kwargs["logic_topology_anchor_weight"] == pytest.approx(0.3)

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
    assert config.logic_topology_trace_weight == pytest.approx(0.6)
    assert config.logic_topology_anchor_weight == pytest.approx(0.3)
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
            "masked_room": {"epochs": 2},
        },
    )

    resolved = merge_config(yaml_path=str(cfg_path), cli_overrides=None)
    kwargs = masked_room_training_kwargs_from_resolved_config(resolved)
    assert kwargs["num_classes"] == 44
    assert kwargs["latent_dim"] == 96

    captured = {}

    def _fake_train_masked_room(config):
        captured["config"] = config

    monkeypatch.setattr(main, "train_masked_room", _fake_train_masked_room)
    main._run_masked_room_stage_from_config(resolved)

    assert captured["config"].num_classes == 44
    assert captured["config"].latent_dim == 96


def test_src_train_is_a_compatibility_wrapper(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def _fake_main(argv=None):
        captured["argv"] = argv

    monkeypatch.setattr(legacy_train.root_main, "main", _fake_main)
    legacy_train.main(["train", "--config", "configs/zelda_hmolqd.yaml", "--stage", "diffusion"])

    assert captured["argv"] == ["train", "--config", "configs/zelda_hmolqd.yaml", "--stage", "diffusion"]
