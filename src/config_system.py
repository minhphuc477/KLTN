"""
Experiment configuration loading, validation, and reproducibility utilities.

Resolution order:
    defaults -> YAML file -> CLI overrides
"""

from __future__ import annotations

import copy
import json
import logging
import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import yaml

from src.core.definitions import ROOM_HEIGHT, ROOM_WIDTH, SEMANTIC_PALETTE

try:
    import torch
except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
    torch = None


logger = logging.getLogger(__name__)

DEFAULT_DATASET_SCHEMA_PROFILE = "zelda_v1"
DATASET_SCHEMA_PROFILES: Dict[str, Dict[str, int]] = {
    DEFAULT_DATASET_SCHEMA_PROFILE: {
        "num_classes": int(max(int(v) for v in SEMANTIC_PALETTE.values()) + 1),
        "room_height": int(ROOM_HEIGHT),
        "room_width": int(ROOM_WIDTH),
        "node_feature_dim": 6,
        "edge_feature_dim": 8,
        "tpe_dim": 8,
    },
}


def get_dataset_schema_requirements(schema_profile: str = DEFAULT_DATASET_SCHEMA_PROFILE) -> Dict[str, int]:
    """Return the locked dataset/schema contract for the requested profile."""
    profile = str(schema_profile).strip().lower()
    if profile not in DATASET_SCHEMA_PROFILES:
        raise ValueError(
            f"Unsupported dataset.schema_profile={schema_profile!r}. "
            f"Expected one of {tuple(DATASET_SCHEMA_PROFILES.keys())!r}."
        )
    return copy.deepcopy(DATASET_SCHEMA_PROFILES[profile])


def dataset_schema_lock_summary(schema_profile: str = DEFAULT_DATASET_SCHEMA_PROFILE) -> str:
    """Human-readable schema lock summary for logs/errors/metadata."""
    req = get_dataset_schema_requirements(schema_profile)
    return (
        f"{schema_profile}: num_classes={req['num_classes']}, "
        f"room_shape={req['room_height']}x{req['room_width']}, "
        f"node_feature_dim={req['node_feature_dim']}, "
        f"edge_feature_dim={req['edge_feature_dim']}, "
        f"tpe_dim={req['tpe_dim']}"
    )


@dataclass(frozen=True)
class ConfigField:
    path: str
    field_type: type
    default: Any
    help: str
    sequence_item_type: Optional[type] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: Optional[Iterable[Any]] = None
    allow_none: bool = False
    cli: Optional[str] = None


CONFIG_FIELDS: List[ConfigField] = [
    ConfigField("training.stage", str, "all", "Training stage selector.", choices=("all", "vqvae", "diffusion", "fast_sampler", "masked_room")),
    ConfigField("runtime.experiment_name", str, "zelda_hmolqd", "Human-readable experiment name."),
    ConfigField("runtime.output_dir", str, "outputs/zelda_hmolqd", "Output directory for checkpoints, logs, and config snapshots."),
    ConfigField("runtime.log_file", str, "training.log", "Training log filename written inside output_dir."),
    ConfigField("runtime.device", str, "auto", "Execution device.", choices=("auto", "cuda", "cpu")),
    ConfigField("runtime.seed", int, 42, "Global random seed.", min_value=0),
    ConfigField("runtime.verbose", bool, False, "Enable verbose logging."),
    ConfigField("runtime.quick", bool, False, "Shorten training for smoke tests."),
    ConfigField("runtime.resume", str, None, "Optional checkpoint to resume from.", allow_none=True),
    ConfigField("distributed.enabled", bool, False, "Enable distributed launch metadata and environment setup.", cli="distributed_enabled"),
    ConfigField("distributed.backend", str, "nccl", "torch.distributed backend.", choices=("nccl", "gloo"), cli="distributed_backend"),
    ConfigField("distributed.nproc_per_node", int, 1, "Processes per node for distributed launch.", min_value=1, cli="nproc_per_node"),
    ConfigField("distributed.master_port", int, 29500, "Master port for torch.distributed.", min_value=1024, max_value=65535, cli="master_port"),
    ConfigField("distributed.cuda_visible_devices", str, "", "Comma-separated CUDA device list. Empty keeps current environment.", cli="cuda_visible_devices"),
    ConfigField("distributed.find_unused_parameters", bool, False, "DDP find_unused_parameters flag.", cli="find_unused_parameters"),
    ConfigField("dataset.data_dir", str, "Data/The Legend of Zelda", "Dataset root directory."),
    ConfigField("dataset.schema_profile", str, DEFAULT_DATASET_SCHEMA_PROFILE, "Named dataset/schema contract for this repository.", choices=tuple(DATASET_SCHEMA_PROFILES.keys())),
    ConfigField("dataset.batch_size", int, 4, "Mini-batch size.", min_value=1),
    ConfigField("dataset.num_workers", int, 0, "DataLoader worker count.", min_value=0),
    ConfigField("dataset.pin_memory", bool, True, "Enable DataLoader pin_memory."),
    ConfigField("dataset.drop_last", bool, True, "Drop incomplete final batch."),
    ConfigField("dataset.shuffle_train", bool, True, "Shuffle the training loader."),
    ConfigField("dataset.shuffle_val", bool, False, "Shuffle the validation loader."),
    ConfigField("dataset.use_vglc", bool, True, "Use VGLC-format dataset adapter."),
    ConfigField("dataset.normalize", bool, True, "Normalize room grids to [0,1]."),
    ConfigField("dataset.room_level", bool, True, "Train on individual rooms instead of stitched dungeons."),
    ConfigField("dataset.load_graphs", bool, True, "Load graph side-information from dataset."),
    ConfigField("dataset.min_samples_per_epoch", int, 64, "Minimum effective VQ-VAE samples per epoch.", min_value=1),
    ConfigField("dataset.num_classes", int, 44, "Semantic tile vocabulary size.", min_value=1),
    ConfigField("dataset.room_height", int, 16, "Supported room height.", min_value=1),
    ConfigField("dataset.room_width", int, 11, "Supported room width.", min_value=1),
    ConfigField("dataset.node_feature_dim", int, 6, "Supported graph node-feature width.", min_value=1),
    ConfigField("dataset.edge_feature_dim", int, 8, "Supported graph edge-feature width.", min_value=1),
    ConfigField("dataset.tpe_dim", int, 8, "Supported topological positional encoding width.", min_value=1),
    ConfigField("vqvae.checkpoint_dir", str, "", "Directory for VQ-VAE checkpoints. Empty means output_dir/checkpoints/vqvae."),
    ConfigField("vqvae.resume_checkpoint", str, None, "Optional VQ-VAE checkpoint to resume.", allow_none=True),
    ConfigField("vqvae.epochs", int, 300, "VQ-VAE training epochs.", min_value=1),
    ConfigField("vqvae.learning_rate", float, 3e-4, "VQ-VAE learning rate.", min_value=1e-8),
    ConfigField("vqvae.weight_decay", float, 1e-5, "VQ-VAE optimizer weight decay.", min_value=0.0),
    ConfigField("vqvae.grad_clip_norm", float, 1.0, "VQ-VAE gradient clipping norm.", min_value=0.0),
    ConfigField("vqvae.save_every", int, 50, "VQ-VAE checkpoint interval.", min_value=1),
    ConfigField("vqvae.latent_dim", int, 64, "VQ-VAE latent width.", min_value=1),
    ConfigField("vqvae.hidden_dim", int, 128, "VQ-VAE base channel width.", min_value=8),
    ConfigField("vqvae.codebook_size", int, 512, "VQ-VAE codebook size.", min_value=8),
    ConfigField("vqvae.commitment_cost", float, 0.25, "VQ-VAE commitment loss weight.", min_value=0.0),
    ConfigField("vqvae.rare_tile_weight", float, 5.0, "Rare-tile reconstruction reweighting.", min_value=1.0),
    ConfigField("vqvae.use_ema", bool, True, "Use EMA VQ codebook updates."),
    ConfigField("vqvae.use_coordconv", bool, True, "Use CoordConv in VQ-VAE encoder."),
    ConfigField("vqvae.mrf_penalty_weight", float, 0.05, "Illegal-adjacency penalty coefficient.", min_value=0.0),
    ConfigField("diffusion.checkpoint_dir", str, "", "Directory for diffusion checkpoints. Empty means output_dir/checkpoints/diffusion."),
    ConfigField("diffusion.vqvae_checkpoint", str, None, "Frozen VQ-VAE checkpoint for diffusion.", allow_none=True),
    ConfigField("diffusion.epochs", int, 100, "Diffusion training epochs.", min_value=1),
    ConfigField("diffusion.learning_rate", float, 1e-4, "Diffusion optimizer learning rate.", min_value=1e-8),
    ConfigField("diffusion.optimizer_weight_decay", float, 1e-5, "Diffusion optimizer weight decay.", min_value=0.0),
    ConfigField("diffusion.grad_clip_norm", float, 1.0, "Diffusion gradient clipping norm.", min_value=0.0),
    ConfigField("diffusion.save_every", int, 10, "Diffusion checkpoint interval.", min_value=1),
    ConfigField("diffusion.latent_dim", int, 64, "Diffusion latent width.", min_value=1),
    ConfigField("diffusion.model_channels", int, 128, "Diffusion U-Net base channels.", min_value=8),
    ConfigField("diffusion.context_dim", int, 256, "Conditioning context width.", min_value=8),
    ConfigField("diffusion.unet_channel_mult", list, [1, 2, 4], "Per-level U-Net channel multipliers.", sequence_item_type=int, min_value=1),
    ConfigField("diffusion.unet_num_res_blocks", int, 2, "Residual blocks per U-Net level.", min_value=1),
    ConfigField("diffusion.unet_attention_resolutions", list, [1, 2], "U-Net level indices that enable attention.", sequence_item_type=int, min_value=0),
    ConfigField("diffusion.unet_num_heads", int, 8, "U-Net attention head count.", min_value=1),
    ConfigField("diffusion.unet_dropout", float, 0.1, "U-Net residual/attention dropout.", min_value=0.0, max_value=1.0),
    ConfigField("diffusion.condition_hidden_dim", int, 256, "Condition-encoder hidden width.", min_value=8),
    ConfigField("diffusion.condition_num_gnn_layers", int, 3, "Condition-encoder GNN depth.", min_value=1),
    ConfigField("diffusion.condition_num_attention_heads", int, 8, "Condition-encoder fusion heads.", min_value=1),
    ConfigField("diffusion.condition_dropout", float, 0.1, "Condition-encoder dropout.", min_value=0.0, max_value=1.0),
    ConfigField("diffusion.condition_gnn_type", str, "gcn", "Condition-encoder graph backbone.", choices=("gcn", "gat", "sage", "gps")),
    ConfigField("diffusion.graph_conditioning_mode", str, "node_sequence", "Graph-conditioning representation.", choices=("node_sequence", "pooled")),
    ConfigField("diffusion.num_timesteps", int, 1000, "Forward diffusion timesteps.", min_value=10),
    ConfigField("diffusion.schedule_type", str, "cosine", "Diffusion noise schedule.", choices=("linear", "cosine")),
    ConfigField("diffusion.topology_refinement_mode", str, "gat2", "Topology refinement inside attention.", choices=("none", "lightweight", "gat2", "upgraded")),
    ConfigField("diffusion.attention_mode", str, "softmax", "Attention kernel.", choices=("softmax", "linear_hedgehog")),
    ConfigField("diffusion.topology_conditioning_mode", str, "additive", "Room-topology conditioning path.", choices=("additive", "spade")),
    ConfigField("diffusion.hedgehog_feature_dim", int, 32, "Linear-attention feature width.", min_value=4),
    ConfigField("diffusion.graph_auto_linear_attention_nodes", int, 128, "Switch graph-to-grid attention to linear mode above this node count. 0 disables the auto-switch.", min_value=0),
    ConfigField("diffusion.spatial_graph_gate_init", float, -2.0, "Initial logit for graph-conditioning gate."),
    ConfigField("diffusion.spatial_topology_gate_init", float, -2.0, "Initial logit for room-topology gate."),
    ConfigField("diffusion.use_teacher_forced_neighbor_latents", bool, True, "Use real adjacent room maps during room-level diffusion training to encode neighbor latents."),
    ConfigField("diffusion.use_current_node_distance_features", bool, True, "Inject current-room distance features into Block III/IV graph conditioning."),
    ConfigField("diffusion.current_node_distance_max", int, 8, "Distance clip used when normalizing current-room graph distances.", min_value=1),
    ConfigField("diffusion.room_topology_channels", int, 18, "Room-topology conditioning channel count.", min_value=1),
    ConfigField("diffusion.cfg_dropout_prob", float, 0.1, "Classifier-free conditioning dropout.", min_value=0.0, max_value=1.0),
    ConfigField("diffusion.cfg_scale", float, 3.0, "Classifier-free guidance scale.", min_value=0.0),
    ConfigField("diffusion.cfg_schedule_mode", str, "constant", "Classifier-free guidance schedule.", choices=("constant", "linear_decay", "cosine_decay")),
    ConfigField("diffusion.cfg_schedule_min_scale", float, 1.0, "Minimum classifier-free guidance scale.", min_value=0.0),
    ConfigField("diffusion.cfg_schedule_power", float, 1.0, "Classifier-free guidance schedule power.", min_value=1e-6),
    ConfigField("diffusion.prediction_type", str, "epsilon", "Diffusion target parameterization.", choices=("epsilon", "v")),
    ConfigField("diffusion.min_snr_gamma", float, 5.0, "Min-SNR-gamma training weight.", min_value=0.0),
    ConfigField("diffusion.num_logic_iterations", int, 30, "LogicNet message-passing iterations.", min_value=1),
    ConfigField("diffusion.logic_topology_trace_weight", float, 0.25, "Additional LogicNet weight on room-topology traversability traces.", min_value=0.0),
    ConfigField("diffusion.logic_topology_anchor_weight", float, 0.25, "Additional LogicNet weight on start/goal/door anchor walkability.", min_value=0.0),
    ConfigField("diffusion.guidance_scale", float, 1.0, "Logic guidance scale.", min_value=0.0),
    ConfigField("diffusion.guidance_clamp_magnitude", float, 1.0, "Logic-guidance gradient clamp magnitude.", min_value=0.0),
    ConfigField("diffusion.guidance_relative_norm_cap", float, 0.25, "Relative guidance norm cap.", min_value=0.0),
    ConfigField("diffusion.guidance_schedule_enabled", bool, True, "Enable timestep-decayed LogicNet guidance."),
    ConfigField("diffusion.guidance_active_fraction", float, 0.30, "Active reverse-process fraction for LogicNet guidance.", min_value=0.05, max_value=1.0),
    ConfigField("diffusion.guidance_decay_power", float, 1.0, "Logic-guidance decay power.", min_value=0.25),
    ConfigField("diffusion.guidance_max_graph_nodes", int, 512, "Maximum graph nodes allowed for LogicNet guidance.", min_value=1),
    ConfigField("diffusion.guidance_max_key_lock_pairs", int, 2048, "Maximum key-lock pairs passed into LogicNet guidance.", min_value=0),
    ConfigField("diffusion.guidance_max_guidance_elements", int, 2_000_000, "Maximum latent elements allowed for autograd guidance.", min_value=1),
    ConfigField("diffusion.alpha_visual", float, 1.0, "Diffusion reconstruction loss coefficient.", min_value=0.0),
    ConfigField("diffusion.alpha_logic", float, 0.1, "Logic regularization coefficient.", min_value=0.0),
    ConfigField("diffusion.logic_loss_mode", str, "predicted_latent", "Logic loss target mode.", choices=("predicted_latent", "detached_real")),
    ConfigField("diffusion.warmup_epochs", int, 5, "Epochs before enabling logic loss.", min_value=0),
    ConfigField("diffusion.scheduler_t0", int, 10, "CosineWarmRestarts T_0.", min_value=1),
    ConfigField("diffusion.scheduler_t_mult", int, 2, "CosineWarmRestarts T_mult.", min_value=1),
    ConfigField("diffusion.scheduler_eta_min", float, 1e-6, "CosineWarmRestarts eta_min.", min_value=0.0),
    ConfigField("diffusion.ema_decay", float, 0.9999, "EMA decay for diffusion weights.", min_value=0.0, max_value=0.999999),
    ConfigField("diffusion.validation_num_samples", int, 4, "Validation sample count.", min_value=1),
    ConfigField("fast_sampler.checkpoint_dir", str, "", "Directory for fast-sampler checkpoints. Empty means output_dir/checkpoints/fast_sampler."),
    ConfigField("fast_sampler.base_diffusion_checkpoint", str, None, "Base diffusion checkpoint for distillation.", allow_none=True),
    ConfigField("fast_sampler.epochs", int, 10, "Fast-sampler distillation epochs.", min_value=1),
    ConfigField("fast_sampler.learning_rate", float, 1e-4, "Fast-sampler learning rate.", min_value=1e-8),
    ConfigField("fast_sampler.optimizer_weight_decay", float, 1e-4, "Fast-sampler optimizer weight decay.", min_value=0.0),
    ConfigField("fast_sampler.grad_clip_norm", float, 1.0, "Fast-sampler gradient clipping norm.", min_value=0.0),
    ConfigField("fast_sampler.num_inference_steps", int, 4, "Target inference steps for distillation.", min_value=1),
    ConfigField("fast_sampler.lora_rank", int, 8, "LoRA rank.", min_value=1),
    ConfigField("fast_sampler.lora_alpha", float, 8.0, "LoRA alpha.", min_value=0.0),
    ConfigField("fast_sampler.prediction_loss_weight", float, 0.25, "Weight on student-vs-teacher prediction loss.", min_value=0.0),
    ConfigField("fast_sampler.save_every", int, 5, "Fast-sampler checkpoint interval.", min_value=1),
    ConfigField("masked_room.checkpoint_dir", str, "", "Directory for masked-room checkpoints. Empty means output_dir/checkpoints/masked_room."),
    ConfigField("masked_room.epochs", int, 100, "Masked-room training epochs.", min_value=1),
    ConfigField("masked_room.learning_rate", float, 1e-4, "Masked-room learning rate.", min_value=1e-8),
    ConfigField("masked_room.optimizer_weight_decay", float, 1e-5, "Masked-room optimizer weight decay.", min_value=0.0),
    ConfigField("masked_room.grad_clip_norm", float, 1.0, "Masked-room gradient clipping norm.", min_value=0.0),
    ConfigField("masked_room.scheduler_eta_min", float, 1e-6, "Masked-room cosine scheduler eta_min.", min_value=0.0),
    ConfigField("masked_room.save_every", int, 10, "Masked-room checkpoint interval.", min_value=1),
    ConfigField("masked_room.context_dim", int, 256, "Masked-room conditioning width.", min_value=8),
    ConfigField("masked_room.condition_hidden_dim", int, 256, "Masked-room condition-encoder hidden width.", min_value=8),
    ConfigField("masked_room.condition_num_gnn_layers", int, 3, "Masked-room condition-encoder GNN depth.", min_value=1),
    ConfigField("masked_room.condition_num_attention_heads", int, 8, "Masked-room fusion heads.", min_value=1),
    ConfigField("masked_room.condition_dropout", float, 0.1, "Masked-room condition-encoder dropout.", min_value=0.0, max_value=1.0),
    ConfigField("masked_room.condition_gnn_type", str, "gcn", "Masked-room graph backbone.", choices=("gcn", "gat", "sage", "gps")),
    ConfigField("masked_room.graph_conditioning_mode", str, "node_sequence", "Masked-room graph-conditioning mode.", choices=("node_sequence", "pooled")),
    ConfigField("masked_room.use_current_node_distance_features", bool, True, "Inject current-room distance features into masked-room graph conditioning."),
    ConfigField("masked_room.current_node_distance_max", int, 8, "Distance clip used when normalizing current-room graph distances for masked-room training.", min_value=1),
    ConfigField("masked_room.model_channels", int, 128, "Masked-room U-Net base channels.", min_value=8),
    ConfigField("masked_room.hidden_dim", int, 64, "Masked-room token hidden width.", min_value=8),
    ConfigField("masked_room.masked_steps", int, 8, "Masked-token corruption steps.", min_value=1),
    ConfigField("masked_room.room_topology_channels", int, 18, "Masked-room topology-channel count.", min_value=1),
]


def cli_name_for_path(path: str) -> str:
    parts = path.split(".")
    if parts[0] in {"training", "runtime", "dataset", "distributed"}:
        return "_".join(parts[1:])
    return "_".join(parts)


def _deep_set(root: Dict[str, Any], path: str, value: Any) -> None:
    cursor = root
    parts = path.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def _deep_get(root: Dict[str, Any], path: str) -> Any:
    cursor = root
    for part in path.split("."):
        cursor = cursor[part]
    return cursor


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def build_default_config() -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    for field in CONFIG_FIELDS:
        _deep_set(config, field.path, copy.deepcopy(field.default))
    return config


def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path)
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file {cfg_path} must contain a YAML mapping at the root.")
    return payload


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    raise TypeError(f"Expected a boolean-compatible value, got {value!r}.")


def _coerce_value(field: ConfigField, value: Any) -> Any:
    if value is None:
        if field.allow_none:
            return None
        raise TypeError(f"{field.path} does not allow null values.")

    target = field.field_type
    if target in {list, tuple}:
        if isinstance(value, str):
            raw_items = [part.strip() for part in value.split(",") if part.strip()]
        elif isinstance(value, (list, tuple)):
            raw_items = list(value)
        else:
            raise TypeError(f"{field.path} expects a sequence, got {type(value).__name__}.")

        item_type = field.sequence_item_type or str
        coerced_items = []
        for raw_item in raw_items:
            if item_type is bool:
                coerced_item = _normalize_bool(raw_item)
            elif item_type is int:
                if isinstance(raw_item, bool):
                    raise TypeError(f"{field.path} expects int items, got bool.")
                coerced_item = int(raw_item)
            elif item_type is float:
                if isinstance(raw_item, bool):
                    raise TypeError(f"{field.path} expects float items, got bool.")
                coerced_item = float(raw_item)
            elif item_type is str:
                coerced_item = str(raw_item)
            else:
                if not isinstance(raw_item, item_type):
                    raise TypeError(
                        f"{field.path} expects {item_type.__name__} items, got {type(raw_item).__name__}."
                    )
                coerced_item = raw_item

            if field.choices is not None and coerced_item not in tuple(field.choices):
                raise ValueError(
                    f"{field.path} contains invalid value {coerced_item!r}; expected one of {tuple(field.choices)!r}."
                )
            if isinstance(coerced_item, (int, float)):
                if field.min_value is not None and coerced_item < field.min_value:
                    raise ValueError(f"{field.path} item {coerced_item!r} is below minimum {field.min_value}.")
                if field.max_value is not None and coerced_item > field.max_value:
                    raise ValueError(f"{field.path} item {coerced_item!r} exceeds maximum {field.max_value}.")
            coerced_items.append(coerced_item)
        coerced = list(coerced_items) if target is list else tuple(coerced_items)
    elif target is bool:
        coerced = _normalize_bool(value)
    elif target is int:
        if isinstance(value, bool):
            raise TypeError(f"{field.path} expects int, got bool.")
        coerced = int(value)
    elif target is float:
        if isinstance(value, bool):
            raise TypeError(f"{field.path} expects float, got bool.")
        coerced = float(value)
    elif target is str:
        coerced = str(value)
    else:
        if not isinstance(value, target):
            raise TypeError(f"{field.path} expects {target.__name__}, got {type(value).__name__}.")
        coerced = value

    if target not in {list, tuple} and field.choices is not None and coerced not in tuple(field.choices):
        raise ValueError(
            f"{field.path}={coerced!r} is invalid; expected one of {tuple(field.choices)!r}."
        )
    if target not in {list, tuple} and isinstance(coerced, (int, float)):
        if field.min_value is not None and coerced < field.min_value:
            raise ValueError(f"{field.path}={coerced!r} is below minimum {field.min_value}.")
        if field.max_value is not None and coerced > field.max_value:
            raise ValueError(f"{field.path}={coerced!r} exceeds maximum {field.max_value}.")
    return coerced


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    validated = copy.deepcopy(config)
    known_paths = {field.path for field in CONFIG_FIELDS}

    def _collect_paths(node: Any, prefix: str = "") -> List[str]:
        if not isinstance(node, dict):
            return [prefix] if prefix else []
        paths: List[str] = []
        for key, value in node.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                paths.extend(_collect_paths(value, child))
            else:
                paths.append(child)
        return paths

    extra_paths = sorted(path for path in _collect_paths(validated) if path not in known_paths)
    if extra_paths:
        raise KeyError(f"Unknown configuration field(s): {extra_paths}")

    for field in CONFIG_FIELDS:
        try:
            raw_value = _deep_get(validated, field.path)
        except KeyError as exc:
            raise KeyError(f"Missing configuration field: {field.path}") from exc
        _deep_set(validated, field.path, _coerce_value(field, raw_value))

    schema_profile = str(validated["dataset"]["schema_profile"]).strip().lower()
    expected_schema = get_dataset_schema_requirements(schema_profile)
    for key, expected_value in expected_schema.items():
        actual_value = int(validated["dataset"][key])
        if actual_value != int(expected_value):
            raise ValueError(
                f"dataset.{key}={actual_value!r} is incompatible with dataset.schema_profile={schema_profile!r}. "
                f"Expected {expected_value}. This repository is currently schema-locked to "
                f"{dataset_schema_lock_summary(schema_profile)}."
            )

    if validated["training"]["stage"] == "fast_sampler" and not validated["fast_sampler"]["base_diffusion_checkpoint"]:
        logger.warning(
            "fast_sampler.base_diffusion_checkpoint is empty. main.py will fall back to output_dir/checkpoints/diffusion/best_model.pth."
        )

    if validated["diffusion"]["context_dim"] != validated["masked_room"]["context_dim"]:
        logger.warning(
            "masked_room.context_dim (%d) differs from diffusion.context_dim (%d). "
            "This is valid, but checkpoints are not interchangeable.",
            int(validated["masked_room"]["context_dim"]),
            int(validated["diffusion"]["context_dim"]),
        )

    channel_mult = [int(v) for v in validated["diffusion"]["unet_channel_mult"]]
    attention_levels = [int(v) for v in validated["diffusion"]["unet_attention_resolutions"]]
    unet_num_heads = int(validated["diffusion"]["unet_num_heads"])
    model_channels = int(validated["diffusion"]["model_channels"])
    if not channel_mult:
        raise ValueError("diffusion.unet_channel_mult must be non-empty.")
    if any(level < 0 or level >= len(channel_mult) for level in attention_levels):
        raise ValueError(
            "diffusion.unet_attention_resolutions contains an out-of-range level for "
            f"diffusion.unet_channel_mult={channel_mult!r}."
        )
    if any((model_channels * mult) % unet_num_heads != 0 for mult in channel_mult):
        raise ValueError(
            "Every diffusion U-Net channel width must be divisible by diffusion.unet_num_heads. "
            f"Got model_channels={model_channels}, unet_channel_mult={channel_mult}, "
            f"unet_num_heads={unet_num_heads}."
        )

    output_dir = Path(validated["runtime"]["output_dir"])
    validated["vqvae"]["checkpoint_dir"] = (
        validated["vqvae"]["checkpoint_dir"] or str(output_dir / "checkpoints" / "vqvae")
    )
    validated["diffusion"]["checkpoint_dir"] = (
        validated["diffusion"]["checkpoint_dir"] or str(output_dir / "checkpoints" / "diffusion")
    )
    validated["fast_sampler"]["checkpoint_dir"] = (
        validated["fast_sampler"]["checkpoint_dir"] or str(output_dir / "checkpoints" / "fast_sampler")
    )
    validated["masked_room"]["checkpoint_dir"] = (
        validated["masked_room"]["checkpoint_dir"] or str(output_dir / "checkpoints" / "masked_room")
    )
    return validated


def merge_config(
    *,
    yaml_path: Optional[str] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    defaults = build_default_config()
    yaml_config = load_yaml_config(yaml_path)
    merged = _deep_merge(defaults, yaml_config)
    if cli_overrides:
        merged = _deep_merge(merged, cli_overrides)
    return validate_config(merged)


def cli_overrides_from_namespace(namespace: Any) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for field in CONFIG_FIELDS:
        cli_name = field.cli or cli_name_for_path(field.path)
        if not hasattr(namespace, cli_name):
            continue
        value = getattr(namespace, cli_name)
        if value is None:
            continue
        _deep_set(overrides, field.path, value)
    return overrides


def apply_runtime_environment(config: Dict[str, Any]) -> None:
    cuda_visible_devices = str(config["distributed"]["cuda_visible_devices"]).strip()
    if cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    os.environ.setdefault("MASTER_PORT", str(config["distributed"]["master_port"]))


def seed_everything(seed: Optional[int]) -> int:
    if seed is None:
        seed = 42
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    return seed


def get_git_commit(cwd: Optional[str] = None) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() or None
    except Exception:
        return None


def save_reproducibility_snapshot(config: Dict[str, Any], *, argv: Optional[List[str]] = None) -> Dict[str, Path]:
    output_dir = Path(config["runtime"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_yaml = output_dir / "resolved_config.yaml"
    resolved_json = output_dir / "resolved_config.json"
    metadata_json = output_dir / "run_metadata.json"

    with open(resolved_yaml, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    with open(resolved_json, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    metadata = {
        "saved_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "seed": int(config["runtime"]["seed"]),
        "git_commit": get_git_commit(cwd=str(Path(__file__).resolve().parent.parent)),
        "command": list(argv or sys.argv),
        "dataset_schema_profile": str(config["dataset"]["schema_profile"]),
        "dataset_schema_lock": dataset_schema_lock_summary(str(config["dataset"]["schema_profile"])),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": (None if torch is None else getattr(torch, "__version__", None)),
    }
    with open(metadata_json, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return {
        "resolved_yaml": resolved_yaml,
        "resolved_json": resolved_json,
        "metadata_json": metadata_json,
    }


def configure_logging(config: Dict[str, Any], *, rank: Optional[int] = None) -> Path:
    output_dir = Path(config["runtime"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_rank = int(os.environ.get("RANK", "0")) if rank is None else int(rank)
    is_main_process = resolved_rank == 0

    log_name = str(config["runtime"]["log_file"])
    if is_main_process:
        log_path = output_dir / log_name
    else:
        base = Path(log_name)
        log_path = output_dir / f"{base.stem}.rank{resolved_rank}{base.suffix or '.log'}"

    log_level = logging.DEBUG if bool(config["runtime"]["verbose"]) else logging.INFO
    stream_level = log_level if is_main_process else logging.WARNING

    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(stream_level)

    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    logger.debug("Logging configured at %s", log_path)
    return log_path
