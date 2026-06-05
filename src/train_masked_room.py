"""
Training pipeline for the graph-conditioned discrete masked room model.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config_system import merge_config, seed_everything
from src.core.condition_encoder import DualStreamConditionEncoder, create_condition_encoder
from src.core.definitions import GRAPH_EDGE_FEATURE_DIM, GRAPH_NODE_FEATURE_DIM, ROOM_TOPOLOGY_CHANNEL_COUNT
from src.core.discrete_masked_model import (
    DiscreteMaskedRoomModel,
    create_discrete_masked_model,
)
from src.core.logic_net import LogicNet
from src.core.puzzle_stage_semantics import (
    DEFAULT_PUZZLE_STAGE_MAX_SEQUENCE_LENGTH,
    DEFAULT_PUZZLE_STAGE_SEMANTICS_HIDDEN_DIM,
    PuzzleStageSemanticsHead,
)
from src.pipeline.room_topology_conditioning import (
    DEFAULT_PUZZLE_STAGE_TOKEN_SCALE,
    DEFAULT_PUZZLE_STAGE_TRACE_DECAY,
    DEFAULT_SEMANTIC_ANCHOR_THRESHOLD,
    DEFAULT_SEMANTIC_PUZZLE_OFFSET,
    DEFAULT_SEMANTIC_ROLE_PRIOR_STRENGTH,
    apply_puzzle_stage_control_to_conditioning,
    apply_puzzle_structure_control_to_conditioning,
    apply_puzzle_structure_dropout_batch,
    build_topology_anchor_policy_metadata,
    build_topology_loss_focus_map,
)
from src.pipeline.graph_features import (
    align_nodewise_tensor,
    build_default_node_positions,
    compute_current_node_distance_features,
    compute_rwse_features,
)
from src.utils.checkpoint import (
    LATEST_RESUME_FILENAME,
    MetricsLogger,
    atomic_torch_save,
    enforce_checkpoint_storage_budget,
    log_checkpoint_artifact,
    prune_checkpoints,
    resolve_resume_checkpoint,
    safe_torch_load,
    write_checkpoint_metadata,
)
from src.utils.data_loading import dataloader_runtime_kwargs
from src.utils.model_capacity import count_parameters, log_capacity_guardrails
from src.utils.optimization import adamw_decay_param_groups_for_modules
from src.zelda_data.zelda_loader import DungeonBatchSampler, create_dataloader, graph_collate_fn
from src.train_vqvae import split_dataset_for_vqvae_validation

logger = logging.getLogger(__name__)


@dataclass(init=False)
class MaskedRoomTrainingConfig:
    def __init__(
        self,
        **kwargs: Any,
    ):
        self._init_from_values(**kwargs)

    def _init_from_values(
        self,
        data_dir: str = "Data/The Legend of Zelda",
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = True,
        shuffle_train: bool = True,
        shuffle_val: bool = False,
        normalize: bool = True,
        train_dungeon_ids: Optional[List[int]] = None,
        test_dungeon_ids: Optional[List[int]] = None,
        variants: Optional[List[int]] = None,
        num_classes: int = 44,
        latent_dim: int = 64,
        node_feature_dim: int = GRAPH_NODE_FEATURE_DIM,
        edge_feature_dim: int = GRAPH_EDGE_FEATURE_DIM,
        epochs: int = 100,
        learning_rate: float = 1e-4,
        context_dim: int = 256,
        condition_hidden_dim: int = 256,
        condition_num_gnn_layers: int = 3,
        condition_num_attention_heads: int = 8,
        condition_dropout: float = 0.1,
        condition_gnn_type: str = "gcn",
        condition_use_reference_room_maps: bool = False,
        condition_reference_tile_vocab_size: int = 44,
        condition_reference_embedding_dim: int = 32,
        condition_reference_hidden_dim: int = 64,
        graph_conditioning_mode: str = "node_sequence",
        use_current_node_distance_features: bool = True,
        current_node_distance_max: int = 8,
        model_channels: int = 64,
        hidden_dim: int = 48,
        masked_steps: int = 8,
        attention_mode: str = "softmax",
        context_attention_mode: str = "concat_encoder",
        topology_conditioning_mode: str = "additive",
        hedgehog_feature_dim: int = 32,
        graph_auto_linear_attention_nodes: int = 128,
        spatial_graph_gate_init: float = -2.0,
        spatial_topology_gate_init: float = -2.0,
        unet_channel_mult: Tuple[int, ...] = (1, 2),
        unet_num_res_blocks: int = 1,
        unet_attention_resolutions: Tuple[int, ...] = (0, 1),
        unet_num_heads: int = 4,
        unet_dropout: float = 0.1,
        min_mask_ratio: float = 0.12,
        max_mask_ratio: float = 0.85,
        topology_alignment_weight: float = 0.25,
        topology_marker_weight: float = 2.0,
        topology_trace_weight: float = 0.75,
        topology_focus_dilation: int = 1,
        logic_net_enabled: bool = False,
        logic_net_trainable: bool = False,
        alpha_logic: float = 0.0,
        logic_global_reach_weight: float = 1.0,
        logic_global_room_weight: float = 0.25,
        logic_topology_trace_weight: float = 0.25,
        logic_topology_anchor_weight: float = 0.25,
        logic_grid_pathfinder: str = "bellman_ford",
        num_logic_iterations: int = 30,
        validation_fraction: float = 0.1,
        validation_max_batches: int = 16,
        best_checkpoint_metric: str = "val_loss",
        optimizer_weight_decay: float = 1e-5,
        scheduler_eta_min: float = 1e-6,
        grad_clip_norm: float = 1.0,
        room_topology_channels: int = ROOM_TOPOLOGY_CHANNEL_COUNT,
        topology_supervision_mode: str = "runtime_aligned",
        semantic_role_prior_strength: float = DEFAULT_SEMANTIC_ROLE_PRIOR_STRENGTH,
        semantic_puzzle_offset: int = DEFAULT_SEMANTIC_PUZZLE_OFFSET,
        puzzle_structure_dropout_prob: float = 0.35,
        puzzle_stage_conditioning_enabled: bool = False,
        puzzle_stage_token_scale: float = DEFAULT_PUZZLE_STAGE_TOKEN_SCALE,
        puzzle_stage_topology_enabled: bool = False,
        puzzle_stage_trace_decay: float = DEFAULT_PUZZLE_STAGE_TRACE_DECAY,
        puzzle_stage_semantics_loss_weight: float = 0.0,
        puzzle_stage_semantics_hidden_dim: int = DEFAULT_PUZZLE_STAGE_SEMANTICS_HIDDEN_DIM,
        puzzle_stage_semantics_max_sequence_length: int = DEFAULT_PUZZLE_STAGE_MAX_SEQUENCE_LENGTH,
        checkpoint_dir: str = "./checkpoints/masked_room",
        save_every: int = 10,
        keep_last: int = 2,
        auto_resume: bool = True,
        resume_checkpoint: Optional[str] = None,
        checkpoint_storage_budget_gb: Optional[float] = None,
        checkpoint_storage_warning_fraction: float = 0.8,
        checkpoint_storage_cleanup_enabled: bool = True,
        checkpoint_storage_cleanup_target_fraction: float = 0.6,
        device: str = "auto",
        seed: int = 42,
        quick: bool = False,
        semantic_anchor_threshold: float = DEFAULT_SEMANTIC_ANCHOR_THRESHOLD,
    ):
        self.data_dir = data_dir
        self.batch_size = int(batch_size)
        self.num_workers = int(max(0, num_workers))
        self.pin_memory = bool(pin_memory)
        self.drop_last = bool(drop_last)
        self.shuffle_train = bool(shuffle_train)
        self.shuffle_val = bool(shuffle_val)
        self.normalize = bool(normalize)
        self.train_dungeon_ids = [int(v) for v in (train_dungeon_ids if train_dungeon_ids is not None else list(range(1, 9)))]
        self.test_dungeon_ids = [int(v) for v in (test_dungeon_ids if test_dungeon_ids is not None else [9])]
        self.variants = [int(v) for v in (variants if variants is not None else [1, 2])]
        self.num_classes = int(max(1, num_classes))
        self.latent_dim = int(max(1, latent_dim))
        self.node_feature_dim = int(max(1, node_feature_dim))
        self.edge_feature_dim = int(max(1, edge_feature_dim))
        self.epochs = 2 if quick else int(epochs)
        self.learning_rate = float(learning_rate)
        self.context_dim = int(context_dim)
        self.condition_hidden_dim = int(condition_hidden_dim)
        self.condition_num_gnn_layers = int(max(1, condition_num_gnn_layers))
        self.condition_num_attention_heads = int(max(1, condition_num_attention_heads))
        self.condition_dropout = float(max(0.0, min(1.0, condition_dropout)))
        self.condition_gnn_type = str(condition_gnn_type).strip().lower()
        self.condition_use_reference_room_maps = bool(condition_use_reference_room_maps)
        self.condition_reference_tile_vocab_size = int(max(2, condition_reference_tile_vocab_size))
        self.condition_reference_embedding_dim = int(max(4, condition_reference_embedding_dim))
        self.condition_reference_hidden_dim = int(max(4, condition_reference_hidden_dim))
        self.graph_conditioning_mode = str(graph_conditioning_mode).strip().lower()
        self.use_current_node_distance_features = bool(use_current_node_distance_features)
        self.current_node_distance_max = int(max(1, current_node_distance_max))
        self.model_channels = int(model_channels)
        self.hidden_dim = int(hidden_dim)
        self.masked_steps = int(max(1, masked_steps))
        self.attention_mode = str(attention_mode).strip().lower()
        cam = str(context_attention_mode).strip().lower()
        if cam in {"concat", "encoder", "original"}:
            cam = "concat_encoder"
        elif cam in {"cross", "decoder", "cross_attention"}:
            cam = "cross_decoder"
        if cam not in {"concat_encoder", "cross_decoder"}:
            raise ValueError("context_attention_mode must be 'concat_encoder' or 'cross_decoder'.")
        self.context_attention_mode = cam
        self.topology_conditioning_mode = str(topology_conditioning_mode).strip().lower()
        self.hedgehog_feature_dim = int(max(4, hedgehog_feature_dim))
        self.graph_auto_linear_attention_nodes = int(max(0, graph_auto_linear_attention_nodes))
        self.spatial_graph_gate_init = float(spatial_graph_gate_init)
        self.spatial_topology_gate_init = float(spatial_topology_gate_init)

        def _normalize_int_sequence(
            name: str,
            values: Any,
            *,
            allow_zero: bool = True,
            allow_empty: bool = False,
        ) -> Tuple[int, ...]:
            if isinstance(values, str):
                parts = [part.strip() for part in values.split(",") if part.strip()]
                values = [int(part) for part in parts]
            elif isinstance(values, torch.Tensor):
                values = values.detach().cpu().tolist()
            elif not isinstance(values, (list, tuple)):
                raise TypeError(f"{name} must be a list/tuple of integers, got {type(values).__name__}.")

            seq = tuple(int(v) for v in values)
            if not seq and not allow_empty:
                raise ValueError(f"{name} must be non-empty.")
            lower_bound = 0 if allow_zero else 1
            if any(v < lower_bound for v in seq):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} must contain only {qualifier} integers, got {seq!r}.")
            return seq

        self.unet_channel_mult = _normalize_int_sequence(
            "unet_channel_mult",
            unet_channel_mult,
            allow_zero=False,
        )
        self.unet_num_res_blocks = int(max(1, unet_num_res_blocks))
        self.unet_attention_resolutions = _normalize_int_sequence(
            "unet_attention_resolutions",
            unet_attention_resolutions,
            allow_zero=True,
            allow_empty=True,
        )
        self.unet_num_heads = int(max(1, unet_num_heads))
        self.unet_dropout = float(max(0.0, min(1.0, unet_dropout)))
        self.min_mask_ratio = float(max(0.0, min(1.0, min_mask_ratio)))
        self.max_mask_ratio = float(max(0.0, min(1.0, max_mask_ratio)))
        self.topology_alignment_weight = float(max(0.0, topology_alignment_weight))
        self.topology_marker_weight = float(max(0.0, topology_marker_weight))
        self.topology_trace_weight = float(max(0.0, topology_trace_weight))
        self.topology_focus_dilation = int(max(0, topology_focus_dilation))
        self.logic_net_enabled = bool(logic_net_enabled)
        self.logic_net_trainable = bool(logic_net_trainable) if self.logic_net_enabled else False
        self.alpha_logic = float(max(0.0, alpha_logic)) if self.logic_net_enabled else 0.0
        self.logic_global_reach_weight = float(max(0.0, logic_global_reach_weight))
        self.logic_global_room_weight = float(max(0.0, logic_global_room_weight))
        self.logic_topology_trace_weight = float(max(0.0, logic_topology_trace_weight))
        self.logic_topology_anchor_weight = float(max(0.0, logic_topology_anchor_weight))
        self.logic_grid_pathfinder = str(logic_grid_pathfinder).strip().lower()
        self.num_logic_iterations = int(max(1, num_logic_iterations))
        self.validation_fraction = float(max(0.0, min(0.5, validation_fraction)))
        self.validation_max_batches = int(max(1, validation_max_batches))
        self.best_checkpoint_metric = str(best_checkpoint_metric).strip().lower()
        self.optimizer_weight_decay = float(max(0.0, optimizer_weight_decay))
        self.scheduler_eta_min = float(max(0.0, scheduler_eta_min))
        self.grad_clip_norm = float(max(0.0, grad_clip_norm))
        self.room_topology_channels = int(max(1, room_topology_channels))
        self.topology_supervision_mode = str(topology_supervision_mode).strip().lower()
        self.semantic_role_prior_strength = float(max(0.0, min(1.0, semantic_role_prior_strength)))
        self.semantic_puzzle_offset = int(max(0, semantic_puzzle_offset))
        self.puzzle_structure_dropout_prob = float(max(0.0, min(1.0, puzzle_structure_dropout_prob)))
        self.puzzle_stage_conditioning_enabled = bool(puzzle_stage_conditioning_enabled)
        self.puzzle_stage_token_scale = float(max(0.0, puzzle_stage_token_scale))
        self.puzzle_stage_topology_enabled = bool(puzzle_stage_topology_enabled)
        self.puzzle_stage_trace_decay = float(max(0.05, min(1.0, puzzle_stage_trace_decay)))
        self.puzzle_stage_semantics_loss_weight = float(max(0.0, puzzle_stage_semantics_loss_weight))
        self.puzzle_stage_semantics_hidden_dim = int(max(16, puzzle_stage_semantics_hidden_dim))
        self.puzzle_stage_semantics_max_sequence_length = int(max(1, puzzle_stage_semantics_max_sequence_length))
        self.checkpoint_dir = str(checkpoint_dir)
        self.save_every = int(save_every)
        self.keep_last = int(max(0, keep_last))
        self.auto_resume = bool(auto_resume)
        self.resume_checkpoint = None if resume_checkpoint is None else str(resume_checkpoint)
        self.checkpoint_storage_budget_gb = (
            None if checkpoint_storage_budget_gb is None else float(max(0.0, checkpoint_storage_budget_gb))
        )
        self.checkpoint_storage_warning_fraction = float(max(0.0, min(1.0, checkpoint_storage_warning_fraction)))
        self.checkpoint_storage_cleanup_enabled = bool(checkpoint_storage_cleanup_enabled)
        self.checkpoint_storage_cleanup_target_fraction = float(
            max(0.0, min(1.0, checkpoint_storage_cleanup_target_fraction))
        )
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else str(device)
        self.seed = int(seed)
        self.quick = bool(quick)
        self.semantic_anchor_threshold = float(max(0.0, min(1.0, semantic_anchor_threshold)))

        if self.condition_gnn_type not in {"gcn", "gat", "sage", "gps"}:
            raise ValueError("condition_gnn_type must be 'gcn', 'gat', 'sage', or 'gps'.")
        if self.graph_conditioning_mode not in {"node_sequence", "pooled"}:
            raise ValueError("graph_conditioning_mode must be 'node_sequence' or 'pooled'.")
        if self.topology_supervision_mode not in {"runtime_aligned", "oracle_room_grid"}:
            raise ValueError("topology_supervision_mode must be 'runtime_aligned' or 'oracle_room_grid'.")
        if self.attention_mode not in {"softmax", "linear_hedgehog"}:
            raise ValueError("attention_mode must be 'softmax' or 'linear_hedgehog'.")
        if self.topology_conditioning_mode not in {"additive", "spade"}:
            raise ValueError("topology_conditioning_mode must be 'additive' or 'spade'.")
        if self.logic_grid_pathfinder not in {"bellman_ford", "conv", "cnn", "vin", "learnable", "perturb_and_map"}:
            raise ValueError(
                "logic_grid_pathfinder must be 'bellman_ford', 'conv'/'cnn', 'vin', 'learnable', or 'perturb_and_map'."
            )
        if any((self.model_channels * mult) % self.unet_num_heads != 0 for mult in self.unet_channel_mult):
            raise ValueError(
                "Every masked-room U-Net channel width must be divisible by unet_num_heads; "
                f"got model_channels={self.model_channels}, unet_channel_mult={self.unet_channel_mult}, "
                f"unet_num_heads={self.unet_num_heads}."
            )
        max_level = len(self.unet_channel_mult) - 1
        if any(level > max_level for level in self.unet_attention_resolutions):
            raise ValueError(
                f"unet_attention_resolutions={self.unet_attention_resolutions!r} contains a level above {max_level}."
            )
        if self.min_mask_ratio > self.max_mask_ratio:
            raise ValueError(
                "min_mask_ratio must be <= max_mask_ratio. "
                f"Got {self.min_mask_ratio} > {self.max_mask_ratio}."
            )
        if self.best_checkpoint_metric not in {"val_loss", "val_topology_focus_loss", "val_puzzle_stage_semantic_loss", "train_loss"}:
            raise ValueError(
                "best_checkpoint_metric must be 'val_loss', 'val_topology_focus_loss', 'val_puzzle_stage_semantic_loss', or 'train_loss'."
            )

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def masked_room_training_kwargs_from_resolved_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build MaskedRoomTrainingConfig kwargs from the validated global config payload."""
    stage = config["masked_room"]
    dataset = config["dataset"]
    runtime = config["runtime"]
    return {
        "data_dir": dataset["data_dir"],
        "batch_size": dataset["batch_size"],
        "num_workers": dataset["num_workers"],
        "pin_memory": dataset["pin_memory"],
        "drop_last": dataset["drop_last"],
        "shuffle_train": dataset["shuffle_train"],
        "shuffle_val": dataset["shuffle_val"],
        "normalize": dataset["normalize"],
        "train_dungeon_ids": dataset.get("train_dungeons", list(range(1, 9))),
        "test_dungeon_ids": dataset.get("test_dungeons", [9]),
        "variants": dataset.get("variants", [1, 2]),
        "num_classes": dataset["num_classes"],
        "latent_dim": config["vqvae"]["latent_dim"],
        "node_feature_dim": dataset["node_feature_dim"],
        "edge_feature_dim": dataset["edge_feature_dim"],
        "epochs": stage["epochs"],
        "learning_rate": stage["learning_rate"],
        "context_dim": stage["context_dim"],
        "condition_hidden_dim": stage["condition_hidden_dim"],
        "condition_num_gnn_layers": stage["condition_num_gnn_layers"],
        "condition_num_attention_heads": stage["condition_num_attention_heads"],
        "condition_dropout": stage["condition_dropout"],
        "condition_gnn_type": stage["condition_gnn_type"],
        "condition_use_reference_room_maps": stage["condition_use_reference_room_maps"],
        "condition_reference_tile_vocab_size": stage["condition_reference_tile_vocab_size"],
        "condition_reference_embedding_dim": stage["condition_reference_embedding_dim"],
        "condition_reference_hidden_dim": stage["condition_reference_hidden_dim"],
        "graph_conditioning_mode": stage["graph_conditioning_mode"],
        "use_current_node_distance_features": stage["use_current_node_distance_features"],
        "current_node_distance_max": stage["current_node_distance_max"],
        "model_channels": stage["model_channels"],
        "hidden_dim": stage["hidden_dim"],
        "masked_steps": stage["masked_steps"],
        "attention_mode": stage["attention_mode"],
        "context_attention_mode": stage.get("context_attention_mode", "concat_encoder"),
        "topology_conditioning_mode": stage["topology_conditioning_mode"],
        "hedgehog_feature_dim": stage["hedgehog_feature_dim"],
        "graph_auto_linear_attention_nodes": stage["graph_auto_linear_attention_nodes"],
        "spatial_graph_gate_init": stage["spatial_graph_gate_init"],
        "spatial_topology_gate_init": stage["spatial_topology_gate_init"],
        "unet_channel_mult": tuple(stage["unet_channel_mult"]),
        "unet_num_res_blocks": stage["unet_num_res_blocks"],
        "unet_attention_resolutions": tuple(stage["unet_attention_resolutions"]),
        "unet_num_heads": stage["unet_num_heads"],
        "unet_dropout": stage["unet_dropout"],
        "min_mask_ratio": stage["min_mask_ratio"],
        "max_mask_ratio": stage["max_mask_ratio"],
        "topology_alignment_weight": stage.get("topology_alignment_weight", 0.25),
        "topology_marker_weight": stage.get("topology_marker_weight", 2.0),
        "topology_trace_weight": stage.get("topology_trace_weight", 0.75),
        "topology_focus_dilation": stage.get("topology_focus_dilation", 1),
        "logic_net_enabled": stage.get("logic_net_enabled", False),
        "logic_net_trainable": stage.get("logic_net_trainable", False),
        "alpha_logic": stage.get("alpha_logic", 0.0),
        "logic_global_reach_weight": stage.get("logic_global_reach_weight", 1.0),
        "logic_global_room_weight": stage.get("logic_global_room_weight", 0.25),
        "logic_topology_trace_weight": stage.get("logic_topology_trace_weight", 0.25),
        "logic_topology_anchor_weight": stage.get("logic_topology_anchor_weight", 0.25),
        "logic_grid_pathfinder": stage.get("logic_grid_pathfinder", "bellman_ford"),
        "num_logic_iterations": stage.get("num_logic_iterations", 30),
        "validation_fraction": stage.get("validation_fraction", 0.1),
        "validation_max_batches": stage.get("validation_max_batches", 16),
        "best_checkpoint_metric": stage.get("best_checkpoint_metric", "val_loss"),
        "optimizer_weight_decay": stage["optimizer_weight_decay"],
        "scheduler_eta_min": stage["scheduler_eta_min"],
        "grad_clip_norm": stage["grad_clip_norm"],
        "room_topology_channels": stage["room_topology_channels"],
        "topology_supervision_mode": dataset["topology_supervision_mode"],
        "semantic_role_prior_strength": config["generation"]["semantic_role_prior_strength"],
        "semantic_puzzle_offset": config["generation"]["semantic_puzzle_offset"],
        "puzzle_structure_dropout_prob": stage.get("puzzle_structure_dropout_prob", 0.35),
        "puzzle_stage_conditioning_enabled": stage.get("puzzle_stage_conditioning_enabled", False),
        "puzzle_stage_token_scale": stage.get("puzzle_stage_token_scale", DEFAULT_PUZZLE_STAGE_TOKEN_SCALE),
        "puzzle_stage_topology_enabled": stage.get("puzzle_stage_topology_enabled", False),
        "puzzle_stage_trace_decay": stage.get("puzzle_stage_trace_decay", DEFAULT_PUZZLE_STAGE_TRACE_DECAY),
        "puzzle_stage_semantics_loss_weight": stage.get("puzzle_stage_semantics_loss_weight", 0.0),
        "puzzle_stage_semantics_hidden_dim": stage.get(
            "puzzle_stage_semantics_hidden_dim",
            DEFAULT_PUZZLE_STAGE_SEMANTICS_HIDDEN_DIM,
        ),
        "puzzle_stage_semantics_max_sequence_length": stage.get(
            "puzzle_stage_semantics_max_sequence_length",
            DEFAULT_PUZZLE_STAGE_MAX_SEQUENCE_LENGTH,
        ),
        "semantic_anchor_threshold": config["generation"]["semantic_anchor_threshold"],
        "checkpoint_dir": stage["checkpoint_dir"],
        "save_every": stage["save_every"],
        "keep_last": stage["keep_last"],
        "auto_resume": runtime["auto_resume"],
        "resume_checkpoint": runtime["resume"],
        "checkpoint_storage_budget_gb": runtime["checkpoint_storage_budget_gb"],
        "checkpoint_storage_warning_fraction": runtime["checkpoint_storage_warning_fraction"],
        "checkpoint_storage_cleanup_enabled": runtime["checkpoint_storage_cleanup_enabled"],
        "checkpoint_storage_cleanup_target_fraction": runtime["checkpoint_storage_cleanup_target_fraction"],
        "device": runtime["device"],
        "seed": runtime["seed"],
        "quick": runtime["quick"],
    }


def _legacy_masked_room_overrides_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}

    def _set(name: str, value: Any, *, transform: Optional[Any] = None) -> None:
        if value is None:
            return
        if transform is not None:
            value = transform(value)
        overrides[name] = value

    _set("data_dir", getattr(args, "data_dir", None))
    _set("batch_size", getattr(args, "batch_size", None))
    _set("train_dungeon_ids", getattr(args, "train_dungeon_ids", None))
    _set("test_dungeon_ids", getattr(args, "test_dungeon_ids", None))
    _set("variants", getattr(args, "variants", None))
    _set("epochs", getattr(args, "epochs", None))
    _set("learning_rate", getattr(args, "lr", None))
    _set("num_classes", getattr(args, "num_classes", None))
    _set("latent_dim", getattr(args, "latent_dim", None))
    _set("context_dim", getattr(args, "context_dim", None))
    _set("condition_hidden_dim", getattr(args, "condition_hidden_dim", None))
    _set("condition_num_gnn_layers", getattr(args, "condition_num_gnn_layers", None))
    _set("condition_num_attention_heads", getattr(args, "condition_num_attention_heads", None))
    _set("condition_dropout", getattr(args, "condition_dropout", None))
    _set("condition_gnn_type", getattr(args, "condition_gnn_type", None))
    _set("condition_use_reference_room_maps", getattr(args, "condition_use_reference_room_maps", None))
    _set("condition_reference_tile_vocab_size", getattr(args, "condition_reference_tile_vocab_size", None))
    _set("condition_reference_embedding_dim", getattr(args, "condition_reference_embedding_dim", None))
    _set("condition_reference_hidden_dim", getattr(args, "condition_reference_hidden_dim", None))
    _set("graph_conditioning_mode", getattr(args, "graph_conditioning_mode", None))
    _set("use_current_node_distance_features", getattr(args, "use_current_node_distance_features", None))
    _set("current_node_distance_max", getattr(args, "current_node_distance_max", None))
    _set("model_channels", getattr(args, "model_channels", None))
    _set("hidden_dim", getattr(args, "hidden_dim", None))
    _set("masked_steps", getattr(args, "masked_steps", None))
    _set("attention_mode", getattr(args, "attention_mode", None))
    _set("context_attention_mode", getattr(args, "context_attention_mode", None))
    _set("topology_conditioning_mode", getattr(args, "topology_conditioning_mode", None))
    _set("hedgehog_feature_dim", getattr(args, "hedgehog_feature_dim", None))
    _set("graph_auto_linear_attention_nodes", getattr(args, "graph_auto_linear_attention_nodes", None))
    _set("spatial_graph_gate_init", getattr(args, "spatial_graph_gate_init", None))
    _set("spatial_topology_gate_init", getattr(args, "spatial_topology_gate_init", None))
    _set("unet_channel_mult", getattr(args, "unet_channel_mult", None), transform=tuple)
    _set("unet_num_res_blocks", getattr(args, "unet_num_res_blocks", None))
    _set("unet_attention_resolutions", getattr(args, "unet_attention_resolutions", None), transform=tuple)
    _set("unet_num_heads", getattr(args, "unet_num_heads", None))
    _set("unet_dropout", getattr(args, "unet_dropout", None))
    _set("min_mask_ratio", getattr(args, "min_mask_ratio", None))
    _set("max_mask_ratio", getattr(args, "max_mask_ratio", None))
    _set("topology_alignment_weight", getattr(args, "topology_alignment_weight", None))
    _set("topology_marker_weight", getattr(args, "topology_marker_weight", None))
    _set("topology_trace_weight", getattr(args, "topology_trace_weight", None))
    _set("topology_focus_dilation", getattr(args, "topology_focus_dilation", None))
    _set("logic_net_enabled", getattr(args, "logic_net_enabled", None))
    _set("logic_net_trainable", getattr(args, "logic_net_trainable", None))
    _set("alpha_logic", getattr(args, "alpha_logic", None))
    _set("logic_global_reach_weight", getattr(args, "logic_global_reach_weight", None))
    _set("logic_global_room_weight", getattr(args, "logic_global_room_weight", None))
    _set("logic_topology_trace_weight", getattr(args, "logic_topology_trace_weight", None))
    _set("logic_topology_anchor_weight", getattr(args, "logic_topology_anchor_weight", None))
    _set("logic_grid_pathfinder", getattr(args, "logic_grid_pathfinder", None))
    _set("num_logic_iterations", getattr(args, "num_logic_iterations", None))
    _set("validation_fraction", getattr(args, "validation_fraction", None))
    _set("validation_max_batches", getattr(args, "validation_max_batches", None))
    _set("best_checkpoint_metric", getattr(args, "best_checkpoint_metric", None))
    _set("puzzle_structure_dropout_prob", getattr(args, "puzzle_structure_dropout_prob", None))
    _set("puzzle_stage_conditioning_enabled", getattr(args, "puzzle_stage_conditioning_enabled", None))
    _set("puzzle_stage_token_scale", getattr(args, "puzzle_stage_token_scale", None))
    _set("puzzle_stage_topology_enabled", getattr(args, "puzzle_stage_topology_enabled", None))
    _set("puzzle_stage_trace_decay", getattr(args, "puzzle_stage_trace_decay", None))
    _set("puzzle_stage_semantics_loss_weight", getattr(args, "puzzle_stage_semantics_loss_weight", None))
    _set("puzzle_stage_semantics_hidden_dim", getattr(args, "puzzle_stage_semantics_hidden_dim", None))
    _set(
        "puzzle_stage_semantics_max_sequence_length",
        getattr(args, "puzzle_stage_semantics_max_sequence_length", None),
    )
    _set("checkpoint_dir", getattr(args, "checkpoint_dir", None))
    _set("save_every", getattr(args, "save_every", None))
    _set("keep_last", getattr(args, "keep_last", None))
    _set("auto_resume", getattr(args, "auto_resume", None))
    _set("resume_checkpoint", getattr(args, "resume", None))
    _set("checkpoint_storage_budget_gb", getattr(args, "checkpoint_storage_budget_gb", None))
    _set("checkpoint_storage_warning_fraction", getattr(args, "checkpoint_storage_warning_fraction", None))
    _set("checkpoint_storage_cleanup_enabled", getattr(args, "checkpoint_storage_cleanup_enabled", None))
    _set("checkpoint_storage_cleanup_target_fraction", getattr(args, "checkpoint_storage_cleanup_target_fraction", None))
    _set("device", getattr(args, "device", None))
    _set("seed", getattr(args, "seed", None))
    _set("quick", getattr(args, "quick", None))
    return overrides


def build_masked_room_training_config_from_args(args: argparse.Namespace) -> MaskedRoomTrainingConfig:
    base_kwargs: Dict[str, Any] = {}
    config_path = getattr(args, "config", None)
    if config_path:
        resolved = merge_config(yaml_path=str(config_path), cli_overrides=None)
        base_kwargs = masked_room_training_kwargs_from_resolved_config(resolved)
        if getattr(args, "verbose", None) is None:
            setattr(args, "verbose", bool(resolved["runtime"]["verbose"]))
    legacy_overrides = _legacy_masked_room_overrides_from_args(args)
    return MaskedRoomTrainingConfig(**{**base_kwargs, **legacy_overrides})


def _create_masked_room_dataloaders(
    config: MaskedRoomTrainingConfig,
) -> tuple[DataLoader, DataLoader, str, int, int]:
    base_loader = create_dataloader(
        config.data_dir,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
        use_vglc=True,
        normalize=config.normalize,
        room_level=True,
        load_graphs=True,
        node_feature_dim=config.node_feature_dim,
        edge_feature_dim=config.edge_feature_dim,
        topology_supervision_mode=config.topology_supervision_mode,
        semantic_role_prior_strength=config.semantic_role_prior_strength,
        semantic_puzzle_offset=config.semantic_puzzle_offset,
        puzzle_stage_topology_enabled=config.puzzle_stage_topology_enabled,
        puzzle_stage_trace_decay=config.puzzle_stage_trace_decay,
        dungeon_ids=config.train_dungeon_ids,
        variants=config.variants,
    )
    try:
        base_loader_batches = int(len(base_loader))
    except Exception:
        base_loader_batches = -1
    if base_loader_batches == 0:
        return base_loader, base_loader, "train", 0, 0
    dataset = base_loader.dataset
    train_dataset, val_dataset = split_dataset_for_vqvae_validation(
        dataset,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )
    eval_source = val_dataset if val_dataset is not None else train_dataset
    eval_split_name = "val" if val_dataset is not None else "train"
    use_dungeon_batches = bool(getattr(config, "logic_net_enabled", False)) and float(
        getattr(config, "alpha_logic", 0.0)
    ) > 0.0
    runtime_kwargs = dataloader_runtime_kwargs(num_workers=config.num_workers, pin_memory=config.pin_memory)
    if use_dungeon_batches:
        train_sampler = DungeonBatchSampler.from_dataset(
            train_dataset,
            shuffle=config.shuffle_train,
            drop_last=config.drop_last,
            seed=config.seed,
        )
        val_sampler = DungeonBatchSampler.from_dataset(
            eval_source,
            shuffle=False,
            drop_last=False,
            seed=config.seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=graph_collate_fn,
            **runtime_kwargs,
        )
        val_loader = DataLoader(
            eval_source,
            batch_sampler=val_sampler,
            collate_fn=graph_collate_fn,
            **runtime_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle_train,
            drop_last=config.drop_last,
            collate_fn=graph_collate_fn,
            **runtime_kwargs,
        )
        val_loader = DataLoader(
            eval_source,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=graph_collate_fn,
            **runtime_kwargs,
        )
    return train_loader, val_loader, eval_split_name, len(train_dataset), len(eval_source)


def _resolve_masked_room_best_metric_name(config: MaskedRoomTrainingConfig) -> str:
    if config.best_checkpoint_metric == "train_loss":
        return "train_loss"
    if config.best_checkpoint_metric == "val_topology_focus_loss":
        return "val_topology_focus_loss"
    if config.best_checkpoint_metric == "val_puzzle_stage_semantic_loss":
        return "val_puzzle_stage_semantic_loss"
    return "val_loss"


class MaskedRoomTrainer:
    def __init__(
        self,
        config: MaskedRoomTrainingConfig,
        *,
        model: Optional[DiscreteMaskedRoomModel] = None,
        condition_encoder: Optional[DualStreamConditionEncoder] = None,
    ):
        self.config = config
        self.device = torch.device(config.device)
        self.model = (model or create_discrete_masked_model(
            num_classes=config.num_classes,
            hidden_dim=config.hidden_dim,
            model_channels=config.model_channels,
            context_dim=config.context_dim,
            num_steps=config.masked_steps,
            attention_mode=config.attention_mode,
            context_attention_mode=config.context_attention_mode,
            topology_conditioning_mode=config.topology_conditioning_mode,
            hedgehog_feature_dim=config.hedgehog_feature_dim,
            graph_auto_linear_attention_nodes=config.graph_auto_linear_attention_nodes,
            spatial_graph_gate_init=config.spatial_graph_gate_init,
            spatial_topology_gate_init=config.spatial_topology_gate_init,
            unet_channel_mult=config.unet_channel_mult,
            unet_num_res_blocks=config.unet_num_res_blocks,
            unet_attention_resolutions=config.unet_attention_resolutions,
            unet_num_heads=config.unet_num_heads,
            unet_dropout=config.unet_dropout,
            room_topology_channels=config.room_topology_channels,
        )).to(self.device)
        self.condition_encoder = (condition_encoder or create_condition_encoder(
            latent_dim=config.latent_dim,
            node_feature_dim=config.node_feature_dim,
            edge_feature_dim=config.edge_feature_dim,
            output_dim=config.context_dim,
            hidden_dim=config.condition_hidden_dim,
            num_gnn_layers=config.condition_num_gnn_layers,
            gnn_type=config.condition_gnn_type,
            num_attention_heads=config.condition_num_attention_heads,
            dropout=config.condition_dropout,
            use_current_node_distance_features=config.use_current_node_distance_features,
            use_reference_room_maps=config.condition_use_reference_room_maps,
            reference_num_tile_types=config.condition_reference_tile_vocab_size,
            reference_embedding_dim=config.condition_reference_embedding_dim,
            reference_hidden_dim=config.condition_reference_hidden_dim,
        )).to(self.device)
        self.puzzle_stage_semantics_head = PuzzleStageSemanticsHead(
            num_tile_classes=int(config.num_classes),
            hidden_dim=int(getattr(config, "puzzle_stage_semantics_hidden_dim", DEFAULT_PUZZLE_STAGE_SEMANTICS_HIDDEN_DIM)),
            max_sequence_length=int(
                getattr(
                    config,
                    "puzzle_stage_semantics_max_sequence_length",
                    DEFAULT_PUZZLE_STAGE_MAX_SEQUENCE_LENGTH,
                )
            ),
        ).to(self.device)
        self.logic_net = self._create_logic_net() if bool(getattr(config, "logic_net_enabled", False)) else None
        if self.logic_net is not None:
            self.logic_net.to(self.device)
            if not bool(getattr(config, "logic_net_trainable", False)):
                self.logic_net.requires_grad_(False)
                self.logic_net.eval()
        optimizer_modules = [
            ("model", self.model),
            ("condition_encoder", self.condition_encoder),
            ("puzzle_stage_semantics_head", self.puzzle_stage_semantics_head),
        ]
        if self.logic_net is not None and bool(getattr(config, "logic_net_trainable", False)):
            optimizer_modules.append(("logic_net", self.logic_net))
        self.optimizer = optim.AdamW(
            adamw_decay_param_groups_for_modules(
                tuple(optimizer_modules),
                weight_decay=float(config.optimizer_weight_decay),
            ),
            lr=config.learning_rate,
            weight_decay=0.0,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, config.epochs),
            eta_min=config.scheduler_eta_min,
        )
        self.global_step = 0
        # Keep -1 until the outer training loop assigns the first epoch index.
        self.epoch = -1

    def _create_logic_net(self) -> LogicNet:
        pathfinder = str(getattr(self.config, "logic_grid_pathfinder", "bellman_ford")).strip().lower()
        if pathfinder == "conv":
            pathfinder = "cnn"
        if pathfinder == "learnable":
            pathfinder = "vin"
        return LogicNet(
            latent_dim=int(self.config.latent_dim),
            num_tile_classes=int(self.config.num_classes),
            num_iterations=int(getattr(self.config, "num_logic_iterations", 30)),
            global_reach_weight=float(getattr(self.config, "logic_global_reach_weight", 1.0)),
            global_room_weight=float(getattr(self.config, "logic_global_room_weight", 0.25)),
            topology_trace_weight=float(getattr(self.config, "logic_topology_trace_weight", 0.25)),
            topology_anchor_weight=float(getattr(self.config, "logic_topology_anchor_weight", 0.25)),
            grid_pathfinder_type=pathfinder,
        )

    @staticmethod
    def _to_token_ids(real_maps: torch.Tensor, num_classes: int) -> torch.Tensor:
        if real_maps.dim() != 4 or int(real_maps.shape[1]) != 1:
            raise ValueError(f"Expected room tensors [B,1,H,W], got {tuple(real_maps.shape)}")
        tile_ids = (real_maps.squeeze(1) * float(num_classes - 1)).round().long()
        return tile_ids.clamp_(0, num_classes - 1)

    @staticmethod
    def _encode_edge_features(graph_dict: dict, device: torch.device) -> Optional[torch.Tensor]:
        edge_features = graph_dict.get("edge_features")
        if edge_features is not None:
            if not isinstance(edge_features, torch.Tensor):
                edge_features = torch.tensor(edge_features, dtype=torch.float32)
            edge_features = edge_features.to(device, dtype=torch.float32)
            if edge_features.numel() == 0:
                return None
            if edge_features.dim() == 1:
                edge_features = edge_features.unsqueeze(-1)
            return edge_features

        edge_attr = graph_dict.get("edge_attr")
        if edge_attr is None:
            return None
        if not isinstance(edge_attr, torch.Tensor):
            edge_attr = torch.tensor(edge_attr, dtype=torch.long)
        edge_attr = edge_attr.to(device)
        if edge_attr.numel() == 0:
            return None
        num_edge_types = GRAPH_EDGE_FEATURE_DIM
        return F.one_hot(edge_attr.clamp(0, num_edge_types - 1), num_classes=num_edge_types).float()

    def _stack_conditioning_vectors(self, cond_vectors: List[torch.Tensor]) -> torch.Tensor:
        if not cond_vectors:
            raise ValueError("cond_vectors must be non-empty")
        if self.config.graph_conditioning_mode == "node_sequence":
            max_nodes = max(int(c.shape[0]) for c in cond_vectors)
            padded = []
            for c in cond_vectors:
                if int(c.shape[0]) < max_nodes:
                    pad = torch.zeros(max_nodes - int(c.shape[0]), int(c.shape[1]), device=c.device, dtype=c.dtype)
                    c = torch.cat([c, pad], dim=0)
                padded.append(c.unsqueeze(0))
            return torch.cat(padded, dim=0)
        return torch.cat(cond_vectors, dim=0)

    def _encode_graph_conditioning(self, graph_dict: dict) -> torch.Tensor:
        node_features = graph_dict["node_features"].to(self.device)
        edge_index = graph_dict["edge_index"].to(self.device)
        edge_features = self._encode_edge_features(graph_dict, self.device)
        tpe = align_nodewise_tensor(
            graph_dict.get("tpe"),
            num_nodes=int(node_features.shape[0]),
            feature_dim=8,
            device=self.device,
            dtype=torch.float32,
            feature_name="tpe",
            default_value=compute_rwse_features(
                edge_index,
                int(node_features.shape[0]),
                steps=8,
                device=self.device,
                dtype=torch.float32,
            ),
        )

        boundary_constraints = graph_dict.get("boundary_constraints")
        room_position = graph_dict.get("room_position")
        current_node_idx = graph_dict.get("current_node_idx")
        style_id = graph_dict.get("style_id")
        current_node_distance = None
        if bool(getattr(self.config, "use_current_node_distance_features", True)):
            current_node_distance = align_nodewise_tensor(
                graph_dict.get("current_node_distance"),
                num_nodes=int(node_features.shape[0]),
                feature_dim=4,
                device=self.device,
                dtype=torch.float32,
                feature_name="current_node_distance",
                default_value=compute_current_node_distance_features(
                    edge_index,
                    int(node_features.shape[0]),
                    current_node_idx=int(current_node_idx) if current_node_idx is not None else None,
                    device=self.device,
                    dtype=torch.float32,
                    max_distance=int(getattr(self.config, "current_node_distance_max", 8)),
                ),
            )
        has_room_anchor = bool(graph_dict.get("has_room_anchor", False)) or (
            isinstance(boundary_constraints, torch.Tensor)
            and isinstance(room_position, torch.Tensor)
        )
        if has_room_anchor:
            boundary_constraints = boundary_constraints.to(self.device, dtype=torch.float32)
            room_position = room_position.to(self.device, dtype=torch.float32)
            if boundary_constraints.dim() == 1:
                boundary_constraints = boundary_constraints.unsqueeze(0)
            if room_position.dim() == 1:
                room_position = room_position.unsqueeze(0)
            reference_room_maps = (
                graph_dict.get("neighbor_maps")
                if bool(getattr(self.config, "condition_use_reference_room_maps", False))
                else None
            )
            condition_out = self.condition_encoder(
                neighbor_latents={"N": None, "S": None, "E": None, "W": None},
                boundary_constraints=boundary_constraints,
                position=room_position,
                node_features=node_features,
                edge_index=edge_index,
                edge_features=edge_features,
                edge_rrwp=graph_dict.get("edge_rrwp"),
                tpe=tpe,
                current_node_distance=current_node_distance,
                node_mask=graph_dict.get("node_mask"),
                current_node_idx=int(current_node_idx) if current_node_idx is not None else None,
                reference_room_maps=reference_room_maps,
                style_id=style_id,
                return_global_tokens=self.config.graph_conditioning_mode == "node_sequence",
            )
            if self.config.graph_conditioning_mode == "node_sequence":
                if not isinstance(condition_out, tuple) or len(condition_out) != 2:
                    raise ValueError(
                        "Condition encoder must return (room_anchor, global_tokens) "
                        "when graph_conditioning_mode='node_sequence' and a room anchor is requested."
                    )
                room_anchor, c_global = condition_out
                if c_global.dim() == 3:
                    if int(c_global.shape[0]) != 1:
                        raise ValueError(
                            f"Expected a single-sample global token batch, got shape {tuple(c_global.shape)}."
                        )
                    c_global = c_global.squeeze(0)
                condition_out = torch.cat([room_anchor, c_global], dim=0)
            conditioning_out = condition_out
            if float(getattr(self.config, "puzzle_structure_dropout_prob", 0.0)) > 0.0:
                conditioning_out = apply_puzzle_structure_control_to_conditioning(
                    conditioning_out,
                    puzzle_structure_enabled=bool(graph_dict.get("puzzle_room_structure_enabled", True)),
                    graph_conditioning_mode=self.config.graph_conditioning_mode,
                )
            if bool(getattr(self.config, "puzzle_stage_conditioning_enabled", False)):
                conditioning_out = apply_puzzle_stage_control_to_conditioning(
                    conditioning_out,
                    puzzle_stage_condition=graph_dict.get("puzzle_stage_condition"),
                    graph_conditioning_mode=self.config.graph_conditioning_mode,
                    scale=float(getattr(self.config, "puzzle_stage_token_scale", DEFAULT_PUZZLE_STAGE_TOKEN_SCALE)),
                )
            return conditioning_out

        c_global = self.condition_encoder.encode_global_only(
            node_features,
            edge_index,
            edge_features=edge_features,
            edge_rrwp=graph_dict.get("edge_rrwp"),
            tpe=tpe,
            current_node_distance=current_node_distance,
            node_mask=graph_dict.get("node_mask"),
        )

        if self.config.graph_conditioning_mode == "node_sequence":
            default_anchor = self.condition_encoder.encode_local_only(
                neighbor_latents={"N": None, "S": None, "E": None, "W": None},
                boundary_constraints=torch.zeros(1, 8, device=self.device, dtype=torch.float32),
                position=torch.zeros(1, 2, device=self.device, dtype=torch.float32),
            )
            conditioning_out = torch.cat([default_anchor, c_global], dim=0)
            if float(getattr(self.config, "puzzle_structure_dropout_prob", 0.0)) > 0.0:
                conditioning_out = apply_puzzle_structure_control_to_conditioning(
                    conditioning_out,
                    puzzle_structure_enabled=bool(graph_dict.get("puzzle_room_structure_enabled", True)),
                    graph_conditioning_mode=self.config.graph_conditioning_mode,
                )
            if bool(getattr(self.config, "puzzle_stage_conditioning_enabled", False)):
                conditioning_out = apply_puzzle_stage_control_to_conditioning(
                    conditioning_out,
                    puzzle_stage_condition=graph_dict.get("puzzle_stage_condition"),
                    graph_conditioning_mode=self.config.graph_conditioning_mode,
                    scale=float(getattr(self.config, "puzzle_stage_token_scale", DEFAULT_PUZZLE_STAGE_TOKEN_SCALE)),
                )
            return conditioning_out
        conditioning_out = c_global.mean(dim=0, keepdim=True)
        if float(getattr(self.config, "puzzle_structure_dropout_prob", 0.0)) > 0.0:
            conditioning_out = apply_puzzle_structure_control_to_conditioning(
                conditioning_out,
                puzzle_structure_enabled=bool(graph_dict.get("puzzle_room_structure_enabled", True)),
                graph_conditioning_mode=self.config.graph_conditioning_mode,
            )
        if bool(getattr(self.config, "puzzle_stage_conditioning_enabled", False)):
            conditioning_out = apply_puzzle_stage_control_to_conditioning(
                conditioning_out,
                puzzle_stage_condition=graph_dict.get("puzzle_stage_condition"),
                graph_conditioning_mode=self.config.graph_conditioning_mode,
                scale=float(getattr(self.config, "puzzle_stage_token_scale", DEFAULT_PUZZLE_STAGE_TOKEN_SCALE)),
            )
        return conditioning_out

    def _normalize_graph_sample(self, graph_dict: dict) -> Dict[str, torch.Tensor]:
        node_features = graph_dict["node_features"]
        edge_index = graph_dict["edge_index"]
        if not isinstance(node_features, torch.Tensor):
            node_features = torch.tensor(node_features, dtype=torch.float32)
        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            node_features = node_features.to(self.device, dtype=torch.float32)
        edge_index = edge_index.to(self.device, dtype=torch.long)

        num_nodes = int(node_features.shape[0])
        num_edges = int(edge_index.shape[1]) if edge_index.dim() == 2 else 0
        current_node_idx = graph_dict.get("current_node_idx")
        if isinstance(current_node_idx, torch.Tensor):
            current_node_idx = int(current_node_idx.detach().flatten()[0].item()) if current_node_idx.numel() else 0
        elif current_node_idx is not None:
            current_node_idx = int(current_node_idx)
        else:
            current_node_idx = 0

        start_node_id = graph_dict.get("start_node_id", graph_dict.get("start_idx", -1))
        if isinstance(start_node_id, torch.Tensor):
            start_node_id = int(start_node_id.detach().flatten()[0].item()) if start_node_id.numel() else -1
        elif start_node_id is not None:
            start_node_id = int(start_node_id)
        else:
            start_node_id = -1

        target_idx = graph_dict.get("target_idx")
        if isinstance(target_idx, torch.Tensor):
            target_idx = int(target_idx.detach().flatten()[0].item()) if target_idx.numel() else -1
        elif target_idx is not None:
            target_idx = int(target_idx)
        elif node_features.shape[1] > 3:
            target_hits = torch.nonzero(node_features[:, 3] > 0.5, as_tuple=False).flatten()
            target_idx = int(target_hits[0].item()) if target_hits.numel() else -1
        else:
            target_idx = -1

        edge_features = self._encode_edge_features(graph_dict, self.device)
        edge_feature_dim = int(max(1, getattr(self.config, "edge_feature_dim", GRAPH_EDGE_FEATURE_DIM)))
        if not isinstance(edge_features, torch.Tensor):
            edge_features = torch.zeros(num_edges, edge_feature_dim, device=self.device, dtype=torch.float32)
        elif edge_features.dim() == 1:
            edge_features = edge_features.unsqueeze(-1)
        if int(edge_features.shape[0]) != num_edges:
            aligned = torch.zeros(num_edges, max(edge_feature_dim, int(edge_features.shape[-1])), device=self.device, dtype=torch.float32)
            rows = min(num_edges, int(edge_features.shape[0]))
            cols = min(int(aligned.shape[1]), int(edge_features.shape[-1]))
            if rows > 0 and cols > 0:
                aligned[:rows, :cols] = edge_features[:rows, :cols].to(self.device, dtype=torch.float32)
            edge_features = aligned

        edge_attr = graph_dict.get("edge_attr")
        if not isinstance(edge_attr, torch.Tensor):
            edge_attr = torch.tensor(edge_attr, dtype=torch.long) if edge_attr is not None else torch.zeros(0, dtype=torch.long)
        edge_attr = edge_attr.to(self.device, dtype=torch.long).flatten()
        if edge_attr.numel() < num_edges:
            edge_attr = F.pad(edge_attr, (0, num_edges - int(edge_attr.numel())), value=0)
        edge_attr = edge_attr[:num_edges]

        tpe = align_nodewise_tensor(
            graph_dict.get("tpe"),
            num_nodes=num_nodes,
            feature_dim=8,
            device=self.device,
            dtype=torch.float32,
            feature_name="tpe",
            default_value=compute_rwse_features(
                edge_index,
                num_nodes,
                steps=8,
                device=self.device,
                dtype=torch.float32,
            ),
        )
        current_node_distance = align_nodewise_tensor(
            graph_dict.get("current_node_distance"),
            num_nodes=num_nodes,
            feature_dim=4,
            device=self.device,
            dtype=torch.float32,
            feature_name="current_node_distance",
            default_value=compute_current_node_distance_features(
                edge_index,
                num_nodes,
                current_node_idx=int(current_node_idx) if current_node_idx is not None else None,
                device=self.device,
                dtype=torch.float32,
                max_distance=int(getattr(self.config, "current_node_distance_max", 8)),
            ),
        )

        node_positions = align_nodewise_tensor(
            graph_dict.get("node_positions"),
            num_nodes=num_nodes,
            feature_dim=2,
            device=self.device,
            dtype=torch.float32,
            feature_name="node_positions",
            default_value=build_default_node_positions(
                num_nodes,
                device=self.device,
                dtype=torch.float32,
            ),
        )

        node_mask = graph_dict.get("node_mask")
        if not isinstance(node_mask, torch.Tensor):
            node_mask = torch.ones(num_nodes, dtype=torch.float32)
        node_mask = node_mask.to(self.device, dtype=torch.float32)
        if node_mask.dim() == 2:
            node_mask = node_mask.squeeze(0)

        room_topology_map = graph_dict.get("room_topology_map")
        if isinstance(room_topology_map, torch.Tensor):
            room_topology_map = room_topology_map.to(self.device, dtype=torch.float32)
            if room_topology_map.dim() == 4:
                room_topology_map = room_topology_map.squeeze(0)

        boundary_constraints = graph_dict.get("boundary_constraints")
        if isinstance(boundary_constraints, torch.Tensor):
            boundary_constraints = boundary_constraints.to(self.device, dtype=torch.float32)
            if boundary_constraints.dim() == 2:
                boundary_constraints = boundary_constraints.squeeze(0)

        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_features": edge_features,
            "edge_attr": edge_attr,
            "tpe": tpe,
            "current_node_distance": current_node_distance,
            "node_positions": node_positions,
            "node_mask": node_mask,
            "current_node_idx": int(current_node_idx),
            "start_node_id": int(start_node_id),
            "target_idx": int(target_idx),
            "has_room_anchor": bool(graph_dict.get("has_room_anchor", False)) or (
                isinstance(graph_dict.get("boundary_constraints"), torch.Tensor)
                and isinstance(graph_dict.get("room_position"), torch.Tensor)
            ),
            **({"boundary_constraints": boundary_constraints} if isinstance(boundary_constraints, torch.Tensor) else {}),
            **({"room_topology_map": room_topology_map} if isinstance(room_topology_map, torch.Tensor) else {}),
        }

    def _stack_graph_batch(self, graph_list: List[dict]) -> Optional[Dict[str, torch.Tensor]]:
        if not graph_list:
            return None
        dungeon_graph = self._try_stack_dungeon_scope_graph_batch(graph_list)
        if dungeon_graph is not None:
            return dungeon_graph
        samples = [self._normalize_graph_sample(graph_dict) for graph_dict in graph_list]
        max_nodes = max(int(sample["node_features"].shape[0]) for sample in samples)
        feat_dim = max(int(sample["node_features"].shape[1]) for sample in samples)
        tpe_dim = max(int(sample["tpe"].shape[1]) for sample in samples)
        distance_dim = max(int(sample["current_node_distance"].shape[1]) for sample in samples)
        max_edges = max(int(sample["edge_index"].shape[1]) if sample["edge_index"].dim() == 2 else 0 for sample in samples)
        edge_feat_dim = max(int(sample["edge_features"].shape[1]) if sample["edge_features"].dim() == 2 else 0 for sample in samples)

        node_features_batch = torch.zeros(len(samples), max_nodes, feat_dim, device=self.device, dtype=torch.float32)
        tpe_batch = torch.zeros(len(samples), max_nodes, tpe_dim, device=self.device, dtype=torch.float32)
        current_node_distance_batch = torch.zeros(len(samples), max_nodes, distance_dim, device=self.device, dtype=torch.float32)
        node_positions_batch = torch.zeros(len(samples), max_nodes, 2, device=self.device, dtype=torch.float32)
        node_mask_batch = torch.zeros(len(samples), max_nodes, device=self.device, dtype=torch.float32)
        edge_index_batch = torch.full((len(samples), 2, max_edges), -1, device=self.device, dtype=torch.long)
        edge_features_batch = torch.zeros(len(samples), max_edges, max(1, edge_feat_dim), device=self.device, dtype=torch.float32)
        edge_attr_batch = torch.full((len(samples), max_edges), -1, device=self.device, dtype=torch.long)
        current_node_idx_batch = torch.zeros(len(samples), device=self.device, dtype=torch.long)
        start_node_id_batch = torch.full((len(samples),), -1, device=self.device, dtype=torch.long)
        target_idx_batch = torch.full((len(samples),), -1, device=self.device, dtype=torch.long)
        topo_maps = []
        boundary_rows = []
        for i, sample in enumerate(samples):
            n = int(sample["node_features"].shape[0])
            if n > 0:
                node_features_batch[i, :n] = sample["node_features"]
                tpe_batch[i, :n] = sample["tpe"]
                current_node_distance_batch[i, :n] = sample["current_node_distance"]
                node_positions_batch[i, :n] = sample["node_positions"]
                node_mask_batch[i, :n] = sample["node_mask"]
            e = int(sample["edge_index"].shape[1]) if sample["edge_index"].dim() == 2 else 0
            if e > 0:
                edge_index_batch[i, :, :e] = sample["edge_index"]
                edge_features_batch[i, :e, : sample["edge_features"].shape[1]] = sample["edge_features"]
                edge_attr_batch[i, :e] = sample["edge_attr"]
            current_node_idx_batch[i] = int(sample.get("current_node_idx", 0))
            start_node_id_batch[i] = int(sample.get("start_node_id", -1))
            target_idx_batch[i] = int(sample.get("target_idx", -1))
            topo = sample.get("room_topology_map")
            if isinstance(topo, torch.Tensor):
                topo_maps.append(topo.unsqueeze(0) if topo.dim() == 3 else topo)
            boundary = sample.get("boundary_constraints")
            if isinstance(boundary, torch.Tensor):
                boundary_rows.append(boundary.reshape(1, -1))

        batch_graph = {
            "node_features": node_features_batch,
            "edge_index": edge_index_batch,
            "edge_features": edge_features_batch,
            "edge_attr": edge_attr_batch,
            "tpe": tpe_batch,
            "current_node_distance": current_node_distance_batch,
            "node_positions": node_positions_batch,
            "node_mask": node_mask_batch,
            "current_node_idx": current_node_idx_batch,
            "start_node_id": start_node_id_batch,
            "target_idx": target_idx_batch,
            "graph_scope": "room_batch",
            "has_room_anchor": bool(self.config.graph_conditioning_mode == "node_sequence") or bool(samples[0].get("has_room_anchor", False)),
        }
        if len(topo_maps) == len(samples):
            batch_graph["room_topology_map"] = torch.cat(topo_maps, dim=0)
        if len(boundary_rows) == len(samples):
            batch_graph["boundary_constraints"] = torch.cat(boundary_rows, dim=0)
        return batch_graph

    def _try_stack_dungeon_scope_graph_batch(self, graph_list: List[dict]) -> Optional[Dict[str, torch.Tensor]]:
        if not graph_list:
            return None
        node_counts = [int(g.get("num_nodes", 0)) for g in graph_list]
        if not node_counts or min(node_counts) <= 0 or len(set(node_counts)) != 1:
            return None
        num_nodes = int(node_counts[0])
        if len(graph_list) != num_nodes:
            return None

        first_node_map = dict(graph_list[0].get("node_to_idx", {}))
        current_indices: List[int] = []
        for graph in graph_list:
            if dict(graph.get("node_to_idx", {})) != first_node_map:
                return None
            current = graph.get("current_node_idx")
            if isinstance(current, torch.Tensor):
                current = int(current.detach().flatten()[0].item()) if current.numel() else -1
            elif current is None:
                return None
            else:
                current = int(current)
            current_indices.append(current)
        if sorted(current_indices) != list(range(num_nodes)):
            return None

        sample = self._normalize_graph_sample(graph_list[0])
        topo_by_node: Dict[int, torch.Tensor] = {}
        boundary_by_node: Dict[int, torch.Tensor] = {}
        for graph, current in zip(graph_list, current_indices):
            normalized = self._normalize_graph_sample(graph)
            topo = normalized.get("room_topology_map")
            if isinstance(topo, torch.Tensor):
                topo_by_node[int(current)] = topo.unsqueeze(0) if topo.dim() == 3 else topo
            boundary = normalized.get("boundary_constraints")
            if isinstance(boundary, torch.Tensor):
                boundary_by_node[int(current)] = boundary.reshape(1, -1)

        node_mask = sample.get("node_mask")
        if not isinstance(node_mask, torch.Tensor):
            node_mask = torch.ones(num_nodes, device=self.device, dtype=torch.float32)
        batch_graph = {
            "node_features": sample["node_features"],
            "edge_index": sample["edge_index"],
            "edge_features": sample["edge_features"],
            "edge_attr": sample["edge_attr"],
            "tpe": sample["tpe"],
            "current_node_distance": sample["current_node_distance"],
            "node_positions": sample["node_positions"],
            "node_mask": node_mask,
            "current_node_idx": torch.tensor(current_indices, device=self.device, dtype=torch.long),
            "start_node_id": torch.tensor(int(sample.get("start_node_id", -1)), device=self.device, dtype=torch.long),
            "target_idx": torch.tensor(int(sample.get("target_idx", -1)), device=self.device, dtype=torch.long),
            "graph_scope": "dungeon",
            "has_room_anchor": bool(self.config.graph_conditioning_mode == "node_sequence") or bool(sample.get("has_room_anchor", False)),
        }
        if len(topo_by_node) == len(graph_list):
            batch_graph["room_topology_map"] = torch.cat([topo_by_node[i] for i in current_indices], dim=0)
        if len(boundary_by_node) == len(graph_list):
            batch_graph["boundary_constraints"] = torch.cat([boundary_by_node[i] for i in current_indices], dim=0)
        return batch_graph

    @staticmethod
    def _tensor_is_finite(value: Any) -> bool:
        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all().item())
        try:
            return bool(torch.isfinite(torch.as_tensor(float(value))).item())
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _logic_loss_to_solvability_proxy(logic_loss: torch.Tensor) -> torch.Tensor:
        if not isinstance(logic_loss, torch.Tensor):
            logic_loss = torch.tensor(float(logic_loss), dtype=torch.float32)
        return torch.exp(-logic_loss.detach().clamp_min(0.0))

    @staticmethod
    def _logic_info_scalar(info: Dict[str, Any], key: str, default: float = 0.0) -> float:
        value = info.get(key, default)
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return float(default)
            return float(value.detach().float().mean().item())
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _gradients_are_finite(parameters: List[torch.nn.Parameter]) -> bool:
        for param in parameters:
            if param.grad is not None and not bool(torch.isfinite(param.grad).all().item()):
                return False
        return True

    def _step(
        self,
        real_maps: torch.Tensor,
        graph_list: Optional[List[dict]],
        *,
        train: bool,
    ) -> Dict[str, float]:
        token_ids = self._to_token_ids(real_maps.to(self.device), num_classes=self.config.num_classes)
        if graph_list:
            cond_vectors = [self._encode_graph_conditioning(graph_dict) for graph_dict in graph_list]
            conditioning = self._stack_conditioning_vectors(cond_vectors)
            graph_batch = self._stack_graph_batch(graph_list)
            topo = graph_batch.get("room_topology_map") if isinstance(graph_batch, dict) else None
        else:
            conditioning = torch.zeros(token_ids.shape[0], 1, self.config.context_dim, device=self.device)
            graph_batch = None
            topo = None

        fixed_tokens, fixed_mask = DiscreteMaskedRoomModel.build_fixed_mask_from_topology_map(
            token_ids,
            topo,
            num_classes=self.config.num_classes,
            semantic_anchor_threshold=self.config.semantic_anchor_threshold,
        )
        topology_focus_map = None
        if topo is not None and float(getattr(self.config, "topology_alignment_weight", 0.0)) > 0.0:
            topology_focus_map = build_topology_loss_focus_map(
                topo,
                semantic_anchor_threshold=self.config.semantic_anchor_threshold,
                marker_weight=float(getattr(self.config, "topology_marker_weight", 2.0)),
                trace_weight=float(getattr(self.config, "topology_trace_weight", 0.75)),
                dilation=int(getattr(self.config, "topology_focus_dilation", 1)),
            )
        need_puzzle_stage_semantics = bool(
            graph_list and float(getattr(self.config, "puzzle_stage_semantics_loss_weight", 0.0)) > 0.0
        )
        need_logic_supervision = bool(
            self.logic_net is not None
            and graph_batch is not None
            and float(getattr(self.config, "alpha_logic", 0.0)) > 0.0
        )
        if need_puzzle_stage_semantics or need_logic_supervision:
            loss, metrics, aux = self.model.training_loss(
                token_ids,
                conditioning,
                graph_data=graph_batch,
                fixed_tokens=fixed_tokens,
                fixed_mask=fixed_mask,
                min_mask_ratio=self.config.min_mask_ratio,
                max_mask_ratio=self.config.max_mask_ratio,
                topology_focus_map=topology_focus_map,
                topology_alignment_weight=float(getattr(self.config, "topology_alignment_weight", 0.0)),
                return_aux=True,
            )
        else:
            loss, metrics = self.model.training_loss(
                token_ids,
                conditioning,
                graph_data=graph_batch,
                fixed_tokens=fixed_tokens,
                fixed_mask=fixed_mask,
                min_mask_ratio=self.config.min_mask_ratio,
                max_mask_ratio=self.config.max_mask_ratio,
                topology_focus_map=topology_focus_map,
                topology_alignment_weight=float(getattr(self.config, "topology_alignment_weight", 0.0)),
            )
            aux = {}
        puzzle_stage_semantic_loss = torch.zeros((), device=self.device, dtype=loss.dtype)
        puzzle_stage_semantic_metrics: Dict[str, float] = {
            "puzzle_stage_semantic_loss": 0.0,
            "puzzle_stage_gate_loss": 0.0,
            "puzzle_stage_sequence_loss": 0.0,
            "puzzle_stage_count_loss": 0.0,
            "puzzle_stage_slot_loss": 0.0,
            "puzzle_stage_gate_acc": 0.0,
            "puzzle_stage_sequence_acc": 0.0,
            "puzzle_stage_count_acc": 0.0,
            "puzzle_stage_slot_acc": 0.0,
        }
        if need_puzzle_stage_semantics and isinstance(aux.get("logits"), torch.Tensor):
            puzzle_stage_semantic_loss, puzzle_stage_semantic_metrics = self.puzzle_stage_semantics_head.compute_loss(
                aux["logits"],
                [graph_dict.get("puzzle_stage_condition") if isinstance(graph_dict, dict) else {} for graph_dict in graph_list],
            )

        logic_loss = torch.zeros((), device=self.device, dtype=loss.dtype)
        logic_info: Dict[str, Any] = {}
        solvability_proxy = torch.zeros((), device=self.device, dtype=loss.dtype)
        if need_logic_supervision and isinstance(aux.get("logits"), torch.Tensor):
            logic_loss, logic_info = self.logic_net(aux["logits"], graph_data=graph_batch)
            if isinstance(logic_loss, torch.Tensor) and logic_loss.numel() != 1:
                logic_loss = logic_loss.mean()
            if self._tensor_is_finite(logic_loss):
                solvability_proxy = self._logic_loss_to_solvability_proxy(logic_loss).to(device=self.device, dtype=loss.dtype)
            else:
                logic_info = dict(logic_info)
                logic_info["global_graph_skipped"] = logic_info.get("global_graph_skipped", "nonfinite_logic_loss")
                logic_loss = torch.zeros((), device=self.device, dtype=loss.dtype)
                solvability_proxy = torch.zeros((), device=self.device, dtype=loss.dtype)

        total_loss = (
            loss
            + float(getattr(self.config, "puzzle_stage_semantics_loss_weight", 0.0)) * puzzle_stage_semantic_loss
            + float(getattr(self.config, "alpha_logic", 0.0)) * logic_loss
        )
        graph_skip_reason = str(logic_info.get("global_graph_skipped", "") or "")
        graph_loss_attempted = bool(need_logic_supervision)
        logic_metrics = {
            "logic_loss": float(logic_loss.detach().item()) if self._tensor_is_finite(logic_loss) else 0.0,
            "logic_loss_contribution": (
                float(getattr(self.config, "alpha_logic", 0.0)) * float(logic_loss.detach().item())
                if self._tensor_is_finite(logic_loss)
                else 0.0
            ),
            "solvability_proxy": float(solvability_proxy.detach().item()) if self._tensor_is_finite(solvability_proxy) else 0.0,
            "solvability": float(solvability_proxy.detach().item()) if self._tensor_is_finite(solvability_proxy) else 0.0,
            "logic_global_graph_loss_skipped": 1.0 if graph_loss_attempted and graph_skip_reason else 0.0,
            "logic_global_graph_supervised": 1.0 if graph_loss_attempted and not graph_skip_reason else 0.0,
            "logic_global_graph_node_coverage": self._logic_info_scalar(logic_info, "global_graph_node_coverage", 0.0),
            "logic_global_graph_reachability": self._logic_info_scalar(logic_info, "global_graph_reachability", 0.0),
            "logic_global_room_passability": self._logic_info_scalar(logic_info, "global_room_passability", 0.0),
        }
        if train:
            if not self._tensor_is_finite(total_loss):
                self.optimizer.zero_grad(set_to_none=True)
                metrics = dict(metrics)
                metrics["loss"] = 0.0
                metrics.update(puzzle_stage_semantic_metrics)
                metrics.update(logic_metrics)
                metrics["skipped_nonfinite_batch"] = 1.0
                return metrics
            self.optimizer.zero_grad()
            total_loss.backward()
            trainable_params = [
                param
                for module in (
                    self.model,
                    self.condition_encoder,
                    self.puzzle_stage_semantics_head,
                    self.logic_net if bool(getattr(self.config, "logic_net_trainable", False)) else None,
                )
                if module is not None
                for param in module.parameters()
                if param.requires_grad
            ]
            if not self._gradients_are_finite(trainable_params):
                self.optimizer.zero_grad(set_to_none=True)
                metrics = dict(metrics)
                metrics["loss"] = float(total_loss.detach().item())
                metrics.update(puzzle_stage_semantic_metrics)
                metrics.update(logic_metrics)
                metrics["skipped_nonfinite_batch"] = 1.0
                return metrics
            if self.config.grad_clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params,
                    max_norm=self.config.grad_clip_norm,
                )
                if not self._tensor_is_finite(grad_norm):
                    self.optimizer.zero_grad(set_to_none=True)
                    metrics = dict(metrics)
                    metrics["loss"] = float(total_loss.detach().item())
                    metrics.update(puzzle_stage_semantic_metrics)
                    metrics.update(logic_metrics)
                    metrics["skipped_nonfinite_batch"] = 1.0
                    return metrics
            self.optimizer.step()
            self.global_step += 1
        metrics = dict(metrics)
        metrics["loss"] = float(total_loss.detach().item())
        metrics.update(puzzle_stage_semantic_metrics)
        metrics.update(logic_metrics)
        metrics["skipped_nonfinite_batch"] = 0.0
        return metrics

    def _build_resume_checkpoint_payload(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        topology_anchor_policy = build_topology_anchor_policy_metadata(
            semantic_role_prior_strength=self.config.semantic_role_prior_strength,
            semantic_anchor_threshold=self.config.semantic_anchor_threshold,
            semantic_puzzle_offset=self.config.semantic_puzzle_offset,
            topology_supervision_mode=self.config.topology_supervision_mode,
        )
        return {
            "epoch": int(self.epoch),
            "global_step": int(self.global_step),
            "model_state_dict": self.model.state_dict(),
            "condition_encoder_state_dict": self.condition_encoder.state_dict(),
            "puzzle_stage_semantics_head_state_dict": self.puzzle_stage_semantics_head.state_dict(),
            **(
                {"logic_net_state_dict": self.logic_net.state_dict()}
                if self.logic_net is not None
                else {}
            ),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config.to_dict(),
            "metrics": dict(metrics),
            "metadata": {
                "topology_anchor_policy": dict(topology_anchor_policy),
            },
        }

    def _build_inference_checkpoint_payload(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        topology_anchor_policy = build_topology_anchor_policy_metadata(
            semantic_role_prior_strength=self.config.semantic_role_prior_strength,
            semantic_anchor_threshold=self.config.semantic_anchor_threshold,
            semantic_puzzle_offset=self.config.semantic_puzzle_offset,
            topology_supervision_mode=self.config.topology_supervision_mode,
        )
        return {
            "epoch": int(self.epoch),
            "global_step": int(self.global_step),
            "model_state_dict": self.model.state_dict(),
            "condition_encoder_state_dict": self.condition_encoder.state_dict(),
            "puzzle_stage_semantics_head_state_dict": self.puzzle_stage_semantics_head.state_dict(),
            **(
                {"logic_net_state_dict": self.logic_net.state_dict()}
                if self.logic_net is not None
                else {}
            ),
            "config": self.config.to_dict(),
            "metrics": dict(metrics),
            "metadata": {
                "topology_anchor_policy": dict(topology_anchor_policy),
            },
        }

    def save_checkpoint(self, path: str, metrics: Dict[str, Any], *, include_optimizer: bool = True) -> None:
        payload = (
            self._build_resume_checkpoint_payload(metrics)
            if bool(include_optimizer)
            else self._build_inference_checkpoint_payload(metrics)
        )
        atomic_torch_save(payload, path)
        checkpoint_kind = "resume" if include_optimizer else "inference"
        contains = ["model", "condition_encoder", "puzzle_stage_semantics_head"]
        if self.logic_net is not None:
            contains.append("logic_net")
        if include_optimizer:
            contains.extend(["optimizer", "scheduler"])
        write_checkpoint_metadata(
            path,
            model_type="masked_room_resume" if include_optimizer else "masked_room_model",
            architecture={
                "num_classes": int(self.config.num_classes),
                "latent_dim": int(self.config.latent_dim),
                "hidden_dim": int(self.model.hidden_dim),
                "model_channels": int(self.config.model_channels),
                "context_dim": int(self.model.context_dim),
                "masked_steps": int(self.config.masked_steps),
                "attention_mode": str(self.config.attention_mode),
                "context_attention_mode": str(getattr(self.config, "context_attention_mode", "concat_encoder")),
                "topology_conditioning_mode": str(self.config.topology_conditioning_mode),
                "hedgehog_feature_dim": int(self.config.hedgehog_feature_dim),
                "graph_auto_linear_attention_nodes": int(self.config.graph_auto_linear_attention_nodes),
                "spatial_graph_gate_init": float(self.config.spatial_graph_gate_init),
                "spatial_topology_gate_init": float(self.config.spatial_topology_gate_init),
                "unet_channel_mult": list(self.config.unet_channel_mult),
                "unet_num_res_blocks": int(self.config.unet_num_res_blocks),
                "unet_attention_resolutions": list(self.config.unet_attention_resolutions),
                "unet_num_heads": int(self.config.unet_num_heads),
                "unet_dropout": float(self.config.unet_dropout),
                "min_mask_ratio": float(self.config.min_mask_ratio),
                "max_mask_ratio": float(self.config.max_mask_ratio),
                "semantic_anchor_threshold": float(self.config.semantic_anchor_threshold),
                "topology_alignment_weight": float(self.config.topology_alignment_weight),
                "topology_marker_weight": float(self.config.topology_marker_weight),
                "topology_trace_weight": float(self.config.topology_trace_weight),
                "topology_focus_dilation": int(self.config.topology_focus_dilation),
                "logic_net_enabled": bool(getattr(self.config, "logic_net_enabled", False)),
                "logic_net_trainable": bool(getattr(self.config, "logic_net_trainable", False)),
                "alpha_logic": float(getattr(self.config, "alpha_logic", 0.0)),
                "logic_grid_pathfinder": str(getattr(self.config, "logic_grid_pathfinder", "bellman_ford")),
                "num_logic_iterations": int(getattr(self.config, "num_logic_iterations", 30)),
            },
            extra={
                "graph_conditioning_mode": self.config.graph_conditioning_mode,
                "epoch": int(self.epoch),
                "global_step": int(self.global_step),
                "checkpoint_kind": checkpoint_kind,
                "contains": contains,
                "topology_anchor_policy": build_topology_anchor_policy_metadata(
                    semantic_role_prior_strength=self.config.semantic_role_prior_strength,
                    semantic_anchor_threshold=self.config.semantic_anchor_threshold,
                    semantic_puzzle_offset=self.config.semantic_puzzle_offset,
                    topology_supervision_mode=self.config.topology_supervision_mode,
                ),
            },
        )
        log_checkpoint_artifact(
            logger,
            path,
            checkpoint_dir=Path(path).parent,
            label="Saved masked-room checkpoint",
        )

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        checkpoint = safe_torch_load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.condition_encoder.load_state_dict(checkpoint["condition_encoder_state_dict"])
        if "puzzle_stage_semantics_head_state_dict" in checkpoint:
            self.puzzle_stage_semantics_head.load_state_dict(checkpoint["puzzle_stage_semantics_head_state_dict"])
        if self.logic_net is not None and "logic_net_state_dict" in checkpoint:
            self.logic_net.load_state_dict(checkpoint["logic_net_state_dict"], strict=False)
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = int(checkpoint.get("epoch", 0))
        self.global_step = int(checkpoint.get("global_step", 0))
        logger.info("Loaded masked-room checkpoint from %s (epoch %d)", path, self.epoch)
        return checkpoint


def train_masked_room(config: MaskedRoomTrainingConfig) -> MaskedRoomTrainer:
    resolved_seed = seed_everything(int(getattr(config, "seed", 42)))
    logger.info("Masked-room trainer seeds initialized: seed=%d", resolved_seed)
    trainer = MaskedRoomTrainer(config)
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, eval_split_name, train_size, eval_size = _create_masked_room_dataloaders(config)
    log_capacity_guardrails(
        logger,
        stage_name="Masked-room trainer",
        dataset_size=int(train_size),
        param_groups={
            "masked_room_model": count_parameters(trainer.model, trainable_only=True),
            "condition_encoder": count_parameters(trainer.condition_encoder, trainable_only=True),
            "puzzle_stage_semantics_head": count_parameters(trainer.puzzle_stage_semantics_head, trainable_only=True),
            **(
                {"logic_net": count_parameters(trainer.logic_net, trainable_only=True)}
                if trainer.logic_net is not None
                else {}
            ),
        },
        recommended_config="configs/zelda_hmolqd_masked_small.yaml",
        capacity_knobs=(
            "masked_room.model_channels, masked_room.hidden_dim, "
            "masked_room.condition_hidden_dim, masked_room.condition_num_gnn_layers, "
            "masked_room.unet_channel_mult, masked_room.unet_num_res_blocks, "
            "masked_room.unet_num_heads"
        ),
    )

    metrics_logger = MetricsLogger(
        log_dir=str(checkpoint_dir / "logs"),
        experiment_name="masked_room_training",
    )
    if float(getattr(config, "validation_fraction", 0.0)) > 0.0:
        best_metric_name = _resolve_masked_room_best_metric_name(config)
    else:
        best_metric_name = "train_loss"
    if config.best_checkpoint_metric == "train_loss":
        best_metric_name = "train_loss"
    best_metric_value = float("inf")
    epoch_metrics: Dict[str, Any] = {}
    logger.info(
        "Masked-room split: train=%d rooms | %s=%d rooms | final_test_dungeons=%s | best_metric=%s",
        int(train_size),
        eval_split_name,
        int(eval_size),
        list(getattr(config, "test_dungeon_ids", [9])),
        best_metric_name,
    )
    resume_path = resolve_resume_checkpoint(
        explicit_path=getattr(config, "resume_checkpoint", None),
        checkpoint_dir=str(checkpoint_dir),
        auto_resume=bool(getattr(config, "auto_resume", True)),
        latest_filename=LATEST_RESUME_FILENAME,
    )
    if resume_path is not None:
        try:
            resume_payload = trainer.load_checkpoint(str(resume_path))
        except (RuntimeError, ValueError) as exc:
            if getattr(config, "resume_checkpoint", None):
                raise
            logger.warning(
                "Skipping auto-resume masked-room checkpoint at %s because it is incompatible with the current architecture: %s",
                resume_path,
                exc,
            )
        else:
            latest_metrics = resume_payload.get("metrics", {})
            if isinstance(latest_metrics, dict):
                best_metric_name = str(latest_metrics.get("best_metric_name", best_metric_name))
                best_metric_value = float(
                    latest_metrics.get("best_metric_value", latest_metrics.get("best_val_loss", best_metric_value))
                )
            logger.info("Auto-resumed masked-room training from %s", resume_path)

    for epoch in range(int(getattr(trainer, "epoch", -1)) + 1, config.epochs):
        trainer.epoch = int(epoch)
        batch_sampler = getattr(train_loader, "batch_sampler", None)
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(int(epoch))
        trainer.model.train()
        trainer.condition_encoder.train()
        trainer.puzzle_stage_semantics_head.train()
        if trainer.logic_net is not None:
            trainer.logic_net.train(bool(getattr(config, "logic_net_trainable", False)))
        train_sum = {
            "loss": 0.0,
            "base_loss": 0.0,
            "mask_ratio": 0.0,
            "masked_fraction": 0.0,
            "topology_focus_loss": 0.0,
            "topology_focus_fraction": 0.0,
            "puzzle_stage_semantic_loss": 0.0,
            "puzzle_stage_gate_loss": 0.0,
            "puzzle_stage_sequence_loss": 0.0,
            "puzzle_stage_count_loss": 0.0,
            "puzzle_stage_slot_loss": 0.0,
            "puzzle_stage_gate_acc": 0.0,
            "puzzle_stage_sequence_acc": 0.0,
            "puzzle_stage_count_acc": 0.0,
            "puzzle_stage_slot_acc": 0.0,
        }
        train_batches = 0
        for batch in train_loader:
            real_maps, graph_list = batch if isinstance(batch, (list, tuple)) and len(batch) == 2 else (batch, None)
            if graph_list is not None and float(getattr(config, "puzzle_structure_dropout_prob", 0.0)) > 0.0:
                real_maps, graph_list = apply_puzzle_structure_dropout_batch(
                    real_maps,
                    graph_list,
                    num_classes=int(config.num_classes),
                    dropout_prob=float(config.puzzle_structure_dropout_prob),
                )
            metrics = trainer._step(real_maps, graph_list, train=True)
            for key, value in metrics.items():
                train_sum[key] = float(train_sum.get(key, 0.0)) + float(value)
            train_batches += 1

        trainer.model.eval()
        trainer.condition_encoder.eval()
        trainer.puzzle_stage_semantics_head.eval()
        if trainer.logic_net is not None:
            trainer.logic_net.eval()
        val_sum: Dict[str, float] = {}
        val_batches = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                real_maps, graph_list = batch if isinstance(batch, (list, tuple)) and len(batch) == 2 else (batch, None)
                metrics = trainer._step(real_maps, graph_list, train=False)
                for key, value in metrics.items():
                    metric_key = f"val_{key}"
                    val_sum[metric_key] = float(val_sum.get(metric_key, 0.0)) + float(value)
                val_batches += 1
                if batch_idx + 1 >= int(getattr(config, "validation_max_batches", 16)):
                    break

        if train_batches > 0:
            trainer.scheduler.step()
        else:
            logger.warning(
                "Skipping masked-room scheduler step for epoch %d because no train batches were processed.",
                epoch,
            )
        epoch_metrics = {
            "epoch": epoch,
            "eval_split": eval_split_name,
            **{k: v / max(1, train_batches) for k, v in train_sum.items()},
            **{k: v / max(1, val_batches) for k, v in val_sum.items()},
        }
        if val_batches <= 0:
            for key in train_sum:
                epoch_metrics.setdefault(f"val_{key}", float("inf"))
        metrics_logger.log(epoch_metrics)
        logger.info(
            "Masked room epoch %d/%d: loss=%.4f val_loss=%.4f val_topo=%.4f",
            epoch + 1,
            config.epochs,
            epoch_metrics["loss"],
            epoch_metrics["val_loss"],
            epoch_metrics["val_topology_focus_loss"],
        )
        if (epoch + 1) % config.save_every == 0:
            trainer.save_checkpoint(
                str(checkpoint_dir / f"masked_room_resume_epoch_{epoch + 1:04d}.pth"),
                epoch_metrics,
                include_optimizer=True,
            )
            prune_checkpoints(
                checkpoint_dir=str(checkpoint_dir),
                pattern="masked_room_resume_epoch_*.pth",
                keep_last=int(getattr(config, "keep_last", 2)),
            )
        if best_metric_name == "val_topology_focus_loss":
            current_metric_value = float(epoch_metrics["val_topology_focus_loss"])
        elif best_metric_name == "val_puzzle_stage_semantic_loss":
            current_metric_value = float(epoch_metrics["val_puzzle_stage_semantic_loss"])
        elif best_metric_name == "val_loss":
            current_metric_value = float(epoch_metrics["val_loss"])
        else:
            current_metric_value = float(epoch_metrics["loss"])
        if current_metric_value < best_metric_value:
            best_metric_value = current_metric_value
            trainer.save_checkpoint(str(checkpoint_dir / "masked_room_best.pth"), epoch_metrics, include_optimizer=False)

        latest_metrics = dict(epoch_metrics)
        latest_metrics["best_metric_name"] = str(best_metric_name)
        latest_metrics["best_metric_value"] = float(best_metric_value)
        latest_metrics["best_val_loss"] = float(
            best_metric_value if best_metric_name == "val_loss" else epoch_metrics["val_loss"]
        )
        trainer.save_checkpoint(
            str(checkpoint_dir / LATEST_RESUME_FILENAME),
            latest_metrics,
            include_optimizer=True,
        )
        enforce_checkpoint_storage_budget(
            logger,
            checkpoint_dir=checkpoint_dir,
            budget_gb=getattr(config, "checkpoint_storage_budget_gb", None),
            warning_fraction=float(getattr(config, "checkpoint_storage_warning_fraction", 0.8)),
            cleanup_enabled=bool(getattr(config, "checkpoint_storage_cleanup_enabled", True)),
            cleanup_target_fraction=float(getattr(config, "checkpoint_storage_cleanup_target_fraction", 0.6)),
            removable_patterns=("masked_room_resume_epoch_*.pth",),
        )

    trainer.save_checkpoint(str(checkpoint_dir / "masked_room_final.pth"), epoch_metrics, include_optimizer=False)
    metrics_logger.save()
    return trainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the graph-conditioned discrete masked room model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config path using the shared validated config system. "
             "When provided, omitted legacy flags inherit values from that config.",
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--context-dim", type=int, default=None)
    parser.add_argument("--condition-hidden-dim", type=int, default=None)
    parser.add_argument("--condition-num-gnn-layers", type=int, default=None)
    parser.add_argument("--condition-num-attention-heads", type=int, default=None)
    parser.add_argument("--condition-dropout", type=float, default=None)
    parser.add_argument("--condition-gnn-type", type=str, default=None, choices=["gcn", "gat", "sage", "gps"])
    parser.add_argument("--condition-use-reference-room-maps", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--condition-reference-tile-vocab-size", type=int, default=None)
    parser.add_argument("--condition-reference-embedding-dim", type=int, default=None)
    parser.add_argument("--condition-reference-hidden-dim", type=int, default=None)
    parser.add_argument("--graph-conditioning-mode", type=str, default=None)
    parser.add_argument("--use-current-node-distance-features", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--current-node-distance-max", type=int, default=None)
    parser.add_argument("--model-channels", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--masked-steps", type=int, default=None)
    parser.add_argument("--attention-mode", type=str, default=None, choices=["softmax", "linear_hedgehog"])
    parser.add_argument("--context-attention-mode", type=str, default=None, choices=["concat_encoder", "cross_decoder"])
    parser.add_argument("--topology-conditioning-mode", type=str, default=None, choices=["additive", "spade"])
    parser.add_argument("--hedgehog-feature-dim", type=int, default=None)
    parser.add_argument("--graph-auto-linear-attention-nodes", type=int, default=None)
    parser.add_argument("--spatial-graph-gate-init", type=float, default=None)
    parser.add_argument("--spatial-topology-gate-init", type=float, default=None)
    parser.add_argument("--unet-channel-mult", type=int, nargs="+", default=None)
    parser.add_argument("--unet-num-res-blocks", type=int, default=None)
    parser.add_argument("--unet-attention-resolutions", type=int, nargs="+", default=None)
    parser.add_argument("--unet-num-heads", type=int, default=None)
    parser.add_argument("--unet-dropout", type=float, default=None)
    parser.add_argument("--min-mask-ratio", type=float, default=None)
    parser.add_argument("--max-mask-ratio", type=float, default=None)
    parser.add_argument("--topology-alignment-weight", type=float, default=None)
    parser.add_argument("--topology-marker-weight", type=float, default=None)
    parser.add_argument("--topology-trace-weight", type=float, default=None)
    parser.add_argument("--topology-focus-dilation", type=int, default=None)
    parser.add_argument("--logic-net-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--logic-net-trainable", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--alpha-logic", type=float, default=None)
    parser.add_argument("--logic-global-reach-weight", type=float, default=None)
    parser.add_argument("--logic-global-room-weight", type=float, default=None)
    parser.add_argument("--logic-topology-trace-weight", type=float, default=None)
    parser.add_argument("--logic-topology-anchor-weight", type=float, default=None)
    parser.add_argument(
        "--logic-grid-pathfinder",
        type=str,
        choices=["bellman_ford", "conv", "cnn", "vin", "learnable", "perturb_and_map"],
        default=None,
    )
    parser.add_argument("--num-logic-iterations", type=int, default=None)
    parser.add_argument("--validation-fraction", type=float, default=None)
    parser.add_argument("--validation-max-batches", type=int, default=None)
    parser.add_argument(
        "--best-checkpoint-metric",
        type=str,
        choices=["val_loss", "val_topology_focus_loss", "val_puzzle_stage_semantic_loss", "train_loss"],
        default=None,
    )
    parser.add_argument("--puzzle-structure-dropout-prob", type=float, default=None)
    parser.add_argument("--puzzle-stage-conditioning-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--puzzle-stage-token-scale", type=float, default=None)
    parser.add_argument("--puzzle-stage-topology-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--puzzle-stage-trace-decay", type=float, default=None)
    parser.add_argument("--puzzle-stage-semantics-loss-weight", type=float, default=None)
    parser.add_argument("--puzzle-stage-semantics-hidden-dim", type=int, default=None)
    parser.add_argument("--puzzle-stage-semantics-max-sequence-length", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--keep-last", type=int, default=None)
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--checkpoint-storage-budget-gb", type=float, default=None)
    parser.add_argument("--checkpoint-storage-warning-fraction", type=float, default=None)
    parser.add_argument("--checkpoint-storage-cleanup-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--checkpoint-storage-cleanup-target-fraction", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--quick", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--verbose", "-v", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    config = build_masked_room_training_config_from_args(args)

    logging.basicConfig(
        level=logging.DEBUG if bool(getattr(args, "verbose", False)) else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    train_masked_room(config)


if __name__ == "__main__":
    main()
