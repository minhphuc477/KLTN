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

from src.core.definitions import (
    GRAPH_EDGE_FEATURE_DIM,
    GRAPH_NODE_FEATURE_DIM,
    GRAPH_TPE_DIM,
    ROOM_HEIGHT,
    ROOM_TOPOLOGY_CHANNEL_COUNT,
    ROOM_WIDTH,
    SEMANTIC_PALETTE,
)

try:
    import torch
except ImportError:
    torch = None


logger = logging.getLogger(__name__)

DEFAULT_DATASET_SCHEMA_PROFILE = "zelda_v1"
DATASET_SCHEMA_PROFILES: Dict[str, Dict[str, int]] = {
    DEFAULT_DATASET_SCHEMA_PROFILE: {
        "num_classes": int(max(int(v) for v in SEMANTIC_PALETTE.values()) + 1),
        "room_height": int(ROOM_HEIGHT),
        "room_width": int(ROOM_WIDTH),
        "node_feature_dim": int(GRAPH_NODE_FEATURE_DIM),
        "edge_feature_dim": int(GRAPH_EDGE_FEATURE_DIM),
        "tpe_dim": int(GRAPH_TPE_DIM),
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
    ConfigField("runtime.cudnn_benchmark", bool, True, "Enable cuDNN convolution autotuning for fixed-size training grids."),
    ConfigField("runtime.cudnn_deterministic", bool, False, "Force deterministic cuDNN kernels; disables benchmark when true."),
    ConfigField("runtime.verbose", bool, False, "Enable verbose logging."),
    ConfigField("runtime.quick", bool, False, "Shorten training for smoke tests."),
    ConfigField("runtime.auto_resume", bool, True, "Automatically resume from checkpoint_dir/latest_resume.pth when present."),
    ConfigField("runtime.checkpoint_storage_budget_gb", float, None, "Optional per-stage checkpoint storage budget in GB. Null disables budget enforcement.", min_value=0.0, allow_none=True),
    ConfigField("runtime.checkpoint_storage_warning_fraction", float, 0.8, "Warn when checkpoint usage reaches this fraction of the storage budget.", min_value=0.0, max_value=1.0),
    ConfigField("runtime.checkpoint_storage_cleanup_enabled", bool, True, "Automatically remove retained resume checkpoints when the storage budget is exceeded."),
    ConfigField("runtime.checkpoint_storage_cleanup_target_fraction", float, 0.6, "After automatic cleanup, aim to reduce checkpoint usage to this fraction of the storage budget.", min_value=0.0, max_value=1.0),
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
    ConfigField("dataset.grid_augmentation", bool, False, "Enable shape-preserving random grid augmentation for non-graph autoencoder training. Graph-conditioned stages keep this disabled unless graph metadata is transformed in lockstep."),
    ConfigField("dataset.room_level", bool, True, "Train on individual rooms instead of stitched dungeons."),
    ConfigField("dataset.dungeon_batch_mode", bool, True, "For room-level diffusion, batch all rooms from one dungeon variant for global graph loss."),
    ConfigField("dataset.load_graphs", bool, True, "Load graph side-information from dataset."),
    ConfigField("dataset.train_dungeons", list, [1, 2, 3, 4, 5, 6, 7, 8], "Dungeon ids allowed in training and internal validation loaders. Dungeon 9 is held out by default.", sequence_item_type=int, min_value=1, max_value=9),
    ConfigField("dataset.test_dungeons", list, [9], "Dungeon ids reserved for final unseen test evaluation.", sequence_item_type=int, min_value=1, max_value=9),
    ConfigField("dataset.variants", list, [1, 2], "Quest variants included for each selected dungeon id.", sequence_item_type=int, min_value=1, max_value=2),
    ConfigField("dataset.topology_supervision_mode", str, "runtime_aligned", "How room-topology training signals are constructed. 'runtime_aligned' forbids peeking at ground-truth room geometry; 'oracle_room_grid' restores the legacy room-grid-derived traces for controlled ablations only.", choices=("runtime_aligned", "oracle_room_grid")),
    ConfigField("dataset.min_samples_per_epoch", int, 64, "Minimum effective VQ-VAE samples per epoch.", min_value=1),
    ConfigField("dataset.num_classes", int, 44, "Semantic tile vocabulary size.", min_value=1),
    ConfigField("dataset.room_height", int, 16, "Supported room height.", min_value=1),
    ConfigField("dataset.room_width", int, 11, "Supported room width.", min_value=1),
    ConfigField("dataset.node_feature_dim", int, GRAPH_NODE_FEATURE_DIM, "Supported graph node-feature width.", min_value=1),
    ConfigField("dataset.edge_feature_dim", int, GRAPH_EDGE_FEATURE_DIM, "Supported graph edge-feature width.", min_value=1),
    ConfigField("dataset.tpe_dim", int, GRAPH_TPE_DIM, "Supported topological positional encoding width.", min_value=1),
    ConfigField("vqvae.checkpoint_dir", str, "", "Directory for VQ-VAE checkpoints. Empty means output_dir/checkpoints/vqvae."),
    ConfigField("vqvae.resume_checkpoint", str, None, "Optional VQ-VAE checkpoint to resume.", allow_none=True),
    ConfigField("vqvae.epochs", int, 300, "VQ-VAE training epochs.", min_value=1),
    ConfigField("vqvae.learning_rate", float, 3e-4, "VQ-VAE learning rate.", min_value=1e-8),
    ConfigField("vqvae.weight_decay", float, 1e-5, "VQ-VAE optimizer weight decay.", min_value=0.0),
    ConfigField("vqvae.grad_clip_norm", float, 1.0, "VQ-VAE gradient clipping norm.", min_value=0.0),
    ConfigField("vqvae.scheduler_eta_min", float, 1e-6, "VQ-VAE cosine scheduler minimum learning rate.", min_value=0.0),
    ConfigField("vqvae.save_every", int, 50, "VQ-VAE checkpoint interval.", min_value=1),
    ConfigField("vqvae.keep_last", int, 2, "Number of retained full-resume VQ-VAE checkpoints besides latest_resume/best.", min_value=0),
    ConfigField("vqvae.latent_dim", int, 64, "VQ-VAE latent width.", min_value=1),
    ConfigField("vqvae.hidden_dim", int, 96, "VQ-VAE base channel width.", min_value=8),
    ConfigField("vqvae.codebook_size", int, 256, "VQ-VAE codebook size.", min_value=8),
    ConfigField("vqvae.architecture", str, "vqvae", "Block-II tokenizer architecture.", choices=("vqvae", "vqvae2", "fsq")),
    ConfigField("vqvae.top_codebook_size", int, None, "Optional top-level codebook size for vqvae2.", min_value=8, allow_none=True),
    ConfigField("vqvae.top_latent_dim", int, None, "Optional top-level latent width for vqvae2.", min_value=1, allow_none=True),
    ConfigField("vqvae.commitment_cost", float, 0.25, "VQ-VAE commitment loss weight.", min_value=0.0),
    ConfigField("vqvae.rare_tile_weight", float, 5.0, "Rare-tile reconstruction reweighting.", min_value=1.0),
    ConfigField("vqvae.use_ema", bool, True, "Use EMA VQ codebook updates."),
    ConfigField("vqvae.use_coordconv", bool, True, "Use CoordConv in VQ-VAE encoder."),
    ConfigField("vqvae.mrf_penalty_weight", float, 0.05, "Illegal-adjacency penalty coefficient.", min_value=0.0),
    ConfigField("vqvae.dead_code_reset_interval", int, 100, "Check for dead VQ codes every N optimizer steps.", min_value=1),
    ConfigField("vqvae.dead_code_threshold", float, 0.05, "EMA assignment-count threshold below which a VQ code is considered dead.", min_value=0.0),
    ConfigField("vqvae.dead_code_warmup_steps", int, 500, "Do not reset VQ codes until at least this many optimizer steps have elapsed.", min_value=0),
    ConfigField("vqvae.protect_active_codes_during_reset", bool, True, "Never reset VQ codes that are still active in the current batch."),
    ConfigField("vqvae.max_dead_code_resets_per_event", int, 16, "Maximum number of VQ codes to reset in a single maintenance event; 0 disables the cap.", min_value=0),
    ConfigField("vqvae.validation_fraction", float, 0.1, "Held-out validation fraction for VQ-VAE model selection and reporting. Set to 0 to disable a validation split.", min_value=0.0, max_value=0.5),
    ConfigField("vqvae.validation_max_batches", int, 16, "Maximum number of mini-batches evaluated on the VQ-VAE validation split each epoch.", min_value=1),
    ConfigField("vqvae.best_checkpoint_metric", str, "val_loss", "Metric used to select the best VQ-VAE checkpoint. Falls back to train_loss when no validation split exists.", choices=("val_loss", "train_loss")),
    ConfigField("diffusion.checkpoint_dir", str, "", "Directory for diffusion checkpoints. Empty means output_dir/checkpoints/diffusion."),
    ConfigField("diffusion.vqvae_checkpoint", str, None, "Frozen VQ-VAE checkpoint for diffusion.", allow_none=True),
    ConfigField("diffusion.epochs", int, 100, "Diffusion training epochs.", min_value=1),
    ConfigField("diffusion.learning_rate", float, 1e-4, "Diffusion optimizer learning rate.", min_value=1e-8),
    ConfigField("diffusion.optimizer_weight_decay", float, 1e-5, "Diffusion optimizer weight decay.", min_value=0.0),
    ConfigField("diffusion.global_lr_warmup_epochs", int, 0, "Optional LR warmup applied to all diffusion optimizer groups.", min_value=0),
    ConfigField("diffusion.grad_clip_norm", float, 1.0, "Diffusion gradient clipping norm.", min_value=0.0),
    ConfigField("diffusion.gradient_accumulation_steps", int, 1, "Number of micro-batches to accumulate before each diffusion optimizer/EMA step.", min_value=1),
    ConfigField("diffusion.gradient_checkpointing", bool, False, "Enable memory-efficient gradient checkpointing."),
    ConfigField("diffusion.use_amp", bool, False, "Enable mixed-precision autocast during diffusion training."),
    ConfigField("diffusion.amp_mixed_precision", str, "fp16", "Mixed-precision mode used when diffusion.use_amp is true.", choices=("fp16", "bf16", "auto")),
    ConfigField("diffusion.use_accelerate", bool, False, "Use HuggingFace Accelerate to prepare the diffusion trainer modules and optimizer when not running torchrun/DDP."),
    ConfigField("diffusion.save_every", int, 1000, "Iterations between saving checkpoints.", min_value=1),
    ConfigField("diffusion.keep_last", int, 2, "Number of retained full-resume diffusion checkpoints besides latest_resume/best/final.", min_value=0),
    ConfigField("diffusion.validation_fraction", float, 0.1, "Held-out validation fraction for diffusion checkpoint selection and reporting. Set to 0 to reuse the training split for evaluation-only ablations.", min_value=0.0, max_value=0.5),
    ConfigField("diffusion.latent_dim", int, 64, "Diffusion latent width.", min_value=1),
    ConfigField("diffusion.model_channels", int, 96, "Diffusion U-Net base channels.", min_value=8),
    ConfigField("diffusion.context_dim", int, 256, "Conditioning context width.", min_value=8),
    ConfigField("diffusion.denoiser_backbone", str, "unet", "Latent denoiser backbone ablation.", choices=("unet", "dit")),
    ConfigField("diffusion.unet_channel_mult", list, [1, 2, 4], "Per-level U-Net channel multipliers.", sequence_item_type=int, min_value=1),
    ConfigField("diffusion.unet_num_res_blocks", int, 2, "Residual blocks per U-Net level.", min_value=1),
    ConfigField("diffusion.unet_attention_resolutions", list, [1, 2], "U-Net level indices that enable attention.", sequence_item_type=int, min_value=0),
    ConfigField("diffusion.unet_num_heads", int, 8, "U-Net attention head count.", min_value=1),
    ConfigField("diffusion.unet_dropout", float, 0.1, "U-Net residual/attention dropout.", min_value=0.0, max_value=1.0),
    ConfigField("diffusion.dit_depth", int, 4, "DiT transformer block count.", min_value=1),
    ConfigField("diffusion.dit_patch_size", int, 1, "DiT latent patch size.", min_value=1),
    ConfigField("diffusion.dit_mlp_ratio", float, 4.0, "DiT MLP expansion ratio.", min_value=1.0),
    ConfigField("diffusion.dit_activation_type", str, "gelu", "DiT MLP activation ablation.", choices=("gelu", "swiglu")),
    ConfigField("diffusion.dit_norm_type", str, "layer", "DiT normalization ablation.", choices=("layer", "rms")),
    ConfigField("diffusion.condition_hidden_dim", int, 192, "Condition-encoder hidden width.", min_value=8),
    ConfigField("diffusion.condition_num_gnn_layers", int, 2, "Condition-encoder GNN depth.", min_value=1),
    ConfigField("diffusion.condition_num_attention_heads", int, 8, "Condition-encoder fusion heads.", min_value=1),
    ConfigField("diffusion.condition_dropout", float, 0.1, "Condition-encoder dropout.", min_value=0.0, max_value=1.0),
    ConfigField("diffusion.condition_gnn_type", str, "gps", "Condition-encoder graph backbone.", choices=("gcn", "gat", "sage", "gps")),
    ConfigField("diffusion.condition_use_reference_room_maps", bool, True, "Enable discrete neighboring-room exemplar conditioning in Block III."),
    ConfigField("diffusion.condition_reference_tile_vocab_size", int, 44, "Tile vocabulary size used by the reference-room exemplar encoder.", min_value=2),
    ConfigField("diffusion.condition_reference_embedding_dim", int, 32, "Embedding width for the reference-room exemplar encoder.", min_value=4),
    ConfigField("diffusion.condition_reference_hidden_dim", int, 64, "CNN hidden width for the reference-room exemplar encoder.", min_value=4),
    ConfigField("diffusion.condition_use_rrwp_edge_features", bool, True, "Inject RRWP edge encodings into graph conditioning edge attributes."),
    ConfigField("diffusion.graph_conditioning_mode", str, "node_sequence", "Graph-conditioning representation.", choices=("node_sequence", "pooled")),
    ConfigField("diffusion.num_timesteps", int, 1000, "Forward diffusion timesteps.", min_value=10),
    ConfigField("diffusion.schedule_type", str, "cosine", "Diffusion noise schedule.", choices=("linear", "cosine")),
    ConfigField("diffusion.topology_refinement_mode", str, "gat2", "Topology refinement inside attention.", choices=("none", "lightweight", "sparse_edge", "sparse_directed", "sparse_semantic", "sparse_directed_semantic", "gat2", "gat2_directed", "gat2_semantic", "gat2_directed_semantic", "graphormer", "graphormer_learned", "graphormer_learned_directed", "graphormer_learned_semantic", "graphormer_learned_directed_semantic")),
    ConfigField("diffusion.attention_mode", str, "softmax", "Attention kernel.", choices=("softmax", "linear_hedgehog")),
    ConfigField("diffusion.topology_conditioning_mode", str, "spade", "Room-topology conditioning path.", choices=("additive", "spade")),
    ConfigField("diffusion.hedgehog_feature_dim", int, 32, "Linear-attention feature width.", min_value=4),
    ConfigField("diffusion.graph_auto_linear_attention_nodes", int, 128, "Switch graph-to-grid attention to linear mode above this node count. 0 disables the auto-switch.", min_value=0),
    ConfigField("diffusion.graph_to_grid_edge_semantics", bool, False, "Ablation: inject edge-label semantics into graph-to-grid spatial attention."),
    ConfigField("diffusion.spatial_graph_gate_init", float, -2.0, "Initial logit for graph-conditioning gate."),
    ConfigField("diffusion.spatial_topology_gate_init", float, -2.0, "Initial logit for room-topology gate."),
    ConfigField("diffusion.use_teacher_forced_neighbor_latents", bool, True, "Use real adjacent room maps during room-level diffusion training to encode neighbor latents."),
    ConfigField("diffusion.puzzle_structure_dropout_prob", float, 0.35, "Train-only augmentation probability for puzzle rooms: strip BLOCK structure from the target room and set puzzle_room_structure_enabled=false in the conditioning metadata so the diffusion teacher learns explicit puzzle-on/puzzle-off control.", min_value=0.0, max_value=1.0),
    ConfigField("diffusion.puzzle_stage_conditioning_enabled", bool, False, "Append deterministic ordered puzzle-stage tokens to graph conditioning so retrains can learn explicit multi-step puzzle semantics."),
    ConfigField("diffusion.puzzle_stage_token_scale", float, 0.20, "Scale of deterministic puzzle-stage control tokens appended to diffusion graph conditioning.", min_value=0.0, max_value=2.0),
    ConfigField("diffusion.puzzle_stage_topology_enabled", bool, False, "Inject ordered stage-trace structure into room_topology_map traversability during diffusion training."),
    ConfigField("diffusion.puzzle_stage_trace_decay", float, 0.75, "Per-stage decay used when rasterizing ordered puzzle traces into topology priors.", min_value=0.05, max_value=1.0),
    ConfigField("diffusion.puzzle_stage_semantics_loss_weight", float, 0.0, "Auxiliary learned loss over decoded room logits that predicts gate family, sequence requirement, stage count, and ordered stage slots.", min_value=0.0, max_value=10.0),
    ConfigField("diffusion.puzzle_stage_semantics_hidden_dim", int, 96, "Hidden width of the puzzle-stage semantics supervision head.", min_value=16, max_value=512),
    ConfigField("diffusion.puzzle_stage_semantics_max_sequence_length", int, 6, "Maximum ordered puzzle stages supervised per room.", min_value=1, max_value=12),
    ConfigField("diffusion.use_current_node_distance_features", bool, True, "Inject current-room distance features into Block III/IV graph conditioning."),
    ConfigField("diffusion.current_node_distance_max", int, 8, "Distance clip used when normalizing current-room graph distances.", min_value=1),
    ConfigField("diffusion.room_topology_channels", int, ROOM_TOPOLOGY_CHANNEL_COUNT, "Room-topology conditioning channel count.", min_value=1),
    ConfigField("diffusion.cfg_dropout_prob", float, 0.1, "Classifier-free conditioning dropout.", min_value=0.0, max_value=1.0),
    ConfigField("diffusion.cfg_scale", float, 3.0, "Classifier-free guidance scale.", min_value=0.0),
    ConfigField("diffusion.cfg_schedule_mode", str, "constant", "Classifier-free guidance schedule.", choices=("constant", "linear_decay", "cosine_decay")),
    ConfigField("diffusion.cfg_schedule_min_scale", float, 1.0, "Minimum classifier-free guidance scale.", min_value=0.0),
    ConfigField("diffusion.cfg_schedule_power", float, 1.0, "Classifier-free guidance schedule power.", min_value=1e-6),
    ConfigField("diffusion.pag_scale", float, 0.0, "Perturbed-attention guidance scale for inference.", min_value=0.0),
    ConfigField("diffusion.prediction_type", str, "epsilon", "Diffusion target parameterization.", choices=("epsilon", "v")),
    ConfigField("diffusion.training_objective", str, "diffusion", "Latent training objective ablation. flow_matching requires the DiT backbone and uses the rectified-flow ODE sampler for generation.", choices=("diffusion", "flow_matching")),
    ConfigField("diffusion.min_snr_gamma", float, 5.0, "Min-SNR-gamma training weight.", min_value=0.0),
    ConfigField("diffusion.logic_net_enabled", bool, True, "Enable LogicNet loss, validation scoring, and gradient guidance."),
    ConfigField("diffusion.logic_net_trainable", bool, True, "Optimize LogicNet parameters jointly with diffusion when LogicNet is enabled."),
    ConfigField("diffusion.logic_learning_rate", float, None, "Optional LogicNet-specific optimizer learning rate. Null reuses diffusion.learning_rate.", min_value=1e-8, allow_none=True),
    ConfigField("diffusion.logic_lr_warmup_epochs", int, 5, "Epochs used to linearly warm up only the LogicNet optimizer group.", min_value=0),
    ConfigField("diffusion.logic_grid_pathfinder", str, "bellman_ford", "Grid-level LogicNet pathfinder ablation.", choices=("cnn", "bellman_ford", "bellman-ford", "soft_bellman_ford", "soft-bellman-ford", "vin", "value_iteration", "value-iteration", "perturb_and_map", "perturb-and-map", "perturb_map", "pmap")),
    ConfigField("diffusion.logic_full_coverage", bool, True, "Use complete Bellman coverage; false is the truncated-planning ablation."),
    ConfigField("diffusion.num_logic_iterations", int, 30, "LogicNet message-passing iterations.", min_value=1),
    ConfigField("diffusion.logic_topology_trace_weight", float, 0.25, "Additional LogicNet weight on room-topology traversability traces.", min_value=0.0),
    ConfigField("diffusion.logic_topology_anchor_weight", float, 0.25, "Additional LogicNet weight on start/goal/door anchor walkability.", min_value=0.0),
    ConfigField("diffusion.logic_global_reach_weight", float, 1.0, "LogicNet weight on dungeon-level mission-graph reachability.", min_value=0.0),
    ConfigField("diffusion.logic_global_room_weight", float, 0.25, "LogicNet weight on lifting room passability into mission-graph node costs.", min_value=0.0),
    ConfigField("diffusion.guidance_scale", float, 1.0, "Logic guidance scale.", min_value=0.0),
    ConfigField("diffusion.guidance_clamp_magnitude", float, 1.0, "Logic-guidance gradient clamp magnitude.", min_value=0.0),
    ConfigField("diffusion.guidance_relative_norm_cap", float, 0.25, "Relative guidance norm cap.", min_value=0.0),
    ConfigField("diffusion.guidance_schedule_enabled", bool, True, "Enable timestep-decayed LogicNet guidance."),
    ConfigField("diffusion.guidance_active_fraction", float, 1.0, "Active reverse-process fraction for LogicNet guidance.", min_value=0.05, max_value=1.0),
    ConfigField("diffusion.guidance_decay_power", float, 1.0, "Logic-guidance decay power.", min_value=0.25),
    ConfigField("diffusion.guidance_max_graph_nodes", int, 512, "Maximum graph nodes allowed for LogicNet guidance.", min_value=1),
    ConfigField("diffusion.guidance_max_key_lock_pairs", int, 2048, "Maximum key-lock pairs passed into LogicNet guidance.", min_value=0),
    ConfigField("diffusion.guidance_max_guidance_elements", int, 2_000_000, "Maximum latent elements allowed for autograd guidance.", min_value=1),
    ConfigField("diffusion.alpha_visual", float, 1.0, "Diffusion reconstruction loss coefficient.", min_value=0.0),
    ConfigField("diffusion.alpha_logic", float, 0.1, "Logic regularization coefficient.", min_value=0.0),
    ConfigField("diffusion.alpha_logic_tile", float, 0.05, "Supervised LogicNet tile-classifier loss coefficient.", min_value=0.0),
    ConfigField("diffusion.alpha_wfc_pseudo", float, 0.0, "Opt-in WFC pseudo-target distillation coefficient.", min_value=0.0),
    ConfigField("diffusion.wfc_pseudo_max_samples", int, 2, "Maximum repaired pseudo-label samples per training batch.", min_value=0),
    ConfigField("diffusion.wfc_pseudo_confidence_threshold", float, 0.75, "Prediction confidence required to pin a cell before WFC pseudo-repair.", min_value=0.0, max_value=1.0),
    ConfigField("diffusion.min_logic_tile_accuracy_for_guidance", float, 0.4, "Minimum validation tile-classifier accuracy before LogicNet sampling guidance is trusted.", min_value=0.0),
    ConfigField("diffusion.graph_spatial_alignment_weight", float, 0.0, "Graph-node to grid-position attention alignment coefficient.", min_value=0.0),
    ConfigField("diffusion.logic_loss_mode", str, "predicted_latent", "Logic loss target mode.", choices=("predicted_latent", "detached_real")),
    ConfigField("diffusion.warmup_epochs", int, 5, "Epochs before enabling logic loss.", min_value=0),
    ConfigField("diffusion.logic_loss_ramp_epochs", int, 2, "Epochs used to linearly ramp alpha_logic after warmup.", min_value=1),
    ConfigField("diffusion.scheduler_t0", int, 10, "CosineWarmRestarts T_0.", min_value=1),
    ConfigField("diffusion.scheduler_t_mult", int, 2, "CosineWarmRestarts T_mult.", min_value=1),
    ConfigField("diffusion.scheduler_eta_min", float, 1e-6, "CosineWarmRestarts eta_min.", min_value=0.0),
    ConfigField("diffusion.ema_decay", float, 0.9999, "EMA decay for diffusion weights.", min_value=0.0, max_value=0.999999),
    ConfigField("diffusion.validation_num_samples", int, 8, "Generated-sample validation count for logic/solvability metrics.", min_value=1),
    ConfigField("diffusion.validation_num_diffusion_samples", int, 64, "Validation sample count for denoising loss on real latents.", min_value=1),
    ConfigField("diffusion.latent_cache_enabled", bool, True, "Cache frozen VQ-VAE latents during diffusion training so repeated room and neighbor maps do not rerun Block II encoding."),
    ConfigField("diffusion.latent_cache_max_items", int, 4096, "Maximum in-memory frozen-latent cache entries for diffusion training. 0 disables caching.", min_value=0),
    ConfigField("topology.default_target_curve", list, [0.2, 0.4, 0.6, 0.8, 1.0], "Default target difficulty/tension curve for evolutionary topology generation.", sequence_item_type=float, min_value=0.0, max_value=1.0),
    ConfigField("topology.num_rooms", int, 8, "Default room budget for generated topologies.", min_value=1),
    ConfigField("topology.population_size", int, 50, "Default evolutionary population size for Block I.", min_value=1),
    ConfigField("topology.generations", int, 100, "Default number of evolutionary generations for Block I.", min_value=1),
    ConfigField("topology.mutation_rate", float, 0.15, "Per-gene mutation probability for Block I search.", min_value=0.0, max_value=1.0),
    ConfigField("topology.crossover_rate", float, 0.7, "Crossover probability for Block I search.", min_value=0.0, max_value=1.0),
    ConfigField("topology.genome_length", int, 0, "Genome length for Block I. Set to 0 to auto-derive from num_rooms.", min_value=0),
    ConfigField("topology.rule_space", str, "full", "Grammar rule-space for Block I.", choices=("core", "full")),
    ConfigField("topology.transition_mix", float, 0.7, "Mixing ratio between transition-biased and global rule priors.", min_value=0.0, max_value=1.0),
    ConfigField("topology.search_strategy", str, "ga", "Topology search backend.", choices=("ga", "cvt_emitter", "map_elites", "cvt", "cvt_map_elites")),
    ConfigField("topology.qd_archive_cells", int, 128, "CVT archive cell count when using QD topology search.", min_value=32),
    ConfigField("topology.qd_init_random_fraction", float, 0.35, "Bootstrap fraction of random samples before CVT emitters dominate.", min_value=0.05, max_value=0.95),
    ConfigField("topology.qd_emitter_mutation_rate", float, 0.18, "Emitter mutation rate when using CVT topology search.", min_value=0.01, max_value=0.95),
    ConfigField("topology.qd_archive_path", str, None, "Optional persisted CVT archive path for warm-started topology QD search.", allow_none=True),
    ConfigField("topology.qd_load_archive", bool, False, "Load topology.qd_archive_path before CVT-emitter search when available."),
    ConfigField("topology.qd_autosave_archive", bool, False, "Persist topology.qd_archive_path during and after CVT-emitter search."),
    ConfigField("topology.max_lock_key_rules", int, 3, "Soft cap on InsertLockKey rule applications per genome execution.", min_value=0),
    ConfigField("topology.enable_rule_credit_assignment", bool, False, "Enable adaptive rule-credit assignment during topology search."),
    ConfigField("topology.enforce_generation_constraints", bool, False, "Reject intermediate topology candidates that violate progression constraints."),
    ConfigField("topology.allow_candidate_repairs", bool, False, "Attempt local repairs when topology generation constraints fail."),
    ConfigField("generation.room_generator_mode", str, "latent_diffusion", "Runtime room generator branch.", choices=("latent_diffusion", "discrete_masked")),
    ConfigField("generation.guidance_scale", float, 3.0, "Default classifier-free guidance scale for runtime generation; matches the distilled/validated teacher regime.", min_value=0.0),
    ConfigField("generation.logic_guidance_scale", float, 0.0, "Default LogicNet guidance scale for runtime generation; extra gradient guidance is opt-in.", min_value=0.0),
    ConfigField("generation.logic_guidance_strategy", str, "late", "Runtime LogicNet gradient guidance window.", choices=("none", "late", "full")),
    ConfigField("generation.logic_guidance_active_fraction", float, 0.2, "Reverse-process fraction used when generation.logic_guidance_strategy='late'.", min_value=0.05, max_value=1.0),
    ConfigField("generation.num_diffusion_steps", int, 50, "Default diffusion or masked-token sampling steps for runtime generation.", min_value=1),
    ConfigField("generation.use_fast_sampling", bool, False, "Prefer fast-sampler inference when available during runtime generation."),
    ConfigField("generation.latent_sampler", str, "diffusion", "Default latent sampling backend for runtime generation.", choices=("diffusion", "categorical")),
    ConfigField("generation.categorical_codebook_size", int, None, "Optional codebook-size cap for categorical runtime sampling.", min_value=1, allow_none=True),
    ConfigField("generation.use_topological_positional_encoding", bool, True, "Enable topological positional encoding during runtime generation."),
    ConfigField("generation.apply_repair", bool, True, "Apply symbolic repair during runtime generation."),
    ConfigField("generation.use_neural_guided_repair", bool, True, "When LogicNet is available, use LogicNet-guided cost maps and topology masks inside symbolic repair."),
    ConfigField("generation.use_neural_repair_feedback", bool, True, "Use diffusion inpainting as M3 neural feedback when symbolic repair hits contradiction regions."),
    ConfigField("generation.repair_inpaint_noise_strength", float, 0.5, "Noise strength for M3 contradiction-region inpainting.", min_value=0.0, max_value=1.0),
    ConfigField("generation.repair_inpaint_guidance_scale_multiplier", float, 1.0, "Multiplier applied to LogicNet guidance scale during M3 inpainting.", min_value=0.0),
    ConfigField("generation.enable_map_elites", bool, False, "Compute MAP-Elites descriptors during runtime generation."),
    ConfigField("generation.symbolic_max_repair_attempts", int, 5, "Maximum symbolic repair passes applied after neural room generation.", min_value=1),
    ConfigField("generation.symbolic_repair_margin", int, 2, "Margin passed to symbolic repair when validating and reconnecting local room structure.", min_value=0),
    ConfigField("generation.symbolic_adjacency_threshold", float, 0.01, "Adjacency threshold used by symbolic repair when evaluating structural connectivity.", min_value=0.0),
    ConfigField("generation.default_start_coord", list, [1, 5], "Fallback room start coordinate used when no room-specific anchor metadata exists.", sequence_item_type=int, min_value=0),
    ConfigField("generation.default_goal_coord", list, [14, 5], "Fallback room goal coordinate used when no room-specific anchor metadata exists.", sequence_item_type=int, min_value=0),
    ConfigField("generation.semantic_role_prior_strength", float, 0.15, "Room-topology role broadcast prior strength used when constructing semantic topology maps at runtime.", min_value=0.0, max_value=1.0),
    ConfigField("generation.semantic_anchor_threshold", float, 0.5, "Threshold used when converting topology anchor channels into fixed semantic tokens for masked-room training/ablation.", min_value=0.0, max_value=1.0),
    ConfigField("generation.semantic_puzzle_offset", int, 2, "Perpendicular offset magnitude used for puzzle anchor placement inside room-topology priors.", min_value=0, max_value=4),
    ConfigField("generation.semantic_constrained_decoding_enabled", bool, True, "Apply graph-aware semantic logit shaping during room decode so planned semantic anchors are encouraged before post-hoc overlay."),
    ConfigField("generation.semantic_marker_logit_bias", float, 10000.0, "Positive logit bias added at planned graph-marker slots during semantic constrained decoding. Large values behave like graph-aware hard constraints during decode.", min_value=0.0),
    ConfigField("generation.semantic_marker_suppression_bias", float, 100.0, "Negative logit bias applied to volatile semantic channels away from planned graph-marker slots during semantic constrained decoding. Large values strongly suppress stray semantic markers.", min_value=0.0),
    ConfigField("generation.puzzle_room_scaffold_enabled", bool, True, "Inject a graph-conditioned constructive obstacle scaffold into under-structured puzzle rooms at runtime.",),
    ConfigField("generation.puzzle_room_structure_enabled", bool, True, "Allow interior BLOCK-structure content in generated rooms. Disable this for strict no-puzzle ablations so learned block clutter is stripped even when it was not produced by the runtime scaffold."),
    ConfigField("generation.puzzle_room_scaffold_min_structure_tiles", int, 10, "Minimum number of interior wall/block tiles that counts as an already-structured puzzle room before runtime scaffolding is skipped.", min_value=0),
    ConfigField("generation.puzzle_room_archetype_mode", str, "auto", "Puzzle-room scaffold archetype selection policy. 'auto' chooses an archetype from graph semantics; the other values force a specific constructive pattern.", choices=("auto", "gate", "serpentine", "hub", "island", "combat")),
    ConfigField("generation.puzzle_room_branch_density", float, 0.75, "How aggressively optional scaffold branches are added to puzzle rooms. Higher values produce denser obstacle patterns.", min_value=0.0, max_value=1.0),
    ConfigField("generation.puzzle_room_block_budget", int, 28, "Soft upper bound on runtime scaffold block placements per puzzle room.", min_value=0),
    ConfigField("generation.puzzle_room_preserve_route_margin", int, 0, "4-neighbour dilation radius used to preserve the planned route around puzzle-room scaffolds. Tiny Zelda rooms usually work best with 0 unless an ablation explicitly wants a larger safety buffer.", min_value=0, max_value=4),
    ConfigField("generation.puzzle_room_switch_pocket_depth", int, 3, "Depth of the local switch pocket used by switch-locked puzzle templates.", min_value=1, max_value=6),
    ConfigField("generation.puzzle_room_resource_bypass_offset", int, 2, "Offset used to route bombable-wall puzzle templates through a side bypass instead of a centered gate opening. Item-gate rooms use the separate item-slot depth control.", min_value=1, max_value=5),
    ConfigField("generation.puzzle_room_key_pocket_depth", int, 3, "Depth of the local key-before-gate pocket when a key-locked room also owns a local key anchor.", min_value=1, max_value=6),
    ConfigField("generation.puzzle_room_item_slot_depth", int, 3, "Depth of the local item-slot alcove used by item-locked puzzle templates.", min_value=1, max_value=6),
    ConfigField("generation.puzzle_room_toggle_corridor_offset", int, 2, "Half-width offset used by toggle-state corridor templates for on_off/state-block puzzle rooms.", min_value=1, max_value=5),
    ConfigField("generation.puzzle_room_novelty_enabled", bool, True, "Evaluate multiple valid puzzle scaffold variants per room and select one using a lightweight quality-diversity score instead of taking the first matching template."),
    ConfigField("generation.puzzle_room_candidate_count", int, 4, "Number of local puzzle scaffold candidates evaluated per room when novelty-aware selection is enabled.", min_value=1, max_value=6),
    ConfigField("generation.puzzle_room_novelty_weight", float, 0.45, "Weight applied to novelty distance when ranking locally valid puzzle scaffold candidates. Higher values prefer more diverse puzzle layouts across rooms.", min_value=0.0, max_value=2.0),
    ConfigField("generation.puzzle_room_min_quality_gain", float, 0.5, "Minimum score improvement a constructive puzzle scaffold must achieve over the cleaned no-scaffold room before it is applied. Higher values make puzzle injection more conservative.", min_value=0.0, max_value=10.0),
    ConfigField("generation.validator_plan_max_states", int, 512, "Maximum validator search states explored per room-local topology-prior segment before falling back to the simpler geometric traversability trace.", min_value=32, max_value=4096),
    ConfigField("generation.puzzle_stage_topology_enabled", bool, False, "Inject ordered multi-step puzzle traces into runtime room-topology priors. Enable only with retrained stage-conditioned checkpoints."),
    ConfigField("generation.puzzle_stage_trace_decay", float, 0.75, "Per-stage decay used when converting ordered puzzle stages into runtime topology traces.", min_value=0.05, max_value=1.0),
    ConfigField("generation.deterministic_graph_marker_overlay_enabled", bool, True, "Apply deterministic graph-owned semantic marker overlay after generation/repair. Disable only for purely-neural semantic ablations."),
    ConfigField("generation.fast_sampler_teacher_fallback_enabled", bool, False, "Allow fast-sampler rooms to fall back to the full diffusion teacher when runtime quality guards trigger. Defaults off so student quality is exposed unless an ablation opts in."),
    ConfigField("generation.masked_room_teacher_fallback_enabled", bool, False, "Allow masked-room outputs with obvious structural noise to fall back to the diffusion teacher during runtime generation. Defaults off so masked-room quality is exposed unless an ablation opts in."),
    ConfigField("generation.masked_room_sampling_temperature", float, 1.0, "Masked-room runtime sampling temperature. 1.0 keeps categorical sampling calibrated; lower sharpens, higher diversifies.", min_value=1e-6),
    ConfigField("generation.masked_room_sampling_schedule", str, "cosine", "Remaining-mask schedule for masked-room iterative decode.", choices=("cosine", "linear")),
    ConfigField("generation.masked_room_sampling_stochastic", bool, True, "Use stochastic categorical masked-room token sampling instead of greedy argmax-only decode."),
    ConfigField("generation.masked_room_corrector_steps", int, 1, "Number of low-confidence masked-room correction rounds after the main decode pass.", min_value=0, max_value=4),
    ConfigField("generation.masked_room_corrector_mask_ratio", float, 0.1, "Fraction of lowest-confidence editable masked-room tokens re-masked in each correction round.", min_value=0.0, max_value=1.0),
    ConfigField("fast_sampler.checkpoint_dir", str, "", "Directory for fast-sampler checkpoints. Empty means output_dir/checkpoints/fast_sampler."),
    ConfigField("fast_sampler.base_diffusion_checkpoint", str, None, "Base diffusion checkpoint for distillation.", allow_none=True),
    ConfigField("fast_sampler.epochs", int, 10, "Fast-sampler distillation epochs.", min_value=1),
    ConfigField("fast_sampler.learning_rate", float, 1e-4, "Fast-sampler learning rate.", min_value=1e-8),
    ConfigField("fast_sampler.optimizer_weight_decay", float, 1e-4, "Fast-sampler optimizer weight decay.", min_value=0.0),
    ConfigField("fast_sampler.grad_clip_norm", float, 1.0, "Fast-sampler gradient clipping norm.", min_value=0.0),
    ConfigField("fast_sampler.num_inference_steps", int, 4, "Target inference steps for distillation.", min_value=1),
    ConfigField("fast_sampler.ema_decay", float, 0.95, "EMA decay for the lower-noise consistency target student.", min_value=0.0, max_value=0.999999),
    ConfigField("fast_sampler.lora_rank", int, 8, "LoRA rank.", min_value=1),
    ConfigField("fast_sampler.lora_alpha", float, 8.0, "LoRA alpha.", min_value=0.0),
    ConfigField("fast_sampler.prediction_loss_weight", float, 0.25, "Weight on student-vs-teacher prediction loss.", min_value=0.0),
    ConfigField("fast_sampler.decode_alignment_weight", float, 0.25, "Weight on decoded room cross-entropy against ground-truth tiles during fast-sampler distillation.", min_value=0.0),
    ConfigField("fast_sampler.topology_alignment_weight", float, 0.25, "Extra weight on topology-critical decoded-room CE during fast-sampler distillation.", min_value=0.0),
    ConfigField("fast_sampler.puzzle_structure_dropout_prob", float, 0.35, "Train-only augmentation probability for puzzle rooms during fast-sampler distillation. Selected puzzle rooms are paired with structure-stripped targets and puzzle_room_structure_enabled=false so the student learns the explicit control exposed at runtime.", min_value=0.0, max_value=1.0),
    ConfigField("fast_sampler.puzzle_stage_conditioning_enabled", bool, False, "Expect and preserve ordered puzzle-stage conditioning when distilling a stage-conditioned diffusion teacher."),
    ConfigField("fast_sampler.puzzle_stage_token_scale", float, 0.20, "Scale of deterministic ordered puzzle-stage tokens during fast-sampler distillation.", min_value=0.0, max_value=2.0),
    ConfigField("fast_sampler.puzzle_stage_topology_enabled", bool, False, "Inject ordered stage traces into room-topology priors during fast-sampler distillation."),
    ConfigField("fast_sampler.puzzle_stage_trace_decay", float, 0.75, "Per-stage decay used when rasterizing ordered puzzle traces for fast-sampler training.", min_value=0.05, max_value=1.0),
    ConfigField("fast_sampler.puzzle_stage_semantics_loss_weight", float, 0.0, "Auxiliary learned loss over decoded student room logits that predicts ordered puzzle semantics during fast-sampler distillation.", min_value=0.0, max_value=10.0),
    ConfigField("fast_sampler.puzzle_stage_semantics_hidden_dim", int, 96, "Hidden width of the fast-sampler puzzle-stage semantics supervision head.", min_value=16, max_value=512),
    ConfigField("fast_sampler.puzzle_stage_semantics_max_sequence_length", int, 6, "Maximum ordered puzzle stages supervised per room during fast-sampler distillation.", min_value=1, max_value=12),
    ConfigField("fast_sampler.topology_marker_weight", float, 2.0, "Relative weight assigned to sparse topology markers, doors, and typed gates when building the fast-sampler topology focus map.", min_value=0.0),
    ConfigField("fast_sampler.topology_trace_weight", float, 0.75, "Relative weight assigned to traversability-trace pixels when building the fast-sampler topology focus map.", min_value=0.0),
    ConfigField("fast_sampler.topology_focus_dilation", int, 1, "4-neighbour dilation radius applied to the fast-sampler topology focus map so sparse anchor pixels influence a small local neighborhood.", min_value=0),
    ConfigField("fast_sampler.validation_fraction", float, 0.1, "Held-out validation fraction for fast-sampler model selection. Set to 0 to disable a validation split and fall back to train_loss.", min_value=0.0, max_value=0.5),
    ConfigField("fast_sampler.validation_max_batches", int, 16, "Maximum number of mini-batches evaluated on the fast-sampler validation split each epoch.", min_value=1),
    ConfigField("fast_sampler.best_checkpoint_metric", str, "val_decode_ce_loss", "Metric used to select the best fast-sampler checkpoint. Falls back to train_loss when no validation split exists.", choices=("val_loss", "val_decode_ce_loss", "val_topology_decode_ce_loss", "train_loss")),
    ConfigField("fast_sampler.save_every", int, 5, "Fast-sampler checkpoint interval.", min_value=1),
    ConfigField("fast_sampler.keep_last", int, 2, "Number of retained full-resume fast-sampler checkpoints besides latest_resume/best/final.", min_value=0),
    ConfigField("masked_room.checkpoint_dir", str, "", "Directory for masked-room checkpoints. Empty means output_dir/checkpoints/masked_room."),
    ConfigField("masked_room.epochs", int, 100, "Masked-room training epochs.", min_value=1),
    ConfigField("masked_room.learning_rate", float, 1e-4, "Masked-room learning rate.", min_value=1e-8),
    ConfigField("masked_room.optimizer_weight_decay", float, 1e-5, "Masked-room optimizer weight decay.", min_value=0.0),
    ConfigField("masked_room.grad_clip_norm", float, 1.0, "Masked-room gradient clipping norm.", min_value=0.0),
    ConfigField("masked_room.scheduler_eta_min", float, 1e-6, "Masked-room cosine scheduler eta_min.", min_value=0.0),
    ConfigField("masked_room.save_every", int, 10, "Masked-room checkpoint interval.", min_value=1),
    ConfigField("masked_room.keep_last", int, 2, "Number of retained full-resume masked-room checkpoints besides latest_resume/best/final.", min_value=0),
    ConfigField("masked_room.context_dim", int, 256, "Masked-room conditioning width.", min_value=8),
    ConfigField("masked_room.condition_hidden_dim", int, 192, "Masked-room condition-encoder hidden width.", min_value=8),
    ConfigField("masked_room.condition_num_gnn_layers", int, 2, "Masked-room condition-encoder GNN depth.", min_value=1),
    ConfigField("masked_room.condition_num_attention_heads", int, 8, "Masked-room fusion heads.", min_value=1),
    ConfigField("masked_room.condition_dropout", float, 0.1, "Masked-room condition-encoder dropout.", min_value=0.0, max_value=1.0),
    ConfigField("masked_room.condition_gnn_type", str, "gcn", "Masked-room graph backbone.", choices=("gcn", "gat", "sage", "gps")),
    ConfigField("masked_room.condition_use_reference_room_maps", bool, True, "Enable discrete neighboring-room exemplar conditioning for the masked-room conditioner."),
    ConfigField("masked_room.condition_reference_tile_vocab_size", int, 44, "Tile vocabulary size used by the masked-room exemplar encoder.", min_value=2),
    ConfigField("masked_room.condition_reference_embedding_dim", int, 32, "Embedding width for the masked-room exemplar encoder.", min_value=4),
    ConfigField("masked_room.condition_reference_hidden_dim", int, 64, "CNN hidden width for the masked-room exemplar encoder.", min_value=4),
    ConfigField("masked_room.graph_conditioning_mode", str, "node_sequence", "Masked-room graph-conditioning mode.", choices=("node_sequence", "pooled")),
    ConfigField("masked_room.use_current_node_distance_features", bool, True, "Inject current-room distance features into masked-room graph conditioning."),
    ConfigField("masked_room.current_node_distance_max", int, 8, "Distance clip used when normalizing current-room graph distances for masked-room training.", min_value=1),
    ConfigField("masked_room.model_channels", int, 64, "Legacy checkpoint field; the masked transformer requires the compatibility value 64.", min_value=8),
    ConfigField("masked_room.hidden_dim", int, 48, "Masked-room token hidden width.", min_value=8),
    ConfigField("masked_room.masked_steps", int, 8, "Masked-token corruption steps.", min_value=1),
    ConfigField("masked_room.attention_mode", str, "softmax", "Masked transformer attention kernel. Only softmax is implemented.", choices=("softmax",)),
    ConfigField("masked_room.context_attention_mode", str, "concat_encoder", "Masked-room context fusion ablation. concat_encoder is the original baseline; cross_decoder routes context through decoder cross-attention.", choices=("concat_encoder", "cross_decoder")),
    ConfigField("masked_room.topology_conditioning_mode", str, "additive", "Masked transformer topology conditioning. Only additive conditioning is implemented.", choices=("additive",)),
    ConfigField("masked_room.hedgehog_feature_dim", int, 32, "Legacy compatibility field; must remain 32.", min_value=4),
    ConfigField("masked_room.graph_auto_linear_attention_nodes", int, 128, "Legacy compatibility field; must remain 128.", min_value=0),
    ConfigField("masked_room.spatial_graph_gate_init", float, -2.0, "Legacy compatibility field; must remain -2.0."),
    ConfigField("masked_room.spatial_topology_gate_init", float, -2.0, "Legacy compatibility field; must remain -2.0."),
    ConfigField("masked_room.unet_channel_mult", list, [1, 2], "Legacy stage list whose length sets masked-transformer depth.", sequence_item_type=int, min_value=1),
    ConfigField("masked_room.unet_num_res_blocks", int, 1, "Transformer layers per legacy stage.", min_value=1),
    ConfigField("masked_room.unet_attention_resolutions", list, [0, 1], "Legacy compatibility field; must remain [0, 1].", sequence_item_type=int, min_value=0),
    ConfigField("masked_room.unet_num_heads", int, 4, "Masked-transformer attention head count.", min_value=1),
    ConfigField("masked_room.unet_dropout", float, 0.1, "Masked-transformer attention/feed-forward dropout.", min_value=0.0, max_value=1.0),
    ConfigField("masked_room.min_mask_ratio", float, 0.0, "Minimum token-mask ratio sampled during masked-room training.", min_value=0.0, max_value=1.0),
    ConfigField("masked_room.max_mask_ratio", float, 1.0, "Maximum token-mask ratio sampled during masked-room training.", min_value=0.0, max_value=1.0),
    ConfigField("masked_room.topology_alignment_weight", float, 0.25, "Extra weight on topology-critical masked-token CE during masked-room training.", min_value=0.0),
    ConfigField("masked_room.logic_net_enabled", bool, False, "Enable LogicNet supervision for masked-room training as an ablation."),
    ConfigField("masked_room.logic_net_trainable", bool, False, "Allow LogicNet parameters to update during masked-room logic-supervised ablations."),
    ConfigField("masked_room.alpha_logic", float, 0.0, "Weight for masked-room LogicNet loss. Kept at 0 unless the ablation is explicitly enabled.", min_value=0.0),
    ConfigField("masked_room.logic_global_reach_weight", float, 1.0, "Weight for dungeon-level reachability in masked-room LogicNet supervision.", min_value=0.0),
    ConfigField("masked_room.logic_global_room_weight", float, 0.25, "Weight for room-passability terms in masked-room LogicNet supervision.", min_value=0.0),
    ConfigField("masked_room.logic_topology_trace_weight", float, 0.25, "Weight for topology-trace walkability anchors in masked-room LogicNet supervision.", min_value=0.0),
    ConfigField("masked_room.logic_topology_anchor_weight", float, 0.25, "Weight for sparse topology-anchor walkability terms in masked-room LogicNet supervision.", min_value=0.0),
    ConfigField("masked_room.logic_grid_pathfinder", str, "bellman_ford", "Grid pathfinder used by masked-room LogicNet ablations.", choices=("bellman_ford", "conv", "cnn", "vin", "learnable", "perturb_and_map")),
    ConfigField("masked_room.logic_full_coverage", bool, True, "Use complete Bellman coverage for masked-room LogicNet."),
    ConfigField("masked_room.num_logic_iterations", int, 30, "Number of LogicNet pathfinding iterations for masked-room ablations.", min_value=1),
    ConfigField("masked_room.puzzle_structure_dropout_prob", float, 0.35, "Train-only augmentation probability for puzzle rooms during masked-room training. Selected puzzle rooms are duplicated as structure-free targets with puzzle_room_structure_enabled=false so the model learns puzzle-on/puzzle-off behavior explicitly.", min_value=0.0, max_value=1.0),
    ConfigField("masked_room.puzzle_stage_conditioning_enabled", bool, False, "Append deterministic ordered puzzle-stage tokens to masked-room conditioning for retrains with explicit multi-step puzzle supervision."),
    ConfigField("masked_room.puzzle_stage_token_scale", float, 0.20, "Scale of deterministic ordered puzzle-stage conditioning tokens during masked-room training.", min_value=0.0, max_value=2.0),
    ConfigField("masked_room.puzzle_stage_topology_enabled", bool, False, "Inject ordered stage traces into room-topology priors during masked-room training."),
    ConfigField("masked_room.puzzle_stage_trace_decay", float, 0.75, "Per-stage decay used when rasterizing ordered puzzle traces for masked-room training.", min_value=0.05, max_value=1.0),
    ConfigField("masked_room.puzzle_stage_semantics_loss_weight", float, 0.0, "Auxiliary learned loss over room-token logits that predicts gate family, sequence requirement, stage count, and ordered stage slots.", min_value=0.0, max_value=10.0),
    ConfigField("masked_room.puzzle_stage_semantics_hidden_dim", int, 96, "Hidden width of the masked-room puzzle-stage semantics supervision head.", min_value=16, max_value=512),
    ConfigField("masked_room.puzzle_stage_semantics_max_sequence_length", int, 6, "Maximum ordered puzzle stages supervised per room for masked-room training.", min_value=1, max_value=12),
    ConfigField("masked_room.topology_marker_weight", float, 2.0, "Relative weight assigned to sparse topology markers, doors, and typed gates when building the masked-room topology focus map.", min_value=0.0),
    ConfigField("masked_room.topology_trace_weight", float, 0.75, "Relative weight assigned to traversability-trace pixels when building the masked-room topology focus map.", min_value=0.0),
    ConfigField("masked_room.topology_focus_dilation", int, 1, "4-neighbour dilation radius applied to the masked-room topology focus map so sparse anchor pixels influence a small local neighborhood.", min_value=0),
    ConfigField("masked_room.validation_fraction", float, 0.1, "Held-out validation fraction for masked-room model selection. Set to 0 to disable a validation split and fall back to train_loss.", min_value=0.0, max_value=0.5),
    ConfigField("masked_room.validation_max_batches", int, 16, "Maximum number of mini-batches evaluated on the masked-room validation split each epoch.", min_value=1),
    ConfigField("masked_room.best_checkpoint_metric", str, "val_loss", "Metric used to select the best masked-room checkpoint. Falls back to train_loss when no validation split exists.", choices=("val_loss", "val_topology_focus_loss", "val_puzzle_stage_semantic_loss", "train_loss")),
    ConfigField("masked_room.room_topology_channels", int, ROOM_TOPOLOGY_CHANNEL_COUNT, "Masked-room topology-channel count.", min_value=1),
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

    if int(validated["diffusion"]["latent_dim"]) != int(validated["vqvae"]["latent_dim"]):
        raise ValueError(
            "diffusion.latent_dim must match vqvae.latent_dim for stage handoff compatibility. "
            f"Got diffusion.latent_dim={validated['diffusion']['latent_dim']} and "
            f"vqvae.latent_dim={validated['vqvae']['latent_dim']}."
        )

    if not bool(validated["diffusion"]["logic_net_enabled"]):
        validated["diffusion"]["logic_net_trainable"] = False
        validated["diffusion"]["guidance_scale"] = 0.0
        validated["diffusion"]["alpha_logic"] = 0.0
        validated["diffusion"]["alpha_logic_tile"] = 0.0
    if not bool(validated["masked_room"]["logic_net_enabled"]):
        validated["masked_room"]["logic_net_trainable"] = False
        validated["masked_room"]["alpha_logic"] = 0.0

    expected_topology_channels = int(ROOM_TOPOLOGY_CHANNEL_COUNT)
    if int(validated["diffusion"]["room_topology_channels"]) != expected_topology_channels:
        raise ValueError(
            "diffusion.room_topology_channels must match the repository room-topology schema. "
            f"Expected {expected_topology_channels}, got {validated['diffusion']['room_topology_channels']}."
        )
    if int(validated["masked_room"]["room_topology_channels"]) != expected_topology_channels:
        raise ValueError(
            "masked_room.room_topology_channels must match the repository room-topology schema. "
            f"Expected {expected_topology_channels}, got {validated['masked_room']['room_topology_channels']}."
        )

    categorical_codebook_size = validated["generation"]["categorical_codebook_size"]
    if categorical_codebook_size is not None and int(categorical_codebook_size) > int(validated["vqvae"]["codebook_size"]):
        raise ValueError(
            "generation.categorical_codebook_size cannot exceed vqvae.codebook_size. "
            f"Got {categorical_codebook_size} > {validated['vqvae']['codebook_size']}."
        )

    if int(validated["fast_sampler"]["num_inference_steps"]) > int(validated["diffusion"]["num_timesteps"]):
        raise ValueError(
            "fast_sampler.num_inference_steps cannot exceed diffusion.num_timesteps. "
            f"Got {validated['fast_sampler']['num_inference_steps']} > {validated['diffusion']['num_timesteps']}."
        )

    if (
        float(validated["runtime"]["checkpoint_storage_cleanup_target_fraction"])
        > float(validated["runtime"]["checkpoint_storage_warning_fraction"])
    ):
        raise ValueError(
            "runtime.checkpoint_storage_cleanup_target_fraction must be <= "
            "runtime.checkpoint_storage_warning_fraction."
        )

    dataset_num_classes = int(validated["dataset"]["num_classes"])
    for section_name in ("diffusion", "masked_room"):
        section = validated[section_name]
        if bool(section["condition_use_reference_room_maps"]):
            vocab_size = int(section["condition_reference_tile_vocab_size"])
            if vocab_size != dataset_num_classes:
                raise ValueError(
                    f"{section_name}.condition_reference_tile_vocab_size={vocab_size} must match "
                    f"dataset.num_classes={dataset_num_classes} when reference-room conditioning is enabled."
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

    topology_curve = [float(v) for v in validated["topology"]["default_target_curve"]]
    if not topology_curve:
        raise ValueError("topology.default_target_curve must be non-empty.")
    if (
        bool(validated["topology"]["allow_candidate_repairs"])
        and not bool(validated["topology"]["enforce_generation_constraints"])
    ):
        logger.warning(
            "topology.allow_candidate_repairs=true has no effect unless "
            "topology.enforce_generation_constraints is also true."
        )

    room_height = int(validated["dataset"]["room_height"])
    room_width = int(validated["dataset"]["room_width"])
    for field_name in ("default_start_coord", "default_goal_coord"):
        coord = [int(v) for v in validated["generation"][field_name]]
        if len(coord) != 2:
            raise ValueError(f"generation.{field_name} must contain exactly two integers [row, col].")
        row, col = int(coord[0]), int(coord[1])
        if row < 0 or row >= room_height or col < 0 or col >= room_width:
            raise ValueError(
                f"generation.{field_name}={coord!r} must lie within the configured room bounds "
                f"{room_height}x{room_width}."
            )

    masked_channel_mult = [int(v) for v in validated["masked_room"]["unet_channel_mult"]]
    masked_unet_num_heads = int(validated["masked_room"]["unet_num_heads"])
    masked_hidden_dim = int(validated["masked_room"]["hidden_dim"])
    if not masked_channel_mult:
        raise ValueError("masked_room.unet_channel_mult must be non-empty.")
    if masked_hidden_dim % masked_unet_num_heads != 0:
        raise ValueError(
            "masked_room.hidden_dim must be divisible by masked_room.unet_num_heads. "
            f"Got hidden_dim={masked_hidden_dim}, unet_num_heads={masked_unet_num_heads}."
        )
    legacy_masked_defaults = {
        "model_channels": 64,
        "hedgehog_feature_dim": 32,
        "graph_auto_linear_attention_nodes": 128,
        "spatial_graph_gate_init": -2.0,
        "spatial_topology_gate_init": -2.0,
        "unet_attention_resolutions": [0, 1],
    }
    changed_legacy = [
        name
        for name, expected in legacy_masked_defaults.items()
        if validated["masked_room"][name] != expected
    ]
    if changed_legacy:
        raise ValueError(
            "The masked-room model is a transformer; these legacy U-Net/linear-attention "
            f"fields are not executable ablations and must remain at defaults: {changed_legacy}."
        )
    min_mask_ratio = float(validated["masked_room"]["min_mask_ratio"])
    max_mask_ratio = float(validated["masked_room"]["max_mask_ratio"])
    if min_mask_ratio > max_mask_ratio:
        raise ValueError(
            "masked_room.min_mask_ratio must be <= masked_room.max_mask_ratio. "
            f"Got {min_mask_ratio} > {max_mask_ratio}."
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
    validated = validate_config(merged)
    # Keep the existing path-aware validation as the source of normalized
    # values, then enforce the Pydantic v2 section schema at runtime.
    from src.pipeline.config_schema import HMOLQDConfigSchema

    HMOLQDConfigSchema.model_validate(validated)
    return validated


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


def find_resolved_config_path(start_path: Optional[str | Path]) -> Optional[Path]:
    """Find the nearest resolved_config snapshot for an artifact or output directory."""
    if start_path is None:
        return None
    anchor = Path(start_path).expanduser().resolve()
    search_root = anchor if anchor.is_dir() else anchor.parent
    candidates = [search_root, *search_root.parents]
    for directory in candidates:
        yaml_path = directory / "resolved_config.yaml"
        if yaml_path.exists():
            return yaml_path
        json_path = directory / "resolved_config.json"
        if json_path.exists():
            return json_path
    return None


def load_resolved_config_for_artifact(start_path: Optional[str | Path]) -> Optional[Dict[str, Any]]:
    """Load and validate the nearest resolved_config snapshot for an artifact path."""
    resolved_path = find_resolved_config_path(start_path)
    if resolved_path is None:
        return None
    suffix = resolved_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return merge_config(yaml_path=str(resolved_path), cli_overrides=None)
    if suffix == ".json":
        with open(resolved_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return validate_config(payload)
    raise ValueError(f"Unsupported resolved config format: {resolved_path}")


def seed_everything(
    seed: Optional[int],
    *,
    cudnn_benchmark: Optional[bool] = None,
    cudnn_deterministic: Optional[bool] = None,
) -> int:
    if seed is None:
        seed = 42
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            if cudnn_deterministic is not None:
                torch.backends.cudnn.deterministic = bool(cudnn_deterministic)
            if cudnn_benchmark is not None:
                torch.backends.cudnn.benchmark = bool(cudnn_benchmark) and not bool(
                    torch.backends.cudnn.deterministic
                )
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
