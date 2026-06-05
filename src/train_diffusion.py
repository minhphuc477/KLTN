"""
Training Pipeline for Latent Diffusion Model
=============================================

Full training pipeline connecting:
- LatentDiffusionModel for generation
- VQ-VAE for latent encoding
- LogicNet for solvability guidance
- DualStreamConditionEncoder for conditioning

Usage:
    python -m src.train_diffusion --data-dir "Data/The Legend of Zelda" --epochs 100
"""

import sys
import argparse
import inspect
import json
import logging
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Set, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.zelda_data.zelda_loader import DungeonBatchSampler, create_dataloader, extract_start_goal, graph_collate_fn
from src.core.latent_diffusion import LatentDiffusionModel, create_latent_diffusion
from src.core.vqvae import SemanticVQVAE as VQVAE, create_vqvae
from src.core.condition_encoder import DualStreamConditionEncoder, create_condition_encoder
from src.core.definitions import (
    GRAPH_EDGE_FEATURE_DIM,
    GRAPH_NODE_FEATURE_DIM,
    GRAPH_TPE_DIM,
    ROOM_HEIGHT,
    ROOM_TOPOLOGY_CHANNEL_COUNT,
    ROOM_WIDTH,
    SEMANTIC_PALETTE,
)
# Use Block V LogicNet (with temperature annealing), not legacy src.ml.logic_net
from src.core.logic_net import LogicNet
from src.pipeline.graph_features import (
    align_nodewise_tensor,
    build_default_node_positions,
    compute_current_node_distance_features,
    compute_rrwp_edge_features,
    compute_rwse_features,
)
from src.pipeline.room_topology_conditioning import (
    DEFAULT_PUZZLE_STAGE_TOKEN_SCALE,
    DEFAULT_SEMANTIC_PUZZLE_OFFSET,
    DEFAULT_SEMANTIC_ROLE_PRIOR_STRENGTH,
    apply_puzzle_stage_control_to_conditioning,
    apply_puzzle_structure_control_to_conditioning,
    apply_puzzle_structure_dropout_batch,
    build_topology_anchor_policy_metadata,
)
from src.config_system import merge_config, seed_everything
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
from src.generation.weighted_bayesian_wfc import (
    TilePrior,
    WeightedBayesianWFCConfig,
    integrate_weighted_wfc_into_pipeline,
)
from src.utils.distributed import (
    DistributedContext,
    average_module_parameters,
    average_gradients,
    destroy_distributed,
    initialize_distributed,
    make_distributed_sampler,
    maybe_barrier,
    reduce_scalar_metrics,
    resolve_device,
)
from src.utils.model_capacity import (
    count_parameters,
    format_parameter_count,
    log_capacity_guardrails,
)
from src.utils.data_loading import dataloader_runtime_kwargs
from src.utils.frozen_latent_cache import FrozenLatentCache

logger = logging.getLogger(__name__)
CARDINAL_DIRECTIONS = ("N", "S", "E", "W")

# ---------------------------------------------------------------------------
# Optional heavy-path libraries (graceful fallback when not installed)
# ---------------------------------------------------------------------------
try:
    from safetensors.torch import load_file as _load_safetensors
    from safetensors.torch import save_file as _save_safetensors
    _HAS_SAFETENSORS = True
except ImportError:
    _HAS_SAFETENSORS = False

try:
    from accelerate import Accelerator
    _HAS_ACCELERATE = True
except ImportError:
    _HAS_ACCELERATE = False
    Accelerator = None  # type: ignore[assignment,misc]

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

@dataclass(init=False)
class DiffusionTrainingConfig:
    """Training configuration for latent diffusion."""

    def __init__(
        self,
        **kwargs: Any,
    ):
        self._init_from_values(**kwargs)

    def _init_from_values(
        self,
        # Data
        data_dir: str = "Data/The Legend of Zelda",
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = True,
        shuffle_train: bool = True,
        shuffle_val: bool = False,
        normalize: bool = True,
        use_vglc: bool = True,
        room_level: bool = True,
        dungeon_batch_mode: bool = True,
        train_dungeon_ids: Optional[List[int]] = None,
        test_dungeon_ids: Optional[List[int]] = None,
        variants: Optional[List[int]] = None,
        num_classes: int = 44,
        node_feature_dim: int = GRAPH_NODE_FEATURE_DIM,
        edge_feature_dim: int = GRAPH_EDGE_FEATURE_DIM,
        
        # VQ-VAE (frozen encoder)
        vqvae_checkpoint: Optional[str] = None,
        vqvae_hidden_dim: int = 96,
        vqvae_codebook_size: int = 256,
        vqvae_architecture: str = "vqvae",
        vqvae_top_codebook_size: Optional[int] = None,
        vqvae_top_latent_dim: Optional[int] = None,
        vqvae_use_coordconv: bool = True,
        vqvae_mrf_penalty_weight: float = 0.05,
        
        # Diffusion Model
        latent_dim: int = 64,
        model_channels: int = 128,
        context_dim: int = 256,
        denoiser_backbone: str = "unet",
        unet_channel_mult: Tuple[int, ...] = (1, 2, 4),
        unet_num_res_blocks: int = 2,
        unet_attention_resolutions: Tuple[int, ...] = (1, 2),
        unet_num_heads: int = 8,
        unet_dropout: float = 0.1,
        dit_depth: int = 4,
        dit_patch_size: int = 1,
        dit_mlp_ratio: float = 4.0,
        condition_hidden_dim: int = 256,
        condition_num_gnn_layers: int = 3,
        condition_num_attention_heads: int = 8,
        condition_dropout: float = 0.1,
        condition_gnn_type: str = "gcn",  # gcn | gat | sage | gps
        condition_use_reference_room_maps: bool = False,
        condition_reference_tile_vocab_size: int = 44,
        condition_reference_embedding_dim: int = 32,
        condition_reference_hidden_dim: int = 64,
        condition_use_rrwp_edge_features: bool = True,
        num_timesteps: int = 1000,
        schedule_type: str = "cosine",
        topology_refinement_mode: str = "gat2",  # none | lightweight | sparse*/gat2* | graphormer
        attention_mode: str = "softmax",
        topology_conditioning_mode: str = "additive",
        hedgehog_feature_dim: int = 32,
        graph_auto_linear_attention_nodes: int = 128,
        spatial_graph_gate_init: float = -2.0,
        spatial_topology_gate_init: float = -2.0,
        use_teacher_forced_neighbor_latents: bool = True,
        puzzle_structure_dropout_prob: float = 0.35,
        use_current_node_distance_features: bool = True,
        current_node_distance_max: int = 8,
        room_topology_channels: int = ROOM_TOPOLOGY_CHANNEL_COUNT,
        topology_supervision_mode: str = "runtime_aligned",
        semantic_role_prior_strength: float = DEFAULT_SEMANTIC_ROLE_PRIOR_STRENGTH,
        semantic_puzzle_offset: int = DEFAULT_SEMANTIC_PUZZLE_OFFSET,
        cfg_dropout_prob: float = 0.1,
        cfg_scale: float = 3.0,
        cfg_schedule_mode: str = "constant",
        cfg_schedule_min_scale: float = 1.0,
        cfg_schedule_power: float = 1.0,
        pag_scale: float = 0.0,
        prediction_type: str = "epsilon",
        diffusion_training_objective: str = "diffusion",
        min_snr_gamma: float = 5.0,
        
        # LogicNet
        logic_net_enabled: bool = True,
        logic_net_trainable: bool = True,
        logic_learning_rate: Optional[float] = None,
        logic_lr_warmup_epochs: int = 5,
        logic_grid_pathfinder: str = "bellman_ford",
        num_logic_iterations: int = 30,
        logic_topology_trace_weight: float = 0.25,
        logic_topology_anchor_weight: float = 0.25,
        logic_global_reach_weight: float = 1.0,
        logic_global_room_weight: float = 0.25,
        guidance_scale: float = 1.0,
        guidance_clamp_magnitude: float = 1.0,
        guidance_relative_norm_cap: float = 0.25,
        guidance_schedule_enabled: bool = True,
        guidance_active_fraction: float = 1.0,
        guidance_decay_power: float = 1.0,
        guidance_max_graph_nodes: int = 512,
        guidance_max_key_lock_pairs: int = 2048,
        guidance_max_guidance_elements: int = 2_000_000,
        
        # Training
        epochs: int = 100,
        learning_rate: float = 1e-4,
        optimizer_weight_decay: float = 1e-5,
        global_lr_warmup_epochs: int = 0,
        alpha_visual: float = 1.0,   # Diffusion loss weight
        alpha_logic: float = 0.1,     # Solvability loss weight
        alpha_logic_tile: float = 0.05,  # Supervised LogicNet tile-classifier loss weight
        alpha_wfc_pseudo: float = 0.0,
        wfc_pseudo_max_samples: int = 2,
        wfc_pseudo_confidence_threshold: float = 0.75,
        min_logic_tile_accuracy_for_guidance: float = 0.4,
        graph_spatial_alignment_weight: float = 0.0,
        logic_loss_mode: str = "predicted_latent",  # predicted_latent | detached_real
        graph_conditioning_mode: str = "node_sequence",  # node_sequence | pooled
        warmup_epochs: int = 5,       # Epochs before adding logic loss
        logic_loss_ramp_epochs: int = 2,
        scheduler_t0: int = 10,
        scheduler_t_mult: int = 2,
        scheduler_eta_min: float = 1e-6,
        ema_decay: float = 0.9999,
        grad_clip_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        gradient_checkpointing: bool = False,
        use_amp: bool = False,
        amp_mixed_precision: str = "fp16",
        use_accelerate: bool = False,
        validation_num_samples: int = 8,
        validation_num_diffusion_samples: int = 64,
        validation_fraction: float = 0.1,
        latent_cache_enabled: bool = True,
        latent_cache_max_items: int = 4096,
        
        # Checkpointing
        checkpoint_dir: str = "./checkpoints",
        save_every: int = 10,
        keep_last: int = 2,
        auto_resume: bool = True,
        resume_checkpoint: Optional[str] = None,
        checkpoint_storage_budget_gb: Optional[float] = None,
        checkpoint_storage_warning_fraction: float = 0.8,
        checkpoint_storage_cleanup_enabled: bool = True,
        checkpoint_storage_cleanup_target_fraction: float = 0.6,
        
        # Device
        device: str = "auto",
        seed: int = 42,
        distributed_enabled: bool = False,
        distributed_backend: str = "nccl",
        distributed_find_unused_parameters: bool = False,
        
        # Quick mode
        quick: bool = False,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = int(max(0, num_workers))
        self.pin_memory = bool(pin_memory)
        self.drop_last = bool(drop_last)
        self.shuffle_train = bool(shuffle_train)
        self.shuffle_val = bool(shuffle_val)
        self.normalize = bool(normalize)
        self.use_vglc = use_vglc
        self.room_level = bool(room_level)
        self.dungeon_batch_mode = bool(dungeon_batch_mode)
        self.train_dungeon_ids = [int(v) for v in (train_dungeon_ids if train_dungeon_ids is not None else list(range(1, 9)))]
        self.test_dungeon_ids = [int(v) for v in (test_dungeon_ids if test_dungeon_ids is not None else [9])]
        self.variants = [int(v) for v in (variants if variants is not None else [1, 2])]
        self.num_classes = int(num_classes)
        self.node_feature_dim = int(max(1, node_feature_dim))
        self.edge_feature_dim = int(max(1, edge_feature_dim))
        
        self.vqvae_checkpoint = vqvae_checkpoint
        self.vqvae_hidden_dim = int(max(8, vqvae_hidden_dim))
        self.vqvae_codebook_size = int(max(8, vqvae_codebook_size))
        self.vqvae_architecture = str(vqvae_architecture or "vqvae")
        self.vqvae_top_codebook_size = (
            None if vqvae_top_codebook_size is None else int(max(8, vqvae_top_codebook_size))
        )
        self.vqvae_top_latent_dim = (
            None if vqvae_top_latent_dim is None else int(max(1, vqvae_top_latent_dim))
        )
        self.vqvae_use_coordconv = bool(vqvae_use_coordconv)
        self.vqvae_mrf_penalty_weight = float(max(0.0, vqvae_mrf_penalty_weight))
        
        self.latent_dim = latent_dim
        self.model_channels = int(model_channels)
        self.context_dim = int(context_dim)
        self.denoiser_backbone = str(denoiser_backbone).strip().lower()
        if self.denoiser_backbone not in {"unet", "dit"}:
            raise ValueError(f"denoiser_backbone must be 'unet' or 'dit', got {denoiser_backbone!r}.")

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
        self.dit_depth = int(max(1, dit_depth))
        self.dit_patch_size = int(max(1, dit_patch_size))
        self.dit_mlp_ratio = float(max(1.0, dit_mlp_ratio))
        if any((self.model_channels * mult) % self.unet_num_heads != 0 for mult in self.unet_channel_mult):
            raise ValueError(
                "Every attention-enabled U-Net channel width must be divisible by unet_num_heads; "
                f"got model_channels={self.model_channels}, unet_channel_mult={self.unet_channel_mult}, "
                f"unet_num_heads={self.unet_num_heads}."
            )
        max_level = len(self.unet_channel_mult) - 1
        if any(level > max_level for level in self.unet_attention_resolutions):
            raise ValueError(
                f"unet_attention_resolutions={self.unet_attention_resolutions!r} contains a level above {max_level}."
            )
        self.condition_hidden_dim = int(condition_hidden_dim)
        self.condition_num_gnn_layers = int(max(1, condition_num_gnn_layers))
        self.condition_num_attention_heads = int(max(1, condition_num_attention_heads))
        self.condition_dropout = float(max(0.0, min(1.0, condition_dropout)))
        if self.condition_hidden_dim % self.condition_num_attention_heads != 0:
            raise ValueError(
                "condition_hidden_dim must be divisible by condition_num_attention_heads; "
                f"got condition_hidden_dim={self.condition_hidden_dim}, "
                f"condition_num_attention_heads={self.condition_num_attention_heads}."
            )
        gnn_type = str(condition_gnn_type).strip().lower()
        if gnn_type not in {"gcn", "gat", "sage", "gps"}:
            raise ValueError(
                f"Invalid condition_gnn_type={condition_gnn_type!r}. "
                "Expected 'gcn', 'gat', 'sage', or 'gps'."
            )
        self.condition_gnn_type = gnn_type
        self.condition_use_reference_room_maps = bool(condition_use_reference_room_maps)
        self.condition_reference_tile_vocab_size = int(max(2, condition_reference_tile_vocab_size))
        self.condition_reference_embedding_dim = int(max(4, condition_reference_embedding_dim))
        self.condition_reference_hidden_dim = int(max(4, condition_reference_hidden_dim))
        self.condition_use_rrwp_edge_features = bool(condition_use_rrwp_edge_features)
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type
        trm = str(topology_refinement_mode).strip().lower()
        if trm == "upgraded":
            trm = "gat2"
        allowed_topology_modes = {
            "none",
            "lightweight",
            "sparse_edge",
            "sparse_directed",
            "sparse_semantic",
            "sparse_directed_semantic",
            "gat2",
            "gat2_directed",
            "gat2_semantic",
            "gat2_directed_semantic",
            "graphormer",
        }
        if trm not in allowed_topology_modes:
            raise ValueError(
                f"Invalid topology_refinement_mode={topology_refinement_mode!r}. "
                "Expected a supported topology refinement ablation."
            )
        self.topology_refinement_mode = trm
        attn_mode = str(attention_mode).strip().lower()
        if attn_mode not in {"softmax", "linear_hedgehog"}:
            raise ValueError(
                f"Invalid attention_mode={attention_mode!r}. "
                "Expected 'softmax' or 'linear_hedgehog'."
            )
        self.attention_mode = attn_mode
        topo_mode = str(topology_conditioning_mode).strip().lower()
        if topo_mode not in {"additive", "spade"}:
            raise ValueError(
                f"Invalid topology_conditioning_mode={topology_conditioning_mode!r}. "
                "Expected 'additive' or 'spade'."
            )
        self.topology_conditioning_mode = topo_mode
        self.hedgehog_feature_dim = int(max(4, hedgehog_feature_dim))
        self.graph_auto_linear_attention_nodes = int(max(0, graph_auto_linear_attention_nodes))
        self.spatial_graph_gate_init = float(spatial_graph_gate_init)
        self.spatial_topology_gate_init = float(spatial_topology_gate_init)
        self.use_teacher_forced_neighbor_latents = bool(use_teacher_forced_neighbor_latents)
        self.puzzle_structure_dropout_prob = float(max(0.0, min(1.0, puzzle_structure_dropout_prob)))
        self.use_current_node_distance_features = bool(use_current_node_distance_features)
        self.current_node_distance_max = int(max(1, current_node_distance_max))
        self.room_topology_channels = int(max(1, room_topology_channels))
        self.topology_supervision_mode = str(topology_supervision_mode).strip().lower()
        if self.topology_supervision_mode not in {"runtime_aligned", "oracle_room_grid"}:
            raise ValueError(
                "topology_supervision_mode must be 'runtime_aligned' or 'oracle_room_grid'."
            )
        self.semantic_role_prior_strength = float(max(0.0, min(1.0, semantic_role_prior_strength)))
        self.semantic_puzzle_offset = int(max(0, semantic_puzzle_offset))
        self.cfg_dropout_prob = float(max(0.0, min(1.0, cfg_dropout_prob)))
        self.cfg_scale = float(max(0.0, cfg_scale))
        self.cfg_schedule_mode = str(cfg_schedule_mode).strip().lower()
        self.cfg_schedule_min_scale = float(max(0.0, cfg_schedule_min_scale))
        self.cfg_schedule_power = float(max(1e-6, cfg_schedule_power))
        self.pag_scale = float(max(0.0, pag_scale))
        self.prediction_type = str(prediction_type).strip().lower()
        self.diffusion_training_objective = str(diffusion_training_objective).strip().lower()
        if self.diffusion_training_objective not in {"diffusion", "flow_matching"}:
            raise ValueError(
                "diffusion_training_objective must be 'diffusion' or 'flow_matching'."
            )
        if self.diffusion_training_objective == "flow_matching" and self.denoiser_backbone != "dit":
            raise ValueError(
                "diffusion_training_objective='flow_matching' requires denoiser_backbone='dit' "
                "for trainable architecture ablations. Use LatentDiffusionModel.flow_matching_loss() "
                "directly only for isolated research probes."
            )
        self.min_snr_gamma = float(max(0.0, min_snr_gamma))
        
        self.logic_net_enabled = bool(logic_net_enabled)
        self.logic_net_trainable = bool(logic_net_trainable) and self.logic_net_enabled
        self.logic_learning_rate = (
            None if logic_learning_rate is None else float(max(1e-8, logic_learning_rate))
        )
        self.logic_lr_warmup_epochs = int(max(0, logic_lr_warmup_epochs))
        self.logic_grid_pathfinder = str(logic_grid_pathfinder).strip().lower()
        if self.logic_grid_pathfinder in {"bellman-ford", "soft_bellman_ford", "soft-bellman-ford"}:
            self.logic_grid_pathfinder = "bellman_ford"
        if self.logic_grid_pathfinder in {"value_iteration", "value-iteration"}:
            self.logic_grid_pathfinder = "vin"
        if self.logic_grid_pathfinder in {"perturb-and-map", "perturb_map", "pmap"}:
            self.logic_grid_pathfinder = "perturb_and_map"
        if self.logic_grid_pathfinder not in {"cnn", "bellman_ford", "vin", "perturb_and_map"}:
            raise ValueError("logic_grid_pathfinder must be 'cnn', 'bellman_ford', 'vin', or 'perturb_and_map'.")
        self.num_logic_iterations = num_logic_iterations
        self.logic_topology_trace_weight = float(max(0.0, logic_topology_trace_weight))
        self.logic_topology_anchor_weight = float(max(0.0, logic_topology_anchor_weight))
        self.logic_global_reach_weight = float(max(0.0, logic_global_reach_weight))
        self.logic_global_room_weight = float(max(0.0, logic_global_room_weight))
        self.guidance_scale = guidance_scale if self.logic_net_enabled else 0.0
        self.guidance_clamp_magnitude = float(max(0.0, guidance_clamp_magnitude))
        self.guidance_relative_norm_cap = float(max(0.0, guidance_relative_norm_cap))
        self.guidance_schedule_enabled = bool(guidance_schedule_enabled)
        self.guidance_active_fraction = float(max(0.05, min(1.0, guidance_active_fraction)))
        self.guidance_decay_power = float(max(0.25, guidance_decay_power))
        self.guidance_max_graph_nodes = int(max(1, guidance_max_graph_nodes))
        self.guidance_max_key_lock_pairs = int(max(0, guidance_max_key_lock_pairs))
        self.guidance_max_guidance_elements = int(max(1, guidance_max_guidance_elements))
        
        self.epochs = epochs if not quick else 2
        self.learning_rate = learning_rate
        self.optimizer_weight_decay = float(max(0.0, optimizer_weight_decay))
        self.global_lr_warmup_epochs = int(max(0, global_lr_warmup_epochs))
        self.alpha_visual = alpha_visual
        self.alpha_logic = alpha_logic if self.logic_net_enabled else 0.0
        self.alpha_logic_tile = float(max(0.0, alpha_logic_tile)) if self.logic_net_enabled else 0.0
        self.alpha_wfc_pseudo = float(max(0.0, alpha_wfc_pseudo)) if self.logic_net_enabled else 0.0
        self.wfc_pseudo_max_samples = int(max(0, wfc_pseudo_max_samples))
        self.wfc_pseudo_confidence_threshold = float(max(0.0, min(1.0, wfc_pseudo_confidence_threshold)))
        self.min_logic_tile_accuracy_for_guidance = float(max(0.0, min_logic_tile_accuracy_for_guidance))
        self.graph_spatial_alignment_weight = float(max(0.0, graph_spatial_alignment_weight))
        mode = str(logic_loss_mode).strip().lower()
        if mode not in {"predicted_latent", "detached_real"}:
            raise ValueError(
                f"Invalid logic_loss_mode={logic_loss_mode!r}. "
                "Expected 'predicted_latent' or 'detached_real'."
            )
        self.logic_loss_mode = mode
        gmode = str(graph_conditioning_mode).strip().lower()
        if gmode not in {"node_sequence", "pooled"}:
            raise ValueError(
                f"Invalid graph_conditioning_mode={graph_conditioning_mode!r}. "
                "Expected 'node_sequence' or 'pooled'."
            )
        self.graph_conditioning_mode = gmode
        self.warmup_epochs = warmup_epochs
        self.logic_loss_ramp_epochs = int(max(1, logic_loss_ramp_epochs))
        self.scheduler_t0 = int(max(1, scheduler_t0))
        self.scheduler_t_mult = int(max(1, scheduler_t_mult))
        self.scheduler_eta_min = float(max(0.0, scheduler_eta_min))
        self.ema_decay = float(min(0.999999, max(0.0, ema_decay)))
        self.grad_clip_norm = float(max(0.0, grad_clip_norm))
        self.gradient_accumulation_steps = int(max(1, gradient_accumulation_steps))
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.use_amp = bool(use_amp)
        amp_mode = str(amp_mixed_precision).strip().lower()
        if amp_mode == "auto":
            amp_mode = "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp16"
        if amp_mode not in {"fp16", "bf16"}:
            raise ValueError("amp_mixed_precision must be 'fp16', 'bf16', or 'auto'.")
        self.amp_mixed_precision = amp_mode
        self.use_accelerate = bool(use_accelerate)
        self.validation_num_samples = int(max(1, validation_num_samples))
        self.validation_num_diffusion_samples = int(max(1, validation_num_diffusion_samples))
        self.validation_fraction = float(max(0.0, min(0.5, validation_fraction)))
        self.latent_cache_enabled = bool(latent_cache_enabled)
        self.latent_cache_max_items = int(max(0, latent_cache_max_items))
        
        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every
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
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.seed = int(seed)
        self.distributed_enabled = bool(distributed_enabled)
        backend = str(distributed_backend).strip().lower()
        if backend not in {"nccl", "gloo"}:
            raise ValueError(
                f"Invalid distributed_backend={distributed_backend!r}. Expected 'nccl' or 'gloo'."
            )
        self.distributed_backend = backend
        self.distributed_find_unused_parameters = bool(distributed_find_unused_parameters)
        
        self.quick = quick
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DiffusionTrainingConfig":
        """Build a training config from either resolved global YAML or flat kwargs."""
        if not isinstance(payload, dict):
            raise TypeError(f"DiffusionTrainingConfig.from_dict expects a dict, got {type(payload).__name__}.")

        if {"dataset", "diffusion", "runtime", "distributed"}.issubset(payload):
            resolved = merge_config(cli_overrides=payload)
            kwargs = diffusion_training_kwargs_from_resolved_config(resolved)
            return cls(**kwargs)

        allowed = set(cls().__dict__.keys())
        return cls(**{key: value for key, value in payload.items() if key in allowed})

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)


def diffusion_training_kwargs_from_resolved_config(
    config: Dict[str, Any],
    *,
    fallback_vqvae_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """Build DiffusionTrainingConfig kwargs from the validated global config payload."""
    stage = config["diffusion"]
    dataset = config["dataset"]
    runtime = config["runtime"]
    distributed = config["distributed"]
    vqvae_stage = config["vqvae"]
    ckpt_path = stage["vqvae_checkpoint"] or fallback_vqvae_checkpoint
    return {
        "data_dir": dataset["data_dir"],
        "batch_size": dataset["batch_size"],
        "num_workers": dataset["num_workers"],
        "pin_memory": dataset["pin_memory"],
        "drop_last": dataset["drop_last"],
        "shuffle_train": dataset["shuffle_train"],
        "shuffle_val": dataset["shuffle_val"],
        "normalize": dataset["normalize"],
        "use_vglc": dataset["use_vglc"],
        "room_level": dataset["room_level"],
        "dungeon_batch_mode": dataset.get("dungeon_batch_mode", True),
        "train_dungeon_ids": dataset.get("train_dungeons", list(range(1, 9))),
        "test_dungeon_ids": dataset.get("test_dungeons", [9]),
        "variants": dataset.get("variants", [1, 2]),
        "num_classes": dataset["num_classes"],
        "node_feature_dim": dataset["node_feature_dim"],
        "edge_feature_dim": dataset["edge_feature_dim"],
        "vqvae_checkpoint": ckpt_path,
        "vqvae_hidden_dim": vqvae_stage["hidden_dim"],
        "vqvae_codebook_size": vqvae_stage["codebook_size"],
        "vqvae_architecture": vqvae_stage["architecture"],
        "vqvae_top_codebook_size": vqvae_stage["top_codebook_size"],
        "vqvae_top_latent_dim": vqvae_stage["top_latent_dim"],
        "vqvae_use_coordconv": vqvae_stage["use_coordconv"],
        "vqvae_mrf_penalty_weight": vqvae_stage["mrf_penalty_weight"],
        "latent_dim": stage["latent_dim"],
        "model_channels": stage["model_channels"],
        "context_dim": stage["context_dim"],
        "denoiser_backbone": stage.get("denoiser_backbone", "unet"),
        "unet_channel_mult": tuple(stage["unet_channel_mult"]),
        "unet_num_res_blocks": stage["unet_num_res_blocks"],
        "unet_attention_resolutions": tuple(stage["unet_attention_resolutions"]),
        "unet_num_heads": stage["unet_num_heads"],
        "unet_dropout": stage["unet_dropout"],
        "dit_depth": stage.get("dit_depth", 4),
        "dit_patch_size": stage.get("dit_patch_size", 1),
        "dit_mlp_ratio": stage.get("dit_mlp_ratio", 4.0),
        "condition_hidden_dim": stage["condition_hidden_dim"],
        "condition_num_gnn_layers": stage["condition_num_gnn_layers"],
        "condition_num_attention_heads": stage["condition_num_attention_heads"],
        "condition_dropout": stage["condition_dropout"],
        "condition_gnn_type": stage["condition_gnn_type"],
        "condition_use_reference_room_maps": stage["condition_use_reference_room_maps"],
        "condition_reference_tile_vocab_size": stage["condition_reference_tile_vocab_size"],
        "condition_reference_embedding_dim": stage["condition_reference_embedding_dim"],
        "condition_reference_hidden_dim": stage["condition_reference_hidden_dim"],
        "condition_use_rrwp_edge_features": stage.get("condition_use_rrwp_edge_features", True),
        "num_timesteps": stage["num_timesteps"],
        "schedule_type": stage["schedule_type"],
        "topology_refinement_mode": stage["topology_refinement_mode"],
        "attention_mode": stage["attention_mode"],
        "topology_conditioning_mode": stage["topology_conditioning_mode"],
        "hedgehog_feature_dim": stage["hedgehog_feature_dim"],
        "graph_auto_linear_attention_nodes": stage["graph_auto_linear_attention_nodes"],
        "spatial_graph_gate_init": stage["spatial_graph_gate_init"],
        "spatial_topology_gate_init": stage["spatial_topology_gate_init"],
        "use_teacher_forced_neighbor_latents": stage["use_teacher_forced_neighbor_latents"],
        "puzzle_structure_dropout_prob": stage.get("puzzle_structure_dropout_prob", 0.35),
        "use_current_node_distance_features": stage["use_current_node_distance_features"],
        "current_node_distance_max": stage["current_node_distance_max"],
        "room_topology_channels": stage["room_topology_channels"],
        "topology_supervision_mode": dataset["topology_supervision_mode"],
        "semantic_role_prior_strength": config["generation"]["semantic_role_prior_strength"],
        "semantic_puzzle_offset": config["generation"]["semantic_puzzle_offset"],
        "cfg_dropout_prob": stage["cfg_dropout_prob"],
        "cfg_scale": stage["cfg_scale"],
        "cfg_schedule_mode": stage["cfg_schedule_mode"],
        "cfg_schedule_min_scale": stage["cfg_schedule_min_scale"],
        "cfg_schedule_power": stage["cfg_schedule_power"],
        "pag_scale": stage.get("pag_scale", 0.0),
        "prediction_type": stage["prediction_type"],
        "diffusion_training_objective": stage.get("training_objective", "diffusion"),
        "min_snr_gamma": stage["min_snr_gamma"],
        "logic_net_enabled": stage["logic_net_enabled"],
        "logic_net_trainable": stage["logic_net_trainable"],
        "logic_learning_rate": stage["logic_learning_rate"],
        "logic_lr_warmup_epochs": stage["logic_lr_warmup_epochs"],
        "logic_grid_pathfinder": stage["logic_grid_pathfinder"],
        "num_logic_iterations": stage["num_logic_iterations"],
        "logic_topology_trace_weight": stage["logic_topology_trace_weight"],
        "logic_topology_anchor_weight": stage["logic_topology_anchor_weight"],
        "logic_global_reach_weight": stage.get("logic_global_reach_weight", 1.0),
        "logic_global_room_weight": stage.get("logic_global_room_weight", 0.25),
        "guidance_scale": stage["guidance_scale"],
        "guidance_clamp_magnitude": stage["guidance_clamp_magnitude"],
        "guidance_relative_norm_cap": stage["guidance_relative_norm_cap"],
        "guidance_schedule_enabled": stage["guidance_schedule_enabled"],
        "guidance_active_fraction": stage["guidance_active_fraction"],
        "guidance_decay_power": stage["guidance_decay_power"],
        "guidance_max_graph_nodes": stage["guidance_max_graph_nodes"],
        "guidance_max_key_lock_pairs": stage["guidance_max_key_lock_pairs"],
        "guidance_max_guidance_elements": stage["guidance_max_guidance_elements"],
        "epochs": stage["epochs"],
        "learning_rate": stage["learning_rate"],
        "optimizer_weight_decay": stage["optimizer_weight_decay"],
        "global_lr_warmup_epochs": stage["global_lr_warmup_epochs"],
        "alpha_visual": stage["alpha_visual"],
        "alpha_logic": stage["alpha_logic"],
        "alpha_logic_tile": stage.get("alpha_logic_tile", 0.05),
        "alpha_wfc_pseudo": stage.get("alpha_wfc_pseudo", 0.0),
        "wfc_pseudo_max_samples": stage.get("wfc_pseudo_max_samples", 2),
        "wfc_pseudo_confidence_threshold": stage.get("wfc_pseudo_confidence_threshold", 0.75),
        "min_logic_tile_accuracy_for_guidance": stage.get("min_logic_tile_accuracy_for_guidance", 0.4),
        "graph_spatial_alignment_weight": stage.get("graph_spatial_alignment_weight", 0.0),
        "logic_loss_mode": stage["logic_loss_mode"],
        "graph_conditioning_mode": stage["graph_conditioning_mode"],
        "warmup_epochs": stage["warmup_epochs"],
        "logic_loss_ramp_epochs": stage.get("logic_loss_ramp_epochs", 2),
        "scheduler_t0": stage["scheduler_t0"],
        "scheduler_t_mult": stage["scheduler_t_mult"],
        "scheduler_eta_min": stage["scheduler_eta_min"],
        "ema_decay": stage["ema_decay"],
        "grad_clip_norm": stage["grad_clip_norm"],
        "gradient_accumulation_steps": stage.get("gradient_accumulation_steps", 1),
        "gradient_checkpointing": stage.get("gradient_checkpointing", False),
        "use_amp": stage.get("use_amp", False),
        "amp_mixed_precision": stage.get("amp_mixed_precision", "fp16"),
        "use_accelerate": stage.get("use_accelerate", False),
        "validation_num_samples": stage["validation_num_samples"],
        "validation_num_diffusion_samples": stage["validation_num_diffusion_samples"],
        "validation_fraction": stage.get("validation_fraction", 0.1),
        "latent_cache_enabled": stage.get("latent_cache_enabled", True),
        "latent_cache_max_items": stage.get("latent_cache_max_items", 4096),
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
        "distributed_enabled": distributed["enabled"],
        "distributed_backend": distributed["backend"],
        "distributed_find_unused_parameters": distributed["find_unused_parameters"],
        "quick": runtime["quick"],
    }


def _load_checkpoint_metadata_sidecar(checkpoint_path: str | Path) -> Dict[str, Any]:
    """Load `<checkpoint>.meta.json` when present."""
    meta_path = Path(f"{checkpoint_path}.meta.json")
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read checkpoint metadata sidecar %s: %s", meta_path, exc)
        return {}


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _validate_vqvae_checkpoint_metadata(
    checkpoint_path: str | Path,
    *,
    metadata: Dict[str, Any],
    expected: Dict[str, Any],
) -> None:
    """Fail fast when a frozen VQ-VAE checkpoint does not match diffusion config."""
    architecture = metadata.get("architecture", {}) if isinstance(metadata, dict) else {}
    if not isinstance(architecture, dict) or not architecture:
        return

    int_fields = (
        "num_classes",
        "latent_dim",
        "hidden_dim",
        "codebook_size",
        "top_codebook_size",
        "top_latent_dim",
    )
    for key in int_fields:
        if key not in architecture or architecture[key] is None:
            continue
        actual = int(architecture[key])
        expected_value = _coerce_optional_int(expected.get(key))
        if expected_value is not None and actual != expected_value:
            raise ValueError(
                f"VQ-VAE checkpoint {checkpoint_path} metadata mismatch for {key}: "
                f"checkpoint={actual}, config={expected_value}. Update the diffusion config "
                "to match the frozen VQ-VAE before training."
            )

    bool_fields = ("use_coordconv",)
    for key in bool_fields:
        if key not in architecture or architecture[key] is None:
            continue
        actual_bool = bool(architecture[key])
        expected_bool = bool(expected.get(key))
        if actual_bool != expected_bool:
            raise ValueError(
                f"VQ-VAE checkpoint {checkpoint_path} metadata mismatch for {key}: "
                f"checkpoint={actual_bool}, config={expected_bool}."
            )

    if "architecture" in architecture and architecture["architecture"] is not None:
        actual_arch = str(architecture["architecture"]).strip().lower()
        expected_arch = str(expected.get("architecture", "vqvae")).strip().lower()
        if actual_arch != expected_arch:
            raise ValueError(
                f"VQ-VAE checkpoint {checkpoint_path} metadata mismatch for architecture: "
                f"checkpoint={actual_arch!r}, config={expected_arch!r}."
            )


def _validate_vqvae_checkpoint_state(
    checkpoint_path: str | Path,
    checkpoint: Dict[str, Any],
    *,
    expected_codebook_size: int,
) -> None:
    """Infer codebook size from old checkpoints without sidecars when possible."""
    state = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        return
    candidate_keys = (
        "quantizer.embedding.weight",
        "bottom_quantizer.embedding.weight",
    )
    for key in candidate_keys:
        value = state.get(key)
        if isinstance(value, torch.Tensor) and value.dim() >= 1:
            actual = int(value.shape[0])
            expected = int(expected_codebook_size)
            if actual != expected:
                raise ValueError(
                    f"VQ-VAE checkpoint {checkpoint_path} codebook size mismatch: "
                    f"checkpoint={actual}, config={expected}. Update vqvae_codebook_size "
                    "or choose the matching frozen VQ-VAE checkpoint."
                )
            return


def _resolve_vqvae_architecture(
    checkpoint_path: Optional[str],
    *,
    num_classes: int,
    latent_dim: int,
    hidden_dim: int,
    codebook_size: int,
    architecture: str = "vqvae",
    top_codebook_size: Optional[int] = None,
    top_latent_dim: Optional[int] = None,
    use_coordconv: bool = True,
    mrf_penalty_weight: float = 0.05,
) -> Dict[str, Any]:
    """Resolve the VQ-VAE architecture and reject checkpoint/config drift."""
    resolved: Dict[str, Any] = {
        "architecture": str(architecture or "vqvae"),
        "num_classes": int(num_classes),
        "latent_dim": int(latent_dim),
        "hidden_dim": int(hidden_dim),
        "codebook_size": int(codebook_size),
        "top_codebook_size": top_codebook_size,
        "top_latent_dim": top_latent_dim,
        "use_coordconv": bool(use_coordconv),
        "mrf_penalty_weight": float(mrf_penalty_weight),
    }
    if not checkpoint_path:
        return resolved

    checkpoint = Path(checkpoint_path)
    metadata = _load_checkpoint_metadata_sidecar(checkpoint)
    architecture = metadata.get("architecture", {}) if isinstance(metadata, dict) else {}
    if isinstance(architecture, dict):
        for key in (
            "architecture",
            "num_classes",
            "latent_dim",
            "hidden_dim",
            "codebook_size",
            "top_codebook_size",
            "top_latent_dim",
            "use_coordconv",
            "mrf_penalty_weight",
        ):
            if key in architecture and architecture[key] is not None:
                resolved[key] = architecture[key]

    return {
        "architecture": str(resolved.get("architecture", "vqvae")),
        "num_classes": int(resolved["num_classes"]),
        "latent_dim": int(resolved["latent_dim"]),
        "hidden_dim": int(resolved["hidden_dim"]),
        "codebook_size": int(resolved["codebook_size"]),
        "top_codebook_size": (
            None if resolved.get("top_codebook_size") is None else int(resolved["top_codebook_size"])
        ),
        "top_latent_dim": (
            None if resolved.get("top_latent_dim") is None else int(resolved["top_latent_dim"])
        ),
        "use_coordconv": bool(resolved["use_coordconv"]),
        "mrf_penalty_weight": float(resolved["mrf_penalty_weight"]),
    }


def _legacy_diffusion_overrides_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Collect only explicitly provided legacy CLI overrides."""
    overrides: Dict[str, Any] = {}

    def _set(name: str, value: Any, *, transform=None) -> None:
        if value is None:
            return
        overrides[name] = transform(value) if transform is not None else value

    _set("data_dir", getattr(args, "data_dir", None))
    _set("batch_size", getattr(args, "batch_size", None))
    _set("room_level", getattr(args, "room_level", None))
    _set("dungeon_batch_mode", getattr(args, "dungeon_batch_mode", None))
    _set("train_dungeon_ids", getattr(args, "train_dungeon_ids", None))
    _set("test_dungeon_ids", getattr(args, "test_dungeon_ids", None))
    _set("variants", getattr(args, "variants", None))
    _set("epochs", getattr(args, "epochs", None))
    _set("learning_rate", getattr(args, "lr", None))
    _set("model_channels", getattr(args, "model_channels", None))
    _set("context_dim", getattr(args, "context_dim", None))
    _set("denoiser_backbone", getattr(args, "denoiser_backbone", None))
    _set("unet_channel_mult", getattr(args, "unet_channel_mult", None), transform=tuple)
    _set("unet_num_res_blocks", getattr(args, "unet_num_res_blocks", None))
    _set(
        "unet_attention_resolutions",
        getattr(args, "unet_attention_resolutions", None),
        transform=tuple,
    )
    _set("unet_num_heads", getattr(args, "unet_num_heads", None))
    _set("unet_dropout", getattr(args, "unet_dropout", None))
    _set("dit_depth", getattr(args, "dit_depth", None))
    _set("dit_patch_size", getattr(args, "dit_patch_size", None))
    _set("dit_mlp_ratio", getattr(args, "dit_mlp_ratio", None))
    _set("pag_scale", getattr(args, "pag_scale", None))
    _set("alpha_logic", getattr(args, "alpha_logic", None))
    _set("alpha_logic_tile", getattr(args, "alpha_logic_tile", None))
    _set("alpha_wfc_pseudo", getattr(args, "alpha_wfc_pseudo", None))
    _set("wfc_pseudo_max_samples", getattr(args, "wfc_pseudo_max_samples", None))
    _set("wfc_pseudo_confidence_threshold", getattr(args, "wfc_pseudo_confidence_threshold", None))
    _set("min_logic_tile_accuracy_for_guidance", getattr(args, "min_logic_tile_accuracy_for_guidance", None))
    _set("graph_spatial_alignment_weight", getattr(args, "graph_spatial_alignment_weight", None))
    _set("logic_loss_mode", getattr(args, "logic_loss_mode", None))
    _set("graph_conditioning_mode", getattr(args, "graph_conditioning_mode", None))
    _set("condition_gnn_type", getattr(args, "condition_gnn_type", None))
    _set(
        "condition_use_reference_room_maps",
        getattr(args, "condition_use_reference_room_maps", None),
    )
    _set(
        "condition_reference_tile_vocab_size",
        getattr(args, "condition_reference_tile_vocab_size", None),
    )
    _set(
        "condition_reference_embedding_dim",
        getattr(args, "condition_reference_embedding_dim", None),
    )
    _set(
        "condition_reference_hidden_dim",
        getattr(args, "condition_reference_hidden_dim", None),
    )
    _set(
        "condition_use_rrwp_edge_features",
        getattr(args, "condition_use_rrwp_edge_features", None),
    )
    _set("vqvae_hidden_dim", getattr(args, "vqvae_hidden_dim", None))
    _set("vqvae_codebook_size", getattr(args, "vqvae_codebook_size", None))
    _set("vqvae_use_coordconv", getattr(args, "vqvae_use_coordconv", None))
    _set("vqvae_mrf_penalty_weight", getattr(args, "vqvae_mrf_penalty_weight", None))
    _set("topology_refinement_mode", getattr(args, "topology_refinement_mode", None))
    _set("attention_mode", getattr(args, "attention_mode", None))
    _set("topology_conditioning_mode", getattr(args, "topology_conditioning_mode", None))
    _set("hedgehog_feature_dim", getattr(args, "hedgehog_feature_dim", None))
    _set(
        "graph_auto_linear_attention_nodes",
        getattr(args, "graph_auto_linear_attention_nodes", None),
    )
    _set("spatial_graph_gate_init", getattr(args, "spatial_graph_gate_init", None))
    _set("spatial_topology_gate_init", getattr(args, "spatial_topology_gate_init", None))
    _set(
        "use_teacher_forced_neighbor_latents",
        getattr(args, "use_teacher_forced_neighbor_latents", None),
    )
    _set(
        "puzzle_structure_dropout_prob",
        getattr(args, "puzzle_structure_dropout_prob", None),
    )
    _set(
        "use_current_node_distance_features",
        getattr(args, "use_current_node_distance_features", None),
    )
    _set("current_node_distance_max", getattr(args, "current_node_distance_max", None))
    _set("logic_net_enabled", getattr(args, "logic_net_enabled", None))
    _set("logic_net_trainable", getattr(args, "logic_net_trainable", None))
    _set("logic_learning_rate", getattr(args, "logic_learning_rate", None))
    _set("logic_lr_warmup_epochs", getattr(args, "logic_lr_warmup_epochs", None))
    _set("logic_grid_pathfinder", getattr(args, "logic_grid_pathfinder", None))
    _set("global_lr_warmup_epochs", getattr(args, "global_lr_warmup_epochs", None))
    _set("logic_loss_ramp_epochs", getattr(args, "logic_loss_ramp_epochs", None))
    _set("guidance_scale", getattr(args, "guidance_scale", None))
    _set("logic_topology_trace_weight", getattr(args, "logic_topology_trace_weight", None))
    _set("logic_topology_anchor_weight", getattr(args, "logic_topology_anchor_weight", None))
    _set("logic_global_reach_weight", getattr(args, "logic_global_reach_weight", None))
    _set("logic_global_room_weight", getattr(args, "logic_global_room_weight", None))
    _set("diffusion_training_objective", getattr(args, "diffusion_training_objective", None))
    _set("latent_cache_enabled", getattr(args, "latent_cache_enabled", None))
    _set("latent_cache_max_items", getattr(args, "latent_cache_max_items", None))
    _set("checkpoint_dir", getattr(args, "checkpoint_dir", None))
    _set("keep_last", getattr(args, "keep_last", None))
    _set("auto_resume", getattr(args, "auto_resume", None))
    _set("resume_checkpoint", getattr(args, "resume", None))
    _set("checkpoint_storage_budget_gb", getattr(args, "checkpoint_storage_budget_gb", None))
    _set("checkpoint_storage_warning_fraction", getattr(args, "checkpoint_storage_warning_fraction", None))
    _set("checkpoint_storage_cleanup_enabled", getattr(args, "checkpoint_storage_cleanup_enabled", None))
    _set("checkpoint_storage_cleanup_target_fraction", getattr(args, "checkpoint_storage_cleanup_target_fraction", None))
    _set("vqvae_checkpoint", getattr(args, "vqvae_checkpoint", None))
    _set("device", getattr(args, "device", None))
    _set("seed", getattr(args, "seed", None))
    _set("distributed_enabled", getattr(args, "distributed_enabled", None))
    _set("distributed_backend", getattr(args, "distributed_backend", None))
    _set("quick", getattr(args, "quick", None))
    return overrides


def build_diffusion_training_config_from_args(args: argparse.Namespace) -> DiffusionTrainingConfig:
    """Resolve the standalone diffusion CLI into a validated DiffusionTrainingConfig."""
    base_kwargs: Dict[str, Any] = {}
    config_path = getattr(args, "config", None)
    if config_path:
        resolved = merge_config(yaml_path=str(config_path), cli_overrides=None)
        base_kwargs = diffusion_training_kwargs_from_resolved_config(resolved)
        if getattr(args, "verbose", None) is None:
            setattr(args, "verbose", bool(resolved["runtime"]["verbose"]))
    legacy_overrides = _legacy_diffusion_overrides_from_args(args)
    return DiffusionTrainingConfig(**{**base_kwargs, **legacy_overrides})


def compute_teacher_validation_total_loss(
    *,
    val_diffusion_loss: float,
    val_logic_loss: float,
    alpha_visual: float,
    alpha_logic: float,
    include_logic_loss: bool,
) -> float:
    """Mirror the training objective for checkpoint selection."""
    total = float(alpha_visual) * float(val_diffusion_loss)
    if include_logic_loss and float(alpha_logic) > 0.0:
        total += float(alpha_logic) * float(val_logic_loss)
    return float(total)


# =============================================================================
# INTEGRATED DIFFUSION TRAINER
# =============================================================================

class DiffusionTrainer:
    """
    Unified trainer for latent diffusion dungeon generation.
    
    Components:
    1. VQ-VAE: Encode real dungeons to latent space (frozen)
    2. ConditionEncoder: Process graph + spatial context
    3. LatentDiffusion: Generate dungeons in latent space
    4. LogicNet: Differentiable solvability (gradient guidance)
    """

    @staticmethod
    def _adamw_decay_param_groups(
        name: str,
        module: nn.Module,
        *,
        weight_decay: float,
    ) -> List[Dict[str, Any]]:
        """Build AdamW groups that exclude biases and 1D scale parameters from decay."""
        decay_params: List[nn.Parameter] = []
        no_decay_params: List[nn.Parameter] = []
        seen: Set[int] = set()
        for param_name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            if param.ndim <= 1 or param_name.endswith(".bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        groups: List[Dict[str, Any]] = []
        if decay_params:
            groups.append(
                {
                    "name": f"{name}_decay",
                    "params": decay_params,
                    "weight_decay": float(max(0.0, weight_decay)),
                }
            )
        if no_decay_params:
            groups.append(
                {
                    "name": f"{name}_no_decay",
                    "params": no_decay_params,
                    "weight_decay": 0.0,
                }
            )
        return groups
    
    def __init__(
        self,
        config: DiffusionTrainingConfig,
        distributed_context: Optional[DistributedContext] = None,
        vqvae: Optional[VQVAE] = None,
        diffusion: Optional[LatentDiffusionModel] = None,
        condition_encoder: Optional[DualStreamConditionEncoder] = None,
        logic_net: Optional[LogicNet] = None,
    ):
        self.config = config
        self.distributed_context = distributed_context or DistributedContext(
            enabled=False,
            backend=str(getattr(config, "distributed_backend", "nccl")),
            world_size=1,
            rank=0,
            local_rank=0,
        )
        self.device = resolve_device(config.device, self.distributed_context)
        
        # Initialize models
        self.vqvae = vqvae or self._create_vqvae()
        self.diffusion = diffusion or self._create_diffusion()
        self.model = self.diffusion
        self.condition_encoder = condition_encoder or self._create_condition_encoder()
        self.logic_net = logic_net or self._create_logic_net()
        diffusion_context_dim = int(getattr(self.diffusion, "context_dim", self.config.context_dim))
        encoder_output_dim = int(getattr(self.condition_encoder, "output_dim", self.config.context_dim))
        if diffusion_context_dim != encoder_output_dim:
            raise ValueError(
                "context_dim mismatch between diffusion and condition encoder: "
                f"diffusion={diffusion_context_dim}, condition_encoder={encoder_output_dim}."
            )
        
        # Move to device
        self.vqvae = self.vqvae.to(self.device)
        self.diffusion = self.diffusion.to(self.device)
        self.condition_encoder = self.condition_encoder.to(self.device)
        self.logic_net = self.logic_net.to(self.device)
        
        # Freeze VQ-VAE
        self.vqvae.eval()
        self.vqvae.requires_grad_(False)

        if not bool(getattr(config, "logic_net_trainable", True)):
            self.logic_net.eval()
            for param in self.logic_net.parameters():
                param.requires_grad = False
        
        # --- Wire LogicNet into diffusion model's GradientGuidance ---
        # This enables gradient guidance during sampling: at each denoising
        # step, ∇_{x_t}L_logic nudges the sample toward solvable configs.
        self._configure_guidance()

        # --- Gradient checkpointing (reduces VRAM at cost of recompute) ---
        if getattr(config, 'gradient_checkpointing', False):
            if hasattr(self.model, 'enable_gradient_checkpointing'):
                self.model.enable_gradient_checkpointing()
            elif hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            # For custom UNet/DiT modules, enable torch.utils.checkpoint on attention blocks
            for module in self.model.modules():
                if hasattr(module, 'use_checkpoint'):
                    module.use_checkpoint = True
            logger.info("Gradient checkpointing enabled on diffusion model.")
        
        optimizer_groups = self._adamw_decay_param_groups(
            "diffusion",
            self.diffusion,
            weight_decay=float(config.optimizer_weight_decay),
        )
        optimizer_groups.extend(
            self._adamw_decay_param_groups(
                "condition_encoder",
                self.condition_encoder,
                weight_decay=float(config.optimizer_weight_decay),
            )
        )
        if bool(getattr(config, "logic_net_trainable", True)):
            logic_groups = self._adamw_decay_param_groups(
                "logic_net",
                self.logic_net,
                weight_decay=float(config.optimizer_weight_decay),
            )
            if getattr(config, "logic_learning_rate", None) is not None:
                for logic_group in logic_groups:
                    logic_group["lr"] = float(config.logic_learning_rate)
            optimizer_groups.extend(logic_groups)

        self.optimizer = optim.AdamW(
            optimizer_groups,
            lr=config.learning_rate,
            weight_decay=0.0,
        )
        for group in self.optimizer.param_groups:
            group.setdefault("base_lr", float(group.get("lr", config.learning_rate)))

        # --- Accelerate / AMP integration ---
        self._accelerator: Optional[Any] = None
        self._amp_enabled = bool(getattr(config, "use_amp", False))
        self._amp_mixed_precision = str(getattr(config, "amp_mixed_precision", "fp16")).strip().lower()
        self._amp_dtype = torch.bfloat16 if self._amp_mixed_precision == "bf16" else torch.float16
        scaler_enabled = (
            self._amp_enabled
            and self._amp_mixed_precision == "fp16"
            and torch.device(self.device).type == "cuda"
        )
        try:
            self._grad_scaler = torch.amp.GradScaler("cuda", enabled=bool(scaler_enabled))
        except TypeError:  # Older PyTorch compatibility.
            self._grad_scaler = torch.cuda.amp.GradScaler(enabled=bool(scaler_enabled))
        should_use_accelerate = bool(getattr(config, "use_accelerate", False))
        if should_use_accelerate and _HAS_ACCELERATE and not bool(getattr(self.distributed_context, "enabled", False)):
            try:
                mixed_precision = self._amp_mixed_precision if self._amp_enabled else "no"
                self._accelerator = Accelerator(mixed_precision=mixed_precision)
                self.diffusion, self.condition_encoder, self.logic_net, self.optimizer = (
                    self._accelerator.prepare(
                        self.diffusion,
                        self.condition_encoder,
                        self.logic_net,
                        self.optimizer,
                    )
                )
                self.model = self.diffusion
                self._configure_guidance()
                logger.info(
                    "Accelerate initialized (mixed_precision=%s).", mixed_precision
                )
            except Exception as _acc_err:  # noqa: BLE001
                logger.warning(
                    "Accelerate init failed (%s); falling back to plain PyTorch.", _acc_err
                )
                self._accelerator = None
        elif should_use_accelerate and not _HAS_ACCELERATE:
            logger.warning("use_accelerate=True but accelerate is not installed; using plain PyTorch.")
        elif should_use_accelerate and bool(getattr(self.distributed_context, "enabled", False)):
            logger.info(
                "Skipping Accelerate in torchrun mode; using DistributedSampler plus explicit gradient averaging."
            )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.scheduler_t0,
            T_mult=config.scheduler_t_mult,
            eta_min=config.scheduler_eta_min,
        )
        
        # Metrics tracking
        self.epoch = 0
        self.global_step = 0
        self._accumulation_micro_steps = 0
        self._estimated_total_steps = self._default_estimated_total_steps()
        self._apply_lr_warmup(completed_steps=0)
        
        # --- Phase 4A: EMA model weights ---
        import copy
        self.ema_diffusion = copy.deepcopy(self.diffusion)
        self._configure_guidance(self.ema_diffusion)
        self.ema_diffusion.eval()
        for param in self.ema_diffusion.parameters():
            param.requires_grad = False
        self.ema_decay = float(config.ema_decay)
        self._latent_cache = FrozenLatentCache(
            enabled=bool(getattr(config, "latent_cache_enabled", True)),
            max_items=int(getattr(config, "latent_cache_max_items", 4096)),
        )

    def _configure_guidance(self, diffusion: Optional[nn.Module] = None) -> None:
        """Wire current LogicNet and config values into gradient guidance."""
        target = self.diffusion if diffusion is None else diffusion
        guidance = target.guidance
        logic_enabled = bool(getattr(self.config, "logic_net_enabled", True))
        if isinstance(getattr(type(guidance), "logic_net", None), property):
            guidance.logic_net = self.logic_net if logic_enabled else None
        else:
            object.__setattr__(guidance, "logic_net", self.logic_net if logic_enabled else None)
        modules = getattr(guidance, "_modules", None)
        if isinstance(modules, dict):
            modules.pop("logic_net", None)
        guidance.guidance_scale = float(getattr(self.config, "guidance_scale", 1.0)) if logic_enabled else 0.0
        guidance.clamp_magnitude = float(getattr(self.config, "guidance_clamp_magnitude", 1.0))
        guidance.relative_norm_cap = float(getattr(self.config, "guidance_relative_norm_cap", 0.25))
        guidance.schedule_enabled = bool(getattr(self.config, "guidance_schedule_enabled", True))
        guidance.active_fraction = float(getattr(self.config, "guidance_active_fraction", 1.0))
        guidance.decay_power = float(getattr(self.config, "guidance_decay_power", 1.0))
        guidance.max_graph_nodes = int(getattr(self.config, "guidance_max_graph_nodes", 512))
        guidance.max_key_lock_pairs = int(getattr(self.config, "guidance_max_key_lock_pairs", 2048))
        guidance.max_guidance_elements = int(getattr(self.config, "guidance_max_guidance_elements", 2_000_000))
    
    def _create_vqvae(self) -> VQVAE:
        """Create or load VQ-VAE."""
        vqvae_arch = _resolve_vqvae_architecture(
            self.config.vqvae_checkpoint,
            num_classes=self.config.num_classes,
            latent_dim=self.config.latent_dim,
            hidden_dim=self.config.vqvae_hidden_dim,
            codebook_size=self.config.vqvae_codebook_size,
            architecture=self.config.vqvae_architecture,
            top_codebook_size=self.config.vqvae_top_codebook_size,
            top_latent_dim=self.config.vqvae_top_latent_dim,
            use_coordconv=self.config.vqvae_use_coordconv,
            mrf_penalty_weight=self.config.vqvae_mrf_penalty_weight,
        )
        self.config.vqvae_hidden_dim = int(vqvae_arch["hidden_dim"])
        self.config.vqvae_codebook_size = int(vqvae_arch["codebook_size"])
        self.config.vqvae_architecture = str(vqvae_arch["architecture"])
        self.config.vqvae_top_codebook_size = vqvae_arch.get("top_codebook_size")
        self.config.vqvae_top_latent_dim = vqvae_arch.get("top_latent_dim")
        self.config.vqvae_use_coordconv = bool(vqvae_arch["use_coordconv"])
        self.config.vqvae_mrf_penalty_weight = float(vqvae_arch["mrf_penalty_weight"])
        vqvae = create_vqvae(
            architecture=vqvae_arch["architecture"],
            num_classes=vqvae_arch["num_classes"],
            latent_dim=vqvae_arch["latent_dim"],
            hidden_dim=vqvae_arch["hidden_dim"],
            codebook_size=vqvae_arch["codebook_size"],
            top_codebook_size=vqvae_arch.get("top_codebook_size"),
            top_latent_dim=vqvae_arch.get("top_latent_dim"),
            use_coordconv=vqvae_arch["use_coordconv"],
            mrf_penalty_weight=vqvae_arch["mrf_penalty_weight"],
        )
        
        if self.config.vqvae_checkpoint:
            checkpoint = safe_torch_load(self.config.vqvae_checkpoint, map_location='cpu')
            _validate_vqvae_checkpoint_state(
                self.config.vqvae_checkpoint,
                checkpoint,
                expected_codebook_size=int(self.config.vqvae_codebook_size),
            )
            vqvae.load_state_dict(checkpoint['model_state_dict'])
            logger.info(
                "Loaded VQ-VAE from %s with architecture=%s num_classes=%d latent_dim=%d hidden_dim=%d codebook_size=%d",
                self.config.vqvae_checkpoint,
                vqvae_arch["architecture"],
                vqvae_arch["num_classes"],
                vqvae_arch["latent_dim"],
                vqvae_arch["hidden_dim"],
                vqvae_arch["codebook_size"],
            )
        
        return vqvae
    
    def _create_diffusion(self) -> LatentDiffusionModel:
        """Create latent diffusion model."""
        return create_latent_diffusion(
            latent_dim=self.config.latent_dim,
            model_channels=self.config.model_channels,
            context_dim=self.config.context_dim,
            denoiser_backbone=self.config.denoiser_backbone,
            unet_channel_mult=self.config.unet_channel_mult,
            unet_num_res_blocks=self.config.unet_num_res_blocks,
            unet_attention_resolutions=self.config.unet_attention_resolutions,
            unet_num_heads=self.config.unet_num_heads,
            unet_dropout=self.config.unet_dropout,
            dit_depth=self.config.dit_depth,
            dit_patch_size=self.config.dit_patch_size,
            dit_mlp_ratio=self.config.dit_mlp_ratio,
            num_timesteps=self.config.num_timesteps,
            schedule_type=self.config.schedule_type,
            prediction_type=self.config.prediction_type,
            cfg_dropout_prob=self.config.cfg_dropout_prob,
            cfg_scale=self.config.cfg_scale,
            cfg_schedule_mode=self.config.cfg_schedule_mode,
            cfg_schedule_min_scale=self.config.cfg_schedule_min_scale,
            cfg_schedule_power=self.config.cfg_schedule_power,
            pag_scale=self.config.pag_scale,
            min_snr_gamma=self.config.min_snr_gamma,
            guidance_scale=self.config.guidance_scale,
            topology_refinement_mode=self.config.topology_refinement_mode,
            attention_mode=self.config.attention_mode,
            topology_conditioning_mode=self.config.topology_conditioning_mode,
            hedgehog_feature_dim=self.config.hedgehog_feature_dim,
            graph_auto_linear_attention_nodes=self.config.graph_auto_linear_attention_nodes,
            spatial_graph_gate_init=self.config.spatial_graph_gate_init,
            spatial_topology_gate_init=self.config.spatial_topology_gate_init,
            room_topology_channels=self.config.room_topology_channels,
            training_objective=self.config.diffusion_training_objective,
        )
    
    def _create_condition_encoder(self) -> DualStreamConditionEncoder:
        """Create condition encoder."""
        return create_condition_encoder(
            latent_dim=self.config.latent_dim,
            node_feature_dim=self.config.node_feature_dim,
            edge_feature_dim=self.config.edge_feature_dim,
            output_dim=self.config.context_dim,
            hidden_dim=self.config.condition_hidden_dim,
            num_gnn_layers=self.config.condition_num_gnn_layers,
            gnn_type=self.config.condition_gnn_type,
            num_attention_heads=self.config.condition_num_attention_heads,
            dropout=self.config.condition_dropout,
            use_current_node_distance_features=self.config.use_current_node_distance_features,
            use_reference_room_maps=self.config.condition_use_reference_room_maps,
            reference_num_tile_types=self.config.condition_reference_tile_vocab_size,
            reference_embedding_dim=self.config.condition_reference_embedding_dim,
            reference_hidden_dim=self.config.condition_reference_hidden_dim,
            use_rrwp_edge_features=self.config.condition_use_rrwp_edge_features,
        )

    def _stack_conditioning_vectors(self, cond_vectors: List[torch.Tensor]) -> torch.Tensor:
        """Stack per-sample conditioning vectors into batch tensor for diffusion."""
        if not cond_vectors:
            raise ValueError("cond_vectors must be non-empty")

        if self.config.graph_conditioning_mode == "node_sequence":
            max_nodes = max(int(c.shape[0]) for c in cond_vectors)
            padded = []
            for c in cond_vectors:
                if c.shape[0] < max_nodes:
                    pad = torch.zeros(
                        max_nodes - c.shape[0],
                        c.shape[1],
                        device=c.device,
                        dtype=c.dtype,
                    )
                    c = torch.cat([c, pad], dim=0)
                padded.append(c.unsqueeze(0))
            return torch.cat(padded, dim=0)  # [B, N_max, context_dim]

        return torch.cat(cond_vectors, dim=0)  # [B, context_dim]
    
    def _create_logic_net(self) -> LogicNet:
        """Create LogicNet for solvability."""
        return LogicNet(
            latent_dim=self.config.latent_dim,
            num_classes=self.config.num_classes,
            num_iterations=self.config.num_logic_iterations,
            grid_pathfinder_type=self.config.logic_grid_pathfinder,
            topology_trace_weight=self.config.logic_topology_trace_weight,
            topology_anchor_weight=self.config.logic_topology_anchor_weight,
            global_reach_weight=self.config.logic_global_reach_weight,
            global_room_weight=self.config.logic_global_room_weight,
        )
    
    def _encode_to_latent_uncached(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to VQ-VAE latent space.
        
        Handles data format conversion:
        - Data loader returns [B, 1, H, W] normalized tile IDs in [0, 1]
        - VQ-VAE expects [B, C=44, H, W] one-hot encoded tiles
        
        Conversion: denormalize â†’ integer tile IDs â†’ one-hot â†’ VQ-VAE encode
        """
        import torch.nn.functional as F
        
        with torch.no_grad():
            num_classes = self.vqvae.num_classes  # 44
            
            if x.shape[1] == 1:
                # Data loader format: [B, 1, H, W] normalized [0, 1]
                # Step 1: Denormalize to integer tile IDs
                tile_ids = (x.squeeze(1) * (num_classes - 1)).round().long()
                tile_ids = tile_ids.clamp(0, num_classes - 1)
                
                # Step 2: One-hot encode â†’ [B, H, W, C] â†’ permute to [B, C, H, W]
                x_onehot = F.one_hot(tile_ids, num_classes=num_classes)
                x_onehot = x_onehot.permute(0, 3, 1, 2).float()
            elif x.shape[1] == num_classes:
                # Already one-hot: [B, C, H, W]
                x_onehot = x
            else:
                raise ValueError(
                    f"Unexpected input channels {x.shape[1]}. "
                    f"Expected 1 (normalized tile IDs) or {num_classes} (one-hot)."
                )
            
            # encode() returns (z_q, indices) â€” 2 values, not 3
            z_q, _indices = self.vqvae.encode(x_onehot)
        return z_q

    def _latent_cache_key(self, x: torch.Tensor) -> Optional[Tuple[Any, ...]]:
        """Build a stable key for one frozen-tokenizer input map."""
        namespace = (
            str(getattr(self.config, "vqvae_checkpoint", "") or ""),
            str(getattr(self.config, "vqvae_architecture", "vqvae")),
            int(getattr(self.config, "num_classes", 44)),
        )
        return self._latent_cache.key_for_tensor(x, namespace=namespace)

    def _cache_get_latent(self, key: Optional[Tuple[Any, ...]]) -> Optional[torch.Tensor]:
        cache = getattr(self, "_latent_cache", None)
        if not isinstance(cache, FrozenLatentCache):
            cache = FrozenLatentCache(
                enabled=bool(getattr(self.config, "latent_cache_enabled", True)),
                max_items=int(getattr(self.config, "latent_cache_max_items", 4096)),
            )
            self._latent_cache = cache
        return cache.get(key, device=self.device)

    def _cache_put_latent(self, key: Optional[Tuple[Any, ...]], latent: torch.Tensor) -> None:
        cache = getattr(self, "_latent_cache", None)
        if not isinstance(cache, FrozenLatentCache):
            cache = FrozenLatentCache(
                enabled=bool(getattr(self.config, "latent_cache_enabled", True)),
                max_items=int(getattr(self.config, "latent_cache_max_items", 4096)),
            )
            self._latent_cache = cache
        cache.put(key, latent)

    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to VQ-VAE latent space.

        The VQ-VAE is frozen during diffusion training, so repeated room maps
        can reuse in-memory latents across epochs and neighbor-conditioning
        calls without changing gradients through Block IV.
        """
        with torch.no_grad():
            if x.dim() != 4:
                raise ValueError(f"Expected rank-4 map tensor [B,C,H,W], got {tuple(x.shape)}.")
            keys = [self._latent_cache_key(sample.unsqueeze(0)) for sample in x]
            cached_latents: List[Optional[torch.Tensor]] = [self._cache_get_latent(key) for key in keys]
            missing_indices = [index for index, latent in enumerate(cached_latents) if latent is None]
            if not missing_indices:
                return torch.cat([latent for latent in cached_latents if latent is not None], dim=0)

            missing_maps = torch.cat([x[index : index + 1] for index in missing_indices], dim=0)
            encoded_missing = self._encode_to_latent_uncached(missing_maps)
            for offset, index in enumerate(missing_indices):
                latent = encoded_missing[offset : offset + 1].detach()
                self._cache_put_latent(keys[index], latent)
                cached_latents[index] = latent

            return torch.cat(
                [latent.to(self.device, non_blocking=True) for latent in cached_latents if latent is not None],
                dim=0,
            )
    
    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent codes back to tile logits.
        
        Returns:
            Tensor [B, C=44, H, W] of tile class logits
        """
        with torch.no_grad():
            return self.vqvae.decode(z, target_size=(ROOM_HEIGHT, ROOM_WIDTH))

    def _encode_edge_features(self, graph_dict: dict) -> Optional[torch.Tensor]:
        """Load explicit edge features when available, else fall back to one-hot labels."""
        edge_features = graph_dict.get("edge_features")
        if edge_features is not None:
            if not isinstance(edge_features, torch.Tensor):
                edge_features = torch.tensor(edge_features, dtype=torch.float32)
            edge_features = edge_features.to(self.device, dtype=torch.float32)
            if edge_features.numel() == 0:
                return None
            if edge_features.dim() == 1:
                edge_features = edge_features.unsqueeze(-1)
            return edge_features

        edge_attr = graph_dict.get('edge_attr')
        if edge_attr is None:
            return None
        if not isinstance(edge_attr, torch.Tensor):
            edge_attr = torch.tensor(edge_attr, dtype=torch.long)
        if edge_attr.numel() == 0:
            return None
        edge_attr = edge_attr.to(self.device, dtype=torch.long)
        num_edge_types = int(getattr(self.config, "edge_feature_dim", GRAPH_EDGE_FEATURE_DIM))
        # Ensure edge_attr is 1D for one_hot
        if edge_attr.dim() > 1:
            edge_attr = edge_attr.squeeze()
        edge_attr_clamped = edge_attr.clamp(0, num_edge_types - 1)
        return F.one_hot(
            edge_attr_clamped, num_classes=num_edge_types
        ).float()

    @staticmethod
    def _call_supports_keyword(callable_obj: Any, keyword: str) -> bool:
        """Return whether a callable can accept a keyword without invoking it."""
        target = getattr(callable_obj, "forward", callable_obj)
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):
            return True
        for parameter in signature.parameters.values():
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                return True
        return keyword in signature.parameters
    
    def get_dummy_conditioning(self, batch_size: int) -> torch.Tensor:
        """
        Get fallback conditioning when graph data is unavailable.
        
        Used only as a fallback during validation or when graph loading fails.
        During training, real graph data from .dot files is used instead.
        """
        if self.config.graph_conditioning_mode == "node_sequence":
            return torch.zeros(batch_size, 1, self.config.context_dim, device=self.device)
        return torch.zeros(batch_size, self.config.context_dim, device=self.device)

    def _encode_neighbor_maps_to_latents(
        self,
        neighbor_maps: Optional[Dict[str, Optional[torch.Tensor]]],
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Encode teacher-forced neighboring room maps into VQ-VAE latent tensors.

        This closes the train/inference gap between:
        - training: room-local conditioner saw only null neighbors, and
        - inference: room-local conditioner receives actual generated neighbors.
        """
        latents: Dict[str, Optional[torch.Tensor]] = {direction: None for direction in CARDINAL_DIRECTIONS}
        if not isinstance(neighbor_maps, dict):
            return latents

        for direction in CARDINAL_DIRECTIONS:
            room_map = neighbor_maps.get(direction)
            if room_map is None:
                continue
            if not isinstance(room_map, torch.Tensor):
                room_map = torch.as_tensor(room_map, dtype=torch.float32)
            if room_map.dim() == 2:
                room_map = room_map.unsqueeze(0).unsqueeze(0)
            elif room_map.dim() == 3:
                room_map = room_map.unsqueeze(0)
            if room_map.dim() != 4:
                raise ValueError(
                    f"Neighbor room map for direction {direction!r} must be rank-2/3/4, got {tuple(room_map.shape)}."
                )
            room_map = room_map.to(self.device, dtype=torch.float32)
            latents[direction] = self.encode_to_latent(room_map).detach()

        return latents

    def _prediction_to_x0(
        self,
        prediction: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Convert diffusion model prediction to predicted clean latent x0."""
        sqrt_alpha_t = self.diffusion.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_t = self.diffusion.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        if self.diffusion.prediction_type == 'v':
            # v-prediction: x0 = sqrt(alpha_bar_t) * x_t - sqrt(1-alpha_bar_t) * v
            return sqrt_alpha_t * x_t - sqrt_one_minus_alpha_t * prediction

        # epsilon-prediction: x0 = (x_t - sqrt(1-alpha_bar_t) * eps) / sqrt(alpha_bar_t)
        return (x_t - sqrt_one_minus_alpha_t * prediction) / (sqrt_alpha_t + 1e-8)

    def _diffusion_objective_loss(
        self,
        z_0: torch.Tensor,
        conditioning: torch.Tensor,
        graph_data: Optional[Dict[str, torch.Tensor]] = None,
        *,
        model: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Dispatch the configured latent objective without changing callers."""
        objective = str(getattr(self.config, "diffusion_training_objective", "diffusion")).strip().lower()
        target_model = model or self.diffusion
        compute_loss = getattr(target_model, "compute_loss", None)
        if callable(compute_loss):
            previous_objective = getattr(target_model, "training_objective", None)
            if hasattr(target_model, "training_objective"):
                target_model.training_objective = objective
            try:
                return compute_loss(z_0, conditioning, graph_data=graph_data)
            finally:
                if previous_objective is not None and hasattr(target_model, "training_objective"):
                    target_model.training_objective = previous_objective
        if objective == "flow_matching":
            flow_loss = getattr(target_model, "flow_matching_loss", None)
            if not callable(flow_loss):
                raise TypeError(
                    "diffusion_training_objective='flow_matching' requires the diffusion model "
                    "to implement flow_matching_loss()."
                )
            return flow_loss(z_0, conditioning, graph_data=graph_data)
        if objective != "diffusion":
            raise ValueError(f"Unsupported diffusion_training_objective={objective!r}.")
        return target_model.training_loss(z_0, conditioning, graph_data=graph_data)

    def _autocast_context(self):
        """Return the active mixed-precision context, or a no-op context."""
        if self._accelerator is not None:
            return self._accelerator.autocast()
        if bool(getattr(self, "_amp_enabled", False)):
            device_type = torch.device(self.device).type
            if device_type in {"cuda", "cpu"}:
                return torch.amp.autocast(
                    device_type=device_type,
                    dtype=getattr(self, "_amp_dtype", torch.float16),
                    enabled=True,
                )
        return nullcontext()

    def _backward_loss(self, loss: torch.Tensor) -> None:
        """Backward pass through either Accelerate, GradScaler, or plain PyTorch."""
        if self._accelerator is not None:
            self._accelerator.backward(loss)
            return
        scaler = getattr(self, "_grad_scaler", None)
        if scaler is not None and bool(getattr(scaler, "is_enabled", lambda: False)()):
            scaler.scale(loss).backward()
            return
        loss.backward()

    def _unscale_gradients_if_needed(self) -> None:
        scaler = getattr(self, "_grad_scaler", None)
        if scaler is not None and bool(getattr(scaler, "is_enabled", lambda: False)()):
            scaler.unscale_(self.optimizer)

    def _optimizer_step_with_scaler(self) -> None:
        scaler = getattr(self, "_grad_scaler", None)
        if scaler is not None and bool(getattr(scaler, "is_enabled", lambda: False)()):
            scaler.step(self.optimizer)
            scaler.update()
            return
        self.optimizer.step()

    def _decode_latent_for_logic(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode a predicted clean latent into tile logits for LogicNet.

        LogicNet pathfinding operates on walkability probabilities derived from
        tile space. Passing raw continuous VQ latents into that loss makes the
        clamp/softmax semantics arbitrary, so predicted-latent training routes
        through the frozen VQ-VAE codebook and decoder while preserving
        straight-through gradients to the diffusion denoiser.
        """
        if not hasattr(self.vqvae, "decode"):
            raise TypeError("Configured VQ-VAE does not expose decode(); cannot compute decoded logic loss.")
        decode_latent = latent
        quantize = getattr(self.vqvae, "quantize", None)
        if callable(quantize):
            quantized = quantize(latent)
            if isinstance(quantized, (tuple, list)) and quantized:
                decode_latent = quantized[0]
            elif isinstance(quantized, torch.Tensor):
                decode_latent = quantized
            if not isinstance(decode_latent, torch.Tensor) or decode_latent.shape != latent.shape:
                raise ValueError(
                    "VQ-VAE quantize() must return a quantized latent tensor matching "
                    f"{tuple(latent.shape)}, got {type(decode_latent).__name__}."
                )
        tile_logits = self.vqvae.decode(decode_latent, target_size=(ROOM_HEIGHT, ROOM_WIDTH))
        if not isinstance(tile_logits, torch.Tensor) or tile_logits.dim() != 4:
            raise ValueError(f"VQ-VAE decode must return [B,C,H,W] logits, got {type(tile_logits).__name__}.")
        if int(tile_logits.shape[1]) != int(self.config.num_classes):
            raise ValueError(
                f"Decoded VQ-VAE logits have {int(tile_logits.shape[1])} channels, "
                f"expected num_classes={int(self.config.num_classes)}."
            )
        return tile_logits

    def _tile_targets_from_maps(self, real_maps: torch.Tensor) -> torch.Tensor:
        """Convert training room tensors to integer tile labels [B,H,W]."""
        if real_maps.dim() == 4 and int(real_maps.shape[1]) == 1:
            targets = real_maps[:, 0]
        elif real_maps.dim() == 4 and int(real_maps.shape[1]) == int(self.config.num_classes):
            targets = real_maps.argmax(dim=1)
        elif real_maps.dim() == 3:
            targets = real_maps
        else:
            raise ValueError(f"Cannot derive tile targets from real_maps shape {tuple(real_maps.shape)}.")

        if targets.dtype.is_floating_point:
            max_value = float(targets.detach().max().item()) if targets.numel() > 0 else 0.0
            if max_value <= 1.0 and int(self.config.num_classes) > 1:
                targets = torch.round(targets * float(int(self.config.num_classes) - 1))
            else:
                targets = torch.round(targets)
        return targets.to(device=self.device, dtype=torch.long).clamp(0, int(self.config.num_classes) - 1)

    def _compute_hard_solvability(self, decoded_logits: torch.Tensor) -> float:
        """Symbolically check generated room grids for a start-to-goal/exit path."""
        if not isinstance(decoded_logits, torch.Tensor) or decoded_logits.dim() != 4:
            return 0.0
        try:
            from src.core.symbolic_refiner import PathAnalyzer
        except Exception as exc:
            logger.debug("Hard solvability skipped: PathAnalyzer unavailable: %s", exc)
            return 0.0

        analyzer = PathAnalyzer()
        tile_maps = decoded_logits.argmax(dim=1).detach().cpu()
        walkable_ids = set(int(v) for v in getattr(analyzer, "walkable_tiles", set()))
        start_id = int(SEMANTIC_PALETTE.get("START", 0))
        goal_ids = {
            int(SEMANTIC_PALETTE.get(name, -1))
            for name in ("TRIFORCE", "DOOR_OPEN", "DOOR_LOCKED", "DOOR_BOMB", "DOOR_PUZZLE", "DOOR_BOSS", "DOOR_SOFT", "STAIR")
        }
        solved = 0
        total = int(tile_maps.shape[0])
        for tile_map_t in tile_maps:
            grid = tile_map_t.numpy()
            start_hits = torch.nonzero(tile_map_t == start_id, as_tuple=False)
            if start_hits.numel() > 0:
                start = tuple(int(v) for v in start_hits[0].tolist())
            else:
                walkable_hits = torch.nonzero(
                    torch.isin(tile_map_t, torch.tensor(sorted(walkable_ids), dtype=tile_map_t.dtype)),
                    as_tuple=False,
                )
                if walkable_hits.numel() == 0:
                    continue
                start = tuple(int(v) for v in walkable_hits[0].tolist())

            goal_positions = []
            for goal_id in goal_ids:
                if goal_id < 0:
                    continue
                hits = torch.nonzero(tile_map_t == goal_id, as_tuple=False)
                goal_positions.extend(tuple(int(v) for v in hit.tolist()) for hit in hits)
            if not goal_positions:
                height, width = tile_map_t.shape
                border_mask = torch.zeros_like(tile_map_t, dtype=torch.bool)
                border_mask[0, :] = True
                border_mask[-1, :] = True
                border_mask[:, 0] = True
                border_mask[:, -1] = True
                walkable_tensor = torch.tensor(sorted(walkable_ids), dtype=tile_map_t.dtype)
                border_walkable = border_mask & torch.isin(tile_map_t, walkable_tensor)
                goal_positions = [
                    tuple(int(v) for v in hit.tolist())
                    for hit in torch.nonzero(border_walkable, as_tuple=False)
                ]
            if any(analyzer.analyze_grid(grid, start=start, goal=goal) == [] for goal in goal_positions):
                solved += 1
        return float(solved / max(1, total))

    def _infer_symbolic_repair_endpoints(
        self,
        tile_map_t: torch.Tensor,
    ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Infer start/goal coordinates for validation-time symbolic repair."""
        try:
            from src.core.symbolic_refiner import PathAnalyzer
        except Exception as exc:
            logger.debug("Validation repair endpoint inference skipped: PathAnalyzer unavailable: %s", exc)
            return None

        analyzer = PathAnalyzer()
        walkable_ids = set(int(v) for v in getattr(analyzer, "walkable_tiles", set()))
        start_id = int(SEMANTIC_PALETTE.get("START", 0))
        goal_ids = {
            int(SEMANTIC_PALETTE.get(name, -1))
            for name in ("TRIFORCE", "DOOR_OPEN", "DOOR_LOCKED", "DOOR_BOMB", "DOOR_PUZZLE", "DOOR_BOSS", "DOOR_SOFT", "STAIR")
        }

        start_hits = torch.nonzero(tile_map_t == start_id, as_tuple=False)
        if start_hits.numel() > 0:
            start = tuple(int(v) for v in start_hits[0].tolist())
        else:
            walkable_tensor = torch.tensor(sorted(walkable_ids), dtype=tile_map_t.dtype, device=tile_map_t.device)
            walkable_hits = torch.nonzero(torch.isin(tile_map_t, walkable_tensor), as_tuple=False)
            if walkable_hits.numel() == 0:
                return None
            start = tuple(int(v) for v in walkable_hits[0].tolist())

        goal_positions: List[Tuple[int, int]] = []
        for goal_id in goal_ids:
            if goal_id < 0:
                continue
            hits = torch.nonzero(tile_map_t == goal_id, as_tuple=False)
            goal_positions.extend(tuple(int(v) for v in hit.tolist()) for hit in hits)
        if not goal_positions:
            height, width = tile_map_t.shape
            border_mask = torch.zeros_like(tile_map_t, dtype=torch.bool)
            border_mask[0, :] = True
            border_mask[-1, :] = True
            border_mask[:, 0] = True
            border_mask[:, -1] = True
            walkable_tensor = torch.tensor(sorted(walkable_ids), dtype=tile_map_t.dtype, device=tile_map_t.device)
            border_walkable = border_mask & torch.isin(tile_map_t, walkable_tensor)
            goal_positions = [
                tuple(int(v) for v in hit.tolist())
                for hit in torch.nonzero(border_walkable, as_tuple=False)
            ]
        if not goal_positions:
            return None
        return start, goal_positions[0]

    def _validation_repairer(self):
        """Return or lazily construct a validation-time NeuralGuidedRepair wrapper."""
        repairer = getattr(self, "validation_neural_guided_repair", None)
        if repairer is not None:
            return repairer
        try:
            from src.core.neural_guided_repair import NeuralGuidedRepair
            from src.core.symbolic_refiner import SymbolicRefiner
        except Exception as exc:
            logger.debug("Validation repair skipped: repair modules unavailable: %s", exc)
            return None
        refiner = (
            getattr(self, "symbolic_refiner", None)
            or getattr(self, "refiner", None)
            or SymbolicRefiner(max_repair_attempts=int(getattr(self.config, "validation_repair_attempts", 2)))
        )
        repairer = NeuralGuidedRepair(
            self.logic_net,
            refiner,
            use_neural_feedback=False,
            use_logicnet_cost=bool(getattr(self.config, "validation_repair_use_logicnet_cost", True)),
            use_logicnet_floor_mask=bool(getattr(self.config, "validation_repair_use_logicnet_floor_mask", True)),
        )
        self.validation_neural_guided_repair = repairer
        return repairer

    def _repair_validation_decoded_logits(
        self,
        decoded_logits: torch.Tensor,
        graph_data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Optional[torch.Tensor], float]:
        """Apply the neural-symbolic repair path to decoded validation samples."""
        if not isinstance(decoded_logits, torch.Tensor) or decoded_logits.dim() != 4:
            return None, 0.0
        repairer = self._validation_repairer()
        if repairer is None:
            return None, 0.0

        tile_maps = decoded_logits.argmax(dim=1).detach().cpu()
        repaired_maps: List[torch.Tensor] = []
        successes = 0
        for sample_idx, tile_map_t in enumerate(tile_maps):
            endpoints = self._infer_symbolic_repair_endpoints(tile_map_t)
            if endpoints is None:
                repaired_maps.append(tile_map_t.to(dtype=torch.long))
                continue
            start, goal = endpoints
            grid = tile_map_t.numpy().astype(np.int64, copy=False)
            sample_graph_data = self._select_validation_graph_sample(
                graph_data,
                sample_idx=sample_idx,
                batch_size=int(tile_maps.shape[0]),
            )
            try:
                repaired_grid, success, _diag = repairer.repair_room_with_neural_guidance(
                    grid,
                    start=start,
                    goal=goal,
                    tile_logits=decoded_logits[sample_idx : sample_idx + 1],
                    graph_data=sample_graph_data,
                    max_feedback_rounds=0,
                )
                if isinstance(repaired_grid, np.ndarray) and repaired_grid.shape == grid.shape:
                    repaired_t = torch.as_tensor(repaired_grid, dtype=torch.long)
                    repaired_maps.append(repaired_t)
                    successes += int(bool(success))
                else:
                    repaired_maps.append(tile_map_t.to(dtype=torch.long))
            except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
                logger.debug("Validation neural-symbolic repair failed for sample %d: %s", sample_idx, exc)
                repaired_maps.append(tile_map_t.to(dtype=torch.long))

        if not repaired_maps:
            return None, 0.0
        repaired_ids = torch.stack(repaired_maps, dim=0).to(device=decoded_logits.device, dtype=torch.long)
        num_classes = int(getattr(self.config, "num_classes", decoded_logits.shape[1]))
        repaired_ids = repaired_ids.clamp(0, num_classes - 1)
        repaired_logits = F.one_hot(repaired_ids, num_classes=num_classes).permute(0, 3, 1, 2).to(
            device=decoded_logits.device,
            dtype=decoded_logits.dtype,
        )
        return repaired_logits, float(successes / max(1, int(repaired_ids.shape[0])))

    def _select_validation_graph_sample(
        self,
        graph_data: Optional[Dict[str, Any]],
        *,
        sample_idx: int,
        batch_size: int,
    ) -> Optional[Dict[str, Any]]:
        """Select one room graph from a stacked validation graph batch."""
        if not isinstance(graph_data, dict):
            return graph_data
        if graph_data.get("graph_scope") != "room_batch":
            return graph_data
        selected: Dict[str, Any] = {}
        for key, value in graph_data.items():
            if isinstance(value, torch.Tensor) and value.dim() > 0 and int(value.shape[0]) == int(batch_size):
                selected[key] = value[int(sample_idx)]
            else:
                selected[key] = value
        selected["graph_scope"] = "room"
        return selected
    
    def _encode_graph_conditioning(
        self,
        graph_dict: dict,
    ) -> torch.Tensor:
        """
        Encode a single graph dict into a conditioning vector using the GNN.
        
        Args:
            graph_dict: Dict from zelda_loader with:
                - node_features: [N, 6]
                - edge_index: [2, E]
                - edge_attr: [E] edge type labels
                
        Returns:
            If graph_conditioning_mode='pooled': [1, context_dim]
            If graph_conditioning_mode='node_sequence': [N+1, context_dim]
            with a canonical room-anchor token prepended.
        """
        node_features = graph_dict['node_features'].to(self.device)
        edge_index = graph_dict['edge_index'].to(self.device)
        
        edge_features = self._encode_edge_features(graph_dict)
        edge_rrwp = graph_dict.get("edge_rrwp")
        if not isinstance(edge_rrwp, torch.Tensor):
            edge_rrwp = compute_rrwp_edge_features(
                edge_index,
                int(node_features.shape[0]),
                steps=int(GRAPH_TPE_DIM),
                device=self.device,
                dtype=torch.float32,
            )
        else:
            edge_rrwp = edge_rrwp.to(self.device, dtype=torch.float32)
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
        graph_batch_idx = graph_dict.get("batch_idx")
        if isinstance(graph_batch_idx, torch.Tensor):
            graph_batch_idx = graph_batch_idx.to(self.device, dtype=torch.long)
        graph_node_mask = graph_dict.get("node_mask")
        if isinstance(graph_node_mask, torch.Tensor):
            graph_node_mask = graph_node_mask.to(self.device, dtype=torch.float32)
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
            if bool(getattr(self.config, "use_teacher_forced_neighbor_latents", True)):
                neighbor_latents = self._encode_neighbor_maps_to_latents(graph_dict.get("neighbor_maps"))
            else:
                neighbor_latents = {direction: None for direction in CARDINAL_DIRECTIONS}
            reference_room_maps = (
                graph_dict.get("neighbor_maps")
                if bool(getattr(self.config, "condition_use_reference_room_maps", False))
                else None
            )
            condition_kwargs = {
                "neighbor_latents": neighbor_latents,
                "boundary_constraints": boundary_constraints,
                "position": room_position,
                "node_features": node_features,
                "edge_index": edge_index,
                "edge_features": edge_features,
                "tpe": tpe,
                "current_node_distance": current_node_distance,
                "batch_idx": graph_batch_idx,
                "node_mask": graph_node_mask,
                "current_node_idx": int(current_node_idx) if current_node_idx is not None else None,
                "reference_room_maps": reference_room_maps,
                "style_id": style_id,
                "return_global_tokens": self.config.graph_conditioning_mode == "node_sequence",
            }
            if self._call_supports_keyword(self.condition_encoder, "edge_rrwp"):
                condition_kwargs["edge_rrwp"] = edge_rrwp
            condition_out = self.condition_encoder(**condition_kwargs)
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

        global_kwargs = {
            "edge_features": edge_features,
            "tpe": tpe,
            "current_node_distance": current_node_distance,
            "batch_idx": graph_batch_idx,
            "node_mask": graph_node_mask,
        }
        encode_global = self.condition_encoder.encode_global_only
        if self._call_supports_keyword(encode_global, "edge_rrwp"):
            global_kwargs["edge_rrwp"] = edge_rrwp
        c_global = encode_global(node_features, edge_index, **global_kwargs)

        if self.config.graph_conditioning_mode == "node_sequence":
            default_anchor = self.condition_encoder.encode_local_only(
                neighbor_latents={direction: None for direction in CARDINAL_DIRECTIONS},
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

        # Pooled baseline.
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

    def _normalize_diffusion_graph_sample(self, graph_dict: dict) -> Dict[str, torch.Tensor]:
        """Prepare one graph sample for diffusion spatial/topological conditioning."""
        node_features = graph_dict["node_features"]
        edge_index = graph_dict["edge_index"]
        if not isinstance(node_features, torch.Tensor):
            node_features = torch.tensor(node_features, dtype=torch.float32)
        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.tensor(edge_index, dtype=torch.long)
        node_features = node_features.to(self.device, dtype=torch.float32)
        edge_index = edge_index.to(self.device, dtype=torch.long)
        if node_features.dim() != 2:
            raise ValueError(f"node_features must have shape [N, F], got {tuple(node_features.shape)}")
        if edge_index.dim() != 2 or int(edge_index.shape[0]) != 2:
            raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")

        num_nodes = int(node_features.shape[0])
        num_edges = int(edge_index.shape[1])
        current_node_idx = graph_dict.get("current_node_idx")
        if isinstance(current_node_idx, torch.Tensor):
            current_node_idx = int(current_node_idx.detach().flatten()[0].item()) if current_node_idx.numel() else 0
        elif current_node_idx is not None:
            current_node_idx = int(current_node_idx)

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

        edge_features = self._encode_edge_features(graph_dict)
        edge_feature_dim = int(max(1, getattr(self.config, "edge_feature_dim", GRAPH_EDGE_FEATURE_DIM)))
        if not isinstance(edge_features, torch.Tensor):
            edge_features = torch.zeros(num_edges, edge_feature_dim, device=self.device, dtype=torch.float32)
        else:
            if edge_features.dim() == 1:
                edge_features = edge_features.unsqueeze(-1)
            if edge_features.dim() != 2:
                raise ValueError(f"edge_features must have shape [E,F], got {tuple(edge_features.shape)}")
            aligned = torch.zeros(num_edges, max(edge_feature_dim, int(edge_features.shape[1])), device=self.device, dtype=torch.float32)
            rows = min(num_edges, int(edge_features.shape[0]))
            cols = min(int(aligned.shape[1]), int(edge_features.shape[1]))
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

        edge_rrwp = graph_dict.get("edge_rrwp")
        if isinstance(edge_rrwp, torch.Tensor):
            edge_rrwp = edge_rrwp.to(self.device, dtype=torch.float32)
            if edge_rrwp.dim() == 1:
                edge_rrwp = edge_rrwp.unsqueeze(-1)
            if edge_rrwp.dim() != 2:
                raise ValueError(f"edge_rrwp must have shape [E,F], got {tuple(edge_rrwp.shape)}")
            aligned_rrwp = torch.zeros(num_edges, int(GRAPH_TPE_DIM), device=self.device, dtype=torch.float32)
            rows = min(num_edges, int(edge_rrwp.shape[0]))
            cols = min(int(GRAPH_TPE_DIM), int(edge_rrwp.shape[1]))
            if rows > 0 and cols > 0:
                aligned_rrwp[:rows, :cols] = edge_rrwp[:rows, :cols]
            edge_rrwp = aligned_rrwp
        else:
            edge_rrwp = compute_rrwp_edge_features(
                edge_index,
                num_nodes,
                steps=int(GRAPH_TPE_DIM),
                device=self.device,
                dtype=torch.float32,
            )

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
            if int(node_mask.shape[0]) != 1:
                raise ValueError(f"Unbatched diffusion sample cannot provide multi-row node_mask: {tuple(node_mask.shape)}")
            node_mask = node_mask.squeeze(0)
        if node_mask.dim() != 1 or int(node_mask.shape[0]) != num_nodes:
            raise ValueError(
                f"node_mask must have shape [N] matching node_features; got {tuple(node_mask.shape)} for N={num_nodes}."
            )

        room_topology_map = graph_dict.get("room_topology_map")
        if isinstance(room_topology_map, torch.Tensor):
            room_topology_map = room_topology_map.to(self.device, dtype=torch.float32)
            if room_topology_map.dim() == 4:
                if int(room_topology_map.shape[0]) != 1:
                    raise ValueError(
                        f"Single graph sample room_topology_map must be [C,H,W] or [1,C,H,W], got {tuple(room_topology_map.shape)}."
                    )
                room_topology_map = room_topology_map.squeeze(0)
            if room_topology_map.dim() != 3:
                raise ValueError(
                    f"room_topology_map must have shape [C,H,W] for one sample, got {tuple(room_topology_map.shape)}."
                )

        boundary_constraints = graph_dict.get("boundary_constraints")
        if isinstance(boundary_constraints, torch.Tensor):
            boundary_constraints = boundary_constraints.to(self.device, dtype=torch.float32)
            if boundary_constraints.dim() == 2:
                if int(boundary_constraints.shape[0]) != 1:
                    raise ValueError(
                        "Single graph sample boundary_constraints must be [8] or [1,8], "
                        f"got {tuple(boundary_constraints.shape)}."
                    )
                boundary_constraints = boundary_constraints.squeeze(0)
            if boundary_constraints.dim() != 1 or int(boundary_constraints.shape[0]) != 8:
                raise ValueError(
                    f"boundary_constraints must have shape [8] for one sample, got {tuple(boundary_constraints.shape)}."
                )

        has_room_anchor = bool(graph_dict.get("has_room_anchor", False)) or (
            isinstance(graph_dict.get("boundary_constraints"), torch.Tensor)
            and isinstance(graph_dict.get("room_position"), torch.Tensor)
        )

        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_features": edge_features,
            "edge_attr": edge_attr,
            "edge_rrwp": edge_rrwp,
            "tpe": tpe,
            "current_node_distance": current_node_distance,
            "node_positions": node_positions,
            "node_mask": node_mask,
            "current_node_idx": int(current_node_idx) if current_node_idx is not None else 0,
            "start_node_id": int(start_node_id),
            "target_idx": int(target_idx),
            "has_room_anchor": has_room_anchor,
            **({"boundary_constraints": boundary_constraints} if isinstance(boundary_constraints, torch.Tensor) else {}),
            **({"room_topology_map": room_topology_map} if isinstance(room_topology_map, torch.Tensor) else {}),
        }

    def _stack_diffusion_graph_batch(self, graph_list: List[dict]) -> Optional[Dict[str, torch.Tensor]]:
        """Pad a batch of variable-size graph tensors for diffusion conditioning."""
        if not graph_list:
            return None

        dungeon_graph = self._try_stack_dungeon_scope_graph_batch(graph_list)
        if dungeon_graph is not None:
            return dungeon_graph

        samples = [self._normalize_diffusion_graph_sample(graph_dict) for graph_dict in graph_list]
        if not samples:
            return None

        anchor_flags = {bool(sample.get("has_room_anchor", False)) for sample in samples}

        max_nodes = max(int(sample["node_features"].shape[0]) for sample in samples)
        feat_dim = max(int(sample["node_features"].shape[1]) if sample["node_features"].dim() == 2 else 0 for sample in samples)
        tpe_dim = max(int(sample["tpe"].shape[1]) if sample["tpe"].dim() == 2 else 0 for sample in samples)
        distance_dim = max(
            int(sample["current_node_distance"].shape[1]) if sample["current_node_distance"].dim() == 2 else 0
            for sample in samples
        )
        pos_dim = max(int(sample["node_positions"].shape[1]) if sample["node_positions"].dim() == 2 else 0 for sample in samples)
        max_edges = max(int(sample["edge_index"].shape[1]) if sample["edge_index"].dim() == 2 else 0 for sample in samples)
        edge_feat_dim = max(
            int(sample["edge_features"].shape[1]) if sample["edge_features"].dim() == 2 else 0
            for sample in samples
        )
        edge_rrwp_dim = max(
            int(sample["edge_rrwp"].shape[1]) if sample["edge_rrwp"].dim() == 2 else 0
            for sample in samples
        )

        node_features_batch = torch.zeros(len(samples), max_nodes, max(1, feat_dim), device=self.device, dtype=torch.float32)
        tpe_batch = torch.zeros(len(samples), max_nodes, max(1, tpe_dim), device=self.device, dtype=torch.float32)
        current_node_distance_batch = torch.zeros(len(samples), max_nodes, max(1, distance_dim), device=self.device, dtype=torch.float32)
        node_positions_batch = torch.zeros(len(samples), max_nodes, max(1, pos_dim), device=self.device, dtype=torch.float32)
        node_mask_batch = torch.zeros(len(samples), max_nodes, device=self.device, dtype=torch.float32)
        edge_index_batch = torch.full((len(samples), 2, max_edges), -1, device=self.device, dtype=torch.long)
        edge_features_batch = torch.zeros(len(samples), max_edges, max(1, edge_feat_dim), device=self.device, dtype=torch.float32)
        edge_rrwp_batch = torch.zeros(len(samples), max_edges, max(1, edge_rrwp_dim), device=self.device, dtype=torch.float32)
        edge_attr_batch = torch.full((len(samples), max_edges), -1, device=self.device, dtype=torch.long)
        current_node_idx_batch = torch.zeros(len(samples), device=self.device, dtype=torch.long)
        start_node_id_batch = torch.full((len(samples),), -1, device=self.device, dtype=torch.long)
        target_idx_batch = torch.full((len(samples),), -1, device=self.device, dtype=torch.long)

        topo_maps = []
        has_topology = [("room_topology_map" in sample) for sample in samples]
        if any(has_topology) and not all(has_topology):
            raise ValueError(
                "room_topology_map must be present for every graph in a diffusion batch or omitted for all of them."
            )
        can_stack_topology = all(has_topology)
        topo_shape = None
        has_boundary = [("boundary_constraints" in sample) for sample in samples]
        if any(has_boundary) and not all(has_boundary):
            raise ValueError(
                "boundary_constraints must be present for every graph in a diffusion batch or omitted for all of them."
            )
        boundary_batch = (
            torch.zeros(len(samples), 8, device=self.device, dtype=torch.float32)
            if all(has_boundary)
            else None
        )

        for i, sample in enumerate(samples):
            num_nodes = int(sample["node_features"].shape[0])
            if num_nodes > 0:
                node_features_batch[i, :num_nodes, : sample["node_features"].shape[1]] = sample["node_features"]
                tpe_batch[i, :num_nodes, : sample["tpe"].shape[1]] = sample["tpe"]
                current_node_distance_batch[i, :num_nodes, : sample["current_node_distance"].shape[1]] = sample["current_node_distance"]
                node_positions_batch[i, :num_nodes, : sample["node_positions"].shape[1]] = sample["node_positions"]
                node_mask_batch[i, :num_nodes] = sample["node_mask"]

            num_edges = int(sample["edge_index"].shape[1]) if sample["edge_index"].dim() == 2 else 0
            if num_edges > 0:
                edge_index_batch[i, :, :num_edges] = sample["edge_index"]
                edge_features_batch[i, :num_edges, : sample["edge_features"].shape[1]] = sample["edge_features"]
                edge_rrwp_batch[i, :num_edges, : sample["edge_rrwp"].shape[1]] = sample["edge_rrwp"]
                edge_attr_batch[i, :num_edges] = sample["edge_attr"]

            current_node_idx_batch[i] = int(sample.get("current_node_idx", 0))
            start_node_id_batch[i] = int(sample.get("start_node_id", -1))
            target_idx_batch[i] = int(sample.get("target_idx", -1))

            if can_stack_topology:
                topo = sample["room_topology_map"]
                if topo.dim() == 3:
                    topo = topo.unsqueeze(0)
                current_shape = tuple(topo.shape[1:])
                if topo_shape is None:
                    topo_shape = current_shape
                if current_shape != topo_shape:
                    if not bool(getattr(self, "_topology_shape_mismatch_warning_emitted", False)):
                        logger.warning(
                            "Disabling batched room_topology_map stacking due to shape mismatch: "
                            "expected %s, got %s. Topology conditioning will be omitted for this batch.",
                            str(topo_shape),
                            str(current_shape),
                        )
                        self._topology_shape_mismatch_warning_emitted = True
                    can_stack_topology = False
                    topo_maps = []
                else:
                    topo_maps.append(topo)
            if boundary_batch is not None:
                boundary_batch[i] = sample["boundary_constraints"]

        batch_graph = {
            "node_features": node_features_batch,
            "edge_index": edge_index_batch,
            "edge_features": edge_features_batch,
            "edge_rrwp": edge_rrwp_batch,
            "edge_attr": edge_attr_batch,
            "tpe": tpe_batch,
            "current_node_distance": current_node_distance_batch,
            "node_positions": node_positions_batch,
            "node_mask": node_mask_batch,
            "current_node_idx": current_node_idx_batch,
            "start_node_id": start_node_id_batch,
            "target_idx": target_idx_batch,
            "graph_scope": "room_batch",
            "has_room_anchor": bool(self.config.graph_conditioning_mode == "node_sequence") or (
                bool(next(iter(anchor_flags))) if anchor_flags else False
            ),
        }
        if can_stack_topology and topo_maps:
            batch_graph["room_topology_map"] = torch.cat(topo_maps, dim=0)
        if boundary_batch is not None:
            batch_graph["boundary_constraints"] = boundary_batch
        return batch_graph

    def _try_stack_dungeon_scope_graph_batch(self, graph_list: List[dict]) -> Optional[Dict[str, torch.Tensor]]:
        """
        Collapse a room batch from one dungeon into one mission graph.

        Returns None when the batch is mixed or incomplete, falling back to the
        legacy per-room graph list path.
        """
        if not graph_list:
            return None
        node_counts = [int(g.get("num_nodes", 0)) for g in graph_list]
        if not node_counts or min(node_counts) <= 0 or len(set(node_counts)) != 1:
            return None
        num_nodes = node_counts[0]
        if len(graph_list) != num_nodes:
            return None

        first = graph_list[0]
        first_node_map = dict(first.get("node_to_idx", {}))
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

        sample = self._normalize_diffusion_graph_sample(first)
        topo_maps: List[torch.Tensor] = []
        boundary_rows: List[torch.Tensor] = []
        for graph in graph_list:
            normalized = self._normalize_diffusion_graph_sample(graph)
            topo = normalized.get("room_topology_map")
            boundary = normalized.get("boundary_constraints")
            if isinstance(topo, torch.Tensor):
                topo_maps.append(topo.unsqueeze(0) if topo.dim() == 3 else topo)
            if isinstance(boundary, torch.Tensor):
                boundary_rows.append(boundary.reshape(1, -1))

        node_mask = sample.get("node_mask")
        if not isinstance(node_mask, torch.Tensor):
            node_mask = torch.ones(num_nodes, device=self.device, dtype=torch.float32)
        batch_graph = {
            "node_features": sample["node_features"],
            "edge_index": sample["edge_index"],
            "edge_features": sample["edge_features"],
            "edge_rrwp": sample["edge_rrwp"],
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
        if topo_maps and len(topo_maps) == len(graph_list):
            batch_graph["room_topology_map"] = torch.cat(topo_maps, dim=0)
        if boundary_rows and len(boundary_rows) == len(graph_list):
            batch_graph["boundary_constraints"] = torch.cat(boundary_rows, dim=0)
        return batch_graph
    
    @staticmethod
    def _logic_loss_to_solvability_proxy(logic_loss: torch.Tensor) -> torch.Tensor:
        """
        Convert unbounded non-negative LogicNet loss into a bounded proxy score.

        We use exp(-loss) so the proxy stays in (0, 1], decreases smoothly as
        constraints are violated, and never becomes negative.
        """
        if not isinstance(logic_loss, torch.Tensor):
            logic_loss = torch.tensor(float(logic_loss), dtype=torch.float32)
        return torch.exp(-logic_loss.detach().clamp_min(0.0))

    def _tile_grid_from_maps(self, maps: torch.Tensor) -> torch.Tensor:
        """Convert normalized room maps or logits/probs to integer tile IDs."""
        if maps.dim() != 4:
            raise ValueError(f"maps must be [B,C,H,W], got {tuple(maps.shape)}")
        if int(maps.shape[1]) == 1:
            scale = float(max(1, int(getattr(self.config, "num_classes", 44)) - 1))
            return (maps[:, 0].detach() * scale).round().long().clamp(0, int(scale))
        return maps.detach().argmax(dim=1).long()

    def _build_wfc_tile_priors(self, real_maps: torch.Tensor) -> Dict[int, TilePrior]:
        """Build lightweight tile-frequency priors from the current real batch."""
        tile_ids = self._tile_grid_from_maps(real_maps).detach().cpu().numpy().astype(np.int64)
        values, counts = np.unique(tile_ids, return_counts=True)
        total = float(max(1, int(counts.sum())))
        return {
            int(tile_id): TilePrior(tile_id=int(tile_id), frequency=float(count) / total)
            for tile_id, count in zip(values.tolist(), counts.tolist())
        }

    def _wfc_pseudo_label_loss(
        self,
        pred_tile_logits: torch.Tensor,
        real_maps: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, torch.Tensor]:
        """
        Distill WFC-repaired pseudo targets into predicted tile logits.

        WFC is non-differentiable here; gradients flow from cross-entropy on
        current logits to repaired tile labels. The loss is opt-in through
        alpha_wfc_pseudo.
        """
        zero = torch.zeros((), device=pred_tile_logits.device, dtype=pred_tile_logits.dtype)
        if float(getattr(self.config, "alpha_wfc_pseudo", 0.0)) <= 0.0:
            return zero, 0.0, zero
        max_samples = int(getattr(self.config, "wfc_pseudo_max_samples", 0))
        if max_samples <= 0 or pred_tile_logits.numel() == 0:
            return zero, 0.0, zero

        priors = self._build_wfc_tile_priors(real_maps)
        if not priors:
            return zero, 0.0, zero

        with torch.no_grad():
            probs = F.softmax(pred_tile_logits.detach(), dim=1)
            confidence, pred_ids = probs.max(dim=1)
            threshold = float(getattr(self.config, "wfc_pseudo_confidence_threshold", 0.75))
            repair_targets: List[torch.Tensor] = []
            limit = min(int(pred_tile_logits.shape[0]), max_samples)
            wfc_config = WeightedBayesianWFCConfig(
                use_vqvae_priors=True,
                max_iterations=max(128, ROOM_HEIGHT * ROOM_WIDTH * 4),
                max_backtracks=32,
                max_restarts=1,
            )
            for sample_idx in range(limit):
                seed_grid = pred_ids[sample_idx].detach().cpu().numpy().astype(np.int64)
                confident = confidence[sample_idx].detach().cpu().numpy() >= threshold
                partial_seed = np.where(confident, seed_grid, -1).astype(np.int64)
                try:
                    repaired = integrate_weighted_wfc_into_pipeline(
                        partial_seed,
                        priors,
                        seed=int(getattr(self.config, "seed", 42)) + int(self.global_step) + sample_idx,
                        config=wfc_config,
                    )["grid"]
                except (RuntimeError, ValueError, TypeError, KeyError):
                    continue
                target = torch.as_tensor(
                    repaired,
                    device=pred_tile_logits.device,
                    dtype=torch.long,
                ).clamp(0, int(getattr(self.config, "num_classes", 44)) - 1)
                repair_targets.append(target)

        if not repair_targets:
            return zero, 0.0, zero
        target_batch = torch.stack(repair_targets, dim=0)
        logits = pred_tile_logits[: target_batch.shape[0]]
        repaired_mean = F.cross_entropy(logits, target_batch, reduction="mean")
        full_batch_loss = F.cross_entropy(logits, target_batch, reduction="sum") / float(
            max(1, int(pred_tile_logits.shape[0]) * int(pred_tile_logits.shape[2]) * int(pred_tile_logits.shape[3]))
        )
        return full_batch_loss, float(target_batch.shape[0]), repaired_mean
    
    def _update_ema(self):
        """
        Update EMA model weights (Phase 4A).
        
        EMA provides more stable weights for sampling/validation.
        Standard in all modern diffusion training (Nichol & Dhariwal, 2021).
        """
        with torch.no_grad():
            for p_ema, p in zip(self.ema_diffusion.parameters(),
                                self.diffusion.parameters()):
                p_ema.data.mul_(self.ema_decay).add_(p.data, alpha=1.0 - self.ema_decay)
        average_module_parameters(
            self.ema_diffusion,
            context=getattr(self, "distributed_context", None),
        )

    @staticmethod
    def _tensor_is_finite(value: Any) -> bool:
        """Return True when a tensor/scalar payload contains only finite values."""
        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all())
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _state_dict_is_finite(state_dict: Any) -> bool:
        """Recursively validate that a state dict does not contain NaN/Inf tensors."""
        if isinstance(state_dict, dict):
            for value in state_dict.values():
                if not DiffusionTrainer._state_dict_is_finite(value):
                    return False
            return True
        if isinstance(state_dict, (list, tuple)):
            return all(DiffusionTrainer._state_dict_is_finite(value) for value in state_dict)
        if isinstance(state_dict, torch.Tensor):
            return bool(torch.isfinite(state_dict).all())
        return True

    @staticmethod
    def _strip_embedded_guidance_logic_net_state(state_dict: Any) -> Tuple[Any, int]:
        """Drop legacy LogicNet weights that were nested inside diffusion guidance."""
        if not isinstance(state_dict, dict):
            return state_dict, 0

        prefix = "guidance.logic_net."
        kept_items = []
        removed = 0
        for key, value in state_dict.items():
            if str(key).startswith(prefix):
                removed += 1
                continue
            kept_items.append((key, value))

        if removed == 0:
            return state_dict, 0
        return dict(kept_items), removed

    def _warn_nonfinite(self, key: str, message: str, *args: Any) -> None:
        """Rate-limit repeated non-finite warnings so logs stay readable."""
        counts = getattr(self, "_nonfinite_warning_counts", None)
        if not isinstance(counts, dict):
            counts = {}
            self._nonfinite_warning_counts = counts
        count = int(counts.get(key, 0)) + 1
        counts[key] = count
        if count <= 5 or count in {10, 20, 50} or count % 100 == 0:
            logger.warning(message, *args)
            if count == 5:
                logger.warning(
                    "Further `%s` non-finite warnings will be rate-limited.",
                    key,
                )

    def _gradients_are_finite(self) -> bool:
        """Check trainable diffusion/condition-encoder gradients before stepping."""
        modules = [self.diffusion, self.condition_encoder]
        if bool(getattr(self.config, "logic_net_trainable", True)):
            modules.append(self.logic_net)
        for module in modules:
            for param in module.parameters():
                if param.grad is None:
                    continue
                if not bool(torch.isfinite(param.grad).all()):
                    return False
        return True

    def _default_estimated_total_steps(self) -> int:
        """Fallback schedule length used before a dataloader length is known."""
        total_epochs = int(max(1, int(getattr(self.config, "epochs", 1))))
        fallback_steps_per_epoch = int(max(1, int(getattr(self.config, "estimated_steps_per_epoch", 100))))
        fallback_steps_per_epoch = int(
            max(1, math.ceil(fallback_steps_per_epoch / float(self._gradient_accumulation_steps())))
        )
        return int(max(1, total_epochs * fallback_steps_per_epoch))

    def _warmup_scale(self, warmup_epochs: int, completed_steps: int) -> float:
        if warmup_epochs <= 0:
            return 1.0
        total_steps = max(1, int(getattr(self, "_estimated_total_steps", 1)))
        total_epochs = max(1, int(getattr(self.config, "epochs", 1)))
        warmup_steps = max(1, int(total_steps * min(1.0, warmup_epochs / total_epochs)))
        return min(1.0, max(1, int(completed_steps) + 1) / float(warmup_steps))

    def _apply_lr_warmup(self, *, completed_steps: Optional[int] = None) -> None:
        """Set optimizer-group learning rates for the next optimizer step."""
        step_index = int(self.global_step if completed_steps is None else completed_steps)
        global_scale = self._warmup_scale(
            int(max(0, getattr(self.config, "global_lr_warmup_epochs", 0))),
            step_index,
        )
        logic_scale = self._warmup_scale(
            int(max(0, getattr(self.config, "logic_lr_warmup_epochs", 0))),
            step_index,
        ) if bool(getattr(self.config, "logic_net_trainable", True)) else 1.0
        for group in self.optimizer.param_groups:
            base_lr = float(group.get("base_lr", group.get("lr", self.config.learning_rate)))
            scale = global_scale
            if group.get("name") == "logic_net":
                scale = min(scale, logic_scale)
            group["lr"] = base_lr * scale

    def _effective_logic_loss_weight(self, include_logic_loss: bool) -> float:
        """Linearly ramp alpha_logic after the warmup boundary."""
        if not bool(include_logic_loss):
            return 0.0
        base = float(max(0.0, getattr(self.config, "alpha_logic", 0.0)))
        if base <= 0.0:
            return 0.0
        warmup = int(max(0, getattr(self.config, "warmup_epochs", 0)))
        ramp_epochs = int(max(1, getattr(self.config, "logic_loss_ramp_epochs", 1)))
        epoch = int(getattr(self, "epoch", 0))
        if epoch < warmup:
            return 0.0
        ramp_step = epoch - warmup + 1
        return base * min(1.0, float(ramp_step) / float(ramp_epochs))

    def _vqvae_codebook_stats(self) -> Dict[str, float]:
        """Return frozen VQ/FSQ code usage metrics for diffusion logs."""
        stats: Dict[str, float] = {}

        def add_usage(prefix: str, usage: torch.Tensor) -> None:
            values = usage.detach().float().view(-1)
            total_codes = int(values.numel())
            if total_codes <= 0:
                return
            active = values > 0
            probs = values
            total = float(probs.sum().item())
            if total > 0.0:
                probs = probs / max(total, 1e-12)
                perplexity = torch.exp(
                    -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum()
                )
                max_usage = float(probs.max().item())
            else:
                perplexity = torch.zeros((), device=values.device, dtype=values.dtype)
                max_usage = 0.0
            stats[f"{prefix}_active_codes"] = float(active.sum().item())
            stats[f"{prefix}_total_codes"] = float(total_codes)
            stats[f"{prefix}_active_fraction"] = float(active.float().mean().item())
            stats[f"{prefix}_perplexity"] = float(perplexity.item())
            stats[f"{prefix}_max_usage"] = max_usage

        with torch.no_grad():
            vqvae = getattr(self, "vqvae", None)
            if vqvae is None:
                return stats
            hierarchical = getattr(vqvae, "get_hierarchical_codebook_usage", None)
            if callable(hierarchical):
                try:
                    payload = hierarchical()
                except RuntimeError:
                    payload = None
                if isinstance(payload, dict):
                    for name, usage in payload.items():
                        if isinstance(usage, torch.Tensor):
                            add_usage(f"vqvae_codebook_{name}", usage)
                    return stats
            getter = getattr(vqvae, "get_codebook_usage", None)
            if callable(getter):
                try:
                    usage = getter()
                except RuntimeError:
                    usage = None
                if isinstance(usage, torch.Tensor):
                    add_usage("vqvae_codebook", usage)
        return stats

    def _gradient_accumulation_steps(self) -> int:
        return int(max(1, int(getattr(self.config, "gradient_accumulation_steps", 1))))

    def _accumulated_micro_steps(self) -> int:
        return int(max(0, int(getattr(self, "_accumulation_micro_steps", 0))))

    def _reset_gradient_accumulation(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        self._accumulation_micro_steps = 0

    def _scale_accumulated_gradients(self, divisor: int) -> None:
        scale = float(max(1, int(divisor)))
        modules = [self.diffusion, self.condition_encoder]
        if bool(getattr(self.config, "logic_net_trainable", True)):
            modules.append(self.logic_net)
        seen: Set[int] = set()
        for module in modules:
            for param in module.parameters():
                if param.grad is None:
                    continue
                param_id = id(param)
                if param_id in seen:
                    continue
                seen.add(param_id)
                param.grad.detach().div_(scale)
    
    def train_step(
        self,
        real_maps: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
        include_logic_loss: bool = True,
        logic_graph_data: Optional[dict] = None,
        diffusion_graph_data: Optional[dict] = None,
        force_optimizer_step: bool = False,
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Training strategy:
        1. Diffusion loss: standard Îµ-prediction on real encoded latents
        2. LogicNet loss: computed on REAL z_0 with graph_data from .dot files,
           enabling both grid-level AND graph-level pathfinding/key-lock checking.
        3. GradientGuidance (wired in __init__): applies âˆ‡_{x_t}L_logic
           during sampling/validation to steer generation toward solvable maps.
        
        Args:
            real_maps: [B, 1, H, W] real dungeon maps
            conditioning: [B, context_dim] conditioning vectors from real graphs
            include_logic_loss: Whether to train LogicNet on real data
            logic_graph_data: Batched graph data dict for LogicNet.
            
        Returns:
            Dict of loss values
        """
        self.diffusion.train()
        self.condition_encoder.train()
        
        batch_size = real_maps.shape[0]
        
        # Get conditioning (use fallback if not provided)
        if conditioning is None:
            conditioning = self.get_dummy_conditioning(batch_size)
        
        # Encode to latent space
        z_0 = self.encode_to_latent(real_maps)

        graph_alignment_weight = float(getattr(self.config, "graph_spatial_alignment_weight", 0.0))
        if (
            graph_alignment_weight > 0.0
            and isinstance(diffusion_graph_data, dict)
            and "spatial_alignment_node_indices" in diffusion_graph_data
            and "spatial_alignment_positions" in diffusion_graph_data
        ):
            diffusion_graph_data = dict(diffusion_graph_data)
            diffusion_graph_data["spatial_alignment_weight"] = graph_alignment_weight
        
        # === Part 1: Diffusion / flow objective ===
        with self._autocast_context():
            diffusion_loss = self._diffusion_objective_loss(z_0, conditioning, diffusion_graph_data)
        if not self._tensor_is_finite(diffusion_loss):
            self._reset_gradient_accumulation()
            self._warn_nonfinite(
                "diffusion_loss",
                "Diffusion training: non-finite diffusion loss detected; skipping optimizer step for this batch.",
            )
            metrics = {
                'loss': 0.0,
                'diffusion_loss': 0.0,
                'logic_loss': 0.0,
                'logic_tile_loss': 0.0,
                'wfc_pseudo_loss': 0.0,
                'wfc_pseudo_loss_sum': 0.0,
                'wfc_pseudo_loss_contribution': 0.0,
                'wfc_pseudo_samples': 0.0,
                'logic_tile_accuracy': 0.0,
                'solvability_proxy': 0.0,
                'solvability': 0.0,
                'logic_loss_mode_predicted': 1.0 if self.config.logic_loss_mode == 'predicted_latent' else 0.0,
                'skipped_nonfinite_batch': 1.0,
            }
            metrics.update(self._vqvae_codebook_stats())
            return metrics
        
        # === Part 2: LogicNet loss on model-predicted latent WITH graph topology ===
        # IMPORTANT: computing logic loss on detached real z_0 does not train diffusion.
        # We instead denoise a noisy latent and apply LogicNet to predicted x0 so
        # logic gradients flow into diffusion + condition encoder.
        logic_loss = torch.tensor(0.0, device=self.device)
        logic_tile_loss = torch.tensor(0.0, device=self.device)
        wfc_pseudo_loss = torch.tensor(0.0, device=self.device)
        wfc_pseudo_mean_loss = torch.tensor(0.0, device=self.device)
        wfc_pseudo_samples = 0.0
        logic_tile_accuracy = torch.tensor(0.0, device=self.device)
        solvability_proxy = torch.tensor(0.0, device=self.device)
        logic_info: Dict[str, Any] = {}
        
        logic_enabled = bool(getattr(self.config, "logic_net_enabled", True))
        logic_loss_weight = self._effective_logic_loss_weight(include_logic_loss)
        if logic_enabled and include_logic_loss and float(getattr(self.config, "alpha_logic_tile", 0.0)) > 0.0:
            tile_targets = self._tile_targets_from_maps(real_maps)
            # Trains LogicNet.tile_classifier only; z_0.detach() prevents this
            # auxiliary CE branch from injecting gradients into diffusion.
            with self._autocast_context():
                tile_logits = self.logic_net.tile_classifier(z_0.detach())
                tile_logits = self.logic_net._project_tile_logits_to_room(tile_logits)
                logic_tile_loss = F.cross_entropy(tile_logits, tile_targets)
            logic_tile_accuracy = (tile_logits.argmax(dim=1) == tile_targets).float().mean()

        if logic_enabled and logic_loss_weight > 0:
            if self.config.logic_loss_mode == "detached_real":
                # Legacy baseline: logic regularization on real latent only.
                z_for_logic = z_0.detach().requires_grad_(True)
                logic_loss, logic_info = self.logic_net(z_for_logic, graph_data=logic_graph_data)
            else:
                # New default: logic supervision on predicted latent (trains diffusion).
                t_logic = torch.randint(0, self.diffusion.num_timesteps, (batch_size,), device=self.device)
                noise_logic = torch.randn_like(z_0)
                x_t_logic = self.diffusion.q_sample(z_0, t_logic, noise_logic)

                # Predict noise/velocity and convert to predicted clean latent x0.
                topology_tensors = self.diffusion._extract_context_topology(
                    conditioning,
                    diffusion_graph_data,
                )
                if len(topology_tensors) == 2:
                    context_edge_index, context_node_mask = topology_tensors
                    context_edge_attr = None
                else:
                    context_edge_index, context_edge_attr, context_node_mask = topology_tensors
                spatial_graph_data = self.diffusion._extract_spatial_graph_context(
                    conditioning,
                    diffusion_graph_data,
                )
                with self._autocast_context():
                    pred_logic = self.diffusion.denoiser(
                        x_t_logic,
                        t_logic,
                        conditioning,
                        context_edge_index=context_edge_index,
                        context_edge_attr=context_edge_attr,
                        context_node_mask=context_node_mask,
                        spatial_graph_data=spatial_graph_data,
                    )
                    pred_x0_logic = self._prediction_to_x0(pred_logic, x_t_logic, t_logic)

                    # Keep latent range bounded similarly to sampling path.
                    pred_x0_logic = torch.clamp(pred_x0_logic, -1.0, 1.0)
                    if not self._tensor_is_finite(pred_x0_logic):
                        self._warn_nonfinite(
                            "logic_pred_x0",
                            "Diffusion training: non-finite predicted x0 for logic supervision; disabling logic loss for this batch.",
                        )
                        pred_x0_logic = None

                    # Decode predicted x0 before LogicNet so pathfinding sees tile
                    # logits/walkability rather than arbitrary continuous latents.
                    if pred_x0_logic is not None:
                        pred_tile_logits = self._decode_latent_for_logic(pred_x0_logic)
                        logic_loss, logic_info = self.logic_net(pred_tile_logits, graph_data=logic_graph_data)
                        wfc_pseudo_loss, wfc_pseudo_samples, wfc_pseudo_mean_loss = self._wfc_pseudo_label_loss(
                            pred_tile_logits,
                            real_maps,
                        )

            if self._tensor_is_finite(logic_loss):
                if isinstance(logic_loss, torch.Tensor) and logic_loss.numel() != 1:
                    logic_loss = logic_loss.mean()
                solvability_proxy = self._logic_loss_to_solvability_proxy(logic_loss)
            else:
                self._warn_nonfinite(
                    "logic_loss",
                    "Diffusion training: non-finite logic loss detected; disabling logic loss for this batch.",
                )
                logic_loss = torch.zeros((), device=self.device, dtype=diffusion_loss.dtype)
                solvability_proxy = torch.zeros((), device=self.device, dtype=diffusion_loss.dtype)
        
        # Combined loss
        total_loss = (
            self.config.alpha_visual * diffusion_loss + 
            logic_loss_weight * logic_loss
            + float(getattr(self.config, "alpha_logic_tile", 0.0)) * logic_tile_loss
            + float(getattr(self.config, "alpha_wfc_pseudo", 0.0)) * wfc_pseudo_loss
        )
        if not self._tensor_is_finite(total_loss):
            self._reset_gradient_accumulation()
            self._warn_nonfinite(
                "total_loss",
                "Diffusion training: non-finite total loss detected; skipping optimizer step for this batch.",
            )
            metrics = {
                'loss': 0.0,
                'diffusion_loss': float(diffusion_loss.detach().item()) if self._tensor_is_finite(diffusion_loss) else 0.0,
                'logic_loss': 0.0,
                'logic_tile_loss': 0.0,
                'wfc_pseudo_loss': 0.0,
                'wfc_pseudo_loss_sum': 0.0,
                'wfc_pseudo_loss_contribution': 0.0,
                'wfc_pseudo_samples': 0.0,
                'logic_tile_accuracy': 0.0,
                'solvability_proxy': 0.0,
                'solvability': 0.0,
                'logic_loss_mode_predicted': 1.0 if self.config.logic_loss_mode == 'predicted_latent' else 0.0,
                'skipped_nonfinite_batch': 1.0,
            }
            metrics.update(self._vqvae_codebook_stats())
            return metrics
        
        # Backward / accumulation. global_step counts optimizer updates, not
        # dataloader micro-batches.
        accum_steps = self._gradient_accumulation_steps()
        if self._accumulated_micro_steps() == 0:
            self.optimizer.zero_grad(set_to_none=True)
        self._backward_loss(total_loss)
        self._accumulation_micro_steps = self._accumulated_micro_steps() + 1
        micro_steps = self._accumulated_micro_steps()
        should_step_optimizer = bool(force_optimizer_step or micro_steps >= accum_steps)

        graph_skip_reason = str(logic_info.get('global_graph_skipped', '') or '')
        graph_loss_attempted = bool(logic_enabled and logic_loss_weight > 0.0)
        metrics = {
            'loss': total_loss.item(),
            'diffusion_loss': diffusion_loss.item(),
            'logic_loss': logic_loss.item(),
            'logic_tile_loss': logic_tile_loss.item(),
            'wfc_pseudo_loss': wfc_pseudo_mean_loss.item(),
            'wfc_pseudo_loss_sum': float(wfc_pseudo_mean_loss.detach().item()) * float(wfc_pseudo_samples),
            'wfc_pseudo_loss_contribution': wfc_pseudo_loss.item(),
            'wfc_pseudo_samples': float(wfc_pseudo_samples),
            'logic_tile_accuracy': logic_tile_accuracy.item(),
            'solvability_proxy': solvability_proxy.item(),
            'solvability': solvability_proxy.item(),
            'logic_loss_mode_predicted': 1.0 if self.config.logic_loss_mode == 'predicted_latent' else 0.0,
            'logic_global_graph_loss_skipped': 1.0 if graph_loss_attempted and graph_skip_reason else 0.0,
            'logic_global_graph_supervised': 1.0 if graph_loss_attempted and not graph_skip_reason else 0.0,
            'logic_global_graph_node_coverage': float(logic_info.get('global_graph_node_coverage', 0.0) or 0.0),
            'gradient_accumulation_steps': float(accum_steps),
            'gradient_accumulation_micro_steps': float(micro_steps),
            'optimizer_step': 0.0,
        }
        if not should_step_optimizer:
            metrics.update(self._vqvae_codebook_stats())
            return metrics

        self._unscale_gradients_if_needed()
        modules_for_average = [self.diffusion, self.condition_encoder]
        if bool(getattr(self.config, "logic_net_trainable", True)):
            modules_for_average.append(self.logic_net)
        average_gradients(
            tuple(modules_for_average),
            context=getattr(self, "distributed_context", None),
        )
        self._scale_accumulated_gradients(micro_steps)
        if not self._gradients_are_finite():
            self._reset_gradient_accumulation()
            self._warn_nonfinite(
                "gradient",
                "Diffusion training: non-finite gradients detected; skipping optimizer step for this batch.",
            )
            metrics['skipped_nonfinite_batch'] = 1.0
            metrics['gradient_accumulation_micro_steps'] = 0.0
            metrics.update(self._vqvae_codebook_stats())
            return metrics
        grad_clip_norm = float(max(0.0, float(getattr(self.config, "grad_clip_norm", 1.0))))
        if grad_clip_norm > 0:
            if self._accelerator is not None:
                grad_norms = [
                    self._accelerator.clip_grad_norm_(self.diffusion.parameters(), max_norm=grad_clip_norm),
                    self._accelerator.clip_grad_norm_(self.condition_encoder.parameters(), max_norm=grad_clip_norm),
                ]
                if bool(getattr(self.config, "logic_net_trainable", True)):
                    grad_norms.append(self._accelerator.clip_grad_norm_(self.logic_net.parameters(), max_norm=grad_clip_norm))
            else:
                grad_norms = [
                    torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), max_norm=grad_clip_norm),
                    torch.nn.utils.clip_grad_norm_(self.condition_encoder.parameters(), max_norm=grad_clip_norm),
                ]
                if bool(getattr(self.config, "logic_net_trainable", True)):
                    grad_norms.append(torch.nn.utils.clip_grad_norm_(self.logic_net.parameters(), max_norm=grad_clip_norm))
            if not all(self._tensor_is_finite(norm) for norm in grad_norms):
                self._reset_gradient_accumulation()
                self._warn_nonfinite(
                    "gradient_norm",
                    "Diffusion training: non-finite gradient norm detected after clipping; skipping optimizer step for this batch.",
                )
                metrics['skipped_nonfinite_batch'] = 1.0
                metrics['gradient_accumulation_micro_steps'] = 0.0
                metrics.update(self._vqvae_codebook_stats())
                return metrics
        self._optimizer_step_with_scaler()
        
        # --- Phase 4A: Update EMA model weights ---
        self._update_ema()
        self._accumulation_micro_steps = 0

        self.global_step += 1
        self._apply_lr_warmup(completed_steps=self.global_step)

        # --- Phase 1D: Anneal LogicNet temperature ---
        # Use estimated total steps from config instead of hardcoded epochs*100
        if logic_enabled and hasattr(self.logic_net, 'anneal_temperature'):
            default_total_steps = max(1, int(getattr(self.config, "epochs", 1)) * 100)
            estimated_total_steps = max(1, getattr(self, '_estimated_total_steps', default_total_steps))
            progress = min(1.0, self.global_step / estimated_total_steps)
            self.logic_net.anneal_temperature(progress)

        metrics['optimizer_step'] = 1.0
        metrics['gradient_accumulation_micro_steps'] = 0.0
        metrics.update(self._vqvae_codebook_stats())
        return metrics

    def dpo_step(
        self,
        chosen_maps: torch.Tensor,
        rejected_maps: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
        graph_data: Optional[Dict[str, torch.Tensor]] = None,
        *,
        rejected_conditioning: Optional[torch.Tensor] = None,
        rejected_graph_data: Optional[Dict[str, torch.Tensor]] = None,
        beta: float = 0.1,
        reference_model: Optional[nn.Module] = None,
    ) -> Dict[str, float]:
        """
        Run one Diffusion-DPO update on preferred/rejected map pairs.

        Preference pairs can be produced by hard symbolic validation, e.g. a
        solvable dungeon as `chosen_maps` and a broken dungeon as
        `rejected_maps` under the same graph condition.
        """
        self.diffusion.train()
        self.condition_encoder.train()
        if conditioning is None:
            conditioning = self.get_dummy_conditioning(int(chosen_maps.shape[0]))
        if rejected_conditioning is None:
            rejected_conditioning = conditioning

        chosen_z = self.encode_to_latent(chosen_maps)
        rejected_z = self.encode_to_latent(rejected_maps)
        objective = str(getattr(self.config, "diffusion_training_objective", "diffusion")).strip().lower()
        ref = reference_model if reference_model is not None else self.ema_diffusion
        if ref is self.diffusion:
            ref = None

        with self._autocast_context():
            loss, dpo_metrics = self.diffusion.diffusion_dpo_loss(
                chosen_z,
                rejected_z,
                conditioning,
                graph_data=graph_data,
                rejected_context=rejected_conditioning,
                rejected_graph_data=rejected_graph_data,
                reference_model=ref,
                beta=float(beta),
                objective=objective,
            )
        if not self._tensor_is_finite(loss):
            self.optimizer.zero_grad(set_to_none=True)
            self._warn_nonfinite(
                "dpo_loss",
                "Diffusion-DPO: non-finite DPO loss detected; skipping optimizer step for this batch.",
            )
            return {
                "loss": 0.0,
                "dpo_loss": 0.0,
                "dpo_margin": 0.0,
                "dpo_accuracy": 0.0,
                "skipped_nonfinite_batch": 1.0,
            }

        self.optimizer.zero_grad(set_to_none=True)
        if self._accelerator is not None:
            self._accelerator.backward(loss)
        else:
            loss.backward()
        modules_for_average = [self.diffusion, self.condition_encoder]
        if bool(getattr(self.config, "logic_net_trainable", True)):
            modules_for_average.append(self.logic_net)
        average_gradients(
            tuple(modules_for_average),
            context=getattr(self, "distributed_context", None),
        )
        if not self._gradients_are_finite():
            self.optimizer.zero_grad(set_to_none=True)
            self._warn_nonfinite(
                "dpo_gradient",
                "Diffusion-DPO: non-finite gradients detected; skipping optimizer step for this batch.",
            )
            return {
                "loss": float(loss.detach().item()) if self._tensor_is_finite(loss) else 0.0,
                "dpo_loss": float(loss.detach().item()) if self._tensor_is_finite(loss) else 0.0,
                "dpo_margin": float(dpo_metrics["dpo_margin"].detach().item()) if self._tensor_is_finite(dpo_metrics.get("dpo_margin")) else 0.0,
                "dpo_accuracy": float(dpo_metrics["dpo_accuracy"].detach().item()) if self._tensor_is_finite(dpo_metrics.get("dpo_accuracy")) else 0.0,
                "skipped_nonfinite_batch": 1.0,
            }
        grad_clip = float(getattr(self.config, "grad_clip_norm", 0.0))
        if grad_clip > 0:
            if self._accelerator is not None:
                grad_norms = [
                    self._accelerator.clip_grad_norm_(self.diffusion.parameters(), grad_clip),
                    self._accelerator.clip_grad_norm_(self.condition_encoder.parameters(), grad_clip),
                ]
                if bool(getattr(self.config, "logic_net_trainable", True)):
                    grad_norms.append(self._accelerator.clip_grad_norm_(self.logic_net.parameters(), grad_clip))
            else:
                grad_norms = [
                    torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), grad_clip),
                    torch.nn.utils.clip_grad_norm_(self.condition_encoder.parameters(), grad_clip),
                ]
                if bool(getattr(self.config, "logic_net_trainable", True)):
                    grad_norms.append(torch.nn.utils.clip_grad_norm_(self.logic_net.parameters(), grad_clip))
            if not all(self._tensor_is_finite(norm) for norm in grad_norms):
                self.optimizer.zero_grad(set_to_none=True)
                self._warn_nonfinite(
                    "dpo_gradient_norm",
                    "Diffusion-DPO: non-finite gradient norm detected after clipping; skipping optimizer step for this batch.",
                )
                return {
                    "loss": float(loss.detach().item()) if self._tensor_is_finite(loss) else 0.0,
                    "dpo_loss": float(loss.detach().item()) if self._tensor_is_finite(loss) else 0.0,
                    "dpo_margin": float(dpo_metrics["dpo_margin"].detach().item()) if self._tensor_is_finite(dpo_metrics.get("dpo_margin")) else 0.0,
                    "dpo_accuracy": float(dpo_metrics["dpo_accuracy"].detach().item()) if self._tensor_is_finite(dpo_metrics.get("dpo_accuracy")) else 0.0,
                    "skipped_nonfinite_batch": 1.0,
                }
        self.optimizer.step()
        self._update_ema()
        self.global_step += 1
        self._apply_lr_warmup(completed_steps=self.global_step)

        return {
            "loss": float(loss.detach().item()),
            "dpo_loss": float(loss.detach().item()),
            "dpo_margin": float(dpo_metrics["dpo_margin"].item()),
            "dpo_accuracy": float(dpo_metrics["dpo_accuracy"].item()),
            "chosen_score": float(dpo_metrics["chosen_score"].item()),
            "rejected_score": float(dpo_metrics["rejected_score"].item()),
            "skipped_nonfinite_batch": 0.0,
        }
    
    def _extract_coords_from_maps(self, real_maps: torch.Tensor) -> Tuple[Tuple[int,int], Tuple[int,int]]:
        """Extract start/goal coordinates from map tensors. Fallback to defaults."""
        start, goal = extract_start_goal(real_maps[0])
        return (start if start else (2, 2)), (goal if goal else (13, 8))

    def train_epoch(
        self,
        dataloader: DataLoader,
        sampler: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        Train for one epoch using real graph data from .dot files.
        
        The dataloader returns (images, graph_list) when load_graphs=True.
        Each graph in graph_list is a dict from zelda_loader._extract_graph()
        containing real mission topology from the VGLC .dot files.
        """
        metrics_sum = {
            'loss': 0,
            'diffusion_loss': 0,
            'logic_loss': 0,
            'logic_tile_loss': 0,
            'wfc_pseudo_loss': 0,
            'wfc_pseudo_loss_sum': 0,
            'wfc_pseudo_loss_contribution': 0,
            'wfc_pseudo_samples': 0,
            'logic_tile_accuracy': 0,
            'solvability_proxy': 0,
            'solvability': 0,
        }
        num_batches = 0
        
        # DESIGN-08: Compute actual optimizer steps for temperature annealing
        total_epochs = int(getattr(self.config, "epochs", self.epoch + 1))
        steps_per_epoch = int(math.ceil(len(dataloader) / float(self._gradient_accumulation_steps())))
        self._estimated_total_steps = max(1, total_epochs * max(1, steps_per_epoch))
        
        include_logic = bool(getattr(self.config, "logic_net_enabled", True)) and self.epoch >= self.config.warmup_epochs
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(int(self.epoch))
        logger.info(
            "Train epoch %d/%d: logic_loss_%s (warmup_epochs=%d)",
            int(self.epoch),
            total_epochs,
            "enabled" if include_logic and self.config.alpha_logic > 0 else "disabled",
            int(self.config.warmup_epochs),
        )
        
        for batch_idx, batch_data in enumerate(dataloader):
            # Handle (images, graph_list) from graph_collate_fn
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                real_maps, graph_list = batch_data
            else:
                real_maps = batch_data
                graph_list = None
            if graph_list is not None and float(getattr(self.config, "puzzle_structure_dropout_prob", 0.0)) > 0.0:
                real_maps, graph_list = apply_puzzle_structure_dropout_batch(
                    real_maps,
                    graph_list,
                    num_classes=int(self.config.num_classes),
                    dropout_prob=float(self.config.puzzle_structure_dropout_prob),
                )
            real_maps = real_maps.to(self.device)
            
            # === Build conditioning from REAL graph data ===
            conditioning = None
            logic_graph_data = None
            diffusion_graph_data = None
            
            if graph_list is not None and self.condition_encoder is not None:
                try:
                    # Encode each graph through GNN and stack.
                    cond_vectors = []
                    for graph_dict in graph_list:
                        c_i = self._encode_graph_conditioning(graph_dict)
                        cond_vectors.append(c_i)
                    conditioning = self._stack_conditioning_vectors(cond_vectors)
                except (AttributeError, RuntimeError, ValueError, TypeError) as e:
                    logger.debug(f"Graph conditioning failed: {e}")
                    conditioning = None

                try:
                    diffusion_graph_data = self._stack_diffusion_graph_batch(graph_list)
                except (AttributeError, RuntimeError, ValueError, TypeError) as e:
                    logger.debug(f"Diffusion graph-data build failed: {e}")
                    diffusion_graph_data = None
                
                if include_logic:
                    logic_graph_data = diffusion_graph_data
            
            is_last_batch = int(batch_idx) + 1 >= len(dataloader)
            metrics = self.train_step(
                real_maps,
                conditioning=conditioning,
                include_logic_loss=include_logic,
                logic_graph_data=logic_graph_data,
                diffusion_graph_data=diffusion_graph_data,
                force_optimizer_step=is_last_batch,
            )
            
            for k, v in metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0.0) + float(v)
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logger.debug(
                    f"Batch {batch_idx}: loss={metrics['loss']:.4f}, "
                    f"diffusion={metrics['diffusion_loss']:.4f}, "
                    f"train_solvability_proxy={metrics.get('solvability_proxy', metrics['solvability']):.4f}, "
                    f"logic_loss={metrics['logic_loss']:.4f}"
                )
        
        self.epoch += 1
        self.scheduler.step()

        metrics_sum["num_batches"] = float(num_batches)
        reduced = reduce_scalar_metrics(
            metrics_sum,
            device=self.device,
            context=getattr(self, "distributed_context", None),
            average=False,
        )
        total_batches = float(max(1.0, reduced.pop("num_batches", float(num_batches))))
        wfc_loss_sum = float(reduced.pop("wfc_pseudo_loss_sum", 0.0))
        wfc_sample_total = float(reduced.get("wfc_pseudo_samples", 0.0))
        epoch_metrics = {k: float(v) / total_batches for k, v in reduced.items()}
        epoch_metrics["wfc_pseudo_loss"] = (
            wfc_loss_sum / wfc_sample_total
            if wfc_sample_total > 0.0
            else 0.0
        )
        epoch_metrics["wfc_pseudo_total_samples"] = wfc_sample_total
        cache = getattr(self, "_latent_cache", None)
        if isinstance(cache, FrozenLatentCache) and cache.total_lookups > 0:
            epoch_metrics["latent_cache_hit_rate"] = float(cache.hit_rate)
            epoch_metrics["latent_cache_size"] = float(len(cache))
        return epoch_metrics
    
    @torch.no_grad()
    def validate(
        self,
        dataloader: DataLoader,
        num_samples: int = 4,
        num_diffusion_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """Validate model using EMA weights and real graph conditioning."""
        eval_model = self.ema_diffusion if hasattr(self, 'ema_diffusion') else self.diffusion
        eval_model.eval()

        if num_diffusion_samples is None:
            num_diffusion_samples = int(getattr(self.config, "validation_num_diffusion_samples", num_samples))

        total_diffusion_loss = 0.0
        num_diffusion_eval = 0
        total_logic_loss = 0.0
        total_solvability_proxy = 0.0
        total_grid_reach_loss = 0.0
        total_graph_reach_loss = 0.0
        total_hard_solvability = 0.0
        total_hard_solvability_after_repair = 0.0
        total_logicnet_score_after_repair = 0.0
        total_validation_repair_success = 0.0
        num_logic_metric_eval = 0
        num_hard_solvability_eval = 0
        num_repaired_solvability_eval = 0
        total_logic_tile_accuracy = 0.0
        num_logic_tile_eval = 0
        logic_eval_enabled = bool(getattr(self.config, "logic_net_enabled", True))
        num_generated_eval = 0 if logic_eval_enabled else int(num_samples)
        skipped_nonfinite = 0
        guidance_suppressed_low_tile_accuracy = 0

        for batch_data in dataloader:
            # Handle (images, graph_list) from graph_collate_fn
            if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                real_maps = batch_data[0]
                graph_list = batch_data[1] if len(batch_data) > 1 else None
            else:
                real_maps = batch_data
                graph_list = None
            real_maps = real_maps.to(self.device)
            batch_size = real_maps.shape[0]
            
            # Build conditioning from real graphs if available
            conditioning = None
            diffusion_graph_data = None
            if graph_list is not None:
                cond_vectors = []
                for idx, graph_dict in enumerate(graph_list):
                    try:
                        c_i = self._encode_graph_conditioning(graph_dict)
                    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
                        logger.debug(
                            "Graph conditioning encode failed for sample %d; using dummy conditioning: %s",
                            idx,
                            exc,
                        )
                        c_i = self.get_dummy_conditioning(1)
                    cond_vectors.append(c_i)
                if cond_vectors:
                    conditioning = self._stack_conditioning_vectors(cond_vectors)
                try:
                    diffusion_graph_data = self._stack_diffusion_graph_batch(graph_list)
                except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
                    logger.debug("Diffusion graph-data build failed during validation: %s", exc)
                    diffusion_graph_data = None
            
            if conditioning is None:
                conditioning = self.get_dummy_conditioning(batch_size)
            
            # Encode real maps to get latent shape
            z_0 = self.encode_to_latent(real_maps)

            batch_logic_tile_accuracy: Optional[float] = None
            if logic_eval_enabled and hasattr(self.logic_net, "tile_classifier"):
                try:
                    tile_targets = self._tile_targets_from_maps(real_maps)
                    tile_logits = self.logic_net.tile_classifier(z_0.detach())
                    if hasattr(self.logic_net, "_project_tile_logits_to_room"):
                        tile_logits = self.logic_net._project_tile_logits_to_room(tile_logits)
                    tile_accuracy = (tile_logits.argmax(dim=1) == tile_targets).float().mean()
                    if self._tensor_is_finite(tile_accuracy):
                        batch_logic_tile_accuracy = float(tile_accuracy.detach().item())
                        total_logic_tile_accuracy += batch_logic_tile_accuracy * batch_size
                        num_logic_tile_eval += batch_size
                except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
                    logger.debug("LogicNet tile-classifier validation failed; omitting tile accuracy: %s", exc)

            if num_diffusion_eval < int(num_diffusion_samples):
                diffusion_loss = self._diffusion_objective_loss(
                    z_0,
                    conditioning,
                    diffusion_graph_data,
                    model=eval_model,
                )
                if self._tensor_is_finite(diffusion_loss):
                    diffusion_batch = min(batch_size, int(num_diffusion_samples) - num_diffusion_eval)
                    total_diffusion_loss += float(diffusion_loss.item()) * diffusion_batch
                    num_diffusion_eval += diffusion_batch
                else:
                    skipped_nonfinite += int(batch_size)
                    self._warn_nonfinite(
                        "validation_diffusion_loss",
                        "Diffusion validation: non-finite denoising loss detected; skipping this validation batch.",
                    )

            if logic_eval_enabled and num_generated_eval < int(num_samples):
                # Generate samples using EMA model
                guidance_module = getattr(eval_model, "guidance", None)
                old_guidance_scale = getattr(guidance_module, "guidance_scale", None)
                suppress_guidance = (
                    batch_logic_tile_accuracy is not None
                    and float(getattr(self.config, "min_logic_tile_accuracy_for_guidance", 0.0)) > 0.0
                    and batch_logic_tile_accuracy < float(getattr(self.config, "min_logic_tile_accuracy_for_guidance", 0.0))
                    and old_guidance_scale is not None
                )
                if suppress_guidance:
                    guidance_suppressed_low_tile_accuracy += int(batch_size)
                    guidance_module.guidance_scale = 0.0
                try:
                    objective = str(getattr(self.config, "diffusion_training_objective", "diffusion")).strip().lower()
                    if objective == "flow_matching" and hasattr(eval_model, "flow_ode_sample"):
                        z_gen = eval_model.flow_ode_sample(
                            conditioning,
                            shape=z_0.shape,
                            graph_data=diffusion_graph_data,
                            num_steps=min(int(getattr(self.config, "num_timesteps", 50)), 50),
                        )
                    else:
                        z_gen = eval_model.sample(conditioning, shape=z_0.shape, graph_data=diffusion_graph_data)
                finally:
                    if suppress_guidance:
                        guidance_module.guidance_scale = old_guidance_scale
                if not self._tensor_is_finite(z_gen):
                    skipped_nonfinite += int(batch_size)
                    self._warn_nonfinite(
                        "validation_sample",
                        "Diffusion validation: generated non-finite latent sample; skipping this validation batch.",
                    )
                    if num_diffusion_eval >= int(num_diffusion_samples):
                        continue
                else:
                    # Build LogicNet graph data if available
                    logic_graph_data = diffusion_graph_data

                    # LogicNet: evaluate with graph topology
                    logic_loss, _logic_info = self.logic_net(z_gen, graph_data=logic_graph_data)
                    if not self._tensor_is_finite(logic_loss):
                        skipped_nonfinite += int(batch_size)
                        self._warn_nonfinite(
                            "validation_logic_loss",
                            "Diffusion validation: non-finite logic loss detected on sampled latent; skipping this validation batch.",
                        )
                    else:
                        generated_batch = min(batch_size, int(num_samples) - num_generated_eval)
                        solvability_proxy = float(self._logic_loss_to_solvability_proxy(logic_loss).item())
                        total_logic_loss += float(logic_loss.item()) * generated_batch
                        total_solvability_proxy += solvability_proxy * generated_batch
                        if isinstance(_logic_info, dict):
                            grid_loss = _logic_info.get("grid_reach_loss")
                            graph_loss = _logic_info.get("graph_reach_loss")
                            if self._tensor_is_finite(grid_loss):
                                total_grid_reach_loss += float(torch.as_tensor(grid_loss).detach().mean().item()) * generated_batch
                            if self._tensor_is_finite(graph_loss):
                                total_graph_reach_loss += float(torch.as_tensor(graph_loss).detach().mean().item()) * generated_batch
                            num_logic_metric_eval += generated_batch
                        if hasattr(self, "vqvae") and hasattr(self.vqvae, "decode"):
                            try:
                                decoded = self._decode_latent_for_logic(z_gen[:generated_batch])
                                hard_solvability = self._compute_hard_solvability(decoded)
                                total_hard_solvability += hard_solvability * generated_batch
                                num_hard_solvability_eval += generated_batch
                                repaired_decoded, repair_success_rate = self._repair_validation_decoded_logits(
                                    decoded,
                                    graph_data=logic_graph_data,
                                )
                                if isinstance(repaired_decoded, torch.Tensor):
                                    repaired_hard = self._compute_hard_solvability(repaired_decoded)
                                    total_hard_solvability_after_repair += repaired_hard * generated_batch
                                    total_validation_repair_success += float(repair_success_rate) * generated_batch
                                    repaired_maps = (
                                        repaired_decoded.argmax(dim=1).float()
                                        / float(max(1, int(self.config.num_classes) - 1))
                                    ).unsqueeze(1)
                                    z_repaired = self.encode_to_latent(repaired_maps.to(self.device))
                                    repaired_logic_loss, _repaired_logic_info = self.logic_net(
                                        z_repaired,
                                        graph_data=logic_graph_data,
                                    )
                                    if self._tensor_is_finite(repaired_logic_loss):
                                        repaired_score = float(
                                            self._logic_loss_to_solvability_proxy(repaired_logic_loss).item()
                                        )
                                        total_logicnet_score_after_repair += repaired_score * generated_batch
                                    num_repaired_solvability_eval += generated_batch
                            except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
                                logger.debug("Hard solvability validation failed; omitting metric: %s", exc)
                        num_generated_eval += generated_batch

            if num_generated_eval >= int(num_samples) and num_diffusion_eval >= int(num_diffusion_samples):
                break

        if num_generated_eval <= 0 or num_diffusion_eval <= 0:
            return {
                'val_diffusion_loss': float("inf"),
                'val_logic_loss': float("inf"),
                'val_total_loss': float("inf"),
                'val_solvability_proxy': 0.0,
                'val_solvability': 0.0,
                'val_logic_tile_accuracy': total_logic_tile_accuracy / max(num_logic_tile_eval, 1),
                'val_grid_reach_loss': total_grid_reach_loss / max(num_logic_metric_eval, 1),
                'val_graph_reach_loss': total_graph_reach_loss / max(num_logic_metric_eval, 1),
                'val_hard_solvability': total_hard_solvability / max(num_hard_solvability_eval, 1),
                'val_hard_solvability_after_repair': total_hard_solvability_after_repair / max(num_repaired_solvability_eval, 1),
                'val_logicnet_score_after_repair': total_logicnet_score_after_repair / max(num_repaired_solvability_eval, 1),
                'val_neural_repair_success_rate': total_validation_repair_success / max(num_repaired_solvability_eval, 1),
                'val_logic_guidance_suppressed_low_tile_accuracy': float(guidance_suppressed_low_tile_accuracy),
                'val_skipped_nonfinite': float(skipped_nonfinite),
            }

        include_logic = (
            bool(getattr(self.config, "logic_net_enabled", True))
            and self.epoch >= self.config.warmup_epochs
            and self.config.alpha_logic > 0
        )
        val_diffusion_loss = total_diffusion_loss / max(num_diffusion_eval, 1)
        val_logic_loss = total_logic_loss / max(num_generated_eval, 1)
        val_total_loss = compute_teacher_validation_total_loss(
            val_diffusion_loss=val_diffusion_loss,
            val_logic_loss=val_logic_loss,
            alpha_visual=float(getattr(self.config, "alpha_visual", 1.0)),
            alpha_logic=float(getattr(self.config, "alpha_logic", 0.0)),
            include_logic_loss=bool(include_logic),
        )

        return {
            'val_diffusion_loss': val_diffusion_loss,
            'val_logic_loss': val_logic_loss,
            'val_total_loss': val_total_loss,
            'val_solvability_proxy': total_solvability_proxy / max(num_generated_eval, 1),
            'val_solvability': total_solvability_proxy / max(num_generated_eval, 1),
            'val_logic_tile_accuracy': total_logic_tile_accuracy / max(num_logic_tile_eval, 1),
            'val_grid_reach_loss': total_grid_reach_loss / max(num_logic_metric_eval, 1),
            'val_graph_reach_loss': total_graph_reach_loss / max(num_logic_metric_eval, 1),
            'val_hard_solvability': total_hard_solvability / max(num_hard_solvability_eval, 1),
            'val_hard_solvability_after_repair': total_hard_solvability_after_repair / max(num_repaired_solvability_eval, 1),
            'val_logicnet_score_after_repair': total_logicnet_score_after_repair / max(num_repaired_solvability_eval, 1),
            'val_neural_repair_success_rate': total_validation_repair_success / max(num_repaired_solvability_eval, 1),
            'val_logic_guidance_suppressed_low_tile_accuracy': float(guidance_suppressed_low_tile_accuracy),
            'val_skipped_nonfinite': float(skipped_nonfinite),
        }
    
    def _build_resume_checkpoint_payload(self, metrics: Optional[Dict] = None) -> Dict[str, Any]:
        payload = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'diffusion_state_dict': self.diffusion.state_dict(),
            'ema_diffusion_state_dict': self.ema_diffusion.state_dict(),
            'condition_encoder_state_dict': self.condition_encoder.state_dict(),
            'logic_net_state_dict': self.logic_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.to_dict(),
            'metrics': metrics,
            # Store schedule/prediction type for inference consistency
            'schedule_type': self.config.schedule_type,
        }
        if bool(getattr(self.config, "logic_net_enabled", True)):
            payload['logic_net_state_dict'] = self.logic_net.state_dict()
        return payload

    def _build_inference_checkpoint_payload(self, metrics: Optional[Dict] = None) -> Dict[str, Any]:
        payload = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'diffusion_state_dict': self.diffusion.state_dict(),
            'ema_diffusion_state_dict': self.ema_diffusion.state_dict(),
            'condition_encoder_state_dict': self.condition_encoder.state_dict(),
            'logic_net_state_dict': self.logic_net.state_dict(),
            'config': self.config.to_dict(),
            'metrics': metrics,
            'schedule_type': self.config.schedule_type,
        }
        if bool(getattr(self.config, "logic_net_enabled", True)):
            payload['logic_net_state_dict'] = self.logic_net.state_dict()
        return payload

    @staticmethod
    def _prefixed_safetensors_state(prefix: str, state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            f"{prefix}.{key}": value.detach().cpu()
            for key, value in state_dict.items()
            if isinstance(value, torch.Tensor)
        }

    @staticmethod
    def _extract_prefixed_safetensors_state(payload: Mapping[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        stem = f"{prefix}."
        return {
            key[len(stem):]: value
            for key, value in payload.items()
            if isinstance(key, str) and key.startswith(stem)
        }

    def save_checkpoint(self, path: str, metrics: Optional[Dict] = None, *, include_optimizer: bool = True):
        """Save training or inference checkpoint."""
        checkpoint = (
            self._build_resume_checkpoint_payload(metrics)
            if bool(include_optimizer)
            else self._build_inference_checkpoint_payload(metrics)
        )
        atomic_torch_save(checkpoint, path)
        safetensors_path: Optional[Path] = None
        # Also write a tensor-only safetensors sidecar with deployable weights.
        if _HAS_SAFETENSORS:
            try:
                safetensors_path = Path(path).with_suffix('.safetensors')
                safetensors_payload: Dict[str, torch.Tensor] = {}
                safetensors_payload.update(self._prefixed_safetensors_state("diffusion", self.diffusion.state_dict()))
                safetensors_payload.update(
                    self._prefixed_safetensors_state("ema_diffusion", self.ema_diffusion.state_dict())
                )
                safetensors_payload.update(
                    self._prefixed_safetensors_state("condition_encoder", self.condition_encoder.state_dict())
                )
                if bool(getattr(self.config, "logic_net_enabled", True)):
                    safetensors_payload.update(self._prefixed_safetensors_state("logic_net", self.logic_net.state_dict()))
                _save_safetensors(safetensors_payload, str(safetensors_path))
                logger.debug("Saved safetensors sidecar: %s", safetensors_path)
            except Exception as _st_err:  # noqa: BLE001
                logger.warning("safetensors save failed (%s); .pth checkpoint is intact.", _st_err)
                safetensors_path = None
        write_checkpoint_metadata(
            path,
            model_type="diffusion_resume" if include_optimizer else "diffusion",
            architecture={
                "latent_dim": int(self.config.latent_dim),
                "context_dim": int(self.config.context_dim),
                "num_timesteps": int(self.config.num_timesteps),
                "schedule_type": str(self.config.schedule_type),
                "diffusion_training_objective": str(getattr(self.config, "diffusion_training_objective", "diffusion")),
                "denoiser_backbone": str(getattr(self.config, "denoiser_backbone", "unet")),
                "pag_scale": float(getattr(self.config, "pag_scale", 0.0)),
                "dit_depth": int(getattr(self.config, "dit_depth", 4)),
                "dit_patch_size": int(getattr(self.config, "dit_patch_size", 1)),
                "dit_mlp_ratio": float(getattr(self.config, "dit_mlp_ratio", 4.0)),
                "num_classes": int(self.config.num_classes),
                "vqvae_hidden_dim": int(self.config.vqvae_hidden_dim),
                "vqvae_codebook_size": int(self.config.vqvae_codebook_size),
                "vqvae_architecture": str(getattr(self.config, "vqvae_architecture", "vqvae")),
                "vqvae_top_codebook_size": getattr(self.config, "vqvae_top_codebook_size", None),
                "vqvae_top_latent_dim": getattr(self.config, "vqvae_top_latent_dim", None),
                "vqvae_use_coordconv": bool(self.config.vqvae_use_coordconv),
            },
            extra={
                "epoch": int(self.epoch),
                "global_step": int(self.global_step),
                "checkpoint_kind": "resume" if include_optimizer else "inference",
                "primary_format": "torch_pth",
                "safetensors_sidecar": str(safetensors_path.name) if safetensors_path is not None else None,
                "safetensors_contains_optimizer": False,
                "contains": (
                    (
                        ["diffusion", "ema_diffusion", "condition_encoder"]
                        + (["logic_net"] if bool(getattr(self.config, "logic_net_enabled", True)) else [])
                        + ["optimizer", "scheduler"]
                    )
                    if include_optimizer
                    else (
                        ["diffusion", "ema_diffusion", "condition_encoder"]
                        + (["logic_net"] if bool(getattr(self.config, "logic_net_enabled", True)) else [])
                    )
                ),
                "vqvae_checkpoint": str(getattr(self.config, "vqvae_checkpoint", "") or ""),
                "topology_anchor_policy": build_topology_anchor_policy_metadata(
                    semantic_role_prior_strength=self.config.semantic_role_prior_strength,
                    semantic_puzzle_offset=self.config.semantic_puzzle_offset,
                    topology_supervision_mode=self.config.topology_supervision_mode,
                ),
            },
        )
        if safetensors_path is not None:
            write_checkpoint_metadata(
                str(safetensors_path),
                model_type="diffusion_safetensors_inference",
                architecture={
                    "latent_dim": int(self.config.latent_dim),
                    "context_dim": int(self.config.context_dim),
                    "num_timesteps": int(self.config.num_timesteps),
                    "schedule_type": str(self.config.schedule_type),
                    "diffusion_training_objective": str(getattr(self.config, "diffusion_training_objective", "diffusion")),
                    "denoiser_backbone": str(getattr(self.config, "denoiser_backbone", "unet")),
                    "num_classes": int(self.config.num_classes),
                },
                extra={
                    "epoch": int(self.epoch),
                    "global_step": int(self.global_step),
                    "checkpoint_kind": "inference",
                    "primary_format": "safetensors",
                    "contains_optimizer": False,
                    "contains": (
                        ["diffusion", "ema_diffusion", "condition_encoder"]
                        + (["logic_net"] if bool(getattr(self.config, "logic_net_enabled", True)) else [])
                    ),
                    "torch_resume_checkpoint": Path(path).name if include_optimizer else None,
                },
            )
        log_checkpoint_artifact(
            logger,
            path,
            checkpoint_dir=Path(path).parent,
            label="Saved checkpoint",
        )
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        if str(path).lower().endswith(".safetensors"):
            if not _HAS_SAFETENSORS:
                raise ImportError("Loading .safetensors checkpoints requires the safetensors package.")
            payload = _load_safetensors(path, device=str(self.device))
            diffusion_state = self._extract_prefixed_safetensors_state(payload, "diffusion")
            ema_state = self._extract_prefixed_safetensors_state(payload, "ema_diffusion")
            condition_state = self._extract_prefixed_safetensors_state(payload, "condition_encoder")
            logic_state = self._extract_prefixed_safetensors_state(payload, "logic_net")
            for name, state in (
                ("diffusion", diffusion_state),
                ("ema_diffusion", ema_state),
                ("condition_encoder", condition_state),
                ("logic_net", logic_state),
            ):
                if state and not self._state_dict_is_finite(state):
                    raise ValueError(
                        f"Checkpoint {path} contains non-finite values in `{name}` and cannot be loaded safely."
                    )
            if not diffusion_state or not condition_state:
                raise ValueError(
                    f"Safetensors checkpoint {path} must contain at least diffusion.* and condition_encoder.* weights."
                )
            self.diffusion.load_state_dict(diffusion_state)
            self.ema_diffusion.load_state_dict(ema_state or diffusion_state)
            self.condition_encoder.load_state_dict(condition_state)
            if logic_state:
                self.logic_net.load_state_dict(logic_state)
            metadata = _load_checkpoint_metadata_sidecar(path)
            extra = dict(metadata.get("extra", {}) or {})
            self.epoch = int(extra.get("epoch", getattr(self, "epoch", 0)))
            self.global_step = int(extra.get("global_step", getattr(self, "global_step", 0)))
            self._reset_gradient_accumulation()
            self._configure_guidance()
            self._configure_guidance(self.ema_diffusion)
            logger.info(
                "Loaded tensor-only safetensors checkpoint from %s (epoch %d, global_step %d); optimizer/scheduler state was not restored.",
                path,
                int(self.epoch),
                int(self.global_step),
            )
            return

        checkpoint = safe_torch_load(path, map_location=self.device)
        for key in ('diffusion_state_dict', 'ema_diffusion_state_dict'):
            if key in checkpoint:
                checkpoint[key], removed = self._strip_embedded_guidance_logic_net_state(checkpoint[key])
                if removed:
                    logger.warning(
                        "Stripped %d legacy guidance.logic_net.* tensor(s) from `%s` while loading %s; "
                        "using `logic_net_state_dict` as the LogicNet source of truth.",
                        int(removed),
                        key,
                        path,
                    )

        for key in (
            'diffusion_state_dict',
            'ema_diffusion_state_dict',
            'condition_encoder_state_dict',
            'logic_net_state_dict',
        ):
            if key in checkpoint and not self._state_dict_is_finite(checkpoint[key]):
                raise ValueError(
                    f"Checkpoint {path} contains non-finite values in `{key}` and cannot be resumed safely."
                )
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self._reset_gradient_accumulation()
        self.diffusion.load_state_dict(checkpoint['diffusion_state_dict'])
        if 'ema_diffusion_state_dict' in checkpoint:
            self.ema_diffusion.load_state_dict(checkpoint['ema_diffusion_state_dict'])
        self.condition_encoder.load_state_dict(checkpoint['condition_encoder_state_dict'])
        if 'logic_net_state_dict' in checkpoint:
            self.logic_net.load_state_dict(checkpoint['logic_net_state_dict'])
        optimizer_state_loaded = False
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                optimizer_state_loaded = True
            except ValueError as exc:
                logger.warning(
                    "Skipping optimizer state from %s because it is incompatible with the current trainer: %s",
                    path,
                    exc,
                )
        if 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except ValueError as exc:
                reason = (
                    "optimizer state was not restored"
                    if not optimizer_state_loaded
                    else "the scheduler state is incompatible with the current trainer"
                )
                logger.warning(
                    "Skipping scheduler state from %s because %s: %s",
                    path,
                    reason,
                    exc,
                )
        
        # Re-wire LogicNet into guidance after loading
        self._configure_guidance()
        self._configure_guidance(self.ema_diffusion)

        logger.info(f"Loaded checkpoint from {path} (epoch {self.epoch})")


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_diffusion(config: DiffusionTrainingConfig) -> DiffusionTrainer:
    """Main training function."""
    logger.info(f"Starting diffusion training with config: {config.to_dict()}")

    distributed_context = initialize_distributed(
        enabled=bool(getattr(config, "distributed_enabled", False)),
        backend=str(getattr(config, "distributed_backend", "nccl")),
    )
    worker_seed = seed_everything(int(getattr(config, "seed", 42)) + int(distributed_context.rank))
    logger.info(
        "Diffusion trainer seeds initialized: base_seed=%d worker_seed=%d rank=%d",
        int(getattr(config, "seed", 42)),
        worker_seed,
        int(distributed_context.rank),
    )

    try:
        base_loader = create_dataloader(
            config.data_dir,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=config.drop_last,
            use_vglc=config.use_vglc,
            normalize=config.normalize,
            room_level=config.room_level,
            load_graphs=True,
            node_feature_dim=config.node_feature_dim,
            edge_feature_dim=config.edge_feature_dim,
            topology_supervision_mode=config.topology_supervision_mode,
            semantic_role_prior_strength=config.semantic_role_prior_strength,
            semantic_puzzle_offset=config.semantic_puzzle_offset,
            dungeon_ids=config.train_dungeon_ids,
            variants=config.variants,
        )
        base_dataset = base_loader.dataset
        from src.train_vqvae import split_dataset_for_vqvae_validation
        train_dataset, val_dataset = split_dataset_for_vqvae_validation(
            base_dataset,
            validation_fraction=float(getattr(config, "validation_fraction", 0.0)),
            seed=int(getattr(config, "seed", 42)),
        )
        use_dungeon_batch_mode = (
            bool(getattr(config, "dungeon_batch_mode", True))
            and bool(getattr(config, "room_level", True))
            and not bool(getattr(distributed_context, "enabled", False))
        )
        if use_dungeon_batch_mode:
            train_sampler = DungeonBatchSampler.from_dataset(
                train_dataset,
                shuffle=config.shuffle_train,
                drop_last=False,
                seed=int(getattr(config, "seed", 42)),
            )
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=train_sampler,
                collate_fn=graph_collate_fn,
                **dataloader_runtime_kwargs(num_workers=config.num_workers, pin_memory=config.pin_memory),
            )
        else:
            train_sampler = make_distributed_sampler(
                train_dataset,
                context=distributed_context,
                shuffle=config.shuffle_train,
                drop_last=config.drop_last,
                seed=int(getattr(config, "seed", 42)),
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=(bool(config.shuffle_train) if train_sampler is None else False),
                sampler=train_sampler,
                drop_last=config.drop_last,
                collate_fn=graph_collate_fn,
                **dataloader_runtime_kwargs(num_workers=config.num_workers, pin_memory=config.pin_memory),
            )

        val_loader = None
        if distributed_context.is_main_process:
            eval_source = val_dataset if val_dataset is not None else train_dataset
            if use_dungeon_batch_mode:
                val_batch_sampler = DungeonBatchSampler.from_dataset(
                    eval_source,
                    shuffle=bool(config.shuffle_val),
                    drop_last=False,
                    seed=int(getattr(config, "seed", 42)) + 10_000,
                )
                val_loader = DataLoader(
                    eval_source,
                    batch_sampler=val_batch_sampler,
                    collate_fn=graph_collate_fn,
                    **dataloader_runtime_kwargs(num_workers=config.num_workers, pin_memory=config.pin_memory),
                )
            else:
                val_loader = DataLoader(
                    eval_source,
                    batch_size=config.batch_size,
                    shuffle=bool(config.shuffle_val),
                    drop_last=False,
                    collate_fn=graph_collate_fn,
                    **dataloader_runtime_kwargs(num_workers=config.num_workers, pin_memory=config.pin_memory),
                )

        sample_kind = "rooms" if config.room_level else "dungeons"
        logger.info(
            "Training samples: %d %s | internal_val=%d | final_test_dungeons=%s%s",
            len(train_dataset),
            sample_kind,
            len(val_dataset) if val_dataset is not None else 0,
            list(getattr(config, "test_dungeon_ids", [9])),
            f" (world_size={distributed_context.world_size})" if distributed_context.enabled else "",
        )
        logger.info(
            "Diffusion corpus split: train/internal-val dungeons=%s variants=%s",
            list(getattr(config, "train_dungeon_ids", list(range(1, 9)))),
            list(getattr(config, "variants", [1, 2])),
        )

        trainer = DiffusionTrainer(config, distributed_context=distributed_context)
        if getattr(trainer, "_accelerator", None) is not None:
            if val_loader is not None:
                train_loader, val_loader = trainer._accelerator.prepare(train_loader, val_loader)
            else:
                train_loader = trainer._accelerator.prepare(train_loader)
            logger.info("Accelerate prepared diffusion train/validation dataloaders.")
        if distributed_context.is_main_process:
            diffusion_trainable = count_parameters(trainer.diffusion, trainable_only=True)
            condition_trainable = count_parameters(trainer.condition_encoder, trainable_only=True)
            logic_subset = count_parameters(trainer.logic_net, trainable_only=True)
            logger.info(
                "LogicNet guidance parameters are optimized separately: logic_net=%s.",
                format_parameter_count(logic_subset),
            )
            log_capacity_guardrails(
                logger,
                stage_name="Diffusion trainer",
                dataset_size=len(train_dataset),
                param_groups={
                    "diffusion": diffusion_trainable,
                    "condition_encoder": condition_trainable,
                    "logic_net": logic_subset,
                },
                recommended_config="configs/zelda_hmolqd.yaml",
                capacity_knobs=(
                    "diffusion.model_channels, diffusion.condition_hidden_dim, "
                    "diffusion.condition_num_gnn_layers"
                ),
            )

        checkpoint_dir = Path(config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        resume_path = resolve_resume_checkpoint(
            explicit_path=getattr(config, "resume_checkpoint", None),
            checkpoint_dir=str(checkpoint_dir),
            auto_resume=bool(getattr(config, "auto_resume", True)),
            latest_filename=LATEST_RESUME_FILENAME,
        )
        if resume_path is not None:
            try:
                trainer.load_checkpoint(str(resume_path))
                logger.info("Auto-resumed diffusion training from %s", resume_path)
            except ValueError as exc:
                explicit_resume = bool(getattr(config, "resume_checkpoint", None))
                if explicit_resume:
                    raise
                logger.warning(
                    "Ignoring auto-resume diffusion checkpoint %s because it is not safe to resume: %s",
                    resume_path,
                    exc,
                )
                resume_path = None

        metrics_logger = (
            MetricsLogger(
                log_dir=str(checkpoint_dir / 'logs'),
                experiment_name='diffusion_training',
            )
            if distributed_context.is_main_process
            else None
        )

        best_solvability = 0.0
        best_teacher_loss = float("inf")
        metrics: Dict[str, float] = {}
        if resume_path is not None:
            latest_ckpt = safe_torch_load(str(resume_path), map_location="cpu")
            latest_metrics = latest_ckpt.get("metrics", {})
            if isinstance(latest_metrics, dict):
                best_solvability = float(latest_metrics.get("best_solvability", latest_metrics.get("val_solvability", 0.0)))
                best_teacher_loss = float(latest_metrics.get("best_teacher_loss", latest_metrics.get("val_total_loss", float("inf"))))

        for epoch in range(int(getattr(trainer, "epoch", 0)) + 1, config.epochs + 1):
            trainer.epoch = int(epoch)
            train_metrics = trainer.train_epoch(train_loader, sampler=train_sampler)
            maybe_barrier(distributed_context)

            if distributed_context.is_main_process:
                assert val_loader is not None
                val_metrics = trainer.validate(
                    val_loader,
                    num_samples=config.validation_num_samples,
                    num_diffusion_samples=config.validation_num_diffusion_samples,
                )
                metrics = {
                    'epoch': epoch,
                    'lr': trainer.scheduler.get_last_lr()[0],
                    **train_metrics,
                    **val_metrics,
                }
                assert metrics_logger is not None
                metrics_logger.log(metrics)

                logger.info(
                    f"Epoch {epoch}/{config.epochs}: "
                    f"loss={train_metrics['loss']:.4f}, "
                    f"diffusion={train_metrics['diffusion_loss']:.4f}, "
                    f"val_diffusion_loss={val_metrics.get('val_diffusion_loss', float('inf')):.4f}, "
                    f"val_logic_loss={val_metrics.get('val_logic_loss', 0.0):.4f}, "
                    f"val_total_loss={val_metrics.get('val_total_loss', float('inf')):.4f}, "
                    f"val_solvability_proxy={val_metrics.get('val_solvability_proxy', val_metrics['val_solvability']):.4f}, "
                    f"logic_loss_{'enabled' if epoch >= config.warmup_epochs and config.alpha_logic > 0 else 'disabled'}"
                )

                if epoch % config.save_every == 0:
                    trainer.save_checkpoint(
                        str(checkpoint_dir / f"resume_epoch_{epoch:04d}.pth"),
                        metrics,
                        include_optimizer=True,
                    )
                    prune_checkpoints(
                        checkpoint_dir=str(checkpoint_dir),
                        pattern="resume_epoch_*.pth",
                        keep_last=int(getattr(config, "keep_last", 2)),
                    )

                current_teacher_loss = float(val_metrics.get("val_total_loss", float("inf")))
                current_solvability = float(val_metrics['val_solvability'])
                is_better_teacher = (
                    current_teacher_loss < (best_teacher_loss - 1e-8)
                    or (
                        abs(current_teacher_loss - best_teacher_loss) <= 1e-8
                        and current_solvability > best_solvability
                    )
                )
                if is_better_teacher:
                    best_teacher_loss = current_teacher_loss
                    trainer.save_checkpoint(
                        str(checkpoint_dir / "best_model.pth"),
                        metrics,
                        include_optimizer=False,
                    )
                if current_solvability > best_solvability:
                    best_solvability = current_solvability
                    trainer.save_checkpoint(
                        str(checkpoint_dir / "best_logic_model.pth"),
                        metrics,
                        include_optimizer=False,
                    )

                latest_metrics = dict(metrics)
                latest_metrics["best_solvability"] = float(best_solvability)
                latest_metrics["best_teacher_loss"] = float(best_teacher_loss)
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
                    cleanup_target_fraction=float(
                        getattr(config, "checkpoint_storage_cleanup_target_fraction", 0.6)
                    ),
                    removable_patterns=("resume_epoch_*.pth",),
                )

            maybe_barrier(distributed_context)

        if distributed_context.is_main_process:
            trainer.save_checkpoint(str(checkpoint_dir / "final_model.pth"), metrics or None, include_optimizer=False)
            assert metrics_logger is not None
            metrics_logger.save()

        maybe_barrier(distributed_context)
        return trainer
    finally:
        destroy_distributed(distributed_context)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train Latent Diffusion for Dungeon Generation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Optional YAML config path using the shared validated config system. '
             'When provided, omitted legacy flags inherit values from that config.',
    )
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--room-level', dest='room_level', action='store_true', help='Train the diffusion model on individual room samples.')
    parser.add_argument('--dungeon-level', dest='room_level', action='store_false', help='Train the diffusion model on whole-dungeon samples.')
    parser.set_defaults(room_level=None)
    parser.add_argument('--dungeon-batch-mode', action=argparse.BooleanOptionalAction, default=None,
                        help='For room-level training, batch all rooms from one dungeon variant so global graph loss receives full node passability.')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--model-channels', type=int, default=None)
    parser.add_argument('--context-dim', type=int, default=None)
    parser.add_argument('--denoiser-backbone', type=str, default=None, choices=['unet', 'dit'])
    parser.add_argument('--unet-channel-mult', type=int, nargs='+', default=None)
    parser.add_argument('--unet-num-res-blocks', type=int, default=None)
    parser.add_argument('--unet-attention-resolutions', type=int, nargs='+', default=None)
    parser.add_argument('--unet-num-heads', type=int, default=None)
    parser.add_argument('--unet-dropout', type=float, default=None)
    parser.add_argument('--dit-depth', type=int, default=None)
    parser.add_argument('--dit-patch-size', type=int, default=None)
    parser.add_argument('--dit-mlp-ratio', type=float, default=None)
    parser.add_argument('--pag-scale', type=float, default=None)
    parser.add_argument('--alpha-logic', type=float, default=None)
    parser.add_argument('--alpha-logic-tile', type=float, default=None)
    parser.add_argument('--alpha-wfc-pseudo', type=float, default=None)
    parser.add_argument('--wfc-pseudo-max-samples', type=int, default=None)
    parser.add_argument('--wfc-pseudo-confidence-threshold', type=float, default=None)
    parser.add_argument('--min-logic-tile-accuracy-for-guidance', type=float, default=None)
    parser.add_argument('--graph-spatial-alignment-weight', type=float, default=None)
    parser.add_argument(
        '--logic-loss-mode',
        type=str,
        default=None,
        choices=['predicted_latent', 'detached_real'],
        help='Logic-loss target mode for A/B: predicted_latent (new) or detached_real (legacy).',
    )
    parser.add_argument(
        '--graph-conditioning-mode',
        type=str,
        default=None,
        choices=['node_sequence', 'pooled'],
        help='Graph conditioning for diffusion cross-attention: node_sequence (GCN node tokens) or pooled baseline.',
    )
    parser.add_argument(
        '--condition-gnn-type',
        type=str,
        default=None,
        choices=['gcn', 'gat', 'sage', 'gps'],
        help='GNN backbone for graph-node conditioning.',
    )
    parser.add_argument('--condition-use-reference-room-maps', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--condition-reference-tile-vocab-size', type=int, default=None)
    parser.add_argument('--condition-reference-embedding-dim', type=int, default=None)
    parser.add_argument('--condition-reference-hidden-dim', type=int, default=None)
    parser.add_argument('--condition-use-rrwp-edge-features', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--vqvae-hidden-dim', type=int, default=None)
    parser.add_argument('--vqvae-codebook-size', type=int, default=None)
    parser.add_argument('--vqvae-use-coordconv', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--vqvae-mrf-penalty-weight', type=float, default=None)
    parser.add_argument(
        '--topology-refinement-mode',
        type=str,
        default=None,
        choices=[
            'none', 'lightweight',
            'sparse_edge', 'sparse_directed', 'sparse_semantic', 'sparse_directed_semantic',
            'gat2', 'gat2_directed', 'gat2_semantic', 'gat2_directed_semantic',
            'graphormer', 'upgraded',
        ],
        help='Topology preprocessing inside diffusion cross-attention (gat2 is explicit 2-layer GAT).',
    )
    parser.add_argument(
        '--attention-mode',
        type=str,
        default=None,
        choices=['softmax', 'linear_hedgehog'],
        help='Cross-attention kernel used in diffusion graph conditioning.',
    )
    parser.add_argument(
        '--topology-conditioning-mode',
        type=str,
        default=None,
        choices=['additive', 'spade'],
        help='Room-topology conditioning path: additive bias or SPADE-style affine modulation.',
    )
    parser.add_argument('--hedgehog-feature-dim', type=int, default=None)
    parser.add_argument('--graph-auto-linear-attention-nodes', type=int, default=None)
    parser.add_argument('--spatial-graph-gate-init', type=float, default=None)
    parser.add_argument('--spatial-topology-gate-init', type=float, default=None)
    parser.add_argument('--use-teacher-forced-neighbor-latents', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--puzzle-structure-dropout-prob', type=float, default=None)
    parser.add_argument('--use-current-node-distance-features', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--current-node-distance-max', type=int, default=None)
    parser.add_argument(
        '--logic-net-enabled',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Enable LogicNet loss/guidance during diffusion training and validation.',
    )
    parser.add_argument(
        '--disable-logic-net',
        dest='logic_net_enabled',
        action='store_false',
        default=None,
        help='Train a clean no-LogicNet diffusion ablation checkpoint.',
    )
    parser.add_argument('--logic-net-trainable', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--logic-learning-rate', type=float, default=None)
    parser.add_argument('--logic-lr-warmup-epochs', type=int, default=None)
    parser.add_argument('--global-lr-warmup-epochs', type=int, default=None)
    parser.add_argument('--logic-loss-ramp-epochs', type=int, default=None)
    parser.add_argument(
        '--logic-grid-pathfinder',
        type=str,
        default=None,
        choices=[
            'cnn',
            'bellman_ford',
            'bellman-ford',
            'soft_bellman_ford',
            'soft-bellman-ford',
            'vin',
            'value_iteration',
            'value-iteration',
            'perturb_and_map',
            'perturb-and-map',
            'perturb_map',
            'pmap',
        ],
        help='Grid-level LogicNet pathfinder ablation: learned CNN, explicit soft Bellman-Ford, VIN, or Perturb-and-MAP straight-through.',
    )
    parser.add_argument('--logic-topology-trace-weight', type=float, default=None)
    parser.add_argument('--logic-topology-anchor-weight', type=float, default=None)
    parser.add_argument('--logic-global-reach-weight', type=float, default=None)
    parser.add_argument('--logic-global-room-weight', type=float, default=None)
    parser.add_argument(
        '--diffusion-training-objective',
        type=str,
        default=None,
        choices=['diffusion', 'flow_matching'],
        help='Latent training objective ablation. flow_matching requires DiT and validates/generates with the rectified-flow ODE sampler.',
    )
    parser.add_argument('--guidance-scale', type=float, default=None)
    parser.add_argument('--latent-cache-enabled', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--latent-cache-max-items', type=int, default=None)
    parser.add_argument('--checkpoint-dir', type=str, default=None)
    parser.add_argument('--keep-last', type=int, default=None)
    parser.add_argument('--auto-resume', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--checkpoint-storage-budget-gb', type=float, default=None)
    parser.add_argument('--checkpoint-storage-warning-fraction', type=float, default=None)
    parser.add_argument('--checkpoint-storage-cleanup-enabled', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--checkpoint-storage-cleanup-target-fraction', type=float, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--vqvae-checkpoint', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--distributed-enabled', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--distributed-backend', type=str, default=None, choices=['nccl', 'gloo'])
    parser.add_argument('--quick', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--verbose', '-v', action=argparse.BooleanOptionalAction, default=None)
    
    args = parser.parse_args()

    config = build_diffusion_training_config_from_args(args)

    log_level = logging.DEBUG if bool(getattr(args, "verbose", False)) else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S',
    )

    try:
        _trainer = train_diffusion(config)
        logger.info("Training complete!")
    except KeyboardInterrupt:
        logger.info("Training interrupted")
    except (AttributeError, RuntimeError, ValueError, TypeError) as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    main()

