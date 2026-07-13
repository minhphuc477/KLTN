"""Configuration contract and resolved-config bridge for diffusion training."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.config_system import merge_config
from src.core.definitions import (
    GRAPH_EDGE_FEATURE_DIM,
    GRAPH_NODE_FEATURE_DIM,
    ROOM_TOPOLOGY_CHANNEL_COUNT,
)
from src.core.puzzle_stage_semantics import (
    DEFAULT_PUZZLE_STAGE_MAX_SEQUENCE_LENGTH,
    DEFAULT_PUZZLE_STAGE_SEMANTICS_HIDDEN_DIM,
)
from src.pipeline.room_topology_conditioning import (
    DEFAULT_PUZZLE_STAGE_TOKEN_SCALE,
    DEFAULT_PUZZLE_STAGE_TRACE_DECAY,
    DEFAULT_SEMANTIC_PUZZLE_OFFSET,
    DEFAULT_SEMANTIC_ROLE_PRIOR_STRENGTH,
)
from src.zelda_data.splits import validate_disjoint_dungeon_splits
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
        latent_scale_factor: float = 1.0,
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
        dit_activation_type: str = "gelu",  # 'gelu' (default) | 'swiglu' (ablation)
        dit_norm_type: str = "layer",  # 'layer' (default) | 'rms' (ablation)
        condition_hidden_dim: int = 256,
        condition_num_gnn_layers: int = 3,
        condition_num_attention_heads: int = 8,
        condition_dropout: float = 0.1,
        condition_gnn_type: str = "gps",  # gcn | gat | sage | gps
        condition_use_reference_room_maps: bool = False,
        condition_reference_tile_vocab_size: int = 44,
        condition_reference_embedding_dim: int = 32,
        condition_reference_hidden_dim: int = 64,
        condition_use_rrwp_edge_features: bool = True,
        condition_strict_schema: bool = True,
        num_timesteps: int = 1000,
        schedule_type: str = "cosine",
        topology_refinement_mode: str = "gat2",  # none | lightweight | sparse*/gat2* | graphormer
        attention_mode: str = "softmax",
        topology_conditioning_mode: str = "additive",
        hedgehog_feature_dim: int = 32,
        graph_auto_linear_attention_nodes: int = 128,
        graphormer_max_distance: int = 16,
        graphormer_max_degree: int = 64,
        graph_to_grid_edge_semantics: bool = False,
        spatial_graph_gate_init: float = -2.0,
        spatial_topology_gate_init: float = -2.0,
        use_teacher_forced_neighbor_latents: bool = True,
        puzzle_structure_dropout_prob: float = 0.0,
        use_current_node_distance_features: bool = True,
        current_node_distance_max: int = 8,
        room_topology_channels: int = ROOM_TOPOLOGY_CHANNEL_COUNT,
        topology_supervision_mode: str = "runtime_aligned",
        semantic_role_prior_strength: float = DEFAULT_SEMANTIC_ROLE_PRIOR_STRENGTH,
        semantic_puzzle_offset: int = DEFAULT_SEMANTIC_PUZZLE_OFFSET,
        puzzle_stage_conditioning_enabled: bool = False,
        puzzle_stage_token_scale: float = DEFAULT_PUZZLE_STAGE_TOKEN_SCALE,
        puzzle_stage_topology_enabled: bool = False,
        puzzle_stage_trace_decay: float = DEFAULT_PUZZLE_STAGE_TRACE_DECAY,
        puzzle_stage_semantics_loss_weight: float = 0.0,
        puzzle_stage_semantics_hidden_dim: int = DEFAULT_PUZZLE_STAGE_SEMANTICS_HIDDEN_DIM,
        puzzle_stage_semantics_max_sequence_length: int = DEFAULT_PUZZLE_STAGE_MAX_SEQUENCE_LENGTH,
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
        logic_graph_pathfinder: str = "dense_bellman_ford",
        logic_resource_gate_mode: str = "hard_ordered",
        logic_full_coverage: bool = True,
        num_logic_iterations: int = 30,
        logic_initial_temperature: float = 1.0,
        logic_final_temperature: float = 0.05,
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
        train_ids, test_ids = validate_disjoint_dungeon_splits(
            train_dungeon_ids if train_dungeon_ids is not None else range(1, 9),
            test_dungeon_ids if test_dungeon_ids is not None else (9,),
        )
        self.train_dungeon_ids = list(train_ids)
        self.test_dungeon_ids = list(test_ids)
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
        self.latent_scale_factor = float(latent_scale_factor)
        if not math.isfinite(self.latent_scale_factor) or self.latent_scale_factor <= 0.0:
            raise ValueError("latent_scale_factor must be finite and greater than zero.")
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
        self.dit_activation_type = str(dit_activation_type).strip().lower()
        self.dit_norm_type = str(dit_norm_type).strip().lower()
        if self.dit_activation_type not in {"gelu", "swiglu"}:
            raise ValueError(f"dit_activation_type must be 'gelu' or 'swiglu', got {dit_activation_type!r}.")
        if self.dit_norm_type not in {"layer", "rms"}:
            raise ValueError(f"dit_norm_type must be 'layer' or 'rms', got {dit_norm_type!r}.")
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
        self.condition_strict_schema = bool(condition_strict_schema)
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type
        trm = str(topology_refinement_mode).strip().lower()
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
            "graphormer_learned",
            "graphormer_learned_directed",
            "graphormer_learned_semantic",
            "graphormer_learned_directed_semantic",
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
        self.graphormer_max_distance = int(max(1, graphormer_max_distance))
        self.graphormer_max_degree = int(max(1, graphormer_max_degree))
        self.graph_to_grid_edge_semantics = bool(graph_to_grid_edge_semantics)
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
        self.puzzle_stage_conditioning_enabled = bool(puzzle_stage_conditioning_enabled)
        self.puzzle_stage_token_scale = float(max(0.0, puzzle_stage_token_scale))
        self.puzzle_stage_topology_enabled = bool(puzzle_stage_topology_enabled)
        self.puzzle_stage_trace_decay = float(max(0.05, min(1.0, puzzle_stage_trace_decay)))
        self.puzzle_stage_semantics_loss_weight = float(max(0.0, puzzle_stage_semantics_loss_weight))
        self.puzzle_stage_semantics_hidden_dim = int(max(16, puzzle_stage_semantics_hidden_dim))
        self.puzzle_stage_semantics_max_sequence_length = int(
            max(1, puzzle_stage_semantics_max_sequence_length)
        )
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
        self.logic_graph_pathfinder = str(logic_graph_pathfinder).strip().lower()
        if self.logic_graph_pathfinder not in {"dense_bellman_ford", "sparse_bellman_ford"}:
            raise ValueError(
                "logic_graph_pathfinder must be 'dense_bellman_ford' or 'sparse_bellman_ford'."
            )
        self.logic_resource_gate_mode = str(logic_resource_gate_mode).strip().lower()
        if self.logic_resource_gate_mode not in {"hard_ordered", "soft_ordered"}:
            raise ValueError("logic_resource_gate_mode must be 'hard_ordered' or 'soft_ordered'.")
        if self.logic_grid_pathfinder in {"bellman-ford", "soft_bellman_ford", "soft-bellman-ford"}:
            self.logic_grid_pathfinder = "bellman_ford"
        if self.logic_grid_pathfinder in {"value_iteration", "value-iteration"}:
            self.logic_grid_pathfinder = "vin"
        if self.logic_grid_pathfinder in {"perturb-and-map", "perturb_map", "pmap"}:
            self.logic_grid_pathfinder = "perturb_and_map"
        if self.logic_grid_pathfinder not in {"cnn", "bellman_ford", "vin", "perturb_and_map"}:
            raise ValueError("logic_grid_pathfinder must be 'cnn', 'bellman_ford', 'vin', or 'perturb_and_map'.")
        self.logic_full_coverage = bool(logic_full_coverage)
        self.num_logic_iterations = int(max(1, num_logic_iterations))
        self.logic_initial_temperature = float(logic_initial_temperature)
        self.logic_final_temperature = float(logic_final_temperature)
        for name in ("logic_initial_temperature", "logic_final_temperature"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and greater than zero.")
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
        "latent_scale_factor": stage.get("latent_scale_factor", 1.0),
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
        "dit_activation_type": stage.get("dit_activation_type", "gelu"),
        "dit_norm_type": stage.get("dit_norm_type", "layer"),
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
        "condition_strict_schema": stage.get("condition_strict_schema", True),
        "num_timesteps": stage["num_timesteps"],
        "schedule_type": stage["schedule_type"],
        "topology_refinement_mode": stage["topology_refinement_mode"],
        "attention_mode": stage["attention_mode"],
        "topology_conditioning_mode": stage["topology_conditioning_mode"],
        "hedgehog_feature_dim": stage["hedgehog_feature_dim"],
        "graph_auto_linear_attention_nodes": stage["graph_auto_linear_attention_nodes"],
        "graphormer_max_distance": stage["graphormer_max_distance"],
        "graphormer_max_degree": stage["graphormer_max_degree"],
        "graph_to_grid_edge_semantics": stage.get("graph_to_grid_edge_semantics", False),
        "spatial_graph_gate_init": stage["spatial_graph_gate_init"],
        "spatial_topology_gate_init": stage["spatial_topology_gate_init"],
        "use_teacher_forced_neighbor_latents": stage["use_teacher_forced_neighbor_latents"],
        "puzzle_structure_dropout_prob": stage.get("puzzle_structure_dropout_prob", 0.0),
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
        "logic_graph_pathfinder": stage.get("logic_graph_pathfinder", "dense_bellman_ford"),
        "logic_resource_gate_mode": stage.get("logic_resource_gate_mode", "hard_ordered"),
        "logic_full_coverage": stage.get("logic_full_coverage", True),
        "num_logic_iterations": stage["num_logic_iterations"],
        "logic_initial_temperature": stage.get("logic_initial_temperature", 1.0),
        "logic_final_temperature": stage.get("logic_final_temperature", 0.05),
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
    _set("gradient_accumulation_steps", getattr(args, "gradient_accumulation_steps", None))
    _set("gradient_checkpointing", getattr(args, "gradient_checkpointing", None))
    _set("use_amp", getattr(args, "use_amp", None))
    _set("amp_mixed_precision", getattr(args, "amp_mixed_precision", None))
    _set("use_accelerate", getattr(args, "use_accelerate", None))
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
    _set("dit_activation_type", getattr(args, "dit_activation_type", None))
    _set("dit_norm_type", getattr(args, "dit_norm_type", None))
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
    _set(
        "condition_strict_schema",
        getattr(args, "condition_strict_schema", None),
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
    _set("graphormer_max_distance", getattr(args, "graphormer_max_distance", None))
    _set("graphormer_max_degree", getattr(args, "graphormer_max_degree", None))
    _set("graph_to_grid_edge_semantics", getattr(args, "graph_to_grid_edge_semantics", None))
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
    _set("logic_graph_pathfinder", getattr(args, "logic_graph_pathfinder", None))
    _set("logic_resource_gate_mode", getattr(args, "logic_resource_gate_mode", None))
    _set("logic_full_coverage", getattr(args, "logic_full_coverage", None))
    _set("num_logic_iterations", getattr(args, "num_logic_iterations", None))
    _set("logic_initial_temperature", getattr(args, "logic_initial_temperature", None))
    _set("logic_final_temperature", getattr(args, "logic_final_temperature", None))
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


