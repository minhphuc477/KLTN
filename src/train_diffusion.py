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
import json
import logging
import math
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.zelda_data.zelda_loader import create_dataloader, extract_start_goal
from src.core.latent_diffusion import LatentDiffusionModel, create_latent_diffusion
from src.core.vqvae import SemanticVQVAE as VQVAE, create_vqvae
from src.core.condition_encoder import DualStreamConditionEncoder, create_condition_encoder
from src.core.definitions import (
    GRAPH_EDGE_FEATURE_DIM,
    GRAPH_NODE_FEATURE_DIM,
    ROOM_HEIGHT,
    ROOM_TOPOLOGY_CHANNEL_COUNT,
    ROOM_WIDTH,
)
# Use Block V LogicNet (with temperature annealing), not legacy src.ml.logic_net
from src.core.logic_net import LogicNet
from src.pipeline.graph_features import (
    align_nodewise_tensor,
    build_default_node_positions,
    compute_current_node_distance_features,
    compute_rwse_features,
)
from src.pipeline.room_topology_conditioning import (
    DEFAULT_SEMANTIC_PUZZLE_OFFSET,
    DEFAULT_SEMANTIC_ROLE_PRIOR_STRENGTH,
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
    write_checkpoint_metadata,
)
from src.utils.distributed import (
    DistributedContext,
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

logger = logging.getLogger(__name__)
CARDINAL_DIRECTIONS = ("N", "S", "E", "W")


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

class DiffusionTrainingConfig:
    """Training configuration for latent diffusion."""
    
    def __init__(
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
        num_classes: int = 44,
        node_feature_dim: int = GRAPH_NODE_FEATURE_DIM,
        edge_feature_dim: int = GRAPH_EDGE_FEATURE_DIM,
        
        # VQ-VAE (frozen encoder)
        vqvae_checkpoint: Optional[str] = None,
        vqvae_hidden_dim: int = 96,
        vqvae_codebook_size: int = 256,
        vqvae_use_coordconv: bool = True,
        vqvae_mrf_penalty_weight: float = 0.05,
        
        # Diffusion Model
        latent_dim: int = 64,
        model_channels: int = 128,
        context_dim: int = 256,
        unet_channel_mult: Tuple[int, ...] = (1, 2, 4),
        unet_num_res_blocks: int = 2,
        unet_attention_resolutions: Tuple[int, ...] = (1, 2),
        unet_num_heads: int = 8,
        unet_dropout: float = 0.1,
        condition_hidden_dim: int = 256,
        condition_num_gnn_layers: int = 3,
        condition_num_attention_heads: int = 8,
        condition_dropout: float = 0.1,
        condition_gnn_type: str = "gcn",  # gcn | gat | sage | gps
        condition_use_reference_room_maps: bool = False,
        condition_reference_tile_vocab_size: int = 44,
        condition_reference_embedding_dim: int = 32,
        condition_reference_hidden_dim: int = 64,
        num_timesteps: int = 1000,
        schedule_type: str = "cosine",
        topology_refinement_mode: str = "gat2",  # none | lightweight | gat2
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
        prediction_type: str = "epsilon",
        min_snr_gamma: float = 5.0,
        
        # LogicNet
        num_logic_iterations: int = 30,
        logic_topology_trace_weight: float = 0.25,
        logic_topology_anchor_weight: float = 0.25,
        guidance_scale: float = 1.0,
        guidance_clamp_magnitude: float = 1.0,
        guidance_relative_norm_cap: float = 0.25,
        guidance_schedule_enabled: bool = True,
        guidance_active_fraction: float = 0.30,
        guidance_decay_power: float = 1.0,
        guidance_max_graph_nodes: int = 512,
        guidance_max_key_lock_pairs: int = 2048,
        guidance_max_guidance_elements: int = 2_000_000,
        
        # Training
        epochs: int = 100,
        learning_rate: float = 1e-4,
        optimizer_weight_decay: float = 1e-5,
        alpha_visual: float = 1.0,   # Diffusion loss weight
        alpha_logic: float = 0.1,     # Solvability loss weight
        logic_loss_mode: str = "predicted_latent",  # predicted_latent | detached_real
        graph_conditioning_mode: str = "node_sequence",  # node_sequence | pooled
        warmup_epochs: int = 5,       # Epochs before adding logic loss
        scheduler_t0: int = 10,
        scheduler_t_mult: int = 2,
        scheduler_eta_min: float = 1e-6,
        ema_decay: float = 0.9999,
        grad_clip_norm: float = 1.0,
        validation_num_samples: int = 8,
        validation_num_diffusion_samples: int = 64,
        
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
        self.num_classes = int(num_classes)
        self.node_feature_dim = int(max(1, node_feature_dim))
        self.edge_feature_dim = int(max(1, edge_feature_dim))
        
        self.vqvae_checkpoint = vqvae_checkpoint
        self.vqvae_hidden_dim = int(max(8, vqvae_hidden_dim))
        self.vqvae_codebook_size = int(max(8, vqvae_codebook_size))
        self.vqvae_use_coordconv = bool(vqvae_use_coordconv)
        self.vqvae_mrf_penalty_weight = float(max(0.0, vqvae_mrf_penalty_weight))
        
        self.latent_dim = latent_dim
        self.model_channels = int(model_channels)
        self.context_dim = int(context_dim)

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
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type
        trm = str(topology_refinement_mode).strip().lower()
        if trm == "upgraded":
            trm = "gat2"
        if trm not in {"none", "lightweight", "gat2"}:
            raise ValueError(
                f"Invalid topology_refinement_mode={topology_refinement_mode!r}. "
                "Expected 'none', 'lightweight', or 'gat2'."
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
        self.prediction_type = str(prediction_type).strip().lower()
        self.min_snr_gamma = float(max(0.0, min_snr_gamma))
        
        self.num_logic_iterations = num_logic_iterations
        self.logic_topology_trace_weight = float(max(0.0, logic_topology_trace_weight))
        self.logic_topology_anchor_weight = float(max(0.0, logic_topology_anchor_weight))
        self.guidance_scale = guidance_scale
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
        self.alpha_visual = alpha_visual
        self.alpha_logic = alpha_logic
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
        self.scheduler_t0 = int(max(1, scheduler_t0))
        self.scheduler_t_mult = int(max(1, scheduler_t_mult))
        self.scheduler_eta_min = float(max(0.0, scheduler_eta_min))
        self.ema_decay = float(min(0.999999, max(0.0, ema_decay)))
        self.grad_clip_norm = float(max(0.0, grad_clip_norm))
        self.validation_num_samples = int(max(1, validation_num_samples))
        self.validation_num_diffusion_samples = int(max(1, validation_num_diffusion_samples))
        
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
        "num_classes": dataset["num_classes"],
        "node_feature_dim": dataset["node_feature_dim"],
        "edge_feature_dim": dataset["edge_feature_dim"],
        "vqvae_checkpoint": ckpt_path,
        "vqvae_hidden_dim": vqvae_stage["hidden_dim"],
        "vqvae_codebook_size": vqvae_stage["codebook_size"],
        "vqvae_use_coordconv": vqvae_stage["use_coordconv"],
        "vqvae_mrf_penalty_weight": vqvae_stage["mrf_penalty_weight"],
        "latent_dim": stage["latent_dim"],
        "model_channels": stage["model_channels"],
        "context_dim": stage["context_dim"],
        "unet_channel_mult": tuple(stage["unet_channel_mult"]),
        "unet_num_res_blocks": stage["unet_num_res_blocks"],
        "unet_attention_resolutions": tuple(stage["unet_attention_resolutions"]),
        "unet_num_heads": stage["unet_num_heads"],
        "unet_dropout": stage["unet_dropout"],
        "condition_hidden_dim": stage["condition_hidden_dim"],
        "condition_num_gnn_layers": stage["condition_num_gnn_layers"],
        "condition_num_attention_heads": stage["condition_num_attention_heads"],
        "condition_dropout": stage["condition_dropout"],
        "condition_gnn_type": stage["condition_gnn_type"],
        "condition_use_reference_room_maps": stage["condition_use_reference_room_maps"],
        "condition_reference_tile_vocab_size": stage["condition_reference_tile_vocab_size"],
        "condition_reference_embedding_dim": stage["condition_reference_embedding_dim"],
        "condition_reference_hidden_dim": stage["condition_reference_hidden_dim"],
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
        "prediction_type": stage["prediction_type"],
        "min_snr_gamma": stage["min_snr_gamma"],
        "num_logic_iterations": stage["num_logic_iterations"],
        "logic_topology_trace_weight": stage["logic_topology_trace_weight"],
        "logic_topology_anchor_weight": stage["logic_topology_anchor_weight"],
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
        "alpha_visual": stage["alpha_visual"],
        "alpha_logic": stage["alpha_logic"],
        "logic_loss_mode": stage["logic_loss_mode"],
        "graph_conditioning_mode": stage["graph_conditioning_mode"],
        "warmup_epochs": stage["warmup_epochs"],
        "scheduler_t0": stage["scheduler_t0"],
        "scheduler_t_mult": stage["scheduler_t_mult"],
        "scheduler_eta_min": stage["scheduler_eta_min"],
        "ema_decay": stage["ema_decay"],
        "grad_clip_norm": stage["grad_clip_norm"],
        "validation_num_samples": stage["validation_num_samples"],
        "validation_num_diffusion_samples": stage["validation_num_diffusion_samples"],
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


def _resolve_vqvae_architecture(
    checkpoint_path: Optional[str],
    *,
    num_classes: int,
    latent_dim: int,
    hidden_dim: int,
    codebook_size: int,
    use_coordconv: bool,
    mrf_penalty_weight: float,
) -> Dict[str, Any]:
    """
    Resolve the VQ-VAE architecture from config first, then checkpoint metadata.

    This keeps stage handoffs compatible when the trained VQ-VAE shape differs
    from historical diffusion defaults.
    """
    resolved: Dict[str, Any] = {
        "num_classes": int(num_classes),
        "latent_dim": int(latent_dim),
        "hidden_dim": int(hidden_dim),
        "codebook_size": int(codebook_size),
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
            "num_classes",
            "latent_dim",
            "hidden_dim",
            "codebook_size",
            "use_coordconv",
            "mrf_penalty_weight",
        ):
            if key in architecture and architecture[key] is not None:
                resolved[key] = architecture[key]

    return {
        "num_classes": int(resolved["num_classes"]),
        "latent_dim": int(resolved["latent_dim"]),
        "hidden_dim": int(resolved["hidden_dim"]),
        "codebook_size": int(resolved["codebook_size"]),
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
    _set("epochs", getattr(args, "epochs", None))
    _set("learning_rate", getattr(args, "lr", None))
    _set("model_channels", getattr(args, "model_channels", None))
    _set("context_dim", getattr(args, "context_dim", None))
    _set("unet_channel_mult", getattr(args, "unet_channel_mult", None), transform=tuple)
    _set("unet_num_res_blocks", getattr(args, "unet_num_res_blocks", None))
    _set(
        "unet_attention_resolutions",
        getattr(args, "unet_attention_resolutions", None),
        transform=tuple,
    )
    _set("unet_num_heads", getattr(args, "unet_num_heads", None))
    _set("unet_dropout", getattr(args, "unet_dropout", None))
    _set("alpha_logic", getattr(args, "alpha_logic", None))
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
    _set("guidance_scale", getattr(args, "guidance_scale", None))
    _set("logic_topology_trace_weight", getattr(args, "logic_topology_trace_weight", None))
    _set("logic_topology_anchor_weight", getattr(args, "logic_topology_anchor_weight", None))
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
        self.condition_encoder = condition_encoder or self._create_condition_encoder()
        self.logic_net = logic_net or self._create_logic_net()
        
        # Move to device
        self.vqvae = self.vqvae.to(self.device)
        self.diffusion = self.diffusion.to(self.device)
        self.condition_encoder = self.condition_encoder.to(self.device)
        self.logic_net = self.logic_net.to(self.device)
        
        # Freeze VQ-VAE
        self.vqvae.eval()
        for param in self.vqvae.parameters():
            param.requires_grad = False
        
        # --- Wire LogicNet into diffusion model's GradientGuidance ---
        # This enables gradient guidance during sampling: at each denoising
        # step, âˆ‡_{x_t}L_logic nudges the sample toward solvable configs.
        self.diffusion.guidance.logic_net = self.logic_net
        self.diffusion.guidance.guidance_scale = config.guidance_scale
        self.diffusion.guidance.clamp_magnitude = float(config.guidance_clamp_magnitude)
        self.diffusion.guidance.relative_norm_cap = float(config.guidance_relative_norm_cap)
        self.diffusion.guidance.schedule_enabled = bool(config.guidance_schedule_enabled)
        self.diffusion.guidance.active_fraction = float(config.guidance_active_fraction)
        self.diffusion.guidance.decay_power = float(config.guidance_decay_power)
        self.diffusion.guidance.max_graph_nodes = int(config.guidance_max_graph_nodes)
        self.diffusion.guidance.max_key_lock_pairs = int(config.guidance_max_key_lock_pairs)
        self.diffusion.guidance.max_guidance_elements = int(config.guidance_max_guidance_elements)
        
        # Setup optimizer: train diffusion + condition encoder
        # Note: LogicNet is now a submodule of diffusion.guidance, so its
        # parameters are already included in self.diffusion.parameters().
        self.optimizer = optim.AdamW(
            list(self.diffusion.parameters()) + 
            list(self.condition_encoder.parameters()),
            lr=config.learning_rate,
            weight_decay=config.optimizer_weight_decay,
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
        
        # --- Phase 4A: EMA model weights ---
        import copy
        self.ema_diffusion = copy.deepcopy(self.diffusion)
        self.ema_diffusion.eval()
        for param in self.ema_diffusion.parameters():
            param.requires_grad = False
        self.ema_decay = float(config.ema_decay)
    
    def _create_vqvae(self) -> VQVAE:
        """Create or load VQ-VAE."""
        vqvae_arch = _resolve_vqvae_architecture(
            self.config.vqvae_checkpoint,
            num_classes=self.config.num_classes,
            latent_dim=self.config.latent_dim,
            hidden_dim=self.config.vqvae_hidden_dim,
            codebook_size=self.config.vqvae_codebook_size,
            use_coordconv=self.config.vqvae_use_coordconv,
            mrf_penalty_weight=self.config.vqvae_mrf_penalty_weight,
        )
        self.config.vqvae_hidden_dim = int(vqvae_arch["hidden_dim"])
        self.config.vqvae_codebook_size = int(vqvae_arch["codebook_size"])
        self.config.vqvae_use_coordconv = bool(vqvae_arch["use_coordconv"])
        self.config.vqvae_mrf_penalty_weight = float(vqvae_arch["mrf_penalty_weight"])
        vqvae = create_vqvae(
            num_classes=vqvae_arch["num_classes"],
            latent_dim=vqvae_arch["latent_dim"],
            hidden_dim=vqvae_arch["hidden_dim"],
            codebook_size=vqvae_arch["codebook_size"],
            use_coordconv=vqvae_arch["use_coordconv"],
            mrf_penalty_weight=vqvae_arch["mrf_penalty_weight"],
        )
        
        if self.config.vqvae_checkpoint:
            checkpoint = torch.load(self.config.vqvae_checkpoint, map_location='cpu')
            vqvae.load_state_dict(checkpoint['model_state_dict'])
            logger.info(
                "Loaded VQ-VAE from %s with architecture num_classes=%d latent_dim=%d hidden_dim=%d codebook_size=%d",
                self.config.vqvae_checkpoint,
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
            unet_channel_mult=self.config.unet_channel_mult,
            unet_num_res_blocks=self.config.unet_num_res_blocks,
            unet_attention_resolutions=self.config.unet_attention_resolutions,
            unet_num_heads=self.config.unet_num_heads,
            unet_dropout=self.config.unet_dropout,
            num_timesteps=self.config.num_timesteps,
            schedule_type=self.config.schedule_type,
            prediction_type=self.config.prediction_type,
            cfg_dropout_prob=self.config.cfg_dropout_prob,
            cfg_scale=self.config.cfg_scale,
            cfg_schedule_mode=self.config.cfg_schedule_mode,
            cfg_schedule_min_scale=self.config.cfg_schedule_min_scale,
            cfg_schedule_power=self.config.cfg_schedule_power,
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
            topology_trace_weight=self.config.logic_topology_trace_weight,
            topology_anchor_weight=self.config.logic_topology_anchor_weight,
        )
    
    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
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
            if bool(getattr(self.config, "use_teacher_forced_neighbor_latents", True)):
                neighbor_latents = self._encode_neighbor_maps_to_latents(graph_dict.get("neighbor_maps"))
            else:
                neighbor_latents = {direction: None for direction in CARDINAL_DIRECTIONS}
            reference_room_maps = (
                graph_dict.get("neighbor_maps")
                if bool(getattr(self.config, "condition_use_reference_room_maps", False))
                else None
            )
            condition_out = self.condition_encoder(
                neighbor_latents=neighbor_latents,
                boundary_constraints=boundary_constraints,
                position=room_position,
                node_features=node_features,
                edge_index=edge_index,
                edge_features=edge_features,
                tpe=tpe,
                current_node_distance=current_node_distance,
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
            return conditioning_out

        c_global = self.condition_encoder.encode_global_only(
            node_features, edge_index,
            edge_features=edge_features,
            tpe=tpe,
            current_node_distance=current_node_distance,
        )

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
            return conditioning_out

        # Pooled baseline.
        conditioning_out = c_global.mean(dim=0, keepdim=True)
        if float(getattr(self.config, "puzzle_structure_dropout_prob", 0.0)) > 0.0:
            conditioning_out = apply_puzzle_structure_control_to_conditioning(
                conditioning_out,
                puzzle_structure_enabled=bool(graph_dict.get("puzzle_room_structure_enabled", True)),
                graph_conditioning_mode=self.config.graph_conditioning_mode,
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
        current_node_idx = graph_dict.get("current_node_idx")
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
            "tpe": tpe,
            "current_node_distance": current_node_distance,
            "node_positions": node_positions,
            "node_mask": node_mask,
            "has_room_anchor": has_room_anchor,
            **({"boundary_constraints": boundary_constraints} if isinstance(boundary_constraints, torch.Tensor) else {}),
            **({"room_topology_map": room_topology_map} if isinstance(room_topology_map, torch.Tensor) else {}),
        }

    def _stack_diffusion_graph_batch(self, graph_list: List[dict]) -> Optional[Dict[str, torch.Tensor]]:
        """Pad a batch of variable-size graph tensors for diffusion conditioning."""
        if not graph_list:
            return None

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

        node_features_batch = torch.zeros(len(samples), max_nodes, max(1, feat_dim), device=self.device, dtype=torch.float32)
        tpe_batch = torch.zeros(len(samples), max_nodes, max(1, tpe_dim), device=self.device, dtype=torch.float32)
        current_node_distance_batch = torch.zeros(len(samples), max_nodes, max(1, distance_dim), device=self.device, dtype=torch.float32)
        node_positions_batch = torch.zeros(len(samples), max_nodes, max(1, pos_dim), device=self.device, dtype=torch.float32)
        node_mask_batch = torch.zeros(len(samples), max_nodes, device=self.device, dtype=torch.float32)
        edge_index_batch = torch.full((len(samples), 2, max_edges), -1, device=self.device, dtype=torch.long)

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
            "tpe": tpe_batch,
            "current_node_distance": current_node_distance_batch,
            "node_positions": node_positions_batch,
            "node_mask": node_mask_batch,
            "has_room_anchor": bool(self.config.graph_conditioning_mode == "node_sequence") or (
                bool(next(iter(anchor_flags))) if anchor_flags else False
            ),
        }
        if can_stack_topology and topo_maps:
            batch_graph["room_topology_map"] = torch.cat(topo_maps, dim=0)
        if boundary_batch is not None:
            batch_graph["boundary_constraints"] = boundary_batch
        return batch_graph
    
    def _build_logic_graph_data(
        self,
        graph_dict: dict,
    ) -> dict:
        """
        Convert a dataset graph_dict to LogicNet's expected format.
        
        LogicNet expects:
            adjacency: [N, N] adjacency matrix
            edge_weights: [N, N] traversal costs (1.0 for open, 2.0 for locked)
            start_idx: int
            target_idx: int (triforce room)
            key_lock_pairs: List[(key_node, lock_target)]
            
        Args:
            graph_dict: Dict from zelda_loader
            
        Returns:
            Dict for LogicNet.forward() or None if graph is empty
        """
        num_nodes = graph_dict['num_nodes']
        if num_nodes == 0:
            return None
        
        edge_index = graph_dict['edge_index']
        edge_attr = graph_dict.get('edge_attr')
        node_features = graph_dict['node_features']
        
        # Build adjacency matrix
        adjacency = torch.zeros(num_nodes, num_nodes, device=self.device)
        edge_weights = torch.zeros(num_nodes, num_nodes, device=self.device)
        
        key_lock_pairs = []
        
        if edge_index.numel() > 0:
            edge_index_dev = edge_index.to(self.device)
            src_indices = edge_index_dev[0]
            dst_indices = edge_index_dev[1]
            
            # Vectorized adjacency construction
            adjacency[src_indices, dst_indices] = 1.0
            
            # Vectorized edge weights: default=1.0, key_locked=2.0, boss_locked=3.0
            edge_weights[src_indices, dst_indices] = 1.0
            if edge_attr is not None:
                edge_attr_dev = edge_attr.to(self.device) if isinstance(edge_attr, torch.Tensor) else torch.tensor(edge_attr, device=self.device)
                num_edges = min(len(edge_attr_dev), edge_index_dev.shape[1])
                key_locked_mask = edge_attr_dev[:num_edges] == 1
                boss_locked_mask = edge_attr_dev[:num_edges] == 4
                if key_locked_mask.any():
                    edge_weights[src_indices[:num_edges][key_locked_mask], dst_indices[:num_edges][key_locked_mask]] = 2.0
                if boss_locked_mask.any():
                    edge_weights[src_indices[:num_edges][boss_locked_mask], dst_indices[:num_edges][boss_locked_mask]] = 3.0
        
        # Find start and target nodes from node_features
        start_idx = graph_dict.get('start_node_id', 0)
        if start_idx < 0:
            start_idx = 0
        
        target_idx = None
        node_feats = node_features if isinstance(node_features, torch.Tensor) else torch.tensor(node_features)
        for i in range(num_nodes):
            # has_triforce is feature[3]
            if node_feats[i, 3] > 0.5:
                target_idx = i
                break
        
        # Find key-lock pairs:
        # Key nodes (feature[1] = has_key) should be reachable before locked doors
        key_nodes = [i for i in range(num_nodes) if node_feats[i, 1] > 0.5]
        # Lock targets: rooms behind key-locked edges
        lock_targets = set()
        if edge_index.numel() > 0 and edge_attr is not None:
            for e in range(edge_index.shape[1]):
                if edge_attr[e].item() == 1:  # key_locked
                    lock_targets.add(edge_index[1, e].item())
        
        # Pair keys to locks (simple: pair by order)
        lock_list = sorted(lock_targets)
        for i, key_node in enumerate(key_nodes):
            if i < len(lock_list):
                key_lock_pairs.append((key_node, lock_list[i]))
        
        return {
            'adjacency': adjacency,
            'edge_weights': edge_weights,
            'start_idx': start_idx,
            'target_idx': target_idx,
            'key_lock_pairs': key_lock_pairs,
        }

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
        for module in (self.diffusion, self.condition_encoder):
            for param in module.parameters():
                if param.grad is None:
                    continue
                if not bool(torch.isfinite(param.grad).all()):
                    return False
        return True
    
    def train_step(
        self,
        real_maps: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
        include_logic_loss: bool = True,
        logic_graph_data: Optional[dict] = None,
        diffusion_graph_data: Optional[dict] = None,
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
            logic_graph_data: Graph data dict for LogicNet (from _build_logic_graph_data)
            
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
        
        # === Part 1: Diffusion loss (standard noise prediction) ===
        diffusion_loss = self.diffusion.training_loss(z_0, conditioning, graph_data=diffusion_graph_data)
        if not self._tensor_is_finite(diffusion_loss):
            self.optimizer.zero_grad(set_to_none=True)
            self._warn_nonfinite(
                "diffusion_loss",
                "Diffusion training: non-finite diffusion loss detected; skipping optimizer step for this batch.",
            )
            self.global_step += 1
            return {
                'loss': 0.0,
                'diffusion_loss': 0.0,
                'logic_loss': 0.0,
                'solvability_proxy': 0.0,
                'solvability': 0.0,
                'logic_loss_mode_predicted': 1.0 if self.config.logic_loss_mode == 'predicted_latent' else 0.0,
                'skipped_nonfinite_batch': 1.0,
            }
        
        # === Part 2: LogicNet loss on model-predicted latent WITH graph topology ===
        # IMPORTANT: computing logic loss on detached real z_0 does not train diffusion.
        # We instead denoise a noisy latent and apply LogicNet to predicted x0 so
        # logic gradients flow into diffusion + condition encoder.
        logic_loss = torch.tensor(0.0, device=self.device)
        solvability_proxy = torch.tensor(0.0, device=self.device)
        
        if include_logic_loss and self.config.alpha_logic > 0:
            if self.config.logic_loss_mode == "detached_real":
                # Legacy baseline: logic regularization on real latent only.
                z_for_logic = z_0.detach().requires_grad_(True)
                logic_loss, _logic_info = self.logic_net(z_for_logic, graph_data=logic_graph_data)
            else:
                # New default: logic supervision on predicted latent (trains diffusion).
                t_logic = torch.randint(0, self.diffusion.num_timesteps, (batch_size,), device=self.device)
                noise_logic = torch.randn_like(z_0)
                x_t_logic = self.diffusion.q_sample(z_0, t_logic, noise_logic)

                # Predict noise/velocity and convert to predicted clean latent x0.
                context_edge_index, context_node_mask = self.diffusion._extract_context_topology(
                    conditioning,
                    diffusion_graph_data,
                )
                spatial_graph_data = self.diffusion._extract_spatial_graph_context(
                    conditioning,
                    diffusion_graph_data,
                )
                pred_logic = self.diffusion.denoiser(
                    x_t_logic,
                    t_logic,
                    conditioning,
                    context_edge_index=context_edge_index,
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

                # Pass predicted latent to LogicNet for graph-level pathfinding loss.
                if pred_x0_logic is not None:
                    logic_loss, _logic_info = self.logic_net(pred_x0_logic, graph_data=logic_graph_data)

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
            self.config.alpha_logic * logic_loss
        )
        if not self._tensor_is_finite(total_loss):
            self.optimizer.zero_grad(set_to_none=True)
            self._warn_nonfinite(
                "total_loss",
                "Diffusion training: non-finite total loss detected; skipping optimizer step for this batch.",
            )
            self.global_step += 1
            return {
                'loss': 0.0,
                'diffusion_loss': float(diffusion_loss.detach().item()) if self._tensor_is_finite(diffusion_loss) else 0.0,
                'logic_loss': 0.0,
                'solvability_proxy': 0.0,
                'solvability': 0.0,
                'logic_loss_mode_predicted': 1.0 if self.config.logic_loss_mode == 'predicted_latent' else 0.0,
                'skipped_nonfinite_batch': 1.0,
            }
        
        # Backward
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        average_gradients(
            (self.diffusion, self.condition_encoder),
            context=getattr(self, "distributed_context", None),
        )
        if not self._gradients_are_finite():
            self.optimizer.zero_grad(set_to_none=True)
            self._warn_nonfinite(
                "gradient",
                "Diffusion training: non-finite gradients detected; skipping optimizer step for this batch.",
            )
            self.global_step += 1
            return {
                'loss': float(total_loss.detach().item()),
                'diffusion_loss': float(diffusion_loss.detach().item()),
                'logic_loss': float(logic_loss.detach().item()) if self._tensor_is_finite(logic_loss) else 0.0,
                'solvability_proxy': float(solvability_proxy.detach().item()) if self._tensor_is_finite(solvability_proxy) else 0.0,
                'solvability': float(solvability_proxy.detach().item()) if self._tensor_is_finite(solvability_proxy) else 0.0,
                'logic_loss_mode_predicted': 1.0 if self.config.logic_loss_mode == 'predicted_latent' else 0.0,
                'skipped_nonfinite_batch': 1.0,
            }
        grad_clip_norm = float(max(0.0, float(getattr(self.config, "grad_clip_norm", 1.0))))
        if grad_clip_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(self.diffusion.parameters()) +
                list(self.condition_encoder.parameters()),
                max_norm=grad_clip_norm
            )
            if not self._tensor_is_finite(grad_norm):
                self.optimizer.zero_grad(set_to_none=True)
                self._warn_nonfinite(
                    "gradient_norm",
                    "Diffusion training: non-finite gradient norm detected after clipping; skipping optimizer step for this batch.",
                )
                self.global_step += 1
                return {
                    'loss': float(total_loss.detach().item()),
                    'diffusion_loss': float(diffusion_loss.detach().item()),
                    'logic_loss': float(logic_loss.detach().item()) if self._tensor_is_finite(logic_loss) else 0.0,
                    'solvability_proxy': float(solvability_proxy.detach().item()) if self._tensor_is_finite(solvability_proxy) else 0.0,
                    'solvability': float(solvability_proxy.detach().item()) if self._tensor_is_finite(solvability_proxy) else 0.0,
                    'logic_loss_mode_predicted': 1.0 if self.config.logic_loss_mode == 'predicted_latent' else 0.0,
                    'skipped_nonfinite_batch': 1.0,
                }
        self.optimizer.step()
        
        # --- Phase 4A: Update EMA model weights ---
        self._update_ema()
        
        # --- Phase 1D: Anneal LogicNet temperature ---
        # Use estimated total steps from config instead of hardcoded epochs*100
        if hasattr(self.logic_net, 'update_temperature'):
            default_total_steps = max(1, int(getattr(self.config, "epochs", 1)) * 100)
            estimated_total_steps = max(1, getattr(self, '_estimated_total_steps', default_total_steps))
            progress = min(1.0, self.global_step / estimated_total_steps)
            self.logic_net.update_temperature(progress)
        
        self.global_step += 1
        
        return {
            'loss': total_loss.item(),
            'diffusion_loss': diffusion_loss.item(),
            'logic_loss': logic_loss.item(),
            'solvability_proxy': solvability_proxy.item(),
            'solvability': solvability_proxy.item(),
            'logic_loss_mode_predicted': 1.0 if self.config.logic_loss_mode == 'predicted_latent' else 0.0,
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
        metrics_sum = {'loss': 0, 'diffusion_loss': 0, 'logic_loss': 0, 'solvability_proxy': 0, 'solvability': 0}
        num_batches = 0
        
        # DESIGN-08: Compute actual total training steps for temperature annealing
        total_epochs = int(getattr(self.config, "epochs", self.epoch + 1))
        self._estimated_total_steps = max(1, total_epochs * len(dataloader))
        
        include_logic = self.epoch >= self.config.warmup_epochs
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(int(self.epoch))
        logger.info(
            "Train epoch %d/%d: logic_loss_%s (warmup_epochs=%d)",
            int(self.epoch + 1),
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
            
            metrics = self.train_step(
                real_maps,
                conditioning=conditioning,
                include_logic_loss=include_logic,
                logic_graph_data=logic_graph_data,
                diffusion_graph_data=diffusion_graph_data,
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
        return {k: float(v) / total_batches for k, v in reduced.items()}
    
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
        num_generated_eval = 0
        skipped_nonfinite = 0

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

            if num_diffusion_eval < int(num_diffusion_samples):
                diffusion_loss = eval_model.training_loss(z_0, conditioning, graph_data=diffusion_graph_data)
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

            if num_generated_eval < int(num_samples):
                # Generate samples using EMA model
                z_gen = eval_model.sample(conditioning, shape=z_0.shape, graph_data=diffusion_graph_data)
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
                'val_skipped_nonfinite': float(skipped_nonfinite),
            }

        include_logic = self.epoch >= self.config.warmup_epochs and self.config.alpha_logic > 0
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
            'val_skipped_nonfinite': float(skipped_nonfinite),
        }
    
    def _build_resume_checkpoint_payload(self, metrics: Optional[Dict] = None) -> Dict[str, Any]:
        return {
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

    def _build_inference_checkpoint_payload(self, metrics: Optional[Dict] = None) -> Dict[str, Any]:
        return {
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

    def save_checkpoint(self, path: str, metrics: Optional[Dict] = None, *, include_optimizer: bool = True):
        """Save training or inference checkpoint."""
        checkpoint = (
            self._build_resume_checkpoint_payload(metrics)
            if bool(include_optimizer)
            else self._build_inference_checkpoint_payload(metrics)
        )
        atomic_torch_save(checkpoint, path)
        write_checkpoint_metadata(
            path,
            model_type="diffusion_resume" if include_optimizer else "diffusion",
            architecture={
                "latent_dim": int(self.config.latent_dim),
                "context_dim": int(self.config.context_dim),
                "num_timesteps": int(self.config.num_timesteps),
                "schedule_type": str(self.config.schedule_type),
                "num_classes": int(self.config.num_classes),
                "vqvae_hidden_dim": int(self.config.vqvae_hidden_dim),
                "vqvae_codebook_size": int(self.config.vqvae_codebook_size),
                "vqvae_use_coordconv": bool(self.config.vqvae_use_coordconv),
            },
            extra={
                "epoch": int(self.epoch),
                "global_step": int(self.global_step),
                "checkpoint_kind": "resume" if include_optimizer else "inference",
                "contains": (
                    ["diffusion", "ema_diffusion", "condition_encoder", "logic_net", "optimizer", "scheduler"]
                    if include_optimizer
                    else ["diffusion", "ema_diffusion", "condition_encoder", "logic_net"]
                ),
                "vqvae_checkpoint": str(getattr(self.config, "vqvae_checkpoint", "") or ""),
                "topology_anchor_policy": build_topology_anchor_policy_metadata(
                    semantic_role_prior_strength=self.config.semantic_role_prior_strength,
                    semantic_puzzle_offset=self.config.semantic_puzzle_offset,
                    topology_supervision_mode=self.config.topology_supervision_mode,
                ),
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
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
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
        self.diffusion.load_state_dict(checkpoint['diffusion_state_dict'])
        if 'ema_diffusion_state_dict' in checkpoint:
            self.ema_diffusion.load_state_dict(checkpoint['ema_diffusion_state_dict'])
        self.condition_encoder.load_state_dict(checkpoint['condition_encoder_state_dict'])
        if 'logic_net_state_dict' in checkpoint:
            self.logic_net.load_state_dict(checkpoint['logic_net_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Re-wire LogicNet into guidance after loading
        self.diffusion.guidance.logic_net = self.logic_net
        self.diffusion.guidance.guidance_scale = float(getattr(self.config, "guidance_scale", 1.0))
        self.diffusion.guidance.clamp_magnitude = float(getattr(self.config, "guidance_clamp_magnitude", 1.0))
        self.diffusion.guidance.relative_norm_cap = float(getattr(self.config, "guidance_relative_norm_cap", 0.25))
        self.diffusion.guidance.schedule_enabled = bool(getattr(self.config, "guidance_schedule_enabled", True))
        self.diffusion.guidance.active_fraction = float(getattr(self.config, "guidance_active_fraction", 0.30))
        self.diffusion.guidance.decay_power = float(getattr(self.config, "guidance_decay_power", 1.0))
        self.diffusion.guidance.max_graph_nodes = int(getattr(self.config, "guidance_max_graph_nodes", 512))
        self.diffusion.guidance.max_key_lock_pairs = int(getattr(self.config, "guidance_max_key_lock_pairs", 2048))
        self.diffusion.guidance.max_guidance_elements = int(getattr(self.config, "guidance_max_guidance_elements", 2_000_000))
        
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
        train_loader_plain = create_dataloader(
            config.data_dir,
            batch_size=config.batch_size,
            shuffle=config.shuffle_train,
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
        )
        train_sampler = make_distributed_sampler(
            train_loader_plain.dataset,
            context=distributed_context,
            shuffle=config.shuffle_train,
            drop_last=config.drop_last,
            seed=int(getattr(config, "seed", 42)),
        )
        train_loader = create_dataloader(
            config.data_dir,
            batch_size=config.batch_size,
            shuffle=config.shuffle_train,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=config.drop_last,
            use_vglc=config.use_vglc,
            normalize=config.normalize,
            room_level=config.room_level,
            load_graphs=True,
            sampler=train_sampler,
            node_feature_dim=config.node_feature_dim,
            edge_feature_dim=config.edge_feature_dim,
            topology_supervision_mode=config.topology_supervision_mode,
            semantic_role_prior_strength=config.semantic_role_prior_strength,
            semantic_puzzle_offset=config.semantic_puzzle_offset,
        )

        val_loader = None
        if distributed_context.is_main_process:
            val_loader = create_dataloader(
                config.data_dir,
                batch_size=config.batch_size,
                shuffle=config.shuffle_val,
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
            )

        sample_kind = "rooms" if config.room_level else "dungeons"
        logger.info(
            "Training samples: %d %s%s",
            len(train_loader.dataset),
            sample_kind,
            f" (world_size={distributed_context.world_size})" if distributed_context.enabled else "",
        )

        trainer = DiffusionTrainer(config, distributed_context=distributed_context)
        if distributed_context.is_main_process:
            diffusion_trainable = count_parameters(trainer.diffusion, trainable_only=True)
            condition_trainable = count_parameters(trainer.condition_encoder, trainable_only=True)
            logic_subset = count_parameters(trainer.logic_net, trainable_only=True)
            logger.info(
                "Diffusion guidance subset (already included in diffusion total): logic_net=%s.",
                format_parameter_count(logic_subset),
            )
            log_capacity_guardrails(
                logger,
                stage_name="Diffusion trainer",
                dataset_size=len(train_loader.dataset),
                param_groups={
                    "diffusion_plus_guidance": diffusion_trainable,
                    "condition_encoder": condition_trainable,
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
            latest_ckpt = torch.load(str(resume_path), map_location="cpu", weights_only=False)
            latest_metrics = latest_ckpt.get("metrics", {})
            if isinstance(latest_metrics, dict):
                best_solvability = float(latest_metrics.get("best_solvability", latest_metrics.get("val_solvability", 0.0)))
                best_teacher_loss = float(latest_metrics.get("best_teacher_loss", latest_metrics.get("val_total_loss", float("inf"))))

        for epoch in range(int(getattr(trainer, "epoch", -1)) + 1, config.epochs):
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
                    f"Epoch {epoch+1}/{config.epochs}: "
                    f"loss={train_metrics['loss']:.4f}, "
                    f"diffusion={train_metrics['diffusion_loss']:.4f}, "
                    f"val_diffusion_loss={val_metrics.get('val_diffusion_loss', float('inf')):.4f}, "
                    f"val_logic_loss={val_metrics.get('val_logic_loss', 0.0):.4f}, "
                    f"val_total_loss={val_metrics.get('val_total_loss', float('inf')):.4f}, "
                    f"val_solvability_proxy={val_metrics.get('val_solvability_proxy', val_metrics['val_solvability']):.4f}, "
                    f"logic_loss_{'enabled' if epoch >= config.warmup_epochs and config.alpha_logic > 0 else 'disabled'}"
                )

                if (epoch + 1) % config.save_every == 0:
                    trainer.save_checkpoint(
                        str(checkpoint_dir / f"resume_epoch_{epoch+1:04d}.pth"),
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
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--model-channels', type=int, default=None)
    parser.add_argument('--context-dim', type=int, default=None)
    parser.add_argument('--unet-channel-mult', type=int, nargs='+', default=None)
    parser.add_argument('--unet-num-res-blocks', type=int, default=None)
    parser.add_argument('--unet-attention-resolutions', type=int, nargs='+', default=None)
    parser.add_argument('--unet-num-heads', type=int, default=None)
    parser.add_argument('--unet-dropout', type=float, default=None)
    parser.add_argument('--alpha-logic', type=float, default=None)
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
    parser.add_argument('--vqvae-hidden-dim', type=int, default=None)
    parser.add_argument('--vqvae-codebook-size', type=int, default=None)
    parser.add_argument('--vqvae-use-coordconv', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--vqvae-mrf-penalty-weight', type=float, default=None)
    parser.add_argument(
        '--topology-refinement-mode',
        type=str,
        default=None,
        choices=['none', 'lightweight', 'gat2', 'upgraded'],
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
    parser.add_argument('--logic-topology-trace-weight', type=float, default=None)
    parser.add_argument('--logic-topology-anchor-weight', type=float, default=None)
    parser.add_argument('--guidance-scale', type=float, default=None)
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

