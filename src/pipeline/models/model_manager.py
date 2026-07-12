"""Model loading and model-management helpers for the dungeon pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch

from src.core import (
    DualStreamConditionEncoder,
    LatentDiffusionModel,
    LogicNet,
    SEMANTIC_PALETTE,
    SymbolicRefiner,
    LearnedTileStatistics,
    create_vqvae,
)
from src.core.definitions import GRAPH_EDGE_FEATURE_DIM, GRAPH_NODE_FEATURE_DIM
from src.core.symbolic_refiner import DEFAULT_ADJACENCY
from src.core.discrete_masked_model import DiscreteMaskedRoomModel
from src.pipeline.block_contracts import summarize_missing_keys
from src.pipeline.room_topology_conditioning import ROOM_TOPOLOGY_CHANNEL_COUNT

logger = logging.getLogger(__name__)


def _require_all_learned_parameters(
    model: torch.nn.Module,
    missing_keys: Iterable[str],
    *,
    component_name: str,
) -> None:
    """Reject checkpoints that leave any learned parameter randomly initialized."""
    parameter_names = {str(name) for name, _parameter in model.named_parameters()}
    missing_parameters = [
        str(name)
        for name in missing_keys
        if str(name) in parameter_names
    ]
    if missing_parameters:
        raise RuntimeError(
            f"{component_name} checkpoint is missing learned parameters; "
            "partial loading would mix trained and random weights: "
            f"{summarize_missing_keys(missing_parameters)}"
        )


def load_vqvae(pipeline, checkpoint_path: Optional[str]) -> torch.nn.Module:
    """Load or create VQ-VAE model."""
    use_coordconv = True
    checkpoint: Optional[Dict[str, Any]] = None
    state_dict: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = {}
    checkpoint_config: Dict[str, Any] = {}
    num_classes = int(np.max(pipeline._valid_semantic_tile_ids_np)) + 1
    architecture_name = "vqvae"
    latent_dim = 64
    codebook_size = 256
    top_codebook_size: Optional[int] = None
    top_latent_dim: Optional[int] = None
    hidden_dim = 128
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint, metadata = pipeline._load_checkpoint_and_metadata(
            checkpoint_path,
            "vqvae",
            accepted_model_types=("diffusion",),
        )
        if isinstance(checkpoint, dict):
            checkpoint_config = pipeline._extract_checkpoint_config(checkpoint)
            declared_model_type = str(metadata.get("model_type", "")).strip().lower()
            explicit_vq_state = checkpoint.get("vqvae_state_dict")
            is_composite_generation_checkpoint = any(
                isinstance(checkpoint.get(key), dict)
                for key in ("diffusion_state_dict", "condition_encoder_state_dict", "logic_net_state_dict")
            )
            if isinstance(explicit_vq_state, dict):
                state_dict = explicit_vq_state
            elif declared_model_type not in {"diffusion"} and not is_composite_generation_checkpoint:
                state_dict = pipeline._extract_checkpoint_state_dict(checkpoint)
        architecture = metadata.get("architecture", {}) if isinstance(metadata, dict) else {}
        if isinstance(architecture, dict):
            architecture_name = str(
                checkpoint_config.get(
                    "architecture",
                    architecture.get("architecture", architecture.get("vqvae_architecture", architecture_name)),
                )
                or architecture_name
            )
        else:
            architecture = {}
        num_classes = int(checkpoint_config.get("num_classes", architecture.get("num_classes", num_classes)))
        latent_dim = int(checkpoint_config.get("latent_dim", latent_dim))
        latent_dim = int(architecture.get("latent_dim", latent_dim))
        codebook_size = int(checkpoint_config.get("codebook_size", codebook_size))
        codebook_size = int(architecture.get("codebook_size", codebook_size))
        top_codebook_candidate = checkpoint_config.get(
            "top_codebook_size",
            architecture.get("top_codebook_size", architecture.get("vqvae_top_codebook_size", None)),
        )
        top_latent_candidate = checkpoint_config.get(
            "top_latent_dim",
            architecture.get("top_latent_dim", architecture.get("vqvae_top_latent_dim", None)),
        )
        if top_codebook_candidate is not None:
            top_codebook_size = int(top_codebook_candidate)
        if top_latent_candidate is not None:
            top_latent_dim = int(top_latent_candidate)
        use_coordconv = bool(checkpoint_config.get("use_coordconv", use_coordconv))
        use_coordconv = bool(architecture.get("use_coordconv", use_coordconv))
        # Backward compatibility: older checkpoints may use plain Conv2d
        # keys (encoder.conv_in.weight) while newer CoordConv checkpoints
        # use encoder.conv_in.conv.weight.
        if isinstance(state_dict, dict):
            has_coordconv_keys = ('encoder.conv_in.conv.weight' in state_dict)
            has_plain_conv_keys = ('encoder.conv_in.weight' in state_dict)
            if has_plain_conv_keys and not has_coordconv_keys:
                use_coordconv = False
            conv_key = 'encoder.conv_in.conv.weight' if has_coordconv_keys else 'encoder.conv_in.weight'
            conv_weight = state_dict.get(conv_key)
            if isinstance(conv_weight, torch.Tensor) and conv_weight.dim() == 4:
                hidden_dim = int(max(1, int(conv_weight.shape[0])))

    model = create_vqvae(
        architecture=architecture_name,
        num_classes=num_classes,
        codebook_size=codebook_size,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        top_codebook_size=top_codebook_size,
        top_latent_dim=top_latent_dim,
        use_coordconv=use_coordconv,
    ).to(pipeline.device)

    if checkpoint_path and Path(checkpoint_path).exists():
        if checkpoint is None:
            checkpoint, _metadata = pipeline._load_checkpoint_and_metadata(
                checkpoint_path,
                "vqvae",
                accepted_model_types=("diffusion",),
            )
        if state_dict is None and isinstance(checkpoint, dict):
            declared_model_type = str(metadata.get("model_type", "")).strip().lower()
            explicit_vq_state = checkpoint.get("vqvae_state_dict")
            is_composite_generation_checkpoint = any(
                isinstance(checkpoint.get(key), dict)
                for key in ("diffusion_state_dict", "condition_encoder_state_dict", "logic_net_state_dict")
            )
            if isinstance(explicit_vq_state, dict):
                state_dict = explicit_vq_state
            elif declared_model_type not in {"diffusion"} and not is_composite_generation_checkpoint:
                state_dict = pipeline._extract_checkpoint_state_dict(checkpoint)
        if isinstance(state_dict, dict):
            incompatible = model.load_state_dict(state_dict, strict=False)
            missing = [str(k) for k in getattr(incompatible, 'missing_keys', [])]
            unexpected = [str(k) for k in getattr(incompatible, 'unexpected_keys', [])]
            _require_all_learned_parameters(
                model,
                missing,
                component_name="VQ-VAE",
            )

            # Legacy checkpoints created before explicit legality buffer registration.
            allowed_missing = {"illegal_adjacency_matrix"}
            unexpected_missing = [k for k in missing if k not in allowed_missing]

            if unexpected_missing or unexpected:
                msg = (
                    "VQ-VAE checkpoint key mismatch. "
                    f"missing={unexpected_missing} unexpected={unexpected}"
                )
                if pipeline.strict_checkpoint_mode:
                    raise RuntimeError(msg)
                logger.warning(msg)
        elif isinstance(checkpoint, dict):
            raise ValueError(
                f"VQ-VAE checkpoint at {checkpoint_path!r} does not contain "
                "a loadable VQ-VAE state_dict. Refusing to continue with a "
                "random Block-II tokenizer."
            )
        else:
            model.load_state_dict(checkpoint)
        logger.info(f"Loaded VQ-VAE from {checkpoint_path}")
    else:
        if pipeline.strict_checkpoint_mode:
            raise FileNotFoundError(
                f"Strict checkpoint mode enabled: missing VQ-VAE checkpoint at {checkpoint_path!r}"
            )
        logger.warning("No VQ-VAE checkpoint provided, using random initialization")

    return model

def load_condition_encoder(
    pipeline, 
    checkpoint_path: Optional[str]
) -> DualStreamConditionEncoder:
    """
    Load or create condition encoder.

    Best-practice behavior:
    - default to richer graph-conditioning schema for fresh training,
    - auto-infer legacy schema from checkpoint weights for compatibility.
    """
    default_node_feature_dim = int(GRAPH_NODE_FEATURE_DIM)
    default_edge_feature_dim = int(GRAPH_EDGE_FEATURE_DIM)
    node_feature_dim = int(default_node_feature_dim)
    edge_feature_dim = int(default_edge_feature_dim)
    checkpoint_state: Optional[Dict[str, Any]] = None
    checkpoint_config: Dict[str, Any] = {}
    fallback_config = dict(pipeline.condition_encoder_fallback_config)
    default_latent_dim = int(
        getattr(getattr(pipeline, "vqvae", None), "latent_dim", 64)
    )

    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint, _metadata = pipeline._load_checkpoint_and_metadata(
            checkpoint_path,
            "condition_encoder",
            accepted_model_types=("diffusion", "masked_room_model"),
        )
        checkpoint_state = pipeline._extract_checkpoint_state_dict(
            checkpoint,
            candidate_keys=["ema_condition_encoder_state_dict", "condition_encoder_state_dict"],
        )
        checkpoint_config = pipeline._extract_checkpoint_config(checkpoint)
        if isinstance(checkpoint_state, dict):
            node_weight = checkpoint_state.get('global_encoder.node_encoder.weight')
            edge_weight = checkpoint_state.get('global_encoder.edge_encoder.weight')
            if isinstance(node_weight, torch.Tensor) and node_weight.dim() == 2:
                node_feature_dim = int(max(1, int(node_weight.shape[1])))
            if isinstance(edge_weight, torch.Tensor) and edge_weight.dim() == 2:
                edge_feature_dim = int(max(1, int(edge_weight.shape[1])))

    if "condition_use_rrwp_edge_features" in checkpoint_config:
        use_rrwp_edge_features = bool(checkpoint_config["condition_use_rrwp_edge_features"])
    elif isinstance(checkpoint_state, dict):
        # Checkpoints created before RRWP edge conditioning existed have no
        # RRWP projection weights and must reconstruct the legacy architecture.
        use_rrwp_edge_features = False
    else:
        use_rrwp_edge_features = bool(
            fallback_config.get(
                "condition_use_rrwp_edge_features",
                pipeline.condition_use_rrwp_edge_features,
            )
        )

    model = DualStreamConditionEncoder(
        latent_dim=int(checkpoint_config.get("latent_dim", fallback_config.get("latent_dim", default_latent_dim))),
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=int(checkpoint_config.get("condition_hidden_dim", fallback_config.get("condition_hidden_dim", 256))),
        output_dim=int(checkpoint_config.get("context_dim", fallback_config.get("context_dim", 256))),
        gnn_type=str(checkpoint_config.get("condition_gnn_type", fallback_config.get("condition_gnn_type", pipeline.condition_gnn_type))),
        num_gnn_layers=int(checkpoint_config.get("condition_num_gnn_layers", fallback_config.get("condition_num_gnn_layers", 3))),
        num_attention_heads=int(checkpoint_config.get("condition_num_attention_heads", fallback_config.get("condition_num_attention_heads", 8))),
        dropout=float(checkpoint_config.get("condition_dropout", fallback_config.get("condition_dropout", 0.1))),
        use_current_node_distance_features=bool(
            checkpoint_config.get(
                "use_current_node_distance_features",
                fallback_config.get("use_current_node_distance_features", pipeline.use_current_node_distance_features),
            )
        ),
        use_reference_room_maps=bool(
            checkpoint_config.get(
                "condition_use_reference_room_maps",
                fallback_config.get("condition_use_reference_room_maps", pipeline.condition_use_reference_room_maps),
            )
        ),
        reference_num_tile_types=int(
            checkpoint_config.get(
                "condition_reference_tile_vocab_size",
                fallback_config.get("condition_reference_tile_vocab_size", pipeline.condition_reference_tile_vocab_size),
            )
        ),
        reference_embedding_dim=int(
            checkpoint_config.get(
                "condition_reference_embedding_dim",
                fallback_config.get("condition_reference_embedding_dim", pipeline.condition_reference_embedding_dim),
            )
        ),
        reference_hidden_dim=int(
            checkpoint_config.get(
                "condition_reference_hidden_dim",
                fallback_config.get("condition_reference_hidden_dim", pipeline.condition_reference_hidden_dim),
            )
        ),
        use_rrwp_edge_features=use_rrwp_edge_features,
        strict_schema=bool(
            checkpoint_config.get(
                "condition_strict_schema",
                fallback_config.get(
                    "condition_strict_schema",
                    bool(getattr(pipeline, "condition_strict_schema", pipeline.strict_checkpoint_mode)),
                ),
            )
        ),
    ).to(pipeline.device)

    if checkpoint_state is not None:
        if isinstance(checkpoint_state, dict):
            model_state = model.state_dict()
            filtered_state: Dict[str, Any] = {}
            dropped_shape: List[str] = []
            for k, v in checkpoint_state.items():
                if k not in model_state:
                    continue
                target_v = model_state[k]
                if hasattr(v, 'shape') and hasattr(target_v, 'shape') and tuple(v.shape) != tuple(target_v.shape):
                    dropped_shape.append(k)
                    continue
                filtered_state[k] = v
            checkpoint_state = filtered_state
        incompatible = model.load_state_dict(checkpoint_state, strict=False)
        missing = list(getattr(incompatible, 'missing_keys', []))
        unexpected = list(getattr(incompatible, 'unexpected_keys', []))
        _require_all_learned_parameters(
            model,
            missing,
            component_name="Condition encoder",
        )
        logger.info(
            "Loaded condition encoder from %s (node_dim=%d edge_dim=%d, missing=%d unexpected=%d)",
            checkpoint_path,
            node_feature_dim,
            edge_feature_dim,
            len(missing),
            len(unexpected),
        )
        if 'dropped_shape' in locals() and dropped_shape:
            logger.warning(
                "Condition encoder dropped incompatible checkpoint keys (shape mismatch): %s",
                summarize_missing_keys(dropped_shape),
            )
        if missing or unexpected:
            msg = (
                "Condition encoder checkpoint/schema mismatch: "
                f"missing={summarize_missing_keys(missing)} unexpected={summarize_missing_keys(unexpected)}"
            )
            if pipeline.strict_checkpoint_mode:
                raise RuntimeError(msg)
            logger.warning(msg)
    else:
        if pipeline.strict_checkpoint_mode:
            raise FileNotFoundError(
                f"Strict checkpoint mode enabled: missing condition encoder checkpoint at {checkpoint_path!r}"
            )
        logger.warning(
            "No condition encoder checkpoint, using random initialization with enhanced schema (node_dim=%d edge_dim=%d)",
            node_feature_dim,
            edge_feature_dim,
        )

    return model

def load_diffusion(pipeline, checkpoint_path: Optional[str]) -> LatentDiffusionModel:
    """Load or create latent diffusion model."""
    checkpoint_config: Dict[str, Any] = {}
    checkpoint_state: Optional[Dict[str, Any]] = None
    fallback_config = dict(pipeline.diffusion_fallback_config)
    default_latent_dim = int(
        checkpoint_config.get(
            "latent_dim",
            fallback_config.get("latent_dim", getattr(getattr(pipeline, "vqvae", None), "latent_dim", 64)),
        )
    )
    default_context_dim = int(
        checkpoint_config.get(
            "context_dim",
            fallback_config.get("context_dim", getattr(getattr(pipeline, "condition_encoder", None), "output_dim", 256)),
        )
    )
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint, _metadata = pipeline._load_checkpoint_and_metadata(checkpoint_path, "diffusion")
        checkpoint_config = pipeline._extract_checkpoint_config(checkpoint)
        if pipeline.strict_checkpoint_mode and "topology_conditioning_mode" not in checkpoint_config:
            raise ValueError(
                "Strict checkpoint mode enabled: diffusion checkpoint config missing required key "
                "'topology_conditioning_mode'."
            )
        checkpoint_state_key = "ema_diffusion_state_dict"
        checkpoint_state = pipeline._extract_checkpoint_state_dict(
            checkpoint,
            "ema_diffusion_state_dict",
            "diffusion_state_dict",
        )
        if checkpoint_state is None:
            checkpoint_state_key = "diffusion_state_dict"
        elif not isinstance(checkpoint.get("ema_diffusion_state_dict"), dict):
            checkpoint_state_key = "diffusion_state_dict"
        default_latent_dim = int(
            checkpoint_config.get(
                "latent_dim",
                getattr(getattr(pipeline, "vqvae", None), "latent_dim", default_latent_dim),
            )
        )
        default_context_dim = int(
            checkpoint_config.get(
                "context_dim",
                getattr(getattr(pipeline, "condition_encoder", None), "output_dim", default_context_dim),
            )
        )
    training_objective = str(
        checkpoint_config.get(
            "diffusion_training_objective",
            checkpoint_config.get("training_objective", fallback_config.get("diffusion_training_objective", "diffusion")),
        )
    ).strip().lower()
    model = LatentDiffusionModel(
        latent_dim=default_latent_dim,
        latent_scale_factor=float(
            checkpoint_config.get(
                "latent_scale_factor",
                fallback_config.get("latent_scale_factor", 1.0),
            )
        ),
        context_dim=default_context_dim,
        num_timesteps=int(checkpoint_config.get("num_timesteps", fallback_config.get("num_timesteps", 1000))),
        prediction_type=str(checkpoint_config.get("prediction_type", fallback_config.get("prediction_type", "epsilon"))),
        cfg_dropout_prob=float(checkpoint_config.get("cfg_dropout_prob", fallback_config.get("cfg_dropout_prob", 0.1))),
        cfg_scale=float(checkpoint_config.get("cfg_scale", fallback_config.get("cfg_scale", 3.0))),
        pag_scale=float(checkpoint_config.get("pag_scale", fallback_config.get("pag_scale", 0.0))),
        cfg_schedule_mode=str(checkpoint_config.get("cfg_schedule_mode", pipeline.diffusion_cfg_schedule_mode)),
        cfg_schedule_min_scale=float(checkpoint_config.get("cfg_schedule_min_scale", pipeline.diffusion_cfg_schedule_min_scale)),
        cfg_schedule_power=float(checkpoint_config.get("cfg_schedule_power", pipeline.diffusion_cfg_schedule_power)),
        min_snr_gamma=float(checkpoint_config.get("min_snr_gamma", fallback_config.get("min_snr_gamma", 5.0))),
        model_channels=int(checkpoint_config.get("model_channels", fallback_config.get("model_channels", 128))),
        topology_refinement_mode=str(checkpoint_config.get("topology_refinement_mode", pipeline.topology_refinement_mode)),
        attention_mode=str(checkpoint_config.get("attention_mode", pipeline.diffusion_attention_mode)),
        topology_conditioning_mode=str(
            checkpoint_config.get("topology_conditioning_mode", fallback_config.get("topology_conditioning_mode", "additive"))
        ),
        hedgehog_feature_dim=int(checkpoint_config.get("hedgehog_feature_dim", pipeline.diffusion_hedgehog_feature_dim)),
        denoiser_backbone=str(checkpoint_config.get("denoiser_backbone", fallback_config.get("denoiser_backbone", "unet"))),
        unet_channel_mult=tuple(checkpoint_config.get("unet_channel_mult", fallback_config.get("unet_channel_mult", (1, 2, 4)))),
        unet_num_res_blocks=int(checkpoint_config.get("unet_num_res_blocks", fallback_config.get("unet_num_res_blocks", 2))),
        unet_attention_resolutions=tuple(
            checkpoint_config.get("unet_attention_resolutions", fallback_config.get("unet_attention_resolutions", (1, 2)))
        ),
        unet_num_heads=int(checkpoint_config.get("unet_num_heads", fallback_config.get("unet_num_heads", 8))),
        unet_dropout=float(checkpoint_config.get("unet_dropout", fallback_config.get("unet_dropout", 0.1))),
        dit_depth=int(checkpoint_config.get("dit_depth", fallback_config.get("dit_depth", 4))),
        dit_patch_size=int(checkpoint_config.get("dit_patch_size", fallback_config.get("dit_patch_size", 1))),
        dit_mlp_ratio=float(checkpoint_config.get("dit_mlp_ratio", fallback_config.get("dit_mlp_ratio", 4.0))),
        graph_auto_linear_attention_nodes=int(
            checkpoint_config.get(
                "graph_auto_linear_attention_nodes",
                fallback_config.get("graph_auto_linear_attention_nodes", 128),
            )
        ),
        graphormer_max_distance=int(
            checkpoint_config.get(
                "graphormer_max_distance",
                fallback_config.get("graphormer_max_distance", 16),
            )
        ),
        graphormer_max_degree=int(
            checkpoint_config.get(
                "graphormer_max_degree",
                fallback_config.get("graphormer_max_degree", 64),
            )
        ),
        spatial_graph_gate_init=float(
            checkpoint_config.get("spatial_graph_gate_init", fallback_config.get("spatial_graph_gate_init", -2.0))
        ),
        spatial_topology_gate_init=float(
            checkpoint_config.get("spatial_topology_gate_init", fallback_config.get("spatial_topology_gate_init", -2.0))
        ),
        room_topology_channels=int(
            checkpoint_config.get("room_topology_channels", fallback_config.get("room_topology_channels", ROOM_TOPOLOGY_CHANNEL_COUNT))
        ),
        training_objective=training_objective,
    ).to(pipeline.device)
    setattr(
        model,
        "training_cfg_scale",
        float(checkpoint_config.get("cfg_scale", fallback_config.get("cfg_scale", 3.0))),
    )
    setattr(model, "training_objective", training_objective)
    if training_objective == "flow_matching":
        logger.info(
            "Loaded a diffusion checkpoint trained with flow_matching; generation will prefer "
            "LatentDiffusionModel.flow_ode_sample() over DDPM/DDIM sampling."
        )
    setattr(
        model,
        "inference_checkpoint_state_key",
        str(locals().get("checkpoint_state_key", "random_init")),
    )
    pipeline.diffusion_puzzle_structure_condition_enabled = bool(
        float(checkpoint_config.get("puzzle_structure_dropout_prob", fallback_config.get("puzzle_structure_dropout_prob", 0.0))) > 0.0
    )

    if checkpoint_path and Path(checkpoint_path).exists():
        if not isinstance(checkpoint_state, dict):
            raise ValueError(
                f"Diffusion checkpoint at {checkpoint_path!r} does not contain a loadable state_dict."
            )
        if isinstance(checkpoint_state, dict):
            model_state = model.state_dict()
            filtered_state: Dict[str, Any] = {}
            dropped_shape: List[str] = []
            for k, v in checkpoint_state.items():
                if k not in model_state:
                    continue
                target_v = model_state[k]
                if hasattr(v, 'shape') and hasattr(target_v, 'shape') and tuple(v.shape) != tuple(target_v.shape):
                    dropped_shape.append(k)
                    continue
                filtered_state[k] = v
            checkpoint_state = filtered_state
        incompatible = model.load_state_dict(checkpoint_state, strict=False)
        missing = list(getattr(incompatible, 'missing_keys', []))
        unexpected = list(getattr(incompatible, 'unexpected_keys', []))
        _require_all_learned_parameters(
            model,
            missing,
            component_name="Diffusion model",
        )
        logger.info(
            "Loaded diffusion model from %s using %s (missing=%d unexpected=%d)",
            checkpoint_path,
            getattr(model, "inference_checkpoint_state_key", "diffusion_state_dict"),
            len(missing),
            len(unexpected),
        )
        if 'dropped_shape' in locals() and dropped_shape:
            logger.warning(
                "Diffusion dropped incompatible checkpoint keys (shape mismatch): %s",
                summarize_missing_keys(dropped_shape),
            )
        if missing or unexpected:
            msg = (
                "Diffusion checkpoint/schema mismatch: "
                f"missing={summarize_missing_keys(missing)} unexpected={summarize_missing_keys(unexpected)}"
            )
            if pipeline.strict_checkpoint_mode:
                raise RuntimeError(msg)
            logger.warning(msg)
    else:
        if pipeline.strict_checkpoint_mode:
            raise FileNotFoundError(
                f"Strict checkpoint mode enabled: missing diffusion checkpoint at {checkpoint_path!r}"
            )
        logger.warning("No diffusion checkpoint, using random initialization")

    if pipeline.fast_sampling_checkpoint:
        fast_ckpt_path = Path(pipeline.fast_sampling_checkpoint)
        if fast_ckpt_path.exists():
            try:
                model.enable_fast_sampling(
                    adapter_checkpoint=str(fast_ckpt_path),
                    num_inference_steps=int(pipeline.fast_sampling_steps),
                    use_fp16=(pipeline.device.type == "cuda"),
                    compile_model=False,
                    strict=pipeline.strict_checkpoint_mode,
                )
                logger.info(
                    "Enabled distilled fast sampling from %s (%d steps).",
                    fast_ckpt_path,
                    int(pipeline.fast_sampling_steps),
                )
            except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
                if pipeline.strict_checkpoint_mode:
                    raise
                logger.warning(
                    "Fast-sampling checkpoint rejected; using standard diffusion sampling: %s",
                    exc,
                )
        elif pipeline.strict_checkpoint_mode:
            raise FileNotFoundError(
                f"Strict checkpoint mode enabled: missing fast-sampling checkpoint at {fast_ckpt_path}"
            )
        else:
            logger.warning(
                "Fast-sampling checkpoint not found at %s; using standard diffusion sampling.",
                fast_ckpt_path,
            )

    return model

def load_logic_net(pipeline, checkpoint_path: Optional[str]) -> Optional[LogicNet]:
    """Load or create LogicNet."""
    checkpoint_state: Optional[Dict[str, Any]] = None
    checkpoint_config: Dict[str, Any] = {}
    fallback_config = dict(pipeline.logic_net_fallback_config)
    default_latent_dim = int(
        getattr(
            getattr(pipeline, "diffusion", None),
            "latent_dim",
            fallback_config.get("latent_dim", getattr(getattr(pipeline, "vqvae", None), "latent_dim", 64)),
        )
    )
    default_num_classes = int(
        fallback_config.get("num_classes", getattr(getattr(pipeline, "vqvae", None), "num_classes", 44))
    )
    model = LogicNet(
        latent_dim=default_latent_dim,
        latent_scale_factor=float(
            getattr(
                getattr(pipeline, "diffusion", None),
                "latent_scale_factor",
                fallback_config.get("latent_scale_factor", 1.0),
            )
        ),
        num_classes=default_num_classes,
        num_iterations=int(fallback_config.get("num_logic_iterations", 20)),
        grid_pathfinder_type=str(fallback_config.get("logic_grid_pathfinder", "bellman_ford")),
        resource_gate_mode=str(fallback_config.get("logic_resource_gate_mode", "hard_ordered")),
        full_coverage=bool(fallback_config.get("logic_full_coverage", True)),
        initial_temperature=float(fallback_config.get("logic_initial_temperature", 1.0)),
        final_temperature=float(fallback_config.get("logic_final_temperature", 0.05)),
        topology_trace_weight=float(fallback_config.get("logic_topology_trace_weight", 0.25)),
        topology_anchor_weight=float(fallback_config.get("logic_topology_anchor_weight", 0.25)),
        global_reach_weight=float(fallback_config.get("logic_global_reach_weight", 1.0)),
        global_room_weight=float(fallback_config.get("logic_global_room_weight", 0.25)),
    ).to(pipeline.device)

    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint, metadata = pipeline._load_checkpoint_and_metadata(
            checkpoint_path,
            "logic_net",
            accepted_model_types=("diffusion",),
        )
        checkpoint_state = pipeline._extract_checkpoint_state_dict(
            checkpoint,
            candidate_keys=["ema_logic_net_state_dict", "logic_net_state_dict"],
        )
        checkpoint_config = pipeline._extract_checkpoint_config(checkpoint)
        architecture = metadata.get("architecture", {}) if isinstance(metadata, dict) else {}
        model = LogicNet(
            latent_dim=int(checkpoint_config.get("latent_dim", architecture.get("latent_dim", default_latent_dim))),
            latent_scale_factor=float(
                checkpoint_config.get(
                    "latent_scale_factor",
                    getattr(getattr(pipeline, "diffusion", None), "latent_scale_factor", 1.0),
                )
            ),
            num_classes=int(checkpoint_config.get("num_classes", architecture.get("num_classes", default_num_classes))),
            num_iterations=int(checkpoint_config.get("num_logic_iterations", 20)),
            grid_pathfinder_type=str(checkpoint_config.get("logic_grid_pathfinder", fallback_config.get("logic_grid_pathfinder", "bellman_ford"))),
            resource_gate_mode=str(checkpoint_config.get("logic_resource_gate_mode", fallback_config.get("logic_resource_gate_mode", "hard_ordered"))),
            full_coverage=bool(checkpoint_config.get("logic_full_coverage", fallback_config.get("logic_full_coverage", True))),
            initial_temperature=float(checkpoint_config.get("logic_initial_temperature", fallback_config.get("logic_initial_temperature", 1.0))),
            final_temperature=float(checkpoint_config.get("logic_final_temperature", fallback_config.get("logic_final_temperature", 0.05))),
            topology_trace_weight=float(checkpoint_config.get("logic_topology_trace_weight", 0.25)),
            topology_anchor_weight=float(checkpoint_config.get("logic_topology_anchor_weight", 0.25)),
            global_reach_weight=float(checkpoint_config.get("logic_global_reach_weight", 1.0)),
            global_room_weight=float(checkpoint_config.get("logic_global_room_weight", 0.25)),
        ).to(pipeline.device)
        if isinstance(checkpoint_state, dict):
            incompatible = model.load_state_dict(checkpoint_state, strict=False)
            missing = list(getattr(incompatible, "missing_keys", []))
            unexpected = list(getattr(incompatible, "unexpected_keys", []))
            parameter_names = {name for name, _param in model.named_parameters()}
            missing_parameters = [name for name in missing if name in parameter_names]
            if missing_parameters:
                raise RuntimeError(
                    "LogicNet checkpoint is missing learned parameters: "
                    f"{summarize_missing_keys(missing_parameters)}"
                )
            compatibility_only = {"locked_edge_role_ids"}
            unexpected = [name for name in unexpected if name not in compatibility_only]
            if missing or unexpected:
                msg = (
                    "LogicNet checkpoint/schema mismatch: "
                    f"missing={summarize_missing_keys(missing)} "
                    f"unexpected={summarize_missing_keys(unexpected)}"
                )
                if pipeline.strict_checkpoint_mode:
                    raise RuntimeError(msg)
                logger.warning(msg)
        else:
            raise ValueError(
                f"LogicNet checkpoint at {checkpoint_path!r} does not contain a loadable state_dict."
            )
        checkpoint_metrics = checkpoint.get("metrics", {}) if isinstance(checkpoint, dict) else {}
        tile_accuracy = None
        if isinstance(checkpoint_metrics, dict):
            raw_accuracy = checkpoint_metrics.get(
                "val_logic_tile_accuracy",
                checkpoint_metrics.get("logic_tile_accuracy"),
            )
            try:
                tile_accuracy = float(raw_accuracy) if raw_accuracy is not None else None
            except (TypeError, ValueError, OverflowError):
                tile_accuracy = None
        min_tile_accuracy = float(
            checkpoint_config.get(
                "min_logic_tile_accuracy_for_guidance",
                fallback_config.get("min_logic_tile_accuracy_for_guidance", 0.4),
            )
        )
        setattr(model, "_hmolqd_logic_tile_accuracy", tile_accuracy)
        setattr(model, "_hmolqd_min_logic_tile_accuracy", max(0.0, min_tile_accuracy))
        setattr(
            model,
            "_hmolqd_guidance_calibrated",
            bool(tile_accuracy is not None and tile_accuracy >= max(0.0, min_tile_accuracy)),
        )
        setattr(model, "_hmolqd_checkpoint_loaded", True)
        pipeline.logic_net_checkpoint_loaded = True
        logger.info(f"Loaded LogicNet from {checkpoint_path}")
    else:
        if pipeline.strict_checkpoint_mode or bool(getattr(pipeline, "require_logic_net", False)):
            raise FileNotFoundError(
                "LogicNet is required for this run but no loadable checkpoint was found "
                f"at {checkpoint_path!r}. Enable a symbolic-only ablation explicitly by "
                "setting require_logic_net=False."
            )
        pipeline.logic_net_checkpoint_loaded = False
        logger.warning("No LogicNet checkpoint; LogicNet-dependent guidance and evaluation are disabled.")
        return None

    return model

def load_masked_room_model(
    pipeline,
    checkpoint_path: Optional[str],
) -> Optional[DiscreteMaskedRoomModel]:
    """Load or create the optional discrete masked room model."""
    if checkpoint_path is None and pipeline.room_generator_mode != "discrete_masked":
        return None

    from src.core.discrete_masked_model import create_discrete_masked_model

    checkpoint_config: Dict[str, Any] = {}
    checkpoint_state: Optional[Dict[str, Any]] = None
    checkpoint: Optional[Dict[str, Any]] = None
    fallback_config = dict(pipeline.masked_room_fallback_config)
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint, _metadata = pipeline._load_checkpoint_and_metadata(checkpoint_path, "masked_room_model")
        checkpoint_config = pipeline._extract_checkpoint_config(checkpoint)
        if pipeline.strict_checkpoint_mode and "topology_conditioning_mode" not in checkpoint_config:
            raise ValueError(
                "Strict checkpoint mode enabled: masked-room checkpoint config missing required key "
                "'topology_conditioning_mode'."
            )
        checkpoint_state = pipeline._extract_checkpoint_state_dict(
            checkpoint,
            candidate_keys=["ema_masked_room_state_dict", "masked_room_state_dict"],
        )

    model = create_discrete_masked_model(
        num_classes=int(
            checkpoint_config.get(
                "num_classes",
                fallback_config.get("num_classes", getattr(getattr(pipeline, "vqvae", None), "num_classes", 44)),
            )
        ),
        hidden_dim=int(checkpoint_config.get("hidden_dim", fallback_config.get("hidden_dim", 48))),
        model_channels=int(checkpoint_config.get("model_channels", fallback_config.get("model_channels", 64))),
        context_dim=int(
            checkpoint_config.get(
                "context_dim",
                fallback_config.get("context_dim", getattr(getattr(pipeline, "condition_encoder", None), "output_dim", 256)),
            )
        ),
        num_steps=int(checkpoint_config.get("masked_steps", pipeline.masked_sampling_steps)),
        attention_mode=str(checkpoint_config.get("attention_mode", pipeline.diffusion_attention_mode)),
        context_attention_mode=str(
            checkpoint_config.get(
                "context_attention_mode",
                fallback_config.get("context_attention_mode", "concat_encoder"),
            )
        ),
        topology_conditioning_mode=str(
            checkpoint_config.get("topology_conditioning_mode", fallback_config.get("topology_conditioning_mode", "additive"))
        ),
        hedgehog_feature_dim=int(checkpoint_config.get("hedgehog_feature_dim", pipeline.diffusion_hedgehog_feature_dim)),
        graph_auto_linear_attention_nodes=int(
            checkpoint_config.get(
                "graph_auto_linear_attention_nodes",
                fallback_config.get("graph_auto_linear_attention_nodes", 128),
            )
        ),
        spatial_graph_gate_init=float(
            checkpoint_config.get("spatial_graph_gate_init", fallback_config.get("spatial_graph_gate_init", -2.0))
        ),
        spatial_topology_gate_init=float(
            checkpoint_config.get("spatial_topology_gate_init", fallback_config.get("spatial_topology_gate_init", -2.0))
        ),
        unet_channel_mult=tuple(checkpoint_config.get("unet_channel_mult", fallback_config.get("unet_channel_mult", (1, 2)))),
        unet_num_res_blocks=int(checkpoint_config.get("unet_num_res_blocks", fallback_config.get("unet_num_res_blocks", 1))),
        unet_attention_resolutions=tuple(
            checkpoint_config.get("unet_attention_resolutions", fallback_config.get("unet_attention_resolutions", (0, 1)))
        ),
        unet_num_heads=int(checkpoint_config.get("unet_num_heads", fallback_config.get("unet_num_heads", 4))),
        unet_dropout=float(checkpoint_config.get("unet_dropout", fallback_config.get("unet_dropout", 0.1))),
        room_topology_channels=int(
            checkpoint_config.get("room_topology_channels", fallback_config.get("room_topology_channels", ROOM_TOPOLOGY_CHANNEL_COUNT))
        ),
    ).to(pipeline.device)
    pipeline.masked_room_puzzle_structure_condition_enabled = bool(
        float(checkpoint_config.get("puzzle_structure_dropout_prob", fallback_config.get("puzzle_structure_dropout_prob", 0.0))) > 0.0
    )

    if checkpoint_path and Path(checkpoint_path).exists():
        assert checkpoint is not None
        state_dict = checkpoint_state
        if not isinstance(state_dict, dict):
            raise ValueError(
                f"Masked-room checkpoint at {checkpoint_path!r} does not contain a loadable state_dict."
            )
        incompatible = model.load_state_dict(state_dict, strict=False)
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        _require_all_learned_parameters(
            model,
            missing,
            component_name="Masked-room model",
        )
        if missing or unexpected:
            msg = (
                "Masked-room checkpoint/schema mismatch: "
                f"missing={summarize_missing_keys(missing)} unexpected={summarize_missing_keys(unexpected)}"
            )
            if pipeline.strict_checkpoint_mode:
                raise RuntimeError(msg)
            logger.warning(msg)

        cond_state = checkpoint.get("condition_encoder_state_dict")
        if cond_state is not None and pipeline.condition_encoder is not None:
            incompatible_cond = pipeline.condition_encoder.load_state_dict(cond_state, strict=False)
            cond_missing = list(getattr(incompatible_cond, "missing_keys", []))
            cond_unexpected = list(getattr(incompatible_cond, "unexpected_keys", []))
            _require_all_learned_parameters(
                pipeline.condition_encoder,
                cond_missing,
                component_name="Masked-room bundled condition encoder",
            )
            if cond_missing or cond_unexpected:
                msg = (
                    "Masked-room checkpoint bundled condition encoder mismatch: "
                    f"missing={summarize_missing_keys(cond_missing)} unexpected={summarize_missing_keys(cond_unexpected)}"
                )
                if pipeline.strict_checkpoint_mode:
                    raise RuntimeError(msg)
                logger.warning(msg)
        logger.info("Loaded discrete masked room model from %s", checkpoint_path)
    else:
        if pipeline.room_generator_mode == "discrete_masked" and pipeline.strict_checkpoint_mode:
            raise FileNotFoundError(
                f"Strict checkpoint mode enabled: missing discrete masked room checkpoint at {checkpoint_path!r}"
            )
        if pipeline.room_generator_mode == "discrete_masked":
            logger.warning("No discrete masked room checkpoint provided, using random initialization")

    return model

def create_refiner(
    use_learned_rules: bool,
    *,
    max_repair_attempts: int = 5,
    margin: int = 2,
    adjacency_threshold: float = 0.01,
) -> SymbolicRefiner:
    """Create symbolic refiner with optional learned rules."""
    learned_stats = LearnedTileStatistics() if use_learned_rules else None

    # Align refiner vocabulary with project semantic palette.
    # Legacy SymbolicRefiner defaults include tile id 50, which is outside
    # this project's canonical semantic range and can leak invalid ids.
    valid_ids = set(int(v) for v in SEMANTIC_PALETTE.values())
    enemy_id = int(SEMANTIC_PALETTE.get("ENEMY", 20))
    legacy_to_canonical = {
        50: enemy_id,
    }
    canonical_adjacency: Dict[int, Set[int]] = {}
    for raw_src, raw_neighbors in dict(DEFAULT_ADJACENCY).items():
        src = int(legacy_to_canonical.get(int(raw_src), int(raw_src)))
        if src not in valid_ids:
            continue
        neigh_set: Set[int] = set()
        for raw_dst in set(raw_neighbors):
            dst = int(legacy_to_canonical.get(int(raw_dst), int(raw_dst)))
            if dst in valid_ids:
                neigh_set.add(dst)
        if not neigh_set:
            neigh_set.add(src)
        neigh_set.add(src)
        canonical_adjacency[src] = neigh_set

    tile_types = sorted(int(t) for t in canonical_adjacency.keys())

    refiner = SymbolicRefiner(
        tile_types=tile_types,
        adjacency=(None if use_learned_rules else canonical_adjacency),
        learned_stats=learned_stats,
        max_repair_attempts=int(max(1, int(max_repair_attempts))),
        margin=int(max(0, int(margin))),
        adjacency_threshold=float(max(0.0, float(adjacency_threshold))),
    )

    logger.info(f"Created SymbolicRefiner (learned_rules={use_learned_rules})")
    return refiner


class ModelManager:
    """
    Model-management boundary for the pipeline facade.

    Checkpoint loading lives here so runtime initialization and sampling do not
    need to know model-specific construction details.
    """

    _MODEL_ATTRS = (
        "vqvae",
        "condition_encoder",
        "diffusion",
        "logic_net",
        "masked_room_model",
    )

    def __init__(self, engine: Any):
        self.engine = engine

    @property
    def device(self) -> torch.device:
        return getattr(self.engine, "device")

    def iter_models(self) -> Iterable[torch.nn.Module]:
        for name in self._MODEL_ATTRS:
            model = getattr(self.engine, name, None)
            if isinstance(model, torch.nn.Module):
                yield model

    def iter_named_models(self) -> Iterable[Tuple[str, torch.nn.Module]]:
        for name in self._MODEL_ATTRS:
            model = getattr(self.engine, name, None)
            if isinstance(model, torch.nn.Module):
                yield name, model

    def to(self, device: str | torch.device) -> "ModelManager":
        target = torch.device(device)
        self.engine.device = target
        for model in self.iter_models():
            model.to(target)
        return self

    def active_model_names(self) -> Set[str]:
        """Return model names that should stay on the runtime device."""
        active = {"vqvae", "condition_encoder", "logic_net"}
        mode = str(getattr(self.engine, "room_generator_mode", "latent_diffusion")).strip().lower()
        if mode == "discrete_masked":
            active.add("masked_room_model")
            if bool(getattr(self.engine, "default_masked_room_teacher_fallback_enabled", False)):
                active.add("diffusion")
        else:
            active.add("diffusion")
        return active

    def offload_inactive(
        self,
        active_names: Optional[Iterable[str]] = None,
        *,
        target_device: Optional[str | torch.device] = None,
        offload_device: str | torch.device = "cpu",
    ) -> "ModelManager":
        """Move inactive model backends off GPU while keeping active components resident."""
        target = torch.device(target_device) if target_device is not None else torch.device(getattr(self.engine, "device"))
        offload = torch.device(offload_device)
        active = set(active_names) if active_names is not None else self.active_model_names()
        for name, model in self.iter_named_models():
            model.to(target if name in active else offload)
        self.engine.device = target
        return self

    def eval(self) -> "ModelManager":
        for model in self.iter_models():
            model.eval()
        return self

    def set_precision(self, dtype: Optional[torch.dtype]) -> "ModelManager":
        if dtype is None:
            return self
        for model in self.iter_models():
            model.to(dtype=dtype)
        return self

    def load_vqvae(self, checkpoint_path: Optional[str]) -> torch.nn.Module:
        return load_vqvae(self.engine, checkpoint_path)

    def load_condition_encoder(self, checkpoint_path: Optional[str]) -> DualStreamConditionEncoder:
        return load_condition_encoder(self.engine, checkpoint_path)

    def load_diffusion(self, checkpoint_path: Optional[str]) -> LatentDiffusionModel:
        return load_diffusion(self.engine, checkpoint_path)

    def load_logic_net(self, checkpoint_path: Optional[str]) -> LogicNet:
        return load_logic_net(self.engine, checkpoint_path)

    def load_masked_room_model(self, checkpoint_path: Optional[str]):
        return load_masked_room_model(self.engine, checkpoint_path)

    @staticmethod
    def create_refiner(
        use_learned_rules: bool,
        *,
        max_repair_attempts: int = 5,
        margin: int = 2,
        adjacency_threshold: float = 0.01,
    ) -> SymbolicRefiner:
        return create_refiner(
            use_learned_rules,
            max_repair_attempts=max_repair_attempts,
            margin=margin,
            adjacency_threshold=adjacency_threshold,
        )

    @property
    def vqvae(self) -> Optional[torch.nn.Module]:
        return getattr(self.engine, "vqvae", None)

    @property
    def diffusion(self) -> Optional[torch.nn.Module]:
        return getattr(self.engine, "diffusion", None)

    @property
    def logic_net(self) -> Optional[torch.nn.Module]:
        return getattr(self.engine, "logic_net", None)

    @property
    def condition_encoder(self) -> Optional[torch.nn.Module]:
        return getattr(self.engine, "condition_encoder", None)


__all__ = [
    "ModelManager",
    "load_vqvae",
    "load_condition_encoder",
    "load_diffusion",
    "load_logic_net",
    "load_masked_room_model",
    "create_refiner",
]
