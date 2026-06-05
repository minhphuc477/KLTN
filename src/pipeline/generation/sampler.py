"""Sampling helpers for room-level neural generation."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch

from src.core import ROOM_HEIGHT, ROOM_WIDTH
from src.core.definitions import DOOR_POSITIONS
from src.core.neural_guided_repair import NeuralGuidedRepair
from src.core.vqvae import canonical_latent_shape
from src.pipeline.block_contracts import BlockShapeContract, validate_tensor_contract
from src.pipeline.repair_feedback import build_neighbor_boundary_inpaint_inputs
from src.pipeline.types import RoomGenerationResult
from src.utils.stable_seed import stable_seed_offset
from src.zelda_data.vglc_utils import validate_room_dimensions

logger = logging.getLogger(__name__)
DEFAULT_ROOM_LATENT_HW: Tuple[int, int] = canonical_latent_shape((ROOM_HEIGHT, ROOM_WIDTH))


def _configure_runtime_logic_guidance(pipeline, logic_guidance_scale: float) -> float:
    """Apply explicit runtime LogicNet guidance strategy to the diffusion model."""
    strategy = str(getattr(pipeline, "default_logic_guidance_strategy", "late") or "late").strip().lower()
    if strategy not in {"none", "late", "full"}:
        raise ValueError(f"default_logic_guidance_strategy must be 'none', 'late', or 'full', got {strategy!r}.")

    effective_scale = max(0.0, float(logic_guidance_scale))
    guidance = getattr(getattr(pipeline, "diffusion", None), "guidance", None)
    if guidance is None:
        return 0.0
    if strategy == "none" or effective_scale <= 0.0 or getattr(pipeline, "logic_net", None) is None:
        guidance.logic_net = None
        guidance.guidance_scale = 0.0
        return 0.0

    guidance.logic_net = pipeline.logic_net
    guidance.guidance_scale = effective_scale
    guidance.schedule_enabled = True
    if strategy == "full":
        guidance.active_fraction = 1.0
        pipeline._bump_diagnostic("logic_guidance_full_dpps_used")
    else:
        guidance.active_fraction = float(
            max(0.05, min(1.0, float(getattr(pipeline, "default_logic_guidance_active_fraction", 0.2))))
        )
        pipeline._bump_diagnostic("logic_guidance_late_dpps_used")
    return effective_scale


def _stable_node_seed_offset(node: Any) -> int:
    """Deterministic integer seed offset for arbitrary node-id types."""
    return stable_seed_offset(node, digest_size=4)


def _default_latent_shape_chw(pipeline, sampler_mode: str) -> Tuple[int, int, int]:
    """Default VQ latent shape without requiring diffusion for categorical sampling."""
    if getattr(pipeline, "room_generator_mode", None) == "discrete_masked":
        hidden_dim = int(getattr(getattr(pipeline, "masked_room_model", None), "hidden_dim", 64))
        return (hidden_dim, int(ROOM_HEIGHT), int(ROOM_WIDTH))
    diffusion = getattr(pipeline, "diffusion", None)
    latent_dim = getattr(diffusion, "latent_dim", None)
    if latent_dim is None and str(sampler_mode or "").strip().lower() == "categorical":
        vqvae = getattr(pipeline, "vqvae", None)
        latent_dim = getattr(vqvae, "latent_dim", None)
        if latent_dim is None:
            latent_dim = getattr(getattr(vqvae, "quantizer", object()), "embedding_dim", None)
    if latent_dim is None:
        diffusion = pipeline._require_component("diffusion", "_default_latent_shape_chw")
        latent_dim = getattr(diffusion, "latent_dim")
    return (
        int(latent_dim),
        int(DEFAULT_ROOM_LATENT_HW[0]),
        int(DEFAULT_ROOM_LATENT_HW[1]),
    )


def _infer_latent_shape_from_neighbors_or_default(
    pipeline,
    neighbor_latents: Dict[str, Optional[Any]],
    *,
    sampler_mode: str,
) -> Tuple[int, int, int, int]:
    """Infer rank-4 latent shape from neighbors, falling back by sampler mode."""
    for latent in neighbor_latents.values():
        if isinstance(latent, torch.Tensor) and latent.dim() == 4:
            return tuple(int(v) for v in latent.shape)  # type: ignore[return-value]
        if isinstance(latent, np.ndarray) and latent.ndim == 4:
            return tuple(int(v) for v in latent.shape)  # type: ignore[return-value]
    c, h, w = _default_latent_shape_chw(pipeline, sampler_mode)
    return (1, int(c), int(h), int(w))


@torch.no_grad()
def generate_room_batch(
    pipeline,
    *,
    room_ids: List[Any],
    mission_graph_physical: nx.Graph,
    graph_data: Dict[str, Any],
    generated_rooms: Dict[Any, RoomGenerationResult],
    room_latents: Dict[int, torch.Tensor],
    guidance_scale: float,
    logic_guidance_scale: float,
    num_diffusion_steps: int,
    use_fast_sampling: bool,
    latent_sampler: str,
    categorical_codebook_size: Optional[int],
    apply_repair: bool,
    seed: Optional[int],
    layer_offset: int,
    latent_shape_chw: Optional[Tuple[int, int, int]] = None,
) -> Dict[Any, RoomGenerationResult]:
    """Generate one dependency-safe room layer with batched diffusion decode."""
    if not room_ids:
        return {}

    sampler_mode = str(latent_sampler or "diffusion").strip().lower()
    pipeline._require_room_generation_components(
        "_generate_room_batch",
        latent_sampler=sampler_mode,
    )
    batch_conditions: List[torch.Tensor] = []
    per_room_inputs: List[Dict[str, Any]] = []

    for j, room_id in enumerate(room_ids):
        neighbor_latents = pipeline._normalize_neighbor_latents(
            pipeline._get_neighbor_latents(room_id, mission_graph_physical, room_latents)
        )
        reference_room_maps = (
            pipeline._get_neighbor_reference_room_maps(room_id, mission_graph_physical, generated_rooms)
            if bool(getattr(pipeline.condition_encoder, "use_reference_room_maps", False))
            else None
        )
        boundary_constraints = pipeline._build_room_boundary_constraints(
            graph=mission_graph_physical,
            room_id=room_id,
        )
        room_position = pipeline._build_room_position_tensor(
            graph=mission_graph_physical,
            room_id=room_id,
            fallback_order_index=layer_offset + j,
        )
        room_seed = None
        if seed is not None:
            room_seed = int(seed) + int(_stable_node_seed_offset(room_id))
        start_goal = pipeline._extract_room_start_goal(mission_graph_physical, room_id)

        room_graph_context = pipeline._build_room_graph_context(
            graph_data=graph_data,
            mission_graph=mission_graph_physical,
            room_id=room_id,
            start_goal=start_goal,
        )
        condition = pipeline._compute_room_condition(
            neighbor_latents=neighbor_latents,
            reference_room_maps=reference_room_maps,
            graph_context=room_graph_context,
            boundary_constraints=boundary_constraints,
            position=room_position,
        )
        batch_conditions.append(condition.detach())
        per_room_inputs.append(
            {
                'batch_index': int(j),
                'room_id': room_id,
                'neighbor_latents': neighbor_latents,
                'reference_room_maps': reference_room_maps,
                'graph_context': room_graph_context,
                'boundary_constraints': boundary_constraints,
                'position': room_position,
                'start_goal': start_goal,
                'seed': room_seed,
            }
        )

    if not batch_conditions or not per_room_inputs:
        return {}

    # Stack per-room conditions into one batch.
    expected_dim = int(batch_conditions[0].dim())
    if any(int(cond.dim()) != expected_dim for cond in batch_conditions):
        dims = [int(cond.dim()) for cond in batch_conditions]
        raise ValueError(f"Inconsistent condition tensor ranks inside batch: {dims}")
    condition_batch = torch.cat(batch_conditions, dim=0)
    first_room_graph_context = per_room_inputs[0]['graph_context']

    graph_ctx_for_guidance = {
        'graph_scope': 'dungeon',
        'node_features': graph_data.get('node_features'),
        'edge_index': graph_data.get('edge_index'),
        'edge_features': graph_data.get('edge_features'),
        'edge_rrwp': graph_data.get('edge_rrwp'),
        'tpe': graph_data.get('tpe'),
        'node_positions': graph_data.get('node_positions'),
        'node_mask': graph_data.get('node_mask'),
        'start_node_id': graph_data.get(
            'start_node_id',
            first_room_graph_context.get('start_node_id', 0),
        ),
        'target_idx': graph_data.get(
            'target_idx',
            first_room_graph_context.get('target_idx', -1),
        ),
        'key_lock_pairs': graph_data.get(
            'key_lock_pairs',
            first_room_graph_context.get('key_lock_pairs', []),
        ),
        'boundary_constraints': torch.cat(
            [inp['boundary_constraints'].to(pipeline.device, dtype=torch.float32) for inp in per_room_inputs],
            dim=0,
        ),
        'room_topology_map': pipeline._stack_room_topology_maps(
            [inp['graph_context']['room_topology_map'] for inp in per_room_inputs]
        ),
    }

    # Map each sampled room latent back to its dungeon graph node.
    node_to_idx = graph_data.get('node_to_idx')
    if isinstance(node_to_idx, dict) and room_ids:
        current_node_idx_batch = []
        for room_id in room_ids:
            idx = node_to_idx.get(room_id, -1)
            current_node_idx_batch.append(int(idx))
        if all(idx >= 0 for idx in current_node_idx_batch):
            graph_ctx_for_guidance['current_node_idx'] = torch.tensor(
                current_node_idx_batch,
                device=pipeline.device,
                dtype=torch.long,
            )

    if pipeline.use_current_node_distance_features:
        current_node_distance_batch: List[torch.Tensor] = []
        for inp in per_room_inputs:
            value = inp['graph_context']['current_node_distance']
            if not isinstance(value, torch.Tensor):
                continue
            tensor = value.to(pipeline.device, dtype=torch.float32)
            if tensor.dim() == 3 and int(tensor.shape[0]) == 1:
                tensor = tensor.squeeze(0)
            current_node_distance_batch.append(tensor.detach())
        if current_node_distance_batch:
            graph_ctx_for_guidance['current_node_distance'] = torch.stack(
                current_node_distance_batch,
                dim=0,
            )

    B = len(room_ids)
    if latent_shape_chw is None:
        latent_shape_chw = _default_latent_shape_chw(pipeline, sampler_mode)

    latent_shape: Tuple[int, int, int, int] = (
        B,
        int(latent_shape_chw[0]),
        int(latent_shape_chw[1]),
        int(latent_shape_chw[2]),
    )

    tokens_batch: Optional[torch.Tensor] = None
    if pipeline.room_generator_mode == "discrete_masked":
        fixed_layouts = [
            pipeline._build_masked_room_fixed_tokens(
                mission_graph_physical,
                inp['room_id'],
                start_goal=inp['start_goal'],
            )
            for inp in per_room_inputs
        ]
        fixed_tokens = torch.cat([layout[0] for layout in fixed_layouts], dim=0)
        fixed_mask = torch.cat([layout[1] for layout in fixed_layouts], dim=0)
        tokens_batch, logits_batch, z_batch = pipeline.masked_room_model.sample(
            context=condition_batch,
            graph_data=graph_ctx_for_guidance,
            fixed_tokens=fixed_tokens,
            fixed_mask=fixed_mask,
            num_steps=max(1, min(int(num_diffusion_steps), int(pipeline.masked_sampling_steps))),
            temperature=float(pipeline.default_masked_room_sampling_temperature),
            schedule_mode=pipeline.default_masked_room_sampling_schedule,
            stochastic=bool(pipeline.default_masked_room_sampling_stochastic),
            corrector_steps=int(pipeline.default_masked_room_corrector_steps),
            corrector_mask_ratio=float(pipeline.default_masked_room_corrector_mask_ratio),
            seed=seed,
        )
    elif sampler_mode == "categorical":
        guidance_scale, logic_guidance_scale = pipeline._resolve_effective_sampling_guidance(
            use_fast_sampling=False,
            guidance_scale=float(guidance_scale),
            logic_guidance_scale=float(logic_guidance_scale),
        )
        if getattr(pipeline, "diffusion", None) is not None:
            pipeline.diffusion.cfg_scale = float(guidance_scale)
        logic_guidance_scale = _configure_runtime_logic_guidance(pipeline, logic_guidance_scale)
        if hasattr(pipeline.vqvae, "codebook_size"):
            num_embeddings = int(getattr(pipeline.vqvae, "codebook_size"))
        else:
            num_embeddings = int(getattr(getattr(pipeline.vqvae, "quantizer", object()), "num_embeddings", 512))
        active_codebook_size = int(max(1, min(num_embeddings, int(categorical_codebook_size or num_embeddings))))

        probs = np.ones(active_codebook_size, dtype=np.float64)
        try:
            usage = pipeline.vqvae.get_codebook_usage()
            if isinstance(usage, torch.Tensor):
                usage_np = usage.detach().float().cpu().numpy()
                if usage_np.size >= active_codebook_size:
                    usage_np = np.asarray(usage_np[:active_codebook_size], dtype=np.float64)
                    if float(np.sum(usage_np)) > 0.0:
                        probs = usage_np
        except (AttributeError, RuntimeError, ValueError, TypeError):
            pass
        probs = probs / max(float(np.sum(probs)), 1e-9)

        sampled = []
        for inp in per_room_inputs:
            local_rng = np.random.default_rng(inp['seed']) if inp['seed'] is not None else np.random.default_rng()
            sampled.append(
                local_rng.choice(
                    active_codebook_size,
                    size=(latent_shape[2], latent_shape[3]),
                    p=probs,
                )
            )
        indices_t = torch.from_numpy(np.stack(sampled, axis=0)).to(pipeline.device, dtype=torch.long)
        logits_batch = pipeline.vqvae.decode_indices(indices_t)
        z_batch = pipeline.vqvae.quantizer.encode_indices(indices_t).permute(0, 3, 1, 2).contiguous()
    else:
        guidance_scale, logic_guidance_scale = pipeline._resolve_effective_sampling_guidance(
            use_fast_sampling=use_fast_sampling,
            guidance_scale=float(guidance_scale),
            logic_guidance_scale=float(logic_guidance_scale),
        )
        pipeline.diffusion.cfg_scale = float(guidance_scale)
        logic_guidance_scale = _configure_runtime_logic_guidance(pipeline, logic_guidance_scale)

        # Verify bucket uniformity for neighbor latent references.
        for inp in per_room_inputs:
            for latent in inp['neighbor_latents'].values():
                shape_here: Optional[Tuple[int, int, int]] = None
                if isinstance(latent, torch.Tensor) and latent.dim() == 4:
                    shape_here = (int(latent.shape[1]), int(latent.shape[2]), int(latent.shape[3]))
                elif isinstance(latent, np.ndarray) and latent.ndim == 4:
                    shape_here = (int(latent.shape[1]), int(latent.shape[2]), int(latent.shape[3]))
                if shape_here is not None and shape_here != tuple(latent_shape_chw):
                    raise ValueError(
                        f"Mixed latent shapes inside one batch: expected {latent_shape_chw}, got {shape_here}"
                    )
        training_objective = str(getattr(pipeline.diffusion, "training_objective", "diffusion")).strip().lower()
        use_flow_ode = training_objective == "flow_matching" and hasattr(pipeline.diffusion, "flow_ode_sample")
        if use_flow_ode:
            if use_fast_sampling:
                pipeline._bump_diagnostic("fast_sampling_unavailable_flow_matching")
            z_batch = pipeline.diffusion.flow_ode_sample(
                context=condition_batch,
                shape=latent_shape,
                graph_data=graph_ctx_for_guidance,
                num_steps=max(2, int(num_diffusion_steps)),
            )
            pipeline._bump_diagnostic("flow_ode_sampling_used")
        elif use_fast_sampling and pipeline.diffusion.supports_fast_sampling():
            z_batch = pipeline.diffusion.fast_sample(
                context=condition_batch,
                shape=latent_shape,
                graph_data=graph_ctx_for_guidance,
                guidance_scale=float(guidance_scale),
                seed=seed,
            )
            pipeline._bump_diagnostic("fast_sampling_used")
        else:
            if use_fast_sampling:
                pipeline._bump_diagnostic("fast_sampling_unavailable_fallback")
            z_batch = pipeline.diffusion.ddim_sample(
                context=condition_batch,
                shape=latent_shape,
                num_steps=max(1, int(num_diffusion_steps)),
                graph_data=graph_ctx_for_guidance,
            )

        if pipeline.use_latent_boundary_masking:
            for i, inp in enumerate(per_room_inputs):
                try:
                    z_ref, boundary_edit_mask, has_boundary_constraints = build_neighbor_boundary_inpaint_inputs(
                        base_latent=z_batch[i:i + 1],
                        neighbor_latents=inp['neighbor_latents'],
                        band=1,
                    )
                    if has_boundary_constraints:
                        room_graph_guidance = pipeline._slice_graph_guidance_batch(graph_ctx_for_guidance, i)
                        z_batch[i:i + 1] = pipeline.diffusion.inpaint(
                            x_0=z_ref,
                            mask=boundary_edit_mask,
                            context=condition_batch[i:i + 1],
                            graph_data=room_graph_guidance,
                            num_steps=max(8, int(num_diffusion_steps) // 2),
                        )
                except (AttributeError, RuntimeError, ValueError, TypeError):
                    continue
        logits_batch = pipeline._decode_latent_with_vqvae(z_batch)

    out: Dict[Any, RoomGenerationResult] = {}
    for i, inp in enumerate(per_room_inputs):
        if int(inp['batch_index']) != int(i):
            raise RuntimeError(
                f"Batch routing mismatch for room {inp['room_id']}: stored index={inp['batch_index']} actual={i}"
            )
        result_i = pipeline.generate_room(
            neighbor_latents=inp['neighbor_latents'],
            graph_context=inp['graph_context'],
            room_id=inp['room_id'],
            boundary_constraints=inp['boundary_constraints'],
            position=inp['position'],
            reference_room_maps=inp['reference_room_maps'],
            guidance_scale=guidance_scale,
            logic_guidance_scale=logic_guidance_scale,
            num_diffusion_steps=num_diffusion_steps,
            use_fast_sampling=use_fast_sampling,
            latent_sampler=latent_sampler,
            categorical_codebook_size=categorical_codebook_size,
            apply_repair=apply_repair,
            start_goal_coords=inp['start_goal'],
            seed=inp['seed'],
            precomputed_condition=condition_batch[i:i + 1],
            precomputed_latent=z_batch[i:i + 1],
            precomputed_logits=logits_batch[i:i + 1],
            precomputed_tokens=(
                tokens_batch[i:i + 1]
                if isinstance(tokens_batch, torch.Tensor)
                else None
            ),
        )
        out[inp['room_id']] = result_i

    return out

@torch.no_grad()
def generate_room(
    pipeline,
    neighbor_latents: Dict[str, Optional[Any]],
    graph_context: Dict[str, Any],
    room_id: int,
    boundary_constraints: Optional[torch.Tensor] = None,
    position: Optional[torch.Tensor] = None,
    reference_room_maps: Optional[Dict[str, Optional[torch.Tensor]]] = None,
    guidance_scale: Optional[float] = None,
    logic_guidance_scale: Optional[float] = None,
    num_diffusion_steps: Optional[int] = None,
    use_fast_sampling: Optional[bool] = None,
    latent_sampler: Optional[str] = None,
    categorical_codebook_size: Optional[int] = None,
    use_ddim: bool = True,
    apply_repair: Optional[bool] = None,
    start_goal_coords: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
    seed: Optional[int] = None,
    precomputed_condition: Optional[torch.Tensor] = None,
    precomputed_latent: Optional[torch.Tensor] = None,
    precomputed_logits: Optional[torch.Tensor] = None,
    precomputed_tokens: Optional[torch.Tensor] = None,
    allow_teacher_fallback: Optional[bool] = None,
    room_generator_override: Optional[str] = None,
) -> RoomGenerationResult:
    """
    Generate a single room using the full 7-block pipeline.

    Args:
        neighbor_latents: Dict of neighboring room latents {'N': tensor, ...}
        graph_context: Graph data dict with:
            - node_features: (num_nodes, feature_dim)
            - edge_index: (2, num_edges)
            - tpe: Topological positional encoding
            - current_node_idx: Index of current room in graph
        room_id: Unique room identifier
        boundary_constraints: (1, 8) door mask tensor
        position: (1, 2) grid position
        guidance_scale: Classifier-free guidance scale
        logic_guidance_scale: LogicNet gradient guidance scale
        num_diffusion_steps: Number of DDIM/DDPM steps
        use_fast_sampling: Use a configured distilled fast sampler when available
        latent_sampler: "diffusion" (default) or "categorical"
        categorical_codebook_size: Optional cap for categorical sampling
        use_ddim: Use DDIM (deterministic) vs DDPM (stochastic)
        apply_repair: Apply symbolic WFC repair
        start_goal_coords: ((start_r, start_c), (goal_r, goal_c)) for repair
        seed: Random seed for reproducibility

    Returns:
        RoomGenerationResult with room grid, latents, and metrics
    """
    local_np_rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    if seed is not None:
        torch.manual_seed(seed)
    neighbor_latents = pipeline._normalize_neighbor_latents(neighbor_latents)
    guidance_scale = pipeline.default_guidance_scale if guidance_scale is None else float(guidance_scale)
    logic_guidance_scale = (
        pipeline.default_logic_guidance_scale
        if logic_guidance_scale is None
        else float(logic_guidance_scale)
    )
    num_diffusion_steps = (
        pipeline.default_num_diffusion_steps if num_diffusion_steps is None else int(num_diffusion_steps)
    )
    use_fast_sampling = (
        pipeline.default_use_fast_sampling if use_fast_sampling is None else bool(use_fast_sampling)
    )
    latent_sampler = pipeline.default_latent_sampler if latent_sampler is None else str(latent_sampler)
    if categorical_codebook_size is None and pipeline.default_categorical_codebook_size is not None:
        categorical_codebook_size = int(pipeline.default_categorical_codebook_size)
    apply_repair = pipeline.default_apply_repair if apply_repair is None else bool(apply_repair)
    if start_goal_coords is None:
        start_goal_coords = pipeline.default_start_goal_coords
    elif start_goal_coords is not None:
        start_goal_coords = pipeline._normalize_start_goal_coords(start_goal_coords)
    effective_room_generator_mode = (
        pipeline.room_generator_mode
        if room_generator_override is None
        else str(room_generator_override).strip().lower()
    )
    if allow_teacher_fallback is None:
        if effective_room_generator_mode == "discrete_masked":
            allow_teacher_fallback = pipeline.default_masked_room_teacher_fallback_enabled
        else:
            allow_teacher_fallback = pipeline.default_fast_sampler_teacher_fallback_enabled
    else:
        allow_teacher_fallback = bool(allow_teacher_fallback)

    if logic_guidance_scale > 0 and pipeline.logic_net is None:
        pipeline._bump_diagnostic("logic_guidance_disabled_missing_component")
        logger.warning(
            "Logic guidance requested for room %s but no logic_net component is configured; disabling guidance.",
            room_id,
        )
        logic_guidance_scale = 0.0
    if apply_repair and pipeline.refiner is None:
        pipeline._bump_diagnostic("repair_disabled_missing_component")
        logger.warning(
            "Symbolic repair requested for room %s but no refiner component is configured; using neural output.",
            room_id,
        )
        apply_repair = False

    sampler_mode = str(latent_sampler or "diffusion").strip().lower()
    pipeline._require_room_generation_components(
        "generate_room",
        latent_sampler=sampler_mode,
        room_generator_mode=effective_room_generator_mode,
    )

    if precomputed_condition is not None:
        condition = precomputed_condition.to(pipeline.device)
    else:
        condition = pipeline._compute_room_condition(
            neighbor_latents=neighbor_latents,
            reference_room_maps=reference_room_maps,
            graph_context=graph_context,
            boundary_constraints=boundary_constraints,
            position=position,
        )

    graph_data = graph_context if isinstance(graph_context, dict) else None
    if graph_data is not None and boundary_constraints is not None and "boundary_constraints" not in graph_data:
        graph_data = {
            **graph_data,
            "boundary_constraints": boundary_constraints.to(pipeline.device, dtype=torch.float32),
        }
    mission_graph_for_room = graph_data.get("mission_graph") if isinstance(graph_data, dict) else None

    sampled_tokens: Optional[torch.Tensor] = None

    if precomputed_latent is not None and precomputed_logits is not None:
        z_latent = precomputed_latent.to(pipeline.device)
        logits = precomputed_logits.to(pipeline.device)
        if precomputed_tokens is not None:
            sampled_tokens = precomputed_tokens.to(pipeline.device, dtype=torch.long)
    elif effective_room_generator_mode == "discrete_masked":
        fixed_tokens = None
        fixed_mask = None
        if mission_graph_for_room is not None:
            fixed_tokens, fixed_mask = pipeline._build_masked_room_fixed_tokens(
                mission_graph_for_room,
                room_id,
                start_goal=start_goal_coords,
            )
        sampled_tokens, logits, z_latent = pipeline.masked_room_model.sample(
            context=condition,
            graph_data=graph_data,
            fixed_tokens=fixed_tokens,
            fixed_mask=fixed_mask,
            num_steps=max(1, min(int(num_diffusion_steps), int(pipeline.masked_sampling_steps))),
            temperature=float(pipeline.default_masked_room_sampling_temperature),
            schedule_mode=pipeline.default_masked_room_sampling_schedule,
            stochastic=bool(pipeline.default_masked_room_sampling_stochastic),
            corrector_steps=int(pipeline.default_masked_room_corrector_steps),
            corrector_mask_ratio=float(pipeline.default_masked_room_corrector_mask_ratio),
            seed=seed,
        )
    elif sampler_mode == "categorical":
        # Infer latent shape from neighbors when possible, otherwise use VQ-VAE spatial downsampling (x4).
        latent_shape = _infer_latent_shape_from_neighbors_or_default(
            pipeline,
            neighbor_latents,
            sampler_mode=sampler_mode,
        )
        logger.debug("Room %s: Sampling with categorical codebook path", room_id)
        latent_h = int(max(1, latent_shape[2]))
        latent_w = int(max(1, latent_shape[3]))
        if hasattr(pipeline.vqvae, "codebook_size"):
            num_embeddings = int(getattr(pipeline.vqvae, "codebook_size"))
        else:
            num_embeddings = int(getattr(getattr(pipeline.vqvae, "quantizer", object()), "num_embeddings", 512))
        active_codebook_size = int(max(1, min(num_embeddings, int(categorical_codebook_size or num_embeddings))))

        probs = np.ones(active_codebook_size, dtype=np.float64)
        try:
            usage = pipeline.vqvae.get_codebook_usage()
            if isinstance(usage, torch.Tensor):
                usage_np = usage.detach().float().cpu().numpy()
                if usage_np.size >= active_codebook_size:
                    usage_np = np.asarray(usage_np[:active_codebook_size], dtype=np.float64)
                    if float(np.sum(usage_np)) > 0.0:
                        probs = usage_np
        except (AttributeError, RuntimeError, ValueError, TypeError) as e:
            pipeline._bump_diagnostic("categorical_prior_fallback")
            logger.debug("Falling back to uniform categorical priors (codebook usage unavailable): %s", e)
        probs = np.asarray(probs, dtype=np.float64)
        probs = probs / max(float(np.sum(probs)), 1e-9)

        sampled_indices = local_np_rng.choice(
            active_codebook_size,
            size=(1, latent_h, latent_w),
            p=probs,
        )
        indices_t = torch.from_numpy(sampled_indices).to(pipeline.device, dtype=torch.long)
        logits = pipeline.vqvae.decode_indices(indices_t)  # (1, 44, 16, 11)
        with torch.no_grad():
            z_latent = pipeline.vqvae.quantizer.encode_indices(indices_t).permute(0, 3, 1, 2).contiguous()
        validate_tensor_contract(
            z_latent,
            BlockShapeContract(name='block_iv_categorical_latent', dims=4, batch_dim=1),
        )
    else:
        # BLOCK V: Logic guidance configuration for diffusion sampler
        guidance_scale, logic_guidance_scale = pipeline._resolve_effective_sampling_guidance(
            use_fast_sampling=use_fast_sampling,
            guidance_scale=float(guidance_scale),
            logic_guidance_scale=float(logic_guidance_scale),
        )
        pipeline.diffusion.cfg_scale = float(guidance_scale)
        logic_guidance_scale = _configure_runtime_logic_guidance(pipeline, logic_guidance_scale)

        # Infer latent shape from neighbors when possible, otherwise use VQ-VAE spatial downsampling (x4).
        latent_shape = _infer_latent_shape_from_neighbors_or_default(
            pipeline,
            neighbor_latents,
            sampler_mode=sampler_mode,
        )

        # BLOCK IV: Latent Diffusion Sampling
        logger.debug(f"Room {room_id}: Sampling with {num_diffusion_steps} steps")
        training_objective = str(getattr(pipeline.diffusion, "training_objective", "diffusion")).strip().lower()
        use_flow_ode = training_objective == "flow_matching" and hasattr(pipeline.diffusion, "flow_ode_sample")
        if use_flow_ode:
            if use_fast_sampling:
                pipeline._bump_diagnostic("fast_sampling_unavailable_flow_matching")
            z_latent = pipeline.diffusion.flow_ode_sample(
                context=condition,
                shape=latent_shape,
                graph_data=graph_data,
                num_steps=max(2, int(num_diffusion_steps)),
            )
            pipeline._bump_diagnostic("flow_ode_sampling_used")
        elif use_fast_sampling and pipeline.diffusion.supports_fast_sampling():
            z_latent = pipeline.diffusion.fast_sample(
                context=condition,
                shape=latent_shape,
                graph_data=graph_data,
                guidance_scale=float(guidance_scale),
                seed=seed,
            )
            pipeline._bump_diagnostic("fast_sampling_used")
        elif use_ddim:
            if use_fast_sampling:
                pipeline._bump_diagnostic("fast_sampling_unavailable_fallback")
            z_latent = pipeline.diffusion.ddim_sample(
                context=condition,
                shape=latent_shape,
                num_steps=max(1, int(num_diffusion_steps)),
                graph_data=graph_data,
            )
        else:
            if use_fast_sampling:
                pipeline._bump_diagnostic("fast_sampling_unavailable_fallback")
            z_latent = pipeline.diffusion.sample(
                context=condition,
                shape=latent_shape,
                graph_data=graph_data,
            )

        validate_tensor_contract(
            z_latent,
            BlockShapeContract(
                name='block_iv_diffusion_latent',
                dims=4,
                batch_dim=1,
                channel_dim=int(pipeline.diffusion.latent_dim),
            ),
        )

        # Autoregressive spatial generation: preserve known boundary latents from generated neighbors.
        if pipeline.use_latent_boundary_masking:
            try:
                z_ref, boundary_edit_mask, has_boundary_constraints = build_neighbor_boundary_inpaint_inputs(
                    base_latent=z_latent,
                    neighbor_latents=neighbor_latents,
                    band=1,
                )
                if has_boundary_constraints:
                    z_latent = pipeline.diffusion.inpaint(
                        x_0=z_ref,
                        mask=boundary_edit_mask,
                        context=condition,
                        graph_data=graph_data,
                        num_steps=max(8, int(num_diffusion_steps) // 2),
                        noise_strength=0.25,  # Lower noise for boundary blending (not full regeneration)
                    )
                    pipeline._bump_diagnostic("boundary_latent_masking_applied")
            except (AttributeError, RuntimeError, ValueError, TypeError) as e:
                pipeline._bump_diagnostic("boundary_latent_masking_fallback")
                logger.debug("Boundary latent masking skipped due to error: %s", e)

        # BLOCK II: VQ-VAE Decoding
        logits = pipeline._decode_latent_with_vqvae(z_latent)  # (1, 44, 16, 11)
    validate_tensor_contract(
        logits,
        BlockShapeContract(
            name='block_ii_decode_logits',
            dims=4,
            batch_dim=1,
            channel_dim=int(getattr(pipeline.vqvae, "num_classes", logits.shape[1])),
            spatial_hw=(ROOM_HEIGHT, ROOM_WIDTH),
        ),
    )

    # BLOCK II.a: Topology-Enforced Constrained Decoding
    # Clamp doorway logits to the exact door type implied by graph semantics
    # before argmax. This keeps the topology constraint inside the decoder
    # instead of stamping a mismatched discrete tile after generation.
    door_tiles_forced = 0
    if isinstance(mission_graph_for_room, nx.Graph) and room_id in mission_graph_for_room:
        try:
            neg_large = float(-1e4)
            pos_large = float(1e4)
            semantics = pipeline._extract_room_topology_semantics(mission_graph_for_room, room_id)
            required_doors = semantics.get("required_doors", {})
            for direction, is_required in required_doors.items():
                if not is_required:
                    continue
                spec = DOOR_POSITIONS.get(direction)
                if spec is None:
                    continue
                door_tile = int(
                    pipeline._edge_tokens_to_door_tile(
                        semantics.get("edge_constraints", {}).get(direction, set())
                    )
                )

                if direction in {"N", "S"}:
                    row = int(max(0, min(ROOM_HEIGHT - 1, spec["row"])))
                    col_start = int(max(0, min(ROOM_WIDTH - 1, spec["col_start"])))
                    col_end = int(max(0, min(ROOM_WIDTH - 1, spec["col_end"])))
                    for c in range(col_start, col_end + 1):
                        if int(logits[0, :, row, c].argmax()) != door_tile:
                            logits[0, :, row, c] = neg_large
                            logits[0, door_tile, row, c] = pos_large
                            door_tiles_forced += 1
                else:
                    col = int(max(0, min(ROOM_WIDTH - 1, spec["col"])))
                    row_start = int(max(0, min(ROOM_HEIGHT - 1, spec["row_start"])))
                    row_end = int(max(0, min(ROOM_HEIGHT - 1, spec["row_end"])))
                    for r in range(row_start, row_end + 1):
                        if int(logits[0, :, r, col].argmax()) != door_tile:
                            logits[0, :, r, col] = neg_large
                            logits[0, door_tile, r, col] = pos_large
                            door_tiles_forced += 1
        except (AttributeError, RuntimeError, ValueError, TypeError) as e:
            logger.debug("Topology door constrained decoding skipped for room %s: %s", room_id, e)

    if door_tiles_forced > 0:
        pipeline._bump_diagnostic("topology_door_tiles_forced")
        logger.debug("Room %s: forced %d door tiles via constrained decoding", room_id, door_tiles_forced)

    semantic_decode_stats = pipeline._apply_semantic_constrained_decoding(
        logits,
        graph=mission_graph_for_room,
        room_id=room_id,
        start_goal=start_goal_coords,
    )
    if int(semantic_decode_stats.get("biased_slots", 0)) > 0:
        pipeline._bump_diagnostic("semantic_constrained_decode_applied")
        logger.debug(
            "Room %s: biased %d/%d planned graph-marker slots via semantic constrained decoding",
            room_id,
            int(semantic_decode_stats.get("biased_slots", 0)),
            int(semantic_decode_stats.get("planned_markers", 0)),
        )

    if effective_room_generator_mode == "discrete_masked" and sampled_tokens is not None:
        neural_grid = sampled_tokens.detach().cpu().numpy()[0].astype(np.int32, copy=False)
    else:
        neural_grid = logits.argmax(dim=1).detach().cpu().numpy()[0]  # (16, 11)
    raw_neural_grid = np.asarray(neural_grid, dtype=np.int32).copy()
    neural_grid, neural_invalid_count, neural_invalid_ids = pipeline._sanitize_semantic_grid(
        neural_grid,
        strip_void=True,
    )
    if neural_invalid_count > 0:
        pipeline._bump_diagnostic("neural_invalid_tile_ids_sanitized")
        logger.warning(
            "Room %s neural decode produced invalid tile IDs %s (count=%d); sanitized.",
            room_id,
            neural_invalid_ids,
            neural_invalid_count,
        )
    neural_grid, neural_semantic_strip_count, neural_semantic_strip_ids, neural_semantic_preserved_count, neural_semantic_preserved_ids = pipeline._strip_volatile_room_semantics(
        neural_grid,
        graph=mission_graph_for_room,
        room_id=room_id,
        start_goal=start_goal_coords,
    )
    if neural_semantic_strip_count > 0:
        pipeline._bump_diagnostic("neural_room_semantics_stripped")
        logger.debug(
            "Room %s stripped %d volatile semantic tiles from neural output: %s",
            room_id,
            neural_semantic_strip_count,
            neural_semantic_strip_ids,
        )
    if neural_semantic_preserved_count > 0:
        pipeline._bump_diagnostic("neural_graph_semantic_hints_salvaged")
        logger.debug(
            "Room %s preserved %d graph-owned semantic hints from neural output: %s",
            room_id,
            neural_semantic_preserved_count,
            neural_semantic_preserved_ids,
        )
    neural_structural_cleanup = {
        "invalid_door_tiles_removed": 0,
        "interior_obstacle_tiles_removed": 0,
        "interior_obstacle_components_removed": 0,
    }
    if effective_room_generator_mode == "latent_diffusion":
        neural_grid, neural_structural_cleanup = pipeline._strip_structural_room_artifacts(
            neural_grid,
            graph=mission_graph_for_room,
            room_id=room_id,
        )
        if any(int(v) > 0 for v in neural_structural_cleanup.values()):
            pipeline._bump_diagnostic("neural_structural_artifacts_stripped")
            logger.debug(
                "Room %s stripped structural artifacts from neural output: %s",
                room_id,
                neural_structural_cleanup,
            )
    neural_probs = logits.softmax(dim=1).detach().cpu().numpy()[0]  # (44, 16, 11)

    # BLOCK III: Removed (Migrated to Block II.a Constrained Decoding)

    # BLOCK VI: Symbolic Repair (if enabled)
    was_repaired = False
    repair_mask = None
    room_plan_mask = None
    final_grid = neural_grid.copy()
    repaired_invalid_count = 0
    repaired_invalid_ids: List[int] = []
    repaired_semantic_strip_count = 0
    repaired_semantic_strip_ids: List[int] = []
    repaired_semantic_preserved_count = 0
    repaired_semantic_preserved_ids: List[int] = []
    repair_time_sec = 0.0
    neural_boundary_shell = {
        "boundary_wall_tiles_forced": 0,
        "boundary_door_tiles_forced": 0,
        "interior_door_apron_tiles_forced": 0,
    }
    repaired_boundary_shell = {
        "boundary_wall_tiles_forced": 0,
        "boundary_door_tiles_forced": 0,
        "interior_door_apron_tiles_forced": 0,
    }
    neural_puzzle_scaffold = {
        "applied": 0,
        "tiles_added": 0,
        "segments_added": 0,
        "existing_structure_tiles": 0,
        "planned_route_pixels": 0,
    }
    final_puzzle_scaffold = {
        "applied": 0,
        "tiles_added": 0,
        "segments_added": 0,
        "existing_structure_tiles": 0,
        "planned_route_pixels": 0,
    }
    neural_no_puzzle_structure_cleanup = {
        "applied": 0,
        "block_tiles_removed": 0,
        "block_components_removed": 0,
    }
    final_no_puzzle_structure_cleanup = {
        "applied": 0,
        "block_tiles_removed": 0,
        "block_components_removed": 0,
    }
    repaired_structural_cleanup = {
        "invalid_door_tiles_removed": 0,
        "interior_obstacle_tiles_removed": 0,
        "interior_obstacle_components_removed": 0,
    }
    repair_diag: Dict[str, Any] = {}
    normalized_start_goal: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    if start_goal_coords is not None:
        normalized_start_goal = pipeline._normalize_start_goal_coords(start_goal_coords)

    if apply_repair and start_goal_coords is not None:
        start, goal = normalized_start_goal if normalized_start_goal is not None else pipeline._normalize_start_goal_coords(start_goal_coords)
        repair_started_at = time.perf_counter()
        try:
            if isinstance(mission_graph_for_room, nx.Graph) and room_id in mission_graph_for_room:
                room_plan_mask = pipeline._build_room_plan_trace(
                    mission_graph_for_room,
                    room_id,
                    neural_grid,
                    start_goal=(start, goal),
                )
        except (AttributeError, RuntimeError, ValueError, TypeError):
            room_plan_mask = None
        try:
            neural_guided_repair = None
            if (
                bool(getattr(pipeline, "default_use_neural_guided_repair", True))
                and getattr(pipeline, "logic_net", None) is not None
                and getattr(pipeline, "refiner", None) is not None
            ):
                neural_guided_repair = NeuralGuidedRepair(
                    logic_net=pipeline.logic_net,
                    refiner=pipeline.refiner,
                    use_neural_feedback=bool(getattr(pipeline, "default_use_neural_repair_feedback", True)),
                    repair_inpaint_noise_strength=float(getattr(pipeline, "default_repair_inpaint_noise_strength", 0.5)),
                    repair_inpaint_guidance_scale_multiplier=float(getattr(pipeline, "default_repair_inpaint_guidance_scale_multiplier", 1.0)),
                )

            if neural_guided_repair is not None:
                try:
                    repaired_grid, success, repair_diag = neural_guided_repair.repair_room_with_neural_guidance(
                        grid=neural_grid,
                        start=start,
                        goal=goal,
                        tile_logits=logits.detach(),
                        graph_data=graph_data,
                        required_floor_mask=room_plan_mask,
                        inpaint_callback=getattr(pipeline, "_logicnet_guided_inpaint_room", None),
                        inpaint_context=condition,
                        num_diffusion_steps=max(12, int(num_diffusion_steps) // 2),
                        seed=seed,
                    )
                    pipeline._bump_diagnostic("neural_guided_repair_used")
                except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
                    pipeline._bump_diagnostic("neural_guided_repair_fallback")
                    logger.debug("Room %s neural-guided repair failed; falling back to symbolic repair: %s", room_id, exc)
                    repaired_grid, success, repair_diag = pipeline.repair_room(
                        grid=neural_grid,
                        start=start,
                        goal=goal,
                        required_floor_mask=room_plan_mask,
                        feedback_callback=None,
                        max_feedback_rounds=0,
                        seed=seed,
                    )
            else:
                repaired_grid, success, repair_diag = pipeline.repair_room(
                    grid=neural_grid,
                    start=start,
                    goal=goal,
                    required_floor_mask=room_plan_mask,
                    feedback_callback=None,
                    max_feedback_rounds=0,
                    seed=seed,
                )

            if success:
                repaired_grid_raw = repaired_grid.copy()
                repaired_grid, repaired_invalid_count, repaired_invalid_ids = pipeline._sanitize_semantic_grid(
                    repaired_grid,
                    fallback_grid=neural_grid,
                    strip_void=True,
                )
                if repaired_invalid_count > 0:
                    pipeline._bump_diagnostic("repair_invalid_tile_ids_sanitized")
                    logger.warning(
                        "Room %s repair produced invalid tile IDs %s (count=%d); replaced using neural fallback.",
                        room_id,
                        repaired_invalid_ids,
                        repaired_invalid_count,
                    )
                    logger.debug(
                        "Room %s neural grid before repair:\n%s",
                        room_id,
                        np.array2string(neural_grid, max_line_width=240),
                    )
                    logger.debug(
                        "Room %s repaired grid before sanitize:\n%s",
                        room_id,
                        np.array2string(repaired_grid_raw, max_line_width=240),
                    )
                repaired_grid, repaired_semantic_strip_count, repaired_semantic_strip_ids, repaired_semantic_preserved_count, repaired_semantic_preserved_ids = (
                    pipeline._strip_volatile_room_semantics(
                        repaired_grid,
                        graph=mission_graph_for_room,
                        room_id=room_id,
                        start_goal=normalized_start_goal if normalized_start_goal is not None else start_goal_coords,
                    )
                )
                if repaired_semantic_strip_count > 0:
                    pipeline._bump_diagnostic("repair_room_semantics_stripped")
                    logger.debug(
                        "Room %s stripped %d volatile semantic tiles after repair: %s",
                        room_id,
                        repaired_semantic_strip_count,
                        repaired_semantic_strip_ids,
                    )
                if repaired_semantic_preserved_count > 0:
                    pipeline._bump_diagnostic("repair_graph_semantic_hints_salvaged")
                    logger.debug(
                        "Room %s preserved %d graph-owned semantic hints after repair: %s",
                        room_id,
                        repaired_semantic_preserved_count,
                        repaired_semantic_preserved_ids,
                    )
                if effective_room_generator_mode == "latent_diffusion":
                    repaired_grid, repaired_structural_cleanup = pipeline._strip_structural_room_artifacts(
                        repaired_grid,
                        graph=mission_graph_for_room,
                        room_id=room_id,
                    )
                    if any(int(v) > 0 for v in repaired_structural_cleanup.values()):
                        pipeline._bump_diagnostic("repair_structural_artifacts_stripped")
                        logger.debug(
                            "Room %s stripped structural artifacts after repair: %s",
                            room_id,
                            repaired_structural_cleanup,
                        )
                repair_mask = (repaired_grid != neural_grid)
                final_grid = repaired_grid
                was_repaired = bool(np.any(repair_mask))
                logger.debug(f"Room {room_id}: Repair successful ({np.sum(repair_mask)} tiles changed)")
            else:
                logger.warning(f"Room {room_id}: Repair failed, using neural output")
            pipeline._bump_diagnostic("wfc_feedback_attempts")
        except (AttributeError, RuntimeError, ValueError, TypeError) as e:
            pipeline._bump_diagnostic("room_repair_exception")
            logger.error(f"Room {room_id}: Repair error: {e}")
        finally:
            repair_time_sec = float(time.perf_counter() - repair_started_at)
    elif start_goal_coords is not None:
        start, goal = normalized_start_goal if normalized_start_goal is not None else pipeline._normalize_start_goal_coords(start_goal_coords)
        try:
            if isinstance(mission_graph_for_room, nx.Graph) and room_id in mission_graph_for_room:
                room_plan_mask = pipeline._build_room_plan_trace(
                    mission_graph_for_room,
                    room_id,
                    neural_grid,
                    start_goal=(start, goal),
                )
        except (AttributeError, RuntimeError, ValueError, TypeError):
            room_plan_mask = None

    neural_grid, neural_boundary_shell = pipeline._enforce_room_boundary_shell(
        neural_grid,
        graph=mission_graph_for_room,
        room_id=room_id,
    )
    final_grid, repaired_boundary_shell = pipeline._enforce_room_boundary_shell(
        final_grid,
        graph=mission_graph_for_room,
        room_id=room_id,
    )
    if any(int(v) > 0 for v in neural_boundary_shell.values()):
        pipeline._bump_diagnostic("neural_boundary_shell_enforced")
        logger.debug(
            "Room %s enforced boundary shell on neural output: %s",
            room_id,
            neural_boundary_shell,
        )
    if any(int(v) > 0 for v in repaired_boundary_shell.values()):
        pipeline._bump_diagnostic("final_boundary_shell_enforced")
        logger.debug(
            "Room %s enforced boundary shell on final output: %s",
            room_id,
            repaired_boundary_shell,
        )

    overlay_start_goal = normalized_start_goal
    if overlay_start_goal is None and isinstance(mission_graph_for_room, nx.Graph) and room_id in mission_graph_for_room:
        overlay_start_goal = pipeline._extract_room_start_goal(mission_graph_for_room, room_id)

    neural_grid, _, _, neural_post_boundary_preserved_count, neural_post_boundary_preserved_ids = (
        pipeline._strip_volatile_room_semantics(
            neural_grid,
            graph=mission_graph_for_room,
            room_id=room_id,
            start_goal=overlay_start_goal,
        )
    )
    if neural_post_boundary_preserved_count > 0:
        pipeline._bump_diagnostic("neural_post_boundary_graph_semantic_hints_salvaged")
        logger.debug(
            "Room %s re-salvaged %d graph-owned semantic hints after boundary enforcement: %s",
            room_id,
            neural_post_boundary_preserved_count,
            neural_post_boundary_preserved_ids,
        )

    final_grid, _, _, final_post_boundary_preserved_count, final_post_boundary_preserved_ids = (
        pipeline._strip_volatile_room_semantics(
            final_grid,
            graph=mission_graph_for_room,
            room_id=room_id,
            start_goal=overlay_start_goal,
        )
    )
    if final_post_boundary_preserved_count > 0:
        pipeline._bump_diagnostic("final_post_boundary_graph_semantic_hints_salvaged")
        logger.debug(
            "Room %s re-salvaged %d graph-owned semantic hints on final grid after boundary enforcement: %s",
            room_id,
            final_post_boundary_preserved_count,
            final_post_boundary_preserved_ids,
        )

    neural_grid, neural_void_cleanup = pipeline._strip_room_void_tiles(neural_grid)
    final_grid, final_void_cleanup = pipeline._strip_room_void_tiles(final_grid)
    if any(int(v) > 0 for v in neural_void_cleanup.values()):
        pipeline._bump_diagnostic("neural_void_tiles_stripped")
        logger.debug(
            "Room %s stripped VOID tiles from neural output: %s",
            room_id,
            neural_void_cleanup,
        )
    if any(int(v) > 0 for v in final_void_cleanup.values()):
        pipeline._bump_diagnostic("final_void_tiles_stripped")
        logger.debug(
            "Room %s stripped VOID tiles from final output: %s",
            room_id,
            final_void_cleanup,
        )

    neural_grid, neural_puzzle_scaffold = pipeline._apply_puzzle_room_scaffold(
        neural_grid,
        graph=mission_graph_for_room,
        room_id=room_id,
        room_plan_mask=room_plan_mask,
        start_goal=overlay_start_goal,
    )
    final_grid, final_puzzle_scaffold = pipeline._apply_puzzle_room_scaffold(
        final_grid,
        graph=mission_graph_for_room,
        room_id=room_id,
        room_plan_mask=room_plan_mask,
        start_goal=overlay_start_goal,
    )
    if int(final_puzzle_scaffold.get("applied", 0)) > 0:
        pipeline._commit_puzzle_novelty_choice(
            room_id=room_id,
            scaffold_stats=final_puzzle_scaffold,
        )
        pipeline._bump_diagnostic("puzzle_room_scaffold_applied")
        puzzle_archetype = str(final_puzzle_scaffold.get("archetype", "")).strip().lower()
        if puzzle_archetype:
            pipeline._bump_diagnostic(f"puzzle_room_scaffold_{puzzle_archetype}")
        puzzle_gate_family = str(final_puzzle_scaffold.get("gate_family", "")).strip().lower()
        if puzzle_gate_family:
            pipeline._bump_diagnostic(f"puzzle_room_scaffold_gate_{puzzle_gate_family}")
        if str(final_puzzle_scaffold.get("variant_name", "")).strip():
            pipeline._bump_diagnostic("puzzle_room_scaffold_novelty_selected")
        logger.debug(
            "Room %s applied puzzle scaffold: %s",
            room_id,
            final_puzzle_scaffold,
        )
    if int(final_puzzle_scaffold.get("contract_valid", 0) or 0) > 0:
        pipeline._bump_diagnostic("puzzle_room_contract_valid")
    elif str(final_puzzle_scaffold.get("gate_family", "")).strip():
        pipeline._bump_diagnostic("puzzle_room_contract_invalid")
    if int(final_puzzle_scaffold.get("contract_gate_skipped", 0) or 0) > 0:
        pipeline._bump_diagnostic("puzzle_room_contract_gate_skipped")
    if int(final_puzzle_scaffold.get("interaction_valid", 0) or 0) > 0:
        pipeline._bump_diagnostic("puzzle_room_interaction_valid")
    elif str(final_puzzle_scaffold.get("gate_family", "")).strip():
        pipeline._bump_diagnostic("puzzle_room_interaction_invalid")
    if int(final_puzzle_scaffold.get("interaction_gate_skipped", 0) or 0) > 0:
        pipeline._bump_diagnostic("puzzle_room_interaction_gate_skipped")
    if int(final_puzzle_scaffold.get("interaction_sequence_valid", 0) or 0) > 0:
        pipeline._bump_diagnostic("puzzle_room_sequence_valid")
    elif int(final_puzzle_scaffold.get("interaction_sequence_required", 0) or 0) > 0:
        pipeline._bump_diagnostic("puzzle_room_sequence_invalid")
    if int(final_puzzle_scaffold.get("sequence_gate_skipped", 0) or 0) > 0:
        pipeline._bump_diagnostic("puzzle_room_sequence_gate_skipped")
    if int(final_puzzle_scaffold.get("quality_gate_skipped", 0) or 0) > 0:
        pipeline._bump_diagnostic("puzzle_room_quality_gate_skipped")

    if not bool(pipeline.default_puzzle_room_structure_enabled):
        neural_grid, neural_no_puzzle_structure_cleanup = pipeline._strip_room_block_structure(
            neural_grid,
            graph=mission_graph_for_room,
            room_id=room_id,
        )
        final_grid, final_no_puzzle_structure_cleanup = pipeline._strip_room_block_structure(
            final_grid,
            graph=mission_graph_for_room,
            room_id=room_id,
        )
        if int(final_no_puzzle_structure_cleanup.get("applied", 0)) > 0:
            pipeline._bump_diagnostic("no_puzzle_block_structure_stripped")

    neural_pre_marker_grid = np.asarray(neural_grid, dtype=np.int32).copy()
    final_pre_marker_grid = np.asarray(final_grid, dtype=np.int32).copy()
    neural_marker_plan = pipeline._plan_room_graph_marker_layout(
        neural_pre_marker_grid,
        graph=mission_graph_for_room,
        room_id=room_id,
        start_goal=overlay_start_goal,
    )
    final_marker_plan = pipeline._plan_room_graph_marker_layout(
        final_pre_marker_grid,
        graph=mission_graph_for_room,
        room_id=room_id,
        start_goal=overlay_start_goal,
    )

    if bool(pipeline.default_deterministic_graph_marker_overlay_enabled):
        neural_grid, neural_marker_count, neural_marker_ids = pipeline._overlay_room_graph_markers(
            neural_grid,
            graph=mission_graph_for_room,
            room_id=room_id,
            start_goal=overlay_start_goal,
        )
        final_grid, final_marker_count, final_marker_ids = pipeline._overlay_room_graph_markers(
            final_grid,
            graph=mission_graph_for_room,
            room_id=room_id,
            start_goal=overlay_start_goal,
        )
        if final_marker_count > 0:
            logger.debug(
                "Room %s placed %d graph-owned semantic markers: %s",
                room_id,
                final_marker_count,
                final_marker_ids,
            )
    else:
        neural_marker_count = 0
        final_marker_count = 0
        neural_marker_ids = []
        final_marker_ids = []
        pipeline._bump_diagnostic("deterministic_graph_marker_overlay_disabled")

    neural_marker_alignment = pipeline._measure_room_graph_marker_alignment(
        neural_pre_marker_grid,
        placements=neural_marker_plan,
        prefix="neural_",
    )
    final_pre_overlay_alignment = pipeline._measure_room_graph_marker_alignment(
        final_pre_marker_grid,
        placements=final_marker_plan,
        prefix="final_pre_overlay_",
    )
    final_post_overlay_alignment = pipeline._measure_room_graph_marker_alignment(
        final_grid,
        placements=final_marker_plan,
        prefix="final_post_overlay_",
    )
    final_marker_overwrites = sum(
        int(final_pre_marker_grid[int(slot[0]), int(slot[1])]) != int(tile_id)
        for tile_id, slot in final_marker_plan
    )
    final_marker_expected = max(1, len(final_marker_plan))
    final_marker_overwrite_rate = float(final_marker_overwrites) / float(final_marker_expected)
    room_puzzle_metadata = pipeline._build_room_puzzle_metadata(
        grid=final_grid,
        graph=mission_graph_for_room,
        room_id=room_id,
        start_goal=overlay_start_goal,
        marker_plan=final_marker_plan,
        scaffold_stats=final_puzzle_scaffold,
    )

    # VGLC Compliance: Validate room dimensions
    valid_dims, dim_msg = validate_room_dimensions(final_grid)
    if not valid_dims:
        logger.error(f"Room {room_id}: VGLC dimension validation FAILED: {dim_msg}")
        raise ValueError(f"Generated room has invalid dimensions: {dim_msg}")
    else:
        logger.debug(f"Room {room_id}: VGLC dimension validation PASSED")

    # Compute metrics
    entropy_val = float(
        np.mean(
            -(
                logits.softmax(dim=1).detach()
                * logits.log_softmax(dim=1).detach()
            ).sum(dim=1).cpu().numpy()
        )
    )
    latent_cpu = z_latent.detach().to(device='cpu', dtype=torch.float32).contiguous()

    metrics = {
        'room_id': room_id,
        'neural_grid_entropy': entropy_val,
        'was_repaired': was_repaired,
        'repair_count': int(bool(was_repaired)),
        'repair_time_sec': float(repair_time_sec),
        'tiles_changed': int(np.sum(repair_mask)) if repair_mask is not None else 0,
        'raw_neural_to_cleaned_tiles_changed': int(np.sum(raw_neural_grid != neural_grid)),
        'raw_neural_to_final_tiles_changed': int(np.sum(raw_neural_grid != final_grid)),
        'neural_invalid_tile_ids': int(neural_invalid_count),
        'repair_invalid_tile_ids': int(repaired_invalid_count),
        'neural_semantic_tiles_stripped': int(neural_semantic_strip_count),
        'neural_graph_semantic_hints_salvaged': int(neural_semantic_preserved_count),
        'repair_semantic_tiles_stripped': int(repaired_semantic_strip_count),
        'repair_graph_semantic_hints_salvaged': int(repaired_semantic_preserved_count),
        'neural_invalid_door_tiles_removed': int(neural_structural_cleanup['invalid_door_tiles_removed']),
        'neural_interior_obstacle_tiles_removed': int(neural_structural_cleanup['interior_obstacle_tiles_removed']),
        'neural_interior_obstacle_components_removed': int(neural_structural_cleanup['interior_obstacle_components_removed']),
        'neural_boundary_void_tiles_removed': int(neural_void_cleanup['boundary_void_tiles_removed']),
        'neural_interior_void_tiles_removed': int(neural_void_cleanup['interior_void_tiles_removed']),
        'neural_boundary_wall_tiles_forced': int(neural_boundary_shell['boundary_wall_tiles_forced']),
        'neural_boundary_door_tiles_forced': int(neural_boundary_shell['boundary_door_tiles_forced']),
        'neural_interior_door_apron_tiles_forced': int(neural_boundary_shell['interior_door_apron_tiles_forced']),
        'repair_invalid_door_tiles_removed': int(repaired_structural_cleanup['invalid_door_tiles_removed']),
        'repair_interior_obstacle_tiles_removed': int(repaired_structural_cleanup['interior_obstacle_tiles_removed']),
        'repair_interior_obstacle_components_removed': int(repaired_structural_cleanup['interior_obstacle_components_removed']),
        'final_boundary_void_tiles_removed': int(final_void_cleanup['boundary_void_tiles_removed']),
        'final_interior_void_tiles_removed': int(final_void_cleanup['interior_void_tiles_removed']),
        'repair_boundary_wall_tiles_forced': int(repaired_boundary_shell['boundary_wall_tiles_forced']),
        'repair_boundary_door_tiles_forced': int(repaired_boundary_shell['boundary_door_tiles_forced']),
        'repair_interior_door_apron_tiles_forced': int(repaired_boundary_shell['interior_door_apron_tiles_forced']),
        'neural_puzzle_scaffold_applied': int(neural_puzzle_scaffold['applied']),
        'neural_puzzle_scaffold_tiles_added': int(neural_puzzle_scaffold['tiles_added']),
        'neural_puzzle_scaffold_segments_added': int(neural_puzzle_scaffold['segments_added']),
        'neural_puzzle_scaffold_optional_segments_requested': int(neural_puzzle_scaffold.get('optional_segments_requested', 0)),
        'neural_puzzle_scaffold_optional_segments_applied': int(neural_puzzle_scaffold.get('optional_segments_applied', 0)),
        'neural_puzzle_scaffold_route_template_used': int(neural_puzzle_scaffold.get('route_template_used', 0)),
        'neural_puzzle_scaffold_noise_components_removed': int(neural_puzzle_scaffold.get('noise_components_removed', 0)),
        'neural_puzzle_scaffold_noise_tiles_removed': int(neural_puzzle_scaffold.get('noise_tiles_removed', 0)),
        'neural_puzzle_scaffold_novelty_score': float(neural_puzzle_scaffold.get('novelty_score', 0.0)),
        'neural_puzzle_scaffold_variant_name': str(neural_puzzle_scaffold.get('variant_name', '') or ''),
        'neural_puzzle_scaffold_variant_style': str(neural_puzzle_scaffold.get('variant_style', '') or ''),
        'neural_puzzle_scaffold_variant_side_bias': int(neural_puzzle_scaffold.get('variant_side_bias', 0) or 0),
        'neural_puzzle_scaffold_interaction_valid': int(neural_puzzle_scaffold.get('interaction_valid', 0) or 0),
        'neural_puzzle_scaffold_interaction_score': float(neural_puzzle_scaffold.get('interaction_score', 0.0) or 0.0),
        'neural_puzzle_scaffold_interaction_push_slot_count': int(neural_puzzle_scaffold.get('interaction_push_slot_count', 0) or 0),
        'neural_puzzle_scaffold_interaction_barrier_axis_tiles': int(neural_puzzle_scaffold.get('interaction_barrier_axis_tiles', 0) or 0),
        'neural_puzzle_scaffold_interaction_route_divergence': float(neural_puzzle_scaffold.get('interaction_route_divergence', 0.0) or 0.0),
        'neural_puzzle_scaffold_interaction_sequence_valid': int(neural_puzzle_scaffold.get('interaction_sequence_valid', 0) or 0),
        'neural_puzzle_scaffold_interaction_sequence_score': float(neural_puzzle_scaffold.get('interaction_sequence_score', 0.0) or 0.0),
        'neural_puzzle_scaffold_interaction_sequence_length': int(neural_puzzle_scaffold.get('interaction_sequence_length', 0) or 0),
        'neural_puzzle_scaffold_interaction_sequence_route_anchor_coverage': float(neural_puzzle_scaffold.get('interaction_sequence_route_anchor_coverage', 0.0) or 0.0),
        'neural_puzzle_scaffold_interaction_sequence_pairwise_path_ratio': float(neural_puzzle_scaffold.get('interaction_sequence_pairwise_path_ratio', 0.0) or 0.0),
        'final_puzzle_scaffold_applied': int(final_puzzle_scaffold['applied']),
        'final_puzzle_scaffold_tiles_added': int(final_puzzle_scaffold['tiles_added']),
        'final_puzzle_scaffold_segments_added': int(final_puzzle_scaffold['segments_added']),
        'final_puzzle_scaffold_optional_segments_requested': int(final_puzzle_scaffold.get('optional_segments_requested', 0)),
        'final_puzzle_scaffold_optional_segments_applied': int(final_puzzle_scaffold.get('optional_segments_applied', 0)),
        'final_puzzle_scaffold_route_template_used': int(final_puzzle_scaffold.get('route_template_used', 0)),
        'final_puzzle_scaffold_noise_components_removed': int(final_puzzle_scaffold.get('noise_components_removed', 0)),
        'final_puzzle_scaffold_noise_tiles_removed': int(final_puzzle_scaffold.get('noise_tiles_removed', 0)),
        'final_puzzle_scaffold_novelty_score': float(final_puzzle_scaffold.get('novelty_score', 0.0)),
        'final_puzzle_scaffold_variant_name': str(final_puzzle_scaffold.get('variant_name', '') or ''),
        'final_puzzle_scaffold_variant_style': str(final_puzzle_scaffold.get('variant_style', '') or ''),
        'final_puzzle_scaffold_variant_side_bias': int(final_puzzle_scaffold.get('variant_side_bias', 0) or 0),
        'final_puzzle_scaffold_interaction_valid': int(final_puzzle_scaffold.get('interaction_valid', 0) or 0),
        'final_puzzle_scaffold_interaction_score': float(final_puzzle_scaffold.get('interaction_score', 0.0) or 0.0),
        'final_puzzle_scaffold_interaction_push_slot_count': int(final_puzzle_scaffold.get('interaction_push_slot_count', 0) or 0),
        'final_puzzle_scaffold_interaction_barrier_axis_tiles': int(final_puzzle_scaffold.get('interaction_barrier_axis_tiles', 0) or 0),
        'final_puzzle_scaffold_interaction_route_divergence': float(final_puzzle_scaffold.get('interaction_route_divergence', 0.0) or 0.0),
        'final_puzzle_scaffold_interaction_sequence_valid': int(final_puzzle_scaffold.get('interaction_sequence_valid', 0) or 0),
        'final_puzzle_scaffold_interaction_sequence_score': float(final_puzzle_scaffold.get('interaction_sequence_score', 0.0) or 0.0),
        'final_puzzle_scaffold_interaction_sequence_length': int(final_puzzle_scaffold.get('interaction_sequence_length', 0) or 0),
        'final_puzzle_scaffold_interaction_sequence_route_anchor_coverage': float(final_puzzle_scaffold.get('interaction_sequence_route_anchor_coverage', 0.0) or 0.0),
        'final_puzzle_scaffold_interaction_sequence_pairwise_path_ratio': float(final_puzzle_scaffold.get('interaction_sequence_pairwise_path_ratio', 0.0) or 0.0),
        'puzzle_plan_stage_count': int(len(list(room_puzzle_metadata.get('stage_sequence', []) or []))),
        'puzzle_plan_controlled_door_count': int(len(list(room_puzzle_metadata.get('controlled_doors_local', []) or []))),
        'neural_no_puzzle_structure_cleanup_applied': int(neural_no_puzzle_structure_cleanup.get('applied', 0)),
        'neural_no_puzzle_block_tiles_removed': int(neural_no_puzzle_structure_cleanup.get('block_tiles_removed', 0)),
        'neural_no_puzzle_block_components_removed': int(neural_no_puzzle_structure_cleanup.get('block_components_removed', 0)),
        'final_no_puzzle_structure_cleanup_applied': int(final_no_puzzle_structure_cleanup.get('applied', 0)),
        'final_no_puzzle_block_tiles_removed': int(final_no_puzzle_structure_cleanup.get('block_tiles_removed', 0)),
        'final_no_puzzle_block_components_removed': int(final_no_puzzle_structure_cleanup.get('block_components_removed', 0)),
        'neural_graph_markers_placed': int(neural_marker_count),
        'final_graph_markers_placed': int(final_marker_count),
        'final_graph_marker_overwrites': int(final_marker_overwrites),
        'final_graph_marker_overwrite_rate': float(final_marker_overwrite_rate),
        'semantic_constrained_decode_planned_markers': float(semantic_decode_stats.get('planned_markers', 0)),
        'semantic_constrained_decode_biased_slots': float(semantic_decode_stats.get('biased_slots', 0)),
        'neural_post_boundary_graph_semantic_hints_salvaged': float(neural_post_boundary_preserved_count),
        'final_post_boundary_graph_semantic_hints_salvaged': float(final_post_boundary_preserved_count),
        'vglc_compliant': valid_dims,
        'wfc_feedback_rounds': float(repair_diag.get('feedback_rounds', 0)),
        'wfc_failures': float(repair_diag.get('wfc_failures', 0)),
        'planned_traversability_pixels': float(np.sum(room_plan_mask)) if isinstance(room_plan_mask, np.ndarray) else 0.0,
        'used_fast_sampling': float(bool(use_fast_sampling)),
        'masked_room_sampling_temperature': float(pipeline.default_masked_room_sampling_temperature),
        'masked_room_sampling_stochastic': float(
            bool(pipeline.default_masked_room_sampling_stochastic)
        ),
        'masked_room_corrector_steps': float(pipeline.default_masked_room_corrector_steps),
        'masked_room_corrector_mask_ratio': float(pipeline.default_masked_room_corrector_mask_ratio),
    }
    metrics.update(neural_marker_alignment)
    metrics.update(final_pre_overlay_alignment)
    metrics.update(final_post_overlay_alignment)

    teacher_fallback_source: Optional[str] = None
    if (
        bool(allow_teacher_fallback)
        and effective_room_generator_mode == "latent_diffusion"
        and bool(use_fast_sampling)
        and pipeline.diffusion is not None
        and pipeline.diffusion.supports_fast_sampling()
        and pipeline._should_retry_room_with_teacher(
            final_grid=final_grid,
            graph=mission_graph_for_room,
            room_id=room_id,
            metrics=metrics,
            source_mode="fast_sampler",
        )
    ):
        teacher_fallback_source = "fast_sampler"
    elif (
        bool(allow_teacher_fallback)
        and effective_room_generator_mode == "discrete_masked"
        and pipeline.diffusion is not None
        and pipeline._should_retry_room_with_teacher(
            final_grid=final_grid,
            graph=mission_graph_for_room,
            room_id=room_id,
            metrics=metrics,
            source_mode="masked_room",
        )
    ):
        teacher_fallback_source = "masked_room"

    if teacher_fallback_source is not None:
        pipeline._bump_diagnostic(f"{teacher_fallback_source}_teacher_fallback")
        logger.debug(
            "Room %s triggered %s teacher fallback; rerunning with full diffusion teacher.",
            room_id,
            teacher_fallback_source.replace("_", "-"),
        )
        if teacher_fallback_source == "masked_room":
            # The masked generator and diffusion teacher can use different CUDA kernels and
            # work queues. Flush queued masked-room work before the recursive teacher rerun
            # so VQ-VAE decode always sees tensors on a consistent stream.
            pipeline._synchronize_cuda_device()
        teacher_result = pipeline.generate_room(
            neighbor_latents=neighbor_latents,
            graph_context=graph_context,
            room_id=room_id,
            boundary_constraints=boundary_constraints,
            position=position,
            reference_room_maps=reference_room_maps,
            guidance_scale=guidance_scale,
            logic_guidance_scale=logic_guidance_scale,
            num_diffusion_steps=max(int(pipeline.default_num_diffusion_steps), int(num_diffusion_steps)),
            use_fast_sampling=False,
            latent_sampler="diffusion",
            categorical_codebook_size=categorical_codebook_size,
            use_ddim=use_ddim,
            apply_repair=apply_repair,
            start_goal_coords=start_goal_coords,
            seed=seed,
            precomputed_condition=condition.detach().clone(),
            allow_teacher_fallback=False,
            room_generator_override="latent_diffusion",
        )
        teacher_result.metrics["teacher_fallback_used"] = 1.0
        teacher_result.metrics[f"teacher_fallback_source_{teacher_fallback_source}"] = 1.0
        teacher_result.metrics["original_fallback_candidate_neural_grid_entropy"] = float(metrics["neural_grid_entropy"])
        teacher_result.metrics["original_fallback_candidate_tiles_changed"] = float(metrics["tiles_changed"])
        return teacher_result

    return RoomGenerationResult(
        room_id=room_id,
        room_grid=final_grid,
        latent=latent_cpu,
        neural_grid=neural_grid,
        was_repaired=was_repaired,
        raw_neural_grid=raw_neural_grid,
        repair_mask=repair_mask,
        room_plan_mask=room_plan_mask,
        neural_probs=neural_probs,
        puzzle_metadata=room_puzzle_metadata,
        metrics=metrics,
    )


class DiffusionSampler:
    """Room-level diffusion/categorical sampling boundary."""

    def __init__(self, engine: Any):
        self.engine = engine

    def generate_room(self, *args: Any, **kwargs: Any) -> Any:
        return generate_room(self.engine, *args, **kwargs)

    def generate_room_batch(self, *args: Any, **kwargs: Any) -> Any:
        return generate_room_batch(self.engine, *args, **kwargs)

    def resolve_guidance(self, *args: Any, **kwargs: Any) -> Any:
        return self.engine._resolve_effective_sampling_guidance(*args, **kwargs)


__all__ = ["DiffusionSampler", "generate_room", "generate_room_batch"]
