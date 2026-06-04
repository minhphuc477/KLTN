"""
Validated configuration-to-pipeline keyword adapters.

Keeping this bridge outside ``dungeon_pipeline`` reduces constructor/config
drift risk and lets config tests import the mapping layer without loading the
full pipeline orchestration class.
"""

from __future__ import annotations

from typing import Any, Dict

from src.pipeline.room_topology_conditioning import (
    DEFAULT_PUZZLE_STAGE_TRACE_DECAY,
    DEFAULT_SEMANTIC_PUZZLE_OFFSET,
    DEFAULT_SEMANTIC_ROLE_PRIOR_STRENGTH,
    DEFAULT_VALIDATOR_PLAN_MAX_STATES,
)


def topology_generation_kwargs_from_resolved_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build Block I generation kwargs from the validated global config payload."""
    stage = config["topology"]
    return {
        "target_curve": list(stage["default_target_curve"]),
        "num_rooms": stage["num_rooms"],
        "population_size": stage["population_size"],
        "generations": stage["generations"],
        "mutation_rate": stage["mutation_rate"],
        "crossover_rate": stage["crossover_rate"],
        "genome_length": stage["genome_length"],
        "rule_space": stage["rule_space"],
        "transition_mix": stage["transition_mix"],
        "search_strategy": stage["search_strategy"],
        "qd_archive_cells": stage["qd_archive_cells"],
        "qd_init_random_fraction": stage["qd_init_random_fraction"],
        "qd_emitter_mutation_rate": stage["qd_emitter_mutation_rate"],
        "qd_archive_path": stage.get("qd_archive_path"),
        "qd_load_archive": stage.get("qd_load_archive", False),
        "qd_autosave_archive": stage.get("qd_autosave_archive", False),
        "max_lock_key_rules": stage["max_lock_key_rules"],
        "enable_rule_credit_assignment": stage["enable_rule_credit_assignment"],
        "enforce_generation_constraints": stage["enforce_generation_constraints"],
        "allow_candidate_repairs": stage["allow_candidate_repairs"],
    }


def generation_runtime_kwargs_from_resolved_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build runtime room/dungeon generation defaults from the validated config payload."""
    stage = config["generation"]
    return {
        "default_guidance_scale": stage.get("guidance_scale", 3.0),
        "default_logic_guidance_scale": stage.get("logic_guidance_scale", 0.0),
        "default_logic_guidance_strategy": stage.get("logic_guidance_strategy", "late"),
        "default_logic_guidance_active_fraction": stage.get("logic_guidance_active_fraction", 0.2),
        "default_num_diffusion_steps": stage.get("num_diffusion_steps", 50),
        "default_use_fast_sampling": stage.get("use_fast_sampling", False),
        "default_latent_sampler": stage.get("latent_sampler", "diffusion"),
        "default_categorical_codebook_size": stage.get("categorical_codebook_size", 256),
        "default_use_topological_positional_encoding": stage.get("use_topological_positional_encoding", True),
        "default_apply_repair": stage.get("apply_repair", True),
        "default_use_neural_guided_repair": stage.get("use_neural_guided_repair", True),
        "default_use_neural_repair_feedback": stage.get("use_neural_repair_feedback", True),
        "default_repair_inpaint_noise_strength": stage.get("repair_inpaint_noise_strength", 0.5),
        "default_repair_inpaint_guidance_scale_multiplier": stage.get("repair_inpaint_guidance_scale_multiplier", 1.0),
        "default_enable_map_elites": stage.get("enable_map_elites", False),
        "symbolic_max_repair_attempts": stage.get("symbolic_max_repair_attempts", 5),
        "symbolic_repair_margin": stage.get("symbolic_repair_margin", 2),
        "symbolic_adjacency_threshold": stage.get("symbolic_adjacency_threshold", 0.01),
        "default_start_goal_coords": (
            tuple(int(v) for v in stage.get("default_start_coord", (1, 5))),
            tuple(int(v) for v in stage.get("default_goal_coord", (14, 5))),
        ),
        "default_semantic_role_prior_strength": stage.get(
            "semantic_role_prior_strength",
            DEFAULT_SEMANTIC_ROLE_PRIOR_STRENGTH,
        ),
        "default_semantic_anchor_threshold": stage.get("semantic_anchor_threshold", 0.5),
        "default_semantic_puzzle_offset": stage.get(
            "semantic_puzzle_offset",
            DEFAULT_SEMANTIC_PUZZLE_OFFSET,
        ),
        "default_semantic_constrained_decoding_enabled": stage.get(
            "semantic_constrained_decoding_enabled",
            True,
        ),
        "default_semantic_marker_logit_bias": stage.get(
            "semantic_marker_logit_bias",
            10000.0,
        ),
        "default_semantic_marker_suppression_bias": stage.get(
            "semantic_marker_suppression_bias",
            100.0,
        ),
        "default_puzzle_room_scaffold_enabled": stage.get(
            "puzzle_room_scaffold_enabled",
            True,
        ),
        "default_puzzle_room_structure_enabled": stage.get(
            "puzzle_room_structure_enabled",
            True,
        ),
        "default_puzzle_room_scaffold_min_structure_tiles": stage.get(
            "puzzle_room_scaffold_min_structure_tiles",
            10,
        ),
        "default_puzzle_room_archetype_mode": stage.get(
            "puzzle_room_archetype_mode",
            "auto",
        ),
        "default_puzzle_room_branch_density": stage.get(
            "puzzle_room_branch_density",
            0.75,
        ),
        "default_puzzle_room_block_budget": stage.get(
            "puzzle_room_block_budget",
            28,
        ),
        "default_puzzle_room_preserve_route_margin": stage.get(
            "puzzle_room_preserve_route_margin",
            0,
        ),
        "default_puzzle_room_switch_pocket_depth": stage.get(
            "puzzle_room_switch_pocket_depth",
            3,
        ),
        "default_puzzle_room_resource_bypass_offset": stage.get(
            "puzzle_room_resource_bypass_offset",
            2,
        ),
        "default_puzzle_room_key_pocket_depth": stage.get(
            "puzzle_room_key_pocket_depth",
            3,
        ),
        "default_puzzle_room_item_slot_depth": stage.get(
            "puzzle_room_item_slot_depth",
            3,
        ),
        "default_puzzle_room_toggle_corridor_offset": stage.get(
            "puzzle_room_toggle_corridor_offset",
            2,
        ),
        "default_puzzle_room_novelty_enabled": stage.get(
            "puzzle_room_novelty_enabled",
            True,
        ),
        "default_puzzle_room_candidate_count": stage.get(
            "puzzle_room_candidate_count",
            4,
        ),
        "default_puzzle_room_novelty_weight": stage.get(
            "puzzle_room_novelty_weight",
            0.45,
        ),
        "default_puzzle_room_min_quality_gain": stage.get(
            "puzzle_room_min_quality_gain",
            0.5,
        ),
        "default_validator_plan_max_states": stage.get(
            "validator_plan_max_states",
            DEFAULT_VALIDATOR_PLAN_MAX_STATES,
        ),
        "default_puzzle_stage_topology_enabled": stage.get(
            "puzzle_stage_topology_enabled",
            False,
        ),
        "default_puzzle_stage_trace_decay": stage.get(
            "puzzle_stage_trace_decay",
            DEFAULT_PUZZLE_STAGE_TRACE_DECAY,
        ),
        "default_deterministic_graph_marker_overlay_enabled": stage.get(
            "deterministic_graph_marker_overlay_enabled",
            True,
        ),
        "default_fast_sampler_teacher_fallback_enabled": stage.get(
            "fast_sampler_teacher_fallback_enabled",
            True,
        ),
        "default_masked_room_teacher_fallback_enabled": stage.get(
            "masked_room_teacher_fallback_enabled",
            True,
        ),
        "default_masked_room_sampling_temperature": stage.get(
            "masked_room_sampling_temperature",
            1.0,
        ),
        "default_masked_room_sampling_schedule": stage.get(
            "masked_room_sampling_schedule",
            "cosine",
        ),
        "default_masked_room_sampling_stochastic": stage.get(
            "masked_room_sampling_stochastic",
            True,
        ),
        "default_masked_room_corrector_steps": stage.get(
            "masked_room_corrector_steps",
            1,
        ),
        "default_masked_room_corrector_mask_ratio": stage.get(
            "masked_room_corrector_mask_ratio",
            0.1,
        ),
    }


def pipeline_kwargs_from_resolved_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build canonical pipeline constructor kwargs from the validated global config payload."""
    diffusion = config["diffusion"]
    fast_sampler = config["fast_sampler"]
    masked_room = config["masked_room"]
    topology_kwargs = topology_generation_kwargs_from_resolved_config(config)
    kwargs = {
        "topology_default_target_curve": list(topology_kwargs["target_curve"]),
        "topology_num_rooms": topology_kwargs["num_rooms"],
        "topology_population_size": topology_kwargs["population_size"],
        "topology_generations": topology_kwargs["generations"],
        "topology_mutation_rate": topology_kwargs["mutation_rate"],
        "topology_crossover_rate": topology_kwargs["crossover_rate"],
        "topology_genome_length": topology_kwargs["genome_length"],
        "topology_rule_space": topology_kwargs["rule_space"],
        "topology_transition_mix": topology_kwargs["transition_mix"],
        "topology_search_strategy": topology_kwargs["search_strategy"],
        "topology_qd_archive_cells": topology_kwargs["qd_archive_cells"],
        "topology_qd_init_random_fraction": topology_kwargs["qd_init_random_fraction"],
        "topology_qd_emitter_mutation_rate": topology_kwargs["qd_emitter_mutation_rate"],
        "topology_qd_archive_path": topology_kwargs.get("qd_archive_path"),
        "topology_qd_load_archive": topology_kwargs.get("qd_load_archive", False),
        "topology_qd_autosave_archive": topology_kwargs.get("qd_autosave_archive", False),
        "topology_max_lock_key_rules": topology_kwargs["max_lock_key_rules"],
        "topology_enable_rule_credit_assignment": topology_kwargs["enable_rule_credit_assignment"],
        "topology_enforce_generation_constraints": topology_kwargs["enforce_generation_constraints"],
        "topology_allow_candidate_repairs": topology_kwargs["allow_candidate_repairs"],
    }
    kwargs.update(generation_runtime_kwargs_from_resolved_config(config))
    kwargs.update(
        {
            "condition_gnn_type": diffusion["condition_gnn_type"],
            "condition_use_reference_room_maps": diffusion["condition_use_reference_room_maps"],
            "condition_reference_tile_vocab_size": diffusion["condition_reference_tile_vocab_size"],
            "condition_reference_embedding_dim": diffusion["condition_reference_embedding_dim"],
            "condition_reference_hidden_dim": diffusion["condition_reference_hidden_dim"],
            "condition_use_rrwp_edge_features": diffusion.get("condition_use_rrwp_edge_features", True),
            "topology_refinement_mode": diffusion["topology_refinement_mode"],
            "diffusion_attention_mode": diffusion["attention_mode"],
            "diffusion_hedgehog_feature_dim": diffusion["hedgehog_feature_dim"],
            "diffusion_cfg_schedule_mode": diffusion["cfg_schedule_mode"],
            "diffusion_cfg_schedule_min_scale": diffusion["cfg_schedule_min_scale"],
            "diffusion_cfg_schedule_power": diffusion["cfg_schedule_power"],
            "use_current_node_distance_features": diffusion["use_current_node_distance_features"],
            "current_node_distance_max": diffusion["current_node_distance_max"],
            "masked_sampling_steps": masked_room["masked_steps"],
            "fast_sampling_steps": fast_sampler["num_inference_steps"],
            "condition_encoder_fallback_config": {
                "latent_dim": diffusion["latent_dim"],
                "condition_hidden_dim": diffusion["condition_hidden_dim"],
                "context_dim": diffusion["context_dim"],
                "condition_gnn_type": diffusion["condition_gnn_type"],
                "condition_num_gnn_layers": diffusion["condition_num_gnn_layers"],
                "condition_num_attention_heads": diffusion["condition_num_attention_heads"],
                "condition_dropout": diffusion["condition_dropout"],
                "use_current_node_distance_features": diffusion["use_current_node_distance_features"],
                "condition_use_reference_room_maps": diffusion["condition_use_reference_room_maps"],
                "condition_reference_tile_vocab_size": diffusion["condition_reference_tile_vocab_size"],
                "condition_reference_embedding_dim": diffusion["condition_reference_embedding_dim"],
                "condition_reference_hidden_dim": diffusion["condition_reference_hidden_dim"],
                "condition_use_rrwp_edge_features": diffusion.get("condition_use_rrwp_edge_features", True),
            },
            "diffusion_fallback_config": {
                "latent_dim": diffusion["latent_dim"],
                "context_dim": diffusion["context_dim"],
                "num_timesteps": diffusion["num_timesteps"],
                "prediction_type": diffusion["prediction_type"],
                "diffusion_training_objective": diffusion.get("training_objective", "diffusion"),
                "denoiser_backbone": diffusion.get("denoiser_backbone", "unet"),
                "cfg_dropout_prob": diffusion["cfg_dropout_prob"],
                "cfg_scale": diffusion["cfg_scale"],
                "pag_scale": diffusion.get("pag_scale", 0.0),
                "min_snr_gamma": diffusion["min_snr_gamma"],
                "model_channels": diffusion["model_channels"],
                "topology_conditioning_mode": diffusion["topology_conditioning_mode"],
                "unet_channel_mult": list(diffusion["unet_channel_mult"]),
                "unet_num_res_blocks": diffusion["unet_num_res_blocks"],
                "unet_attention_resolutions": list(diffusion["unet_attention_resolutions"]),
                "unet_num_heads": diffusion["unet_num_heads"],
                "unet_dropout": diffusion["unet_dropout"],
                "dit_depth": diffusion.get("dit_depth", 4),
                "dit_patch_size": diffusion.get("dit_patch_size", 1),
                "dit_mlp_ratio": diffusion.get("dit_mlp_ratio", 4.0),
                "graph_auto_linear_attention_nodes": diffusion["graph_auto_linear_attention_nodes"],
                "spatial_graph_gate_init": diffusion["spatial_graph_gate_init"],
                "spatial_topology_gate_init": diffusion["spatial_topology_gate_init"],
                "room_topology_channels": diffusion["room_topology_channels"],
                "puzzle_structure_dropout_prob": diffusion.get("puzzle_structure_dropout_prob", 0.0),
            },
            "logic_net_fallback_config": {
                "latent_dim": diffusion["latent_dim"],
                "num_classes": config["dataset"]["num_classes"],
                "num_logic_iterations": diffusion["num_logic_iterations"],
                "logic_grid_pathfinder": diffusion.get("logic_grid_pathfinder", "bellman_ford"),
                "logic_topology_trace_weight": diffusion["logic_topology_trace_weight"],
                "logic_topology_anchor_weight": diffusion["logic_topology_anchor_weight"],
                "logic_global_reach_weight": diffusion.get("logic_global_reach_weight", 1.0),
                "logic_global_room_weight": diffusion.get("logic_global_room_weight", 0.25),
            },
            "masked_room_fallback_config": {
                "num_classes": config["dataset"]["num_classes"],
                "hidden_dim": masked_room["hidden_dim"],
                "model_channels": masked_room["model_channels"],
                "context_dim": masked_room["context_dim"],
                "topology_conditioning_mode": masked_room["topology_conditioning_mode"],
                "graph_auto_linear_attention_nodes": masked_room["graph_auto_linear_attention_nodes"],
                "spatial_graph_gate_init": masked_room["spatial_graph_gate_init"],
                "spatial_topology_gate_init": masked_room["spatial_topology_gate_init"],
                "unet_channel_mult": list(masked_room["unet_channel_mult"]),
                "unet_num_res_blocks": masked_room["unet_num_res_blocks"],
                "unet_attention_resolutions": list(masked_room["unet_attention_resolutions"]),
                "unet_num_heads": masked_room["unet_num_heads"],
                "unet_dropout": masked_room["unet_dropout"],
                "room_topology_channels": masked_room["room_topology_channels"],
                "puzzle_structure_dropout_prob": masked_room.get("puzzle_structure_dropout_prob", 0.0),
            },
        }
    )
    return kwargs


__all__ = [
    "topology_generation_kwargs_from_resolved_config",
    "generation_runtime_kwargs_from_resolved_config",
    "pipeline_kwargs_from_resolved_config",
]
