"""Frozen design primitives for the publication experiment.

This module prevents two high-risk protocol mistakes:

* splitting Zelda rooms instead of whole dungeon identities; and
* presenting inference-time switches as separately trained ablations.

It contains no training code.  Callers materialize a complete, validated
configuration for every outer fold, model variant, and training seed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from src.zelda_data.splits import (
    ALL_DUNGEONS,
    DEFAULT_VARIANTS,
    normalize_dungeon_ids,
    normalize_variants,
    validate_disjoint_dungeon_splits,
)


PROTOCOL_SCHEMA_VERSION = "2.0"
_MODEL_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Serialize a protocol payload deterministically for hashing."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def payload_sha256(payload: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 digest for a JSON-compatible mapping."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def seal_protocol_manifest(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a copy whose digest covers every field except the digest itself."""
    sealed = copy.deepcopy(dict(payload))
    sealed.pop("protocol_sha256", None)
    sealed["protocol_sha256"] = payload_sha256(sealed)
    return sealed


def _deep_set(target: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor = target
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = copy.deepcopy(value)


@dataclass(frozen=True)
class OuterFold:
    """One grouped outer fold with complete dungeon identities held out."""

    fold_id: str
    train_dungeons: Tuple[int, ...]
    test_dungeons: Tuple[int, ...]
    variants: Tuple[int, ...] = DEFAULT_VARIANTS

    def __post_init__(self) -> None:
        train, test = validate_disjoint_dungeon_splits(
            self.train_dungeons,
            self.test_dungeons,
        )
        variants = normalize_variants(self.variants)
        if len(test) != 1:
            raise ValueError(
                "Publication outer folds must hold out exactly one dungeon "
                f"identity; got {test}."
            )
        if set(train).union(test) != set(ALL_DUNGEONS):
            raise ValueError(
                "Publication outer folds must partition all canonical dungeon "
                f"identities {ALL_DUNGEONS}; got train={train}, test={test}."
            )
        if tuple(self.train_dungeons) != train:
            raise ValueError("train_dungeons must be sorted and unique.")
        if tuple(self.test_dungeons) != test:
            raise ValueError("test_dungeons must be sorted and unique.")
        if tuple(self.variants) != variants:
            raise ValueError("variants must be sorted and unique.")
        expected_fold_id = f"holdout_dungeon_{test[0]}"
        if self.fold_id != expected_fold_id:
            raise ValueError(
                f"fold_id must be {expected_fold_id!r} for test dungeon {test[0]}; "
                f"got {self.fold_id!r}."
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "train_dungeons": list(self.train_dungeons),
            "test_dungeons": list(self.test_dungeons),
            "variants": list(self.variants),
        }


def build_leave_one_dungeon_out_folds(
    dungeon_ids: Iterable[int] = ALL_DUNGEONS,
    *,
    variants: Iterable[int] = DEFAULT_VARIANTS,
) -> Tuple[OuterFold, ...]:
    """Build deterministic leave-one-dungeon-identity-out folds."""
    normalized_dungeons = normalize_dungeon_ids(dungeon_ids)
    if normalized_dungeons is None:
        raise ValueError("dungeon_ids must be non-empty.")
    if normalized_dungeons != ALL_DUNGEONS:
        raise ValueError(
            "The confirmatory publication protocol requires all canonical "
            f"dungeons {ALL_DUNGEONS}; got {normalized_dungeons}."
        )
    normalized_variants = normalize_variants(variants)
    return tuple(
        OuterFold(
            fold_id=f"holdout_dungeon_{held_out}",
            train_dungeons=tuple(
                dungeon_id
                for dungeon_id in normalized_dungeons
                if dungeon_id != held_out
            ),
            test_dungeons=(held_out,),
            variants=normalized_variants,
        )
        for held_out in normalized_dungeons
    )


@dataclass(frozen=True)
class ModelVariant:
    """One separately trained causal model variant."""

    model_id: str
    description: str
    global_graph_conditioning: bool
    local_logic_loss: bool
    global_logic_loss: bool
    tile_auxiliary_loss: bool

    def __post_init__(self) -> None:
        if not _MODEL_ID_PATTERN.fullmatch(self.model_id):
            raise ValueError(
                "model_id must use lowercase letters, digits, underscores, or hyphens "
                "and must start with a letter or digit."
            )
        if not self.description.strip():
            raise ValueError("description must be non-empty.")
        if self.global_logic_loss and not self.global_graph_conditioning:
            raise ValueError("Global logic loss requires global graph conditioning.")
        if self.tile_auxiliary_loss and not (
            self.local_logic_loss or self.global_logic_loss
        ):
            raise ValueError("Tile auxiliary loss requires an active logic objective.")

    @property
    def logic_net_enabled(self) -> bool:
        return bool(
            self.local_logic_loss
            or self.global_logic_loss
            or self.tile_auxiliary_loss
        )

    def training_overrides(self) -> Dict[str, Any]:
        """Return explicit settings that define this trained condition."""
        return {
            "diffusion.condition_global_graph_enabled": bool(
                self.global_graph_conditioning
            ),
            "diffusion.condition_strict_schema": True,
            "diffusion.graph_conditioning_mode": (
                "node_sequence" if self.global_graph_conditioning else "pooled"
            ),
            # Neighbor latents already carry room-local context.  Enabling the
            # separate discrete reference-room encoder only for graph models
            # would make the N-vs-G comparison change two causal factors.
            "diffusion.condition_use_reference_room_maps": False,
            "diffusion.graph_to_grid_edge_semantics": False,
            "diffusion.graph_spatial_alignment_weight": 0.0,
            "diffusion.logic_net_enabled": bool(self.logic_net_enabled),
            "diffusion.logic_net_trainable": bool(self.logic_net_enabled),
            "diffusion.alpha_logic": (
                0.1 if self.local_logic_loss or self.global_logic_loss else 0.0
            ),
            "diffusion.alpha_logic_tile": (
                0.05 if self.tile_auxiliary_loss else 0.0
            ),
            "diffusion.logic_topology_trace_weight": (
                0.25 if self.local_logic_loss else 0.0
            ),
            "diffusion.logic_topology_anchor_weight": (
                0.25 if self.local_logic_loss else 0.0
            ),
            "diffusion.logic_global_reach_weight": (
                1.0 if self.global_logic_loss else 0.0
            ),
            "diffusion.logic_global_room_weight": (
                0.25 if self.global_logic_loss else 0.0
            ),
            # Training-time supervision must not be confounded with sampling
            # guidance.  Guidance is crossed as an inference factor later.
            "diffusion.guidance_scale": 0.0,
            "generation.logic_guidance_scale": 0.0,
            "generation.enable_map_elites": False,
            # Repair must be the same symbolic operator for every model arm.
            # Otherwise LogicNet variants receive a neural repair capability
            # that N/G variants cannot use.
            "generation.use_neural_guided_repair": False,
            "generation.use_neural_repair_feedback": False,
            # Return failed and indeterminate attempts to the publication
            # recorder instead of filtering them out before aggregation.
            "generation.end_to_end_validation_mode": "report",
            "diffusion.puzzle_stage_conditioning_enabled": False,
            "diffusion.puzzle_stage_topology_enabled": False,
            "diffusion.puzzle_stage_semantics_loss_weight": 0.0,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "description": self.description,
            "global_graph_conditioning": bool(self.global_graph_conditioning),
            "local_logic_loss": bool(self.local_logic_loss),
            "global_logic_loss": bool(self.global_logic_loss),
            "tile_auxiliary_loss": bool(self.tile_auxiliary_loss),
            "logic_net_enabled": bool(self.logic_net_enabled),
            "training_overrides": self.training_overrides(),
        }


PUBLICATION_VARIANTS: Tuple[ModelVariant, ...] = (
    ModelVariant(
        model_id="n_local",
        description="Local room context without global mission-graph conditioning or LogicNet.",
        global_graph_conditioning=False,
        local_logic_loss=False,
        global_logic_loss=False,
        tile_auxiliary_loss=False,
    ),
    ModelVariant(
        model_id="g_graph",
        description="Global mission-graph conditioning without differentiable logic losses.",
        global_graph_conditioning=True,
        local_logic_loss=False,
        global_logic_loss=False,
        tile_auxiliary_loss=False,
    ),
    ModelVariant(
        model_id="g_l_local_logic",
        description="Graph conditioning with local room reachability supervision.",
        global_graph_conditioning=True,
        local_logic_loss=True,
        global_logic_loss=False,
        tile_auxiliary_loss=False,
    ),
    ModelVariant(
        model_id="g_lg_full_logic",
        description="Graph conditioning with local and global mission-logic supervision.",
        global_graph_conditioning=True,
        local_logic_loss=True,
        global_logic_loss=True,
        tile_auxiliary_loss=False,
    ),
)


def apply_publication_variant(
    base_config: Mapping[str, Any],
    *,
    fold: OuterFold,
    variant: ModelVariant,
    training_seed: int,
    output_root: str = "outputs/paper_v1",
) -> Dict[str, Any]:
    """Return a complete config for one independently trained run."""
    if int(training_seed) < 0:
        raise ValueError("training_seed must be non-negative.")
    config = copy.deepcopy(dict(base_config))
    run_id = f"{fold.fold_id}__{variant.model_id}__seed_{int(training_seed)}"
    run_output_dir = f"{output_root.rstrip('/')}/{run_id}"
    explicit = {
        "dataset.train_dungeons": list(fold.train_dungeons),
        "dataset.test_dungeons": list(fold.test_dungeons),
        "dataset.variants": list(fold.variants),
        "dataset.topology_supervision_mode": "runtime_aligned",
        "runtime.seed": int(training_seed),
        "runtime.cudnn_benchmark": False,
        "runtime.cudnn_deterministic": True,
        "runtime.deterministic_algorithms": True,
        "runtime.experiment_name": run_id,
        "runtime.output_dir": run_output_dir,
        "runtime.auto_resume": False,
        "runtime.resume": None,
        "diffusion.warm_start_checkpoint": None,
        "diffusion.checkpoint_dir": f"{run_output_dir}/checkpoints/diffusion",
        # Use a criterion shared by every arm. val_total_loss changes meaning
        # when a variant activates the logic objective.
        "diffusion.best_checkpoint_metric": "val_diffusion_loss",
        # Equal epochs are not equal optimization budgets if one arm silently
        # skips unstable batches. Confirmatory runs therefore fail closed.
        "diffusion.nonfinite_policy": "error",
        **variant.training_overrides(),
    }
    for path, value in explicit.items():
        _deep_set(config, path, value)
    return config


def build_protocol_manifest(
    *,
    training_seeds: Sequence[int] = (42, 43, 44, 45, 46),
    folds: Sequence[OuterFold] | None = None,
    variants: Sequence[ModelVariant] = PUBLICATION_VARIANTS,
) -> Dict[str, Any]:
    """Build a deterministic manifest for the full confirmatory run matrix."""
    normalized_seeds = tuple(int(seed) for seed in training_seeds)
    if not normalized_seeds or len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("training_seeds must be non-empty and unique.")
    if any(seed < 0 for seed in normalized_seeds):
        raise ValueError("training_seeds must be non-negative.")
    resolved_folds = tuple(
        build_leave_one_dungeon_out_folds() if folds is None else folds
    )
    resolved_variants = tuple(variants)
    if not resolved_folds:
        raise ValueError("At least one outer fold is required.")
    if not resolved_variants:
        raise ValueError("At least one model variant is required.")
    fold_ids = [fold.fold_id for fold in resolved_folds]
    if len(set(fold_ids)) != len(fold_ids):
        raise ValueError("Publication fold IDs must be unique.")
    model_ids = [variant.model_id for variant in resolved_variants]
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("Publication model IDs must be unique.")

    manifest: Dict[str, Any] = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "design": "leave_one_dungeon_identity_out",
        "observational_unit": "complete_generated_dungeon_attempt",
        "confirmatory_inference_unit": "outer_held_out_dungeon_identity",
        "topology_supervision_mode": "runtime_aligned",
        "canonical_generation_entrypoint": (
            "src.pipeline.stitching.dungeon_assembler.generate_dungeon"
        ),
        "publication_stages": {
            "raw": "pre-constraint neural room proposals after neutral assembly",
            "constrained": "deterministically constrained room outputs after neutral assembly",
            "repaired": "explicitly repaired room outputs after neutral assembly",
            "compiled": "final dungeon after mission-graph and entity compilation",
        },
        "stage_assembly_contract": (
            "raw, constrained, and repaired use identical slots, boundary sealing, "
            "and physical-edge carving"
        ),
        "inference_factors": {
            "primary_model_ablation": {
                "condition": "unguided",
                "eligible_model_ids": [variant.model_id for variant in resolved_variants],
                "reason": (
                    "All model arms can realize this condition; it isolates trained "
                    "graph and logic-supervision factors."
                ),
            },
            "secondary_logic_guidance": {
                "status": "requires_separate_sealed_sampling_manifest",
                "conditions": ["unguided", "logic_guided"],
                "eligible_model_ids": [
                    variant.model_id for variant in resolved_variants if variant.logic_net_enabled
                ],
                "scale_selection": (
                    "Select one nonzero scale using outer-training identities only, "
                    "then seal it before held-out sampling."
                ),
                "reason": (
                    "N and G have no LogicNet and silently reduce requested logic "
                    "guidance to zero, so they cannot form a valid guided comparison."
                ),
            },
            "paired_by": [
                "outer_fold",
                "train_seed",
                "sampling_seed",
                "mission_graph_id",
            ],
            "stage_note": (
                "stage is a repeated measurement of one attempt, not a separately "
                "sampled inference variant"
            ),
        },
        "training_seeds": list(normalized_seeds),
        "folds": [fold.to_dict() for fold in resolved_folds],
        "model_variants": [variant.to_dict() for variant in resolved_variants],
        "run_count": (
            len(resolved_folds) * len(resolved_variants) * len(normalized_seeds)
        ),
        "evidence_policy": {
            "allow_existing_checkpoints": False,
            "allow_existing_artifacts": False,
            "require_raw_stage": True,
            "require_all_stage_attempts": True,
            "count_timeout_or_error_as_invalid_in_all_attempt_rate": True,
            "require_independent_exact_validation": True,
            "generation_validation_mode": "report",
        },
    }
    return seal_protocol_manifest(manifest)
