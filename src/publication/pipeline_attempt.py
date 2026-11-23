"""Canonical generation and exact stage validation for sealed experiment attempts."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import networkx as nx
import numpy as np
import yaml

from src.config_system import seed_everything, validate_config
from src.pipeline.config_bridge import pipeline_kwargs_from_resolved_config
from src.pipeline.evaluation import _validate_dungeon
from src.pipeline.room_stitching import validate_connection_realizations
from src.pipeline.stitching.dungeon_assembler import _validate_mission_graph_exact
from src.pipeline.validation_context import (
    build_stitched_validation_context, has_complete_stitched_validation_context,
)
from src.validation.end_to_end import build_end_to_end_validation_report
from src.validation.global_state import validate_attached_global_state_contract

from .attempt_runner import AttemptSpec
from .protocol import PUBLICATION_VARIANTS, payload_sha256


REQUIRED_ARTIFACTS = frozenset({"resolved_config", "mission_graph", "vqvae_checkpoint", "diffusion_checkpoint"})
MASKED_ROOM_REQUIRED_ARTIFACTS = frozenset({"resolved_config", "mission_graph", "masked_room_checkpoint"})


class CanonicalAttemptPipeline:
    """Load only sealed checkpoints and reuse at most one model on the device."""

    def __init__(self, pipeline_factory=None):
        if pipeline_factory is None:
            from src.pipeline.dungeon_pipeline import NeuralSymbolicDungeonPipeline
            pipeline_factory = NeuralSymbolicDungeonPipeline.from_kwargs
        self._factory = pipeline_factory
        self._pipeline = None
        self._model_key = None
        self._references = {}

    def _load(self, spec: AttemptSpec):
        if "resolved_config" not in spec.artifacts or "mission_graph" not in spec.artifacts:
            raise ValueError("Canonical attempts require artifact hashes for resolved_config and mission_graph.")
        config = validate_config(yaml.safe_load(Path(spec.artifacts["resolved_config"]["path"]).read_text(encoding="utf-8")))
        if payload_sha256(config) != spec.metadata["config_sha256"]:
            raise ValueError("Resolved config does not match the attempt's training config digest.")
        is_masked_room = (
            ("masked_room_checkpoint" in spec.artifacts and "vqvae_checkpoint" not in spec.artifacts)
            or str(config["training"]["stage"]).strip().lower() == "masked_room"
        )
        required_artifacts = MASKED_ROOM_REQUIRED_ARTIFACTS if is_masked_room else REQUIRED_ARTIFACTS
        if not required_artifacts.issubset(spec.artifacts):
            raise ValueError(f"Canonical attempts require artifact hashes for {sorted(required_artifacts)}.")
        model_key = payload_sha256({
            key: dict(spec.artifacts[key]) for key in sorted(required_artifacts - {"mission_graph"})
        })
        variants = {variant.model_id: variant for variant in PUBLICATION_VARIANTS}
        variant = variants.get(spec.metadata["model_id"])
        if variant is None:
            raise ValueError("Canonical executor requires a declared publication model variant.")
        if is_masked_room:
            if bool(config["masked_room"]["logic_net_enabled"]) != variant.local_logic_loss:
                raise ValueError("Checkpoint condition does not match the declared model variant.")
            if (str(config["masked_room"]["condition_gnn_type"]).strip().lower() == "gps") != variant.global_graph_conditioning:
                raise ValueError("Graph conditioning differs from the declared model variant.")
            overrides = variant.masked_room_training_overrides()
        else:
            if bool(config["diffusion"]["logic_net_enabled"]) != variant.local_logic_loss:
                raise ValueError("Checkpoint condition does not match the declared model variant.")
            if bool(config["diffusion"]["condition_global_graph_enabled"]) != variant.global_graph_conditioning:
                raise ValueError("Graph conditioning differs from the declared model variant.")
            overrides = variant.training_overrides()
        for path, expected in overrides.items():
            actual = config
            for part in path.split("."):
                actual = actual[part]
            if actual != expected:
                raise ValueError(f"Configuration violates the declared variant: {path}")
        if config["runtime"]["seed"] != spec.metadata["train_seed"]:
            raise ValueError("Training seed differs from the sealed attempt.")
        test_dungeons = config["dataset"]["test_dungeons"]
        if len(test_dungeons) != 1 or spec.metadata["outer_fold"] != f"holdout_dungeon_{test_dungeons[0]}":
            raise ValueError("Held-out dungeon differs from the sealed attempt.")
        if spec.metadata["inference_variant"] == "logic_guided" and not variant.local_logic_loss:
            raise ValueError("Logic guidance requires a trained LogicNet in this model arm.")
        if config["generation"]["end_to_end_validation_mode"] != "report":
            raise ValueError("Attempt generation must retain invalid outputs using report mode.")
        if model_key != self._model_key:
            self._pipeline = None
            gc.collect()
            kwargs = pipeline_kwargs_from_resolved_config(config)
            if is_masked_room:
                masked_path = spec.artifacts["masked_room_checkpoint"]["path"]
                kwargs.update(
                    room_generator_mode="discrete_masked",
                    masked_room_checkpoint=masked_path,
                    condition_encoder_checkpoint=masked_path,
                    logic_net_checkpoint=masked_path if variant.local_logic_loss else None,
                    strict_checkpoint_mode=True,
                    require_logic_net=variant.local_logic_loss,
                    device=config["runtime"]["device"],
                    enable_logging=False,
                    use_learned_refiner_rules=False,
                )
            else:
                diffusion_path = spec.artifacts["diffusion_checkpoint"]["path"]
                kwargs.update(
                    vqvae_checkpoint=spec.artifacts["vqvae_checkpoint"]["path"],
                    diffusion_checkpoint=diffusion_path,
                    condition_encoder_checkpoint=diffusion_path,
                    logic_net_checkpoint=diffusion_path if variant.local_logic_loss else None,
                    strict_checkpoint_mode=True,
                    require_logic_net=variant.local_logic_loss,
                    device=config["runtime"]["device"],
                    enable_logging=False,
                    use_learned_refiner_rules=False,
                )
            self._pipeline = self._factory(**kwargs)
            self._model_key = model_key
        return self._pipeline

    def generate(self, spec: AttemptSpec):
        spec.verify_artifacts()
        pipeline = self._load(spec)
        payload = json.loads(Path(spec.artifacts["mission_graph"]["path"]).read_text(encoding="utf-8"))
        graph = nx.node_link_graph(payload, edges="links")
        if not graph.number_of_nodes():
            raise ValueError("The sealed mission graph must not be empty.")
        kwargs = dict(spec.generation_kwargs)
        if "mission_graph" in kwargs or bool(kwargs.get("generate_topology", False)):
            raise ValueError("Generation cannot replace the sealed mission graph.")
        kwargs["seed"] = spec.metadata["sampling_seed"]
        if spec.metadata["inference_variant"] == "unguided":
            if float(kwargs.get("logic_guidance_scale", 0.0)) != 0.0:
                raise ValueError("Unguided attempt cannot enable LogicNet guidance.")
            kwargs["logic_guidance_scale"] = 0.0
        elif spec.metadata["inference_variant"] != "logic_guided":
            raise ValueError("Unknown sealed inference condition.")
        elif float(kwargs.get("logic_guidance_scale", 0.0)) <= 0.0:
            raise ValueError("A guided attempt requires an explicit positive guidance scale.")
        # Model construction consumes random numbers. Reset afterward so model
        # size, arm ordering, and cache hits cannot change the sampling stream.
        seed_everything(kwargs["seed"], cudnn_benchmark=False, cudnn_deterministic=True,
                        deterministic_algorithms=True)
        result = pipeline.generate_dungeon(mission_graph=graph, **kwargs)
        if getattr(result, "rooms", None):
            from .room_evidence import TrainingRoomReference
            config = validate_config(yaml.safe_load(Path(spec.artifacts["resolved_config"]["path"]).read_text(encoding="utf-8")))
            key = payload_sha256(config["dataset"])
            if key not in self._references:
                from .dataset_provenance import verify_dataset_manifest
                provenance = verify_dataset_manifest(config["dataset"]["data_dir"])
                if provenance["content_sha256"] != spec.metadata["dataset_manifest_sha256"]:
                    raise ValueError("Training reference corpus differs from the sealed dataset.")
                self._references[key] = TrainingRoomReference.from_config(config)
            for room in result.rooms.values():
                room.metrics["training_reference"] = self._references[key].compare(room.raw_neural_grid)
        return result

    def validator_factory(self, spec: AttemptSpec, generated):
        pipeline = self._load(spec)
        graph = generated.physical_mission_graph
        layout = generated.stitched_layout
        context = build_stitched_validation_context(graph, layout)
        if not has_complete_stitched_validation_context(context):
            raise ValueError("Exact stage validation requires the actual stitched graph/room mapping.")
        graph_evidence = _validate_mission_graph_exact(pipeline, graph)
        global_state = validate_attached_global_state_contract(
            graph, max_states=int(getattr(pipeline, "graph_oracle_max_states", 200_000)),
        )

        compiled_grid_ref = getattr(generated, "dungeon_grid", None)
        compiled_tile_cache = getattr(generated, "compiled_tile_validation", None)

        def validate(grid: np.ndarray, stage: str):
            # Recompute tile and spatial evidence for raw/constrained/repaired grids,
            # and reuse the already-computed compiled tile oracle when validating the
            # identical compiled grid.
            if (
                stage == "compiled"
                and isinstance(compiled_tile_cache, dict)
                and bool(compiled_tile_cache.get("is_exact", False))
                and bool(compiled_tile_cache.get("graph_tile_context_applied") is True)
                and compiled_grid_ref is not None
                and np.array_equal(np.asarray(grid, dtype=np.int32), np.asarray(compiled_grid_ref, dtype=np.int32))
            ):
                tile = dict(compiled_tile_cache)
            else:
                tile = _validate_dungeon(
                    pipeline, grid, room_puzzle_metadata=getattr(generated, "puzzle_metadata", {}),
                    **context,
                )
            spatial = validate_connection_realizations(
                grid, dict(getattr(layout, "connection_realizations", {}) or {}),
            )
            report = build_end_to_end_validation_report(
                dungeon_grid=grid, graph_validation=graph_evidence,
                global_state_validation=global_state.to_dict() if global_state else None,
                spatial_validation={**dict(getattr(layout, "realization_metrics", {}) or {}), **spatial},
                tile_validation=tile,
            )
            applicable = [item for item in report.hard_stages if item.applicable]
            unresolved = any(item.passed is None or not item.exact for item in applicable)
            unresolved = unresolved or graph_evidence.get("termination_status") == "not_applicable_missing_roles"
            unresolved = unresolved or tile.get("graph_tile_context_applied") is not True
            timeout = (any(item.status in {"timeout", "budget_exhausted"} for item in applicable)
                       or graph_evidence.get("all_rooms_reachability_status") == "budget_exhausted")
            unresolved = unresolved or timeout
            termination = ("timeout" if timeout else "validator_error" if unresolved
                           else "solved" if report.accepted else "invalid")
            return {"is_exact": not unresolved, "accepted": report.accepted,
                    "termination_status": termination, "stage": stage,
                    "contract": report.to_metrics()}
        return validate
