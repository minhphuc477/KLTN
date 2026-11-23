"""Freeze evaluation requests before training without opening model artifacts."""

from numbers import Integral
from pathlib import Path
import json

import networkx as nx

from .dataset_provenance import file_sha256
from .protocol import CONTROL_MODELS


def validate_mission_support(path):
    """Reject a test bank whose primary endpoint cannot be certified by the code.

    This checks representation support only, never model outcomes or whether a
    particular generated dungeon succeeds. Do not silently remove gate labels
    or drop unsupported source missions to make this check pass.
    """
    from src.validation.end_to_end import graph_tile_projection_issues
    graph = nx.node_link_graph(json.loads(Path(path).read_text(encoding="utf-8")), edges="links")
    if not graph.number_of_nodes():
        raise ValueError(f"Evaluation mission is empty: {path}")
    from src.simulation.graph_progression import starting_bomb_count
    starting_bomb_count(graph, 0)
    issues = graph_tile_projection_issues(graph, physical_bridge=True)
    undirected = graph.to_undirected()
    recorded_transitions = graph.graph.get('spatial_transition_policy') == 'recorded'
    if not recorded_transitions and max(dict(undirected.degree()).values(), default=0) > 4:
        issues.append("mission exceeds the current stitcher's four-neighbor room layout support")
    if not recorded_transitions and not nx.check_planarity(undirected)[0]:
        issues.append("nonplanar mission requires explicit crossing/transport support in the physical layout")
    if issues:
        raise ValueError(
            f"Evaluation mission {path} has unsupported graph-to-tile mechanics; "
            f"the primary exact-validity endpoint cannot be certified: {issues}. "
            "Implement and test the shared progression semantics before training."
        )


def seal_evaluation_design(*, mission_bank, sampling_seeds, generation_kwargs, fold_ids):
    seeds = list(sampling_seeds)
    if (not seeds or any(isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Sampling seeds must be unique nonnegative integers.")
    if set(mission_bank) != set(fold_ids) or any(not missions for missions in mission_bank.values()):
        raise ValueError("Evaluation missions must cover every declared fold.")
    if (any(key in generation_kwargs for key in ("seed", "mission_graph"))
            or generation_kwargs.get("generate_topology")
            or float(generation_kwargs.get("logic_guidance_scale", 0.0)) != 0.0):
        raise ValueError("Primary evaluation must use the sealed mission and seed with unguided inference.")
    missions = {}
    for fold, bank in mission_bank.items():
        missions[fold] = {}
        for mission_id, value in bank.items():
            if not isinstance(mission_id, str) or not mission_id.strip():
                raise ValueError("Evaluation mission IDs must be nonempty strings.")
            path = Path(value).resolve()
            validate_mission_support(path)
            missions[fold][mission_id] = {"path": str(path), "sha256": file_sha256(path)}
    return {"schema_version": "2.0", "missions": missions, "sampling_seeds": [int(seed) for seed in seeds],
            "independent_controls": list(CONTROL_MODELS),
            "control_interpretation": "training_only_room_sources_with_common_symbolic_repair_and_compilation",
            "generation_kwargs": {**generation_kwargs, "logic_guidance_scale": 0.0}}


def verify_evaluation_design(design, fold_ids):
    if not isinstance(design, dict) or design.get("schema_version") != "2.0":
        raise ValueError("Freeze an evaluation design before starting publication training.")
    current = seal_evaluation_design(
        mission_bank={fold: {name: item["path"] for name, item in bank.items()} for fold, bank in design["missions"].items()},
        sampling_seeds=design["sampling_seeds"], generation_kwargs=design["generation_kwargs"], fold_ids=fold_ids,
    )
    if current != design:
        raise ValueError("Evaluation missions changed after sealing.")
