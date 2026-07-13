#!/usr/bin/env python
"""Materialize every selected topology-QD elite into exact final maps.

This command deliberately consumes only provenance-bearing pickle archives.
It recompiles each archived *phenotype* through the final graph export gate,
generates rooms for one or more paired seeds, and records the end-to-end hard
validation contract. Topology archive coverage and surviving final-artifact
coverage are reported separately.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
from networkx.readwrite import json_graph


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_system import merge_config  # noqa: E402
from src.generation.evolutionary_director import EvolutionaryTopologyGenerator  # noqa: E402
from src.generation.evolutionary_director.converters import (  # noqa: E402
    networkx_to_mission_graph,
)
from src.generation.grammar import MissionGraph  # noqa: E402
from src.pipeline.config_bridge import pipeline_kwargs_from_resolved_config  # noqa: E402
from src.pipeline.dungeon_pipeline import NeuralSymbolicDungeonPipeline  # noqa: E402


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return repr(value)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_value(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_archive_payload(path: Path, *, trust_pickle: bool) -> Dict[str, Any]:
    """Load a native QD archive only after explicit pickle trust consent."""
    if not trust_pickle:
        raise ValueError(
            "Topology QD archives are Python pickle files and can execute code. "
            "Pass --trust-pickle only for an archive you created and trust."
        )
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("QD archive must be a versioned dictionary payload.")
    if int(payload.get("version", 0) or 0) < 3:
        raise ValueError(
            "QD archive predates phenotype snapshots. Genome-only archives "
            "cannot reproduce evaluated elites and must be regenerated."
        )
    if payload.get("archive") is None:
        raise ValueError("QD archive payload has no archive object.")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("QD archive has no generator provenance contract.")
    return payload


def generator_from_archive_payload(
    payload: Mapping[str, Any],
) -> EvolutionaryTopologyGenerator:
    provenance = dict(payload.get("provenance", {}) or {})
    kwargs = dict(provenance.get("generator_kwargs", {}) or {})
    if not kwargs:
        raise ValueError("QD archive provenance has no generator kwargs.")
    kwargs.update(
        {
            "qd_archive_path": None,
            "qd_load_archive": False,
            "qd_autosave_archive": False,
        }
    )
    generator = EvolutionaryTopologyGenerator(**kwargs)
    generator.validate_qd_archive_provenance(payload)
    return generator


def _parse_seeds(raw: str) -> list[int]:
    values = [token.strip() for token in str(raw).split(",") if token.strip()]
    if not values:
        raise ValueError("At least one materialization seed is required.")
    seeds = [int(value) for value in values]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Materialization seeds must be unique.")
    return seeds


def _ordered_elites(archive: Any) -> list[tuple[int, Any]]:
    storage = dict(getattr(archive, "archive", {}) or {})
    return sorted(
        ((int(cell), elite) for cell, elite in storage.items()),
        key=lambda item: item[0],
    )


def _final_graph_features(
    generator: EvolutionaryTopologyGenerator,
    graph: Any,
) -> tuple[tuple[float, ...], Dict[str, Any]]:
    mission_graph = networkx_to_mission_graph(graph)
    evaluation = dict(generator.evaluator.evaluate_graph(mission_graph))
    metrics = dict(evaluation.get("descriptor_metrics", {}) or {})
    features = tuple(
        float(np.clip(metrics.get(name, 0.0), 0.0, 1.0))
        for name in (
            "linearity",
            "leniency",
            "progression_complexity",
            "topology_complexity",
        )
    )
    return features, evaluation


def _build_pipeline(args: argparse.Namespace) -> NeuralSymbolicDungeonPipeline:
    resolved = merge_config(yaml_path=str(args.config), cli_overrides=None)
    kwargs = pipeline_kwargs_from_resolved_config(resolved)
    overrides = {
        "vqvae_checkpoint": args.vqvae_checkpoint,
        "diffusion_checkpoint": args.diffusion_checkpoint,
        "condition_encoder_checkpoint": args.condition_encoder_checkpoint,
        "logic_net_checkpoint": args.logic_net_checkpoint,
        "masked_room_checkpoint": args.masked_room_checkpoint,
        "fast_sampling_checkpoint": args.fast_sampling_checkpoint,
    }
    kwargs.update({key: str(value) for key, value in overrides.items() if value})
    kwargs.update(
        {
            "strict_checkpoint_mode": True,
            "require_logic_net": bool(args.require_logic_net),
            "device": str(args.device),
            "default_end_to_end_validation_mode": "reject",
            "default_verify_solver_consistency": bool(
                args.verify_solver_consistency
            ),
        }
    )
    return NeuralSymbolicDungeonPipeline.from_kwargs(**kwargs)


def materialize(args: argparse.Namespace) -> Dict[str, Any]:
    payload = load_archive_payload(args.archive, trust_pickle=args.trust_pickle)
    archive = payload["archive"]
    generator = generator_from_archive_payload(payload)
    pipeline = _build_pipeline(args)
    seeds = _parse_seeds(args.seeds)
    elites = _ordered_elites(archive)
    if args.max_elites is not None:
        elites = elites[: max(0, int(args.max_elites))]
    if not elites:
        raise ValueError("No topology elites were selected for materialization.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[Dict[str, Any]] = []
    robust_source_cells: set[int] = set()
    any_valid_source_cells: set[int] = set()
    robust_final_descriptor_cells: set[int] = set()
    any_valid_final_descriptor_cells: set[int] = set()
    started_at = time.time()

    for cell, elite in elites:
        metadata = dict(getattr(elite, "metadata", {}) or {})
        phenotype = metadata.get("phenotype")
        source_features = tuple(float(value) for value in elite.features)
        cell_record: Dict[str, Any] = {
            "source_cell": int(cell),
            "source_fitness": float(elite.fitness),
            "source_features": source_features,
            "genome": [int(gene) for gene in elite.solution],
            "runs": [],
        }
        if not isinstance(phenotype, MissionGraph):
            cell_record["status"] = "missing_archived_phenotype"
            records.append(cell_record)
            _atomic_json(
                output_dir / "materialization_progress.json",
                {"records": records},
            )
            continue

        try:
            graph = generator.finalize_archived_phenotype(
                phenotype,
                directed_output=True,
            )
            final_features, final_graph_evaluation = _final_graph_features(
                generator,
                graph,
            )
            final_cell = int(archive.find_cell(final_features))
            feature_delta = np.asarray(final_features) - np.asarray(source_features)
            cell_record.update(
                {
                    "status": "graph_finalized",
                    "final_graph_features": final_features,
                    "final_graph_cell": final_cell,
                    "graph_cell_preserved": final_cell == int(cell),
                    "graph_feature_l1_drift": float(np.abs(feature_delta).sum()),
                    "graph_feature_l2_drift": float(np.linalg.norm(feature_delta)),
                    "final_graph_evaluation": final_graph_evaluation,
                }
            )
            graph_path = output_dir / f"cell_{cell:05d}" / "mission_graph.json"
            _atomic_json(
                graph_path,
                json_graph.node_link_data(graph, edges="links"),
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            cell_record.update(
                {"status": "graph_finalization_failed", "error": str(exc)}
            )
            records.append(cell_record)
            _atomic_json(
                output_dir / "materialization_progress.json",
                {"records": records},
            )
            continue

        accepted_seeds = 0
        for seed in seeds:
            run_dir = output_dir / f"cell_{cell:05d}" / f"seed_{seed}"
            run_started = time.time()
            try:
                result = pipeline.generate_dungeon(
                    mission_graph=graph.copy(),
                    generate_topology=False,
                    apply_repair=bool(args.apply_repair),
                    enable_map_elites=True,
                    seed=int(seed),
                    num_diffusion_steps=int(args.diffusion_steps),
                )
                metrics = dict(result.metrics or {})
                accepted = bool(metrics.get("end_to_end_validation_accepted", False))
                if not accepted:
                    raise RuntimeError(
                        "Pipeline returned without an accepted end-to-end contract."
                    )
                run_dir.mkdir(parents=True, exist_ok=True)
                np.save(run_dir / "dungeon_grid.npy", result.dungeon_grid)
                _atomic_json(run_dir / "metrics.json", metrics)
                accepted_seeds += 1
                run_record = {
                    "seed": int(seed),
                    "status": "accepted",
                    "elapsed_sec": float(time.time() - run_started),
                    "grid_shape": list(np.asarray(result.dungeon_grid).shape),
                    "end_to_end_validation_accepted": True,
                }
            except Exception as exc:  # Preserve the complete archive audit ledger.
                run_record = {
                    "seed": int(seed),
                    "status": "failed",
                    "elapsed_sec": float(time.time() - run_started),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            cell_record["runs"].append(run_record)

        if accepted_seeds > 0:
            any_valid_source_cells.add(int(cell))
            any_valid_final_descriptor_cells.add(int(cell_record["final_graph_cell"]))
        if accepted_seeds == len(seeds):
            robust_source_cells.add(int(cell))
            robust_final_descriptor_cells.add(int(cell_record["final_graph_cell"]))
        cell_record["accepted_seed_count"] = int(accepted_seeds)
        cell_record["seed_count"] = int(len(seeds))
        cell_record["all_seeds_accepted"] = accepted_seeds == len(seeds)
        records.append(cell_record)
        _atomic_json(output_dir / "materialization_progress.json", {"records": records})

    num_cells = int(getattr(archive, "num_cells", 0) or 0)
    selected_count = len(elites)
    accepted_runs = sum(
        1
        for record in records
        for run in list(record.get("runs", []) or [])
        if run.get("status") == "accepted"
    )
    attempted_runs = sum(len(list(record.get("runs", []) or [])) for record in records)
    topology_elite_count = len(getattr(archive, "archive", {}) or {})
    complete_archive_scan = (
        args.max_elites is None and selected_count == topology_elite_count
    )
    graph_finalized_count = sum(
        record.get("status") == "graph_finalized" for record in records
    )
    graph_cell_preserved_count = sum(
        bool(record.get("graph_cell_preserved", False)) for record in records
    )
    complete_archive_materialization = bool(
        complete_archive_scan
        and graph_finalized_count == topology_elite_count
        and len(robust_source_cells) == topology_elite_count
    )
    summary = {
        "schema_version": 1,
        "archive_path": str(args.archive.resolve()),
        "archive_cells": num_cells,
        "topology_elite_count": int(topology_elite_count),
        "topology_archive_coverage": float(
            len(getattr(archive, "archive", {}) or {}) / max(1, num_cells)
        ),
        "selected_elite_count": int(selected_count),
        "complete_archive_scan": bool(complete_archive_scan),
        "graph_finalized_count": int(graph_finalized_count),
        "all_selected_graphs_finalized": bool(
            graph_finalized_count == selected_count
        ),
        "graph_cell_preserved_count": int(graph_cell_preserved_count),
        "graph_cell_preservation_rate": float(
            graph_cell_preserved_count / max(1, graph_finalized_count)
        ),
        "complete_archive_materialization": complete_archive_materialization,
        "paired_seeds": seeds,
        "attempted_final_map_runs": int(attempted_runs),
        "accepted_final_map_runs": int(accepted_runs),
        "final_map_sample_acceptance_rate": float(
            accepted_runs / max(1, attempted_runs)
        ),
        "any_seed_valid_source_cell_count": int(len(any_valid_source_cells)),
        "all_seeds_valid_source_cell_count": int(len(robust_source_cells)),
        "any_seed_source_elite_survival_rate": float(
            len(any_valid_source_cells) / max(1, topology_elite_count)
        ),
        "all_seeds_source_elite_survival_rate": float(
            len(robust_source_cells) / max(1, topology_elite_count)
        ),
        "any_seed_final_descriptor_cell_count": int(
            len(any_valid_final_descriptor_cells)
        ),
        "all_seeds_final_descriptor_cell_count": int(
            len(robust_final_descriptor_cells)
        ),
        "any_seed_valid_archive_coverage": float(
            len(any_valid_final_descriptor_cells) / max(1, num_cells)
        ),
        "robust_final_map_archive_coverage": float(
            len(robust_final_descriptor_cells) / max(1, num_cells)
        ),
        "elapsed_sec": float(time.time() - started_at),
        "records": records,
    }
    _atomic_json(output_dir / "materialization_summary.json", summary)
    return summary


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register CLI arguments on a standalone or repository-root parser."""
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--trust-pickle", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "zelda_hmolqd.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--max-elites", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--apply-repair", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify-solver-consistency", action="store_true")
    parser.add_argument("--require-logic-net", action="store_true")
    parser.add_argument("--vqvae-checkpoint", type=Path)
    parser.add_argument("--diffusion-checkpoint", type=Path)
    parser.add_argument("--condition-encoder-checkpoint", type=Path)
    parser.add_argument("--logic-net-checkpoint", type=Path)
    parser.add_argument("--masked-room-checkpoint", type=Path)
    parser.add_argument("--fast-sampling-checkpoint", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return parser


def run_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Root-CLI compatible materialization entrypoint."""
    return materialize(args)


def main() -> None:
    args = build_parser().parse_args()
    summary = run_from_args(args)
    print(
        json.dumps(
            {
                "complete_archive_materialization": summary[
                    "complete_archive_materialization"
                ],
                "robust_final_map_archive_coverage": summary[
                    "robust_final_map_archive_coverage"
                ],
                "final_map_sample_acceptance_rate": summary[
                    "final_map_sample_acceptance_rate"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
