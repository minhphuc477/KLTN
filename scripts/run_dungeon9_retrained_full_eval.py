#!/usr/bin/env python3
"""Evaluate a retrained neural pipeline on the held-out Dungeon 9 graphs.

This runner is intentionally separate from the lightweight audit protocol:
it consumes freshly retrained VQ-VAE/diffusion checkpoints, generates rooms on
the actual Dungeon 9 mission graphs, and reports both room-level novelty and
dungeon-level oracle solvability for the required core ablations.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_dungeon9_holdout_protocol import (  # noqa: E402
    RoomSpec,
    _boundary_artifacts,
    _leniency_score,
    _load_rooms,
    _logic_violations,
    _missing_required_doors,
    _nearest_hamming_stats,
    _pairwise_hamming,
    _room_playable,
    _write_csv,
)
from src.config_system import load_resolved_config_for_artifact  # noqa: E402
from src.pipeline.dungeon_pipeline import (  # noqa: E402
    NeuralSymbolicDungeonPipeline,
    pipeline_kwargs_from_resolved_config,
)
from src.simulation.validator import StateSpaceAStar, ZeldaLogicEnv  # noqa: E402
from src.zelda_data.splits import DEFAULT_TRAIN_DUNGEONS, DEFAULT_VARIANTS  # noqa: E402
from src.zelda_data.zelda_core import ZeldaDungeonAdapter  # noqa: E402


@dataclass(frozen=True)
class EvalConfig:
    name: str
    use_graph: bool = True
    use_logicnet: bool = True
    use_wfc: bool = True
    topology_refinement_mode: str = "gat2"


EVAL_CONFIGS: Tuple[EvalConfig, ...] = (
    EvalConfig("FULL", use_graph=True, use_logicnet=True, use_wfc=True, topology_refinement_mode="gat2"),
    EvalConfig("NO_GRAPH", use_graph=False, use_logicnet=True, use_wfc=True, topology_refinement_mode="none"),
    EvalConfig("NO_LOGICNET", use_graph=True, use_logicnet=False, use_wfc=True, topology_refinement_mode="gat2"),
    EvalConfig("NO_WFC", use_graph=True, use_logicnet=True, use_wfc=False, topology_refinement_mode="gat2"),
    EvalConfig("NO_LOGICNET_NO_WFC", use_graph=True, use_logicnet=False, use_wfc=False, topology_refinement_mode="gat2"),
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _load_pipeline(
    *,
    vqvae_checkpoint: str,
    diffusion_checkpoint: str,
    logic_net_checkpoint: Optional[str],
    condition_encoder_checkpoint: Optional[str],
    device: str,
) -> NeuralSymbolicDungeonPipeline:
    resolved = load_resolved_config_for_artifact(diffusion_checkpoint)
    pipeline_kwargs: Dict[str, Any] = {}
    if isinstance(resolved, dict):
        pipeline_kwargs.update(pipeline_kwargs_from_resolved_config(resolved))
    pipeline_kwargs.update(
        {
            "vqvae_checkpoint": vqvae_checkpoint,
            "diffusion_checkpoint": diffusion_checkpoint,
            "logic_net_checkpoint": logic_net_checkpoint or diffusion_checkpoint,
            "condition_encoder_checkpoint": condition_encoder_checkpoint or diffusion_checkpoint,
            "condition_use_reference_room_maps": True,
            "room_generator_mode": "latent_diffusion",
            "device": device,
            "use_learned_refiner_rules": True,
            "enable_logging": False,
        }
    )
    return NeuralSymbolicDungeonPipeline(**pipeline_kwargs)


def _specs_for_generated_rooms(
    *,
    variant: int,
    dungeon: Any,
    graph: nx.Graph,
    room_ids: Sequence[Any],
) -> List[RoomSpec]:
    node_to_room: Dict[Any, Any] = {}
    for room in getattr(dungeon, "rooms", {}).values():
        node_id = getattr(room, "graph_node_id", None)
        if node_id is not None:
            node_to_room[node_id] = room

    specs: List[RoomSpec] = []
    for room_id in room_ids:
        attrs = dict(graph.nodes[room_id]) if room_id in graph else {}
        source_room = node_to_room.get(room_id)
        source_doors = dict(getattr(source_room, "doors", {}) or {}) if source_room is not None else {}
        required_doors = tuple(
            direction for direction in ("N", "S", "W", "E") if bool(source_doors.get(direction, False))
        )
        specs.append(
            RoomSpec(
                dungeon_num=9,
                variant=int(variant),
                coord=(0, int(len(specs))),
                required_doors=required_doors,
                is_start=bool(
                    attrs.get("is_start")
                    or attrs.get("is_start_pointer")
                    or attrs.get("is_entry")
                    or bool(getattr(source_room, "is_start", False))
                ),
                is_goal=bool(
                    attrs.get("is_triforce")
                    or attrs.get("is_goal")
                    or bool(getattr(source_room, "has_triforce", False))
                ),
            )
        )
    return specs


def _dungeon_solvable(grid: np.ndarray, timeout: int) -> Tuple[bool, int, str]:
    try:
        env = ZeldaLogicEnv(semantic_grid=np.asarray(grid, dtype=np.int32))
        solver = StateSpaceAStar(env, timeout=int(timeout), search_mode="astar")
        success, path, stats = solver.solve()
        path_len = len(path or []) if success else 0
        reason = ""
        if not success and isinstance(stats, Mapping):
            reason = str(stats.get("reason", ""))
        return bool(success), int(path_len), reason
    except Exception as exc:  # noqa: BLE001 - persisted as diagnostic, not re-raised per sample.
        return False, 0, type(exc).__name__


def _linearity_from_graph(graph: nx.Graph) -> float:
    degrees: List[int] = []
    for node in graph.nodes:
        if graph.is_directed():
            degree = int(graph.in_degree(node) + graph.out_degree(node))
        else:
            degree = int(graph.degree(node))
        degrees.append(degree)
    if not degrees:
        return 0.0
    scores: List[float] = []
    for degree in degrees:
        if degree <= 2:
            scores.append(1.0)
        elif degree == 3:
            scores.append(0.35)
        else:
            scores.append(0.0)
    return float(statistics.fmean(scores))


def _summarize_config(
    *,
    config_name: str,
    rows: Sequence[Mapping[str, Any]],
    all_rooms: Sequence[np.ndarray],
    all_specs: Sequence[RoomSpec],
    train_rooms: Sequence[np.ndarray],
) -> Dict[str, Any]:
    solvable = [bool(row.get("dungeon_solvable", False)) for row in rows]
    room_playable = [_room_playable(room, spec) for room, spec in zip(all_rooms, all_specs)]
    missing_doors = [_missing_required_doors(room, spec) for room, spec in zip(all_rooms, all_specs)]
    required_door_count = [len(spec.required_doors) for spec in all_specs]
    logic_violations = [_logic_violations(room, spec) for room, spec in zip(all_rooms, all_specs)]
    leniency = [_leniency_score(room) for room in all_rooms]
    boundary_artifacts = [_boundary_artifacts(room) for room in all_rooms]
    repair_rates = [float(row.get("room_repair_rate", 0.0) or 0.0) for row in rows]
    generation_times = [float(row.get("generation_time_sec", 0.0) or 0.0) for row in rows]
    graph_linearity = [float(row.get("graph_linearity", 0.0) or 0.0) for row in rows]
    novelty = _nearest_hamming_stats(all_rooms, train_rooms)
    return {
        "config": config_name,
        "dungeon_runs": int(len(rows)),
        "rooms": int(len(all_rooms)),
        "dungeon_solvable_rate": float(statistics.fmean(float(v) for v in solvable)) if solvable else 0.0,
        "room_playable_rate": float(statistics.fmean(float(v) for v in room_playable)) if room_playable else 0.0,
        "linearity": float(statistics.fmean(graph_linearity)) if graph_linearity else 0.0,
        "leniency": float(statistics.fmean(leniency)) if leniency else 0.0,
        "graph_door_violation_rate": float(sum(missing_doors) / max(1, int(sum(required_door_count)))),
        "logic_violation_rate": float(
            sum(1 for value in logic_violations if int(value) > 0) / max(1, len(logic_violations))
        ),
        "boundary_artifacts_mean": float(statistics.fmean(boundary_artifacts)) if boundary_artifacts else 0.0,
        "boundary_artifacts_total": int(sum(boundary_artifacts)),
        "room_repair_rate": float(statistics.fmean(repair_rates)) if repair_rates else 0.0,
        "generation_time_sec_mean": float(statistics.fmean(generation_times)) if generation_times else 0.0,
        "generation_time_sec_total": float(sum(generation_times)),
        "pairwise_diversity_hamming": _pairwise_hamming(all_rooms),
        **novelty,
    }


def run_eval(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = [int(v.strip()) for v in str(args.variants).split(",") if v.strip()]
    seeds = [int(v.strip()) for v in str(args.seeds).split(",") if v.strip()]
    selected = {v.strip().upper() for v in str(args.configs).split(",") if v.strip()}
    configs = [cfg for cfg in EVAL_CONFIGS if not selected or cfg.name in selected]
    if not configs:
        raise ValueError("No evaluation configs selected.")

    train_rooms, _train_meta = _load_rooms(
        Path(args.data_root),
        tuple(int(v) for v in DEFAULT_TRAIN_DUNGEONS),
        tuple(int(v) for v in DEFAULT_VARIANTS),
    )

    pipeline = _load_pipeline(
        vqvae_checkpoint=str(args.vqvae_checkpoint),
        diffusion_checkpoint=str(args.diffusion_checkpoint),
        logic_net_checkpoint=str(args.logic_net_checkpoint) if args.logic_net_checkpoint else None,
        condition_encoder_checkpoint=(
            str(args.condition_encoder_checkpoint) if args.condition_encoder_checkpoint else None
        ),
        device=str(args.device),
    )
    adapter = ZeldaDungeonAdapter(str(args.data_root))

    raw_rows: List[Dict[str, Any]] = []
    room_rows: List[Dict[str, Any]] = []
    rooms_by_config: Dict[str, List[np.ndarray]] = {cfg.name: [] for cfg in configs}
    specs_by_config: Dict[str, List[RoomSpec]] = {cfg.name: [] for cfg in configs}

    for cfg in configs:
        pipeline.use_graph_node_cross_attention = bool(cfg.use_graph)
        try:
            pipeline.diffusion.set_topology_refinement_mode(str(cfg.topology_refinement_mode))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            logger.debug("Unable to set topology refinement mode for %s.", cfg.name, exc_info=True)

        for variant in variants:
            dungeon = adapter.load_dungeon(9, int(variant))
            graph = copy.deepcopy(dungeon.graph)
            graph_linearity = _linearity_from_graph(graph)
            for seed in seeds:
                print(
                    f"[eval] config={cfg.name} variant={variant} seed={seed} "
                    f"nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}",
                    flush=True,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                started = time.perf_counter()
                try:
                    result = pipeline.generate_dungeon(
                        mission_graph=graph,
                        generate_topology=False,
                        guidance_scale=float(args.guidance_scale),
                        logic_guidance_scale=(
                            float(args.logic_guidance_scale) if cfg.use_logicnet else 0.0
                        ),
                        num_diffusion_steps=int(args.diffusion_steps),
                        latent_sampler="diffusion",
                        apply_repair=bool(cfg.use_wfc),
                        enable_map_elites=False,
                        seed=int(seed) + int(variant) * 1000,
                        batch_independent_rooms=not bool(args.sequential_rooms),
                        max_batch_size=int(args.max_batch_size),
                    )
                    generation_time = float(getattr(result, "generation_time", time.perf_counter() - started))
                    grid = np.asarray(result.dungeon_grid, dtype=np.int32)
                    dungeon_solvable, path_len, failure_reason = _dungeon_solvable(
                        grid,
                        timeout=int(args.astar_timeout),
                    )
                    room_items = list(result.rooms.items())
                    generated_rooms = [np.asarray(room.room_grid, dtype=np.int32) for _, room in room_items]
                    generated_specs = _specs_for_generated_rooms(
                        variant=int(variant),
                        dungeon=dungeon,
                        graph=result.mission_graph,
                        room_ids=[room_id for room_id, _ in room_items],
                    )
                    rooms_by_config[cfg.name].extend(generated_rooms)
                    specs_by_config[cfg.name].extend(generated_specs)
                    repaired = [bool(getattr(room, "was_repaired", False)) for _, room in room_items]
                    tiles_repaired = [
                        float((getattr(room, "metrics", {}) or {}).get("tiles_changed", 0.0))
                        for _, room in room_items
                    ]
                    raw_row = {
                        "config": cfg.name,
                        "variant": int(variant),
                        "seed": int(seed),
                        "graph_nodes": int(graph.number_of_nodes()),
                        "graph_edges": int(graph.number_of_edges()),
                        "generated_rooms": int(len(generated_rooms)),
                        "generation_time_sec": generation_time,
                        "dungeon_solvable": bool(dungeon_solvable),
                        "astar_path_length": int(path_len),
                        "failure_reason": failure_reason,
                        "room_repair_rate": float(statistics.fmean(float(v) for v in repaired))
                        if repaired
                        else 0.0,
                        "tiles_repaired": float(sum(tiles_repaired)),
                        "graph_linearity": float(graph_linearity),
                    }
                    raw_rows.append(raw_row)
                    for (room_id, _room_result), room_grid, spec in zip(room_items, generated_rooms, generated_specs):
                        room_rows.append(
                            {
                                "config": cfg.name,
                                "variant": int(variant),
                                "seed": int(seed),
                                "room_id": str(room_id),
                                "playable": bool(_room_playable(room_grid, spec)),
                                "missing_required_doors": int(_missing_required_doors(room_grid, spec)),
                                "logic_violations": int(_logic_violations(room_grid, spec)),
                                "leniency": float(_leniency_score(room_grid)),
                                "boundary_artifacts": int(_boundary_artifacts(room_grid)),
                            }
                        )
                    print(
                        f"[eval] done config={cfg.name} variant={variant} seed={seed} "
                        f"rooms={len(generated_rooms)} solvable={bool(dungeon_solvable)} "
                        f"time={generation_time:.2f}s",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001 - one failed sample should not erase the run.
                    print(
                        f"[eval] failed config={cfg.name} variant={variant} seed={seed}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    raw_rows.append(
                        {
                            "config": cfg.name,
                            "variant": int(variant),
                            "seed": int(seed),
                            "graph_nodes": int(graph.number_of_nodes()),
                            "graph_edges": int(graph.number_of_edges()),
                            "generated_rooms": 0,
                            "generation_time_sec": float(time.perf_counter() - started),
                            "dungeon_solvable": False,
                            "astar_path_length": 0,
                            "failure_reason": f"{type(exc).__name__}: {exc}",
                            "room_repair_rate": 0.0,
                            "tiles_repaired": 0.0,
                            "graph_linearity": float(graph_linearity),
                        }
                    )

    summary_rows = [
        _summarize_config(
            config_name=cfg.name,
            rows=[row for row in raw_rows if row.get("config") == cfg.name],
            all_rooms=rooms_by_config[cfg.name],
            all_specs=specs_by_config[cfg.name],
            train_rooms=train_rooms,
        )
        for cfg in configs
    ]

    _write_csv(output_dir / "full_retrained_raw.csv", raw_rows)
    _write_csv(output_dir / "full_retrained_rooms.csv", room_rows)
    _write_csv(output_dir / "full_retrained_summary.csv", summary_rows)
    payload = {
        "manifest": {
            "name": "dungeon9_retrained_full_eval",
            "variants": variants,
            "seeds": seeds,
            "configs": [cfg.__dict__ for cfg in configs],
            "diffusion_steps": int(args.diffusion_steps),
            "guidance_scale": float(args.guidance_scale),
            "logic_guidance_scale": float(args.logic_guidance_scale),
            "astar_timeout": int(args.astar_timeout),
        },
        "summary": summary_rows,
        "raw": raw_rows,
    }
    _write_json(output_dir / "full_retrained_results.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full retrained Dungeon 9 holdout evaluation.")
    parser.add_argument("--data-root", type=Path, default=Path("Data") / "The Legend of Zelda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vqvae-checkpoint", type=Path, required=True)
    parser.add_argument("--diffusion-checkpoint", type=Path, required=True)
    parser.add_argument("--logic-net-checkpoint", type=Path, default=None)
    parser.add_argument("--condition-encoder-checkpoint", type=Path, default=None)
    parser.add_argument("--variants", type=str, default="1,2")
    parser.add_argument("--seeds", type=str, default="20260515,20260516,20260517")
    parser.add_argument("--configs", type=str, default="FULL,NO_GRAPH,NO_LOGICNET,NO_WFC")
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--logic-guidance-scale", type=float, default=1.0)
    parser.add_argument("--astar-timeout", type=int, default=250000)
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--sequential-rooms", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    payload = run_eval(parse_args())
    print(json.dumps(_json_safe({"summary": payload["summary"]}), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
