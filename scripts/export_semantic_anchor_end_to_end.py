"""
Export end-to-end dungeon artifacts from a semantic-anchor retrain run.

This reuses one generated mission graph and exports:

- diffusion teacher
- fast sampler
- masked-room branch

side by side so the new checkpoints can be compared on the same topology.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import networkx as nx
import numpy as np
from networkx.readwrite import json_graph

try:
    import torch
except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
    torch = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_fast_sampler_visual_audit import (
    _resolve_vqvae_checkpoint,
    _resolve_export_device,
    _resolve_export_execution_kwargs,
    _generate_dungeon_with_oom_backoff,
    add_generation_override_args,
    build_validation_context_from_generation_result,
    build_pipeline,
    _compute_generation_validation,
    build_validation_search_stats_payload,
    export_variant,
    generation_overrides_from_namespace,
    _generation_policy_summary,
    save_grid_png,
    save_grid_json,
    save_grid_txt,
    save_rooms_sheet,
    save_stylized_grid_png,
    save_stylized_rooms_sheet,
    write_room_layout_artifacts,
)
from src.pipeline.dungeon_pipeline import NeuralSymbolicDungeonPipeline, pipeline_kwargs_from_resolved_config


def _emit_progress(message: str) -> None:
    print(f"[export-semantic-anchor] {message}", flush=True)


def _release_torch_memory() -> None:
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolve_masked_room_checkpoint(run_dir: Path) -> Path:
    for candidate in (
        run_dir / "checkpoints" / "masked_room" / "masked_room_best.pth",
        run_dir / "checkpoints" / "masked_room" / "masked_room_final.pth",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not resolve a masked-room checkpoint under "
        f"{run_dir / 'checkpoints' / 'masked_room'}."
    )


def build_masked_room_pipeline(
    run_dir: Path,
    *,
    generation_overrides: Dict[str, Any] | None = None,
    device_override: str | None = None,
) -> NeuralSymbolicDungeonPipeline:
    resolved = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    if generation_overrides:
        generation = resolved.setdefault("generation", {})
        for key, value in generation_overrides.items():
            generation[str(key)] = value
    export_device = str(device_override).strip().lower() if device_override else _resolve_export_device(resolved)
    pipeline_kwargs = pipeline_kwargs_from_resolved_config(resolved)
    vqvae_checkpoint = _resolve_vqvae_checkpoint(run_dir)
    masked_room_checkpoint = _resolve_masked_room_checkpoint(run_dir)
    diffusion_checkpoint = run_dir / "checkpoints" / "diffusion" / "best_model.pth"

    pipeline_kwargs.update(
        {
            "room_generator_mode": "discrete_masked",
            "masked_room_checkpoint": str(masked_room_checkpoint),
        }
    )

    return NeuralSymbolicDungeonPipeline(
        vqvae_checkpoint=str(vqvae_checkpoint),
        diffusion_checkpoint=str(diffusion_checkpoint) if diffusion_checkpoint.exists() else None,
        condition_encoder_checkpoint=str(masked_room_checkpoint),
        logic_net_checkpoint=str(diffusion_checkpoint) if diffusion_checkpoint.exists() else None,
        device=export_device,
        enable_logging=False,
        **pipeline_kwargs,
    )


def export_masked_variant(
    *,
    run_dir: Path,
    mission_graph: nx.Graph,
    variant_name: str,
    out_dir: Path,
    seed: int,
    generation_overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    execution_kwargs = _resolve_export_execution_kwargs()
    pipeline, result, generation_execution = _generate_dungeon_with_oom_backoff(
        pipeline_builder=build_masked_room_pipeline,
        run_dir=run_dir,
        mission_graph=mission_graph,
        generation_overrides=generation_overrides,
        execution_kwargs=execution_kwargs,
        status_writer=lambda *_args, **_kwargs: None,
        generation_kwargs={
            "generate_topology": False,
            "apply_repair": True,
            "enable_map_elites": False,
            "seed": int(seed),
        },
    )

    variant_dir = out_dir / variant_name
    rooms_dir = variant_dir / "rooms"
    room_grids: Dict[int, np.ndarray] = {}
    room_hashes: Dict[str, str] = {}

    for room_id, room in sorted(result.rooms.items(), key=lambda kv: int(kv[0])):
        grid = np.asarray(room.room_grid, dtype=np.int32)
        room_grids[int(room_id)] = grid
        room_text = save_grid_txt(grid, rooms_dir / f"room_{room_id}.txt")
        room_hashes[str(room_id)] = hashlib.sha256(room_text.encode("utf-8")).hexdigest()[:16]
        save_grid_png(grid, rooms_dir / f"room_{room_id}.png", tile_px=20)
        save_stylized_grid_png(grid, rooms_dir / f"room_{room_id}_stylized.png", tile_px=20, crop_void=False)

    dungeon_grid = np.asarray(result.dungeon_grid, dtype=np.int32)
    save_grid_png(dungeon_grid, variant_dir / "dungeon_grid.png", tile_px=16, crop_void=False)
    save_grid_json(dungeon_grid, variant_dir / "dungeon_grid_ids.json")
    preview = save_grid_txt(dungeon_grid, variant_dir / "dungeon_grid.txt")
    (variant_dir / "preview.txt").write_text(preview, encoding="utf-8")
    save_stylized_grid_png(dungeon_grid, variant_dir / "dungeon_grid_stylized.png", tile_px=20, crop_void=True)
    save_rooms_sheet(room_grids, variant_dir / "rooms_sheet.png", tile_px=16, columns=4)
    save_stylized_rooms_sheet(room_grids, variant_dir / "rooms_sheet_stylized.png", tile_px=18, columns=4)
    layout_payload = write_room_layout_artifacts(
        dungeon_grid=dungeon_grid,
        rooms=result.rooms,
        mission_graph=mission_graph,
        variant_dir=variant_dir,
        tile_px=20,
    )
    result_metrics = dict(result.metrics)
    generation_time_sec = float(result.generation_time)
    runtime_diagnostics = dict(pipeline.runtime_diagnostics)
    topology_anchor_policy = _generation_policy_summary(pipeline)
    validation_context = build_validation_context_from_generation_result(result)
    del room_grids
    del pipeline
    del result
    _release_torch_memory()
    validation = _compute_generation_validation(
        dungeon_grid=dungeon_grid,
        mission_graph=mission_graph,
        **validation_context,
    )

    summary = {
        "name": variant_name,
        "room_generator_mode": "discrete_masked",
        "generation_overrides_applied": dict(generation_overrides or {}),
        "generation_execution": generation_execution,
        "metrics": {
            **result_metrics,
            "generation_time_sec": generation_time_sec,
        },
        "runtime_diagnostics": runtime_diagnostics,
        "topology_anchor_policy": topology_anchor_policy,
        "semantic_metrics": {
            key: result_metrics.get(key)
            for key in (
                "total_graph_marker_expected",
                "total_graph_marker_overwrites",
                "avg_neural_graph_marker_exact_match_rate",
                "avg_final_pre_overlay_graph_marker_exact_match_rate",
                "avg_final_post_overlay_graph_marker_exact_match_rate",
                "avg_final_graph_marker_overwrite_rate",
                "avg_neural_semantic_anchor_error",
                "avg_final_pre_overlay_semantic_anchor_error",
                "avg_final_post_overlay_semantic_anchor_error",
            )
        },
        "tile_hist": {str(int(k)): int(v) for k, v in Counter(int(v) for v in dungeon_grid.ravel()).items()},
        "room_hashes": room_hashes,
        "layout": {
            "room_count": int(layout_payload.get("room_count", 0)),
            "primary_quality_metric_name": layout_payload.get("primary_quality_metric_name"),
            "primary_quality_metric_value": layout_payload.get("primary_quality_metric_value"),
            **dict(layout_payload.get("layout_quality", {})),
        },
        "validation": validation,
    }
    (variant_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (variant_dir / "validation_search_stats.json").write_text(
        json.dumps(build_validation_search_stats_payload(summary.get("validation", {})), indent=2),
        encoding="utf-8",
    )
    _release_torch_memory()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export end-to-end artifacts for semantic-anchor retrain checkpoints.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260404)
    parser.add_argument("--num-rooms", type=int, default=8)
    parser.add_argument("--topology-population", type=int, default=50)
    parser.add_argument("--topology-generations", type=int, default=100)
    add_generation_override_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generation_overrides = generation_overrides_from_namespace(args)
    _emit_progress(f"loading run directory: {args.run_dir}")
    diffusion_pipeline = build_pipeline(args.run_dir, generation_overrides=generation_overrides)
    _emit_progress(
        "generating shared topology "
        f"(seed={int(args.seed)}, rooms={int(args.num_rooms)}, population={int(args.topology_population)}, "
        f"generations={int(args.topology_generations)})"
    )
    topology_started = time.perf_counter()
    prepared = diffusion_pipeline.prepare_dungeon_generation(
        mission_graph=None,
        generate_topology=True,
        num_rooms=int(args.num_rooms),
        population_size=int(args.topology_population),
        generations=int(args.topology_generations),
        seed=int(args.seed),
    )
    _emit_progress(f"shared topology ready in {time.perf_counter() - topology_started:.1f}s")
    mission_graph = copy.deepcopy(prepared.mission_graph)
    del prepared
    del diffusion_pipeline
    _release_torch_memory()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "mission_graph.json").write_text(
        json.dumps(json_graph.node_link_data(mission_graph, edges="links"), indent=2),
        encoding="utf-8",
    )
    _emit_progress(f"wrote mission graph: {args.output_dir / 'mission_graph.json'}")

    summaries: Dict[str, Any] = {}

    _emit_progress("starting diffusion export (50-step teacher)")
    started = time.perf_counter()
    summaries["diffusion_cfg3_logic0_steps50"] = export_variant(
            run_dir=args.run_dir,
            mission_graph=mission_graph,
            variant_name="diffusion_cfg3_logic0_steps50",
            out_dir=args.output_dir,
            guidance_scale=3.0,
            logic_guidance_scale=0.0,
            num_diffusion_steps=50,
            use_fast_sampling=False,
            seed=int(args.seed),
            generation_overrides=generation_overrides,
        )
    _emit_progress(f"finished diffusion export in {time.perf_counter() - started:.1f}s")
    _release_torch_memory()

    _emit_progress("starting fast-sampler export (4-step)")
    started = time.perf_counter()
    summaries["fast_cfg3_logic0_steps4"] = export_variant(
            run_dir=args.run_dir,
            mission_graph=mission_graph,
            variant_name="fast_cfg3_logic0_steps4",
            out_dir=args.output_dir,
            guidance_scale=3.0,
            logic_guidance_scale=0.0,
            num_diffusion_steps=4,
            use_fast_sampling=True,
            seed=int(args.seed),
            generation_overrides=generation_overrides,
        )
    _emit_progress(f"finished fast-sampler export in {time.perf_counter() - started:.1f}s")
    _release_torch_memory()

    _emit_progress("starting masked-room export")
    started = time.perf_counter()
    summaries["masked_room_full"] = export_masked_variant(
            run_dir=args.run_dir,
            mission_graph=mission_graph,
            variant_name="masked_room_full",
            out_dir=args.output_dir,
            seed=int(args.seed),
            generation_overrides=generation_overrides,
        )
    _emit_progress(f"finished masked-room export in {time.perf_counter() - started:.1f}s")
    _release_torch_memory()
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "generation_overrides": generation_overrides,
                "variants": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _emit_progress(f"wrote summary: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
