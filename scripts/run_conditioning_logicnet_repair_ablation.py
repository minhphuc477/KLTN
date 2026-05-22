"""Paired conditioning, LogicNet, and repair ablation protocol.

Default mode is plan-only. Pass ``--execute`` when checkpoints and compute are
ready. The run matrix separates:

- full conditioning vs no graph tokens vs no stage tokens
- repair disabled vs enabled
- LogicNet guidance disabled vs enabled
- pre-repair vs post-repair semantics and validity
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_system import merge_config  # noqa: E402
from src.core.definitions import TileID  # noqa: E402
from src.evaluation.pcbs_validation import evaluate_astar_vs_pcbs  # noqa: E402
from src.pipeline.dungeon_pipeline import (  # noqa: E402
    DungeonGenerationResult,
    NeuralSymbolicDungeonPipeline,
    RoomGenerationResult,
    pipeline_kwargs_from_resolved_config,
    topology_generation_kwargs_from_resolved_config,
)
from src.zelda_data.vglc_utils import filter_virtual_nodes  # noqa: E402


@dataclass(frozen=True)
class VariantSpec:
    name: str
    conditioning: str
    repair_enabled: bool
    logic_enabled: bool
    notes: str


def build_experiment_matrix() -> List[VariantSpec]:
    variants: List[VariantSpec] = []
    conditioning_notes = {
        "full": "graph node tokens, TPE, and stage topology signal enabled",
        "no_graph_tokens": "graph node cross-attention and TPE disabled",
        "no_stage_tokens": "stage topology signal disabled while graph tokens remain enabled",
    }
    for conditioning in ("full", "no_graph_tokens", "no_stage_tokens"):
        for repair_enabled in (False, True):
            for logic_enabled in (False, True):
                variants.append(
                    VariantSpec(
                        name=(
                            f"{conditioning}__repair_{'on' if repair_enabled else 'off'}"
                            f"__logic_{'on' if logic_enabled else 'off'}"
                        ),
                        conditioning=conditioning,
                        repair_enabled=bool(repair_enabled),
                        logic_enabled=bool(logic_enabled),
                        notes=conditioning_notes[conditioning],
                    )
                )
    return variants


def _parse_seeds(raw: str) -> List[int]:
    seeds = [int(token.strip()) for token in str(raw).split(",") if token.strip()]
    return seeds or [42]


def _existing_or_none(path: Optional[str | Path]) -> Optional[str]:
    if not path:
        return None
    candidate = Path(path)
    return str(candidate) if candidate.exists() else None


def resolve_checkpoints(args: argparse.Namespace, resolved_config: Mapping[str, Any]) -> Dict[str, Optional[str]]:
    output_dir = Path(str(resolved_config["runtime"]["output_dir"]))
    vqvae = (
        args.vqvae_checkpoint
        or resolved_config.get("diffusion", {}).get("vqvae_checkpoint")
        or output_dir / "checkpoints" / "vqvae" / "vqvae_pretrained.pth"
    )
    diffusion = args.diffusion_checkpoint or output_dir / "checkpoints" / "diffusion" / "best_model.pth"
    logic = args.logic_net_checkpoint or diffusion
    return {
        "vqvae_checkpoint": _existing_or_none(vqvae),
        "diffusion_checkpoint": _existing_or_none(diffusion),
        "logic_net_checkpoint": _existing_or_none(logic),
    }


def pipeline_kwargs_for_variant(
    resolved_config: Mapping[str, Any],
    checkpoints: Mapping[str, Optional[str]],
    variant: VariantSpec,
) -> Dict[str, Any]:
    kwargs = pipeline_kwargs_from_resolved_config(dict(resolved_config))
    kwargs.update({key: value for key, value in checkpoints.items() if value is not None})

    if variant.conditioning == "no_graph_tokens":
        kwargs["use_graph_node_cross_attention"] = False
        kwargs["default_use_topological_positional_encoding"] = False
    else:
        kwargs["use_graph_node_cross_attention"] = True
        kwargs["default_use_topological_positional_encoding"] = True

    kwargs["default_puzzle_stage_topology_enabled"] = variant.conditioning != "no_stage_tokens"
    kwargs["default_apply_repair"] = bool(variant.repair_enabled)
    return kwargs


def _semantic_counts(grid: Any) -> Dict[str, int]:
    array = np.asarray(grid, dtype=np.int32)
    return {
        "start_count": int(np.sum(array == int(TileID.START))),
        "triforce_count": int(np.sum(array == int(TileID.TRIFORCE))),
        "key_count": int(np.sum(np.isin(array, [int(TileID.KEY_SMALL), int(TileID.KEY_BOSS), int(TileID.KEY_ITEM)]))),
        "lock_count": int(np.sum(np.isin(array, [int(TileID.DOOR_LOCKED), int(TileID.DOOR_BOSS)]))),
        "puzzle_count": int(np.sum(array == int(TileID.PUZZLE))),
        "enemy_count": int(np.sum(np.isin(array, [int(TileID.ENEMY), int(TileID.BOSS)]))),
    }


def _flatten_eval(prefix: str, result: Mapping[str, Any]) -> Dict[str, Any]:
    oracle = dict(result.get("oracle", {}) or {})
    pcbs = dict(result.get("pcbs", {}) or {})
    comparison = dict(result.get("comparison", {}) or {})
    return {
        f"{prefix}_oracle_solved": bool(oracle.get("success", False)),
        f"{prefix}_oracle_status": str(oracle.get("status", "")),
        f"{prefix}_oracle_path_length": int(oracle.get("path_length", 0) or 0),
        f"{prefix}_pcbs_solved": bool(pcbs.get("success", False)),
        f"{prefix}_pcbs_status": str(pcbs.get("status", "")),
        f"{prefix}_pcbs_path_length": int(pcbs.get("path_length", 0) or 0),
        f"{prefix}_pcbs_trajectory_length": int(pcbs.get("trajectory_length", 0) or 0),
        f"{prefix}_pcbs_states_explored": int(pcbs.get("states_explored", 0) or 0),
        f"{prefix}_bounded_rationality_index": float(comparison.get("bounded_rationality_index", 0.0) or 0.0),
        f"{prefix}_readability_score": float(comparison.get("readability_score", 0.0) or 0.0),
        f"{prefix}_cognitive_effort_index": float(comparison.get("cognitive_effort_index", 0.0) or 0.0),
        f"{prefix}_confusion_ratio_vs_oracle": comparison.get("confusion_ratio_vs_oracle"),
        f"{prefix}_oracle_pcbs_path_delta": comparison.get("oracle_pcbs_path_delta"),
    }


def _safe_evaluate(source: Any, *, persona: str, timeout_astar: int, timeout_pcbs: int, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    try:
        payload = evaluate_astar_vs_pcbs(
            source,
            persona=persona,
            timeout_astar=int(timeout_astar),
            timeout_pcbs=int(timeout_pcbs),
            seed=int(seed),
        )
        return payload, {}
    except Exception as exc:  # evaluation artifacts should record failures, not abort the run
        return {}, {"error_type": type(exc).__name__, "error": str(exc)}


def _pre_repair_result(
    pipeline: NeuralSymbolicDungeonPipeline,
    result: DungeonGenerationResult,
) -> Tuple[DungeonGenerationResult, np.ndarray]:
    physical_graph = filter_virtual_nodes(result.mission_graph)
    pre_rooms: Dict[Any, RoomGenerationResult] = {}
    for room_id, room in result.rooms.items():
        pre_grid = np.asarray(room.neural_grid if room.neural_grid is not None else room.room_grid, dtype=np.int32)
        pre_rooms[room_id] = replace(
            room,
            room_grid=pre_grid,
            was_repaired=False,
            repair_mask=None,
        )
    layout = pipeline.stitch_room_layout(pre_rooms, physical_graph)
    pre_grid = np.asarray(layout.dungeon_grid, dtype=np.int32)
    return replace(result, dungeon_grid=pre_grid, rooms=pre_rooms, stitched_layout=layout), pre_grid


def _row_for_result(
    *,
    variant: VariantSpec,
    seed: int,
    result: DungeonGenerationResult,
    pre_result: DungeonGenerationResult,
    pre_eval: Mapping[str, Any],
    post_eval: Mapping[str, Any],
    pre_eval_error: Mapping[str, Any],
    post_eval_error: Mapping[str, Any],
) -> Dict[str, Any]:
    pre_counts = _semantic_counts(pre_result.dungeon_grid)
    post_counts = _semantic_counts(result.dungeon_grid)
    row: Dict[str, Any] = {
        "variant": variant.name,
        "conditioning": variant.conditioning,
        "repair_enabled": bool(variant.repair_enabled),
        "logic_enabled": bool(variant.logic_enabled),
        "seed": int(seed),
        "notes": variant.notes,
        "generation_time_sec": float(result.generation_time),
        "repair_count": int(result.metrics.get("repair_count", 0) or 0),
        "repair_time_sec": float(result.metrics.get("repair_time_sec", 0.0) or 0.0),
        "total_tiles_repaired": int(result.metrics.get("total_tiles_repaired", 0) or 0),
        "repair_rate": float(result.metrics.get("repair_rate", 0.0) or 0.0),
        "logicnet_dungeon_solvability": float(result.metrics.get("logicnet_dungeon_solvability", 0.0) or 0.0),
        "logicnet_room_solvability": float(result.metrics.get("logicnet_room_solvability", 0.0) or 0.0),
        "pre_eval_error_type": str(pre_eval_error.get("error_type", "")),
        "post_eval_error_type": str(post_eval_error.get("error_type", "")),
    }
    for key, value in pre_counts.items():
        row[f"pre_{key}"] = int(value)
    for key, value in post_counts.items():
        row[f"post_{key}"] = int(value)
        row[f"delta_{key}"] = int(value) - int(pre_counts.get(key, 0))
    row.update(_flatten_eval("pre", pre_eval))
    row.update(_flatten_eval("post", post_eval))
    return row


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({str(key) for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _mean(values: Iterable[Any]) -> float:
    nums: List[float] = []
    for value in values:
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(val):
            nums.append(val)
    return float(np.mean(nums)) if nums else 0.0


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, bool, bool], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("conditioning")), bool(row.get("repair_enabled")), bool(row.get("logic_enabled")))
        groups.setdefault(key, []).append(row)

    summary: List[Dict[str, Any]] = []
    for (conditioning, repair_enabled, logic_enabled), group in sorted(groups.items()):
        summary.append(
            {
                "conditioning": conditioning,
                "repair_enabled": bool(repair_enabled),
                "logic_enabled": bool(logic_enabled),
                "n": int(len(group)),
                "pre_oracle_valid_rate": _mean(row.get("pre_oracle_solved", False) for row in group),
                "post_oracle_valid_rate": _mean(row.get("post_oracle_solved", False) for row in group),
                "pre_pcbs_valid_rate": _mean(row.get("pre_pcbs_solved", False) for row in group),
                "post_pcbs_valid_rate": _mean(row.get("post_pcbs_solved", False) for row in group),
                "repair_count_mean": _mean(row.get("repair_count", 0) for row in group),
                "repair_time_sec_mean": _mean(row.get("repair_time_sec", 0.0) for row in group),
                "total_tiles_repaired_mean": _mean(row.get("total_tiles_repaired", 0) for row in group),
                "generation_time_sec_mean": _mean(row.get("generation_time_sec", 0.0) for row in group),
                "post_readability_score_mean": _mean(row.get("post_readability_score", 0.0) for row in group),
                "post_bounded_rationality_index_mean": _mean(
                    row.get("post_bounded_rationality_index", 0.0) for row in group
                ),
                "logicnet_dungeon_solvability_mean": _mean(
                    row.get("logicnet_dungeon_solvability", 0.0) for row in group
                ),
            }
        )
    return summary


def _save_visuals(output_dir: Path, rows: Sequence[Mapping[str, Any]], grids: Sequence[Tuple[str, np.ndarray]], *, tile_px: int) -> None:
    if not grids:
        return
    from PIL import Image, ImageDraw, ImageFont
    from src.gui.rendering.level_image_export import save_level_grid_png

    image_dir = output_dir / "visual_sheet_tiles"
    image_dir.mkdir(parents=True, exist_ok=True)
    rendered: List[Tuple[str, Path]] = []
    for label, grid in grids:
        safe_label = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in label)[:120]
        path = image_dir / f"{safe_label}.png"
        save_level_grid_png(grid, path, tile_px=int(tile_px))
        rendered.append((label, path))

    images = [(label, Image.open(path).convert("RGB")) for label, path in rendered]
    max_w = max(image.width for _label, image in images)
    max_h = max(image.height for _label, image in images)
    label_h = 28
    cols = 2
    rows_count = int(np.ceil(len(images) / float(cols)))
    sheet = Image.new("RGB", (cols * max_w, rows_count * (max_h + label_h)), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 12)
    except (OSError, ValueError):
        font = ImageFont.load_default()
    for idx, (label, image) in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * max_w
        y = row * (max_h + label_h)
        draw.rectangle((x, y, x + max_w - 1, y + label_h - 1), fill=(230, 230, 230))
        draw.text((x + 4, y + 7), label[:90], fill=(20, 20, 20), font=font)
        sheet.paste(image, (x, y + label_h))
    sheet.save(output_dir / "visual_sheet.png")
    (output_dir / "visual_sheet_manifest.json").write_text(
        json.dumps(
            {"tiles": [{"label": label, "path": str(path)} for label, path in rendered], "rows": list(rows)},
            indent=2,
        ),
        encoding="utf-8",
    )


def write_plan(
    output_dir: Path,
    variants: Sequence[VariantSpec],
    args: argparse.Namespace,
    checkpoints: Mapping[str, Optional[str]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "conditioning_logicnet_repair_ablation",
        "mode": "execute" if bool(args.execute) else "plan_only",
        "seeds": _parse_seeds(args.seeds),
        "variants": [asdict(variant) for variant in variants],
        "checkpoints": dict(checkpoints),
        "required_outputs": [
            "conditioning_logicnet_repair_rows.csv",
            "conditioning_logicnet_repair_summary.csv",
            "conditioning_logicnet_repair_payload.json",
            "visual_sheet.png",
            "visual_sheet_manifest.json",
        ],
        "semantics_contract": "Report pre-repair and post-repair A*/P-CBS validity, semantic counts, repair count, repair time, and LogicNet metrics separately.",
    }
    (output_dir / "conditioning_logicnet_repair_plan.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Conditioning LogicNet Repair Ablation Plan",
        "",
        "This file is generated by `scripts/run_conditioning_logicnet_repair_ablation.py`.",
        "",
        f"- mode: `{payload['mode']}`",
        f"- seeds: `{', '.join(str(seed) for seed in payload['seeds'])}`",
        f"- variants: `{len(variants)}`",
        "",
        "| Variant | Conditioning | Repair | LogicNet | Notes |",
        "|---|---|---:|---:|---|",
    ]
    for variant in variants:
        lines.append(
            f"| `{variant.name}` | `{variant.conditioning}` | {int(variant.repair_enabled)} | "
            f"{int(variant.logic_enabled)} | {variant.notes} |"
        )
    lines.extend(
        [
            "",
            "## Execute Later",
            "",
            "```powershell",
            "python scripts\\run_conditioning_logicnet_repair_ablation.py --execute --output results\\conditioning_logicnet_repair_ablation",
            "```",
        ]
    )
    (output_dir / "conditioning_logicnet_repair_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute_protocol(args: argparse.Namespace, variants: Sequence[VariantSpec]) -> Tuple[List[Dict[str, Any]], List[Tuple[str, np.ndarray]]]:
    resolved = merge_config(yaml_path=str(args.config))
    checkpoints = resolve_checkpoints(args, resolved)
    topology_kwargs = topology_generation_kwargs_from_resolved_config(resolved)
    rows: List[Dict[str, Any]] = []
    visual_grids: List[Tuple[str, np.ndarray]] = []
    for seed in _parse_seeds(args.seeds):
        for variant in variants:
            kwargs = pipeline_kwargs_for_variant(resolved, checkpoints, variant)
            pipeline = NeuralSymbolicDungeonPipeline(device=str(args.device), **kwargs)
            started = time.time()
            result = pipeline.generate_dungeon(
                generate_topology=True,
                seed=int(seed),
                num_rooms=int(args.num_rooms or topology_kwargs["num_rooms"]),
                population_size=int(args.population_size or topology_kwargs["population_size"]),
                generations=int(args.generations or topology_kwargs["generations"]),
                num_diffusion_steps=int(args.num_diffusion_steps),
                logic_guidance_scale=float(args.logic_guidance_scale if variant.logic_enabled else 0.0),
                apply_repair=bool(variant.repair_enabled),
                use_topological_positional_encoding=(variant.conditioning != "no_graph_tokens"),
                enable_map_elites=False,
                batch_independent_rooms=not bool(args.disable_batch_independent_rooms),
                max_batch_size=int(args.max_batch_size),
            )
            result.metrics["ablation_wall_time_sec"] = float(time.time() - started)
            pre_result, pre_grid = _pre_repair_result(pipeline, result)
            pre_eval, pre_error = _safe_evaluate(
                pre_result,
                persona=str(args.persona),
                timeout_astar=int(args.timeout_astar),
                timeout_pcbs=int(args.timeout_pcbs),
                seed=int(seed),
            )
            post_eval, post_error = _safe_evaluate(
                result,
                persona=str(args.persona),
                timeout_astar=int(args.timeout_astar),
                timeout_pcbs=int(args.timeout_pcbs),
                seed=int(seed),
            )
            row = _row_for_result(
                variant=variant,
                seed=int(seed),
                result=result,
                pre_result=pre_result,
                pre_eval=pre_eval,
                post_eval=post_eval,
                pre_eval_error=pre_error,
                post_eval_error=post_error,
            )
            rows.append(row)
            if bool(args.write_visual_sheet):
                visual_grids.append((f"{variant.name} seed={seed} pre", pre_grid))
                visual_grids.append((f"{variant.name} seed={seed} post", np.asarray(result.dungeon_grid, dtype=np.int32)))
            _write_csv(args.output / "conditioning_logicnet_repair_rows.partial.csv", rows)
    return rows, visual_grids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or run paired conditioning/LogicNet/repair ablations.")
    parser.add_argument("--execute", action="store_true", help="Run generation. Omit to write a plan only.")
    parser.add_argument("--config", type=Path, default=Path("configs") / "zelda_hmolqd.yaml")
    parser.add_argument("--output", type=Path, default=Path("results") / "conditioning_logicnet_repair_ablation")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--vqvae-checkpoint", type=str, default=None)
    parser.add_argument("--diffusion-checkpoint", type=str, default=None)
    parser.add_argument("--logic-net-checkpoint", type=str, default=None)
    parser.add_argument("--num-rooms", type=int, default=None)
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--num-diffusion-steps", type=int, default=50)
    parser.add_argument("--logic-guidance-scale", type=float, default=1.0)
    parser.add_argument("--persona", type=str, default="novice")
    parser.add_argument("--timeout-astar", type=int, default=200000)
    parser.add_argument("--timeout-pcbs", type=int, default=50000)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--disable-batch-independent-rooms", action="store_true")
    parser.add_argument("--write-visual-sheet", action="store_true", default=True)
    parser.add_argument("--tile-px", type=int, default=8)
    parser.add_argument("--quick", action="store_true", help="Plan or execute a one-seed smoke subset.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    variants = build_experiment_matrix()
    if args.quick:
        args.seeds = str(_parse_seeds(args.seeds)[0])
        variants = variants[:4]
        args.num_rooms = args.num_rooms or 6
        args.population_size = args.population_size or 8
        args.generations = args.generations or 3
        args.num_diffusion_steps = min(int(args.num_diffusion_steps), 8)
        args.timeout_astar = min(int(args.timeout_astar), 2000)
        args.timeout_pcbs = min(int(args.timeout_pcbs), 1000)

    resolved = merge_config(yaml_path=str(args.config))
    checkpoints = resolve_checkpoints(args, resolved)
    write_plan(args.output, variants, args, checkpoints)
    if not args.execute:
        print(f"Wrote conditioning/LogicNet/repair ablation plan to {args.output}")
        return 0

    rows, visual_grids = execute_protocol(args, variants)
    summary = summarize_rows(rows)
    _write_csv(args.output / "conditioning_logicnet_repair_rows.csv", rows)
    _write_csv(args.output / "conditioning_logicnet_repair_summary.csv", summary)
    payload = {
        "rows": rows,
        "summary": summary,
        "variants": [asdict(variant) for variant in variants],
        "checkpoints": checkpoints,
    }
    (args.output / "conditioning_logicnet_repair_payload.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if bool(args.write_visual_sheet):
        _save_visuals(args.output, rows, visual_grids, tile_px=int(args.tile_px))
    print(f"Wrote conditioning/LogicNet/repair ablation outputs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
