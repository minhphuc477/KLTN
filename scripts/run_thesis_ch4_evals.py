"""Run report-facing fixed-graph evaluations for completed thesis model runs.

This script reuses the existing export helpers to:
1. generate representative dungeon artifacts on one fixed mission graph;
2. cache per-variant summaries so reruns are cheap;
3. collect the main Chapter 4 metrics into CSV / JSON / Markdown tables.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_manual_rich_topology_compare import build_manual_rich_topology_graph
from scripts.export_semantic_anchor_end_to_end import export_masked_variant
from scripts.run_fast_sampler_visual_audit import export_variant


@dataclass(frozen=True)
class VariantSpec:
    name: str
    kind: str
    guidance_scale: float = 3.0
    logic_guidance_scale: float = 0.0
    num_diffusion_steps: int = 50
    use_fast_sampling: bool = False


@dataclass(frozen=True)
class RunSpec:
    label: str
    run_dir: Path
    variants: tuple[VariantSpec, ...]


DIFFUSION = VariantSpec(
    name="diffusion_cfg3_logic0_steps50",
    kind="diffusion",
    guidance_scale=3.0,
    logic_guidance_scale=0.0,
    num_diffusion_steps=50,
    use_fast_sampling=False,
)

FAST = VariantSpec(
    name="fast_cfg3_logic0_steps4",
    kind="diffusion",
    guidance_scale=3.0,
    logic_guidance_scale=0.0,
    num_diffusion_steps=4,
    use_fast_sampling=True,
)

MASKED = VariantSpec(
    name="masked_room_full",
    kind="masked",
)


GROUPS: Dict[str, tuple[RunSpec, ...]] = {
    "pdrop_sweep": (
        RunSpec("pdrop015", ROOT / "outputs" / "zelda_hmolqd_puzzlecookbook_baseline_pdrop015_v1", (DIFFUSION,)),
        RunSpec("pdrop035", ROOT / "outputs" / "zelda_hmolqd_puzzlecookbook_pdrop035_v1", (DIFFUSION,)),
        RunSpec("pdrop055", ROOT / "outputs" / "zelda_hmolqd_puzzlecookbook_pdrop055_v1", (DIFFUSION,)),
    ),
    "tokenizer_compare": (
        RunSpec(
            "codebook256",
            ROOT / "outputs" / "zelda_hmolqd_downstream_baseline_puzzle_subtype_v2_rerun_heldout_20260419_182314",
            (DIFFUSION, MASKED),
        ),
        RunSpec(
            "codebook512",
            ROOT / "outputs" / "zelda_hmolqd_downstream_codebook512_puzzle_subtype_v2_rerun_heldout_20260419_182314",
            (DIFFUSION, MASKED),
        ),
    ),
    "stage_semantics_compare": (
        RunSpec(
            "baseline_branch",
            ROOT / "outputs" / "zelda_hmolqd_downstream_baseline_puzzle_subtype_v2_rerun_heldout_20260419_182314",
            (DIFFUSION, MASKED),
        ),
        RunSpec(
            "stageconditioned_semantics",
            ROOT / "outputs" / "zelda_hmolqd_downstream_stageconditioned_semantics_v2",
            (DIFFUSION, MASKED),
        ),
    ),
    "structure_control_compare": (
        RunSpec(
            "structure_baseline",
            ROOT / "outputs" / "zelda_hmolqd_downstream_baseline_puzzle_structure_control_v2",
            (DIFFUSION,),
        ),
        RunSpec(
            "structure_masked_branch",
            ROOT / "outputs" / "zelda_hmolqd_downstream_puzzle_structure_control_v2",
            (DIFFUSION, MASKED),
        ),
    ),
    "branch_compare_pdrop035": (
        RunSpec(
            "pdrop035",
            ROOT / "outputs" / "zelda_hmolqd_puzzlecookbook_pdrop035_v1",
            (DIFFUSION, FAST, MASKED),
        ),
    ),
}


def _safe_optional_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(v) for v in value]
    return value


def _summary_path(run_output_dir: Path, variant_name: str) -> Path:
    return run_output_dir / variant_name / "summary.json"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_variant(
    *,
    run_spec: RunSpec,
    variant: VariantSpec,
    mission_graph,
    run_output_dir: Path,
    seed: int,
    reuse_existing: bool,
) -> Dict[str, Any]:
    summary_path = _summary_path(run_output_dir, variant.name)
    if reuse_existing and summary_path.exists():
        return _load_json(summary_path)

    run_output_dir.mkdir(parents=True, exist_ok=True)
    if variant.kind == "masked":
        return export_masked_variant(
            run_dir=run_spec.run_dir,
            mission_graph=copy.deepcopy(mission_graph),
            variant_name=variant.name,
            out_dir=run_output_dir,
            seed=int(seed),
            generation_overrides={},
        )

    return export_variant(
        run_dir=run_spec.run_dir,
        mission_graph=copy.deepcopy(mission_graph),
        variant_name=variant.name,
        out_dir=run_output_dir,
        guidance_scale=float(variant.guidance_scale),
        logic_guidance_scale=float(variant.logic_guidance_scale),
        num_diffusion_steps=int(variant.num_diffusion_steps),
        use_fast_sampling=bool(variant.use_fast_sampling),
        seed=int(seed),
        generation_overrides={},
    )


def _row_from_summary(*, group_name: str, run_spec: RunSpec, variant: VariantSpec, run_output_dir: Path, summary: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = dict(summary.get("metrics", {}) or {})
    validation = dict(summary.get("validation", {}) or {})
    end_to_end = dict(summary.get("end_to_end_evaluation", {}) or {})
    astar = dict(validation.get("astar_grid", {}) or {})
    softlock = dict(validation.get("softlock_check", {}) or {})
    graph_guided = dict(validation.get("graph_guided_oracle", {}) or {})
    mechanical = dict(validation.get("mechanical_contract", {}) or {})
    cbs = dict(validation.get("cbs_balanced", {}) or {})

    stylized_path = run_output_dir / variant.name / "dungeon_grid_stylized.png"
    rooms_sheet_path = run_output_dir / variant.name / "rooms_sheet_stylized.png"

    return {
        "group": group_name,
        "label": run_spec.label,
        "run_dir": str(run_spec.run_dir),
        "variant": variant.name,
        "generator_kind": variant.kind,
        "generation_time_sec": _safe_optional_float(metrics.get("generation_time_sec")),
        "repair_rate": _safe_optional_float(metrics.get("repair_rate")),
        "tiles_repaired": _safe_optional_float(metrics.get("total_tiles_repaired")),
        "overwrite_rate": _safe_optional_float(metrics.get("avg_final_graph_marker_overwrite_rate")),
        "anchor_error_post": _safe_optional_float(metrics.get("avg_final_post_overlay_semantic_anchor_error")),
        "anchor_error_pre": _safe_optional_float(metrics.get("avg_final_pre_overlay_semantic_anchor_error")),
        "astar_solvable": bool(astar.get("solvable", False)),
        "softlock_safe": bool(softlock.get("is_safe", False)),
        "graph_guided_oracle_solvable": bool(graph_guided.get("solvable", False)),
        "hybrid_oracle_pass": bool(mechanical.get("hybrid_oracle_pass", False)),
        "cbs_success": bool(cbs.get("success", False)),
        "cbs_confusion_ratio_vs_astar": _safe_optional_float(cbs.get("confusion_ratio_vs_astar")),
        "cbs_confusion_index": _safe_optional_float(cbs.get("confusion_index")),
        "room_unique_ratio": _safe_optional_float(end_to_end.get("room_unique_ratio")),
        "room_pairwise_ncd_mean": _safe_optional_float(dict(end_to_end.get("room_pairwise_ncd", {}) or {}).get("mean")),
        "room_nearest_reference_ncd_mean": _safe_optional_float(
            dict(end_to_end.get("room_nearest_reference_ncd", {}) or {}).get("mean")
        ),
        "dungeon_symbol_entropy_non_void": _safe_optional_float(end_to_end.get("dungeon_symbol_entropy_non_void")),
        "image_stylized": str(stylized_path),
        "image_rooms_sheet": str(rooms_sheet_path),
        "summary_json": str(_summary_path(run_output_dir, variant.name)),
    }


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows_list = [dict(row) for row in rows]
    fieldnames: List[str] = []
    for row in rows_list:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_list:
            writer.writerow(row)


def _markdown_table(rows: List[Mapping[str, Any]]) -> str:
    headers = [
        "group",
        "label",
        "variant",
        "generation_time_sec",
        "repair_rate",
        "overwrite_rate",
        "hybrid_oracle_pass",
        "cbs_success",
        "room_unique_ratio",
        "room_nearest_reference_ncd_mean",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values: List[str] = []
        for header in headers:
            value = row.get(header)
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-graph Chapter 4 evaluations for completed model runs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "thesis_ch4_evals",
    )
    parser.add_argument(
        "--groups",
        type=str,
        default="pdrop_sweep,tokenizer_compare,stage_semantics_compare,structure_control_compare,branch_compare_pdrop035",
        help="Comma-separated group names.",
    )
    parser.add_argument("--seed", type=int, default=20260404)
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    requested_groups = [part.strip() for part in str(args.groups).split(",") if part.strip()]
    unknown = [name for name in requested_groups if name not in GROUPS]
    if unknown:
        raise ValueError(f"Unknown groups: {unknown}. Valid groups: {sorted(GROUPS)}")

    mission_graph = build_manual_rich_topology_graph()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    payload: Dict[str, Any] = {
        "seed": int(args.seed),
        "groups": {},
    }

    for group_name in requested_groups:
        group_dir = args.output_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        payload["groups"][group_name] = {}
        for run_spec in GROUPS[group_name]:
            run_output_dir = group_dir / run_spec.label
            payload["groups"][group_name][run_spec.label] = {}
            for variant in run_spec.variants:
                print(
                    f"[thesis-ch4] group={group_name} label={run_spec.label} variant={variant.name}",
                    flush=True,
                )
                summary = _run_variant(
                    run_spec=run_spec,
                    variant=variant,
                    mission_graph=mission_graph,
                    run_output_dir=run_output_dir,
                    seed=int(args.seed),
                    reuse_existing=bool(args.reuse_existing),
                )
                payload["groups"][group_name][run_spec.label][variant.name] = summary
                rows.append(
                    _row_from_summary(
                        group_name=group_name,
                        run_spec=run_spec,
                        variant=variant,
                        run_output_dir=run_output_dir,
                        summary=summary,
                    )
                )

    csv_path = args.output_dir / "chapter4_eval_rows.csv"
    json_path = args.output_dir / "chapter4_eval_payload.json"
    md_path = args.output_dir / "chapter4_eval_rows.md"
    _write_csv(csv_path, rows)
    json_path.write_text(json.dumps(_json_sanitize(payload), indent=2), encoding="utf-8")
    md_path.write_text(_markdown_table(rows), encoding="utf-8")

    print(
        json.dumps(
            {
                "csv": str(csv_path),
                "json": str(json_path),
                "markdown": str(md_path),
                "rows": len(rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
