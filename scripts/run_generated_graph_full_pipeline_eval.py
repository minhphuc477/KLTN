#!/usr/bin/env python3
"""Run end-to-end room generation and validation on generated mission graphs.

This script fills the gap between the graph-only OOD benchmark and the fixed-graph
visual audit: it first generates real mission graphs, then sends the same graph
through one or more room-generation variants, and finally records the downstream
oracle / P-CBS validation already produced by the export pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from networkx.readwrite import json_graph

from export_semantic_anchor_end_to_end import export_masked_variant
from run_fast_sampler_visual_audit import _json_sanitize, export_variant
from run_ood_scaling_and_blinded_eval import _room_budgets
from src.evaluation.benchmark_suite import (
    extract_graph_descriptor,
    generate_block_i_graphs,
    load_vglc_reference_graphs,
)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    kind: str
    guidance_scale: float = 3.0
    logic_guidance_scale: float = 0.0
    num_diffusion_steps: int = 50
    use_fast_sampling: bool = False


VARIANT_PRESETS: Dict[str, VariantSpec] = {
    "diffusion": VariantSpec(
        name="diffusion_cfg3_logic0_steps50",
        kind="diffusion",
        guidance_scale=3.0,
        logic_guidance_scale=0.0,
        num_diffusion_steps=50,
        use_fast_sampling=False,
    ),
    "fast": VariantSpec(
        name="fast_cfg3_logic0_steps4",
        kind="fast_sampler",
        guidance_scale=3.0,
        logic_guidance_scale=0.0,
        num_diffusion_steps=4,
        use_fast_sampling=True,
    ),
    "masked": VariantSpec(
        name="masked_room_full",
        kind="masked_room",
    ),
    "diffusion_no_logic": VariantSpec(
        name="diffusion_cfg3_logic0_steps50",
        kind="diffusion_no_logic",
        guidance_scale=3.0,
        logic_guidance_scale=0.0,
        num_diffusion_steps=50,
        use_fast_sampling=False,
    ),
    "diffusion_logic1": VariantSpec(
        name="diffusion_cfg3_logic1_steps50",
        kind="diffusion_logic_guided",
        guidance_scale=3.0,
        logic_guidance_scale=1.0,
        num_diffusion_steps=50,
        use_fast_sampling=False,
    ),
    "diffusion_logic05": VariantSpec(
        name="diffusion_cfg3_logic05_steps50",
        kind="diffusion_logic_guided",
        guidance_scale=3.0,
        logic_guidance_scale=0.5,
        num_diffusion_steps=50,
        use_fast_sampling=False,
    ),
}


def _parse_csv_tokens(raw: str) -> List[str]:
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def _parse_seeds(raw: str) -> List[int]:
    seeds: List[int] = []
    for token in _parse_csv_tokens(raw):
        if ":" in token:
            parts = [int(v.strip()) for v in token.split(":") if v.strip()]
            if len(parts) not in {2, 3}:
                raise ValueError(f"Invalid seed range '{token}'. Use start:end[:step].")
            start, end = parts[0], parts[1]
            step = parts[2] if len(parts) == 3 else 1
            if step == 0:
                raise ValueError("Seed range step cannot be zero.")
            stop = end + (1 if step > 0 else -1)
            seeds.extend(range(start, stop, step))
        else:
            seeds.append(int(token))
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _mean(values: Iterable[Any]) -> Optional[float]:
    cleaned = [_safe_float(v) for v in values]
    cleaned = [v for v in cleaned if v is not None]
    if not cleaned:
        return None
    return float(statistics.fmean(cleaned))


def _rate(values: Iterable[Any]) -> Optional[float]:
    vals = list(values)
    if not vals:
        return None
    return float(sum(1 for v in vals if bool(v)) / len(vals))


def _variant_specs(raw: str) -> List[VariantSpec]:
    specs: List[VariantSpec] = []
    for token in _parse_csv_tokens(raw):
        key = token.lower()
        if key not in VARIANT_PRESETS:
            raise ValueError(f"Unknown variant '{token}'. Valid: {sorted(VARIANT_PRESETS)}")
        specs.append(VARIANT_PRESETS[key])
    if not specs:
        raise ValueError("At least one variant is required.")
    return specs


def _descriptor_targets(refs: Sequence[nx.Graph]) -> Dict[str, float]:
    if not refs:
        return {}
    descriptors = [extract_graph_descriptor(g) for g in refs]
    keys = sorted({k for desc in descriptors for k in desc.keys()})
    targets: Dict[str, float] = {}
    for key in keys:
        values = [_safe_float(desc.get(key)) for desc in descriptors]
        values = [v for v in values if v is not None]
        if values:
            targets[key] = float(statistics.fmean(values))
    return targets


def _row_from_summary(
    *,
    regime: str,
    graph_index: int,
    seed: int,
    run_dir: Path,
    variant: VariantSpec,
    graph_path: Path,
    output_dir: Path,
    graph_generation_time_sec: float,
    summary: Mapping[str, Any],
) -> Dict[str, Any]:
    metrics = dict(summary.get("metrics", {}) or {})
    validation = dict(summary.get("validation", {}) or {})
    end_to_end = dict(summary.get("end_to_end_evaluation", {}) or {})
    astar = dict(validation.get("astar_grid", {}) or {})
    softlock = dict(validation.get("softlock_check", {}) or {})
    graph_guided = dict(validation.get("graph_guided_oracle", {}) or {})
    mechanical = dict(validation.get("mechanical_contract", {}) or {})
    cbs = dict(validation.get("cbs_balanced", {}) or {})
    room_ncd = dict(end_to_end.get("room_pairwise_ncd", {}) or {})
    ref_ncd = dict(end_to_end.get("room_nearest_reference_ncd", {}) or {})

    variant_dir = output_dir / variant.name
    return {
        "regime": regime,
        "graph_index": int(graph_index),
        "seed": int(seed),
        "run_dir": str(run_dir),
        "variant": variant.name,
        "generator_kind": variant.kind,
        "mission_graph": str(graph_path),
        "graph_nodes": int(summary.get("layout", {}).get("room_count", 0) or 0),
        "graph_generation_time_sec": float(graph_generation_time_sec),
        "generation_time_sec": _safe_float(metrics.get("generation_time_sec")),
        "repair_rate": _safe_float(metrics.get("repair_rate")),
        "tiles_repaired": _safe_float(metrics.get("total_tiles_repaired")),
        "overwrite_rate": _safe_float(metrics.get("avg_final_graph_marker_overwrite_rate")),
        "anchor_error_post": _safe_float(metrics.get("avg_final_post_overlay_semantic_anchor_error")),
        "anchor_error_pre": _safe_float(metrics.get("avg_final_pre_overlay_semantic_anchor_error")),
        "astar_solvable": bool(astar.get("solvable", False)),
        "softlock_safe": bool(softlock.get("is_safe", False)),
        "graph_guided_oracle_solvable": bool(graph_guided.get("solvable", False)),
        "hybrid_oracle_pass": bool(mechanical.get("hybrid_oracle_pass", False)),
        "cbs_success": bool(cbs.get("success", False)),
        "cbs_path_length": _safe_float(cbs.get("path_length")),
        "cbs_confusion_ratio_vs_astar": _safe_float(cbs.get("confusion_ratio_vs_astar")),
        "cbs_confusion_index": _safe_float(cbs.get("confusion_index")),
        "room_unique_ratio": _safe_float(end_to_end.get("room_unique_ratio")),
        "room_pairwise_ncd_mean": _safe_float(room_ncd.get("mean")),
        "room_nearest_reference_ncd_mean": _safe_float(ref_ncd.get("mean")),
        "dungeon_symbol_entropy_non_void": _safe_float(end_to_end.get("dungeon_symbol_entropy_non_void")),
        "image_stylized": str(variant_dir / "dungeon_grid_stylized.png"),
        "image_rooms_sheet": str(variant_dir / "rooms_sheet_stylized.png"),
        "summary_json": str(variant_dir / "summary.json"),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("regime")), str(row.get("variant")), str(row.get("generator_kind")))
        grouped.setdefault(key, []).append(row)

    out: List[Dict[str, Any]] = []
    for (regime, variant, kind), group in sorted(grouped.items()):
        out.append(
            {
                "regime": regime,
                "variant": variant,
                "generator_kind": kind,
                "n": len(group),
                "generation_time_sec_mean": _mean(row.get("generation_time_sec") for row in group),
                "repair_rate_mean": _mean(row.get("repair_rate") for row in group),
                "tiles_repaired_mean": _mean(row.get("tiles_repaired") for row in group),
                "hybrid_oracle_pass_rate": _rate(row.get("hybrid_oracle_pass") for row in group),
                "cbs_success_rate": _rate(row.get("cbs_success") for row in group),
                "room_unique_ratio_mean": _mean(row.get("room_unique_ratio") for row in group),
                "room_nearest_reference_ncd_mean": _mean(row.get("room_nearest_reference_ncd_mean") for row in group),
                "dungeon_symbol_entropy_non_void_mean": _mean(
                    row.get("dungeon_symbol_entropy_non_void") for row in group
                ),
                "cbs_confusion_ratio_vs_astar_mean": _mean(
                    row.get("cbs_confusion_ratio_vs_astar") for row in group
                ),
            }
        )
    return out


def _paired_logicnet(rows: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    by_graph: Dict[Tuple[str, int, int], Dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        kind = str(row.get("generator_kind", ""))
        if kind not in {"diffusion_no_logic", "diffusion_logic_guided"}:
            continue
        key = (str(row.get("regime")), int(row.get("graph_index", -1)), int(row.get("seed", -1)))
        by_graph.setdefault(key, {})[kind] = row

    pairs = [pair for pair in by_graph.values() if {"diffusion_no_logic", "diffusion_logic_guided"} <= set(pair)]
    if not pairs:
        return None

    def deltas(metric: str) -> List[float]:
        vals: List[float] = []
        for pair in pairs:
            a = _safe_float(pair["diffusion_logic_guided"].get(metric))
            b = _safe_float(pair["diffusion_no_logic"].get(metric))
            if a is not None and b is not None:
                vals.append(float(a - b))
        return vals

    def bool_deltas(metric: str) -> List[float]:
        return [
            float(bool(pair["diffusion_logic_guided"].get(metric)))
            - float(bool(pair["diffusion_no_logic"].get(metric)))
            for pair in pairs
        ]

    rng = np.random.default_rng(20260514)

    def bootstrap_ci(values: Sequence[float], draws: int = 5000) -> Optional[List[float]]:
        if not values:
            return None
        arr = np.asarray(values, dtype=np.float64)
        samples = rng.choice(arr, size=(draws, arr.size), replace=True).mean(axis=1)
        return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]

    def sign_permutation_p(values: Sequence[float], draws: int = 20000) -> Optional[float]:
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.abs(arr) > 1e-12]
        if arr.size == 0:
            return None
        observed = abs(float(arr.mean()))
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(draws, arr.size), replace=True)
        means = np.abs((signs * arr).mean(axis=1))
        return float((np.count_nonzero(means >= observed) + 1) / (draws + 1))

    metrics = {
        "hybrid_oracle_pass": bool_deltas("hybrid_oracle_pass"),
        "cbs_success": bool_deltas("cbs_success"),
        "generation_time_sec": deltas("generation_time_sec"),
        "repair_rate": deltas("repair_rate"),
        "room_nearest_reference_ncd_mean": deltas("room_nearest_reference_ncd_mean"),
    }
    return {
        "paired_n": len(pairs),
        "deltas_logic_minus_no_logic": {
            metric: {
                "mean": float(statistics.fmean(vals)) if vals else None,
                "bootstrap_ci95": bootstrap_ci(vals),
                "sign_permutation_p_two_sided": sign_permutation_p(vals),
            }
            for metric, vals in metrics.items()
        },
    }


def _markdown_report(rows: Sequence[Mapping[str, Any]], aggregate: Sequence[Mapping[str, Any]]) -> str:
    headers = [
        "regime",
        "variant",
        "n",
        "hybrid_oracle_pass_rate",
        "cbs_success_rate",
        "repair_rate_mean",
        "room_nearest_reference_ncd_mean",
    ]
    lines = [
        "# Generated-Graph Full-Pipeline Evaluation",
        "",
        f"- Rows: {len(rows)}",
        f"- Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in aggregate:
        values: List[str] = []
        for key in headers:
            value = row.get(key)
            if isinstance(value, float):
                values.append(f"{value:.4g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _regimes(args: argparse.Namespace, refs: Sequence[nx.Graph]) -> Dict[str, Tuple[int, int]]:
    if args.include_ood:
        budgets = _room_budgets(refs, room_budget_cap=int(args.room_budget_cap))
        requested = _parse_csv_tokens(args.regimes)
        return {name: budgets[name] for name in requested if name in budgets}
    return {args.regime_name: (int(args.min_rooms), int(args.max_rooms))}


def _generate_graph(
    *,
    seed: int,
    min_rooms: int,
    max_rooms: int,
    population_size: int,
    generations: int,
    descriptor_targets: Mapping[str, float],
    search_strategy: str,
    qd_archive_cells: int,
) -> Tuple[nx.Graph, float]:
    graphs, times = generate_block_i_graphs(
        num_samples=1,
        seed=int(seed),
        min_rooms=int(min_rooms),
        max_rooms=int(max_rooms),
        population_size=int(population_size),
        generations=int(generations),
        rule_space="full",
        descriptor_targets=dict(descriptor_targets),
        room_count_bias=0.45,
        search_strategy=str(search_strategy),
        qd_archive_cells=int(qd_archive_cells),
    )
    if not graphs:
        raise RuntimeError("Graph generation returned no graph.")
    return graphs[0], float(times[0] if times else 0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate mission graphs, run room generation variants, and collect oracle/P-CBS validation."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", type=str, default="diffusion,fast,masked")
    parser.add_argument("--seeds", type=str, default="20260514:20260518")
    parser.add_argument("--min-rooms", type=int, default=12)
    parser.add_argument("--max-rooms", type=int, default=12)
    parser.add_argument("--regime-name", type=str, default="generated_12")
    parser.add_argument("--include-ood", action="store_true")
    parser.add_argument("--regimes", type=str, default="in_dist,ood_small,ood_large")
    parser.add_argument("--room-budget-cap", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=Path("data/zelda/processed"))
    parser.add_argument("--reference-limit", type=int, default=256)
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--generations", type=int, default=24)
    parser.add_argument("--search-strategy", choices=["ga", "cvt"], default="ga")
    parser.add_argument("--qd-archive-cells", type=int, default=128)
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seeds = _parse_seeds(args.seeds)
    variants = _variant_specs(args.variants)
    refs = load_vglc_reference_graphs(args.data_root, limit=int(args.reference_limit))
    targets = _descriptor_targets(refs)
    regimes = _regimes(args, refs)
    if not regimes:
        raise ValueError("No regimes selected.")

    rows: List[Dict[str, Any]] = []
    manifest = {
        "run_dir": str(args.run_dir),
        "variants": [spec.__dict__ for spec in variants],
        "seeds": seeds,
        "regimes": regimes,
        "population_size": int(args.population_size),
        "generations": int(args.generations),
        "search_strategy": str(args.search_strategy),
        "reference_count": len(refs),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(_json_sanitize(manifest), indent=2),
        encoding="utf-8",
    )

    graph_index = 0
    for regime_name, (min_rooms, max_rooms) in regimes.items():
        for seed in seeds:
            graph_index += 1
            graph_dir = args.output_dir / regime_name / f"seed_{seed}"
            graph_dir.mkdir(parents=True, exist_ok=True)
            graph_path = graph_dir / "mission_graph.json"
            timing_path = graph_dir / "graph_generation_timing.json"
            if args.reuse_existing and graph_path.exists():
                graph = json_graph.node_link_graph(json.loads(graph_path.read_text(encoding="utf-8")), edges="links")
                timing = json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.exists() else {}
                graph_generation_time = float(timing.get("graph_generation_time_sec", 0.0))
            else:
                print(
                    f"[graph] regime={regime_name} seed={seed} rooms={min_rooms}-{max_rooms} "
                    f"pop={args.population_size} gen={args.generations}",
                    flush=True,
                )
                graph, graph_generation_time = _generate_graph(
                    seed=int(seed),
                    min_rooms=int(min_rooms),
                    max_rooms=int(max_rooms),
                    population_size=int(args.population_size),
                    generations=int(args.generations),
                    descriptor_targets=targets,
                    search_strategy=str(args.search_strategy),
                    qd_archive_cells=int(args.qd_archive_cells),
                )
                graph_path.write_text(
                    json.dumps(_json_sanitize(json_graph.node_link_data(graph, edges="links")), indent=2),
                    encoding="utf-8",
                )
                timing_path.write_text(
                    json.dumps({"graph_generation_time_sec": graph_generation_time}, indent=2),
                    encoding="utf-8",
                )

            for variant in variants:
                variant_dir = graph_dir / variant.name
                summary_path = variant_dir / "summary.json"
                if args.reuse_existing and summary_path.exists():
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    print(f"[reuse] {regime_name}/seed_{seed}/{variant.name}", flush=True)
                else:
                    print(f"[export] {regime_name}/seed_{seed}/{variant.name}", flush=True)
                    if variant.kind == "masked_room":
                        summary = export_masked_variant(
                            run_dir=args.run_dir,
                            mission_graph=graph,
                            variant_name=variant.name,
                            out_dir=graph_dir,
                            seed=int(seed),
                            generation_overrides={},
                        )
                    else:
                        summary = export_variant(
                            run_dir=args.run_dir,
                            mission_graph=graph,
                            variant_name=variant.name,
                            out_dir=graph_dir,
                            guidance_scale=float(variant.guidance_scale),
                            logic_guidance_scale=float(variant.logic_guidance_scale),
                            num_diffusion_steps=int(variant.num_diffusion_steps),
                            use_fast_sampling=bool(variant.use_fast_sampling),
                            seed=int(seed),
                            generation_overrides={},
                        )

                rows.append(
                    _row_from_summary(
                        regime=regime_name,
                        graph_index=graph_index,
                        seed=int(seed),
                        run_dir=args.run_dir,
                        variant=variant,
                        graph_path=graph_path,
                        output_dir=graph_dir,
                        graph_generation_time_sec=graph_generation_time,
                        summary=summary,
                    )
                )

            _write_csv(args.output_dir / "full_pipeline_rows.partial.csv", rows)

    aggregate = _aggregate(rows)
    paired = _paired_logicnet(rows)
    _write_csv(args.output_dir / "full_pipeline_rows.csv", rows)
    _write_csv(args.output_dir / "full_pipeline_aggregate.csv", aggregate)
    payload = {"manifest": manifest, "rows": rows, "aggregate": aggregate, "paired_logicnet": paired}
    (args.output_dir / "full_pipeline_results.json").write_text(
        json.dumps(_json_sanitize(payload), indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(_markdown_report(rows, aggregate), encoding="utf-8")
    if paired is not None:
        (args.output_dir / "paired_logicnet_summary.json").write_text(
            json.dumps(_json_sanitize(paired), indent=2),
            encoding="utf-8",
        )
    print(json.dumps(_json_sanitize({"output": str(args.output_dir), "rows": len(rows)}), indent=2))


if __name__ == "__main__":
    main()
