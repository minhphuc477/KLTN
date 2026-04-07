"""
Run a fixed-graph multi-seed audit for the diffusion / fast-sampler branch.

This keeps the mission graph constant and reruns the same graph-conditioned
generation with multiple random seeds. It is useful after sampler/runtime
fixes because it separates visual stability from Block I topology variance.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from networkx.readwrite import json_graph


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_fast_sampler_visual_audit import (
    add_generation_override_args,
    export_variant,
    generation_overrides_from_namespace,
)
from scripts.export_manual_rich_topology_compare import build_manual_rich_topology_graph


def _load_mission_graph(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return json_graph.node_link_graph(payload, edges="links")


def _safe_mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def _safe_median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def _aggregate_variant(entries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not entries:
        return {}

    repair_rates = [float(entry["metrics"]["repair_rate"]) for entry in entries]
    repaired_tiles = [int(entry["metrics"]["total_tiles_repaired"]) for entry in entries]
    generation_times = [float(entry["metrics"]["generation_time_sec"]) for entry in entries]
    marker_overwrite_rates = [
        float(entry["metrics"].get("avg_final_graph_marker_overwrite_rate", 0.0))
        for entry in entries
    ]
    neural_anchor_errors = [
        float(entry["metrics"].get("avg_neural_semantic_anchor_error", 0.0))
        for entry in entries
    ]
    room_hash_signatures = [
        "|".join(f"{room_id}:{digest}" for room_id, digest in sorted(entry["room_hashes"].items()))
        for entry in entries
    ]

    return {
        "num_runs": len(entries),
        "avg_repair_rate": _safe_mean(repair_rates),
        "median_repair_rate": _safe_median(repair_rates),
        "avg_total_tiles_repaired": _safe_mean(repaired_tiles),
        "median_total_tiles_repaired": _safe_median(repaired_tiles),
        "avg_generation_time_sec": _safe_mean(generation_times),
        "median_generation_time_sec": _safe_median(generation_times),
        "avg_final_graph_marker_overwrite_rate": _safe_mean(marker_overwrite_rates),
        "median_final_graph_marker_overwrite_rate": _safe_median(marker_overwrite_rates),
        "avg_neural_semantic_anchor_error": _safe_mean(neural_anchor_errors),
        "median_neural_semantic_anchor_error": _safe_median(neural_anchor_errors),
        "unique_layout_count": len(set(room_hash_signatures)),
        "all_layouts_identical": len(set(room_hash_signatures)) == 1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fixed-graph multi-seed audit for diffusion / fast-sampler.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training run directory containing checkpoints and resolved_config.json",
    )
    parser.add_argument(
        "--mission-graph",
        type=Path,
        default=None,
        help="Optional path to a fixed mission_graph.json file. If omitted, the built-in rich manual topology is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write per-seed exports and aggregate summary",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[20260404, 20260405, 20260406],
        help="Seeds to audit",
    )
    add_generation_override_args(parser)
    return parser.parse_args()


def run_from_args(args: argparse.Namespace) -> Dict[str, str]:
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"--run-dir does not exist: {run_dir}")

    if args.mission_graph is None:
        mission_graph = build_manual_rich_topology_graph()
        mission_graph_source = "built_in_manual_rich_topology"
    else:
        mission_graph = _load_mission_graph(Path(args.mission_graph))
        mission_graph_source = str(Path(args.mission_graph))

    generation_overrides = generation_overrides_from_namespace(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "mission_graph.json").write_text(
        json.dumps(json_graph.node_link_data(mission_graph, edges="links"), indent=2),
        encoding="utf-8",
    )
    print(
        f"[topology-audit-fixed-graph] run_dir={run_dir} graph={mission_graph_source} "
        f"seeds={[int(seed) for seed in args.seeds]} overrides={generation_overrides}",
        flush=True,
    )

    variants = [
        {
            "variant_name": "diffusion_cfg3_logic0_steps50",
            "guidance_scale": 3.0,
            "logic_guidance_scale": 0.0,
            "num_diffusion_steps": 50,
            "use_fast_sampling": False,
        },
        {
            "variant_name": "fast_cfg3_logic0_steps4",
            "guidance_scale": 3.0,
            "logic_guidance_scale": 0.0,
            "num_diffusion_steps": 4,
            "use_fast_sampling": True,
        },
    ]

    per_seed: List[Dict[str, Any]] = []
    aggregate_inputs: Dict[str, List[Dict[str, Any]]] = {
        str(variant["variant_name"]): [] for variant in variants
    }

    for seed in args.seeds:
        seed_dir = args.output_dir / f"seed_{int(seed)}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_summary: Dict[str, Any] = {"seed": int(seed), "variants": {}}
        for variant in variants:
            print(
                f"[topology-audit-fixed-graph] seed={int(seed)} variant={variant['variant_name']} starting",
                flush=True,
            )
            summary = export_variant(
                run_dir=run_dir,
                mission_graph=copy.deepcopy(mission_graph),
                out_dir=seed_dir,
                seed=int(seed),
                generation_overrides=generation_overrides,
                **variant,
            )
            variant_name = str(variant["variant_name"])
            seed_summary["variants"][variant_name] = summary
            aggregate_inputs[variant_name].append(summary)
            metrics = summary.get("metrics", {})
            print(
                f"[topology-audit-fixed-graph] seed={int(seed)} variant={variant_name} done "
                f"repair_rate={metrics.get('repair_rate')} "
                f"overwrite={metrics.get('avg_final_graph_marker_overwrite_rate')}",
                flush=True,
            )
        per_seed.append(seed_summary)

    aggregate = {
        variant_name: _aggregate_variant(entries)
        for variant_name, entries in aggregate_inputs.items()
    }

    output = {
        "run_dir": str(run_dir),
        "mission_graph": mission_graph_source,
        "seeds": [int(seed) for seed in args.seeds],
        "generation_overrides": generation_overrides,
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[topology-audit-fixed-graph] wrote {summary_path}", flush=True)
    return {"output": str(summary_path)}


def main() -> None:
    args = parse_args()
    print(json.dumps(run_from_args(args), indent=2))


if __name__ == "__main__":
    main()
