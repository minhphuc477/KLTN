"""
Run component ablations for Persona-Driven Cognitive Bounded Search (P-CBS).

This script isolates the contribution of the bounded-rational terms:
- revisit penalty
- conditional uncertainty penalty
- deliberation budget
- progression-affordance memory

It is intended for thesis/report tables where P-CBS must be defended as more
than a generic weighted search controller.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.pcbs_validation import prepare_dungeon_grid_for_validation
from src.evaluation.search_benchmark_utils import confusion_ratio_vs_oracle, run_astar_oracle
from src.simulation.cognitive_bounded_search import AgentPersona, CognitiveBoundedSearch, PersonaConfig
from src.simulation.validator import ZeldaLogicEnv
from src.zelda_data.zelda_core import ZeldaDungeonAdapter


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_json_safe(inner) for inner in value]
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    if math.isfinite(numeric):
        return value
    return None


def _ablation_config(persona_name: str, variant: str) -> PersonaConfig:
    persona = AgentPersona(persona_name.lower())
    base = PersonaConfig.get_persona(persona)
    variant = str(variant).lower()
    if variant == "full":
        return replace(base, name=f"{base.name} [full]")
    if variant == "no_revisit":
        return replace(base, name=f"{base.name} [no_revisit]", revisit_penalty_weight=0.0)
    if variant == "no_uncertainty":
        return replace(
            base,
            name=f"{base.name} [no_uncertainty]",
            conditional_uncertainty_penalty_weight=0.0,
            puzzle_complexity_weight=0.0,
        )
    if variant == "no_deliberation":
        return replace(
            base,
            name=f"{base.name} [no_deliberation]",
            deliberation_budget=0.0,
            deliberation_recovery=0.0,
            deliberation_trigger=10.0,
            deliberation_cost_weight=0.0,
        )
    if variant == "no_affordance":
        return replace(
            base,
            name=f"{base.name} [no_affordance]",
            affordance_memory_bonus_weight=0.0,
            affordance_forgetting_penalty_weight=0.0,
            affordance_reactivation_boost=0.0,
        )
    if variant == "no_focus":
        return replace(
            base,
            name=f"{base.name} [no_focus]",
            focus_commitment_bonus_weight=0.0,
            task_switch_penalty_weight=0.0,
        )
    raise ValueError(f"Unsupported ablation variant: {variant}")


def _build_markdown(summary: Dict[str, Any], *, persona: str) -> str:
    lines = [
        f"# P-CBS Component Ablation ({persona})",
        "",
        "| Variant | Success % | Oracle-Cond. Success % | Oracle Solved | CGR | Confusion | Nav Entropy | Cog Load | Aha | Delib | Budget Exhaust | Peak Frustration | Focus Switches | Affordance Reactivations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, stats in summary["variants"].items():
        success_oracle = stats["success_rate_given_oracle_solved"]
        cgr = stats["cognitive_gap_rate_given_oracle_solved"]
        success_oracle_text = f"{success_oracle*100:.1f}" if success_oracle is not None else "n/a"
        cgr_text = f"{cgr*100:.1f}" if cgr is not None else "n/a"
        lines.append(
            f"| {variant} | {stats['success_rate']*100:.1f} | {success_oracle_text} | "
            f"{stats['oracle_solved_maps']} | {cgr_text} | {stats['avg_confusion_index']:.3f} | "
            f"{stats['avg_navigation_entropy']:.3f} | {stats['avg_cognitive_load']:.3f} | "
            f"{stats['avg_aha_latency']:.1f} | {stats['avg_deliberation_events']:.1f} | "
            f"{stats['avg_budget_exhaustion_events']:.1f} | {stats['avg_peak_frustration']:.3f} | "
            f"{stats['avg_focus_switches']:.1f} | {stats['avg_affordance_reactivations']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def run_ablation(
    *,
    dungeon_nums: Iterable[int],
    variants: Iterable[int],
    persona: str,
    ablations: Iterable[str],
    timeout_astar: int,
    timeout_pcbs: int,
    seed: int,
    out_csv: Path,
    verbose: bool,
) -> List[Dict[str, Any]]:
    adapter = ZeldaDungeonAdapter("Data/The Legend of Zelda")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "map_id",
        "ablation",
        "persona",
        "success",
        "path_length",
        "trajectory_length",
        "states_explored",
        "confusion_index",
        "navigation_entropy",
        "cognitive_load",
        "aha_latency",
        "deliberation_events",
        "budget_exhaustion_events",
        "peak_frustration",
        "affordance_reactivations",
        "affordance_guided_steps",
        "inventory_change_events",
        "focus_switches",
        "focus_guided_steps",
        "oracle_status",
        "oracle_success",
        "confusion_ratio",
        "solver_status",
        "time_ms",
    ]

    rows: List[Dict[str, Any]] = []
    for dungeon_num in dungeon_nums:
        for variant in variants:
            map_id = f"D{int(dungeon_num)}_v{int(variant)}"
            dungeon = adapter.load_dungeon(int(dungeon_num), variant=int(variant))
            stitched = adapter.stitch_dungeon(dungeon)
            prepared = prepare_dungeon_grid_for_validation(stitched)
            grid = prepared.grid

            oracle_env = ZeldaLogicEnv(semantic_grid=grid)
            oracle = run_astar_oracle(oracle_env, timeout=int(timeout_astar))

            for ablation_name in ablations:
                env = ZeldaLogicEnv(semantic_grid=grid)
                config = _ablation_config(persona, ablation_name)
                started = time.perf_counter()
                solver = CognitiveBoundedSearch(
                    env,
                    persona=persona,
                    timeout=int(timeout_pcbs),
                    seed=seed,
                    custom_config=config,
                )
                success, path, states, metrics = solver.solve()
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                solver_status = "solved" if success else ("timeout" if int(states) >= int(timeout_pcbs) else "failed")
                trajectory_length = int(len(path))
                solution_path_length = trajectory_length if bool(success) else 0
                confusion_ratio = confusion_ratio_vs_oracle(
                    int(oracle["path_length"]),
                    solution_path_length,
                    oracle_status=str(oracle["status"]),
                    candidate_success=bool(success),
                )
                row = {
                    "map_id": map_id,
                    "ablation": str(ablation_name),
                    "persona": str(persona),
                    "success": int(success),
                    "path_length": int(solution_path_length),
                    "trajectory_length": int(trajectory_length),
                    "states_explored": int(states),
                    "confusion_index": round(float(metrics.confusion_index), 4),
                    "navigation_entropy": round(float(metrics.navigation_entropy), 4),
                    "cognitive_load": round(float(metrics.cognitive_load), 4),
                    "aha_latency": int(metrics.aha_latency),
                    "deliberation_events": int(getattr(metrics, "deliberation_events", 0) or 0),
                    "budget_exhaustion_events": int(getattr(metrics, "budget_exhaustion_events", 0) or 0),
                    "peak_frustration": round(float(getattr(metrics, "peak_frustration", 0.0) or 0.0), 4),
                    "affordance_reactivations": int(getattr(metrics, "affordance_reactivations", 0) or 0),
                    "affordance_guided_steps": int(getattr(metrics, "affordance_guided_steps", 0) or 0),
                    "inventory_change_events": int(getattr(metrics, "inventory_change_events", 0) or 0),
                    "focus_switches": int(getattr(metrics, "focus_switches", 0) or 0),
                    "focus_guided_steps": int(getattr(metrics, "focus_guided_steps", 0) or 0),
                    "oracle_status": str(oracle["status"]),
                    "oracle_success": int(bool(oracle["success"])),
                    "confusion_ratio": round(float(confusion_ratio), 4) if np.isfinite(confusion_ratio) else float("nan"),
                    "solver_status": solver_status,
                    "time_ms": round(float(elapsed_ms), 3),
                }
                rows.append(row)
                if verbose:
                    ratio_text = f"{float(confusion_ratio):.2f}" if np.isfinite(confusion_ratio) else "n/a"
                    print(
                        f"{map_id} {ablation_name}: status={solver_status} path={solution_path_length} "
                        f"trajectory={trajectory_length} "
                        f"states={states} confusion_ratio={ratio_text}"
                    )

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"total_runs": len(rows), "variants": {}}
    by_variant = sorted({str(row["ablation"]) for row in rows})
    for variant in by_variant:
        variant_rows = [row for row in rows if str(row["ablation"]) == variant]
        successful = [row for row in variant_rows if int(row["success"]) == 1]
        oracle_solved_rows = [row for row in variant_rows if int(row["oracle_success"]) == 1]
        oracle_conditioned_success = [row for row in oracle_solved_rows if int(row["success"]) == 1]
        success_given_oracle = (
            len(oracle_conditioned_success) / len(oracle_solved_rows)
            if oracle_solved_rows else None
        )
        summary["variants"][variant] = {
            "success_rate": len(successful) / max(1, len(variant_rows)),
            "success_rate_given_oracle_solved": success_given_oracle,
            "cognitive_gap_rate_given_oracle_solved": (1.0 - success_given_oracle) if success_given_oracle is not None else None,
            "oracle_solved_maps": len(oracle_solved_rows),
            "avg_path_length": float(np.mean([row["path_length"] for row in successful])) if successful else 0.0,
            "avg_confusion_index": float(np.mean([row["confusion_index"] for row in variant_rows])) if variant_rows else 0.0,
            "avg_navigation_entropy": float(np.mean([row["navigation_entropy"] for row in variant_rows])) if variant_rows else 0.0,
            "avg_cognitive_load": float(np.mean([row["cognitive_load"] for row in variant_rows])) if variant_rows else 0.0,
            "avg_aha_latency": float(np.mean([row["aha_latency"] for row in variant_rows])) if variant_rows else 0.0,
            "avg_deliberation_events": float(np.mean([row["deliberation_events"] for row in variant_rows])) if variant_rows else 0.0,
            "avg_budget_exhaustion_events": float(np.mean([row["budget_exhaustion_events"] for row in variant_rows])) if variant_rows else 0.0,
            "avg_peak_frustration": float(np.mean([row["peak_frustration"] for row in variant_rows])) if variant_rows else 0.0,
            "avg_affordance_reactivations": float(np.mean([row["affordance_reactivations"] for row in variant_rows])) if variant_rows else 0.0,
            "avg_affordance_guided_steps": float(np.mean([row["affordance_guided_steps"] for row in variant_rows])) if variant_rows else 0.0,
            "avg_inventory_change_events": float(np.mean([row["inventory_change_events"] for row in variant_rows])) if variant_rows else 0.0,
            "avg_focus_switches": float(np.mean([row["focus_switches"] for row in variant_rows])) if variant_rows else 0.0,
            "avg_focus_guided_steps": float(np.mean([row["focus_guided_steps"] for row in variant_rows])) if variant_rows else 0.0,
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run P-CBS component ablations on Zelda dungeons")
    parser.add_argument("--levels", type=str, default="1,2,3", help="Comma-separated dungeon numbers")
    parser.add_argument("--variants", type=str, default="1,2", help="Comma-separated variant numbers")
    parser.add_argument("--persona", type=str, default="novice", help="Persona to ablate")
    parser.add_argument(
        "--ablations",
        type=str,
        default="full,no_revisit,no_uncertainty,no_deliberation,no_affordance,no_focus",
        help="Comma-separated ablation variants",
    )
    parser.add_argument("--timeout-astar", type=int, default=200000, help="A* timeout")
    parser.add_argument("--timeout-pcbs", type=int, default=50000, help="P-CBS timeout")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="results/pcbs_component_ablation", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Quick smoke run on D1_v1")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    dungeon_nums = [1] if args.quick else [int(token.strip()) for token in args.levels.split(",") if token.strip()]
    variants = [1] if args.quick else [int(token.strip()) for token in args.variants.split(",") if token.strip()]
    ablations = [token.strip() for token in args.ablations.split(",") if token.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = run_ablation(
        dungeon_nums=dungeon_nums,
        variants=variants,
        persona=str(args.persona),
        ablations=ablations,
        timeout_astar=int(args.timeout_astar),
        timeout_pcbs=int(args.timeout_pcbs),
        seed=int(args.seed),
        out_csv=output_dir / "pcbs_component_ablation.csv",
        verbose=not args.quiet,
    )
    summary = summarize(rows)
    (output_dir / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(_build_markdown(summary, persona=str(args.persona)), encoding="utf-8")
    if not args.quiet:
        print(f"Wrote {output_dir / 'pcbs_component_ablation.csv'}")
        print(f"Wrote {output_dir / 'summary.json'}")
        print(f"Wrote {output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
