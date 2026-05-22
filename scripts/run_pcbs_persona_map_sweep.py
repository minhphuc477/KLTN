#!/usr/bin/env python3
"""Run P-CBS persona sweeps on oracle-solvable Zelda maps.

The component-ablation runner is intentionally expensive because it evaluates
several disabled-term variants per map. This runner keeps the P-CBS model intact
and broadens the evidence across personas/maps, reporting the cognitive gap rate
only on maps that the hard A* oracle solves.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.pcbs_validation import prepare_dungeon_grid_for_validation
from src.evaluation.search_benchmark_utils import confusion_ratio_vs_oracle, run_astar_oracle
from src.simulation.cognitive_bounded_search import CognitiveBoundedSearch
from src.simulation.validator import ZeldaLogicEnv
from src.zelda_data.zelda_core import ZeldaDungeonAdapter


def _tokens(raw: str) -> List[str]:
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def _ints(raw: str) -> List[int]:
    return [int(token) for token in _tokens(raw)]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    if math.isfinite(numeric):
        return value
    return None


def _mean(values: Iterable[Any]) -> Optional[float]:
    cleaned: List[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            cleaned.append(numeric)
    if not cleaned:
        return None
    return float(statistics.fmean(cleaned))


def _rate(rows: Iterable[Mapping[str, Any]], key: str) -> Optional[float]:
    vals = list(rows)
    if not vals:
        return None
    return float(sum(1 for row in vals if bool(int(row.get(key, 0)))) / len(vals))


def _headers() -> List[str]:
    return [
        "map_id",
        "level",
        "variant",
        "persona",
        "oracle_status",
        "oracle_success",
        "oracle_path_length",
        "oracle_states_explored",
        "pcbs_success",
        "pcbs_status",
        "pcbs_path_length",
        "pcbs_trajectory_length",
        "pcbs_states_explored",
        "confusion_ratio",
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
        "time_ms",
    ]


def _row_for_persona(
    *,
    grid: np.ndarray,
    level: int,
    variant: int,
    persona: str,
    oracle: Mapping[str, Any],
    timeout_pcbs: int,
    seed: int,
) -> Dict[str, Any]:
    env = ZeldaLogicEnv(semantic_grid=grid)
    started = time.perf_counter()
    solver = CognitiveBoundedSearch(
        env,
        persona=str(persona),
        timeout=int(timeout_pcbs),
        seed=int(seed),
    )
    try:
        success, path, states, metrics = solver.solve()
    finally:
        try:
            env.close()
        except Exception:
            pass
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    pcbs_status = "solved" if success else ("timeout" if int(states) >= int(timeout_pcbs) else "failed")
    pcbs_trajectory_length = int(len(path or []))
    pcbs_solution_length = pcbs_trajectory_length if bool(success) else 0
    confusion_ratio = confusion_ratio_vs_oracle(
        int(oracle.get("path_length", 0) or 0),
        pcbs_solution_length,
        oracle_status=str(oracle.get("status", "unknown")),
        candidate_success=bool(success),
    )
    return {
        "map_id": f"D{int(level)}_v{int(variant)}",
        "level": int(level),
        "variant": int(variant),
        "persona": str(persona),
        "oracle_status": str(oracle.get("status", "unknown")),
        "oracle_success": int(bool(oracle.get("success", False))),
        "oracle_path_length": int(oracle.get("path_length", 0) or 0),
        "oracle_states_explored": int(oracle.get("states_explored", 0) or 0),
        "pcbs_success": int(bool(success)),
        "pcbs_status": pcbs_status,
        "pcbs_path_length": int(pcbs_solution_length),
        "pcbs_trajectory_length": int(pcbs_trajectory_length),
        "pcbs_states_explored": int(states or 0),
        "confusion_ratio": round(float(confusion_ratio), 4) if np.isfinite(confusion_ratio) else float("nan"),
        "confusion_index": round(float(getattr(metrics, "confusion_index", 0.0) or 0.0), 4),
        "navigation_entropy": round(float(getattr(metrics, "navigation_entropy", 0.0) or 0.0), 4),
        "cognitive_load": round(float(getattr(metrics, "cognitive_load", 0.0) or 0.0), 4),
        "aha_latency": int(getattr(metrics, "aha_latency", 0) or 0),
        "deliberation_events": int(getattr(metrics, "deliberation_events", 0) or 0),
        "budget_exhaustion_events": int(getattr(metrics, "budget_exhaustion_events", 0) or 0),
        "peak_frustration": round(float(getattr(metrics, "peak_frustration", 0.0) or 0.0), 4),
        "affordance_reactivations": int(getattr(metrics, "affordance_reactivations", 0) or 0),
        "affordance_guided_steps": int(getattr(metrics, "affordance_guided_steps", 0) or 0),
        "inventory_change_events": int(getattr(metrics, "inventory_change_events", 0) or 0),
        "focus_switches": int(getattr(metrics, "focus_switches", 0) or 0),
        "focus_guided_steps": int(getattr(metrics, "focus_guided_steps", 0) or 0),
        "time_ms": round(float(elapsed_ms), 3),
    }


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"total_runs": len(rows), "personas": {}}
    for persona in sorted({str(row["persona"]) for row in rows}):
        persona_rows = [row for row in rows if str(row["persona"]) == persona]
        oracle_rows = [row for row in persona_rows if int(row["oracle_success"]) == 1]
        solved_rows = [row for row in oracle_rows if int(row["pcbs_success"]) == 1]
        success_given_oracle = len(solved_rows) / len(oracle_rows) if oracle_rows else None
        summary["personas"][persona] = {
            "n": len(persona_rows),
            "oracle_solved_maps": len(oracle_rows),
            "success_rate": _rate(persona_rows, "pcbs_success"),
            "success_rate_given_oracle_solved": success_given_oracle,
            "cognitive_gap_rate_given_oracle_solved": (
                1.0 - success_given_oracle if success_given_oracle is not None else None
            ),
            "avg_confusion_ratio_oracle_solved": _mean(row.get("confusion_ratio") for row in solved_rows),
            "avg_confusion_index": _mean(row.get("confusion_index") for row in persona_rows),
            "avg_navigation_entropy": _mean(row.get("navigation_entropy") for row in persona_rows),
            "avg_cognitive_load": _mean(row.get("cognitive_load") for row in persona_rows),
            "avg_deliberation_events": _mean(row.get("deliberation_events") for row in persona_rows),
            "avg_affordance_reactivations": _mean(row.get("affordance_reactivations") for row in persona_rows),
            "avg_focus_switches": _mean(row.get("focus_switches") for row in persona_rows),
            "avg_time_ms": _mean(row.get("time_ms") for row in persona_rows),
        }
    return summary


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# P-CBS Persona/Map Sweep",
        "",
        "| Persona | Runs | Oracle-solvable maps | Success on oracle-solvable % | CGR % | Confusion ratio | Confusion index | Cognitive load |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for persona, stats in dict(summary.get("personas", {})).items():
        success = stats.get("success_rate_given_oracle_solved")
        cgr = stats.get("cognitive_gap_rate_given_oracle_solved")
        prefix = f"| {persona} | {int(stats.get('n', 0))} | {int(stats.get('oracle_solved_maps', 0))} | "
        if success is not None:
            line = prefix + f"{(float(success) * 100.0):.1f}"
            lines.append(
                line
                + f" | {(float(cgr) * 100.0):.1f} | "
                + f"{_format_float(stats.get('avg_confusion_ratio_oracle_solved'))} | "
                + f"{_format_float(stats.get('avg_confusion_index'))} | "
                + f"{_format_float(stats.get('avg_cognitive_load'))} |"
            )
        else:
            line = prefix + "n/a"
            lines.append(
                line
                + " | n/a | "
                + f"{_format_float(stats.get('avg_confusion_ratio_oracle_solved'))} | "
                + f"{_format_float(stats.get('avg_confusion_index'))} | "
                + f"{_format_float(stats.get('avg_cognitive_load'))} |"
            )
    return "\n".join(lines) + "\n"


def _format_float(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(numeric):
        return "n/a"
    return f"{numeric:.3f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full P-CBS across personas and maps.")
    parser.add_argument("--levels", type=str, default="1,2,3,4,5,6")
    parser.add_argument("--variants", type=str, default="1")
    parser.add_argument("--personas", type=str, default="novice,balanced,speedrunner")
    parser.add_argument("--timeout-astar", type=int, default=200000)
    parser.add_argument("--timeout-pcbs", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--output-dir", type=Path, default=Path("results/pcbs_persona_map_sweep"))
    parser.add_argument("--oracle-solved-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "pcbs_persona_map_sweep.csv"
    adapter = ZeldaDungeonAdapter("Data/The Legend of Zelda")
    levels = _ints(args.levels)
    variants = _ints(args.variants)
    personas = _tokens(args.personas)
    rows: List[Dict[str, Any]] = []

    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_headers())
        writer.writeheader()
        for level in levels:
            for variant in variants:
                map_id = f"D{int(level)}_v{int(variant)}"
                if not args.quiet:
                    print(f"[oracle] {map_id}", flush=True)
                dungeon = adapter.load_dungeon(int(level), variant=int(variant))
                stitched = adapter.stitch_dungeon(dungeon)
                prepared = prepare_dungeon_grid_for_validation(stitched)
                grid = prepared.grid
                oracle_env = ZeldaLogicEnv(semantic_grid=grid)
                try:
                    oracle = run_astar_oracle(oracle_env, timeout=int(args.timeout_astar))
                finally:
                    try:
                        oracle_env.close()
                    except Exception:
                        pass
                if args.oracle_solved_only and not bool(oracle.get("success", False)):
                    if not args.quiet:
                        print(f"[skip] {map_id}: oracle_status={oracle.get('status')}", flush=True)
                    continue
                for persona in personas:
                    if not args.quiet:
                        print(f"[pcbs] {map_id} persona={persona}", flush=True)
                    row = _row_for_persona(
                        grid=grid,
                        level=int(level),
                        variant=int(variant),
                        persona=str(persona),
                        oracle=oracle,
                        timeout_pcbs=int(args.timeout_pcbs),
                        seed=int(args.seed),
                    )
                    rows.append(row)
                    writer.writerow(row)
                    handle.flush()

    summary = _summarize(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    (args.output_dir / "report.md").write_text(_markdown(summary), encoding="utf-8")
    if not args.quiet:
        print(f"Wrote {rows_path}")
        print(f"Wrote {args.output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
