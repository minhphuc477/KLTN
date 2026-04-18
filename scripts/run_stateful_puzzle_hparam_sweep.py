"""
Run a resume-friendly sweep over the stateful puzzle grammar controls.

This gives the puzzle module the same treatment as the VQ-VAE codebook studies:
multiple named settings, one fixed graph, one fixed room branch, and a concrete
ranking artifact instead of hand-picked defaults.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping

import networkx as nx

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.export_manual_rich_topology_compare import (  # type: ignore  # noqa: E402
    build_manual_rich_topology_graph,
    _load_mission_graph,
)
from scripts.run_fast_sampler_visual_audit import export_variant  # type: ignore  # noqa: E402


PUZZLE_PROFILES: Dict[str, Dict[str, Any]] = {
    "baseline_default": {},
    "conservative_quality": {
        "puzzle_room_branch_density": 0.65,
        "puzzle_room_block_budget": 24,
        "puzzle_room_novelty_weight": 0.30,
        "puzzle_room_min_quality_gain": 0.80,
    },
    "route_safe_stateful": {
        "puzzle_room_preserve_route_margin": 1,
        "puzzle_room_candidate_count": 5,
        "puzzle_room_min_quality_gain": 0.70,
        "validator_plan_max_states": 768,
    },
    "dense_stateful": {
        "puzzle_room_branch_density": 0.90,
        "puzzle_room_block_budget": 32,
        "puzzle_room_candidate_count": 6,
        "puzzle_room_novelty_weight": 0.60,
        "puzzle_room_min_quality_gain": 0.60,
    },
    "deterministic_low_novelty": {
        "puzzle_room_novelty_enabled": False,
        "puzzle_room_candidate_count": 1,
        "puzzle_room_min_quality_gain": 0.50,
    },
    "no_puzzle_control": {
        "puzzle_room_scaffold_enabled": False,
        "puzzle_room_structure_enabled": False,
    },
}


def _parse_profiles(raw: str) -> List[str]:
    requested = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not requested:
        raise ValueError("--profiles resolved to an empty set.")
    unsupported = [name for name in requested if name not in PUZZLE_PROFILES]
    if unsupported:
        raise ValueError(
            f"Unsupported puzzle profile(s): {unsupported}. "
            f"Supported: {', '.join(sorted(PUZZLE_PROFILES))}"
        )
    ordered: List[str] = []
    seen = set()
    for name in requested:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def _combined_overrides(
    base_overrides: Mapping[str, Any] | None,
    profile_overrides: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    combined = dict(base_overrides or {})
    combined.update(dict(profile_overrides or {}))
    return combined


def _load_summary(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _profile_score(summary: Mapping[str, Any]) -> float:
    metrics = summary.get("metrics", {}) if isinstance(summary, Mapping) else {}
    validation = summary.get("validation", {}) if isinstance(summary, Mapping) else {}
    runtime = summary.get("runtime_diagnostics", {}) if isinstance(summary, Mapping) else {}
    astar = validation.get("astar_grid", {}) if isinstance(validation, Mapping) else {}
    mechanical_contract = validation.get("mechanical_contract", {}) if isinstance(validation, Mapping) else {}
    softlock = validation.get("softlock_check", {}) if isinstance(validation, Mapping) else {}
    progression = validation.get("graph_progression", {}) if isinstance(validation, Mapping) else {}
    pcbs = validation.get("cbs_balanced", {}) if isinstance(validation, Mapping) else {}

    hard_ok = bool(mechanical_contract.get("hybrid_oracle_pass", False))
    if not hard_ok:
        hard_ok = bool(astar.get("solvable", False)) and bool(softlock.get("is_safe", False)) and bool(
            progression.get("goal_gauntlet_valid", False)
        )
    pcbs_ok = bool(pcbs.get("success", False))
    pcbs_status = str(pcbs.get("status", "") or "").strip().lower()

    confusion_ratio = pcbs.get("confusion_ratio_vs_astar", 0.0)
    confusion_ratio = float(confusion_ratio) if confusion_ratio is not None else 0.0
    if not math.isfinite(confusion_ratio):
        confusion_ratio = 0.0
    confusion_index = float(pcbs.get("confusion_index", 0.0) or 0.0)
    cognitive_load = float(pcbs.get("cognitive_load", 0.0) or 0.0)
    peak_frustration = float(pcbs.get("peak_frustration", 0.0) or 0.0)
    contract_valid = float(runtime.get("puzzle_room_contract_valid", 0.0) or 0.0)
    interaction_valid = float(runtime.get("puzzle_room_interaction_valid", 0.0) or 0.0)
    sequence_valid = float(runtime.get("puzzle_room_sequence_valid", 0.0) or 0.0)
    contract_invalid = float(runtime.get("puzzle_room_contract_invalid", 0.0) or 0.0)
    interaction_invalid = float(runtime.get("puzzle_room_interaction_invalid", 0.0) or 0.0)
    quality_gate_skipped = float(runtime.get("puzzle_room_quality_gate_skipped", 0.0) or 0.0)

    score = 0.0
    score += 100.0 if hard_ok else -500.0
    if pcbs_ok:
        score += 15.0
    elif pcbs_status == "budget_exhausted":
        score -= 6.0
    else:
        score -= 12.0
    score += 20.0 * float(metrics.get("repair_rate", 0.0) or 0.0)
    score -= 30.0 * float(metrics.get("avg_final_graph_marker_overwrite_rate", 0.0) or 0.0)
    score -= 2.0 * float(metrics.get("avg_final_post_overlay_semantic_anchor_error", 0.0) or 0.0)
    score -= 0.05 * float(metrics.get("generation_time_sec", 0.0) or 0.0)
    score -= 2.0 * confusion_ratio
    score -= 0.20 * confusion_index
    score -= 1.50 * cognitive_load
    score -= 4.0 * peak_frustration
    score += 0.5 * float(metrics.get("puzzle_stage_count", 0.0) or 0.0)
    score += 0.25 * float(metrics.get("puzzle_plan_count", 0.0) or 0.0)
    score += 4.0 * contract_valid
    score += 6.0 * interaction_valid
    score += 3.0 * sequence_valid
    score -= 5.0 * contract_invalid
    score -= 4.0 * interaction_invalid
    score -= 2.0 * quality_gate_skipped
    return float(score)


def _build_report(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "# Stateful Puzzle Hyperparameter Sweep",
        "",
        "| Rank | Profile | Score | Hard Oracle | P-CBS | Repair | Overwrite | Anchor Error | Time (s) | Plans | Stages |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    str(row["profile"]),
                    f"{float(row['score']):.3f}",
                    "pass" if bool(row["hard_oracle"]) else "fail",
                    "pass" if bool(row["pcbs_success"]) else "fail",
                    f"{float(row['repair_rate']):.3f}",
                    f"{float(row['overwrite_rate']):.3f}",
                    f"{float(row['anchor_error']):.3f}",
                    f"{float(row['generation_time_sec']):.2f}",
                    str(int(row["puzzle_plan_count"])),
                    str(int(row["puzzle_stage_count"])),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a diffusion-only sweep over stateful puzzle-grammar hyperparameters.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mission-graph", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260418)
    parser.add_argument(
        "--profiles",
        type=str,
        default="baseline_default,conservative_quality,route_safe_stateful,dense_stateful,deterministic_low_novelty,no_puzzle_control",
    )
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = _parse_profiles(args.profiles)
    graph: nx.Graph = (
        _load_mission_graph(args.mission_graph)
        if args.mission_graph is not None
        else build_manual_rich_topology_graph()
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    summaries: Dict[str, Any] = {}

    for profile_name in profiles:
        profile_dir = args.output_dir / profile_name
        summary_path = profile_dir / "summary.json"
        if bool(args.reuse_existing) and summary_path.exists():
            summary = _load_summary(summary_path)
        else:
            summary = export_variant(
                run_dir=args.run_dir,
                mission_graph=copy.deepcopy(graph),
                variant_name=profile_name,
                out_dir=args.output_dir,
                guidance_scale=3.0,
                logic_guidance_scale=0.0,
                num_diffusion_steps=50,
                use_fast_sampling=False,
                seed=int(args.seed),
                generation_overrides=_combined_overrides({}, PUZZLE_PROFILES[profile_name]),
            )
        summaries[profile_name] = summary

        metrics = summary.get("metrics", {})
        validation = summary.get("validation", {})
        row = {
            "profile": profile_name,
            "score": _profile_score(summary),
            "hard_oracle": bool(validation.get("mechanical_contract", {}).get("hybrid_oracle_pass", False))
            or (
                bool(validation.get("astar_grid", {}).get("solvable", False))
                and bool(validation.get("softlock_check", {}).get("is_safe", False))
                and bool(validation.get("graph_progression", {}).get("goal_gauntlet_valid", False))
            ),
            "pcbs_success": bool(validation.get("cbs_balanced", {}).get("success", False)),
            "repair_rate": float(metrics.get("repair_rate", 0.0) or 0.0),
            "overwrite_rate": float(metrics.get("avg_final_graph_marker_overwrite_rate", 0.0) or 0.0),
            "anchor_error": float(metrics.get("avg_final_post_overlay_semantic_anchor_error", 0.0) or 0.0),
            "generation_time_sec": float(metrics.get("generation_time_sec", 0.0) or 0.0),
            "puzzle_plan_count": int(metrics.get("puzzle_plan_count", 0) or 0),
            "puzzle_stage_count": int(metrics.get("puzzle_stage_count", 0) or 0),
            "generation_overrides": dict(summary.get("generation_overrides_applied", {}) or {}),
        }
        rows.append(row)

    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    payload = {
        "profiles": rows,
        "raw_summaries": summaries,
        "best_profile": rows[0]["profile"] if rows else None,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.output_dir / "report.md").write_text(_build_report(rows), encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir / "summary.json"), "best_profile": payload["best_profile"]}, indent=2))


if __name__ == "__main__":
    main()
