"""
Compare fixed-graph protocol outputs against already-generated topology baselines.

This is an evidence report, not a claim engine. The fixed-graph protocol measures
room-generation stability on a held mission graph, while the matched-budget and
PCG Benchmark reports evaluate topology generation. The script keeps those
boundaries explicit and only compares dimensions that are actually meaningful.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _safe_optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [_json_sanitize(v) for v in value]
    return value


def _rank_matched_budget_rows(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    rows_list = [dict(row) for row in rows]
    if not rows_list:
        return {}
    return {
        "best_fitness": max(rows_list, key=lambda row: _safe_float(row.get("fitness", 0.0))),
        "best_completeness": max(rows_list, key=lambda row: _safe_float(row.get("overall_completeness", 0.0))),
        "best_novelty": max(rows_list, key=lambda row: _safe_float(row.get("novelty_vs_reference", 0.0))),
        "best_generation_time": min(rows_list, key=lambda row: _safe_float(row.get("generation_time_sec", float("inf")))),
    }


def _rank_external_rows(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        row_dict = dict(row)
        grouped.setdefault(str(row_dict.get("problem_name", "unknown")), []).append(row_dict)

    ranked: Dict[str, Dict[str, Any]] = {}
    for problem_name, problem_rows in grouped.items():
        ranked[problem_name] = {
            "best_quality": max(problem_rows, key=lambda row: _safe_float(row.get("external_quality_pass_rate", 0.0))),
            "best_controlability": max(problem_rows, key=lambda row: _safe_float(row.get("external_controlability_pass_rate", 0.0))),
            "best_diversity": max(problem_rows, key=lambda row: _safe_float(row.get("external_diversity_pass_rate", 0.0))),
            "best_internal_completeness": max(problem_rows, key=lambda row: _safe_float(row.get("internal_overall_completeness", 0.0))),
        }
    return ranked


def _summarize_protocol_variants(aggregate: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for variant_name, payload in dict(aggregate).items():
        payload_dict = dict(payload)
        search = dict(payload_dict.get("search_algorithm_aggregate", {}))
        oracle_stack = dict(search.get("oracle_stack", {}))
        behavioral = dict(search.get("behavioral_probe", {})).get("cbs_balanced", {})
        summary[str(variant_name)] = {
            "avg_repair_rate": _safe_float(payload_dict.get("avg_repair_rate", 0.0)),
            "avg_generation_time_sec": _safe_float(payload_dict.get("avg_generation_time_sec", 0.0)),
            "avg_final_graph_marker_overwrite_rate": _safe_float(
                payload_dict.get("avg_final_graph_marker_overwrite_rate", 0.0)
            ),
            "avg_final_pre_overlay_semantic_anchor_error": _safe_float(
                payload_dict.get("avg_final_pre_overlay_semantic_anchor_error", 0.0)
            ),
            "avg_final_post_overlay_semantic_anchor_error": _safe_float(
                payload_dict.get("avg_final_post_overlay_semantic_anchor_error", 0.0)
            ),
            "astar_grid_solvable_rate": _safe_float(oracle_stack.get("astar_grid_solvable_rate", 0.0)),
            "graph_guided_oracle_solvable_rate": _safe_float(
                oracle_stack.get("graph_guided_oracle_solvable_rate", 0.0)
            ),
            "softlock_safe_rate": _safe_float(oracle_stack.get("softlock_safe_rate", 0.0)),
            "goal_gauntlet_valid_rate": _safe_float(oracle_stack.get("goal_gauntlet_valid_rate", 0.0)),
            "hybrid_oracle_pass_rate": _safe_float(oracle_stack.get("hybrid_oracle_pass_rate", 0.0)),
            "cbs_success_rate": _safe_float(behavioral.get("success_rate", 0.0)),
            "avg_cbs_confusion_ratio_vs_astar": _safe_optional_float(
                behavioral.get("avg_confusion_ratio_vs_astar")
            ),
        }
    return summary


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Protocol vs Baselines")
    lines.append("")
    lines.append("## Claim status")
    lines.append("")
    lines.append(f"- can_claim_surpasses_publications: `{report['claim_status']['can_claim_surpasses_publications']}`")
    for item in report["claim_status"]["reasons"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Fixed-graph protocol")
    lines.append("")
    for variant_name, payload in report["fixed_graph_protocol"]["variants"].items():
        cbs_confusion = payload["avg_cbs_confusion_ratio_vs_astar"]
        cbs_confusion_str = f"{cbs_confusion:.4f}" if isinstance(cbs_confusion, (int, float)) else "n/a"
        lines.append(
            f"- `{variant_name}`: repair={payload['avg_repair_rate']:.4f}, "
                f"overwrite={payload['avg_final_graph_marker_overwrite_rate']:.4f}, "
                f"pre_anchor={payload['avg_final_pre_overlay_semantic_anchor_error']:.4f}, "
                f"post_anchor={payload['avg_final_post_overlay_semantic_anchor_error']:.4f}, "
                f"A*={payload['astar_grid_solvable_rate']:.4f}, "
                f"graph_guided={payload['graph_guided_oracle_solvable_rate']:.4f}, "
                f"softlock={payload['softlock_safe_rate']:.4f}, "
                f"gauntlet={payload['goal_gauntlet_valid_rate']:.4f}, "
                f"hybrid_contract={payload['hybrid_oracle_pass_rate']:.4f}, "
                f"CBS={payload['cbs_success_rate']:.4f}, "
                f"CBS_confusion={cbs_confusion_str}, "
                f"time={payload['avg_generation_time_sec']:.2f}s"
        )
    lines.append("")
    lines.append("## Matched-budget topology baselines")
    lines.append("")
    for label, payload in report["matched_budget"]["ranked"].items():
        lines.append(
            f"- `{label}`: method={payload.get('method')}, "
            f"fitness={_safe_float(payload.get('fitness', 0.0)):.4f}, "
            f"overall_completeness={_safe_float(payload.get('overall_completeness', 0.0)):.4f}, "
            f"linearity={_safe_float(payload.get('linearity', 0.0)):.4f}, "
            f"leniency={_safe_float(payload.get('leniency', 0.0)):.4f}, "
            f"generation_time_sec={_safe_float(payload.get('generation_time_sec', 0.0)):.2f}"
        )
    lines.append("")
    lines.append("## PCG Benchmark alignment")
    lines.append("")
    for problem_name, payload in report["pcg_benchmark_alignment"]["ranked_by_problem"].items():
        quality = payload["best_quality"]
        control = payload["best_controlability"]
        diversity = payload["best_diversity"]
        lines.append(
            f"- `{problem_name}`: best_quality={quality.get('method')} ({_safe_float(quality.get('external_quality_pass_rate', 0.0)):.4f}), "
            f"best_controlability={control.get('method')} ({_safe_float(control.get('external_controlability_pass_rate', 0.0)):.4f}), "
            f"best_diversity={diversity.get('method')} ({_safe_float(diversity.get('external_diversity_pass_rate', 0.0)):.4f})"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for note in report["comparison_notes"]:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def build_report(
    *,
    fixed_graph_summary: Path,
    matched_budget_report: Path,
    pcg_benchmark_report: Path,
) -> Dict[str, Any]:
    fixed_graph = _load_json(fixed_graph_summary)
    matched_budget = _load_json(matched_budget_report)
    pcg_report = _load_json(pcg_benchmark_report)

    fixed_graph_variants = _summarize_protocol_variants(dict(fixed_graph.get("aggregate", {})))
    matched_budget_summary = [dict(row) for row in matched_budget.get("summary", [])]
    pcg_summary = [dict(row) for row in pcg_report.get("summary", [])]

    reasons = [
        "The fixed-graph protocol measures room-generation stability on a fixed mission graph, not the full topology-generation task used by the matched-budget and PCG Benchmark baselines.",
        "The strongest external rows are mixed rather than dominant across all benchmark problems, especially on diversity and smaller-problem controlability.",
        "Strict puzzle/no-puzzle and no-fallback ablations are internal evidence, not direct external matched-budget replacements.",
    ]
    can_claim = False

    return {
        "fixed_graph_protocol": {
            "summary_path": str(fixed_graph_summary),
            "variants": fixed_graph_variants,
        },
        "matched_budget": {
            "report_path": str(matched_budget_report),
            "ranked": _rank_matched_budget_rows(matched_budget_summary),
        },
        "pcg_benchmark_alignment": {
            "report_path": str(pcg_benchmark_report),
            "ranked_by_problem": _rank_external_rows(pcg_summary),
        },
        "comparison_notes": [
            "Use the fixed-graph protocol to compare branch quality and validation stability under the same topology.",
            "Use matched-budget and PCG Benchmark outputs to judge topology-generation competitiveness.",
            "Do not collapse those two evidence layers into one scalar; they measure different stages of the stack.",
        ],
        "claim_status": {
            "can_claim_surpasses_publications": can_claim,
            "reasons": reasons,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare fixed-graph protocol results against saved topology baselines.")
    parser.add_argument("--fixed-graph-summary", type=Path, required=True)
    parser.add_argument("--matched-budget-report", type=Path, required=True)
    parser.add_argument("--pcg-benchmark-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    report = build_report(
        fixed_graph_summary=args.fixed_graph_summary,
        matched_budget_report=args.matched_budget_report,
        pcg_benchmark_report=args.pcg_benchmark_report,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "protocol_vs_baselines.json"
    md_path = args.output_dir / "protocol_vs_baselines.md"
    json_path.write_text(json.dumps(_json_sanitize(report), indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    print(f"[compare-protocol-to-baselines] wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
