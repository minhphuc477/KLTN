"""
Automatic pairwise rule-conflict detector for MissionGrammar.

The detector estimates how often applying rule A causes:
- rule B to become unavailable (blocking conflict),
- invalid mission constraints after B (invalidation conflict),
- exceptions while applying B (execution conflict).
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generation.grammar import Difficulty, MissionGrammar, MissionGraph

logger = logging.getLogger(__name__)


def _new_context(seed: int, difficulty: float = 0.5) -> Dict[str, Any]:
    return {
        "rng": random.Random(int(seed)),
        "difficulty": float(difficulty),
        "goal_row": 5,
        "goal_col": 5,
    }


def _path_exists(graph: MissionGraph) -> bool:
    start = graph.get_start_node()
    goal = graph.get_goal_node()
    if start is None or goal is None:
        return False
    try:
        path = graph.find_path(start.id, goal.id)
        return bool(path)
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect pairwise conflicts between grammar rules.")
    parser.add_argument("--output", type=Path, default=Path("results") / "rule_conflicts")
    parser.add_argument("--num-samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-rooms", type=int, default=12)
    parser.add_argument("--max-keys", type=int, default=3)
    parser.add_argument("--difficulty", type=str, default="MEDIUM", choices=["EASY", "MEDIUM", "HARD", "EXPERT"])
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    if not args.verbose:
        logging.getLogger("src.generation.grammar").setLevel(logging.ERROR)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_grammar = MissionGrammar(seed=int(args.seed))
    rules = list(base_grammar.rules[1:])  # Skip StartRule.
    rule_names = [r.name for r in rules]
    validator = MissionGrammar(seed=int(args.seed) + 999)
    difficulty = Difficulty[str(args.difficulty)]

    pair_stats: Dict[Tuple[str, str], Dict[str, float]] = {}
    for ra in rule_names:
        for rb in rule_names:
            if ra == rb:
                continue
            pair_stats[(ra, rb)] = {
                "trials": 0.0,
                "a_applied": 0.0,
                "a_apply_exceptions": 0.0,
                "b_applicable_before": 0.0,
                "b_applicable_after": 0.0,
                "blocked": 0.0,
                "synergy": 0.0,
                "b_applied": 0.0,
                "b_apply_exceptions": 0.0,
                "invalid_after_pair": 0.0,
            }

    for sample_idx in range(int(args.num_samples)):
        sample_seed = int(args.seed) + sample_idx
        sample_grammar = MissionGrammar(seed=sample_seed)
        base_graph = sample_grammar.generate(
            difficulty=difficulty,
            num_rooms=int(args.num_rooms),
            max_keys=int(args.max_keys),
            validate_all=True,
        )
        base_graph.sanitize()

        for i, rule_a in enumerate(rules):
            for j, rule_b in enumerate(rules):
                if i == j:
                    continue
                key = (rule_a.name, rule_b.name)
                stat = pair_stats[key]
                stat["trials"] += 1.0

                # Evaluate B applicability on the untouched base graph.
                graph_before = copy.deepcopy(base_graph)
                graph_before.sanitize()
                can_b_before = False
                try:
                    can_b_before = bool(
                        rule_b.can_apply(
                            graph_before,
                            _new_context(seed=(sample_seed * 100000) + (i * 1000) + j),
                        )
                    )
                except Exception:
                    can_b_before = False
                if can_b_before:
                    stat["b_applicable_before"] += 1.0

                # Apply A.
                graph_after_a = copy.deepcopy(base_graph)
                graph_after_a.sanitize()
                can_a = False
                try:
                    can_a = bool(
                        rule_a.can_apply(
                            graph_after_a,
                            _new_context(seed=(sample_seed * 100000) + (i * 1000) + j + 1),
                        )
                    )
                except Exception:
                    can_a = False

                if not can_a:
                    continue

                try:
                    graph_after_a = rule_a.apply(
                        graph_after_a,
                        _new_context(seed=(sample_seed * 100000) + (i * 1000) + j + 2),
                    )
                    graph_after_a.sanitize()
                    stat["a_applied"] += 1.0
                except Exception:
                    stat["a_apply_exceptions"] += 1.0
                    continue

                # Evaluate B after A.
                can_b_after = False
                try:
                    can_b_after = bool(
                        rule_b.can_apply(
                            graph_after_a,
                            _new_context(seed=(sample_seed * 100000) + (i * 1000) + j + 3),
                        )
                    )
                except Exception:
                    can_b_after = False

                if can_b_after:
                    stat["b_applicable_after"] += 1.0

                if can_b_before and not can_b_after:
                    stat["blocked"] += 1.0
                elif (not can_b_before) and can_b_after:
                    stat["synergy"] += 1.0

                if not can_b_after:
                    continue

                graph_after_pair = copy.deepcopy(graph_after_a)
                graph_after_pair.sanitize()
                try:
                    graph_after_pair = rule_b.apply(
                        graph_after_pair,
                        _new_context(seed=(sample_seed * 100000) + (i * 1000) + j + 4),
                    )
                    graph_after_pair.sanitize()
                    stat["b_applied"] += 1.0
                except Exception:
                    stat["b_apply_exceptions"] += 1.0
                    continue

                valid = bool(validator.validate_all_constraints(graph_after_pair))
                valid = valid and _path_exists(graph_after_pair)
                if not valid:
                    stat["invalid_after_pair"] += 1.0

    rows: List[Dict[str, Any]] = []
    for (rule_a, rule_b), stat in pair_stats.items():
        a_applied = max(1.0, float(stat["a_applied"]))
        b_applied = max(1.0, float(stat["b_applied"]))
        blocked_rate = float(stat["blocked"] / a_applied)
        invalid_rate = float(stat["invalid_after_pair"] / b_applied)
        exception_rate = float(stat["b_apply_exceptions"] / a_applied)
        conflict_rate = float((stat["blocked"] + stat["invalid_after_pair"] + stat["b_apply_exceptions"]) / a_applied)
        synergy_rate = float(stat["synergy"] / a_applied)
        rows.append(
            {
                "rule_a": rule_a,
                "rule_b": rule_b,
                "trials": int(stat["trials"]),
                "a_applied": int(stat["a_applied"]),
                "b_applicable_before": int(stat["b_applicable_before"]),
                "b_applicable_after": int(stat["b_applicable_after"]),
                "b_applied": int(stat["b_applied"]),
                "blocked": int(stat["blocked"]),
                "invalid_after_pair": int(stat["invalid_after_pair"]),
                "b_apply_exceptions": int(stat["b_apply_exceptions"]),
                "synergy": int(stat["synergy"]),
                "blocked_rate": blocked_rate,
                "invalid_rate": invalid_rate,
                "exception_rate": exception_rate,
                "conflict_rate": conflict_rate,
                "synergy_rate": synergy_rate,
            }
        )

    df = pd.DataFrame(rows).sort_values(["conflict_rate", "blocked_rate", "invalid_rate"], ascending=False)
    matrix = df.pivot(index="rule_a", columns="rule_b", values="conflict_rate").fillna(0.0)
    top_conflicts = df.head(40).copy()

    raw_path = out_dir / "rule_conflicts_raw.csv"
    matrix_path = out_dir / "rule_conflicts_matrix.csv"
    top_path = out_dir / "rule_conflicts_top40.csv"
    report_json = out_dir / "rule_conflicts_report.json"
    report_md = out_dir / "rule_conflicts_report.md"
    df.to_csv(raw_path, index=False)
    matrix.to_csv(matrix_path)
    top_conflicts.to_csv(top_path, index=False)

    summary = {
        "num_samples": int(args.num_samples),
        "num_rules": int(len(rule_names)),
        "mean_conflict_rate": float(df["conflict_rate"].mean()) if not df.empty else 0.0,
        "median_conflict_rate": float(df["conflict_rate"].median()) if not df.empty else 0.0,
        "max_conflict_rate": float(df["conflict_rate"].max()) if not df.empty else 0.0,
        "top_conflicts": top_conflicts.to_dict(orient="records"),
    }
    report_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    try:
        top_md = top_conflicts.to_markdown(index=False)
    except Exception:
        top_md = top_conflicts.to_string(index=False)
    lines = [
        "# Rule Conflict Report",
        "",
        f"- samples: {int(args.num_samples)}",
        f"- rules: {int(len(rule_names))}",
        f"- mean conflict rate: {summary['mean_conflict_rate']:.4f}",
        f"- max conflict rate: {summary['max_conflict_rate']:.4f}",
        "",
        "## Top Conflicts",
        "",
        top_md,
    ]
    report_md.write_text("\n".join(lines), encoding="utf-8")

    logger.info("Saved rule conflict outputs to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
