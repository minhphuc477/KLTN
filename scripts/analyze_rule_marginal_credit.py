"""Per-rule marginal fitness credit analysis.

Runs evolutionary search once, then performs leave-one-out ablations over the
best genome to estimate rule-level marginal contributions.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generation.evolutionary_director import EvolutionaryTopologyGenerator

logger = logging.getLogger(__name__)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer")
    return parsed


def _parse_target_curve(value: str) -> List[float]:
    items = [part.strip() for part in value.split(",") if part.strip()]
    if len(items) < 2:
        raise argparse.ArgumentTypeError("Target curve must contain at least 2 comma-separated values")
    try:
        return [float(item) for item in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Target curve values must be numeric") from exc


def _write_json_atomic(output_path: Path, payload: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze marginal fitness credit per rule")
    parser.add_argument("--population-size", type=_positive_int, default=24)
    parser.add_argument("--generations", type=_positive_int, default=18)
    parser.add_argument("--genome-length", type=_positive_int, default=24)
    parser.add_argument("--max-nodes", type=_positive_int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=_positive_int, default=12, help="Max gene positions to ablate")
    parser.add_argument(
        "--target-curve",
        type=_parse_target_curve,
        default=[0.2, 0.35, 0.5, 0.68, 0.82, 0.95],
        help="Comma-separated target tension curve values",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--output", type=Path, default=Path("results/rule_marginal_credit.json"))
    return parser.parse_args()


def _topk(items: Dict[str, float], k: int, reverse: bool) -> List[Dict[str, Any]]:
    ranked = sorted(items.items(), key=lambda x: x[1], reverse=reverse)
    return [{"rule": str(name), "avg_delta": float(delta)} for name, delta in ranked[:k]]


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    generator = EvolutionaryTopologyGenerator(
        target_curve=args.target_curve,
        population_size=int(args.population_size),
        generations=int(args.generations),
        genome_length=int(args.genome_length),
        max_nodes=int(args.max_nodes),
        seed=int(args.seed),
    )

    logger.info("Running marginal-credit analysis with seed=%s", args.seed)
    generator.evolve(directed_output=True)
    best = generator.last_best_individual
    if best is None:
        raise RuntimeError("No best individual captured from evolution run")

    generation_stats = getattr(best.phenotype, "generation_stats", {}) if best.phenotype else {}

    deltas_by_index, deltas_by_rule = generator.compute_rule_marginal_credit(
        best.genome,
        max_samples=int(args.max_samples),
    )
    sampled_positions = sorted(int(idx) for idx in deltas_by_index.keys())

    result = {
        "seed": int(args.seed),
        "population_size": int(args.population_size),
        "generations": int(args.generations),
        "genome_length": int(args.genome_length),
        "max_nodes": int(args.max_nodes),
        "best_fitness": float(best.fitness),
        "best_genome": [int(x) for x in best.genome],
        "rule_trace_available": bool(generation_stats.get("rule_trace")),
        "max_samples_requested": int(args.max_samples),
        "num_positions_sampled": int(len(sampled_positions)),
        "sampled_positions": sampled_positions,
        "marginal_delta_by_index": {str(idx): float(delta) for idx, delta in deltas_by_index.items()},
        "marginal_delta_by_rule": {str(rule): float(delta) for rule, delta in deltas_by_rule.items()},
        "top_helpful_rules": _topk(deltas_by_rule, k=8, reverse=True),
        "top_harmful_rules": _topk(deltas_by_rule, k=8, reverse=False),
    }

    _write_json_atomic(args.output, result)
    logger.info("Wrote marginal-credit report: %s", args.output)

    print(json.dumps(
        {
            "best_fitness": result["best_fitness"],
            "num_rule_deltas": len(result["marginal_delta_by_rule"]),
            "output": str(args.output),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
