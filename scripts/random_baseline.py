"""
Random Baseline for MAP-Elites
===============================

Establishes a null hypothesis: what if we just generate random topologies
without any evolutionary optimization?

The candidates are scored by the same validator-grounded objective as the
experimental method. This is a lower-bound control, not a performance ceiling.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.zelda_data.vglc_utils import validate_topology  # noqa: E402
from src.evaluation.benchmark_suite import (  # noqa: E402
    extract_graph_descriptor,
)
from src.evaluation.map_elites import EliteArchive  # noqa: E402
from src.evaluation.validator import ExternalValidator  # noqa: E402
from src.generation.evolutionary_director import (  # noqa: E402
    mission_graph_to_networkx,
)
from src.generation.grammar import MissionGrammar  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class RandomBaselineResult:
    """Result from random generation baseline."""
    seed: int
    num_samples: int  # Attempted fixed-budget draws.
    archive_cells: int
    mean_fitness_random: float
    std_fitness_random: float
    mean_coverage_random: float
    std_coverage_random: float
    mean_qd_score_random: float
    solvability_rate_random: float = 0.0
    generated_samples: int = 0
    generation_success_rate_random: float = 0.0
    num_elites_random: int = 0
    feature_diversity_random: float = 0.0
    description: str = "Random generation with no optimization"


def generate_random_topologies(
    num_samples: int,
    seed: int,
    eval_budget: int = 100,
) -> List[Dict[str, Any]]:
    """
    Generate topologies with zero optimization, just random grammar applications.
    
    Returns:
        List of random graph candidates with validator-grounded fitness values.
    """
    
    np.random.seed(seed)
    random.seed(seed)
    
    grammar = MissionGrammar(seed=seed)
    validator = ExternalValidator()
    
    # Preserve every budgeted attempt. Dropping failed grammar draws would
    # condition the baseline on success while still reporting ``num_samples``
    # as the requested budget, inflating its apparent quality.
    topologies: List[Dict[str, Any]] = []
    
    for i in range(num_samples):
        candidate: Dict[str, Any] = {
            "graph": None,
            "fitness": 0.0,
            "solvable": False,
            "is_random": True,
            "failure_reason": None,
        }
        try:
            # Generate one mission graph directly from grammar
            mission_graph = grammar.generate(num_rooms=8)

            if mission_graph is None:
                candidate["failure_reason"] = "grammar_returned_none"
            else:
                graph = mission_graph_to_networkx(mission_graph, directed=True)
                if graph is None or graph.number_of_nodes() <= 0:
                    candidate["failure_reason"] = "empty_graph"
                else:
                    validation = validator.validate(graph)
                    topology_valid = bool(validate_topology(graph).is_valid)
                    candidate.update(
                        {
                            "graph": graph,
                            "fitness": float(validation.is_solvable) + (0.25 if topology_valid else 0.0),
                            "solvable": bool(validation.is_solvable),
                        }
                    )
        except Exception as e:
            candidate["failure_reason"] = f"{type(e).__name__}: {e}"
            logger.debug("Failed to generate random topology %d: %s", i, e)
        topologies.append(candidate)
    
    return topologies


def run_random_baseline_with_archive(
    num_samples: int,
    archive_cells: int,
    seed: int,
    output_dir: Path,
) -> RandomBaselineResult:
    """
    Run random topologies through MAP-Elites archive to see how they distribute.
    """
    
    logger.info(f"Running random baseline: {num_samples} samples with {archive_cells} cells")
    
    # Generate random topologies
    topologies = generate_random_topologies(num_samples, seed)
    
    valid_topologies = [topology for topology in topologies if topology.get("graph") is not None]
    logger.info(
        "Generated %d/%d non-empty random topology candidates",
        len(valid_topologies),
        len(topologies),
    )
    
    # Create archive and add them
    try:
        # EliteArchive API expects per-dimension resolution, not total cells.
        cells_per_dim = max(1, int(np.sqrt(archive_cells)))
        archive = EliteArchive(
            feature_dims=2,
            cells_per_dim=cells_per_dim,
        )
        
        for topo in valid_topologies:
            # Extract features
            graph = topo["graph"]
            descriptor = extract_graph_descriptor(graph)
            features = (
                float(getattr(descriptor, "linearity", 0.5)),
                float(getattr(descriptor, "leniency", 0.5)),
            )
            
            # Add to archive
            archive.add(
                solution=topo["graph"],
                fitness=topo["fitness"],
                features=features,
            )
        
        # Archive statistics are conditional on valid archive candidates, while
        # candidate-level quality and solvability remain fixed-budget rates.
        stats_obj = archive.get_stats()
        attempt_fitness = [float(topo["fitness"]) for topo in topologies]
        attempted_solvability = [bool(topo.get("solvable", False)) for topo in topologies]
        
        return RandomBaselineResult(
            seed=seed,
            num_samples=num_samples,
            archive_cells=archive_cells,
            mean_fitness_random=float(np.mean(attempt_fitness)) if attempt_fitness else 0.0,
            std_fitness_random=float(np.std(attempt_fitness)) if attempt_fitness else 0.0,
            mean_coverage_random=stats_obj.coverage,
            std_coverage_random=0,  # Single run
            mean_qd_score_random=stats_obj.total_fitness,
            solvability_rate_random=float(np.mean(attempted_solvability)) if attempted_solvability else 0.0,
            generated_samples=int(len(valid_topologies)),
            generation_success_rate_random=float(len(valid_topologies) / max(1, len(topologies))),
            num_elites_random=int(stats_obj.num_elites),
            feature_diversity_random=float(stats_obj.feature_diversity),
            description="Random generation: no optimization, validator-grounded fitness",
        )
        
    except Exception as e:
        logger.error(f"Archive operation failed: {e}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run random baseline for MAP-Elites"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=96,
        help="Number of random samples to generate",
    )
    parser.add_argument(
        "--archive-cells",
        type=int,
        default=256,
        help="Archive size",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        help="Seeds to run (default: 42 43 44)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/random_baseline"),
        help="Output directory",
    )
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    
    logger.info("="*80)
    logger.info("Random Baseline: MAP-Elites with No Optimization")
    logger.info("="*80)
    logger.info(f"Samples: {args.num_samples}, Archive Cells: {args.archive_cells}")
    logger.info(f"Seeds: {args.seeds}\n")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for seed in args.seeds:
        logger.info(f"Running seed {seed}...")
        result = run_random_baseline_with_archive(
            num_samples=args.num_samples,
            archive_cells=args.archive_cells,
            seed=seed,
            output_dir=output_dir,
        )
        
        if result:
            results.append(asdict(result))
            logger.info(
                "  [OK] Coverage: %.4f, QD-Score: %.2f",
                result.mean_coverage_random,
                result.mean_qd_score_random,
            )
    
    if not results:
        logger.error("No random-baseline seed completed; refusing to write an empty evidence summary.")
        return 1

    # Save results
    output_file = output_dir / "random_baseline_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "title": "Random Baseline Results",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": results,
            "summary": {
                "mean_coverage": float(np.mean([r["mean_coverage_random"] for r in results])),
                "mean_qd_score": float(np.mean([r["mean_qd_score_random"] for r in results])),
                "mean_candidate_generation_success_rate": float(
                    np.mean([r["generation_success_rate_random"] for r in results])
                ),
                "successful_seed_runs": int(len(results)),
                "interpretation": "This is the null hypothesis: random generation achieves this level without optimization.",
            },
        }, f, indent=2)
    
    logger.info("\n[OK] Results saved to %s", output_file)
    logger.info("="*80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
