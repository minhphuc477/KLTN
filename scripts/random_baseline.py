"""
Random Baseline for MAP-Elites
===============================

Establishes a null hypothesis: what if we just generate random topologies
and assign them random fitness values, without any optimization?

This gives us a performance ceiling to compare against.
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
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.zelda_data.vglc_utils import validate_topology
from src.evaluation.benchmark_suite import (
    extract_graph_descriptor,
    load_vglc_reference_graphs,
)
from src.evaluation.map_elites import EliteArchive, LinearityLeniencyExtractor
from src.generation.evolutionary_director import (
    EvolutionaryTopologyGenerator,
    TensionCurveEvaluator,
    mission_graph_to_networkx,
)
from src.generation.grammar import MissionGrammar

logger = logging.getLogger(__name__)


@dataclass
class RandomBaselineResult:
    """Result from random generation baseline."""
    seed: int
    num_samples: int
    archive_cells: int
    mean_fitness_random: float
    std_fitness_random: float
    mean_coverage_random: float
    std_coverage_random: float
    mean_qd_score_random: float
    description: str = "Random generation with no optimization"


def generate_random_topologies(
    num_samples: int,
    seed: int,
    eval_budget: int = 100,
) -> List[Dict[str, Any]]:
    """
    Generate topologies with zero optimization, just random grammar applications.
    
    Returns:
        list of {"graph": nx.DiGraph, "fitness": float (random)} dicts
    """
    
    np.random.seed(seed)
    random.seed(seed)
    
    grammar = MissionGrammar(seed=seed)
    
    topologies = []
    
    for i in range(num_samples):
        try:
            # Generate one mission graph directly from grammar
            mission_graph = grammar.generate(num_rooms=8)

            if mission_graph is not None:
                graph = mission_graph_to_networkx(mission_graph, directed=True)
                if graph is not None and graph.number_of_nodes() > 0:
                    # Assign random fitness instead of optimization fitness.
                    random_fitness = float(np.random.uniform(0, 1))

                    topologies.append({
                        "graph": graph,
                        "fitness": random_fitness,
                        "is_random": True,
                    })
        except Exception as e:
            logger.debug(f"Failed to generate topology {i}: {e}")
            continue
    
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
    
    if not topologies:
        logger.error("Failed to generate any topologies")
        return None
    
    logger.info(f"Generated {len(topologies)} topologies")
    
    # Create archive and add them
    try:
        # EliteArchive API expects per-dimension resolution, not total cells.
        cells_per_dim = max(1, int(np.sqrt(archive_cells)))
        archive = EliteArchive(
            feature_dims=2,
            cells_per_dim=cells_per_dim,
        )
        
        for topo in topologies:
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
        
        # Get stats
        stats_obj = archive.get_stats()
        
        return RandomBaselineResult(
            seed=seed,
            num_samples=num_samples,
            archive_cells=archive_cells,
            mean_fitness_random=stats_obj.mean_fitness,
            std_fitness_random=np.std([topo["fitness"] for topo in topologies]),
            mean_coverage_random=stats_obj.coverage,
            std_coverage_random=0,  # Single run
            mean_qd_score_random=stats_obj.total_fitness,
            description="Random generation: no optimization, random fitness assignment",
        )
        
    except Exception as e:
        logger.error(f"Archive operation failed: {e}")
        return None


def main():
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
            logger.info(f"  ✓ Coverage: {result.mean_coverage_random:.4f}, QD-Score: {result.mean_qd_score_random:.2f}")
    
    # Save results
    output_file = output_dir / "random_baseline_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "title": "Random Baseline Results",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": results,
            "summary": {
                "mean_coverage": float(np.mean([r["mean_coverage_random"] for r in results])),
                "mean_qd_score": float(np.mean([r["mean_qd_score_random"] for r in results])),
                "interpretation": "This is the null hypothesis: random generation achieves this level without optimization.",
            },
        }, f, indent=2)
    
    logger.info(f"\n✓ Results saved to {output_file}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
