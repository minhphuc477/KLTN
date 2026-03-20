"""Evaluation utilities for the H-MOLQD pipeline.

This module provides a lightweight framework for computing key evaluation
metrics on a collection of dungeon graphs.  It is intended to be run locally
from the command line, producing simple summary statistics that mirror the
unit tests but operate on larger datasets.

Usage example::

    python -m src.evaluation.evaluate --num-random 50

The current metrics include:

* solvability rate (via ExternalValidator)
* average solver path length (AgentSimulator)
* MAP-Elites coverage, QD-score, diversity

The module also demonstrates how to generate a small set of query graphs and
collect corresponding responses, which can later be exported to JSON for
use with the evaluation-planner system.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx
import numpy as np

from src.evaluation.validator import ExternalValidator
from src.evaluation.map_elites import (
    LinearityLeniencyExtractor,
    EliteArchive,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def make_simple_graph(
    soluble: bool = True,
    length: int = 5,
    include_locked: bool = False,
) -> nx.DiGraph:
    """Create a simple linear graph for testing.

    If ``soluble`` is False it adds a locked edge without a key.
    ``include_locked`` adds periodically some key-locked edges with keys
    earlier in the graph (so always solvable).
    """
    G = nx.DiGraph()
    for i in range(length):
        label = ''
        if i == 0:
            label = 's'
        elif i == length - 1:
            label = 't'
        elif include_locked and i % 3 == 0:
            label = 'k'
        G.add_node(i, label=label)
        if i > 0:
            if include_locked and i % 4 == 0:
                edge_type = 'key_locked'
            else:
                edge_type = 'open'
            G.add_edge(i - 1, i, edge_type=edge_type)
    if not soluble:
        # break by changing first edge to locked without key
        if G.number_of_nodes() > 1:
            G.edges[0, 1]['edge_type'] = 'key_locked'
    return G


def evaluate_solvability(graphs: List[nx.DiGraph]) -> Dict[str, Any]:
    """Run solvability evaluation on a set of graphs."""
    validator = ExternalValidator()

    total = len(graphs)
    solvable = 0
    path_lengths: List[int] = []

    for g in graphs:
        result = validator.validate(g)
        if getattr(result, 'is_solvable', False):
            solvable += 1
            if getattr(result, 'path_length', 0):
                path_lengths.append(int(result.path_length))
            elif getattr(result, 'solution_path', None):
                path_lengths.append(len(result.solution_path))

    stats = {
        'total': total,
        'solvable': solvable,
        'solvability_rate': solvable / total if total else 0.0,
        'avg_path_length': float(np.mean(path_lengths)) if path_lengths else 0.0,
    }
    return stats


def evaluate_map_elites(graphs: List[nx.DiGraph]) -> Dict[str, float]:
    """Run a small MAP-Elites analysis on a set of graphs."""
    extractor = LinearityLeniencyExtractor()
    archive = EliteArchive(feature_dims=2, cells_per_dim=10)
    validator = ExternalValidator()

    for g in graphs:
        # use solvability as fitness
        fitness = 1.0 if validator.validate(g).is_solvable else 0.0
        # attempt to add with automatic feature extraction
        archive.add(solution=g, fitness=fitness, features=extractor.extract(g))

    stats = archive.get_stats()
    stats = {
        'coverage': float(stats.coverage),
        'qd_score': float(stats.total_fitness),
        'mean_fitness': float(stats.mean_fitness),
        'feature_diversity': float(stats.feature_diversity),
        'num_elites': int(stats.num_elites),
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Run simple evaluation")
    parser.add_argument("--num-random", type=int, default=10,
                        help="number of random graphs to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for reproducible graph generation")
    parser.add_argument("--output", type=Path, default=None,
                        help="path to dump queries/responses JSON")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # generate query graphs
    graphs = [make_simple_graph(soluble=bool(rng.getrandbits(1)),
                                length=rng.randint(3, 8),
                                include_locked=True)
              for _ in range(args.num_random)]

    solv_stats = evaluate_solvability(graphs)
    logger.info("Solvability stats: %s", solv_stats)

    map_stats = evaluate_map_elites(graphs)
    logger.info("MAP-Elites stats: %s", map_stats)

    # optionally write a dataset file
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        data = []
        for g in graphs:
            nodes = [{'id': n, **g.nodes[n]} for n in g.nodes()]
            edges = [{'u': u, 'v': v, **g[u][v]} for u, v in g.edges()]
            data.append({'nodes': nodes, 'edges': edges})
        args.output.write_text(
            json.dumps(
                {
                    'seed': args.seed,
                    'num_graphs': len(graphs),
                    'solvability': solv_stats,
                    'map_elites': map_stats,
                    'graphs': data,
                },
                indent=2,
            )
        )
        logger.info("Wrote %d query graphs to %s", len(graphs), args.output)

if __name__ == '__main__':
    main()
