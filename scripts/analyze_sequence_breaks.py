"""Sequence-break detector for evolutionary mission graphs.

Runs multiple evolutionary generations and reports whether critical-path gated
segments can be bypassed when gated edges are removed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generation.evolutionary_director import EvolutionaryTopologyGenerator
from src.generation.grammar import EdgeType, MissionGraph


GATED_EDGE_TYPES: Set[EdgeType] = {
    EdgeType.LOCKED,
    EdgeType.BOSS_LOCKED,
    EdgeType.ITEM_GATE,
    EdgeType.STATE_BLOCK,
    EdgeType.MULTI_LOCK,
    EdgeType.ON_OFF_GATE,
    EdgeType.SHUTTER,
}

TRAVERSABLE_EDGE_TYPES: Set[EdgeType] = {
    EdgeType.PATH,
    EdgeType.SHORTCUT,
    EdgeType.ONE_WAY,
    EdgeType.HIDDEN,
    EdgeType.WARP,
    EdgeType.STAIRS,
    EdgeType.HAZARD,
}


def _build_adjacency_without_gates(graph: MissionGraph) -> Dict[int, List[int]]:
    adj: Dict[int, List[int]] = {int(nid): [] for nid in graph.nodes.keys()}
    for edge in graph.edges:
        if edge.edge_type not in TRAVERSABLE_EDGE_TYPES:
            continue
        src = int(edge.source)
        dst = int(edge.target)
        adj.setdefault(src, []).append(dst)
        if edge.edge_type in graph.BIDIRECTIONAL_EDGE_TYPES:
            adj.setdefault(dst, []).append(src)

    for node_id, neighbors in list(adj.items()):
        seen: Set[int] = set()
        deduped: List[int] = []
        for neighbor in neighbors:
            if neighbor in seen:
                continue
            seen.add(neighbor)
            deduped.append(neighbor)
        adj[node_id] = deduped
    return adj


def _find_path(adjacency: Dict[int, List[int]], start_id: int, goal_id: int) -> Optional[List[int]]:
    if start_id == goal_id:
        return [start_id]
    visited = {start_id}
    queue: List[Tuple[int, List[int]]] = [(start_id, [start_id])]
    while queue:
        current, path = queue.pop(0)
        for nxt in adjacency.get(current, []):
            if nxt in visited:
                continue
            new_path = path + [nxt]
            if nxt == goal_id:
                return new_path
            visited.add(nxt)
            queue.append((nxt, new_path))
    return None


def _reachable(adjacency: Dict[int, List[int]], start_id: int, target_id: int) -> bool:
    return _find_path(adjacency, start_id, target_id) is not None


def detect_sequence_breaks(graph: MissionGraph) -> Dict[str, Any]:
    start = graph.get_start_node()
    goal = graph.get_goal_node()
    if not start or not goal:
        return {
            "valid_for_analysis": False,
            "sequence_break_count": 0,
            "checked_gate_count": 0,
            "break_rate": 0.0,
            "details": [],
        }

    critical_path = graph.find_path(int(start.id), int(goal.id))
    if not critical_path:
        return {
            "valid_for_analysis": False,
            "sequence_break_count": 0,
            "checked_gate_count": 0,
            "break_rate": 0.0,
            "details": [],
        }

    critical_nodes = set(int(nid) for nid in critical_path)
    ungated_adj = _build_adjacency_without_gates(graph)

    details: List[Dict[str, Any]] = []
    checked = 0
    breaks = 0

    for edge in graph.edges:
        if edge.edge_type not in GATED_EDGE_TYPES:
            continue
        if int(edge.target) not in critical_nodes:
            continue
        checked += 1
        bypass = _reachable(ungated_adj, int(start.id), int(edge.target))
        if bypass:
            breaks += 1
            details.append(
                {
                    "source": int(edge.source),
                    "target": int(edge.target),
                    "edge_type": str(edge.edge_type.name),
                    "reason": "target reachable without gated edges",
                }
            )

    return {
        "valid_for_analysis": True,
        "sequence_break_count": int(breaks),
        "checked_gate_count": int(checked),
        "break_rate": float(breaks / checked) if checked > 0 else 0.0,
        "details": details,
    }


def run_analysis(args: argparse.Namespace) -> Dict[str, Any]:
    per_sample: List[Dict[str, Any]] = []
    break_rates: List[float] = []
    total_breaks = 0
    total_checked = 0

    for offset in range(int(args.num_samples)):
        seed = int(args.seed) + offset
        generator = EvolutionaryTopologyGenerator(
            target_curve=[0.2, 0.35, 0.5, 0.68, 0.82, 0.95],
            population_size=int(args.population_size),
            generations=int(args.generations),
            genome_length=int(args.genome_length),
            max_nodes=int(args.max_nodes),
            seed=seed,
        )
        generator.evolve(directed_output=True)
        best = generator.last_best_individual
        if best is None or best.phenotype is None:
            per_sample.append({"seed": seed, "valid_for_analysis": False})
            continue

        result = detect_sequence_breaks(best.phenotype)
        result["seed"] = seed
        result["fitness"] = float(best.fitness)
        per_sample.append(result)

        if result.get("valid_for_analysis", False):
            break_rates.append(float(result["break_rate"]))
            total_breaks += int(result["sequence_break_count"])
            total_checked += int(result["checked_gate_count"])

    return {
        "num_samples": int(args.num_samples),
        "seed_start": int(args.seed),
        "population_size": int(args.population_size),
        "generations": int(args.generations),
        "genome_length": int(args.genome_length),
        "max_nodes": int(args.max_nodes),
        "mean_break_rate": float(sum(break_rates) / len(break_rates)) if break_rates else 0.0,
        "aggregate_break_rate": float(total_breaks / total_checked) if total_checked > 0 else 0.0,
        "total_sequence_breaks": int(total_breaks),
        "total_checked_gates": int(total_checked),
        "per_sample": per_sample,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze critical-path sequence breaks")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--population-size", type=int, default=16)
    parser.add_argument("--generations", type=int, default=14)
    parser.add_argument("--genome-length", type=int, default=20)
    parser.add_argument("--max-nodes", type=int, default=26)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/sequence_break_analysis.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_analysis(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "mean_break_rate": report["mean_break_rate"],
        "aggregate_break_rate": report["aggregate_break_rate"],
        "total_checked_gates": report["total_checked_gates"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
