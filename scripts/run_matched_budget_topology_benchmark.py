"""Matched-budget Block-I topology benchmark."""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.vglc_utils import filter_virtual_nodes, validate_topology
from src.evaluation.benchmark_suite import (
    calibrate_rule_weights_to_vglc,
    extract_graph_descriptor,
    load_vglc_reference_graphs,
    run_block_i_benchmark,
)
from src.generation.evolutionary_director import (
    EvolutionaryTopologyGenerator,
    GraphGrammarExecutor,
    TensionCurveEvaluator,
    mission_graph_to_networkx,
    networkx_to_mission_graph,
)
from src.generation.grammar import MissionGraph, MissionGrammar

logger = logging.getLogger(__name__)
METHODS_ALL = ("RANDOM", "ES", "GA", "MAP_ELITES", "FULL")


@dataclass
class SearchResult:
    graph: nx.DiGraph
    generation_time_sec: float
    fitness: float
    feasible: bool
    constraint_violation: float
    evaluations_used: int


@dataclass
class SeedScenario:
    seed: int
    room_count: int
    target_curve: List[float]


def _target_curve_for_rooms(rng: np.random.Generator, n_rooms: int) -> List[float]:
    if n_rooms <= 1:
        return [0.5]
    xs = np.linspace(0.0, 1.0, n_rooms)
    slope = rng.uniform(0.45, 0.95)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    wave = 0.15 * np.sin((2.0 * np.pi * xs) + phase)
    return [float(v) for v in np.clip(0.15 + slope * xs + wave, 0.0, 1.0)]


def _sample_room_count(rng: np.random.Generator, min_rooms: int, max_rooms: int, room_count_bias: float) -> int:
    if max_rooms <= min_rooms:
        return int(max(1, min_rooms))
    bias = float(np.clip(room_count_bias, 0.0, 1.0))
    if bias <= 1e-9:
        return int(rng.integers(min_rooms, max_rooms + 1))
    u = float(rng.beta(1.0 + (2.0 * bias), 1.0))
    rc = int(min_rooms + round(u * float(max_rooms - min_rooms)))
    return int(np.clip(rc, min_rooms, max_rooms))


def _effective_room_budget(reference_graphs: Sequence[nx.Graph], min_rooms: int, max_rooms: int, align: bool, cap: int) -> Tuple[int, int]:
    if (not align) or (not reference_graphs):
        return int(min_rooms), int(max(max_rooms, min_rooms + 1))
    ref_nodes = np.asarray([max(1, int(g.number_of_nodes())) for g in reference_graphs], dtype=np.float64)
    q25 = int(np.floor(np.quantile(ref_nodes, 0.25)))
    q75 = int(np.ceil(np.quantile(ref_nodes, 0.75)))
    c = max(6, int(cap))
    lo = int(np.clip(q25, 6, c))
    hi = int(np.clip(max(q75, lo + 2), lo + 1, c))
    out_min = max(int(min_rooms), lo)
    out_max = min(max(int(max_rooms), hi), c)
    if out_min >= out_max:
        out_min = max(6, out_max - 2)
    return out_min, out_max


def _descriptor_targets(reference_graphs: Sequence[nx.Graph]) -> Dict[str, float]:
    desc = [extract_graph_descriptor(g, grammar=None) for g in reference_graphs]
    if not desc:
        return {
            "linearity": 0.45,
            "leniency": 0.5,
            "progression_complexity": 0.68,
            "topology_complexity": 0.45,
            "path_length": 9.0,
            "num_nodes": 20.0,
        }
    return {
        "linearity": float(np.mean([d.linearity for d in desc])),
        "leniency": float(np.mean([d.leniency for d in desc])),
        "progression_complexity": float(np.mean([d.progression_complexity for d in desc])),
        "topology_complexity": float(np.mean([d.topology_complexity for d in desc])),
        "path_length": float(np.mean([d.path_length for d in desc])),
        "num_nodes": float(np.mean([d.num_nodes for d in desc])),
    }


def _descriptor_vec(G: nx.Graph) -> np.ndarray:
    d = extract_graph_descriptor(G, grammar=None)
    return np.array([d.linearity, d.leniency, d.progression_complexity, d.topology_complexity], dtype=np.float64)


def _candidate_key(feasible: bool, fitness: float, violation: float) -> Tuple[int, float, float]:
    return (0 if feasible else 1, -float(fitness), float(violation))


def _is_better(current: Optional[Tuple[bool, float, float]], candidate: Tuple[bool, float, float]) -> bool:
    if current is None:
        return True
    return _candidate_key(*candidate) < _candidate_key(*current)


def _graph_edit_distance_proxy(Ga: nx.Graph, Gb: nx.Graph) -> float:
    na, nb = float(Ga.number_of_nodes()), float(Gb.number_of_nodes())
    ea, eb = float(Ga.number_of_edges()), float(Gb.number_of_edges())
    node_term = abs(na - nb) / max(1.0, na, nb)
    edge_term = abs(ea - eb) / max(1.0, ea, eb)

    def _hist(G: nx.Graph) -> Dict[str, int]:
        h: Dict[str, int] = {}
        for _, attrs in G.nodes(data=True):
            t = str(attrs.get("type", attrs.get("label", "unknown"))).lower()
            h[t] = h.get(t, 0) + 1
        return h

    ha, hb = _hist(Ga), _hist(Gb)
    keys = sorted(set(ha.keys()) | set(hb.keys()))
    if keys:
        va = np.array([ha.get(k, 0) for k in keys], dtype=np.float64)
        vb = np.array([hb.get(k, 0) for k in keys], dtype=np.float64)
        va /= max(np.sum(va), 1.0)
        vb /= max(np.sum(vb), 1.0)
        type_term = float(np.mean(np.abs(va - vb)))
    else:
        type_term = 0.0
    return float(0.4 * node_term + 0.35 * edge_term + 0.25 * type_term)


def _nearest_graph_edit_distance(G: nx.Graph, refs: Sequence[nx.Graph], max_refs: int = 24) -> float:
    if not refs:
        return 0.0
    cand = refs[: max(1, int(max_refs))]
    return float(min(_graph_edit_distance_proxy(G, r) for r in cand))


def _mission_to_graph(mission: MissionGraph) -> nx.DiGraph:
    graph = mission_graph_to_networkx(mission, directed=True)
    physical = filter_virtual_nodes(graph)
    report = validate_topology(physical)
    if not report.is_valid:
        logger.debug("Topology warnings: %s", report.summary())
    return physical


def _random_genome(rng: random.Random, rule_ids: Sequence[int], weights: Sequence[float], genome_length: int) -> List[int]:
    return [int(rng.choices(rule_ids, weights=weights, k=1)[0]) for _ in range(genome_length)]


def _derive_pop_gen(eval_budget: int, pop_hint: int) -> Tuple[int, int]:
    b = max(16, int(eval_budget))
    p = int(max(8, min(int(pop_hint), max(8, b // 4))))
    g = int(max(1, (b // p) - 1))
    return p, g

def _run_random(
    scenario: SeedScenario,
    eval_budget: int,
    seed: int,
    descriptor_targets: Dict[str, float],
    rule_space: str,
    rule_weight_overrides: Optional[Dict[str, float]],
) -> SearchResult:
    started = time.time()
    executor = GraphGrammarExecutor(
        seed=seed,
        use_full_rule_space=(str(rule_space).strip().lower() == "full"),
        rule_weight_overrides=rule_weight_overrides,
    )
    evaluator = TensionCurveEvaluator(scenario.target_curve, descriptor_targets=descriptor_targets)
    rng = random.Random(seed + 911)
    rule_ids = list(range(1, len(executor.rules)))
    weights = [max(1e-6, float(executor.rules[rid].weight)) for rid in rule_ids]
    genome_length = int(max(8, min(28, scenario.room_count + 4)))

    best_graph: Optional[MissionGraph] = None
    best_triplet: Optional[Tuple[bool, float, float]] = None

    for _ in range(max(1, int(eval_budget))):
        genome = _random_genome(rng, rule_ids, weights, genome_length)
        mission = executor.execute(genome, difficulty=0.5, max_nodes=scenario.room_count)
        result = evaluator.evaluate_graph(mission)
        triplet = (
            bool(result.get("feasible", False)),
            float(result.get("fitness", 0.0)),
            float(result.get("constraint_violation", 1.0)),
        )
        if _is_better(best_triplet, triplet):
            best_triplet = triplet
            best_graph = mission

    if best_graph is None:
        raise RuntimeError("RANDOM search produced no graph")

    feasible, fitness, violation = best_triplet if best_triplet is not None else (False, 0.0, 1.0)
    return SearchResult(
        graph=_mission_to_graph(best_graph),
        generation_time_sec=float(time.time() - started),
        fitness=float(fitness),
        feasible=bool(feasible),
        constraint_violation=float(violation),
        evaluations_used=max(1, int(eval_budget)),
    )


def _run_evolution(
    scenario: SeedScenario,
    eval_budget: int,
    seed: int,
    descriptor_targets: Dict[str, float],
    rule_space: str,
    pop_hint: int,
    mutation_rate: float,
    crossover_rate: float,
    rule_weight_overrides: Optional[Dict[str, float]] = None,
    transition_mix: float = 0.7,
) -> SearchResult:
    started = time.time()
    pop, gens = _derive_pop_gen(eval_budget, pop_hint)
    generator = EvolutionaryTopologyGenerator(
        target_curve=scenario.target_curve,
        population_size=pop,
        generations=gens,
        mutation_rate=float(mutation_rate),
        crossover_rate=float(crossover_rate),
        genome_length=int(max(8, min(28, scenario.room_count + 4))),
        max_nodes=int(scenario.room_count),
        rule_space=rule_space,
        rule_weight_overrides=rule_weight_overrides,
        descriptor_targets=descriptor_targets,
        transition_mix=float(np.clip(transition_mix, 0.0, 1.0)),
        seed=seed,
    )
    graph = generator.evolve(directed_output=True)
    mission = networkx_to_mission_graph(graph)
    mission.sanitize()
    evaluator = TensionCurveEvaluator(scenario.target_curve, descriptor_targets=descriptor_targets)
    ev = evaluator.evaluate_graph(mission)

    stats = generator.get_statistics()
    generations_run = int(max(0, stats.get("generations_run", gens)))
    converged = bool(stats.get("converged", False))
    evaluations_used = int(pop * max(1, generations_run)) if converged else int(pop * (max(1, generations_run) + 1))

    return SearchResult(
        graph=graph,
        generation_time_sec=float(time.time() - started),
        fitness=float(ev.get("fitness", 0.0)),
        feasible=bool(ev.get("feasible", False)),
        constraint_violation=float(ev.get("constraint_violation", 1.0)),
        evaluations_used=max(1, evaluations_used),
    )


def _run_map_elites(
    scenario: SeedScenario,
    eval_budget: int,
    seed: int,
    descriptor_targets: Dict[str, float],
    rule_space: str,
    rule_weight_overrides: Optional[Dict[str, float]],
    archive_cells: int,
    init_random_frac: float,
    mutation_rate: float,
) -> SearchResult:
    started = time.time()
    pop, gens = _derive_pop_gen(eval_budget, pop_hint=max(16, int(math.sqrt(max(16, eval_budget)))))
    generator = EvolutionaryTopologyGenerator(
        target_curve=scenario.target_curve,
        population_size=pop,
        generations=gens,
        mutation_rate=float(np.clip(mutation_rate, 0.01, 0.95)),
        crossover_rate=0.68,
        genome_length=int(max(8, min(28, scenario.room_count + 4))),
        max_nodes=int(scenario.room_count),
        rule_space=rule_space,
        rule_weight_overrides=rule_weight_overrides,
        descriptor_targets=descriptor_targets,
        transition_mix=0.70,
        seed=seed,
        search_strategy="cvt_emitter",
        qd_archive_cells=max(32, int(archive_cells)),
        qd_init_random_fraction=float(np.clip(init_random_frac, 0.05, 0.95)),
        qd_emitter_mutation_rate=float(np.clip(mutation_rate, 0.01, 0.95)),
    )
    graph = generator.evolve(directed_output=True)
    mission = networkx_to_mission_graph(graph)
    mission.sanitize()
    evaluator = TensionCurveEvaluator(scenario.target_curve, descriptor_targets=descriptor_targets)
    ev = evaluator.evaluate_graph(mission)
    stats = generator.get_statistics()
    generations_run = int(max(0, stats.get("generations_run", gens)))
    evaluations_used = int(pop * max(1, generations_run))

    return SearchResult(
        graph=graph,
        generation_time_sec=float(time.time() - started),
        fitness=float(ev.get("fitness", 0.0)),
        feasible=bool(ev.get("feasible", False)),
        constraint_violation=float(ev.get("constraint_violation", 1.0)),
        evaluations_used=max(1, evaluations_used),
    )

def _paired_bootstrap_ci(deltas: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    if deltas.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, deltas.size, size=(n_boot, deltas.size))
    means = np.mean(deltas[idx], axis=1)
    return float(np.quantile(means, alpha / 2.0)), float(np.quantile(means, 1.0 - alpha / 2.0))


def _paired_sign_permutation_pvalue(deltas: np.ndarray, n_perm: int = 5000, seed: int = 0) -> float:
    if deltas.size == 0:
        return 1.0
    observed = abs(float(np.mean(deltas)))
    if observed <= 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    abs_d = np.abs(deltas)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, deltas.size))
    perm_means = np.mean(signs * abs_d[None, :], axis=1)
    return (1.0 + float(np.sum(np.abs(perm_means) >= observed))) / float(n_perm + 1)


def _benjamini_hochberg(p_values: Sequence[float]) -> List[float]:
    arr = np.asarray([float(p) for p in p_values], dtype=np.float64)
    n = int(arr.size)
    if n <= 0:
        return []
    order = np.argsort(arr)
    ranked = arr[order]
    q = np.zeros(n, dtype=np.float64)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = float(i + 1)
        raw = float(ranked[i]) * float(n) / rank
        prev = min(prev, raw)
        q[i] = prev
    out = np.empty(n, dtype=np.float64)
    out[order] = np.clip(q, 0.0, 1.0)
    return [float(v) for v in out.tolist()]


def _build_method_list(raw: str) -> List[str]:
    if not raw.strip():
        return list(METHODS_ALL)
    methods = [m.strip().upper() for m in raw.split(",") if m.strip()]
    invalid = [m for m in methods if m not in METHODS_ALL]
    if invalid:
        raise ValueError(f"Unsupported methods: {invalid}. Valid: {METHODS_ALL}")
    return methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run matched-budget Block-I topology benchmark.")
    parser.add_argument("--output", type=Path, default=Path("results") / "matched_budget")
    parser.add_argument("--data-root", type=Path, default=Path("Data") / "The Legend of Zelda")
    parser.add_argument("--reference-limit", type=int, default=None)
    parser.add_argument("--methods", type=str, default=",".join(METHODS_ALL))
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-budget", type=int, default=512)
    parser.add_argument("--population-hint", type=int, default=24)
    parser.add_argument("--min-rooms", type=int, default=8)
    parser.add_argument("--max-rooms", type=int, default=16)
    parser.add_argument("--room-count-bias", type=float, default=0.45)
    parser.add_argument("--no-align-rooms-to-reference", action="store_true")
    parser.add_argument("--room-budget-cap", type=int, default=42)
    parser.add_argument("--rule-space", type=str, default="full", choices=["core", "full"])
    parser.add_argument("--archive-cells", type=int, default=128)
    parser.add_argument("--map-elites-init-frac", type=float, default=0.35)
    parser.add_argument("--map-elites-mutation-rate", type=float, default=0.18)
    parser.add_argument("--calibrate-full", action="store_true")
    parser.add_argument("--calibration-iterations", type=int, default=4)
    parser.add_argument("--calibration-sample-size", type=int, default=12)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--kaggle-t4x2", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _build_seed_scenarios(seeds: Sequence[int], min_rooms: int, max_rooms: int, room_count_bias: float) -> Dict[int, SeedScenario]:
    scenarios: Dict[int, SeedScenario] = {}
    for seed in seeds:
        rng = np.random.default_rng(int(seed) + 101)
        rc = _sample_room_count(rng, min_rooms=min_rooms, max_rooms=max_rooms, room_count_bias=room_count_bias)
        scenarios[int(seed)] = SeedScenario(seed=int(seed), room_count=int(rc), target_curve=_target_curve_for_rooms(rng, int(rc)))
    return scenarios


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    if not args.verbose:
        # Grammar internals emit many expected repair warnings during search.
        # Keep console output focused on benchmark progress.
        logging.getLogger("src.generation.grammar").setLevel(logging.ERROR)
        logging.getLogger("src.data.vglc_utils").setLevel(logging.WARNING)

    if args.quick:
        args.num_samples = min(int(args.num_samples), 3)
        args.eval_budget = min(int(args.eval_budget), 128)
        args.population_hint = min(int(args.population_hint), 16)

    if args.kaggle_t4x2:
        args.num_samples = max(int(args.num_samples), 12)
        args.eval_budget = max(int(args.eval_budget), 768)
        args.population_hint = max(int(args.population_hint), 32)
        args.calibrate_full = True

    methods = _build_method_list(str(args.methods))
    seeds = [int(args.seed) + i for i in range(int(args.num_samples))]
    refs = load_vglc_reference_graphs(data_root=args.data_root, limit=args.reference_limit)
    if not refs:
        raise RuntimeError(f"No reference graphs loaded from {args.data_root}")

    descriptor_targets = _descriptor_targets(refs)
    min_rooms, max_rooms = _effective_room_budget(
        refs,
        min_rooms=int(args.min_rooms),
        max_rooms=int(args.max_rooms),
        align=not bool(args.no_align_rooms_to_reference),
        cap=int(args.room_budget_cap),
    )
    scenarios = _build_seed_scenarios(seeds, min_rooms=min_rooms, max_rooms=max_rooms, room_count_bias=float(args.room_count_bias))
    constraint_grammar = MissionGrammar(seed=int(args.seed) + 9001)

    full_rule_weights: Optional[Dict[str, float]] = None
    calibration_payload: Dict[str, Any] = {"enabled": bool(args.calibrate_full), "rule_weight_overrides": {}, "history": [], "best_gap": 0.0}
    if bool(args.calibrate_full):
        cal = calibrate_rule_weights_to_vglc(
            refs,
            seed=int(args.seed),
            iterations=int(max(1, args.calibration_iterations)),
            sample_size=int(max(4, args.calibration_sample_size)),
            population_size=max(16, int(args.population_hint)),
            generations=max(8, int(args.eval_budget) // max(8, int(args.population_hint))),
            min_rooms=min_rooms,
            max_rooms=max_rooms,
        )
        full_rule_weights = {str(k): float(v) for k, v in dict(cal.get("calibrated_rule_weights", {})).items() if np.isfinite(v)}
        calibration_payload.update({"rule_weight_overrides": full_rule_weights, "history": cal.get("history", []), "best_gap": float(cal.get("best_gap", 0.0))})

    ref_vec = np.stack([_descriptor_vec(g) for g in refs], axis=0)
    rows: List[Dict[str, Any]] = []
    gen_by_method: Dict[str, List[nx.Graph]] = {m: [] for m in methods}
    time_by_method: Dict[str, List[float]] = {m: [] for m in methods}

    logger.info("Matched-budget benchmark: methods=%s seeds=%d eval_budget=%d room_budget=[%d,%d]", methods, len(seeds), int(args.eval_budget), min_rooms, max_rooms)

    for method in methods:
        logger.info("Method=%s", method)
        for seed in seeds:
            scenario = scenarios[int(seed)]
            if method == "RANDOM":
                result = _run_random(
                    scenario=scenario,
                    eval_budget=int(args.eval_budget),
                    seed=int(seed) + 1000,
                    descriptor_targets=descriptor_targets,
                    rule_space=str(args.rule_space),
                    rule_weight_overrides=None,
                )
            elif method == "ES":
                result = _run_evolution(
                    scenario=scenario,
                    eval_budget=int(args.eval_budget),
                    seed=int(seed) + 2000,
                    descriptor_targets=descriptor_targets,
                    rule_space=str(args.rule_space),
                    pop_hint=int(args.population_hint),
                    mutation_rate=0.22,
                    crossover_rate=0.0,
                    rule_weight_overrides=None,
                    transition_mix=0.65,
                )
            elif method == "GA":
                result = _run_evolution(
                    scenario=scenario,
                    eval_budget=int(args.eval_budget),
                    seed=int(seed) + 3000,
                    descriptor_targets=descriptor_targets,
                    rule_space=str(args.rule_space),
                    pop_hint=int(args.population_hint),
                    mutation_rate=0.15,
                    crossover_rate=0.70,
                    rule_weight_overrides=None,
                    transition_mix=0.70,
                )
            elif method == "MAP_ELITES":
                result = _run_map_elites(
                    scenario=scenario,
                    eval_budget=int(args.eval_budget),
                    seed=int(seed) + 4000,
                    descriptor_targets=descriptor_targets,
                    rule_space=str(args.rule_space),
                    rule_weight_overrides=None,
                    archive_cells=int(args.archive_cells),
                    init_random_frac=float(args.map_elites_init_frac),
                    mutation_rate=float(args.map_elites_mutation_rate),
                )
            elif method == "FULL":
                result = _run_evolution(
                    scenario=scenario,
                    eval_budget=int(args.eval_budget),
                    seed=int(seed) + 5000,
                    descriptor_targets=descriptor_targets,
                    rule_space="full",
                    pop_hint=max(int(args.population_hint), 24),
                    mutation_rate=0.14,
                    crossover_rate=0.72,
                    rule_weight_overrides=full_rule_weights,
                    transition_mix=0.78,
                )
            else:
                raise ValueError(f"Unsupported method {method}")

            graph = result.graph
            desc = extract_graph_descriptor(graph, grammar=constraint_grammar)
            has_start_goal = float(desc.has_start and desc.has_goal)
            overall = float(0.25 * has_start_goal + 0.25 * float(desc.connected) + 0.25 * float(desc.path_exists) + 0.25 * float(desc.constraint_valid))
            dvec = np.array([desc.linearity, desc.leniency, desc.progression_complexity, desc.topology_complexity], dtype=np.float64)
            novelty = float(np.min(np.linalg.norm(ref_vec - dvec[None, :], axis=1)) / math.sqrt(4.0)) if ref_vec.size else 0.0
            ged = _nearest_graph_edit_distance(graph, refs, max_refs=24)

            rows.append(
                {
                    "method": method,
                    "seed": int(seed),
                    "room_count": int(scenario.room_count),
                    "target_curve_len": int(len(scenario.target_curve)),
                    "generation_time_sec": float(result.generation_time_sec),
                    "eval_budget": int(args.eval_budget),
                    "evaluations_used": int(result.evaluations_used),
                    "fitness": float(result.fitness),
                    "feasible_search": float(1.0 if result.feasible else 0.0),
                    "feasible_operational": float(
                        1.0
                        if (
                            bool(desc.has_start and desc.has_goal)
                            and bool(desc.connected)
                            and bool(desc.path_exists)
                            and bool(desc.constraint_valid)
                        )
                        else 0.0
                    ),
                    "constraint_violation": float(result.constraint_violation),
                    "overall_completeness": float(overall),
                    "constraint_valid": float(desc.constraint_valid),
                    "linearity": float(desc.linearity),
                    "leniency": float(desc.leniency),
                    "progression_complexity": float(desc.progression_complexity),
                    "topology_complexity": float(desc.topology_complexity),
                    "path_length": float(desc.path_length),
                    "num_nodes": float(desc.num_nodes),
                    "num_edges": float(desc.num_edges),
                    "repair_applied": float(desc.repair_applied),
                    "total_repairs": float(desc.total_repairs),
                    "generation_constraint_rejections": float(desc.generation_constraint_rejections),
                    "candidate_repairs_applied": float(desc.candidate_repairs_applied),
                    "novelty_vs_reference": float(novelty),
                    "graph_edit_distance": float(ged),
                }
            )
            gen_by_method[method].append(graph)
            time_by_method[method].append(float(result.generation_time_sec))

    raw_df = pd.DataFrame(rows)

    summary_rows: List[Dict[str, Any]] = []
    benchmark_payload_by_method: Dict[str, Dict[str, Any]] = {}
    for method in methods:
        sub = raw_df[raw_df["method"] == method]
        bench = run_block_i_benchmark(generated_graphs=gen_by_method[method], reference_graphs=refs, generation_times=time_by_method[method])
        payload = asdict(bench)
        benchmark_payload_by_method[method] = payload
        summary_rows.append(
            {
                "method": method,
                "n": int(len(sub)),
                "fitness": float(sub["fitness"].mean(skipna=True)),
                "feasible_search_rate": float(sub["feasible_search"].mean(skipna=True)),
                "feasible_operational_rate": float(sub["feasible_operational"].mean(skipna=True)),
                "overall_completeness": float(sub["overall_completeness"].mean(skipna=True)),
                "constraint_valid_rate": float(sub["constraint_valid"].mean(skipna=True)),
                "linearity": float(sub["linearity"].mean(skipna=True)),
                "leniency": float(sub["leniency"].mean(skipna=True)),
                "progression_complexity": float(sub["progression_complexity"].mean(skipna=True)),
                "topology_complexity": float(sub["topology_complexity"].mean(skipna=True)),
                "path_length": float(sub["path_length"].mean(skipna=True)),
                "num_nodes": float(sub["num_nodes"].mean(skipna=True)),
                "repair_rate": float(sub["repair_applied"].mean(skipna=True)),
                "mean_generation_constraint_rejections": float(sub["generation_constraint_rejections"].mean(skipna=True)),
                "mean_candidate_repairs_applied": float(sub["candidate_repairs_applied"].mean(skipna=True)),
                "novelty_vs_reference": float(sub["novelty_vs_reference"].mean(skipna=True)),
                "graph_edit_distance": float(sub["graph_edit_distance"].mean(skipna=True)),
                "generation_time_sec": float(sub["generation_time_sec"].mean(skipna=True)),
                "evaluations_used": float(sub["evaluations_used"].mean(skipna=True)),
                "fidelity_js_divergence": float(payload["reference_comparison"]["fidelity_js_divergence"]),
                "expressive_overlap_reference": float(payload["reference_comparison"]["expressive_overlap_reference"]),
                "coverage_linearity_leniency": float(payload["expressive_range"]["coverage_linearity_leniency"]),
                "coverage_progression_topology": float(payload["expressive_range"]["coverage_progression_topology"]),
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    metrics_for_sig = [
        "fitness", "feasible_search", "feasible_operational", "overall_completeness", "constraint_valid", "linearity", "leniency", "progression_complexity",
        "topology_complexity", "path_length", "num_nodes", "repair_applied", "generation_constraint_rejections",
        "candidate_repairs_applied", "novelty_vs_reference", "graph_edit_distance", "generation_time_sec",
    ]
    baseline = "FULL" if "FULL" in methods else methods[0]
    sig_rows: List[Dict[str, Any]] = []
    base = raw_df[raw_df["method"] == baseline]
    p_values: List[float] = []

    for method in methods:
        if method == baseline:
            continue
        merged = base.merge(raw_df[raw_df["method"] == method], on="seed", suffixes=("_base", "_cfg"))
        if merged.empty:
            continue
        for idx, metric in enumerate(metrics_for_sig):
            bcol, ccol = f"{metric}_base", f"{metric}_cfg"
            if bcol not in merged.columns or ccol not in merged.columns:
                continue
            deltas = (merged[ccol].astype(np.float64) - merged[bcol].astype(np.float64)).to_numpy(dtype=np.float64)
            deltas = deltas[np.isfinite(deltas)]
            if deltas.size == 0:
                continue
            mean_delta = float(np.mean(deltas))
            ci_low, ci_high = _paired_bootstrap_ci(deltas, n_boot=2000, alpha=0.05, seed=int(args.seed) + (17 * (idx + 1)))
            p_value = _paired_sign_permutation_pvalue(deltas, n_perm=4000, seed=int(args.seed) + (31 * (idx + 1)))
            std = float(np.std(deltas))
            effect = float(mean_delta / std) if std > 1e-9 else 0.0
            sig_rows.append({
                "method": method,
                "metric": metric,
                "n_pairs": int(deltas.size),
                "delta_mean_cfg_minus_full": mean_delta,
                "delta_ci_low": ci_low,
                "delta_ci_high": ci_high,
                "p_value": p_value,
                "effect_size_d": effect,
            })
            p_values.append(float(p_value))

    sig_df = pd.DataFrame(sig_rows)
    if not sig_df.empty:
        sig_df["p_value_bh_fdr"] = _benjamini_hochberg(p_values)
        sig_df["significant_fdr_0_05"] = sig_df["p_value_bh_fdr"] < 0.05

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "matched_budget_raw.csv"
    summary_path = out_dir / "matched_budget_summary.csv"
    sig_path = out_dir / "matched_budget_significance.csv"
    json_path = out_dir / "matched_budget_report.json"
    md_path = out_dir / "matched_budget_report.md"

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    sig_df.to_csv(sig_path, index=False)

    payload = {
        "methods": methods,
        "baseline": baseline,
        "seeds": seeds,
        "settings": {
            "eval_budget": int(args.eval_budget),
            "population_hint": int(args.population_hint),
            "min_rooms": int(min_rooms),
            "max_rooms": int(max_rooms),
            "room_count_bias": float(args.room_count_bias),
            "rule_space": str(args.rule_space),
            "archive_cells": int(args.archive_cells),
            "map_elites_init_frac": float(args.map_elites_init_frac),
            "map_elites_mutation_rate": float(args.map_elites_mutation_rate),
            "calibration": calibration_payload,
            "descriptor_targets": descriptor_targets,
        },
        "summary": summary_df.to_dict(orient="records"),
        "significance": sig_df.to_dict(orient="records"),
        "benchmark_payload_by_method": benchmark_payload_by_method,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _fmt(df: pd.DataFrame) -> str:
        try:
            return df.to_markdown(index=False)
        except (TypeError, ValueError, AttributeError):
            return df.to_string(index=False)

    lines = [
        "# Matched-Budget Block-I Benchmark",
        "",
        "## Methods",
    ]
    lines.extend([f"- `{m}`" for m in methods])
    lines.extend([
        "",
        "## Settings",
        "",
        f"- `eval_budget`: {int(args.eval_budget)}",
        f"- `num_samples`: {int(args.num_samples)}",
        f"- `room_budget`: [{int(min_rooms)}, {int(max_rooms)}]",
        f"- `rule_space`: {str(args.rule_space)}",
        f"- `baseline_for_significance`: {baseline}",
        "",
        "## Summary",
        "",
        _fmt(summary_df),
        "",
        "## Paired Significance",
        "",
        _fmt(sig_df) if not sig_df.empty else "_No paired significance rows available_",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")

    logger.info("Saved matched-budget outputs to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
