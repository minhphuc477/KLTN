"""CBS-based fitness function for MAP-Elites / H-MOLQD."""

from __future__ import annotations

from typing import Any, Dict

import networkx as nx
import numpy as np

from src.evaluation.search_benchmark_utils import normalized_confusion_ratio
from src.simulation.cognitive_bounded_search import CognitiveBoundedSearch
from src.simulation.validator import StateSpaceAStar, ZeldaLogicEnv


def _compute_graph_cognitive_proxy(
    graph: nx.Graph,
    target_confusion_ratio: float,
) -> Dict[str, float]:
    """
    Approximate cognitive navigation metrics directly from topology.

    This proxy is used when room-level semantic grids are unavailable
    (e.g., MAP-Elites graph feature extraction stage).
    """
    n = int(graph.number_of_nodes())
    e = int(graph.number_of_edges())
    if n <= 0:
        return {
            'fitness': -10.0,
            'solvable_astar': False,
            'solvable_cbs': False,
            'confusion_ratio': float('inf'),
            'normalized_confusion_ratio': float('inf'),
            'path_efficiency': 0.0,
            'room_entropy': 0.0,
            'confusion_index': 0.0,
            'astar_path_length': 0,
            'cbs_path_length': 0,
            'astar_states': 0,
            'is_proxy': 1.0,
        }

    def _has_semantic(node_data: Dict[str, Any], *tokens: str) -> bool:
        label_tokens = {
            str(part).strip().lower()
            for part in str(node_data.get('label', '')).split(',')
            if str(part).strip()
        }
        node_type = str(node_data.get('type', '')).strip().lower()
        semantics = set(label_tokens)
        if node_type:
            semantics.add(node_type)
        if bool(node_data.get('is_start', False)):
            semantics.add('start')
        if bool(node_data.get('is_goal', False) or node_data.get('has_triforce', False) or node_data.get('is_triforce', False)):
            semantics.add('goal')
            semantics.add('triforce')
        return any(token in semantics for token in tokens)

    # Cognitive navigation is a physical traversal descriptor, not a mission
    # dependency descriptor. Directed exports often encode one bidirectional
    # corridor as two directed edges; using the DiGraph directly makes trees
    # look cyclic and hides physical dead ends.
    physical = nx.Graph()
    physical.add_nodes_from(graph.nodes(data=True))
    for src, dst, data in graph.edges(data=True):
        if src == dst:
            continue
        physical.add_edge(src, dst, **dict(data))

    dg = graph
    n_physical = int(physical.number_of_nodes())
    e_physical = int(physical.number_of_edges())

    explicit_starts = [
        node for node, data in dg.nodes(data=True)
        if _has_semantic(data, 'start', 's')
    ]
    explicit_goals = [
        node for node, data in dg.nodes(data=True)
        if _has_semantic(data, 'goal', 'triforce', 't')
    ]

    # Prefer explicit semantics when present; fall back to degree extrema only for legacy graphs.
    start = (
        min(explicit_starts, key=str)
        if explicit_starts
        else min(physical.nodes(), key=lambda node: (physical.degree(node), str(node)))
    )
    goal = (
        min(explicit_goals, key=str)
        if explicit_goals
        else max(physical.nodes(), key=lambda node: (physical.degree(node), str(node)))
    )

    try:
        shortest = nx.shortest_path_length(physical, source=start, target=goal)
        solvable = True
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        shortest = 0
        solvable = False

    degrees = [deg for _node, deg in physical.degree()]
    mean_deg = float(np.mean(degrees)) if degrees else 0.0
    deg_std = float(np.std(degrees)) if degrees else 0.0

    # Dead-end pressure and loop pressure proxy human confusion tendencies.
    # Exclude both goal AND start nodes: start typically has degree 1 but
    # is not a navigational dead-end.
    dead_count = sum(
        1
        for n_id in physical.nodes()
        if int(physical.degree(n_id)) <= 1
        and not _has_semantic(dict(physical.nodes[n_id]), 'goal', 'triforce', 't')
        and not _has_semantic(dict(physical.nodes[n_id]), 'start', 's')
    )
    dead_ends = float(dead_count) / float(max(1, n_physical))
    components = nx.number_connected_components(physical) if n_physical > 0 else 0
    cycle_rank = max(0, e_physical - n_physical + components)
    loop_pressure = float(cycle_rank) / float(max(1, n_physical))
    branch_pressure = float(np.clip((mean_deg - 1.0) / 3.0, 0.0, 1.0))

    confusion_index = float(np.clip((0.45 * branch_pressure) + (0.35 * dead_ends) + (0.20 * loop_pressure), 0.0, 3.0))
    confusion_ratio = 1.0 + confusion_index

    if not solvable:
        fitness = -10.0
        path_efficiency = 0.0
        cbs_path_len = int(max(1, n))
        normalized_confusion = float('inf')
    else:
        # Assume bounded agent pays structural overhead over optimal shortest path.
        cbs_path_len = 0 if int(shortest) == 0 else int(max(1, round(shortest * (1.0 + confusion_index))))
        path_efficiency = float(shortest) / float(max(1, cbs_path_len))
        normalized_confusion = normalized_confusion_ratio(
            shortest,
            cbs_path_len,
            shortest,
            oracle_status="solved",
            candidate_success=True,
        )
        if not np.isfinite(normalized_confusion):
            normalized_confusion = 1.0
        target_normalized = max(0.0, float(target_confusion_ratio) - 1.0)
        cr_penalty = (normalized_confusion - target_normalized) ** 2
        fitness = 1.0 / (1.0 + cr_penalty)

    # Entropy proxy from degree spread (more uniform branching -> higher entropy).
    room_entropy = float(np.clip((deg_std / max(1.0, mean_deg + 1e-8)), 0.0, 1.0))

    return {
        'fitness': float(fitness),
        'solvable_astar': bool(solvable),
        'solvable_cbs': bool(solvable),
        'confusion_ratio': float(confusion_ratio),
        'normalized_confusion_ratio': float(normalized_confusion),
        'path_efficiency': float(path_efficiency),
        'room_entropy': float(room_entropy),
        'confusion_index': float(confusion_index),
        'astar_path_length': int(shortest),
        'cbs_path_length': int(cbs_path_len),
        'astar_states': int(max(0, n_physical + e_physical)),
        'is_proxy': 1.0,
    }


def compute_cbs_fitness(
    grid: Any,
    target_confusion_ratio: float = 2.0,
    persona: str = 'balanced',
    astar_timeout: int = 100000,
    cbs_timeout: int = 5000,
    seed: int = 42
) -> dict:
    """
    Compute CBS-based fitness for a generated dungeon.
    
    Returns dict with:
        fitness: float (higher = better, penalizes deviation from target CR)
        solvable_astar: bool
        solvable_cbs: bool
        confusion_ratio: float
        path_efficiency: float
        room_entropy: float
        confusion_index: float
    """
    # Graph mode: use cognitive-topology proxy when no semantic grid is available.
    if isinstance(grid, (nx.Graph, nx.DiGraph)):
        return _compute_graph_cognitive_proxy(
            graph=grid,
            target_confusion_ratio=float(target_confusion_ratio),
        )

    if not isinstance(grid, np.ndarray):
        raise TypeError(
            "compute_cbs_fitness expects a numpy grid or networkx graph, "
            f"got {type(grid).__name__}"
        )

    grid_for_astar = np.array(grid, dtype=np.int64, copy=True)
    grid_for_pcbs = np.array(grid, dtype=np.int64, copy=True)

    env_a = ZeldaLogicEnv(semantic_grid=grid_for_astar)
    solver_a = StateSpaceAStar(env_a, timeout=astar_timeout)
    success_a, path_a, states_a = solver_a.solve()
    if success_a and len(path_a) <= 0:
        return {
            'fitness': -10.0,
            'solvable_astar': False,
            'solvable_cbs': False,
            'confusion_ratio': float('inf'),
            'normalized_confusion_ratio': float('inf'),
            'path_efficiency': 0.0,
            'exploration_efficiency': 0.0,
            'room_entropy': 0.0,
            'confusion_index': 0.0,
            'astar_path_length': 0,
            'cbs_path_length': 0,
            'astar_states': states_a,
            'is_proxy': 0.0,
        }
    
    env_c = ZeldaLogicEnv(semantic_grid=grid_for_pcbs)
    cbs = CognitiveBoundedSearch(env_c, persona=persona, timeout=cbs_timeout, seed=seed)
    success_c, path_c, _states_c, metrics = cbs.solve()
    
    # Confusion Ratio
    astar_steps = len(path_a) if success_a and len(path_a) > 0 else 0
    cbs_steps = len(path_c) if success_c and len(path_c) > 0 else 0
    raw_cbs_steps = len(path_c) if len(path_c) > 0 else cbs_timeout
    confusion_ratio = (
        float(raw_cbs_steps) / float(astar_steps)
        if astar_steps > 0 and success_c
        else float('inf')
    )
    if env_a.start_pos is not None and env_a.goal_pos is not None:
        manhattan = abs(int(env_a.start_pos[0]) - int(env_a.goal_pos[0])) + abs(int(env_a.start_pos[1]) - int(env_a.goal_pos[1]))
    else:
        manhattan = 0
    normalized_confusion = normalized_confusion_ratio(
        astar_steps,
        cbs_steps,
        manhattan,
        oracle_status="solved" if success_a else "failed",
        candidate_success=bool(success_c),
    )
    
    # Fitness: penalize deviation from target CR
    # Also penalize unsolvable levels
    if not success_a:
        fitness = -10.0  # Unsolvable = very bad
    elif not success_c:
        fitness = -5.0   # Solvable but too hard for humans
    else:
        # Penalize normalized excess path length rather than raw CR.
        target_normalized = max(0.0, float(target_confusion_ratio) - 1.0)
        cr_penalty = (float(normalized_confusion) - target_normalized) ** 2
        fitness = 1.0 / (1.0 + cr_penalty)
    
    return {
        'fitness': fitness,
        'solvable_astar': success_a,
        'solvable_cbs': success_c,
        'confusion_ratio': confusion_ratio,
        'normalized_confusion_ratio': float(normalized_confusion) if np.isfinite(normalized_confusion) else float('inf'),
        'path_efficiency': (
            float(astar_steps) / float(max(1, cbs_steps))
            if success_a and success_c and cbs_steps > 0
            else 0.0
        ),
        'exploration_efficiency': getattr(metrics, 'exploration_efficiency', 0.0),
        'room_entropy': getattr(metrics, 'room_entropy', 0.0),
        'confusion_index': getattr(metrics, 'confusion_index', 0.0),
        'astar_path_length': len(path_a) if success_a else 0,
        'cbs_path_length': len(path_c),
        'astar_states': states_a,
        'is_proxy': 0.0,
    }


def cbs_loss_term(confusion_ratio: float, target: float = 2.0) -> float:
    """
    Loss term for training: penalizes deviation from target Confusion Ratio.
    
    For use in training loop:
        total_loss = recon_loss + alpha * cbs_loss_term(CR, target=2.0)
    
    Returns (target - CR)^2
    """
    return (target - confusion_ratio) ** 2
