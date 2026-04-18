"""CBS-based fitness function for MAP-Elites / H-MOLQD."""

from __future__ import annotations

from typing import Any, Dict

import networkx as nx
import numpy as np

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

    # Ensure directed reachability is judged on directed graph semantics.
    dg = graph if graph.is_directed() else nx.DiGraph(graph)

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
        else min(dg.nodes(), key=lambda node: (dg.in_degree(node), str(node)))
    )
    goal = (
        min(explicit_goals, key=str)
        if explicit_goals
        else max(dg.nodes(), key=lambda node: (dg.out_degree(node), str(node)))
    )

    try:
        shortest = nx.shortest_path_length(dg, source=start, target=goal)
        solvable = True
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        shortest = 0
        solvable = False

    degrees = [deg for _node, deg in dg.degree()]
    mean_deg = float(np.mean(degrees)) if degrees else 0.0
    deg_std = float(np.std(degrees)) if degrees else 0.0

    # Dead-end pressure and loop pressure proxy human confusion tendencies.
    dead_ends = float(sum(1 for _n, deg in dg.out_degree() if deg == 0)) / float(max(1, n))
    loop_pressure = max(0.0, float(e - n + 1)) / float(max(1, n))
    branch_pressure = float(np.clip((mean_deg - 1.0) / 3.0, 0.0, 1.0))

    confusion_index = float(np.clip((0.45 * branch_pressure) + (0.35 * dead_ends) + (0.20 * loop_pressure), 0.0, 3.0))
    confusion_ratio = 1.0 + confusion_index

    if not solvable:
        fitness = -10.0
        path_efficiency = 0.0
        cbs_path_len = int(max(1, n))
    else:
        # Assume bounded agent pays structural overhead over optimal shortest path.
        cbs_path_len = int(max(1, round(shortest * (1.0 + confusion_index))))
        path_efficiency = float(shortest) / float(max(1, cbs_path_len))
        cr_penalty = (confusion_ratio - float(target_confusion_ratio)) ** 2
        fitness = 1.0 / (1.0 + cr_penalty)

    # Entropy proxy from degree spread (more uniform branching -> higher entropy).
    room_entropy = float(np.clip((deg_std / max(1.0, mean_deg + 1e-8)), 0.0, 1.0))

    return {
        'fitness': float(fitness),
        'solvable_astar': bool(solvable),
        'solvable_cbs': bool(solvable),
        'confusion_ratio': float(confusion_ratio),
        'path_efficiency': float(path_efficiency),
        'room_entropy': float(room_entropy),
        'confusion_index': float(confusion_index),
        'astar_path_length': int(shortest),
        'cbs_path_length': int(cbs_path_len),
        'astar_states': int(max(0, n + e)),
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

    env_a = ZeldaLogicEnv(semantic_grid=grid)
    solver_a = StateSpaceAStar(env_a, timeout=astar_timeout)
    success_a, path_a, states_a = solver_a.solve()
    
    env_c = ZeldaLogicEnv(semantic_grid=grid.copy())
    cbs = CognitiveBoundedSearch(env_c, persona=persona, timeout=cbs_timeout, seed=seed)
    success_c, path_c, _states_c, metrics = cbs.solve()
    
    # Confusion Ratio
    astar_steps = len(path_a) if success_a and len(path_a) > 0 else float('inf')
    cbs_steps = len(path_c) if len(path_c) > 0 else cbs_timeout
    confusion_ratio = cbs_steps / astar_steps if astar_steps > 0 and astar_steps != float('inf') else float('inf')
    
    # Fitness: penalize deviation from target CR
    # Also penalize unsolvable levels
    if not success_a:
        fitness = -10.0  # Unsolvable = very bad
    elif not success_c:
        fitness = -5.0   # Solvable but too hard for humans
    else:
        # Gaussian penalty around target CR
        cr_penalty = (confusion_ratio - target_confusion_ratio) ** 2
        fitness = 1.0 / (1.0 + cr_penalty)
    
    return {
        'fitness': fitness,
        'solvable_astar': success_a,
        'solvable_cbs': success_c,
        'confusion_ratio': confusion_ratio,
        'path_efficiency': getattr(metrics, 'exploration_efficiency', 0.0),
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
