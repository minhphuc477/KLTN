"""CBS-based fitness function for MAP-Elites / H-MOLQD."""

import numpy as np
from src.simulation.cognitive_bounded_search import CognitiveBoundedSearch
from src.simulation.validator import StateSpaceAStar, ZeldaLogicEnv


def compute_cbs_fitness(
    grid: np.ndarray,
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
    env_a = ZeldaLogicEnv(semantic_grid=grid)
    solver_a = StateSpaceAStar(env_a, timeout=astar_timeout)
    success_a, path_a, states_a = solver_a.solve()
    
    env_c = ZeldaLogicEnv(semantic_grid=grid.copy())
    cbs = CognitiveBoundedSearch(env_c, persona=persona, timeout=cbs_timeout, seed=seed)
    success_c, path_c, states_c, metrics = cbs.solve()
    
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
    }


def cbs_loss_term(confusion_ratio: float, target: float = 2.0) -> float:
    """
    Loss term for training: penalizes deviation from target Confusion Ratio.
    
    For use in training loop:
        total_loss = recon_loss + alpha * cbs_loss_term(CR, target=2.0)
    
    Returns (target - CR)^2
    """
    return (target - confusion_ratio) ** 2
