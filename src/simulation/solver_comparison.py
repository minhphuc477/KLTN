"""
Solver Comparison Mode - Compare Multiple Search Algorithms
==========================================================

Compare 4 classical search algorithms side-by-side:
1. A* - Optimal with heuristic (f = g + h)
2. BFS - Breadth-first (optimal for unit costs)
3. Dijkstra - Uniform cost search (optimal)
4. Greedy Best-First - Fast but not optimal (f = h only)

Research:
- Russell & Norvig "Artificial Intelligence: A Modern Approach" Ch. 3
- Hart, Nilsson, Raphael (1968) - Original A* paper

Educational Value:
- Visualize trade-offs between algorithms
- Understand impact of heuristics
- Compare exploration patterns
"""

import heapq
import time
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
from .validator import GameState, ZeldaLogicEnv, ACTION_DELTAS, game_state_key

logger = logging.getLogger(__name__)


def _reconstruct_state_path(
    end_key: Any,
    parents: Dict[Any, Optional[Any]],
    positions: Dict[Any, Tuple[int, int]],
) -> List[Tuple[int, int]]:
    path: List[Tuple[int, int]] = []
    current_key: Optional[Any] = end_key
    while current_key is not None:
        path.append(positions[current_key])
        current_key = parents[current_key]
    path.reverse()
    return path


@dataclass
class SolverMetrics:
    """Performance metrics for a solver."""
    name: str
    success: bool
    path: List[Tuple[int, int]]
    path_length: int
    states_explored: int
    time_taken: float  # seconds
    optimality: float  # 1.0 = optimal, >1.0 = suboptimal
    
    def __str__(self):
        status = "[OK]" if self.success else "[FAIL]"
        return (f"{status} {self.name}: "
                f"Length={self.path_length}, "
                f"Explored={self.states_explored}, "
                f"Time={self.time_taken:.3f}s, "
                f"Optimality={self.optimality:.2f}x")


class SolverComparison:
    """
    Run multiple search algorithms in parallel and compare results.
    
    Features:
    - Side-by-side execution of 4 algorithms
    - Fair comparison (same environment, same start/goal)
    - Detailed metrics (time, nodes, optimality)
    - Visual split-screen rendering
    """
    
    def __init__(self, env: ZeldaLogicEnv):
        """Initialize with environment."""
        self.env = env
    
    def compare_all(self, start_state: GameState, max_time: float = 30.0) -> Dict[str, SolverMetrics]:
        """
        Run all 4 solvers and collect metrics.
        
        Args:
            start_state: Initial game state
            max_time: Maximum time per solver (seconds)
            
        Returns:
            Dict mapping solver name to metrics
        """
        results = {}
        
        # Run each solver
        logger.info("=== Starting Solver Comparison ===")
        
        results['A*'] = self._run_astar(start_state, max_time)
        results['BFS'] = self._run_bfs(start_state, max_time)
        results['Dijkstra'] = self._run_dijkstra(start_state, max_time)
        results['Greedy'] = self._run_greedy(start_state, max_time)
        
        # Compute optimality scores (relative to best path)
        successful = [r for r in results.values() if r.success]
        if successful:
            optimal_length = min(r.path_length for r in successful)
            for metrics in results.values():
                if metrics.success:
                    metrics.optimality = metrics.path_length / optimal_length
                else:
                    metrics.optimality = float('inf')
        
        # Log summary
        logger.info("=== Comparison Results ===")
        for _name, metrics in results.items():
            logger.info(str(metrics))
        
        # Determine winner (best optimality x speed trade-off)
        if successful:
            winner = min(successful, key=lambda m: m.optimality * m.time_taken)
            logger.info(f"[WIN] Winner: {winner.name}")
        
        return results
    
    def _run_astar(self, start_state: GameState, max_time: float) -> SolverMetrics:
        """Run A* search."""
        from .validator import StateSpaceAStar
        
        start_time = time.time()
        solver = StateSpaceAStar(self.env)
        
        try:
            success, path, states = solver.solve(start_state)
            elapsed = time.time() - start_time
            
            return SolverMetrics(
                name="A*",
                success=success,
                path=path,
                path_length=max(0, len(path) - 1),
                states_explored=states,
                time_taken=elapsed,
                optimality=1.0  # Will be updated later
            )
        except (AttributeError, RuntimeError, ValueError, TypeError) as e:
            logger.error(f"A* failed: {e}")
            return SolverMetrics("A*", False, [], 0, 0, time.time() - start_time, float('inf'))
    
    def _run_bfs(self, start_state: GameState, max_time: float) -> SolverMetrics:
        """Run Breadth-First Search."""
        start_time = time.time()
        
        # BFS implementation
        start_key = game_state_key(start_state)
        queue = deque([start_state])
        visited = {start_key}
        parents: Dict[Any, Optional[Any]] = {start_key: None}
        positions = {start_key: start_state.position}
        states_explored = 0
        
        while queue and (time.time() - start_time) < max_time:
            current_state = queue.popleft()
            current_key = game_state_key(current_state)
            states_explored += 1
            
            # Check goal
            if current_state.position == self.env.goal_pos:
                elapsed = time.time() - start_time
                path = _reconstruct_state_path(current_key, parents, positions)
                return SolverMetrics(
                    name="BFS",
                    success=True,
                    path=path,
                    path_length=max(0, len(path) - 1),
                    states_explored=states_explored,
                    time_taken=elapsed,
                    optimality=1.0
                )
            
            # Expand neighbors
            for _action, (dr, dc) in ACTION_DELTAS.items():
                new_r = current_state.position[0] + dr
                new_c = current_state.position[1] + dc
                
                if not (0 <= new_r < self.env.height and 0 <= new_c < self.env.width):
                    continue
                
                target_pos = (new_r, new_c)
                target_tile = self.env.grid[new_r, new_c]
                
                can_move, new_state = self._simple_move(current_state, target_pos, target_tile)
                
                if not can_move:
                    continue
                
                new_key = game_state_key(new_state)
                if new_key not in visited:
                    visited.add(new_key)
                    parents[new_key] = current_key
                    positions[new_key] = new_state.position
                    queue.append(new_state)
        
        elapsed = time.time() - start_time
        return SolverMetrics("BFS", False, [], 0, states_explored, elapsed, float('inf'))
    
    def _run_dijkstra(self, start_state: GameState, max_time: float) -> SolverMetrics:
        """Run Dijkstra's algorithm (uniform cost search)."""
        start_time = time.time()
        
        # Priority queue: (cost, counter, state)
        start_key = game_state_key(start_state)
        open_set = [(0, 0, start_state)]
        g_scores = {start_key: 0}
        parents: Dict[Any, Optional[Any]] = {start_key: None}
        positions = {start_key: start_state.position}
        visited = set()
        counter = 1
        states_explored = 0
        
        while open_set and (time.time() - start_time) < max_time:
            cost, _, current_state = heapq.heappop(open_set)
            
            state_key = game_state_key(current_state)
            if state_key in visited:
                continue
            
            visited.add(state_key)
            states_explored += 1
            
            # Check goal
            if current_state.position == self.env.goal_pos:
                elapsed = time.time() - start_time
                path = _reconstruct_state_path(state_key, parents, positions)
                return SolverMetrics(
                    name="Dijkstra",
                    success=True,
                    path=path,
                    path_length=max(0, len(path) - 1),
                    states_explored=states_explored,
                    time_taken=elapsed,
                    optimality=1.0
                )
            
            # Expand neighbors
            for _action, (dr, dc) in ACTION_DELTAS.items():
                new_r = current_state.position[0] + dr
                new_c = current_state.position[1] + dc
                
                if not (0 <= new_r < self.env.height and 0 <= new_c < self.env.width):
                    continue
                
                target_pos = (new_r, new_c)
                target_tile = self.env.grid[new_r, new_c]
                
                can_move, new_state = self._simple_move(current_state, target_pos, target_tile)
                
                if not can_move:
                    continue
                
                new_key = game_state_key(new_state)
                if new_key in visited:
                    continue
                
                new_cost = cost + 1  # Uniform cost
                
                if new_key in g_scores and new_cost >= g_scores[new_key]:
                    continue
                
                g_scores[new_key] = new_cost
                parents[new_key] = state_key
                positions[new_key] = new_state.position
                heapq.heappush(open_set, (new_cost, counter, new_state))
                counter += 1
        
        elapsed = time.time() - start_time
        return SolverMetrics("Dijkstra", False, [], 0, states_explored, elapsed, float('inf'))
    
    def _run_greedy(self, start_state: GameState, max_time: float) -> SolverMetrics:
        """Run Greedy Best-First Search (heuristic only)."""
        start_time = time.time()
        
        # Priority queue: (heuristic, counter, state)
        h_start = self._heuristic(start_state)
        start_key = game_state_key(start_state)
        open_set = [(h_start, 0, start_state)]
        parents: Dict[Any, Optional[Any]] = {start_key: None}
        positions = {start_key: start_state.position}
        visited = set()
        counter = 1
        states_explored = 0
        
        while open_set and (time.time() - start_time) < max_time:
            _, _, current_state = heapq.heappop(open_set)
            
            state_key = game_state_key(current_state)
            if state_key in visited:
                continue
            
            visited.add(state_key)
            states_explored += 1
            
            # Check goal
            if current_state.position == self.env.goal_pos:
                elapsed = time.time() - start_time
                path = _reconstruct_state_path(state_key, parents, positions)
                return SolverMetrics(
                    name="Greedy",
                    success=True,
                    path=path,
                    path_length=max(0, len(path) - 1),
                    states_explored=states_explored,
                    time_taken=elapsed,
                    optimality=1.0
                )
            
            # Expand neighbors
            for _action, (dr, dc) in ACTION_DELTAS.items():
                new_r = current_state.position[0] + dr
                new_c = current_state.position[1] + dc
                
                if not (0 <= new_r < self.env.height and 0 <= new_c < self.env.width):
                    continue
                
                target_pos = (new_r, new_c)
                target_tile = self.env.grid[new_r, new_c]
                
                can_move, new_state = self._simple_move(current_state, target_pos, target_tile)
                
                if not can_move:
                    continue
                
                new_key = game_state_key(new_state)
                if new_key in visited:
                    continue
                
                h = self._heuristic(new_state)
                if new_key not in parents:
                    parents[new_key] = state_key
                    positions[new_key] = new_state.position
                heapq.heappush(open_set, (h, counter, new_state))
                counter += 1
        
        elapsed = time.time() - start_time
        return SolverMetrics("Greedy", False, [], 0, states_explored, elapsed, float('inf'))
    
    def _heuristic(self, state: GameState) -> float:
        """Manhattan distance heuristic."""
        if self.env.goal_pos is None:
            return 0
        
        pos = state.position
        goal = self.env.goal_pos
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    def _simple_move(self, state: GameState, target_pos: Tuple[int, int],
                     target_tile: int) -> Tuple[bool, GameState]:
        """Apply the same full game-state transition used by the hard oracle."""
        return self.env.try_move_pure(state, target_pos, int(target_tile))
