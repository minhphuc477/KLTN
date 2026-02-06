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
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
from .validator import GameState, ZeldaLogicEnv, ACTION_DELTAS, SEMANTIC_PALETTE

logger = logging.getLogger(__name__)


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
        status = "✓" if self.success else "✗"
        return (f"{status} {self.name}: "
                f"Length={self.path_length}, "
                f"Explored={self.states_explored}, "
                f"Time={self.time_taken:.3f}s, "
                f"Optimality={self.optimality:.2f}×")


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
        for name, metrics in results.items():
            logger.info(str(metrics))
        
        # Determine winner (best optimality × speed trade-off)
        if successful:
            winner = min(successful, key=lambda m: m.optimality * m.time_taken)
            logger.info(f"🏆 Winner: {winner.name}")
        
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
                path_length=len(path),
                states_explored=states,
                time_taken=elapsed,
                optimality=1.0  # Will be updated later
            )
        except Exception as e:
            logger.error(f"A* failed: {e}")
            return SolverMetrics("A*", False, [], 0, 0, time.time() - start_time, float('inf'))
    
    def _run_bfs(self, start_state: GameState, max_time: float) -> SolverMetrics:
        """Run Breadth-First Search."""
        start_time = time.time()
        
        # BFS implementation
        queue = deque([(start_state, [start_state.position])])
        visited = {hash(start_state)}
        states_explored = 0
        
        while queue and (time.time() - start_time) < max_time:
            current_state, path = queue.popleft()
            states_explored += 1
            
            # Check goal
            if current_state.position == self.env.goal_pos:
                elapsed = time.time() - start_time
                return SolverMetrics(
                    name="BFS",
                    success=True,
                    path=path,
                    path_length=len(path),
                    states_explored=states_explored,
                    time_taken=elapsed,
                    optimality=1.0
                )
            
            # Expand neighbors
            for action, (dr, dc) in ACTION_DELTAS.items():
                new_r = current_state.position[0] + dr
                new_c = current_state.position[1] + dc
                
                if not (0 <= new_r < self.env.height and 0 <= new_c < self.env.width):
                    continue
                
                target_pos = (new_r, new_c)
                target_tile = self.env.grid[new_r, new_c]
                
                can_move, new_state = self._simple_move(current_state, target_pos, target_tile)
                
                if not can_move:
                    continue
                
                new_hash = hash(new_state)
                if new_hash not in visited:
                    visited.add(new_hash)
                    queue.append((new_state, path + [target_pos]))
        
        elapsed = time.time() - start_time
        return SolverMetrics("BFS", False, [], 0, states_explored, elapsed, float('inf'))
    
    def _run_dijkstra(self, start_state: GameState, max_time: float) -> SolverMetrics:
        """Run Dijkstra's algorithm (uniform cost search)."""
        start_time = time.time()
        
        # Priority queue: (cost, counter, state, path)
        open_set = [(0, 0, start_state, [start_state.position])]
        g_scores = {hash(start_state): 0}
        visited = set()
        counter = 1
        states_explored = 0
        
        while open_set and (time.time() - start_time) < max_time:
            cost, _, current_state, path = heapq.heappop(open_set)
            
            state_hash = hash(current_state)
            if state_hash in visited:
                continue
            
            visited.add(state_hash)
            states_explored += 1
            
            # Check goal
            if current_state.position == self.env.goal_pos:
                elapsed = time.time() - start_time
                return SolverMetrics(
                    name="Dijkstra",
                    success=True,
                    path=path,
                    path_length=len(path),
                    states_explored=states_explored,
                    time_taken=elapsed,
                    optimality=1.0
                )
            
            # Expand neighbors
            for action, (dr, dc) in ACTION_DELTAS.items():
                new_r = current_state.position[0] + dr
                new_c = current_state.position[1] + dc
                
                if not (0 <= new_r < self.env.height and 0 <= new_c < self.env.width):
                    continue
                
                target_pos = (new_r, new_c)
                target_tile = self.env.grid[new_r, new_c]
                
                can_move, new_state = self._simple_move(current_state, target_pos, target_tile)
                
                if not can_move:
                    continue
                
                new_hash = hash(new_state)
                if new_hash in visited:
                    continue
                
                new_cost = cost + 1  # Uniform cost
                
                if new_hash in g_scores and new_cost >= g_scores[new_hash]:
                    continue
                
                g_scores[new_hash] = new_cost
                heapq.heappush(open_set, (new_cost, counter, new_state, path + [target_pos]))
                counter += 1
        
        elapsed = time.time() - start_time
        return SolverMetrics("Dijkstra", False, [], 0, states_explored, elapsed, float('inf'))
    
    def _run_greedy(self, start_state: GameState, max_time: float) -> SolverMetrics:
        """Run Greedy Best-First Search (heuristic only)."""
        start_time = time.time()
        
        # Priority queue: (heuristic, counter, state, path)
        h_start = self._heuristic(start_state)
        open_set = [(h_start, 0, start_state, [start_state.position])]
        visited = set()
        counter = 1
        states_explored = 0
        
        while open_set and (time.time() - start_time) < max_time:
            _, _, current_state, path = heapq.heappop(open_set)
            
            state_hash = hash(current_state)
            if state_hash in visited:
                continue
            
            visited.add(state_hash)
            states_explored += 1
            
            # Check goal
            if current_state.position == self.env.goal_pos:
                elapsed = time.time() - start_time
                return SolverMetrics(
                    name="Greedy",
                    success=True,
                    path=path,
                    path_length=len(path),
                    states_explored=states_explored,
                    time_taken=elapsed,
                    optimality=1.0
                )
            
            # Expand neighbors
            for action, (dr, dc) in ACTION_DELTAS.items():
                new_r = current_state.position[0] + dr
                new_c = current_state.position[1] + dc
                
                if not (0 <= new_r < self.env.height and 0 <= new_c < self.env.width):
                    continue
                
                target_pos = (new_r, new_c)
                target_tile = self.env.grid[new_r, new_c]
                
                can_move, new_state = self._simple_move(current_state, target_pos, target_tile)
                
                if not can_move:
                    continue
                
                new_hash = hash(new_state)
                if new_hash in visited:
                    continue
                
                h = self._heuristic(new_state)
                heapq.heappush(open_set, (h, counter, new_state, path + [target_pos]))
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
        """Simplified movement check."""
        if target_tile in [SEMANTIC_PALETTE['WALL'], SEMANTIC_PALETTE['VOID']]:
            return False, state
        
        new_state = state.copy()
        new_state.position = target_pos
        
        # Handle pickups
        if target_tile == SEMANTIC_PALETTE['KEY_SMALL'] and target_pos not in state.collected_items:
            new_state.collected_items = state.collected_items | {target_pos}
            new_state.keys = state.keys + 1
        elif target_tile == SEMANTIC_PALETTE['KEY_BOSS'] and target_pos not in state.collected_items:
            new_state.collected_items = state.collected_items | {target_pos}
            new_state.has_boss_key = True
        
        # Handle doors
        if target_tile == SEMANTIC_PALETTE['DOOR_LOCKED']:
            if target_pos in state.opened_doors or state.keys > 0:
                new_state.keys = max(0, state.keys - 1)
                new_state.opened_doors = state.opened_doors | {target_pos}
                return True, new_state
            return False, state
        
        if target_tile == SEMANTIC_PALETTE['DOOR_BOSS']:
            if target_pos in state.opened_doors or state.has_boss_key:
                new_state.opened_doors = state.opened_doors | {target_pos}
                return True, new_state
            return False, state
        
        return True, new_state
