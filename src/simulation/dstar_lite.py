"""
D* Lite Implementation for Real-Time Replanning
===============================================

Based on: Koenig, S., & Likhachev, M. (2002). "D* Lite." AAAI Conference.

D* Lite is an incremental heuristic search algorithm that:
1. Efficiently replans when the environment changes
2. Maintains g(s) and rhs(s) values for all states
3. Uses priority queue with two-component keys
4. Achieves O(log N) update time vs O(N²) for full A* restart

Key Concepts:
- g(s): Current best cost from start to s
- rhs(s): One-step lookahead value (min over predecessors)
- Locally inconsistent: g(s) != rhs(s)
- Priority queue key: [min(g,rhs) + h, min(g,rhs)]
"""

import heapq
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from .validator import GameState, ACTION_DELTAS, SEMANTIC_PALETTE

logger = logging.getLogger(__name__)


@dataclass(order=True)
class DStarKey:
    """Priority queue key for D* Lite."""
    k1: float  # min(g, rhs) + h
    k2: float  # min(g, rhs)
    state_hash: int = field(compare=False)
    state: GameState = field(compare=False)


class DStarLiteSolver:
    """
    D* Lite incremental search for real-time replanning.
    
    Features:
    - Efficient replanning when environment changes
    - Maintains consistency constraints
    - Incremental updates instead of full restart
    
    Performance:
    - Initial search: Similar to A* (O(N log N))
    - Replan after change: O(M log N) where M = affected states
    """
    
    def __init__(self, env, heuristic_mode: str = "balanced"):
        """
        Initialize D* Lite solver.
        
        Args:
            env: ZeldaLogicEnv instance
            heuristic_mode: Heuristic type (balanced/speedrunner/completionist)
        """
        self.env = env
        self.heuristic_mode = heuristic_mode
        
        # Core D* Lite data structures
        self.g_scores: Dict[int, float] = {}  # g(s) values
        self.rhs_scores: Dict[int, float] = {}  # rhs(s) values
        self.open_set: List[DStarKey] = []  # Priority queue
        self.open_set_hashes: Set[int] = set()  # Fast membership check
        
        # Environment change detection
        self.last_opened_doors: Set[Tuple[int, int]] = set()
        self.last_pushed_blocks: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
        
        # Statistics
        self.replans_count = 0
        self.states_updated = 0
        
        # Current path
        self.current_path: List[Tuple[int, int]] = []
        self.path_index = 0
    
    def calculate_key(self, state: GameState, state_hash: int) -> DStarKey:
        """
        Calculate priority queue key for state.
        
        Key = [min(g(s), rhs(s)) + h(s), min(g(s), rhs(s))]
        
        Args:
            state: Game state
            state_hash: Hash of state
            
        Returns:
            DStarKey with k1 and k2 values
        """
        g = self.g_scores.get(state_hash, float('inf'))
        rhs = self.rhs_scores.get(state_hash, float('inf'))
        min_val = min(g, rhs)
        h = self._heuristic(state)
        
        return DStarKey(k1=min_val + h, k2=min_val, state_hash=state_hash, state=state)
    
    def _heuristic(self, state: GameState) -> float:
        """Manhattan distance heuristic."""
        if self.env.goal_pos is None:
            return float('inf')
        
        pos = state.position
        goal = self.env.goal_pos
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    def update_vertex(self, state: GameState, state_hash: int):
        """
        Update vertex consistency (Algorithm 2 from paper).
        
        If state is locally inconsistent (g != rhs), add to priority queue.
        Otherwise, remove from priority queue.
        """
        # Start state has rhs = 0
        if state.position == self.env.start_pos:
            self.rhs_scores[state_hash] = 0
        else:
            # Compute rhs(s) = min over predecessors: g(s') + c(s', s)
            min_rhs = float('inf')
            
            # Get all predecessor states (neighbors that can reach this state)
            for action, (dr, dc) in ACTION_DELTAS.items():
                pred_r = state.position[0] - dr
                pred_c = state.position[1] - dc
                
                if not (0 <= pred_r < self.env.height and 0 <= pred_c < self.env.width):
                    continue
                
                # Create predecessor state (simplified - assume same inventory)
                pred_state = state.copy()
                pred_state.position = (pred_r, pred_c)
                pred_hash = hash(pred_state)
                
                pred_g = self.g_scores.get(pred_hash, float('inf'))
                if pred_g < float('inf'):
                    cost = self._get_edge_cost(pred_state, state)
                    min_rhs = min(min_rhs, pred_g + cost)
            
            self.rhs_scores[state_hash] = min_rhs
        
        # Remove from OPEN if present
        if state_hash in self.open_set_hashes:
            # Mark for removal (lazy deletion)
            self.open_set_hashes.discard(state_hash)
        
        # Add to OPEN if locally inconsistent
        g = self.g_scores.get(state_hash, float('inf'))
        rhs = self.rhs_scores.get(state_hash, float('inf'))
        
        if g != rhs:
            key = self.calculate_key(state, state_hash)
            heapq.heappush(self.open_set, key)
            self.open_set_hashes.add(state_hash)
            self.states_updated += 1
    
    def compute_shortest_path(self) -> bool:
        """
        Main D* Lite algorithm (Algorithm 1 from paper).
        
        Expands states until goal is consistent and has lowest key.
        
        Returns:
            True if path found, False otherwise
        """
        if self.env.goal_pos is None:
            return False
        
        goal_state = GameState(position=self.env.goal_pos)
        goal_hash = hash(goal_state)
        
        iterations = 0
        max_iterations = 100000
        
        while iterations < max_iterations:
            # Clean lazy-deleted entries from open set
            while self.open_set and self.open_set[0].state_hash not in self.open_set_hashes:
                heapq.heappop(self.open_set)
            
            if not self.open_set:
                break
            
            # Check termination condition
            top_key = self.open_set[0]
            goal_key = self.calculate_key(goal_state, goal_hash)
            
            g_goal = self.g_scores.get(goal_hash, float('inf'))
            rhs_goal = self.rhs_scores.get(goal_hash, float('inf'))
            
            if top_key >= goal_key and rhs_goal == g_goal:
                # Goal is consistent and has lowest key
                return True
            
            # Pop minimum key state
            current_key = heapq.heappop(self.open_set)
            state_hash = current_key.state_hash
            state = current_key.state
            self.open_set_hashes.discard(state_hash)
            
            g = self.g_scores.get(state_hash, float('inf'))
            rhs = self.rhs_scores.get(state_hash, float('inf'))
            
            if g > rhs:
                # Locally overconsistent - update g
                self.g_scores[state_hash] = rhs
                
                # Update all successors
                for successor_state in self._get_successors(state):
                    successor_hash = hash(successor_state)
                    self.update_vertex(successor_state, successor_hash)
            else:
                # Locally underconsistent - set g to infinity
                self.g_scores[state_hash] = float('inf')
                
                # Update vertex and all successors
                self.update_vertex(state, state_hash)
                for successor_state in self._get_successors(state):
                    successor_hash = hash(successor_state)
                    self.update_vertex(successor_state, successor_hash)
            
            iterations += 1
        
        logger.warning(f"D* Lite: Max iterations ({max_iterations}) reached")
        return False
    
    def solve(self, start_state: GameState) -> Tuple[bool, List[Tuple[int, int]], int]:
        """
        Initial planning phase.
        
        Args:
            start_state: Initial game state
            
        Returns:
            (success, path, states_explored)
        """
        self.g_scores.clear()
        self.rhs_scores.clear()
        self.open_set.clear()
        self.open_set_hashes.clear()
        
        # Initialize start
        start_hash = hash(start_state)
        self.rhs_scores[start_hash] = 0
        self.update_vertex(start_state, start_hash)
        
        # Run search
        success = self.compute_shortest_path()
        
        if not success:
            return False, [], len(self.g_scores)
        
        # Extract path
        path = self._extract_path(start_state)
        self.current_path = path
        self.path_index = 0
        
        # Store initial state for change detection
        self.last_opened_doors = start_state.opened_doors.copy()
        self.last_pushed_blocks = start_state.pushed_blocks.copy()
        
        return True, path, len(self.g_scores)
    
    def replan(self, current_state: GameState) -> Tuple[bool, List[Tuple[int, int]], int]:
        """
        Replan after environment change.
        
        Args:
            current_state: Current game state (after change)
            
        Returns:
            (success, new_path, states_updated)
        """
        self.replans_count += 1
        self.states_updated = 0
        
        # Detect changes
        new_doors = current_state.opened_doors - self.last_opened_doors
        new_blocks = current_state.pushed_blocks - self.last_pushed_blocks
        
        logger.info(f"D* Lite Replan #{self.replans_count}: {len(new_doors)} doors, {len(new_blocks)} blocks")
        
        # Update affected vertices
        for door_pos in new_doors:
            # Find all states that could be affected by this door opening
            # (simplified: just update neighbors)
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                neighbor_pos = (door_pos[0] + dr, door_pos[1] + dc)
                if 0 <= neighbor_pos[0] < self.env.height and 0 <= neighbor_pos[1] < self.env.width:
                    neighbor_state = current_state.copy()
                    neighbor_state.position = neighbor_pos
                    self.update_vertex(neighbor_state, hash(neighbor_state))
        
        # Recompute shortest path
        success = self.compute_shortest_path()
        
        if not success:
            return False, [], self.states_updated
        
        # Extract new path
        path = self._extract_path(current_state)
        self.current_path = path
        
        # Update last state
        self.last_opened_doors = current_state.opened_doors.copy()
        self.last_pushed_blocks = current_state.pushed_blocks.copy()
        
        return True, path, self.states_updated
    
    def needs_replan(self, current_state: GameState) -> bool:
        """Check if replanning is needed."""
        return (current_state.opened_doors != self.last_opened_doors or
                current_state.pushed_blocks != self.last_pushed_blocks)
    
    def _extract_path(self, start_state: GameState) -> List[Tuple[int, int]]:
        """Extract path from g_scores by greedy descent."""
        path = [start_state.position]
        current = start_state
        visited = set()
        
        for _ in range(1000):  # Max path length
            if current.position == self.env.goal_pos:
                break
            
            current_hash = hash(current)
            if current_hash in visited:
                break
            visited.add(current_hash)
            
            # Find best successor (lowest g-score)
            best_successor = None
            best_g = float('inf')
            
            for successor in self._get_successors(current):
                successor_hash = hash(successor)
                g = self.g_scores.get(successor_hash, float('inf'))
                if g < best_g:
                    best_g = g
                    best_successor = successor
            
            if best_successor is None:
                break
            
            current = best_successor
            path.append(current.position)
        
        return path
    
    def _get_successors(self, state: GameState) -> List[GameState]:
        """Get all valid successor states."""
        successors = []
        
        for action, (dr, dc) in ACTION_DELTAS.items():
            new_r = state.position[0] + dr
            new_c = state.position[1] + dc
            
            if not (0 <= new_r < self.env.height and 0 <= new_c < self.env.width):
                continue
            
            target_tile = self.env.grid[new_r, new_c]
            
            # Simplified movement check (full logic in validator.py)
            if target_tile not in [SEMANTIC_PALETTE['WALL'], SEMANTIC_PALETTE['VOID']]:
                new_state = state.copy()
                new_state.position = (new_r, new_c)
                successors.append(new_state)
        
        return successors
    
    def _get_edge_cost(self, from_state: GameState, to_state: GameState) -> float:
        """Calculate edge cost between states."""
        # Use diagonal cost if moving diagonally
        dr = abs(to_state.position[0] - from_state.position[0])
        dc = abs(to_state.position[1] - from_state.position[1])
        
        if dr + dc == 2:  # Diagonal
            return 1.414
        else:  # Cardinal
            return 1.0
