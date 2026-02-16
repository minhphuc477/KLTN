"""
State-Space DFS/IDDFS Implementation for Zelda Dungeon Solving
==============================================================

Depth-First Search (DFS) and Iterative Deepening DFS (IDDFS) for complete 
graph exploration in state-space graphs. Useful for feasibility checking 
in small subgraphs and validating puzzle solvability.

Key Features:
- Complete exploration guarantee with IDDFS
- Memory-efficient deep searches with iterative version
- State-space aware: tracks (position, inventory) tuples
- Cycle detection to avoid infinite loops
- Configurable depth limits for large dungeons

Scientific Basis:
- Korf, R. E. (1985). "Depth-First Iterative-Deepening: An Optimal Admissible 
  Tree Search." Artificial Intelligence, 27(1), 97-109.
- IDDFS combines BFS completeness with DFS memory efficiency: O(bd) space, 
  O(b^d) time
"""

import logging
from typing import Dict, List, Tuple, Optional, Set, FrozenSet
from dataclasses import dataclass
from collections import deque

from .validator import (
    GameState, ZeldaLogicEnv, SolverOptions, SolverDiagnostics,
    SEMANTIC_PALETTE, ACTION_DELTAS, CARDINAL_COST, DIAGONAL_COST,
    WALKABLE_IDS, BLOCKING_IDS, PICKUP_IDS, PUSHABLE_IDS, WATER_IDS
)

logger = logging.getLogger(__name__)


@dataclass
class DFSMetrics:
    """Performance metrics for DFS search."""
    max_depth_reached: int = 0
    nodes_at_depth: Dict[int, int] = None
    backtrack_count: int = 0
    cycle_detections: int = 0
    
    def __post_init__(self):
        if self.nodes_at_depth is None:
            self.nodes_at_depth = {}


class StateSpaceDFS:
    """
    Depth-First Search solver for Zelda state-space graphs.
    
    Features:
    - State-space DFS: tracks (position, inventory) tuples
    - Iterative (stack-based) and recursive implementations
    - Depth limiting to prevent stack overflow
    - Cycle detection with visited set
    - Branching factor handling for multiple door choices
    
    Performance:
    - Time: O(b^d) where b=branching factor, d=solution depth
    - Space: O(bd) for iterative, O(d) for recursive (excl. visited set)
    - Best for: Small dungeons, feasibility checking, depth-limited searches
    - Worst case: Exponential time on large dungeons without depth limit
    
    Integration:
    - Used by fitness function to verify local connectivity
    - Validates puzzle solvability in isolated room clusters
    - Complements A* for completeness checks
    """
    
    def __init__(self, env: ZeldaLogicEnv, timeout: int = 50000, 
                 max_depth: int = 500, allow_diagonals: bool = False,
                 use_iddfs: bool = True):
        """
        Initialize DFS solver.
        
        Args:
            env: ZeldaLogicEnv instance
            timeout: Maximum states to explore (safeguard)
            max_depth: Maximum search depth (default 500)
            allow_diagonals: Enable diagonal movement
            use_iddfs: Use Iterative Deepening DFS (recommended for large dungeons)
        """
        self.env = env
        self.timeout = timeout
        self.max_depth = max_depth
        self.allow_diagonals = allow_diagonals
        self.use_iddfs = use_iddfs
        
        # Statistics
        self.metrics = DFSMetrics()
        self.states_explored = 0
        
        # Read-only grid reference
        self.grid = self.env.original_grid
        self.height, self.width = self.grid.shape
        
    def solve(self) -> Tuple[bool, List[Tuple[int, int]], int]:
        """
        Find a solution path using DFS or IDDFS.
        
        Returns:
            success: Whether a solution was found
            path: List of positions visited (if found)
            states_explored: Number of states explored
        """
        self.env.reset()
        
        if self.env.goal_pos is None or self.env.start_pos is None:
            return False, [], 0
        
        if self.use_iddfs:
            return self._solve_iddfs()
        else:
            return self._solve_iterative_dfs()
    
    def _solve_iddfs(self) -> Tuple[bool, List[Tuple[int, int]], int]:
        """
        Iterative Deepening DFS.
        
        Progressively increases depth limit, combining BFS completeness 
        with DFS memory efficiency.
        
        Returns:
            (success, path, states_explored)
        """
        logger.debug(f'StateSpaceDFS: Starting IDDFS (max_depth={self.max_depth})')
        
        # Start with shallow depth, double each iteration
        depth_limit = 10
        total_states = 0
        
        while depth_limit <= self.max_depth and total_states < self.timeout:
            # Reset metrics for this iteration
            self.states_explored = 0
            self.metrics = DFSMetrics()
            
            logger.debug(f'IDDFS: Trying depth_limit={depth_limit}')
            
            # Run depth-limited DFS
            success, path = self._dfs_recursive(
                self.env.state.copy(),
                visited=set(),
                path=[self.env.start_pos],
                depth=0,
                depth_limit=depth_limit
            )
            
            total_states += self.states_explored
            
            if success:
                logger.debug(f'IDDFS succeeded at depth {len(path)}, explored {total_states} states')
                return True, path, total_states
            
            # Double depth limit for next iteration
            depth_limit *= 2
        
        logger.debug(f'IDDFS exhausted: max_depth={self.max_depth}, states={total_states}')
        return False, [], total_states
    
    def _dfs_recursive(self, state: GameState, visited: Set[int], 
                      path: List[Tuple[int, int]], depth: int, 
                      depth_limit: int) -> Tuple[bool, List[Tuple[int, int]]]:
        """
        Recursive DFS implementation with depth limiting.
        
        Args:
            state: Current game state
            visited: Set of visited state hashes
            path: Current path (positions)
            depth: Current depth in search tree
            depth_limit: Maximum depth for this iteration
            
        Returns:
            (success, path) tuple
        """
        # Check termination conditions
        if depth >= depth_limit:
            return False, []
        
        if self.states_explored >= self.timeout:
            return False, []
        
        state_hash = hash(state)
        
        # Cycle detection
        if state_hash in visited:
            self.metrics.cycle_detections += 1
            return False, []
        
        visited.add(state_hash)
        self.states_explored += 1
        
        # Track depth metrics
        self.metrics.max_depth_reached = max(self.metrics.max_depth_reached, depth)
        self.metrics.nodes_at_depth[depth] = \
            self.metrics.nodes_at_depth.get(depth, 0) + 1
        
        # Goal check
        if state.position == self.env.goal_pos:
            return True, path
        
        # Generate successors
        successors = self._get_successors(state)
        
        # Try each successor
        for next_state in successors:
            next_hash = hash(next_state)
            if next_hash in visited:
                continue
            
            new_path = path + [next_state.position]
            success, result_path = self._dfs_recursive(
                next_state, visited, new_path, depth + 1, depth_limit
            )
            
            if success:
                return True, result_path
            else:
                # Backtrack
                self.metrics.backtrack_count += 1
        
        # Dead end - backtrack
        visited.remove(state_hash)
        return False, []
    
    def _solve_iterative_dfs(self) -> Tuple[bool, List[Tuple[int, int]], int]:
        """
        Iterative (stack-based) DFS implementation.
        
        Uses explicit stack instead of recursion to avoid stack overflow
        on deep searches.
        
        Returns:
            (success, path, states_explored)
        """
        logger.debug(f'StateSpaceDFS: Starting iterative DFS (max_depth={self.max_depth})')
        
        start_state = self.env.state.copy()
        
        # Stack: (state, path, depth)
        stack = [(start_state, [start_state.position], 0)]
        visited = set()
        
        while stack and self.states_explored < self.timeout:
            state, path, depth = stack.pop()
            
            # Depth limit check
            if depth >= self.max_depth:
                continue
            
            state_hash = hash(state)
            
            # Cycle detection
            if state_hash in visited:
                self.metrics.cycle_detections += 1
                continue
            
            visited.add(state_hash)
            self.states_explored += 1
            
            # Track depth metrics
            self.metrics.max_depth_reached = max(self.metrics.max_depth_reached, depth)
            self.metrics.nodes_at_depth[depth] = \
                self.metrics.nodes_at_depth.get(depth, 0) + 1
            
            # Goal check
            if state.position == self.env.goal_pos:
                logger.debug(f'Iterative DFS succeeded: depth={depth}, states={self.states_explored}')
                return True, path, self.states_explored
            
            # Generate and push successors (reverse order for left-to-right exploration)
            successors = self._get_successors(state)
            for next_state in reversed(successors):
                next_hash = hash(next_state)
                if next_hash not in visited:
                    new_path = path + [next_state.position]
                    stack.append((next_state, new_path, depth + 1))
        
        logger.debug(f'Iterative DFS exhausted: states={self.states_explored}')
        return False, [], self.states_explored
    
    def _get_successors(self, state: GameState) -> List[GameState]:
        """
        Generate all valid successor states from current state.
        
        Handles:
        - 4-directional or 8-directional movement
        - Door unlocking with keys/bombs/boss key
        - Item pickup and inventory updates
        - Block pushing
        - Water crossing with ladder
        
        Args:
            state: Current game state
            
        Returns:
            List of valid successor GameStates
        """
        successors = []
        curr_r, curr_c = state.position
        
        # Cardinal directions (UP, DOWN, LEFT, RIGHT)
        cardinal_deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Diagonal directions (if enabled)
        diagonal_deltas = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        # Try cardinal moves
        for dr, dc in cardinal_deltas:
            new_r, new_c = curr_r + dr, curr_c + dc
            
            if not (0 <= new_r < self.height and 0 <= new_c < self.width):
                continue
            
            target_tile = self.grid[new_r, new_c]
            can_move, new_state = self._try_move(state, (new_r, new_c), target_tile)
            
            if can_move:
                successors.append(new_state)
        
        # Try diagonal moves (if enabled)
        if self.allow_diagonals:
            for dr, dc in diagonal_deltas:
                new_r, new_c = curr_r + dr, curr_c + dc
                
                if not (0 <= new_r < self.height and 0 <= new_c < self.width):
                    continue
                
                # Check corner cutting (can't move diagonally through walls)
                adj_r_tile = self.grid[curr_r + dr, curr_c]
                adj_c_tile = self.grid[curr_r, curr_c + dc]
                
                if adj_r_tile in BLOCKING_IDS or adj_c_tile in BLOCKING_IDS:
                    continue
                
                target_tile = self.grid[new_r, new_c]
                can_move, new_state = self._try_move(state, (new_r, new_c), target_tile)
                
                if can_move:
                    successors.append(new_state)
        
        return successors
    
    def _try_move(self, state: GameState, target_pos: Tuple[int, int], 
                  target_tile: int) -> Tuple[bool, GameState]:
        """
        Attempt to move to target position (pure state-based).
        
        COMPLETE IMPLEMENTATION matching StateSpaceAStar._try_move_pure.
        Handles ALL game mechanics: keys, doors, blocks, items, bombs, water/element tiles.
        
        Args:
            state: Current game state
            target_pos: Target position (row, col)
            target_tile: Tile ID at target position
            
        Returns:
            (can_move, new_state) tuple
        """
        # Blocking tiles - cannot pass
        if target_tile in BLOCKING_IDS:
            return False, state
        
        new_state = state.copy()
        new_state.position = target_pos
        
        # Check if this door was already opened (in state)
        if target_pos in state.opened_doors:
            # Door is open, can pass freely
            return True, new_state
        
        # Check if this item was already collected (in state)
        if target_pos in state.collected_items:
            # Item already collected, treat as floor
            return True, new_state
        
        # CRITICAL FIX: Check if a block was pushed FROM this position
        # If so, the position is now empty (treat as floor)
        for (from_pos, to_pos) in state.pushed_blocks:
            if from_pos == target_pos:
                # Block was pushed away from here - position is now empty floor
                return True, new_state
        
        # CRITICAL FIX 2: Check if a block was pushed TO this position
        # If so, we need to handle it as a BLOCK (pushable), not as floor
        for (from_pos, to_pos) in state.pushed_blocks:
            if to_pos == target_pos:
                # There's a pushed block here! Need to try pushing it further
                # Calculate direction of push
                dr = target_pos[0] - state.position[0]
                dc = target_pos[1] - state.position[1]
                push_dest_r = target_pos[0] + dr
                push_dest_c = target_pos[1] + dc
                
                # Check bounds
                if not (0 <= push_dest_r < self.height and 0 <= push_dest_c < self.width):
                    return False, state  # Can't push off map
                
                # Check destination - but also check if another block is there!
                push_dest_tile = self.grid[push_dest_r, push_dest_c]
                dest_has_block = any(tp == (push_dest_r, push_dest_c) for (_, tp) in state.pushed_blocks)
                
                if push_dest_tile in WALKABLE_IDS and not dest_has_block:
                    # Can push - update pushed_blocks
                    # CRITICAL: Preserve ORIGINAL from_pos to keep track of empty positions!
                    new_pushed = set()
                    for (fp, tp) in state.pushed_blocks:
                        if tp == target_pos:
                            # Keep original from_pos, update destination to new position
                            new_pushed.add((from_pos, (push_dest_r, push_dest_c)))
                        else:
                            new_pushed.add((fp, tp))
                    new_state.pushed_blocks = new_pushed
                    return True, new_state
                else:
                    return False, state  # Can't push
        
        # Walkable tiles - free movement
        if target_tile in WALKABLE_IDS:
            # Handle item pickup (add to collected_items)
            if target_tile in PICKUP_IDS:
                new_state.collected_items = state.collected_items | {target_pos}
                
                if target_tile == SEMANTIC_PALETTE['KEY_SMALL']:
                    new_state.keys = state.keys + 1
                elif target_tile == SEMANTIC_PALETTE['KEY_BOSS']:
                    new_state.has_boss_key = True
                elif target_tile == SEMANTIC_PALETTE['KEY_ITEM']:
                    new_state.has_item = True
                    new_state.bomb_count = state.bomb_count + 4  # Consumable bombs
                elif target_tile == SEMANTIC_PALETTE['ITEM_MINOR']:
                    # ITEM_MINOR represents bomb pickups in VGLC Zelda dungeons
                    new_state.bomb_count = state.bomb_count + 4  # Consumable: add 4 bombs
            
            return True, new_state
        
        # Conditional tiles - require inventory items
        if target_tile == SEMANTIC_PALETTE['DOOR_LOCKED']:
            if state.keys > 0:
                new_state.keys = state.keys - 1
                new_state.opened_doors = state.opened_doors | {target_pos}
                return True, new_state
            return False, state
        
        if target_tile == SEMANTIC_PALETTE['DOOR_BOMB']:
            if state.bomb_count > 0:
                new_state.bomb_count = state.bomb_count - 1  # Consume one bomb
                new_state.opened_doors = state.opened_doors | {target_pos}
                return True, new_state
            return False, state
        
        if target_tile == SEMANTIC_PALETTE['DOOR_BOSS']:
            if state.has_boss_key:
                new_state.opened_doors = state.opened_doors | {target_pos}
                return True, new_state
            return False, state
        
        if target_tile == SEMANTIC_PALETTE['DOOR_PUZZLE']:
            # Puzzle doors can be passed (simplified)
            return True, new_state
        
        if target_tile == SEMANTIC_PALETTE['DOOR_OPEN']:
            return True, new_state
        
        # DOOR_SOFT - One-way/soft-locked door
        if target_tile == SEMANTIC_PALETTE['DOOR_SOFT']:
            return True, new_state
        
        # TRIFORCE - goal tile
        if target_tile == SEMANTIC_PALETTE['TRIFORCE']:
            return True, new_state
        
        # BOSS - Boss enemy tile (must fight boss)
        # Walkable like regular enemies
        if target_tile == SEMANTIC_PALETTE['BOSS']:
            return True, new_state
        
        # PUZZLE - Puzzle element tile (interact to solve)
        # Walkable - player interacts with puzzle to progress
        if target_tile == SEMANTIC_PALETTE['PUZZLE']:
            return True, new_state
        
        # PUSH BLOCK LOGIC (Zelda mechanic)
        # Agent can push blocks if there's empty space behind the block
        if target_tile in PUSHABLE_IDS:
            # Calculate direction of push (from agent's current position to target)
            dr = target_pos[0] - state.position[0]
            dc = target_pos[1] - state.position[1]
            
            # Determine where block would land if pushed
            push_dest_r = target_pos[0] + dr
            push_dest_c = target_pos[1] + dc
            
            # Check if push destination is in bounds
            if not (0 <= push_dest_r < self.height and 0 <= push_dest_c < self.width):
                return False, state  # Can't push block off map
            
            # Check if push destination is empty
            # CRITICAL FIX: Also check if another pushed block occupies the destination
            push_dest_tile = self.grid[push_dest_r, push_dest_c]
            dest_has_block = any(tp == (push_dest_r, push_dest_c) for (_, tp) in state.pushed_blocks)
            
            if push_dest_tile in WALKABLE_IDS and not dest_has_block:
                # Block can be pushed - agent moves onto block's original position
                # Track pushed block state
                new_state.pushed_blocks = state.pushed_blocks | {(target_pos, (push_dest_r, push_dest_c))}
                return True, new_state
            else:
                # Can't push block (destination is blocked or has another pushed block)
                return False, state
        
        # WATER/LADDER LOGIC (Zelda mechanic)
        # ELEMENT tiles (water/lava) require KEY_ITEM (Ladder) to cross
        if target_tile in WATER_IDS:
            if state.has_item:  # has_item represents KEY_ITEM (Ladder)
                # Can cross water with ladder
                return True, new_state
            else:
                # Can't cross water without ladder
                return False, state
        
        # Default case: treat unknown tiles as walkable
        return True, new_state
    
    def solve_with_diagnostics(self) -> Tuple[bool, List[Tuple[int, int]], SolverDiagnostics]:
        """
        Solve with detailed diagnostics.
        
        Returns:
            (success, path, diagnostics)
        """
        import time
        start_time = time.perf_counter()
        
        success, path, states = self.solve()
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        diag = SolverDiagnostics(
            success=success,
            states_explored=states,
            states_pruned_dominated=self.metrics.cycle_detections,
            max_queue_size=0,  # DFS uses stack, not priority queue
            time_taken_ms=elapsed_ms,
            failure_reason="" if success else "DFS exhausted search space",
            path_length=len(path) if success else 0,
            final_inventory=None
        )
        
        return success, path, diag


# ==========================================
# STANDALONE TESTING
# ==========================================

if __name__ == "__main__":
    import numpy as np
    
    # Configure logging
    logging.basicConfig(level=logging.DEBUG, 
                       format='%(levelname)s - %(message)s')
    
    # Create simple test dungeon
    from src.core.definitions import SEMANTIC_PALETTE
    
    # 10x10 test dungeon with key and locked door
    test_grid = np.full((10, 10), SEMANTIC_PALETTE['FLOOR'], dtype=np.int64)
    
    # Add walls
    test_grid[0, :] = SEMANTIC_PALETTE['WALL']
    test_grid[-1, :] = SEMANTIC_PALETTE['WALL']
    test_grid[:, 0] = SEMANTIC_PALETTE['WALL']
    test_grid[:, -1] = SEMANTIC_PALETTE['WALL']
    
    # Add start and goal
    test_grid[1, 1] = SEMANTIC_PALETTE['START']
    test_grid[8, 8] = SEMANTIC_PALETTE['TRIFORCE']
    
    # Add key and locked door
    test_grid[1, 5] = SEMANTIC_PALETTE['KEY_SMALL']
    test_grid[5, 1] = SEMANTIC_PALETTE['DOOR_LOCKED']
    
    logger.info('Testing StateSpaceDFS on simple dungeon...')
    
    # Test IDDFS
    env = ZeldaLogicEnv(test_grid)
    solver = StateSpaceDFS(env, use_iddfs=True, max_depth=100)
    success, path, states = solver.solve()
    
    logger.info(f'IDDFS Result: success={success}, path_len={len(path)}, states={states}')
    logger.info(f'Metrics: max_depth={solver.metrics.max_depth_reached}, '
               f'cycles={solver.metrics.cycle_detections}, '
               f'backtracks={solver.metrics.backtrack_count}')
    
    # Test iterative DFS
    env2 = ZeldaLogicEnv(test_grid)
    solver2 = StateSpaceDFS(env2, use_iddfs=False, max_depth=100)
    success2, path2, states2 = solver2.solve()
    
    logger.info(f'Iterative DFS Result: success={success2}, path_len={len(path2)}, states={states2}')
