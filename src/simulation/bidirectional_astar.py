"""
Bidirectional A* Implementation for Zelda State-Space Search
============================================================

Meet-in-the-middle search for reversible, uniform-cost grid movement.

Inventory-changing Zelda mechanics are not generally reversible from a single
guessed goal inventory. Those maps are delegated to the canonical full-state
A* solver so this comparison solver never certifies an invalid reverse path.

Key Features:
- Dual frontier expansion (forward and backward)
- Collision detection when frontiers meet
- Path reconstruction from both directions
    - Reversible-grid fast path
    - Canonical A* fallback for stateful or directed mechanics

Scientific Basis:
- Pohl, I. (1971). "Bi-directional Search." Machine Intelligence, 6, 127-140.
- Kaindl, H., & Kainz, G. (1997). "Bidirectional Heuristic Search Reconsidered."  
  Journal of Artificial Intelligence Research, 7, 283-317.

Critical limitation:
- A single backward state cannot enumerate every predecessor inventory for
  consumable resources, movable blocks, staged puzzles, or directed warps.
"""

import heapq
import logging
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from .validator import (
    GameState, ZeldaLogicEnv, SEMANTIC_PALETTE, WALKABLE_IDS, BLOCKING_IDS,
    CONDITIONAL_IDS, PUSHABLE_IDS, WATER_IDS, PICKUP_IDS, CARDINAL_COST,
    DIAGONAL_COST, game_state_key,
)

logger = logging.getLogger(__name__)


@dataclass
class SearchNode:
    """Node in bidirectional search frontier."""
    state: GameState
    g_score: float  # Cost from start/goal
    f_score: float  # g + h
    parent: Optional['SearchNode'] = None
    path: List[Tuple[int, int]] = field(default_factory=list)


class BidirectionalAStar:
    """
    Bidirectional search with a correctness-preserving full-state fallback.
    
    Features:
    - Forward search from start toward goal
    - Backward search from goal toward start on reversible grids
    - Collision detection when frontiers meet
    - Path reconstruction by concatenating forward and backward paths
    - Lower-bound certification before accepting a first frontier meeting
    
    Bidirectional search can reduce expansions on suitable reversible problems,
    but no domain-independent speedup is claimed; frontier balance, obstacles,
    heuristic quality, and optimality proof overhead determine actual cost.
    
    Stateful Zelda transitions are delegated to ``StateSpaceAStar`` because a
    guessed goal inventory does not define a complete reverse transition graph.
    
    Integration:
    - Provides path length baseline for expressive range analysis
    - Useful for dungeons with clear start-to-goal corridor structure
    - Complements A* for long-distance pathfinding benchmarks
    """
    
    def __init__(self, env: ZeldaLogicEnv, timeout: int = 100000,
                 allow_diagonals: bool = False, heuristic_mode: str = "balanced"):
        """
        Initialize Bidirectional A* solver.
        
        Args:
            env: ZeldaLogicEnv instance
            timeout: Maximum states to explore (combined forward + backward)
            allow_diagonals: Enable diagonal movement
            heuristic_mode: Heuristic type (balanced/speedrunner/completionist)
        """
        self.env = env
        self.timeout = timeout
        self.allow_diagonals = allow_diagonals
        self.heuristic_mode = heuristic_mode
        
        # Read-only grid reference
        self.grid = self.env.original_grid
        self.height, self.width = self.grid.shape
        
        # Frontier tracking
        self.forward_open: List[Tuple] = []  # Priority queue
        self.backward_open: List[Tuple] = []  # Priority queue
        
        self.forward_closed: Dict[Any, SearchNode] = {}  # state key -> node
        self.backward_closed: Dict[Any, SearchNode] = {}  # state key -> node
        # Position indexes to avoid O(|closed|) scans during collision checks.
        self.forward_closed_by_pos: Dict[Tuple[int, int], List[SearchNode]] = {}
        self.backward_closed_by_pos: Dict[Tuple[int, int], List[SearchNode]] = {}
        
        self.forward_g_scores: Dict[Any, float] = {}
        self.backward_g_scores: Dict[Any, float] = {}
        
        # Statistics
        self.states_explored = 0
        self.collision_checks = 0
        self.meeting_point: Optional[Tuple[int, int]] = None
        self.used_fallback = False
    
    def solve(self) -> Tuple[bool, List[Tuple[int, int]], int]:
        """
        Find solution using bidirectional A*.
        
        Returns:
            success: Whether a solution was found
            path: List of positions visited
            states_explored: Number of states explored
        """
        self.env.reset()
        
        if self.env.goal_pos is None or self.env.start_pos is None:
            return False, [], 0

        if self._requires_canonical_fallback():
            logger.debug(
                "BidirectionalAStar: stateful or directed mechanics detected; "
                "using canonical full-state A*"
            )
            return self._fallback_to_astar()
        
        logger.debug('BidirectionalAStar: Starting search')
        self.used_fallback = False
        self.forward_closed.clear()
        self.backward_closed.clear()
        self.forward_closed_by_pos.clear()
        self.backward_closed_by_pos.clear()
        self.forward_open.clear()
        self.backward_open.clear()
        self.forward_g_scores.clear()
        self.backward_g_scores.clear()
        self.states_explored = 0
        self.collision_checks = 0
        self.meeting_point = None
        
        # Initialize forward search from start
        start_state = self.env.state.copy()
        start_node = SearchNode(
            state=start_state,
            g_score=0,
            f_score=self._heuristic_forward(start_state),
            path=[start_state.position]
        )
        
        start_hash = game_state_key(start_state)
        self.forward_g_scores[start_hash] = 0
        heapq.heappush(self.forward_open, (start_node.f_score, 0, start_hash, start_node))
        
        # Initialize backward search from goal
        # CRITICAL: Guess goal inventory (assume agent collected everything)
        goal_state = self._create_goal_state()
        goal_node = SearchNode(
            state=goal_state,
            g_score=0,
            f_score=self._heuristic_backward(goal_state),
            path=[goal_state.position]
        )
        
        goal_hash = game_state_key(goal_state)
        self.backward_g_scores[goal_hash] = 0
        heapq.heappush(self.backward_open, (goal_node.f_score, 0, goal_hash, goal_node))
        
        # Alternating expansion
        counter = 1
        iterations = 0
        max_iterations = max(10_000, int(self.timeout) * 20)
        best_meeting_cost = float('inf')
        best_forward_node = None
        best_backward_node = None
        
        while (self.forward_open or self.backward_open) and \
              self.states_explored < self.timeout:
            iterations += 1
            if iterations > max_iterations:
                logger.warning(
                    "BidirectionalAStar: iteration budget reached (%d), aborting",
                    max_iterations,
                )
                break

            if (
                best_forward_node is not None
                and best_backward_node is not None
                and best_meeting_cost <= self._frontier_lower_bound() + 1e-6
            ):
                path = self._reconstruct_path(best_forward_node, best_backward_node)
                logger.debug(
                    "Bidirectional A* certified incumbent path: cost=%.3f, states=%d",
                    best_meeting_cost,
                    self.states_explored,
                )
                return True, path, self.states_explored
            
            # Alternate between forward and backward expansion
            if len(self.forward_open) <= len(self.backward_open) and self.forward_open:
                # Expand forward
                success, meeting_node_f, meeting_node_b = self._expand_forward(counter)
                if success:
                    path = self._reconstruct_path(meeting_node_f, meeting_node_b)
                    cost = meeting_node_f.g_score + meeting_node_b.g_score
                    if cost < best_meeting_cost:
                        best_meeting_cost = cost
                        best_forward_node = meeting_node_f
                        best_backward_node = meeting_node_b
                    if self._candidate_is_provably_optimal(path, candidate_cost=cost):
                        logger.debug(
                            "Bidirectional A* certified a shortest path: path_len=%d, states=%d",
                            len(path),
                            self.states_explored,
                        )
                        return True, path, self.states_explored
                    logger.debug(
                        "Bidirectional meeting cost %.3f is not yet certified; continuing",
                        cost,
                    )
                    counter += 1
                    continue
                
                # Update best meeting point
                if meeting_node_f and meeting_node_b:
                    cost = meeting_node_f.g_score + meeting_node_b.g_score
                    if cost < best_meeting_cost:
                        best_meeting_cost = cost
                        best_forward_node = meeting_node_f
                        best_backward_node = meeting_node_b
                
                counter += 1
            
            elif self.backward_open:
                # Expand backward
                success, meeting_node_b, meeting_node_f = self._expand_backward(counter)
                if success:
                    path = self._reconstruct_path(meeting_node_f, meeting_node_b)
                    cost = meeting_node_f.g_score + meeting_node_b.g_score
                    if cost < best_meeting_cost:
                        best_meeting_cost = cost
                        best_forward_node = meeting_node_f
                        best_backward_node = meeting_node_b
                    if self._candidate_is_provably_optimal(path, candidate_cost=cost):
                        logger.debug(
                            "Bidirectional A* certified a shortest path: path_len=%d, states=%d",
                            len(path),
                            self.states_explored,
                        )
                        return True, path, self.states_explored
                    logger.debug(
                        "Bidirectional meeting cost %.3f is not yet certified; continuing",
                        cost,
                    )
                    counter += 1
                    continue
                
                # Update best meeting point
                if meeting_node_f and meeting_node_b:
                    cost = meeting_node_f.g_score + meeting_node_b.g_score
                    if cost < best_meeting_cost:
                        best_meeting_cost = cost
                        best_forward_node = meeting_node_f
                        best_backward_node = meeting_node_b
                
                counter += 1
        
        # If we have a meeting point (even if not optimal), use it
        if best_forward_node and best_backward_node:
            path = self._reconstruct_path(best_forward_node, best_backward_node)
            if self._candidate_is_provably_optimal(path):
                return True, path, self.states_explored
            logger.debug(
                "Bidirectional candidate was not lower-bound optimal; using canonical A*"
            )
            return self._fallback_to_astar()

        logger.warning(
            'Bidirectional A* exhausted/aborted without meet point (states=%d); '
            'falling back to StateSpaceAStar for correctness',
            self.states_explored,
        )
        return self._fallback_to_astar()

    def _requires_canonical_fallback(self) -> bool:
        """Return whether the environment contains non-reversible mechanics."""
        stateful_ids = (
            set(CONDITIONAL_IDS)
            | set(PICKUP_IDS)
            | set(PUSHABLE_IDS)
            | set(WATER_IDS)
            | {
                SEMANTIC_PALETTE['DOOR_SOFT'],
                SEMANTIC_PALETTE['ENEMY'],
                SEMANTIC_PALETTE['BOSS'],
                SEMANTIC_PALETTE['PUZZLE'],
            }
        )
        present_ids = {int(value) for value in self.grid.reshape(-1)}
        return bool(
            present_ids.intersection(stateful_ids)
            or getattr(self.env, "graph", None)
            or getattr(self.env, "_puzzle_stage_lookup", None)
            or getattr(self.env, "block_underlay_tiles", None)
        )

    def _candidate_is_provably_optimal(
        self,
        path: List[Tuple[int, int]],
        *,
        candidate_cost: Optional[float] = None,
    ) -> bool:
        """Certify a reversible-grid candidate against frontier lower bounds."""
        if not path or path[0] != self.env.start_pos or path[-1] != self.env.goal_pos:
            return False
        candidate = float(candidate_cost if candidate_cost is not None else self._path_cost(path))
        geometric_lower_bound = self._grid_distance(self.env.start_pos, self.env.goal_pos)
        if candidate <= geometric_lower_bound + 1e-6:
            return True
        frontier_lower_bound = self._frontier_lower_bound()
        return candidate <= frontier_lower_bound + 1e-6

    def _path_cost(self, path: List[Tuple[int, int]]) -> float:
        cost = 0.0
        for a, b in zip(path[:-1], path[1:]):
            cost += self._step_cost(a, b)
        return float(cost)

    def _step_cost(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        dr = abs(int(a[0]) - int(b[0]))
        dc = abs(int(a[1]) - int(b[1]))
        return float(DIAGONAL_COST if dr == 1 and dc == 1 else CARDINAL_COST)

    def _frontier_lower_bound(self) -> float:
        """
        Return the strongest admissible front-to-end lower bound.

        Each direction's minimum f-value independently lower-bounds the optimal
        solution cost. Their maximum is therefore also admissible and avoids
        continuing merely because the opposite frontier has a weaker bound.
        """
        candidates: List[float] = []
        for heap, closed in (
            (self.forward_open, self.forward_closed),
            (self.backward_open, self.backward_closed),
        ):
            while heap and heap[0][2] in closed:
                heapq.heappop(heap)
            if heap:
                candidates.append(float(heap[0][0]))
        return max(candidates) if candidates else float("inf")

    def _fallback_to_astar(self) -> Tuple[bool, List[Tuple[int, int]], int]:
        """Fallback to canonical A* when bidirectional search cannot complete reliably."""
        from .validator import StateSpaceAStar

        self.used_fallback = True
        remaining_budget = int(max(0, int(self.timeout) - int(self.states_explored)))
        if remaining_budget <= 0:
            return False, [], int(self.states_explored)
        fallback = StateSpaceAStar(
            self.env,
            timeout=remaining_budget,
            heuristic_mode=self.heuristic_mode,
            priority_options={'allow_diagonals': self.allow_diagonals},
            search_mode='astar',
        )
        success, path, states = fallback.solve()
        return success, path, int(self.states_explored) + int(states)
    
    def _create_goal_state(self) -> GameState:
        """
        Create a reasonable goal state for backward search.
        
        Challenge: We don't know what inventory the agent will have at the goal.
        Heuristic: Assume agent has collected all items and opened all doors
        (maximal inventory state).
        
        Returns:
            GameState at goal position with maximal inventory
        """
        if not self._requires_canonical_fallback():
            goal_state = self.env.state.copy()
            goal_state.position = self.env.goal_pos
            return goal_state

        # Count all collectable items in dungeon
        all_keys = len(self.env.find_all_positions(SEMANTIC_PALETTE['KEY_SMALL']))
        all_bombs = len(self.env.find_all_positions(SEMANTIC_PALETTE['ITEM_MINOR'])) * 4
        
        # Check for boss key
        has_boss_key = len(self.env.find_all_positions(SEMANTIC_PALETTE['KEY_BOSS'])) > 0
        
        # Check for key item (ladder)
        has_item = len(self.env.find_all_positions(SEMANTIC_PALETTE['KEY_ITEM'])) > 0

        # Find all doors (for opened_doors set)
        all_door_positions = set()
        for door_type in [SEMANTIC_PALETTE['DOOR_LOCKED'], 
                         SEMANTIC_PALETTE['DOOR_BOMB'],
                         SEMANTIC_PALETTE['DOOR_BOSS']]:
            all_door_positions.update(self.env.find_all_positions(door_type))
        
        # Find all items (for collected_items set)
        all_item_positions = set()
        for item_type in [SEMANTIC_PALETTE['KEY_SMALL'],
                         SEMANTIC_PALETTE['KEY_BOSS'],
                         SEMANTIC_PALETTE['KEY_ITEM'],
                         SEMANTIC_PALETTE['ITEM_MINOR']]:
            all_item_positions.update(self.env.find_all_positions(item_type))
        
        goal_state = GameState(
            position=self.env.goal_pos,
            keys=all_keys,
            bomb_count=all_bombs,
            has_boss_key=has_boss_key,
            has_item=has_item,
            opened_doors=all_door_positions,
            collected_items=all_item_positions
        )
        
        return goal_state
    
    def _expand_forward(self, counter: int) -> Tuple[bool, Optional[SearchNode], Optional[SearchNode]]:
        """
        Expand one node from forward frontier.
        
        Returns:
            (collision_found, forward_node, backward_node)
        """
        if not self.forward_open:
            return False, None, None
        
        # Drain duplicate queued entries for already-closed states.
        while self.forward_open:
            _, _, state_hash, current_node = heapq.heappop(self.forward_open)
            if state_hash not in self.forward_closed:
                break
        else:
            return False, None, None
        
        self.forward_closed[state_hash] = current_node
        self.forward_closed_by_pos.setdefault(current_node.state.position, []).append(current_node)
        self.states_explored += 1
        
        # Check for collision with backward frontier
        if state_hash in self.backward_closed:
            self.collision_checks += 1
            backward_node = self.backward_closed[state_hash]
            self.meeting_point = current_node.state.position
            logger.debug(f'Collision detected at {self.meeting_point}')
            return True, current_node, backward_node
        
        # Check approximate collision (same position, compatible inventory)
        collision_node = self._check_approximate_collision(
            current_node, self.backward_closed_by_pos.get(current_node.state.position, []), is_forward=True
        )
        if collision_node:
            self.collision_checks += 1
            self.meeting_point = current_node.state.position
            logger.debug(f'Approximate collision at {self.meeting_point}')
            return True, current_node, collision_node
        
        # Expand successors
        for next_state in self._get_forward_successors(current_node.state):
            next_hash = game_state_key(next_state)
            
            if next_hash in self.forward_closed:
                continue
            
            g_score = current_node.g_score + self._step_cost(current_node.state.position, next_state.position)
            
            if next_hash in self.forward_g_scores and \
               g_score >= self.forward_g_scores[next_hash]:
                continue
            
            self.forward_g_scores[next_hash] = g_score
            h_score = self._heuristic_forward(next_state)
            f_score = g_score + h_score
            
            next_node = SearchNode(
                state=next_state,
                g_score=g_score,
                f_score=f_score,
                parent=current_node,
            )
            
            heapq.heappush(self.forward_open, (f_score, counter, next_hash, next_node))
        
        return False, None, None
    
    def _expand_backward(self, counter: int) -> Tuple[bool, Optional[SearchNode], Optional[SearchNode]]:
        """
        Expand one node from backward frontier.
        
        CRITICAL: Backward expansion inverts actions:
        - Moving "back" from a door means we must have had the key BEFORE passing
        - Moving "back" from an item means we did NOT have it yet
        
        Returns:
            (collision_found, backward_node, forward_node)
        """
        if not self.backward_open:
            return False, None, None
        
        # Drain duplicate queued entries for already-closed states.
        while self.backward_open:
            _, _, state_hash, current_node = heapq.heappop(self.backward_open)
            if state_hash not in self.backward_closed:
                break
        else:
            return False, None, None
        
        self.backward_closed[state_hash] = current_node
        self.backward_closed_by_pos.setdefault(current_node.state.position, []).append(current_node)
        self.states_explored += 1
        
        # Check for collision with forward frontier
        if state_hash in self.forward_closed:
            self.collision_checks += 1
            forward_node = self.forward_closed[state_hash]
            self.meeting_point = current_node.state.position
            logger.debug(f'Collision detected at {self.meeting_point}')
            return True, current_node, forward_node
        
        # Check approximate collision
        collision_node = self._check_approximate_collision(
            current_node, self.forward_closed_by_pos.get(current_node.state.position, []), is_forward=False
        )
        if collision_node:
            self.collision_checks += 1
            self.meeting_point = current_node.state.position
            logger.debug(f'Approximate collision at {self.meeting_point}')
            return True, current_node, collision_node
        
        # Expand predecessors (reversed actions)
        for prev_state in self._get_backward_predecessors(current_node.state):
            prev_hash = game_state_key(prev_state)
            
            if prev_hash in self.backward_closed:
                continue
            
            g_score = current_node.g_score + self._step_cost(prev_state.position, current_node.state.position)
            
            if prev_hash in self.backward_g_scores and \
               g_score >= self.backward_g_scores[prev_hash]:
                continue
            
            self.backward_g_scores[prev_hash] = g_score
            h_score = self._heuristic_backward(prev_state)
            f_score = g_score + h_score
            
            prev_node = SearchNode(
                state=prev_state,
                g_score=g_score,
                f_score=f_score,
                parent=current_node,
            )
            
            heapq.heappush(self.backward_open, (f_score, counter, prev_hash, prev_node))
        
        return False, None, None
    
    def _check_approximate_collision(self, node: SearchNode, 
                                    other_at_position: List[SearchNode],
                                    is_forward: bool) -> Optional[SearchNode]:
        """
        Check if this node approximately collides with opposite frontier.
        
        Approximate collision: same position, inventory compatible
        (forward inventory subset of backward inventory).
        
        CRITICAL FIX: Also check opened_doors and collected_items compatibility!
        
        Args:
            node: Current node
            other_at_position: Opposite frontier nodes at the same position
            is_forward: True if checking forward node against backward frontier
            
        Returns:
            Matching node from opposite frontier, or None
        """
        def _inventory_can_execute_suffix(forward_state: GameState, backward_state: GameState) -> bool:
            """Return whether the forward prefix state can execute the backward suffix."""
            return bool(
                int(forward_state.keys) >= int(backward_state.keys)
                and int(forward_state.bomb_count) >= int(backward_state.bomb_count)
                and (not bool(backward_state.has_boss_key) or bool(forward_state.has_boss_key))
                and (not bool(backward_state.has_item) or bool(forward_state.has_item))
            )

        for other_node in other_at_position:
            
            # Check inventory compatibility
            if is_forward:
                # Approximate frontier splicing is only sound when the forward
                # prefix state is contained in the backward suffix state. Exact
                # set equality almost never occurs in puzzle dungeons because
                # the backward side carries suffix doors/items/enemies.
                inventory_compatible = _inventory_can_execute_suffix(node.state, other_node.state)
                
                # Forward persistent effects must be a subset of the backward
                # suffix state at the same physical position.
                state_sets_compatible = (
                    set(node.state.opened_doors).issubset(other_node.state.opened_doors) and
                    set(node.state.collected_items).issubset(other_node.state.collected_items) and
                    set(node.state.defeated_enemies).issubset(other_node.state.defeated_enemies) and
                    set(node.state.completed_puzzle_stages).issubset(other_node.state.completed_puzzle_stages) and
                    set(node.state.pushed_blocks).issubset(other_node.state.pushed_blocks)
                )
                
                if inventory_compatible and state_sets_compatible:
                    return other_node
            else:
                inventory_compatible = _inventory_can_execute_suffix(other_node.state, node.state)
                
                # Reversed call: other_node is the forward prefix, node is the
                # backward suffix. The forward effects must be contained in the
                # backward suffix state.
                state_sets_compatible = (
                    set(other_node.state.opened_doors).issubset(node.state.opened_doors) and
                    set(other_node.state.collected_items).issubset(node.state.collected_items) and
                    set(other_node.state.defeated_enemies).issubset(node.state.defeated_enemies) and
                    set(other_node.state.completed_puzzle_stages).issubset(node.state.completed_puzzle_stages) and
                    set(other_node.state.pushed_blocks).issubset(node.state.pushed_blocks)
                )
                
                if inventory_compatible and state_sets_compatible:
                    return other_node
        
        return None
    
    def _get_forward_successors(self, state: GameState) -> List[GameState]:
        """
        Generate forward successors (same as standard A*).
        
        Args:
            state: Current game state
            
        Returns:
            List of successor states
        """
        successors = []
        curr_r, curr_c = state.position
        
        # Cardinal directions
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_r, new_c = curr_r + dr, curr_c + dc
            
            if not (0 <= new_r < self.height and 0 <= new_c < self.width):
                continue
            
            target_tile = self.grid[new_r, new_c]
            can_move, new_state = self._try_move_forward(state, (new_r, new_c), target_tile)
            
            if can_move:
                successors.append(new_state)
        
        # Diagonals (if enabled)
        if self.allow_diagonals:
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                new_r, new_c = curr_r + dr, curr_c + dc
                
                if not (0 <= new_r < self.height and 0 <= new_c < self.width):
                    continue
                
                # Check corner cutting
                adj_r = self.grid[curr_r + dr, curr_c]
                adj_c = self.grid[curr_r, curr_c + dc]
                if (
                    adj_r in BLOCKING_IDS
                    or adj_c in BLOCKING_IDS
                    or adj_r in CONDITIONAL_IDS
                    or adj_c in CONDITIONAL_IDS
                    or adj_r in PUSHABLE_IDS
                    or adj_c in PUSHABLE_IDS
                    or adj_r in WATER_IDS
                    or adj_c in WATER_IDS
                ):
                    continue
                
                target_tile = self.grid[new_r, new_c]
                can_move, new_state = self._try_move_forward(state, (new_r, new_c), target_tile)
                
                if can_move:
                    successors.append(new_state)
        
        return successors
    
    def _get_backward_predecessors(self, state: GameState) -> List[GameState]:
        """
        Generate backward predecessors (INVERTED actions).
        
        Challenge: Given a state, what states could have reached it?
        
        Inversion rules:
        - If we're past a door, predecessor must have had the key BEFORE
        - If we have an item, predecessor did NOT have it yet
        - Position: move in reverse direction
        
        Args:
            state: Current game state
            
        Returns:
            List of predecessor states
        """
        predecessors = []
        curr_r, curr_c = state.position
        
        # Cardinal directions (reversed)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            # Move backward = reverse direction
            prev_r, prev_c = curr_r - dr, curr_c - dc
            
            if not (0 <= prev_r < self.height and 0 <= prev_c < self.width):
                continue
            
            prev_tile = self.grid[prev_r, prev_c]
            can_move, prev_state = self._try_move_backward(state, (prev_r, prev_c), prev_tile)
            
            if can_move:
                predecessors.append(prev_state)
        
        # Diagonals (if enabled)
        if self.allow_diagonals:
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                prev_r, prev_c = curr_r - dr, curr_c - dc
                
                if not (0 <= prev_r < self.height and 0 <= prev_c < self.width):
                    continue
                
                # Check corner cutting
                adj_r = self.grid[curr_r - dr, curr_c]
                adj_c = self.grid[curr_r, curr_c - dc]
                if (
                    adj_r in BLOCKING_IDS
                    or adj_c in BLOCKING_IDS
                    or adj_r in CONDITIONAL_IDS
                    or adj_c in CONDITIONAL_IDS
                    or adj_r in PUSHABLE_IDS
                    or adj_c in PUSHABLE_IDS
                    or adj_r in WATER_IDS
                    or adj_c in WATER_IDS
                ):
                    continue
                
                prev_tile = self.grid[prev_r, prev_c]
                can_move, prev_state = self._try_move_backward(state, (prev_r, prev_c), prev_tile)
                
                if can_move:
                    predecessors.append(prev_state)
        
        return predecessors
    
    def _try_move_forward(self, state: GameState, target_pos: Tuple[int, int],
                         target_tile: int) -> Tuple[bool, GameState]:
        """
        Forward move using canonical environment transition logic.
        """
        return self.env.try_move_pure(state, target_pos, target_tile)
    
    def _try_move_backward(self, state: GameState, prev_pos: Tuple[int, int],
                          prev_tile: int) -> Tuple[bool, GameState]:
        """
        Backward move (INVERTED logic).
        
        Moving backward from current state to predecessor:
        - If current state opened a door, predecessor must have had key
        - If current state collected item, predecessor did NOT have it
        
        Args:
            state: Current state (moving backward FROM this)
            prev_pos: Previous position (moving TO this)
            prev_tile: Tile at previous position
            
        Returns:
            (can_move, prev_state)
        """
        if prev_tile in BLOCKING_IDS:
            return False, state
        
        prev_state = state.copy()
        prev_state.position = prev_pos
        
        # Check if we need to UNDO door opening
        curr_tile = self.grid[state.position[0], state.position[1]]
        
        # If current position is an opened door, predecessor must have opened it
        if (curr_tile in {SEMANTIC_PALETTE['DOOR_LOCKED'],
                         SEMANTIC_PALETTE['DOOR_BOMB'],
                         SEMANTIC_PALETTE['DOOR_BOSS']} and
            state.position in state.opened_doors):
            
            # Predecessor state did NOT have door opened yet
            prev_state.opened_doors = state.opened_doors - {state.position}
            
            # And must have had required item
            if curr_tile == SEMANTIC_PALETTE['DOOR_LOCKED']:
                prev_state.keys = state.keys + 1  # Add key back
            elif curr_tile == SEMANTIC_PALETTE['DOOR_BOMB']:
                prev_state.bomb_count = state.bomb_count + 1  # Add bomb back
        
        # Check if we need to UNDO item collection
        if (curr_tile in PICKUP_IDS and 
            state.position in state.collected_items):
            
            # Predecessor did NOT have item yet
            prev_state.collected_items = state.collected_items - {state.position}
            
            # Remove item effects
            if curr_tile == SEMANTIC_PALETTE['KEY_SMALL']:
                prev_state.keys = max(0, state.keys - 1)
            elif curr_tile == SEMANTIC_PALETTE['KEY_BOSS']:
                prev_state.has_boss_key = False
            elif curr_tile == SEMANTIC_PALETTE['KEY_ITEM']:
                prev_state.has_item = False
            elif curr_tile == SEMANTIC_PALETTE['ITEM_MINOR']:
                prev_state.bomb_count = max(0, state.bomb_count - 4)
        
        # Check if previous position is walkable
        if prev_tile not in WALKABLE_IDS and \
           prev_tile not in {SEMANTIC_PALETTE['DOOR_OPEN'],
                            SEMANTIC_PALETTE['DOOR_SOFT'],
                            SEMANTIC_PALETTE['DOOR_PUZZLE']}:
            return False, state
        
        return True, prev_state
    
    def _grid_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        dr = abs(int(a[0]) - int(b[0]))
        dc = abs(int(a[1]) - int(b[1]))
        if self.allow_diagonals:
            diagonal = min(dr, dc)
            straight = max(dr, dc) - diagonal
            return float((DIAGONAL_COST * diagonal) + (CARDINAL_COST * straight))
        return float(dr + dc)

    def _heuristic_forward(self, state: GameState) -> float:
        """Admissible distance to goal."""
        if self.env.goal_pos is None:
            return float('inf')
        
        return self._grid_distance(state.position, self.env.goal_pos)
    
    def _heuristic_backward(self, state: GameState) -> float:
        """Admissible distance to start."""
        if self.env.start_pos is None:
            return float('inf')
        
        return self._grid_distance(state.position, self.env.start_pos)
    
    def _reconstruct_path(self, forward_node: SearchNode, 
                         backward_node: SearchNode) -> List[Tuple[int, int]]:
        """
        Reconstruct complete path from start to goal.
        
        Concatenates:
        - Forward path: start -> meeting point
        - Backward path (reversed): meeting point -> goal
        
        Args:
            forward_node: Node from forward search at meeting point
            backward_node: Node from backward search at meeting point
            
        Returns:
            Complete path from start to goal
        """
        forward_path: List[Tuple[int, int]] = []
        node: Optional[SearchNode] = forward_node
        while node is not None:
            forward_path.append(node.state.position)
            node = node.parent
        forward_path.reverse()

        backward_path: List[Tuple[int, int]] = []
        node = backward_node
        while node is not None:
            backward_path.append(node.state.position)
            node = node.parent

        return forward_path + backward_path[1:]


# ==========================================
# STANDALONE TESTING
# ==========================================

if __name__ == "__main__":
    import numpy as np
    
    # Configure logging
    logging.basicConfig(level=logging.DEBUG,
                       format='%(levelname)s - %(message)s')
    
    from src.core.definitions import SEMANTIC_PALETTE
    
    # Create test dungeon
    test_grid = np.full((20, 20), SEMANTIC_PALETTE['FLOOR'], dtype=np.int64)
    
    # Add walls
    test_grid[0, :] = SEMANTIC_PALETTE['WALL']
    test_grid[-1, :] = SEMANTIC_PALETTE['WALL']
    test_grid[:, 0] = SEMANTIC_PALETTE['WALL']
    test_grid[:, -1] = SEMANTIC_PALETTE['WALL']
    
    # Add start and goal (far apart for bidirectional to be effective)
    test_grid[1, 1] = SEMANTIC_PALETTE['START']
    test_grid[18, 18] = SEMANTIC_PALETTE['TRIFORCE']
    
    # Add some obstacles
    test_grid[10, :] = SEMANTIC_PALETTE['WALL']
    test_grid[10, 5] = SEMANTIC_PALETTE['DOOR_OPEN']  # Opening
    test_grid[10, 15] = SEMANTIC_PALETTE['DOOR_OPEN']  # Opening
    
    logger.info('Testing BidirectionalAStar on long-path dungeon...')
    
    from .validator import ZeldaLogicEnv
    env = ZeldaLogicEnv(test_grid)
    solver = BidirectionalAStar(env, timeout=100000)
    success, path, states = solver.solve()
    
    logger.info(f'Result: success={success}, path_len={len(path)}, states={states}')
    logger.info(f'Meeting point: {solver.meeting_point}')
    logger.info(f'Collision checks: {solver.collision_checks}')
