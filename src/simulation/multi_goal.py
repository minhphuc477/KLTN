"""Bounded, state-preserving multi-objective route planning.

This module is a routing utility, not a canonical dungeon-solvability oracle.
It searches one augmented state space containing the complete ``GameState`` and
the objectives that remain.  It deliberately does not chain position-only A*
segments, because doing so loses consumed inventory and changed world state.

The planner shares Zelda's authoritative transition functions with
``StateSpaceAStar``.  Search is bounded by both objective count and explored
states.  Four-neighbor movement is the default; diagonal movement is available
only when explicitly requested.  Graph transitions are supported, but use a
zero heuristic so graph shortcuts cannot invalidate optimality.
"""

from __future__ import annotations

import colorsys
import heapq
import logging
from dataclasses import dataclass, field
from itertools import count
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple

from src.core.definitions import ROOM_HEIGHT, ROOM_WIDTH, SEMANTIC_PALETTE

from .state import (
    BLOCKING_IDS,
    CARDINAL_COST,
    CONDITIONAL_IDS,
    DIAGONAL_COST,
    PICKUP_IDS,
    PUSHABLE_IDS,
    WATER_IDS,
    GameState,
    game_state_key,
)
from .validator import StateSpaceAStar, ZeldaLogicEnv


logger = logging.getLogger(__name__)

Position = Tuple[int, int]
SearchKey = Tuple[Tuple[Any, ...], FrozenSet[Position]]


@dataclass
class MultiGoalResult:
    """Route plus state evidence for a bounded multi-objective search."""

    success: bool
    full_path: List[Position]
    waypoints: List[Position]
    segment_paths: List[List[Position]]
    total_cost: float
    exploration_count: int
    terminal_state: Optional[GameState] = None
    segment_end_states: List[GameState] = field(default_factory=list)
    collected_targets: List[Position] = field(default_factory=list)
    remaining_targets: List[Position] = field(default_factory=list)
    optimal: bool = False
    state_limit_reached: bool = False
    failure_reason: Optional[str] = None


class MultiGoalPathfinder:
    """Find a minimum-cost route that collects target tiles before the goal.

    The search key includes every field represented by ``game_state_key`` plus
    the remaining target positions.  Successful results are therefore evidence
    of one continuous stateful execution, including consumed keys/bombs,
    opened doors and graph edges, moved blocks, defeated enemies, and completed
    puzzle stages.

    Limitations:
    - This is an explicitly bounded utility and may return an inconclusive
      failure when ``max_states`` is reached.
    - At most ``max_targets`` distinct, initially-unsatisfied target positions
      are accepted.
    - It is intentionally not registered as a canonical solvability oracle.
    """

    DEFAULT_GOAL_TYPES = (
        SEMANTIC_PALETTE["KEY_SMALL"],
        SEMANTIC_PALETTE["KEY_BOSS"],
        SEMANTIC_PALETTE["KEY_ITEM"],
    )

    def __init__(
        self,
        env: ZeldaLogicEnv,
        *,
        max_states: int = 250_000,
        max_targets: int = 10,
        allow_diagonals: bool = False,
    ) -> None:
        if max_states <= 0:
            raise ValueError("max_states must be positive")
        if max_targets < 0:
            raise ValueError("max_targets must be non-negative")

        self.env = env
        self.max_states = int(max_states)
        self.max_targets = int(max_targets)
        self.allow_diagonals = bool(allow_diagonals)
        self._transition_solver = StateSpaceAStar(
            env,
            timeout=self.max_states,
            priority_options={
                "representation": "tile",
                "allow_diagonals": self.allow_diagonals,
            },
        )
        self._mst_cache: Dict[FrozenSet[Position], float] = {}

    def find_optimal_collection_order(
        self,
        start_state: GameState,
        goal_types: Optional[List[int]] = None,
    ) -> MultiGoalResult:
        """Search one continuous state space for all targets and the final goal.

        ``goal_types`` identifies tile types whose positions must be visited and,
        for pickup tiles, canonically collected.  The environment's Triforce is
        always the final objective when present.  The input state and environment
        state are not mutated.
        """
        initial_state = start_state.copy()
        invalid_reason = self._validate_start_state(initial_state)
        if invalid_reason is not None:
            return self._failure(initial_state, (), 0, invalid_reason)

        target_types = self.DEFAULT_GOAL_TYPES if goal_types is None else tuple(goal_types)
        target_positions = {
            tuple(position)
            for tile_id in target_types
            for position in self.env.find_all_positions(int(tile_id))
        }
        final_goal = tuple(self.env.goal_pos) if self.env.goal_pos is not None else None
        if final_goal is not None:
            target_positions.discard(final_goal)

        initial_remaining = frozenset(
            position
            for position in target_positions
            if not self._objective_satisfied(initial_state, position)
        )
        if len(initial_remaining) > self.max_targets:
            return self._failure(
                initial_state,
                initial_remaining,
                0,
                (
                    f"target limit exceeded: {len(initial_remaining)} objectives "
                    f"is greater than max_targets={self.max_targets}"
                ),
            )

        logger.info(
            "MultiGoal: searching %d targets with max_states=%d",
            len(initial_remaining),
            self.max_states,
        )
        return self._search(initial_state, initial_remaining, final_goal)

    def _search(
        self,
        initial_state: GameState,
        initial_remaining: FrozenSet[Position],
        final_goal: Optional[Position],
    ) -> MultiGoalResult:
        initial_key: SearchKey = (game_state_key(initial_state), initial_remaining)
        sequence = count()
        initial_h = self._heuristic(initial_state.position, initial_remaining, final_goal)
        open_set = [(initial_h, next(sequence), 0.0, initial_key, initial_state)]
        g_scores: Dict[SearchKey, float] = {initial_key: 0.0}
        parents: Dict[SearchKey, Optional[SearchKey]] = {initial_key: None}
        states: Dict[SearchKey, GameState] = {initial_key: initial_state}
        explored = 0

        while open_set:
            if explored >= self.max_states:
                return self._failure(
                    initial_state,
                    initial_remaining,
                    explored,
                    f"state limit reached: max_states={self.max_states}",
                    state_limit_reached=True,
                )

            _f_score, _order, current_g, current_key, current_state = heapq.heappop(open_set)
            if current_g != g_scores.get(current_key):
                continue
            explored += 1
            remaining = current_key[1]

            if not remaining and (
                final_goal is None or current_state.position == final_goal
            ):
                return self._build_success(
                    current_key,
                    parents,
                    states,
                    g_scores,
                    initial_remaining,
                    final_goal,
                    explored,
                )

            for next_state, transition_cost in self._successors(current_state):
                next_remaining = self._remaining_after_state(remaining, next_state)
                next_key: SearchKey = (game_state_key(next_state), next_remaining)
                next_g = current_g + transition_cost
                if next_g >= g_scores.get(next_key, float("inf")):
                    continue

                g_scores[next_key] = next_g
                parents[next_key] = current_key
                states[next_key] = next_state
                next_h = self._heuristic(next_state.position, next_remaining, final_goal)
                heapq.heappush(
                    open_set,
                    (next_g + next_h, next(sequence), next_g, next_key, next_state),
                )

        return self._failure(
            initial_state,
            initial_remaining,
            explored,
            "search frontier exhausted: no stateful route satisfies all objectives",
        )

    def _successors(self, state: GameState) -> Iterable[Tuple[GameState, float]]:
        """Yield the same tile and graph transitions used by canonical A*."""
        grid = self.env.original_grid
        height, width = grid.shape
        current_row, current_col = state.position
        transitions: List[Tuple[Position, int, float, Optional[str]]] = []

        for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            row, col = current_row + delta_row, current_col + delta_col
            if 0 <= row < height and 0 <= col < width:
                transitions.append(
                    ((row, col), int(grid[row, col]), float(CARDINAL_COST), None)
                )

        if self.allow_diagonals:
            for delta_row, delta_col in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                row, col = current_row + delta_row, current_col + delta_col
                if not (0 <= row < height and 0 <= col < width):
                    continue
                corner_tiles = (
                    int(grid[current_row + delta_row, current_col]),
                    int(grid[current_row, current_col + delta_col]),
                )
                if any(
                    tile in BLOCKING_IDS
                    or tile in CONDITIONAL_IDS
                    or tile in PUSHABLE_IDS
                    or tile in WATER_IDS
                    for tile in corner_tiles
                ):
                    continue
                transitions.append(
                    ((row, col), int(grid[row, col]), float(DIAGONAL_COST), None)
                )

        current_tile = int(grid[current_row, current_col])
        is_stair = current_tile == int(SEMANTIC_PALETTE["STAIR"])
        is_door = current_tile in {
            int(SEMANTIC_PALETTE["DOOR_OPEN"]),
            int(SEMANTIC_PALETTE["DOOR_SOFT"]),
            int(SEMANTIC_PALETTE["DOOR_LOCKED"]),
            int(SEMANTIC_PALETTE["DOOR_BOMB"]),
            int(SEMANTIC_PALETTE["DOOR_BOSS"]),
        }
        is_at_boundary = self._is_at_room_boundary(state.position)
        can_teleport = is_stair or is_door
        if not self._transition_solver.vglc_strict_mode:
            can_teleport = can_teleport or is_at_boundary
        if self._transition_solver.strict_original_mode:
            can_teleport = is_stair

        if is_stair:
            for destination in self._transition_solver.get_stair_destinations(state.position):
                row, col = destination
                if 0 <= row < height and 0 <= col < width:
                    transitions.append(
                        (tuple(destination), int(grid[row, col]), 1.0, "stair")
                    )

        if can_teleport and not self._transition_solver.strict_original_mode:
            graph_destinations = (
                self._transition_solver.get_controlled_virtual_destinations(
                    state.position, state
                )
                + self._transition_solver.get_graph_warp_destinations(
                    state.position, state
                )
            )
            for destination, base_cost, edge_type in graph_destinations:
                row, col = destination
                if 0 <= row < height and 0 <= col < width:
                    transitions.append(
                        (
                            tuple(destination),
                            int(grid[row, col]),
                            float(base_cost),
                            str(edge_type),
                        )
                    )

        for target_pos, target_tile, base_cost, edge_type in transitions:
            transition_state = state
            if edge_type not in {None, "stair"}:
                allowed, transition_state = (
                    self._transition_solver.apply_graph_edge_transition(
                        state,
                        state.position,
                        target_pos,
                        edge_type,
                    )
                )
                if not allowed:
                    continue

            allowed, next_state = self.env.try_move_pure(
                transition_state,
                target_pos,
                target_tile,
            )
            if not allowed:
                continue
            next_state.current_floor = self.env.floor_for_position(
                target_pos,
                default=state.current_floor,
            )
            movement_cost = self._transition_solver._get_movement_cost(
                target_tile,
                target_pos,
                state,
            )
            yield next_state, float(movement_cost) * base_cost

    def _is_at_room_boundary(self, position: Position) -> bool:
        if not self.env.room_positions:
            return False
        row, col = position
        for row_offset, col_offset in self.env.room_positions.values():
            if (
                row_offset <= row < row_offset + ROOM_HEIGHT
                and col_offset <= col < col_offset + ROOM_WIDTH
            ):
                local_row = row - row_offset
                local_col = col - col_offset
                return bool(
                    local_row <= 1
                    or local_row >= ROOM_HEIGHT - 2
                    or local_col <= 1
                    or local_col >= ROOM_WIDTH - 2
                )
        return False

    def _objective_satisfied(self, state: GameState, position: Position) -> bool:
        tile = int(self.env.original_grid[position[0], position[1]])
        if tile in PICKUP_IDS:
            return position in state.collected_items
        return state.position == position

    def _remaining_after_state(
        self,
        remaining: FrozenSet[Position],
        state: GameState,
    ) -> FrozenSet[Position]:
        if state.position not in remaining:
            return remaining
        if not self._objective_satisfied(state, state.position):
            return remaining
        return remaining - {state.position}

    def _heuristic(
        self,
        position: Position,
        remaining: FrozenSet[Position],
        final_goal: Optional[Position],
    ) -> float:
        # Graph transitions can shortcut Manhattan distance, so Dijkstra is the
        # only generally admissible choice when topology links are available.
        if self.env.graph:
            return 0.0
        if not remaining:
            return float(self._grid_distance(position, final_goal)) if final_goal else 0.0

        nearest = min(self._grid_distance(position, target) for target in remaining)
        lower_bound = float(nearest) + self._mst_cost(remaining)
        if final_goal is not None:
            lower_bound += float(
                min(self._grid_distance(target, final_goal) for target in remaining)
            )
        return lower_bound

    def _mst_cost(self, positions: FrozenSet[Position]) -> float:
        cached = self._mst_cache.get(positions)
        if cached is not None:
            return cached
        if len(positions) <= 1:
            self._mst_cache[positions] = 0.0
            return 0.0

        remaining = set(positions)
        connected = {remaining.pop()}
        total = 0.0
        while remaining:
            distance, target = min(
                (self._grid_distance(source, candidate), candidate)
                for source in connected
                for candidate in remaining
            )
            total += float(distance)
            connected.add(target)
            remaining.remove(target)
        self._mst_cache[positions] = total
        return total

    def _grid_distance(self, left: Position, right: Position) -> int:
        row_distance = abs(left[0] - right[0])
        col_distance = abs(left[1] - right[1])
        if self.allow_diagonals:
            return max(row_distance, col_distance)
        return row_distance + col_distance

    def _build_success(
        self,
        terminal_key: SearchKey,
        parents: Dict[SearchKey, Optional[SearchKey]],
        states: Dict[SearchKey, GameState],
        g_scores: Dict[SearchKey, float],
        initial_remaining: FrozenSet[Position],
        final_goal: Optional[Position],
        explored: int,
    ) -> MultiGoalResult:
        chain: List[SearchKey] = []
        current: Optional[SearchKey] = terminal_key
        while current is not None:
            chain.append(current)
            current = parents[current]
        chain.reverse()

        full_path = [states[key].position for key in chain]
        waypoints: List[Position] = []
        waypoint_indices: List[int] = []
        for index in range(1, len(chain)):
            removed = chain[index - 1][1] - chain[index][1]
            for position in sorted(removed):
                waypoints.append(position)
                waypoint_indices.append(index)

        collected_targets = list(waypoints)
        if final_goal is not None:
            waypoints.append(final_goal)
            waypoint_indices.append(len(chain) - 1)

        segment_paths: List[List[Position]] = []
        segment_end_states: List[GameState] = []
        segment_start = 0
        for waypoint_index in waypoint_indices:
            segment_paths.append(full_path[segment_start : waypoint_index + 1])
            segment_end_states.append(states[chain[waypoint_index]].copy())
            segment_start = waypoint_index

        terminal_state = states[terminal_key].copy()
        return MultiGoalResult(
            success=True,
            full_path=full_path,
            waypoints=waypoints,
            segment_paths=segment_paths,
            total_cost=float(g_scores[terminal_key]),
            exploration_count=explored,
            terminal_state=terminal_state,
            segment_end_states=segment_end_states,
            collected_targets=collected_targets,
            remaining_targets=[],
            optimal=True,
        )

    def _validate_start_state(self, state: GameState) -> Optional[str]:
        row, col = state.position
        height, width = self.env.original_grid.shape
        if not (0 <= row < height and 0 <= col < width):
            return f"start position {state.position} is outside the environment"
        return None

    @staticmethod
    def _failure(
        initial_state: GameState,
        remaining: Iterable[Position],
        explored: int,
        reason: str,
        *,
        state_limit_reached: bool = False,
    ) -> MultiGoalResult:
        logger.warning("MultiGoal: %s", reason)
        return MultiGoalResult(
            success=False,
            full_path=[],
            waypoints=[],
            segment_paths=[],
            total_cost=float("inf"),
            exploration_count=explored,
            terminal_state=None,
            remaining_targets=sorted(remaining),
            optimal=False,
            state_limit_reached=state_limit_reached,
            failure_reason=reason,
        )


def get_waypoint_colors(num_waypoints: int) -> List[Tuple[int, int, int]]:
    """Generate distinct RGB colors for waypoint rendering."""
    if num_waypoints <= 0:
        return []
    colors = []
    for index in range(num_waypoints):
        hue = index / num_waypoints
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append(tuple(int(channel * 255) for channel in rgb))
    return colors


def render_waypoint_numbers(
    surface: Any,
    waypoints: List[Position],
    tile_size: int,
    font: Any,
) -> None:
    """Render numbered waypoint markers on a Pygame surface."""
    import pygame

    for index, (row, col) in enumerate(waypoints):
        center_x = col * tile_size + tile_size // 2
        center_y = row * tile_size + tile_size // 2
        pygame.draw.circle(
            surface, (255, 215, 0), (center_x, center_y), tile_size // 3
        )
        pygame.draw.circle(
            surface, (0, 0, 0), (center_x, center_y), tile_size // 3, 2
        )
        text = font.render(str(index + 1), True, (0, 0, 0))
        surface.blit(text, text.get_rect(center=(center_x, center_y)))


__all__ = [
    "MultiGoalPathfinder",
    "MultiGoalResult",
    "get_waypoint_colors",
    "render_waypoint_numbers",
]
