"""
Backward D* Lite replanning for reversible Zelda grid movement.

The textbook D* Lite recurrence is rooted at the goal: ``rhs(goal) = 0`` and
the search propagates costs backward toward the agent's current position.
Inventory consumption, item pickup, block pushing, directed graph transitions,
and staged puzzles make transitions path-dependent and generally irreversible.
Those maps are delegated to the canonical full-state A* solver rather than
being represented by an unsound reverse search.

Reference:
    Koenig, S., & Likhachev, M. (2002). D* Lite. AAAI, 476-483.
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from .validator import (
    ACTION_DELTAS,
    BLOCKING_IDS,
    CARDINAL_COST,
    CONDITIONAL_IDS,
    DIAGONAL_COST,
    PICKUP_IDS,
    PUSHABLE_IDS,
    SEMANTIC_PALETTE,
    WATER_IDS,
    GameState,
    game_state_key,
)

logger = logging.getLogger(__name__)


@dataclass(order=True)
class DStarKey:
    """Lexicographic D* Lite priority key."""

    k1: float
    k2: float
    state_hash: Any = field(compare=False)
    state: GameState = field(compare=False)


class DStarLiteSolver:
    """Backward incremental replanner with a full-state A* safety boundary."""

    def __init__(
        self,
        env: Any,
        heuristic_mode: str = "balanced",
        timeout: int = 100000,
        allow_diagonals: bool = False,
    ) -> None:
        self.env = env
        self.heuristic_mode = str(heuristic_mode)
        self.timeout = int(max(1, timeout))
        self.allow_diagonals = bool(allow_diagonals)

        self.g_scores: Dict[Any, float] = {}
        self.rhs_scores: Dict[Any, float] = {}
        self.open_set: List[DStarKey] = []
        self.open_set_hashes: Dict[Any, Tuple[float, float]] = {}
        self.predecessors: Dict[Any, Optional[Any]] = {}
        self.states_by_hash: Dict[Any, GameState] = {}

        self.km = 0.0
        self.current_start_state: Optional[GameState] = None
        self.last_start_state: Optional[GameState] = None
        self._grid_snapshot: Optional[np.ndarray] = None
        self._textbook_core_active = False
        self._last_compute_budget_exhausted = False

        self.last_opened_doors: Set[Tuple[int, int]] = set()
        self.last_pushed_blocks: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
        self.replans_count = 0
        self.states_updated = 0
        self.used_fallback = False
        self.current_path: List[Tuple[int, int]] = []
        self.path_index = 0

    def _iter_action_deltas(self) -> Iterable[Tuple[int, int]]:
        for dr, dc in ACTION_DELTAS.values():
            if not self.allow_diagonals and abs(int(dr)) + abs(int(dc)) == 2:
                continue
            yield int(dr), int(dc)

    def _distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        dr = abs(int(a[0]) - int(b[0]))
        dc = abs(int(a[1]) - int(b[1]))
        if not self.allow_diagonals:
            return float(dr + dc)
        diagonal = min(dr, dc)
        straight = max(dr, dc) - diagonal
        return float((DIAGONAL_COST * diagonal) + (CARDINAL_COST * straight))

    def _heuristic(self, state: GameState) -> float:
        """Goal-distance helper retained for diagnostics and compatibility."""
        if self.env.goal_pos is None:
            return float("inf")
        return self._distance(tuple(state.position), tuple(self.env.goal_pos))

    def _supports_textbook_core(self) -> bool:
        """Return whether the current problem is reversible and position-only."""
        if self.env.start_pos is None or self.env.goal_pos is None:
            return False
        if getattr(self.env, "graph", None):
            return False
        if getattr(self.env, "_puzzle_stage_lookup", None):
            return False
        if getattr(self.env, "block_underlay_tiles", None):
            return False

        stateful_ids = (
            set(CONDITIONAL_IDS)
            | set(PICKUP_IDS)
            | set(PUSHABLE_IDS)
            | set(WATER_IDS)
            | {
                int(SEMANTIC_PALETTE["DOOR_SOFT"]),
                int(SEMANTIC_PALETTE["ENEMY"]),
                int(SEMANTIC_PALETTE["BOSS"]),
                int(SEMANTIC_PALETTE["PUZZLE"]),
            }
        )
        present = {int(value) for value in np.asarray(self.env.grid).reshape(-1)}
        return not bool(present.intersection(stateful_ids))

    def _state_at(self, position: Tuple[int, int], template: GameState) -> GameState:
        state = template.copy()
        state.position = tuple(position)
        return state

    def _position_is_traversable(self, position: Tuple[int, int]) -> bool:
        row, col = position
        if not (0 <= row < self.env.height and 0 <= col < self.env.width):
            return False
        tile = int(self.env.grid[row, col])
        return tile not in BLOCKING_IDS and tile not in CONDITIONAL_IDS and tile not in PUSHABLE_IDS

    def calculate_key(self, state: GameState, state_hash: Any) -> DStarKey:
        """Calculate ``[min(g,rhs)+h(start,s)+km, min(g,rhs)]``."""
        g = self.g_scores.get(state_hash, float("inf"))
        rhs = self.rhs_scores.get(state_hash, float("inf"))
        minimum = min(g, rhs)
        if self.current_start_state is None:
            heuristic = 0.0
        else:
            heuristic = self._distance(
                tuple(self.current_start_state.position),
                tuple(state.position),
            )
        return DStarKey(
            k1=float(minimum + heuristic + self.km),
            k2=float(minimum),
            state_hash=state_hash,
            state=state.copy(),
        )

    @staticmethod
    def _key_tuple(key: DStarKey) -> Tuple[float, float]:
        return float(key.k1), float(key.k2)

    def _queue_if_inconsistent(self, state: GameState, state_hash: Any) -> None:
        g = self.g_scores.get(state_hash, float("inf"))
        rhs = self.rhs_scores.get(state_hash, float("inf"))
        if g == rhs:
            self.open_set_hashes.pop(state_hash, None)
            return
        key = self.calculate_key(state, state_hash)
        heapq.heappush(self.open_set, key)
        self.open_set_hashes[state_hash] = self._key_tuple(key)
        self.states_updated += 1

    def _clean_open(self) -> None:
        while self.open_set:
            top = self.open_set[0]
            if self.open_set_hashes.get(top.state_hash) == self._key_tuple(top):
                return
            heapq.heappop(self.open_set)

    def _get_successors(self, state: GameState) -> List[GameState]:
        """Return reversible grid neighbors under the configured movement model."""
        if not self._position_is_traversable(tuple(state.position)):
            return []

        successors: List[GameState] = []
        row, col = state.position
        for dr, dc in self._iter_action_deltas():
            next_pos = row + dr, col + dc
            if not self._position_is_traversable(next_pos):
                continue
            if abs(dr) == 1 and abs(dc) == 1:
                if (
                    not self._position_is_traversable((row + dr, col))
                    or not self._position_is_traversable((row, col + dc))
                ):
                    continue
            target_tile = int(self.env.grid[next_pos[0], next_pos[1]])
            allowed, successor = self.env.try_move_pure(state, next_pos, target_tile)
            if allowed:
                successors.append(successor)
        return successors

    def _get_edge_cost(self, from_state: GameState, to_state: GameState) -> float:
        dr = abs(int(to_state.position[0]) - int(from_state.position[0]))
        dc = abs(int(to_state.position[1]) - int(from_state.position[1]))
        return float(DIAGONAL_COST if dr == 1 and dc == 1 else CARDINAL_COST)

    def update_vertex(self, state: GameState, state_hash: Any) -> None:
        """Update one vertex using the backward D* Lite recurrence."""
        self.states_by_hash[state_hash] = state.copy()
        goal = tuple(self.env.goal_pos) if self.env.goal_pos is not None else None

        if goal is not None and tuple(state.position) == goal:
            self.rhs_scores[state_hash] = 0.0
            self.predecessors[state_hash] = None
        elif not self._position_is_traversable(tuple(state.position)):
            self.rhs_scores[state_hash] = float("inf")
            self.predecessors.pop(state_hash, None)
        else:
            best_cost = float("inf")
            best_successor: Optional[Any] = None
            for successor in self._get_successors(state):
                successor_hash = game_state_key(successor)
                self.states_by_hash[successor_hash] = successor.copy()
                candidate = (
                    self._get_edge_cost(state, successor)
                    + self.g_scores.get(successor_hash, float("inf"))
                )
                if candidate < best_cost:
                    best_cost = candidate
                    best_successor = successor_hash
            self.rhs_scores[state_hash] = float(best_cost)
            if best_successor is None:
                self.predecessors.pop(state_hash, None)
            else:
                self.predecessors[state_hash] = best_successor

        self.open_set_hashes.pop(state_hash, None)
        self._queue_if_inconsistent(state, state_hash)

    def compute_shortest_path(self) -> bool:
        """Run textbook backward D* Lite until the current start is consistent."""
        self._last_compute_budget_exhausted = False
        if self.current_start_state is None:
            return False
        start_hash = game_state_key(self.current_start_state)
        self.states_by_hash[start_hash] = self.current_start_state.copy()
        iterations = 0

        while iterations < self.timeout:
            self._clean_open()
            start_key = self.calculate_key(self.current_start_state, start_hash)
            start_g = self.g_scores.get(start_hash, float("inf"))
            start_rhs = self.rhs_scores.get(start_hash, float("inf"))
            top_key = self._key_tuple(self.open_set[0]) if self.open_set else (float("inf"), float("inf"))

            if not (top_key < self._key_tuple(start_key) or start_rhs != start_g):
                return bool(start_g < float("inf"))
            if not self.open_set:
                return False

            queued_key = heapq.heappop(self.open_set)
            state_hash = queued_key.state_hash
            state = queued_key.state
            self.open_set_hashes.pop(state_hash, None)
            current_key = self.calculate_key(state, state_hash)

            if self._key_tuple(queued_key) < self._key_tuple(current_key):
                self._queue_if_inconsistent(state, state_hash)
            elif self.g_scores.get(state_hash, float("inf")) > self.rhs_scores.get(
                state_hash, float("inf")
            ):
                self.g_scores[state_hash] = self.rhs_scores.get(state_hash, float("inf"))
                for predecessor in self._get_successors(state):
                    self.update_vertex(predecessor, game_state_key(predecessor))
            else:
                self.g_scores[state_hash] = float("inf")
                self.update_vertex(state, state_hash)
                for predecessor in self._get_successors(state):
                    self.update_vertex(predecessor, game_state_key(predecessor))
            iterations += 1

        logger.warning("D* Lite reached its update budget (%d)", self.timeout)
        self._last_compute_budget_exhausted = True
        return False

    def _extract_path(self, start_state: GameState) -> List[Tuple[int, int]]:
        if self.env.goal_pos is None:
            return []
        current = start_state.copy()
        path = [tuple(current.position)]
        visited = {game_state_key(current)}
        max_steps = max(1, int(self.env.height) * int(self.env.width))

        for _ in range(max_steps):
            if tuple(current.position) == tuple(self.env.goal_pos):
                return path
            candidates: List[Tuple[float, Tuple[int, int], GameState]] = []
            for successor in self._get_successors(current):
                successor_hash = game_state_key(successor)
                score = (
                    self._get_edge_cost(current, successor)
                    + self.g_scores.get(successor_hash, float("inf"))
                )
                candidates.append((float(score), tuple(successor.position), successor))
            if not candidates:
                return []
            _score, _position, next_state = min(candidates, key=lambda item: (item[0], item[1]))
            next_hash = game_state_key(next_state)
            if next_hash in visited or not np.isfinite(_score):
                return []
            visited.add(next_hash)
            current = next_state
            path.append(tuple(current.position))
        return []

    def _clear_search(self) -> None:
        self.g_scores.clear()
        self.rhs_scores.clear()
        self.open_set.clear()
        self.open_set_hashes.clear()
        self.predecessors.clear()
        self.states_by_hash.clear()
        self.km = 0.0
        self.states_updated = 0

    def _fallback_to_astar(
        self,
        start_state: GameState,
    ) -> Tuple[bool, List[Tuple[int, int]], int]:
        from .validator import StateSpaceAStar

        self.used_fallback = True
        previous_state = self.env.state
        self.env.state = start_state.copy()
        try:
            solver = StateSpaceAStar(
                self.env,
                timeout=self.timeout,
                heuristic_mode=self.heuristic_mode,
                priority_options={"allow_diagonals": self.allow_diagonals},
                search_mode="astar",
            )
            return solver.solve()
        finally:
            self.env.state = previous_state

    def solve(self, start_state: GameState) -> Tuple[bool, List[Tuple[int, int]], int]:
        """Plan from ``start_state`` using D* Lite or the exact stateful oracle."""
        self._clear_search()
        self.used_fallback = False
        self.current_start_state = start_state.copy()
        self.last_start_state = start_state.copy()
        self._textbook_core_active = self._supports_textbook_core()

        if not self._textbook_core_active:
            result = self._fallback_to_astar(start_state)
            self.current_path = list(result[1] or [])
            self.last_opened_doors = set(start_state.opened_doors)
            self.last_pushed_blocks = set(start_state.pushed_blocks)
            return result

        goal_state = self._state_at(tuple(self.env.goal_pos), start_state)
        goal_hash = game_state_key(goal_state)
        self.rhs_scores[goal_hash] = 0.0
        self.states_by_hash[goal_hash] = goal_state.copy()
        self._queue_if_inconsistent(goal_state, goal_hash)
        self._grid_snapshot = np.asarray(self.env.grid).copy()

        success = self.compute_shortest_path()
        if not success and not self._last_compute_budget_exhausted:
            return False, [], len(self.g_scores)
        path = self._extract_path(start_state) if success else []
        if not path or path[-1] != self.env.goal_pos:
            logger.warning("Backward D* Lite could not certify a path; using state-space A*")
            return self._fallback_to_astar(start_state)

        self.current_path = path
        self.path_index = 0
        self.last_opened_doors = set(start_state.opened_doors)
        self.last_pushed_blocks = set(start_state.pushed_blocks)
        return True, path, len(self.g_scores)

    def _changed_grid_positions(self) -> List[Tuple[int, int]]:
        current = np.asarray(self.env.grid)
        if self._grid_snapshot is None or self._grid_snapshot.shape != current.shape:
            return [
                (row, col)
                for row in range(int(current.shape[0]))
                for col in range(int(current.shape[1]))
            ]
        return [tuple(map(int, pos)) for pos in np.argwhere(current != self._grid_snapshot)]

    def replan(self, current_state: GameState) -> Tuple[bool, List[Tuple[int, int]], int]:
        """Incrementally replan after movement or reversible edge-cost changes."""
        self.replans_count += 1
        self.states_updated = 0

        if not self._textbook_core_active or not self._supports_textbook_core():
            result = self._fallback_to_astar(current_state)
            self.current_path = list(result[1] or [])
            self.last_opened_doors = set(current_state.opened_doors)
            self.last_pushed_blocks = set(current_state.pushed_blocks)
            return result
        if self.current_start_state is None:
            return self.solve(current_state)

        old_start = self.current_start_state.copy()
        self.km += self._distance(tuple(old_start.position), tuple(current_state.position))
        self.last_start_state = old_start
        self.current_start_state = current_state.copy()

        changed = self._changed_grid_positions()
        affected: Set[Tuple[int, int]] = set(changed)
        for row, col in changed:
            for dr, dc in self._iter_action_deltas():
                neighbor = row + dr, col + dc
                if 0 <= neighbor[0] < self.env.height and 0 <= neighbor[1] < self.env.width:
                    affected.add(neighbor)
        for position in affected:
            state = self._state_at(position, current_state)
            self.update_vertex(state, game_state_key(state))
        self._grid_snapshot = np.asarray(self.env.grid).copy()

        success = self.compute_shortest_path()
        if not success and not self._last_compute_budget_exhausted:
            return False, [], self.states_updated
        path = self._extract_path(current_state) if success else []
        if not path or path[-1] != self.env.goal_pos:
            return self._fallback_to_astar(current_state)

        self.current_path = path
        self.path_index = 0
        return True, path, self.states_updated

    def needs_replan(self, current_state: GameState) -> bool:
        grid_changed = bool(self._changed_grid_positions())
        return bool(
            grid_changed
            or current_state.opened_doors != self.last_opened_doors
            or current_state.pushed_blocks != self.last_pushed_blocks
        )

    def _predecessor_state_candidates(
        self,
        state: GameState,
        *,
        pred_pos: Tuple[int, int],
        target_tile: int,
    ) -> List[GameState]:
        """Compatibility helper for auditing reverse inventory transitions."""
        base = state.copy()
        base.position = pred_pos
        candidates = [base]
        target_pos = tuple(state.position)
        if target_pos in base.opened_doors and target_tile in {
            int(SEMANTIC_PALETTE["DOOR_LOCKED"]),
            int(SEMANTIC_PALETTE["DOOR_BOMB"]),
            int(SEMANTIC_PALETTE["DOOR_BOSS"]),
        }:
            restored = base.copy()
            restored.opened_doors = set(restored.opened_doors) - {target_pos}
            if target_tile == int(SEMANTIC_PALETTE["DOOR_LOCKED"]):
                restored.keys += 1
            elif target_tile == int(SEMANTIC_PALETTE["DOOR_BOMB"]):
                restored.bomb_count += 1
            candidates.append(restored)
        return candidates

    def _has_consistent_goal_state(self) -> bool:
        """Compatibility diagnostic: report any finite consistent goal state."""
        if self.env.goal_pos is None:
            return False
        for state_hash in set(self.g_scores) | set(self.rhs_scores):
            state = self.states_by_hash.get(state_hash)
            position = tuple(state.position) if state is not None else tuple(state_hash[0])
            if position != tuple(self.env.goal_pos):
                continue
            g = self.g_scores.get(state_hash, float("inf"))
            rhs = self.rhs_scores.get(state_hash, float("inf"))
            if g < float("inf") and g == rhs:
                return True
        return False

    def _can_reach(
        self,
        from_state: GameState,
        to_state: GameState,
        target_tile: int,
    ) -> bool:
        allowed, _ = self.env.try_move_pure(from_state, to_state.position, target_tile)
        return bool(allowed)
