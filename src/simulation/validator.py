"""BLOCK VI: EXTERNAL VALIDATOR (Validation Orchestration & Facade).

Re-exports:
- `ZeldaLogicEnv`: Discrete state machine simulator (from `src.simulation.environment`)
- `StateSpaceAStar`: A* search over inventory state space (from `src.simulation.state_space_astar`)
- `GameState`: Player and world inventory state (from `src.simulation.state`)
- `ZeldaValidator`: Main orchestrator coordinating validation, sanity checks, and metrics.
"""

from __future__ import annotations

import heapq
import logging
import math
from collections import defaultdict, deque
from dataclasses import replace
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from src.core.definitions import (
    ID_TO_NAME,
    ROOM_HEIGHT,
    ROOM_WIDTH,
    SEMANTIC_PALETTE,
    parse_edge_type_tokens,
)
from src.simulation.edge_logic import (
    can_traverse_edge_type,
    edge_constraints_from_data,
    edge_type_from_data,
    graph_edge_state_key,
    graph_predecessors,
    graph_successors,
)
from src.simulation.environment import ZeldaLogicEnv
from src.simulation.graph_progression import (
    PhysicalGraphBridge,
    apply_graph_gate,
    required_switch_ids,
    starting_bomb_count,
    switch_provider_ids,
)
from src.simulation.state import (
    ACTION_DELTAS,
    BLOCKING_IDS,
    BRIDGE_FILL_IDS,
    CARDINAL_COST,
    CONDITIONAL_IDS,
    DIAGONAL_COST,
    EDGE_TYPE_MAP,
    PICKUP_IDS,
    PUSHABLE_IDS,
    TRANSITION_IDS,
    WALKABLE_IDS,
    WATER_IDS,
    Action,
    GameState,
    dominates,
    dynamic_geometry_key,
    game_state_key,
    has_pushed_block_at,
    is_push_destination_available,
    was_block_vacated,
)
from src.simulation.state_space_astar import StateSpaceAStar
from src.simulation.validation_helpers import (
    DiversityEvaluator,
    MetricsEngine,
    SanityChecker,
)
from src.simulation.validation_types import (
    BatchValidationResult,
    SolverDiagnostics,
    SolverOptions,
    ValidationResult,
)
from src.simulation import validator_rendering as _validator_rendering

logger = logging.getLogger(__name__)

GraphTransitionPath = Tuple[Tuple[Any, Any, str], ...]
GraphTransitionSpec = str | GraphTransitionPath

__all__ = [
    "ZeldaLogicEnv",
    "StateSpaceAStar",
    "ZeldaValidator",
    "ValidationMAPElitesEvaluator",
    "MultiPersonaAgent",
    "create_test_map",
    "GameState",
    "ValidationResult",
    "BatchValidationResult",
    "SolverDiagnostics",
    "SolverOptions",
]

# MODULE 6: MAIN VALIDATOR
# ==========================================

class ZeldaValidator:
    """
    Main validation orchestrator.
    
    Coordinates sanity checking, solving, and metrics calculation.
    """

    def __init__(self, calibration_map: np.ndarray = None):
        """
        Initialize validator.
        
        Args:
            calibration_map: Known-solvable map for calibration test
        """
        self.calibration_map = calibration_map
        self.is_calibrated = False

        if calibration_map is not None:
            self._run_calibration()

    def _run_calibration(self) -> bool:
        """
        Run calibration test on known-solvable map.
        
        This verifies the solver is working correctly.
        """
        if self.calibration_map is None:
            return True

        print("Running calibration test...")
        calib_result = self.validate_single(self.calibration_map)

        if not calib_result.is_solvable:
            raise RuntimeError(
                f"CALIBRATION FAILED: Known-solvable map was not solved! "
                f"Error: {calib_result.error_message}"
            )

        print(f"Calibration passed. Path length: {calib_result.path_length}")
        self.is_calibrated = True
        return True

    def validate_single(self, semantic_grid: np.ndarray,
                       render: bool = False,
                       persona_mode: str = "balanced",
                       graph=None,
                       room_to_node=None,
                       room_positions=None,
                       node_to_room=None,
                       room_puzzle_metadata: Optional[Mapping[str, Any]] = None,
                       solver_timeout: int = 200000,
                       run_dijkstra_fallback: bool = False,
                       verify_dijkstra_consistency: bool = False,
                       connection_realizations: Optional[Mapping] = None) -> ValidationResult:
        """
        Validate a single map.
        
        Args:
            semantic_grid: 2D numpy array of semantic IDs
            render: If True, show Pygame visualization
            persona_mode: Heuristic profile for solver (balanced, speedrunner, completionist)
            
        Returns:
            ValidationResult with all metrics
        """
        # Step 1: Sanity Check
        checker = SanityChecker(semantic_grid)
        is_valid, errors = checker.check_all()

        if not is_valid:
            return ValidationResult(
                is_solvable=False,
                is_valid_syntax=False,
                reachability=0.0,
                path_length=0,
                backtracking_score=0.0,
                logical_errors=errors,
                error_message="; ".join(errors),
                termination_status="invalid",
            )

        env_kwargs = {
            "render_mode": render,
            "graph": graph,
            "room_to_node": room_to_node,
            "room_positions": room_positions,
            "node_to_room": node_to_room,
            "room_puzzle_metadata": room_puzzle_metadata,
            "connection_realizations": connection_realizations,
        }
        typed_item_contract = False
        if graph is not None and hasattr(graph, "edges"):
            try:
                for _source, _target, edge_data in graph.edges(data=True):
                    if not isinstance(edge_data, Mapping):
                        continue
                    if any(
                        str(edge_data.get(field, "") or "").strip()
                        for field in ("item_required", "protection_item_id")
                    ):
                        typed_item_contract = True
                        break
            except (AttributeError, TypeError, ValueError):
                typed_item_contract = False
        graph_guided_primary = bool(
            graph is not None
            and room_to_node
            and room_positions
            and not typed_item_contract
            and connection_realizations is None
        )

        def _run_solver(search_mode: str) -> Tuple[bool, List[Tuple[int, int]], SolverDiagnostics]:
            local_env = ZeldaLogicEnv(semantic_grid, **env_kwargs)
            try:
                use_hybrid_frontend = bool(search_mode == "astar" and graph_guided_primary)
                priority_options = {
                    "allow_diagonals": False,
                    "representation": "hybrid" if use_hybrid_frontend else "tile",
                    "rules_profile": "vglc_strict",
                    "enable_hierarchical": bool(use_hybrid_frontend),
                }
                solver = StateSpaceAStar(
                    local_env,
                    timeout=int(max(1, solver_timeout)),
                    heuristic_mode=persona_mode,
                    priority_options=priority_options,
                    search_mode=search_mode,
                )
                if use_hybrid_frontend:
                    success_i, path_i, states_i = solver.solve()
                    solver_termination = str(
                        getattr(solver, "last_termination_status", "incomplete")
                    )
                    solver_failure = str(getattr(solver, "last_failure_reason", "") or "")
                    diagnostics_i = SolverDiagnostics(
                        success=bool(success_i),
                        states_explored=int(states_i or 0),
                        failure_reason=solver_failure,
                        path_length=max(0, len(path_i or []) - 1),
                        path_cost=(
                            float(solver.last_solution_cost)
                            if getattr(solver, "last_solution_cost", None) is not None
                            else None
                        ),
                        final_inventory=dict(
                            getattr(solver, "last_solution_inventory", None) or {}
                        ),
                        termination_status=solver_termination,
                    )
                else:
                    if (
                        search_mode == "astar"
                        and bool(np.any(semantic_grid == int(SEMANTIC_PALETTE["BLOCK"])))
                    ):
                        preflight_grid = np.array(semantic_grid, copy=True)
                        preflight_grid[preflight_grid == int(SEMANTIC_PALETTE["BLOCK"])] = int(
                            SEMANTIC_PALETTE["WALL"]
                        )
                        preflight_env = ZeldaLogicEnv(preflight_grid, **env_kwargs)
                        try:
                            preflight_solver = StateSpaceAStar(
                                preflight_env,
                                timeout=int(max(1, solver_timeout)),
                                heuristic_mode=persona_mode,
                                priority_options=priority_options,
                                search_mode=search_mode,
                            )
                            pre_ok, pre_path, pre_diag = preflight_solver.solve_with_verified_diagnostics()
                        finally:
                            try:
                                preflight_env.close()
                            except Exception:
                                logger.debug("Failed to close static-block preflight environment.", exc_info=True)
                        if pre_ok and str(pre_diag.termination_status) == "solved":
                            replay_ok, _replay_err, replay_state, replay_cost = solver.verify_position_path(pre_path)
                            reported_cost = getattr(pre_diag, "path_cost", None)
                            if (
                                replay_ok
                                and replay_cost is not None
                                and reported_cost is not None
                                and not math.isclose(
                                    float(replay_cost),
                                    float(reported_cost),
                                    rel_tol=1e-9,
                                    abs_tol=1e-9,
                                )
                            ):
                                replay_ok = False
                            if replay_ok:
                                pre_diag.route_replay_status = "verified"
                                pre_diag.route_replay_error = ""
                                pre_diag.route_replay_path_cost = replay_cost
                                if replay_state is not None:
                                    pre_diag.final_inventory = {
                                        'keys': replay_state.keys,
                                        'bomb_count': replay_state.bomb_count,
                                        'has_bomb': replay_state.has_bomb,
                                        'has_boss_key': replay_state.has_boss_key,
                                        'has_item': replay_state.has_item,
                                        'item_names': sorted(
                                            str(name) for name in replay_state.item_names
                                        ),
                                        'current_floor': replay_state.current_floor,
                                        'doors_opened': len(replay_state.opened_doors),
                                        'items_collected': len(replay_state.collected_items),
                                    }
                                return True, list(pre_path), pre_diag
                    success_i, path_i, diagnostics_i = solver.solve_with_verified_diagnostics()
                if success_i and diagnostics_i.route_replay_status != "verified":
                    replay_ok, replay_error, replay_final_state, replay_path_cost = (
                        solver.verify_position_path(path_i)
                    )
                    diagnostics_i.route_replay_path_cost = replay_path_cost
                    reported_path_cost = getattr(diagnostics_i, "path_cost", None)
                    if (
                        replay_ok
                        and replay_path_cost is not None
                        and reported_path_cost is not None
                        and not math.isclose(
                            float(replay_path_cost),
                            float(reported_path_cost),
                            rel_tol=1e-9,
                            abs_tol=1e-9,
                        )
                    ):
                        replay_ok = False
                        replay_error = (
                            "replayed route cost does not match the solver cost: "
                            f"replay={float(replay_path_cost):.12g}, "
                            f"solver={float(reported_path_cost):.12g}"
                        )
                    diagnostics_i.route_replay_status = (
                        "verified" if replay_ok else "failed"
                    )
                    diagnostics_i.route_replay_error = str(replay_error or "")
                    if replay_ok and replay_final_state is not None:
                        diagnostics_i.final_inventory = {
                            'keys': replay_final_state.keys,
                            'bomb_count': replay_final_state.bomb_count,
                            'has_bomb': replay_final_state.has_bomb,
                            'has_boss_key': replay_final_state.has_boss_key,
                            'has_item': replay_final_state.has_item,
                            'item_names': sorted(
                                str(name) for name in replay_final_state.item_names
                            ),
                            'current_floor': replay_final_state.current_floor,
                            'doors_opened': len(replay_final_state.opened_doors),
                            'items_collected': len(replay_final_state.collected_items),
                        }
                    if not replay_ok:
                        success_i = False
                        path_i = []
                        diagnostics_i.success = False
                        diagnostics_i.failure_reason = (
                            "Post-route replay rejected the reconstructed solution: "
                            f"{replay_error}"
                        )
                        diagnostics_i.termination_status = "route_replay_failed"
                return bool(success_i), list(path_i or []), diagnostics_i
            finally:
                try:
                    local_env.close()
                except Exception:
                    logger.debug("Failed to close local validation environment.", exc_info=True)

        # Step 2: Run primary A* oracle.
        success, path, diagnostics = _run_solver("astar")
        primary_failure = str(getattr(diagnostics, "failure_reason", "") or "")
        solver_used = "hybrid_astar" if graph_guided_primary else "astar"

        # Step 3: Exact fallback. Stateful puzzle mechanics increased
        # path-dependence, so use uniform-cost search as the completeness
        # preserving fallback when heuristic A* underperforms.
        primary_status = str(getattr(diagnostics, "termination_status", "unknown"))
        fallback_eligible = primary_status == "route_replay_failed"
        if not success and bool(run_dijkstra_fallback) and fallback_eligible:
            primary_states = int(getattr(diagnostics, "states_explored", 0) or 0)
            remaining_budget = max(0, int(max(1, solver_timeout)) - primary_states)
            if remaining_budget <= 0:
                fallback_success = False
                fallback_path = []
                fallback_diagnostics = diagnostics
            else:
                original_timeout = solver_timeout
                try:
                    solver_timeout = remaining_budget
                    fallback_success, fallback_path, fallback_diagnostics = _run_solver("dijkstra")
                finally:
                    solver_timeout = original_timeout
                fallback_diagnostics.states_explored = (
                    int(fallback_diagnostics.states_explored) + primary_states
                )
            if fallback_success:
                success = True
                path = fallback_path
                diagnostics = fallback_diagnostics
                solver_used = "dijkstra_fallback"
            elif fallback_diagnostics is not diagnostics:
                diagnostics = fallback_diagnostics
                solver_used = "astar_then_dijkstra"

        if not success:
            fallback_diagnostics = diagnostics
            exact_states = int(
                getattr(
                    fallback_diagnostics,
                    "states_explored",
                    0,
                )
                or 0
            )
            final_status = str(
                getattr(fallback_diagnostics, "termination_status", "unknown")
            )
            from src.simulation.search_status import (
                proven_unsolvable_from_outcome,
            )

            proven_unsolvable = proven_unsolvable_from_outcome(False, final_status)
            return ValidationResult(
                is_solvable=False,
                is_valid_syntax=True,
                reachability=0.0,
                path_length=0,
                backtracking_score=0.0,
                logical_errors=["Exact tile-state oracle failed to find path"],
                error_message=(
                    str(getattr(diagnostics, "failure_reason", "") or final_status)
                ),
                solver_used=solver_used,
                primary_solver_solved=False,
                primary_solver_error=primary_failure,
                states_explored=exact_states,
                termination_status=final_status,
                proven_unsolvable=proven_unsolvable,
                final_inventory=dict(getattr(diagnostics, "final_inventory", {}) or {}),
                route_replay_status=str(
                    getattr(diagnostics, "route_replay_status", "not_run")
                ),
                route_replay_error=str(
                    getattr(diagnostics, "route_replay_error", "") or ""
                ),
                route_replay_path_cost=getattr(
                    diagnostics,
                    "route_replay_path_cost",
                    None,
                ),
            )

        consistency_status = "not_requested"
        solver_consistent: Optional[bool] = None
        consistency_path_length: Optional[int] = None
        consistency_path_cost: Optional[float] = None
        consistency_states_explored = 0
        if bool(verify_dijkstra_consistency):
            reference_success, reference_path, reference_diagnostics = _run_solver(
                "dijkstra"
            )
            consistency_states_explored = int(
                getattr(reference_diagnostics, "states_explored", 0) or 0
            )
            if reference_success:
                consistency_path_length = max(0, len(reference_path) - 1)
                reference_cost = getattr(reference_diagnostics, "path_cost", None)
                primary_cost = getattr(diagnostics, "path_cost", None)
                consistency_path_cost = (
                    float(reference_cost) if reference_cost is not None else None
                )
                if primary_cost is None or reference_cost is None:
                    consistency_status = "indeterminate_missing_path_cost"
                else:
                    solver_consistent = math.isclose(
                        float(primary_cost),
                        float(reference_cost),
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    )
                    consistency_status = (
                        "consistent" if solver_consistent else "path_cost_mismatch"
                    )
            elif str(
                getattr(reference_diagnostics, "termination_status", "unknown")
            ) == "budget_exhausted":
                consistency_status = "indeterminate_budget_exhausted"
            else:
                # A* found a solution but exhaustive uniform-cost search did
                # not. This is a solver-contract disagreement, not proof that
                # the map itself is unsolvable.
                solver_consistent = False
                consistency_status = "reachability_mismatch"

        # Step 4: Recreate environment for metrics/rendering on the winning path.
        env = ZeldaLogicEnv(semantic_grid, **env_kwargs)
        try:
            reachability = MetricsEngine.calculate_reachability(env, path)
            backtracking = MetricsEngine.calculate_backtracking(path)
            logical_errors = MetricsEngine.find_logical_errors(env, path)

            if solver_used != "astar" and primary_failure:
                logical_errors = list(logical_errors) + [
                    f"primary_astar_failed: {primary_failure}",
                    "exact_fallback_solver_used: dijkstra",
                ]

            if render:
                self._visualize_solution(env, path)
        finally:
            env.close()

        path_positions = {
            (int(position[0]), int(position[1]))
            for position in path
            if 0 <= int(position[0]) < int(semantic_grid.shape[0])
            and 0 <= int(position[1]) < int(semantic_grid.shape[1])
        }
        path_interactions = {
            "small_keys_collected": sum(
                int(int(semantic_grid[row, col]) == int(SEMANTIC_PALETTE['KEY_SMALL']))
                for row, col in path_positions
            ),
            "boss_keys_collected": sum(
                int(int(semantic_grid[row, col]) == int(SEMANTIC_PALETTE['KEY_BOSS']))
                for row, col in path_positions
            ),
            "key_items_collected": sum(
                int(int(semantic_grid[row, col]) == int(SEMANTIC_PALETTE['KEY_ITEM']))
                for row, col in path_positions
            ),
            "locked_doors_traversed": sum(
                int(int(semantic_grid[row, col]) == int(SEMANTIC_PALETTE['DOOR_LOCKED']))
                for row, col in path_positions
            ),
            "boss_doors_traversed": sum(
                int(int(semantic_grid[row, col]) == int(SEMANTIC_PALETTE['DOOR_BOSS']))
                for row, col in path_positions
            ),
        }
        final_inventory = dict(getattr(diagnostics, "final_inventory", {}) or {})
        for key, value in path_interactions.items():
            final_inventory.setdefault(f"path_{key}", int(value))

        return ValidationResult(
            is_solvable=True,
            is_valid_syntax=True,
            reachability=reachability,
            path_length=max(0, len(path) - 1),
            path_cost=(
                float(diagnostics.path_cost)
                if diagnostics.path_cost is not None
                else None
            ),
            backtracking_score=backtracking,
            logical_errors=logical_errors,
            path=path,
            solver_used=solver_used,
            primary_solver_solved=bool(solver_used in {"astar", "hybrid_astar"}),
            primary_solver_error=primary_failure if solver_used != "astar" else "",
            states_explored=int(getattr(diagnostics, "states_explored", 0) or 0),
            termination_status="solved",
            final_inventory=final_inventory,
            path_interactions=path_interactions,
            route_replay_status=str(
                getattr(diagnostics, "route_replay_status", "not_run")
            ),
            route_replay_error=str(
                getattr(diagnostics, "route_replay_error", "") or ""
            ),
            route_replay_path_cost=getattr(
                diagnostics,
                "route_replay_path_cost",
                None,
            ),
            solver_consistency_status=consistency_status,
            solver_consistent=solver_consistent,
            solver_consistency_path_length=consistency_path_length,
            solver_consistency_path_cost=consistency_path_cost,
            solver_consistency_states_explored=consistency_states_explored,
        )

    def check_soft_locks(
        self,
        semantic_grid: np.ndarray,
        sample_count: int = 10,
        deterministic: bool = True,
        graph=None,
        room_to_node=None,
        room_positions=None,
        node_to_room=None,
    ) -> Tuple[bool, List[str]]:
        """
        Detect soft-lock traps (one-way rooms where player gets stuck).
        
        ALGORITHM:
        1. Find all reachable floor positions from START
        2. Randomly sample N positions
        3. For each position, test if GOAL is still reachable from there
        4. If any position has no path to goal, it's a soft-lock trap
        
        This detects scenarios like:
        - One-way doors (ledges, shutters) that trap the player
        - Rooms where the player can walk in but door closes with no key
        - Unreachable key islands
        
        Args:
            semantic_grid: The map to check
            sample_count: How many random positions to test (default: 10)
            
        Returns:
            (is_safe, trap_descriptions): True if no soft-locks found, plus list of trap locations
        """
        if deterministic:
            return self.check_soft_locks_deterministic(
                semantic_grid,
                graph=graph,
                room_to_node=room_to_node,
                room_positions=room_positions,
                node_to_room=node_to_room,
            )

        import random

        # Create environment
        env = ZeldaLogicEnv(semantic_grid, render_mode=False)

        if env.goal_pos is None or env.start_pos is None:
            return False, ["No start or goal position defined"]

        # Step 1: Get all reachable walkable tiles from START
        solver = StateSpaceAStar(env, timeout=50000)
        success, winning_path, _ = solver.solve()

        if not success:
            # If START->GOAL already fails, map is unsolvable (not a soft-lock issue)
            return True, []  # We don't count this as a soft-lock (it's a regular failure)

        # Get all walkable positions
        reachable_spots = []
        h, w = semantic_grid.shape
        for r in range(h):
            for c in range(w):
                tile = semantic_grid[r, c]
                if tile in WALKABLE_IDS or tile in CONDITIONAL_IDS:
                    reachable_spots.append((r, c))

        # Sample random positions to test (limit to reasonable count)
        if len(reachable_spots) > sample_count:
            test_positions = random.sample(reachable_spots, sample_count)
        else:
            test_positions = reachable_spots[:sample_count]

        # Add positions from the winning path (these MUST be safe)
        test_positions.extend(winning_path[::len(winning_path)//3] if len(winning_path) > 3 else winning_path)

        # Step 2: For each test position, check if we can still reach GOAL
        trap_positions = []

        for test_pos in test_positions:
            # Create a modified environment starting from test_pos
            # We need to simulate "player teleported here, can they escapex"

            # Simple heuristic: Check if test_pos is on the winning path
            # If not on winning path and isolated, it might be a trap

            # Create new env with modified start
            test_env = ZeldaLogicEnv(semantic_grid, render_mode=False)
            test_env.start_pos = test_pos
            test_env.reset()

            test_solver = StateSpaceAStar(test_env, timeout=10000)
            can_escape, _, _ = test_solver.solve()

            if not can_escape:
                # This position cannot reach the goal - potential soft-lock!
                trap_positions.append(test_pos)

            test_env.close()

        env.close()

        # Step 3: Report results
        if trap_positions:
            trap_descriptions = [
                f"Soft-lock trap at position {pos}: player cannot reach goal from here"
                for pos in trap_positions[:5]  # Limit output
            ]
            return False, trap_descriptions

        return True, []

    def check_soft_locks_deterministic(self, semantic_grid: np.ndarray,
                                        graph=None, room_to_node=None,
                                        room_positions=None,
                                        node_to_room=None,
                                        room_puzzle_metadata: Optional[Mapping[str, Any]] = None,
                                        solver_timeout: int = 200000) -> Tuple[bool, List[str]]:
        """Deterministic soft-lock detection via reverse reachability.

        Unlike :meth:`check_soft_locks` which uses random sampling,
        this method uses bidirectional BFS (forward from START, backward
        from GOAL with reversed one-way edges) to *prove* the existence
        or absence of trap regions.

        Reference: Holzer & Schwoon (2011) - Reachability vs. Safety.

        Args:
            semantic_grid: The map to check.
            graph, room_to_node, room_positions, node_to_room:
                Optional graph topology for graph-level analysis.

        Returns:
            (is_safe, trap_descriptions): True if no proven traps found.
        """
        env = ZeldaLogicEnv(
            semantic_grid, render_mode=False,
            graph=graph, room_to_node=room_to_node,
            room_positions=room_positions, node_to_room=node_to_room,
            room_puzzle_metadata=room_puzzle_metadata,
        )
        if env.goal_pos is None or env.start_pos is None:
            return False, ['No start or goal position defined']

        solver = StateSpaceAStar(
            env,
            timeout=int(max(1, solver_timeout)),
            priority_options={
                "allow_diagonals": False,
                "representation": "tile",
                "rules_profile": "vglc_strict",
            },
        )
        traps = solver.find_proven_traps()

        descriptions: List[str] = []
        graph_traps = traps.get('graph_traps', set())
        grid_traps = traps.get('grid_traps', set())

        if graph_traps:
            descriptions.append(
                f'{len(graph_traps)} graph-level trap node(s): '
                f'{", ".join(str(n) for n in sorted(graph_traps, key=str)[:5])}'
            )
        if grid_traps:
            sample = sorted(grid_traps)[:5]
            descriptions.append(
                f'{len(grid_traps)} grid-level trap tile(s), e.g. {sample}'
            )

        env.close()
        is_safe = len(graph_traps) == 0 and len(grid_traps) == 0
        return is_safe, descriptions

    def validate_batch(self, grids: List[np.ndarray],
                      verbose: bool = True,
                      persona_mode: str = "balanced") -> BatchValidationResult:
        """
        Validate a batch of maps.
        
        Args:
            grids: List of semantic grids to validate
            verbose: Print progress
            
        Returns:
            BatchValidationResult with aggregate metrics
        """
        results = []
        solvable_count = 0
        valid_count = 0

        total_reachability = 0.0
        total_path_length = 0
        total_backtracking = 0.0

        paths = []

        for i, grid in enumerate(grids):
            if verbose and (i + 1) % 10 == 0:
                print(f"Validating {i + 1}/{len(grids)}...")

            single_result = self.validate_single(grid, render=False, persona_mode=persona_mode)
            results.append(single_result)

            if single_result.is_valid_syntax:
                valid_count += 1

            if single_result.is_solvable:
                solvable_count += 1
                total_reachability += single_result.reachability
                total_path_length += single_result.path_length
                total_backtracking += single_result.backtracking_score
                paths.append(single_result.path)

        # Calculate averages
        n = len(grids)
        n_solvable = max(1, solvable_count)

        # Calculate diversity
        diversity = DiversityEvaluator.batch_diversity(grids)

        return BatchValidationResult(
            total_maps=n,
            valid_syntax_count=valid_count,
            solvable_count=solvable_count,
            solvability_rate=solvable_count / n if n > 0 else 0.0,
            avg_reachability=total_reachability / n_solvable,
            avg_path_length=total_path_length / n_solvable,
            avg_backtracking=total_backtracking / n_solvable,
            diversity_score=diversity,
            individual_results=results
        )

    def validate_batch_multi_persona(self, grids: List[np.ndarray],
                                     personas: List[str] = None,
                                     verbose: bool = True) -> Dict[str, BatchValidationResult]:
        """
        Validate a batch of maps using multiple persona modes.
        
        This evaluates the same maps with different heuristic profiles:
        - Speedrunner: Prefers direct routes, ignores optional pickups
        - Completionist: Explores all rooms, collects all items
        - Balanced: Standard pathfinding
        
        Args:
            grids: List of semantic grids to validate
            personas: List of persona modes (default: all three)
            verbose: Print progress
            
        Returns:
            Dict mapping persona_mode -> BatchValidationResult
        """
        if personas is None:
            personas = ["speedrunner", "balanced", "completionist"]

        results_by_persona = {}

        for persona in personas:
            if verbose:
                print(f"\n=== Evaluating with '{persona}' persona ===")

            batch_result = self.validate_batch(
                grids,
                verbose=verbose,
                persona_mode=persona
            )

            results_by_persona[persona] = batch_result

            if verbose:
                print(f"{persona.capitalize()} Results:")
                print(f"  Solvability: {batch_result.solvability_rate:.1%}")
                print(f"  Avg Path Length: {batch_result.avg_path_length:.1f}")
                print(f"  Avg Reachability: {batch_result.avg_reachability:.1%}")

        return results_by_persona

    def _visualize_solution(self, env: ZeldaLogicEnv, path: List[Tuple[int, int]]):
        """Show animated solution using Pygame."""
        import time

        try:
            import pygame
        except ImportError:
            print("Pygame not available for visualization")
            return

        env.reset()
        env.render()
        time.sleep(0.5)

        for i, pos in enumerate(path[1:], 1):
            # Determine action
            prev = path[i - 1]
            dr = pos[0] - prev[0]
            dc = pos[1] - prev[1]

            if dr == -1:
                action = 0
            elif dr == 1:
                action = 1
            elif dc == -1:
                action = 2
            else:
                action = 3

            env.step(action)
            env.render()

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:  # pylint: disable=no-member
                    return

            time.sleep(0.1)

        # Show final state
        time.sleep(2)


# ==========================================
# MODULE 7: GRAPH-GUIDED VALIDATOR
# ==========================================

# Compatibility re-exports. Existing imports and serialized references through
# src.simulation.validator continue to resolve to the canonical class objects.
from src.simulation.graph_validator import (  # noqa: E402, F401
    GraphGuidedValidator,
    GraphValidationResult,
)


# ==========================================
# MODULE 8: ADVANCED ANALYTICS
# ==========================================

class ValidationMAPElitesEvaluator:
    """
    Validator-local diversity heatmap helper.

    This is intentionally separate from the canonical runtime evaluator in
    `src.simulation.map_elites`.
    """

    def __init__(self, bins: int = 10, danger_cap: int = 50):
        self.bins = bins
        self.danger_cap = danger_cap
        self.heatmap = np.zeros((bins, bins), dtype=int)

    def _find_tile(self, grid: np.ndarray, target_id: int) -> Optional[Tuple[int, int]]:
        positions = np.where(grid == target_id)
        if len(positions[0]) > 0:
            return (int(positions[0][0]), int(positions[1][0]))
        return None

    def evaluate_batch(self, results: List[ValidationResult], grids: List[np.ndarray]) -> Tuple[np.ndarray, List[float], List[float]]:
        self.heatmap[:, :] = 0
        x_coords: List[float] = []
        y_coords: List[float] = []

        for res, grid in zip(results, grids):
            if not res.is_solvable or grid.size == 0 or not res.path:
                continue

            start_pos = self._find_tile(grid, SEMANTIC_PALETTE['START']) or res.path[0]
            goal_pos = self._find_tile(grid, SEMANTIC_PALETTE['TRIFORCE']) or res.path[-1]

            if start_pos is None or goal_pos is None:
                continue

            linearity = MetricsEngine.calculate_linearity(res.path, start_pos, goal_pos)
            enemy_count = int(np.sum(grid == SEMANTIC_PALETTE['ENEMY']))
            danger = min(1.0, enemy_count / max(1, self.danger_cap))

            x_bin = min(self.bins - 1, int(linearity * self.bins))
            y_bin = min(self.bins - 1, int(danger * self.bins))
            self.heatmap[y_bin, x_bin] += 1

            x_coords.append(linearity)
            y_coords.append(danger)

        return self.heatmap.copy(), x_coords, y_coords


# Backward compatibility alias. Prefer ValidationMAPElitesEvaluator in new code.
MAPElitesEvaluator = ValidationMAPElitesEvaluator


class MultiPersonaAgent:
    """Convenience wrapper for persona-specific solver settings."""

    def __init__(self, env: ZeldaLogicEnv):
        self.env = env

    def solve_speedrunner(self) -> Tuple[bool, List[Tuple[int, int]], int]:
        return StateSpaceAStar(self.env, heuristic_mode="speedrunner").solve()

    def solve_completionist(self) -> Tuple[bool, List[Tuple[int, int]], int]:
        return StateSpaceAStar(self.env, heuristic_mode="completionist").solve()

    def solve_balanced(self) -> Tuple[bool, List[Tuple[int, int]], int]:
        return StateSpaceAStar(self.env, heuristic_mode="balanced").solve()


# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def create_test_map() -> np.ndarray:
    """Create a simple test map for validation testing."""
    W = SEMANTIC_PALETTE['WALL']
    F = SEMANTIC_PALETTE['FLOOR']
    S = SEMANTIC_PALETTE['START']
    T = SEMANTIC_PALETTE['TRIFORCE']
    K = SEMANTIC_PALETTE['KEY_SMALL']
    L = SEMANTIC_PALETTE['DOOR_LOCKED']

    # Demo map with 11 rows x 16 columns for standalone validator smoke tests.
    # Player starts at top-left, gets key, unlocks door, reaches triforce
    # Path: Start -> Key -> Door -> Triforce
    test_map = np.array([
        [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
        [W, S, F, F, F, F, F, F, F, F, F, F, F, F, F, W],
        [W, F, F, F, F, F, F, F, F, F, F, F, F, F, F, W],
        [W, F, F, K, F, F, F, F, F, F, F, F, F, F, F, W],
        [W, F, F, F, F, F, F, F, F, F, F, F, F, F, F, W],
        [W, F, F, F, F, F, W, W, L, W, W, F, F, F, F, W],
        [W, F, F, F, F, F, W, F, F, F, W, F, F, F, F, W],
        [W, F, F, F, F, F, W, F, T, F, W, F, F, F, F, W],
        [W, F, F, F, F, F, W, F, F, F, W, F, F, F, F, W],
        [W, F, F, F, F, F, W, W, W, W, W, F, F, F, F, W],
        [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
    ], dtype=np.int64)

    return test_map


# ==========================================
# MAIN ENTRY POINT
# ==========================================

if __name__ == "__main__":
    print("=== ZAVE: Zelda AI Validation Environment ===\n")

    # Create test map
    demo_map = create_test_map()
    print("Created test map (11 rows x 16 cols)")
    print(f"Start: {np.where(demo_map == SEMANTIC_PALETTE['START'])}")
    print(f"Goal: {np.where(demo_map == SEMANTIC_PALETTE['TRIFORCE'])}")

    # Run validation
    validator = ZeldaValidator()

    print("\n--- Validating Test Map ---")
    result = validator.validate_single(demo_map, render=False)

    print(f"Solvable: {result.is_solvable}")
    print(f"Valid Syntax: {result.is_valid_syntax}")
    print(f"Path Length: {result.path_length}")
    print(f"Reachability: {result.reachability:.2%}")
    print(f"Backtracking: {result.backtracking_score:.2%}")

    if result.logical_errors:
        print(f"Logical Errors: {result.logical_errors}")

    if result.error_message:
        print(f"Error: {result.error_message}")

    print("\n--- Test Complete ---")
