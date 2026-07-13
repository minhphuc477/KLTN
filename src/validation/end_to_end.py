"""Publication-facing validation contract for generated Zelda dungeons.

The stages deliberately use different representations. A graph oracle proves
mission progression before generation, connection records prove that graph
edges survived spatial realization, and the state-space tile oracle proves the
final playable artifact. LogicNet is useful evidence, but never a hard proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np

from src.core.definitions import SEMANTIC_PALETTE


@dataclass(frozen=True)
class ValidationStageEvidence:
    """One independently interpretable stage in the validation contract."""

    name: str
    passed: Optional[bool]
    status: str
    applicable: bool = True
    exact: bool = True
    details: Optional[Mapping[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "status": self.status,
            "applicable": bool(self.applicable),
            "exact": bool(self.exact),
            "details": dict(self.details or {}),
        }


@dataclass(frozen=True)
class EndToEndValidationReport:
    """Combined hard-evidence report; advisory model scores stay separate."""

    representation: ValidationStageEvidence
    graph_progression: ValidationStageEvidence
    global_state_progression: ValidationStageEvidence
    spatial_realization: ValidationStageEvidence
    tile_solvability: ValidationStageEvidence
    route_replay: ValidationStageEvidence
    solver_consistency: ValidationStageEvidence
    logicnet_agreement: Optional[bool] = None
    advisory_metrics: Optional[Mapping[str, Any]] = None

    @property
    def hard_stages(self) -> tuple[ValidationStageEvidence, ...]:
        return (
            self.representation,
            self.graph_progression,
            self.global_state_progression,
            self.spatial_realization,
            self.tile_solvability,
            self.route_replay,
            self.solver_consistency,
        )

    @property
    def accepted(self) -> bool:
        return all(
            stage.passed is True
            for stage in self.hard_stages
            if stage.applicable
        )

    @property
    def failed_stages(self) -> list[str]:
        return [
            stage.name
            for stage in self.hard_stages
            if stage.applicable and stage.passed is False
        ]

    @property
    def indeterminate_stages(self) -> list[str]:
        return [
            stage.name
            for stage in self.hard_stages
            if stage.applicable and stage.passed is None
        ]

    def require_accepted(self) -> None:
        if self.accepted:
            return
        raise RuntimeError(
            "End-to-end dungeon validation did not produce an accepted artifact: "
            f"failed={self.failed_stages}, indeterminate={self.indeterminate_stages}."
        )

    def to_metrics(self) -> Dict[str, Any]:
        return {
            "end_to_end_validation_accepted": bool(self.accepted),
            "end_to_end_validation_failed_stages": self.failed_stages,
            "end_to_end_validation_indeterminate_stages": self.indeterminate_stages,
            "end_to_end_validation_stages": {
                stage.name: stage.to_dict() for stage in self.hard_stages
            },
            "end_to_end_logicnet_hard_agreement": self.logicnet_agreement,
            "end_to_end_advisory_metrics": dict(self.advisory_metrics or {}),
        }


def validate_grid_representation(grid: np.ndarray) -> ValidationStageEvidence:
    """Validate the semantic-grid boundary before interpreting game rules."""
    array = np.asarray(grid)
    failures: list[str] = []
    if array.ndim != 2:
        failures.append(f"rank={array.ndim}, expected 2")
    if array.size == 0:
        failures.append("empty grid")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
            failures.append(f"non-integral dtype {array.dtype}")
        elif not np.equal(array, np.floor(array)).all():
            failures.append("fractional tile IDs")

    valid_ids = {int(value) for value in SEMANTIC_PALETTE.values()}
    observed_ids: set[int] = set()
    if array.size and np.issubdtype(array.dtype, np.number):
        flattened = array.reshape(-1)
        if np.issubdtype(array.dtype, np.floating):
            flattened = flattened[np.isfinite(flattened)]
        observed_ids = {int(value) for value in flattened}
    invalid_ids = sorted(observed_ids - valid_ids)
    if invalid_ids:
        failures.append(f"unknown tile IDs {invalid_ids}")

    start_count = int(np.sum(array == int(SEMANTIC_PALETTE["START"]))) if array.ndim == 2 else 0
    goal_count = int(np.sum(array == int(SEMANTIC_PALETTE["TRIFORCE"]))) if array.ndim == 2 else 0
    if start_count != 1:
        failures.append(f"start count {start_count}, expected 1")
    if goal_count < 1:
        failures.append("no goal tile")
    return ValidationStageEvidence(
        name="representation",
        passed=not failures,
        status="valid" if not failures else "invalid",
        details={
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "start_count": start_count,
            "goal_count": goal_count,
            "failures": failures,
        },
    )


def build_end_to_end_validation_report(
    *,
    dungeon_grid: np.ndarray,
    graph_validation: Optional[Mapping[str, Any]],
    spatial_validation: Optional[Mapping[str, Any]],
    tile_validation: Optional[Mapping[str, Any]],
    global_state_validation: Optional[Mapping[str, Any]] = None,
    logicnet_agreement: Optional[bool] = None,
    advisory_metrics: Optional[Mapping[str, Any]] = None,
) -> EndToEndValidationReport:
    """Normalize existing validator outputs into one fail-closed contract."""
    graph = dict(graph_validation or {})
    graph_status = str(graph.get("termination_status", "not_run"))
    graph_applicable = graph_status != "not_applicable_missing_roles"
    graph_solved = graph.get("solvable")
    all_rooms = graph.get("all_rooms_reachable")
    graph_replay_status = str(graph.get("route_replay_status", "not_run"))
    if not graph_applicable:
        graph_passed: Optional[bool] = None
    elif graph_solved is None or all_rooms is None:
        graph_passed = None
    elif bool(graph_solved) and graph_replay_status == "not_run":
        graph_passed = None
    elif graph_replay_status == "failed":
        graph_passed = False
    else:
        graph_passed = bool(
            graph_solved
            and all_rooms
            and graph_replay_status == "verified"
        )

    global_state = dict(global_state_validation or {})
    global_state_applicable = bool(global_state)
    global_state_status = str(
        global_state.get("termination_status", "not_applicable")
    )
    if not global_state_applicable:
        global_state_passed: Optional[bool] = None
    elif not bool(global_state.get("complete", False)):
        # An exhausted finite-state search is indeterminate, not evidence of
        # either validity or invalidity.
        global_state_passed = None
    elif "accepted" not in global_state:
        global_state_passed = None
    else:
        global_state_passed = bool(global_state.get("accepted", False))

    spatial = dict(spatial_validation or {})
    broken = spatial.get("final_spatial_edge_records_broken")
    uncarved = spatial.get("spatial_graph_edges_uncarved", 0)
    missing_endpoints = spatial.get("spatial_graph_edges_missing_room_endpoint", 0)
    exact_topology = spatial.get("spatial_topology_exact_invariants_preserved")
    if broken is None:
        spatial_passed = None
    else:
        spatial_passed = bool(
            int(broken) == 0
            and int(uncarved or 0) == 0
            and int(missing_endpoints or 0) == 0
            and (exact_topology is None or bool(exact_topology))
        )

    tile = dict(tile_validation or {})
    tile_status = str(tile.get("termination_status", "not_run"))
    tile_exact = bool(tile.get("is_exact", False))
    if not tile_exact or tile_status in {"unknown", "budget_exhausted", "validator_error", "not_run"}:
        tile_passed: Optional[bool] = None
    else:
        tile_passed = bool(tile.get("solvable", False))

    replay_status = str(tile.get("route_replay_status", "not_run"))
    replay_applicable = bool(tile.get("solvable", False)) or replay_status != "not_run"
    if not replay_applicable:
        replay_passed: Optional[bool] = None
    elif replay_status == "verified":
        replay_passed = True
    elif replay_status == "failed":
        replay_passed = False
    else:
        replay_passed = None

    consistency_status = str(
        tile.get("solver_consistency_status", "not_requested")
    )
    consistency_applicable = consistency_status != "not_requested"
    if not consistency_applicable:
        consistency_passed: Optional[bool] = None
    elif consistency_status == "consistent":
        consistency_passed = True
    elif consistency_status in {"path_cost_mismatch", "reachability_mismatch"}:
        consistency_passed = False
    else:
        consistency_passed = None

    return EndToEndValidationReport(
        representation=validate_grid_representation(dungeon_grid),
        graph_progression=ValidationStageEvidence(
            name="graph_progression",
            passed=graph_passed,
            status=graph_status,
            applicable=graph_applicable,
            details=graph,
        ),
        global_state_progression=ValidationStageEvidence(
            name="global_state_progression",
            passed=global_state_passed,
            status=global_state_status,
            applicable=global_state_applicable,
            details=global_state,
        ),
        spatial_realization=ValidationStageEvidence(
            name="spatial_realization",
            passed=spatial_passed,
            status=("intact" if spatial_passed is True else "broken" if spatial_passed is False else "not_run"),
            details=spatial,
        ),
        tile_solvability=ValidationStageEvidence(
            name="tile_solvability",
            passed=tile_passed,
            status=tile_status,
            exact=tile_exact,
            details=tile,
        ),
        route_replay=ValidationStageEvidence(
            name="route_replay",
            passed=replay_passed,
            status=replay_status,
            applicable=replay_applicable,
            details={
                "path_length": tile.get("path_length"),
                "path_cost": tile.get("path_cost"),
                "replayed_path_cost": tile.get("route_replay_path_cost"),
                "error": tile.get("route_replay_error", ""),
            },
        ),
        solver_consistency=ValidationStageEvidence(
            name="solver_consistency",
            passed=consistency_passed,
            status=consistency_status,
            applicable=consistency_applicable,
            details={
                "astar_path_length": tile.get("path_length"),
                "astar_path_cost": tile.get("path_cost"),
                "dijkstra_path_length": tile.get(
                    "solver_consistency_path_length"
                ),
                "dijkstra_path_cost": tile.get(
                    "solver_consistency_path_cost"
                ),
                "dijkstra_states_explored": tile.get(
                    "solver_consistency_states_explored", 0
                ),
            },
        ),
        logicnet_agreement=logicnet_agreement,
        advisory_metrics=dict(advisory_metrics or {}),
    )
