"""Exact graph-level validation for explicit cross-room global state.

The validator explores ``(room, state assignment)`` pairs. It proves that the
goal is reachable, every declared transition can be triggered from a reachable
state, and every state-gated graph edge is traversable in at least one reachable
state. It does not claim that rendered room variants preserve those semantics.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import networkx as nx


StateKey = Tuple[Tuple[str, Any], ...]


@dataclass(frozen=True)
class GlobalStateValidationResult:
    accepted: bool
    goal_reachable: bool
    complete: bool
    termination_status: str
    states_explored: int
    solution_rooms: Tuple[Any, ...] = ()
    solution_actions: Tuple[str, ...] = ()
    unreachable_transition_indices: Tuple[int, ...] = ()
    untraversable_requirement_indices: Tuple[int, ...] = ()
    errors: Tuple[str, ...] = ()
    reachable_rooms: Tuple[Any, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": bool(self.accepted),
            "goal_reachable": bool(self.goal_reachable),
            "complete": bool(self.complete),
            "termination_status": self.termination_status,
            "states_explored": int(self.states_explored),
            "solution_rooms": list(self.solution_rooms),
            "solution_actions": list(self.solution_actions),
            "unreachable_transition_indices": list(
                self.unreachable_transition_indices
            ),
            "untraversable_requirement_indices": list(
                self.untraversable_requirement_indices
            ),
            "errors": list(self.errors),
            "reachable_rooms": list(self.reachable_rooms),
        }


def _state_key(values: Mapping[str, Any]) -> StateKey:
    key: list[tuple[str, Any]] = []
    for name, value in sorted(values.items()):
        try:
            hash(value)
        except TypeError as exc:
            raise TypeError(
                f"Global-state value for {name!r} must be a hashable scalar, "
                f"got {type(value).__name__}."
            ) from exc
        key.append((str(name), value))
    return tuple(key)


def _state_dict(key: StateKey) -> Dict[str, Any]:
    return dict(key)


def _requirements_hold(
    state: Mapping[str, Any],
    requirements: Mapping[str, Any],
) -> bool:
    return all(state.get(str(name)) == value for name, value in requirements.items())


def _node_matches_role(attrs: Mapping[str, Any], role: str) -> bool:
    value = attrs.get("node_type", attrs.get("type", attrs.get("label", "")))
    if hasattr(value, "name"):
        value = value.name
    tokens = {
        token.strip().upper().split(".")[-1]
        for token in str(value).replace("|", ",").split(",")
        if token.strip()
    }
    if role == "START":
        return bool(attrs.get("is_start", False)) or bool(tokens & {"START", "S"})
    return bool(
        attrs.get("is_goal", False)
        or attrs.get("has_goal", False)
        or attrs.get("is_triforce", False)
        or tokens & {"GOAL", "TRIFORCE", "T"}
    )


def _resolve_endpoint(
    graph: nx.Graph,
    explicit: Optional[Any],
    role: str,
) -> Optional[Any]:
    if explicit is not None:
        return explicit if explicit in graph else None
    matches = [
        node_id
        for node_id, attrs in graph.nodes(data=True)
        if _node_matches_role(dict(attrs), role)
    ]
    return matches[0] if len(matches) == 1 else None


def validate_global_state_progression(
    graph: nx.Graph,
    specification: Mapping[str, Any],
    *,
    start_node: Optional[Any] = None,
    goal_node: Optional[Any] = None,
    max_states: int = 100_000,
) -> GlobalStateValidationResult:
    """Exhaustively validate a finite explicit global-state graph contract."""
    start = _resolve_endpoint(graph, start_node, "START")
    goal = _resolve_endpoint(graph, goal_node, "GOAL")
    errors: list[str] = []
    if start is None:
        errors.append("Global-state validation requires exactly one valid START room.")
    if goal is None:
        errors.append("Global-state validation requires exactly one valid GOAL room.")

    variables = list(specification.get("variables", []) or [])
    transitions = list(specification.get("transitions", []) or [])
    edge_requirements = list(specification.get("edge_requirements", []) or [])
    initial: Dict[str, Any] = {}
    for index, raw in enumerate(variables):
        if not isinstance(raw, Mapping):
            errors.append(f"Variable {index} is not a mapping.")
            continue
        name = str(raw.get("name", "")).strip()
        if not name:
            errors.append(f"Variable {index} has no name.")
            continue
        if name in initial:
            errors.append(f"Duplicate global-state variable {name!r}.")
            continue
        if "initial" not in raw:
            errors.append(f"Variable {name!r} has no initial value.")
            continue
        initial[name] = raw.get("initial")

    normalized_transitions: list[tuple[Any, str, Dict[str, Any]]] = []
    for index, raw in enumerate(transitions):
        if not isinstance(raw, Mapping):
            errors.append(f"Transition {index} is not a mapping.")
            continue
        room = raw.get("from_room")
        trigger = str(raw.get("trigger", "")).strip()
        changes = dict(raw.get("changes", {}) or {})
        if room not in graph:
            errors.append(f"Transition {index} references unknown room {room!r}.")
        if not trigger:
            errors.append(f"Transition {index} has no trigger.")
        unknown = set(changes) - set(initial)
        if unknown:
            errors.append(
                f"Transition {index} changes unknown variables {sorted(unknown)}."
            )
        normalized_transitions.append((room, trigger, changes))

    normalized_requirements: list[tuple[Any, Any, Dict[str, Any]]] = []
    for index, raw in enumerate(edge_requirements):
        if not isinstance(raw, Mapping):
            errors.append(f"Edge requirement {index} is not a mapping.")
            continue
        source = raw.get("source")
        target = raw.get("target")
        requirements = dict(raw.get("requires", {}) or {})
        if not graph.has_edge(source, target):
            errors.append(
                f"Edge requirement {index} references missing edge "
                f"{source!r}->{target!r}."
            )
        unknown = set(requirements) - set(initial)
        if unknown:
            errors.append(
                f"Edge requirement {index} uses unknown variables {sorted(unknown)}."
            )
        normalized_requirements.append((source, target, requirements))

    if errors:
        return GlobalStateValidationResult(
            accepted=False,
            goal_reachable=False,
            complete=True,
            termination_status="invalid_contract",
            states_explored=0,
            errors=tuple(errors),
        )

    requirements_by_edge: Dict[tuple[Any, Any], list[tuple[int, Dict[str, Any]]]] = {}
    for index, (source, target, requirements) in enumerate(normalized_requirements):
        requirements_by_edge.setdefault((source, target), []).append(
            (index, requirements)
        )

    initial_key = _state_key(initial)
    initial_search = (start, initial_key)
    queue = deque([initial_search])
    visited = {initial_search}
    parents: Dict[tuple[Any, StateKey], tuple[tuple[Any, StateKey], str]] = {}
    reachable_rooms = {start}
    reachable_transitions: set[int] = set()
    traversed_requirements: set[int] = set()
    goal_state: Optional[tuple[Any, StateKey]] = None
    explored = 0
    budget = max(1, int(max_states))

    while queue and explored < budget:
        current = queue.popleft()
        room, state_key = current
        state = _state_dict(state_key)
        explored += 1
        reachable_rooms.add(room)
        if room == goal and goal_state is None:
            goal_state = current

        for index, (trigger_room, trigger, changes) in enumerate(
            normalized_transitions
        ):
            if trigger_room != room:
                continue
            reachable_transitions.add(index)
            updated = dict(state)
            updated.update(changes)
            next_state = (room, _state_key(updated))
            if next_state == current or next_state in visited:
                continue
            visited.add(next_state)
            parents[next_state] = (current, f"trigger:{trigger}")
            queue.append(next_state)

        neighbors = graph.successors(room) if graph.is_directed() else graph.neighbors(room)
        for neighbor in neighbors:
            requirement_entries = requirements_by_edge.get((room, neighbor), [])
            if any(
                not _requirements_hold(state, requirements)
                for _, requirements in requirement_entries
            ):
                continue
            traversed_requirements.update(index for index, _ in requirement_entries)
            next_state = (neighbor, state_key)
            if next_state in visited:
                continue
            visited.add(next_state)
            parents[next_state] = (current, f"move:{room}->{neighbor}")
            queue.append(next_state)

    complete = not queue
    missing_transitions = tuple(
        sorted(set(range(len(normalized_transitions))) - reachable_transitions)
    )
    missing_requirements = tuple(
        sorted(set(range(len(normalized_requirements))) - traversed_requirements)
    )
    solution_rooms: list[Any] = []
    solution_actions: list[str] = []
    if goal_state is not None:
        cursor = goal_state
        solution_rooms.append(cursor[0])
        while cursor in parents:
            previous, action = parents[cursor]
            solution_actions.append(action)
            solution_rooms.append(previous[0])
            cursor = previous
        solution_rooms.reverse()
        solution_actions.reverse()

    accepted = bool(
        complete
        and goal_state is not None
        and not missing_transitions
        and not missing_requirements
    )
    return GlobalStateValidationResult(
        accepted=accepted,
        goal_reachable=goal_state is not None,
        complete=complete,
        termination_status=(
            "accepted"
            if accepted
            else "budget_exhausted"
            if not complete
            else "unreachable_contract"
        ),
        states_explored=explored,
        solution_rooms=tuple(solution_rooms),
        solution_actions=tuple(solution_actions),
        unreachable_transition_indices=missing_transitions,
        untraversable_requirement_indices=missing_requirements,
        reachable_rooms=tuple(
            sorted(
                reachable_rooms,
                key=lambda value: (type(value).__name__, repr(value)),
            )
        ),
    )


def validate_attached_global_state_contract(
    graph: nx.Graph,
    *,
    max_states: int = 100_000,
) -> Optional[GlobalStateValidationResult]:
    """Revalidate a graph's attached global-state contract, when present.

    Stored validation payloads are deliberately ignored. Generation and repair
    may have changed the graph after the payload was produced, so publication
    evidence must be recomputed against the exact graph being reported.
    """
    specification = graph.graph.get("global_state_contract")
    if not isinstance(specification, Mapping) or not specification:
        return None
    return validate_global_state_progression(
        graph,
        specification,
        max_states=max_states,
    )


__all__ = [
    "GlobalStateValidationResult",
    "validate_attached_global_state_contract",
    "validate_global_state_progression",
]
