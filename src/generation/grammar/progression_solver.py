"""Exact mission-graph progression validation for resource-gated edges.

The grammar-level graph uses persistent affordances for named keys, boss keys,
items, protection equipment, switches, and collection tokens. Fungible small
keys are different: opening a ``requires_key_count`` edge consumes inventory.
A monotone reachability closure therefore cannot prove those graphs by itself;
it can accidentally reuse one key for several locks.

This module keeps the persistent part as a fixed-point closure and performs a
state search over the subset of opened fungible-key gates. The representation
is substantially smaller than tile-level player search while preserving the
resource ordering contract required before room materialization.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import FrozenSet, Iterable, Optional, Set, Tuple

from src.generation.grammar.graph_types import EdgeType, MissionEdge, MissionGraph, NodeType


@dataclass(frozen=True)
class ProgressionSearchResult:
    """Evidence returned by :func:`solve_mission_progression`."""

    reachable_nodes: FrozenSet[int]
    required_nodes: FrozenSet[int]
    opened_fungible_edges: FrozenSet[int]
    open_order: Tuple[int, ...]
    explored_states: int
    exhausted: bool

    @property
    def all_reachable(self) -> bool:
        return bool(self.required_nodes.issubset(self.reachable_nodes)) and not self.exhausted

    @property
    def unreachable_nodes(self) -> FrozenSet[int]:
        return frozenset(self.required_nodes - self.reachable_nodes)


@dataclass(frozen=True)
class _ResourceSnapshot:
    named_key_ids: FrozenSet[int]
    items: FrozenSet[str]
    switch_ids: FrozenSet[int]
    token_counts: Tuple[Tuple[str, int], ...]
    untyped_token_count: int
    small_key_supply: int

    def token_count(self, token_id: Optional[str]) -> int:
        if token_id is None or not str(token_id).strip():
            return int(self.untyped_token_count)
        return int(dict(self.token_counts).get(str(token_id), 0))


def _required_nodes(graph: MissionGraph, start: int) -> FrozenSet[int]:
    """Return nodes participating in the traversable mission contract."""
    required = {int(start)}
    for edge in graph.edges:
        if edge.edge_type in graph.NON_TRAVERSABLE_EDGE_TYPES:
            continue
        if edge.source in graph.nodes:
            required.add(int(edge.source))
        if edge.target in graph.nodes:
            required.add(int(edge.target))
    return frozenset(required)


def _resources(graph: MissionGraph, reachable: Iterable[int]) -> _ResourceSnapshot:
    named_key_ids: Set[int] = set()
    items: Set[str] = set()
    switch_ids: Set[int] = set()
    token_counts: Counter[str] = Counter()
    total_tokens = 0
    small_key_supply = 0

    for node_id in reachable:
        node = graph.nodes.get(node_id)
        if node is None:
            continue
        if node.node_type in {NodeType.KEY, NodeType.BIG_KEY} and node.key_id is not None:
            named_key_ids.add(int(node.key_id))
        if node.node_type == NodeType.KEY:
            small_key_supply += max(1, int(getattr(node, "key_count_hint", 0) or 0))
        if node.node_type in {NodeType.ITEM, NodeType.PROTECTION_ITEM} and node.item_type:
            items.add(str(node.item_type))
        if node.node_type == NodeType.RESOURCE_FARM and node.drops_resource:
            items.add(str(node.drops_resource))
        if node.node_type == NodeType.SWITCH:
            switch_ids.add(int(node.id))
            if node.switch_id is not None:
                switch_ids.add(int(node.switch_id))
        if node.node_type == NodeType.TOKEN:
            total_tokens += 1
            if node.token_id is not None and str(node.token_id).strip():
                token_counts[str(node.token_id)] += 1

    return _ResourceSnapshot(
        named_key_ids=frozenset(named_key_ids),
        items=frozenset(items),
        switch_ids=frozenset(switch_ids),
        token_counts=tuple(sorted(token_counts.items())),
        untyped_token_count=int(total_tokens),
        small_key_supply=int(small_key_supply),
    )


def _node_gate_open(graph: MissionGraph, node_id: int, resources: _ResourceSnapshot) -> bool:
    node = graph.nodes.get(node_id)
    if node is None:
        return False
    if node.node_type in {NodeType.LOCK, NodeType.BOSS_DOOR}:
        return node.key_id is not None and int(node.key_id) in resources.named_key_ids
    return True


def _persistent_edge_gate_open(edge: MissionEdge, resources: _ResourceSnapshot) -> bool:
    """Check every non-consumable requirement on an edge."""
    if edge.edge_type in {EdgeType.LOCKED, EdgeType.BOSS_LOCKED} and edge.key_required is not None:
        if int(edge.key_required) not in resources.named_key_ids:
            return False

    if edge.edge_type == EdgeType.ITEM_GATE:
        required_item = str(edge.item_required or "").strip()
        if not required_item or required_item not in resources.items:
            return False

    if edge.edge_type == EdgeType.HAZARD:
        protection = str(edge.protection_item_id or "").strip()
        if not protection or protection not in resources.items:
            return False

    if edge.edge_type == EdgeType.MULTI_LOCK:
        required_tokens = int(max(0, edge.token_count))
        if required_tokens > 0 and resources.token_count(edge.token_id) < required_tokens:
            return False

    if edge.edge_type in {EdgeType.STATE_BLOCK, EdgeType.ON_OFF_GATE}:
        required_switches = {int(value) for value in edge.switches_required}
        if edge.switch_id is not None:
            required_switches.add(int(edge.switch_id))
        if required_switches and not required_switches.issubset(resources.switch_ids):
            return False

    return True


def _closure(
    graph: MissionGraph,
    start: int,
    opened_fungible_edges: FrozenSet[int],
) -> FrozenSet[int]:
    """Compute the persistent-resource closure for one small-key decision state."""
    reachable: Set[int] = {int(start)}
    changed = True
    while changed:
        changed = False
        resources = _resources(graph, reachable)
        for edge_index, edge in enumerate(graph.edges):
            if edge.edge_type in graph.NON_TRAVERSABLE_EDGE_TYPES:
                continue
            if edge.source not in reachable or edge.target in reachable:
                continue
            if int(max(0, edge.requires_key_count)) > 0 and edge_index not in opened_fungible_edges:
                continue
            if not _node_gate_open(graph, edge.target, resources):
                continue
            if not _persistent_edge_gate_open(edge, resources):
                continue
            reachable.add(int(edge.target))
            changed = True
    return frozenset(reachable)


def solve_mission_progression(
    graph: MissionGraph,
    start: int,
    *,
    max_states: Optional[int] = None,
) -> ProgressionSearchResult:
    """Prove mission progression while accounting for consumable small keys.

    ``max_states=None`` performs an exact search over all reachable fungible
    gate subsets. Supplying a positive bound is appropriate for untrusted or
    externally supplied graphs; hitting the bound returns ``exhausted=True``
    and never certifies the graph as valid.
    """
    graph.sanitize()
    required = _required_nodes(graph, start)
    if start not in graph.nodes:
        return ProgressionSearchResult(
            reachable_nodes=frozenset(),
            required_nodes=required,
            opened_fungible_edges=frozenset(),
            open_order=(),
            explored_states=0,
            exhausted=False,
        )
    if max_states is not None and int(max_states) < 1:
        raise ValueError("max_states must be positive or None")

    queue = deque([(frozenset(), tuple())])
    visited: Set[FrozenSet[int]] = {frozenset()}
    best_reachable: FrozenSet[int] = frozenset({int(start)})
    best_opened: FrozenSet[int] = frozenset()
    best_order: Tuple[int, ...] = ()
    explored = 0

    while queue:
        if max_states is not None and explored >= int(max_states):
            return ProgressionSearchResult(
                reachable_nodes=best_reachable,
                required_nodes=required,
                opened_fungible_edges=best_opened,
                open_order=best_order,
                explored_states=explored,
                exhausted=True,
            )

        opened, order = queue.popleft()
        explored += 1
        reachable = _closure(graph, start, opened)
        if len(reachable) > len(best_reachable):
            best_reachable, best_opened, best_order = reachable, opened, order
        if required.issubset(reachable):
            return ProgressionSearchResult(
                reachable_nodes=reachable,
                required_nodes=required,
                opened_fungible_edges=opened,
                open_order=order,
                explored_states=explored,
                exhausted=False,
            )

        resources = _resources(graph, reachable)
        spent_keys = sum(
            int(max(0, graph.edges[index].requires_key_count))
            for index in opened
        )
        available_keys = int(resources.small_key_supply) - int(spent_keys)
        if available_keys <= 0:
            continue

        for edge_index, edge in enumerate(graph.edges):
            cost = int(max(0, edge.requires_key_count))
            if cost <= 0 or edge_index in opened or cost > available_keys:
                continue
            if edge.edge_type in graph.NON_TRAVERSABLE_EDGE_TYPES:
                continue
            if edge.source not in reachable or edge.target in reachable:
                continue
            if not _node_gate_open(graph, edge.target, resources):
                continue
            if not _persistent_edge_gate_open(edge, resources):
                continue
            next_opened = frozenset((*opened, edge_index))
            if next_opened in visited:
                continue
            visited.add(next_opened)
            queue.append((next_opened, (*order, edge_index)))

    return ProgressionSearchResult(
        reachable_nodes=best_reachable,
        required_nodes=required,
        opened_fungible_edges=best_opened,
        open_order=best_order,
        explored_states=explored,
        exhausted=False,
    )


__all__ = ["ProgressionSearchResult", "solve_mission_progression"]
