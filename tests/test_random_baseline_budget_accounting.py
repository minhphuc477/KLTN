"""The random control must account for every fixed-budget grammar draw."""

from __future__ import annotations

from types import SimpleNamespace

import networkx as nx

from scripts import random_baseline


def test_random_baseline_keeps_failed_generation_attempts_in_budget(monkeypatch):
    graph = nx.DiGraph()
    graph.add_edge("start", "goal")

    class _Grammar:
        def __init__(self, seed):
            self._draws = iter([None, "valid", "invalid"])

        def generate(self, num_rooms):
            return next(self._draws)

    monkeypatch.setattr(random_baseline, "MissionGrammar", _Grammar)
    monkeypatch.setattr(
        random_baseline,
        "mission_graph_to_networkx",
        lambda mission_graph, directed: graph if mission_graph == "valid" else None,
    )
    monkeypatch.setattr(
        random_baseline,
        "ExternalValidator",
        lambda: SimpleNamespace(validate=lambda _graph: SimpleNamespace(is_solvable=True)),
    )
    monkeypatch.setattr(
        random_baseline,
        "validate_topology",
        lambda _graph: SimpleNamespace(is_valid=True),
    )

    candidates = random_baseline.generate_random_topologies(num_samples=3, seed=7)

    assert len(candidates) == 3
    assert [candidate["graph"] is not None for candidate in candidates] == [False, True, False]
    assert [candidate["fitness"] for candidate in candidates] == [0.0, 1.25, 0.0]
