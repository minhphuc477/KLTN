"""Tests for src.evaluation.evaluate utility module."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src.evaluation import evaluate as eval_mod


def test_make_simple_graph_basic_shape():
    """Generated graph has expected node/edge counts and endpoint labels."""
    g = eval_mod.make_simple_graph(soluble=True, length=6, include_locked=True)

    assert g.number_of_nodes() == 6
    assert g.number_of_edges() == 5
    assert g.nodes[0]["label"] == "s"
    assert g.nodes[5]["label"] == "t"


def test_evaluate_solvability_returns_required_metrics():
    """Solvability evaluator returns stable metric keys."""
    graphs = [
        eval_mod.make_simple_graph(soluble=True, length=5, include_locked=False),
        eval_mod.make_simple_graph(soluble=False, length=5, include_locked=False),
    ]
    stats = eval_mod.evaluate_solvability(graphs)

    assert set(stats.keys()) == {
        "total",
        "solvable",
        "solvability_rate",
        "avg_path_length",
    }
    assert stats["total"] == 2
    assert 0.0 <= stats["solvability_rate"] <= 1.0
    assert stats["avg_path_length"] >= 0.0


def test_evaluate_map_elites_returns_required_metrics():
    """MAP-Elites evaluator returns archive stats from current API."""
    graphs = [
        eval_mod.make_simple_graph(soluble=True, length=5, include_locked=True),
        eval_mod.make_simple_graph(soluble=False, length=6, include_locked=True),
    ]
    stats = eval_mod.evaluate_map_elites(graphs)

    assert set(stats.keys()) == {
        "coverage",
        "qd_score",
        "mean_fitness",
        "feature_diversity",
        "num_elites",
    }
    assert 0.0 <= stats["coverage"] <= 1.0
    assert stats["num_elites"] >= 0


def test_main_writes_json_with_seed_and_metrics(monkeypatch, tmp_path: Path):
    """CLI main() writes output file with expected top-level structure."""
    out_file = tmp_path / "eval.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--num-random",
            "3",
            "--seed",
            "7",
            "--output",
            str(out_file),
        ],
    )

    eval_mod.main()

    assert out_file.exists()
    payload = json.loads(out_file.read_text())
    assert payload["seed"] == 7
    assert payload["num_graphs"] == 3
    assert "solvability" in payload
    assert "map_elites" in payload
    assert isinstance(payload["graphs"], list)
    assert len(payload["graphs"]) == 3
