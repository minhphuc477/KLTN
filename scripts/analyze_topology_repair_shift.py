"""Summarize pre/post topology-repair fitness shifts from exported graph artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import networkx as nx
import numpy as np


def _load_graph(path: Path, *, trust_pickle: bool = False) -> nx.Graph:
    if path.suffix.lower() == ".graphml":
        return nx.read_graphml(path)
    if path.suffix.lower() in {".gpickle", ".pickle", ".pkl"}:
        if not trust_pickle:
            raise ValueError(
                f"Refusing to unpickle {path}. Pickle can execute code; "
                "rerun with --trust-pickle only for trusted local artifacts."
            )
        import pickle

        with open(path, "rb") as handle:
            graph = pickle.load(handle)
        if not isinstance(graph, nx.Graph):
            raise TypeError(f"{path} does not contain a NetworkX graph.")
        return graph
    raise ValueError(f"Unsupported graph artifact: {path}")


def _iter_graph_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    for suffix in ("*.graphml", "*.gpickle", "*.pickle", "*.pkl"):
        yield from sorted(root.rglob(suffix))


def _repair_record(graph: nx.Graph) -> Dict[str, Any]:
    stats = graph.graph.get("generation_stats", {})
    if not isinstance(stats, dict):
        return {}
    record = stats.get("final_repair_evaluation", {})
    return dict(record) if isinstance(record, dict) else {}


def summarize(paths: Iterable[Path], *, trust_pickle: bool = False) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        try:
            graph = _load_graph(path, trust_pickle=trust_pickle)
            record = _repair_record(graph)
        except (OSError, ValueError, TypeError) as exc:
            rows.append({"path": str(path), "error": str(exc)})
            continue
        if not record:
            rows.append({"path": str(path), "missing_final_repair_evaluation": True})
            continue
        rows.append({"path": str(path), **record})

    measured = [row for row in rows if "fitness_delta" in row]
    deltas = [float(row["fitness_delta"]) for row in measured]
    pre_feasible = [bool(row.get("pre_feasible", False)) for row in measured]
    post_feasible = [bool(row.get("post_feasible", False)) for row in measured]
    summary = {
        "artifact_count": len(rows),
        "measured_count": len(measured),
        "missing_count": sum(bool(row.get("missing_final_repair_evaluation", False)) for row in rows),
        "error_count": sum("error" in row for row in rows),
        "fitness_delta_mean": float(np.mean(deltas)) if deltas else 0.0,
        "fitness_delta_std": float(np.std(deltas)) if deltas else 0.0,
        "fitness_delta_abs_mean": float(np.mean(np.abs(deltas))) if deltas else 0.0,
        "pre_feasible_rate": float(np.mean(pre_feasible)) if pre_feasible else 0.0,
        "post_feasible_rate": float(np.mean(post_feasible)) if post_feasible else 0.0,
        "feasibility_repair_rate": float(
            np.mean([not before and after for before, after in zip(pre_feasible, post_feasible)])
        ) if measured else 0.0,
    }
    return {"summary": summary, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze post-hoc topology repair shifts.")
    parser.add_argument("--input", required=True, type=Path, help="Graph artifact file or directory.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--trust-pickle",
        action="store_true",
        help="Allow loading .pkl/.pickle/.gpickle graph artifacts from trusted local runs.",
    )
    args = parser.parse_args()

    payload = summarize(_iter_graph_files(args.input), trust_pickle=bool(args.trust_pickle))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
