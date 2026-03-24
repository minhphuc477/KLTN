"""Paired A/B benchmark for Block I cognitive objective.

Compares topology generation with cognitive objective disabled vs enabled
under matched seeds and outputs per-seed rows plus aggregated deltas.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.benchmark_suite import (
    extract_graph_descriptor,
    generate_block_i_graphs,
    load_vglc_reference_graphs,
)
from src.evaluation.cbs_fitness import compute_cbs_fitness

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArmConfig:
    name: str
    cognitive_weight: float


def _descriptor_target_means(reference_graphs: Sequence[Any]) -> Dict[str, float]:
    if not reference_graphs:
        return {
            "linearity": 0.45,
            "leniency": 0.50,
            "progression_complexity": 0.65,
            "topology_complexity": 0.45,
            "path_length": 8.0,
            "num_nodes": 12.0,
        }

    desc = [extract_graph_descriptor(g, grammar=None) for g in reference_graphs]

    def _mean(name: str) -> float:
        return float(np.mean([float(getattr(d, name)) for d in desc]))

    return {
        "linearity": _mean("linearity"),
        "leniency": _mean("leniency"),
        "progression_complexity": _mean("progression_complexity"),
        "topology_complexity": _mean("topology_complexity"),
        "path_length": _mean("path_length"),
        "num_nodes": _mean("num_nodes"),
    }


def _run_one_seed(
    *,
    seed: int,
    arm: ArmConfig,
    descriptor_targets: Dict[str, float],
    min_rooms: int,
    max_rooms: int,
    population_size: int,
    generations: int,
    rule_space: str,
    room_count_bias: float,
    persona: str,
    target_confusion_ratio: float,
) -> Dict[str, Any]:
    targets = dict(descriptor_targets)
    targets["cognitive_score_weight"] = float(max(0.0, arm.cognitive_weight))
    targets["cognitive_persona"] = str(persona)
    targets["cognitive_target_confusion_ratio"] = float(target_confusion_ratio)

    graphs, times = generate_block_i_graphs(
        num_samples=1,
        seed=int(seed),
        min_rooms=int(min_rooms),
        max_rooms=int(max_rooms),
        population_size=int(population_size),
        generations=int(generations),
        rule_space=str(rule_space),
        descriptor_targets=targets,
        room_count_bias=float(room_count_bias),
        search_strategy="ga",
    )
    graph = graphs[0]
    desc = extract_graph_descriptor(graph, grammar=None)
    cbs = compute_cbs_fitness(
        graph,
        persona=str(persona),
        target_confusion_ratio=float(target_confusion_ratio),
    )

    return {
        "seed": int(seed),
        "arm": str(arm.name),
        "cognitive_weight": float(arm.cognitive_weight),
        "generation_time_sec": float(times[0]) if times else float("nan"),
        "nodes": int(graph.number_of_nodes()),
        "edges": int(graph.number_of_edges()),
        "linearity": float(desc.linearity),
        "leniency": float(desc.leniency),
        "progression_complexity": float(desc.progression_complexity),
        "topology_complexity": float(desc.topology_complexity),
        "cycle_density": float(desc.cycle_density),
        "shortcut_density": float(desc.shortcut_density),
        "gate_depth_ratio": float(desc.gate_depth_ratio),
        "path_depth_ratio": float(desc.path_depth_ratio),
        "directionality_gap": float(desc.directionality_gap),
        "constraint_valid": float(desc.constraint_valid),
        "path_exists": float(desc.path_exists),
        "cbs_fitness": float(cbs.get("fitness", 0.0)),
        "cbs_confusion_ratio": float(cbs.get("confusion_ratio", 0.0)),
        "cbs_path_efficiency": float(cbs.get("path_efficiency", 0.0)),
        "cbs_room_entropy": float(cbs.get("room_entropy", 0.0)),
        "cbs_proxy_mode": float(cbs.get("is_proxy", 0.0)),
    }


def _aggregate(df: pd.DataFrame, metrics: Sequence[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for arm in sorted(df["arm"].unique().tolist()):
        sub = df[df["arm"] == arm]
        out[str(arm)] = {
            m: float(sub[m].mean(skipna=True))
            for m in metrics
        }
    return out


def _paired_delta(
    df: pd.DataFrame,
    *,
    control_name: str,
    treatment_name: str,
    metrics: Sequence[str],
) -> pd.DataFrame:
    left = df[df["arm"] == treatment_name]
    right = df[df["arm"] == control_name]
    merged = left.merge(right, on="seed", suffixes=("_treat", "_ctrl"))
    rows: List[Dict[str, Any]] = []
    if merged.empty:
        return pd.DataFrame(rows)

    for metric in metrics:
        t = merged[f"{metric}_treat"].astype(float)
        c = merged[f"{metric}_ctrl"].astype(float)
        delta = t - c
        rows.append(
            {
                "metric": metric,
                "n_pairs": int(delta.shape[0]),
                "control_mean": float(c.mean()),
                "treatment_mean": float(t.mean()),
                "delta_mean_treatment_minus_control": float(delta.mean()),
                "delta_std": float(delta.std(ddof=0)),
            }
        )
    return pd.DataFrame(rows)


def _to_markdown(df: pd.DataFrame) -> str:
    try:
        return str(df.to_markdown(index=False))
    except Exception:
        return str(df.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paired A/B benchmark for cognitive topology objective")
    parser.add_argument("--output", type=Path, default=Path("results") / "cognitive_objective_ab")
    parser.add_argument("--data-root", type=Path, default=Path("Data") / "The Legend of Zelda")
    parser.add_argument("--reference-limit", type=int, default=96)
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-rooms", type=int, default=8)
    parser.add_argument("--max-rooms", type=int, default=16)
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--rule-space", type=str, default="full", choices=["core", "full"])
    parser.add_argument("--room-count-bias", type=float, default=0.45)
    parser.add_argument("--persona", type=str, default="balanced")
    parser.add_argument("--target-confusion-ratio", type=float, default=1.8)
    parser.add_argument("--cognitive-weight", type=float, default=0.08)
    parser.add_argument("--control-name", type=str, default="COGNITIVE_OFF")
    parser.add_argument("--treatment-name", type=str, default="COGNITIVE_ON")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if bool(args.verbose) else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    refs = load_vglc_reference_graphs(
        data_root=Path(args.data_root),
        limit=int(max(1, args.reference_limit)),
    )
    descriptor_targets = _descriptor_target_means(refs)

    control = ArmConfig(name=str(args.control_name), cognitive_weight=0.0)
    treatment = ArmConfig(
        name=str(args.treatment_name),
        cognitive_weight=float(max(0.0, args.cognitive_weight)),
    )

    rows: List[Dict[str, Any]] = []
    for i in range(int(max(1, args.num_samples))):
        run_seed = int(args.seed) + (97 * i)
        rows.append(
            _run_one_seed(
                seed=run_seed,
                arm=control,
                descriptor_targets=descriptor_targets,
                min_rooms=int(args.min_rooms),
                max_rooms=int(args.max_rooms),
                population_size=int(args.population_size),
                generations=int(args.generations),
                rule_space=str(args.rule_space),
                room_count_bias=float(args.room_count_bias),
                persona=str(args.persona),
                target_confusion_ratio=float(args.target_confusion_ratio),
            )
        )
        rows.append(
            _run_one_seed(
                seed=run_seed,
                arm=treatment,
                descriptor_targets=descriptor_targets,
                min_rooms=int(args.min_rooms),
                max_rooms=int(args.max_rooms),
                population_size=int(args.population_size),
                generations=int(args.generations),
                rule_space=str(args.rule_space),
                room_count_bias=float(args.room_count_bias),
                persona=str(args.persona),
                target_confusion_ratio=float(args.target_confusion_ratio),
            )
        )

    df = pd.DataFrame(rows)
    metrics = [
        "generation_time_sec",
        "nodes",
        "edges",
        "linearity",
        "leniency",
        "progression_complexity",
        "topology_complexity",
        "cycle_density",
        "shortcut_density",
        "gate_depth_ratio",
        "path_depth_ratio",
        "directionality_gap",
        "constraint_valid",
        "path_exists",
        "cbs_fitness",
        "cbs_confusion_ratio",
        "cbs_path_efficiency",
        "cbs_room_entropy",
    ]

    means = _aggregate(df, metrics)
    paired = _paired_delta(
        df,
        control_name=control.name,
        treatment_name=treatment.name,
        metrics=metrics,
    )

    config = {
        "num_samples": int(args.num_samples),
        "seed": int(args.seed),
        "min_rooms": int(args.min_rooms),
        "max_rooms": int(args.max_rooms),
        "population_size": int(args.population_size),
        "generations": int(args.generations),
        "rule_space": str(args.rule_space),
        "room_count_bias": float(args.room_count_bias),
        "persona": str(args.persona),
        "target_confusion_ratio": float(args.target_confusion_ratio),
        "cognitive_weight": float(args.cognitive_weight),
        "control_name": str(control.name),
        "treatment_name": str(treatment.name),
    }

    summary = {
        "config": config,
        "means_by_arm": means,
        "paired_delta": paired.to_dict(orient="records"),
        "rows": int(df.shape[0]),
    }

    (out_dir / "runs.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    (out_dir / "paired_delta.csv").write_text(paired.to_csv(index=False), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Cognitive Objective A/B",
        "",
        "## Config",
        "",
        json.dumps(config, indent=2),
        "",
        "## Means By Arm",
        "",
        _to_markdown(pd.DataFrame([{"arm": k, **v} for k, v in means.items()])),
        "",
        "## Paired Delta (treatment - control)",
        "",
        _to_markdown(paired),
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")

    logger.info("Saved outputs to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
