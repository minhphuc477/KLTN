"""Designer controllability proof protocol for Block-I topology generation.

The default mode writes an experiment plan only. Pass ``--execute`` when the
machine is ready for the expensive generation sweep.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.benchmark_suite import (  # noqa: E402
    GraphDescriptor,
    extract_graph_descriptor,
    generate_block_i_graphs,
    load_vglc_reference_graphs,
)


@dataclass(frozen=True)
class MethodConfig:
    name: str
    rule_space: str
    search_strategy: str


@dataclass
class TargetSpec:
    name: str
    family: str
    min_rooms: int
    max_rooms: int
    descriptor_targets: Dict[str, float]
    evaluation_targets: Dict[str, float] = field(default_factory=dict)
    tolerances: Dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def merged_targets(self) -> Dict[str, float]:
        out = dict(self.descriptor_targets)
        out.update(self.evaluation_targets)
        return out


METHODS: Dict[str, MethodConfig] = {
    "FULL_GA": MethodConfig("FULL_GA", "full", "ga"),
    "FULL_CVT": MethodConfig("FULL_CVT", "full", "cvt_emitter"),
    "CORE_GA": MethodConfig("CORE_GA", "core", "ga"),
}


DEFAULT_TOLERANCES: Dict[str, float] = {
    "num_nodes": 0.20,
    "num_edges": 0.28,
    "path_length": 0.30,
    "key_count": 0.35,
    "lock_count": 0.35,
    "enemy_count": 0.45,
    "puzzle_count": 0.45,
    "item_count": 0.45,
    "linearity": 0.12,
    "leniency": 0.15,
    "progression_complexity": 0.15,
    "topology_complexity": 0.16,
    "cycle_density": 0.16,
    "shortcut_density": 0.12,
    "gating_density": 0.12,
    "gate_depth_ratio": 0.14,
    "path_depth_ratio": 0.14,
    "puzzle_density": 0.12,
    "item_density": 0.12,
    "gate_variety": 0.18,
}

COUNT_METRICS = {
    "num_nodes",
    "num_edges",
    "path_length",
    "key_count",
    "lock_count",
    "enemy_count",
    "puzzle_count",
    "item_count",
}

DESCRIPTOR_TARGET_KEYS = {
    "linearity",
    "leniency",
    "progression_complexity",
    "topology_complexity",
    "path_length",
    "num_nodes",
    "num_edges",
    "key_count",
    "lock_count",
    "cycle_density",
    "shortcut_density",
    "gate_depth_ratio",
    "path_depth_ratio",
    "gating_density",
    "puzzle_density",
    "item_density",
    "gate_variety",
    "bombable_ratio",
    "soft_lock_ratio",
    "switch_ratio",
    "stair_ratio",
    "difficulty_curve_min_alignment",
    "difficulty_curve_min_trend_corr",
}


def _clip(value: float, lo: float, hi: float) -> float:
    return float(np.clip(float(value), float(lo), float(hi)))


def _mean(values: Iterable[float], default: float = 0.0) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float(default)


def _descriptor_to_metrics(desc: GraphDescriptor) -> Dict[str, float]:
    payload = asdict(desc)
    metrics: Dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, bool):
            metrics[key] = float(1.0 if value else 0.0)
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            metrics[key] = float(value)
    num_nodes = max(1.0, metrics.get("num_nodes", 0.0))
    num_edges = max(1.0, metrics.get("num_edges", 0.0))
    metrics["room_count"] = metrics.get("num_nodes", 0.0)
    metrics["puzzle_density"] = metrics.get("puzzle_count", 0.0) / num_nodes
    metrics["item_density"] = metrics.get("item_count", 0.0) / num_nodes
    metrics["enemy_density"] = metrics.get("enemy_count", 0.0) / num_nodes
    metrics["key_density"] = metrics.get("key_count", 0.0) / num_nodes
    metrics["lock_density"] = metrics.get("lock_count", 0.0) / num_edges
    metrics["gate_count"] = metrics.get("lock_count", 0.0)
    return metrics


def _reference_target_means(reference_graphs: Sequence[nx.Graph]) -> Dict[str, float]:
    if not reference_graphs:
        return {
            "num_nodes": 24.0,
            "num_edges": 48.0,
            "path_length": 9.0,
            "key_count": 3.0,
            "lock_count": 3.0,
            "enemy_count": 5.0,
            "puzzle_count": 5.0,
            "item_count": 3.0,
            "linearity": 0.45,
            "leniency": 0.50,
            "progression_complexity": 0.65,
            "topology_complexity": 0.45,
            "cycle_density": 0.30,
            "shortcut_density": 0.08,
            "gating_density": 0.16,
            "gate_depth_ratio": 0.25,
            "path_depth_ratio": 0.40,
            "puzzle_density": 0.20,
            "item_density": 0.12,
            "gate_variety": 0.30,
        }
    descriptors = [_descriptor_to_metrics(extract_graph_descriptor(graph, grammar=None)) for graph in reference_graphs]
    keys = sorted({key for row in descriptors for key in row})
    return {key: _mean(row.get(key, 0.0) for row in descriptors) for key in keys}


def _targets_for_generation(targets: Mapping[str, float]) -> Dict[str, float]:
    return {key: float(value) for key, value in targets.items() if key in DESCRIPTOR_TARGET_KEYS}


def _target_spec(
    *,
    name: str,
    family: str,
    targets: Mapping[str, float],
    min_rooms: Optional[int] = None,
    max_rooms: Optional[int] = None,
    notes: str = "",
) -> TargetSpec:
    merged = {str(key): float(value) for key, value in targets.items()}
    room_goal = int(round(float(merged.get("num_nodes", merged.get("room_count", 24.0)))))
    min_r = int(min_rooms if min_rooms is not None else max(6, round(0.88 * room_goal)))
    max_r = int(max_rooms if max_rooms is not None else max(min_r + 1, round(1.12 * room_goal)))
    descriptor_targets = _targets_for_generation(merged)
    evaluation_targets = {
        key: value for key, value in merged.items() if key not in descriptor_targets or key in COUNT_METRICS
    }
    return TargetSpec(
        name=name,
        family=family,
        min_rooms=min_r,
        max_rooms=max_r,
        descriptor_targets=descriptor_targets,
        evaluation_targets=evaluation_targets,
        tolerances=dict(DEFAULT_TOLERANCES),
        notes=notes,
    )


def build_target_suite(reference_means: Optional[Mapping[str, float]] = None) -> List[TargetSpec]:
    """Build the controllability sweep without running generation."""

    ref = dict(reference_means or _reference_target_means([]))
    base_edges = max(1.0, float(ref.get("num_edges", 2.0 * ref.get("num_nodes", 24.0))))

    def common(**updates: float) -> Dict[str, float]:
        target = {
            "num_nodes": float(ref.get("num_nodes", 24.0)),
            "num_edges": float(ref.get("num_edges", base_edges)),
            "path_length": float(ref.get("path_length", 9.0)),
            "key_count": float(ref.get("key_count", 3.0)),
            "lock_count": float(ref.get("lock_count", 3.0)),
            "linearity": float(ref.get("linearity", 0.45)),
            "leniency": float(ref.get("leniency", 0.50)),
            "progression_complexity": float(ref.get("progression_complexity", 0.65)),
            "topology_complexity": float(ref.get("topology_complexity", 0.45)),
            "cycle_density": float(ref.get("cycle_density", 0.30)),
            "shortcut_density": float(ref.get("shortcut_density", 0.08)),
            "gating_density": float(ref.get("gating_density", 0.16)),
            "gate_depth_ratio": float(ref.get("gate_depth_ratio", 0.25)),
            "path_depth_ratio": float(ref.get("path_depth_ratio", 0.40)),
            "puzzle_density": float(ref.get("puzzle_density", 0.20)),
            "item_density": float(ref.get("item_density", 0.12)),
            "gate_variety": float(ref.get("gate_variety", 0.30)),
        }
        target.update({key: float(value) for key, value in updates.items()})
        if "lock_count" in updates and "num_edges" in target:
            target["gating_density"] = _clip(target["lock_count"] / max(1.0, target["num_edges"]), 0.02, 0.65)
        if "key_count" in target and "lock_count" in target:
            target["leniency"] = _clip(target["key_count"] / max(1.0, target["lock_count"]), 0.12, 0.85)
        return target

    specs = [
        _target_spec(
            name="reference_center",
            family="reference",
            targets=common(),
            notes="Reference-mean target derived from VGLC graph descriptors when available.",
        ),
        _target_spec(
            name="p_quick_linear_easy",
            family="pereira_style",
            targets=common(
                num_nodes=12,
                num_edges=20,
                path_length=8,
                key_count=2,
                lock_count=1,
                linearity=0.68,
                progression_complexity=0.48,
                topology_complexity=0.28,
                cycle_density=0.08,
                gate_depth_ratio=0.16,
                path_depth_ratio=0.64,
            ),
            notes="Small, mostly linear, forgiving dungeon target.",
        ),
        _target_spec(
            name="p_balanced_keylock",
            family="pereira_style",
            targets=common(
                num_nodes=24,
                num_edges=44,
                path_length=11,
                key_count=3,
                lock_count=3,
                linearity=0.50,
                progression_complexity=0.62,
                topology_complexity=0.42,
                cycle_density=0.22,
                gate_depth_ratio=0.28,
            ),
            notes="Medium key-lock target similar to a thesis controllability table row.",
        ),
        _target_spec(
            name="p_hard_backtracking",
            family="pereira_style",
            targets=common(
                num_nodes=32,
                num_edges=70,
                path_length=15,
                key_count=3,
                lock_count=6,
                linearity=0.36,
                progression_complexity=0.78,
                topology_complexity=0.62,
                cycle_density=0.45,
                shortcut_density=0.14,
                gate_depth_ratio=0.45,
                path_depth_ratio=0.48,
                gate_variety=0.48,
            ),
            notes="High gate pressure and stronger backtracking target.",
        ),
        _target_spec(
            name="p_large_stress",
            family="pereira_style",
            targets=common(
                num_nodes=48,
                num_edges=104,
                path_length=22,
                key_count=5,
                lock_count=7,
                linearity=0.44,
                progression_complexity=0.82,
                topology_complexity=0.70,
                cycle_density=0.50,
                shortcut_density=0.16,
                gate_depth_ratio=0.52,
                path_depth_ratio=0.46,
                puzzle_density=0.26,
                item_density=0.15,
            ),
            notes="Large stress row; intended to reveal scaling and target drift.",
        ),
        _target_spec(
            name="p_large_stress_100",
            family="pereira_style",
            targets=common(
                num_nodes=100,
                num_edges=220,
                path_length=46,
                key_count=10,
                lock_count=15,
                linearity=0.42,
                progression_complexity=0.84,
                topology_complexity=0.74,
                cycle_density=0.50,
                shortcut_density=0.16,
                gate_depth_ratio=0.52,
                path_depth_ratio=0.46,
                puzzle_density=0.26,
                item_density=0.15,
            ),
            min_rooms=92,
            max_rooms=108,
            notes="Large stress row comparable to Pereira-style 100-room scalability discussion.",
        ),
        _target_spec(
            name="p_large_stress_250",
            family="pereira_style",
            targets=common(
                num_nodes=250,
                num_edges=550,
                path_length=115,
                key_count=25,
                lock_count=38,
                linearity=0.41,
                progression_complexity=0.85,
                topology_complexity=0.76,
                cycle_density=0.51,
                shortcut_density=0.165,
                gate_depth_ratio=0.535,
                path_depth_ratio=0.46,
                puzzle_density=0.26,
                item_density=0.15,
            ),
            min_rooms=230,
            max_rooms=270,
            notes="Intermediate stress row between the 100-room and 500-room endpoints.",
        ),
        _target_spec(
            name="p_large_stress_500",
            family="pereira_style",
            targets=common(
                num_nodes=500,
                num_edges=1100,
                path_length=230,
                key_count=50,
                lock_count=75,
                linearity=0.40,
                progression_complexity=0.86,
                topology_complexity=0.78,
                cycle_density=0.52,
                shortcut_density=0.17,
                gate_depth_ratio=0.55,
                path_depth_ratio=0.46,
                puzzle_density=0.26,
                item_density=0.15,
            ),
            min_rooms=460,
            max_rooms=540,
            notes="Extreme stress row comparable to Pereira-style 500-room scalability discussion; run separately if compute is constrained.",
        ),
    ]

    for value in (0.36, 0.52, 0.68):
        specs.append(
            _target_spec(
                name=f"axis_linearity_{value:.2f}".replace(".", "p"),
                family="axis_linearity",
                targets=common(num_nodes=24, num_edges=46, path_length=11, linearity=value),
                notes="One-axis linearity sweep with other controls held near reference.",
            )
        )
    for key_count, lock_count in ((5, 2), (3, 3), (2, 6)):
        specs.append(
            _target_spec(
                name=f"axis_keylock_k{key_count}_l{lock_count}",
                family="axis_keylock",
                targets=common(
                    num_nodes=24,
                    num_edges=46,
                    path_length=11,
                    key_count=key_count,
                    lock_count=lock_count,
                    progression_complexity=0.56 + (0.04 * lock_count),
                ),
                notes="Key/lock target sweep; raw key_count and lock_count are direct search targets and post-hoc evaluation targets.",
            )
        )
    for rooms in (12, 24, 36):
        specs.append(
            _target_spec(
                name=f"axis_size_{rooms}",
                family="axis_size",
                targets=common(
                    num_nodes=rooms,
                    num_edges=max(rooms + 8, int(round(2.0 * rooms))),
                    path_length=max(5, int(round(0.48 * rooms))),
                ),
                notes="Room-count controllability sweep.",
            )
        )
    return specs


def method_list(raw: str) -> List[MethodConfig]:
    out: List[MethodConfig] = []
    for token in [item.strip().upper() for item in str(raw).split(",") if item.strip()]:
        if token not in METHODS:
            raise ValueError(f"Unknown method '{token}'. Valid methods: {sorted(METHODS)}")
        out.append(METHODS[token])
    return out or [METHODS["FULL_GA"]]


def _target_error_fields(actual: Mapping[str, float], spec: TargetSpec) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    metric_passes: List[bool] = []
    for metric, target_value in sorted(spec.merged_targets().items()):
        if metric not in actual:
            continue
        observed = float(actual[metric])
        target = float(target_value)
        abs_error = abs(observed - target)
        if metric in COUNT_METRICS:
            denom = max(1.0, abs(target))
            norm_error = abs_error / denom
        else:
            norm_error = abs_error
        tolerance = float(spec.tolerances.get(metric, DEFAULT_TOLERANCES.get(metric, 0.15)))
        passed = bool(norm_error <= tolerance)
        fields[f"target_{metric}"] = target
        fields[f"actual_{metric}"] = observed
        fields[f"abs_error_{metric}"] = float(abs_error)
        fields[f"norm_error_{metric}"] = float(norm_error)
        fields[f"pass_{metric}"] = int(passed)
        metric_passes.append(passed)
    fields["controlled_metric_count"] = int(len(metric_passes))
    fields["controlled_pass_rate"] = float(np.mean(metric_passes)) if metric_passes else 0.0
    fields["pass_all_controlled_targets"] = int(all(metric_passes)) if metric_passes else 0
    return fields


def _stable_seed(base_seed: int, *parts: Any) -> int:
    text = ":".join(str(part) for part in parts)
    value = int(base_seed) & 0xFFFFFFFF
    for ch in text:
        value = ((value * 131) + ord(ch)) & 0xFFFFFFFF
    return int(value)


def execute_protocol(
    *,
    specs: Sequence[TargetSpec],
    methods: Sequence[MethodConfig],
    output_dir: Path,
    samples_per_target: int,
    seed: int,
    population_size: int,
    generations: int,
    room_count_bias: float,
    qd_archive_cells: int,
    qd_init_random_fraction: float,
    qd_emitter_mutation_rate: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    raw_rows: List[Dict[str, Any]] = []
    graph_payload: List[Dict[str, Any]] = []
    for spec in specs:
        for method in methods:
            run_seed = _stable_seed(seed, spec.name, method.name)
            started = time.time()
            graphs, generation_times = generate_block_i_graphs(
                num_samples=int(samples_per_target),
                seed=run_seed,
                min_rooms=int(spec.min_rooms),
                max_rooms=int(spec.max_rooms),
                population_size=int(population_size),
                generations=int(generations),
                rule_space=str(method.rule_space),
                descriptor_targets=dict(spec.descriptor_targets),
                room_count_bias=float(room_count_bias),
                search_strategy=str(method.search_strategy),
                qd_archive_cells=int(qd_archive_cells),
                qd_init_random_fraction=float(qd_init_random_fraction),
                qd_emitter_mutation_rate=float(qd_emitter_mutation_rate),
            )
            wall_time_sec = float(time.time() - started)
            for idx, graph in enumerate(graphs):
                desc = extract_graph_descriptor(graph, grammar=None)
                metrics = _descriptor_to_metrics(desc)
                row: Dict[str, Any] = {
                    "target_name": spec.name,
                    "target_family": spec.family,
                    "method": method.name,
                    "sample_idx": int(idx),
                    "seed": int(run_seed + idx),
                    "generation_time_sec": float(generation_times[idx]) if idx < len(generation_times) else 0.0,
                    "run_wall_time_sec": wall_time_sec,
                    "min_rooms": int(spec.min_rooms),
                    "max_rooms": int(spec.max_rooms),
                    "notes": spec.notes,
                }
                for key, value in metrics.items():
                    row[f"metric_{key}"] = value
                row.update(_target_error_fields(metrics, spec))
                raw_rows.append(row)
                graph_payload.append(
                    {
                        "target_name": spec.name,
                        "method": method.name,
                        "sample_idx": int(idx),
                        "graph": nx.node_link_data(graph),
                    }
                )
    output_dir.mkdir(parents=True, exist_ok=True)
    return raw_rows, graph_payload


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("target_family")), str(row.get("target_name")), str(row.get("method")))
        groups.setdefault(key, []).append(row)

    summary: List[Dict[str, Any]] = []
    for (family, target_name, method), group_rows in sorted(groups.items()):
        metric_names = sorted(
            {
                key[len("norm_error_") :]
                for row in group_rows
                for key in row.keys()
                if str(key).startswith("norm_error_")
            }
        )
        out: Dict[str, Any] = {
            "target_family": family,
            "target_name": target_name,
            "method": method,
            "n": int(len(group_rows)),
            "controlled_pass_rate_mean": _mean(row.get("controlled_pass_rate", 0.0) for row in group_rows),
            "pass_all_rate": _mean(row.get("pass_all_controlled_targets", 0.0) for row in group_rows),
            "generation_time_sec_mean": _mean(row.get("generation_time_sec", 0.0) for row in group_rows),
        }
        per_metric_pass: List[float] = []
        per_metric_error: List[float] = []
        for metric in metric_names:
            pass_key = f"pass_{metric}"
            err_key = f"norm_error_{metric}"
            abs_key = f"abs_error_{metric}"
            pass_rate = _mean(row.get(pass_key, 0.0) for row in group_rows)
            mean_norm_error = _mean(row.get(err_key, 0.0) for row in group_rows)
            mean_abs_error = _mean(row.get(abs_key, 0.0) for row in group_rows)
            out[f"pass_rate_{metric}"] = pass_rate
            out[f"mean_norm_error_{metric}"] = mean_norm_error
            out[f"mean_abs_error_{metric}"] = mean_abs_error
            out[f"target_{metric}_mean"] = _mean(row.get(f"target_{metric}", 0.0) for row in group_rows)
            out[f"actual_{metric}_mean"] = _mean(row.get(f"actual_{metric}", 0.0) for row in group_rows)
            per_metric_pass.append(pass_rate)
            per_metric_error.append(mean_norm_error)
        out["macro_metric_pass_rate"] = _mean(per_metric_pass)
        out["macro_norm_error"] = _mean(per_metric_error)
        summary.append(out)
    return summary


def build_target_response_rows(summary_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Build monotonic target-response rows for paper tables."""
    families: Dict[str, Sequence[str]] = {
        "axis_linearity": ("linearity",),
        "axis_size": ("num_nodes",),
        "axis_keylock": ("lock_count", "gating_density"),
    }
    rows: List[Dict[str, Any]] = []
    for family, metrics in families.items():
        family_rows = [row for row in summary_rows if str(row.get("target_family")) == family]
        methods = sorted({str(row.get("method", "")) for row in family_rows})
        for method in methods:
            method_rows = [row for row in family_rows if str(row.get("method", "")) == method]
            for metric in metrics:
                ordered = sorted(
                    method_rows,
                    key=lambda row: (
                        float(row.get(f"target_{metric}_mean", 0.0) or 0.0),
                        str(row.get("target_name", "")),
                    ),
                )
                previous_actual: Optional[float] = None
                for row in ordered:
                    target_mean = float(row.get(f"target_{metric}_mean", 0.0) or 0.0)
                    actual_mean = float(row.get(f"actual_{metric}_mean", 0.0) or 0.0)
                    monotonic = 1
                    if previous_actual is not None and actual_mean + 1e-9 < previous_actual:
                        monotonic = 0
                    previous_actual = actual_mean
                    rows.append(
                        {
                            "target_family": family,
                            "metric": metric,
                            "target_name": str(row.get("target_name", "")),
                            "method": method,
                            "target_mean": target_mean,
                            "actual_mean": actual_mean,
                            "mean_norm_error": float(row.get(f"mean_norm_error_{metric}", 0.0) or 0.0),
                            "pass_rate": float(row.get(f"pass_rate_{metric}", 0.0) or 0.0),
                            "monotonic_non_decreasing_from_previous": int(monotonic),
                        }
                    )
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({str(key) for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def write_plan(output_dir: Path, specs: Sequence[TargetSpec], methods: Sequence[MethodConfig], args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "designer_controllability_proof",
        "mode": "execute" if bool(args.execute) else "plan_only",
        "samples_per_target": int(args.samples_per_target),
        "population_size": int(args.population_size),
        "generations": int(args.generations),
        "methods": [asdict(method) for method in methods],
        "targets": [asdict(spec) for spec in specs],
        "research_basis": [
            "PCG Benchmark evaluates quality, diversity, and controllability.",
            "Pereira-style dungeon work reports target matching for rooms, keys, locks, and linearity.",
            "Controllable PCG literature treats constraint satisfaction and target tracking as first-class evidence.",
        ],
    }
    (output_dir / "designer_controllability_plan.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Designer Controllability Proof Plan",
        "",
        "This file is generated by `scripts/run_designer_controllability_proof.py`.",
        "",
        f"- mode: `{payload['mode']}`",
        f"- methods: `{', '.join(method.name for method in methods)}`",
        f"- targets: `{len(specs)}`",
        f"- samples per target: `{int(args.samples_per_target)}`",
        "",
        "## Target Rows",
        "",
        "| Target | Family | Rooms | Controlled targets | Notes |",
        "|---|---|---:|---|---|",
    ]
    for spec in specs:
        controlled = ", ".join(sorted(spec.merged_targets().keys()))
        lines.append(
            f"| `{spec.name}` | `{spec.family}` | `{spec.min_rooms}-{spec.max_rooms}` | {controlled} | {spec.notes} |"
        )
    lines.extend(
        [
            "",
            "## Execute Later",
            "",
            "```powershell",
            "python scripts\\run_designer_controllability_proof.py --execute --output results\\designer_controllability_proof",
            "```",
        ]
    )
    (output_dir / "designer_controllability_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or run designer-controllability target sweeps.")
    parser.add_argument("--execute", action="store_true", help="Run generation. Omit this to write the plan only.")
    parser.add_argument("--output", type=Path, default=Path("results") / "designer_controllability_proof")
    parser.add_argument("--data-root", type=Path, default=Path("Data") / "The Legend of Zelda")
    parser.add_argument("--reference-limit", type=int, default=None)
    parser.add_argument("--methods", type=str, default="FULL_GA,FULL_CVT")
    parser.add_argument("--samples-per-target", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--room-count-bias", type=float, default=0.35)
    parser.add_argument("--qd-archive-cells", type=int, default=128)
    parser.add_argument("--qd-init-random-fraction", type=float, default=0.35)
    parser.add_argument("--qd-emitter-mutation-rate", type=float, default=0.18)
    parser.add_argument("--write-graphs", action="store_true", help="Persist generated graphs in node-link JSON.")
    parser.add_argument(
        "--target-names",
        type=str,
        default="",
        help="Optional comma-separated target subset, such as p_large_stress_100,p_large_stress_250,p_large_stress_500.",
    )
    parser.add_argument("--quick", action="store_true", help="Small smoke execution profile for local validation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    methods = method_list(args.methods)
    refs = load_vglc_reference_graphs(args.data_root, limit=args.reference_limit)
    reference_means = _reference_target_means(refs)
    specs = build_target_suite(reference_means)
    requested_targets = {token.strip() for token in str(args.target_names).split(",") if token.strip()}
    if requested_targets:
        available_targets = {spec.name for spec in specs}
        unknown_targets = sorted(requested_targets - available_targets)
        if unknown_targets:
            raise ValueError(f"Unknown target names: {unknown_targets}. Available: {sorted(available_targets)}")
        specs = [spec for spec in specs if spec.name in requested_targets]
    if args.quick:
        args.samples_per_target = min(int(args.samples_per_target), 2)
        args.population_size = min(int(args.population_size), 12)
        args.generations = min(int(args.generations), 6)
        specs = specs[:3]
        methods = methods[:1]

    write_plan(args.output, specs, methods, args)
    if not args.execute:
        print(f"Wrote designer-controllability plan to {args.output}")
        return 0

    rows, graph_payload = execute_protocol(
        specs=specs,
        methods=methods,
        output_dir=args.output,
        samples_per_target=int(args.samples_per_target),
        seed=int(args.seed),
        population_size=int(args.population_size),
        generations=int(args.generations),
        room_count_bias=float(args.room_count_bias),
        qd_archive_cells=int(args.qd_archive_cells),
        qd_init_random_fraction=float(args.qd_init_random_fraction),
        qd_emitter_mutation_rate=float(args.qd_emitter_mutation_rate),
    )
    summary = summarize_rows(rows)
    target_response = build_target_response_rows(summary)
    write_csv(args.output / "designer_controllability_raw.csv", rows)
    write_csv(args.output / "designer_controllability_summary.csv", summary)
    write_csv(args.output / "designer_target_response_monotonicity.csv", target_response)
    payload = {
        "reference_means": reference_means,
        "rows": rows,
        "summary": summary,
        "target_response_monotonicity": target_response,
    }
    (args.output / "designer_controllability_payload.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    if args.write_graphs:
        (args.output / "designer_controllability_graphs.json").write_text(
            json.dumps(graph_payload, indent=2),
            encoding="utf-8",
        )
    print(f"Wrote designer-controllability outputs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
