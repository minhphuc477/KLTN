"""
Fixed-seed A/B benchmark for grammar rule-weight schedules.

This compares two rule-weight policies on the same seed scenarios and reports:
- mean metrics per arm,
- paired deltas (treatment - control),
- paired bootstrap CIs and sign-permutation p-values,
- BH-FDR adjusted significance.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.benchmark_suite import (
    calibrate_rule_weights_to_vglc,
    extract_graph_descriptor,
    generate_block_i_graphs,
    load_vglc_reference_graphs,
)
from src.generation.grammar import MissionGrammar

logger = logging.getLogger(__name__)


@dataclass
class ArmConfig:
    name: str
    rule_weights: Optional[Dict[str, float]]


def _load_weights(path: Optional[Path]) -> Optional[Dict[str, float]]:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    out: Dict[str, float] = {}
    for key, value in data.items():
        try:
            v = float(value)
        except Exception:
            continue
        if np.isfinite(v) and v >= 0.0:
            out[str(key)] = float(v)
    return out


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


def _paired_bootstrap_ci(
    deltas: np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    if deltas.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, deltas.size, size=(n_boot, deltas.size))
    means = np.mean(deltas[idx], axis=1)
    return (
        float(np.quantile(means, alpha / 2.0)),
        float(np.quantile(means, 1.0 - alpha / 2.0)),
    )


def _paired_sign_permutation_pvalue(
    deltas: np.ndarray,
    *,
    n_perm: int = 5000,
    seed: int = 0,
) -> float:
    if deltas.size == 0:
        return 1.0
    observed = abs(float(np.mean(deltas)))
    if observed <= 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    abs_d = np.abs(deltas)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, deltas.size))
    perm_means = np.mean(signs * abs_d[None, :], axis=1)
    return float((1.0 + np.sum(np.abs(perm_means) >= observed)) / float(n_perm + 1))


def _benjamini_hochberg(p_values: Sequence[float]) -> List[float]:
    arr = np.asarray([float(p) for p in p_values], dtype=np.float64)
    n = int(arr.size)
    if n <= 0:
        return []
    order = np.argsort(arr)
    ranked = arr[order]
    q = np.zeros(n, dtype=np.float64)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = float(i + 1)
        raw = float(ranked[i]) * float(n) / rank
        prev = min(prev, raw)
        q[i] = prev
    out = np.empty(n, dtype=np.float64)
    out[order] = np.clip(q, 0.0, 1.0)
    return [float(v) for v in out.tolist()]


def _overall_completeness(desc: Any) -> float:
    return float(
        0.25 * float(desc.has_start and desc.has_goal)
        + 0.25 * float(desc.connected)
        + 0.25 * float(desc.path_exists)
        + 0.25 * float(desc.constraint_valid)
    )


def _descriptor_gap(desc: Any, targets: Dict[str, float]) -> float:
    keys = (
        ("linearity", 0.25),
        ("leniency", 0.20),
        ("progression_complexity", 0.25),
        ("topology_complexity", 0.30),
    )
    total = 0.0
    for key, w in keys:
        total += float(w) * abs(float(getattr(desc, key)) - float(targets.get(key, 0.5)))
    return float(total)


def _run_single_seed(
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
    search_strategy: str,
    qd_archive_cells: int,
    qd_init_random_fraction: float,
    qd_emitter_mutation_rate: float,
) -> Dict[str, Any]:
    graphs, times = generate_block_i_graphs(
        num_samples=1,
        seed=int(seed),
        min_rooms=int(min_rooms),
        max_rooms=int(max_rooms),
        population_size=int(population_size),
        generations=int(generations),
        rule_space=rule_space,
        rule_weight_overrides=arm.rule_weights,
        descriptor_targets=descriptor_targets,
        room_count_bias=float(room_count_bias),
        search_strategy=search_strategy,
        qd_archive_cells=int(qd_archive_cells),
        qd_init_random_fraction=float(qd_init_random_fraction),
        qd_emitter_mutation_rate=float(qd_emitter_mutation_rate),
    )
    graph = graphs[0]
    desc = extract_graph_descriptor(graph, grammar=MissionGrammar(seed=int(seed) + 777))
    row = {
        "seed": int(seed),
        "arm": arm.name,
        "generation_time_sec": float(times[0]) if times else float("nan"),
        "overall_completeness": _overall_completeness(desc),
        "constraint_valid": float(desc.constraint_valid),
        "path_exists": float(desc.path_exists),
        "linearity": float(desc.linearity),
        "leniency": float(desc.leniency),
        "progression_complexity": float(desc.progression_complexity),
        "topology_complexity": float(desc.topology_complexity),
        "cycle_density": float(desc.cycle_density),
        "shortcut_density": float(desc.shortcut_density),
        "gate_depth_ratio": float(desc.gate_depth_ratio),
        "path_depth_ratio": float(desc.path_depth_ratio),
        "directionality_gap": float(desc.directionality_gap),
        "repair_rate": float(desc.repair_applied),
        "total_repairs": float(desc.total_repairs),
        "generation_constraint_rejections": float(desc.generation_constraint_rejections),
        "candidate_repairs_applied": float(desc.candidate_repairs_applied),
        "rule_applied": float(desc.rule_applied),
        "rule_skipped": float(desc.rule_skipped),
        "descriptor_gap": _descriptor_gap(desc, descriptor_targets),
    }
    return row


def _summary_table(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for arm_name in sorted(df["arm"].unique().tolist()):
        sub = df[df["arm"] == arm_name]
        row = {"arm": arm_name, "n": int(len(sub))}
        for m in metrics:
            row[m] = float(sub[m].mean(skipna=True))
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_significance(
    df: pd.DataFrame,
    *,
    control_name: str,
    treatment_name: str,
    metrics: Sequence[str],
    seed: int,
) -> pd.DataFrame:
    left = df[df["arm"] == treatment_name]
    right = df[df["arm"] == control_name]
    merged = left.merge(right, on="seed", suffixes=("_treat", "_ctrl"))
    rows: List[Dict[str, Any]] = []
    if merged.empty:
        return pd.DataFrame(rows)
    for idx, metric in enumerate(metrics):
        a = merged[f"{metric}_treat"].astype(float).to_numpy()
        b = merged[f"{metric}_ctrl"].astype(float).to_numpy()
        deltas = a - b
        deltas = deltas[np.isfinite(deltas)]
        if deltas.size == 0:
            continue
        ci_low, ci_high = _paired_bootstrap_ci(
            deltas,
            n_boot=2000,
            alpha=0.05,
            seed=int(seed) + (17 * (idx + 1)),
        )
        p_value = _paired_sign_permutation_pvalue(
            deltas,
            n_perm=4000,
            seed=int(seed) + (31 * (idx + 1)),
        )
        std = float(np.std(deltas))
        effect_size = float(np.mean(deltas) / std) if std > 1e-9 else 0.0
        rows.append(
            {
                "metric": metric,
                "n_pairs": int(deltas.size),
                "delta_mean_treatment_minus_control": float(np.mean(deltas)),
                "delta_ci_low": float(ci_low),
                "delta_ci_high": float(ci_high),
                "p_value": float(p_value),
                "effect_size_d": float(effect_size),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_value_bh_fdr"] = _benjamini_hochberg(out["p_value"].astype(float).tolist())
    out["significant_fdr_0_05"] = out["p_value_bh_fdr"] < 0.05
    return out


def _fmt(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-seed A/B test for rule weight schedules.")
    parser.add_argument("--output", type=Path, default=Path("results") / "rule_weight_ab")
    parser.add_argument("--data-root", type=Path, default=Path("Data") / "The Legend of Zelda")
    parser.add_argument("--reference-limit", type=int, default=96)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-rooms", type=int, default=8)
    parser.add_argument("--max-rooms", type=int, default=16)
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--rule-space", type=str, default="full", choices=["core", "full"])
    parser.add_argument("--room-count-bias", type=float, default=0.45)
    parser.add_argument("--search-strategy", type=str, default="ga", choices=["ga", "cvt_emitter"])
    parser.add_argument("--qd-archive-cells", type=int, default=128)
    parser.add_argument("--qd-init-random-fraction", type=float, default=0.35)
    parser.add_argument("--qd-emitter-mutation-rate", type=float, default=0.18)
    parser.add_argument("--control-name", type=str, default="CONTROL")
    parser.add_argument("--treatment-name", type=str, default="TREATMENT")
    parser.add_argument("--control-weights", type=Path, default=None)
    parser.add_argument("--treatment-weights", type=Path, default=None)
    parser.add_argument("--calibrate-treatment", action="store_true")
    parser.add_argument("--calibration-iterations", type=int, default=4)
    parser.add_argument("--calibration-sample-size", type=int, default=12)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    if not args.verbose:
        logging.getLogger("src.data.vglc_utils").setLevel(logging.ERROR)
        logging.getLogger("src.generation.grammar").setLevel(logging.ERROR)
        logging.getLogger("src.generation.evolutionary_director").setLevel(logging.ERROR)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_graphs = load_vglc_reference_graphs(
        data_root=args.data_root,
        limit=args.reference_limit,
        filter_virtual_start_pointers=True,
    )
    descriptor_targets = _descriptor_target_means(reference_graphs)

    control_weights = _load_weights(args.control_weights)
    treatment_weights = _load_weights(args.treatment_weights)
    calibration_payload: Dict[str, Any] = {"enabled": bool(args.calibrate_treatment), "history": []}
    if args.calibrate_treatment:
        cal = calibrate_rule_weights_to_vglc(
            reference_graphs,
            seed=int(args.seed),
            iterations=int(args.calibration_iterations),
            sample_size=int(args.calibration_sample_size),
            min_rooms=int(args.min_rooms),
            max_rooms=int(args.max_rooms),
            population_size=int(args.population_size),
            generations=int(args.generations),
            rule_space=args.rule_space,
            room_count_bias=float(args.room_count_bias),
            search_strategy=args.search_strategy,
            qd_archive_cells=int(args.qd_archive_cells),
            qd_init_random_fraction=float(args.qd_init_random_fraction),
            qd_emitter_mutation_rate=float(args.qd_emitter_mutation_rate),
        )
        treatment_weights = {
            str(k): float(v)
            for k, v in dict(cal.get("calibrated_rule_weights", {})).items()
            if np.isfinite(v) and float(v) >= 0.0
        }
        calibration_payload = {
            "enabled": True,
            "best_gap": float(cal.get("best_gap", 0.0)),
            "history": cal.get("history", []),
        }

    arms = [
        ArmConfig(name=str(args.control_name), rule_weights=control_weights),
        ArmConfig(name=str(args.treatment_name), rule_weights=treatment_weights),
    ]
    seeds = [int(args.seed) + i for i in range(int(args.num_samples))]

    records: List[Dict[str, Any]] = []
    for s in seeds:
        logger.info("Seed %d", s)
        for arm in arms:
            row = _run_single_seed(
                seed=int(s),
                arm=arm,
                descriptor_targets=descriptor_targets,
                min_rooms=int(args.min_rooms),
                max_rooms=int(args.max_rooms),
                population_size=int(args.population_size),
                generations=int(args.generations),
                rule_space=str(args.rule_space),
                room_count_bias=float(args.room_count_bias),
                search_strategy=str(args.search_strategy),
                qd_archive_cells=int(args.qd_archive_cells),
                qd_init_random_fraction=float(args.qd_init_random_fraction),
                qd_emitter_mutation_rate=float(args.qd_emitter_mutation_rate),
            )
            records.append(row)

    raw_df = pd.DataFrame(records)
    metrics = [
        "overall_completeness",
        "constraint_valid",
        "path_exists",
        "linearity",
        "leniency",
        "progression_complexity",
        "topology_complexity",
        "cycle_density",
        "shortcut_density",
        "gate_depth_ratio",
        "path_depth_ratio",
        "directionality_gap",
        "repair_rate",
        "total_repairs",
        "generation_constraint_rejections",
        "generation_time_sec",
        "descriptor_gap",
    ]
    summary_df = _summary_table(raw_df, metrics=metrics)
    sig_df = _paired_significance(
        raw_df,
        control_name=str(args.control_name),
        treatment_name=str(args.treatment_name),
        metrics=metrics,
        seed=int(args.seed) + 999,
    )

    raw_path = out_dir / "rule_weight_ab_raw.csv"
    summary_path = out_dir / "rule_weight_ab_summary.csv"
    sig_path = out_dir / "rule_weight_ab_significance.csv"
    report_json = out_dir / "rule_weight_ab_report.json"
    report_md = out_dir / "rule_weight_ab_report.md"

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    sig_df.to_csv(sig_path, index=False)

    payload = {
        "control_name": str(args.control_name),
        "treatment_name": str(args.treatment_name),
        "seeds": seeds,
        "descriptor_targets": descriptor_targets,
        "control_weights": control_weights or {},
        "treatment_weights": treatment_weights or {},
        "calibration": calibration_payload,
        "summary": summary_df.to_dict(orient="records"),
        "significance": sig_df.to_dict(orient="records"),
    }
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Rule Weight A/B Report",
        "",
        f"- control: `{args.control_name}`",
        f"- treatment: `{args.treatment_name}`",
        f"- seeds: {len(seeds)}",
        f"- calibrated_treatment: {bool(args.calibrate_treatment)}",
        "",
        "## Mean Metrics by Arm",
        "",
        _fmt(summary_df),
        "",
        "## Paired Significance (Treatment - Control)",
        "",
        _fmt(sig_df) if not sig_df.empty else "_No paired significance rows available_",
    ]
    report_md.write_text("\n".join(lines), encoding="utf-8")

    logger.info("Saved A/B outputs to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
