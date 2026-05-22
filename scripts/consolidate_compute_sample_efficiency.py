"""Consolidate compute and sample-efficiency evidence from local artifacts.

This script does not launch training or generation. It scans existing JSON/CSV
artifacts and produces a thesis-facing inventory of runtime, sample count, and
best observed metric fields. Missing fields are reported explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RUNTIME_KEYS = {
    "wall_time_sec",
    "runtime_sec",
    "duration_sec",
    "elapsed_sec",
    "time_elapsed",
    "total_runtime_sec",
    "generation_time_sec",
    "mean_generation_time_sec",
    "train_time_sec",
    "training_time_sec",
}

SAMPLE_KEYS = {
    "num_samples",
    "n_samples",
    "sample_count",
    "num_generated",
    "n",
    "reference_room_count",
    "room_count",
    "event_count",
}

STEP_KEYS = {
    "epoch",
    "epochs",
    "best_epoch",
    "step",
    "steps",
    "global_step",
    "iteration",
    "iterations",
}

LOSS_KEYS = {
    "loss",
    "train_loss",
    "val_loss",
    "validation_loss",
    "reconstruction_error",
    "tile_prior_kl",
    "graph_edit_distance",
    "macro_norm_error",
}

SUCCESS_KEYS = {
    "solvable",
    "constraint_valid",
    "hybrid_oracle_pass",
    "path_exists_rate",
    "constraint_valid_rate",
    "overall_completeness",
    "success_rate",
    "controlled_pass_rate",
    "controlled_pass_rate_mean",
    "pass_all_rate",
    "macro_metric_pass_rate",
}

ALL_INTERESTING_KEYS = RUNTIME_KEYS | SAMPLE_KEYS | STEP_KEYS | LOSS_KEYS | SUCCESS_KEYS


@dataclass
class Observation:
    run_id: str
    source_path: str
    source_format: str
    metric: str
    value: float
    row_index: Optional[int] = None

    def to_row(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "source_path": self.source_path,
            "source_format": self.source_format,
            "metric": self.metric,
            "value": self.value,
            "row_index": "" if self.row_index is None else int(self.row_index),
        }


def _normalized_key(key: str) -> str:
    return str(key).strip().lower().replace("-", "_").replace(" ", "_")


def _is_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _flatten_json(obj: Any, *, prefix: str = "", depth: int = 0, max_depth: int = 6) -> Dict[str, Any]:
    if depth > max_depth:
        return {}
    if isinstance(obj, Mapping):
        out: Dict[str, Any] = {}
        for key, value in obj.items():
            norm = _normalized_key(str(key))
            child_prefix = f"{prefix}.{norm}" if prefix else norm
            out.update(_flatten_json(value, prefix=child_prefix, depth=depth + 1, max_depth=max_depth))
        return out
    if isinstance(obj, list):
        out = {}
        for idx, value in enumerate(obj[:200]):
            child_prefix = f"{prefix}[{idx}]"
            out.update(_flatten_json(value, prefix=child_prefix, depth=depth + 1, max_depth=max_depth))
        return out
    return {prefix: obj}


def _metric_leaf(path_key: str) -> str:
    cleaned = str(path_key).replace("[", ".").replace("]", "")
    return _normalized_key(cleaned.split(".")[-1])


def _interesting_metric(path_key: str) -> Optional[str]:
    leaf = _metric_leaf(path_key)
    if leaf in ALL_INTERESTING_KEYS:
        return leaf
    for key in ALL_INTERESTING_KEYS:
        if leaf.endswith(f"_{key}") or leaf == key:
            return key
    return None


def _infer_run_id(path: Path, roots: Sequence[Path]) -> str:
    resolved = path.resolve()
    for root in roots:
        try:
            rel = resolved.relative_to(root.resolve())
        except ValueError:
            continue
        parts = rel.parts
        if not parts:
            return root.name
        if len(parts) == 1:
            return parts[0]
        if Path(parts[1]).suffix:
            return parts[0]
        if parts[1].lower() in {"checkpoints", "figures", "logs", "metrics", "plots", "reports"}:
            return parts[0]
        return str(Path(parts[0]) / parts[1])
    return str(path.parent)


def _iter_artifact_files(roots: Sequence[Path], max_file_mb: float) -> Iterable[Path]:
    suffixes = {".json", ".jsonl", ".csv"}
    max_bytes = int(max(1.0, float(max_file_mb)) * 1024 * 1024)
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            try:
                if path.stat().st_size > max_bytes:
                    continue
            except OSError:
                continue
            yield path


def _observations_from_json(path: Path, run_id: str) -> List[Observation]:
    observations: List[Observation] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return observations
    flat = _flatten_json(payload)
    for key, value in flat.items():
        metric = _interesting_metric(key)
        if metric is None or not _is_number(value):
            continue
        observations.append(
            Observation(
                run_id=run_id,
                source_path=str(path),
                source_format="json",
                metric=metric,
                value=float(value),
            )
        )
    return observations


def _observations_from_jsonl(path: Path, run_id: str, max_rows: int) -> List[Observation]:
    observations: List[Observation] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError:
        return observations
    with handle:
        for idx, line in enumerate(handle):
            if idx >= int(max_rows):
                break
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            for key, value in _flatten_json(payload).items():
                metric = _interesting_metric(key)
                if metric is None or not _is_number(value):
                    continue
                observations.append(
                    Observation(
                        run_id=run_id,
                        source_path=str(path),
                        source_format="jsonl",
                        metric=metric,
                        value=float(value),
                        row_index=idx,
                    )
                )
    return observations


def _observations_from_csv(path: Path, run_id: str, max_rows: int) -> List[Observation]:
    observations: List[Observation] = []
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except OSError:
        return observations
    with handle:
        try:
            reader = csv.DictReader(handle)
        except csv.Error:
            return observations
        for idx, row in enumerate(reader):
            if idx >= int(max_rows):
                break
            for key, value in row.items():
                metric = _interesting_metric(str(key))
                if metric is None or not _is_number(value):
                    continue
                observations.append(
                    Observation(
                        run_id=run_id,
                        source_path=str(path),
                        source_format="csv",
                        metric=metric,
                        value=float(value),
                        row_index=idx,
                    )
                )
    return observations


def collect_compute_observations(
    roots: Sequence[Path],
    *,
    max_file_mb: float = 25.0,
    max_rows_per_table: int = 20000,
) -> Tuple[List[Observation], List[Dict[str, Any]]]:
    resolved_roots = [Path(root) for root in roots]
    observations: List[Observation] = []
    inventory: Dict[str, Dict[str, Any]] = {}

    for path in _iter_artifact_files(resolved_roots, max_file_mb=max_file_mb):
        run_id = _infer_run_id(path, resolved_roots)
        inv = inventory.setdefault(
            run_id,
            {
                "run_id": run_id,
                "file_count": 0,
                "json_count": 0,
                "jsonl_count": 0,
                "csv_count": 0,
                "checkpoint_count": 0,
                "total_bytes": 0,
                "latest_mtime": 0.0,
            },
        )
        try:
            stat = path.stat()
            inv["file_count"] += 1
            inv["total_bytes"] += int(stat.st_size)
            inv["latest_mtime"] = max(float(inv["latest_mtime"]), float(stat.st_mtime))
        except OSError:
            continue

        suffix = path.suffix.lower()
        if suffix == ".json":
            inv["json_count"] += 1
            observations.extend(_observations_from_json(path, run_id))
        elif suffix == ".jsonl":
            inv["jsonl_count"] += 1
            observations.extend(_observations_from_jsonl(path, run_id, max_rows=max_rows_per_table))
        elif suffix == ".csv":
            inv["csv_count"] += 1
            observations.extend(_observations_from_csv(path, run_id, max_rows=max_rows_per_table))

    # Checkpoints are counted separately because they are often binary files.
    for root in resolved_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".pth", ".pt", ".ckpt"}:
                continue
            run_id = _infer_run_id(path, resolved_roots)
            inv = inventory.setdefault(
                run_id,
                {
                    "run_id": run_id,
                    "file_count": 0,
                    "json_count": 0,
                    "jsonl_count": 0,
                    "csv_count": 0,
                    "checkpoint_count": 0,
                    "total_bytes": 0,
                    "latest_mtime": 0.0,
                },
            )
            try:
                stat = path.stat()
                inv["checkpoint_count"] += 1
                inv["total_bytes"] += int(stat.st_size)
                inv["latest_mtime"] = max(float(inv["latest_mtime"]), float(stat.st_mtime))
            except OSError:
                continue

    return observations, sorted(inventory.values(), key=lambda row: str(row["run_id"]))


def _values_by_metric(observations: Sequence[Observation]) -> Dict[str, List[float]]:
    grouped: Dict[str, List[float]] = {}
    for obs in observations:
        grouped.setdefault(obs.metric, []).append(float(obs.value))
    return grouped


def _safe_max(values: Sequence[float]) -> Optional[float]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return max(finite) if finite else None


def _safe_min(values: Sequence[float]) -> Optional[float]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return min(finite) if finite else None


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(finite) / len(finite)) if finite else None


def summarize_compute_efficiency(observations: Sequence[Observation]) -> List[Dict[str, Any]]:
    by_run: Dict[str, List[Observation]] = {}
    for obs in observations:
        by_run.setdefault(obs.run_id, []).append(obs)

    rows: List[Dict[str, Any]] = []
    for run_id, run_obs in sorted(by_run.items()):
        grouped = _values_by_metric(run_obs)
        runtime_candidates = [value for key in RUNTIME_KEYS for value in grouped.get(key, [])]
        sample_candidates = [value for key in SAMPLE_KEYS for value in grouped.get(key, [])]
        step_candidates = [value for key in STEP_KEYS for value in grouped.get(key, [])]
        loss_candidates = [value for key in LOSS_KEYS for value in grouped.get(key, [])]
        success_candidates = [value for key in SUCCESS_KEYS for value in grouped.get(key, [])]

        runtime_sec = _safe_max(runtime_candidates)
        sample_count = _safe_max(sample_candidates)
        rows.append(
            {
                "run_id": run_id,
                "observation_count": int(len(run_obs)),
                "observed_runtime_sec": "" if runtime_sec is None else runtime_sec,
                "observed_runtime_hours": "" if runtime_sec is None else runtime_sec / 3600.0,
                "observed_sample_count": "" if sample_count is None else sample_count,
                "samples_per_sec": (
                    ""
                    if runtime_sec is None or sample_count is None or runtime_sec <= 0.0
                    else float(sample_count) / float(runtime_sec)
                ),
                "sec_per_sample": (
                    ""
                    if runtime_sec is None or sample_count is None or sample_count <= 0.0
                    else float(runtime_sec) / float(sample_count)
                ),
                "max_step_or_epoch": "" if _safe_max(step_candidates) is None else _safe_max(step_candidates),
                "best_loss_like_metric": "" if _safe_min(loss_candidates) is None else _safe_min(loss_candidates),
                "best_success_like_metric": "" if _safe_max(success_candidates) is None else _safe_max(success_candidates),
                "mean_generation_time_sec": (
                    ""
                    if _safe_mean(grouped.get("generation_time_sec", []) + grouped.get("mean_generation_time_sec", [])) is None
                    else _safe_mean(grouped.get("generation_time_sec", []) + grouped.get("mean_generation_time_sec", []))
                ),
                "missing_runtime": int(runtime_sec is None),
                "missing_sample_count": int(sample_count is None),
                "missing_loss_metric": int(_safe_min(loss_candidates) is None),
                "missing_success_metric": int(_safe_max(success_candidates) is None),
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


def write_report(path: Path, summary: Sequence[Mapping[str, Any]], inventory: Sequence[Mapping[str, Any]]) -> None:
    missing_runtime = sum(int(row.get("missing_runtime", 0)) for row in summary)
    missing_samples = sum(int(row.get("missing_sample_count", 0)) for row in summary)
    missing_losses = sum(int(row.get("missing_loss_metric", 0)) for row in summary)
    missing_success = sum(int(row.get("missing_success_metric", 0)) for row in summary)
    lines = [
        "# Consolidated Compute And Sample-Efficiency Report",
        "",
        "Generated by `scripts/consolidate_compute_sample_efficiency.py`.",
        "",
        "## Coverage",
        "",
        f"- runs with metric observations: `{len(summary)}`",
        f"- artifact run roots inventoried: `{len(inventory)}`",
        f"- runs missing runtime fields: `{missing_runtime}`",
        f"- runs missing sample-count fields: `{missing_samples}`",
        f"- runs missing loss-like metrics: `{missing_losses}`",
        f"- runs missing success-like metrics: `{missing_success}`",
        "",
        "## Interpretation Rules",
        "",
        "- `observed_runtime_sec` is the largest runtime-like value found for a run, not guaranteed GPU time.",
        "- `samples_per_sec` is only populated when both runtime and sample-count fields exist.",
        "- `best_loss_like_metric` is the minimum among loss/KL/error fields; compare only within compatible experiment families.",
        "- `best_success_like_metric` is the maximum among pass/success/completeness fields; compare only within compatible experiment families.",
        "",
        "## Missing Field Contract",
        "",
        "For final thesis tables, each long run should export at least:",
        "",
        "- `wall_time_sec` or `runtime_sec`",
        "- `num_samples` or task-specific generated sample count",
        "- one loss/error metric for model training branches",
        "- one success/pass metric for generation/evaluation branches",
        "- seed, config snapshot, and checkpoint path metadata",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidate compute/sample-efficiency evidence from artifacts.")
    parser.add_argument("--roots", nargs="*", type=Path, default=[Path("outputs"), Path("results")])
    parser.add_argument("--output", type=Path, default=Path("results") / "compute_sample_efficiency")
    parser.add_argument("--max-file-mb", type=float, default=25.0)
    parser.add_argument("--max-rows-per-table", type=int, default=20000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    observations, inventory = collect_compute_observations(
        args.roots,
        max_file_mb=float(args.max_file_mb),
        max_rows_per_table=int(args.max_rows_per_table),
    )
    summary = summarize_compute_efficiency(observations)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "metric_observations.csv", [obs.to_row() for obs in observations])
    write_csv(args.output / "artifact_inventory.csv", inventory)
    write_csv(args.output / "compute_sample_efficiency_summary.csv", summary)
    payload = {
        "roots": [str(root) for root in args.roots],
        "observation_count": len(observations),
        "inventory_count": len(inventory),
        "summary": summary,
    }
    (args.output / "compute_sample_efficiency_payload.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    write_report(args.output / "compute_sample_efficiency_report.md", summary, inventory)
    print(f"Wrote compute/sample-efficiency consolidation to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
