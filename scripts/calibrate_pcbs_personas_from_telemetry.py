#!/usr/bin/env python3
"""Calibrate P-CBS persona parameters from local playtest telemetry."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.pcbs_telemetry_calibration import (
    aggregate_sessions,
    calibrate_persona_overrides,
    load_telemetry_sessions,
    render_calibration_markdown,
)


def _split_csv(raw: str) -> List[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive P-CBS PersonaConfig overrides from playtest telemetry."
    )
    parser.add_argument(
        "--telemetry",
        nargs="+",
        required=True,
        help="Telemetry JSON/JSONL/CSV files or directories from src.utils.playtest_telemetry.",
    )
    parser.add_argument(
        "--pcbs-sweep-csv",
        nargs="*",
        default=[],
        help="Optional CSV output from scripts/run_pcbs_persona_map_sweep.py for baseline correction.",
    )
    parser.add_argument(
        "--personas",
        default="novice,balanced,speedrunner,explorer,cautious,forgetful,completionist,greedy",
        help="Comma-separated P-CBS personas to calibrate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "pcbs_telemetry_calibration",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    telemetry_sessions = load_telemetry_sessions(args.telemetry)
    if not telemetry_sessions:
        raise SystemExit("No telemetry sessions found.")
    targets = aggregate_sessions(telemetry_sessions)

    baselines = None
    if args.pcbs_sweep_csv:
        baseline_sessions = load_telemetry_sessions(args.pcbs_sweep_csv)
        baselines = aggregate_sessions(baseline_sessions)

    overrides = calibrate_persona_overrides(
        targets,
        pcbs_baselines=baselines,
        personas=_split_csv(args.personas),
    )
    if not overrides:
        raise SystemExit("No persona overrides were produced; check telemetry persona labels.")

    targets_payload: Dict[str, Any] = {name: asdict(value) for name, value in targets.items()}
    baselines_payload = None if baselines is None else {name: asdict(value) for name, value in baselines.items()}

    (args.output_dir / "pcbs_telemetry_targets.json").write_text(
        json.dumps(targets_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if baselines_payload is not None:
        (args.output_dir / "pcbs_baseline_metrics.json").write_text(
            json.dumps(baselines_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    (args.output_dir / "pcbs_persona_overrides.json").write_text(
        json.dumps(overrides, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.output_dir / "pcbs_calibration_report.md").write_text(
        render_calibration_markdown(targets, overrides),
        encoding="utf-8",
    )

    print(f"Wrote calibration artifacts to {args.output_dir}")
    print(f"Telemetry sessions: {len(telemetry_sessions)}")
    print(f"Calibrated personas: {', '.join(sorted(overrides))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
