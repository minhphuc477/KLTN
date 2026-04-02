"""Run a lightweight structural fairness / bias audit for generated room maps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.fairness_assessment import run_fairness_assessment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight structural fairness/bias audit over generated room maps.")
    parser.add_argument("--generated-dir", type=Path, required=True, help="Directory containing generated .npy room maps.")
    parser.add_argument("--reference-dir", type=Path, required=True, help="Directory containing reference .npy room maps.")
    parser.add_argument("--num-classes", type=int, default=44, help="Semantic vocabulary size.")
    parser.add_argument("--max-samples", type=int, default=200, help="Maximum number of maps to audit from each directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "fairness_bias_audit", help="Output directory for JSON and Markdown artifacts.")
    return parser.parse_args()


def _risk_band(report: Dict[str, Any]) -> str:
    jsd = float(report.get("jsd", 0.0))
    invalid_generated = int(report.get("generated_invalid_tile_count", 0))
    invalid_reference = int(report.get("reference_invalid_tile_count", 0))
    invalid_any = max(invalid_generated, invalid_reference)
    if invalid_any > 0 or jsd >= 0.20:
        return "high"
    if jsd >= 0.10:
        return "medium"
    return "low"


def _build_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Fairness / Bias Audit",
        "",
        "## Summary",
        "",
        f"- `risk_band`: {report.get('risk_band', 'unknown')}",
        f"- `generated_count`: {report.get('generated_count', 0)}",
        f"- `reference_count`: {report.get('reference_count', 0)}",
        f"- `jsd`: {float(report.get('jsd', 0.0)):.6f}",
        f"- `l1`: {float(report.get('l1', 0.0)):.6f}",
        f"- `l2`: {float(report.get('l2', 0.0)):.6f}",
        f"- `generated_invalid_tile_count`: {int(report.get('generated_invalid_tile_count', 0))}",
        f"- `reference_invalid_tile_count`: {int(report.get('reference_invalid_tile_count', 0))}",
        "",
        "## Distribution Statistics",
        "",
        f"- `generated_entropy`: {float(report.get('generated_entropy', 0.0)):.6f}",
        f"- `reference_entropy`: {float(report.get('reference_entropy', 0.0)):.6f}",
        f"- `generated_active_class_count`: {float(report.get('generated_active_class_count', 0.0)):.1f}",
        f"- `reference_active_class_count`: {float(report.get('reference_active_class_count', 0.0)):.1f}",
        f"- `generated_max_tile_share`: {float(report.get('generated_max_tile_share', 0.0)):.6f}",
        f"- `reference_max_tile_share`: {float(report.get('reference_max_tile_share', 0.0)):.6f}",
        "",
        "## Limits",
        "",
        "- This is a structural smoke test, not a full demographic fairness study.",
        "- It measures distribution drift, invalid tile IDs, entropy, and active-class coverage.",
        "- Human-subject evaluation and downstream difficulty/fairness studies remain separate research work.",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = run_fairness_assessment(
        generated_dir=args.generated_dir,
        reference_dir=args.reference_dir,
        num_classes=int(args.num_classes),
        max_samples=(int(args.max_samples) if args.max_samples else None),
    )
    report["risk_band"] = _risk_band(report)

    json_path = out_dir / "fairness_bias_audit.json"
    md_path = out_dir / "fairness_bias_audit.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_build_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
