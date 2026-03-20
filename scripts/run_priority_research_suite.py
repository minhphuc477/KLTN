"""Run consolidated P0/P1/P2(+others) research suite and produce one report.

This orchestrates key research scripts already in the repo and aggregates
artifacts/metrics into a single JSON + Markdown summary.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class StepResult:
    name: str
    command: List[str]
    exit_code: int
    duration_sec: float
    stdout_tail: str
    stderr_tail: str
    output_path: Optional[str] = None


@dataclass(frozen=True)
class StepSpec:
    name: str
    priority: str
    command: List[str]
    output: Path


def _run(command: List[str], cwd: Path, timeout_sec: Optional[int]) -> StepResult:
    started = time.time()
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=(int(timeout_sec) if timeout_sec and timeout_sec > 0 else None),
        )
        duration = float(time.time() - started)
        out = str(proc.stdout or "")
        err = str(proc.stderr or "")
        return StepResult(
            name="",
            command=command,
            exit_code=int(proc.returncode),
            duration_sec=duration,
            stdout_tail=out[-2000:],
            stderr_tail=err[-2000:],
        )
    except subprocess.TimeoutExpired as exc:
        duration = float(time.time() - started)
        out = str(exc.stdout or "")
        err = str(exc.stderr or "")
        return StepResult(
            name="",
            command=command,
            exit_code=124,
            duration_sec=duration,
            stdout_tail=out[-2000:],
            stderr_tail=(err + "\n[timeout] step exceeded timeout_sec")[ -2000:],
        )


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _build_steps(args: argparse.Namespace) -> List[StepSpec]:
    py = str(args.python_exe)
    out = Path(args.output_dir)
    quick_samples = int(max(2, min(4, int(args.num_samples)))) if bool(args.quick) else int(args.num_samples)
    quick_eval_budget = int(max(64, min(128, int(args.eval_budget)))) if bool(args.quick) else int(args.eval_budget)
    return [
        StepSpec(
            name="matched_budget",
            priority="p0",
            command=[
                py,
                "scripts/run_matched_budget_topology_benchmark.py",
                "--methods",
                "RANDOM,ES,GA,MAP_ELITES,FULL",
                "--num-samples",
                str(int(quick_samples)),
                "--seed",
                str(int(args.seed)),
                "--eval-budget",
                str(int(quick_eval_budget)),
                "--output",
                str(out / "matched_budget"),
            ],
            output=out / "matched_budget" / "matched_budget_report.json",
        ),
        StepSpec(
            name="ablation_fixed_seed",
            priority="p1",
            command=[
                py,
                "scripts/run_ablation_study.py",
                "--num-samples",
                str(int(quick_samples)),
                "--evolution-population",
                str(int(12 if bool(args.quick) else 24)),
                "--evolution-generations",
                str(int(12 if bool(args.quick) else 30)),
                "--diffusion-steps",
                str(int(8 if bool(args.quick) else 25)),
                *( ["--quick"] if bool(args.quick) else [] ),
                *( ["--max-runtime-sec", "240"] if bool(args.quick) else [] ),
                *( ["--configs", "FULL,NO_EVOLUTION"] if bool(args.quick) else [] ),
                "--seed",
                str(int(args.seed)),
                "--output",
                str(out / "ablation"),
                *( ["--core-only"] if bool(args.quick) else [] ),
            ],
            output=out / "ablation" / "ablation_report.json",
        ),
        StepSpec(
            name="sequence_break_analysis",
            priority="p2",
            command=[
                py,
                "scripts/analyze_sequence_breaks.py",
                "--num-samples",
                str(int(max(4, quick_samples))),
                "--seed",
                str(int(args.seed)),
                "--output",
                str(out / "sequence_break_analysis.json"),
            ],
            output=out / "sequence_break_analysis.json",
        ),
        StepSpec(
            name="rule_marginal_credit",
            priority="p2",
            command=[
                py,
                "scripts/analyze_rule_marginal_credit.py",
                "--seed",
                str(int(args.seed)),
                "--output",
                str(out / "rule_marginal_credit.json"),
            ],
            output=out / "rule_marginal_credit.json",
        ),
        StepSpec(
            name="ood_blinded_eval",
            priority="others",
            command=[
                py,
                "scripts/run_ood_scaling_and_blinded_eval.py",
                "--num-samples",
                str(int(max(4, quick_samples))),
                "--seed",
                str(int(args.seed)),
                "--output",
                str(out / "ood_blinded_eval"),
            ],
            output=out / "ood_blinded_eval" / "ood_scaling_report.json",
        ),
        StepSpec(
            name="rule_weight_ab_test",
            priority="p1",
            command=[
                py,
                "scripts/run_rule_weight_ab_test.py",
                "--num-samples",
                str(int(quick_samples)),
                "--seed",
                str(int(args.seed)),
                "--output",
                str(out / "rule_weight_ab"),
            ],
            output=out / "rule_weight_ab" / "rule_weight_ab_report.json",
        ),
        StepSpec(
            name="feature_distribution",
            priority="p2",
            command=[
                py,
                "scripts/analyze_block_i_feature_distribution.py",
                "--num-samples",
                str(int(100 if bool(args.full_research) else (6 if bool(args.quick) else max(12, quick_samples)))),
                "--population-size",
                str(int(16 if bool(args.quick) else 32)),
                "--generations",
                str(int(16 if bool(args.quick) else 40)),
                "--seed",
                str(int(args.seed)),
                "--output",
                str(out / "feature_distribution"),
            ],
            output=out / "feature_distribution" / "feature_distribution_summary.json",
        ),
        StepSpec(
            name="topology_rubric",
            priority="others",
            command=[
                py,
                "scripts/score_topology_stack_rubric.py",
                "--output-dir",
                str(out / "topology_rubric"),
            ],
            output=out / "topology_rubric" / "topology_stack_rubric_report.json",
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run consolidated thesis research suite")
    parser.add_argument("--python-exe", type=Path, default=Path(sys.executable))
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "priority_research_suite")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--eval-budget", type=int, default=256)
    parser.add_argument(
        "--priority",
        type=str,
        default="all",
        help="Priority filter when --steps=all. One of: all,p0,p1,p2,others.",
    )
    parser.add_argument("--quick", action="store_true", help="Use bounded sample/budget defaults for faster runs.")
    parser.add_argument("--full-research", action="store_true", help="Enable heavier research defaults (e.g., 100-sample feature distribution).")
    parser.add_argument(
        "--steps",
        type=str,
        default="all",
        help="Comma-separated step names or 'all'. Names: matched_budget,ablation_fixed_seed,sequence_break_analysis,rule_marginal_credit,ood_blinded_eval,rule_weight_ab_test,feature_distribution,topology_rubric",
    )
    parser.add_argument("--step-timeout-sec", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_steps = _build_steps(args)
    if str(args.steps).strip().lower() == "all":
        priority = str(args.priority).strip().lower()
        allowed = {"all", "p0", "p1", "p2", "others"}
        if priority not in allowed:
            raise ValueError(f"Unknown --priority '{priority}'. Expected one of {sorted(allowed)}")
        steps = all_steps if priority == "all" else [spec for spec in all_steps if spec.priority == priority]
    else:
        requested = {part.strip() for part in str(args.steps).split(",") if part.strip()}
        steps = [spec for spec in all_steps if str(spec.name) in requested]
        missing = sorted(requested - {str(spec.name) for spec in all_steps})
        if missing:
            raise ValueError(f"Unknown --steps values: {missing}")
        if not steps:
            raise ValueError("No steps selected. Use --steps all or valid step names.")
    executed: List[StepResult] = []

    if bool(args.dry_run):
        payload = {
            "seed": int(args.seed),
            "num_samples": int(args.num_samples),
            "eval_budget": int(args.eval_budget),
            "output_dir": str(out_dir),
            "steps": [
                {
                    "name": str(spec.name),
                    "priority": str(spec.priority),
                    "command": spec.command,
                    "output": str(spec.output),
                }
                for spec in steps
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0

    for spec in steps:
        result = _run(spec.command, cwd=ROOT, timeout_sec=args.step_timeout_sec)
        result.name = str(spec.name)
        result.output_path = str(spec.output)
        executed.append(result)
        if result.exit_code != 0 and bool(args.stop_on_error):
            break

    suite_payload: Dict[str, Any] = {
        "seed": int(args.seed),
        "num_samples": int(args.num_samples),
        "eval_budget": int(args.eval_budget),
        "output_dir": str(out_dir),
        "all_steps_passed": bool(all(r.exit_code == 0 for r in executed)),
        "steps": [
            {
                "name": r.name,
                "command": r.command,
                "exit_code": r.exit_code,
                "duration_sec": r.duration_sec,
                "output_path": r.output_path,
                "stdout_tail": r.stdout_tail,
                "stderr_tail": r.stderr_tail,
            }
            for r in executed
        ],
        "artifacts": {},
    }

    for r in executed:
        out_path = Path(r.output_path) if r.output_path else None
        if out_path is None:
            continue
        payload = _safe_read_json(out_path)
        if payload is not None:
            suite_payload["artifacts"][r.name] = payload

    json_path = out_dir / "priority_research_suite_report.json"
    md_path = out_dir / "priority_research_suite_report.md"
    json_path.write_text(json.dumps(suite_payload, indent=2), encoding="utf-8")

    lines: List[str] = [
        "# Priority Research Suite Report",
        "",
        f"- all_steps_passed: {suite_payload['all_steps_passed']}",
        f"- seed: {suite_payload['seed']}",
        f"- num_samples: {suite_payload['num_samples']}",
        f"- eval_budget: {suite_payload['eval_budget']}",
        "",
        "## Steps",
        "",
    ]
    for r in executed:
        lines.append(f"- {r.name}: exit={r.exit_code}, duration_sec={r.duration_sec:.2f}, output={r.output_path}")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report_json": str(json_path), "report_md": str(md_path)}, indent=2))
    return 0 if bool(suite_payload["all_steps_passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
