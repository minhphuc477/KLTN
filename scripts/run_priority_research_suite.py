"""Run consolidated P0/P1/P2(+others) research suite and produce one report.

This orchestrates key research scripts already in the repo and aggregates
artifacts/metrics into a single JSON + Markdown summary.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
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
    category: str
    command: List[str]
    output: Path


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _artifact_summary_lines(step_name: str, payload: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    summary_rows = payload.get("summary")
    if isinstance(summary_rows, list) and summary_rows:
        first = summary_rows[0] if isinstance(summary_rows[0], dict) else None
        if isinstance(first, dict):
            if step_name == "matched_budget":
                for row in summary_rows:
                    if not isinstance(row, dict):
                        continue
                    method = str(row.get("method", "unknown"))
                    completeness = _safe_float(row.get("overall_completeness"))
                    valid_rate = _safe_float(row.get("constraint_valid_rate"))
                    lines.append(
                        f"- `{method}`: completeness={completeness:.3f}, constraint_valid_rate={valid_rate:.3f}"
                        if completeness is not None and valid_rate is not None
                        else f"- `{method}`: summary row present"
                    )
            elif step_name == "room_branch_benchmark":
                for row in summary_rows:
                    if not isinstance(row, dict):
                        continue
                    cfg = str(row.get("config", row.get("name", "unknown")))
                    valid_rate = _safe_float(row.get("valid_rate"))
                    repair_rate = _safe_float(row.get("room_repair_rate"))
                    diversity = _safe_float(row.get("diversity"))
                    parts = [f"`{cfg}`"]
                    if valid_rate is not None:
                        parts.append(f"valid_rate={valid_rate:.3f}")
                    if repair_rate is not None:
                        parts.append(f"repair_rate={repair_rate:.3f}")
                    if diversity is not None:
                        parts.append(f"diversity={diversity:.3f}")
                    lines.append("- " + ", ".join(parts))
            elif step_name == "ablation_fixed_seed":
                for row in summary_rows:
                    if not isinstance(row, dict):
                        continue
                    cfg = str(row.get("config", row.get("name", "unknown")))
                    valid_rate = _safe_float(row.get("valid_rate"))
                    repair_rate = _safe_float(row.get("room_repair_rate"))
                    novelty = _safe_float(row.get("novelty_vs_reference"))
                    parts = [f"`{cfg}`"]
                    if valid_rate is not None:
                        parts.append(f"valid_rate={valid_rate:.3f}")
                    if repair_rate is not None:
                        parts.append(f"repair_rate={repair_rate:.3f}")
                    if novelty is not None:
                        parts.append(f"novelty={novelty:.3f}")
                    lines.append("- " + ", ".join(parts))
    if not lines:
        notes = payload.get("notes")
        if isinstance(notes, list) and notes:
            for note in notes[:2]:
                lines.append(f"- {note}")
    return lines


def _run(command: List[str], cwd: Path, timeout_sec: Optional[int]) -> StepResult:
    started = time.time()
    timeout = int(timeout_sec) if timeout_sec and timeout_sec > 0 else None
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    start_new_session = os.name != "nt"
    proc: Optional[subprocess.Popen[str]] = None
    try:
        proc = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
            start_new_session=start_new_session,
        )
        out, err = proc.communicate(timeout=timeout)
        duration = float(time.time() - started)
        return StepResult(
            name="",
            command=command,
            exit_code=int(proc.returncode),
            duration_sec=duration,
            stdout_tail=out[-2000:],
            stderr_tail=err[-2000:],
        )
    except subprocess.TimeoutExpired:
        if proc is not None:
            _terminate_process_tree(proc)
            try:
                out, err = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                out, err = "", ""
        else:
            out, err = "", ""
        duration = float(time.time() - started)
        return StepResult(
            name="",
            command=command,
            exit_code=124,
            duration_sec=duration,
            stdout_tail=str(out or "")[-2000:],
            stderr_tail=(str(err or "") + "\n[timeout] step exceeded timeout_sec")[-2000:],
        )


def _terminate_process_tree(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


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
            category="topology",
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
            category="full_stack",
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
            name="room_branch_benchmark",
            priority="p1",
            category="room_branch",
            command=[
                py,
                "scripts/run_room_branch_benchmark.py",
                "--num-samples",
                str(int(quick_samples)),
                "--diffusion-steps",
                str(int(8 if bool(args.quick) else 25)),
                "--evolution-population",
                str(int(12 if bool(args.quick) else 24)),
                "--evolution-generations",
                str(int(12 if bool(args.quick) else 30)),
                "--seed",
                str(int(args.seed)),
                "--output",
                str(out / "room_branch_benchmark"),
                *( ["--quick"] if bool(args.quick) else [] ),
            ],
            output=out / "room_branch_benchmark" / "room_branch_benchmark_report.json",
        ),
        StepSpec(
            name="sequence_break_analysis",
            priority="p2",
            category="full_stack",
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
            category="topology",
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
            category="full_stack",
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
            category="topology",
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
            category="topology",
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
            category="topology",
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
        help="Comma-separated step names or 'all'. Names: matched_budget,ablation_fixed_seed,room_branch_benchmark,sequence_break_analysis,rule_marginal_credit,ood_blinded_eval,rule_weight_ab_test,feature_distribution,topology_rubric",
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
                    "category": str(spec.category),
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
                "priority": str(next((spec.priority for spec in steps if spec.name == r.name), "unknown")),
                "category": str(next((spec.category for spec in steps if spec.name == r.name), "unknown")),
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
        spec = next((item for item in steps if item.name == r.name), None)
        category = getattr(spec, "category", "unknown")
        priority = getattr(spec, "priority", "unknown")
        lines.append(
            f"- {r.name}: category={category}, priority={priority}, exit={r.exit_code}, duration_sec={r.duration_sec:.2f}, output={r.output_path}"
        )

    category_titles = {
        "topology": "Topology Evidence",
        "room_branch": "Room-Branch Evidence",
        "full_stack": "Full-Stack Evidence",
    }
    for category_key, title in category_titles.items():
        category_steps = [r for r in executed if next((spec.category for spec in steps if spec.name == r.name), None) == category_key]
        if not category_steps:
            continue
        lines.extend(["", f"## {title}", ""])
        for r in category_steps:
            lines.append(
                f"### {r.name}"
            )
            lines.append("")
            lines.append(
                f"- exit_code: {r.exit_code}"
            )
            lines.append(
                f"- duration_sec: {r.duration_sec:.2f}"
            )
            lines.append(
                f"- output: {r.output_path}"
            )
            artifact_payload = suite_payload["artifacts"].get(r.name)
            artifact_lines = (
                _artifact_summary_lines(r.name, artifact_payload)
                if isinstance(artifact_payload, dict)
                else []
            )
            if artifact_lines:
                lines.append("- highlights:")
                lines.extend([f"  {line}" for line in artifact_lines])

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report_json": str(json_path), "report_md": str(md_path)}, indent=2))
    return 0 if bool(suite_payload["all_steps_passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
