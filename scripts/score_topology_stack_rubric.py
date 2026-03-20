"""Automated scoring for the topology stack rubric.

Runs reproducible checks and emits:
- JSON machine-readable report
- Markdown checklist report

Usage:
    f:/KLTN/.venv/Scripts/python.exe scripts/score_topology_stack_rubric.py
    f:/KLTN/.venv/Scripts/python.exe scripts/score_topology_stack_rubric.py --output-dir results/topology_rubric
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class CommandResult:
    command: str
    exit_code: int
    duration_sec: float
    stdout: str
    stderr: str


@dataclass
class CriterionScore:
    id: str
    name: str
    score: int
    max_score: int
    passed: bool
    rationale: str
    evidence: List[str]


def _run_command(command: Sequence[str], cwd: Path) -> CommandResult:
    started = time.time()
    proc = subprocess.run(
        list(command),
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    duration = float(time.time() - started)
    return CommandResult(
        command=" ".join(command),
        exit_code=int(proc.returncode),
        duration_sec=duration,
        stdout=str(proc.stdout or ""),
        stderr=str(proc.stderr or ""),
    )


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _has_all(text: str, needles: Sequence[str]) -> bool:
    return all(n in text for n in needles)


def _extract_pytest_pass_count(output: str) -> int:
    m = re.search(r"(\d+)\s+passed", output)
    if not m:
        return 0
    return int(m.group(1))


def _score_binary(
    cid: str,
    name: str,
    passed: bool,
    rationale_ok: str,
    rationale_bad: str,
    evidence: List[str],
    max_score: int = 5,
) -> CriterionScore:
    return CriterionScore(
        id=cid,
        name=name,
        score=max_score if passed else 0,
        max_score=max_score,
        passed=bool(passed),
        rationale=rationale_ok if passed else rationale_bad,
        evidence=evidence,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Score topology stack rubric reproducibly.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/topology_rubric",
        help="Directory for JSON/Markdown rubric outputs.",
    )
    parser.add_argument(
        "--python-exe",
        type=str,
        default=sys.executable,
        help="Python executable for pytest invocations.",
    )
    args = parser.parse_args()

    root = ROOT
    out_dir = (root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline_py = _load_text(root / "src/pipeline/dungeon_pipeline.py")
    grammar_py = _load_text(root / "src/generation/grammar.py")
    evolution_py = _load_text(root / "src/generation/evolutionary_director.py")
    graph_utils_py = _load_text(root / "src/utils/graph_utils.py")
    vglc_py = _load_text(root / "src/data/vglc_utils.py")
    bench_py = _load_text(root / "src/evaluation/benchmark_suite.py")

    # Reproducible test runs (C6, C7)
    cmd_topology = _run_command(
        [args.python_exe, "-m", "pytest", "tests/test_topology_generation_fixes.py", "-q"],
        cwd=root,
    )
    cmd_vglc = _run_command(
        [args.python_exe, "-m", "pytest", "tests/test_vglc_compliance.py", "-q"],
        cwd=root,
    )

    topology_passes = _extract_pytest_pass_count(cmd_topology.stdout + "\n" + cmd_topology.stderr)
    vglc_passes = _extract_pytest_pass_count(cmd_vglc.stdout + "\n" + cmd_vglc.stderr)

    criteria: List[CriterionScore] = []

    # C1
    c1_needles = [
        "topology_generator = EvolutionaryTopologyGenerator(",
        "mission_graph = topology_generator.evolve()",
        "is_valid, errors = validate_graph_topology(mission_graph)",
        "mission_graph_physical = filter_virtual_nodes(mission_graph)",
    ]
    c1_pass = _has_all(pipeline_py, c1_needles)
    criteria.append(
        _score_binary(
            "C1",
            "Topology Pipeline Integration",
            c1_pass,
            "Pipeline includes generation, topology validation, and virtual-node filtering path.",
            "Missing one or more required topology integration anchors in pipeline.",
            [
                "src/pipeline/dungeon_pipeline.py",
            ],
        )
    )

    # C2
    c2_needles = [
        "def validate_all_constraints(self, graph: MissionGraph) -> bool:",
        "def validate_lock_key_ordering(self, graph: MissionGraph) -> bool:",
        "def validate_progression_constraints(self, graph: MissionGraph) -> bool:",
        "graph = self._repair_progression_constraints(graph)",
    ]
    c2_pass = _has_all(grammar_py, c2_needles)
    criteria.append(
        _score_binary(
            "C2",
            "Constraint Completeness in Graph Grammar",
            c2_pass,
            "Grammar contains comprehensive constraints and repair hooks.",
            "Constraint validators and repair hooks are incomplete.",
            [
                "src/generation/grammar.py",
            ],
        )
    )

    # C3
    c3_needles = [
        "offspring = self._evaluate_population(offspring, gen)",
        "def _individual_sort_key(ind: Individual)",
        "self.feasible_ratio_history",
        "self.avg_violation_history",
    ]
    c3_pass = _has_all(evolution_py, c3_needles)
    criteria.append(
        _score_binary(
            "C3",
            "Evolutionary Search Correctness",
            c3_pass,
            "Evolution loop and survivor pressure anchors are present.",
            "Evolution loop is missing one or more correctness anchors.",
            [
                "src/generation/evolutionary_director.py",
            ],
        )
    )

    # C4 (partial scoring by realism-pressure coverage)
    c4_needles = [
        "def _apply_target_aware_rule_prior(self) -> None:",
        "def _adapt_global_rule_prior_from_population(self, population: Sequence[Individual]) -> None:",
        "def _relax_rule_weights_to_target_prior(self, decay: float = 0.08) -> None:",
    ]
    c4_count = sum(1 for needle in c4_needles if needle in evolution_py)
    if c4_count == len(c4_needles):
        c4_score = 4
        c4_pass = True
        c4_rationale = "Adaptive and target-aware realism pressure mechanisms are implemented."
    elif c4_count >= 2:
        c4_score = 3
        c4_pass = False
        c4_rationale = "Realism pressure mechanisms are partially implemented."
    else:
        c4_score = 0
        c4_pass = False
        c4_rationale = "Realism pressure mechanisms are missing or minimal."
    criteria.append(
        CriterionScore(
            id="C4",
            name="Descriptor/Realism Pressure Mechanisms",
            score=c4_score,
            max_score=5,
            passed=c4_pass,
            rationale=c4_rationale,
            evidence=["src/generation/evolutionary_director.py"],
        )
    )

    # C5
    c5_pass = _has_all(graph_utils_py, [
        "def validate_goal_subgraph(G: nx.Graph) -> Tuple[bool, List[str]]:",
        "def validate_graph_topology(G: nx.Graph) -> Tuple[bool, List[str]]:",
    ]) and _has_all(vglc_py, [
        "def validate_topology(graph: nx.Graph) -> TopologyReport:",
    ])
    criteria.append(
        _score_binary(
            "C5",
            "Topology Validation Redundancy",
            c5_pass,
            "Two complementary validator layers are present.",
            "Validator layering is incomplete.",
            [
                "src/utils/graph_utils.py",
                "src/data/vglc_utils.py",
            ],
        )
    )

    # C6
    c6_pass = (cmd_topology.exit_code == 0 and topology_passes >= 1)
    criteria.append(
        _score_binary(
            "C6",
            "Regression Test Coverage for Topology Fixes",
            c6_pass,
            f"Topology-fix regression suite passed ({topology_passes} passed).",
            "Topology-fix regression suite failed.",
            ["tests/test_topology_generation_fixes.py"],
        )
    )

    # C7
    c7_pass = (cmd_vglc.exit_code == 0 and vglc_passes >= 1)
    criteria.append(
        _score_binary(
            "C7",
            "VGLC Compliance Test Coverage",
            c7_pass,
            f"VGLC compliance suite passed ({vglc_passes} passed).",
            "VGLC compliance suite failed.",
            ["tests/test_vglc_compliance.py"],
        )
    )

    # C8 (partial by instrumentation anchors)
    c8_needles = [
        "def run_wfc_robustness_probe(",
        "mask_ratio_target",
        "mask_ratio_applied",
        "wfc_probe_results = run_wfc_robustness_probe(",
    ]
    c8_count = sum(1 for needle in c8_needles if needle in bench_py)
    if c8_count == len(c8_needles):
        c8_score = 4
        c8_pass = True
        c8_rationale = "Robustness probe + mask instrumentation are implemented in benchmark suite."
    elif c8_count >= 2:
        c8_score = 2
        c8_pass = False
        c8_rationale = "Reproducibility instrumentation is partial."
    else:
        c8_score = 0
        c8_pass = False
        c8_rationale = "Reproducibility instrumentation is largely missing."
    criteria.append(
        CriterionScore(
            id="C8",
            name="Reproducibility Instrumentation",
            score=c8_score,
            max_score=5,
            passed=c8_pass,
            rationale=c8_rationale,
            evidence=["src/evaluation/benchmark_suite.py"],
        )
    )

    total_score = int(sum(c.score for c in criteria))
    total_max = int(sum(c.max_score for c in criteria))
    percent = float((100.0 * total_score / total_max) if total_max > 0 else 0.0)

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "workspace_root": str(root),
        "total_score": total_score,
        "total_max": total_max,
        "percent": percent,
        "criteria": [asdict(c) for c in criteria],
        "commands": {
            "topology_fixes_pytest": asdict(cmd_topology),
            "vglc_compliance_pytest": asdict(cmd_vglc),
        },
    }

    json_path = out_dir / "topology_stack_rubric_report.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    grade = "Strong" if percent >= 90.0 else ("Moderate" if percent >= 70.0 else "Weak")
    lines: List[str] = []
    lines.append("# Topology Stack Rubric Report")
    lines.append("")
    lines.append(f"- Total: **{total_score}/{total_max}** ({percent:.1f}%)")
    lines.append(f"- Grade: **{grade}**")
    lines.append("")
    lines.append("## Criteria")
    lines.append("")
    for c in criteria:
        check = "x" if c.passed else " "
        lines.append(f"- [{check}] **{c.id} {c.name}** — **{c.score}/{c.max_score}**")
        lines.append(f"  - {c.rationale}")
        if c.evidence:
            lines.append(f"  - Evidence: {', '.join(c.evidence)}")
    lines.append("")
    lines.append("## Reproducible test outputs")
    lines.append("")
    lines.append(f"- `tests/test_topology_generation_fixes.py`: exit={cmd_topology.exit_code}, passed={topology_passes}")
    lines.append(f"- `tests/test_vglc_compliance.py`: exit={cmd_vglc.exit_code}, passed={vglc_passes}")
    lines.append("")
    lines.append("## Commands")
    lines.append("")
    lines.append(f"- `{cmd_topology.command}`")
    lines.append(f"- `{cmd_vglc.command}`")

    md_path = out_dir / "topology_stack_rubric_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    print(f"Score: {total_score}/{total_max} ({percent:.1f}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
