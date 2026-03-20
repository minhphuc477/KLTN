# Topology Stack Evaluation Rubric (2026-03-08)

## Purpose

This document defines a strict, reproducible rubric for evaluating the current topology generation stack (Block I + topology validation integration) and scoring it with objective evidence.

Scope includes:
- mission graph generation correctness and constraints,
- pipeline integration and validation gates,
- reproducible test and benchmark hooks.

---

## Reproducibility Protocol

### Environment

- Workspace root: `F:\KLTN`
- Python interpreter: `f:/KLTN/.venv/Scripts/python.exe`

### Required commands

Run the following commands from repository root:

```powershell
f:/KLTN/.venv/Scripts/python.exe -m pytest tests/test_topology_generation_fixes.py -q
f:/KLTN/.venv/Scripts/python.exe -m pytest tests/test_vglc_compliance.py -q
```

Expected for current snapshot:
- `tests/test_topology_generation_fixes.py`: `12 passed`
- `tests/test_vglc_compliance.py`: `36 passed`

### Code evidence checks (static)

Verify the following implementation anchors exist:
- Topology generation in pipeline:
  - `src/pipeline/dungeon_pipeline.py`: `EvolutionaryTopologyGenerator(...)`, `.evolve()`, `validate_graph_topology(...)`, `filter_virtual_nodes(...)`
- Grammar constraint validators:
  - `src/generation/grammar.py`: `validate_all_constraints`, `validate_lock_key_ordering`, `validate_progression_constraints`
- Topology validators:
  - `src/utils/graph_utils.py`: `validate_goal_subgraph`, `validate_graph_topology`
  - `src/data/vglc_utils.py`: `validate_topology`
- Evolutionary robustness hooks:
  - `src/generation/evolutionary_director.py`: `evolve`, offspring evaluation before selection, `_individual_sort_key`, `_adapt_global_rule_prior_from_population`
- WFC robustness probe with masking:
  - `src/evaluation/benchmark_suite.py`: `run_wfc_robustness_probe`, `mask_ratio_target`, `mask_ratio_applied`

---

## Strict Rubric (8 Criteria, 0-5 each)

Scoring scale:
- **0** = absent / broken
- **1** = minimal partial implementation
- **2** = weak and unreliable
- **3** = functional baseline
- **4** = strong with minor gaps
- **5** = production-grade with strong evidence

Maximum score: **40**

### C1. Topology Pipeline Integration (0-5)

**Measure**
- Pipeline can generate topology automatically when no mission graph is provided.
- Generated graph is validated before downstream room synthesis.
- Virtual nodes are filtered before room generation.

**Pass threshold for 5**
- All three conditions implemented in runtime path.

### C2. Constraint Completeness in Graph Grammar (0-5)

**Measure**
- Presence of comprehensive constraint validation methods for anchors, lock/key ordering, and progression edge constraints.
- Repair loop exists for post-rule correction.

**Pass threshold for 5**
- All required validators plus repair loop are present and reachable in generation flow.

### C3. Evolutionary Search Correctness (0-5)

**Measure**
- Offspring are evaluated before survivor selection.
- Feasibility-first survivor ordering exists.
- Search loop records convergence diagnostics.

**Pass threshold for 5**
- All three features present in `evolve` and selection internals.

### C4. Descriptor/Realism Pressure Mechanisms (0-5)

**Measure**
- Rule priors adapt from descriptor errors.
- Target-aware prior shaping exists.
- Mechanisms cover topology and gating realism dimensions.

**Pass threshold for 5**
- Adaptive and target-aware pressure both implemented and integrated in loop.

### C5. Topology Validation Redundancy (Defense-in-Depth) (0-5)

**Measure**
- At least two distinct validator layers are used (lightweight and rich report style).
- Validation checks include connectivity/start-goal/goal-subgraph semantics.

**Pass threshold for 5**
- Two layers present and actively called in generation/pipeline flow.

### C6. Regression Test Coverage for Topology Fixes (0-5)

**Measure**
- `tests/test_topology_generation_fixes.py` passes fully.
- Tests cover max-node constraints and algorithmic bugfix regressions.

**Pass threshold for 5**
- Full pass with no skipped/failing tests.

### C7. VGLC Compliance Test Coverage (0-5)

**Measure**
- `tests/test_vglc_compliance.py` passes fully.
- Includes virtual-node handling, shape constraints, and topology checks.

**Pass threshold for 5**
- Full pass with no skipped/failing tests.

### C8. Reproducibility Instrumentation (0-5)

**Measure**
- Benchmark suite contains explicit robustness probes with controlled masking metadata.
- Inputs/outputs are scriptable from CLI.

**Pass threshold for 5**
- WFC probe and mask-ratio diagnostics are implemented and callable.

---

## Current Scored Checklist (2026-03-08)

### Evidence snapshot

- Test run:
  - `tests/test_topology_generation_fixes.py`: `12 passed in 12.55s`
  - `tests/test_vglc_compliance.py`: `36 passed in 2.86s`
- Code anchors confirmed in current workspace (see criteria notes below).

### Checklist

- [x] **C1 Topology Pipeline Integration** — **5/5**
  - Evidence: `src/pipeline/dungeon_pipeline.py` lines around topology creation, evolution call, graph validation, virtual-node filtering.

- [x] **C2 Constraint Completeness in Graph Grammar** — **5/5**
  - Evidence: `src/generation/grammar.py` includes `validate_all_constraints`, `validate_lock_key_ordering`, `validate_progression_constraints`, and repair/validation loop in `generate`.

- [x] **C3 Evolutionary Search Correctness** — **5/5**
  - Evidence: `src/generation/evolutionary_director.py` evaluates offspring before survivor selection and uses feasibility-first sort key.

- [x] **C4 Descriptor/Realism Pressure Mechanisms** — **4/5**
  - Evidence: target-aware priors + generation-level adaptive rule prior updates are present.
  - Remaining gap: realism still configuration-sensitive per topology tuning logs.

- [x] **C5 Topology Validation Redundancy** — **5/5**
  - Evidence: `src/utils/graph_utils.py` + `src/data/vglc_utils.py` both provide topology checks, and pipeline/evolution paths call validation.

- [x] **C6 Regression Test Coverage for Topology Fixes** — **5/5**
  - Evidence: `12/12` tests passed in `tests/test_topology_generation_fixes.py`.

- [x] **C7 VGLC Compliance Test Coverage** — **5/5**
  - Evidence: `36/36` tests passed in `tests/test_vglc_compliance.py`.

- [x] **C8 Reproducibility Instrumentation** — **4/5**
  - Evidence: `run_wfc_robustness_probe` with `mask_ratio_target` and `mask_ratio_applied` in `src/evaluation/benchmark_suite.py`.
  - Remaining gap: full matched-budget/seed-stable benchmark execution is still expensive and should be routinely automated.

### Total

- **Score: 38/40 (95.0%)**
- **Grade: Strong research-engineering maturity**

Interpretation:
- The stack is robust, integrated, and strongly validated at implementation level.
- The primary remaining risk is not correctness; it is consistent realism alignment under broader benchmark budgets/configurations.

---

## Optional Automation Extension

Automation is now implemented:

```powershell
f:/KLTN/.venv/Scripts/python.exe scripts/score_topology_stack_rubric.py --output-dir results/topology_rubric
```

Outputs:
- `results/topology_rubric/topology_stack_rubric_report.json`
- `results/topology_rubric/topology_stack_rubric_report.md`

For CI reproducibility, run this command and publish both artifacts.
