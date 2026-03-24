# Topology Stack Rubric Report

- Total: **38/40** (95.0%)
- Grade: **Strong**

## Criteria

- [x] **C1 Topology Pipeline Integration** — **5/5**
  - Pipeline includes generation, topology validation, and virtual-node filtering path.
  - Evidence: src/pipeline/dungeon_pipeline.py
- [x] **C2 Constraint Completeness in Graph Grammar** — **5/5**
  - Grammar contains comprehensive constraints and repair hooks.
  - Evidence: src/generation/grammar.py
- [x] **C3 Evolutionary Search Correctness** — **5/5**
  - Evolution loop and survivor pressure anchors are present.
  - Evidence: src/generation/evolutionary_director.py
- [x] **C4 Descriptor/Realism Pressure Mechanisms** — **4/5**
  - Adaptive and target-aware realism pressure mechanisms are implemented.
  - Evidence: src/generation/evolutionary_director.py
- [x] **C5 Topology Validation Redundancy** — **5/5**
  - Two complementary validator layers are present.
  - Evidence: src/utils/graph_utils.py, src/data/vglc_utils.py
- [x] **C6 Regression Test Coverage for Topology Fixes** — **5/5**
  - Topology-fix regression suite passed (12 passed).
  - Evidence: tests/test_topology_generation_fixes.py
- [x] **C7 VGLC Compliance Test Coverage** — **5/5**
  - VGLC compliance suite passed (36 passed).
  - Evidence: tests/test_vglc_compliance.py
- [x] **C8 Reproducibility Instrumentation** — **4/5**
  - Robustness probe + mask instrumentation are implemented in benchmark suite.
  - Evidence: src/evaluation/benchmark_suite.py

## Reproducible test outputs

- `tests/test_topology_generation_fixes.py`: exit=0, passed=12
- `tests/test_vglc_compliance.py`: exit=0, passed=36

## Commands

- `F:\KLTN\.venv\Scripts\python.exe -m pytest tests/test_topology_generation_fixes.py -q`
- `F:\KLTN\.venv\Scripts\python.exe -m pytest tests/test_vglc_compliance.py -q`
