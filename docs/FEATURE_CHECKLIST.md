# Feature roadmap & PR checklist — Topology Solvers

This document lists the high-level features, the split into tracked PRs, acceptance criteria, and tests to add. Use this to track progress and review PRs.

## Overall Goals
- Deterministic room↔node mapping and canonical graph normalization
- Unified traversal semantics (edge_type canonical)
- Production-grade JPS: correctness (forced neighbors), diagonal + corner policy, tracing, GUI overlay
- Bench harness + Numba hot-paths and microbenching
- Tests, docs, and CI smoke jobs

## Tracked PRs (priority order)
1. PR: `pr/deterministic-mapping-normalizer` (current priority)
   - Changes:
     - Add/extend `Data/zelda_core._normalize_graph()` to canonicalize node labels and edge_types (ensure backwards compatibility)
     - Harden `RoomGraphMatcher` mapping for determinism and add validation helpers
     - Tests: deterministic mapping, mapping consistency, infer_missing_mappings coverage
   - Acceptance: mapping deterministic across 50 shuffles; mapping consistency >= current thresholds
   - Estimated effort: 4–8 hours

2. PR: `feat/jps-productionization`
   - Changes:
     - Reimplement `bench/grid_solvers.jps` forced-neighbor logic to canonical spec
     - Add `allow_corner_cutting` policy, `trace` container (expanded segments + jump points)
     - GUI: overlay rendering of JPS trace and toggles (`use_jps`, `show_jps_overlay`) with safe defaults
     - Tests: JPS vs A* randomized parity, corner cases, trace format tests
   - Acceptance: JPS path costs equal A* (or fallback) across test suite; trace content stable and rendered by GUI
   - Estimated effort: 1–2 days

3. PR: `feat/traversal-unification`
   - Changes:
     - Centralize can_traverse_edge (Data + Pathfinder), unify semantics and add label->edge_type fallback
     - Add tests comparing both modules
   - Estimated effort: 3–6 hours

4. PR: `feat/bench-ci-and-numba`
   - Changes:
     - Add CI bench smoke job, integrate `bench/bench_harness.py`
     - Extend `bench/numba_solvers` -> numba A* microkernel prototype
     - Tests: microbench correctness + smoke runs in CI
   - Estimated effort: 1–2 days

5. PR: `docs/usage-and-flags`
   - Changes: update README and pathfinding docs to describe flags and GUI toggles, add examples, add changelog and PR entry
   - Estimated effort: 2–4 hours

## Cross-PR tests to add (regression protection)
- deterministic_mapping_test (50 iter), test_jps_vs_astar_randomized_consistency, test_traversal_consistency_core_vs_pathfinder, bench smoke workflow in CI, test_gui_overlay_smoke

---

Use branch naming from above and incrementally add small, reviewable commits and tests.
