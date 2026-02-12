# KLTN — Current Problems Snapshot

This document summarizes the key problems discovered during the repository audit and points to likely root causes to fix (prefer fixing underlying code rather than only adjusting tests).

## High-level issues

- Failing unit tests across multiple modules that indicate API drift or logic bugs rather than test errors.
  - `tests/test_block_integration.py::test_block_iii_condition_encoder` — Shape mismatch between data and Linear layers in `src/core/condition_encoder.py`.
  - `tests/test_cognitive_bounded_search.py::TestBeliefMap::test_observe_tile` — `BeliefMap.get_tile()` returns a `TileID` object or wrong type; tests expect a `(tile_type, confidence)` tuple.
  - `tests/test_data_integrity.py::test_graph_room_consistency[3-2]` — Rooms without graph nodes: mapping between parsed rooms and topology graph nodes is inconsistent.
  - `tests/test_hmolqd/*` — Multiple API mismatches (constructor signatures/return types) in `src/core/*` (VQ-VAE, diffusion, etc.).

- Large, monolithic modules with high complexity and limited unit tests:
  - `src/data/zelda_core.py` (≈4.4k LOC)
  - `src/simulation/validator.py` (≈4.9k LOC)

- Missing repository hygiene items:
  - No top-level `LICENSE` file (should be added).
  - No CI workflow (`.github/workflows/ci.yml`) currently present.

## Likely root causes (summary)

- API refactors were performed without updating tests or dependent callers — leading to signature/return-type mismatches.
- Parsing & mapping logic in `zelda_core` may not guarantee a 1:1 mapping between rooms and topology graph nodes; missing guard or fallback when rooms are isolated or filtered.
- Optional dependencies (e.g., `pygame`) cause GUI tests to fail if not present; tests should guard or skip.

## Immediate recommended fixes (prioritized)

1. Fix `src/core/condition_encoder.py` to ensure consistent input dimensions and add explicit assertions with informative errors. Add tests that exercise both input shapes.
2. Fix `BeliefMap.get_tile()` return type or adapt callers/tests to the new return format; ensure tuple `(tile_type, confidence)` is returned where expected.
3. Add a regression test reproducing the graph-room mapping issue in `tests/test_data_integrity.py` and patch parsing/mapping logic in `src/data/zelda_core.py` accordingly.
4. Add `LICENSE` file and a minimal CI workflow (pytest + ruff + optional safety) to catch regressions early.

## Mapping bug: unmapped graph nodes (detailed)

- **Symptom:** After matching, several graph nodes have no corresponding room (i.e., `node_to_room` missing entries). This causes items and door types associated with those nodes to be missing from the stitched dungeon and leads to validation/test failures (see `tests/test_data_integrity.py::test_graph_room_consistency[3-2]`).

- **Where observed:** `src/data/zelda_core.py` — primarily in `RoomGraphMatcher.match`, `_match_rooms_to_nodes_bfs`, and the later `DungeonStitcher.stitch` logic. Symptoms are surfaced during stitching and in unit tests.

- **Probable root causes:**
  - DOT/topology contains more nodes than parsed rooms (virtual/pointer nodes or extra labels) and the greedy BFS matching leaves nodes unmapped.
  - `start` pointer nodes (`is_start_pointer`) are sometimes treated as regular nodes and consume mapping slots.
  - Room adjacency detection (door-center or semantic mismatch) produces incomplete spatial adjacency making matching fail.
  - Fallback assignment can fail silently or not persist the assignment.
  - Malformed or uncommon DOT labels produce nodes with missing flags, confusing heuristics.

- **Quick repro snippet:**

```py
from src.data.zelda_core import VGLCParser, DOTParser, RoomGraphMatcher
rooms = VGLCParser().parse('Data/The Legend of Zelda/example_map.txt')
graph = DOTParser().parse('dungeons/3.dot')
dungeon = RoomGraphMatcher().match(rooms, graph)
unmapped = [n for n in graph.nodes if n not in dungeon.node_to_room]
print('unmapped nodes:', unmapped)
```

If `unmapped` is non-empty for the failing case, the bug is reproducible.

- **Immediate fixes:**
  1. Add a post-match invariant check to surface `unmapped_nodes` (populate `dungeon.mapping_issues` or raise `MappingError`).
  2. Exclude `is_start_pointer` nodes from normal matching.
  3. Ensure `_find_fallback_room` is invoked for every unmapped node and its assignment is persisted and logged.
  4. Harden door-center detection and adjacency building.
  5. Implement a `strict` mode in `stitch` that fails on `missing_items` or `unmapped_nodes` > 0.

- **Tests to add:**
  - Unit test for `DOTParser` labeling and `RoomGraphMatcher._match_rooms_to_nodes_bfs` deterministic behavior with extra virtual nodes.
  - Integration test for the real failing case (dungeon 3 v2): load VGLC + DOT, run `match` + `stitch`, and assert `len(unmapped_nodes) == 0` and `missing_items == 0`.

---

## Additional repo-wide problems discovered (summary of scan)

Below are further problems discovered in a read-only repo scan. Each entry includes severity, evidence (files/commands), and concise next steps to validate or remedy.

1) `gui_runner.py` is extremely large (High)
- Evidence & command: `Get-Content gui_runner.py | Measure-Object -Line` → **9431 lines**.
- Impact: Hard to maintain, test, and review. Refactor recommended.
- Next steps: add an issue to split into modules and extract one small component with unit tests.

2) Large `ruff` backlog (Medium→High)
- Evidence & command: `ruff check --statistics .` → ~**550** errors (top issues: F401 unused-import, E402 import-not-at-top, F821 undefined-name).
- Impact: Hidden bugs and maintenance friction.
- Next steps: run `ruff --fix` on safe targets (tests and small modules) and add `ruff` to CI.

3) Unsafe `pickle.load` usage (Security: High)
- Evidence & command: `grep -R "pickle.load" -n` → multiple hits (e.g., `gui_runner.py` lines with `pickle.load(f)`).
- Impact: Arbitrary code execution if loading untrusted files.
- Next steps: open security issue to migrate to `json`/`npz` or add validation / signed files; add tests for safe loading.

4) Bare `except:` usage (Medium)
- Evidence: `grep -R "except:\" -n` and `ruff` E722 flagged occurrences in `src/simulation/*`.
- Impact: Swallowed exceptions hide real errors.
- Next steps: replace bare `except:` with `except Exception as e:` and add tests for error logging.

5) Hard-coded user data path (Medium)
- Evidence: `src/data/zelda_core.py` contains `default="C:\\Users\\MPhuc\\Desktop\\KLTN\\Data\\The Legend of Zelda"`.
- Impact: Non-portable, leaks local paths.
- Next steps: remove hard-coded path, use relative default or require `--data-root`.

6) Legacy packaging / deprecated `Data/` imports (Medium)
- Evidence: Tests emit `DeprecationWarning` for `Data` import; both `Data/` and `src/data` exist.
- Impact: Confusing imports and maintenance debt.
- Next steps: migrate imports in tests and provide a deprecation removal plan.

7) Missing repository metadata & CI (High)
- Evidence: No `LICENSE` in repo root; no GitHub Actions workflows found for CI.
- Impact: Legal uncertainty and no automated checks on PRs.
- Next steps: add `LICENSE`, `CONTRIBUTING.md`, and a basic GitHub Actions workflow to run linters and fast tests.

8) Heavy test dependencies & flakiness (Medium)
- Evidence: Some tests import/require `torch`, `pygame`; some tests hang or are slow when run together.
- Impact: Slow or flaky CI runs.
- Next steps: add pytest markers (`slow`, `gpu`) and run fast tests in PRs, full heavy tests in nightly runs.

9) Known failing test: mapping bug already documented (Low/Medium)
- Evidence: `pytest tests/test_data_integrity.py` → failing case for Dungeon 3.
- Next steps: add regression test and fix (`RoomGraphMatcher` / `stitch`) per mapping bug notes above.

10) `.venv/` present in workspace (Low)
- Evidence: virtualenv files in workspace; ensure `.gitignore` excludes them.
- Next steps: confirm `.venv` is in `.gitignore` and add workspace exclusion to `.vscode/settings.json`.

---

If you want, I can convert these findings into GitHub issues (one per item) with acceptance criteria and suggested PR changes; say "create issues" and I will draft them (no code changes will be made).