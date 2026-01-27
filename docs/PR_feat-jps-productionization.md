# PR Draft: feat/jps-productionization

## Title
Productionize Jump Point Search (JPS): correctness, tracing, GUI overlay

## Summary
- Implement canonical forced-neighbor checks and ensure diagonal/corner-cut policy is enforced correctly.
- Add `trace` output (segments + jumps + expanded nodes) and GUI overlay (`show_jps_overlay`) for visualization.
- Add robust tests comparing JPS to A* and targeted corner-cut tests.

## Files (proposed)
- `bench/grid_solvers.py` — rework JPS implementation, add `trace` container and `allow_corner_cutting` toggle
- `gui_runner.py` — add overlay rendering and toggles, ensure headless-friendly
- `tests/test_jps_cornercut_and_consistency.py` — randomized and corner tests

## Acceptance
- JPS matches A* path cost across randomized test families (open, maze, corridor) for many seeds
- Trace format stable and GUI overlay renders segments without errors

## Notes
Aim for correctness first; micro-optimizations (Numba) to follow in separate PRs.
