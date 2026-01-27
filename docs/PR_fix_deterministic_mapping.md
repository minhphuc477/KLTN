# PR Draft: fix/deterministic-mapping-heuristic

## Title
Fix deterministic room→node mapping & make heuristic admissible by default

## Summary
- Ensure deterministic local room→node assignment in `ZeldaPathfinder` by sorting candidate graph neighbors deterministically and preferring degree as tie-breaker.
- Add `admissible_heuristic` flag to `ZeldaPathfinder` (default True) which disables the heuristic key-deficit penalty when strict admissibility is required.
- Add unit test `tests/test_zelda_heuristic_admissibility.py` to validate admissible behavior.

## Files changed
- `zelda_pathfinder.py` — add `admissible_heuristic` flag and deterministic ordering.
- `tests/test_zelda_heuristic_admissibility.py` — new test.

## Rationale
- Deterministic mapping makes tests reproducible across runs and CI.
- Using an admissible heuristic by default prevents A* from being misled by heuristic penalties and preserves optimality.

## How to review
- Verify `zelda_pathfinder.py` changes are minimal and well-commented.
- Run tests to confirm behaviour: `pytest tests/test_zelda_heuristic_admissibility.py`.

## Follow-ups
- Normalize assignment logic between `RoomGraphMatcher` and `ZeldaPathfinder` to reuse canonical code.
- Document `admissible_heuristic` in README.
