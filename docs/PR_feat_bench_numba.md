# PR Draft: feat/bench-numba

## Title
Add benchmark harness and Numba helper sketch for solver optimizations

## Summary
- Add `bench/bench_harness.py` — a lightweight deterministic harness for running solver experiments and exporting CSV results.
- Add `bench/numba_solvers.py` — a small Numba helper (neighbors on flattened grids) demonstrating the pattern for accelerating inner loops.
- Add `tests/test_bench_numba_smoke.py` to ensure the sketch imports and runs (skips gracefully when Numba not available).

## Files changed
- `bench/bench_harness.py` — new CLI harness (seeded, repeatable).
- `bench/numba_solvers.py` — Numba helper and Python fallback.
- `tests/test_bench_numba_smoke.py` — smoke test.

## How to review
- Run a small harness example: `python -m bench.bench_harness --solver astar --map open --size 16 --seed 0 --repeats 3` and inspect `bench_results.csv`.
- Confirm `tests/test_bench_numba_smoke.py` passes locally (it will skip if Numba not installed).

## Follow-ups
- Implement full Numba-accelerated A*/JPS kernels and add microbench harness.
- Add CI job to run bench harness in a short smoke configuration and collect artifacts.
