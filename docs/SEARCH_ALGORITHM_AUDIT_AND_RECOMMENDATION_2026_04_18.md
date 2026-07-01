# Search Algorithm Audit And Recommendation

Last updated: 2026-07-02.

This note answers which pathfinding algorithms should be used in H-MOLQD and
which algorithms should remain diagnostics or ablations. It is intentionally
conservative: a solver is only a hard oracle when its assumptions match the
Zelda state model implemented in `src/simulation/validator.py`.

## Bottom Line

- Use full-state `A*` as the report-facing mechanical oracle.
- Use `Dijkstra` as an exact cost fallback/baseline when heuristic behavior is
  under scrutiny.
- Use `Bidirectional A*` only for reversible, stateless grid comparisons.
- Use `D* Lite` only for incremental replanning experiments.
- Use `P-CBS` as a bounded-agent behavioral probe, not as a hard solvability
  oracle.
- Use `JPS`, `HPA*`, or `Theta*` only in separate ablations whose assumptions
  are explicitly restricted to static grids or any-angle movement.

## Code Truth

The current selector lives in
[`src/simulation/search_factory.py`](../src/simulation/search_factory.py):

- `A*`: `validation_role="oracle"`, `canonical_use="hard_oracle"`.
- `Dijkstra`: exact fallback and baseline.
- `D* Lite`: `validation_role="replanning_diagnostic"`.
- `Bidirectional A*`: `validation_role="reversible_grid_diagnostic"`.
- `recommended_game_state_algorithm_specs(...)` selects algorithms by
  environment class.
- `environment_requires_full_state_oracle(...)` returns true when the map
  includes inventory, pickups, doors, water/item traversal, push blocks,
  enemies, puzzles, graph transitions, or staged puzzle lookup.

This means the code no longer exposes all solvers as equivalent validators.

## Literature Basis

Primary references checked:

- Hart, Nilsson, and Raphael, "A Formal Basis for the Heuristic Determination
  of Minimum Cost Paths" (1968): basis for admissible A* path search.
- Koenig and Likhachev, "D* Lite" (AAAI 2002): incremental heuristic search
  for repeated similar replanning tasks.
- Harabor and Grastien, "Online Graph Pruning for Pathfinding on Grid Maps"
  (AAAI 2011): Jump Point Search for uniform-cost grid maps.
- Botea, Mueller, and Schaeffer, "Near Optimal Hierarchical Path-Finding"
  (Journal of Game Development, 2004): HPA* for large static game maps.

The important mismatch is state representation. H-MOLQD validation states are
not only `(row, col)`: they include keys, bombs, opened doors, collected items,
pushed blocks, defeated enemies, puzzle stages, floors, and opened graph edges.
Two visits to the same tile can have different legal futures. Plain-grid
symmetry pruning and one-shot backward search are therefore diagnostics unless
the environment is proven stateless and reversible.

## Algorithm Roles

| Algorithm | Use in this repo | Do not claim |
|---|---|---|
| Full-state A* | Hard mechanical oracle | Human playability |
| Dijkstra | Exact no-heuristic baseline/fallback | Fast default |
| BFS | Small uniform-cost sanity baseline | Scalable validator |
| Greedy | Inadmissible heuristic baseline | Correctness oracle |
| D* Lite | Dynamic replanning diagnostic | Primary static oracle |
| DFS/IDDFS | Bounded exhaustive probe | Optimal default |
| Bidirectional A* | Reversible stateless-grid comparison | Valid on inventory maps |
| P-CBS | Bounded-agent behavioral/readability probe | Mechanical oracle |
| JPS | Future static-grid speed ablation | Valid on stateful Zelda mechanics |
| HPA* | Future large static-map speed ablation | Exact optimal oracle |
| Theta* | Future any-angle geometry ablation | Zelda tile-interaction oracle |

## Implementation Notes

Current fixes applied:

- Runtime algorithm recommendation is centralized in `search_factory.py`.
- The role-separated benchmark entry point is
  [`scripts/run_search_role_benchmark.py`](../scripts/run_search_role_benchmark.py).
- Bidirectional A* now continues after the first frontier meeting until the
  incumbent path is certified against the remaining frontier lower bound.
- Reversible-grid bidirectional search initializes backward inventory from the
  start state, avoiding false collision rejection on stateless maps.
- D* Lite documentation now explicitly states this implementation is a forward
  LPA*/D* Lite-style variant.
- Diagonal movement uses the canonical `sqrt(2)` cost consistently across
  validator, D* Lite, and Bidirectional A*.
- Hazard graph edges parse and traverse as risk-bearing open edges rather than
  silently becoming non-traversable.
- Graph-level QD fitness now checks progression solvability through
  `ExternalValidator` before scoring graph cognitive proxies. Undirected
  connectivity is not treated as enough for lock/key mission graphs.

### Benchmark Commands

Use a smoke run only to verify wiring:

```bash
python scripts/run_search_role_benchmark.py --synthetic-smoke --include-diagnostics --include-static-grid-ablation --output-dir results/search_role_benchmark_smoke --timeout 5000
```

Use final generated artifacts for report tables:

```bash
python scripts/run_search_role_benchmark.py --input results/final_generated_maps --include-diagnostics --include-static-grid-ablation --pcbs-personas novice,balanced,expert --output-dir results/search_role_benchmark_final
```

The CSV/JSON outputs include `validation_role` and `canonical_use` columns.
Keep those columns in downstream tables.

## Publication Contract

For final tables:

1. Report hard solvability with full-state A* plus graph progression and
   softlock checks.
2. Report Dijkstra only as an exact comparator or fallback.
3. Report D* Lite under a replanning section with dynamic-change scenarios.
4. Report Bidirectional A* only on reversible stateless grids, or explicitly
   mark `fallback_used=True` when it delegates to A*.
5. Report P-CBS separately as bounded-agent readability and difficulty.
6. Do not merge these into one "solver success" metric without role labels.
7. For Quality-Diversity evolution, archive quality must be based on
   progression-solvable mission graphs or validated semantic grids. Behavioral
   descriptors can be topological, but the quality score must not reward a
   graph that is only connected after ignoring gates, keys, switches, or
   consumable locks.

## Remaining Search Work

- Run the search-only benchmark over final generated maps with `A*`,
  `Dijkstra`, `P-CBS` personas, and diagnostic solvers separated by role.
- If large stitched maps become a runtime bottleneck, extend the static-grid
  ablation with HPA*. Keep it separate from Zelda-state validation just like
  the current JPS ablation.
- Keep human/player claims separate from oracle claims until calibrated
  playtest data exists.
