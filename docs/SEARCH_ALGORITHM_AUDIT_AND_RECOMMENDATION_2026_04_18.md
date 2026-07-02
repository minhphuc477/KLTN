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
- Alvarez et al., "Empowering Quality Diversity in Dungeon Design with
  Interactive Constrained MAP-Elites" (CoG 2019): dungeon QD separates
  feasibility constraints from behavior-space illumination.
- Khalifa et al., "Talakat: Bullet Hell Generation through Constrained
  MAP-Elites" (GECCO 2018): feasible-infeasible constraint handling and
  simulation-based content evaluation.
- van der Linden, Lopes, and Bidarra, "Designing Procedurally Generated
  Levels" (AIIDE 2013): action/mission graphs must be compiled into layouts
  while preserving authored gameplay constraints.

Primary-source links:

- [A* formal basis](https://ai.stanford.edu/~nilsson/OnlinePubs-Nils/PublishedPapers/astar.pdf)
- [D* Lite](https://aaai.org/Papers/AAAI/2002/AAAI02-072.pdf)
- [Interactive Constrained MAP-Elites](https://arxiv.org/abs/2003.03377)
- [Talakat constrained MAP-Elites](https://arxiv.org/abs/1806.04718)
- [Designing Procedurally Generated Levels](https://ojs.aaai.org/index.php/AIIDE/article/view/12592)

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
- Bidirectional fallback uses only the unspent expansion budget and reports
  the sum of bidirectional and fallback expansions. A single solver request
  can no longer silently spend the full budget twice.
- Reversible-grid bidirectional search initializes backward inventory from the
  start state, avoiding false collision rejection on stateless maps.
- D* Lite documentation now explicitly states this implementation is a forward
  LPA*/D* Lite-style variant.
- Diagonal movement uses the canonical `sqrt(2)` cost consistently across
  validator, D* Lite, and Bidirectional A*.
- Hazard graph edges parse and traverse as risk-bearing open edges rather than
  silently becoming non-traversable.
- A protected hazard compiles to the canonical `ELEMENT` tile and its
  `PROTECTION_ITEM` provider compiles to `KEY_ITEM`. Because the semantic tile
  vocabulary has only one generic protection-item identity, end-to-end
  generation rejects graphs containing multiple distinct named protection
  requirements instead of collapsing them.
- Graph-level QD fitness now checks progression solvability through
  `ExternalValidator` before scoring graph cognitive proxies. Undirected
  connectivity is not treated as enough for lock/key mission graphs.
- `ExternalValidator` now represents consumable small keys, specific key IDs,
  permanent boss keys, named items, switches, tokens, resource providers,
  hazards, and opened gates in the search state. One key can no longer satisfy
  multiple unopened locks.
- The graph-to-grid compiler preserves supported typed gates and assigns one
  consumable gate tile per physical room connection. The opposite boundary
  cell is an open door, avoiding double key consumption.
- START, GOAL, keys, boss keys, traversal items, enemies, bosses, and hazards
  are materialized on the semantic grid seen by the final oracle.
- The advanced pipeline runs a graph oracle before room generation and a
  tile-state oracle on the exact stitched semantic artifact before QD
  insertion.
- Grid and CVT MAP-Elites archives reject infeasible candidates instead of
  allowing zero-fitness artifacts to occupy empty cells.
- Mechanical feasibility is now separate from soft target/style mismatch.
  Missing a preferred linearity, curve, realism, or Pareto target reduces
  quality but does not relabel a playable graph as functionally infeasible.
- Runtime topology QD uses seeded CVT initialization and seeded elite
  selection. Paired runs with the same seed now reproduce both the selected
  graph and archive statistics.
- Advanced generation defaults to the `cvt_emitter` topology search rather
  than running a conventional GA and applying MAP-Elites only afterward.
- Search-budget exhaustion is reported as `budget_exhausted`, not as proven
  unsolvability, for both graph and final tile oracles.
- Search budgets in this document are state-expansion budgets. Wall-clock
  limits are separate orchestration timeouts and must not be compared as if
  they were expansion counts.
- Composite graph gates enforce every listed constraint, and typed token locks
  count only matching token identities.
- End-to-end generation defaults to the `core` mission rule space. The full
  grammar remains available for graph-only experiments, but a full-grammar
  graph is rejected by spatial compilation when its mechanic has no faithful
  tile/entity/oracle representation.

## End-To-End Solvability Contract

The authoritative order is:

1. Generate a mission graph without destructive open-edge connectivity repair.
2. Validate graph progression with full resource state, not weak connectivity.
3. Reject graph mechanics that the current tile vocabulary cannot represent.
4. Generate and stitch rooms while preserving supported edge types.
5. Materialize graph roles and progression entities on the semantic grid.
6. Validate the final semantic grid with full-state A*.
7. Admit only final-map-feasible artifacts to the publishable QD archive.
8. Run P-CBS afterward as a bounded-player diagnostic. The robust wrapper has
   an opt-in post-refinement P-CBS acceptance policy for calibrated ablations,
   but its rejection is reported separately and never relabeled mechanical
   unsolvability.

The deterministic linear mission fallback is disabled by default. It may be
enabled for diagnostics, but fallback artifacts must not be mixed with
evolutionary/QD outputs in result tables.

## QD Boundary

Two archives currently exist and must not be conflated:

- The topology CVT archive drives mission-genome evolution.
- The final-map archive receives only fully generated, tile-oracle-valid
  artifacts and accumulates across pipeline calls.

The advanced pipeline still materializes only the selected topology from each
topology-QD run. Therefore it is topology-level QD plus post-generation final
map archiving, not yet a single end-to-end QD search over all rendered map
elites. A publication claiming end-to-end map QD must generate and validate
the final map for every selected topology elite under the same budget.

Graph validity and final-map validity must be reported separately. A graph can
be valid while neural room geometry, stitching, entity placement, or door
compilation makes the final artifact invalid.

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
5. Report P-CBS separately as bounded-agent readability and difficulty. If the
   opt-in acceptance policy is used, publish its persona, budget, threshold,
   calibration data, and rejection rate.
6. Do not merge these into one "solver success" metric without role labels.
7. For Quality-Diversity evolution, archive quality must be based on
   progression-solvable mission graphs or validated semantic grids. Behavioral
   descriptors can be topological, but the quality score must not reward a
   graph that is only connected after ignoring gates, keys, switches, or
   consumable locks.

## Remaining Search Work

- Run the search-only benchmark over final generated maps with `A*`,
  `Dijkstra`, `P-CBS` personas, and diagnostic solvers separated by role.
- Run the complete graph-to-final-map pipeline on real checkpoints and report
  both pre-compilation graph solvability and post-compilation tile solvability.
- Add an archive-materialization runner that generates rooms, stitches, and
  tile-validates every selected topology elite. Until then, call the current
  method "topology QD with post-hoc final-map archiving," not end-to-end map QD.
- For full-grammar topology QD, recompute descriptors after final export
  repairs or disable repairs. Pre-repair descriptor cells must not be reported
  as properties of a materially changed post-repair graph.
- Add explicit tile/entity/oracle semantics before enabling `ITEM_GATE`,
  `MULTI_LOCK`, `ONE_WAY`, `STATE_BLOCK`, or `WARP` rules in end-to-end
  generation. A single named protected-hazard class is supported; multiple
  protection identities require distinct semantic tile/entity identities.
- If large stitched maps become a runtime bottleneck, extend the static-grid
  ablation with HPA*. Keep it separate from Zelda-state validation just like
  the current JPS ablation.
- Keep human/player claims separate from oracle claims until calibrated
  playtest data exists.
