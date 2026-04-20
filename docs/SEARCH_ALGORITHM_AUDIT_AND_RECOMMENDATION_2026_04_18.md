# Search Algorithm Audit And Recommendation

Last updated: 2026-04-18

This note answers a specific question:

- is `D* Lite` actually the primary search algorithm in this repo?
- if not, what should be the primary search stack for the current
  `H-MOLQD` architecture?

Short answer:

- `No`, `D* Lite` is not the repo's hard oracle.
- the correct primary stack is still:
  - `graph_guided_oracle`
  - `A*` / hybrid A* tile-state validation
  - `Dijkstra` exact fallback when heuristic A* underperforms on staged puzzles
- `D* Lite` should remain a `replanning probe`, not the primary static
  correctness oracle.

## 1. Code Truth

Current code says:

- [`src/simulation/search_factory.py`](../src/simulation/search_factory.py)
  marks:
  - `A*` as `validation_role="oracle"` and `canonical_use="hard_oracle"`
  - `D* Lite` as `validation_role="replanning"` and
    `canonical_use="incremental_replanning"`
- [`src/simulation/dstar_search.py`](../src/simulation/dstar_search.py)
  explicitly labels `D* Lite` as:
  - `intended_use="incremental_replanning"`
  - `independent_oracle=False`
- [`src/simulation/validator.py`](../src/simulation/validator.py)
  runs:
  - primary `A*` / hybrid A*
  - then `Dijkstra` fallback if needed
- [`scripts/run_fast_sampler_visual_audit.py`](../scripts/run_fast_sampler_visual_audit.py)
  explicitly states:
  - `A* remains the hard grid-level oracle in this suite.`

So the repo’s actual primary correctness contract is not `D* Lite`.

## 2. Why D* Lite Is Not The Best Primary Oracle Here

The current architecture does not validate simple static occupancy grids only.
It validates:

- inventory state
- key / lock progression
- staged puzzle semantics
- room-to-room graph guidance
- stateful multi-step puzzle plans

That breaks the clean assumptions behind most grid-speedup papers.

### 2.1 Why `JPS` is not the hard oracle here

`JPS` is excellent for static, uniform-cost, plain occupancy grids. But the
repo’s full game-state search space is not that:

- state includes keys, opened doors, items, and puzzle-stage completion
- two identical `(x, y)` tiles can represent different legal futures
- symmetry pruning assumptions become unreliable once the transition model is
  stateful rather than purely geometric

So `JPS` is a good future research branch for simple static tile validation, but
not the right hard oracle for the current stateful dungeon validator.

### 2.2 Why `Theta*` is not the hard oracle here

`Theta*` is useful when any-angle movement matters. But Zelda dungeon movement
in this repo is tile-semantic and interaction-heavy:

- doors
- locks
- puzzle gates
- push-block semantics

The key problem is not path smoothing. It is exact state transition validity.
So `Theta*` is not the right report-facing correctness oracle either.

### 2.3 Why `D* Lite` is still valuable

`D* Lite` is valuable when:

- the map changes online
- hidden obstacles are revealed incrementally
- the planner must repair paths repeatedly instead of solving one fixed static
  state-space instance

That matches the GUI replanning story and dynamic probe story.
It does not match the repo’s main export-time correctness contract.

## 3. Literature Basis

Primary sources:

- Koenig and Likhachev, *D* Lite*, AAAI 2002:
  <https://aaai-25.aaai.org/Papers/AAAI/2002/AAAI02-072.pdf>
- Koenig, Likhachev, Furcy, *Lifelong Planning A**,
  Artificial Intelligence 2004:
  <https://doi.org/10.1016/j.artint.2003.12.001>
- Harabor and Grastien, *Online Graph Pruning for Pathfinding on Grid Maps*,
  AAAI 2011:
  <https://pathfinding.ai/pdf/harabor-grastien-aaai11.pdf>
- Daniel, Nash, Koenig, Felner, *Theta*: Any-Angle Path Planning on Grids*:
  <https://arxiv.org/abs/1401.3843>
- Botea, Müller, Schaeffer, *Near Optimal Hierarchical Path-Finding*:
  <https://webdocs.cs.ualberta.ca/~jonathan/publications/ai_publications/jogd.pdf>
- Björnsson et al., *Fringe Search: Beating A* at Pathfinding on Game Maps*:
  archived at:
  <https://web.archive.org/web/20090219220415/http://www.cs.ualberta.ca/~games/pathfind/publications/cig2005.pdf>

Research conclusion:

- `D* Lite` is strong for incremental replanning.
- `JPS` is strong for static uniform grids.
- `Theta*` is strong for any-angle geometry.
- `HPA*` and `Fringe Search` are strong game-AI speedups in large static maps.
- none of those automatically dominate `A*` when the search state includes
  inventory, door state, and staged puzzle progress.

## 4. Best Search Stack For This Repo

### 4.1 Hard correctness oracle

Keep:

1. `graph_guided_oracle`
2. `hybrid A*`
3. `Dijkstra` exact fallback
4. `graph_progression`
5. `softlock_check`

Reason:

- this stack matches the actual stateful mechanics
- it is conservative and thesis-safe
- it already aligns with the repo’s current validation contract

### 4.2 Comparison / ablation solvers

Keep:

- `BFS`
- `Dijkstra`
- `Greedy`
- `D* Lite`
- `DFS/IDDFS`
- `Bidirectional A*`

Reason:

- they help characterize the search space
- they should not be promoted to hard oracle status without stronger evidence

### 4.3 Behavioral validator

Keep:

- `P-CBS`

Reason:

- it measures bounded-rational player-like difficulty
- it is not the hard oracle

### 4.4 Optional future research additions

Promising but not implemented as primary oracle:

- `JPS` for simple static occupancy-grid validation only
- `HPA*` for very large static stitched maps if a clean abstraction layer is
  built
- `Theta*` only if the movement model becomes any-angle and not strictly
  tile-semantic

## 5. Concrete Fixes Applied In This Pass

Implemented:

- solver registry now exposes explicit canonical-use metadata
- `D* Lite` is now labeled `replanning`, not generic comparison-only metadata
- export search-suite payloads now carry `canonical_use`
- GUI `D* Lite` logging now says `incremental replanning probe` and makes the
  `A*` fallback explicit

Main files:

- [`src/simulation/search_factory.py`](../src/simulation/search_factory.py)
- [`scripts/run_fast_sampler_visual_audit.py`](../scripts/run_fast_sampler_visual_audit.py)
- [`src/gui/gameplay/path_strategies.py`](../src/gui/gameplay/path_strategies.py)

## 6. Honest Remaining Search Gaps

Still not done:

1. full latest-code rerun of the long persona benchmark
2. a dedicated search-only benchmark table comparing the canonical solvers on
   the current staged puzzle slice
3. a clean research branch for `JPS` or `HPA*` on simplified static grids

What is *not* needed right now:

- replacing the current hard oracle with `D* Lite`
- replacing the current hard oracle with `Theta*`
- replacing the current hard oracle with sampling planners

## 7. Bottom Line

If the question is:

- `what is the best search for this current model/architecture?`

then the answer is:

- `A* + graph guidance + Dijkstra exact fallback` for correctness
- `D* Lite` only for incremental replanning experiments / GUI dynamic behavior
- `P-CBS` only for bounded-rational behavioral evaluation
