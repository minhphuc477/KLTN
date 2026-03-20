# Architecture Research Judgment (2026-02-24)

## Scope
This judgment evaluates whether the current block architecture is technically coherent, implemented as claimed, and competitive with research-oriented baselines.

Evidence used:
- Local architecture and benchmark docs.
- Runtime code path checks in Block 0 to VII.
- Fresh benchmark run on this repo (`tmp/block_i_benchmark_judgement.json`).
- External literature on mission-space PCG, QD, PCGRL, WFC, and dataset standards.

## Executive Judgment
The architecture is structurally strong and research-aligned (mission-space first, neural-symbolic split, explicit repair and QD evaluation), but it is not yet at SOTA competitiveness in current measured topology quality.

Main reason:
- Completeness is high, but topology/progression realism is still far from VGLC references.
- Current WFC robustness probe under-stresses contradiction behavior, so some robustness claims are not yet fully evidenced.
- Ablation protocol exists in code, but full fixed-seed runs are expensive and not yet completed at thesis-scale sample counts.

## What Is Strong
1. Mission-first topology architecture is correct for Zelda-like progression design.
- Block I grammar+evolution precedes room rendering, matching mission-space theory.
- Code: `src/generation/grammar.py`, `src/generation/evolutionary_director.py`.

2. Canonical pipeline is clearly decomposed and wired.
- Code: `src/pipeline/dungeon_pipeline.py` and canonical map in docs.

3. QD and benchmark instrumentation is much better than baseline projects.
- Repair-rate and WFC robustness outputs are wired.
- Code: `src/evaluation/benchmark_suite.py:1003`, `src/evaluation/benchmark_suite.py:1035`, `src/evaluation/benchmark_suite.py:1166`.

4. Extended ablation protocol is implemented.
- Includes FULL/NO_EVOLUTION/NO_WFC/NO_LOGIC and extended VQ/latent/TPE sweeps.
- Code: `scripts/run_ablation_study.py:560`, `scripts/run_ablation_study.py:567`.

## What Is Not Yet Good Enough
1. Topology quality gap remains large vs reference.
From `tmp/block_i_benchmark_judgement.json`:
- `overall_completeness`: `0.89375`
- `constraint_valid_rate`: `0.575`
- `generated topology_complexity`: `0.147` vs reference `0.461`
- `generated path_length`: `3.575` vs reference `10.944`
- `generated num_nodes`: `10.825` vs reference `29.5`
- `expressive_overlap_reference`: `0.0476`

Interpretation: generation is valid often enough, but mission graphs are still too small/simple relative to VGLC topology targets.

2. WFC contradiction/restart probe is currently weak as a stress test.
- Probe calls WFC with full observed room grids as `initial_grid`.
- Code: `src/evaluation/benchmark_suite.py:711` -> `integrate_weighted_wfc_into_pipeline(... neural_room=room ...)`.
- In WFC implementation, known cells are immediately pinned/collapsed.
- Code: `src/generation/weighted_bayesian_wfc.py:306`.

Observed result in benchmark run: near-zero contradictions/restarts can occur because many runs are effectively pre-collapsed.

3. Exception-swallowing reduces observability in critical paths.
- `src/pipeline/dungeon_pipeline.py:407`
- `src/generation/evolutionary_director.py:580`
- `src/data/zelda_core.py:3587`

This is not a missing implementation, but it can hide logic failures and overstate robustness.

4. "Zero placeholder" wording is too absolute in older docs.
- Canonical blocks are implemented, but abstract methods/no-op branches still exist.
- Audit file already documents this nuance.
- Doc: `docs/BLOCK_BY_BLOCK_ARCHITECTURE_AND_IMPLEMENTATION_AUDIT.md:287`.

## Architecture Drift Status
Drift is reduced but not fully eliminated:
- Good: `MAPElitesEvaluator` in runtime now mirrors into advanced CVT archive when available.
- Code: `src/simulation/map_elites.py:74`, `src/simulation/map_elites.py:77`.
- Remaining risk: dual MAP-Elites modules (`src/simulation/map_elites.py` and `src/evaluation/map_elites.py`) require strict ownership boundaries to avoid future divergence.

## Is It SOTA Today?
Not yet.

Why:
- No completed thesis-scale ablation table in-repo yet.
- No statistically strong head-to-head baseline table against external methods (PCGRL-like, QD variants, etc.) with equal seeds/budgets.
- Current measured topology realism gap is still significant.

## Highest-Impact Next Actions (Ordered)
1. Fix WFC robustness probe to force partial uncertainty.
- Mask a controlled percentage of tiles before WFC probing (for contradiction/restart measurement that is actually informative).

2. Increase Block-I structural difficulty targets.
- Raise node/edge complexity pressure and descriptor-target calibration strength so generated graphs approach reference path length/node count/topology complexity.

3. Make full fixed-seed ablations tractable.
- Add smaller "thesis-smoke" defaults and checkpoint reuse so `scripts/run_ablation_study.py` can complete regularly in CI/local runs.

4. Replace broad `except Exception: pass` in runtime-critical code.
- Convert to scoped exceptions + warning logs with counters.

## External Research Alignment (Used for Judgment)
- Mission/space PCG split (Dormans & Bakkes): https://research.tilburguniversity.edu/en/publications/generating-missions-and-spaces-for-adaptable-play-experiences
- MAP-Elites: https://arxiv.org/abs/1504.04909
- CVT-MAP-Elites: https://arxiv.org/abs/1610.05729
- CMA-ME: https://arxiv.org/abs/1912.02400
- PCGRL: https://arxiv.org/abs/2001.09212
- Constraint handling in evolutionary search (Deb feasibility rules): https://www.researchgate.net/publication/222658699_Deb_K_An_Efficient_Constraint_Handling_Method_for_Genetic_Algorithm_Computer_Methods_in_Applied_Mechanics_and_Engieering_186_311-338
- WFC framing and variants: https://adamsmith.as/papers/wfc_is_constraint_solving_in_the_wild.pdf and https://www.mdpi.com/1999-4893/15/2/60
- VGLC paper: https://arxiv.org/abs/1606.07487
- NumPy RNG reproducibility guidance: https://numpy.org/doc/stable/reference/random/generator.html

## Bottom Line
Your architecture is technically credible and ahead of many prototype PCG systems in block clarity and instrumentation, but current evidence does not support a "best/SOTA" claim yet. The key blocker is measurable topology realism and stronger, stress-valid robustness/ablation evidence.

## Implementation Update (2026-02-24)
The following next-impact actions were implemented after this judgment:
- WFC robustness probe now uses partial-mask stress inputs and reports mask-intensity metrics.
- Block-I benchmark now supports reference-aligned room budget and room-count bias controls.
- Calibration objective now includes path-length and node-count pressure.
- Ablation runner now supports quick profile and runtime-budgeted partial export.
- Ablation significance now includes Benjamini-Hochberg FDR-adjusted p-values.
- Ablation metric semantics corrected: `confusion_ratio` (CBS vs optimal path ratio) is now distinct from CBS `confusion_index`.
- Runtime fallback observability was improved in Block 0 / Block II-IV paths (no silent precheck pass-through in critical exception branches).

## Implementation Update (2026-02-24, Round 2)
Additional research-driven fixes were implemented and measured:

1. Mission-graph metadata preservation across conversions.
- `mission_graph_to_networkx(...)` / `networkx_to_mission_graph(...)` now preserve progression-critical edge/node metadata (fungible keys, item gates, switch requirements, hazard metadata, etc.).
- Impact on benchmark (same command profile):
  - `constraint_valid_rate`: `0.50 -> 1.00`
  - `overall_completeness`: `0.875 -> 1.000`

2. WFC collapse-selection logic corrected.
- Low-entropy unresolved cells are now collapsed instead of skipped.
- This removed false "no valid cell" failures and restart/fallback inflation.
- Impact on benchmark (same command profile):
  - `wfc_restart_rate`: `1.00 -> 0.00`
  - `wfc_fallback_rate`: `1.00 -> 0.00`
  - `wfc_mean_restarts`: `3.0 -> 0.0`

3. Evolutionary fitness strengthened with critical-path pressure.
- Added path-depth and path-coverage terms (in addition to curve/backtracking) to reduce trivial short critical paths.
- Impact on benchmark (same command profile):
  - `generated path_length`: `2.65 -> 7.20` (reference mean `10.94`)
  - `generated linearity`: `0.189 -> 0.371` (reference mean `0.454`)
  - `expressive_overlap_reference`: `0.103 -> 0.179`
  - `fidelity_js_divergence`: `0.410 -> 0.323` (lower is better)

Remaining blocker for SOTA-level claim:
- `topology_complexity` is still under target (`0.108` vs reference `0.461`), meaning cycle/gating richness remains insufficient despite stronger path depth.

## Implementation Update (2026-02-24, Round 3)
Follow-up research/implementation focused on search correctness and descriptor-aligned optimization:

1. Evolutionary loop correctness bug fixed.
- Offspring are now evaluated before survivor selection in `(mu+lambda)` search.
- Before fix, offspring fitness stayed `0.0`, effectively freezing evolution quality improvement.
- Code: `src/generation/evolutionary_director.py` (`evolve`, offspring evaluation before `_select_survivors`).

2. Descriptor-aware fitness integrated in Block I search.
- Fitness now includes descriptor target matching (linearity, leniency, progression complexity, topology complexity), not only tension curve and path terms.
- Block-I benchmark now feeds reference descriptor means from VGLC into generation-time fitness targets.
- Code: `src/generation/evolutionary_director.py` (descriptor extraction + fitness blend), `src/evaluation/benchmark_suite.py` (descriptor target wiring).

3. Backtracking metric logic corrected.
- Previous metric used `unique_nodes / total_steps` on a BFS simple path, which is near-constant and non-informative.
- Replaced with structural proxy tied to linearity + cycle density.
- Code: `src/generation/evolutionary_director.py` (`_calculate_backtracking_score`).

Measured impact (full benchmark profile, `tmp/block_i_benchmark_full_final_current.json`):
- `overall_completeness`: `1.000` (stable)
- `constraint_valid_rate`: `1.000` (stable)
- `wfc_restart_rate`: `0.000` (stable)
- `wfc_fallback_rate`: `0.000` (stable)
- `topology_complexity`: `0.194` (up from earlier `0.108`, reference `0.461`)
- `leniency`: `0.532` (close to reference `0.475`)
- `progression_complexity`: `0.672` (close to reference `0.691`)

Current blocker remains:
- Critical path depth is still short in this configuration (`path_length=4.7` vs reference `10.94`), so topology/branching realism and path-depth realism are not simultaneously matched yet.

## Implementation Update (2026-02-24, Round 4)
Research-backed best-practice update was applied to Block I search control:

1. Feasibility-first evolutionary selection (constraint handling).
- Implemented Deb-style ordering in tournament and survivor selection:
  - feasible solutions dominate infeasible ones,
  - infeasible solutions are ranked by total normalized violation,
  - fitness remains the quality objective within same feasibility class.
- Code: `src/generation/evolutionary_director.py` (`_tournament_selection`, `_select_survivors`, evaluator diagnostics).

2. Descriptor target wiring now includes path depth and graph size.
- `path_length` and `num_nodes` reference means are now passed into:
  - calibration generation runs,
  - production benchmark generation runs,
  - evaluator target bands (critical-path and node-count constraints).
- Code: `src/evaluation/benchmark_suite.py`, `src/generation/evolutionary_director.py`.

3. Evolution diagnostics expanded for search observability.
- Added `feasible_ratio_history` and `avg_violation_history` to evolution stats, so search runs can be judged by constraint convergence, not only scalar fitness.
- Code: `src/generation/evolutionary_director.py`.

Benchmark signal (after Round 4):
- File: `tmp/block_i_benchmark_round4_full_compare.json`
- `overall_completeness`: `1.000` (stable)
- `constraint_valid_rate`: `1.000` (stable)
- `path_length`: `4.90` (up from prior full-profile `4.70`)
- `num_nodes`: `24.90` (up from prior `23.25`)
- `topology_complexity`: `0.183` (still far from reference `0.461`)

Interpretation:
- Feasibility-first search improves path-depth and size pressure while preserving completeness.
- Gains are modest and topology richness remains under target, so next step is still stronger cycle/shortcut objective pressure and/or archive-driven QD selection.

## Implementation Update (2026-02-24, Round 5)
External-comparability gap implementation is now in place:

1. Matched-budget multi-method benchmark runner added.
- New script: `scripts/run_matched_budget_topology_benchmark.py`
- Compares `RANDOM`, `ES`, `GA`, `MAP_ELITES`, `FULL` under the same evaluation budget per seed.

2. Publication-style statistical reporting added for method comparisons.
- Outputs paired bootstrap CIs, paired sign-permutation p-values, and BH-FDR corrected significance.
- Writes comparable artifacts:
  - `matched_budget_raw.csv`
  - `matched_budget_summary.csv`
  - `matched_budget_significance.csv`
  - `matched_budget_report.json`
  - `matched_budget_report.md`

3. Kaggle-oriented execution profile added.
- `--kaggle-t4x2` preset for larger run scale and calibrated FULL baseline.
- Protocol document: `docs/MATCHED_BUDGET_BENCHMARK_PROTOCOL.md`.
