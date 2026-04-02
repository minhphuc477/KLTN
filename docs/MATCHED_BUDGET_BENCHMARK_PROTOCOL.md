# Matched-Budget Topology Benchmark Protocol

This protocol compares Block-I topology generators under the same evaluation budget per seed.

## Methods
- `RANDOM`: random genome search
- `ES`: mutation-only evolutionary search (`crossover_rate=0`)
- `GA`: crossover + mutation evolutionary search
- `MAP_ELITES`: CVT-emitter search using the same runtime `EvolutionaryTopologyGenerator` backend (`search_strategy=cvt_emitter`)
- `FULL`: project default tuned evolutionary profile

## External Baseline Panel
Use these when comparing the repo against publishable topology-generation baselines rather than only internal ablations.

- `Random grammar rollout`: lower bound for any search-based method.
- `Smith et al. 2018 ASP`: symbolic action-adventure dungeon generator for Zelda-like progression graphs.
- `Pereira et al. 2021 EA`: locked-door dungeon evolutionary baseline with explicit mission-structure evaluation.
- `GraphRNN`: generic graph-generation baseline when a reference graph corpus is available.
- `DiGress`: modern labeled-graph generation baseline when training data is available.

Minimum recommended paper-ready panel:
- `Random grammar`
- `FULL`
- `ASP`
- `Pereira EA`
- `GraphRNN` or `DiGress`

## Metrics Reported
- feasibility / completeness: `feasible_rate`, `overall_completeness`, `constraint_valid_rate`
- structure: `linearity`, `leniency`, `progression_complexity`, `topology_complexity`, `path_length`, `num_nodes`
- progression-correctness: `key_before_lock_rate`, `switch_before_gate_rate`, `battery_satisfaction_rate`
- structure-depth: `path_redundancy`, `articulation_count`, `articulation_ratio`
- optional-content quality: `branch_count`, `branch_utility_rate`, `secret_component_count`, `secret_content_discoverability_rate`
- generation-time robustness: `generation_constraint_rejections`, `candidate_repairs_applied`
- quality/diversity proxies: `novelty_vs_reference`, `graph_edit_distance`, `fidelity_js_divergence`, expressive coverages
- runtime: `generation_time_sec`, `evaluations_used`
- statistical testing: paired bootstrap CIs, paired sign-permutation p-values, BH-FDR correction

Interpretation rules:
- `key_before_lock_rate`, `switch_before_gate_rate`, and `battery_satisfaction_rate` are conditional correctness rates. When a graph has no such gate type, the rate is treated as `1.0`; inspect the corresponding gate counts before drawing conclusions.
- `branch_utility_rate` is the fraction of optional side branches that contain meaningful reward/progression content or reconnect as a loop; inspect `branch_count` alongside it.
- `secret_content_discoverability_rate` is a heuristic repo-level proxy for whether secret content is attached through explicit hidden/bombable semantics and yields useful content.

## Reporting Rules
- Match room-count budgets across methods before comparing descriptor means.
- Report `95%` bootstrap confidence intervals for all primary completeness and structure metrics.
- Report means on at least `64` generated graphs per method when possible; `128` to `256` is preferred for final comparison.
- Keep reference graphs fixed across all methods and seeds.
- Do not compare room/layout baselines such as `HouseDiffusion` or `LayoutDM` against Block I topology generation; those belong to the room/layout stage, not mission-graph generation.

## Run Command
```bash
python scripts/run_matched_budget_topology_benchmark.py \
  --data-root "Data/The Legend of Zelda" \
  --methods RANDOM,ES,GA,MAP_ELITES,FULL \
  --num-samples 10 \
  --seed 42 \
  --eval-budget 512 \
  --output results/matched_budget
```

## Kaggle T4 x2 Preset
```bash
python scripts/run_matched_budget_topology_benchmark.py \
  --kaggle-t4x2 \
  --data-root "Data/The Legend of Zelda" \
  --output results/matched_budget_kaggle
```

Optional ablation run on Kaggle T4 x2:
```bash
python scripts/run_ablation_study.py \
  --kaggle-t4x2 \
  --output results/ablation_kaggle
```

## Output Files
- `matched_budget_raw.csv`
- `matched_budget_summary.csv`
- `matched_budget_significance.csv`
- `matched_budget_report.json`
- `matched_budget_report.md`

## External Benchmark Alignment (Next Step)
- PCG Benchmark framework repo (used in FDG 2025 paper): `https://github.com/amidos2006/pcg_benchmark`
- Benchmark paper: `https://arxiv.org/abs/2503.21474`

The current script gives matched-budget internal head-to-head evidence.  
Next, map this project's graph representation to `pcg_benchmark` Zelda content/control spaces for direct cross-publication comparability.

Implemented bridge:
```bash
python scripts/run_pcg_benchmark_alignment.py \
  --data-root "Data/The Legend of Zelda" \
  --methods FULL_GA,FULL_CVT,CORE_GA \
  --problems zelda-v0,zelda-enemies-v0,zelda-large-v0 \
  --control-mode graph \
  --pcg-benchmark-repo path/to/pcg_benchmark \
  --output results/pcg_benchmark_alignment
```

Notes:
- `control-mode graph` now evaluates against benchmark-aligned controls derived from the mission graph's split ratio, then scaled into the legal Zelda control range. This preserves relative progression shape while reserving enough path budget to satisfy the benchmark's solution-length regime.
- The mapper now reserves an explicit `player -> key -> door` corridor budget and caps enemy placement to the benchmark target for each Zelda variant. This specifically fixes the previous `zelda-v0` over-enemy failure mode.
- `zelda-large-v0` now has a control-preserving fallback path: if the richer free-routed mapper collapses the requested path budget or falls below the benchmark solution-length floor, the bridge swaps in a corridor-only realization that preserves the aligned `player_key` / `key_door` controls exactly.
- The bridge now fails closed on missing progression semantics. If a graph does not carry explicit `start`, `key`, and `goal` anchors with valid `start -> key -> goal` connectivity, the mapper marks it as `semantic_valid=false` instead of inventing anchors.
- The bridge is intentionally lossy: advanced mission-graph semantics such as multi-locks, switch batteries, and secret gating are collapsed into the simpler Zelda benchmark domain.
- Reports now show both:
  - strict pass rates from `pcg_benchmark.evaluate(...)`
  - continuous detail means averaged from the per-sample `quality` / `diversity` / `controlability` arrays
- This distinction matters because a run can be close to benchmark compliance on the continuous scores while still failing the strict pass threshold.
- Reports now also include mapper diagnostics:
  - fallback rate
  - semantic-valid rate
  - initial vs final realized `player_key` / `key_door` means
  - initial vs final absolute control errors
- Output files:
  - `pcg_benchmark_alignment_raw.csv`
  - `pcg_benchmark_alignment_summary.csv`
  - `pcg_benchmark_alignment_report.json`
  - `pcg_benchmark_alignment_report.md`

## Room-Branch Internal Benchmark
This closes the internal reproducibility gap for Block III/IV and masked-room comparisons under matched topology/search budgets.

```bash
python scripts/run_room_branch_benchmark.py \
  --data-root "Data/The Legend of Zelda" \
  --num-samples 8 \
  --seed 42 \
  --output results/room_branch_benchmark
```

Outputs:
- `room_branch_raw.csv`
- `room_branch_summary.csv`
- `room_branch_significance.csv`
- `room_branch_benchmark_report.json`
- `room_branch_benchmark_report.md`

## Multi-GPU Training Note
- For two GPUs on one machine, PyTorch recommends `DistributedDataParallel` with one process per GPU:
  - `https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html`
  - `https://docs.pytorch.org/tutorials/beginner/dist_overview.html`
