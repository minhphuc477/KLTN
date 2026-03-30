# Block I Topology Realism Tuning (2026-02-24)

## Scope
This note tracks the implementation pass focused on the Block I topology realism gap against VGLC references, specifically:

- `cycle_density`
- `shortcut_density`
- `gate_depth_ratio`

## Research Basis

- VGLC dataset and graph references: https://arxiv.org/abs/1606.07487
- Zelda-like lock/key + player-validation framing: https://www.sciencedirect.com/science/article/pii/S0950705121003361
- Mission-vs-space graph grammar framing (Dormans lineage, used throughout Block I design).
- Quality-Diversity pressure framing for descriptor targeting:
  - CVT-MAP-Elites: https://arxiv.org/abs/1610.05729
  - CMA-ME: https://arxiv.org/abs/2003.04389
  - ME-MAP-Elites survey/update: https://arxiv.org/abs/2503.14760

## Implemented Changes

### 1. Bidirectional semantics consistency

- File: `src/generation/grammar.py`
- File: `src/generation/evolutionary_director.py`

Changes:
- `MissionGraph` now treats `{PATH, SHORTCUT, WARP, STAIRS, HIDDEN}` as bidirectional traversal types in adjacency/matrix rebuild.
- Directed export now mirrors bidirectional traversal edges so generated `DiGraph` semantics match generation-time traversal semantics.
- Reverse traversable edge set in evaluator updated to include `STAIRS` and `HIDDEN`.

Why:
- Previously, internal traversal and exported graph directionality were inconsistent, inflating directionality-gap and skewing descriptor realism.

### 2. Loop closure rule correction (MergeRule)

- File: `src/generation/grammar.py`

Changes:
- `MergeRule` now creates loop closures as `PATH` edges (with `metadata.loop_closure`), not explicit `SHORTCUT` edges.
- Added protected-node filtering (avoid direct start/goal coupling).
- Added bounded loop span (`2 <= span <= 0.35 * |V|`) to prevent trivializing the critical path.

Why:
- This increases cycle rank using Zelda-style loop doors/corridors instead of overproducing explicit shortcut-labeled edges.

### 3. Target-aware prior shaping in evolutionary director

- File: `src/generation/evolutionary_director.py`

Changes:
- Added normalized rule-weight renormalization utility.
- Added exact rule-ID selectors for targeted priors.
- Added target-aware damping for explicit shortcut operators:
  - `AddTeleport`, `AddItemShortcut`, `AddResourceLoop`
- Added target-aware damping for gate-heavy operators when target gate depth is low:
  - `InsertLockKey`, `AddFungibleLock`, `AddItemGate`, `AddBossGauntlet`, `InsertSwitch`, `AddEntangledBranches`, `AddHazardGate`, `AddCollectionChallenge`, `AddMultiLock`
- Added shortcut-overflow remapping in offspring injection: explicit shortcut genes are rewritten toward non-shortcut topology/gating rules when shortcut excess is detected.

Why:
- Convert descriptor targets into direct search pressure (instead of post-hoc comparison only).

## Benchmark Trend (same config, quick 8-sample smoke)

Reference means:
- `cycle_density=0.3025`
- `shortcut_density=0.0156`
- `gate_depth_ratio=0.1471`

Progress snapshots:

1. `results/block_i_topology_tuned_round7c_2026_02_24.json`
- `cycle_density=0.2254`
- `shortcut_density=0.0636`
- `gate_depth_ratio=0.2074`
- `directionality_gap=0.8750`

2. `results/block_i_topology_tuned_round10_merge_loop_path_2026_02_24.json`
- `cycle_density=0.3189`
- `shortcut_density=0.0581`
- `gate_depth_ratio=0.1605`
- `directionality_gap=0.0500`

3. `results/block_i_topology_tuned_round11_shortcut_damped_2026_02_24.json`
- `cycle_density=0.3042`
- `shortcut_density=0.0128`
- `gate_depth_ratio=0.2158`
- `directionality_gap=0.0000`

4. `results/block_i_topology_tuned_round13_gate_damped_2026_02_24.json`
- `cycle_density=0.2977`
- `shortcut_density=0.0335`
- `gate_depth_ratio=0.1518`
- `directionality_gap=0.0750`

## Current Judgment

- `cycle_density`: now near reference.
- `gate_depth_ratio`: now near reference in latest run.
- `shortcut_density`: much improved versus early baseline, but still variant-sensitive because edge budget remains below reference (`~40` vs `~63` edges mean), making each shortcut edge contribute more ratio mass.

## Remaining Gap (highest impact)

- Raise structural edge/node budget toward reference while preserving solvability.
- Add matched-budget runs at larger sample sizes (>=64 graphs) for stable significance on descriptor gaps.
- Keep shortcut damping but add edge-budget-aware normalization pressure so shortcut ratio is not dominated by low total edges.

## Additional Objective-Pressure Pass (Round 14-17)

### What changed in code

- File: `src/generation/evolutionary_director.py`
  - Added generation-level adaptive rule-prior update (`_adapt_global_rule_prior_from_population`) using population descriptor errors for:
    - `cycle_density`
    - `shortcut_density`
    - `gate_depth_ratio`
    - `path_depth_ratio`
  - Added drift control toward target-aware priors (`_relax_rule_weights_to_target_prior`) to avoid runaway weight collapse/explosion.
  - Upgraded deficit estimation to use quantiles (p25/p75) instead of simple mean, so one strong parent does not hide structural deficits.
  - Added explicit overshoot correction for gate depth (`gate_excess`) in both mutation-rate adaptation and rule-remapping.
  - Added `topology_realism_error` in evaluator diagnostics and integrated it into fitness/constraint pressure.
  - Added survivor tie-break pressure on `topology_realism_error` (after fitness) for feasible and infeasible sorting.

- File: `src/evaluation/benchmark_suite.py`
  - Extended default `descriptor_targets` in benchmark-from-scratch to include feature ratios:
    - `puzzle_density`, `item_density`, `gate_variety`
    - `bombable_ratio`, `soft_lock_ratio`, `switch_ratio`, `stair_ratio`
  - Corrected shortcut semantics:
    - `shortcut_density` now excludes generic stairs (`stair/stairs`) so shortcut realism is not conflated with vertical-connector frequency.
    - `STAIR_EDGE_TYPES` remains separate and is still reported via `stair_count`/`stair_ratio`.

### Why this was necessary

- Before this pass, shortcut realism pressure was partially confounded by stair connectors in benchmark descriptors.
- That made the topology objective push against a mixed signal (true shortcuts + staircase connectors), causing unstable shortcut matching.
- The updated pressure now directly targets topology realism while keeping stair semantics as a separate feature axis.

### New benchmark snapshots

1. `results/block_i_topology_tuned_round14_adaptive_pressure_2026_02_24.json`
- `cycle_density=0.3092` (ref `0.3025`)
- `shortcut_density=0.0340` (ref `0.0156`, pre-semantics-fix metric)
- `gate_depth_ratio=0.1319` (ref `0.1471`)

2. `results/block_i_topology_tuned_round16_feature_targeted_2026_02_24.json`
- `cycle_density=0.3077`
- `shortcut_density=0.0384`
- `gate_depth_ratio=0.1139`

3. `results/block_i_topology_tuned_round17_shortcut_metric_fix_2026_02_24.json`
- `cycle_density=0.3139` (ref `0.3025`)
- `shortcut_density=0.0000` (ref `0.0000`, after shortcut/stair semantic separation)
- `gate_depth_ratio=0.1278` (ref `0.1471`)
- `overall_completeness=1.0`, `constraint_valid_rate=1.0`

### Visual artifacts

- `results/topology_viz_round17/generated_gallery.png`
- `results/topology_viz_round17/reference_gallery.png`
- `results/topology_viz_round17/descriptor_scatter.png`
- `results/topology_viz_round17/summary.json`

### Current judgment (after round 17)

- Cycle realism: close to reference mean.
- Shortcut realism: aligned under corrected semantics (shortcuts separated from stairs).
- Gate-depth realism: improved but still below reference mean in this seed/config, so gate-depth pressure should be increased next while preserving feasibility.
- Note: with current VGLC graph labels, explicit shortcut annotations are sparse; after semantics correction, the shortcut reference mean becomes ~0.0. For stronger shortcut supervision, add explicit shortcut tags in reference preprocessing.
