# P-CBS Affordance Memory And Ablation Note

Last updated: 2026-04-17

## Gap

The earlier `P-CBS` implementation already had:

- revisit penalty
- conditional uncertainty penalty
- bounded working memory
- bounded deliberation budget

But it still lacked a Zelda-specific cognitive mechanism for remembering
unresolved progression affordances such as:

- locked doors
- bombable gates
- puzzle anchors

This made the novelty claim too generic. A reviewer could still argue that the
solver was mostly a weighted heuristic controller with generic bounded-rational
terms.

## Upgrade

`P-CBS` now includes explicit `progression-affordance memory` in
[cognitive_bounded_search.py](../src/simulation/cognitive_bounded_search.py):

- affordances are remembered as explicit episodic memory items
- each affordance stores its requirement family (`small_key`, `bomb`,
  `boss_key`, `item_or_observation`)
- when inventory changes, remembered affordances that become actionable are
  reactivated by a salience boost
- move scoring now includes:
  - `+ ξ affordance_resume`
  - `- ζ affordance_forgetting`

This is implemented as:

- `MemoryItemType.AFFORDANCE`
- `_affordance_requirement(...)`
- `_is_affordance_satisfied(...)`
- `_reactivate_affordances_after_inventory_change(...)`
- `_estimate_affordance_resumption_bonus(...)`
- `_estimate_affordance_forgetting_penalty(...)`

## Why This Is More Defensible

The upgraded novelty claim is:

`P-CBS is a bounded-rational persona validator that models not only memory decay
and uncertainty, but also inventory-triggered recall of previously observed
progression affordances in Zelda-like dungeons.`

This remains a system contribution, not a claim of a brand-new universal search
family.

## New Metrics

The benchmark/report path now records:

- `affordance_reactivations`
- `affordance_guided_steps`
- `inventory_change_events`

These are surfaced in:

- [run_cbs_benchmarks.py](../scripts/run_cbs_benchmarks.py)
- [pcbs_validation.py](../src/evaluation/pcbs_validation.py)

## New Ablation Runner

Component ablations are now reproducible with:

- [run_pcbs_component_ablation.py](../scripts/run_pcbs_component_ablation.py)

Supported variants:

- `full`
- `no_revisit`
- `no_uncertainty`
- `no_deliberation`
- `no_affordance`

## Current Quick Evidence

Explorer quick ablation on `D1_v1`:

- artifact path: `results/pcbs_component_ablation_explorer_quick_v1/report.md`
- full `P-CBS` solved with path length `574`
- `no_affordance` still solved, but path length degraded to `846`
- `no_affordance` also increased confusion, deliberation, budget exhaustion, and
  peak frustration

Novice quick ablation on `D1_v1`:

- artifact path: `results/pcbs_component_ablation_novice_quick_v1/report.md`
- all variants timed out under the bounded budget
- removing revisit or uncertainty terms still changes confusion/frustration
  statistics, which is useful for thesis-facing sensitivity analysis

## Remaining Limitation

The quick ablation slices are not enough to claim publication-level superiority.
The remaining required evidence is:

- multi-map component ablation on oracle-solvable maps
- full `CGR = P(P-CBS fails | hard oracle solves)` tables
- final alignment against the matched-budget external baselines already tracked
  elsewhere in the repo
