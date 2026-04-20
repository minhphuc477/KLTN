# Final Protocol And Playability Status 2026-04-17

## Current Production Judgment

The best end-to-end branch remains:

- `outputs/zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1`
- production room branch: `diffusion`

Do **not** switch production to:

- `outputs/zelda_hmolqd_downstream_puzzle_structure_control_v1`

Reason:

- its fixed-graph audit does not clearly beat the earlier branch on the metrics
  that matter most for stable room generation
- the external baseline comparison still does **not** support a claim that this
  repo surpasses prior publications

## Current Evidence

Primary artifacts:

- new branch fixed-graph audit:
  `outputs/zelda_hmolqd_downstream_puzzle_structure_control_v1/protocol_ablation_hybrid_puzzle_control_default_v3/summary.json`
- new branch vs matched-budget / PCG benchmark comparison:
  `outputs/zelda_hmolqd_downstream_puzzle_structure_control_v1/protocol_ablation_hybrid_puzzle_control_default_v3/baseline_comparison/protocol_vs_baselines.json`
- new branch vs old branch comparison:
  `outputs/zelda_hmolqd_downstream_puzzle_structure_control_v1/protocol_ablation_hybrid_puzzle_control_default_v3/branch_comparison_vs_codebook512_puzzle_subtype_v1.json`
- fresh readable manual export of the new branch:
  `outputs/zelda_hmolqd_downstream_puzzle_structure_control_v1/protocol_manual_compare_puzzle_control_v3/summary.json`
- fresh readable manual export of the best branch:
  `outputs/zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1/protocol_manual_compare_currentcode_v5/summary.json`

Key fixed-graph result:

- `puzzle_structure_control_v1` diffusion slightly improves overwrite against the
  old branch, but is much slower and much worse on `CBS` confusion
- `puzzle_structure_control_v1` fast sampler does not beat the old branch overall
- `masked_room` still is not the production branch

Claim-status artifact:

- `protocol_vs_baselines.json` explicitly records
  `"can_claim_surpasses_publications": false`

## Playability Research Decision

Do **not** replace bounded validation with one unlimited `CBS/CBD` solver.

Use a two-layer contract instead:

- report-facing hard oracle: `graph_guided_oracle + graph progression validator + soft-lock checker`
- stricter stress probe: monolithic stitched tile-state `A*`
- behavioral probe: `CBS+` (`CognitiveBoundedSearch`)

Why:

- the hard oracle is for correctness
- bounded `CBS+` is for human-like difficulty / navigation behavior
- an unlimited `CBS+` stops being clearly bounded-rational and collapses toward
  an expensive approximate oracle

Research note:

- `docs/PLAYABILITY_EVALUATION_AND_CBS_RESEARCH_2026_04_16.md`

## Code Improvements Applied

Current codebase improvements that now apply to future exports:

- regular export PNGs now draw readable semantic overlays for push blocks,
  enemies, keys, items, and puzzle markers
- puzzle scaffold adds a readable local push-block prop near interaction zones
  instead of only generic block clutter
- puzzle scaffold now supports staged multi-step puzzle plans with validator-side
  progression gating for ordered `key/item/puzzle -> DOOR_PUZZLE` and
  push-block-to-switch unlocks
- room-local validator planning now uses a bounded but complexity-adaptive state
  budget instead of one fixed cap for every room

These are code improvements, not retroactive checkpoint improvements. Older
exports do not reflect them unless re-run.

## Solver Benchmark Status

The older full `1-9 x variants 1,2 x all personas` CSV:

- `results/cbs_benchmark_levels1_9_variants12_all_personas_v2.csv`

is still useful as a raw artifact, but it predates the latest benchmark-accounting
cleanup and is **not** the preferred publication-grade benchmark output.

A patched rerun was started to:

- `results/cbs_benchmark_levels1_9_variants12_all_personas_v3.csv`

That long benchmark is still in progress in the background. Until it finishes,
use:

- `results/cbs_benchmark_levels1_9_variants12_all_personas_v2_summary/summary.json`

only as interim behavioral evidence, not as the final report table.
