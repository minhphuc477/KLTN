# Next Chat Context

Use this as the first prompt in a new chat if you need continuity on the
current `H-MOLQD` state.

## Project

Repo root: current checkout root.

Goal: finalize a thesis-grade Zelda dungeon generator with:

1. topology generation
2. room generation
3. symbolic repair / refinement
4. stitched-dungeon validation
5. bounded-rational persona validation

Current final architecture is hybrid, not fully neural:

`MAP-Elites DAG -> graph-conditioned VQ-VAE + diffusion -> symbolic repair / WFC-style cleanup -> deterministic semantic marker overlay -> stitching -> hybrid mechanical contract -> P-CBS`

## Current Best Thesis-Safe Position

- canonical run dir:
  `outputs/zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1`
- canonical tokenizer:
  `codebook512`
- canonical room backbone:
  `diffusion`
- `fast sampler` is now competitive on the latest fixed-graph `v22` slice, but
  not a clean overall replacement
- `masked room` remains auxiliary
- puzzle generation is now hybrid and stateful, not purely decorative
- hard validation is now the hybrid mechanical contract, not only monolithic
  stitched tile-state `A*`

## Hard Validation Contract

Report-facing hard oracle:

- `graph_guided_oracle`
- `graph_progression`
- `softlock_check`
- derived flag:
  `mechanical_contract.hybrid_oracle_pass`

Important:

- monolithic stitched tile-state `A*` now times out on the current stateful
  puzzle slice and must be treated as a stricter stress probe, not the only
  pass/fail criterion
- `P-CBS` is the experiential / bounded-rational validator, not the hard
  correctness oracle

## Puzzle Module Status

Stateful multi-step puzzle mechanics were implemented in hybrid form:

- ordered staged plans such as:
  `collect_key -> collect_item -> step_on_puzzle / push_block_to_switch -> DOOR_PUZZLE`
- generator writes puzzle metadata through room and dungeon results
- validator tracks completed puzzle stages and gates `DOOR_PUZZLE` on stage
  completion
- runtime puzzle grammar now uses interaction and sequence contracts, not only
  clutter density

Main implementation files already patched:

- `src/pipeline/dungeon_pipeline.py`
- `src/simulation/validator.py`
- `src/evaluation/pcbs_validation.py`
- `scripts/run_fast_sampler_visual_audit.py`

New on 2026-04-18 / 2026-04-19:

- a `learned staged-puzzle conditioning path` now exists in code
- shared `puzzle_stage_condition` metadata is built in:
  - `src/pipeline/room_topology_conditioning.py`
  - `src/zelda_data/zelda_loader.py`
- diffusion / masked-room conditioning can now append ordered stage tokens
- room-topology priors can optionally inject ordered stage traces
- diffusion / masked-room / fast-sampler training now also have an explicit
  `puzzle-stage semantics head`
  - supervised on gate family, sequence-required flag, stage count, and ordered
    stage slots from generated room logits

Important boundary:

- this closes the `code gap`
- it does **not** close the `evidence gap`
- all current checkpoints, including `stageconditioned_v1`, are outdated for
  any claim about `learned multi-step puzzle semantics` until retrained with
  the new staged-puzzle flags and semantic-loss branch

## P-CBS Status

Thesis-safe novelty claim:

- `P-CBS` is a repo-novel bounded-rational persona validator integrated into
  dungeon PCG evaluation
- it is not the first persona playtester
- it is not a new universal shortest-path family like `A*`

Main implementation file:

- `src/simulation/cognitive_bounded_search.py`

Current literature-bound claim boundary:

- okay: bounded cognition, revisit penalty, uncertainty penalty, deliberation
  budget, affordance memory, frustration, focus persistence, PCG-loop
  integration
- not okay: "first persona validator ever" or "replaces A* as the hard oracle"

## OOM / CUDA Safety Fixes Already Applied

OOM-safe export retry ladder is now in:

- `scripts/run_fast_sampler_visual_audit.py`
- `scripts/export_semantic_anchor_end_to_end.py`

Behavior:

1. configured execution
2. sequential CUDA with `max_batch_size=1`
3. sequential CPU fallback with `max_batch_size=1`

Generated `summary.json` files now include:

- `generation_execution.attempt`
- `generation_execution.attempt_name`
- `generation_execution.device`
- `generation_execution.execution_kwargs`
- `generation_execution.oom_retry_count`

GPU-safe training launcher:

- `scripts/run_parallel_training_suite_2026_04_17.ps1`

It now:

- schedules one heavy training job per visible GPU
- sets `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`
- avoids the earlier oversubscribed launcher behavior that caused VRAM pressure

## Latest Current-Code Evidence

Manual compare:

- `outputs/zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1/protocol_manual_compare_statefulmultistep_v22`

Fixed-graph 3-seed audit:

- `outputs/zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1/protocol_ablation_statefulmultistep_v22`

Baseline comparison:

- `outputs/zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1/protocol_ablation_statefulmultistep_v22/baseline_comparison/protocol_vs_baselines.json`
- `outputs/zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1/protocol_ablation_statefulmultistep_v22/baseline_comparison/protocol_vs_baselines.md`

Key `v22` aggregate numbers:

- `diffusion`:
  repair `0.8611`, overwrite `0.2500`, pre-anchor `6.75`,
  hybrid contract `1.0`, P-CBS success `0.6667`, time `92.92s`
- `fast sampler`:
  repair `0.8611`, overwrite `0.2222`, pre-anchor `6.00`,
  hybrid contract `1.0`, P-CBS success `0.6667`, time `92.30s`
- `masked room`:
  repair `0.6944`, overwrite `0.2778`, pre-anchor `7.50`,
  hybrid contract `1.0`, P-CBS success `0.3333`, time `54.44s`

Current claim status:

- `can_claim_surpasses_publications = false`

Reason:

- fixed-graph room-generation evidence and matched-budget topology-generation
  baselines are different evidence layers
- external baseline rows are still mixed rather than dominant across all
  problems

## Reporting Fixes Applied On 2026-04-18

These issues were fixed:

- stateful puzzle hyperparameter sweep no longer produces `-Infinity` scores
  from non-finite confusion ratios
- fixed-graph audit now exports strict JSON instead of `NaN` / `Infinity`
- baseline comparison now preserves undefined values as `null / n/a`, not fake
  `0.0`
- verdict doc was updated to reflect the actual `v22` evidence rather than
  stale `v8` wording

Files changed in the latest pass:

- `scripts/run_stateful_puzzle_hparam_sweep.py`
- `scripts/run_fixed_graph_multi_seed_audit.py`
- `scripts/compare_protocol_to_baselines.py`
- `scripts/run_fast_sampler_visual_audit.py`
- `scripts/export_semantic_anchor_end_to_end.py`
- `scripts/run_parallel_training_suite_2026_04_17.ps1`
- `tests/test_protocol_reporting.py`
- `docs/FINAL_ABLATION_AND_ARCHITECTURE_VERDICT_2026_04_17.md`

## Current Sweep Result

Stateful puzzle runtime sweep:

- `results/stateful_puzzle_hparam_sweep_v2/summary.json`
- `results/stateful_puzzle_hparam_sweep_v2/report.md`

Current ranking:

- `baseline_default`: `116.015`
- `route_safe_stateful`: `94.604`
- `no_puzzle_control`: `88.283`

Interpretation:

- keep the current baseline puzzle grammar as default
- `route_safe_stateful` is a valid ablation, not the new default

## Tests Already Run

Latest checks passed:

- `python -m py_compile scripts\run_fixed_graph_multi_seed_audit.py scripts\compare_protocol_to_baselines.py`
- `python -m pytest tests\test_protocol_reporting.py -q` -> `5 passed`

Earlier in the same patch line:

- targeted protocol / validation / search checks passed
- strict JSON outputs were regenerated successfully

## What Still Remains

Main remaining work is evidence-side, not another large code rewrite:

1. rerun matched-budget external baselines on the final latest-code path if a
   completely fresh external table is required
2. rerun the full long `1..9 x variants x personas` P-CBS benchmark on the
   patched solver stack if the thesis needs full persona tables
3. if stronger puzzle claims are needed, extend staged mechanics to richer
   multi-object dependencies such as chained switches, explicit movable-object
   identity, and multi-room persistent puzzle state

## Suggested First Actions In A New Chat

1. open:
   `docs/FINAL_ABLATION_AND_ARCHITECTURE_VERDICT_2026_04_17.md`
2. open:
   `outputs/zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1/protocol_ablation_statefulmultistep_v22/summary.json`
3. open:
   `outputs/zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1/protocol_ablation_statefulmultistep_v22/baseline_comparison/protocol_vs_baselines.md`
4. preserve the current claim boundary:
   thesis-finalizable, but not yet publication-surpassing
