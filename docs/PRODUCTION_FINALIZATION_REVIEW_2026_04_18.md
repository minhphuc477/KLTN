# Production Finalization Review

Last updated: 2026-04-18

This note is the current end-to-end review of the production path:

- topology generation
- tokenizer / room backbone
- symbolic repair
- stateful puzzle module
- hard validation
- P-CBS

It answers one practical question:

- `can this be finalized now, and if not, what is the shortest honest path to finish?`

## Executive Verdict

`Thesis-finalizable: yes.`

`Production-finalizable with disciplined claims: yes.`

`Publication-surpassing claim: no.`

The current stack is strong enough to finalize as a `graph-first hybrid
neuro-symbolic dungeon generator` if you keep the claim boundary correct:

- final room backbone: `diffusion`
- strongest tokenizer evidence: `codebook512`
- puzzle semantics: `hybrid stateful multi-step grammar`
- hard correctness contract:
  - `graph_guided_oracle`
  - `graph_progression`
  - `softlock_check`
  - stitched tile-state `A*` as a stricter stress probe
- behavioral validator: `P-CBS`

Do **not** finalize around these claims:

- monolithic stitched tile-state `A*` alone proves the dungeons end to end
- the network has already learned staged puzzle semantics without hybrid grammar
- the repo already surpasses prior publications

## What Is Sound Right Now

### 1. Core architecture

Still sound:

- mission/topology graph remains necessary
- VQ-VAE remains necessary
- diffusion remains the safest room branch
- symbolic repair remains necessary
- deterministic graph marker overlay remains necessary
- hybrid validation remains necessary

The existing ablations still support the hybrid stack over pure-neural or
pure-symbolic simplifications.

### 2. Puzzle module

Current puzzle semantics are no longer decorative only.

The system now supports:

- local interaction grammars
- staged room-level puzzle plans
- ordered `collect_key -> collect_item -> step_on_puzzle -> DOOR_PUZZLE`
- `push_block_to_switch` unlocking

That is sufficient for a thesis-safe claim of `meaningful hybrid stateful puzzle
mechanics`.

It is **not** sufficient for the stronger claim of a fully learned
multi-object Sokoban-like puzzle generator.

### 3. Search / validation

The correct solver hierarchy is:

- hard oracle:
  - `graph_guided_oracle`
  - `A*` / hybrid A* tile-state probe
  - `Dijkstra` exact fallback
  - `graph_progression`
  - `softlock_check`
- replanning probe:
  - `D* Lite`
- behavioral probe:
  - `P-CBS`

`D* Lite` should not be promoted to the primary static correctness oracle for
this repo.

### 4. P-CBS

Current novelty boundary is still:

- `P-CBS` is a bounded-rational persona validator integrated into dungeon PCG
- `P-CBS` is **not** a new universal shortest-path family
- `P-CBS` is **not** the hard oracle

That claim remains sound.

## Real Bugs / Logic Gaps Fixed In This Review

### 1. Graph-role alias bug in graph-guided validation

Bug:

- topology-generated graphs use `type=START` / `type=GOAL`
- graph-guided validation previously only recognized
  `is_start=True` / `has_triforce=True`
- this caused valid topology-generated dungeons to fail with
  `No START or TRIFORCE node found in graph`

Fix:

- `src/simulation/validator.py` now accepts both the old boolean flags and the
  repo's typed mission-graph schema

Impact:

- topology-generated evaluation is now aligned with the repo's own graph schema

### 2. Non-strict JSON artifact bug

Bug:

- export artifacts could still serialize `Infinity` in
  `cbs_balanced.confusion_ratio_vs_astar`
- that made some summary files non-strict JSON

Fix:

- strict recursive JSON sanitization now applies in:
  - `scripts/run_fast_sampler_visual_audit.py`
  - `scripts/export_semantic_anchor_end_to_end.py`
  - `scripts/run_stateful_puzzle_hparam_sweep.py`

Impact:

- future artifacts are valid JSON instead of Python-style non-finite dumps

## Current Blocking Gaps

These are the remaining blockers that still matter.

### 1. Manual compare should be rerun once after the validator/reporting fixes

Reason:

- the current `protocol_manual_compare_statefulmultistep_v23` artifacts were
  generated before the graph-role alias fix and strict-JSON patch

Action:

```powershell
python scripts\export_semantic_anchor_end_to_end.py `
  --run-dir outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1 `
  --output-dir outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_statefulmultistep_v24 `
  --seed 20260404
```

### 2. External baseline evidence is still not synchronized to the latest runtime patch layer

Reason:

- the fixed-graph branch evidence is current enough
- the matched-budget / external comparison layer is still mixed and not
  sufficiently dominant for a publication-surpassing claim

Action:

```powershell
python scripts\run_matched_budget_topology_benchmark.py `
  --output results\matched_budget_topology_v2 `
  --samples-per-method 64 `
  --seed 42

python scripts\run_pcg_benchmark_alignment.py `
  --output-dir results\pcg_benchmark_alignment_v3 `
  --seed 42
```

### 3. Long-form latest-code P-CBS persona table is still missing

Reason:

- the thesis claim boundary is supported
- the publication-grade long-form persona table is still not fully refreshed

Action:

```powershell
python scripts\run_cbs_benchmarks.py `
  --levels 1,2,3,4,5,6,7,8,9 `
  --variants 1,2 `
  --all-personas `
  --timeout-astar 200000 `
  --timeout-cbs 50000 `
  --seed 42 `
  --output results\cbs_benchmark_levels1_9_variants12_all_personas_v_latest.csv
```

### 4. Puzzle cookbook is still not strong enough for a full training-selection claim

Reason:

- runtime puzzle grammar is good enough now
- training-time puzzle cookbook evidence is still incomplete

Action only if you need the stronger claim:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_parallel_training_suite_2026_04_17.ps1 -Phase puzzle-cookbook-diffusion -GpuIds 0,1
powershell -ExecutionPolicy Bypass -File scripts\run_parallel_training_suite_2026_04_17.ps1 -Phase puzzle-cookbook-aux -GpuIds 0,1
```

If you do **not** need the stronger train-time puzzle-learning claim, do not
wait on this before finalizing the thesis.

## Minimal Finish Plan

If the goal is to finish without wasting time, do only this:

1. rerun manual compare after the validator/reporting fixes
2. keep `diffusion + codebook512 + hybrid contract` as the final path
3. cite the latest fixed-graph `v23` branch evidence
4. rerun the long-form persona benchmark only if you need the final table
5. do **not** block finalization on a full puzzle-cookbook retrain unless you
   want to claim learned puzzle semantics

## Final Honest Answer

This repo is close enough to finalize, but only as a `hybrid graph-first`
system with conservative claims.

It is not yet honest to say:

- the entire pipeline is solved by a single monolithic tile-state oracle
- puzzle semantics are fully learned end to end
- the model already beats prior publications across the board

It is honest to say:

- the production stack is coherent
- the puzzle module is now meaningful and stateful in hybrid form
- the hard validation contract is sound for the current architecture
- `P-CBS` adds bounded-rational behavioral evidence beyond exact solvability
