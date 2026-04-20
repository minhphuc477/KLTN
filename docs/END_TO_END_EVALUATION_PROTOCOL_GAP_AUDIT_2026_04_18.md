# End-to-End Evaluation Protocol Gap Audit

Last updated: 2026-04-18

This note maps the literature-style PCG evaluation checklist onto the current
repo. The goal is not to claim that every desirable metric already exists. The
goal is to state, precisely, what is covered now, what is partially covered,
and what still needs new experiments.

## 1. Research Basis

This audit follows the evaluation taxonomy and cautionary guidance in:

- Withington, Cook, Tokarchuk, *On the Evaluation of Procedural Level
  Generation Systems* (FDG 2024 / arXiv 2024):
  <https://arxiv.org/abs/2404.18657>
- Mariño, Reis, Lelis, *An Empirical Evaluation of Evaluation Metrics of
  Procedurally Generated Mario Levels* (AIIDE 2015):
  <https://webdocs.cs.ualberta.ca/~santanad/papers/2015/marinoRL15.pdf>
- Schaa, Barriga, *Evaluating the Expressive Range of Super Mario Bros Level
  Generators* (Algorithms 2024):
  <https://www.mdpi.com/1999-4893/17/7/307>
- Horn et al., *A Comparative Evaluation of Procedural Level Generators in the
  Mario AI Framework* (FDG 2014):
  <https://www.fdg2014.org/papers/fdg2014_paper_14.pdf>
- Yuan et al., *MOPCGRL: Multi-Objective Procedural Content Generation via
  Reinforcement Learning* (CSMS 2026):
  <https://doi.org/10.23919/CSMS.2025.0034>

The practical lesson from those papers is consistent:

1. do not rely on a single metric
2. separate topology-level and final-level evidence
3. keep solver-based playability, diversity, novelty, and compute reporting all
   visible
4. do not confuse computational metrics with human-perceived quality

## 2. What The Repo Already Covers Well

### 2.1 Playability / solvability

Status: `strong`

Implemented evidence:

- hybrid mechanical contract in protocol exports
  - `graph_guided_oracle`
  - `graph_progression`
  - `softlock_check`
  - `mechanical_contract.hybrid_oracle_pass`
- stitched-tile `A*` stress probe
- comparison solvers in `validation_search_stats.json`
- `P-CBS` behavioral probe and persona metrics

Main code:

- `scripts/run_fast_sampler_visual_audit.py`
- `scripts/run_fixed_graph_multi_seed_audit.py`
- `src/evaluation/pcbs_validation.py`
- `src/simulation/*`

Judgment:

- The repo has a real mechanical playability contract.
- The honest report-facing pass/fail gate is the hybrid oracle stack, not
  monolithic stitched `A*` alone.

### 2.2 Topology-side diversity, novelty, expressive range, controllability

Status: `strong`

Implemented evidence:

- `linearity`
- `leniency`
- `progression_complexity`
- `topology_complexity`
- `novelty_vs_reference`
- descriptor-space coverage / expressive overlap
- fidelity JS divergence
- matched-budget topology comparisons
- controllability through descriptor targets and target-tracking summaries

Main code:

- `src/evaluation/benchmark_suite.py`
- `scripts/run_matched_budget_topology_benchmark.py`

Judgment:

- The topology generator is not under-measured.
- The main weakness was the final exported dungeon layer, not Block I.

### 2.3 Reproducibility

Status: `strong`

Implemented evidence:

- resolved config snapshots
- stable checkpoint metadata
- artifact / checkpoint status note
- exact training and evaluation commands in docs
- seed-based protocols and aggregate summaries

Main docs:

- `docs/ARTIFACT_AND_CHECKPOINT_STATUS_2026_04_18.md`
- `docs/FINAL_ABLATION_AND_ARCHITECTURE_VERDICT_2026_04_17.md`
- `docs/NEXT_CHAT_CONTEXT_2026_04_18.md`

## 3. What Was Missing And Is Now Implemented

The literature you pasted exposed a real gap:

- the repo already had graph-level novelty/diversity
- but the exported stitched dungeons did not expose a report-facing structural
  diversity / novelty metric comparable to compression-distance style measures

That gap is now patched.

### 3.1 New end-to-end structural diversity / novelty layer

Added code:

- `src/evaluation/end_to_end_level_metrics.py`
- `scripts/run_fast_sampler_visual_audit.py`
- `scripts/export_semantic_anchor_end_to_end.py`
- `scripts/run_fixed_graph_multi_seed_audit.py`
- `scripts/compare_protocol_to_baselines.py`

New per-export fields:

- `end_to_end_evaluation.room_unique_ratio`
- `end_to_end_evaluation.room_pairwise_ncd`
- `end_to_end_evaluation.room_nearest_reference_ncd`
- `end_to_end_evaluation.room_symbol_entropy_mean`
- `end_to_end_evaluation.dungeon_symbol_entropy_non_void`

New fixed-graph aggregate fields:

- `avg_room_unique_ratio`
- `avg_room_pairwise_ncd_mean`
- `avg_room_nearest_reference_ncd_mean`
- `avg_room_symbol_entropy_mean`
- `avg_dungeon_symbol_entropy_non_void`

Why this patch matters:

- it gives the final room-generation pipeline a structural diversity metric
  closer to the NCD / compression-distance family used in PCG evaluation papers
- it makes end-to-end novelty measurable against reference room content instead
  of only against graph descriptors

### 3.2 Quick backfill on the current canonical `v22` artifacts

I backfilled the new metrics once on the existing canonical
`protocol_ablation_statefulmultistep_v22` room text artifacts without rerunning
generation.

Observed on the 3-seed canonical slice:

| Variant | Mean room pairwise NCD | Mean nearest-reference room NCD | Mean room symbol entropy | Note |
|---|---:|---:|---:|---|
| diffusion | `0.6376` | `0.6184` | `1.2903` | highest internal room diversity of the three |
| fast sampler | `0.6337` | `0.6163` | `1.2576` | close to diffusion on structural spread |
| masked room | `0.6171` | `0.5818` | `1.2346` | closest to reference room corpus, but less internally varied |

All three variants had `room_unique_ratio = 1.0` on that slice.

Interpretation:

- the new metric is not degenerate
- it surfaces a real tradeoff between internal room variety and reference
  proximity
- the current canonical artifacts are still usable for qualitative judgment,
  but they should be regenerated if the report wants these fields stored in the
  official JSON summaries

## 4. Coverage Matrix

| Dimension | Current repo coverage | Status | Evidence / note |
|---|---|---|---|
| Playability / solvability | Hybrid mechanical contract + `A*` stress probe + solver suite + `P-CBS` | `strong` | Implemented and already used in protocol exports |
| Diversity (topology) | Descriptor coverage, expressive range proxies, novelty vs reference | `strong` | `benchmark_suite.py`, matched-budget benchmark |
| Diversity (end-to-end final rooms) | Room uniqueness + pairwise NCD + room symbol entropy | `implemented, rerun pending` | Code exists now, but latest canonical artifacts predate this patch |
| Novelty (topology) | `novelty_vs_reference`, graph edit distance proxies | `strong` | Already in topology benchmarks |
| Novelty (end-to-end final rooms) | Nearest-reference room NCD | `implemented, rerun pending` | Code exists now, needs latest protocol rerun |
| Difficulty | Topology leniency/linearity + solver difficulty proxies + `P-CBS` confusion/load | `partial but useful` | Good computational coverage, weak human calibration |
| Aesthetics / visual quality | Visual sheets and optional blinded-eval scaffolding | `weak` | No completed validated human study on latest branch |
| Controllability | Strong on topology; partial on room semantics | `partial to strong` | Topology targets are explicit; room-level target sweeps are less complete |
| Generalization / OOD | Protocol/docs exist; latest-code full rerun incomplete | `partial` | Not absent, but not fully refreshed on current branch |
| Sample efficiency | Training logs/checkpoints exist | `partial` | No consolidated cross-branch sample-efficiency table yet |
| Compute cost | Generation time reported; parameter-count warnings exist | `partial` | No single GPU-hour / wall-clock comparison table across all cookbook runs |
| Statistical significance | Present for some ablations | `partial` | Stronger on core ablation study than on final protocol exports |
| Reproducibility | Config snapshots, seed protocols, command docs | `strong` | Good enough for thesis-grade reporting |

## 5. Honest Remaining Gaps

These are the real remaining gaps after the code patch above.

### 5.1 Latest-code reruns still needed

Status: `not code-blocked`

Still needed:

- rerun the canonical fixed-graph protocol on the current branch so the new
  end-to-end metrics appear in the canonical export artifacts
- rerun the long patched `1..9 x variants x personas` `P-CBS` benchmark if the
  thesis needs final persona tables on current code

Why:

- the new metric layer is implemented now
- older protocol artifacts do not automatically contain it

### 5.2 Human-perceived quality is still under-evidenced

Status: `still missing`

What is missing:

- validated questionnaire study
- blinded expert review packet on the latest branch
- stronger aesthetics / coherence evidence beyond visual inspection

Why:

- Mariño et al. explicitly warn that computational metrics should not replace
  user studies for player-perceived quality
- this repo currently has qualitative sheets and some blinded-eval scaffolding,
  but not a completed latest-code human study

### 5.3 Sample-efficiency reporting is still incomplete

Status: `still missing`

What is missing:

- one consolidated table for VQ-VAE cookbook runs
- one consolidated table for downstream branch retrains
- wall-clock / checkpoint-quality curves summarized in one report-facing note

Why:

- the raw evidence exists in checkpoints and logs
- the synthesis does not yet exist in one canonical table

### 5.4 Puzzle cookbook evidence is still incomplete

Status: `still missing`

What is missing:

- complete downstream retrain sweep for the stateful puzzle cookbook family
- final best-setting selection backed by full downstream evidence, not only
  partial or interrupted runs

Why:

- the runtime/stateful puzzle grammar is much stronger now
- but a strong "best train-time puzzle recipe" claim still needs the full
  cookbook training matrix completed

## 6. What This Means For The Current Architecture Claim

Current honest claim:

- the architecture is thesis-finalizable
- the playability and structural reporting stack is materially stronger than it
  was before this patch
- the repo now measures both topology-level and final-room structural diversity
  more defensibly

Current non-defensible claim:

- "the system has already completed every evaluation dimension recommended by
  the PCG literature"
- "computational metrics alone prove aesthetics / player experience"
- "the current latest-code branch already surpasses prior publications"

## 7. Recommended Next Runs

### 7.1 Refresh the canonical fixed-graph protocol with the new metric layer

```powershell
python scripts\run_fixed_graph_multi_seed_audit.py `
  --run-dir outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1 `
  --output-dir outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_ablation_statefulmultistep_v23 `
  --seeds 20260404 20260405 20260406
```

### 7.2 Regenerate the manual compare sheet on the same branch

```powershell
python scripts\export_semantic_anchor_end_to_end.py `
  --run-dir outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1 `
  --output-dir outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_statefulmultistep_v23 `
  --seed 20260404
```

### 7.3 If final thesis tables need latest-code persona evidence

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

## 8. Bottom Line

Before this patch, the biggest evaluation gap was not "lack of metrics in
general". It was the absence of a clear final-level structural diversity /
novelty layer for the stitched dungeon exports.

That gap is now implemented.

What remains is mostly evidence refresh and human-evaluation depth, not another
large architecture rewrite.
