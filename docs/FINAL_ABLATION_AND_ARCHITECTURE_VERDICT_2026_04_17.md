# Final Ablation And Architecture Verdict

Last updated: 2026-04-18

## Executive Verdict

The repo is close to a final thesis-grade system, but it is not honest to claim
that it already surpasses prior publications.

The safest final production recommendation is still:

- run directory:
  `outputs/zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1`
- tokenizer:
  `codebook512`
- room backbone:
  `diffusion`
- runtime strategy:
  `graph-first hybrid` with symbolic repair, deterministic marker overlay, and
  hard validation by the hybrid mechanical contract
- experiential validator:
  `P-CBS`

What is good enough now:

- VQ-VAE is not the main bottleneck.
- `codebook512` remains the strongest tokenizer variant tested.
- evolution, symbolic repair, and logic/topology guidance are still necessary.
- the current hybrid stack is materially stronger than pure neural or pure
  symbolic simplifications.
- puzzle scaffolding now enforces local interaction grammars instead of only
  obstacle density, so empty or decorative stateful rooms are penalized while
  readable push/bypass/alcove layouts are favored.
- puzzle scaffolding now also supports simple multi-anchor interaction
  sequences, so complex puzzle rooms are no longer scored only on local pocket
  geometry; staged route coverage across multiple anchors is measured and can
  veto bad candidates.
- puzzle validation now uses a hybrid mechanical contract:
  `graph-guided oracle + graph progression + soft-lock`, while the monolithic
  tile-state A* run is kept as a stricter stress probe rather than the only
  report-facing pass/fail signal.
- P-CBS is now defensible as a bounded-rational persona validator, not just a
  weighted shortest-path controller.

What is still not fully solved:

- puzzle generation now has validator-backed staged mechanics for ordered
  `key -> item -> puzzle/switch -> DOOR_PUZZLE` plans and push-block switch
  unlocks, but it is still not full Sokoban-grade multi-object state search.
- masked-room is promising, but not yet strong enough to replace diffusion as
  the default room backbone for the whole thesis.
- the repo still lacks a fully current matched-budget external baseline table
  that would justify "better than other publications" language.

## Current Patched-Protocol Status

The latest code-path evidence after the stateful multi-step puzzle upgrade is:

- manual compare:
  `outputs/zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1/protocol_manual_compare_statefulmultistep_v22`
- fixed-graph 3-seed audit:
  `outputs/zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1/protocol_ablation_statefulmultistep_v22`

What that current evidence says:

- diffusion is still the safest canonical thesis branch
- fast-sampler is now competitive with diffusion on the current `v22`
  fixed-graph slice and slightly better on overwrite / pre-overlay anchor
  error, but it is not a clean overall takeover
- masked-room remains an auxiliary branch: faster, but clearly weaker on repair
  rate and bounded-rational success on the current aggregate

Current `v22` headline numbers:

- manual compare:
  - `diffusion`: `99.49s`, repair `0.917`, overwrite `0.250`, hybrid contract `pass`, `P-CBS budget exhausted`
  - `fast-sampler`: `88.68s`, repair `0.917`, overwrite `0.333`, hybrid contract `pass`, `P-CBS budget exhausted`
  - `masked-room`: `54.97s`, repair `0.583`, overwrite `0.167`, hybrid contract `pass`, `P-CBS budget exhausted`
- fixed-graph 3-seed aggregate:
  - `diffusion`: repair `0.861`, overwrite `0.250`, pre-overlay anchor error `6.75`, hybrid contract `1.000`, `P-CBS success_rate=0.667`, time `92.92s`
  - `fast-sampler`: repair `0.861`, overwrite `0.222`, pre-overlay anchor error `6.00`, hybrid contract `1.000`, `P-CBS success_rate=0.667`, time `92.30s`
  - `masked-room`: repair `0.694`, overwrite `0.278`, pre-overlay anchor error `7.50`, hybrid contract `1.000`, `P-CBS success_rate=0.333`, time `54.44s`

The correct reading is not that masked-room or fast-sampler have cleanly
overtaken diffusion. The correct reading is:

- hard playability remains solved for all three branches on the current fixed
  graph slice under the hybrid mechanical contract
- puzzle semantics are more coherent than before because the new interaction
  and sequence gates are active in real exports
- monolithic stitched tile-state `A*` now times out on this stateful puzzle
  slice for every branch, so it should be treated as a stricter stress probe,
  not the only report-facing hard oracle
- branch ranking is still mixed enough that diffusion remains the safest
  canonical backbone for the thesis, while fast-sampler is now competitive on
  repair and overwrite but not clearly dominant
- the latest baseline-comparison report still sets
  `can_claim_surpasses_publications = false`, because fixed-graph room
  generation and matched-budget topology generation remain different evidence
  layers and the external rows are still mixed rather than dominant

## Final Architecture Recommendation

Use the current architecture as:

1. `MAP-Elites` mission DAG generation
2. graph-conditioned `VQ-VAE + latent diffusion` room generation
3. symbolic repair / WFC-style refinement
4. puzzle grammar + interaction-grammar cleanup
5. deterministic semantic marker overlay
6. room stitching
7. hard validation:
   - hybrid mechanical contract:
     - graph-guided room oracle
     - graph progression validator
     - deterministic soft-lock checker
   - monolithic tile-state `A*` as a stricter timeout-prone stress probe
8. behavioral validation:
   - `P-CBS`

Do not switch to fully neural semantics as the canonical thesis path. The
repo's own ablations still support the hybrid contract.

## What The Ablations Already Prove

### VQ-VAE

Existing tokenizer ablations and audits support:

- keep the current VQ-VAE family
- keep `latent_dim=64`
- prefer `codebook_size=512` over the smaller alternatives already tested
- do not remove `CoordConv`
- do not remove the light local-structure penalty without evidence

Current evidence base:

- `outputs/vqvae_audit_baseline_v1`
- `outputs/vqvae_ablation_codebook128_v1`
- `outputs/vqvae_ablation_codebook512_v1`
- `outputs/vqvae_ablation_hidden64_v1`
- `outputs/vqvae_ablation_no_coordconv_v1`
- `outputs/vqvae_ablation_no_mrf_v1`
- [`VQVAE_RESEARCH_AUDIT_2026_04_10.md`](./VQVAE_RESEARCH_AUDIT_2026_04_10.md)

Verdict:

- VQ-VAE does not require a new architecture first.
- The main remaining bottlenecks are downstream semantics, puzzle structure,
  and full-stack validation evidence.

### Core Architecture

Quick matched-budget ablation results:

- `FULL` and `TOPO_LIGHTWEIGHT` are nearly tied on the current quick slice.
- `NO_EVOLUTION` is materially worse on topology preservation and repair cost.
- `RANDOM_TOPOLOGY` degrades controllability and representability.
- `NO_WFC`, `NO_LOGIC`, and `PURE_WFC` are not credible replacements for the
  full hybrid path.

Current artifacts:

- `results/ablation_core_quick_v3`
- `results/ablation_core_quick_part2_v1`

Verdict:

- evolution is necessary
- symbolic repair is necessary
- logic / topology guidance is necessary
- lightweight topology refinement is not yet proven necessary over the current
  full setting

### Room Branches

Current matched-budget room-branch evidence:

- latent diffusion is faster and remains the safest default
- masked-room can look cleaner on some fixed-graph slices
- reference-room maps are not a decisive win in the current quick benchmark

Current artifact:

- `results/room_branch_benchmark_quick_v2`

Verdict:

- keep diffusion as the canonical room branch
- treat masked-room as an auxiliary branch and report it honestly as such

### Stateful Puzzle Module

Current runtime puzzle-profile sweep:

- `results/stateful_puzzle_hparam_sweep_v2`

Current ranking:

- `baseline_default`: `116.015`
- `route_safe_stateful`: `94.604`
- `no_puzzle_control`: `88.283`

Verdict:

- the current baseline puzzle grammar is better than removing puzzle structure
- the stricter route-safe runtime overrides are defensible as an ablation, but
  they are not the new default
- the meaningful gain is no longer "more blocks"; it is better interaction and
  staged-sequence validity under the hybrid oracle contract

### P-CBS

Current P-CBS evidence is now good enough to support a thesis claim with a
carefully bounded novelty statement.

What is already true:

- persona-driven automated playtesting is not new
- procedural personas are not new
- cognitive pathfinding and resource-rational planning are not new

What is defensible here:

- the repo combines explicit bounded-cognition penalties, affordance memory,
  metacognitive deliberation budget, frustration, focus persistence, and
  dungeon-PCG-loop integration in a single heuristic search validator

Current evidence base:

- `results/pcbs_component_ablation_balanced_l123_v3`
- `results/pcbs_component_ablation_explorer_l123_v3`
- `results/cbs_benchmark_levels123_all_personas_v5.csv`
- `results/cbs_benchmark_levels123_all_personas_v5_summary.json`
- [`PCBS_REVIEWER2_NOVELTY_AUDIT_2026_04_17.md`](./PCBS_REVIEWER2_NOVELTY_AUDIT_2026_04_17.md)
- [`PCBS_AFFORDANCE_MEMORY_AND_ABLATION_2026_04_17.md`](./PCBS_AFFORDANCE_MEMORY_AND_ABLATION_2026_04_17.md)

Verdict:

- do not claim P-CBS is a new universal search family
- do claim it is a repo-novel bounded-rational persona validator integrated
  into dungeon PCG evaluation
- the strongest personas on the current `levels 1..3` slice are `novice` and
  `explorer`, not `balanced`

## Search And Validation Contract

The canonical correctness contract should remain:

- hybrid mechanical contract:
  - graph-guided room oracle
  - graph progression validator
  - deterministic soft-lock checker
- monolithic tile-state `A*` as a stricter secondary oracle / stress probe

Use these only as comparison or behavioral tools:

- `BFS`
- `Dijkstra`
- `Greedy`
- `D* Lite`
- bidirectional A*
- `P-CBS`

Reason:

- `P-CBS` is intentionally bounded and persona-dependent
- monolithic tile-state `A*` times out on the current stitched stateful-puzzle
  dungeons often enough that it is better treated as a stress probe than as the
  only report-facing hard validator
- the graph-guided oracle is the architecture-consistent exact check for the
  graph-first hybrid stack because it validates the actual stitched room path
  induced by the mission DAG
- `D* Lite` and bidirectional A* are not currently stronger independent static
  dungeon oracles in this repo than the hybrid oracle stack

## Final Model Quality Judgment

Is the current architecture final enough to write the thesis around?

Yes, with disciplined claims.

Use these thesis-safe claims:

- the final system is a graph-first neuro-symbolic hybrid dungeon generator
- the hybrid handoff is necessary because pure-neural and pure-symbolic
  simplifications underperform
- the current tokenizer choice is justified by existing ablations
- P-CBS adds experiential validation beyond exact solvability
- puzzle grammar anchors improve readability and semantic structure

Do not use these claims:

- the system already surpasses prior publications
- puzzle rooms are now fully meaningful multi-step puzzles
- P-CBS replaces A* as the hard oracle
- masked-room has already replaced diffusion as the canonical backbone

## Remaining High-Value Work

These are the real remaining gaps:

1. finish the full latest-code strict multi-seed protocol matrix
2. rerun matched-budget external baseline tables on the final code path
3. rerun the patched long-form `1..9 x variants x personas` P-CBS benchmark
4. if the thesis needs stronger qualitative claims than the current staged
   `key/item/puzzle/switch -> DOOR_PUZZLE` state machine, extend the grammar
   to richer multi-object mechanics such as chained switches, explicit movable
   object identity, and multi-room state dependencies

One operational gap is now fixed as well:

- `scripts/run_parallel_training_suite_2026_04_17.ps1` is now GPU-aware and
  OOM-safer. It schedules one heavy training job per visible GPU, sets
  `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`, and avoids the old
  oversubscribed `Start-Process ... -NoExit` pattern that could leave windows
  open and keep VRAM reserved unnecessarily.

One protocol gap is now fixed in code: `scripts/run_fixed_graph_multi_seed_audit.py`
supports `--reuse-existing-seed-summaries` and `--aggregate-only`, so long
multi-seed audits can be resumed and aggregated after a wall-clock interruption
instead of re-running every seed export.

If you do not finish those, the thesis is still viable, but the claim ceiling
must stay conservative.

## Exact Training Commands

Use fresh `v2` output folders for clean reruns. Do not reuse old folders unless
you explicitly want resume behavior.

### 1. VQ-VAE Ablations

Run these in parallel if you have the memory budget.

Baseline / canonical-width tokenizer:

```powershell
python -m src.train_vqvae `
  --config configs\zelda_hmolqd.yaml `
  --save-dir outputs\vqvae_audit_baseline_v2\checkpoints\vqvae `
  --data-dir "Data\The Legend of Zelda" `
  --epochs 300 `
  --hidden-dim 96 `
  --latent-dim 64 `
  --codebook-size 256 `
  --use-coordconv `
  --mrf-penalty-weight 0.05 `
  --validation-fraction 0.1 `
  --validation-max-batches 16 `
  --best-checkpoint-metric val_loss `
  --seed 42
```

Codebook `128`:

```powershell
python -m src.train_vqvae `
  --config configs\zelda_hmolqd.yaml `
  --save-dir outputs\vqvae_ablation_codebook128_v2\checkpoints\vqvae `
  --data-dir "Data\The Legend of Zelda" `
  --epochs 300 `
  --hidden-dim 96 `
  --latent-dim 64 `
  --codebook-size 128 `
  --use-coordconv `
  --mrf-penalty-weight 0.05 `
  --validation-fraction 0.1 `
  --validation-max-batches 16 `
  --best-checkpoint-metric val_loss `
  --seed 42
```

Codebook `512`:

```powershell
python -m src.train_vqvae `
  --config configs\zelda_hmolqd.yaml `
  --save-dir outputs\vqvae_ablation_codebook512_v2\checkpoints\vqvae `
  --data-dir "Data\The Legend of Zelda" `
  --epochs 300 `
  --hidden-dim 96 `
  --latent-dim 64 `
  --codebook-size 512 `
  --use-coordconv `
  --mrf-penalty-weight 0.05 `
  --validation-fraction 0.1 `
  --validation-max-batches 16 `
  --best-checkpoint-metric val_loss `
  --seed 42
```

Hidden `64`:

```powershell
python -m src.train_vqvae `
  --config configs\zelda_hmolqd.yaml `
  --save-dir outputs\vqvae_ablation_hidden64_v2\checkpoints\vqvae `
  --data-dir "Data\The Legend of Zelda" `
  --epochs 300 `
  --hidden-dim 64 `
  --latent-dim 64 `
  --codebook-size 256 `
  --use-coordconv `
  --mrf-penalty-weight 0.05 `
  --validation-fraction 0.1 `
  --validation-max-batches 16 `
  --best-checkpoint-metric val_loss `
  --seed 42
```

No `CoordConv`:

```powershell
python -m src.train_vqvae `
  --config configs\zelda_hmolqd.yaml `
  --save-dir outputs\vqvae_ablation_no_coordconv_v2\checkpoints\vqvae `
  --data-dir "Data\The Legend of Zelda" `
  --epochs 300 `
  --hidden-dim 96 `
  --latent-dim 64 `
  --codebook-size 256 `
  --no-use-coordconv `
  --mrf-penalty-weight 0.05 `
  --validation-fraction 0.1 `
  --validation-max-batches 16 `
  --best-checkpoint-metric val_loss `
  --seed 42
```

No local-structure penalty:

```powershell
python -m src.train_vqvae `
  --config configs\zelda_hmolqd.yaml `
  --save-dir outputs\vqvae_ablation_no_mrf_v2\checkpoints\vqvae `
  --data-dir "Data\The Legend of Zelda" `
  --epochs 300 `
  --hidden-dim 96 `
  --latent-dim 64 `
  --codebook-size 256 `
  --use-coordconv `
  --mrf-penalty-weight 0.0 `
  --validation-fraction 0.1 `
  --validation-max-batches 16 `
  --best-checkpoint-metric val_loss `
  --seed 42
```

### 2. Canonical Downstream Branch (`codebook512_puzzle_subtype`)

Diffusion teacher:

```powershell
python main.py train `
  --config configs\zelda_hmolqd.yaml `
  --stage diffusion `
  --output-dir outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v2 `
  --diffusion-vqvae-checkpoint outputs\vqvae_ablation_codebook512_v2\checkpoints\vqvae\vqvae_pretrained.pth `
  --seed 42 `
  --no-auto-resume `
  --verbose
```

Fast sampler:

```powershell
python main.py train `
  --config configs\zelda_hmolqd.yaml `
  --stage fast_sampler `
  --output-dir outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v2 `
  --fast-sampler-base-diffusion-checkpoint outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v2\checkpoints\diffusion\best_model.pth `
  --seed 42 `
  --no-auto-resume `
  --verbose
```

Masked-room:

```powershell
python main.py train `
  --config configs\zelda_hmolqd.yaml `
  --stage masked_room `
  --output-dir outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v2 `
  --seed 42 `
  --no-auto-resume `
  --verbose
```

### 3. Puzzle-Structure-Controlled Downstream Branch

Diffusion teacher:

```powershell
python main.py train `
  --config configs\zelda_hmolqd.yaml `
  --stage diffusion `
  --output-dir outputs\zelda_hmolqd_downstream_puzzle_structure_control_v2 `
  --diffusion-vqvae-checkpoint outputs\vqvae_ablation_codebook512_v2\checkpoints\vqvae\vqvae_pretrained.pth `
  --diffusion-puzzle-structure-dropout-prob 0.35 `
  --seed 42 `
  --no-auto-resume `
  --verbose
```

Fast sampler:

```powershell
python main.py train `
  --config configs\zelda_hmolqd.yaml `
  --stage fast_sampler `
  --output-dir outputs\zelda_hmolqd_downstream_puzzle_structure_control_v2 `
  --fast-sampler-base-diffusion-checkpoint outputs\zelda_hmolqd_downstream_puzzle_structure_control_v2\checkpoints\diffusion\best_model.pth `
  --fast-sampler-puzzle-structure-dropout-prob 0.35 `
  --seed 42 `
  --no-auto-resume `
  --verbose
```

Masked-room:

```powershell
python main.py train `
  --config configs\zelda_hmolqd.yaml `
  --stage masked_room `
  --output-dir outputs\zelda_hmolqd_downstream_puzzle_structure_control_v2 `
  --masked-room-puzzle-structure-dropout-prob 0.35 `
  --seed 42 `
  --no-auto-resume `
  --verbose
```

## Exact Evaluation And Ablation Commands

### Fixed-Graph Protocol On Final Branch

Manual compare:

```powershell
python main.py topology-compare-manual `
  --run-dir outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v2 `
  --output-dir outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v2\protocol_manual_compare_currentcode_v8 `
  --seed 20260417
```

Fixed-graph audit:

```powershell
python main.py topology-audit-fixed-graph `
  --run-dir outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v2 `
  --output-dir outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v2\protocol_ablation_currentcode_v8 `
  --seeds 20260404 20260405 20260406
```

Strict no-fallback / puzzle / pure-neural runtime matrix:

```powershell
python main.py topology-audit-fixed-graph `
  --run-dir outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v2 `
  --output-dir outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v2\protocol_ablation_currentcode_v8_full `
  --seeds 20260404 20260405 20260406 `
  --include-no-fallback-ablations `
  --include-puzzle-ablations
```

### Core Architecture Ablation

```powershell
python scripts\run_ablation_study.py `
  --output results\ablation_core_full_v1 `
  --data-root "Data\The Legend of Zelda" `
  --num-samples 8 `
  --seed 42 `
  --num-rooms 8 `
  --target-curve "0.2,0.4,0.6,0.8,0.7,0.5,0.3,0.2" `
  --diffusion-steps 25 `
  --cbs-timeout 120000 `
  --evolution-population 24 `
  --evolution-generations 30 `
  --vqvae-checkpoint outputs\vqvae_ablation_codebook512_v2\checkpoints\vqvae\vqvae_pretrained.pth `
  --diffusion-checkpoint outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v2\checkpoints\diffusion\best_model.pth `
  --masked-room-checkpoint outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v2\checkpoints\masked_room\masked_room_best.pth
```

### Room Branch Benchmark

```powershell
python scripts\run_room_branch_benchmark.py `
  --output results\room_branch_benchmark_full_v1 `
  --data-root "Data\The Legend of Zelda" `
  --num-samples 8 `
  --seed 42 `
  --num-rooms 8 `
  --target-curve 0.2 0.4 0.6 0.8 1.0 `
  --diffusion-steps 25 `
  --cbs-timeout 1000 `
  --evolution-population 24 `
  --evolution-generations 30 `
  --vqvae-checkpoint outputs\vqvae_ablation_codebook512_v2\checkpoints\vqvae\vqvae_pretrained.pth `
  --diffusion-checkpoint outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v2\checkpoints\diffusion\best_model.pth `
  --masked-room-checkpoint outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v2\checkpoints\masked_room\masked_room_best.pth
```

### P-CBS Component Ablations

Balanced:

```powershell
python scripts\run_pcbs_component_ablation.py `
  --levels 1,2,3 `
  --variants 1,2 `
  --persona balanced `
  --ablations full,no_revisit,no_uncertainty,no_deliberation,no_affordance,no_focus `
  --timeout-astar 200000 `
  --timeout-pcbs 50000 `
  --seed 42 `
  --output-dir results\pcbs_component_ablation_balanced_l123_v4
```

Explorer:

```powershell
python scripts\run_pcbs_component_ablation.py `
  --levels 1,2,3 `
  --variants 1,2 `
  --persona explorer `
  --ablations full,no_revisit,no_uncertainty,no_deliberation,no_affordance,no_focus `
  --timeout-astar 200000 `
  --timeout-pcbs 50000 `
  --seed 42 `
  --output-dir results\pcbs_component_ablation_explorer_l123_v4
```

Novice:

```powershell
python scripts\run_pcbs_component_ablation.py `
  --levels 1,2,3 `
  --variants 1,2 `
  --persona novice `
  --ablations full,no_revisit,no_uncertainty,no_deliberation,no_affordance,no_focus `
  --timeout-astar 200000 `
  --timeout-pcbs 50000 `
  --seed 42 `
  --output-dir results\pcbs_component_ablation_novice_l123_v1
```

### Full P-CBS Persona Benchmark

```powershell
python scripts\run_cbs_benchmarks.py `
  --levels 1,2,3,4,5,6,7,8,9 `
  --variants 1,2 `
  --all-personas `
  --timeout-astar 200000 `
  --timeout-cbs 50000 `
  --seed 42 `
  --output results\cbs_benchmark_levels1_9_variants12_all_personas_v4.csv
```

### Matched-Budget External Topology Baselines

```powershell
python scripts\run_matched_budget_topology_benchmark.py `
  --output results\matched_budget_topology_v2 `
  --data-root "Data\The Legend of Zelda" `
  --methods "RANDOM,ES,GA,MAP_ELITES,FULL" `
  --num-samples 64 `
  --seed 42 `
  --eval-budget 512 `
  --population-hint 24 `
  --min-rooms 8 `
  --max-rooms 16 `
  --room-count-bias 0.45 `
  --room-budget-cap 42 `
  --rule-space full `
  --archive-cells 128 `
  --map-elites-init-frac 0.35 `
  --map-elites-mutation-rate 0.18
```

### PCG Benchmark Alignment

```powershell
python scripts\run_pcg_benchmark_alignment.py `
  --output results\pcg_benchmark_alignment_v3 `
  --data-root "Data\The Legend of Zelda" `
  --methods "FULL_GA,FULL_CVT,CORE_GA" `
  --problems "zelda-v0,zelda-enemies-v0,zelda-large-v0" `
  --num-samples 8 `
  --seed 42 `
  --population-size 24 `
  --generations 24 `
  --room-budget-cap 42 `
  --qd-archive-cells 128 `
  --qd-init-random-fraction 0.35 `
  --qd-emitter-mutation-rate 0.18 `
  --control-mode graph `
  --pcg-benchmark-repo "tmp\pcg_benchmark_upstream"
```

## PowerShell Parallel Launch Pattern

Independent VQ-VAE runs can be launched in parallel like this:

```powershell
Start-Process powershell -ArgumentList '-NoExit','-Command','Set-Location "<repo root>"; python -m src.train_vqvae --config configs\zelda_hmolqd.yaml --save-dir outputs\vqvae_ablation_codebook128_v2\checkpoints\vqvae --data-dir "Data\The Legend of Zelda" --epochs 300 --hidden-dim 96 --latent-dim 64 --codebook-size 128 --use-coordconv --mrf-penalty-weight 0.05 --validation-fraction 0.1 --validation-max-batches 16 --best-checkpoint-metric val_loss --seed 42'
Start-Process powershell -ArgumentList '-NoExit','-Command','Set-Location "<repo root>"; python -m src.train_vqvae --config configs\zelda_hmolqd.yaml --save-dir outputs\vqvae_ablation_codebook512_v2\checkpoints\vqvae --data-dir "Data\The Legend of Zelda" --epochs 300 --hidden-dim 96 --latent-dim 64 --codebook-size 512 --use-coordconv --mrf-penalty-weight 0.05 --validation-fraction 0.1 --validation-max-batches 16 --best-checkpoint-metric val_loss --seed 42'
Start-Process powershell -ArgumentList '-NoExit','-Command','Set-Location "<repo root>"; python -m src.train_vqvae --config configs\zelda_hmolqd.yaml --save-dir outputs\vqvae_ablation_hidden64_v2\checkpoints\vqvae --data-dir "Data\The Legend of Zelda" --epochs 300 --hidden-dim 64 --latent-dim 64 --codebook-size 256 --use-coordconv --mrf-penalty-weight 0.05 --validation-fraction 0.1 --validation-max-batches 16 --best-checkpoint-metric val_loss --seed 42'
```

After diffusion finishes for a downstream branch, fast sampler and masked-room
can be launched in parallel because they are independent of each other.
