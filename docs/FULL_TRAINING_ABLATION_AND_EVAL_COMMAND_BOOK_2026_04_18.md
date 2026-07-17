# Full Training, Ablation, and Evaluation Command Book

Last verified: 2026-07-13

For the shortest dependency-ordered training sequence, use
[`TRAINING_COMMANDS_IN_ORDER.md`](TRAINING_COMMANDS_IN_ORDER.md). This document
remains the expanded ablation and evaluation reference.

The ordered runbook is the authoritative source for the baseline training
commands and checkpoint hand-offs. Keep this document for optional model
variants, ablations, and evaluation commands; do not combine its alternatives
with the baseline sequence in a single output directory.

This is the practical runbook for rebuilding the repo from scratch and
reproducing the main ablations. It is intentionally operational.

## 1. Environment Assumptions

- run commands from the repository root
- shell: `PowerShell`
- CUDA GPU available
- dataset root: `Data\The Legend of Zelda`
- canonical config: `configs\zelda_hmolqd.yaml`

OOM-safe defaults that should be set for long runs:

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128'
$env:KLTN_EXPORT_SEQUENTIAL='1'
$env:KLTN_EXPORT_MAX_BATCH_SIZE='1'
```

## 1.1 Protocol Alerts

- The completed six-run `VQ-VAE` ablation currently supports `codebook_size=256`, `hidden_dim=96`, `latent_dim=64`, `coordconv=on`, and `mrf_penalty_weight=0.05` as the best-tested tokenizer setting by held-out validation loss.
- Designer-controllability and compute/sample-efficiency commands are maintained in `docs\DESIGNER_CONTROLLABILITY_AND_COMPUTE_PROTOCOL.md`.
- `codebook512` remains a valid downstream comparison branch, but it is not the best tokenizer by the completed `VQ-VAE` evidence.
- `diffusion.validation_fraction=0.1` is now part of the canonical config, so new `main.py train --stage diffusion` launches use a real held-out split for checkpoint selection.
- Any diffusion run started before the 2026-04-19 held-out-validation patch must be treated as interim only for thesis purposes and rerun before final Chapter 4 claims.
- Thesis-facing hyperparameter rationale and the current empirical summary are maintained in:
  - `docs\THESIS_HYPERPARAMETER_SEARCH_AND_PROTOCOL_JUSTIFICATION_2026_04_19.md`
  - `results\thesis_hparam_evidence_2026_04_19.md`
- Run the preflight checker before long training launches:

```powershell
python scripts\check_training_hyperparameters.py `
  --config configs\zelda_hmolqd.yaml `
  --output results\training_hyperparameter_check `
  --probe-data
```

The checker writes JSON/CSV/Markdown reports and fails on hard issues such as
zero train batches, CUDA requested when unavailable, latent-dimension mismatch,
invalid attention-head divisibility, and runtime sampling steps exceeding the
trained diffusion timestep count.

## 2. Recommended Search Strategy For Training / Eval

This is not hyperparameter search for search algorithms. It is the repo's
recommended evaluation stack:

- hard correctness oracle:
  - `graph_guided_oracle`
  - `A*`
  - `Dijkstra` exact fallback
  - `graph_progression`
  - `softlock_check`
- comparison solvers:
  - `BFS`
  - `Dijkstra`
  - `Greedy`
  - `D* Lite`
  - `DFS/IDDFS`
  - `Bidirectional A*`
- behavioral probe:
  - `P-CBS`

Do not treat `D* Lite` as the primary static correctness oracle for the current
stateful dungeon validator.

## 3. Hyperparameter Ranges

### 3.1 VQ-VAE cookbook

Primary axes already used in this repo:

| Axis | Values |
|---|---|
| `codebook_size` | `128`, `256`, `512` |
| `hidden_dim` | `64`, `96` |
| `latent_dim` | `64` |
| `coordconv` | `on`, `off` |
| `mrf_penalty_weight` | `0.0`, `0.05` |

Canonical best-tested tokenizer evidence:

- `codebook_size=256`
- `hidden_dim=96`
- `latent_dim=64`
- `coordconv=on`
- `mrf_penalty_weight=0.05`

Comparison tokenizer evidence worth keeping in the thesis:

- `codebook_size=512` is a downstream high-capacity stress-test branch, not the tokenizer winner by held-out validation loss
- `hidden_dim=64` is close enough to the `96`-channel baseline that width is not currently the dominant bottleneck
- removing `coordconv` or the MRF adjacency prior hurts held-out reconstruction materially

### 3.2 Downstream branch cookbook

Primary axes:

| Axis | Values |
|---|---|
| teacher branch | `diffusion` |
| auxiliary branches | `fast_sampler`, `masked_room` |
| VQ-VAE source | `codebook256`, `codebook512` |
| puzzle structure dropout | `0.15`, `0.35`, `0.55` |
| stage-conditioning tokens | `off`, `on` |
| ordered stage-trace prior | `off`, `on` |
| stage-semantics loss weight | `0.0`, `0.10`, `0.25`, `0.50` |
| stage-semantics head width | `96` |
| max supervised stage slots | `6` |

### 3.3 Runtime stateful puzzle sweep

Primary axes already exposed in `scripts/run_stateful_puzzle_hparam_sweep.py`:

| Profile | Main idea |
|---|---|
| `baseline_default` | current canonical runtime |
| `conservative_quality` | lower branch density, stronger quality gate |
| `route_safe_stateful` | stronger route preservation |
| `dense_stateful` | denser puzzle branching |
| `deterministic_low_novelty` | low novelty / low stochasticity |
| `no_puzzle_control` | no-puzzle ablation |

## 4. From-Scratch Training Order

### 4.1 VQ-VAE baseline + ablations

```powershell
python -m src.train_vqvae --config configs\zelda_hmolqd.yaml --save-dir outputs\vqvae_audit_baseline_v2\checkpoints\vqvae --data-dir "Data\The Legend of Zelda" --epochs 300 --hidden-dim 96 --latent-dim 64 --codebook-size 256 --use-coordconv --mrf-penalty-weight 0.05 --validation-fraction 0.1 --validation-max-batches 16 --best-checkpoint-metric val_loss --seed 42

python -m src.train_vqvae --config configs\zelda_hmolqd.yaml --save-dir outputs\vqvae_ablation_codebook128_v2\checkpoints\vqvae --data-dir "Data\The Legend of Zelda" --epochs 300 --hidden-dim 96 --latent-dim 64 --codebook-size 128 --use-coordconv --mrf-penalty-weight 0.05 --validation-fraction 0.1 --validation-max-batches 16 --best-checkpoint-metric val_loss --seed 42

python -m src.train_vqvae --config configs\zelda_hmolqd.yaml --save-dir outputs\vqvae_ablation_codebook512_v2\checkpoints\vqvae --data-dir "Data\The Legend of Zelda" --epochs 300 --hidden-dim 96 --latent-dim 64 --codebook-size 512 --use-coordconv --mrf-penalty-weight 0.05 --validation-fraction 0.1 --validation-max-batches 16 --best-checkpoint-metric val_loss --seed 42

python -m src.train_vqvae --config configs\zelda_hmolqd.yaml --save-dir outputs\vqvae_ablation_hidden64_v2\checkpoints\vqvae --data-dir "Data\The Legend of Zelda" --epochs 300 --hidden-dim 64 --latent-dim 64 --codebook-size 256 --use-coordconv --mrf-penalty-weight 0.05 --validation-fraction 0.1 --validation-max-batches 16 --best-checkpoint-metric val_loss --seed 42

python -m src.train_vqvae --config configs\zelda_hmolqd.yaml --save-dir outputs\vqvae_ablation_no_coordconv_v2\checkpoints\vqvae --data-dir "Data\The Legend of Zelda" --epochs 300 --hidden-dim 96 --latent-dim 64 --codebook-size 256 --no-use-coordconv --mrf-penalty-weight 0.05 --validation-fraction 0.1 --validation-max-batches 16 --best-checkpoint-metric val_loss --seed 42

python -m src.train_vqvae --config configs\zelda_hmolqd.yaml --save-dir outputs\vqvae_ablation_no_mrf_v2\checkpoints\vqvae --data-dir "Data\The Legend of Zelda" --epochs 300 --hidden-dim 96 --latent-dim 64 --codebook-size 256 --use-coordconv --mrf-penalty-weight 0.0 --validation-fraction 0.1 --validation-max-batches 16 --best-checkpoint-metric val_loss --seed 42
```

### 4.2 Primary downstream diffusion comparison branches

Use identical settings for both branches except the tokenizer checkpoint. This
is the cleanest Chapter 4 comparison for asking whether the larger codebook
helps downstream controllability despite weaker tokenizer reconstruction.

```powershell
python main.py train --config configs\zelda_hmolqd.yaml --stage diffusion --output-dir outputs\zelda_hmolqd_downstream_baseline_puzzle_subtype_v2 --diffusion-vqvae-checkpoint outputs\vqvae_audit_baseline_v2\checkpoints\vqvae\vqvae_pretrained.pth --seed 42 --no-auto-resume --verbose

python main.py train --config configs\zelda_hmolqd.yaml --stage fast_sampler --output-dir outputs\zelda_hmolqd_downstream_baseline_puzzle_subtype_v2 --fast-sampler-base-diffusion-checkpoint outputs\zelda_hmolqd_downstream_baseline_puzzle_subtype_v2\checkpoints\diffusion\best_model.pth --seed 42 --no-auto-resume --verbose

python main.py train --config configs\zelda_hmolqd.yaml --stage masked_room --output-dir outputs\zelda_hmolqd_downstream_baseline_puzzle_subtype_v2 --seed 42 --no-auto-resume --verbose

python main.py train --config configs\zelda_hmolqd.yaml --stage diffusion --output-dir outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v2 --diffusion-vqvae-checkpoint outputs\vqvae_ablation_codebook512_v2\checkpoints\vqvae\vqvae_pretrained.pth --seed 42 --no-auto-resume --verbose

python main.py train --config configs\zelda_hmolqd.yaml --stage fast_sampler --output-dir outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v2 --fast-sampler-base-diffusion-checkpoint outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v2\checkpoints\diffusion\best_model.pth --seed 42 --no-auto-resume --verbose

python main.py train --config configs\zelda_hmolqd.yaml --stage masked_room --output-dir outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v2 --seed 42 --no-auto-resume --verbose
```

Thesis rule:

- if the currently running `baseline` and `codebook512` diffusion jobs were started before the 2026-04-19 validation patch, rerun them under the current trainer before using `best_model.pth` for any final branch comparison

### 4.3 Training-time puzzle-control branch

Do not run the historical structure-dropout branch. Its augmentation removed
BLOCK tiles while retaining the original staged puzzle plan, creating
contradictory labels. The production default is `0.0`. Re-enable this axis
only after a solver-validated counterfactual compiler updates the room grid,
stage plan, controlled doors, and oracle proof together.

### 4.4 Puzzle cookbook sweep

This sweep is retired for the same label-consistency reason. It is not an
evidence-producing ablation until the counterfactual compiler exists.

### 4.5 Learned staged-puzzle branch with explicit semantic supervision

Use this branch if and only if you want to claim that the model learned ordered
multi-step puzzle semantics rather than relying only on the hybrid runtime
grammar.

Clear alert:

- every older checkpoint family is outdated for that claim
- you must retrain and rerun protocol exports
- boolean CLI flags in `main.py train` use `argparse.BooleanOptionalAction`
  - use `--diffusion-puzzle-stage-conditioning-enabled`
  - do not write `--diffusion-puzzle-stage-conditioning-enabled true`

Recommended full branch name:

- `outputs\zelda_hmolqd_downstream_stageconditioned_semantics_v2`

Recommended full-loss setting:

- `puzzle_stage_conditioning_enabled = on`
- `puzzle_stage_topology_enabled = on`
- `puzzle_stage_semantics_loss_weight = 0.25`
- `puzzle_stage_semantics_hidden_dim = 96`
- `puzzle_stage_semantics_max_sequence_length = 6`

```powershell
python main.py train --config configs\zelda_hmolqd.yaml --stage diffusion --output-dir outputs\zelda_hmolqd_downstream_stageconditioned_semantics_v2 --diffusion-vqvae-checkpoint outputs\vqvae_ablation_codebook512_v2\checkpoints\vqvae\vqvae_pretrained.pth --diffusion-puzzle-stage-conditioning-enabled --diffusion-puzzle-stage-token-scale 0.20 --diffusion-puzzle-stage-topology-enabled --diffusion-puzzle-stage-trace-decay 0.75 --diffusion-puzzle-stage-semantics-loss-weight 0.25 --diffusion-puzzle-stage-semantics-hidden-dim 96 --diffusion-puzzle-stage-semantics-max-sequence-length 6 --seed 42 --no-auto-resume --verbose

python main.py train --config configs\zelda_hmolqd.yaml --stage fast_sampler --output-dir outputs\zelda_hmolqd_downstream_stageconditioned_semantics_v2 --fast-sampler-base-diffusion-checkpoint outputs\zelda_hmolqd_downstream_stageconditioned_semantics_v2\checkpoints\diffusion\best_model.pth --fast-sampler-puzzle-stage-conditioning-enabled --fast-sampler-puzzle-stage-token-scale 0.20 --fast-sampler-puzzle-stage-topology-enabled --fast-sampler-puzzle-stage-trace-decay 0.75 --fast-sampler-puzzle-stage-semantics-loss-weight 0.25 --fast-sampler-puzzle-stage-semantics-hidden-dim 96 --fast-sampler-puzzle-stage-semantics-max-sequence-length 6 --seed 42 --no-auto-resume --verbose

python main.py train --config configs\zelda_hmolqd.yaml --stage masked_room --output-dir outputs\zelda_hmolqd_downstream_stageconditioned_semantics_v2 --masked-room-puzzle-stage-conditioning-enabled --masked-room-puzzle-stage-token-scale 0.20 --masked-room-puzzle-stage-topology-enabled --masked-room-puzzle-stage-trace-decay 0.75 --masked-room-puzzle-stage-semantics-loss-weight 0.25 --masked-room-puzzle-stage-semantics-hidden-dim 96 --masked-room-puzzle-stage-semantics-max-sequence-length 6 --seed 42 --no-auto-resume --verbose
```

Recommended ablation around the new branch:

```powershell
python main.py train --config configs\zelda_hmolqd.yaml --stage diffusion --output-dir outputs\zelda_hmolqd_downstream_stageconditioned_semantics_tokens_only_v2 --diffusion-vqvae-checkpoint outputs\vqvae_ablation_codebook512_v2\checkpoints\vqvae\vqvae_pretrained.pth --diffusion-puzzle-stage-conditioning-enabled --diffusion-puzzle-stage-token-scale 0.20 --no-diffusion-puzzle-stage-topology-enabled --diffusion-puzzle-stage-semantics-loss-weight 0.25 --diffusion-puzzle-stage-semantics-hidden-dim 96 --diffusion-puzzle-stage-semantics-max-sequence-length 6 --seed 42 --no-auto-resume --verbose

python main.py train --config configs\zelda_hmolqd.yaml --stage diffusion --output-dir outputs\zelda_hmolqd_downstream_stageconditioned_semantics_trace_only_v2 --diffusion-vqvae-checkpoint outputs\vqvae_ablation_codebook512_v2\checkpoints\vqvae\vqvae_pretrained.pth --no-diffusion-puzzle-stage-conditioning-enabled --diffusion-puzzle-stage-topology-enabled --diffusion-puzzle-stage-trace-decay 0.75 --diffusion-puzzle-stage-semantics-loss-weight 0.25 --diffusion-puzzle-stage-semantics-hidden-dim 96 --diffusion-puzzle-stage-semantics-max-sequence-length 6 --seed 42 --no-auto-resume --verbose

python main.py train --config configs\zelda_hmolqd.yaml --stage diffusion --output-dir outputs\zelda_hmolqd_downstream_stageconditioned_semantics_loss010_v2 --diffusion-vqvae-checkpoint outputs\vqvae_ablation_codebook512_v2\checkpoints\vqvae\vqvae_pretrained.pth --diffusion-puzzle-stage-conditioning-enabled --diffusion-puzzle-stage-token-scale 0.20 --diffusion-puzzle-stage-topology-enabled --diffusion-puzzle-stage-trace-decay 0.75 --diffusion-puzzle-stage-semantics-loss-weight 0.10 --diffusion-puzzle-stage-semantics-hidden-dim 96 --diffusion-puzzle-stage-semantics-max-sequence-length 6 --seed 42 --no-auto-resume --verbose

python main.py train --config configs\zelda_hmolqd.yaml --stage diffusion --output-dir outputs\zelda_hmolqd_downstream_stageconditioned_semantics_loss050_v2 --diffusion-puzzle-stage-conditioning-enabled --diffusion-puzzle-stage-token-scale 0.20 --diffusion-puzzle-stage-topology-enabled --diffusion-puzzle-stage-trace-decay 0.75 --diffusion-puzzle-stage-semantics-loss-weight 0.50 --diffusion-puzzle-stage-semantics-hidden-dim 96 --diffusion-puzzle-stage-semantics-max-sequence-length 6 --seed 42 --no-auto-resume --verbose
```

Interpretation:

- `stageconditioned_v1` is now a transitional branch
- `stageconditioned_semantics_v2` is the first branch that actually combines:
  - ordered stage tokens
  - ordered stage traces
  - explicit learned semantic loss on generated room logits
- if you want the stronger thesis claim, do not cite `stageconditioned_v1`

### 4.1 Masked-room spatial graph-attention ablation

The canonical masked-room checkpoint remains the `additive` baseline. Train
the graph branch from scratch because enabling it adds parameters; loading an
additive checkpoint with `strict=False` is not a matched ablation.

```powershell
python -m src.train_masked_room --config configs\zelda_hmolqd.yaml --checkpoint-dir outputs\masked_topology_additive_seed42\checkpoints\masked_room --topology-conditioning-mode additive --attention-mode softmax --seed 42 --no-auto-resume

python -m src.train_masked_room --config configs\zelda_hmolqd.yaml --checkpoint-dir outputs\masked_topology_graph_softmax_seed42\checkpoints\masked_room --topology-conditioning-mode graph_cross_attention --attention-mode softmax --spatial-graph-gate-init -2.0 --seed 42 --no-auto-resume

python -m src.train_masked_room --config configs\zelda_hmolqd.yaml --checkpoint-dir outputs\masked_topology_graph_linear_seed42\checkpoints\masked_room --topology-conditioning-mode graph_cross_attention --attention-mode linear_hedgehog --hedgehog-feature-dim 32 --graph-auto-linear-attention-nodes 128 --spatial-graph-gate-init -2.0 --seed 42 --no-auto-resume
```

Repeat unchanged for the declared paired seeds. Compare held-out loss,
raw-neural and final exact-oracle validity, graph-to-grid topology drift,
fallback rate, wall time, peak memory, and the reported attention-pair counts.
The softmax-versus-linear comparison is secondary to additive-versus-graph
conditioning and must not be pooled across different parameter budgets.
The same three runs are emitted by
`scripts/generate_model_architecture_ablation_manifest.py`; use
`--no-masked-room` only when intentionally planning the diffusion-only table.

## 5. Parallel GPU Launcher

If you want parallel phase execution instead of launching by hand:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_parallel_training_suite_2026_04_17.ps1 -Phase vqvae -GpuIds 0,1
powershell -ExecutionPolicy Bypass -File scripts\run_parallel_training_suite_2026_04_17.ps1 -Phase diffusion-codebook512 -GpuIds 0
powershell -ExecutionPolicy Bypass -File scripts\run_parallel_training_suite_2026_04_17.ps1 -Phase aux-codebook512 -GpuIds 0,1
powershell -ExecutionPolicy Bypass -File scripts\run_parallel_training_suite_2026_04_17.ps1 -Phase diffusion-puzzle-control -GpuIds 0
powershell -ExecutionPolicy Bypass -File scripts\run_parallel_training_suite_2026_04_17.ps1 -Phase aux-puzzle-control -GpuIds 0,1
powershell -ExecutionPolicy Bypass -File scripts\run_parallel_training_suite_2026_04_17.ps1 -Phase puzzle-cookbook-diffusion -GpuIds 0,1
powershell -ExecutionPolicy Bypass -File scripts\run_parallel_training_suite_2026_04_17.ps1 -Phase puzzle-cookbook-aux -GpuIds 0,1
```

## 6. Evaluation Commands

### 6.1 Canonical fixed-graph protocol

```powershell
python scripts\run_fixed_graph_multi_seed_audit.py `
  --run-dir outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v2 `
  --output-dir outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v2\protocol_ablation_statefulmultistep_v23 `
  --seeds 20260404 20260405 20260406
```

For the staged-puzzle branch:

```powershell
python scripts\run_fixed_graph_multi_seed_audit.py `
  --run-dir outputs\zelda_hmolqd_downstream_stageconditioned_semantics_v2 `
  --output-dir outputs\zelda_hmolqd_downstream_stageconditioned_semantics_v2\protocol_ablation_stageconditioned_semantics_v2 `
  --seeds 20260404 20260405 20260406
```

```powershell
python scripts\export_semantic_anchor_end_to_end.py `
  --run-dir outputs\zelda_hmolqd_downstream_stageconditioned_semantics_v2 `
  --output-dir outputs\zelda_hmolqd_downstream_stageconditioned_semantics_v2\protocol_manual_compare_stageconditioned_semantics_v2 `
  --seed 20260404
```

### 6.2 Manual side-by-side export

```powershell
python scripts\export_semantic_anchor_end_to_end.py `
  --run-dir outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v2 `
  --output-dir outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v2\protocol_manual_compare_statefulmultistep_v23 `
  --seed 20260404
```

### 6.3 Runtime puzzle-profile sweep

```powershell
python scripts\run_stateful_puzzle_hparam_sweep.py `
  --run-dir outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v2 `
  --output-dir results\stateful_puzzle_hparam_sweep_v3 `
  --seed 20260418
```

### 6.4 Core architecture ablation

```powershell
python scripts\run_ablation_study.py `
  --config configs\zelda_hmolqd.yaml `
  --output-dir results\ablation_core_quick_v4 `
  --quick `
  --max-runtime-sec 7200 `
  --vqvae-checkpoint outputs\vqvae_ablation_codebook512_v2\checkpoints\vqvae\vqvae_pretrained.pth
```

### 6.5 VQ-VAE-2 tokenizer ablation

```powershell
python src\train_vqvae.py `
  --config configs\zelda_hmolqd.yaml `
  --architecture vqvae2 `
  --top-codebook-size 128 `
  --top-latent-dim 64 `
  --save-dir outputs\vqvae2_hierarchical_v1\checkpoints\vqvae
```

### 6.6 Conditioning, LogicNet, and repair matrix

Plan-only:

```powershell
python scripts\run_conditioning_logicnet_repair_ablation.py `
  --output results\conditioning_logicnet_repair_ablation
```

Execute later:

```powershell
python scripts\run_conditioning_logicnet_repair_ablation.py `
  --execute `
  --config configs\zelda_hmolqd.yaml `
  --output results\conditioning_logicnet_repair_ablation `
  --seeds 42,43,44 `
  --vqvae-checkpoint outputs\vqvae2_hierarchical_v1\checkpoints\vqvae\vqvae_pretrained.pth `
  --diffusion-checkpoint outputs\YOUR_DIFFUSION_RUN\checkpoints\diffusion\best_model.pth `
  --logic-net-checkpoint outputs\YOUR_DIFFUSION_RUN\checkpoints\diffusion\best_model.pth
```

The runner writes `conditioning_logicnet_repair_logic_deltas.csv` for paired
LogicNet ON/OFF deltas. It refuses to execute without trained checkpoints unless
`--allow-random-fallback` is passed for a code-only smoke run.

### 6.7 Designer controllability proof

```powershell
python scripts\run_designer_controllability_proof.py `
  --execute `
  --output results\designer_controllability_proof `
  --methods FULL_GA,FULL_CVT,CORE_GA `
  --samples-per-target 8 `
  --population-size 32 `
  --generations 40 `
  --seed 42
```

### 6.8 Compute and sample-efficiency consolidation

```powershell
python scripts\consolidate_compute_sample_efficiency.py `
  --roots outputs results `
  --output results\compute_sample_efficiency
```

### 6.9 Room branch benchmark

```powershell
python scripts\run_room_branch_benchmark.py `
  --config configs\zelda_hmolqd.yaml `
  --output-dir results\room_branch_benchmark_quick_v3 `
  --vqvae-checkpoint outputs\vqvae_ablation_codebook512_v2\checkpoints\vqvae\vqvae_pretrained.pth `
  --quick
```

### 6.10 Matched-budget topology baseline

```powershell
python scripts\run_matched_budget_topology_benchmark.py `
  --output results\matched_budget_topology_v2 `
  --samples-per-method 64 `
  --seed 42
```

### 6.11 P-CBS component ablations

```powershell
python scripts\run_pcbs_component_ablation.py --levels 1,2,3 --persona balanced --output-dir results\pcbs_component_ablation_balanced_l123_v4
python scripts\run_pcbs_component_ablation.py --levels 1,2,3 --persona explorer --output-dir results\pcbs_component_ablation_explorer_l123_v4
python scripts\run_pcbs_component_ablation.py --levels 1,2,3 --persona novice --output-dir results\pcbs_component_ablation_novice_l123_v1
```

### 6.12 Full persona benchmark

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

## 7. Validation / Sanity Commands

```powershell
python -m pytest tests\test_search_factory.py tests\test_protocol_reporting.py tests\test_cognitive_bounded_search.py tests\test_architecture_audit_fixes.py -q
python -m py_compile src\simulation\search_factory.py scripts\run_fast_sampler_visual_audit.py
```

## 8. Honest Status

This command book is complete enough to rebuild the main current ablations.

It is not a claim that every one of these runs has already been completed on the
latest code path. In particular:

- the puzzle cookbook family is still incomplete as full evidence
- the latest-code full persona benchmark still needs a clean rerun
- the latest-code canonical protocol should be rerun after any new reporting
  patch you want reflected in official JSON artifacts
- the new learned staged-puzzle branch is now implemented in code, but it is
  not evidenced until the retraining and protocol reruns above are complete
- no honest report should say `surpasses prior publications` until the new
  `stageconditioned_semantics_v2` branch is actually retrained and compared
  against matched-budget baselines on the refreshed protocol
