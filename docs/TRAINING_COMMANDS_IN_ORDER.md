
# Training Commands In Execution Order

Last verified against the live CLI parsers: 2026-07-23

This is the canonical, strictly sequential PowerShell runbook for training, auditing, and reproducing every experiment in the repository from a clean checkout. All commands should be executed from the repository root.

#### **Phase 1: Setup & Preflight**

* **Step 1** : Install dependencies (`pip install -r requirements-hmolqd.txt`) & set PowerShell environment variables.
* **Step 2** : Run hyperparameter preflight check (`python scripts/check_training_hyperparameters.py`).
* **Step 3** : Run quick 2-batch smoke test (`python main.py train --stage all --quick`).

#### **Phase 2: Core Four-Stage Model Stack**

* **Step 4** : Train Stage 1 — VQ-VAE Tokenizer (`python main.py train --stage vqvae`).
* **Step 5** : Train Stage 2 — Latent Diffusion Teacher & Joint LogicNet (`python main.py train --stage diffusion`).
* *Step 5a* : Optional reduced-capacity 48-channel U-Net profile.
* **Step 6** : Train Stage 3 — LCM-LoRA Fast Sampler (`python main.py train --stage fast_sampler`).
* **Step 7** : Train Stage 4 — Discrete Masked-Room Model (`python main.py train --stage masked_room`).
* *Step 7a* : Alternative single-command full-stack run (`python main.py train --stage all`).

#### **Phase 3: Post-Training Inspection & Evaluation**

* **Step 8** : Run fixed-topology visual audit (`python scripts/run_fast_sampler_visual_audit.py`).
* **Step 9** : Audit original Zelda solvability (`python main.py validate --all --mode full`).
* **Step 10** : Run fixed-graph multi-seed audit (`python scripts/run_fixed_graph_multi_seed_audit.py`).
* **Step 11** : Export manual topology comparison & audit fixed graphs.

#### **Phase 4: Thesis Chapter 4 Publication Reproduction Runs**

* **Step 12** : Dungeon 9 holdout retraining & zero-leakage novelty audit (Table 4.1 & 4.2).
* **Step 13** : Multi-solver search comparison on generated dungeons (Table 4.5).
* **Step 14** : MAP-Elites / QD archive coverage & materialization (Table 4.15 & Fig 4.10).
* **Step 15** : OOD dungeon size & complexity scaling probe (Table 4.16 & 4.17).
* **Step 16** : P-CBS 8-persona cognitive playability benchmark (Table 4.13 & Fig 4.8).

#### **Phase 5: Multi-Seed Retraining & Model Ablations**

* **Step 17** : Retrain baseline across seeds 7, 3407, 0.
* **Step 18** : Train learned puzzle-stage candidate experiments.
* **Step 19** : Tokenizer ablation sweeps (VQ-VAE-2, FSQ, Codebook capacity).
* **Step 20** : Run standalone LogicNet tile classifier calibration.
* **Step 21** : Generate & execute model-architecture ablation manifest.
* **Step 22** : Run multi-GPU distributed diffusion (DDP / `gloo`).
* **Step 23** : Train independent baselines (Gaussian VAE, DCGAN, Autoregressive LLM).
* **Step 24** : Train optional RL persona playtesters (SB3).

---

## Master Execution Order Roadmap

```
Phase 1: Setup & Preflight      (Steps 1 – 3)
Phase 2: Core Model Stack       (Steps 4 – 7)
Phase 3: Inspection & Eval      (Steps 8 – 11)
Phase 4: Chapter 4 Thesis Runs  (Steps 12 – 16)
Phase 5: Multi-Seed & Ablations (Steps 17 – 24)
```

---

## Phase 1: Setup & Preflight

### Step 1: Shell Setup & Environment Variables

```powershell
python -m pip install -r requirements-hmolqd.txt
# Optional visualization dependencies for GUI and visual audits:
python -m pip install -r requirements-visual.txt

Set-Location F:\KLTN

$Config = "configs\zelda_hmolqd.yaml"
$Data = "Data\The Legend of Zelda"
$Seed = 42
$Run = "outputs\paper_baseline_seed_$Seed"

$VqCheckpoint = Join-Path $Run "checkpoints\vqvae\vqvae_pretrained.pth"
$DiffusionCheckpoint = Join-Path $Run "checkpoints\diffusion\best_model.pth"
$FastSamplerCheckpoint = Join-Path $Run "checkpoints\fast_sampler\fast_sampler_best.pth"
$MaskedRoomCheckpoint = Join-Path $Run "checkpoints\masked_room\masked_room_best.pth"

$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128"
$env:KLTN_EXPORT_SEQUENTIAL = "1"
$env:KLTN_EXPORT_MAX_BATCH_SIZE = "1"
```

### Step 2: Hyperparameter & Data Preflight Check

```powershell
python scripts\check_training_hyperparameters.py `
  --config $Config `
  --output "results\training_preflight_seed_$Seed" `
  --probe-data
```

### Step 3: Quick 2-Batch Smoke Test

```powershell
python main.py train `
  --config $Config `
  --data-dir $Data `
  --stage all `
  --output-dir "outputs\smoke_full_stack_seed_$Seed" `
  --seed $Seed `
  --quick `
  --no-auto-resume `
  --verbose
```

---

## Phase 2: Core Four-Stage Model Stack

### Step 4: Stage 1 — Train VQ-VAE Tokenizer

```powershell
python main.py train `
  --config $Config `
  --data-dir $Data `
  --stage vqvae `
  --output-dir $Run `
  --seed $Seed `
  --no-auto-resume `
  --verbose

if (!(Test-Path $VqCheckpoint)) {
  throw "VQ-VAE checkpoint was not created: $VqCheckpoint"
}
```

### Step 5: Stage 2 — Train Latent Diffusion Teacher & Joint LogicNet

```powershell
python main.py train `
  --config $Config `
  --data-dir $Data `
  --stage diffusion `
  --output-dir $Run `
  --diffusion-vqvae-checkpoint $VqCheckpoint `
  --seed $Seed `
  --no-auto-resume `
  --verbose

if (!(Test-Path $DiffusionCheckpoint)) {
  throw "Diffusion checkpoint was not created: $DiffusionCheckpoint"
}
```

#### Step 5a (Optional): Reduced-Capacity Small Diffusion Profile

The 96-channel teacher contains ~66M parameters. To run a smaller 12.4M parameter model (48 channels, 1 res block, 192 context dim), run:

```powershell
$SmallRun = "outputs\diffusion_small_seed_$Seed"
$SmallDiffusionCheckpoint = Join-Path $SmallRun "checkpoints\diffusion\best_model.pth"

python main.py train `
  --config $Config `
  --data-dir $Data `
  --stage diffusion `
  --output-dir $SmallRun `
  --diffusion-vqvae-checkpoint $VqCheckpoint `
  --diffusion-model-channels 48 `
  --diffusion-context-dim 192 `
  --diffusion-unet-num-res-blocks 1 `
  --diffusion-unet-num-heads 4 `
  --diffusion-condition-hidden-dim 128 `
  --diffusion-condition-num-gnn-layers 2 `
  --seed $Seed `
  --no-auto-resume `
  --verbose
```

**Follow-up Add-on for Step 5a**: Because this checkpoint is architecture-incompatible with the 96-channel model, you must train the LCM Fast Sampler and run the visual audit explicitly pointing to `$SmallRun`:

```powershell
# 1. Train Fast Sampler on the Small Diffusion Teacher:
python main.py train `
  --config $Config `
  --data-dir $Data `
  --stage fast_sampler `
  --output-dir $SmallRun `
  --fast-sampler-base-diffusion-checkpoint $SmallDiffusionCheckpoint `
  --seed $Seed `
  --no-auto-resume `
  --verbose

# 2. Visual Audit for the Small Model Branch:
$SmallFastSamplerCheckpoint = Join-Path $SmallRun "checkpoints\fast_sampler\fast_sampler_best.pth"

python scripts\run_fast_sampler_visual_audit.py `
  --run-dir $SmallRun `
  --output-dir (Join-Path $SmallRun "fast_sampler_visual_audit") `
  --lcm-checkpoint $SmallFastSamplerCheckpoint `
  --seed $Seed `
  --num-rooms 8
```

### Step 6: Stage 3 — Train LCM-LoRA Fast Sampler

```powershell
python main.py train `
  --config $Config `
  --data-dir $Data `
  --stage fast_sampler `
  --output-dir $Run `
  --fast-sampler-base-diffusion-checkpoint $DiffusionCheckpoint `
  --seed $Seed `
  --no-auto-resume `
  --verbose

if (!(Test-Path $FastSamplerCheckpoint)) {
  throw "Fast-sampler checkpoint was not created: $FastSamplerCheckpoint"
}
```

### Step 7: Stage 4 — Train Discrete Masked-Room Generator

```powershell
python main.py train `
  --config $Config `
  --data-dir $Data `
  --stage masked_room `
  --output-dir $Run `
  --seed $Seed `
  --no-auto-resume `
  --verbose

if (!(Test-Path $MaskedRoomCheckpoint)) {
  throw "Masked-room checkpoint was not created: $MaskedRoomCheckpoint"
}
```

#### Step 7a (Alternative): Single-Command Full-Stack Run (Steps 4–7 Combined)

```powershell
python main.py train `
  --config $Config `
  --data-dir $Data `
  --stage all `
  --output-dir $Run `
  --seed $Seed `
  --no-auto-resume `
  --verbose
```

---

## Phase 3: Post-Training Inspection & Evaluation

### Step 8: Fixed-Topology Visual Audit (Diffusion Teacher vs LCM Fast Sampler)

```powershell
$AuditOutputDir = Join-Path $Run "fast_sampler_visual_audit"

python scripts\run_fast_sampler_visual_audit.py `
  --run-dir $Run `
  --output-dir $AuditOutputDir `
  --lcm-checkpoint $FastSamplerCheckpoint `
  --seed $Seed `
  --num-rooms 8
```

### Step 9: Original Zelda Solvability Audit

```powershell
python main.py validate --all --mode full
```

### Step 10: Fixed-Graph Multi-Seed Audit

```powershell
$EvalOutput = "results\fixed_graph_multi_seed_audit_seed_$Seed"

python scripts\run_fixed_graph_multi_seed_audit.py `
  --run-dir $Run `
  --output-dir $EvalOutput `
  --seeds 42,43,44 `
  --num-samples 10
```

### Step 11: Rich Topology Comparison & Materialize QD Archive

```powershell
python main.py topology-compare-manual `
  --run-dir $Run `
  --output-dir "outputs\manual_compare"

python main.py topology-audit-fixed-graph `
  --run-dir $Run `
  --output-dir "outputs\fixed_graph_audit"
```

---

## Phase 4: Thesis Chapter 4 Publication Reproduction Runs

### Step 12: Dungeon 9 Holdout Retraining & Zero-Leakage Novelty Audit (Table 4.1 & Table 4.2)

```powershell
$HoldoutRun = "outputs\dungeon9_holdout_seed_$Seed"
$HoldoutVq = Join-Path $HoldoutRun "checkpoints\vqvae\vqvae_pretrained.pth"

python main.py train `
  --config $Config `
  --data-dir $Data `
  --stage vqvae `
  --output-dir $HoldoutRun `
  --train-dungeons 1 2 3 4 5 6 7 8 `
  --seed $Seed `
  --no-auto-resume

python main.py train `
  --config $Config `
  --data-dir $Data `
  --stage diffusion `
  --output-dir $HoldoutRun `
  --diffusion-vqvae-checkpoint $HoldoutVq `
  --train-dungeons 1 2 3 4 5 6 7 8 `
  --seed $Seed `
  --no-auto-resume
```

### Step 13: Multi-Solver Search Comparison on Generated Dungeons (Table 4.5)

```powershell
python scripts\export_manual_rich_topology_compare.py `
  --run-dir $Run `
  --output-dir "results\solver_comparison_benchmark" `
  --compare-all-solvers
```

### Step 14: MAP-Elites / QD Archive Coverage & Materialization (Table 4.15 & Figure 4.10)

```powershell
python main.py topology-materialize-archive `
  --archive "results\topology_qd.pkl" `
  --trust-pickle `
  --output-dir "results\materialized_qd_archive"
```

### Step 15: OOD Dungeon Size & Complexity Scaling Probe (Table 4.16 & Table 4.17)

```powershell
python scripts\run_fixed_graph_multi_seed_audit.py `
  --run-dir $Run `
  --output-dir "results\ood_scaling_audit" `
  --min-rooms 6 `
  --max-rooms 42 `
  --seeds 42,43,44
```

### Step 16: P-CBS 8-Persona Cognitive Playability Benchmark (Table 4.13 & Figure 4.8)

```powershell
python main.py validate --all --mode full --gui
```

---

## Phase 5: Multi-Seed Retraining & Model Ablations

### Step 17: Publication Multi-Seed Retraining Loop

```powershell
foreach ($Seed in 7, 3407, 0) {
  $Run = "outputs\paper_baseline_seed_$Seed"
  $VqCheckpoint = Join-Path $Run "checkpoints\vqvae\vqvae_pretrained.pth"
  $DiffusionCheckpoint = Join-Path $Run "checkpoints\diffusion\best_model.pth"

  python main.py train --config $Config --data-dir $Data --stage vqvae --output-dir $Run --seed $Seed --no-auto-resume --verbose
  if (!(Test-Path $VqCheckpoint)) { throw "Missing $VqCheckpoint" }

  python main.py train --config $Config --data-dir $Data --stage diffusion --output-dir $Run --diffusion-vqvae-checkpoint $VqCheckpoint --seed $Seed --no-auto-resume --verbose
  if (!(Test-Path $DiffusionCheckpoint)) { throw "Missing $DiffusionCheckpoint" }

  python main.py train --config $Config --data-dir $Data --stage fast_sampler --output-dir $Run --fast-sampler-base-diffusion-checkpoint $DiffusionCheckpoint --seed $Seed --no-auto-resume --verbose
  python main.py train --config $Config --data-dir $Data --stage masked_room --output-dir $Run --seed $Seed --no-auto-resume --verbose
}
```

### Step 18: Learned Puzzle-Stage Candidate Experiments

```powershell
$PuzzleSeed = 3407
$PuzzleRun = "outputs\paper_stage_semantics_seed_$PuzzleSeed"
$BaselineVq = "outputs\paper_baseline_seed_$PuzzleSeed\checkpoints\vqvae\vqvae_pretrained.pth"
$PuzzleDiffusion = Join-Path $PuzzleRun "checkpoints\diffusion\best_model.pth"

python main.py train `
  --config $Config `
  --data-dir $Data `
  --stage diffusion `
  --output-dir $PuzzleRun `
  --diffusion-vqvae-checkpoint $BaselineVq `
  --diffusion-puzzle-stage-conditioning-enabled `
  --diffusion-puzzle-stage-token-scale 0.20 `
  --diffusion-puzzle-stage-topology-enabled `
  --diffusion-puzzle-stage-trace-decay 0.75 `
  --diffusion-puzzle-stage-semantics-loss-weight 0.25 `
  --diffusion-puzzle-stage-semantics-hidden-dim 96 `
  --diffusion-puzzle-stage-semantics-max-sequence-length 6 `
  --seed $PuzzleSeed `
  --no-auto-resume `
  --verbose

python main.py train `
  --config $Config `
  --data-dir $Data `
  --stage fast_sampler `
  --output-dir $PuzzleRun `
  --fast-sampler-base-diffusion-checkpoint $PuzzleDiffusion `
  --fast-sampler-puzzle-stage-conditioning-enabled `
  --fast-sampler-puzzle-stage-token-scale 0.20 `
  --fast-sampler-puzzle-stage-topology-enabled `
  --fast-sampler-puzzle-stage-trace-decay 0.75 `
  --fast-sampler-puzzle-stage-semantics-loss-weight 0.25 `
  --fast-sampler-puzzle-stage-semantics-hidden-dim 96 `
  --fast-sampler-puzzle-stage-semantics-max-sequence-length 6 `
  --seed $PuzzleSeed `
  --no-auto-resume `
  --verbose

python main.py train `
  --config $Config `
  --data-dir $Data `
  --stage masked_room `
  --output-dir $PuzzleRun `
  --masked-room-puzzle-stage-conditioning-enabled `
  --masked-room-puzzle-stage-token-scale 0.20 `
  --masked-room-puzzle-stage-topology-enabled `
  --masked-room-puzzle-stage-trace-decay 0.75 `
  --masked-room-puzzle-stage-semantics-loss-weight 0.25 `
  --masked-room-puzzle-stage-semantics-hidden-dim 96 `
  --masked-room-puzzle-stage-semantics-max-sequence-length 6 `
  --seed $PuzzleSeed `
  --no-auto-resume `
  --verbose
```

### Step 19: Tokenizer Ablation Sweeps (VQ-VAE-2, FSQ, Codebook Capacity)

#### Step 19.1: VQ-VAE-2

```powershell
python -m src.train_vqvae `
  --config $Config `
  --data-dir $Data `
  --architecture vqvae2 `
  --top-codebook-size 128 `
  --top-latent-dim 64 `
  --save-dir "outputs\vqvae2_seed_$Seed\checkpoints\vqvae" `
  --seed $Seed `
  --no-auto-resume
```

#### Step 19.2: Finite Scalar Quantization (FSQ)

```powershell
python -m src.train_vqvae `
  --config $Config `
  --data-dir $Data `
  --architecture fsq `
  --save-dir "outputs\fsq_seed_$Seed\checkpoints\vqvae" `
  --seed $Seed `
  --no-auto-resume
```

#### Step 19.3: Capacity and Prior Matrix Sweeps

```powershell
python -m src.train_vqvae --config $Config --data-dir $Data --save-dir "outputs\vq_codebook128_seed_$Seed\checkpoints\vqvae" --epochs 300 --hidden-dim 96 --latent-dim 64 --codebook-size 128 --use-coordconv --mrf-penalty-weight 0.05 --seed $Seed --no-auto-resume
python -m src.train_vqvae --config $Config --data-dir $Data --save-dir "outputs\vq_codebook512_seed_$Seed\checkpoints\vqvae" --epochs 300 --hidden-dim 96 --latent-dim 64 --codebook-size 512 --use-coordconv --mrf-penalty-weight 0.05 --seed $Seed --no-auto-resume
python -m src.train_vqvae --config $Config --data-dir $Data --save-dir "outputs\vq_hidden64_seed_$Seed\checkpoints\vqvae" --epochs 300 --hidden-dim 64 --latent-dim 64 --codebook-size 256 --use-coordconv --mrf-penalty-weight 0.05 --seed $Seed --no-auto-resume
python -m src.train_vqvae --config $Config --data-dir $Data --save-dir "outputs\vq_no_coordconv_seed_$Seed\checkpoints\vqvae" --epochs 300 --hidden-dim 96 --latent-dim 64 --codebook-size 256 --no-use-coordconv --mrf-penalty-weight 0.05 --seed $Seed --no-auto-resume
python -m src.train_vqvae --config $Config --data-dir $Data --save-dir "outputs\vq_no_mrf_seed_$Seed\checkpoints\vqvae" --epochs 300 --hidden-dim 96 --latent-dim 64 --codebook-size 256 --use-coordconv --mrf-penalty-weight 0.0 --seed $Seed --no-auto-resume
```

### Step 20: Standalone LogicNet Calibration Run

```powershell
$LogicRun = "outputs\logicnet_calibration_seed_$Seed"
$LogicVqCheckpoint = "outputs\paper_baseline_seed_$Seed\checkpoints\vqvae\vqvae_pretrained.pth"

python scripts\train_logicnet_tile_classifier.py `
  --data-dir $Data `
  --vqvae-checkpoint $LogicVqCheckpoint `
  --checkpoint-out "$LogicRun\checkpoints\logicnet\tile_classifier.pth" `
  --metrics-out "$LogicRun\metrics\logicnet_tile_classifier.json" `
  --device cuda `
  --epochs 100 `
  --batch-size 32
```

### Step 21: Model-Architecture Ablation Matrix

```powershell
$AblationVqCheckpoint = "outputs\paper_baseline_seed_42\checkpoints\vqvae\vqvae_pretrained.pth"

# 1. Generate ablation manifest:
python scripts\generate_model_architecture_ablation_manifest.py `
  --config $Config `
  --output-dir "results\model_architecture_ablation" `
  --seeds 42,43,44 `
  --epochs 100 `
  --vqvae-checkpoint $AblationVqCheckpoint

# 2. Execute ablation manifest:
python scripts\generate_model_architecture_ablation_manifest.py `
  --execute `
  --config $Config `
  --output-dir "results\model_architecture_ablation" `
  --seeds 42,43,44 `
  --epochs 100 `
  --vqvae-checkpoint $AblationVqCheckpoint
```

### Step 22: Multi-GPU Distributed Diffusion (DDP / gloo)

```powershell
python main.py train `
  --config $Config `
  --data-dir $Data `
  --stage diffusion `
  --output-dir $Run `
  --diffusion-vqvae-checkpoint $VqCheckpoint `
  --distributed-enabled `
  --distributed-backend gloo `
  --nproc-per-node 2 `
  --cuda-visible-devices "0,1" `
  --seed $Seed `
  --no-auto-resume `
  --verbose
```

### Step 23: Independent Publication Baselines

#### Step 23.1: Gaussian VAE

```powershell
python -m src.train_gaussian_vae `
  --config $Config `
  --data-dir $Data `
  --save-dir "outputs\gaussian_vae_seed_$Seed\checkpoints" `
  --seed $Seed `
  --no-auto-resume
```

#### Step 23.2: DCGAN

```powershell
python experiments\baselines\train_dcgan_baseline.py `
  --data-dir $Data `
  --output-dir "outputs\dcgan_baseline_seed_$Seed" `
  --epochs 100 `
  --seed $Seed `
  --device cuda
```

#### Step 23.3: Autoregressive LLM Baseline

```powershell
python experiments\baselines\train_llm_baseline.py `
  --data-dir $Data `
  --output-dir "outputs\transformer_baseline_seed_$Seed" `
  --epochs 100 `
  --seed $Seed `
  --device cuda
```

### Step 24: Optional RL Persona Playtesters (SB3)

```powershell
$Grid = "outputs\paper_baseline_seed_42\final_grid.npy"

foreach ($Persona in "speedrunner", "explorer", "combatant", "cautious") {
  python scripts\train_rl_persona_sb3.py `
    --grid $Grid `
    --output "outputs\rl_personas\${Persona}_seed_$Seed" `
    --persona $Persona `
    --timesteps 1000000 `
    --seed $Seed `
    --device cuda
}
```

---

## Resume Protocol Summary

- **Interrupted Experiment**: Same output directory + `--auto-resume`.
- **Full State Resume**: `--resume <checkpoint.pth>` (restores model, optimizer, scheduler, epoch, step, AMP scaler).
- **Diffusion Warm-Start**: `--warm-start <weights.pth|safetensors>` (transfers model weights, resets optimizer & epoch counters).

Viewed TRAINING_COMMANDS_IN_ORDER.md:1-678
Edited TRAINING_COMMANDS_IN_ORDER.md

I have restructured [docs/TRAINING_COMMANDS_IN_ORDER.md](file:///f:/KLTN/docs/TRAINING_COMMANDS_IN_ORDER.md) into a **strictly ordered, 24-step master execution roadmap** from start to finish.

---

### Master Step-by-Step Execution Order Summary

```
Phase 1: Setup & Preflight      (Steps 1 – 3)
Phase 2: Core Model Stack       (Steps 4 – 7)
Phase 3: Inspection & Eval      (Steps 8 – 11)
Phase 4: Chapter 4 Thesis Runs  (Steps 12 – 16)
Phase 5: Multi-Seed & Ablations (Steps 17 – 24)
```

#### **Phase 1: Setup & Preflight**

* **Step 1**: Install dependencies (`pip install -r requirements-hmolqd.txt`) & set PowerShell environment variables.
* **Step 2**: Run hyperparameter preflight check (`python scripts/check_training_hyperparameters.py`).
* **Step 3**: Run quick 2-batch smoke test (`python main.py train --stage all --quick`).

#### **Phase 2: Core Four-Stage Model Stack**

* **Step 4**: Train Stage 1 — VQ-VAE Tokenizer (`python main.py train --stage vqvae`).
* **Step 5**: Train Stage 2 — Latent Diffusion Teacher & Joint LogicNet (`python main.py train --stage diffusion`).
  * *Step 5a*: Optional reduced-capacity 48-channel U-Net profile.
* **Step 6**: Train Stage 3 — LCM-LoRA Fast Sampler (`python main.py train --stage fast_sampler`).
* **Step 7**: Train Stage 4 — Discrete Masked-Room Model (`python main.py train --stage masked_room`).
  * *Step 7a*: Alternative single-command full-stack run (`python main.py train --stage all`).

#### **Phase 3: Post-Training Inspection & Evaluation**

* **Step 8**: Run fixed-topology visual audit (`python scripts/run_fast_sampler_visual_audit.py`).
* **Step 9**: Audit original Zelda solvability (`python main.py validate --all --mode full`).
* **Step 10**: Run fixed-graph multi-seed audit (`python scripts/run_fixed_graph_multi_seed_audit.py`).
* **Step 11**: Export manual topology comparison & audit fixed graphs.

#### **Phase 4: Thesis Chapter 4 Publication Reproduction Runs**

* **Step 12**: Dungeon 9 holdout retraining & zero-leakage novelty audit (Table 4.1 & 4.2).
* **Step 13**: Multi-solver search comparison on generated dungeons (Table 4.5).
* **Step 14**: MAP-Elites / QD archive coverage & materialization (Table 4.15 & Fig 4.10).
* **Step 15**: OOD dungeon size & complexity scaling probe (Table 4.16 & 4.17).
* **Step 16**: P-CBS 8-persona cognitive playability benchmark (Table 4.13 & Fig 4.8).

#### **Phase 5: Multi-Seed Retraining & Model Ablations**

* **Step 17**: Retrain baseline across seeds 7, 3407, 0.
* **Step 18**: Train learned puzzle-stage candidate experiments.
* **Step 19**: Tokenizer ablation sweeps (VQ-VAE-2, FSQ, Codebook capacity).
* **Step 20**: Run standalone LogicNet tile classifier calibration.
* **Step 21**: Generate & execute model-architecture ablation manifest.
* **Step 22**: Run multi-GPU distributed diffusion (DDP / `gloo`).
* **Step 23**: Train independent baselines (Gaussian VAE, DCGAN, Autoregressive LLM).
* **Step 24**: Train optional RL persona playtesters (SB3).

---

The document is now 100% linear, sequential, and fully up-to-date!
