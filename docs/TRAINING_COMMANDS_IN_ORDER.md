# Training Commands In Execution Order

Last verified against the live CLI parsers: 2026-07-17

This is the canonical PowerShell runbook for training the repository from a
clean checkout. Run commands from the repository root. Commands under
**Required full stack** are ordered by checkpoint dependency. Later sections
are optional ablations or independent publication baselines.

## 1. Dependency Order

The required model order is:

1. data and configuration preflight
2. VQ-VAE tokenizer
3. latent-diffusion teacher, using the VQ-VAE checkpoint
4. LCM-LoRA fast sampler, using the diffusion checkpoint
5. discrete masked-room generator
6. evaluation and ablations

LogicNet is created and trained jointly by the diffusion trainer. It can also
be enabled in masked-room training. The standalone LogicNet tile-classifier
trainer in Section 8 is a calibration ablation, not an extra mandatory stage.

The masked-room model is tile-categorical and does not depend on the VQ-VAE or
diffusion checkpoint. It is kept fifth so `main.py train --stage all` and the
explicit sequence have the same stage order.

## 2. Shell Setup

Create or activate the Python environment first. The project dependency file
contains the training stack; visualization dependencies are only required for
the optional visual audits and GUI.

```powershell
python -m pip install -r requirements-hmolqd.txt
# Optional:
python -m pip install -r requirements-visual.txt
```

```powershell
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

Do not reuse an output directory for a clean comparison. `--no-auto-resume`
prevents an old `latest_resume.pth` from changing the experiment silently.

## 3. Preflight

```powershell
python scripts\check_training_hyperparameters.py `
  --config $Config `
  --output "results\training_preflight_seed_$Seed" `
  --probe-data
```

Stop if this command reports a hard error. In particular, do not start a long
run with an empty train split, a latent-width mismatch, or an invalid attention
head configuration.

Optional two-batch smoke test of the complete stage wiring:

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

Smoke checkpoints are not evidence and must not be used for evaluation.

## 4. Required Full Stack

### 4.1 Train VQ-VAE

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

Expected artifact: `checkpoints\vqvae\vqvae_pretrained.pth`.

### 4.2 Train Diffusion Teacher and Joint LogicNet

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

The canonical config enables trainable LogicNet supervision. The diffusion
checkpoint therefore contains the diffusion, condition encoder, and LogicNet
states. Expected artifact: `checkpoints\diffusion\best_model.pth`.

### 4.3 Train LCM-LoRA Fast Sampler

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

Expected artifact: `checkpoints\fast_sampler\fast_sampler_best.pth`.

### 4.4 Train Discrete Masked-Room Model

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

Expected artifact: `checkpoints\masked_room\masked_room_best.pth`.

### 4.5 One-Command Alternative

This is equivalent to Sections 4.1-4.4 and uses the same internal order:

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

Use either the explicit sequence or this command, not both. The explicit
sequence is preferred for cluster jobs because every dependency is checked
before the next job starts.

## 5. Publication Multi-Seed Retraining

Use independent output directories. This loop retrains the complete stack for
seeds 42, 43, and 44, including a tokenizer per seed.

```powershell
foreach ($Seed in 42, 43, 44) {
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

For a cheaper downstream-only variance estimate, train one fixed tokenizer and
reuse its checkpoint across the three downstream seeds. Report that protocol
explicitly; it does not measure tokenizer-seed variance.

## 6. Learned Puzzle-Stage Candidate

This is a separate experiment family. Do not write these checkpoints into the
baseline directory.

```powershell
$Seed = 42
$PuzzleRun = "outputs\paper_stage_semantics_seed_$Seed"
$VqCheckpoint = "outputs\paper_baseline_seed_$Seed\checkpoints\vqvae\vqvae_pretrained.pth"
$PuzzleDiffusion = Join-Path $PuzzleRun "checkpoints\diffusion\best_model.pth"

python main.py train `
  --config $Config `
  --data-dir $Data `
  --stage diffusion `
  --output-dir $PuzzleRun `
  --diffusion-vqvae-checkpoint $VqCheckpoint `
  --diffusion-puzzle-stage-conditioning-enabled `
  --diffusion-puzzle-stage-token-scale 0.20 `
  --diffusion-puzzle-stage-topology-enabled `
  --diffusion-puzzle-stage-trace-decay 0.75 `
  --diffusion-puzzle-stage-semantics-loss-weight 0.25 `
  --diffusion-puzzle-stage-semantics-hidden-dim 96 `
  --diffusion-puzzle-stage-semantics-max-sequence-length 6 `
  --seed $Seed `
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
  --seed $Seed `
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
  --seed $Seed `
  --no-auto-resume `
  --verbose
```

Repeat unchanged for seeds 43 and 44. The puzzle-stage loss weight, token
conditioning, and topology trace must each be ablated independently before
claiming which component causes an improvement.

## 7. Tokenizer Ablations

These are alternatives to Section 4.1, not additional stages in one model.

### 7.1 VQ-VAE-2

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

### 7.2 Finite Scalar Quantization

```powershell
python -m src.train_vqvae `
  --config $Config `
  --data-dir $Data `
  --architecture fsq `
  --save-dir "outputs\fsq_seed_$Seed\checkpoints\vqvae" `
  --seed $Seed `
  --no-auto-resume
```

### 7.3 Capacity and Prior Matrix

```powershell
python -m src.train_vqvae --config $Config --data-dir $Data --save-dir "outputs\vq_codebook128_seed_$Seed\checkpoints\vqvae" --epochs 300 --hidden-dim 96 --latent-dim 64 --codebook-size 128 --use-coordconv --mrf-penalty-weight 0.05 --validation-fraction 0.1 --validation-max-batches 16 --best-checkpoint-metric val_loss --seed $Seed --no-auto-resume
python -m src.train_vqvae --config $Config --data-dir $Data --save-dir "outputs\vq_codebook256_seed_$Seed\checkpoints\vqvae" --epochs 300 --hidden-dim 96 --latent-dim 64 --codebook-size 256 --use-coordconv --mrf-penalty-weight 0.05 --validation-fraction 0.1 --validation-max-batches 16 --best-checkpoint-metric val_loss --seed $Seed --no-auto-resume
python -m src.train_vqvae --config $Config --data-dir $Data --save-dir "outputs\vq_codebook512_seed_$Seed\checkpoints\vqvae" --epochs 300 --hidden-dim 96 --latent-dim 64 --codebook-size 512 --use-coordconv --mrf-penalty-weight 0.05 --validation-fraction 0.1 --validation-max-batches 16 --best-checkpoint-metric val_loss --seed $Seed --no-auto-resume
python -m src.train_vqvae --config $Config --data-dir $Data --save-dir "outputs\vq_hidden64_seed_$Seed\checkpoints\vqvae" --epochs 300 --hidden-dim 64 --latent-dim 64 --codebook-size 256 --use-coordconv --mrf-penalty-weight 0.05 --validation-fraction 0.1 --validation-max-batches 16 --best-checkpoint-metric val_loss --seed $Seed --no-auto-resume
python -m src.train_vqvae --config $Config --data-dir $Data --save-dir "outputs\vq_no_coordconv_seed_$Seed\checkpoints\vqvae" --epochs 300 --hidden-dim 96 --latent-dim 64 --codebook-size 256 --no-use-coordconv --mrf-penalty-weight 0.05 --validation-fraction 0.1 --validation-max-batches 16 --best-checkpoint-metric val_loss --seed $Seed --no-auto-resume
python -m src.train_vqvae --config $Config --data-dir $Data --save-dir "outputs\vq_no_mrf_seed_$Seed\checkpoints\vqvae" --epochs 300 --hidden-dim 96 --latent-dim 64 --codebook-size 256 --use-coordconv --mrf-penalty-weight 0.0 --validation-fraction 0.1 --validation-max-batches 16 --best-checkpoint-metric val_loss --seed $Seed --no-auto-resume
```

## 8. LogicNet Calibration Ablation

The normal diffusion command already trains LogicNet jointly. Use this command
only to train and evaluate its latent-to-tile classifier independently.

```powershell
$Seed = 42
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

## 9. Model-Architecture Ablation Matrix

Generate and inspect the manifest before executing it. It covers U-Net versus
DiT, DiT normalization/activation, softmax versus Hedgehog attention,
topology-refinement variants, SPADE versus additive conditioning, and the
masked-room graph-attention variants.

```powershell
$AblationVqCheckpoint = "outputs\paper_baseline_seed_42\checkpoints\vqvae\vqvae_pretrained.pth"

python scripts\generate_model_architecture_ablation_manifest.py `
  --config $Config `
  --output-dir "results\model_architecture_ablation" `
  --seeds 42,43,44 `
  --epochs 100 `
  --vqvae-checkpoint $AblationVqCheckpoint
```

After reviewing the emitted commands and available compute budget:

```powershell
python scripts\generate_model_architecture_ablation_manifest.py `
  --execute `
  --config $Config `
  --output-dir "results\model_architecture_ablation" `
  --seeds 42,43,44 `
  --epochs 100 `
  --vqvae-checkpoint $AblationVqCheckpoint
```

Do not compare variants trained for different epochs, seeds, data splits, or
checkpoint-selection metrics as a matched ablation.

## 10. Multi-GPU Diffusion

Only the diffusion stage currently supports multi-process DDP through the main
entry point. VQ-VAE, fast sampler, and masked-room training remain single
process.

```powershell
python main.py train `
  --config $Config `
  --data-dir $Data `
  --stage diffusion `
  --output-dir $Run `
  --diffusion-vqvae-checkpoint $VqCheckpoint `
  --distributed-enabled `
  --nproc-per-node 2 `
  --cuda-visible-devices "0,1" `
  --seed $Seed `
  --no-auto-resume `
  --verbose
```

On Windows, use the `gloo` backend if the installed PyTorch build does not
support NCCL:

```powershell
python main.py train --config $Config --data-dir $Data --stage diffusion --output-dir $Run --diffusion-vqvae-checkpoint $VqCheckpoint --distributed-enabled --distributed-backend gloo --nproc-per-node 2 --cuda-visible-devices "0,1" --seed $Seed --no-auto-resume --verbose
```

## 11. Independent Publication Baselines

These trainers do not feed checkpoints into the four-stage architecture. They
produce matched external baselines and must use the same data split and seeds.

### 11.1 Gaussian VAE

```powershell
python -m src.train_gaussian_vae `
  --config $Config `
  --data-dir $Data `
  --save-dir "outputs\gaussian_vae_seed_$Seed\checkpoints" `
  --seed $Seed `
  --no-auto-resume
```

### 11.2 DCGAN

```powershell
python experiments\baselines\train_dcgan_baseline.py `
  --data-dir $Data `
  --output-dir "outputs\dcgan_baseline_seed_$Seed" `
  --epochs 100 `
  --seed $Seed `
  --device cuda
```

### 11.3 Autoregressive Transformer Baseline

```powershell
python experiments\baselines\train_llm_baseline.py `
  --data-dir $Data `
  --output-dir "outputs\transformer_baseline_seed_$Seed" `
  --epochs 100 `
  --seed $Seed `
  --device cuda
```

## 12. Optional RL Persona Playtesters

Train these only after exporting a final semantic grid. They evaluate generated
levels and are not generators.

```powershell
$Grid = "outputs\YOUR_GENERATION_RUN\final_grid.npy"

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

## 13. Resume Rules

- Clean experiment: new output directory plus `--no-auto-resume`.
- Interrupted experiment: same output directory plus `--auto-resume`.
- `--resume` is only for a complete `.pth` training-state checkpoint containing
  model, optimizer, scheduler, epoch, global step, and AMP scaler state when AMP
  is active. It fails closed for inference-only checkpoints.
- `--warm-start` is the explicit diffusion-only weights transfer path. It accepts
  compatible `.pth` or `.safetensors` weights and resets epoch, global step,
  optimizer, scheduler, scaler, and historical best metrics.
- Never resume one ablation from another architecture's checkpoint.
- Preserve `resolved_config.yaml`, `run_metadata.json`, logs, and the selected
  best checkpoint together.
- Do not evaluate `final_model.pth` by default when the trainer selected
  `best_model.pth` on held-out validation.

Example resume:

```powershell
python main.py train --config $Config --data-dir $Data --stage diffusion --output-dir $Run --diffusion-vqvae-checkpoint $VqCheckpoint --seed $Seed --auto-resume --verbose
```

Explicit diffusion warm start from an inference artifact:

```powershell
python -m src.train_diffusion `
  --config $Config `
  --data-dir $Data `
  --vqvae-checkpoint $VqCheckpoint `
  --checkpoint-dir (Join-Path $Run "checkpoints\diffusion") `
  --warm-start "outputs\source_run\checkpoints\diffusion\best_model.safetensors" `
  --no-auto-resume `
  --seed $Seed
```

## 14. What To Run First

For one working end-to-end model, run Sections 2, 3, and 4 only.

For publication evidence, run:

1. Section 5 for the baseline multi-seed stack
2. Section 6 for the learned puzzle-stage candidate
3. Section 7 for tokenizer ablations
4. Section 9 for architecture ablations
5. Section 11 for independent baselines

Evaluation commands are intentionally maintained separately in
[`FULL_TRAINING_ABLATION_AND_EVAL_COMMAND_BOOK_2026_04_18.md`](FULL_TRAINING_ABLATION_AND_EVAL_COMMAND_BOOK_2026_04_18.md)
and the protocol documents linked from [`INDEX.md`](INDEX.md).
