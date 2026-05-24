# VQ-VAE Research Audit 2026-04-10

This document answers one specific question for the current KLTN stack:

> Does Block II (the Semantic VQ-VAE) actually need improvement, and if so, what kind?

Short answer:

- **No urgent architectural replacement is justified right now.**
- **Yes, Block II needed protocol and reproducibility improvements.**
- The strongest held-out tokenizer tested in this repo was the `codebook512`
  ablation, but the canonical YAML still keeps the conservative `256`-entry
  default and upgrades via explicit checkpoint handoff.
- The most important fixes were:
  - real held-out validation
  - best-checkpoint selection by validation loss instead of training loss
  - fuller checkpoint metadata
  - explicit codebook-health logging

The current evidence says the main quality bottlenecks are still downstream in Block I / Block III / runtime semantics, not in the tokenizer itself.

Latest linked outcomes:

- tokenizer-side result summary:
  [`archive/2026-q2/VQVAE_PROTOCOL_RESULTS_2026_04_10.md`](archive/2026-q2/VQVAE_PROTOCOL_RESULTS_2026_04_10.md)
- downstream follow-up with puzzle-subtype conditioning:
  [`archive/2026-q2/DOWNSTREAM_CODEBOOK512_PUZZLE_SUBTYPE_PROTOCOL_RESULTS_2026_04_15.md`](archive/2026-q2/DOWNSTREAM_CODEBOOK512_PUZZLE_SUBTYPE_PROTOCOL_RESULTS_2026_04_15.md)

## Scope

This audit is intentionally focused on Block II:

- [src/core/vqvae.py](../src/core/vqvae.py)
- [src/train_vqvae.py](../src/train_vqvae.py)
- [configs/zelda_hmolqd.yaml](../configs/zelda_hmolqd.yaml)
- [src/config_system.py](../src/config_system.py)

## Phase 1 - Research

### Step 1 - Literature Review

Primary sources used:

1. VQ-VAE, NeurIPS 2017  
   van den Oord et al., *Neural Discrete Representation Learning*  
   https://arxiv.org/abs/1711.00937

2. VQ-VAE-2, NeurIPS 2019  
   Razavi et al., *Generating Diverse High-Fidelity Images with VQ-VAE-2*  
   https://arxiv.org/abs/1906.00446

3. VQGAN / Taming Transformers, CVPR 2021  
   Esser et al., *Taming Transformers for High-Resolution Image Synthesis*  
   https://arxiv.org/abs/2012.09841

4. CoordConv, NeurIPS 2018  
   Liu et al., *An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution*  
   https://arxiv.org/abs/1807.03247

5. Large-codebook scaling, NeurIPS 2024  
   Zhu et al., *Scaling the Codebook Size of VQGAN to 100,000 with a Utilization Rate of 99%*  
   https://arxiv.org/abs/2406.11837

Relevant structured-generation references for system fit:

6. Graph2Plan, CVPR 2020  
   https://arxiv.org/abs/2004.13204

7. HouseDiffusion, CVPR 2023  
   https://arxiv.org/abs/2211.13287

8. LayoutDM, 2023  
   https://arxiv.org/abs/2303.08137

### Key findings from the literature

- VQ-VAE remains a strong fit when the downstream model benefits from a **discrete latent vocabulary** and the data has meaningful repeated patterns rather than photorealistic texture detail.
- VQ-VAE-2 and VQGAN improve richness mostly for **higher-resolution, multi-scale natural-image settings**. They are helpful when visual realism and fine texture are central bottlenecks.
- CoordConv is well motivated for domains where **absolute position matters**, especially fixed-size grids with structured semantics.
- Very large codebooks can work at scale, but the latest large-codebook papers assume **much larger datasets and more expressive visual regimes** than this Zelda symbolic room corpus.

### Research-backed implication for this repo

For 16x11 semantic rooms, the question is not "how do we make the tokenizer bigger?" but "is the tokenizer expressive enough, stable enough, and measured honestly enough?".

That distinction matters. In this codebase, most recent failures were:

- topology semantics
- teacher/runtime mismatch
- puzzle logic
- stitching/layout

not raw reconstruction collapse in Block II.

## Phase 2 - Deep Analysis

### Step 2 - Assumptions Validation

#### Current architectural assumptions

1. **Rooms are small, fixed-size symbolic grids.**  
   Holds for the current Zelda corpus. Fragile if the dataset expands to variable-size or richer tile schemas.

2. **A single-level discrete tokenizer is sufficient.**  
   Reasonable for 16x11 symbolic rooms. Less safe for large-scale or art-heavy texture generation.

3. **Absolute spatial position matters.**  
   Supported by CoordConv literature and by Zelda room semantics.

4. **Codebook size 256 is enough.**  
   Plausible on this corpus. Not obviously underpowered given the held-out results below.

5. **The main Block II failure mode is dead-code collapse.**  
   Not supported by the current checkpoint: all 256 codes are used at nonzero EMA usage.

6. **Training loss is a safe best-checkpoint selector.**  
   This was fragile and undocumented. It is not a good default in general.

#### Hardcoded implementation assumptions found in code

These existed before this pass:

- `num_res_blocks = 2` in [src/core/vqvae.py](../src/core/vqvae.py)
- encoder `channel_mult = (1, 2, 4)` in [src/core/vqvae.py](../src/core/vqvae.py)
- decoder `channel_mult = (4, 2, 1)` in [src/core/vqvae.py](../src/core/vqvae.py)
- validation was effectively the training loader, capped to 5 batches, in [src/train_vqvae.py](../src/train_vqvae.py)
- best checkpoint was selected by `train loss`

Candidates for configuration promotion:

- `vqvae.validation_fraction`
- `vqvae.validation_max_batches`
- `vqvae.best_checkpoint_metric`

The first two were promoted in this pass. The fixed architecture knobs were recorded in metadata but not promoted yet because downstream loading currently assumes the canonical architecture shape.

### Step 3 - Logical Audit

#### Issue A - Best-checkpoint selection used training loss

Why this was problematic:

- It rewards memorization even when generalization is worse.
- It makes the downstream stack depend on the most optimistic possible tokenizer snapshot.

Research support:

- This is inference-based rather than paper-specific. The VQ-VAE papers describe objective design, but not this repo-specific checkpoint-selection shortcut.

Status:

- **Fixed in this pass.**

#### Issue B - Evaluation reused the training stream

Why this was problematic:

- It could not reveal train/validation divergence.
- It made the stored `accuracy` too easy to over-interpret.

Status:

- **Fixed in this pass with a held-out validation split.**

#### Issue C - Checkpoint metadata was incomplete

Why this was problematic:

- Diffusion-stage architecture resolution already tries to consume VQ-VAE metadata.
- The VQ-VAE metadata omitted key fields such as `hidden_dim`, `num_classes`, `commitment_cost`, and `use_ema`.
- That creates silent config fallback instead of exact stage handoff.

Status:

- **Fixed in this pass for newly written checkpoints.**

### Step 4 - Theory vs Implementation

The implementation is broadly faithful to the intended model:

- discrete latent bottleneck
- EMA codebook
- rare-tile reweighting
- CoordConv input handling
- soft illegal-adjacency penalty

But before this pass there were theory/implementation gaps in the *training protocol*:

- claimed "best checkpoint" was not validation-backed
- reconstruction quality was not reported on a held-out split
- codebook health was not logged in a stable way

So the architecture was acceptable, but the **measurement contract** around it was weaker than the rest of the stack.

### Step 5 - Gap and Bug Analysis

#### Confirmed gaps

1. No real validation split
2. Best model selected by training loss
3. Incomplete checkpoint metadata
4. No explicit codebook-health summary in history/metadata

#### New configurable parameters added

| Parameter | Type | Default | Valid Range | Why |
|---|---:|---:|---:|---|
| `vqvae.validation_fraction` | `float` | `0.1` | `0.0..0.5` | held-out model selection |
| `vqvae.validation_max_batches` | `int` | `16` | `>=1` | bounded validation cost |
| `vqvae.best_checkpoint_metric` | `str` | `val_loss` | `{val_loss, train_loss}` | explicit model-selection rule |

### Step 6 - Redundancy and Unnecessary Work

No major architectural redundancy was found inside the tokenizer itself.

However, there was one procedural redundancy:

- evaluating on the training stream after already logging training loss

That evaluation consumed compute without giving trustworthy generalization information.

Status:

- **Replaced with real validation.**

### Step 7 - Computational Complexity

#### Current model size

Local checkpoint inspection on the canonical model:

- **Parameters:** `17,623,948`
- **Latent grid:** `4 x 3` for a `16 x 11` room
- **Codebook:** `256 x 64`

#### Complexity summary

Let:

- input room size be `H x W`
- latent grid be `H' x W'`
- codebook size be `K`
- latent width be `D`

Then:

- encoder/decoder conv stack: approximately `O(HW * C^2 * k^2)` per level
- vector-quantization lookup: `O(H'W'KD)`

For the current Zelda setting:

- `H'W' = 12`
- `K = 256`
- `D = 64`

So quantization cost is modest relative to the CNN stack. The current VQ step is not the part of the pipeline that is blowing up runtime.

#### Comparison against larger alternatives

- VQ-VAE-2 adds hierarchy and more complexity than the current symbolic regime clearly needs.
- VQGAN-style perceptual/adversarial tokenizers increase optimization complexity and are aimed more at natural-image realism.
- Large-codebook methods like VQGAN-LC solve a different regime problem: scaling visual tokenizers to much richer data.

### Step 8 - Hyperparameter Sensitivity

Most sensitive VQ-VAE knobs in this repo:

| Parameter | Safe Range | Risk |
|---|---|---|
| `vqvae.codebook_size` | `128..512` | too large wastes codes on this tiny corpus |
| `vqvae.hidden_dim` | `64..128` | larger capacity likely increases overfitting-friendly slack |
| `vqvae.latent_dim` | `32..96` | too small may bottleneck semantics; too large increases cost |
| `vqvae.commitment_cost` | `0.1..0.5` | too low destabilizes commitment, too high can over-regularize |
| `vqvae.rare_tile_weight` | `2..8` | too high can over-bias rare semantics |
| `vqvae.mrf_penalty_weight` | `0.0..0.1` | too high can distort the recon objective |

Research-backed or inference-based note:

- these ranges are partly inference-based from the current repo scale
- the large-tokenizer literature does **not** imply we should raise codebook size on this corpus

### Step 9 - Failure Modes and Edge Cases

Known Block II failure patterns:

1. **Selection optimism** when choosing best checkpoint by train loss  
   Fixed.

2. **Silent architecture mismatch** downstream if metadata is incomplete  
   Fixed for new checkpoints.

3. **Over-capacity relative to tiny dataset**  
   Not catastrophic right now, but still a medium-term risk.

4. **Codebook under-utilization if codebook size is pushed upward**  
   Supported by both prior VQ experience and the recent large-codebook literature.

### Step 10 - Scalability and Generalization Boundary

Current VQ-VAE is practical at the current corpus scale.

But the generalization ceiling is clear:

- it is tuned for **small symbolic rooms**
- not for large natural-image tokens
- not for multi-resolution texture hierarchies

If the project later shifts toward:

- richer sprite fidelity
- larger rooms
- multi-room direct tokenization

then Block II should be revisited.

### Step 11 - Comparison Against SOTA

Against modern visual tokenizer literature:

- the current model is **not SOTA** in the broad image-tokenization sense
- but it is still **appropriate** for this repo's data regime and downstream needs

Assessment:

- **Competitive enough for symbolic Zelda room tokenization**
- **Not competitive as a modern large-scale vision tokenizer**
- **Meaningfully simpler and safer** than VQGAN/VQ-VAE-2 for the current dataset size

### Step 12 - Bias and Ethical Risk

For Block II specifically, architectural ethical risk is low. The model reconstructs symbolic rooms and does not directly encode demographic or human-sensitive attributes.

Main reproducibility risk was technical:

- hidden assumptions
- incomplete metadata
- optimistic checkpoint selection

## Phase 3 - Synthesis

### Step 13 - Evidence-Based Decision

#### Repo evidence

Quick held-out evaluation on the existing canonical checkpoint using a deterministic 90/10 split of the current corpus:

- dataset size: `459` rooms
- train split: `413`
- val split: `46`
- train loss: `3.46e-05`
- val loss: `6.50e-05`
- train accuracy: `1.0`
- val accuracy: `1.0`
- train perplexity: `71.0`
- val perplexity: `67.3`

Interpretation:

- generalization on the current reconstruction task is still very strong
- Block II does **not** look like the immediate quality bottleneck
- the larger issue was that this was not being measured honestly in the original trainer

#### Consolidated config table

| Parameter Name | Type | Default | Valid Range | Source | Notes |
|---|---:|---:|---:|---|---|
| `vqvae.validation_fraction` | `float` | `0.1` | `0.0..0.5` | inference-based | added in this pass |
| `vqvae.validation_max_batches` | `int` | `16` | `>=1` | inference-based | keeps validation cheap |
| `vqvae.best_checkpoint_metric` | `str` | `val_loss` | `{val_loss,train_loss}` | inference-based | explicit selection contract |
| `vqvae.codebook_size` | `int` | `256` | `128..512` | literature + repo evidence | current size is still reasonable |
| `vqvae.hidden_dim` | `int` | `96` | `64..128` | inference-based | current size is already conservative |
| `vqvae.latent_dim` | `int` | `64` | `32..96` | literature + repo evidence | no evidence yet that it is the bottleneck |

### Step 14 - Recommended Ablations

Do not retrain blindly. If Block II is revisited, use these commands.

Baseline:

```powershell
python main.py train --config configs/zelda_hmolqd.yaml --stage vqvae --output-dir outputs/vqvae_audit_baseline_v1 --no-auto-resume --verbose
```

Smaller codebook:

```powershell
python main.py train --config configs/zelda_hmolqd.yaml --stage vqvae --output-dir outputs/vqvae_ablation_codebook128_v1 --vqvae-codebook-size 128 --no-auto-resume --verbose
```

Larger codebook:

```powershell
python main.py train --config configs/zelda_hmolqd.yaml --stage vqvae --output-dir outputs/vqvae_ablation_codebook512_v1 --vqvae-codebook-size 512 --no-auto-resume --verbose
```

Smaller backbone:

```powershell
python main.py train --config configs/zelda_hmolqd.yaml --stage vqvae --output-dir outputs/vqvae_ablation_hidden64_v1 --vqvae-hidden-dim 64 --no-auto-resume --verbose
```

No CoordConv:

```powershell
python main.py train --config configs/zelda_hmolqd.yaml --stage vqvae --output-dir outputs/vqvae_ablation_no_coordconv_v1 --no-vqvae-use-coordconv --no-auto-resume --verbose
```

No adjacency penalty:

```powershell
python main.py train --config configs/zelda_hmolqd.yaml --stage vqvae --output-dir outputs/vqvae_ablation_no_mrf_v1 --vqvae-mrf-penalty-weight 0.0 --no-auto-resume --verbose
```

### Step 15 - Priority Ranking

#### Critical

- None found in the tokenizer architecture itself.

#### High

1. Best-checkpoint selection by training loss  
   Status: fixed

2. No true held-out validation  
   Status: fixed

3. Incomplete VQ-VAE metadata for stage handoff  
   Status: fixed

#### Medium

4. Hidden architecture assumptions not fully promoted into config  
   Status: documented, metadata-expanded, not yet exposed as runtime knobs

5. Codebook health not visible enough in history  
   Status: fixed

#### Low

6. Possible moderate over-capacity relative to dataset size  
   Status: monitor via ablations, not urgent

#### Reproducibility Risk

- best-checkpoint metric
- validation split policy
- metadata completeness

## Phase 4 - Immediate Implementation

### Step 16 - Fixes applied now

Implemented in:

- [src/train_vqvae.py](../src/train_vqvae.py)
- [src/config_system.py](../src/config_system.py)
- [configs/zelda_hmolqd.yaml](../configs/zelda_hmolqd.yaml)
- [tests/test_hmolqd/test_vqvae.py](../tests/test_hmolqd/test_vqvae.py)
- [tests/test_config_system.py](../tests/test_config_system.py)

Applied changes:

1. Added real held-out validation support
2. Added bounded validation evaluation per epoch
3. Changed best-checkpoint logic to prefer `val_loss`
4. Added explicit codebook-health metrics
5. Expanded checkpoint metadata so downstream stages can recover Block II shape/config more faithfully

## Final Recommendation

### Does VQ-VAE need improvement?

**Yes, but mostly in protocol and reproducibility, not in architecture.**

### Does it need retraining right now?

**Not solely because of the tokenizer.**

The evidence does not support "VQ-VAE is the main generation bottleneck" as the current diagnosis.

### What should we do next?

1. Keep the current Block II architecture.
2. Use the improved trainer/metadata path for future Block II runs.
3. Only retrain VQ-VAE if an explicit ablation shows that tokenizer capacity or reconstruction is limiting downstream quality.
4. Keep focusing primary quality work on:
   - Block I distribution quality
   - diffusion teacher quality
   - stateful puzzle semantics
   - hybrid runtime semantics
