# VQ-VAE Protocol Results 2026-04-10

This note records the direct comparison of all completed Block II runs on the same held-out split of the Zelda room corpus.

Evaluation artifact:

- [outputs/vqvae_protocol_check_20260410/summary.json](../outputs/vqvae_protocol_check_20260410/summary.json)

## Protocol

All checkpoints were evaluated on the same deterministic split:

- dataset size: `459`
- train subset: `413`
- validation subset: `46`

Metrics compared:

- `val_loss`
- `val_recon_loss`
- `val_accuracy`
- `val_perplexity`
- codebook-health statistics

Compared checkpoints:

- canonical old checkpoint: `outputs/zelda_hmolqd_fulltrain_rerun/checkpoints/vqvae/vqvae_pretrained.pth`
- new baseline
- codebook `128`
- codebook `512`
- hidden dim `64`
- no CoordConv
- no MRF penalty

## Results

### Best validation loss

1. `codebook512`  
   `val_loss = 6.2839e-05`

2. `old_fulltrain_rerun`  
   `val_loss = 6.4996e-05`

3. `baseline`  
   `val_loss = 6.6234e-05`

### Clear degradations

- `no_coordconv` worsened notably
- `hidden64` worsened notably
- `codebook128` worsened notably
- `no_mrf` degraded badly and is not a safe choice

## Interpretation

### What improved

- A larger `512` codebook gave the best held-out reconstruction among the tested variants.
- The gain over the old canonical checkpoint is real but **small**.

### What stayed true

- All strong variants still reached effectively perfect tile accuracy on the held-out split.
- Codebook collapse is not the current issue.
- Block II still does not look like the primary system bottleneck.

### What the ablations teach us

- **CoordConv matters** for fixed-size Zelda rooms.
- **The MRF adjacency penalty matters** enough to keep.
- Shrinking the model to `hidden_dim=64` is possible, but it loses some held-out quality.
- Shrinking the codebook to `128` also loses quality.

## Recommendation

If we need a single Block II checkpoint to carry forward later:

- **Best quality pick:** `codebook512`
- **Best conservative pick:** keep the existing canonical `256` codebook checkpoint if we want to avoid any downstream re-tokenization churn for a tiny gain

Practical recommendation:

- Do **not** treat VQ-VAE as the urgent bottleneck.
- If we retrain downstream later and want the best current tokenizer, use `codebook512`.
- Otherwise, it is reasonable to keep the current canonical tokenizer and keep spending effort on Block I / diffusion / puzzle semantics first.

