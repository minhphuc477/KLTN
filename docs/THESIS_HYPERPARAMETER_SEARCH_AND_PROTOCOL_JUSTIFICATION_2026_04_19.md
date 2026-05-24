# Thesis Hyperparameter Search and Protocol Justification

Last updated: 2026-04-19

This note is the thesis-facing justification for the hyperparameter ranges,
search strategy, and evaluation protocol used by the current H-MOLQD stack:

- tokenizer: `VQ-VAE`
- room generator: graph-conditioned latent `diffusion`
- auxiliary generators: `fast_sampler`, `masked_room`
- runtime controller: topology generation, symbolic repair, hybrid validator

The local empirical evidence currently available is summarized in:

- `results/thesis_hparam_evidence_2026_04_19.md`
- `results/thesis_hparam_evidence_2026_04_19.json`

## Strict Status Before Writing Chapter 4

Two facts must be stated explicitly in the thesis:

1. the six `VQ-VAE` ablations are complete and already provide valid tokenizer-screening evidence
2. the currently running diffusion branches were launched before the 2026-04-19 held-out diffusion validation fix, so their in-training validation values are interim and must not be treated as thesis-final checkpoint-selection evidence

That diffusion weakness has now been fixed in code:

- [`src/train_diffusion.py`](../src/train_diffusion.py)
- [`src/config_system.py`](../src/config_system.py)

As of this patch, diffusion now uses the same deterministic held-out validation
policy already used by `VQ-VAE`, `fast_sampler`, and `masked_room`.

## Chapter 3: Methodology

### 3.1 System-Specific Problem Setting

This repository is not operating in the regime assumed by large-scale image
generation papers. The room corpus currently used for model training contains
only `459` room samples. The in-flight diffusion logs report:

- trainable parameters: `70.42M`
- effective train samples: `459` before the new held-out split, `413` after the new `0.1` validation split
- parameter-to-sample ratio: approximately `153k` trainable parameters per room before the split

Therefore the search space must be deliberately narrow. A wide unconstrained
hyperparameter search would be methodologically weak because:

- it would overfit a tiny corpus
- it would multiply compute without increasing inferential value
- it would make negative results uninterpretable because capacity, optimization, and conditioning would be changing simultaneously

The correct strategy is a bounded, staged, hypothesis-driven search rather than
global black-box optimization over the full pipeline.

### 3.2 Hyperparameter Inventory and Roles

#### 3.2.1 VQ-VAE / Tokenizer

| Parameter | Role | Current default | Tested repo values | Thesis-bounded range | Reason for range |
| --- | --- | --- | --- | --- | --- |
| `latent_dim` | width of continuous encoder output before quantization; also the latent interface to diffusion | `64` | `64` | keep fixed at `64` for main thesis runs | downstream compatibility matters more than marginal tokenizer gains; current evidence does not identify tokenizer width as the bottleneck |
| `hidden_dim` | encoder/decoder capacity | `96` | `64`, `96` | `64` to `96` | larger widths increase overfitting risk on `459` rooms; current `64` vs `96` comparison already covers the plausible small-data envelope |
| `codebook_size` | discrete latent capacity and semantic vocabulary granularity | `256` | `128`, `256`, `512` | `128` to `512` | below `128` likely underfits rare room semantics; above `512` is not justified on the current corpus because `512` already shows lower EMA live-code rate and worse validation loss |
| `commitment_cost` | forces encoder outputs to stay close to chosen code vectors | `0.25` | fixed | `0.1`, `0.25`, `0.5` | the original VQ-VAE reports robustness for `beta` from `0.1` to `2.0` and uses `0.25`; on this corpus a narrower range is safer because reconstruction loss scale is small |
| `learning_rate` | optimizer step size | `3e-4` | fixed | `1e-4` to `5e-4` log-uniform | VQ-VAE used Adam at `2e-4`; this repo uses `3e-4`, which is reasonable for small batches but still should be bounded tightly |
| `weight_decay` | weak regularization against overfitting | `1e-5` | fixed | `0` to `1e-4` | higher values risk hurting reconstruction on tiny structured maps |
| `use_coordconv` | injects absolute spatial coordinates | `on` | `on`, `off` | binary ablation only | room semantics are location-sensitive; this must be tested as an inductive-bias ablation, not continuously tuned |
| `mrf_penalty_weight` | discourages illegal local tile adjacencies | `0.05` | `0.0`, `0.05` | `0.0` to `0.1` | this is a structural prior; a light penalty is enough to test whether local legality helps without dominating reconstruction |
| `rare_tile_weight` | reweights rare semantic tiles | `5.0` | fixed | `1` to `8` | useful for sparse keys, locks, puzzle markers, but too large a value would distort the global tile distribution |

#### 3.2.2 Diffusion Teacher

| Parameter | Role | Current default | Tested repo values | Thesis-bounded range | Reason for range |
| --- | --- | --- | --- | --- | --- |
| `num_timesteps` | depth of the forward noising process used during training | `1000` | fixed | `500`, `1000` | `1000` is the canonical DDPM-family setting; fewer training timesteps are a compute ablation, not the default |
| `schedule_type` | noise schedule shape | `cosine` | fixed | `linear`, `cosine` | `Improved DDPM` showed cosine schedules as a strong practical modification, so linear is an ablation baseline and cosine is the default |
| `prediction_type` | denoising target parameterization | `epsilon` | fixed | `epsilon`, `v` | `epsilon` follows DDPM; `v` is a valid modern alternative but should be a discrete ablation, not a continuous search axis |
| `learning_rate` | diffusion optimizer step size | `1e-4` | fixed | `5e-5` to `2e-4` log-uniform | the current model is large relative to data; rates above `2e-4` are likely unstable, and below `5e-5` are inefficient |
| `optimizer_weight_decay` | regularization | `1e-5` | fixed | `0` to `1e-4` | higher decay may underfit an already data-poor setting |
| `model_channels` | U-Net base width | `96` | fixed | `64`, `96`, `128` | this is the main capacity knob; the repo already warns that the current model is in a small-data danger zone, so widths above `128` are not defensible |
| `unet_channel_mult` | per-level channel expansion | `[1, 2, 4]` | fixed | keep fixed for main thesis runs | changing depth and width together would confound capacity conclusions; the current topology is already sufficient for `16x11` rooms |
| `unet_num_res_blocks` | per-scale depth | `2` | fixed | `1`, `2` | deeper stacks increase parameters quickly on limited data |
| `unet_dropout` | regularization within the U-Net | `0.1` | fixed | `0.0` to `0.15` | small-data regularization is useful, but large dropout would weaken spatial precision |
| `condition_hidden_dim` | graph conditioner hidden width | `192` | fixed | `128`, `192`, `256` | condition capacity should be co-tuned with U-Net width; larger values add parameters faster than evidence |
| `condition_num_gnn_layers` | graph depth | `2` | fixed | `1`, `2`, `3` | shallow stacks reduce over-smoothing and are adequate for the mission graphs in this repo |
| `condition_gnn_type` | graph backbone | `gps` | fixed | `gcn`, `gat`, `sage`, `gps` | `GPS` is justified because it combines local message passing with scalable global attention; the other types are valid ablations |
| `condition_use_reference_room_maps` | neighbor-room exemplar conditioning | `on` | fixed | `off`, `on` | binary structural ablation; not a continuous search axis |
| `cfg_dropout_prob` | classifier-free conditioning dropout during training | `0.1` | fixed | `0.05` to `0.2` | enough to teach conditional/unconditional behavior without starving the model of conditioning |
| `cfg_scale` | classifier-free guidance strength used by the teacher and runtime default | `3.0` | fixed | `1.5` to `4.0` | CFG explicitly trades fidelity against diversity; very high values risk low-diversity, overconstrained samples |
| `min_snr_gamma` | timestep reweighting clamp for diffusion loss | `5.0` | fixed | `0`, `1`, `5` | `Min-SNR-gamma` is a justified modern training heuristic; these three values cleanly separate off / weak / canonical |
| `alpha_logic` | LogicNet loss weight | `0.1` | fixed | `0.05` to `0.25` | enough to matter without overpowering visual denoising |
| `warmup_epochs` | delay before logic loss enters the objective | `5` | fixed | `3` to `10` | early logic pressure destabilizes denoising; long warmups waste supervision |
| `puzzle_structure_dropout_prob` | paired puzzle-on / puzzle-off control augmentation | `0.35` | `0.15`, `0.35`, `0.55` | same three-point sweep | low / medium / high control-strength triad; already implemented in the repo and interpretable |
| `puzzle_stage_semantics_loss_weight` | auxiliary supervision for ordered puzzle semantics | `0.0` in base branch, `0.25` in staged branch | `0.0`, `0.10`, `0.25`, `0.50` | same four-point sweep | this is an interpretable semantic-ablation axis, not an arbitrary continuous search |
| `puzzle_stage_token_scale` | amplitude of ordered puzzle-stage tokens | `0.20` | fixed | `0.10` to `0.30` | larger values risk dominating graph conditioning; smaller values may be ignored |
| `puzzle_stage_trace_decay` | attenuation over later stage traces | `0.75` | fixed | `0.5` to `0.9` | this directly controls how strongly later stages influence topology priors |
| `puzzle_stage_semantics_hidden_dim` | size of the semantics head | `96` | fixed | `64`, `96`, `128` | enough capacity to classify puzzle semantics without making the auxiliary head itself a major model |
| `puzzle_stage_semantics_max_sequence_length` | maximum ordered puzzle stages supervised | `6` | fixed | keep fixed at `6` | domain-dependent ceiling from Zelda room semantics; changing it would change the task definition |

#### 3.2.3 Auxiliary Branches and Runtime-Coupled Parameters

| Parameter | Role | Current default | Tested repo values | Thesis-bounded range | Reason for range |
| --- | --- | --- | --- | --- | --- |
| `fast_sampler.num_inference_steps` | few-step student sampling depth | `4` | `4` | `2`, `4`, `8` | primary speed-quality trade-off; higher values reduce the point of distillation |
| `fast_sampler.lora_rank` | distillation adapter capacity | `8` | fixed | `4`, `8`, `16` | rank should stay small because the student is a speed path, not a new full-capacity model |
| `fast_sampler.prediction_loss_weight`, `decode_alignment_weight`, `topology_alignment_weight` | distillation balance | `0.25` each | fixed | `0.1` to `0.5` each, one-at-a-time | use bounded one-factor sweeps only; changing all three together would be hard to interpret |
| `masked_room.model_channels` | masked-room model width | `64` | fixed | `48`, `64`, `96` | smaller than diffusion by design because this branch is an efficient auxiliary generator |
| `generation.num_diffusion_steps` | runtime diffusion sampling depth | `50` | `50` | `20` to `50` | runtime speed-quality trade-off; distinct from training `num_timesteps` |
| `generation.guidance_scale` | runtime CFG strength | `3.0` | `3.0` | `2.0` to `4.0` | must stay aligned with training-time CFG unless explicitly studying deployment mismatch |

### 3.3 Why These Ranges Were Chosen

#### 3.3.1 Prior Literature Constraint

- `DDPM` and `Improved DDPM` establish the standard diffusion training setup: many noising steps, noise schedules over `beta_t`, and `epsilon`-prediction as a strong default.
- `Improved DDPM` specifically motivates practical schedule changes such as cosine noise schedules and highlights that better variance handling reduces sampling cost.
- `Latent Diffusion Models` justify operating diffusion in autoencoder latent space when direct pixel-space diffusion would be wasteful.
- `Classifier-Free Diffusion Guidance` justifies treating CFG as a fidelity-diversity trade-off knob rather than pushing guidance monotonically upward.
- `VQ-VAE` justifies the commitment term and directly reports that the method is robust to `beta` over a broad range, with `beta=0.25` used in their experiments.
- `GraphGPS` justifies `gps` as a default graph encoder because it combines local message passing and global attention with linear complexity.

#### 3.3.2 Theoretical and Implementation Constraint

Some ranges are not merely empirical preferences; they are constrained by the
implementation:

- `diffusion.model_channels * unet_channel_mult[level]` must be divisible by `unet_num_heads`
- `fast_sampler.num_inference_steps <= diffusion.num_timesteps`
- `diffusion.latent_dim == vqvae.latent_dim` for stage handoff compatibility
- `room_topology_channels` is schema-locked to the repository topology representation

These are hard feasibility constraints. They reduce the valid search space
before empirical tuning begins.

#### 3.3.3 Local Empirical Constraint

The strongest local reason to keep the search bounded is the completed
tokenizer evidence:

- `baseline256 + CoordConv + MRF` is currently the best VQ-VAE by validation loss
- `hidden64` is close enough to show that width is not the dominant bottleneck
- `codebook512` is materially worse on validation loss and has a lower EMA live-code rate than the `256`-code baseline
- removing `CoordConv` or the MRF-style adjacency prior hurts validation performance sharply

Therefore:

- `codebook512` should not be described as the universally best tokenizer
- `codebook512` remains justified as a downstream hypothesis branch only: it tests whether a larger discrete vocabulary helps semantic controllability even when it does not minimize tokenizer reconstruction loss
- widening the codebook beyond `512` is not defensible on the present corpus

### 3.4 Search Strategy Design

The correct search strategy is **stage-wise and mixed**, not a single global
optimizer over the whole neuro-symbolic stack.

#### 3.4.1 Stage 1: Tokenizer screening

Use a **small discrete ablation grid** for:

- `codebook_size`
- `hidden_dim`
- `CoordConv on/off`
- `MRF penalty on/off`

Reason:

- these are structural design choices with clear interpretability
- the search space is already tiny
- grid-style ablation is preferable because the goal is causal attribution, not only best-score hunting

#### 3.4.2 Stage 2: Diffusion continuous tuning

Use **random search** or **BOHB/ASHA-style early-stopped search** for:

- `learning_rate`
- `model_channels`
- `condition_hidden_dim`
- `condition_num_gnn_layers`
- `cfg_dropout_prob`
- `alpha_logic`
- `min_snr_gamma`

Reason:

- Bergstra and Bengio show that random search is superior to grid search when only a few dimensions matter
- Hyperband is justified when full deep-learning training runs are expensive and early resource allocation matters
- BOHB is justified only after the search space is already narrowed, because the diffusion stage is expensive and highly conditional

Recommended practical design:

1. sample `12` to `20` truncated diffusion trials
2. train only `10` to `20` epochs for these coarse trials
3. keep the top `2` to `3` by held-out `val_total_loss`
4. retrain those finalists for the full `100` epochs

Do **not** use a dense Cartesian grid over continuous diffusion parameters.

#### 3.4.3 Stage 3: Coupled semantic-control ablations

Use **small factorial or one-factor-at-a-time sweeps** for:

- `puzzle_structure_dropout_prob`
- `puzzle_stage_conditioning_enabled`
- `puzzle_stage_topology_enabled`
- `puzzle_stage_semantics_loss_weight`

Reason:

- these are semantically meaningful ablations
- they have strong interaction effects
- the goal is explanatory evidence for Chapter 4, not only score maximization

#### 3.4.4 Why full Bayesian optimization is not the primary strategy here

Full Bayesian optimization is not the right first tool for this thesis because:

- the search space mixes categorical architectural toggles and continuous optimization parameters
- many axes are only meaningful conditionally on earlier stage choices
- the sample budget is too small for a wide expensive loop to be credible

Bayesian optimization becomes reasonable only after:

- the tokenizer is fixed
- diffusion capacity is bounded
- the search is reduced to a few continuous knobs

### 3.5 Experimental Protocol for Reproducible and Fair Comparison

#### 3.5.1 Data splitting

For all trainable stages:

- use a deterministic held-out split with `seed=42`
- use `validation_fraction=0.1`
- record `train_size`, `eval_size`, and `eval_split` in saved metrics

Important protocol note:

- diffusion runs started before 2026-04-19 did not satisfy this requirement
- those runs are exploratory only and must be rerun for thesis-final claims

#### 3.5.2 Checkpoint selection

- `VQ-VAE`: select by `val_loss`
- `diffusion`: select by held-out `val_total_loss`; also save `best_logic_model` by `val_solvability`
- `fast_sampler`: select by `val_decode_ce_loss`
- `masked_room`: select by `val_loss` unless a topology-only ablation explicitly changes the target metric

#### 3.5.3 Final evaluation layer

Model-selection metrics are not the thesis endpoint. Final claims must be made
on external generation and validation artifacts:

- fixed-graph multi-seed audit
- manual side-by-side export
- matched-budget topology benchmark
- room-branch benchmark
- P-CBS / hybrid playability evaluation

This separation is essential. It protects the thesis from the criticism that
the same held-out split was tuned repeatedly and then reused as the sole source
of evidence.

#### 3.5.4 Number of runs

Minimum acceptable standard:

- tokenizer screening ablations: current single-seed results may be reported as screening evidence
- final tokenizer comparison: rerun the finalists with at least `3` seeds if compute allows
- final diffusion branch comparison: at least `3` full reruns per branch
- report `95%` confidence intervals over matched seeds / matched graphs

If compute is tight, prioritize repeated runs for:

- `baseline256` diffusion
- `codebook512` diffusion
- final staged semantic branch

#### 3.5.5 Statistical validation

Use:

- mean and `95%` bootstrap confidence interval
- paired permutation test or Wilcoxon signed-rank test on matched graphs/seeds
- effect size, not only `p`-value

Avoid:

- unpaired tests on small matched design slices
- declaring superiority from overlapping or unstable confidence intervals

#### 3.5.6 Control variables for fair comparison

Keep fixed when comparing branches:

- train/validation split
- random seed set
- number of epochs
- optimizer family
- runtime guidance defaults
- symbolic repair settings
- fixed-graph audit seeds
- hybrid validator budget

The baseline and codebook-512 diffusion branches must differ only in the
tokenizer checkpoint if the goal is to isolate tokenizer capacity effects.

## Chapter 4: Results and Discussion

### 4.1 Result Presentation Structure

Report results in four layers.

#### Table A: VQ-VAE ablations

Columns:

- variant
- `codebook_size`
- `hidden_dim`
- `CoordConv`
- `MRF weight`
- best epoch
- best `val_loss`
- best `val_recon_loss`
- best `val_perplexity`
- codebook utilization
- EMA live-code rate

Current local result:

- the best tokenizer by validation loss is `codebook256 + hidden96 + CoordConv + MRF`

#### Table B: Diffusion branch comparison

Columns:

- tokenizer branch
- held-out `val_total_loss`
- held-out `val_diffusion_loss`
- fixed-graph repair rate
- overwrite rate
- anchor error
- hybrid contract pass rate
- P-CBS success rate
- generation time

Important wording:

- do not mix interim pre-patch diffusion validation with final branch ranking
- if codebook-512 improves downstream controllability, frame that as a downstream trade-off, not as evidence that the tokenizer is globally better

#### Table C: Semantic-control ablations

Columns:

- puzzle structure dropout
- stage tokens on/off
- stage topology on/off
- stage semantics loss weight
- fixed-graph control accuracy
- P-CBS cognitive gap rate
- repair dependence

#### Table D: Failure analysis

Include:

- representative good cases
- representative failure cases
- whether failure is neural, symbolic, or validator-budget related

### 4.2 What Can Already Be Claimed from Current Evidence

Claims already supported:

- tokenizer performance is sensitive to `CoordConv` and the MRF-style local legality prior
- increasing tokenizer capacity from `256` to `512` does not automatically improve held-out tokenizer reconstruction
- `codebook512` is therefore a downstream hypothesis branch, not the tokenizer default by reconstruction evidence
- the diffusion model is operating in a small-data high-capacity regime, so regularization and strict evaluation separation are necessary

Claims that are **not** yet safe until rerun:

- any final ranking between the currently running baseline and codebook-512 diffusion branches based on their in-training validation logs
- any claim that diffusion generalization improved because of the pre-patch logged `val_total_loss`

### 4.3 Interpretation Rules

Use these rules in the text:

- if one branch wins tokenizer validation loss but another wins downstream controllability, say the pipeline exhibits a **rate-distortion versus controllability trade-off**
- if confidence intervals overlap meaningfully, say the methods are **comparable** rather than superior
- if a branch requires much more repair, treat that as a real weakness even if its raw neural outputs look better
- if a method improves P-CBS but not the hard oracle, interpret that as improved readability or bounded-rational navigation, not guaranteed mechanical solvability

### 4.4 Anticipated Reviewer Criticisms and Defenses

#### Criticism 1: "The hyperparameter search is biased."

Defense:

- the search space is declared in advance and is small
- structural ablations are separated from continuous optimization tuning
- final claims rely on external fixed-graph multi-seed audits, not only the model-selection metric

#### Criticism 2: "The search space is too narrow."

Defense:

- the narrowness is deliberate because the dataset is tiny and the model is already large
- widening the space further would be more vulnerable to overfitting than informative
- the ranges are bounded by both literature defaults and local capacity guardrails

#### Criticism 3: "You overfit the validation set."

Defense:

- diffusion now uses an explicit held-out validation split
- final evaluation is separated from model selection
- matched fixed-graph audits and repeated seeds reduce single-split optimism

#### Criticism 4: "Why run codebook-512 downstream if it is not the best tokenizer?"

Defense:

- because the downstream hypothesis is not reconstruction alone
- larger codebooks can preserve rarer semantic distinctions and may help control fidelity even if reconstruction validation loss is worse
- comparing `baseline256` against `codebook512` isolates tokenizer capacity effects better than comparing two nearly equivalent `256`-code models

#### Criticism 5: "Single-seed ablations are weak."

Defense:

- agree for final branch claims
- use current single-seed VQ-VAE runs as screening evidence only
- reserve final superiority language for multi-seed branch reruns

## Mandatory Remaining Jobs Before Final Thesis Lock

1. Rerun `baseline256` diffusion from scratch under the patched held-out validation protocol.
2. Rerun `codebook512` diffusion from scratch under the patched held-out validation protocol.
3. Keep the rest of the downstream stack fixed while comparing those two branches.
4. Run the fixed-graph multi-seed audit on the rerun checkpoints.
5. Report confidence intervals and paired significance on the fixed-graph outputs.
6. Keep the current six VQ-VAE results as tokenizer-screening evidence; if time allows, rerun the top tokenizer finalists with `3` seeds for stronger Chapter 4 reporting.

## References

- Bergstra, J., and Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. JMLR. https://www.jmlr.org/papers/v13/bergstra12a.html
- Snoek, J., Larochelle, H., and Adams, R. P. (2012). Practical Bayesian Optimization of Machine Learning Algorithms. NeurIPS. https://papers.nips.cc/paper/4522-practical-bayesian-optimization
- Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., and Talwalkar, A. (2018). Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization. JMLR. https://www.jmlr.org/beta/papers/v18/16-558.html
- Falkner, S., Klein, A., and Hutter, F. (2018). BOHB: Robust and Efficient Hyperparameter Optimization at Scale. ICML. https://proceedings.mlr.press/v80/falkner18a.html
- Kingma, D. P., and Welling, M. (2013). Auto-Encoding Variational Bayes. https://arxiv.org/abs/1312.6114
- van den Oord, A., Vinyals, O., and Kavukcuoglu, K. (2017). Neural Discrete Representation Learning. https://papers.nips.cc/paper/7210-neural-discrete-representation-learning.pdf
- Ho, J., Jain, A. N., and Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. https://papers.nips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf
- Nichol, A. Q., and Dhariwal, P. (2021). Improved Denoising Diffusion Probabilistic Models. https://proceedings.mlr.press/v139/nichol21a.html
- Ho, J., and Salimans, T. (2022). Classifier-Free Diffusion Guidance. https://arxiv.org/abs/2207.12598
- Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. https://arxiv.org/abs/2112.10752
- Rampasek, L., Galkin, M., Dwivedi, V. P., Luu, A. T., Wolf, G., and Beaini, D. (2022). Recipe for a General, Powerful, Scalable Graph Transformer. https://arxiv.org/abs/2205.12454
- Hang, T., Gu, S., Li, C., Bao, J., Chen, D., Hu, H., Geng, X., and Guo, B. (2023). Efficient Diffusion Training via Min-SNR Weighting Strategy. https://arxiv.org/abs/2303.09556
- Cohen, M., Quispe, G., Le Corff, S., Ollion, C., and Moulines, E. (2022). Diffusion bridges vector quantized variational autoencoders. https://proceedings.mlr.press/v162/cohen22b.html
