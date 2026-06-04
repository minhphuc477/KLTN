# Architecture Audit Research Notes

Last cleaned: 2026-06-04.
Basic ML engineering pass: 2026-06-05.
Core math re-audit pass: 2026-06-05.

These notes are a current audit ledger for H-MOLQD, a neuro-symbolic dungeon
generation system combining VQ-VAE/FSQ latents, graph-conditioned latent
diffusion, GPS/GAT-style graph conditioning, LogicNet differentiable
pathfinding, symbolic WFC repair, and solver/persona evaluation. This document
records defensible architecture claims, known limitations, and the tests that
currently support the claims. It is not an experimental-results substitute.

## Current SOTA Anchors

- **DiT backbones:** Peebles and Xie, "Scalable Diffusion Models with
  Transformers", https://arxiv.org/abs/2212.09748.
  Use this as support for DiT as a scalable diffusion backbone, not as evidence
  that this codebase's DiT is publication-ready without conditioning ablations.
- **Flow Matching / Rectified Flow family:** Lipman et al., "Flow Matching for
  Generative Modeling", https://arxiv.org/abs/2210.02747.
  Flow Matching is an objective-family change; sampling must use the matching
  flow ODE path, not DDPM/DDIM buffers by accident.
- **Discrete Flow Matching:** Gat et al., "Discrete Flow Matching", NeurIPS
  2024, https://proceedings.neurips.cc/paper_files/paper/2024/hash/f0d629a734b56a642701bba7bc8bb3ed-Abstract-Conference.html.
  This is a future categorical-generation baseline; it is not a drop-in
  replacement for the current continuous VQ-latent objective.
- **Few-step diffusion:** Song et al., "Consistency Models",
  https://arxiv.org/abs/2303.01469, and Luo et al., "Latent Consistency
  Models", https://arxiv.org/abs/2310.04378.
  Few-step claims require a trained/distilled consistency backend.
- **Tokenizer upgrades:** FSQ is a real ablation path; LFQ/MAGVIT-v2-style
  tokenizers remain research-track. See MAGVIT-v2 / "Language Model Beats
  Diffusion -- Tokenizer is Key to Visual Generation",
  https://arxiv.org/abs/2310.05737, and FSQ,
  https://arxiv.org/abs/2309.15505.
- **Graph conditioning:** GraphGPS is the correct comparison point for local
  message passing plus global attention:
  https://arxiv.org/abs/2205.12454. RRWP-style edge features should be
  consumed by edge-aware GAT/GPS paths or rejected for GCN/SAGE.
- **Graphormer:** Ying et al., "Do Transformers Really Perform Bad for Graph
  Representation?", NeurIPS 2021, https://arxiv.org/abs/2106.05234.
  Graphormer-style shortest-path and edge attention biases are a clean future
  comparison for key-lock dependency conditioning.
- **Constraint guidance:** Diffusion Posterior Sampling motivates
  sampling-time external-gradient guidance:
  https://arxiv.org/abs/2209.14687. LogicNet guidance is an inference-time
  latent update unless explicitly trained end-to-end through a differentiable
  loss.
- **Posterior/proximal sampling caution:** DPPS targets restoration by
  selecting measurement-consistent candidates during denoising
  (https://arxiv.org/abs/2402.16907), while theoretical work shows exact
  posterior sampling can be computationally intractable in general
  (https://arxiv.org/abs/2402.12727). Treat this as a constrained-sampling
  research direction, not a current correctness claim.
- **Planning losses:** Value Iteration Networks and Neural Bellman-Ford
  Networks are the relevant differentiable planning precedents:
  https://arxiv.org/abs/1602.02867 and https://arxiv.org/abs/2106.06935.

## Fixed Audit Items

- **DiT conditioning gap:** `DiTDenoiser` now has spatial graph conditioning
  and context-token topology refinement. The remaining paper requirement is an
  ablation showing graph/spatial tokens change outputs and improve metrics.
- **RRWP silent failure:** `GlobalStreamEncoder` rejects RRWP with GCN/SAGE,
  and GPS/GAT paths consume projected RRWP edge features.
- **Batched GPS leakage:** GPS global attention now respects `batch_idx` so
  nodes from separate graphs do not attend to each other in a packed batch.
- **Dual-stream feature collapse:** style and reference-room features are kept
  separate before projection instead of being summed.
- **Unreachable graph distances:** current-node distance features use a
  sentinel for unreachable nodes rather than aliasing them with max-distance
  reachable nodes.
- **Strict checkpoint metadata:** strict checkpoint mode now fails when
  topology-conditioning metadata is missing instead of silently falling back to
  legacy additive conditioning.
- **VQ-VAE geometry:** the decoder produces native Zelda room geometry
  `(16, 11)` without relying on resize correction.
- **VIN pathfinder scale:** source cells are anchored at zero and VIN distances
  are clamped to the Bellman-Ford infinity scale.
- **Legacy `src.ml.logic_net.SoftBellmanFord`:** compatibility path now uses
  soft distance relaxation with wall costs instead of saturating probability
  mass flood fill.
- **CBS belief map:** the persona simulator now keeps a categorical posterior
  over tile IDs; scalar `tile_type/confidence` is the MAP view, not the entire
  belief state.
- **Validation repair metrics:** validation now reports raw hard solvability
  separately from `val_hard_solvability_after_repair`,
  `val_logicnet_score_after_repair`, and `val_neural_repair_success_rate`.
  Repaired metrics run through `NeuralGuidedRepair` instead of scoring only
  pre-repair generated samples.
- **Neighbor-latent detach:** inference neighbor latents remain detached by
  design and are documented as greedy autoregressive context, not
  differentiable multi-room stitching.
- **Flow-matching loss weighting:** `flow_matching_loss` now computes
  per-sample velocity MSE and applies continuous-time Min-SNR weighting before
  reducing the batch.
- **Neural repair floor masks:** neural-guided repair now skips LogicNet floor
  target resolution when `graph_data` is absent and transposes reversed
  `[W,H]` masks before interpolation.
- **Neural repair eval-state restoration:** `logic_net.eval()` is now inside
  the guarded repair call, so a failing eval hook still restores the previous
  training state.
- **Unified diffusion objective dispatch:** `LatentDiffusionModel` stores
  `training_objective`, exposes `compute_loss()`, and construction paths pass
  checkpoint/config objective metadata into the model instead of relying only
  on trainer-side branching.
- **Symbolic A* cost maps:** LogicNet-derived repair cost maps are clipped to a
  minimum step cost of `1.0`, preserving Manhattan heuristic admissibility.
- **WFC required-floor protection:** local WFC reset masks exclude forced floor
  cells so post-collapse floor overwrites do not violate adjacency constraints.
- **Latent inpainting orientation:** neighbor-boundary inpaint constraints now
  transpose reversed latent `[H,W]` axes before resizing, and latent edit masks
  transpose only when room-mask aspect is clearly reversed relative to the
  target latent aspect.
- **DataLoader worker RNG seeding:** shared training dataloader kwargs now add a
  top-level `worker_init_fn` whenever `num_workers > 0`. The hook derives NumPy
  and Python `random` seeds from PyTorch's per-worker `torch.initial_seed()`,
  avoiding repeated external-library RNG streams in multi-process loading.
- **Per-graph batch assignment metadata:** Zelda graph collation preserves the
  existing list-of-graphs API but now injects a per-sample all-zero `batch_idx`
  tensor when graph node features are present. This does not convert the batch
  into one packed PyG graph; it prevents downstream single-graph encoders from
  silently defaulting missing batch metadata.
- **cuDNN runtime flags:** `runtime.cudnn_benchmark` and
  `runtime.cudnn_deterministic` are explicit config keys. `seed_everything()`
  now sets `torch.backends.cudnn.benchmark` only when deterministic cuDNN mode
  is disabled.
- **Flow-matching endpoint loss:** rectified-flow velocity training no longer
  applies DDPM Min-SNR-gamma weighting. The previous weighting sent the clean
  endpoint (`t -> 0`) contribution toward zero when `min_snr_gamma > 0`,
  suppressing velocity learning where the data manifold is visible.
- **VQ commitment scale:** VQ codebook and commitment losses now sum squared
  error across latent embedding channels per token before averaging spatial
  tokens. This keeps the latent commitment term on a comparable per-token scale
  instead of diluting it by `embedding_dim`.
- **VQ EMA warmup:** `VectorQuantizer` has an early-update EMA decay warmup so
  the codebook can move quickly away from random initialization before settling
  into the configured long-horizon decay.
- **FSQ saturation regularization:** `FSQuantizer` now reports and optimizes a
  small saturation penalty on pre-`tanh` latents outside the useful scalar
  quantization range. This keeps the FSQ ablation from silently driving
  pre-quantization activations into zero-derivative saturation.
- **LogicNet unreachable sentinel preservation:** grid and graph soft-min
  Bellman-Ford updates keep nodes/cells at `inf_distance` when every candidate
  is still sentinel-valued. This prevents log-sum-exp from gradually lowering
  unreachable regions into fake reachability.
- **Isolated graph regression:** GAT graph conditioning is now covered by a
  one-node zero-edge regression test. Current loaders already emit empty edge
  tensors rather than `None`, and the encoder must return finite embeddings for
  isolated rooms.
- **P-CBS locked-door contact:** failed conditional-door moves now refresh
  durable `DOOR` memory as well as affordance memory, and impossible locked
  doors carry direct risk when the agent lacks the required key, bomb, or boss
  key. This reduces timeout loops caused by memory decay erasing blocked gates.
- **WFC pseudo-label loss scaling:** WFC repaired pseudo-label CE now enters
  the training objective as a full-batch mean (`sum CE over repaired samples /
  BHW`) instead of a mean over only the repaired subset. This prevents a
  `B/K` gradient amplification when only a few samples are repairable.
- **WFC pseudo-label metrics:** training now tracks `wfc_pseudo_loss` as the
  repaired-sample CE averaged over repaired samples, plus
  `wfc_pseudo_loss_contribution` for the scaled objective contribution and
  `wfc_pseudo_total_samples` for denominator auditing. Empty-repair batches no
  longer dilute the reported repaired-sample loss.
- **AdamW parameter decay:** diffusion training now splits optimizer groups so
  matrix/tensor weights receive configured weight decay while biases and 1D
  scale parameters, including norm weights, use `weight_decay=0.0`.
- **Directed structural metrics:** structural topology analysis now preserves
  directed dead-end semantics by counting directed sources/sinks instead of
  collapsing every metric through `to_undirected()`.
- **CBS graph proxy directionality:** graph-mode CBS fitness no longer converts
  undirected legacy graphs into bidirectional `DiGraph` objects before using
  directed out-degree. Directed mission graphs keep out-degree dead-end
  pressure, and undirected graphs use undirected degree.
- **Bounded confusion normalization:** normalized confusion ratios are clipped
  to `[0, 1]` in the shared search-benchmark utility and the graph proxy uses
  that helper instead of an unbounded hand-rolled formula.
- **P-CBS heuristic scale:** goal- and item-seeking heuristics now use bounded
  one-step absolute progress in `[-1, 1]`, so long-distance goal pursuit does
  not vanish relative to constant curiosity scores.
- **P-CBS forgetting threshold:** belief decay now applies forgetting
  thresholds to the decayed confidence before posterior resynchronization, so
  the posterior's uniform floor cannot make `UNKNOWN` unreachable.

## Remaining Publication Risks

- **Flow/DiT objective and sampler parity:** flow-trained checkpoints must be
  evaluated with the matching flow ODE sampler. Do not report DDPM/DDIM metrics
  for a flow objective as if the objective and sampler are aligned.
- **Gradient accumulation:** the main diffusion trainer still lacks a
  first-class gradient-accumulation loop. Low-VRAM ablations need smaller
  batches or a follow-up trainer patch.
- **AMP / Accelerate:** training still does not have a unified AMP or
  HuggingFace Accelerate path. Adding this safely requires moving optimizer
  stepping out of per-batch trainer methods and validating EMA, gradient
  clipping, distributed reduction, and nonfinite-batch handling together.
- **Metadata-safe D4 augmentation:** Zelda room tensors have an existing
  transform hook, but graph metadata includes boundary constraints, neighbor
  maps, topology maps, and room-position features. Random flips/rotations must
  rotate all of that metadata consistently before they are enabled for
  graph-conditioned training.
- **Gradient checkpointing:** DiT/U-Net activation checkpointing is still a
  memory-scaling TODO. It should be enabled behind a config flag and tested
  against dropout/RNG behavior and PAG/attention capture paths.
- **Safe checkpoint format:** checkpoint loading uses the local
  `safe_torch_load()` path, but trainer checkpoints are still `.pth` bundles
  containing optimizer/scheduler state. A `safetensors` migration should be
  dual-format because optimizer metadata is not tensor-only model weights.
- **Dead graph helper:** `DiffusionTrainer._build_logic_graph_data()` remains
  private dead code. Either wire it into graph-data construction or remove it
  after confirming no downstream scripts call it.
- **Global graph supervision coverage:** room-level training can only supervise
  full global graph losses when a complete dungeon-room batch provides one
  passability value per graph node. Skipped graph losses must be reported.
- **LogicNet proxy-vs-hard metric gap:** soft LogicNet losses and hard
  solver-based solvability are separate evidence streams. Paper tables should
  include both and should not equate `exp(-logic_loss)` with playability.
- **DPS/LogicNet guidance scope:** guidance detaches graph tensors and updates
  latents. Claims should say inference-time latent guidance unless a training
  loss demonstrably backpropagates through the graph-conditioned denoiser.
- **WFC pseudo-labels:** `alpha_wfc_pseudo` is opt-in. If disabled, do not
  claim WFC pseudo-label distillation contributes to training.
- **Tokenizer SOTA:** FSQ exists as an ablation. LFQ/RVQ are not implemented
  default paths and require new ablation tables before being claimed.

## Constraint-Guided Generation Protocol

DPPS/DPS-style LogicNet guidance is a hypothesis until the gradient probe and
ablation ladder pass. Current implementation status:

- `scripts/gradient_probe.py` probes LogicNet compatibility-mode gradients on
  clean semantic room logits corrupted with VP-style Gaussian noise. It reports
  loss, score, gradient norm, finite rate, relative gradient norm, and
  walkability statistics by noise level.
- The first CPU smoke run
  `python scripts/gradient_probe.py --noise-levels 0,0.5,1.0 --samples-per-level 1 --num-iterations 2 --device cpu`
  returned finite gradients but relative gradient norms grew from `1.0` at
  clean input to roughly `4.9e2` at medium noise and `3.1e4` at pure noise.
  This supports late-stage LogicNet guidance as the first ablation, not
  full-trajectory DPPS.
- Before sampler integration, run the probe with the trained LogicNet/VQ-VAE
  checkpoint path and at least 8 samples per noise level. A stable guidance
  window should require finite gradients plus bounded relative norms across
  seeds.
- Required ablations remain: no guidance, post-hoc WFC repair, late-stage
  LogicNet guidance, full-trajectory LogicNet guidance, and any retrained
  CFG/DFM variant. Report solvability, hard constraint violations,
  distribution distance, and seconds per room separately.
- Runtime ablation configs now exist for both LogicNet guidance windows:
  `configs/ablation_inference_guidance_only.yaml` sets
  `generation.logic_guidance_strategy: late` with a `0.2` reverse-process
  active fraction, while `configs/ablation_inference_guidance_full_dpps.yaml`
  sets `generation.logic_guidance_strategy: full`. Use the full variant only
  as a stress ablation because the noisy-gradient probe showed high-noise
  relative-gradient explosion.

## Current Verification Ledger

Focused suites run during the latest audit pass:

- `python -m pytest tests/test_cognitive_bounded_search.py tests/test_cbs_full.py -q`
  passed with 75 tests.
- `python -m pytest tests/test_ml_components.py -q` passed with 55 tests.
- `python -m pytest tests/test_train_diffusion_conditioning_shapes.py -q`
  passed with 41 tests.
- `python -m pytest tests/test_hmolqd/test_vqvae.py -q` passed with 24 tests.
- `python -m pytest tests/test_critical_review_fixes.py -q` passed with 14 tests.
- `python -m pytest tests/test_logicnet_fixes.py -q` passed with 17 tests.
- `python -m pytest tests/test_advanced_architecture_ablations.py -q` passed
  with 16 tests.
- `python -m pytest tests/test_hmolqd/test_logic_net.py -q` passed with 17 tests.
- `git diff --check` passed; only Git line-ending warnings were reported.

Latest targeted additions:

- `python -m pytest tests/test_advanced_architecture_ablations.py::test_flow_matching_loss_applies_per_sample_continuous_min_snr_weight tests/test_neural_guided_repair.py::test_neural_guided_repair_omits_logic_floor_mask_without_graph_data tests/test_neural_guided_repair.py::test_resize_mask_transposes_reversed_spatial_axes -q`
  passed with 3 tests.
- `python -m pytest tests/test_hmolqd/test_symbolic_refiner.py::TestPathAnalyzer::test_cost_map_normalization_preserves_astar_admissibility tests/test_hmolqd/test_symbolic_refiner.py::TestSymbolicRefiner::test_repair_room_excludes_required_floor_mask_from_wfc_reset -q`
  passed with 2 tests.
- `python -m pytest tests/test_train_diffusion_conditioning_shapes.py::test_validate_reports_post_repair_solvability_metrics tests/test_train_diffusion_conditioning_shapes.py::test_diffusion_objective_loss_delegates_to_model_compute_loss tests/test_neural_guided_repair.py::test_logicnet_eval_failure_restores_training_state tests/test_advanced_architecture_ablations.py::test_latent_diffusion_compute_loss_dispatches_configured_objective -q`
  passed as part of the focused suites above.
- `python -m pytest tests/test_repair_feedback.py -q` passed with 7 tests,
  including reversed latent-neighbor and reversed room-mask orientation
  regressions.
- `python -m pytest tests/test_neural_guided_repair.py tests/test_train_diffusion_conditioning_shapes.py tests/test_advanced_architecture_ablations.py -q`
  passed with 63 tests.
- `python -m pytest tests/test_noisy_logicnet_gradients.py -q` passed with 3
  tests.
- `python -m pytest tests/test_logicnet_gradient_flow.py tests/test_logicnet_fixes.py -q`
  passed with 25 tests.
- `python scripts/gradient_probe.py --noise-levels 0,0.5,1.0 --samples-per-level 1 --num-iterations 2 --device cpu`
  completed and emitted JSON probe statistics plus a late-stage guidance
  recommendation.
- `python -m pytest tests/test_config_system.py tests/test_runtime_logic_guidance_strategy.py tests/test_noisy_logicnet_gradients.py -q`
  passed with 39 tests.

## Required Reporting Discipline

- Report ablations by feature flag and checkpoint metadata, not by aspirational
  architecture name.
- Keep hard solver metrics, differentiable LogicNet metrics, WFC repair rates,
  and persona-simulator metrics in separate table columns.
- Any fallback path used during generation must be exported as a diagnostic
  count. Silent fallback is a publication bug.
- Claims about Bayesian agents must reference the categorical posterior
  implementation, not scalar confidence alone.
