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
- **External PCG baselines:** the implemented comparison baselines now anchor
  to the PCGML survey (https://arxiv.org/abs/1702.00539), VGLC corpus
  (https://arxiv.org/abs/1606.07487), MarioGPT/text-to-level LLMs
  (https://arxiv.org/abs/2302.05981), TOAD-GAN token-level GAN generation
  (https://arxiv.org/abs/2008.01531), and WFC as constraint solving plus
  pattern learning (https://doi.org/10.1109/TG.2021.3076368). These are
  baseline anchors, not evidence that any short local dry-run has converged.

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
- **FSQ bounded projection:** `FSQuantizer` no longer uses a pre-rounding
  `tanh` squash. It clamps to the finite scalar range with a straight-through
  gradient and keeps the saturation penalty as a diagnostic/regularizer for
  out-of-range activations.
- **RRWP width mismatch preservation:** GPS/GAT edge RRWP projections are
  aligned to the hidden edge width instead of being dropped when a projection
  or checkpoint produces a narrower/wider edge feature tensor. Row-count
  mismatches are still rejected because they indicate an invalid edge mapping.
- **LogicNet fallback source gradients:** when topology metadata does not
  provide a semantic source, Bellman-Ford/VIN modes use a differentiable soft
  single-source distribution over walkability. Perturb-and-MAP keeps a hard
  anchored source because its discrete shortest-path estimator requires one.
- **LogicNet guidance domain:** the main DDPM, DDIM, and flow sampling paths
  apply LogicNet guidance to a clean-latent estimate (`pred_x0`/`x0_hat`) before
  rebuilding the posterior or velocity step. The older low-level
  `GradientGuidance.apply_guidance()` helper is retained as a diagnostic API
  and should not be used as evidence for the main sampler formulation.
- **FSQ index decoding:** FSQ now works as an implicit scalar codebook in the
  same `decode_indices()` path used by VQ. Index tokens are decomposed into
  per-dimension scalar digits, mapped back to quantized values, projected to the
  public latent width, and decoded without a manual architecture trap.
- **DPO distributed call path:** `LatentDiffusionModel.forward()` dispatches
  `forward_mode` values such as `compute_loss` and `dpo_preference_loss`, so
  DPO fine-tuning can call `model(...)` and trigger DDP/Accelerate hooks. The
  DPO experiment loop now uses Accelerate for prepare/backward/clip/save while
  retaining a single-process fallback.
- **Graph padding masks:** GPS global attention accepts `node_mask` and uses it
  as a `key_padding_mask`, while cross-attention uses finite masked scores plus
  post-softmax renormalization. Fully masked graph contexts stay finite instead
  of producing `softmax(-inf)` NaNs.
- **Generation RRWP wiring:** generation-time `_prepare_graph_context()` now
  computes and returns `edge_rrwp`, and room generation forwards both `edge_rrwp`
  and `node_mask` into the condition encoder.
- **Room-context RRWP preservation:** `_build_room_graph_context()` now carries
  `edge_rrwp` through from generation graph data into the per-room conditioning
  dictionary, so inference does not silently drop RRWP after computing it.
- **Finite masked attention logits:** graph-conditioning fusion, latent
  topology refinement, latent cross-attention, and graph-to-grid attention now
  use large finite mask biases instead of `-inf` on padded nodes. Fully masked
  graph rows are covered by finite-backward regressions.
- **Bounded spatial alignment gradients:** graph-to-grid spatial alignment no
  longer uses `-log(attention.clamp_min(1e-8))`. The loss uses a bounded
  probability-to-logit BCE surrogate with a `1e-4` floor, preventing missed-node
  supervision from producing `1e8` local slopes.
- **EMA VQ codebook optimizer isolation:** EMA quantizers freeze
  `embedding.weight` and update the codebook with in-place `copy_()` under
  `torch.no_grad()`, preventing AdamW momentum from competing with EMA updates.
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
- **External baseline experiment harness:** `experiments/baselines/` now
  contains isolated LLM, DCGAN, and overlapping-pattern WFC scripts. All three
  scripts support `--dry-run`, emit JSON reports, share the same hard-oracle
  evaluation path, and report terminal injection/normalization rates so
  standalone room validation does not silently reward generated grids that lack
  START/TRIFORCE semantics. The LLM baseline trains next-token CE on flattened
  `[H*W]` room tokens; the GAN baseline uses straight-through Gumbel-Softmax
  during generator training and reserves `argmax` for `torch.no_grad()`
  sampling; the WFC baseline is local and dependency-free.
- **Strict evidence audit for baseline reports:** fixed-graph protocol
  aggregation now reports teacher-fallback rates, and
  `scripts/compare_protocol_to_baselines.py` emits a strict evidence audit that
  flags teacher fallback, repair-heavy outputs, and overlay-assisted semantic
  gains before a result can be described as standalone neural-generator
  evidence. The audit also blocks standalone-neural publication claims when
  hard oracle metrics are absent, when hard solvability is reported only
  post-repair, or when repaired runs lack raw/pre-repair hard oracle rates.

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
- **Baseline convergence risk:** the new baseline scripts are reproducibility
  scaffolds, not finished results. Dry-run reports are smoke tests only. Paper
  tables require matched sample counts, fixed seeds, identical validation
  timeout budgets, explicit `--run-pcbs` where P-CBS metrics are claimed, and
  separate reporting of `terminal_normalization_rate`.

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
- `python -m pytest tests/test_protocol_reporting.py::test_protocol_baseline_report_flags_fallback_repair_and_overlay_evidence tests/test_protocol_reporting.py::test_fixed_graph_audit_aggregate_tracks_post_overlay_semantic_error -q`
  passed with 2 tests.
- `python -m pytest tests/test_ablation_scripts.py -q` passed with 4 tests.

## Required Reporting Discipline

- Report ablations by feature flag and checkpoint metadata, not by aspirational
  architecture name.
- Keep hard solver metrics, differentiable LogicNet metrics, WFC repair rates,
  and persona-simulator metrics in separate table columns.
- Any fallback path used during generation must be exported as a diagnostic
  count. Silent fallback is a publication bug.
- Claims about Bayesian agents must reference the categorical posterior
  implementation, not scalar confidence alone.

## 2026-06-17 Verification Addendum

This pass rechecked the final vulnerability ledger against the current
workspace instead of treating old audit text as ground truth.

Fixed and verified:

- VQ-VAE no-checkpoint fallback now uses the project codebook contract
  (`256`) instead of the stale `512` value.
- Torch import guards now suppress only `ImportError`; broken Torch runtime
  errors are no longer hidden as optional dependency absence.
- End-to-end export timing now synchronizes CUDA before and after timed
  generation blocks.
- Priority research subprocess timeouts now terminate process trees.
- Dungeon 9 holdout GAN/profiling helpers and MaskGIT tests no longer mutate
  global Torch RNG state.
- D* Lite and parallel A* localized fallback paths now use parent pointers
  instead of copying full path lists per expansion.
- Parallel A*, D* Lite predecessor inversion, and Bidirectional A* inverse
  transitions now match canonical item/bomb/boss-key mechanics.
- PyG-less batched graph conditioning now has a local dense-batch fallback, so
  graph tokens are isolated per sample instead of copied across the batch.
- Collision-prone advanced rules now route token/switch/reward/skill-chain
  node placement through bounded free-position selection.
- Stochastic advanced-rule integration checks now skip unexercised features
  explicitly rather than passing silently.

Scientific boundary resolution:

- `graphormer` remains the checkpoint-compatible static shortest-path-bias
  ablation. `graphormer_learned` is a separate, opt-in ablation with learned
  in/out-degree centrality encodings, per-head shortest-path distance bias,
  and learned edge-type bias. Reports must keep the two names separate. The
  learned mode is Graphormer-style rather than a claim of reproducing every
  training and dataset detail from Ying et al.
- MAP-Elites graph descriptors now use a progression-feasible state search.
  Critical-path length, consumed keys, collected keys, and key surplus are
  measured on a route that satisfies consumable locks and persistent item
  gates. If no feasible macro route exists, the evaluator does not claim that
  graph descriptors were used and falls back to the explicitly named legacy
  grid proxy. Exported metrics include `graph_descriptor_feasible` and the
  path key-economy fields, so archive coverage and QD-score tables can state
  the descriptor definition they used.
- The discrete masked generator now exports requested and executed refinement
  steps, corrector rounds, committed-token counts, unresolved-token counts,
  schedule identity, and stochastic-mode diagnostics. Masked-room results
  must report these fields together with `teacher_fallback_used`; the branch
  is described as iterative masked-token generation, never autoregressive.

Remaining empirical boundary:

- These changes make the hypotheses testable; they do not establish that the
  learned Graphormer-style mode, feasible QD descriptors, or MaskGIT branch is
  superior. Publication claims still require fixed-seed paired ablations,
  confidence intervals, runtime and memory measurements, and no-teacher
  fallback generation runs. An unexecuted experiment is not evidence.

Focused verification commands from this pass:

- `python -m pytest tests/test_verified_vulnerability_fixes.py -q` passed with
  18 tests.
- `python -m pytest tests/test_discrete_masked_room_model.py::test_masked_concat_encoder_respects_context_node_mask tests/test_discrete_masked_room_model.py::test_masked_backbone_ignores_masked_context_tokens -q`
  passed with 2 tests.
- `python -m pytest tests/test_advanced_rules_integration.py::TestAdvancedRulesIntegration::test_fungible_key_economy tests/test_advanced_rules_integration.py::TestAdvancedRulesIntegration::test_big_room_merging tests/test_advanced_rules_integration.py::TestAdvancedRulesIntegration::test_cycle_valves tests/test_advanced_rules_integration.py::TestAdvancedRulesIntegration::test_visual_foreshadowing tests/test_advanced_rules_integration.py::TestAdvancedRulesIntegration::test_collection_challenge tests/test_advanced_rules_integration.py::TestAdvancedRulesIntegration::test_combat_arenas tests/test_advanced_rules_integration.py::TestAdvancedRulesIntegration::test_thematic_sectors tests/test_advanced_rules_integration.py::TestAdvancedRulesIntegration::test_entangled_branches tests/test_advanced_rules_integration.py::TestAdvancedRulesIntegration::test_hazard_gates tests/test_advanced_rules_integration.py::TestAdvancedRulesIntegration::test_virtual_room_layers -q`
  passed as 1 exercised test and 9 explicit skips.

## 2026-06-20 Round 2 Topology Audit

Confirmed implementation fixes:

- Cognitive topology descriptors now project directed mission exports onto a
  simple physical traversal graph before computing dead ends, degree spread,
  shortest paths, and cycle rank. Reciprocal directed corridor edges no longer
  create artificial loops.
- Evolutionary cognitive objectives use physical room degree. Corridors have
  degree 2, junctions have degree at least 3, and non-goal rooms with physical
  degree at most 1 are treated as dead ends.
- `LOCKED`, `ITEM_GATE`, and `BOSS_LOCKED` edges are physically bidirectional.
  Their requirements still gate traversal, but passing a gate does not turn a
  normal doorway into a one-way drop.
- `InsertLockKeyRule` attaches the key to a reachable side branch and places
  the lock later on the trunk path. This restores an actual retrieve-and-return
  pattern instead of making every key an unavoidable corridor pickup.
- Boss keys are reusable authorization items across boss doors. Small keys and
  bombs remain consumable. Forward, parallel, bounded-cognitive, D* Lite, and
  bidirectional transition logic now agree on this rule.

Focused verification:

- `python -m pytest tests/test_critical_review_fixes.py tests/test_verified_vulnerability_fixes.py -q -p no:cacheprovider --basetemp=tmp/pytest-round2`
  passed with 44 tests.
- `python -m pytest tests/test_pathfinding_unified_game_logic.py tests/test_quick_unified_logic.py -q`
  passed with 15 tests.
- `python -m pytest tests/test_evolutionary_director.py::TestEvolutionaryDirector::test_large_population -q`
  passed without relaxing its fitness threshold.
- `python scripts/run_pcbs_component_ablation.py --quick --quiet --output-dir results/pcbs_component_ablation_round2_smoke`
  completed, but the oracle solved 0 of 1 maps. The artifact is therefore
  marked invalid for component comparison and is diagnostic only.
- `python scripts/run_conditioning_logicnet_repair_ablation.py --quick --output results/conditioning_logicnet_repair_round2_plan`
  generated a one-seed, four-variant plan. It did not execute model inference.
- Full P-CBS ablation runs now exit nonzero after writing diagnostics when no
  evaluated map is oracle-solvable. This prevents an all-unsolved benchmark
  from being mistaken for evidence about bounded-agent components.

## Outstanding Empirical Evidence

The following are publication gaps, not completed results. Do not claim that
they have been validated until fresh result artifacts and manifests are
committed or archived:

- Human calibration and blinded preference data:
  `python scripts/run_ood_scaling_and_blinded_eval.py --num-samples 8 --blinded-per-condition 6`.
- Neural contribution versus deterministic repair:
  `python scripts/run_conditioning_logicnet_repair_ablation.py --execute --seeds 42,43,44`.
- Fixed-graph diversity, NCD, entropy, fallback, and overlay evidence:
  `python scripts/run_fixed_graph_multi_seed_audit.py --run-dir <run> --output-dir <output> --include-no-fallback-ablations`.
- P-CBS component and failure-mode breakdown:
  `python scripts/run_pcbs_component_ablation.py --levels 1,2,3 --variants 1,2 --persona novice`.
- Designer controllability, sample efficiency, and compute overhead still need
  centralized 100-room and 500-room stress tables with fixed hardware,
  checkpoint hashes, seeds, confidence intervals, failure counts, and repair
  rates. Existing code paths alone are not empirical evidence.

## 2026-06-20 Non-GUI Continuation

Confirmed fixes from the continued repository audit:

- All training entry points now reject overlapping train/test dungeon IDs.
  VQ-VAE, diffusion, masked-room, and fast-sampler runs cannot silently train
  on the configured holdout.
- Internal room-level validation now groups samples by dungeon when metadata
  exists. Rooms and quest variants from one dungeon remain in one split.
- VQ-VAE evaluation restores the caller's train/eval mode even when evaluation
  exits through an exception.
- DFS/IDDFS and the BFS/Dijkstra/Greedy comparison helpers reconstruct paths
  from parent links instead of copying full path lists per expansion.
- BFS, Dijkstra, and Greedy comparison runs now use the same canonical Zelda
  transition function as the hard oracle. Search algorithms no longer differ
  because one baseline ignored bomb doors, traversal items, blocks, or shutters.
- Hedgehog linear attention remains an opt-in ablation. Its feature softmax and
  sequence reductions now accumulate in FP32 under mixed precision.
- Advanced big-room generation is now initialized when enabled and tiles
  canonical room samples into patches without mutating trained model shapes.
  Patch overlaps use categorical selection instead of arithmetic tile-ID
  interpolation.

Important limitation:

- `ParallelAStarSolver` is currently a first-goal multiprocessing feasibility
  race with a shared closed set. It is not a strict optimal A* implementation
  and must not be used for shortest-path or optimality claims.

Focused verification added in this continuation includes training split guards,
grouped validation, solver semantic parity, parent-link path reconstruction,
mixed-precision Hedgehog attention, and big-room patch assembly.

## 2026-06-20 Model Architecture Audit

Confirmed implementation fixes:

- DiT blocks now implement AdaLN-Zero gating literally: zero-initialized gates
  produce an identity residual block instead of a half-strength residual update.
  This keeps the DiT ablation stable at initialization.
- Fresh U-Net and DiT denoiser heads now start as zero predictors. Tests that
  inspect routing explicitly open the relevant test gates/heads, so ablation
  sensitivity is not confused with unstable random initialization.
- Learned Graphormer centrality encodings now exclude synthetic self-loops.
  Self-loops remain available for stable attention, but centrality buckets
  describe real graph degree only.
- Graph-to-grid conditioning now applies the node mask before the lightweight
  GCN prepass and before degree-feature extraction. Padded nodes and invalid
  padded edges can no longer send topology messages into real graph nodes.
- Spatial graph-to-grid conditioning now has a default-off
  `graph_to_grid_edge_semantics` ablation. When enabled, edge labels/features
  add a learned per-head destination-node attention bias, so locked/open/boss
  edges can be compared against the edge-blind spatial-conditioning baseline.
- MaskGIT training step embeddings now track corruption level. High mask ratio
  maps to late reverse-refinement steps, and near-clean corruption maps to step
  zero, matching inference semantics.
- Graph-level macro validation no longer consumes boss keys in the graph and
  virtual-node branches. Boss keys are persistent authorization items; small
  keys and bombs remain consumable.
- `graphormer_learned_directed`, `graphormer_learned_semantic`, and
  `graphormer_learned_directed_semantic` are now explicit ablation modes.
  The undirected `graphormer_learned` mode remains the checkpoint-compatible
  learned Graphormer-style baseline.

Closed implementation gap, still requiring ablation evidence:

- MaskGIT topology helpers now route directional gate-family channels
  (`gate_key_*`, `gate_boss_*`, `gate_bomb_*`, and related families) into the
  edge-aware boundary logit bias. This makes edge semantics change the
  preferred door class without changing checkpoint parameter shapes. It remains
  an ablation hypothesis until compared against the room-topology-only
  baseline.

Focused verification:

- `python -m pytest --basetemp .tmp/pytest-arch2 tests/test_advanced_architecture_ablations.py tests/test_ml_components.py tests/test_discrete_masked_room_model.py tests/test_critical_review_fixes.py -q`
  passed with 159 tests.
- `python -m compileall -q src/core/latent_diffusion.py src/core/graph_grid_attention.py src/core/discrete_masked_model.py src/simulation/validator.py`
  passed.

## 2026-06-23 Gap Implementation Pass

Confirmed fixes:

- Mission-grammar lock/key and resource-gate reachability now uses a
  progression-aware fixed point instead of raw BFS. Mutual softlocks where key
  A is behind lock B and key B is behind lock A are rejected before Map-Elites
  archive insertion.
- Standalone resource-loop validation now uses the same gate-aware pre-gate
  reachability and still accepts valid early-provider / late-provider dungeons.
- Room-level macro solver switch checks reconstruct `GameState` with
  `bomb_count` and current-room defeated enemies, removing the undefined
  `_GameState` path and matching the room-level abstraction that rooms are
  cleared before shutter traversal.
- Differentiable tortuosity now relaxes via the minimum cardinal neighbor
  distance. The previous convolution summed neighbors and was not a
  shortest-path relaxation.
- Learned A* heuristic calibration no longer claims that a constant 0.9 scale
  proves admissibility. The API can subtract the maximum observed held-out
  overestimate margin; scale-only use is documented as conservative shrinkage.
- LCM distillation auxiliary `pred_loss` now compares converted x0 predictions
  rather than forcing the student to match the teacher's raw local ODE tangent.
- DiT `dit_activation_type` and `dit_norm_type` are exposed through the
  validated config schema and CLI. `scripts/generate_model_architecture_ablation_manifest.py`
  generates plan-only or executable manifests for U-Net-vs-DiT and
  RMSNorm/SwiGLU DiT ablations.

Research boundary:

- Graphormer-style edge/spatial/centrality encodings are supported as
  controlled ablations, consistent with the Graphormer paper's structural
  encoding premise. FlashAttention/SDPA remains a kernel efficiency choice and
  should be reported as runtime optimization, not a new model contribution.
- The new architecture manifest is an experiment scaffold. It is not evidence
  until executed with fixed seeds, checkpoint hashes, confidence intervals,
  wall-clock/VRAM logs, quality metrics, and fallback-use metrics.

## 2026-06-23 Round 5 Diagnostic Fixes

Confirmed fixes:

- Evolutionary output capping now protects item providers required by
  `ITEM_GATE` edges. Final topology cleanup no longer prunes the only provider
  for a required traversal item.
- Output connectivity repair now connects protected goal-only components
  instead of silently merging their component sets without adding an edge.
- MAP-Elites leniency counts boss locks and Big Keys separately from small
  key locks. Boss-door layouts without a Big Key no longer receive maximum
  leniency.
- Zero-length oracle paths now produce finite confusion ratios. Start-equals-
  goal cases no longer inject NaN into MAP-Elites replacement decisions.
- `InventoryAwareLogicNet` and `DifferentiableTortuosity` use grid-cardinality
  sentinels rather than `H+W`, so winding valid routes are not truncated as
  unreachable.
- MaskGIT edge-aware boundary logit bias now accumulates overlapping boundary
  evidence at corners instead of overwriting the prior boundary pass.
- Quest-2 VGLC graph discovery now prefers canonical `LoZ2_X.dot` names before
  legacy `_q2` names and only then falls back to Quest-1 graphs with a warning.
- Core VGLC door parsing and stitched-room connection carving now use the
  canonical `DOOR_POSITIONS` boundary slots, eliminating inner-wall door
  detection and oversized connection carving.
- `scripts/run_ablation_study.py` now includes `PURE_WFC_FLAT_PRIOR`, making
  weighted Bayesian WFC versus flat-prior WFC a controlled ablation.
- `scripts/generate_round5_scientific_gap_manifest.py` creates a reproducible
  manifest for SPADE-vs-additive topology conditioning, fixed-graph fast
  sampler latency-quality, and weighted-vs-flat WFC prior experiments.
- `scripts/statistical_validation_and_analysis.py` now emits coverage curves
  from measured per-seed histories when present and marks older aggregate-only
  artifacts as `curve_source="aggregate_final_only"` instead of treating final
  coverage as a measured convergence curve.

Research boundary:

- SPADE, fast-sampler, and WFC-prior rows are ablation hypotheses until their
  generated manifests are executed and archived with fixed seeds, checkpoint
  hashes, paired metrics, and confidence intervals. The script prevents the
  gap from being forgotten; it does not by itself prove the claims.

Focused verification:

- `python -m pytest tests/test_round5_audit_fixes.py tests/test_search_benchmark_utils.py -q`
  passed with 15 tests.
- `python scripts/generate_round5_scientific_gap_manifest.py --seeds 1 --epochs 1 --output-dir temp_round5_manifest_check`
  wrote JSON/CSV successfully; the temporary output was removed after the
  smoke test.
- `python -m pytest tests/test_round5_audit_fixes.py tests/test_statistical_validation_analysis.py -q`
  passed with 20 tests after the MaskGIT edge-semantics and coverage-curve
  fixes.

Round 5 re-verification closure:

- Output node capping now also protects generic `ITEM` providers when a legacy
  edge encodes only `edge_type="ITEM_GATE"` without an explicit
  `item_required` field.
- Connectivity repair no longer inserts an ordinary raw `PATH` edge into a
  protected goal/boss-only component when no boss anchor exists. Such emergency
  links are marked as progression-gate repairs and use a boss-locked edge type
  so downstream validation cannot mistake them for a free bypass.
- MAP-Elites leniency now keeps ordinary key locks, `MULTI_LOCK` edges, boss
  locks, small keys, and Big Keys in separate economies. Surplus Big Keys no
  longer hide missing small keys.
- The legacy `src.ml.logic_net.SoftBellmanFord` path now uses a grid-area
  sentinel, matching the already repaired inventory-aware pathfinder family.
- MaskGIT fixed-token logit forcing now builds a forced-logit tensor only for
  fixed cells. Editable cells, including valid class-0/VOID logits, are left
  unchanged.
- Zelda parser/adapter consistency is restored: core parser open-boundary door
  detection accepts `D`, `d`, `F`, `f`, and `.`, and the adapter applies the
  same doored-room interior void-fill rule as the core parser.
- `src/zelda_data/stitching/graph_placement.py` now derives boundary door
  coordinates from canonical `DOOR_POSITIONS`, rather than returning hardcoded
  5- or 6-tile ranges.
- `scripts/run_ablation_study.py` now has explicit
  `DIFFUSION_TOPO_ADDITIVE` and `DIFFUSION_TOPO_SPADE` extended ablation rows,
  plan metadata, runtime fallback-config wiring, and a cache key that prevents
  reusing one loaded diffusion variant for another topology-conditioning mode.
- The Round-5 scientific-gap manifest now forwards `--lcm-checkpoint` into the
  fixed-graph fast-sampler benchmark. `run_fixed_graph_multi_seed_audit.py` and
  `run_fast_sampler_visual_audit.py` accept and pass the explicit checkpoint to
  pipeline construction, raising `FileNotFoundError` when the user supplies a
  missing path.

Focused verification after re-verification:

- `python -m pytest tests/test_round5_audit_fixes.py -q` passed with 22 tests.
- `python -m pytest tests/test_round5_audit_fixes.py tests/test_protocol_reporting.py::test_ablation_extended_plan_documents_logic_guidance_timing_sweep tests/test_protocol_reporting.py::test_round5_manifest_passes_lcm_checkpoint_to_fast_sampler_command -q`
  passed with 24 tests.
- `python -m compileall src/generation/evolutionary_director/generator.py src/evaluation/map_elites.py src/ml/logic_net.py src/core/discrete_masked_model.py src/zelda_data/stitching/graph_placement.py src/zelda_data/parsers/core_parsers.py src/data_processing/data_adapter.py scripts/run_ablation_study.py scripts/generate_round5_scientific_gap_manifest.py scripts/run_fixed_graph_multi_seed_audit.py scripts/run_fast_sampler_visual_audit.py tests/test_round5_audit_fixes.py tests/test_protocol_reporting.py`
  completed successfully.

## 2026-06-23 Round 7 Verification Fixes

Confirmed fixes:

- Goal-gauntlet repair now selects reachable approach nodes before preserving
  stale boss-door predecessors. Non-selected boss-door predecessor edges are
  removed during normalization, and orphan cleanup is prevented from deleting
  the protected `GOAL -> BOSS -> BOSS_DOOR` terminal chain.
- `src/data_processing/data_adapter.py` no longer keeps a standalone VGLC
  parser implementation. Its adapter-facing `VGLCParser` delegates canonical
  room parsing, door detection, and semantic conversion to
  `src.zelda_data.parsers.core_parsers.VGLCParser`, then converts the parsed
  records into `RoomTensor`.
- `scripts/generate_round5_scientific_gap_manifest.py` now includes Round-7
  pending protocols for generated-branch A*/P-CBS pre/post repair reporting,
  100-room and 500-room designer-controllability stress rows, matched-budget
  P-CBS component ablations, paired-seed statistical significance, and
  target-response semantic checks.

Scientific boundary:

- The new manifest rows are executable protocols, not results. They remain
  `planned` unless the manifest is run with `--execute`, and publication claims
  still require archived outputs with checkpoint hashes, fixed seeds,
  pre-repair and post-repair metrics, confidence intervals, and hardware logs.

## 2026-06-23 Repository Cleanup Pass

Confirmed fixes:

- Advanced-rule diversity coverage now counts the advanced feature families
  actually produced by `advanced_rules.py`, including switches, stairs,
  secrets, mini-bosses, tutorial chains, hidden edges, switch gates, and
  stair edges. The previous test only counted an older narrow subset and could
  fail even when the generated graph contained multiple advanced mechanics.
- Broken mojibake strings in the advanced-rule integration test output were
  replaced with ASCII output.
- Tracked temporary smoke outputs under `temp_round5_*` were removed. These
  were generated manifest/plan artifacts and are already covered by `.gitignore`
  temp-directory rules.

Verification:

- `python -m pytest tests/test_advanced_rules_integration.py -q` passed with
  22 tests and 8 skips.
- `python -m pytest tests/test_round5_audit_fixes.py tests/test_hmolqd/test_data_adapter.py::TestVGLCParser -q`
  passed with 32 tests.

## 2026-06-23 Major Non-GUI Audit Pass

Confirmed fixes:

- The centralized Zelda VGLC parser now detects door glyphs on both the outer
  canonical room boundary and the inner wall shell used by several VGLC files.
  This restored graph-room matching for Dungeon 2 Quest 1, Dungeon 4 Quest 1,
  and Dungeon 9 Quest 2, where every room previously parsed with zero doors.
- P-CBS avoids repeated posterior rebuilds for stable explored tiles, caches
  static-grid field-of-view sets during a solve, and skips redundant Bayesian
  updates for already-stable visible observations. These changes preserve the
  cognitive model but remove hot-path work that dominated long route searches.
- LogicNet projects latent-scale tile logits smaller than a canonical room back
  to `ROOM_HEIGHT x ROOM_WIDTH`, while preserving native resolution for
  already-canonical and larger dynamic maps. `_resolve_room_logic_targets`
  again accepts omitted `spatial_hw` for compatibility.
- Diffusion trainer optimizer groups retain decay/no-decay behavior while
  exposing stable module-level group names (`diffusion`, `condition_encoder`,
  `logic_net`) plus a `decay_policy` field for logging.
- The D* Lite key-door regression fixture now actually requires crossing the
  locked door; the previous open-floor layout allowed a valid route around the
  key and was a false alarm.

Research boundary:

- Current literature supports reporting quality, diversity, and controllability
  rather than relying on a single validity score, and treats symbolic repair as
  a distinct hybrid-PCG component that must be ablated separately. The code now
  has stronger non-GUI implementation coverage, but SOTA or human-calibration
  claims still require executing the planned external-baseline, pre/post-repair,
  paired-seed, and P-CBS persona protocols with archived artifacts.

Verification:

- `python -m pytest tests/test_data_integrity.py -q --maxfail=5` passed with
  110 tests.
- `python -m pytest tests/test_cognitive_bounded_search.py tests/test_validator_block_push_regressions.py -q --maxfail=5`
  passed with 60 tests.
- `python -m pytest tests/test_hmolqd/test_logic_net.py tests/test_logicnet_optimizer.py tests/test_round5_audit_fixes.py tests/test_topology_generation_fixes.py tests/test_train_diffusion_conditioning_shapes.py::test_diffusion_adamw_groups_exclude_bias_and_norm_from_weight_decay -q --maxfail=5`
  passed with 67 tests.
- `python -m pytest tests -q -k "not gui" --maxfail=5` passed with
  1271 tests, 8 skips, and 280 deselections.
- Full `python -m pytest tests -q --maxfail=5` still times out in
  `tests/test_gui_demo_validated_level_artifact.py::test_gui_real_full_pipeline_pdrop035_demo_solves_live_with_pcbs_balanced`;
  this is a live GUI artifact test and remains outside the non-GUI audit scope.
