# Architecture Audit Research Notes

These notes record the external references used while addressing the H-MOLQD
architecture audit. They are not an experimental-results claim; they document
why the implemented fixes are defensible and where empirical validation is
still required.

## References

- Pre-LN residual placement and Transformer training stability: Xiong et al.,
  "On Layer Normalization in the Transformer Architecture",
  https://arxiv.org/abs/2002.04745.
- Temperature annealing for categorical relaxations: Jang et al.,
  "Categorical Reparameterization with Gumbel-Softmax",
  https://arxiv.org/abs/1611.01144.
- Discrete level-corpus baselines and tile-pattern comparison context:
  Summerville et al., "The VGLC: The Video Game Level Corpus",
  https://arxiv.org/abs/1606.07487.
- Few-step latent diffusion acceleration requires a trained/distilled
  consistency model or LoRA: Luo et al., "Latent Consistency Models",
  https://arxiv.org/abs/2310.04378, and "LCM-LoRA",
  https://arxiv.org/abs/2311.05556. Song et al. provide the broader
  consistency-model basis at https://arxiv.org/abs/2303.01469.
- The authors' full-model LCM distillation script separates a frozen diffusion
  teacher, online student, and EMA target student:
  https://github.com/luosiallen/latent-consistency-model/blob/main/LCM_Training_Script/consistency_distillation/train_lcm_distill_sd_wds.py.
  Their LoRA example still uses adjacent trajectory points but omits the
  separate EMA target:
  https://github.com/luosiallen/latent-consistency-model/blob/main/LCM_Training_Script/consistency_distillation/train_lcm_distill_lora_sd_wds.py.
- PyTorch documents epoch-level cosine learning-rate annealing through
  `CosineAnnealingLR`:
  https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html.
- WFC should preserve recursively propagated constraint supports, not only
  one-hop neighbor updates: Karth and Smith, "WaveFunctionCollapse is
  Constraint Solving in the Wild",
  https://www.pcgworkshop.com/archive/karth2017wavefunctioncollapse.pdf.
- MAP-Elites archives elites by user-selected dimensions of phenotypic
  variation: Mouret and Clune, "Illuminating Search Spaces by Mapping Elites",
  https://arxiv.org/abs/1504.04909.
- Pydantic v2 models validate input through `model_validate()` and can forbid
  extra fields through model configuration:
  https://docs.pydantic.dev/latest/concepts/models/.
- `pytest-timeout` supports a global configured timeout:
  https://pypi.org/project/pytest-timeout/.
- NumPy documents that `convolve(..., mode="same")` still exposes boundary
  effects and that `pad(..., mode="edge")` replicates array edge values:
  https://numpy.org/doc/stable/reference/generated/numpy.convolve and
  https://numpy.org/doc/stable/reference/generated/numpy.pad.html.
- NetworkX defines undirected degree as the number of adjacent edges and
  directed out-degree as the number of outgoing edges. Branching metrics must
  distinguish corridor adjacency from forward choices:
  https://networkx.org/documentation/stable/reference/classes/generated/networkx.Graph.degree.html
  and
  https://networkx.org/documentation/stable/_modules/networkx/classes/digraph.html.
- The University of Alberta Sokoban rules specify one-at-a-time stone pushes.
  Dynamic block occupancy therefore has to be resolved from search state, not
  only from the immutable source grid:
  https://webdocs.cs.ualberta.ca/~games/Sokoban/thegame.html.
- PyTorch autograd grad modes are nested context managers; `no_grad` is useful
  for explicit non-recorded operations, while local grad-enabled regions should
  be stated directly in code that may run guidance:
  https://docs.pytorch.org/docs/stable/notes/autograd.html and
  https://docs.pytorch.org/docs/2.11/generated/torch.no_grad.html.
- PyTorch documents `index_copy_` separately from `torch.autograd.grad`; the
  revision fix keeps the functional `index_copy` path and adds a regression
  test that gradients reach the copied source tensor:
  https://docs.pytorch.org/docs/stable/generated/torch.Tensor.index_copy_.html
  and https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html.
- Diffusion posterior sampling motivates guidance on clean/posterior-mean
  predictions rather than arbitrary noisy states for inverse-problem-style
  constraints: Chung et al., "Diffusion Posterior Sampling for General Noisy
  Inverse Problems", https://arxiv.org/abs/2209.14687.
- Classifier-free guidance is a separate sampling-time guidance family from
  classifier/DPS-style external-gradient guidance: Ho and Salimans,
  "Classifier-Free Diffusion Guidance", https://arxiv.org/abs/2207.12598.
- PyTorch batched matrix multiplication and boolean masking support replacing
  per-sample topology loops with dense padded adjacency for the small graph
  sizes used by room-token conditioning:
  https://docs.pytorch.org/docs/stable/generated/torch.bmm.html and
  https://docs.pytorch.org/docs/stable/notes/broadcasting.html.
- Spatial attention supervision is a standard way to make attention maps serve
  as grounding/alignment signals instead of only diagnostics; the implemented
  graph-node alignment loss follows the same negative-log-attention form used
  by attention-transfer and visual-grounding objectives:
  Zagoruyko and Komodakis, "Paying More Attention to Attention",
  https://arxiv.org/abs/1612.03928.
- Lock/key progression is a state-space reachability problem when keys can be
  collected and consumed. The implemented graph analyzer therefore searches
  `(node, small_key_count, has_boss_key)` states rather than validating only a
  static shortest path. This matches the general BFS/state-augmentation pattern
  for inventory-constrained planning:
  https://www.redblobgames.com/pathfinding/a-star/introduction.html.
- VQ-VAE decoding should use discrete codebook latents, not arbitrary
  continuous predictions, when measuring decoder-visible tile semantics. The
  VQ-VAE paper introduces discrete vector-quantized latents and uses
  straight-through gradient estimation around the quantization step:
  https://arxiv.org/abs/1711.00937.
- Differentiable planning losses are defensible only when they expose a real
  backpropagated path from map logits to the upstream network. Value Iteration
  Networks provide the grid-planning precedent, and Neural Bellman-Ford
  Networks provide a path-relaxation graph precedent:
  https://arxiv.org/abs/1602.02867 and https://arxiv.org/abs/2106.06935.
- Hard solvability remains a validation metric, not a training surrogate. A*
  style minimum-cost path search is the classic hard pathfinding check for
  confirming generated grids after sampling:
  https://doi.org/10.1109/TSSC.1968.300136.
- Graphify was installed as `graphifyy` and the repo graph was generated with
  AST-only indexing so future architecture work can query local code structure:
  https://github.com/safishamsi/graphify.
- Min-SNR diffusion weighting uses the recommended gamma clamp from Hang et al.,
  "Efficient Diffusion Training via Min-SNR Weighting Strategy",
  https://arxiv.org/abs/2303.09556.
- Relative random-walk positional features provide edge-level pairwise walk
  information, unlike node-local RWSE-only summaries. This matches the RRWP
  motivation in GRIT: Ma et al., "Graph Recurrent Neural Networks are More
  Powerful Than Transformers", https://arxiv.org/abs/2305.17589.
- Full GraphGPS replacement remains an ablation item because it changes model
  capacity and training behavior rather than fixing a semantic bug:
  Rampasek et al., "Recipe for a General, Powerful, Scalable Graph Transformer",
  https://arxiv.org/abs/2205.12454.
- FSQ, VIN pathfinding, hierarchical codebooks, and flow/DiT backbones remain
  research-track changes because each trades capacity, inductive bias, or
  training dynamics and therefore needs an ablation table before becoming the
  default. See FSQ at https://arxiv.org/abs/2309.15505 and VIN at
  https://arxiv.org/abs/1602.02867.

## Implementation Implications

- The U-Net attention residual now lives inside the Pre-LN self-attention
  block, avoiding a double residual around attention.
- LogicNet temperature annealing remains explicit and is called through
  `anneal_temperature`, with `update_temperature` retained as a compatibility
  alias.
- Fast-sampler configuration defaults do not claim LCM-LoRA acceleration unless
  a real distilled backend is provided. The implemented backend is repo-specific
  `consistency_lora` from `src/train_lcm.py`: a frozen graph-aware diffusion
  teacher advances adjacent DDIM trajectory points, an online LoRA student
  learns the high-noise consistency output, and an EMA target student provides
  the lower-noise target. Resume checkpoints retain both adapters and the
  deployable adapter export uses the EMA target. It remains metadata-gated
  through the underlying graph-aware latent diffusion runtime; arbitrary Stable
  Diffusion LCM-LoRA checkpoints remain incompatible with this custom latent
  space.
- Gaussian-VAE training now follows the existing VQ-VAE policy with resumable
  epoch-level `CosineAnnealingLR` state.
- Tile-pattern distribution metrics were added as a discrete corpus comparison
  primitive; they are intended to supplement, not replace, human playtests and
  solver-based validation.
- Weighted WFC now treats zero-support cells as contradictions and recursively
  propagates support reductions instead of restoring an unconstrained prior.
- Topology generation records the pre/post repair fitness and feasibility
  shift so exported phenotypes can be analyzed separately from pre-repair
  individuals.
- Pacing smoothing edge-pads the normalized tension curve before applying the
  three-tap convolution, preserving first-room and final-boss tension.
- Frustration scoring uses excess decision branching rather than raw average
  degree and uses room-level goal density rather than an all-or-nothing goal
  flag. A single boss room no longer erases branching confusion.
- Flat and underspecified dungeon inputs no longer receive positive flow or
  pacing progression credit.
- Push-block search resolves current block destinations before static vacated
  origins. Sequential validation, parallel A*, and P-CBS now share dynamic
  occupancy semantics.
- Advanced pipeline fun evaluation follows a resolved start-to-goal graph path,
  receives the NetworkX graph expected by the evaluator, and retains graph and
  entity semantics for bosses, goals, puzzles, locks, rewards, and recovery.
- Revision 2 fixes added supervised LogicNet tile-classifier loss during
  diffusion training, so the latent-to-tile projection used by guidance is no
  longer left as a random module when `logic_net_trainable` is enabled.
- LogicNet grid supervision no longer falls back to an all-door source mask
  when topology is absent. It uses a single current walkability source, avoiding
  false supervision that every unconditioned room should connect all doors.
- LogicNet temperature annealing now propagates to the grid pathfinder and
  sampling synchronizes temperature to denoising confidence: high noise keeps
  the constraint soft, while near-clean steps sharpen it.
- WFC tile IDs are derived from `SEMANTIC_PALETTE`; row/col entropy access is
  explicit through `entropy_at(row, col)`; and default adjacency no longer
  forces entity/door self-adjacency.
- Context-token topology refinement now builds a batched dense normalized
  adjacency with node-mask-aware padded rows/columns, then uses `torch.bmm` for
  both lightweight graph convolution and GAT-style topology refinement. This
  removes the previous per-sample refinement loop while preserving support for
  `[2,E]`, `[1,2,E]`, and `[B,2,E]` edge-index forms.
- Graph-to-grid attention capture now has two paths: detached CPU maps for
  visualization remain available through `get_last_attention_map`, while
  `spatial_alignment_loss` keeps an opt-in differentiable attention tensor for
  real training loss. `LatentDiffusionModel.training_loss` accepts
  `spatial_alignment_node_indices`, `spatial_alignment_positions`, optional
  `spatial_alignment_valid_mask`, and nonzero `spatial_alignment_weight` in
  `graph_data`.
- Diffusion config now validates `alpha_logic_tile` and
  `graph_spatial_alignment_weight`, and CLI training exposes
  `--alpha-logic-tile` and `--graph-spatial-alignment-weight`.
- Symbolic graph validation now performs inventory-state BFS. Small keys are
  consumed by `locked` / `key_locked` edges, boss-key access is tracked as a
  boolean, and failure reports distinguish `missing_key` from
  `missing_boss_key`.
- `scripts/train_logicnet_tile_classifier.py` trains only the LogicNet tile
  classifier on frozen VQ-VAE latents, writes checkpoint and JSON metrics, and
  fails the run when validation accuracy is below `--min-accuracy` unless
  `--no-enforce-threshold` is set.
- `scripts/generate_logic_loss_ablation_manifest.py` writes named config files
  for full, no tile classifier, no topology trace/anchor, no global graph
  reach, no global room lift, no spatial alignment, and no logic-grid-reach
  variants. These files are intended to be run with identical seeds and
  Dungeon 9 holdout reporting.
- Revision 3 fixes route predicted clean latents through the frozen VQ-VAE
  quantizer before decoder-based LogicNet supervision. This keeps the decoder
  on the codebook manifold while retaining straight-through gradients to the
  diffusion denoiser.
- Dungeon-scope global graph loss now requires one room-passability value per
  graph node. Room-level batches that only provide current-room passability no
  longer pretend missing dungeon rooms are fully passable; the graph loss
  returns zero with an explicit `global_graph_skipped` reason. Full dungeon
  batching remains the correct long-term architecture for training the global
  graph objective.
- `WalkabilityPredictor` and `SoftBellmanFordGridPathfinder` now derive their
  walkable tile IDs from `SEMANTIC_PALETTE`, matching the canonical symbolic
  palette instead of maintaining separate hardcoded lists.
- Validation now reports `val_logic_tile_accuracy`. Validation sampling also
  suppresses LogicNet guidance for a batch when tile-classifier accuracy is
  below `min_logic_tile_accuracy_for_guidance`, restoring the model guidance
  scale immediately after that sample call.
- LogicNet parameters are now included in `grad_clip_norm` when
  `logic_net_trainable=True`, and the key-lock dependency checker uses a
  vectorized pair tensor instead of a Python loop over scalar violations.
- Revision 4 makes the Bellman-Ford grid pathfinder the default LogicNet path,
  while retaining `cnn` as an ablation option. This makes the default training
  signal an explicit differentiable reachability calculation instead of a
  learned black-box proxy.
- Diffusion validation now emits `val_grid_reach_loss`,
  `val_graph_reach_loss`, and `val_hard_solvability` alongside
  `val_logic_tile_accuracy`. Hard solvability is computed from quantized
  VQ-VAE-decoded sampled logits through the symbolic `PathAnalyzer`, keeping
  solver evidence separate from differentiable losses.
- `tests/test_logicnet_gradient_flow.py` now verifies that grid losses
  backpropagate to the latent tensor for both Bellman-Ford and CNN pathfinders,
  that graph losses backpropagate to room passability, and that palette-derived
  walkability and temperature annealing remain stable.
- `experiments/logicnet_ablation.py` writes reproducible baseline, tile-only,
  and full Bellman-Ford LogicNet configs with the proof metrics recorded in the
  manifest. `experiments/gradient_magnitude_probe.py` records gradient
  magnitudes for latent inputs, tile logits, walkability, and tile-classifier
  parameters so the "neural architecture works" claim has direct signal-flow
  evidence before long training runs.
- H-MOLQD upgrade pass adds Graphify repo indexing (`python -m graphify`),
  verifies Min-SNR gamma defaults to 5.0, guides DDPM/DDIM through predicted
  clean latents with adaptive sqrt-alpha scaling, and keeps GradientProbe
  available for LogicNet backward-signal inspection.
- LogicNet edge semantics are now handled by a learnable
  `SemanticEdgeEncoder`. The graph-loss path also filters `edge_attr` /
  `edge_features` with the same valid-edge mask as `edge_index`, fixing a
  silent edge-label misalignment bug when invalid endpoints are removed.
- Zelda graph extraction and diffusion graph batching now carry
  `edge_rrwp: [E, GRAPH_TPE_DIM]` relative random-walk features. The current
  pass exposes these features without overwriting existing semantic edge
  feature slots; encoder consumption should be ablated against the current
  graph conditioning path.
- Straight-upgrade items implemented in this pass are production fixes:
  predicted-clean guidance, adaptive guidance scale, semantic edge costs,
  RRWP feature plumbing, GradientProbe diagnostics, Graphify integration, and
  Min-SNR verification. FSQ, GraphGPS, VIN, WFC pseudo-labels, PAG, flow
  matching, RLHF fine-tuning, hierarchical VQ-VAE, and DiT remain ablation
  work rather than unconditional defaults.
- Harsh-review v4 pass converts several previously unwired claims into
  measurable code paths. `architecture="fsq"` now builds a finite-scalar
  tokenizer ablation with implicit scalar codes, zero commitment/codebook
  losses, STE gradients, and code-usage/perplexity diagnostics. This is an
  ablation option, not a silent replacement for the default VQ tokenizer.
- `logic_grid_pathfinder="vin"` now selects a learnable Value Iteration
  Network-style grid pathfinder. The default remains Bellman-Ford because VIN
  changes inductive bias and must earn its place in the ablation table, but the
  code path is now real and differentiable.
- RRWP is now consumed by the graph conditioner. `edge_rrwp` is projected into
  hidden edge attributes and added to semantic edge features for GPS/GAT-style
  message passing, so RRWP no longer exists only as an unused side tensor.
- Room-level diffusion training now supports `dungeon_batch_mode=True`.
  `DungeonBatchSampler` groups all rooms from one dungeon variant, and the
  trainer collapses complete room batches into `graph_scope="dungeon"` graph
  data with one current-node index per room. This gives the global graph loss a
  full node-passability vector instead of random rooms from unrelated dungeons.
- WFC pseudo-label distillation is now opt-in through `alpha_wfc_pseudo`.
  The trainer builds batch priors, pins only high-confidence predicted cells,
  lets WFC fill/repair uncertain cells, and adds cross-entropy to the repaired
  pseudo-target. This keeps the self-training loop measurable and avoids a
  placebo full-grid seed that would simply return the original prediction.

## Validation Commands

- `python -m py_compile src/core/graph_grid_attention.py src/core/latent_diffusion.py src/core/symbolic_refiner.py src/train_diffusion.py scripts/train_logicnet_tile_classifier.py scripts/generate_logic_loss_ablation_manifest.py`
- `python -m pytest tests/test_ml_components.py::TestGraphGridAttention tests/test_audit_regressions.py::test_context_topology_refinement_uses_batched_padded_adjacency tests/test_hmolqd/test_symbolic_refiner.py::TestPathAnalyzer::test_analyze_graph_consumes_small_keys_across_locked_edges tests/test_hmolqd/test_symbolic_refiner.py::TestPathAnalyzer::test_analyze_graph_accepts_path_with_enough_small_keys -q`
- `python -m pytest tests/test_train_diffusion_conditioning_shapes.py::test_predicted_latent_logic_branch_backpropagates_from_vqvae_decode_to_unet -q`
- `python -m pytest tests/test_config_system.py::test_diffusion_helper_preserves_yaml_only_methodology_knobs tests/test_audit_regressions.py tests/test_hmolqd/test_symbolic_refiner.py::TestPathAnalyzer tests/test_ml_components.py::TestGraphGridAttention tests/test_train_diffusion_conditioning_shapes.py::test_predicted_latent_logic_branch_backpropagates_from_vqvae_decode_to_unet -q`
- `python scripts/train_logicnet_tile_classifier.py --help`
- `python scripts/generate_logic_loss_ablation_manifest.py --help`
- Final broad regression: `python -m pytest tests/test_audit_regressions.py tests/test_architecture_audit_fixes.py tests/test_logicnet_fixes.py tests/test_train_diffusion_conditioning_shapes.py tests/test_hmolqd/test_symbolic_refiner.py tests/test_hmolqd/test_gaussian_vae.py tests/test_ml_components.py tests/test_config_system.py -q` passed with 212 tests.
- Revision 3 focused regression: `python -m pytest tests/test_logicnet_fixes.py tests/test_train_diffusion_conditioning_shapes.py::test_decode_latent_for_logic_quantizes_before_vqvae_decode tests/test_train_diffusion_conditioning_shapes.py::test_train_step_predicted_latent_decodes_to_tile_logits_for_logic_loss tests/test_train_diffusion_conditioning_shapes.py::test_train_step_trains_logicnet_tile_classifier_when_enabled tests/test_train_diffusion_conditioning_shapes.py::test_validate_reports_logic_tile_accuracy tests/test_train_diffusion_conditioning_shapes.py::test_validate_suppresses_sampling_guidance_when_tile_accuracy_below_gate tests/test_config_system.py::test_diffusion_helper_preserves_yaml_only_methodology_knobs tests/test_audit_regressions.py::test_pydantic_config_schema_returns_cross_field_normalization -q` passed with 15 tests.
- Revision 3 broad regression: `python -m pytest tests/test_audit_regressions.py tests/test_architecture_audit_fixes.py tests/test_logicnet_fixes.py tests/test_train_diffusion_conditioning_shapes.py tests/test_hmolqd/test_symbolic_refiner.py tests/test_hmolqd/test_gaussian_vae.py tests/test_ml_components.py tests/test_config_system.py -q` passed with 217 tests.
- Revision 4 compile check: `python -m py_compile src/train_diffusion.py src/core/logic_net.py src/config_system.py src/pipeline/config_bridge.py src/pipeline/models/model_manager.py experiments/logicnet_ablation.py experiments/gradient_magnitude_probe.py tests/test_logicnet_gradient_flow.py tests/test_train_diffusion_conditioning_shapes.py tests/test_audit_regressions.py tests/test_config_system.py`.
- Revision 4 focused regression: `python -m pytest tests/test_logicnet_gradient_flow.py tests/test_audit_regressions.py::test_logic_net_defaults_to_bellman_ford_grid_pathfinder tests/test_train_diffusion_conditioning_shapes.py::test_validate_reports_hard_solvability_from_decoded_samples tests/test_config_system.py::test_diffusion_helper_preserves_yaml_only_methodology_knobs -q` passed with 8 tests.
- H-MOLQD upgrade focused regression: `python -m pytest tests/test_architecture_audit_fixes.py::test_p_sample_guides_pred_x0_before_rebuilding_posterior tests/test_architecture_audit_fixes.py::test_min_snr_gamma_defaults_to_five_and_clamps_snr_weights tests/test_logicnet_fixes.py::test_semantic_edge_encoder_defaults_and_receives_gradients tests/test_logicnet_fixes.py::test_logicnet_edge_attr_penalties_follow_valid_edge_filter tests/test_logicnet_gradient_flow.py::test_gradient_probe_records_logicnet_module_gradients tests/test_zelda_loader_graph_conditioning.py::test_dungeon_dataset_getitem_preserves_spatial_graph_fields tests/test_zelda_loader_graph_conditioning.py::test_compute_rrwp_edge_features_preserves_edge_order_and_invalid_rows -q` passed with 7 tests.
- H-MOLQD upgrade broad targeted regression: `python -m pytest tests/test_architecture_audit_fixes.py tests/test_logicnet_fixes.py tests/test_logicnet_gradient_flow.py tests/test_zelda_loader_graph_conditioning.py tests/test_train_diffusion_conditioning_shapes.py tests/test_ml_components.py tests/test_config_system.py -q` passed with 209 tests.
- Harsh-review v4 compile check: `python -m py_compile src/core/vqvae.py src/core/logic_net.py src/core/condition_encoder.py src/zelda_data/zelda_loader.py src/train_diffusion.py src/config_system.py src/pipeline/config_bridge.py tests/test_hmolqd/test_vqvae.py tests/test_logicnet_fixes.py tests/test_ml_components.py tests/test_zelda_loader_graph_conditioning.py tests/test_train_diffusion_conditioning_shapes.py`.
- Harsh-review v4 focused regression: `python -m pytest tests/test_hmolqd/test_vqvae.py::TestFSQuantizer tests/test_logicnet_fixes.py::test_vin_pathfinder_is_selectable_and_backpropagates tests/test_ml_components.py::test_global_stream_encoder_rrwp_changes_gps_edge_messages tests/test_zelda_loader_graph_conditioning.py::test_dungeon_batch_sampler_groups_room_samples_by_dungeon_variant tests/test_train_diffusion_conditioning_shapes.py::test_try_stack_dungeon_scope_graph_batch_collapses_full_room_set tests/test_train_diffusion_conditioning_shapes.py::test_wfc_pseudo_label_loss_is_opt_in_and_backpropagates -q` passed with 7 tests.
- Harsh-review v4 broad targeted regression: `python -m pytest tests/test_hmolqd/test_vqvae.py tests/test_logicnet_fixes.py tests/test_logicnet_gradient_flow.py tests/test_ml_components.py tests/test_zelda_loader_graph_conditioning.py tests/test_train_diffusion_conditioning_shapes.py tests/test_config_system.py -q` passed with 179 tests.
- Revision 4 smoke checks: `python experiments/logicnet_ablation.py --base-config configs/zelda_hmolqd.yaml --output-dir results/tmp_logicnet_ablation_smoke --epochs 1 --validation-samples 1 --validation-diffusion-samples 1 --quick` wrote a manifest, and `python experiments/gradient_magnitude_probe.py --output results/tmp_logicnet_gradient_probe.json --batch-size 1 --height 4 --width 4 --latent-dim 8 --hidden-dim 16 --num-iterations 3` recorded nonzero latent, tile-classifier, tile-logit, and walkability gradients.
- Revision 4 broad regression: `python -m pytest tests/test_audit_regressions.py tests/test_architecture_audit_fixes.py tests/test_logicnet_fixes.py tests/test_logicnet_gradient_flow.py tests/test_train_diffusion_conditioning_shapes.py tests/test_hmolqd/test_symbolic_refiner.py tests/test_hmolqd/test_gaussian_vae.py tests/test_ml_components.py tests/test_config_system.py -q` passed with 224 tests.
