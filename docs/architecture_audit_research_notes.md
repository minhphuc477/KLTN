# Architecture Audit Research Notes

Last consolidated: 2026-06-24.

This is the current non-GUI audit ledger for H-MOLQD. It replaces the previous
chronological append-only log. Historical reviews remain under `docs/archive/`.

## Scope

The audited system combines:

- Block 0: VGLC parsing, room extraction, graph alignment, and split control.
- Block I: directed mission-graph generation and evolutionary/QD search.
- Block II: semantic VQ-VAE/FSQ room tokenization.
- Block III: local-neighbor and graph condition encoding.
- Block IV: latent diffusion, DiT, masked-token, categorical, and fast samplers.
- Block V: LogicNet differentiable feasibility guidance.
- Block VI: symbolic repair, WFC, overlays, and puzzle scaffolds.
- Block VII: hard validation, A*/Dijkstra, P-CBS, metrics, and MAP-Elites.

GUI code is outside this audit unless it changes model, solver, or experiment
semantics.

## Audit Method

- Graphify semantic graph queries are the first code-navigation step.
- Focused source review follows graph relationships into implementation.
- Reported bugs require a code trace or counterexample.
- Tests are evidence only when their asserted contract is independently valid.
- Static checks include `compileall` and Ruff undefined-name/duplicate checks.
- Research claims are checked against primary papers or canonical project
  references. Network search was unavailable during the latest pass because
  external requests returned access errors; no new web-only claim is treated as
  verified.

Graphify status:

- `graphify-out/graph.json` and the local query/explain commands are active.
- The Groq provider is configured through the user environment without storing
  credentials in the repository.
- Groq semantic enrichment is currently limited by the provider account's
  request token budget. Local AST/community semantic search remains usable.

## Evidence Taxonomy

Use these labels consistently:

| Label | Meaning |
|---|---|
| Implemented | Code path exists. |
| Unit-tested | A bounded invariant or counterexample passes. |
| Smoke-executed | The command ran on a tiny or synthetic workload. |
| Scientifically executed | Fixed seeds, valid checkpoints, required metrics, and provenance exist. |
| Invalid | Execution completed but cannot support the intended claim. |
| Blocked | Required data, checkpoint, compute, or calibration is absent. |
| Planned | Manifest or command exists but has not produced evidence. |

An exit code of zero is not by itself scientific evidence.

## Research Anchors

The architecture is positioned against these primary references:

- VQ-VAE: van den Oord et al., 2017, arXiv:1711.00937.
- FSQ: Mentzer et al., 2023, arXiv:2309.15505.
- Latent Diffusion: Rombach et al., 2022, CVPR.
- DiT: Peebles and Xie, 2023, arXiv:2212.09748.
- Flow Matching: Lipman et al., 2023, arXiv:2210.02747.
- Consistency Models/LCM: Song et al., 2023 and Luo et al., 2023.
- GraphGPS: Rampasek et al., 2022, arXiv:2205.12454.
- Graphormer: Ying et al., 2021, arXiv:2106.05234.
- Value Iteration Networks: Tamar et al., 2016, arXiv:1602.02867.
- Neural Bellman-Ford Networks: Zhu et al., 2021, arXiv:2106.06935.
- PCGML survey: Summerville et al., 2018, arXiv:1702.00539.
- VGLC: Summerville et al., 2016, arXiv:1606.07487.
- WFC analysis: Karth and Smith, IEEE Transactions on Games, 2021.

These references motivate ablations. They do not prove this implementation is
better than the referenced methods.

## Current Block Status

| Block | Implementation status | Remaining scientific requirement |
|---|---|---|
| 0 | Text padding, lazy VGLC access, categorical-token mode, and graph alignment have explicit contracts. | Publish split/provenance hashes and verify all final training data. |
| I | Directed graph generation, progression anchors, node-cap protection, and topology metrics are implemented. | Execute controllability and 100/500-room stress protocols. |
| II | VQ/FSQ paths, EMA updates, dead-code reset, strict checkpoint validation, and counter persistence are implemented. | Supply a valid trained VQ-VAE artifact and report utilization/sample efficiency. |
| III | Graph/local conditioning, GPS/RRWP options, topology maps, and reference-room paths exist. | Run matched-budget target-response and conditioning ablations. |
| IV | Diffusion, DiT, flow matching, masked generation, categorical baseline, and fast sampling exist as alternatives. | Train valid checkpoints and compare quality, latency, memory, and pre-repair validity. |
| V | LogicNet grid/graph reachability and guidance paths exist. | Run LogicNet ON/OFF with raw pre-repair hard-oracle rates. |
| VI | WFC, flat-prior WFC, repair feedback, overlays, and puzzle scaffolds exist. | Quantify repair contribution, failure rate, and runtime separately. |
| VII | Tile solvers, graph validation, P-CBS, QD metrics, and statistical scripts exist. | Execute final persona tables, paired tests, and human calibration. |

## Confirmed Fixes In Latest Pass

### Pipeline and data

- Pure MaskGIT no longer loads unused VQ-VAE/diffusion models.
- Categorical codebook sampling is explicitly an unconditional baseline and no
  longer loads or computes an unused condition encoder.
- Enabling a teacher fallback loads the required diffusion teacher stack.
- Strict VQ-VAE loading rejects missing learned keys.
- Text datasets scan all files before `pad_to_max` batching.
- Lazy VGLC `get_raw_grid()` uses the indexed VGLC loader.
- Categorical masked-room training keeps integer token IDs.
- Graph-to-room alignment cannot silently map two graph nodes to one room.
- Robust block timeouts return without waiting for the timed-out worker.
- NetworkX 3.5 artifact serialization uses standard pickle APIs.
- Invalid BSP dimensions fail early instead of creating out-of-bounds rooms.

### Model and training

- CFG dropout now masks context, topology maps, node masks, and other batched
  graph conditioning together.
- Diffusion epoch ownership is centralized in the outer training loop, avoiding
  resume skips and one-epoch-early validation schedules.
- WFC pseudo-label targets retain their original successful sample indices.
- AMP GradScaler state is included in resume checkpoints.
- VQ EMA/dead-code schedule counters persist through state dictionaries while
  legacy checkpoints remain loadable.
- LogicNet validation uses the effective ramped logic weight and includes tile
  classifier and optional WFC pseudo-label components in checkpoint selection.
- Warmup no longer gets overwritten by the cosine scheduler; parameter groups
  retain their actual base learning rates.
- Masked-room legacy U-Net controls now fail explicitly when changed instead of
  creating fake no-op ablations.
- Fast-sampling telemetry separates requested from actually executed fast paths.
- Advanced-pipeline global water state now reaches the existing state-aware
  room transformation instead of remaining unused graph metadata. This is a
  deterministic state transformation, not learned state conditioning.

### Solvers and metrics

- A pushed block occupying the goal no longer counts as a solution.
- Graph-guided room validation checks traversability and fails closed on
  unsupported constraints.
- Directed graph validation no longer invents reverse paths.
- Graph validation models consumable keys/bombs and persistent boss/item keys.
- Graph teleport edges consume resources once and track opened graph edges in
  the full search state.
- Graph-level A* uses an admissible hop lower bound rather than average-room
  area scaling.
- Key-economy search collects each provider once and handles adversarial dead
  ends without unpacking `None`.
- D* Lite stale queue entries are versioned and simple cases use the primary
  algorithm instead of silently falling back to A*.
- IDDFS checks the goal at the depth boundary and visits every permitted depth.
- Path length means transitions (`len(path) - 1`) across validation, solver
  comparison, P-CBS, and benchmark adapters.
- Confusion ratio is normalized as excess path overhead, where an oracle-match
  is `0.0`.
- A* timeout is reported as indeterminate rather than mislabeled unsolvable in
  CBS fitness.
- P-CBS suboptimal-decision accounting uses the same graph-aware navigation
  distance as action scoring.

## Architecture Ablation Policy

Architectural upgrades are hypotheses, not replacements:

- U-Net versus DiT.
- LayerNorm/GELU versus RMSNorm/SwiGLU in DiT.
- additive topology maps versus SPADE.
- softmax attention versus supported linear-attention modes.
- topology refinement off/lightweight/GAT variants.
- VQ-VAE versus FSQ and hierarchical tokenizers.
- diffusion versus masked-token generation.
- diffusion teacher versus trained LCM/fast sampler.
- LogicNet off/on and guidance off/on.
- weighted Bayesian WFC versus flat-prior WFC versus no repair.

Every comparison must keep data split, seed set, optimization budget, parameter
budget, generation budget, and post-processing policy explicit.

The categorical codebook-usage sampler is an unconditional prior baseline. It
must not be described as graph conditioned.

## Experiment Status

### Executed but limited

- `results/random_baseline/random_baseline_results.json`: 96 samples for each
  of seeds 42, 43, and 44. This is topology baseline evidence only.
- `results/matched_budget/`: executed on a small budget, but all reported
  feasible-search rates are zero. Treat as diagnostic.
- `results/cognitive_objective_ab/`: four paired cases with zero constraint
  validity in both arms. Treat as diagnostic.
- Local baseline reports are tiny dry runs, not converged external baselines.

### Invalid

- P-CBS round-2 smoke used one map and the oracle solved none. Its own
  `experiment_valid=false` flag is correct.

### Planned, not executed

- `results/round5_scientific_gaps/` is a plan manifest.
- SPADE versus additive conditioning.
- model architecture and attention ablations.
- LogicNet loss-component ablations.
- diffusion versus LCM-LoRA fast sampling.
- weighted versus flat-prior WFC.
- full persona/component tables and paired significance tests.
- target-response and large-dungeon controllability studies.

### Hard blocker

`outputs/zelda_hmolqd/checkpoints/vqvae/vqvae_pretrained.pth` is a diffusion
bundle rather than a valid VQ-VAE checkpoint. Conditioning, LogicNet, repair,
and end-to-end neural experiments using that path are invalid until Block II is
retrained or a correct artifact is supplied.

Experiment manifests now merge their base configuration and require an
explicit VQ-VAE checkpoint for execution. Plan-only manifests may contain a
placeholder, but cannot be promoted to executed evidence.

## Publication Claim Boundary

Currently defensible:

- the repository implements a seven-block neuro-symbolic PCG architecture;
- major components have explicit ablation switches or separate baselines;
- hard validation and repair-aware reporting are separated;
- bounded counterexample tests cover key solver and training invariants.

Not currently defensible:

- state-of-the-art or “surpasses publications” claims;
- human-likeness claims without calibrated human traces;
- standalone neural solvability claims based only on post-repair output;
- fast-sampler quality claims without a trained LCM/consistency checkpoint;
- attention improvements without matched-budget results;
- throughput or memory claims without target-GPU profiling;
- significance claims based on aggregate means or smoke runs.

Required final evidence:

1. Valid Block-II and downstream checkpoints with hashes and metadata.
2. Fixed-seed raw/pre-repair and repaired hard-oracle rates.
3. Matched-budget architecture and attention ablations.
4. Quality, diversity, controllability, runtime, and memory tables.
5. P-CBS persona/component results with oracle status separated from timeout.
6. Paired significance tests and effect sizes.
7. Human calibration or appropriately limited proxy-language.

## Verification Policy

Before publication lock:

- run the complete non-GUI suite serially;
- run focused gradient/finite-value probes on the target GPU;
- archive exact commands, configs, seeds, checkpoint hashes, environment, and
  required metric files;
- fail experiment completion when required outputs or provenance are missing;
- update Graphify with `python -m graphify update .`;
- regenerate this ledger from current evidence rather than appending another
  dated audit section.
