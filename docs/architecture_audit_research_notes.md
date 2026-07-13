# Architecture Audit Research Notes

Last consolidated: 2026-07-02.

This is the current non-GUI audit ledger for H-MOLQD. It replaces the previous
chronological append-only log. Historical reviews remain under `docs/archive/`.

Publication-facing claim boundaries now live in
[`PUBLICATION_GUIDANCE_AND_CARDS.md`](PUBLICATION_GUIDANCE_AND_CARDS.md) and
the matching machine-readable
[`publication_guidance_card.json`](publication_guidance_card.json). This ledger
tracks implementation/evidence status; the publication card defines the
research question, system boundary, data card, artifact card, metric contract,
baseline taxonomy, failure taxonomy, human-calibration stance, reproducibility
requirements, and IP note.

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

- Source code, call sites, and executable counterexamples are authoritative.
- Graphify may narrow navigation, but its generated graph is not correctness
  evidence and is never substituted for reading the implementation.
- Reported bugs require a code trace or counterexample.
- Tests are evidence only when their asserted contract is independently valid.
- Static checks include `compileall` and Ruff undefined-name/duplicate checks.
- Research claims are checked against primary papers or canonical project
  references. Provider behavior is verified against the provider's current
  first-party API documentation; inaccessible sources remain unverified.

Optional navigation status:

- `graphify-out/graph.json` and the local query/explain commands are available,
  but this consolidation pass did not rely on them.
- ApiFreeLLM is configured through a user-level environment variable and a
  localhost OpenAI-protocol adapter; no credential is stored in the repository.
- The free endpoint has a 32k-token context and a documented per-request delay.
  Full-repository semantic extraction therefore requires bounded chunks and
  serial execution. Local AST/community query and explain remain available
  independently of the external provider.
- The 200 largest communities were labeled successfully in 25-community
  batches; remaining small communities retain deterministic placeholders.

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
| I | Directed graph generation, exact resource-state feasibility, progression anchors, node-cap protection, and topology metrics are implemented. | Execute controllability and 100/500-room stress protocols. |
| II | VQ/FSQ paths, EMA updates, dead-code reset, strict checkpoint validation, and counter persistence are implemented. | Supply a valid trained VQ-VAE artifact and report utilization/sample efficiency. |
| III | Graph/local conditioning, GPS/RRWP options, topology maps, and reference-room paths exist. | Run matched-budget target-response and conditioning ablations. |
| IV | Diffusion, DiT, flow matching, masked generation, categorical baseline, and fast sampling exist as alternatives. | Train valid checkpoints and compare quality, latency, memory, and pre-repair validity. |
| V | LogicNet grid/graph reachability and guidance paths exist. | Run LogicNet ON/OFF with raw pre-repair hard-oracle rates. |
| VI | WFC, flat-prior WFC, repair feedback, overlays, and puzzle scaffolds exist. | Quantify repair contribution, failure rate, and runtime separately. |
| VII | Tile solvers, resource-aware graph validation, feasible-only QD archives, P-CBS, QD metrics, and statistical scripts exist. | Execute final graph-to-map oracle tables, persona tables, paired tests, and human calibration. |

## Confirmed Fixes In Latest Pass

### Pipeline and data

- The 2026-07-03 integration audit removed several silent method changes:
  `AdvancedNeuralSymbolicPipeline` now defaults to strict checkpoint loading,
  delegates graph tensors to the canonical conditioning schema, and makes
  bordered-room and WFC-failure fallbacks opt-in. Saved stats include fallback
  counts rather than reporting fallback output as neural output.
- Weighted Bayesian WFC priors are no longer estimated from Gaussian random
  VQ-VAE decodes. The advanced pipeline requires explicit training-only
  semantic grids, records the resolved source, grid count, and SHA-256 hash,
  and leaves weighted WFC disabled when no empirical prior source is supplied.
- Fast-sampler speedup is no longer computed from a fabricated 45-second
  per-room baseline. It is `null` unless a paired baseline duration is supplied
  for the same run protocol.
- The legacy `enable_ara` path was verified against the ARA* algorithm
  definition. It performs fixed-weight weighted A*, not Anytime Repairing A*.
  Canonical config keys and user-facing documentation now use
  `enable_weighted_astar`/`heuristic_weight`; legacy keys remain accepted only
  for compatibility.
- `PipelineBlock` catches ordinary contract exceptions, honors
  `retryable=False`, and does not launch concurrent retries after a thread
  timeout that Python cannot safely cancel.
- The advanced pipeline no longer emits a private six-feature graph schema; it
  uses the condition encoder's actual node/edge dimensions, edge semantics,
  positional encodings, masks, and stable node mapping.
- The key-economy diagnostic now preserves explicit cyclic START nodes, excludes
  keys trapped behind the lock being evaluated, ignores empty `key_id` values,
  and computes route membership using reachability intersections instead of
  exponential all-simple-path enumeration.
- Canonical layout validation no longer counts tile `0` (VOID) as FLOOR.
  Legacy zero-floor grids must declare `legacy_zero_is_floor=True`.
- Hung GUI solver recovery deletes tracked IPC artifacts after the process has
  stopped and before recovery clears their paths.
- A stale regression test that treated unprotected hazard edges as freely
  traversable was corrected. The graph and tile validators consistently require
  the permanent traversal/protection item.
- Broad autonomous pass on 2026-07-02 rechecked the recent high-risk claims
  against code rather than Graphify alone. The damped tension-spike regression,
  exact-match bidirectional inventory regression, ModelContextContractError
  swallowing regression, PhaseAligner wraparound, categorical-token
  normalization, and node-cap bridge-to-key deletion are not present in the
  current implementation.
- Pure MaskGIT no longer loads unused VQ-VAE/diffusion models.
- Categorical codebook sampling is explicitly an unconditional baseline and no
  longer loads or computes an unused condition encoder, including batched
  generation.
- Enabling a teacher fallback loads the required diffusion teacher stack.
- Strict VQ-VAE loading rejects missing learned keys.
- Text datasets scan all files before `pad_to_max` batching.
- Lazy VGLC `get_raw_grid()` uses the indexed VGLC loader.
- Categorical masked-room training keeps integer token IDs.
- Graph-to-room alignment cannot silently map two graph nodes to one room.
- Robust block timeouts return without waiting for the timed-out worker.
- NetworkX 3.5 artifact serialization uses standard pickle APIs.
- Invalid BSP dimensions fail early instead of creating out-of-bounds rooms.
- Room-dataset construction fails with dungeon/variant context on parser or
  graph-extraction errors instead of silently dropping samples.
- Script cleanup removed one-off debug probes, duplicate wrappers, stale dated
  queue launchers, GUI/demo generators, asset cutters, and unreferenced status
  probes from `scripts/`. The physical `scripts/` tree currently contains 71
  Python files focused on canonical `run_*`, `generate_*`, `validate_*`,
  analysis, and training utilities.
- Top-level docs cleanup removed stale GUI, handoff, final-verdict, and
  superseded P-CBS/evaluation notes. Current claim boundaries live in this
  ledger.
- Active ablation manifests now fail closed on evidence. A zero exit code
  without required metric artifacts is labelled
  `completed_needs_metric_artifact`, not treated as passed scientific evidence.
- The canonical rationale guide now matches the active config: teacher fallbacks
  default off and are only enabled for explicit guarded-runtime ablations.

### Model and training

- MaskGIT corruption sampling now covers the complete `[0, 1]` mask-ratio
  range by default. Inference derives the step embedding from the actual
  unresolved-token fraction, so shortened/extended sampling schedules do not
  use inconsistent or out-of-range step indices. The bounds remain
  configurable ablations.
- MaskGIT cross-entropy uses `ignore_index=-100` for unmasked context tokens,
  and concat mode does not allocate an unused Transformer decoder.
- Masked-room configuration no longer advertises unsupported linear-attention,
  SPADE, or U-Net controls as executable ablations. Transformer hidden width,
  head count, depth, and dropout are validated against the actual backbone.
- VQ-VAE DDP collectives no longer catch `RuntimeError` and continue with
  replica-local EMA statistics. Single-process mode is detected explicitly;
  initialized distributed jobs fail if synchronization fails.
- Categorical codebook sampling stays on-device and no longer suppresses
  failures raised by an available codebook-usage API.
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
- Additive-versus-SPADE execution now requires separate architecture-matched
  checkpoints and verifies the loaded model's conditioning mode before sampling.
- Topology-refinement ablations verify set/readback state and fail instead of
  silently retaining a checkpoint's default attention mode.
- Fast-sampling telemetry separates requested from actually executed fast paths.
- Advanced-pipeline global water state now reaches the existing state-aware
  room transformation instead of remaining unused graph metadata. This is a
  deterministic state transformation, not learned state conditioning.
- Generation-time room probability and entropy metrics now use fp32 softmax /
  log-softmax before detaching for NumPy conversion, matching the AMP-safe
  pattern used inside the model layers.
- WFC pseudo-label confidence extraction now applies the same fp32 softmax
  pattern, preventing mixed-precision overflow from corrupting repair targets.
- MaskGIT iterative filling ranks stochastic token commits with
  `log(confidence) + Gumbel` rather than adding Gumbel noise directly to raw
  `[0, 1]` probabilities. This restores the intended Gumbel-Max scale.
- Deterministic graph-context contract failures are no longer subclasses of
  `ValueError`. `ModelContextContractError` and strict conditioning schema
  errors are non-retryable, so robust generation loops cannot quietly convert
  schema overflow into endless zero-fitness retries.
- Graph conditioning now has a `condition_strict_schema` ablation flag wired
  through shared config, inference runtime, checkpoint fallback construction,
  diffusion training, and masked-room training. Compatibility pad/truncate
  remains the default; strict mode fails closed for schema-drift audits.
- The learned A* heuristic no longer leaks or duplicates key count in the
  collected-item feature. Its locked-door feature uses dynamic `opened_doors`
  state against actual normal locked-door coordinates instead of a static grid
  count throughout the solve.

### Solvers and metrics

- The legacy `ParallelAStarSolver` API now delegates to canonical state-space
  A*. Its former workers raced from one root behind a shared closed set, so
  only one worker could expand the root and the implementation was neither
  HDA* nor a real parallel speedup. Duplicate top-level `simulation/` shims
  were removed; `src/simulation/` is the only implementation.
- IDDFS deepening iterations share one global expansion budget rather than
  each consuming the full timeout.
- Perturb-and-MAP backward propagation now follows each Dijkstra predecessor
  chain. The previous surrogate marked the entire reachable component and
  incorrectly sent a goal gradient through off-path cells and the source.
- Evolutionary feasibility now requires weak connectivity across the complete
  mission graph, not only a surviving START-to-GOAL route. Final selection
  therefore cannot promote an elite with detached keys, switches, tokens, or
  challenge rooms.
- Descriptor-driven grammar repairs reject candidates that disconnect the
  mission. A failed multi-lock transaction rolls back the switches it created
  instead of leaving unowned state nodes behind.
- `InsertLockKeyRule` rolls back the whole transaction if it cannot place both
  the key spur and a causally reachable lock. The fallback search now starts
  from the trunk source where the key spur attaches, not from the key spur
  itself, which has no forward continuation.
- Final topology connectivity handling prunes disconnected optional decoration
  and unreferenced surplus providers, but rejects providers referenced by an
  active gate and all other disconnected progression anchors. It no longer
  fabricates PATH/unkeyed BOSS_LOCKED edges to make an invalid mission graph
  appear valid.
- Goal-gauntlet repair now discards unreachable preserved approaches and selects
  the deepest directed traversable progression node. It no longer fabricates a
  `START -> orphan` bridge that bypasses the generated dungeon.
- The terminal chain validator now requires canonical `PATH` edges and one
  correctly keyed `BOSS_LOCKED` approach edge. Visual-only and unlocked edges
  cannot satisfy the gauntlet contract.
- Gauntlet cleanup is scoped to rejected approach artifacts; it no longer
  deletes unrelated disconnected components.
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
- Grid path linearity is geometric route directness
  (`Manhattan displacement / movement steps`) on a concrete solver path.
  Mission-graph linearity is a separate structural descriptor based on
  alternate-route chords, reconnecting branches, and cycle pressure. Adding
  arbitrary dead-end leaves does not improve either descriptor. The former
  inverse route-coverage proxy is retained only as explicitly named
  `route_sparsity`.
- Confusion ratio is normalized as excess path overhead, where an oracle-match
  is `0.0`.
- The main ablation runner now uses transition counts, the canonical
  A*/Dijkstra oracle wrapper, and the same excess-path confusion definition.
- Legacy benchmark, fast-sampler audit, Dungeon-9 evaluation, and Chapter-4
  figure scripts now use the same transition-count helper; they no longer mix
  visited-state counts with movement counts.
- Path efficiency is consistently bounded as `oracle_length / candidate_length`
  in `[0, 1]`. Telemetry calibration no longer treats the inverse path-effort
  ratio as efficiency, and higher efficiency now reduces inferred boundedness.
- The explicitly labeled `cbs_steps_per_unique_tile` metric remains a separate
  revisit-density descriptor and must not be reported as oracle-relative
  confusion.
- The priority-mode A/B benchmark is import-safe and headless; it constructs a
  prepared `ZeldaLogicEnv` directly instead of instantiating the GUI runner.
- A* timeout is reported as indeterminate rather than mislabeled unsolvable in
  CBS fitness.
- P-CBS suboptimal-decision accounting uses the same graph-aware navigation
  distance as action scoring.
- Graph feasibility now uses an exact resource-state oracle for cumulative
  small-key consumption, specific key identities, permanent boss keys, named
  items, switches, tokens, resource providers, and protected hazards.
- Canonical edge parsing preserves `MULTI_LOCK` instead of silently reducing it
  to a one-key lock.
- Evolutionary feasibility runs the exact graph oracle after grammar metadata
  checks; weak connectivity and pre-gate provider counts are not accepted as a
  complete solvability proof.
- Grid and CVT MAP-Elites archives reject infeasible candidates before cell
  insertion. An unsolvable zero-fitness graph can no longer occupy an empty
  behavior cell.
- Graph-to-grid compilation preserves supported gate types and emits one
  consumable gate tile per physical room connection. The opposite boundary is
  open, so one graph lock does not consume two keys.
- Advanced generation materializes START, GOAL, keys, boss keys, traversal
  items, enemies, bosses, and hazards on the semantic artifact before running
  the tile-state oracle.
- Protected hazards are validated by named item at graph level and compile to
  `ELEMENT` plus the generic `KEY_ITEM` protection at tile level. Multiple
  distinct protection identities are rejected because the current tile
  vocabulary cannot preserve that distinction.
- Mission normalization no longer trims arbitrary topological prefixes,
  reassigns the last node as GOAL, or connects islands with open edges.
  Oversized valid graphs are preserved; disconnected phenotypes are rejected;
  requested extra rooms become optional reachable branches.
- Deterministic linear mission fallback is disabled by default. Evolutionary
  generation failures now remain failures unless a diagnostic run explicitly
  enables `allow_linear_graph_fallback`.
- End-to-end generation defaults to the explicit `spatial` grammar profile.
  It retains mechanics with a faithful tile/entity/oracle contract; `core` is
  the minimal grammar ablation and `full` remains graph-only until every
  additional mechanic has matching compiler and validator semantics.
- Progression and gate-economy repair use the same grammar profile as the
  evolutionary run. Repairs can no longer inject graph-only item gates,
  state blocks, shutters, or multi-locks into a spatially compilable graph.
- Constrained spatial/full populations include one fully evaluated feasible
  anchor genome. It is not a fallback artifact and receives no privileged
  score; it prevents a small random initial population from containing no
  feasible parent at all.
- Topology CVT-emitter search is now the advanced pipeline default. Its archive
  rejects mechanically infeasible genomes, while soft target mismatch remains
  a quality penalty rather than an impossible exact-zero feasibility test.
- CVT centroid initialization and elite sampling use archive-local seeded RNGs,
  making paired-seed topology-QD runs reproducible.
- Final-map MAP-Elites state is no longer cleared before every insertion.
  Archive size and coverage are reported so a one-artifact archive cannot be
  presented as population diversity.
- Graph and tile oracles distinguish `budget_exhausted` from an exhausted
  state space. Capped searches are indeterminate, not proven unsolvable.
- Bidirectional search and canonical A* fallback share one expansion budget;
  diagnostic reporting includes work from both phases.
- P-CBS filtering is an explicit, disabled-by-default post-refinement
  ablation. Its persona, expansion budget, normalized-confusion threshold, and
  rejection rate must be reported separately from exact solvability.
- Composite gate labels enforce all conjuncts, and token counts are tracked by
  token identity instead of pooling unrelated token types.
- QD candidate export now tries ranked feasible candidates and returns the
  highest-quality phenotype that survives final export validation.

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

### Implemented Publication Matrix

`scripts/generate_round5_scientific_gap_manifest.py` now defaults to the five
requested experiment families and emits paired seed-42/43/44 jobs. Plan rows
are classified as `ready_to_execute`, `waiting_for_manifest_dependencies`, or
`blocked_missing_inputs`; a planned row is never treated as evidence:

- additive versus SPADE diffusion training and paired evaluation, including
  topology preservation, validity, runtime, and parameter-count disclosure;
- 50-step diffusion versus 4-step graph-aware `consistency_lora` generation on
  the identical mission graph and seed, with paired speed, hard-oracle,
  P-CBS, repair, NCD, and entropy deltas;
- novice, balanced, and expert P-CBS component ablations with identical maps,
  seeds, A* budgets, and P-CBS state budgets;
- weighted Bayesian versus flat-prior scaffolded symbolic WFC under paired
  seeds and sample counts;
- 100-, 250-, and 500-room controllability stress rows reporting normalized
  target error, node-count pass rate, and generation time.

Execution is fail-closed. A zero process exit is insufficient: required input
artifacts, output files, and metric keys must exist before a run receives
`passed` status. The fast-sampler preflight accepts only this repository's
`consistency_lora` artifact and records that it is not a paper-faithful
LCM-LoRA runtime.

Research basis:

- [SPADE](https://arxiv.org/abs/1903.07291) motivates learned spatial affine
  modulation versus direct additive conditioning.
- [Latent Consistency Models](https://arxiv.org/abs/2310.04378) motivates the
  paired few-step quality/latency test, without overriding the repository's
  stricter artifact-type boundary.
- [G-PCGRL](https://arxiv.org/abs/2407.10483) motivates explicit graph node-count
  controllability metrics.
- [PCGRL+](https://arxiv.org/abs/2408.12525) motivates out-of-distribution scale
  evaluation rather than ordinary-size extrapolation.
- [WaveFunctionCollapse is Constraint Solving in the Wild](https://doi.org/10.1145/3102071.3110566)
  motivates holding constraints fixed while ablating learned pattern priors.

## Experiment Status

### Executed but limited

- `results/random_baseline/random_baseline_results.json`: 96 samples for each
  of seeds 42, 43, and 44. This is scientifically executed but scope-limited
  topology-null evidence; it lacks a raw graph archive and full environment
  manifest and cannot support model-quality comparisons.
- `results/baselines/wfc_dryrun_codex/wfc_baseline_report.json` contains one
  four-sample `wfc_overlapping_patterns` dry run with P-CBS disabled. It has no
  flat-prior arm and is smoke-executed plumbing evidence, not an ablation result.
- `results/wfc_prior_paired_3seed_scaffold_smoke/` verifies the operationally
  neural-free but graph-scaffolded paired WFC protocol with one sample per seed.
  Weighted priors were
  oracle-solvable on 3/3 seeds versus 0/3 for flat priors, while weighted tile
  KL was worse on all three seeds. This is a low-power tradeoff diagnostic, not
  publication evidence; the planned multi-sample run remains required.
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
- SPADE versus additive conditioning has not been executed. A short SPADE run
  (`epoch=3`, `global_step=32`) exists without a matched additive checkpoint or
  comparison metrics, so it is not ablation evidence. The runner is now wired
  to reject mismatched or missing arm-specific checkpoints.
- model architecture and attention ablations.
- LogicNet loss-component ablations.
- diffusion versus the fast sampler has not been executed. The implemented path
  identifies itself as repo-specific `consistency_lora` with DDIM semantics and
  explicitly rejects paper `lcm_lora` artifacts; it must not be reported as a
  paper-faithful LCM-LoRA baseline.
- a scientifically powered weighted versus flat-prior WFC run.
- full persona/component tables and paired significance tests.
- room-semantic target-response before repair remains unimplemented. The
  current executable target-response protocol tests mission-graph descriptor
  response only and must be labeled as such.
- large-dungeon controllability studies.

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

- state-of-the-art or "surpasses publications" claims;
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

Non-experiment publication contract work now completed:

- A canonical publication guidance/card document was added.
- A machine-readable publication card was generated and validated.
- `scripts/validate_publication_readiness.py` now checks card completeness,
  artifact status/hash discipline, required metric names, human-calibration
  status, and conservative claim-language boundaries.
- Documentation entrypoints now route readers to the publication contract before
  older benchmark/SOTA-named notes.

## External Pathfinding And SOTA Audit Triage

The attached broad audit was checked against the current source and primary
literature. Its recommendations cannot be accepted as a single upgrade plan:

- P-CBS in this repository is **Persona-Driven Cognitive Bounded Search**, a
  single-agent behavioral simulator. It is not Conflict-Based Search for
  multi-agent path finding, so ECBS/EECBS optimality claims do not apply.
- Bidirectional search had a real correctness boundary: first-frontier meeting
  was not an optimality certificate, and one guessed goal inventory cannot
  define complete reverse transitions for keys, blocks, staged puzzles, or
  directed warps. It now uses the reversible-grid path only when a returned path
  attains the geometric lower bound; stateful cases use canonical full-state A*.
- CVT-MAP-Elites is already implemented in
  `src/evaluation/map_elites.py`; describing it as absent was stale.
- MaskGIT-style discrete generation and a DiT rectified-flow objective/sampler
  ablation are already implemented. Neither has publication evidence without
  trained matched-budget checkpoints.
- Theta* is an any-angle planner for continuous line-of-sight motion. Zelda tile
  actions and door/item transitions are discrete, so adding it would change the
  game rules rather than improve the current oracle.
- The name `PCGBench` is ambiguous: a 2024 work uses it for parallel code
  generation, while Khalifa et al.'s FDG 2025 **PCG Benchmark** is a relevant
  game-content testbed with quality, diversity, and controllability criteria.
  Its Zelda task is a small key-door maze, so it is a useful external baseline
  but not a drop-in replacement for this repository's multi-room rule set.
- SAT/SMT/CDCL and learned probabilistic graph grammars remain possible separate
  baselines. They are hypotheses requiring explicit encodings, budgets, and
  paired evidence, not correctness patches for Weighted Bayesian WFC.
- `FunMetricsEvaluator` remains a compatibility name. Its values are structural
  experience proxies and cannot support human-fun claims without calibration.

Primary references used for this triage:

- Li, Ruml, and Koenig, EECBS: https://arxiv.org/abs/2010.01367
- Vassiliades et al., CVT-MAP-Elites:
  https://doi.org/10.1109/TEVC.2017.2735550
- Chang et al., MaskGIT: https://arxiv.org/abs/2202.04200
- Lipman et al., Flow Matching: https://arxiv.org/abs/2210.02747
- Nash et al., Theta*: https://doi.org/10.1609/aaai.v21i1.788
- Khalifa et al., PCGRL: https://arxiv.org/abs/2001.09212
- Earle et al., PCGRL+: https://arxiv.org/abs/2408.12525
- Khalifa et al., PCG Benchmark:
  https://github.com/amidos2006/pcg_benchmark
- Cooper, Sturgeon constraint-based level generation:
  https://doi.org/10.1609/aiide.v18i1.21944
- Summerville, expressive-range evaluation:
  https://doi.org/10.1609/aiide.v14i1.13012
- PyTorch scaled-dot-product attention mask contract:
  https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- Likhachev, Gordon, and Thrun, ARA*:
  https://papers.neurips.cc/paper/2382-ara-anytime-a-with-provable-bounds-on-sub-optimality

## Independent Production-Path Audit (2026-07-03)

This pass traced production call paths rather than accepting the accumulated
audit ledger as ground truth. The following reported issues were already fixed
and were excluded as false alarms: categorical normalization in masked-room
training, LCM zero-time anchoring, Gumbel confidence ranking, exact scheduler
duration after dataloader construction, and cumulative LogicNet key counting.

Six remaining correctness failures were confirmed and fixed:

- Diffusion training and validation no longer replace failed graph
  conditioning with dummy conditioning. A graph-schema failure now aborts the
  run instead of producing an unconditional checkpoint or invalid validation
  metrics under a graph-conditioned label.
- Mixed `room_topology_map` shapes now fail the batch contract. Topology is no
  longer silently omitted for one malformed batch.
- MaskGIT training now derives fixed semantic entity tokens from topology
  roles, matching runtime generation. This matters because the real VGLC room
  corpus often stores graph keys only in DOT labels: a Dungeon 1 Quest 1 probe
  found six key-role rooms, with five requiring floor-to-key anchor
  materialization.
- Structural cycle rank, branching choices, and dead ends use a simple
  undirected physical projection. Reciprocal directed arcs no longer turn a
  corridor tree into fake cycles or duplicate choices.
- Checkpoint sidecars now include SHA-256. Pipeline loading validates size and
  hash before tensor deserialization, and all model loaders reject missing
  learned parameters even in compatibility mode.
- The Round-5 experiment executor now rejects unchanged pre-existing outputs,
  header-only CSVs, empty JSON, and missing required metrics. Passed runs record
  output hashes, byte sizes, and record counts.

Research boundary:

- MaskGIT supports iterative masked-token prediction and confidence-based
  scheduled decoding; the repository's fixed topology anchors are a
  task-specific inpainting constraint and must be described as such, not as an
  original MaskGIT mechanism.
- MAP-Elites behavior descriptors define the archive cells. Physical loop and
  branch descriptors therefore must not depend on whether one corridor was
  serialized as one edge or two reciprocal arcs.
- These are implementation-contract fixes. They do not establish model quality
  until matched checkpoints and planned experiments are executed.

## Verification Policy

Before publication lock:

- run the complete non-GUI suite serially;
- run focused gradient/finite-value probes on the target GPU;
- archive exact commands, configs, seeds, checkpoint hashes, environment, and
  required metric files;
- fail experiment completion when required outputs or provenance are missing;
- regenerate optional navigation indexes only after source verification;
- regenerate this ledger from current evidence rather than appending another
  dated audit section.

Test policy:

- tests must call production behavior rather than inspect source strings;
- deterministic failures must not be converted to skips or conditional passes;
- required dependencies such as PyTorch and NetworkX must fail import normally;
- GPU, external-dataset, and genuinely optional-format tests may skip when
  their prerequisite is absent;
- generated demo artifacts and long-running experiment subprocesses are not
  unit tests and must not block the default correctness suite.

Latest local verification (2026-07-03):

- This independent pass ran 223 focused behavior tests across checkpoint
  retention, protocol reporting, MaskGIT, structural metrics, graph-conditioned
  diffusion, LogicNet optimization, block integration, and pipeline reliability;
  all passed.
- A real-data probe loaded Dungeon 1 Quest 1 through the production
  `ZeldaRoomDataset` and verified topology-derived key anchors.
- Direct production probes rejected both a tampered checkpoint and a checkpoint
  missing a learned parameter.
- Ruff passed on every Python file changed in this pass with `E402` ignored for
  `train_diffusion.py`, whose pre-existing path-bootstrap import layout is
  intentional.
- Compile checks passed on every Python file changed in this pass.
- The complete repository suite, target-GPU finite-gradient probes, and final
  checkpoint-backed end-to-end run were not executed in this pass and remain
  publication-lock requirements.

### Component And Pipeline Recheck (2026-07-03)

Component map checked in this pass:

- Data and alignment: `src/zelda_data/`, `src/data_processing/`, and
  room/topology conditioning helpers.
- Graph generation and QD: `src/generation/grammar/`,
  `src/generation/evolutionary_director/`, MAP-Elites evaluators, and
  controllability scripts.
- Model stack: VQ-VAE/FSQ, condition encoder, latent diffusion/DiT, MaskGIT,
  LogicNet, LCM-LoRA fast sampler, categorical sampler, and model manager.
- Inference pipeline: graph context, room sampler, room stitching, semantic
  constrained decoding, WFC repair, overlay, and robust retry orchestration.
- Validation and search: full-state A*, Dijkstra fallback, role-limited
  Bidirectional A*, role-limited D* Lite, P-CBS, graph key-economy validation,
  and external benchmark adapters.
- Experiment layer: SPADE/additive topology conditioning, LCM-LoRA paired
  quality, P-CBS matched-budget personas, WFC prior controls, random baseline,
  fixed-graph multi-seed audit, and 100/250/500-room controllability stress.

Confirmed implementation state:

- The strict publication-facing solvability oracle remains full-state A*;
  Bidirectional A* and D* Lite are diagnostics only on reversible stateless
  grids.
- Categorical data loading keeps discrete token IDs unnormalized when
  `categorical_tokens=True`.
- Categorical sampler decoding keeps exact `decode_indices` logits and samples
  on-device.
- MaskGIT context shape failures and condition-encoder schema failures are
  non-retryable contract errors in the robust block loop.
- The Round-5 manifest wires SPADE/additive, LCM-LoRA, WFC-prior, P-CBS, and
  controllability commands, and rejects stale/empty/missing-metric outputs.

New fix in this pass:

- Graph-conditioning schema drift now fails closed by default across
  `config_system`, `PipelineConfig`, runtime initialization, config bridging,
  diffusion training, masked-room training, and direct condition-encoder
  construction. Legacy pad/truncate compatibility remains available only by
  explicitly setting `condition_strict_schema=False`.

### Search And Key-Economy Audit Addendum (2026-07-03)

Confirmed and fixed:

- `KeyEconomyValidator` now normalizes lock semantics from both the legacy
  `lock_type` field and the newer `edge_type`/`type`/`label` fields. A graph
  edge marked only as `edge_type="key_locked"` is no longer treated as open by
  greedy/adversarial traversal or key-surplus analysis.
- `KeyEconomyValidator` now enforces `item_gate`/`item_locked` resources
  instead of treating them as passable unknown lock types. It also recognizes
  `item_type` and `drops_resource` as providers, so grammar item gates and
  resource farms use the same schema in validation.
- Specific boss-door key identifiers are honored via `key_required`/`key_id`;
  persistent boss keys remain reusable, but a non-matching or missing provider
  no longer silently passes.
- `LOCK`, `BOSS_DOOR`, and other consumer nodes no longer become key providers
  merely because their metadata contains `key_id`. This closes a false-positive
  route where the required-key annotation could be collected after traversal.
- Graph traversal now infers omitted `key_required` / `item_required` fields
  from target consumer nodes when exporters store the requirement on
  `LOCK`, `BOSS_DOOR`, or puzzle nodes instead of duplicating it on the edge.
- MAP-Elites macro-feasibility and the external `AgentSimulator` now use the
  same consumer/provider distinction, so QD descriptors and benchmark
  solvability do not reward keys or items that only exist as requirement
  metadata.
- External graph validation no longer treats `required_item` as an item
  provider. Legacy untyped `I` nodes still act as wildcard item pickups, but
  consumer-only requirement metadata does not create inventory.
- Masked-room checkpoint resume now refuses LogicNet states that omit learned
  parameters, preventing mixed trained/random LogicNet supervision after
  partial checkpoints.
- `BidirectionalAStar` now treats `DOOR_SOFT` and `PUZZLE` tiles as outside
  its reversible-grid problem class. Those maps are delegated to canonical
  full-state A*, matching the documented boundary that bidirectional reverse
  search is only a diagnostic fast path for reversible position-only grids.

Checked and retained:

- D* Lite is already goal-rooted and disabled for irreversible Zelda state
  mechanics. It remains a reversible-grid replanning diagnostic, not the
  publication-facing solvability oracle.
- Tension-curve event matching already detects spikes on the raw, unsmoothed
  curves before normalized-progress interpolation.
- ML heuristic inference no longer constructs a trainer per query, no longer
  leaks the target label into features, and scales calibration in absolute cost
  units.

Focused verification:

- `python -m pytest` passed the key-economy own-lock regression, the new
  `edge_type="key_locked"` regression, the bidirectional stateful fallback
  regression, and the new one-way soft-door fallback regression.
- A follow-up focused run passed the item-gate softlock, specific boss-key,
  consumer-node key metadata, own-lock, edge-type locked, and persistent
  boss-key key-economy regressions.
- A second follow-up run passed endpoint-inferred boss-key/item-gate
  regressions across `KeyEconomyValidator`, MAP-Elites macro feasibility, and
  the external `AgentSimulator`; the full `tests/test_critical_review_fixes.py`
  file passed with 53 behavior tests.
- Ruff passed on the touched search/key-economy files and their focused tests.
- `python -m graphify update .` completed after one timeout retry; the final
  run reported no code-graph topology changes.

### Training, Data, And GUI Reliability Addendum (2026-07-03)

Confirmed and fixed:

- AMP non-finite-gradient skip paths in diffusion and DPO training now still
  notify `GradScaler.update()` outside Accelerate. This lets AMP reduce its
  scale after detected non-finite gradients instead of repeatedly retrying the
  same invalid scale.
- WFC pseudo-label supervision now keeps proportional full-batch weighting:
  successful repaired samples are still averaged for diagnostics, but the loss
  used by training is divided by the original batch's `B * H * W` support so
  one repaired sample does not carry the same weight as a fully repaired batch.
- Zelda semantic-grid normalization no longer special-cases binary-looking
  rooms. When `normalize=True`, raw tile IDs are always scaled by the semantic
  palette size; categorical-token loaders still keep integer token IDs.
- VGLC dungeon and room dataset ingestion now builds the optional graph sample
  before appending the aligned map/metadata rows. Failed graph extraction no
  longer leaves `samples`, `sample_metadata`, and `graphs` with different
  lengths.
- GUI font construction now flows through `src/gui/rendering/font_cache.py`.
  Direct `pygame.font.SysFont` / `pygame.font.Font(None, ...)` calls outside
  the cache were removed from GUI code, preventing repeated OS font registry
  churn during rendering.

Checked and retained:

- Frame-local overlay `Surface` allocations were not globally pooled in this
  pass. Many of those surfaces are intentional transient alpha buffers whose
  safe reuse depends on size, alpha, and clipping state; pooling should be a
  dedicated renderer refactor with visual regression screenshots.

Focused verification:

- Compile and Ruff checks passed on the touched training, data-loading, and GUI
  files.
- A direct production `ZeldaRoomDataset.__getitem__` probe confirmed that a
  binary semantic grid now normalizes max tile value to `1 / 43` instead of
  leaking a raw `1.0`.
- `python -m graphify update .` completed after this pass.

### Reliability Guard Addendum (2026-07-03)

Confirmed and fixed:

- MAP-Elites archives now reject non-finite fitness values before inserting a
  new elite. If an older saved archive already contains a non-finite elite, a
  future finite candidate for that cell can replace it, and aggregate archive
  statistics ignore non-finite legacy scores.
- Distributed gradient averaging no longer skips local `None` gradients before
  collectives. Each parameter first reduces a per-rank gradient-presence flag;
  if any rank has a gradient, all ranks participate in the gradient all-reduce
  with zeros for inactive local parameters. Parameters unused on every rank
  remain `grad=None`.
- `CheckpointManager.save()` now uses `atomic_torch_save()` for the epoch
  checkpoint and for the `best_model.pth` / `checkpoint_latest.pth` aliases,
  avoiding half-written target files after interruption.
- Pygame heatmap rendering now ignores invalid visit counts and clamps
  non-finite interpolation values before converting colors to integers.
- Attention-map PNG export now closes each matplotlib figure in a `finally`
  block, so failed `savefig()` calls do not leak figures into global state.
- Evolutionary genealogy lineage traversal now detects cyclic parent links and
  missing parent records instead of looping forever.

Focused verification:

- Compile and Ruff checks passed on the touched archive, distributed,
  checkpoint, renderer, attention visualization, and explainability modules.
- Direct probes confirmed MAP-Elites NaN rejection/replacement, heatmap
  non-finite color handling, and genealogy cycle detection.
- The distributed fix was not exercised with a real multi-rank GPU job in this
  pass; it was patched at the collective ordering level and still needs a
  torchrun MoE/dynamic-graph smoke run before publication claims.

### Repository-Wide Audit Pass (2026-07-03)

Components checked in this pass:

- Training loops: diffusion, masked-room, LCM-LoRA, VQ-VAE, Gaussian VAE, and
  LogicNet tile-classifier scripts.
- Pipeline artifacts: advanced pipeline result export, generation sampler
  entry points, robust orchestration, and room stitching surfaces.
- Quality-diversity archives: simulation MAP-Elites, evaluation MAP-Elites,
  CVT emitter archives, archive JSON/export paths, and QD metric aggregation.
- Validation/search surface: full-state A*, diagnostic bidirectional/D* Lite
  boundaries, P-CBS evaluation utilities, and timeout reporting helpers.
- Data and model I/O: Zelda dataset loaders, processed-data adapters,
  checkpoint utilities, heuristic-learning model save/load, and experiment
  output writers.
- Ablation scripts: Round-5 study runner, LogicNet repair ablation,
  fast-sampler visual audit, P-CBS component/persona sweeps, designer
  controllability, random baseline, and paired-seed reporting scripts.

Confirmed and fixed:

- The lightweight simulation MAP-Elites wrapper now rejects all non-finite
  scores, not just `NaN`, before inserting into its legacy grid or mirroring
  into the CVT archive.
- MAP-Elites and CVT archive persistence now uses temporary files followed by
  `replace()` so interrupted archive saves do not corrupt the active archive.
- Evolutionary-director CVT QD archive persistence now writes atomically.
- Processed Zelda adapter dumps now write through a temporary pickle file
  before replacing the requested output.
- Heuristic-learning model checkpoints and LogicNet tile-classifier
  checkpoints now use the repository `atomic_torch_save()` helper.
- Advanced pipeline artifact export now writes dungeon grids, visual grids,
  mission graph, and stats through temp files before replacement. Stats JSON is
  written with `allow_nan=False` so non-standard NaN evidence cannot silently
  enter artifact bundles.

Checked and retained:

- The remaining explicit `torch.save()` call is inside `atomic_torch_save()`.
- The remaining `while True` loops in production search/generation code are
  bounded by visited sets, Bresenham endpoint equality, or interactive CLI
  commands; no new non-interactive infinite loop was confirmed in this pass.
- Several experiment scripts still intentionally emit planned manifests or
  `NaN` placeholders before sanitization. Those are acceptable only when they
  are explicitly labeled as planned/missing evidence and are not promoted to
  executed result tables.

### Direct Source Audit And Architecture Wiring Pass (2026-07-03)

Directly inspected components:

- Block I: grammar generation, evolutionary director, MAP-Elites/CVT archives,
  graph descriptors, and node-cap/feasibility boundaries.
- Block II: VQ-VAE and Gaussian-VAE models, codebook maintenance, standalone
  trainers, validation aggregation, and checkpoint selection.
- Block III: strict graph-conditioning schemas, padded-node masks, all-masked
  attention rows, topology maps, and edge-semantic conditioning.
- Block IV: U-Net/DiT diffusion, LCM-LoRA fast sampling, categorical and
  MaskGIT paths, attention/topology refinement modes, and sampler decoding.
- Block V: LogicNet supervision and guidance wiring in diffusion and
  masked-room training.
- Block VI: weighted Bayesian WFC propagation, backtracking/restart bounds,
  symbolic repair, and pre/post-repair metric separation.
- Block VII: finite archive contracts, QD replacement semantics, persistence,
  and archive statistics.
- Cross-cutting: full-state A* oracle, restricted bidirectional/D* Lite roles,
  P-CBS behavioral evaluation, retry contracts, dataset alignment,
  distributed gradients, experiment manifests, and atomic artifacts.

Confirmed and fixed:

- Uniform-grid and CVT archives now reject non-finite or wrong-dimensional
  behavior descriptors before cell lookup. Loaded uniform archives discard
  invalid legacy elites, and feature-diversity statistics use only valid rows.
- The simulation MAP-Elites archive now lets a finite candidate replace a
  legacy non-finite cell and discards invalid bins while loading.
- `atomic_torch_save()` now uses a unique sibling temp file and removes it on
  failure. Filename-only heuristic checkpoints work without requiring a
  directory component.
- `CheckpointManager` now updates `best_metric` even when regular epoch
  checkpoints are enabled. Missing/non-finite best metrics cannot overwrite
  `best_model.pth`; invalid comparison modes fail at construction.
- VQ-VAE and Gaussian-VAE trainers now skip optimizer updates for non-finite
  losses, gradients, or clipped gradient norms. Their training/evaluation
  loops exclude skipped batches and fail instead of saving a zero-valued model
  selection metric when every batch is invalid.
- Diffusion, masked-room, and LCM-LoRA epoch aggregation now excludes
  explicitly skipped non-finite batches and reports the skipped count.
- Diffusion and DPO loss-finiteness decisions are reduced across all workers
  before backward. One rank can no longer return early while peers enter
  gradient collectives for the same batch.
- MaskGIT edge-aware topology bias now raises a non-retryable contract error
  on topology/logit shape drift instead of silently generating without graph
  constraints.
- Learned Graphormer distance and degree clipping are configurable through
  config, training CLI, checkpoints, and inference reconstruction instead of
  being fixed at 16/64.
- The architecture-ablation manifest now has explicit single-factor attention,
  topology-refinement, and topology-conditioning arms. Runs disable implicit
  auto-resume, declare `final_model.pth` as evidence, and validate metrics that
  training actually writes. Publication-only quality/efficiency metrics remain
  separately labeled as requiring downstream evaluation.
- Weighted Bayesian WFC now validates prior domains at construction,
  precomputes directional compatibility matrices, and uses a deque in its
  propagation loop. This removes repeated tile-pair recomputation without
  changing the compatibility rule.
- A stale WFC pseudo-label test was corrected: one repaired sample in a
  four-sample batch contributes one quarter of the repaired-sample mean,
  preserving proportional full-batch supervision.

Verification:

- Direct behavior probes passed for archive descriptor rejection and legacy
  replacement, checkpoint best-epoch selection, atomic temp cleanup,
  non-finite VAE update rejection, Graphormer bucket wiring, MaskGIT topology
  contract failure, architecture-ablation command wiring, and WFC generation.
- `72` VQ-VAE/Gaussian-VAE/evaluation tests passed.
- `91` vulnerability, topology, and diffusion-conditioning tests passed after
  correcting the stale WFC expectation.
- `14` weighted-WFC and mathematical-rigor tests passed.
- Compile and Ruff checks passed on all files changed in this pass.

Still empirical, not code-complete evidence:

- Real multi-rank dynamic-parameter gradient synchronization must be exercised
  with `torchrun`.
- SPADE/additive, attention-kernel, topology-refinement, LCM-LoRA, WFC-prior,
  P-CBS persona, and 100/500-room controllability arms still require matched
  seeds, trained checkpoints, timing/memory capture, and statistical analysis.
- Human-likeness claims still require calibrated playtest telemetry; P-CBS is
  a behavioral proxy and must not be presented as a validated human model.

### Evidence And Refinement Audit (2026-07-11)

Implemented safeguards from a direct source audit:

- `validate_controllability.py` now requires readable trained VQ-VAE and
  diffusion checkpoints by default. It measures a room-grid semantic proxy and
  refuses to substitute the requested tension curve as an observed result.
  Its optional surrogate path is labeled `surrogate_smoke` and is not evidence.
- The advanced pipeline now treats a Weighted Bayesian WFC best-effort fill as
  a failed refinement. It either aborts under the default strict policy or
  returns the original neural room only when WFC refinement failure is
  explicitly allowed. Both attempted WFC fallbacks and rejected refinements are
  exported in pipeline stats.
- `random_baseline.py` now retains all fixed-budget grammar draws in candidate
  quality and solvability statistics. Archive-only metrics remain conditional
  on valid graph candidates, and the report exposes the random generation
  success rate instead of silently dropping failed draws.
- `validate_all_features.py` is labeled as a synthetic component smoke harness.
  It is not a checkpoint evaluation or a thesis-readiness signal.

Verification in this pass covered the WFC fallback boundary, the
controllability observation contract, and fixed-budget random-baseline
accounting. These checks establish code behavior only; they do not replace
matched-seed checkpoint experiments or human calibration.

### Topology Preservation Decision (2026-07-11)

The flat graph-to-grid layer now exports a topology-invariant realization
report and the advanced pipeline rejects a stitched artifact when its configured
flat-spatial invariant score is below the required threshold. The report compares
only spatial edges that the 2D renderer is responsible for and records edge
recall, connected components, cycle rank, branch-node identity, and articulation
point identity before and after carving. Stairs, warps, cross-floor links,
directionality, and resource gates are deliberately excluded from this projection;
they remain the responsibility of the typed graph and full state-space oracle.

Persistent homology is a future **offline ablation**, not a default training
loss. It can measure multi-scale geometry of an unlocked walkability mask using a
distance-transform filtration and compare H0/H1 persistence summaries before and
after repair. It cannot establish Zelda progression semantics: two maps with equal
Betti numbers can differ in one-way traversal, key consumption, switch order, or
hazard affordances. A publishable PH ablation must therefore compare it against
the exact realization metrics and tile oracle, report compute cost, and show an
incremental gain over those cheaper domain-aligned baselines before it is added to
the training objective.

### Training-Contract Audit (2026-07-11)

- The diffusion configuration now carries all registered puzzle-stage controls
  into `DiffusionTrainingConfig`. When the semantic-loss ablation is enabled,
  the trainer creates the corresponding prediction head, includes it in the
  optimizer, AMP/gradient checks, distributed averaging, and resume checkpoint,
  and computes the loss from denoised predicted room logits rather than ground
  truth grids. The data loader also receives the stage-trace topology controls.
  These fields are no longer inert YAML keys.
- The shared `CheckpointManager` now optionally persists and restores an AMP
  `GradScaler`; the diffusion trainer already did so in its specialized resume
  payload. Resume continuity is therefore available to callers of either
  checkpoint API.
- The LCM-LoRA configuration exposes whether validation uses the EMA target,
  and the registered model-selection choices now include its puzzle-stage
  semantic metric. Its time-zero target now observes the teacher ODE endpoint
  (`x_previous`), which is the consistency identity boundary, rather than the
  inaccessible clean training latent. Gradient-finiteness and clipping cover
  the optional puzzle-stage head as well as LoRA parameters.
- Masked-room normalization is fixed to categorical IDs by contract. It is no
  longer read from an unregistered configuration field that suggested a
  supported but unsafe normalization mode.

Verified false alarms retained as audit outcomes:

- `ExternalValidator.validate()` returns `ValidationResult`, not a dictionary;
  `result.is_solvable` in the robust-pipeline graph validator is therefore the
  correct access path.
- `ModelContextContractError` inherits directly from `Exception` and carries
  `retryable = False`; the robust block executor terminates the attempt instead
  of retrying it under its `ValueError` handling.
- Distributed gradient synchronization first reduces an all-rank gradient
  presence flag and then supplies zero gradients on ranks where a parameter was
  inactive. The dynamic-parameter deadlock report does not apply to the current
  implementation. This still needs a real multi-rank run before it is claimed
  as empirical scalability evidence.

### Phase III Expansion Decision (2026-07-12)

The proposed expansion is useful only after correcting three overstatements.

- Full-state A* is the mechanical oracle, not a human-player model. Removing it
  would make learned-policy failure indistinguishable from true unsolvability.
  Human-like behavior remains a separate P-CBS/RL/human-telemetry question.
  RL playtesting literature likewise treats learned play styles as automated
  testing behavior, not a proof of reachability:
  [Le Pelletier de Woillemont et al., 2022](https://ojs.aaai.org/index.php/AIIDE/article/view/21958).
- The architecture is not wholly 2D. Mission nodes already carry `(row, col,
  floor)`, the grammar has `STAIRS_UP`, `STAIRS_DOWN`, and `EdgeType.STAIRS`,
  and the validator traverses explicit stair graph edges. The actual limitation
  is the final renderer: it exports one flat atlas and records cross-floor links
  as non-spatial rather than producing a floor-indexed artifact.
- The model is not in a complete semantic vacuum. It has explicit style/theme
  IDs and room-role conditioning. It does lack generated lore and dialogue.
  LLM narrative is therefore a separate content modality, not a repair for
  spatial solvability. Existing narrative systems require their own alignment
  and validation layer:
  [Buongiorno et al., 2024](https://ojs.aaai.org/index.php/AIIDE/article/view/31876).

Implemented first expansion:

- `run_pcbs_persona_map_sweep.py --include-rl-ablation` now evaluates optional
  goal, exploration, safety, and combat tabular policies only on maps certified
  by the hard oracle. It exports completion, traversal, combat, pickup,
  confusion, and entropy metrics. These results are never used as hard
  feasibility labels. This follows the broader requirement to evaluate a
  generator across expressive behavior rather than a single cherry-picked
  structural score:
  [Summerville, 2018](https://ojs.aaai.org/index.php/AIIDE/article/view/13012).
- The VGLC validation handoff now preserves `StitchedDungeon.graph`, room/node
  mappings, and puzzle metadata. Earlier grid-only benchmark runners silently
  removed stair/warp transitions: for example, `D1_v1` changed from an oracle
  failure to a 28-transition certified path once the actual graph context was
  supplied. The canonical A* fallback now shares one state budget across A*
  and Dijkstra and reports aggregate work, rather than spending and hiding a
  second full budget.
- The current tabular RL policy has four cardinal actions. Maps whose certified
  oracle path uses stairs or warps are therefore exported as explicit
  `rl_playtester_skips.csv` rows, not counted as policy failures. A graph-action
  RL arm is a future ablation and must define transition actions in its MDP.
- Grid MAP-Elites selection now uses a seeded archive-local RNG and persists
  its RNG state in archive checkpoints. Resumed runs therefore continue the
  same stochastic process instead of depending on unrelated global Python RNG
  calls.

Required verticality ablations, in order:

1. `M0`: current flat atlas with an explicit non-spatial stair ledger.
2. `M1`: floor-partitioned artifact `Dict[floor_id, grid]` with verified paired
   stair anchors and a cross-floor state-space oracle. Reuse the trained 2D room
   generator unchanged.
3. `M2`: shared 2D room model plus explicit floor/layer conditioning and
   floor-level QD descriptors. Compare against `M1` under identical graphs.
4. `M3`: volumetric latent generation only after acquiring or constructing a
   justified multi-floor training corpus. Without that data, a 3D DiT is an
   untrained shape change rather than a scientific upgrade.

Implemented M2 conditioning contract (not yet an empirical result):

- `dataset.node_feature_dim=14` remains the checkpoint-compatible baseline.
- `dataset.node_feature_dim=15` appends normalized mission-node floor/z from
  the authoritative `(row, col, floor)` grammar position. The shared extractor
  is used by dataset loading and runtime generation, and fast-sampler
  distillation now preserves the configured graph dimensions.
- Training fails before optimization when width 15 is selected without at
  least two distinct observed floor labels. This prevents reporting a nominal floor
  ablation on the single-floor VGLC corpus. A valid M2 run therefore requires
  generated or curated multi-floor graphs with authoritative floor labels.

LogicNet resource-semantics audit:

- Global graph supervision now keeps later gates closed during ordered
  key/item acquisition instead of deleting every locked edge at once.
- Small keys are not reused across locks; permanent traversal items can be
  reused; collection gates require all paired token providers before opening.
- Explicitly unsatisfied resource pairs remain unsatisfied. Generic key/lock
  pairing is retained only for legacy graph payloads that omit the pair field,
  not payloads that explicitly provide an empty list.
- These changes improve differentiable supervision but do not replace the hard
  state-space oracle. LogicNet remains a training/guidance surrogate and its
  ON/OFF claim still requires raw pre-repair oracle rates.

Narrative remains an optional downstream ablation. Any future module must emit
schema-validated, cached JSON linked to immutable mission-node IDs; it must not
modify locks, keys, puzzle stages, or solvability after validation. Evaluation
requires blinded human ratings of coherence, controllability, and repetition,
plus a no-narrative control. It is not part of the current mechanical claim.

### Model And Self-Correction Pass (2026-07-12)

Implemented contracts:

- Latent diffusion now has an explicit `latent_scale_factor`. Raw VQ-VAE
  latents are multiplied before diffusion and divided before every production
  VQ decode, including LCM alignment, neural inpainting, autoregressive neighbor
  caching, and preference preparation. The default is `1.0`, so existing
  checkpoints retain their original geometry. New scaled runs must use
  `scripts/calibrate_vqvae_latent_scale.py` over the declared training corpus
  and store the measured reciprocal standard deviation in config. Automatic
  first-batch calibration was rejected because it is order- and rank-dependent.
  This follows the symmetric encode/decode scale contract in the
  [CompVis LDM implementation](https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/diffusion/ddpm.py)
  and remains an ablation, not a presumed universal improvement.
- LogicNet node passability no longer multiplies ordinary probabilities by its
  one-million infinity sentinel. It uses a negative-log traversal barrier
  scaled to the reachability horizon, restoring graph-loss gradients for values
  throughout `(0, 1]`. A direct probe confirmed non-zero graph gradients for
  intermediate room passabilities.
- `logic_resource_gate_mode={hard_ordered,soft_ordered}` is wired through
  diffusion, MaskGIT, checkpoint reconstruction, runtime fallback config, YAML,
  and CLI. `soft_ordered` differentiably composes key reachability and supports
  conjunctive multi-token providers; `hard_ordered` preserves the prior
  discrete rollout as the baseline. This is closer to an explicit planning
  computation such as [Value Iteration Networks](https://papers.nips.cc/paper/2016/hash/c21002f464c5fc5bee3b98ced83963b8-Abstract.html),
  but it is not called Neural A*: that method requires supervised path traces
  and a differentiable search objective
  ([Yonetani et al., 2021](https://proceedings.mlr.press/v139/yonetani21a.html)).
- The final state-space validator now receives the exact stitched graph,
  room/node mapping, room offsets, and puzzle metadata. Its public result
  records final inventory and distinct path interactions, so a real integration
  protocol can verify key and lock usage rather than inferring it from graph
  labels.
- `scripts/run_master_pipeline_integration.py` is the strict artifact protocol:
  it builds exactly 20 mission nodes and three lock/key stages, requires real
  VQ-VAE/diffusion/condition/LogicNet checkpoints, probes condition-encoder
  gradients, runs generation/stitching/repair/the stateful oracle, and requires
  a non-empty solution path with exactly three key pickups and three lock
  traversals. It intentionally has no random-weight success mode.

Self-correction and playtesting:

- MAP-Elites can retain a bounded replay buffer of pre-repair neural rooms.
  Preferences are formed only between different samples for the same immutable
  graph fingerprint and room ID. Solvability is primary and room repair burden
  supplies local credit; path length is not treated as quality. Export and
  preparation scripts bind pairs to checkpoint hashes, reconstruct the true
  condition encoder context, apply the checkpoint latent scale, and refuse
  cross-checkpoint DPO training. This observes the same-condition requirement
  implied by [DPO](https://arxiv.org/abs/2305.18290) and
  [Diffusion-DPO](https://arxiv.org/abs/2311.12908); arbitrary elites from
  different mission graphs are not valid preference pairs.
- `HeadlessZeldaPersonaEnv` exposes the canonical stateful tile mechanics via
  Gymnasium's reset/step contract with deterministic cardinal and graph
  transition actions. A measured 100,000-step CPU run completed in 1.62 seconds
  (about 61.8k steps/s, excluding interpreter import). The optional SB3 PPO
  runner is an executable baseline. SB3 is preferred here for the single-node
  baseline and evaluation utilities; CleanRL remains useful for transparent
  single-file algorithm audits, while RLlib's distributed services are
  unnecessary until environment throughput becomes the bottleneck. RL policy
  completion is a behavioral metric and never replaces the hard oracle.

Neural floor conditioning decision:

- The 15th node feature remains implemented but disabled in canonical config.
  Training with it now requires preserved raw floor metadata for every
  conditioned node and at least two floors within the same dungeon. Merely
  mixing separate single-floor dungeons at different elevations is rejected,
  as is treating a missing label as floor zero. Offline loaders and runtime
  graph contexts share the same normalization constant and provenance fields.
  The single-floor VGLC corpus therefore cannot support a meaningful learned-
  floor claim. The next valid experiment is a matched M1/M2 comparison on
  curated or generated multi-floor graphs, not enabling a zero-valued feature
  on current data.

LogicNet follow-up audit:

- Complete Bellman coverage already scales to `N-1` graph relaxations and
  `H*W` room-grid relaxations, with checkpointed training relaxation for larger
  problems. Replacing it with Neural A* would be scientifically mismatched
  without expert path-trace supervision. Recent long-horizon VIN work instead
  reinforces planning-depth and gradient-transport ablations as the relevant
  comparison ([Wang et al., ICML 2025](https://proceedings.mlr.press/v267/wang25do.html)).
- The checkpointed `current_temperature` buffer previously disagreed with the
  actual Python temperature attributes: a new model reported 1.0 while its
  solvers ran at 0.1, and loading the buffer did not restore those attributes.
  Initialization, checkpoint post-load synchronization, diffusion timestep
  guidance, and MaskGIT optimizer-step annealing now use one schedule.
  `logic_initial_temperature` and `logic_final_temperature` are explicit YAML,
  CLI, checkpoint, and runtime reconstruction parameters.
- Multi-resource supervision no longer invents a provider/lock assignment by
  zipping independently sorted lists. Only a unique one-provider/one-lock case
  may be inferred; ambiguous assignments are reported, and locked graphs with
  no explicit provider pairs receive a non-zero structural violation. Batched
  logs now retain blocked-stage, unmatched-lock, and open-probability metrics.
- Switch gates are no longer synonymous with block-push puzzles. A switch room
  is represented as `step_on_puzzle` unless an actual block structure is
  observed or explicitly declared; this repaired all-zero puzzle traces caused
  by asking the validator to push a block that did not exist.

Verification completed in this pass:

- 60 focused pipeline/training tests passed before the model changes.
- 113 LogicNet, architecture, and vulnerability tests passed after the
  passability and resource-gate work.
- A subsequent trainer and fast-sampler regression suite passed all 98 tests
  after preserving the baseline `train_step` override contract and making
  latent-scale metadata backward compatible with legacy trainer fixtures.
- The latent/LogicNet/DPO/RL/master scripts compile, and the hard/soft resource
  rollout probe produces finite losses and non-zero passability gradients.
- The expanded floor/LogicNet/puzzle/trainer/loader/fast-sampler suite passed
  268 tests. A direct checkpoint probe also restored a non-default LogicNet
  temperature exactly and rejected ambiguous two-key/one-lock inference with a
  non-zero violation.
- A repository-wide `pytest -q` attempt terminated inside the native PyTorch
  convolution constructor on Windows/Python 3.13 at roughly 8 percent. This
  was not a Python assertion failure, so the focused suites above are the
  reproducible verification evidence for this pass rather than a false claim
  of full-suite completion.

Still requiring real artifacts rather than more code:

- Calibrate the latent scale on the frozen VQ-VAE training split, then train
  identity-scale and calibrated-scale seeds under an otherwise fixed protocol.
- Execute the strict 20-room integration protocol with final checkpoints.
- Generate repeated samples per fixed graph before exporting DPO pairs; a
  single sample per condition cannot form a scientifically valid preference.
- Install the optional Gymnasium/SB3 experiment dependencies and train matched
  persona seeds. Human-likeness still requires human calibration data.

### Topology Contract And Repair Audit (2026-07-13)

Verified corrections:

- `LightweightGCNLayer` and graph degree-feature extraction now flatten a
  batch into disjoint offset index spaces and aggregate with `index_add_`.
  This removes the Python batch loop while preserving the normalized
  `D^-1/2 A_hat D^-1/2` operator used by the GCN baseline
  ([Kipf and Welling, 2017](https://arxiv.org/abs/1609.02907)). Batched tests
  cover different edge sets, invalid padded edges, masked nodes, numerical
  parity with independent dense graphs, and gradient flow.
- The advanced-pipeline neural failure path no longer emits a bordered blank
  room. Its opt-in fallback invokes the canonical boundary, topology, graph
  marker, and puzzle-scaffold contracts. It fails closed if those helpers are
  unavailable. A production-helper probe confirms that a key room with a
  locked outgoing edge retains one key marker and the exact locked-door strip.
- `AddBossGauntlet` treats every incoming boss-door approach as gated when
  selecting a pre-lock Big Key provider. `AddCollectionChallengeRule` is now
  transactional on every failure path and cannot leave token rewards without
  a corresponding multi-lock.
- Symbolic WFC entropy reset cannot modify door tiles, START, TRIFORCE, or
  STAIRS, even when a contradiction mask is dilated across them. Geometry
  repair therefore cannot erase the room-to-room topology contract.

Rejected or narrowed audit claims:

- The reported two-lock deadlock bypass was valid in the local
  `validate_lock_key_ordering` prefilter: checking each lock independently
  could accept two keys hidden behind one another's locks. The validator now
  performs a fixed-point acquisition pass. Every lock starts closed; reachable
  keys open one wave of locks; the process repeats until all locks open or no
  progress is possible. A mutually locked cycle is rejected, while valid
  `key1 -> lock1 -> key2 -> lock2` progression remains accepted. Merely
  excluding every lock globally would reject that valid staged case.
- LogicNet is not truncated at its configured default iteration count when
  full coverage is enabled. It runs at least `N-1` graph relaxations or `H*W`
  grid relaxations and checkpoints larger differentiable rollouts. Exact
  black-box shortest-path differentiation remains a separate ablation, not a
  drop-in replacement
  ([Vlastelica et al., 2020](https://arxiv.org/abs/1912.02175)).
- Puzzle-stage auxiliary loss is computed from generator tile logits and is
  included in the generator objective, so it does backpropagate into the room
  model. Inference does not load the auxiliary classifier; it instead applies
  the canonical deterministic interaction-sequence contract and records valid,
  invalid, and skipped gates. The learned head should only become an inference
  gate as a separately calibrated ablation with saved head weights and a
  declared threshold.
- MaskGIT is graph-conditioned: it supports concat-encoder and cross-decoder
  context fusion, receives the explicit room-topology tensor, fixes topology
  anchors, and applies semantic boundary logits. It does not implement the
  diffusion model's SPADE or linear-attention ablations, and config validation
  correctly refuses to label those unavailable variants as MaskGIT results.
- `src/simulation/map_elites.py` and `src/evaluation/map_elites.py` are not
  duplicate implementations. The first is the runtime grid evaluator/GUI
  adapter; the second owns general elite archives, feature extractors, and CVT
  support, which the runtime adapter imports.

Focused verification for this audit: 29 advanced-rule tests, 31 symbolic
refiner tests, 35 graph-grid attention tests, and 17 advanced-pipeline contract
tests passed (112 total). The repository-wide native PyTorch crash noted above
remains outside this focused evidence boundary.

Deferred by scientific design rather than missing code:

- Global-state metadata currently drives deterministic state-aware room
  modification, but it is not a learned condition token. Adding a neural
  global-state embedding without paired before/after room states would create
  an untrained input branch. Implement it only with a corpus containing the
  same room under multiple authoritative global states and report it as an
  ablation against deterministic state compilation.

Repository modularization boundary:

- Search state and movement semantics have been extracted from the validator
  god module into `src/simulation/state.py`: `Action`, `GameState`, canonical
  movement costs and tile sets, dynamic block/bridge geometry, immutable state
  keys, and dominance pruning now have one owner. `validator.py` re-exports the
  same names so existing callers and old pickle lookup paths remain valid. This
  reduced the validator by roughly 350 lines without changing solver behavior;
  36 state-space, block-push, search-factory, and block-integration tests pass.
- Validation result, solver-option, diagnostics, and batch-result contracts now
  live in `src/simulation/validation_types.py`. `validator.py` re-exports the
  same class objects, preserving imports and serialized class identity while
  removing another dependency-light responsibility from the simulator/search
  implementation.
- The forwarding-only `src/gui/services`, `src/gui/controls`, and
  `src/gui/overlay` packages were retained during migration, then removed once
  production imports and behavior tests used canonical domain paths. The GUI
  module catalog is filesystem-backed so removed paths cannot remain listed.
- `train_diffusion.py` remains oversized. Frozen VQ-VAE checkpoint metadata,
  architecture resolution, and legacy-state compatibility now live in
  `src/training/diffusion_checkpoint_contracts.py`, with compatibility aliases
  retained by the CLI module. The complete diffusion configuration schema,
  resolved-config bridge, and CLI override builder now live in
  `src/training/diffusion_config.py`; `src/train_diffusion.py` re-exports the
  established public names. Its remaining safe extraction order is checkpoint
  I/O, validation/sampling, then the trainer core.
  Moving thousands of lines in one patch would obscure behavioral review and
  break downstream imports. Continue as compatibility-preserving module
  extractions with the public script re-exporting established names; do not use
  file length itself as evidence of duplicated scientific logic.

Quality-Diversity feasibility boundary:

- The runtime MAP-Elites adapter already computed an inventory-aware macro
  path for graph descriptors, but previously fell back to grid descriptors
  when that graph path was infeasible. Hybrid mode now rejects archive
  admission in that case. The explicit `legacy` mode remains the grid-only
  ablation. This prevents a tile-solvable rendering from becoming an elite
  when its intended mission key/item economy is impossible.

### Multi-Floor Conditioning And LogicNet Scaling (2026-07-13)

- Neural floor conditioning was already implemented as an appended normalized
  graph-node coordinate at feature index 14, propagated through the dataset,
  all three trainers, checkpoint architecture inference, and runtime graph
  construction. The branch was nevertheless unreachable through the strict
  config system because the only registered schema required 14 features.
  `zelda_multifloor_v1` now locks the complete 15-feature contract while
  `zelda_v1` remains the checkpoint-compatible default.
- Enabling the multi-floor profile is intentionally fail-closed. Training must
  observe at least one dungeon whose every conditioned node has authoritative
  floor metadata and which spans at least two floor values. Zero padding,
  variation between unrelated single-floor dungeons, and inferred labels do
  not count as evidence for the ablation.
- LogicNet's dense mission-graph Bellman-Ford backend performs `N-1` rounds over
  an `N x N` matrix under full coverage. A new opt-in
  `sparse_bellman_ford` backend performs the same conservative soft relaxation
  over real directed edges, changing the planning work from dense `O(N^3)` to
  `O(N E)` for sparse mission graphs. Dense remains the baseline; the sparse
  backend is wired through diffusion, masked-room training, checkpoint
  reconstruction, CLI/config resolution, and the LogicNet ablation manifest.
- Numerical verification compares the dense and sparse distance fields and
  edge-weight gradients on the same branching graph. Focused LogicNet tests
  pass (115 pre-change and 35 sparse/config tests after integration). A local
  CPU probe measured approximately 0.092 s dense versus 0.046 s sparse at 100
  nodes and 0.262 s sparse at 500 nodes. These timings are engineering probes,
  not evidence claims; matched-device repeated timing, peak memory, holdout
  solvability, and key-lock violations remain required in the ablation.

Research basis:

- [Value Iteration Networks](https://arxiv.org/abs/1602.02867) motivates an
  explicit differentiable planning computation rather than an unconstrained
  proxy.
- [Generalized Value Iteration Networks](https://arxiv.org/abs/1706.02416)
  extends that planning bias to irregular graphs.
- [Graph neural induction of value iteration](https://arxiv.org/abs/2009.12604)
  supports direct algorithmic supervision on graph planning steps.
- [DataSP](https://proceedings.mlr.press/v244/lahoud24a.html) is relevant to a
  future learned contextual-cost ablation, but it is not silently substituted
  for the current single-source resource-aware objective.

### Publication Validator And Pacing Contract (2026-07-13)

- Final generation now emits one staged end-to-end report with four hard
  evidence boundaries: semantic-grid representation, exact resource-aware
  graph progression plus all-room reachability, graph-to-grid connection
  realization, and exact full-state tile solvability. Canonical generation
  uses `generation.end_to_end_validation_mode: reject`; an exhausted exact
  oracle is indeterminate and cannot be relabeled unsolvable or accepted.
- LogicNet remains advisory evidence. Its agreement with the exact oracle is
  reported, but a differentiable probability is never used as a proof of
  mechanical feasibility.
- Grammar edge gates now receive a collective fixed-point progression closure
  after their per-gate schema/provider checks. This catches dependency cycles
  in which a provider is only reachable after assuming a different unresolved
  gate is open. The final exact oracle still owns consumable-resource proofs.
- Exact graph solution paths now produce advisory pacing evidence: normalized
  landmark positions, edge-spacing variation, setup-before-gate-before-climax
  ordering, unsmoothed tension-event positions, rest count, revisit ratio, and
  revisit depth. These fields are not hard rejection thresholds and are not
  human-fun measurements. They are suitable response variables for matched
  controllability experiments and later human calibration.
- A nondifferentiable digital Betti-curve ablation reports raw-neural versus
  final-map connected-component and loop drift across thresholds. It is an
  image-topology descriptor, not a stateful Zelda solvability proof. Persistent
  topology should enter the training objective only as a separately matched
  ablation with measured gradient/runtime cost and final hard-oracle rates.
- The GUI services/controls/overlay deletion proposal was rejected after
  import tracing: those files are exercised compatibility paths. The obsolete
  `src/zelda_data/modules` re-export package was unreferenced and was removed.

Research interpretation follows the evaluation-taxonomy warning that no single
automatic metric establishes generator quality
([Withington et al., 2024](https://arxiv.org/abs/2404.18657)). Constraint-aware
QD is consistent with treating feasibility separately from diversity/local
competition ([Gravina et al., 2019](https://arxiv.org/abs/1907.04053)). The
locked-door literature likewise validates both mission feasibility and spatial
fit rather than weak connectivity alone
([Pereira et al., 2021](https://doi.org/10.1016/j.eswa.2021.115009)).

### Validator Hardening Follow-up (2026-07-13)

Current-source verification rejected several stale audit claims:

- lock-node ordering already uses a least fixed-point acquisition process that
  keeps every unresolved lock closed; globally excluding every lock in one
  pass would incorrectly reject valid staged progression;
- the advanced room fallback is opt-in and topology-preserving rather than an
  empty all-zero room;
- the boss gauntlet excludes every incoming boss-door approach when placing
  the Big Key;
- `LightweightGCNLayer` already batches disjoint graphs with offset indices and
  `index_add_`, without a Python batch loop;
- the forwarding-only GUI compatibility packages had no production consumers;
  their tests were migrated to canonical modules and the packages were removed.

The deeper pass found and corrected real contract gaps:

- `LOCK` and `BOSS_DOOR` nodes without a key identity, or without a correctly
  typed `KEY`/`BIG_KEY` provider, now fail schema validation. A malformed lock
  can no longer become an implicitly open room.
- benchmark, ablation, and master-integration `constraint_valid` results now
  include the exact consumable-resource oracle. The monotone grammar closure
  remains a fast diagnostic prefilter; it is not relabeled as an exact proof
  because it does not spend small keys.
- final evolutionary repairs are transactional. If descriptor-oriented repair
  invalidates the selected feasible phenotype, the repair is rolled back. Node
  cap and connectivity transformations are followed by a fresh exact graph
  oracle and all-room reachability proof on the artifact that is actually
  exported.
- the advanced pipeline now requires exact all-room progression reachability,
  not only START-to-GOAL reachability. Its graph proof, optional finite
  global-state proof, spatial realization, final tile oracle, and optional
  Dijkstra consistency comparison are emitted through the same staged
  end-to-end report as the canonical pipeline.
- an attached global-state contract is revalidated against the current graph;
  stale stored validation payloads are never trusted after graph mutation.
- `GraphGuidedValidator` and `GraphValidationResult` now live in
  `src/simulation/graph_validator.py`; `validator.py` re-exports the identical
  class objects for compatibility. This removes a self-contained graph/room
  validation responsibility from the validator god module without changing
  serialized or import-facing contracts.
- Laplacian graph positional encoding now parses enum-valued and serialized
  edge semantics through the canonical edge-token parser. Locked edges no
  longer become open-weight edges merely because a loader used `EdgeType`
  rather than a lowercase string.
- switch-conditioned puzzle traces now distinguish explicit/observed
  block-on-switch structure from ordinary step-on switches. Ambiguous
  `switch_locked` metadata no longer silently fabricates a block interaction.

Scientific boundary:

- Automatic validation can establish representation integrity, finite-state
  graph progression, graph-to-grid realization, and exact tile-state
  solvability within a declared complete search budget.
- It cannot establish enjoyment, perceived pacing, or human-likeness without
  calibrated player evidence. The 2024 evaluation survey explicitly warns
  against treating one automatic metric as a universal quality measure
  ([Withington et al., 2024](https://arxiv.org/abs/2404.18657)).
- BSP is not a mandatory scientific stage. It is a layout baseline or
  alternative compiler and should be added only as a matched placement
  ablation against the current strict graph-aware stitcher, reporting success
  rate, topology preservation, runtime, and final oracle validity.
- Persistent-homology guidance is still a hypothesis. Current code reports a
  nondifferentiable Betti-curve preservation ablation; topology-aware graph
  diffusion work supports investigating learned guidance, but it does not make
  such a loss a correctness oracle
  ([TAGG, NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/bb88dfbebcb21022d32086bed631bfc5-Abstract-Conference.html)).

Focused evidence after this integration: 33 end-to-end/grammar tests, 16 room
stitching tests, 37 evaluation tests, and 110 configuration/trainer-shape tests
passed. The evolutionary suite passed 44 of 45 checks; one seeded 20-generation
run obtained quality `0.481` against a test-local `0.5` threshold. This is a
quality-convergence result to report and investigate, not grounds to weaken the
new feasibility closure or claim full-suite success.

### Topology Evidence Hardening (2026-07-13)

- Spatial graph preservation now has an exact verdict in addition to its
  descriptive composite score. Node/edge/component counts, cycle rank,
  branch nodes, articulation nodes, and biconnected-component count must all
  agree. A high average can no longer conceal one destroyed invariant in the
  end-to-end hard contract.
- Final artifacts now report graph node/edge/component counts, normalized cycle
  rank, branch/leaf counts, articulation ratio, biconnected-region count,
  START-to-GOAL node connectivity, mandatory articulation checkpoints, and
  their positions on the exact solution path.
- These graph characteristics remain advisory controls. Articulation points
  and low redundancy are not automatically defects in lock-and-key dungeons;
  hard validity continues to come from progression, realization, and tile-state
  oracles. This follows controllable graph-PCG work, which treats requested
  graph properties as explicit controls rather than universal quality labels
  ([G-PCGRL, 2024](https://arxiv.org/abs/2407.10483)).
- Current automated evidence is still not a replacement for player data.
  PCGRL-style metric optimization demonstrates controllability under computable
  objectives, not perceived fun ([PCGRL](https://arxiv.org/abs/2001.09212));
  scale and out-of-distribution generalization require dedicated experiments
  rather than inference from in-distribution validity
  ([PCGRL+, 2024](https://arxiv.org/abs/2408.12525)).

### End-To-End QD Archive Materialization (2026-07-13)

- CVT archive schema version 4 stores the evaluated `MissionGraph` phenotype
  for every admitted elite plus the ordered grammar rule schema and complete
  fitness/feasibility contract. It also stores every mutable grammar/search RNG
  stream required for exact warm-start continuation. A genome alone is not reproducible here:
  grammar rules consume a population-global RNG, so replaying one genome from
  the initial seed can produce a different graph.
- Archive warm starts now fail closed when rule IDs, target curves, descriptor
  targets, node/resource limits, or feasibility settings differ. Stored
  fitness values are not compared under a new objective contract.
- `python main.py topology-materialize-archive` compiles each archived
  phenotype through the final graph export oracle and generates paired-seed
  final maps under `end_to_end_validation_mode=reject`. It reports topology
  coverage separately from any-seed and all-seeds surviving final-map
  coverage, preserves per-cell failures, and labels limited runs incomplete.
- Native archives remain pickle files, so materialization requires explicit
  `--trust-pickle`. Legacy genome-only archives cannot support defensible
  final-map claims and must be regenerated.

### Masked-Room Graph-To-Grid Attention Ablation (2026-07-13)

- The masked-room branch now has an executable, opt-in
  `topology_conditioning_mode=graph_cross_attention`. The canonical
  `additive` topology-map path remains unchanged for checkpoint compatibility.
- The ablation uses each spatial token as a query over node-aligned graph
  context, with a vectorized edge-aware GCN prepass, optional graph positions,
  topological encodings, current-room distance features, and padding masks.
  Its learned sigmoid gate blends only the cross-attention delta into the
  MaskGIT hidden grid.
- Missing or misaligned node context is a non-retryable model contract error.
  The implementation does not silently truncate nodes or degrade to additive
  conditioning while reporting the graph-attention ablation.
- `attention_mode=linear_hedgehog` and the graph attention width/threshold/gate
  controls are valid only for this ablation. Configuration rejects inactive
  non-default controls, and masked-room inference reads them from the
  masked-room checkpoint/fallback contract rather than diffusion defaults.
- Reported complexity now separates transformer attention pairs from the
  additional `H * W * N` graph-to-grid interactions. Quality, exact raw/final
  solvability, topology drift, runtime, peak memory, and fallback rate must be
  compared under paired seeds; the new branch is a hypothesis, not a claimed
  improvement.

This design keeps MaskGIT's parallel masked-token objective
([Chang et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Chang_MaskGIT_Masked_Generative_Image_Transformer_CVPR_2022_paper.html))
while using query-to-context cross-attention in the style of variable-input
latent routing
([Perceiver IO, ICLR 2022](https://arxiv.org/abs/2107.14795)). Graph structure
is encoded before spatial attention, consistent with the local-plus-global
separation studied by
[GraphGPS](https://arxiv.org/abs/2205.12454). These papers motivate the
ablation but do not establish superiority for Zelda room generation.

### Validator Cost Contract And Cohesive Refactors (2026-07-13)

- The publication validator was already wired into canonical, advanced, and
  symbolic generation. The attached proposal's claims that lock closure,
  topology-preserving room fallback, boss-gauntlet exclusion, and batched GCN
  processing were absent are stale against the current source.
- Solver consistency previously compared transition counts even though enemy,
  pickup, door, and puzzle transitions have different costs. `ValidationResult`
  and `SolverDiagnostics` now retain the winning full-state `g` cost; the
  optional A* versus uniform-cost check compares accumulated costs with a tight
  numerical tolerance and reports path lengths only as descriptive evidence.
  Missing costs are indeterminate rather than silently accepted.
- Every successful tile-oracle route is now replayed from a fresh `GameState`
  through the canonical movement and graph-transition rules before it can be
  reported as solved. The replay covers consumable keys and bombs, persistent
  items, opened doors, pushed-block geometry, staged puzzles, stairs, virtual
  nodes, graph warps, one-way constraints, and floor changes. A failed replay
  changes the oracle result to `route_replay_failed`; a solved result without a
  replay certificate is indeterminate in the publication contract.
- Tile replay also recomputes accumulated transition cost from the reconstructed
  route. A legal path whose replayed cost disagrees with the solver's stored
  `g` value is rejected, preventing stale parent/cost bookkeeping from
  contaminating solver-consistency or pacing evidence.
- The replay retains all distinct legal states compatible with each reported
  position. This avoids choosing an arbitrary inventory interpretation when a
  position-only route can represent more than one graph transition. Behavioral
  checks exercise mandatory small-key use, block pushing, cross-floor stairs,
  and rejection of a fabricated route through a wall.
- The resource-state graph oracle now applies the same rule at its own
  representation boundary. `AgentSimulator.replay_path()` is the single graph
  transition replay owner, `PathVerifier` delegates to it, and solved graph
  evidence without `route_replay_status=verified` is indeterminate. This keeps
  mission-graph path reconstruction and final tile-route reconstruction under
  parallel fail-closed contracts.
- Advisory pacing now reports consecutive landmark segments with graph path
  cost, corridor-edge count, and intermediate-room count. It operates on the
  original discrete route before smoothing. No threshold is used as a hard
  definition of fun or quality.
- Safetensors inference sidecars now include the optional puzzle-stage
  semantics head, matching `.pth` checkpoint behavior. Loading a checkpoint
  whose optional LogicNet or puzzle head is disabled in the current ablation
  emits an explicit warning instead of dereferencing a missing module.
- Pygame rendering was extracted from `simulation/validator.py` into
  `simulation/validator_rendering.py`, and diffusion checkpoint I/O was
  extracted into `training/diffusion_checkpoint_io.py`. Public trainer and
  validator methods remain compatibility delegates; search rules and training
  semantics were not moved.
- The forwarding-only `src/gui/services`, `src/gui/controls`, and
  `src/gui/overlay` packages were removed after every repository caller and
  behavior test migrated to its canonical domain module. The two MAP-Elites
  implementations and two LogicNet modules remain because import tracing and
  source inspection show intentionally different APIs; file count alone is
  not evidence of duplication.

Scientific boundary: the staged contract can reject representation errors,
resource-progression failures, graph-to-grid drift, invalid reconstructed
routes, incomplete exact search, and solver-cost disagreement. Route replay is
an executable certificate against the same canonical transition model; it is
not an independent implementation of the game rules. The contract still
cannot prove player enjoyment or human pacing, and automatic validation does
not remove the need for a human study when making experience claims. This
separation follows the evaluation taxonomy of
[Withington et al. (FDG 2024)](https://arxiv.org/abs/2404.18657) and the
feasibility/quality distinction used in constrained PCG search
([Gallotta et al., 2022](https://arxiv.org/abs/2205.05834)). BSP therefore
remains a matched layout baseline, not a mandatory replacement for the current
graph-aware stitcher.

The attached hardening proposal's four immediate source claims are stale in
the current revision. `MissionGrammar.validate_all_constraints()` runs the
consumable-state `validate_exact_progression()` oracle after its diagnostic
prefilters, so cyclic key dependencies are not certified by excluding one lock
at a time. `AddBossGauntlet` blocks every approach edge while placing the big
key. The advanced pipeline's opt-in fallback reconstructs the canonical
boundary, graph markers, and puzzle scaffold, then reapplies boundary authority.
`LightweightGCNLayer` flattens `[B, N]` indices and uses batched `index_add_`
without a Python batch loop. Replacing these with the proposal's blanket
"exclude all locks" check would reject valid sequential key progression and is
therefore not adopted.

Verification for this pass: 253 checks passed across tile mechanics, stitching,
the end-to-end validator, rendering, diffusion conditioning/checkpoint I/O,
masked-room architecture/configuration/ablation manifests, topology generation,
the neural pipeline, and search-factory behavior. The masked graph-to-grid
softmax and linear branches passed forward, backward, fail-closed, and
complexity checks; `compileall` passed for `src/` and `tests/`.

Route-certificate extension verification: 268 non-overlapping checks passed
across full-state tile search, graph evaluation, grammar integration, topology
reproducibility, advanced and canonical pipelines, batched GCN behavior, and
fallback contracts. `compileall` and `git diff --check` also passed. Dependency
warnings from PyG and `requests` remain environment warnings, not suppressed
test failures.

### Fail-Closed Grammar Generation And Canonical GUI Imports (2026-07-13)

- A direct multi-seed generation probe found complete-contract acceptance of
  `9/10` at 8 rooms, `6/10` at 12 rooms, and `2/10` at 20 rooms. Consequently,
  one grammar attempt cannot be described as guaranteed-valid output.
- `MissionGrammar.generate()` remains the raw one-attempt operator needed to
  report failure rates and expressive range. It now marks output as
  `certified`, `invalid_candidate`, or `partial_lock_check_only` in generation
  metadata, so downstream analyses cannot confuse an attempted graph with a
  validated artifact.
- `MissionGrammar.generate_validated()` is the production contract. It retries
  independent deterministic seed streams, reruns the complete diagnostic plus
  consumable-state oracle, records the accepted seed/attempt count, and raises
  after exhaustion. GUI generation and non-evolution ablation/export paths use
  this fail-closed API.
- The proposed blanket test that removes every lock simultaneously was not
  adopted. The current lock-node validator keeps all unresolved locks closed,
  collects reachable keys, and unlocks matching locks in waves. This rejects
  mutual key-lock cycles while accepting valid nested progression. The exact
  state-space oracle remains the final authority for consumed small keys.
- Fifty-six forwarding-only GUI files were deleted after test imports migrated
  to canonical packages. The module catalog now discovers real modules from
  the filesystem instead of retaining a hand-maintained list of deleted paths.
- The migrated GUI behavior suites passed in two batches (`80 + 66` tests).
  Their execution exposed a real cross-context font-cache defect: font objects
  were keyed only by style and could leak between Pygame-compatible runtimes.
  Cache keys now include the owning font subsystem identity.
- The proposed Zelda-data deletion list was re-audited rather than applied
  mechanically. `src/zelda_data/modules/` is already empty, while
  `adapter_io.py`, `conversion.py`, and `visual_extractor.py` have active
  production importers and unique implementations, so deleting them would be
  data-path regression rather than cleanup.
- The large-file refactor is already staged by ownership: validator state,
  result types, helper metrics, and rendering live in separate modules;
  diffusion configuration, checkpoint contracts, and checkpoint I/O are also
  separate. Remaining solver/trainer extraction must preserve public APIs and
  is not justified by line-count targets alone.

Scientific boundary: exact validators establish feasibility under the encoded
mechanics, not fun. Pacing and topology statistics remain advisory objectives,
and persistent-homology losses remain ablations until paired experiments show
improved graph-to-grid fidelity without reducing valid diversity. This follows
the evaluation-taxonomy caution in
[Withington et al. (2024)](https://arxiv.org/abs/2404.18657), the explicit
designer-constraint framing of
[Linden et al. (2013)](https://ojs.aaai.org/index.php/AIIDE/article/view/12592),
and the use of topology layers as learnable priors rather than correctness
proofs in
[Gabrielsson et al. (2020)](https://proceedings.mlr.press/v108/gabrielsson20a.html).

Verification for this pass used executable contracts rather than source-only
claims: 85 graph-to-tile/advanced-pipeline tests and 207
puzzle-semantics/MaskGIT/neural-pipeline/config/ablation tests passed. A
five-seed 20-room generation probe also produced certified graphs in 1-5
attempts, and every accepted graph passed the complete validation contract.
