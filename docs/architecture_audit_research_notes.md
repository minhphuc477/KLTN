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
