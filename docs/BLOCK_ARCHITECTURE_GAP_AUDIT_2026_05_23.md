# Block Architecture Gap Audit

Last updated: 2026-05-23

This audit starts from the basic contract of a graph-first neural-symbolic PCG
system:

1. parse trustworthy data
2. generate controllable mission structure
3. learn compact room representations
4. condition room generation on graph semantics
5. generate rooms
6. enforce symbolic/gameplay constraints
7. evaluate mechanics, distribution, controllability, compute, and human
   experience

The repo is architecturally complete enough for thesis work. The remaining
gaps are mostly evidence, exact controls, and latest-code reruns.

## Research Anchors

- PCG Benchmark: quality, diversity, and controllability should be reported as
  separate evidence dimensions: <https://arxiv.org/abs/2503.21474>
- Controllable PCG: target/constraint satisfaction must be measured directly,
  not only described as an intent: <https://www.ijcai.org/Abstract/16/116>
- PCGML: small data, representation choice, parameter tuning, and limited data
  are known open problems: <https://arxiv.org/abs/1702.00539>
- VGLC: parseable level corpora support reproducible PCGML and game-AI
  experiments: <https://arxiv.org/abs/1606.07487>
- PCG evaluation survey: evaluation practice is inconsistent, so code/method
  reuse and explicit methodology matter: <https://arxiv.org/abs/2404.18657>
- Pereira dissertation/article: designer target matching plus user validation is
  the closest domain-specific standard for Zelda-like locked-door dungeons:
  <https://repositorio.usp.br/item/002964434>,
  <https://repositorio.usp.br/item/003032388>

## Block 0: Data Adapter

Current code:

- `src/data_processing/data_adapter.py`
- `src/zelda_data/zelda_core.py`
- `src/zelda_data/vglc_utils.py`
- `tests/test_vglc_compliance.py`

What exists:

- VGLC text/DOT parsing
- graph-room alignment helpers
- schema constants and validation tests
- Block-0 audit script

Missing or weak:

- final thesis table proving exact dataset version, room count, graph count,
  split policy, and excluded/virtual-node handling
- explicit data-card style provenance for all Zelda assets and graph files

Needed experiment/artifact:

- rerun `scripts/audit_block0_data.py` and cite the resulting JSON/MD in the
  final report.

## Block I: Mission/Topology Generator

Current code:

- `src/generation/evolutionary_director.py`
- `src/generation/grammar.py`
- `src/evaluation/benchmark_suite.py`
- `scripts/run_designer_controllability_proof.py`

What exists:

- evolutionary grammar search
- optional CVT-emitter QD search
- descriptor targets for topology realism
- VGLC reference alignment
- PCG Benchmark alignment script

Missing or weak:

- execution of the new exact designer-controllability proof
- final analysis of the monotonic target-response table for linearity, size,
  and gate pressure
- actual compute run of the 100/500-room stress rows

Needed experiment/artifact:

- run `scripts/run_designer_controllability_proof.py --execute`.
- report both successes and failures. Raw key/lock counts are now direct search
  targets; high residual error would mean the search space cannot reliably
  satisfy those controls under the chosen budget.

## Block II: VQ-VAE Room Representation

Current code:

- `src/core/vqvae.py`
- `src/train_vqvae.py`
- `tests/test_hmolqd/test_vqvae.py`

What exists:

- VQ-VAE tokenizer with codebook sweeps
- VQ-VAE-2 hierarchical tokenizer ablation path
- CoordConv and local-structure prior ablations
- held-out validation support in current training commands

Missing or weak:

- consolidated sample-efficiency table across tokenizer branches
- final codebook-utilization table aligning top/bottom utilization with
  reconstruction and downstream success
- final compute/runtime table after rerunning all tokenizer branches

Needed experiment/artifact:

- run `scripts/consolidate_compute_sample_efficiency.py`.
- ensure each VQ-VAE run exports runtime, epoch-to-best, val loss, codebook
  utilization, and checkpoint size.

## Block III: Graph Conditioning

Current code:

- `src/core/condition_encoder.py`
- `src/pipeline/room_topology_conditioning.py`
- `src/core/puzzle_stage_semantics.py`

What exists:

- node/edge graph conditioning
- puzzle subtype channels
- ordered stage-condition metadata
- tests for conditioning shapes and topology helpers

Missing or weak:

- executed ablation isolating graph conditioning from symbolic repair
- target-response proof that changing graph semantics changes generated room
  semantics before repair

Needed experiment/artifact:

- run `scripts/run_conditioning_logicnet_repair_ablation.py --execute`.
- compare full conditioning vs no graph tokens vs no stage tokens with repair
  disabled and enabled. Report both pre-repair and post-repair semantics.

## Block IV: Room Generator

Current code:

- `src/core/latent_diffusion.py`
- `src/core/discrete_masked_model.py`
- `src/train_diffusion.py`
- `src/train_masked_room.py`

What exists:

- latent diffusion branch
- masked-room branch
- fast-sampler branch
- downstream branch command book

Missing or weak:

- final latest-code branch reruns after validation changes
- one table tying branch quality to compute/sample efficiency
- human-facing room naturalness validation

Needed experiment/artifact:

- refresh branch runs, then consolidate with
  `scripts/consolidate_compute_sample_efficiency.py`.
- add human/expert room sheet from final generated rooms.

## Block V: LogicNet / Differentiable Guidance

Current code:

- `src/core/logic_net.py`
- `src/ml/logic_net.py`
- `scripts/compare_logic_loss_ab.py`
- LogicNet audit docs

What exists:

- logic/guidance modules and tests
- ablation scripts and historical audit notes

Missing or weak:

- final clean ON/OFF table on current branch with identical seeds and fixed
  downstream checkpoints
- calibration showing the neural guide improves pre-repair validity, not only
  post-repair output

Needed experiment/artifact:

- run `scripts/run_conditioning_logicnet_repair_ablation.py --execute` and use
  the paired LogicNet ON/OFF rows with pre-repair and post-repair metrics
  separated.

## Block VI: Symbolic Repair, Overlay, Puzzle Grammar, Stitching

Current code:

- `src/core/symbolic_refiner.py`
- `src/generation/wfc_refiner.py`
- `src/generation/weighted_bayesian_wfc.py`
- `src/pipeline/dungeon_pipeline.py`
- `src/pipeline/room_stitching.py`

What exists:

- symbolic repair
- deterministic graph marker overlay
- stateful puzzle scaffolding
- WFC diagnostics and stress probes
- end-to-end structural metrics

Missing or weak:

- final separation between "model generated this" and "repair made this valid"
- human-facing evidence that repaired rooms look intentional
- compute cost of repair as a share of total generation time

Needed experiment/artifact:

- run repair-off, overlay-off, WFC-off, and full variants under identical seeds.
- include repair count, repair time, pre/post validity, and visual sheet.
- for the conditioning/LogicNet subset, the new paired ablation script already
  emits repair count, repair time, pre/post validity, and `visual_sheet.png`.

## Block VII: Evaluation, QD, Playability, Human Evidence

Current code:

- `src/evaluation/*`
- `src/simulation/*`
- `scripts/run_ablation_study.py`
- `scripts/run_pcg_benchmark_alignment.py`
- `scripts/run_ood_scaling_and_blinded_eval.py`
- `src/utils/playtest_telemetry.py`

What exists:

- fixed-seed ablations
- PCG Benchmark alignment
- OOD/blinded packet builder
- P-CBS behavioral validator
- telemetry collector

Missing or weak:

- completed latest-branch human study
- P-CBS calibration against human traces
- consolidated compute/sample-efficiency table
- final external baseline table

Needed experiment/artifact:

- run the blinded packet workflow and collect ratings.
- use telemetry to calibrate P-CBS personas.
- run `scripts/consolidate_compute_sample_efficiency.py`.

## Final Missing List

Highest priority:

1. execute designer controllability proof and inspect exact key/lock drift
2. run conditioning/LogicNet/repair matrix with visual sheet
3. latest-code fixed-graph and P-CBS reruns
4. compute/sample-efficiency consolidation
5. human/blinded evaluation packet completion
6. P-CBS calibration against human traces

Architectural change probably needed only if experiments fail:

- room semantics may need stronger pre-repair supervision if ablations show the
  repair/overlay layer is doing most of the semantic work.
