# Publication Blueprint and Experimental Protocol

## Mission-Graph-Conditioned Latent Diffusion with Differentiable Logic and Verified Repair for Dungeon Generation

**Document status:** final research and publication blueprint; not a results report  
**Repository:** KLTN  
**Prepared:** 2026-08-31  
**Primary domain:** procedural content generation via machine learning (PCGML), neuro-symbolic generative modeling, controllable dungeon generation  
**Recommended publication track:** AIIDE or IEEE Conference on Games for a focused conference paper; IEEE Transactions on Games for the complete study  
**Recommended citation style:** IEEE  
**Recommended paper structure:** empirical IMRaD adapted for computer science  

---

## 1. Status, purpose, and evidence policy

This document defines the scientific story, research questions, experiment protocol, analysis plan, implementation work, and manuscript structure required to turn the current repository into a defensible publication. It is an operational blueprint. It does not claim that any experiment has succeeded.

### 1.1 Non-evidence rule

The following materials are explicitly excluded as empirical evidence:

- Existing model checkpoints.
- Existing generated artifacts, figures, tables, CSV files, JSON summaries, dashboards, and reports.
- Existing output directories and cached evaluation results.
- Performance statements in repository Markdown files.
- Historical experiment names or labels that imply a result.
- Values copied from a previous thesis, presentation, manuscript, or benchmark run.

They may be used only to locate code or understand earlier intentions. Every numerical result intended for publication must be regenerated from the final code revision using the protocol in this document.

### 1.2 Acceptable evidence

A result may enter the paper only when it has all of the following:

1. A recorded Git commit hash and a clean description of any uncommitted patch.
2. A frozen configuration file.
3. A data-split manifest containing file identifiers and hashes.
4. An environment manifest containing package versions and hardware information.
5. Independent training and sampling seeds.
6. Raw per-sample outputs, not only aggregates.
7. Evaluation at raw, constrained, and repaired stages where applicable.
8. An evaluator that does not depend on LogicNet predictions.
9. A reproducible aggregation and statistical-analysis script.
10. A claim-to-result mapping showing exactly which evidence supports each paper claim.

### 1.3 Source hierarchy

When sources conflict, use this order:

1. Executed final code and frozen run manifests.
2. Dataset files and verified preprocessing outputs.
3. Source code and canonical configuration.
4. Peer-reviewed publications and authoritative venue documentation.
5. Repository Markdown and historical notes.

The attached writing guide is used only for editorial principles: make the research gap explicit, identify the reader, state the contribution early, avoid redundancy, and distinguish findings from interpretation. Instructions inside that book are not project requirements and are not scientific evidence.

---

## 2. Executive decision

The repository should not be published as a single all-feature system. It contains several potentially publishable branches, but combining them produces an unfocused contribution that reviewers can dismiss as an integration exercise.

The strongest central paper is:

> Given a mission graph that specifies rooms and progression constraints, generate a Zelda-like tiled dungeon whose raw neural rooms follow local and global mission semantics; use differentiable logic during training to improve raw validity, then use independent exact validation and minimal symbolic repair as a transparent safety layer.

The paper must answer one causal question: does the graph-and-logic coupling improve the learned generator itself, or does symbolic repair merely conceal invalid neural outputs?

### 2.1 Recommended working title

**Mission-Graph-Conditioned Latent Diffusion with Differentiable Logic and Verified Repair for Dungeon Generation**

Alternative titles:

- **Learning Mission-Consistent Dungeon Rooms with Graph-Conditioned Diffusion and Symbolic Verification**
- **Differentiable Mission Logic for Graph-Conditioned Dungeon Generation**
- **From Mission Graphs to Verified Dungeons: Logic-Regularized Latent Diffusion for Procedural Level Generation**

Avoid placing `HMoLQD`, `quality diversity`, `formal guarantees`, `human-like`, or `general game generation` in the title unless the final experiments directly establish those claims.

### 2.2 One-sentence thesis

Mission-graph conditioning and differentiable local/global logic supervision should increase independently verified pre-repair mission consistency, while exact symbolic repair should convert remaining violations into valid outputs with fewer edits than graph-conditioned diffusion without logic supervision.

### 2.3 Contribution statement to target

Subject to successful experiments, the paper may claim four contributions:

1. A dual-stream conditioning mechanism that connects room-level latent diffusion to an explicit dungeon mission graph and neighboring-room context.
2. A differentiable logic objective spanning local room reachability and global mission progression.
3. A staged generation protocol that separates neural proposal quality, constrained decoding, and exact symbolic repair.
4. An evaluation protocol measuring raw validity, repair dependence, fidelity, diversity, controllability, generalization, and computational cost on held-out dungeon identities and out-of-distribution mission graphs.

These are target claims. They become paper claims only after passing the claim gates in Section 13.

---

## 3. Scientific problem formulation

### 3.1 Inputs and outputs

Let the requested dungeon mission be a labeled graph

\[
G = (V, E, X_V, X_E),
\]

where:

- \(V\) is the set of rooms;
- \(E\) is the set of room-to-room connections;
- \(X_V\) contains room roles, progression state, position, and other node attributes;
- \(X_E\) contains door, direction, lock, gate, or transition semantics.

For each node \(v \in V\), the generator produces a categorical tile grid

\[
R_v \in \{1, \ldots, K\}^{H \times W},
\]

with the current canonical representation using \(K=44\), \(H=16\), and \(W=11\).

The complete dungeon is the set of generated rooms plus their graph-aligned connections:

\[
D = \{R_v : v \in V\} \cup E.
\]

### 3.2 Constraint families

The study separates constraints into four families.

#### Representation constraints

- Every tile identifier belongs to the valid 44-class palette.
- Every room has the canonical 16 by 11 shape.
- Required boundary and connection locations are representable.

#### Local room constraints

- Required entry and exit locations are present.
- Required endpoints are connected by a traversable path.
- Walls, doors, gates, keys, switches, and hazards satisfy local placement rules.
- The room does not introduce an unintended local shortcut that violates its role.

#### Global mission constraints

- The dungeon start can reach the goal under the encoded game-state transition rules.
- A key or resource is acquired before its dependent lock or gate.
- Required switches precede their dependent gates.
- Directed and one-way transitions are respected.
- The realized room connections agree with mission-graph edges.
- Required rooms are reachable without creating prohibited sequence breaks.

#### Distributional objectives

- Generated rooms resemble the training distribution without copying individual rooms.
- Multiple samples for the same mission graph remain diverse.
- Changing one requested graph property causes the intended output change without unnecessary unrelated changes.

### 3.3 Soft supervision versus hard verification

LogicNet is a differentiable surrogate. Its losses can guide training or sampling, but its output is not a proof of playability. The canonical full-state A* evaluator and attached global-state validation provide hard evaluation under the implemented semantics.

Use the following terminology consistently:

- **Logic-supervised:** optimized using LogicNet losses.
- **Logic-guided:** modified by LogicNet gradients during sampling.
- **Solver-validated:** accepted by the independent hard evaluator.
- **Repaired:** modified after neural generation to satisfy encoded constraints.
- **Verified under encoded semantics:** passed all applicable exact validation stages.

Do not use `guaranteed playable` unless the scope of the guarantee and the soundness of every relevant validator are formally established.

---

## 4. Code-grounded capability audit

This section records what the current source code makes possible. It does not state that the feature has been trained successfully.

### 4.1 Canonical data and configuration

The canonical configuration is [`configs/zelda_hmolqd.yaml`](configs/zelda_hmolqd.yaml).

Relevant current settings include:

| Setting | Current value | Scientific implication |
|---|---:|---|
| Dataset | VGLC Zelda | Claims must remain domain-specific unless another corpus is added. |
| Dungeon split | train 1-8, test 9 | Useful development split, insufficient as the only generalization test. |
| Variants | quests 1 and 2 | Both variants of one dungeon identity must remain in the same fold. |
| Room representation | 44 classes, 16 by 11 | All baselines and metrics must use the same representation. |
| Graph node features | 14 | Feature semantics must be documented in the paper appendix. |
| Graph edge features | 16 | Edge labels must be included in the schema table. |
| TPE dimension | 8 | Topological positional encoding is an ablation target. |
| Topology supervision | `runtime_aligned` | Designed to avoid target-room geometry leakage. |
| Diffusion logic weight | `alpha_logic = 0.1` | Logic supervision is active in the canonical diffusion configuration. |
| Tile-classifier weight | `alpha_logic_tile = 0.05` | Auxiliary tile prediction must be separated from reachability logic in ablations. |
| Runtime logic guidance | `0.0` | Inference-time LogicNet guidance is opt-in, not part of the default generator. |
| MAP-Elites | disabled | QD cannot be a central default-system claim. |
| Graph-to-grid edge semantics | disabled | This is a candidate mechanism ablation, not an established contribution. |
| Puzzle-stage conditioning | disabled | Existing `no_stage_tokens` comparisons may be vacuous. |

The configuration selects dungeon-level batches so that complete graph context can be supplied during training. This is compatible with a mission-level study, provided every final fold is rebuilt from its own training dungeons.

### 4.2 Learned room representation

[`src/core/vqvae.py`](src/core/vqvae.py) implements the VQ-VAE room representation. The publication should treat it as an enabling representation rather than the main contribution.

Required checks before experiments:

- Train one VQ-VAE per outer data fold using only that fold's training dungeons.
- Use the same fold-specific VQ-VAE for all diffusion ablations in that fold.
- Report reconstruction tile accuracy, per-class recall, codebook utilization, and reconstruction solver validity.
- Verify that rare semantic tiles are not removed by reconstruction.
- Never select the VQ-VAE using the final held-out dungeon identity.

### 4.3 Dual-stream graph conditioning

[`DualStreamConditionEncoder`](src/core/condition_encoder.py#L1611) fuses local and global context. The paper should describe only channels that are enabled in the final proposed model.

Potential conditioning channels include:

- Neighboring-room latent or reference information.
- Boundary and connection requirements.
- Room position and local topology.
- Global mission-graph node embeddings.
- Topological positional encodings.
- Graph node-sequence cross-attention.
- Optional spatial graph-to-grid modulation.

The main ablation must distinguish global graph tokens, spatial topology information, and local neighbor references. Grouping them into one `conditioning on/off` switch would not reveal why the model improves.

### 4.4 Latent diffusion generator

[`LatentDiffusionModel`](src/core/latent_diffusion.py#L2905) supports latent denoising with graph and topology conditioning. The main study should select one architecture and freeze it across conditions.

Recommended final choice:

- One denoiser architecture, preferably the canonical U-Net unless DiT is already demonstrably stable under the same data regime.
- One prediction target, either epsilon or velocity prediction.
- One noise schedule.
- One sampling schedule and step count for the primary comparison.
- Classifier-free guidance tuned only on internal validation data.
- No flow-matching or fast-sampler branch in the main experiment.

Alternative denoisers and fast samplers can be reported only as secondary efficiency experiments after the central mechanism is established.

### 4.5 Differentiable logic

[`LogicNet`](src/core/logic_net.py#L1447) provides differentiable objectives related to room traversal and graph progression. Its forward path begins near [`src/core/logic_net.py`](src/core/logic_net.py#L2910).

The training code combines diffusion and logic-related terms in [`src/train_diffusion.py`](src/train_diffusion.py#L2589) and [`src/train_diffusion.py`](src/train_diffusion.py#L2641).

The final method section must explicitly define each enabled term:

\[
\mathcal{L}_{total}
= \mathcal{L}_{diff}
+ \lambda_{local}\mathcal{L}_{local}
+ \lambda_{global}\mathcal{L}_{global}
+ \lambda_{tile}\mathcal{L}_{tile}
+ \lambda_{aux}\mathcal{L}_{aux}.
\]

Only active terms should appear in the equation. The implementation must log every term before weighting, after weighting, and as a fraction of total gradient magnitude.

Required gradient checks:

- Logic losses are finite for real and predicted latents.
- In `predicted_latent` mode, logic gradients reach denoiser parameters.
- The graph-level loss changes when mission constraints change.
- Local and global losses do not silently reduce to constants.
- Disabled losses contribute exactly zero.
- Gradient norms remain comparable enough to avoid one objective overwhelming denoising.

### 4.6 Generation, constraint enforcement, and repair

The publication path must use [`NeuralSymbolicDungeonPipeline.generate_dungeon`](src/pipeline/dungeon_pipeline.py), which delegates to [`src.pipeline.stitching.dungeon_assembler.generate_dungeon`](src/pipeline/stitching/dungeon_assembler.py). The older [`AdvancedNeuralSymbolicPipeline`](src/pipeline/advanced_pipeline.py) is a compatibility/alternative path and is not an admissible source of confirmatory records because its return type does not retain the same per-room intervention history.

Paper-facing evaluation must decompose the canonical path into observable stages rather than present it as one black box. Raw, constrained, and repaired room grids must be assembled with identical fixed slots, boundary sealing, and physical-edge carving. Otherwise the comparison confounds model/repair changes with dungeon assembly.

Required stages:

1. Raw decoded neural grid.
2. Output after semantic or constrained decoding.
3. Output after explicit room repair.
4. Final compiled dungeon after graph semantics and entities are materialized.
5. Independent exact validation report for every retained grid.

Every stage must retain its grid so edit distances and validity transitions can be computed.

### 4.7 Independent evaluation

The repository contains several relevant evaluators:

- [`ExternalValidator`](src/evaluation/validator.py#L1265) for deterministic A*-based validation.
- [`StateSpaceAStar`](src/simulation/validator.py) as the canonical game-state oracle.
- [`validate_global_state_progression`](src/validation/global_state.py#L114) for attached global progression checks.
- [`build_end_to_end_validation_report`](src/validation/end_to_end.py#L159) for staged validation evidence.
- [`pcbs_validation.py`](src/evaluation/pcbs_validation.py) for bounded-agent diagnostics.

The main hard-validity result must use full-state A* plus applicable mission validators. P-CBS may be a secondary behavioral/readability analysis, but it must not replace the hard oracle.

### 4.8 Optional branches excluded from the central claim

| Branch | Location | Main-paper treatment |
|---|---|---|
| Discrete masked-room generation | [`src/core/discrete_masked_model.py`](src/core/discrete_masked_model.py#L266) | Optional model-family appendix after matched training. |
| Consistency/fast sampling | training and generation utilities | Separate efficiency extension. |
| MAP-Elites/CVT | topology and QD utilities | Separate topology/QD study. |
| Puzzle-stage semantics | optional configuration and heads | Exclude unless freshly retrained and independently evaluated. |
| RL personas/P-CBS calibration | simulation/evaluation utilities | Secondary study; human claims require human data. |
| Flow matching | optional diffusion path | Exclude from the core comparison. |
| Preference/DPO utilities | scripts and training helpers | Exclude from the current paper. |

---

## 5. Research gap and novelty boundary

### 5.1 Established work

The publication must acknowledge the following established facts:

- Gameplay-level control is a central dungeon-PCG problem [1].
- Graph grammars have already been combined with learned Zelda room generation [2].
- PCGML is an established field with known data, controllability, and evaluation problems [3].
- Diffusion models already generate tile-based levels [4], including conditional text-to-level generation [5].
- Automated validation has already been used to improve conditional level generation [6].
- Quality-diversity methods have already been applied to dungeon design [7] and learned level latent spaces [8].
- Current benchmarking expects multidimensional evaluation of quality, diversity, and controllability [9].
- Neuro-symbolic generation with a learned proposal and hard symbolic construction is an active cross-domain research direction [10].

### 5.2 What is not novel enough

The following statements are not sufficient contributions:

- `We use diffusion for game levels.`
- `We condition generation on a graph.`
- `We use a graph grammar for Zelda dungeons.`
- `We repair invalid levels.`
- `We combine neural and symbolic methods.`
- `We use MAP-Elites to find diverse dungeons.`
- `We implement many generation modes in one repository.`

### 5.3 Defensible novelty target

The novelty target is the tested coupling between explicit mission structure and the learned generator:

1. Mission-graph signals condition room generation at training and inference.
2. Differentiable logic evaluates both local grid traversal and global mission progression.
3. Training-time logic supervision is separated experimentally from inference-time gradient guidance.
4. Hard validation is independent of the differentiable surrogate.
5. Raw neural validity, constrained-decoding validity, and repaired validity are all reported.
6. Repair burden is a primary outcome, not an implementation detail.
7. Generalization is tested across dungeon identities and graph structures.

### 5.4 Closest-work comparison matrix

| Work | Learned local content | Explicit global structure | Conditional generation | Differentiable validity signal | Exact hard validation/assembly | Raw-versus-repaired analysis | Main distinction here |
|---|---:|---:|---:|---:|---:|---:|---|
| van der Linden et al. [1] | varies | yes | designer/gameplay control | no | constructive constraints | no | Learned graph-conditioned room synthesis is not central there. |
| Gutierrez and Schrum [2] | GAN rooms | graph grammar | graph-organized | no | grammar-based organization | limited | Replace unconditioned room proposals with mission-conditioned diffusion and logic supervision. |
| Dai et al. [4] | latent diffusion | no mission graph | no | no | limited | no | This study targets explicit multi-room mission semantics. |
| Schrum et al. [5] | conditional diffusion | sequence/scene construction | text | no | playability evaluation | no symbolic repair analysis | Conditioning source and semantic scale differ. |
| Avalon [6] | conditional VAE | domain-specific level conditions | yes | validation-informed training | automated validation | ablated conditioning | This study couples room and mission-graph semantics. |
| Alvarez et al. [7] | evolutionary content | dungeon structure | designer-selected descriptors | fitness/constraints | constrained MAP-Elites | no neural-repair decomposition | QD is not the proposed central contribution. |
| Sarkar and Cooper [8] | VAE | latent descriptors | indirect | fitness-based | playability objective | no | Latent QD differs from graph-conditioned logic supervision. |
| NSGGM [10] | neural graph proposals | symbolic graph constraints | user constraints | neural soft guidance | SMT assembly | neural-versus-symbolic separation | Different domain and representation; conceptually important comparator. |

The complete related-work phase should extend this table to at least 20 directly relevant papers and verify every bibliographic field before manuscript submission.

---

## 6. Research questions, hypotheses, and scope

### 6.1 Primary research question

> For unseen mission graphs, does joint mission-graph conditioning and differentiable local/global logic supervision increase pre-repair mission-consistent playability relative to neural-only and symbolic-only generation at comparable fidelity, diversity, and computational cost?

### 6.2 Subquestions

**RQ1: Conditioning.** Which mission-graph conditioning channels improve local connection alignment and global progression compliance?

**RQ2: Logic supervision.** Does training-time differentiable logic improve independently verified raw validity and reduce repair burden?

**RQ3: Inference guidance.** Does inference-time LogicNet guidance add value after logic-supervised training, and what computational cost does it introduce?

**RQ4: Symbolic intervention.** How much validity comes from constrained decoding and repair, and how much content do those interventions change?

**RQ5: Generalization and control.** Do improvements persist for held-out dungeon identities and out-of-distribution graph structures?

### 6.3 Confirmatory hypotheses

**H1.** Graph-conditioned diffusion has higher raw mission-constraint satisfaction than local-only diffusion.

**H2.** Adding local logic supervision increases raw room reachability and required-door alignment relative to graph conditioning alone.

**H3.** Adding global mission logic increases raw full-dungeon progression validity relative to local logic alone.

**H4.** The full logic-supervised model requires fewer repairs and a smaller changed-tile fraction than graph-conditioned diffusion without logic.

**H5.** The proposed model is non-inferior in fidelity and conditional diversity to graph-conditioned diffusion without logic.

**H6.** The direction of the primary effect persists across most held-out dungeon identities and on satisfiable OOD mission graphs.

**H7.** Inference-time logic guidance produces a smaller marginal gain than training-time logic supervision and increases sampling latency.

### 6.4 Exploratory questions

- Which constraint types remain difficult after logic supervision?
- Does logic supervision improve rare semantic tiles or cause conservative layouts?
- Does repair reduce novelty or diversity?
- Under which graph structures does symbolic repair become expensive?
- Can the system correctly identify or report unsatisfiable requests?

### 6.5 Scope boundaries

#### In scope

- Zelda-like 2D tile rooms and dungeons.
- Mission graphs with explicit room and edge semantics.
- VQ-VAE latent representations.
- Latent diffusion room generation.
- Local and global differentiable logic.
- Independent solver validation.
- Constrained decoding and minimal repair.
- Quantitative automatic evaluation.

#### Out of scope

- Generalization to all games.
- Claims about player enjoyment or perceived difficulty.
- Human-likeness claims based only on P-CBS.
- Multi-floor semantics without authoritative labels.
- A central quality-diversity contribution.
- A central fast-sampling contribution.
- A universal formal guarantee.
- Production deployment or real-time designer studies.

---

## 7. Study design

### 7.1 Research paradigm and method

The study is a quantitative controlled benchmarking experiment. The main causal factors are graph conditioning and logic supervision. Postprocessing is analyzed as a separate intervention axis.

The design combines:

- Factorial training ablations.
- Paired sampling experiments.
- Grouped cross-validation by dungeon identity.
- Out-of-distribution graph stress tests.
- Independent hard-solver evaluation.
- Distributional and controllability analysis.

### 7.2 Experimental unit

The primary experimental unit is a complete generated dungeon conditioned on one mission graph. Rooms from the same generated dungeon are not independent observations.

The hierarchy is:

```text
training seed
  └── held-out dungeon identity / graph family
        └── mission graph
              └── sampling seed
                    └── complete generated dungeon
                          └── rooms
```

Statistical analysis must respect this nesting.

### 7.3 Data split protocol

Use leave-one-dungeon-identity-out evaluation across dungeon identities 1 through 9.

For fold \(j\):

- Test data: both quest variants belonging to dungeon identity \(j\).
- Training/validation data: both variants of the other eight identities.
- Internal validation groups: whole dungeon identities, never randomly selected rooms.
- VQ-VAE: trained only on the fold's training data.
- Diffusion and LogicNet: trained only on the fold's training data.
- Hyperparameter selection: performed only through internal grouped validation.
- Test fold: evaluated once after all choices are frozen.

If nine-fold training is computationally impossible, the fallback is a preregistered grouped split covering at least three distinct held-out identities. This weakens the generalization claim and must be stated as a limitation.

### 7.4 Leakage controls

Before training, verify:

- Both quest variants of an identity remain in the same outer fold.
- No overlapping room windows cross train/test boundaries.
- No test-room tile frequencies are used to fit WFC rules or repair costs.
- No test graphs are used to tune graph generators or constraint thresholds.
- No target-room grid is supplied through topology conditioning.
- `topology_supervision_mode` remains `runtime_aligned` in confirmatory models.
- Nearest-neighbor analyses include all train/test pairs to detect duplicates.
- Synthetic OOD graphs are generated from a frozen specification independent of final model outputs.

### 7.5 Training replications and sampling

Preferred confirmatory design:

- Five independent training seeds per trained condition and outer fold.
- At least 30 paired sampling seeds per test mission graph and trained model.
- The same sampling seeds across comparable model variants.
- The same fold-specific VQ-VAE across diffusion ablations.

Minimum acceptable design if compute is constrained:

- Three independent training seeds.
- All nine outer held-out identities.
- At least 30 paired samples per graph.

Do not substitute hundreds of sampling seeds from one trained model for independent training replications.

### 7.6 Hyperparameter policy

Use a two-stage policy:

1. **Exploratory pilot:** tune learning rates, logic weights, guidance scales, and repair parameters using designated development folds only.
2. **Confirmatory freeze:** publish the chosen search space and selected values, freeze them, then execute all outer test folds without further tuning.

Every method receives a comparable tuning budget. The proposed method must not receive extensive tuning while baselines retain arbitrary defaults.

---

## 8. Model and baseline matrix

### 8.1 Core trained models

| ID | Graph context | Local logic loss | Global logic loss | Tile auxiliary loss | Purpose |
|---|---:|---:|---:|---:|---|
| N | no | no | no | matched/off | Local neural baseline. |
| G | yes | no | no | matched/off | Isolates graph conditioning. |
| G-L | yes | yes | no | off | Isolates local reachability supervision. |
| G-LG | yes | yes | yes | off | Proposed logic-supervised neural model. |

Tile auxiliary supervision is off in all four confirmatory variants. If later enabled, it becomes a separately trained exploratory ablation rather than an unnoticed second factor in G-L or G-LG.

### 8.2 Inference variants

For each applicable trained model, sample two paired inference conditions: `unguided` and `logic_guided`. Match them on outer fold, training seed, sampling seed, and mission-graph identity. The guidance scale and all other sampler settings must be frozen before the outer test runs.

`raw`, `constrained`, `repaired`, and `compiled` are repeated stages from one attempted sample, not separately sampled inference variants. Treating `-RAW`, `-CONSTR`, and `-REPAIR` as distinct generation arms would break paired attribution and can give each stage different randomness or retry opportunities.

Do not describe inference guidance as logic-supervised training. Training and inference interventions answer different questions. Report generation errors, validator errors, and timeouts for both conditions; never compare only surviving pairs.

### 8.3 External and symbolic baselines

#### Required baselines

1. **Symbolic-only WFC/constraint generator.** Uses only training-fold statistics and mission constraints.
2. **Generate-and-filter.** Uses the same graph-conditioned neural generator, rejects invalid samples, and receives the same wall-clock or attempt budget as repair-based generation.
3. **Graph grammar plus learned rooms.** A faithful graph-grammar-plus-GAN or equivalent learned-room baseline modeled after the closest Zelda work [2].
4. **Unconditional/local diffusion.** Same VQ-VAE and denoiser capacity without mission-graph context.
5. **Graph-conditioned diffusion without logic.** The most important internal baseline.

#### Optional baselines

- Discrete masked-room generator under matched folds and compute.
- Search in the VQ-VAE latent space.
- A lightweight autoregressive categorical model.
- QD/MAP-Elites as a separate diversity study.

Optional baselines should not delay the required causal ablations.

### 8.4 Fairness requirements

- Same train/test folds and tile schema.
- Same number of requested outputs.
- Same validity evaluator.
- Same hardware reporting.
- Same sample or wall-clock budget for generate-and-filter comparisons.
- Comparable model-selection budget.
- No baseline may use test data to build grammar, adjacency, or repair statistics.
- Report parameter counts and effective training updates.
- Report failures and timeouts as failures, not missing observations.

---

## 9. Experiment program

### E0. Data, representation, and evaluator integrity

**Purpose:** establish that the study inputs and hard evaluator are trustworthy before model comparisons.

Checks:

- File-level split hashes.
- Dungeon/quest grouping.
- Room shape and tile vocabulary.
- Graph-node-to-room alignment.
- Edge direction and semantic labels.
- Runtime-aligned topology contains no target-room tiles.
- VQ-VAE reconstruction validity and rare-tile recall.
- A* and global-state validation on hand-constructed positive and negative cases.
- Validator consistency against an independent Dijkstra comparator where supported.
- Correct rejection of unsatisfiable test specifications.

Deliverable: a machine-readable integrity report and a short paper appendix table.

### E1. Main effectiveness comparison

Compare N, G, G-L, and G-LG on every outer held-out identity.

Primary outcome:

- Exact mission validity of the raw room proposal after neutral fixed-slot assembly and physical-edge carving. Boundary sealing and connection realization are held constant across raw, constrained, and repaired stages.

Secondary outcomes:

- Raw room reachability.
- Graph-edge/door agreement.
- Global progression constraints.
- Fidelity.
- Conditional diversity.
- Sampling time.

This experiment tests H1-H3.

### E2. Conditioning mechanism ablation

Starting from G-LG, disable one channel at a time:

1. Graph node tokens.
2. Topological positional encodings.
3. Spatial topology map.
4. Neighbor-room reference context.
5. Boundary/door requirements.
6. Edge-semantic features.
7. Graph-to-grid spatial modulation if enabled in the proposed model.

Do not use the current `no_stage_tokens` comparison unless puzzle-stage conditioning is enabled by a separately trained model. With the canonical configuration, that switch can be non-operative.

### E3. Logic mechanism ablation

Compare separately trained models:

- Graph only.
- Graph plus tile auxiliary loss.
- Graph plus local reachability logic.
- Graph plus global progression logic.
- Graph plus local and global logic.
- Full model with detached-real logic mode.
- Full model with predicted-latent gradient flow.

Then evaluate each relevant trained model with inference guidance off and on. This creates a clean distinction between learning a better generator and steering an existing generator.

### E4. Symbolic intervention and repair burden

For the same attempted samples, record metrics at:

1. Raw neural room proposals after neutral assembly.
2. Deterministically constrained room outputs after the same neutral assembly.
3. Explicitly repaired room outputs after the same neutral assembly.
4. Final compiled dungeon after mission-graph semantics and entities are materialized.

Report validity transitions and content modifications. Compare against generate-and-filter under matched attempt and wall-clock budgets.

This experiment tests H4 and exposes whether postprocessing dominates the system.

### E5. Controllability and counterfactual response

Construct mission-graph pairs that differ in exactly one target property while holding the sampling seed fixed.

Target properties:

- Number of rooms.
- Critical-path length.
- Branch count.
- Lock/key count.
- Switch/gate count.
- Required room role.
- Door direction or one-way edge.

For each pair, measure:

- Target-property change achieved.
- Unrequested property changes.
- Mission validity.
- Structural edit localization.
- Monotonic response across ordered target sweeps.

### E6. Out-of-distribution graph generalization

Create a frozen graph test suite with:

- In-range sizes with unseen compositions.
- Larger-than-training graphs.
- Higher branching factors.
- Longer dependency chains.
- Rare but satisfiable lock/key and switch/gate combinations.
- Directed and one-way transitions.
- Deliberately unsatisfiable cases.

Every graph must be labeled as satisfiable or unsatisfiable by a graph-level oracle before room generation.

Required outcomes:

- Validity by OOD category.
- Attempts and time per valid dungeon.
- Repair burden by graph complexity.
- Correct UNSAT reporting or graceful failure.
- Calibration between soft LogicNet score and exact validity.

### E7. Efficiency and scalability

Measure:

- Training GPU-hours.
- Peak accelerator memory.
- Parameter count.
- Sampling latency per room and dungeon.
- Logic-guidance overhead.
- Validation latency.
- Repair latency.
- Attempts per valid dungeon.
- Valid dungeons per wall-clock minute.

Report results for at least two graph sizes. Use the same hardware and synchronization policy.

### E8. Failure analysis

Build a taxonomy from exact failures:

- Missing required door or marker.
- Disconnected local path.
- Incorrect boundary alignment.
- Premature gate or missing prerequisite.
- Unintended shortcut.
- Unreachable required room.
- Rare-tile collapse.
- Excessive repair.
- Solver timeout.
- Unsatisfiable request not detected.

For each category, include frequency, representative examples, and which intervention first resolves it.

### E9. Optional human evaluation

Human evaluation is not required for the core algorithmic paper. If added later, it should answer aesthetic quality, perceived coherence, or designer usefulness rather than replace objective validity testing.

It requires:

- Ethics/IRB determination before recruitment.
- Informed consent.
- Power analysis.
- Randomized and blinded presentation.
- Clear separation between expert designers and general players.
- No claim of human-likeness from automatic agents alone.

---

## 10. Metrics and exact definitions

### 10.1 Evaluation stages

Let \(D^{raw}\), \(D^{con}\), and \(D^{rep}\) denote the raw, constrained, and repaired dungeon outputs generated from the same initial sample.

Every metric must include a stage suffix. For example:

- `mission_valid_raw`
- `mission_valid_constrained`
- `mission_valid_repaired`

Never aggregate stages into one ambiguous `validity` field.

### 10.2 Hard validity

For constraint set \(\mathcal{C}(G)\), define:

\[
V(D,G) = \mathbf{1}\left[\bigwedge_{c \in \mathcal{C}(G)} c(D)=\text{true}\right].
\]

Report:

- Full mission validity: mean of \(V(D,G)\).
- Per-constraint satisfaction.
- Room reachability rate.
- Required-edge realization rate.
- Start-to-goal solution rate.
- Softlock rate.
- Timeout and evaluator-error rates.

An evaluator error is not a valid sample and must be reported separately from a proven invalid dungeon.

### 10.3 Door and semantic-marker alignment

For required markers or graph-aligned door endpoints:

\[
Precision = \frac{TP}{TP+FP}, \qquad
Recall = \frac{TP}{TP+FN}, \qquad
F_1 = \frac{2PR}{P+R}.
\]

Report macro averages across semantic types as well as micro averages. Rare locks, keys, switches, and gates must not disappear inside a dominant ordinary-door class.

### 10.4 Repair dependence

Any-repair rate:

\[
R_{any}=\frac{1}{N}\sum_i \mathbf{1}[D_i^{raw} \neq D_i^{rep}].
\]

Changed-tile fraction:

\[
R_{edit}=\frac{1}{N}\sum_i
\frac{\lVert D_i^{raw}-D_i^{rep}\rVert_0}{|D_i|}.
\]

Also report:

- Number of added, removed, and moved semantic markers.
- Room-level and dungeon-level edit counts.
- Repair latency.
- Repair failure rate.
- Valid-to-invalid regression rate.
- Raw-invalid-to-final-valid conversion rate.

### 10.5 Fidelity

Use multiple complementary measures:

- Tile-frequency Jensen-Shannon divergence.
- Local 2 by 2 and 3 by 3 pattern distribution divergence.
- Precision and recall in a frozen representation space.
- Nearest-training-room distance.
- Exact duplicate and near-duplicate rates.
- Structural statistics such as walkable ratio, connected components, and path length.

No single perceptual metric should define fidelity.

### 10.6 Diversity and novelty

Normalized pairwise Hamming diversity for samples conditioned on the same graph:

\[
Div(G)=\frac{2}{n(n-1)}\sum_{i<j}
\frac{\lVert D_i-D_j\rVert_0}{|D|}.
\]

Also report:

- Unique room and dungeon fractions.
- Descriptor-space coverage.
- Conditional diversity at fixed mission graph.
- Cross-graph structural diversity.
- Novelty relative to the nearest training dungeon.

Separate diversity from novelty: samples may differ from each other while all remain close to training examples.

### 10.7 Controllability

For requested scalar target \(t_i\) and realized value \(\hat{t}_i\):

\[
MAE_t = \frac{1}{N}\sum_i |t_i-\hat{t}_i|.
\]

Report:

- MAE or exact match where appropriate.
- Spearman correlation across target sweeps.
- Monotonicity violation rate.
- Counterfactual success rate.
- Unrequested-change magnitude.
- Validity conditional on target difficulty.

### 10.8 Constraint-satisfaction efficiency

Attempt efficiency:

\[
CSE_{attempt}=\frac{N_{valid}}{N_{attempts}}.
\]

Time efficiency:

\[
CSE_{time}=\frac{N_{valid}}{T_{generation}+T_{validation}+T_{repair}}.
\]

Use these metrics when comparing repair against rejection sampling.

### 10.9 Calibration

Compare the soft LogicNet score against exact validity using:

- Reliability diagrams.
- Expected calibration error.
- AUROC and area under the precision-recall curve.
- Brier score.

Calibration analysis can reveal whether LogicNet is useful as a prioritization heuristic even when it is not an exact validator.

---

## 11. Statistical analysis plan

### 11.1 Primary endpoint

The primary endpoint is raw full-dungeon exact mission validity for G-LG versus G.

The primary model is a mixed-effects logistic regression:

\[
logit\,P(Y_{ijkl}=1)
= \beta_0 + \beta_{method}
+ u_{dungeon_i}+u_{trainseed_j}+u_{graph_k},
\]

where random effects account for held-out dungeon identity, training seed, and mission graph where identifiable.

Report:

- Odds ratio.
- Absolute marginal probability difference.
- 95% confidence interval.
- Per-dungeon effects.
- Predicted probabilities, not only regression coefficients.

### 11.2 Continuous and count outcomes

- Changed-tile fraction: beta or zero-inflated model if distribution warrants it.
- Repair edit count: negative-binomial mixed model if overdispersed.
- Latency: log transformation or nonparametric clustered bootstrap.
- Diversity and fidelity: hierarchical linear model or graph-clustered bootstrap.
- Attempts per valid output: survival or count analysis with capped failures retained.

Model choice must be based on outcome distribution and recorded before final hypothesis testing.

### 11.3 Paired comparisons

Use paired sampling seeds and mission graphs. Bootstrap entire mission-graph or dungeon clusters rather than individual rooms.

### 11.4 Multiplicity

Define one primary comparison. Apply Holm correction within each secondary hypothesis family:

- Conditioning ablations.
- Logic ablations.
- Repair-stage comparisons.
- OOD categories.

Exploratory analyses should be labeled exploratory rather than folded into confirmatory significance claims.

### 11.5 Effect sizes and practical thresholds

Before confirmatory runs, define:

- Minimum important absolute gain in raw validity, \(\delta_V\).
- Maximum acceptable fidelity degradation, \(\delta_F\).
- Maximum acceptable diversity degradation, \(\delta_D\).
- Minimum useful repair-edit reduction, \(\delta_R\).

Estimate candidate margins from pilot inter-seed variability and domain reasoning, then freeze them before test-fold evaluation. Do not redefine success after seeing confirmatory results.

### 11.6 Power analysis

Use pilot estimates to simulate the hierarchical design. Vary:

- Baseline raw validity.
- Expected method effect.
- Between-dungeon variance.
- Between-training-seed variance.
- Samples per graph.

Prioritize additional independent training seeds or graph identities over excessive sampling from one model when variance is dominated by training.

### 11.7 Missingness and failures

- Training divergence remains a method failure and must be counted.
- Solver timeout is reported separately and treated conservatively in validity analysis.
- Generation exceptions remain in the denominator.
- No failed seed may be silently replaced after test results are inspected.
- A replacement is allowed only under a preregistered infrastructure-failure rule and must be logged.

---

## 12. Reproducibility and provenance protocol

### 12.1 Run manifest

Every run should record:

- Git commit hash.
- Dirty-worktree diff hash or explicit clean-worktree flag.
- Configuration content hash.
- Dataset manifest hash.
- Outer fold and internal validation split.
- Training, data-loader, and sampling seeds.
- Python, PyTorch, CUDA, and driver versions.
- GPU/CPU model and memory.
- Start/end timestamps.
- Model parameter count.
- Training updates and wall-clock time.
- Checkpoint-selection rule.
- Evaluator version and constraint schema version.

Checkpoints may be stored for reproducibility, but no pre-existing checkpoint is valid evidence for the new paper.

### 12.2 Raw result schema

One row per generated dungeon should include at least:

```text
run_id
git_commit
config_hash
data_manifest_hash
outer_fold
train_seed
sampling_seed
mission_graph_id
mission_graph_family
model_id
inference_variant
generation_status
raw_grid_path
constrained_grid_path
repaired_grid_path
raw_validation_fields...
constrained_validation_fields...
repaired_validation_fields...
repair_edit_fields...
fidelity_fields...
diversity_group_id
timing_fields...
error_type
```

Aggregate tables must be regenerable from this raw schema.

### 12.3 Environment and execution

- Freeze dependencies using the repository lock file or a generated environment lock.
- Record deterministic-mode settings and known nondeterministic operations.
- Synchronize GPU timing before latency measurements.
- Warm up models before timed inference.
- Separate model-loading time from generation time.
- Run all compared methods on the same hardware class.

### 12.4 Preregistration

Register the confirmatory protocol on OSF before outer-fold evaluation. The registration should include:

- Primary and secondary hypotheses.
- Frozen model variants.
- Data splits.
- Metrics.
- Exclusion/failure rules.
- Statistical models.
- Multiplicity correction.
- Claim gates.

Pilot experiments should be labeled exploratory and excluded from confirmatory p-values.

---

## 13. Claim gates

No claim should enter the title, abstract, or conclusion until its gate passes.

| Candidate claim | Required evidence | Disallowed shortcut |
|---|---|---|
| Graph conditioning improves semantic control. | G exceeds N on raw exact metrics across held-out identities with practically meaningful effect and confidence interval. | Comparing only repaired outputs. |
| Local logic improves room playability. | Separately trained G-L exceeds G on independent room-reachability metrics. | Switching inference guidance on for one checkpoint. |
| Global logic improves mission progression. | G-LG exceeds G-L on independent full-state progression validity. | Reusing LogicNet's own score as ground truth. |
| Logic reduces symbolic dependence. | G-LG has lower any-repair rate and edit burden than G. | Reporting final validity alone. |
| Quality is preserved. | Fidelity and conditional diversity meet frozen non-inferiority margins. | Showing a few favorable images. |
| The model generalizes. | Effect persists over grouped held-out identities and OOD graph families. | One Dungeon-9 run. |
| The system is efficient. | Better valid-output throughput or justified quality/cost tradeoff on matched hardware. | Reporting neural sampling time without validation and repair. |
| The output is guaranteed. | Every returned output is solver-verified; guarantee explicitly limited to encoded semantics; validator assumptions documented. | Treating differentiable loss as proof. |
| The system supports quality diversity. | QD enabled, archive metrics reported across seeds, and compared against QD baselines. | Referring to disabled MAP-Elites code. |

Recommended primary success gate:

- G-LG improves raw exact mission validity over G.
- The confidence interval excludes the null and the preregistered negligible-effect region.
- The effect direction is positive in a clear majority of held-out identities.
- Fidelity and conditional diversity remain within frozen non-inferiority margins.
- Repair edit burden decreases.

If only final repaired validity improves, the paper must be reframed as a symbolic repair or hybrid systems paper rather than a logic-supervised generative-model paper.

---

## 14. Existing protocol weaknesses that must be corrected

### 14.1 Conditioning/LogicNet/repair ablation script

[`scripts/run_conditioning_logicnet_repair_ablation.py`](scripts/run_conditioning_logicnet_repair_ablation.py) is useful scaffolding but is not a sufficient confirmatory experiment.

Problems:

- `no_stage_tokens` changes a signal disabled in the canonical configuration, so the comparison may be vacuous.
- The logic flag changes runtime guidance, not training-time logic supervision.
- Multiple inference variants can reuse one trained model, which does not isolate training causality.
- Three sampling seeds do not replace independent training seeds.

Required correction:

- Create distinct frozen training configurations for N, G, G-L, and G-LG.
- Add explicit channel-level condition ablations.
- Retain inference guidance as a separate crossed factor.
- Use grouped outer folds and multiple training seeds.

### 14.2 Dungeon-9 holdout protocol

[`scripts/run_dungeon9_holdout_protocol.py`](scripts/run_dungeon9_holdout_protocol.py) explicitly uses a lightweight GAN and diagnostic ablations without retraining the full diffusion model. It cannot support the main generalization claim.

Required correction:

- Replace the single holdout with grouped outer folds.
- Retrain the full model within each fold.
- Retain the script only as baseline scaffolding or a smoke test.

### 14.3 Designer controllability scripts

Topology target sweeps can support E5, but topology-generator performance must not be confused with room-generator controllability. Report graph synthesis and graph-conditioned room realization as separate stages.

### 14.4 Existing outputs

Existing output and artifact directories must not be scanned to choose favorable metrics, thresholds, examples, or seeds. Example selection for the paper must use a frozen rule, such as median case, best/worst by predefined metric, or a fixed sampling seed.

---

## 15. Required implementation work

The publication code path should be smaller than the complete repository.

### 15.1 Proposed publication experiment package

```text
experiments/paper_v1/
  README_PROTOCOL.md
  configs/
    n_local.yaml
    g_graph.yaml
    g_l_local_logic.yaml
    g_lg_full_logic.yaml
  manifests/
    dataset_manifest.json
    outer_folds.json
    ood_graph_suite.json
  run_train.py
  run_generate.py
  run_validate.py
  run_analysis.py
  schemas/
    raw_result.schema.json
    run_manifest.schema.json
  analysis/
    aggregate.py
    mixed_effects.R_or_Python
    figures.py
    tables.py
```

This layout is a proposal, not yet implemented by this document.

### 15.2 Training runner requirements

- Accept one outer fold, model ID, and training seed.
- Derive no scientific setting from a checkpoint.
- Validate the frozen configuration before training.
- Write a complete run manifest before the first update.
- Fail loudly on missing graph context or inactive requested loss.
- Log raw and weighted loss terms and gradient norms.
- Apply one checkpoint-selection rule using internal validation only.

### 15.3 Generation runner requirements

- Accept a trained run ID and frozen mission-graph suite.
- Generate paired seeds across variants.
- Save raw decoded grids before any intervention.
- Save each subsequent constrained and repaired grid.
- Record all runtime overrides.
- Reject incompatible inference flags rather than silently disabling them.

### 15.4 Validation runner requirements

- Consume saved grids without invoking the generative model.
- Use the canonical hard oracle and global-state validation.
- Record applicability of every constraint.
- Distinguish invalid, timeout, evaluator error, and unsatisfiable specification.
- Produce per-stage evidence.
- Optionally run an independent Dijkstra consistency check.

### 15.5 Analysis runner requirements

- Consume raw records only.
- Validate schema and uniqueness keys.
- Produce all tables from code.
- Preserve cluster identifiers.
- Generate confidence intervals and corrected p-values.
- Produce a claim-gate report automatically.
- Refuse to mix exploratory and confirmatory run IDs.

### 15.6 Tests required before expensive training

- Fold grouping and leakage tests.
- Configuration-to-model activation tests.
- Loss-gradient tests for every logic ablation.
- Condition permutation tests.
- Raw/constrained/repaired snapshot retention tests.
- Validator positive/negative fixture tests.
- Repair edit accounting tests.
- Seed pairing tests.
- Manifest completeness tests.
- Analysis schema and aggregation tests.

---

## 16. Paper architecture

### 16.1 Conference version

Target length depends on the venue. For a roughly 5,000-word body:

| Section | Target words | Function |
|---|---:|---|
| Abstract | 180-220 | Problem, method, fresh quantitative result, significance. |
| 1. Introduction | 650-750 | Establish validity/control gap, RQ, and contributions. |
| 2. Related Work | 700-850 | Position against dungeon PCG, PCGML/diffusion, and constrained generation. |
| 3. Problem and Method | 1,150-1,350 | Define graph/grid task, conditioning, logic losses, and repair stages. |
| 4. Experimental Protocol | 850-1,000 | Data folds, models, baselines, metrics, and statistics. |
| 5. Results | 950-1,150 | Main comparison, ablations, repair burden, OOD. |
| 6. Discussion and Limitations | 450-600 | Interpret mechanism, failures, scope, and threats. |
| 7. Conclusion | 150-220 | Qualified answer to the RQ. |

### 16.2 Journal version

A Transactions on Games version should add:

- All outer folds.
- A broader closest-work review.
- Full mechanism ablations.
- OOD graph suite.
- Calibration and failure analysis.
- Efficiency and scalability.
- Expanded validity threats.
- Optional designer or player evaluation if separately approved.

### 16.3 Detailed section blueprint

#### Abstract

Write only after confirmatory analysis. Required sequence:

1. Problem: learned level generators can reproduce local patterns but often violate global progression rules.
2. Method: mission-graph-conditioned latent diffusion with local/global differentiable logic and independent repair.
3. Protocol: grouped unseen-dungeon evaluation with raw, constrained, and repaired stages.
4. Results: exact effect sizes with confidence intervals.
5. Significance: reduced gap between learned proposal quality and verified dungeon generation.

Do not include results placeholders in a submitted abstract.

#### 1. Introduction

Argument chain:

1. Dungeon quality depends on both room-level tile structure and global progression.
2. Learned generators model local distributions but do not inherently satisfy explicit mission constraints.
3. Pure symbolic generators satisfy rules but may lack learned stylistic fidelity and conditional variety.
4. Existing graph-grammar-plus-learned-room systems do not establish that room generation learns graph-specific semantics.
5. Existing diffusion-level work does not solve multi-room mission consistency.
6. The paper tests whether differentiable mission logic improves the neural proposal before exact repair.

End with one RQ and a short numbered contribution list.

#### 2. Related work

Organize by problem, not by algorithm chronology:

1. Dungeon structure and mission control.
2. Learned tile-level generation and diffusion.
3. Conditional generation and automated validation.
4. Neuro-symbolic generation, constraints, and repair.

Each subsection should end by stating the unresolved point needed by this paper.

#### 3. Problem formulation

Define:

- Mission graph schema.
- Tile grid schema.
- Local constraints.
- Global stateful constraints.
- Conditional generation objective.
- Raw, constrained, repaired, and verified outputs.

#### 4. Method

Recommended order:

1. Fold-specific VQ-VAE representation.
2. Dual-stream condition encoder.
3. Graph-conditioned latent diffusion.
4. Local differentiable reachability.
5. Global mission progression loss.
6. Optional inference-time logic guidance.
7. Constrained decoding and symbolic repair.
8. Exact validation contract.

Do not describe disabled branches.

#### 5. Experimental protocol

Include:

- Dataset and grouped folds.
- OOD mission-graph suite.
- Baselines.
- Training and selection.
- Sampling design.
- Metrics at each stage.
- Statistical tests.
- Compute and reproducibility.

#### 6. Results

Recommended order:

1. Main raw-validity result.
2. Local versus global constraint breakdown.
3. Repair burden and stage transitions.
4. Fidelity/diversity tradeoff.
5. Conditioning and logic ablations.
6. OOD controllability.
7. Efficiency.

State results before interpreting them.

#### 7. Discussion

Answer:

- Did logic improve the generator or only correlate with validity?
- Which graph channel mattered?
- How much did repair contribute?
- What quality cost accompanied higher validity?
- Where did the method fail?
- Which conclusions are Zelda-specific?

Address at least two alternative explanations:

1. Gains came from extra auxiliary supervision rather than logical structure.
2. Gains came from conservative layouts that are valid but less diverse.

#### 8. Limitations

Required limitations:

- Small single-game corpus.
- Potential evaluator dependence on encoded rules.
- Compute cost of grouped retraining.
- Incomplete coverage of player-perceived quality.
- Constraint schema limits.
- Possible repair-induced distribution shift.
- OOD graphs may not represent human-authored missions.

#### 9. Conclusion

Answer the primary RQ using qualified language. Do not restate every component. State what the evidence supports, what remains unresolved, and one concrete next research direction.

---

## 17. Planned figures and tables

### 17.1 Figures

**Figure 1: Method overview.** Mission graph to condition encoder, latent diffusion, raw room grids, LogicNet supervision, symbolic intervention, and exact validation. Distinguish training-only and inference-only arrows.

**Figure 2: Same-graph qualitative comparison.** Fixed graph and sampling seed for N, G, G-LG, and G-LG-REPAIR. Mark violated constraints and repaired tiles.

**Figure 3: Raw-to-final validity decomposition.** Paired validity or transition plot for raw, constrained, and repaired stages.

**Figure 4: Held-out identity effects.** Forest plot of per-dungeon raw-validity effects with pooled interval.

**Figure 5: Validity-quality tradeoff.** Raw validity versus fidelity/diversity, with training seeds shown separately.

**Figure 6: OOD generalization.** Validity and repair burden against graph size or dependency depth.

**Figure 7: Logic calibration.** Reliability curve of LogicNet score against exact solver validity.

Use colorblind-safe colors, vector formats where possible, and visible training-seed uncertainty.

### 17.2 Tables

**Table 1:** Closest-work comparison.  
**Table 2:** Dataset and graph schema.  
**Table 3:** Model and ablation definitions.  
**Table 4:** Main raw/constrained/repaired metrics.  
**Table 5:** Conditioning and logic ablations.  
**Table 6:** OOD and controllability results.  
**Table 7:** Runtime, attempts per valid output, parameters, and GPU-hours.  
**Appendix Table A1:** Full hyperparameters.  
**Appendix Table A2:** Per-fold and per-seed results.  
**Appendix Table A3:** Constraint-by-constraint failures.  

---

## 18. Results table shells

All cells remain `TBD` until fresh confirmatory runs complete.

### 18.1 Main result

| Model | Raw mission valid | Constrained valid | Repaired valid | Any repair | Changed tiles | Fidelity | Conditional diversity | Seconds/valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Symbolic-only | TBD | N/A | TBD | N/A | N/A | TBD | TBD | TBD |
| N | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| G | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| G-L | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| G-LG | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| G-LG + guidance | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 18.2 Constraint breakdown

| Model | Room path | Required doors | Global reachability | Key-lock order | Switch-gate order | No shortcut | All constraints |
|---|---:|---:|---:|---:|---:|---:|---:|
| N | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| G | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| G-L | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| G-LG | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 18.3 Claim-gate outcome

| Claim gate | Estimate | 95% CI | Preregistered threshold | Pass/fail |
|---|---:|---:|---:|---|
| Raw validity: G-LG versus G | TBD | TBD | TBD | TBD |
| Repair burden: G-LG versus G | TBD | TBD | TBD | TBD |
| Fidelity non-inferiority | TBD | TBD | TBD | TBD |
| Diversity non-inferiority | TBD | TBD | TBD | TBD |
| OOD direction consistency | TBD | TBD | TBD | TBD |

---

## 19. Threats to validity and mitigation

### 19.1 Internal validity

**Threat:** model variants differ in more than the intended mechanism.  
**Mitigation:** share VQ-VAE, denoiser capacity, training schedule, data folds, and tuning budget; change one factor per ablation.

**Threat:** graph conditioning leaks target-room geometry.  
**Mitigation:** runtime-aligned topology, input audits, permutation tests, and explicit prohibition of target grid features.

**Threat:** evaluator shares LogicNet assumptions.  
**Mitigation:** full-state A* and exact mission validators independent of LogicNet.

**Threat:** repair creates the observed validity.  
**Mitigation:** report raw, constrained, and repaired outputs plus edit burden.

**Threat:** seed selection.  
**Mitigation:** preregister seed-generation rule and retain all completed runs.

### 19.2 Construct validity

**Threat:** solver validity is equated with gameplay quality.  
**Mitigation:** call it validity under encoded mission semantics; separately measure fidelity, diversity, and structural properties.

**Threat:** diversity metric rewards meaningless tile noise.  
**Mitigation:** combine tile diversity with structural descriptor coverage and validity-conditioned diversity.

**Threat:** LogicNet loss is described as logic correctness.  
**Mitigation:** present it as a differentiable surrogate and measure calibration against exact validation.

### 19.3 External validity

**Threat:** one game and a small number of dungeon identities.  
**Mitigation:** grouped leave-one-identity-out evaluation, OOD graph tests, narrow claims, and future cross-game validation.

**Threat:** synthetic mission graphs are unrealistic.  
**Mitigation:** report their construction rules, separate in-distribution and synthetic results, and include human-authored graph cases where possible.

### 19.4 Statistical conclusion validity

**Threat:** pseudoreplication from treating rooms or samples as independent.  
**Mitigation:** hierarchical models and cluster bootstrap.

**Threat:** many ablations inflate false positives.  
**Mitigation:** one primary endpoint, hypothesis families, Holm correction, and exploratory labels.

**Threat:** underpowered training-seed comparisons.  
**Mitigation:** simulation-based power analysis and prioritization of independent seeds.

---

## 20. Devil's-advocate review

### 20.1 Strongest objection

> The neural model does not learn mission-consistent generation. It produces approximate rooms that a handcrafted symbolic system fixes, so the principal contribution is repair rather than graph-conditioned diffusion.

The study survives this objection only if G-LG improves raw exact validity over G and reduces repair burden without unacceptable fidelity or diversity loss.

### 20.2 Other likely reviewer objections

1. **Known-component assembly:** VQ-VAE, diffusion, GNN conditioning, logic losses, WFC, and repair are individually established.
2. **Small-data overfitting:** room-level samples may exaggerate the effective dataset size.
3. **Single-domain evidence:** conclusions may be Zelda-specific.
4. **Self-referential evaluation:** the learned loss and validator may encode the same biases.
5. **Weak baseline:** a tiny GAN does not fairly represent prior hybrid PCGML.
6. **Vacuous ablation:** a disabled conditioning path is presented as evidence.
7. **Post-hoc thresholds:** success criteria are selected after results are visible.
8. **Guarantee overclaim:** solver success under an implementation is generalized to all playability.

### 20.3 Required responses

- Provide equations and a causal ablation design for the coupling.
- Use complete dungeon identities as independent groups.
- Limit claims and add OOD graph tests.
- Use independent full-state validation.
- Implement a credible closest learned baseline.
- Assert that every ablated channel was active before disabling it.
- Preregister margins and analysis.
- Use `verified under encoded semantics` rather than universal guarantee language.

---

## 21. Venue strategy

### 21.1 AIIDE

Best fit when the final story is concise and centered on a clear game-AI contribution. The paper should emphasize mission-level controllability, algorithmic evaluation, failure cases, and practical implications for procedural dungeon design.

Minimum package:

- Fresh grouped evaluation.
- Strong internal ablations.
- One credible external baseline.
- Raw-versus-repaired decomposition.
- Reproducible code and examples.

### 21.2 IEEE Conference on Games

Best fit when the work emphasizes benchmarking, controllability, search/generation comparisons, or a broader systems evaluation.

### 21.3 IEEE Transactions on Games

Best fit for the complete version with:

- All held-out identities.
- Full OOD suite.
- Detailed statistical modeling.
- Complete failure taxonomy.
- Efficiency analysis.
- Extended related work.
- Optional human/designer study.

### 21.4 Submission rule

Choose one venue before final formatting. Verify its current page limit, anonymity rule, supplemental-material policy, code policy, and AI-use disclosure requirements from the official venue site immediately before submission.

---

## 22. Project execution order

### Milestone 1: Freeze the scientific core

- Confirm title, RQ, and target claims.
- Confirm excluded branches.
- Select one denoiser and one training objective.
- Define exact local and global constraint sets.

Exit criterion: every main-paper component maps to one hypothesis or required evaluation stage.

### Milestone 2: Establish data and evaluator integrity

- Build grouped split manifests.
- Audit topology inputs for leakage.
- Validate graph-room mappings.
- Establish exact evaluator fixtures.
- Test VQ-VAE reconstruction semantics.

Exit criterion: E0 passes with no unresolved critical defect.

### Milestone 3: Implement publication runners

- Frozen training configurations.
- Manifest-aware runner.
- Stage-preserving generation.
- Independent validation.
- Raw-result schema.
- Automated aggregation.

Exit criterion: a small smoke run produces a complete, schema-valid record from training through analysis.

### Milestone 4: Pilot study

- Use development folds only.
- Estimate variance and failure rates.
- Tune logic weights and guidance.
- Select practical-effect and non-inferiority margins.
- Run power simulation.

Exit criterion: confirmatory protocol and preregistration are frozen.

### Milestone 5: Confirmatory training and evaluation

- Execute all folds and seeds.
- Retain all failures.
- Generate paired samples.
- Run independent validation.
- Lock raw records before inferential analysis.

Exit criterion: complete expected-run matrix or documented preregistered exceptions.

### Milestone 6: Analysis and claim audit

- Produce descriptive and inferential tables.
- Apply multiplicity corrections.
- Run claim gates.
- Select examples through frozen rules.
- Complete failure analysis.

Exit criterion: every intended claim has passed, been weakened, or been removed.

### Milestone 7: Manuscript and review

- Draft results from generated tables.
- Complete citation verification.
- Run an adversarial paper review.
- Revise no more than two structured rounds before logging remaining limitations.
- Format for the selected venue.

---

## 23. Reporting statements

### 23.1 Data availability template

> The study uses tile maps and mission-graph data derived from the Video Game Level Corpus. The exact file manifest, grouped train/validation/test splits, preprocessing configuration, and hashes used in each experiment will be released with the code, subject to the source dataset's distribution terms.

### 23.2 Code availability template

> Code for training, generation, independent validation, statistical analysis, and figure production will be released at [repository URL] with frozen configurations and environment instructions. Each reported result will map to a run manifest and raw per-dungeon record.

### 23.3 Ethics declaration template

> The core study uses game-level data and automated evaluation and does not involve human participants or personal data. If a later player or designer study is added, it will be conducted only after the applicable institutional ethics determination and informed-consent process.

### 23.4 Conflict-of-interest template

> The authors declare no competing interests. [Replace if a conflict exists.]

### 23.5 Funding template

> This work was supported by [funder and grant number]. The funder had no role in study design, analysis, interpretation, or publication decisions. [Replace with the accurate statement, including `no external funding` when applicable.]

### 23.6 CRediT author-contribution template

- Conceptualization: [names]
- Methodology: [names]
- Software: [names]
- Validation: [names]
- Formal analysis: [names]
- Investigation: [names]
- Data curation: [names]
- Writing, original draft: [names]
- Writing, review and editing: [names]
- Visualization: [names]
- Supervision: [names]
- Project administration: [names]
- Funding acquisition: [names]

### 23.7 AI-use disclosure template

> Generative AI tools were used to assist with code inspection, research planning, language editing, and/or software development. The authors verified the source code, references, experimental design, analyses, and final text and remain responsible for all content. No generated numerical result was accepted without execution and independent validation.

Adapt this statement to the selected venue's current policy.

---

## 24. Final publication-readiness checklist

### Scientific framing

- [ ] One primary RQ appears consistently in title, introduction, method, results, and conclusion.
- [ ] The contribution is the tested coupling, not the number of modules.
- [ ] Disabled or optional branches are absent from the core story.
- [ ] Claims are limited to Zelda-like mission-graph-conditioned dungeons.

### Data and leakage

- [ ] Both quest variants remain grouped by dungeon identity.
- [ ] Every outer fold has a manifest and hashes.
- [ ] VQ-VAE and WFC statistics use training folds only.
- [ ] Runtime topology does not contain target-room geometry.
- [ ] Duplicate and near-duplicate analysis is complete.

### Models and baselines

- [ ] N, G, G-L, and G-LG are separately trained.
- [ ] Inference guidance is analyzed separately.
- [ ] A credible graph-grammar-plus-learned-room baseline is included.
- [ ] Generate-and-filter receives a matched budget.
- [ ] All methods use the same hard evaluator.

### Evaluation

- [ ] Raw, constrained, and repaired grids are retained.
- [ ] Full-state A* and global mission checks are independent of LogicNet.
- [ ] Repair edits and latency are reported.
- [ ] Fidelity, diversity, controllability, and cost accompany validity.
- [ ] OOD graphs include satisfiable and unsatisfiable cases.

### Statistics

- [ ] Primary endpoint and comparison are preregistered.
- [ ] Training seeds are independent replications.
- [ ] Hierarchical dependence is modeled.
- [ ] Confidence intervals and effect sizes are reported.
- [ ] Multiplicity correction is applied by hypothesis family.
- [ ] Missing runs and timeouts follow frozen rules.

### Reproducibility

- [ ] Commit, configuration, dataset, environment, and hardware are recorded.
- [ ] Raw result schema is complete.
- [ ] Tables and figures regenerate from raw records.
- [ ] Existing artifacts and checkpoints are not mixed into confirmatory evidence.

### Writing and submission

- [ ] Every external claim has a verified citation.
- [ ] Every numerical claim maps to a fresh result table.
- [ ] Abstract contains no unsupported result.
- [ ] Limitations are substantive.
- [ ] Data, code, ethics, funding, conflict, CRediT, and AI-use statements are present.
- [ ] Venue requirements are checked from the current official page.
- [ ] The paper has undergone adversarial review.

---

## 25. Working bibliography

Every item below was selected for direct relevance. Metadata and venue formatting must receive a final DOI and bibliographic audit before submission.

[1] R. van der Linden, R. Lopes, and R. Bidarra, “Procedural Generation of Dungeons,” *IEEE Transactions on Computational Intelligence and AI in Games*, vol. 6, no. 1, pp. 78-89, 2014. https://doi.org/10.1109/TCIAIG.2013.2290371

[2] J. Gutierrez and J. Schrum, “Generative Adversarial Network Rooms in Generative Graph Grammar Dungeons for The Legend of Zelda,” 2020. https://arxiv.org/abs/2001.05065

[3] A. Summerville, S. Snodgrass, M. Guzdial, C. Holmgård, A. K. Hoover, A. Isaksen, A. Nealen, and J. Togelius, “Procedural Content Generation via Machine Learning (PCGML),” *IEEE Transactions on Games*, 2018. https://arxiv.org/abs/1702.00539

[4] S. Dai, X. Zhu, N. Li, T. Dai, and Z. Wang, “Procedural Level Generation with Diffusion Models from a Single Example,” *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 38, no. 9, pp. 10021-10029, 2024. https://doi.org/10.1609/aaai.v38i9.28865

[5] J. Schrum, O. Kilday, E. Salas, B. Hagan, and R. Williams, “Text-to-Level Diffusion Models With Various Text Encoders for Super Mario Bros,” 2025. https://arxiv.org/abs/2507.00184

[6] M. Villanueva Aylagas, J. Bergdahl, J. Gillberg, A. Sestini, T. Tolstoy, and L. Gisslén, “Improving Conditional Level Generation Using Automated Validation in Match-3 Games,” *IEEE Transactions on Games*, 2024. https://doi.org/10.1109/TG.2024.3440214

[7] A. Alvarez, S. Dahlskog, J. Font, and J. Togelius, “Empowering Quality Diversity in Dungeon Design with Interactive Constrained MAP-Elites,” 2019. https://arxiv.org/abs/1906.05175

[8] A. Sarkar and S. Cooper, “Generating and Blending Game Levels via Quality-Diversity in the Latent Space of a Variational Autoencoder,” 2021. https://arxiv.org/abs/2102.12463

[9] A. Khalifa, R. Gallotta, M. Barthet, A. Liapis, J. Togelius, and G. N. Yannakakis, “The Procedural Content Generation Benchmark: An Open-source Testbed for Generative Challenges in Games,” 2025. https://arxiv.org/abs/2503.21474

[10] C. Geng, L. Zhang, M. Zhang, H. Ye, Z. Zhao, and X. Si, “Neural Proposals, Symbolic Guarantees: Neuro-Symbolic Graph Generation with Hard Constraints,” 2026. https://arxiv.org/abs/2602.16954

[11] A. van den Oord, O. Vinyals, and K. Kavukcuoglu, “Neural Discrete Representation Learning,” *Advances in Neural Information Processing Systems*, 2017. https://arxiv.org/abs/1711.00937

[12] J. Ho, A. Jain, and P. Abbeel, “Denoising Diffusion Probabilistic Models,” *Advances in Neural Information Processing Systems*, 2020. https://arxiv.org/abs/2006.11239

[13] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-Resolution Image Synthesis with Latent Diffusion Models,” *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2022. https://doi.org/10.1109/CVPR52688.2022.01042

[14] T. Karras, M. Aittala, T. Aila, and S. Laine, “Elucidating the Design Space of Diffusion-Based Generative Models,” *Advances in Neural Information Processing Systems*, 2022. https://arxiv.org/abs/2206.00364

[15] Z. Hang, S. Gu, C. Li, J. Bao, D. Chen, H. Hu, X. Geng, and B. Guo, “Efficient Diffusion Training via Min-SNR Weighting Strategy,” 2023. https://arxiv.org/abs/2303.09556

[16] L. Rampášek, M. Galkin, V. Dwivedi, A. Luu, G. Wolf, and D. Beaini, “Recipe for a General, Powerful, Scalable Graph Transformer,” *Advances in Neural Information Processing Systems*, 2022. https://arxiv.org/abs/2205.12454

[17] V. Karth and A. M. Smith, “WaveFunctionCollapse is Constraint Solving in the Wild,” 2021. https://arxiv.org/abs/2106.06033

[18] A. J. Summerville, S. Snodgrass, M. Mateas, and S. Ontañón, “The Video Game Level Corpus,” 2016. https://arxiv.org/abs/1606.07487

---

## 26. Final decision record

The repository is potentially publishable, but the publishable unit is smaller than the repository.

Proceed with one central paper on:

> mission-graph-conditioned latent diffusion, differentiable local/global logic supervision, and transparent independent verification/repair.

Treat every other major branch as excluded, supplementary, or a future paper until fresh evidence justifies its inclusion.

The first implementation priority is not manuscript prose. It is a clean publication experiment path that separately trains the causal variants, preserves raw outputs, evaluates them with an independent solver, and records enough provenance to regenerate every reported number.

---

## 27. Architecture-cleanup implementation record

This section records code state, not experimental evidence. It must not be cited as a result.

### 27.1 Architecture now enforced

- The confirmatory model matrix uses four separately trained conditions: local context (`N`), graph conditioning (`G`), graph plus local LogicNet supervision (`G-L`), and graph plus local/global LogicNet supervision (`G-LG`).
- The local baseline bypasses the global graph encoder during both training and inference. Dormant graph, style, fusion, and reference-encoder parameters are frozen; only the local encoder remains trainable.
- The paper variants disable discrete reference-room-map conditioning. Neighbor latents already provide local context, so enabling a second reference encoder only for graph models would confound the `N` versus `G` comparison.
- Conditioning schema failures are fail-closed for paper runs. Zero-conditioning fallback exists only when non-strict legacy compatibility is selected explicitly.
- Graph-node identifiers retain the representation owned by the NetworkX graph, preventing integer room IDs from being silently converted to unmatched strings during constrained sampling.
- Per-room puzzle-structure metadata overrides the pipeline-wide default instead of being discarded when the runtime graph context is built.
- Style-token inputs are normalized to the model device and validated as a scalar or one-dimensional batch before embedding.
- Resource locks in LogicNet are limited to key, boss-key, and item locks. Structural bomb and switch gates are not treated as consumable resources without a matching provider/unlock model.
- Mission-label parsing is case-insensitive and accepts both `type` and `node_type`, including key and item roles required by the exact validator.
- Raw, constrained, repaired, and compiled result stages have separate record fields. A later stage cannot omit its predecessors.

### 27.2 Publication execution path now present

- `src/publication/protocol.py` defines grouped leave-one-dungeon-identity-out folds, independent training seeds, causal model variants, and a hashed protocol manifest.
- `experiments/paper_v1/run_protocol.py` materializes fresh VQ-VAE and diffusion configurations without starting training. It rejects non-empty destinations and disables resume/warm-start routes.
- `src/publication/records.py` defines stage-aware, hashable result records and exact intervention-size metrics.
- With nine outer folds, five training seeds, and four trained model variants, the default plan contains 45 fold-specific tokenizer jobs and 180 diffusion jobs.

### 27.3 Verification completed in this cleanup pass

- Python compilation passed for `src`, `scripts`, `experiments`, and the main entry point.
- Architecture-focused Ruff checks passed for syntax, undefined names, unused imports, invalid f-strings, compound statements, and lambda-assignment defects.
- The focused publication/configuration/conditioning rerun passed 73 tests.
- The broader architecture tests run during cleanup passed in separate groups covering configuration, conditioning, diffusion training, LogicNet, exact validation, graph grammar, WFC, and pipeline integration.
- The complete repository suite was not allowed to run to completion because of its runtime; therefore this document does not claim a clean full-suite result.
- Remaining environment warnings concern PyTorch Geometric deprecation and an installed `requests` dependency-version mismatch. They are not treated as model evidence.

### 27.4 Work still required before a submission claim

- Execute the fresh training matrix; existing checkpoints and artifacts remain excluded from confirmatory evidence.
- Connect the successful canonical generation-to-record bridge to the batch experiment scheduler, including attempted-record creation when generation itself raises or times out.
- Run evaluator-integrity tests on known solvable, known unsolvable, resource-gated, switch-gated, and adversarial maps before scoring models.
- Run the preregistered confirmatory experiments, statistical analysis, duplicate/near-duplicate audit, efficiency measurements, and failure taxonomy.
- Re-run the full repository test suite in an environment where its long runtime can complete, and resolve dependency warnings in the publication environment lock.
- Recheck venue rules and every bibliography entry against current primary sources immediately before submission.

### 27.5 Search-algorithm audit and corrections

The search stack was audited as a transition system rather than by solver name alone. The canonical A*, BFS, Dijkstra, and greedy wrappers share `StateSpaceAStar`, so they now use the same complete `GameState`, pure tile transition function, graph-gate effects, and path replay logic. A* remains the hard-validity oracle; P-CBS remains a behavioral model and must not replace it.

Corrections made during this audit:

- Directed graph frontends now expand outgoing edges only. Reverse edges are used only by reverse goal-distance computations. Graph-only and macro search can no longer report a path by traversing an ordinary directed edge backward.
- Opened-edge identity distinguishes genuine antiparallel directed locks. Converter-generated mirrored arcs for one bidirectional physical connection still share an opened state, so returning through the same opened door does not consume a second resource.
- Virtual-node and transition-room routes preserve the complete graph-edge path. Every lock, bomb wall, item gate, hazard, and other edge along the abstract route is applied sequentially; multi-edge routes are no longer collapsed to one “most restrictive” label.
- P-CBS candidate enumeration is pure. Previewing a candidate no longer mutates the live inventory, collects an item, or opens a gate before the candidate is selected and executed.
- Graph-only validation tracks activated switch IDs and named traversal/protection items. A switch edge without a declared controller fails closed.
- Tile-state and room-level search also fail closed on abstract switch state they do not encode. Ordinary tile puzzle progression and the mission-graph progression oracle remain the authoritative paths for proving those gates.
- Shared directed-successor, reverse-predecessor, and opened-edge identity helpers replaced duplicated edge logic.

Regression coverage includes genuine antiparallel locks, converter-mirrored locks, backward traversal of directed edges, multi-lock virtual paths, P-CBS preview purity, and declared switch activation. The widened search/validation run passed 185 tests, and the independent critical-review file passed 59 tests, for 244 passing tests in these two non-overlapping runs. Focused Ruff checks passed for every touched search/validation module and regression file. These are implementation checks, not experimental paper results.

### 27.6 Continuation audit: provenance, stages, and failure semantics

The next cleanup pass checked contracts that success-path tests do not normally expose.

- The protocol digest now seals the complete manifest, including base-configuration provenance, output location, run counts, and job definitions. An explicit empty fold set is rejected, duplicate fold IDs are rejected, model IDs cannot escape their output directory, and generated PowerShell commands quote apostrophes safely.
- Publication result constructors now validate schema and SHA-256 fields directly. A completed run must contain a real raw grid; any retained grid requires its shape and digest; contradictory validation statuses are rejected.
- Latent generation snapshots `raw_neural_grid` before topology door clamps or semantic logit biases mutate decoder logits. Publication result schema 2.0 separates four meanings: pre-constraint neural proposal (`raw`), deterministic room constraint output (`constrained`), explicit room repair output (`repaired`), and final graph/entity materialization (`compiled`). Raw, constrained, and repaired rooms use the same fixed slots, boundary sealing, and physical-edge carving.
- Hard-oracle timeout, exhaustion, invalid-map, and validator-error outcomes use explicit termination metadata rather than English-message matching when that metadata is available. An indeterminate hard-oracle result is rejected with the same penalty as other oracle failures and P-CBS is not run against an uncertified reference.
- A*, BFS, Dijkstra, greedy search, DFS/IDDFS, D* Lite, bidirectional A*, and learned A* wrappers now retain machine-readable termination status. DFS depth exhaustion is conservatively reported as incomplete rather than as proof that no path exists.
- D* Lite and its A* safety fallback now share one configured search budget. The fallback no longer receives a new full budget after D* Lite has already spent part or all of the original allowance, and combined work is returned in the accounting.

Verification for this continuation was split into focused, partly overlapping groups so a misleading grand total is not reported: 63 architecture regressions; 63 search, multi-goal, benchmark, and topology tests; 25 publication/search-contract tests; 103 advanced pathfinding and generation-to-validation contract tests; and 56 neural-pipeline plus unified Zelda-mechanics tests. Every group passed. Focused Ruff checks also passed. The same PyTorch Geometric deprecation and local `requests` dependency warnings remain environment issues, not model results.

Still deliberately unverified or excluded:

- No existing artifact or checkpoint was evaluated, as requested; no model-quality claim follows from these code tests.
- The fresh confirmatory training matrix and paired sampling runs have not been executed.
- The publication plan materializes training configurations and provenance, and successful canonical outputs can now be converted to independently validated `PublicationResultRecord` files. The remaining orchestration gap is a batch scheduler that records generation-level exceptions/timeouts and regenerates all tables from attempted records.
- Multiprocessing `parallel_astar`, GUI-only solver routes, and benchmark harnesses that are not independent solvers remain outside the canonical validity path. They must not be presented as oracle corroboration.
- A single uninterrupted full-repository test run is still outstanding. The focused suites cover the publication-critical architecture but do not prove every peripheral module clean.

### 27.7 Search proof semantics correction

A further review found that the earlier status cleanup was still too coarse. It sometimes inferred “no path” or “budget exhausted” from a state counter or an English error message, and the benchmark oracle could start a Dijkstra run after A* had already emptied its complete frontier. Those behaviors obscure what the computation actually proves.

The corrected contract follows the distinction required by the underlying algorithms:

- A result is a **solution witness** only when search reaches the goal and the reconstructed route replays under the canonical transition function with the same accumulated cost.
- A result is a **no-path proof** only when a complete finite-state search empties its live frontier without reaching the goal.
- A state-budget limit, depth limit, comparison-search abort, route-replay failure, unknown status, or validator exception is **inconclusive**. It must never set `proven_unsolvable=True`.
- Invalid input is a separate outcome and is not a no-path proof.
- Human-readable failure text is diagnostic only. Status classification now uses structured termination metadata and does not parse phrases such as “timeout” or “no path.”

This separation is consistent with the completeness/admissibility conditions in Hart, Nilsson, and Raphael’s A* formulation, Korf’s depth-first iterative-deepening analysis, and Koenig and Likhachev’s definition of D* Lite as an incremental replanning algorithm for related search tasks:

- P. E. Hart, N. J. Nilsson, and B. Raphael, “A Formal Basis for the Heuristic Determination of Minimum Cost Paths,” 1968. https://doi.org/10.1109/TSSC.1968.300136
- R. E. Korf, “Depth-First Iterative-Deepening: An Optimal Admissible Tree Search,” 1985. https://doi.org/10.1016/0004-3702(85)90084-0
- S. Koenig and M. Likhachev, “D* Lite,” AAAI 2002. https://publications.ri.cmu.edu/d-lite

Implementation consequences:

- `StateSpaceAStar` now exposes its actual structured stopping condition from both diagnostic and compatibility APIs.
- Successful A*, BFS, Dijkstra, greedy, and learned-A* wrapper results pass through one shared route-and-cost verification method.
- The publication A* oracle no longer performs a redundant Dijkstra search after a definitive frontier-empty result. It reports `proof_status`, `proven_unsolvable`, `indeterminate`, termination status, and route-replay evidence separately.
- The exact validator preserves the solver’s real terminal status and grants `proven_unsolvable` only for complete frontier exhaustion. Optional Dijkstra recovery is not attempted after a proof, invalid input, or resource cutoff.
- IDDFS now implements success/cutoff/failure semantics. A per-iteration full-state transposition table prevents exponential duplicate expansion without restoring the previous unsafe global pruning. Iterative DFS checks the goal before applying the depth cutoff, so a goal exactly at the configured limit is accepted.
- D* Lite and bidirectional wrappers conservatively return `incomplete` when they lack a retained complete-frontier certificate, instead of manufacturing a no-path proof from an expansion count.
- P-CBS is run only after the hard oracle supplies a verified solution witness. Oracle no-path proofs and indeterminate oracle runs receive different outcome classes and cannot produce cognitive-comparison metrics.

The widened regression run for this correction passed 221 search, pathfinding, Zelda-mechanics, hard-validation, pipeline-contract, and P-CBS tests. Focused Ruff checks passed. This verifies software contracts only; it is not model-performance evidence.

---

## 28. Full pipeline and experimental-design audit (current state)

This is the current source-of-truth checklist for the publication path. It distinguishes code evidence from experimental evidence. No existing checkpoint or artifact was inspected, and no model-performance claim is made here.

### 28.1 End-to-end publication path

```text
locked corpus + manifest
    -> leave-one-dungeon-identity-out outer fold
    -> grouped internal validation using training identities only
    -> one fresh fold/seed-specific VQ-VAE
    -> four fresh causal diffusion variants (N, G, G-L, G-LG)
    -> primary unguided sampling for all four models on fixed graphs/seeds
    -> secondary paired guidance study only for LogicNet-capable models
    -> canonical room-wise dungeon generation
    -> raw / constrained / repaired neutral assemblies
    -> compiled graph-and-entity dungeon
    -> independent exact validation of every retained stage
    -> immutable attempt record, including errors and timeouts
    -> within-fold aggregation, then inference across held-out identities
    -> tables, figures, claims, and manuscript
```

The canonical generation entry point is `src.pipeline.stitching.dungeon_assembler.generate_dungeon`, reached through `NeuralSymbolicDungeonPipeline.generate_dungeon`. GUI adapters, `AdvancedNeuralSymbolicPipeline`, benchmark wrappers, P-CBS, and multiprocessing search are outside the confirmatory path.

### 28.2 Audit checklist and disposition

| Area | What was checked | Current disposition |
|---|---|---|
| Corpus identity | Manifest paths, byte sizes, per-file SHA-256 values, unexpected/missing files, and a path-order-independent corpus digest | **Verified and enforced.** The currently locked 93-file corpus verifies with content digest `fb0e10e726a6f89108ec51111ed3d4063cf693e8699aa2b1cf00fccfda5457bc`. |
| Outer split | Both quest variants of one dungeon identity are held out together; training and test IDs are disjoint and cover all nine identities | **Verified and tested.** Nine leave-one-dungeon-out folds are materialized. |
| Internal validation | Room samples are grouped by dungeon identity rather than split independently | **Fixed and tested.** All training entry points now fail closed if a requested internal validation split lacks complete dungeon grouping metadata. |
| Tokenizer isolation | VQ-VAE training sees only the outer training identities and is shared only by model variants in the same fold and seed | **Verified in the materialized run plan.** This is a matched control, not cross-fold checkpoint reuse. |
| Run isolation | Resume, auto-resume, and diffusion warm-start routes are disabled; output directories are fold/model/seed specific | **Verified and tested.** Existing artifacts and checkpoints are excluded by the sealed protocol. |
| Reproducibility | Python, NumPy, PyTorch, CUDA RNGs, DataLoader workers, cuDNN flags, and deterministic PyTorch algorithms | **Fixed and tested.** Publication configs use `cudnn_benchmark=false`, `cudnn_deterministic=true`, and `deterministic_algorithms=true`; these values now reach every training entry point and the run snapshot. |
| Causal model matrix | N vs G changes graph context; G vs G-L changes local logic; G-L vs G-LG changes global logic | **Fixed.** Tile auxiliary loss is off in all four primary variants, avoiding an unreported second factor. Reference-room maps, MAP-Elites, puzzle-stage branches, training guidance, and oracle room-grid supervision are also off. |
| Evaluation selection | Whether invalid or indeterminate outputs are returned or rejected before recording | **Fixed.** Publication generation uses `end_to_end_validation_mode=report`; failures cannot disappear through rejection sampling. |
| Canonical generation | Mission graph preparation, room generation, stitching, graph compilation, entity materialization, exact evaluation, and returned provenance | **Audited.** The canonical result now retains the physical mission graph used for stitching. The alternative advanced pipeline is not a publication record source. |
| Stage attribution | Whether raw, constrained, repair, stitching, and graph compilation are separable | **Fixed with schema 2.0.** Raw, constrained, and repaired rooms receive identical fixed-slot assembly and edge carving; compilation is a fourth stage. This removes assembly and entity materialization from the estimated repair effect. |
| Stage retention | Whether each evaluated grid is durable and identifiable | **Implemented for successful generation.** Each stage is written as a non-pickled NumPy array with shape and SHA-256 metadata before independent validation. |
| Validator semantics | Solution witness, finite-frontier no-path proof, cutoff, invalid input, and internal failure | **Audited and corrected.** Only verified route replay proves success; only complete frontier exhaustion proves no path; cutoffs remain inconclusive. P-CBS is secondary and cannot certify validity. |
| Search algorithms | Directionality, antiparallel locks, mirrored physical edges, virtual multi-edge transitions, item/switch state, candidate preview purity, and budget accounting | **Corrected with regression coverage.** The detailed record is in Sections 27.5 and 27.7. |
| Attempt denominator | Generation errors, solver timeouts, validator failures, and missing stages | **Fixed.** The all-attempt exact-validity rate retains them in the denominator; an evaluable-only rate may be shown only as a clearly labeled diagnostic. |
| Pairing | Matching across models and guidance conditions | **Implemented in analysis contracts.** Model comparisons enforce exact keys `(outer fold, training seed, sampling seed, mission graph, inference condition)` and reject missing pairs. Guidance comparisons pair conditions within one LogicNet-capable model using the analogous key without inference condition. The remaining gap is the batch attempt executor. |
| Statistical independence | Whether repeated seeds and samples are treated as independent datasets | **Design fixed; final model still to execute.** Repeats are averaged within each outer held-out identity before the included exact sign-permutation and outer-fold bootstrap analysis. A preregistered hierarchical model may supplement this, but may not turn room/sample counts into the confirmatory sample size. |
| Claims | Whether tests or existing artifacts are presented as model evidence | **Enforced in this document.** Current evidence supports implementation contracts only. |

### 28.3 Defects found and corrected in this audit

1. Runtime determinism options existed in YAML but were discarded by several trainer bridges. They now propagate to VQ-VAE, diffusion, masked-room, and consistency-model training.
2. Strict PyTorch deterministic algorithms were absent. The runtime contract now enables them for publication runs and records the setting.
3. The dataset manifest was not verified when the protocol was materialized. Materialization now fails on missing, unexpected, resized, or content-modified files and records a stable corpus digest.
4. Internal validation could silently fall back to a room-level random split when metadata was malformed. Actual training entry points now require dungeon grouping metadata.
5. G-L and G-LG enabled tile auxiliary supervision while claiming to isolate local and global logic. Tile auxiliary supervision is now off in the confirmatory matrix.
6. Publication configs inherited reject-mode end-to-end validation. They now use report mode so invalid and inconclusive attempts remain observable.
7. The result schema had no reliable bridge from canonical generation. Successful canonical results can now be converted into hashed, stage-aware records with independent validator outcomes.
8. Raw and constrained publication canvases were built without the connection carving applied to the final dungeon. This would make full-dungeon raw validity structurally incomparable and often trivially fail. All pre-compilation stages now use the same neutral assembly.
9. The former `repaired` publication stage used the final compiled dungeon, conflating room repair with graph enforcement and entity placement. Schema 2.0 separates `repaired` from `compiled`.
10. Aggregation could encourage success-only reporting. The status-aware aggregator reports both the conservative all-attempt estimand and the diagnostic evaluable-only estimand.
11. Seed-level comparisons risked pseudoreplication. The paired analysis now aggregates training/sampling repeats inside each independent outer dungeon before permutation testing and bootstrapping.
12. `best_model.pth` was selected with `val_total_loss`, whose meaning changed when a variant enabled logic loss, and used solvability as a tie-break. Publication configs now select every arm by `val_diffusion_loss`; exact ties retain the earlier checkpoint. The separate `best_logic_model.pth` remains diagnostic only.
13. Non-finite losses or gradients could silently skip different numbers of updates in different arms. General runs retain the compatibility `skip` policy, while publication configs use `nonfinite_policy=error` and abort the affected run.
14. Neural-guided repair was available only when a calibrated LogicNet was loaded. Publication configs now disable both neural-guided repair and neural repair feedback, leaving the same symbolic repair operator in every model arm.
15. The original inference-factor description crossed logic guidance with N and G, although those models contain no LogicNet and runtime correctly reduces the requested scale to zero. The primary four-model ablation is now unguided. A secondary paired guidance experiment is allowed only for G-L and G-LG after one nonzero scale is selected using outer-training identities and sealed separately.
16. Resolved model configs were not mechanically checked for accidental arm-specific drift. The run planner now rejects every cross-arm difference outside the declared graph/logic factors and run paths, and records a SHA-256 digest for each planned config.
17. Loss logs exposed raw losses but not their effective weighted contributions. Training now records weighted visual, local/global logic, tile, WFC-pseudo, and puzzle-stage contributions plus the logic fraction of the absolute combined objective.
18. Canonical dungeon generation seeded PyTorch but did not explicitly reset Python, NumPy, and all CUDA generators at the attempt boundary. It now resets all four when a sampling seed is supplied.
19. Validation generated extra stochastic solvability samples only for LogicNet-enabled arms. Those diagnostics advanced the main process RNG and could change the following training epochs, while other distributed ranks did not advance equivalently. Validation now runs inside a Python/NumPy/PyTorch/CUDA RNG-preservation guard. Each validation-loss batch is also reseeded from `(training seed, batch index)`, so logic-only diagnostics in one batch cannot shift the checkpoint-metric noise of later batches. Thus checkpoint evaluation reuses a matched held-out noise/timestep stream across epochs and arms without perturbing later optimization draws.

### 28.4 Statistical estimands and reporting rules

The primary observational unit is one requested complete-dungeon attempt. The confirmatory inference unit is the held-out dungeon identity. Training seeds, sampling seeds, room nodes, and repeated mission graphs improve measurement precision but do not create additional independently sampled source dungeons.

For each model and inference condition, report:

- requested, completed, invalid, exact-valid, timed-out, validator-error, and generation-error counts;
- exact-validity rate over all requested attempts, where non-success outcomes contribute zero;
- the evaluable-only validity rate only as a labeled sensitivity diagnostic;
- per-fold rates and the mean paired fold difference;
- a confidence interval obtained by resampling outer folds, not individual rooms;
- an exact paired random-sign test across fold differences when the nine-fold design is used;
- effect size with an explicit undefined status when fold-difference variance is zero;
- repair/constraint transition rates and changed-tile count/fraction for `raw -> constrained`, `constrained -> repaired`, `repaired -> compiled`, and `raw -> compiled`;
- wall-clock time and search work, with the hardware/software environment and timeouts reported.

The fold-level nonparametric analysis is conservative and transparent for nine source identities. If a mixed-effects logistic analysis is added, the hierarchy must reflect outer dungeon, training seed, mission graph, and sampling repeat. Its assumptions and convergence diagnostics must be reported; it is not a license to call every room independent.

### 28.5 Research basis for the design corrections

- Group separation follows the purpose of grouped cross-validation: samples from the same group must not appear in both train and test partitions. See the [scikit-learn GroupKFold documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html) and Kaufman et al., [Leakage in Data Mining](https://doi.org/10.1145/2382577.2382579).
- Determinism follows the official [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html), [deterministic-algorithm API](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html), and [DataLoader worker-randomness guidance](https://docs.pytorch.org/docs/stable/data.html).
- Repeated random seeds measure an implementation's variability but do not replace independent datasets; see Bouthillier et al., [Accounting for Variance in Machine Learning Benchmarks](https://proceedings.mlr.press/v97/bouthillier19a.html).
- Seed-dependent initialization and data order can materially change selected models, especially on small datasets; see Dodge et al., [Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping](https://arxiv.org/abs/2002.06305). This supports matched seed schedules, explicit checkpoint rules, and reporting distributions rather than best-seed scores.
- Reimers and Gurevych, [Reporting Score Distributions Makes a Difference](https://aclanthology.org/D17-1035/), directly motivates reporting the distribution across independent training runs instead of one selected score.
- Pineau et al., [Improving Reproducibility in Machine Learning Research](https://www.jmlr.org/papers/v22/20-303.html), motivates the sealed configuration, data/code provenance, and explicit reporting checklist used here.
- Fold-level paired nonparametric comparisons follow the small-sample comparison principles discussed by Demsar, [Statistical Comparisons of Classifiers over Multiple Data Sets](https://www.jmlr.org/papers/v7/demsar06a.html). The exact paired randomization implementation is consistent with the [SciPy permutation-test definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html).
- If room/sample hierarchy is analyzed rather than first collapsed by fold, uncertainty must preserve the nested sampling structure; see Saravanan et al., [The Hierarchical Bootstrap](https://arxiv.org/abs/2007.07797).

The attached *English for Writing Research Papers* was treated only as writing guidance. Its reader-centered principles - clarity, logical organization, concision, and unambiguous attribution - informed the wording of this blueprint; it is not evidence for the model or experiment design.

### 28.6 Verification performed after the corrections

- The focused publication, configuration, and critical-review run passed **135 tests**.
- The stitching, neural-pipeline, reliability, reporting, and grouped-split run passed **121 tests**.
- These runs overlap, so their counts must not be added and presented as one suite total.
- Focused Ruff checks passed on all files modified by this audit.
- The dataset verifier passed on the locked 93-file corpus.
- Expected environment warnings remain: a PyTorch Geometric deprecation, a local `requests` dependency-version warning, and PyTorch Geometric/Python 3.13 typing deprecations. They are environment debt, not experimental evidence.

### 28.7 Remaining gates before a paper claim

These are the next tasks in dependency order:

1. Add the batch attempt executor that catches generation-level errors/timeouts, always writes an attempted `PublicationResultRecord`, and atomically appends it to a run ledger.
2. Freeze the mission-graph test bank and paired sampling seeds without looking at model outcomes.
3. Freeze the nonzero logic-guidance scale using training identities only; then implement the guidance-condition paired analysis.
4. Complete evaluator-integrity fixtures for solvable, frontier-exhausted, resource-gated, switch-gated, malformed, and budget-cutoff dungeons, including an independent comparator on the supported subset.
5. Implement or faithfully adapt the required external/symbolic baselines under the same folds, attempt budget, wall-clock accounting, and exact validator.
6. Execute fresh training and sampling. Do not reuse current checkpoints or results.
7. Generate all tables from immutable records, run the near-duplicate audit, and perform the frozen statistical analysis.
8. Run one uninterrupted full repository test suite in the final locked environment and record exact dependency, driver, CUDA, and hardware versions.
9. Write the paper from generated evidence, then perform claim-to-record, citation, grammar, figure, and venue-compliance audits.

Until these gates are complete, the defensible statement is: **the repository has a substantially cleaner and test-backed publication architecture, but it does not yet contain fresh confirmatory model evidence.**

### 28.8 Fair-ablation contract after the continuation audit

The confirmatory comparison is now explicitly asymmetric where the architecture is asymmetric:

| Comparison | Changed causal factor | Required inference condition | Forbidden interpretation |
|---|---|---|---|
| N vs G | Add global mission-graph conditioning | Unguided | Do not attribute a result to LogicNet or repair. |
| G vs G-L | Add differentiable local reachability supervision | Unguided | Do not enable tile auxiliary loss or neural-guided repair. |
| G-L vs G-LG | Add global mission-logic supervision | Unguided | Report weighted logic contribution; a larger composite logic magnitude is not automatically an isolated semantic effect. |
| G-L unguided vs guided | Add inference-time LogicNet gradient guidance | Same checkpoint, graph, and sampling seed | Do not compare this condition against N/G as if they received equivalent guidance. |
| G-LG unguided vs guided | Add inference-time LogicNet gradient guidance | Same checkpoint, graph, and sampling seed | Do not select the guidance scale on held-out outcomes. |

Every primary arm must share the outer training identities, internal grouped validation identities, fresh fold/seed tokenizer, training batches and order, epoch count, optimizer defaults, common checkpoint metric, failure policy, sampling graphs/seeds, diffusion steps, classifier-free guidance, symbolic repair configuration, exact validator, timeout budget, and all-attempt denominator. Resolved-config parity is checked automatically. Architecture-dependent parameter counts and optimizer groups are expected differences and must be reported per arm rather than falsely described as equal compute.

Teacher-forced neighbor latents remain enabled during training. Canonical batched generation proceeds in topological layers and feeds already generated predecessor latents into later rooms, so the inference path does use neighbor context. The remaining train-inference difference is exposure bias from real versus generated neighbor rooms. It affects all four arms under the same protocol and should be acknowledged as a limitation; disabling it would redefine the local baseline rather than merely remove a hidden confound.

The current WFC, DCGAN, and flattened-token language-model scripts are room-level diagnostics only. They do not yet implement the grouped outer folds, mission-graph conditioning, complete-dungeon attempt records, neutral stage assembly, or matched exact validation required for a primary external-baseline claim. Their README now states this boundary, and WFC collapse failures remain in the attempted denominator as invalid fallbacks rather than being excluded.

Verification after this continuation: the combined publication/configuration/diffusion focused run passed **143 tests**, followed by a publication-only four-arm materialization rerun of **21 tests** after the parity, inference-factor, and synchronized non-finite changes. A final focused rerun including RNG isolation is required after the current long repository-wide run; counts will be updated only from completed commands. These runs overlap and must not be added. They verify code contracts, not model quality. One uninterrupted repository-wide test run, a sealed sampling manifest with a training-only-selected guidance scale, evaluator-integrity fixtures, the batch attempt ledger, and fresh model execution remain required.
