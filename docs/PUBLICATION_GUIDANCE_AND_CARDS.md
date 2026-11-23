# Publication Guidance And Cards

Last updated: 2026-11-12.

This is the canonical pre-paper contract for this repository. It defines what
the project should claim, what it must not claim, which artifacts must exist,
and how evidence must be packaged before a paper draft is written.

The most publishable direction is not "best diffusion model" or "generic
dungeon generator." The strongest direction is:

> **G-LMDM (`Graph-Conditioned Logic-Regularized Masked Discrete Model`):** A repair-aware, graph-conditioned discrete neuro-symbolic PCG framework for Zelda-like dungeons that explicitly separates mission-graph controllability, discrete masked room synthesis (`3.58M` train / `3.36M` inference) vs. continuous `VQ-VAE` comparators (`47.13M` `UNet[1,2]`, `22.44M` `DiT+Flow`), direct `16x11` Soft Bellman-Ford `LogicNet` regularization, attributed 4-stage symbolic repair, formal `StateSpaceAStar` proofs, and bounded `P-CBS` diagnostics.

This direction is aligned with PCGML's emphasis on learned content modeling and
repair/critique uses, PCG Benchmark's quality/diversity/controllability axes,
WFC's role as constraint solving, graph-conditioned layout generation, and
procedural personas as synthetic playtesters.

## 1. Scientific Framework: Problem Statements, Objectives, Research Questions, and Hypotheses

The operational protocol spans two coupled tracks documented across
[PUBLICATION_BLUEPRINT.md](../PUBLICATION_BLUEPRINT.md) and
[EXPERIMENT_PLAN.md](../refine-logs/EXPERIMENT_PLAN.md):

1. **System-level Research Question (Contract Boundary):**

> Can G-LMDM (`Graph-Conditioned Logic-Regularized Masked Discrete Model`) synthesize Zelda-like dungeons from unseen mission graphs while explicitly attributing raw neural validity, symbolic repair burden, formal `StateSpaceAStar` solvability proofs, and bounded `P-CBS` diagnostics?

2. **Causal Confirmatory Research Question (`G = (V, E, X_V, X_E) -> D`):**

> On held-out Dungeon-9 mission graphs, do causal Block III `GraphGPS+RRWP` conditioning, absorbing-state discrete masked synthesis on `16x11` tile tokens, and direct `16x11` Soft Bellman-Ford `LogicNet` supervision jointly increase pre-repair oracle solvability and reduce symbolic repair burden (`repair_rate`, `tiles_repaired_mean`) relative to unconditioned, continuous `VQ-VAE` latent diffusion/flow, and external symbolic baselines at matched diversity and lower parameter cost?

### 1.0 Literature Research Gaps (`RG1–RG4` / `FG1–FG4`)

Independent external literature synthesis establishes four field research gaps mapped 1-to-1 through [`PS1–PS4`](#11-problem-statements-ps1ps4), [`RO1–RO4`](#12-research-objectives-ro1ro4), [`RQ1–RQ4`](#13-research-questions-rq1rq4), and [`H1–H4`](#14-hypotheses-h1h4) (canonical machine-readable schema in [`publication_guidance_card.json`](./publication_guidance_card.json)):

- **`RG1` (`FG1`: Macro-to-Micro Semantic-Geometric Disconnect → `PS1`, `RO1`, `RQ1`, `H1`):** Two-stage hybrid generators such as *Graph-GAN* (Gutierrez & Schrum 2020, `arXiv:2001.05065`) and mission-space grammars (Dormans 2010; Lavender & Thompson 2016) populate graph nodes with unconditional or PCA-synthesized rooms and carve doors post-hoc, omitting directional boundary constraints and global random-walk graph structure (*GRIT / RRWP*, Ma et al. 2023, `arXiv:2305.17589`).
- **`RG2` (`FG2`: Continuous-Latent Bottleneck & Rarity Erasure on Small Categorical Grids → `PS2`, `RO2`, `RQ2`, `H2`):** On small VGLC tile corpora (`349` Dungeon 1–8 training rooms, `217` unique `16x11` grids, `43` unique `12x7` structural interiors; `88.60%` `WALL+FLOOR` vs. `0.68%` `ENEMY` and `0.05%` `STAIR`), continuous `VQ-VAE` (`arXiv:1711.00937`) latent diffusion (`47.13M` `UNet[1,2]`, `22.44M` `DiT+Flow`) adds tokenization distortion, parameter bloat, and low raw playability (`12–27%`, Schrum et al. 2025, `arXiv:2507.00184`), whereas absorbing-state discrete masked synthesis (*D3PM* `arXiv:2107.03006`, *MaskGIT* `arXiv:2202.04200`, *LayoutDM* `arXiv:2303.08137`, *MDLM* `arXiv:2406.07524`) operates natively on `16x11` discrete tokens (`3.58M` train / `3.36M` inference) with exact $t=0$ boundary clamping.
- **`RG3` (`FG3`: Traversability-Blind Likelihood Training & Unattributed Post-Hoc Repair → `PS3`, `RO3`, `RQ3`, `H3`):** Cross-entropy treats single-tile corridor cut-sets (`1/176 = 0.57%`) identically to interior walls (`24.3%` pre-repair GAN validity before MILP repair, Zhang et al. 2020, `arXiv:2010.06627`; Cooper 2022/2023). Differentiable planning and semantic losses (*VIN* Tamar et al. 2016 `arXiv:1602.02867`; Xu et al. 2018 `arXiv:1711.11157`) reduce repair edits only when coupled directly onto `16x11` tile logits and paired with 4-stage `raw -> constrained -> repaired -> compiled` attribution (`repair_rate`, `tiles_repaired_mean`).
- **`RG4` (`FG4`: Dungeon-Split Leakage, External Baseline Deficit & Proxy-vs-Oracle Conflation → `PS4`, `RO4`, `RQ4`, `H4`):** Room-level splits leak dungeon style across `9` VGLC dungeons (Summerville et al. 2016), internal-only ablations omit external baselines (`Random`, `WFC`, `Grammar-only`) (Withington et al. 2024, `arXiv:2404.18657`; Khalifa et al. 2025, `arXiv:2503.21474`), dependent multi-fold averages yield pseudoreplicated intervals, and bounded-agent personas (`P-CBS`, Holmgård et al. 2018) cannot substitute for exact `StateSpaceAStar` solvability proofs.

### 1.1 Problem Statements (`PS1–PS4`)

- **`PS1` (`FG1`) — Causal Multi-Scale Conditioning Deficit:**
  Unconditioned or neighbor-only room synthesis (`N`, `No-RRWP`) without Block III
  mission-role embeddings, `GraphGPS + RRWP` encodings, directional neighbor context,
  and incident door-boundary signals breaks inter-room continuity and lock-key role
  placement prior to repair.
- **`PS2` (`FG2`) — Continuous Latent Compression Pathology vs. Discrete Token Synthesis:**
  Compressing `349` training rooms (`217` unique `16x11` grids, `43` unique `12x7`
  structural interiors, `8` active tile IDs of `44`, `88.60%` `WALL+FLOOR`) via `17.37M`
  `VQ-VAE` + `26.33M` `UNet[1,2]` (`47.13M` total) or `DiT+Flow` (`22.44M` total) introduces
  reconstruction error and rare-tile erasure (`<0.7%` `ENEMY`/`STAIR`), whereas absorbing-state
  discrete `G-LMDM` (`3.58M` train / `3.36M` inference) models discrete `16x11` tokens directly.
- **`PS3` (`FG3`) — Traversability-Blind Objective & Unattributed Repair Burden:**
  Likelihood-only training ignores 1-tile corridor cut-sets and progression ordering, while
  evaluating only post-repair WFC/BFS outputs masks raw neural defects behind repair edits
  (`repair_rate`, `tiles_repaired_mean`).
- **`PS4` (`FG4`) — Dungeon-Level Leakage, Omitted External Baselines & Proof Conflation:**
  Room-level splits, omitted external symbolic baselines (`Random`, `WFC`, `Grammar-only`),
  dependent multi-fold pooling without within-identity pairing, and equating bounded `P-CBS`
  agents with formal `StateSpaceAStar` proofs invalidate empirical generalization claims.

### 1.2 Research Objectives (`RO1–RO4`)

- **`RO1` (Causal Conditioning — maps to `PS1`, `B1`):**
  Isolate `G-LMDM`'s Block III `GraphGPS + RRWP` + directional neighbor/boundary
  conditioner (`G`) against unconditioned/neighbor-only (`N`) and `No-RRWP` ablations.
- **`RO2` (Discrete vs. Continuous Synthesis — maps to `PS2`, `B2–B3`):**
  Benchmark discrete absorbing-state `G-LMDM` (`3.58M` train / `3.36M` inference) against
  calibrated continuous `VQ-VAE` + `UNet[1,2]` (`47.13M`) and `DiT+Flow` (`22.44M`) on
  rare-tile recall, pre-repair validity, parameter efficiency, and `4 GB` GPU latency.
- **`RO3` (Direct 16x11 Logic Regularization & 4-Stage Attribution — maps to `PS3`, `B4`, Block VI):**
  Couple differentiable Soft Bellman-Ford `LogicNet` losses directly onto `16x11` discrete
  logits (`G-L`/`G-LG`, `alpha_logic=0.1` vs. `G` and inference-only guidance) and report
  4-stage attribution (`raw -> constrained -> repaired -> compiled`) under neutral assembly.
- **`RO4` (Confirmatory Statistical Generalization & Oracle Separation — maps to `PS4`, `B5`, Blocks I/VII):**
  Enforce strict Dungeon-9 holdout (`Dungeons 1–8` train vs. unseen `Dungeon 9` test),
  external symbolic baselines (`Random`, `WFC`, `Grammar-only`), paired Holm-corrected
  inference, descriptive LODO sensitivity checks, and separation of `StateSpaceAStar` proofs
  from bounded `P-CBS` proxy diagnostics.

### 1.3 Research Questions (`RQ1–RQ4`)

- **`RQ1` (`RO1` — Causal GraphGPS+RRWP Conditioning):** Relative to `N` and `No-RRWP`,
  does causal Block III `GraphGPS + RRWP` conditioning (`G`) improve pre-repair door
  alignment, boundary consistency, and mission-role adherence on unseen Dungeon-9 graphs?
- **`RQ2` (`RO2` — Discrete G-LMDM vs. Continuous Latent Diffusion/Flow):** On the
  small-data VGLC `16x11` corpus (`349` train rooms, `43` unique `12x7` interiors), does
  discrete absorbing-state `G-LMDM` preserve rare tiles (`<0.7%`) and improve pre-repair
  validity at lower parameter/latency cost than continuous `VQ-VAE + UNet[1,2]` and
  `DiT+Flow` baselines?
- **`RQ3` (`RO3` — Direct 16x11 LogicNet Supervision & Repair Attribution):** Holding `G`
  fixed, does direct `16x11` Soft Bellman-Ford `LogicNet` supervision (`G-LG`,
  `alpha_logic=0.1`) increase pre-repair oracle solvability and reduce `repair_rate` and
  `tiles_repaired_mean` across `raw -> constrained -> repaired -> compiled` stages relative
  to `G` and inference-only steering without collapsing diversity?
- **`RQ4` (`RO4` — Dungeon-9 Generalization, External Baselines & Proof-Proxy Separation):**
  Under held-out Dungeon-9 evaluation, external symbolic baselines (`Random`, `WFC`,
  `Grammar-only`), paired non-parametric inference, and explicit separation of
  `StateSpaceAStar` proofs from bounded `P-CBS` proxies, do `G-LMDM` gains hold on unseen
  dungeon topologies without human-likeness overclaiming?

### 1.4 Hypotheses (`H1–H4`)

- **`H1` (`RQ1`):** Causal `GraphGPS + RRWP` conditioning (`G`) significantly improves
  pre-repair door alignment, boundary match accuracy, and mission-role consistency over `N`
  and `No-RRWP` on unseen Dungeon-9 mission graphs.
- **`H2` (`RQ2`):** Discrete absorbing-state `G-LMDM` achieves higher rare-tile recall and
  pre-repair room validity with an order-of-magnitude smaller footprint (`3.58M` train /
  `3.36M` inference vs. `47.13M` `UNet[1,2]` and `22.44M` `DiT+Flow`) and lower latency
  on `4 GB` GPUs, or establishes an explicit discrete–continuous Pareto trade-off.
- **`H3` (`RQ3`):** Direct `16x11` training-time Soft Bellman-Ford `LogicNet` supervision
  (`G-LG`) increases raw oracle solvability and reduces symbolic `repair_rate` and
  `tiles_repaired_mean` across `raw -> constrained -> repaired -> compiled` stages relative
  to `G` and inference-only guidance at matched diversity.
- **`H4` (`RQ4`):** Within-identity Dungeon-9 paired Holm-corrected tests, descriptive LODO
  sensitivity checks, external symbolic baselines (`Random`, `WFC`, `Grammar-only`), and
  Block I topology bounds preserve `G-LMDM` pre-repair gains while maintaining clear
  separation between formal `StateSpaceAStar` proofs and bounded `P-CBS` behavioral proxies.
- **Null-Outcome & Refutation Policy:** If `G-LG` fails to surpass `G` on raw pre-repair
  endpoints after Holm correction, or if continuous `VQ-VAE + UNet[1,2]` / `DiT+Flow`
  surpasses discrete `G-LMDM`, the negative or trade-off outcome is reported verbatim and
  claims revert to the empirically supported boundary.

In scope:

- Zelda-like, room-and-mission-graph dungeons under the repository's `zelda_v1`
  schema.
- Graph-conditioned generation, not unconstrained image generation.
- Repair-aware evidence, not hidden post-processing.
- P-CBS as a bounded-agent proxy unless calibrated human telemetry exists.

Out of scope unless separately proven:

- generic dungeon generation beyond the locked schema;
- human-likeness claims;
- state-of-the-art claims;
- paper-faithful LCM-LoRA claims;
- standalone neural solvability when only post-repair outputs are valid.

## 2. System Boundary

The proposed method includes these blocks:

| Block | Role | Evidence boundary |
|---|---|---|
| 0 | VGLC parsing, split lock, room/graph extraction | Data card and split hashes |
| I | Mission-graph generation and QD/evolutionary search | Topology validity, controllability, diversity |
| II | VQ/FSQ tokenizer | Reconstruction, code utilization, checkpoint provenance |
| III | Graph/local condition encoder | Target-response and conditioning ablations |
| IV | Room generator: diffusion, DiT, masked-token, categorical, fast sampler | Raw room validity, diversity, runtime |
| V | LogicNet diagnostics/guidance | ON/OFF and guidance ablations |
| VI | Symbolic repair/WFC/overlays | Repair delta, repair time, failure rate |
| VII | Hard oracle, P-CBS, metrics, MAP-Elites | Solver correctness, persona proxy tables |

Anything outside this boundary is a baseline, ablation, diagnostic, or GUI/demo.

## 3. Claim Language

Allowed wording:

- graph-conditioned;
- repair-aware;
- neural-symbolic;
- bounded-agent proxy;
- matched-budget internal ablation;
- raw/pre-repair validity;
- post-repair validity;
- external benchmark alignment.

Forbidden unless extra evidence exists:

- state-of-the-art;
- surpasses publications;
- human-like or human-likeness;
- paper-faithful LCM-LoRA;
- standalone neural solvability based only on post-repair results;
- generic dungeon generator.

## 4. Data Card

Required fields:

| Field | Required content |
|---|---|
| Dataset | VGLC The Legend of Zelda |
| Source path | `Data/The Legend of Zelda` |
| Manifest | `Data/The Legend of Zelda/dataset_manifest.json` from `scripts/lock_dataset_split.py` |
| Train split | Dungeon IDs 1-8 unless explicitly changed |
| Test split | Dungeon ID 9 unless explicitly changed |
| Room shape | `[16, 11]` |
| Tile classes | `44` |
| Graph schema | active `dataset.schema_profile` and config snapshot |
| Augmentation | exact commands and random seeds |
| Leakage policy | no Dungeon 9 room or graph in training/model selection |
| License/IP note | research-use VGLC-derived data, Zelda IP acknowledged |

Command:

```bash
python scripts/lock_dataset_split.py --data-root "Data/The Legend of Zelda"
```

## 5. Artifact Card

Every model or result used in a table needs:

| Field | Meaning |
|---|---|
| name | short artifact name |
| path | file or run directory |
| status | `current`, `stale`, `invalid`, `smoke_only`, or `blocked` |
| sha256 | required for current checkpoint files |
| config | resolved config path |
| git_commit | commit used to create it |
| seed_list | exact seeds |
| data_manifest | dataset manifest hash |
| notes | limitations and claim restrictions |

Use `current` only when the artifact was produced by the current code path and
matches the locked data split. Smoke runs stay `smoke_only` even if they pass.

## 6. Metric Contract

These metrics are required in publication-facing result tables:

- `raw_oracle_solved_rate`
- `post_oracle_solved_rate`
- `raw_pcbs_valid_rate`
- `post_pcbs_valid_rate`
- `repair_rate`
- `tiles_repaired_mean`
- `repair_time_sec`
- `generation_time_sec`
- `diversity`
- `ncd`
- `controllability_error`
- `teacher_fallback_used`
- `oracle_timeout_rate`
- `pcbs_timeout_rate`

Rules:

- Raw/pre-repair and post-repair rates must never be merged.
- Timeout, invalid input, contradiction, and unsolved are separate outcomes.
- Path length means transitions, not visited-state count.
- Linearity means route directness, not coverage.
- P-CBS overhead is excess path/search cost relative to the oracle, not a
  generic "confusion" claim.

## 7. Baseline Taxonomy

Symbolic baselines:

- random topology;
- grammar-only topology;
- flat-prior WFC;
- weighted-prior WFC;
- no repair.

Neural baselines:

- categorical/codebook prior;
- MaskGIT/discrete masked room generator;
- latent diffusion;
- DiT/flow matching if trained under the same budget.

Pipeline ablations:

- no graph conditioning;
- no LogicNet;
- LogicNet diagnostics only;
- LogicNet guidance;
- no symbolic repair;
- full pipeline.

External alignment:

- PCG Benchmark-style quality;
- diversity;
- controllability;
- matched-budget compute/runtime.

## 8. Failure Taxonomy

Every failed sample should be assigned at least one label:

- `disconnected_graph`
- `missing_key_or_token`
- `door_mismatch`
- `raw_room_unsolvable`
- `repair_failed`
- `oracle_timeout`
- `pcbs_timeout`
- `teacher_fallback_used`
- `invalid_checkpoint`
- `invalid_dataset_split`
- `metric_artifact_missing`
- `unsupported_claim`

Do not collapse these into "failed." The failure mix is part of the result.

## 9. Human Calibration Decision

There are two valid paths:

1. Proxy-only path:
   - call P-CBS a bounded-agent diagnostic;
   - report persona sensitivity;
   - do not claim human-likeness.

2. Human-calibrated path:
   - collect consent-marked human traces;
   - validate telemetry provenance;
   - fit persona parameters against traces;
   - report calibration error and held-out human agreement.

Until path 2 is complete, the paper should use path 1.

## 10. Reproducibility Package

Each paper table needs:

- command;
- config;
- seed list;
- git commit;
- checkpoint hashes;
- data manifest;
- hardware;
- runtime;
- output paths;
- metric schema;
- known limitations.

An exit code of zero is not enough. Required metric files must exist and contain
the required metric names.

## 11. Architecture Simplification Policy

Keep only claims that have a table or a validated card behind them.

Demote or remove from the main paper:

- unused attention upgrades;
- untrained fast sampler claims;
- duplicate refiner/WFC descriptions;
- old SOTA language;
- GUI/demo-only behavior;
- planned features without executable manifests.

## 12. Ethics And IP Note

The project uses Zelda/VGLC-derived data. Any publication should state:

- data is used for research evaluation;
- generated content is analyzed as game-level structure, not distributed as a
  commercial Zelda asset pack;
- Nintendo/Zelda IP is not owned by the project;
- VGLC provenance and split hashes are documented.

## Machine-Readable Card

The matching machine-readable card is generated and validated by:

```bash
python scripts/validate_publication_readiness.py --init-template --card docs/publication_guidance_card.json
python scripts/validate_publication_readiness.py --card docs/publication_guidance_card.json
```

The validator is intentionally conservative. It checks completeness and claim
discipline, not model quality.

## Literature Anchors

- PCGML survey: https://arxiv.org/abs/1702.00539
- PCG Benchmark: https://arxiv.org/abs/2503.21474
- WFC as constraint solving: https://doi.org/10.1145/3102071.3110566
- Procedural personas: https://arxiv.org/abs/1802.06881
- Graph2Plan: https://arxiv.org/abs/2004.13204
- HouseDiffusion: https://arxiv.org/abs/2211.13287
- LayoutDM: https://arxiv.org/abs/2303.08137
- DiGress: https://arxiv.org/abs/2209.14734
