# Publication Guidance And Cards

Last updated: 2026-07-02.

This is the canonical pre-paper contract for this repository. It defines what
the project should claim, what it must not claim, which artifacts must exist,
and how evidence must be packaged before a paper draft is written.

The most publishable direction is not "best diffusion model" or "generic
dungeon generator." The strongest direction is:

> A repair-aware, graph-conditioned neural-symbolic PCG pipeline for Zelda-like
> dungeons that separates mission-graph controllability, neural room synthesis,
> symbolic repair, hard oracle validation, and bounded-agent diagnostics.

This direction is aligned with PCGML's emphasis on learned content modeling and
repair/critique uses, PCG Benchmark's quality/diversity/controllability axes,
WFC's role as constraint solving, graph-conditioned layout generation, and
procedural personas as synthetic playtesters.

## 1. Research Question

Use this as the locked research question until there is stronger evidence for a
different one:

> Can a graph-conditioned neural-symbolic PCG pipeline generate Zelda-like
> dungeons while explicitly separating raw neural validity, symbolic repair,
> hard-oracle validation, and bounded-agent diagnostics?

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
