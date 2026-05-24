# Architecture Research Audit Addendum: Diffusion Conditioning

This addendum updates the existing architecture audits in:

- `ARCHITECTURE_RESEARCH_AUDIT_2026_03_31.md`
- `TOPOLOGY_GRAPH_RESEARCH_AUDIT_2026_03_31.md`

It focuses on what changed after reviewing recent diffusion-for-PCG and prompt-free conditioning papers, then re-checking the current implementation. The important conclusion is unchanged: keep topology generation explicit and constraint-driven, and improve the room generator's local conditioning rather than replacing Block I with a monolithic diffusion model.

## Step 1. Deep Research and Literature Review

### Primary literature reviewed

| Paper | Venue | What it establishes | Architectural implication here |
|---|---|---|---|
| Text-to-Level Diffusion Models With Various Text Encoders for Super Mario Bros | arXiv preprint, July 2025 | Small in-domain encoders beat larger generic encoders in this constrained PCG setting; long levels remain harder to make beatable; mixed-initiative composition is useful. | Prefer compact domain-native conditioning over heavy generic encoders. Do not collapse topology and room generation into one long-horizon diffusion model. |
| Procedural Level Generation with Diffusion Models from a Single Example | AAAI 2024 | Dense semantic token embeddings and locally constrained receptive fields help diffusion learn level structure from very little data. | Strengthen room-local semantic conditioning and locality bias. |
| Prompt-Free Diffusion: Taking "Text" out of Text-to-Image Diffusion Models | CVPR 2024 | Replacing text prompts with a learned reference encoder can work well; position-aware reference encoding helps structural fidelity. | Add a lightweight reference-room encoder instead of expanding text/prompt conditioning. |
| MaskGIT: Masked Generative Image Transformer | CVPR 2022 | Masked-token iterative generation is effective for discrete spatial synthesis when conditioning is well structured. | Supports the masked-room branch, but does not solve mission-graph validity. |
| LayoutDM: Discrete Diffusion Model for Controllable Layout Generation | CVPR 2023 | Discrete diffusion is strong for controllable structured layouts. | Relevant to Block IV room/layout synthesis, not a substitute for mission-graph generation. |
| HouseDiffusion: Vector Floorplan Generation via a Diffusion Model With Discrete and Continuous Denoising | CVPR 2023 | Hybrid discrete/continuous conditioning helps structured geometry generation. | Supports richer room/layout conditioning, again downstream of topology. |
| Recipe for a General, Powerful, Scalable Graph Transformer (GraphGPS) | arXiv / ICLR-era graph-transformer literature | Hybrid local-message-passing plus global attention scales better than naive graph transformers. | Supports the current `gps` option in the condition encoder. |
| GraphRNN | ICML 2018 | Canonical graph-generation baseline and graph-distribution evaluation family. | Still a baseline reference for topology generation. |
| DiGress | ICLR 2023 | Strong modern discrete graph diffusion baseline. | Relevant for Block I benchmarking, not for replacing room-local denoising. |

### Literature summary

1. The July 2025 Mario paper argues against over-engineering the conditioner. Its discussion section explicitly reports that a small transformer with a small vocabulary worked best, while negative prompts, absence captions, larger language models, and larger U-Nets mostly increased cost without consistent gains.
2. The AAAI 2024 single-example paper argues for dense semantic level representations and constrained local receptive fields. That is a much better match for room generation than mission-graph generation.
3. Prompt-Free Diffusion shows that replacing text with a learned reference encoder is a viable way to preserve style and structure. That transfers cleanly to neighboring-room exemplar conditioning.
4. The broader layout-diffusion literature supports discrete, structured room generation, but it does not remove the need for explicit progression and topology constraints.

## Step 2. Assumptions Validation

### Valid assumptions

- The topology branch should remain explicit and rule-constrained.
  Source: the 2025 Mario diffusion paper reports that longer levels are less reliable and benefits from GUI-assisted composition; this supports keeping long-horizon progression explicit.
- Local room structure benefits from dense semantics and local receptive fields.
  Source: AAAI 2024 single-example diffusion.
- A compact domain-specific conditioner is preferable to a heavyweight generic language encoder for this task family.
  Source: the 2025 Mario diffusion paper.

### Fragile or undocumented assumptions

- `style_id` exists in Block III but there was no end-to-end dataset or pipeline path supplying a real style token.
  Status: fixed for the repo's canonical sector-theme vocabulary. Explicit numeric metadata and compound symbolic sector-theme labels are now forwarded end-to-end as stable style tokens.
- Masked-room training previously assumed null local context at the room-anchor stage.
  Status: fragile, code-confirmed. This contradicted the literature's emphasis on local semantics and also created a train-time information gap versus inference.
- The pipeline previously assumed `context_dim == 256` in one validation/fallback path even though the config system exposes `context_dim` as configurable.
  Status: fragile, code-confirmed runtime bug.

### Hardcoded assumptions promoted into config

The following new assumptions are now explicit YAML/CLI fields:

- `diffusion.condition_use_reference_room_maps`
- `diffusion.condition_reference_tile_vocab_size`
- `diffusion.condition_reference_embedding_dim`
- `diffusion.condition_reference_hidden_dim`
- `masked_room.condition_use_reference_room_maps`
- `masked_room.condition_reference_tile_vocab_size`
- `masked_room.condition_reference_embedding_dim`
- `masked_room.condition_reference_hidden_dim`

## Step 3. Logical Audit

### Finding A: the architecture described a style/global-consistency path, but the implementation had no real data source for it

This was not a crash bug, but it was conceptually weak. The encoder had capacity reserved for a style token without an actual end-to-end style signal. That is architectural overhead without evidence of value.

Decision:

- Keep the style path for backward compatibility.
- Add a real neighboring-room exemplar path, because that is directly supported by the literature and by the data already present in the repo.

### Finding B: masked-room conditioning underused the available room-local context

Diffusion training already used teacher-forced neighboring room maps to build local latents. Masked-room training did not. That made the masked-room branch less consistent with the locality-focused literature and less consistent with the rest of the stack.

Decision:

- Add reference-room map conditioning for masked-room training instead of trying to retrofit a VQ-VAE latent path into that trainer.

### Finding C: one runtime path silently violated the advertised configurability of `context_dim`

This was a real theory-vs-implementation mismatch: the repo exposed `context_dim`, but the pipeline still had a hardcoded `256` assumption in its validation/fallback path.

Decision:

- Remove the hardcoded width and bind it to the loaded condition encoder's actual `output_dim`.

## Step 4. Theory vs. Implementation Consistency Check

### Silent gaps that existed before this patch

- "Global style token" was implemented but not meaningfully sourced from the data pipeline.
- Room-level exemplar conditioning was absent even though the dataset already exposed `neighbor_maps` and the inference pipeline already had neighboring `room_grid` outputs.
- The pipeline advertised configurable conditioning width, but one fallback path still assumed `256`.

### Current status after implementation

- The condition encoder now has an optional `ReferenceRoomMapEncoder`.
- Diffusion training can pass `neighbor_maps` into the condition encoder as reference exemplars.
- Masked-room training can do the same.
- Inference now derives reference room maps from previously generated neighboring room grids.
- Pipeline fallback validation now uses the encoder's actual width instead of a hardcoded constant.

## Step 5. Gap and Bug Analysis

### Fixed now

- Missing exemplar/reference-room conditioning in Block III.
- Masked-room room-anchor path ignoring available neighboring room maps.
- `context_dim=256` hardcode in pipeline conditioning validation/fallback.

### Still open

- External room-branch comparisons against matched-budget layout-diffusion baselines are still not completed.
  Status: internal matched-budget room-branch harness is now implemented, but external baseline runs remain experiment-only.
  Priority: High, but research-only.

## Step 6. Redundancy and Unnecessary Work

- The old style-token path was partially redundant in practice because it had no real source signal.
  We did not remove it for checkpoint compatibility, but the new reference-room path now carries the local exemplar role that the literature actually supports.
- Large generic conditioning models are not justified here.
  The 2025 Mario paper is direct evidence against spending extra compute on them in this task regime.

## Step 7. Computational Complexity Analysis

### Existing dominant costs

- Block IV U-Net / denoiser remains the dominant cost.
- Graph conditioning cost depends on the selected GNN/GPS backbone and graph size.

### New overhead

The new reference-room path is lightweight:

- Time: `O(B * K * H * W * d_ref)` for `K <= 4` neighboring rooms.
- Memory: linear in room area and reference embedding width.
- In this repo's schema (`16x11` rooms, at most 4 neighbors), this is negligible relative to the room denoiser.

### Complexity-relevant new knobs

- `condition_reference_embedding_dim`
- `condition_reference_hidden_dim`

Safe operating guidance:

- `embedding_dim`: `16..64`
- `hidden_dim`: `32..128`

## Step 8. Hyperparameter Sensitivity Analysis

### New sensitive parameters

| Parameter | Type | Default | Valid range | Source | Notes |
|---|---|---:|---|---|---|
| `condition_use_reference_room_maps` | bool | `true` | `{true,false}` | literature-backed + code-backed | Enabled by default in the current validated small-data profile. |
| `style_id` | int | derived from graph/dataset metadata | `0..5` under current canonical theme vocabulary | code-backed + inference-based | Now resolves both numeric IDs and canonical sector-theme labels such as `fire-temple` or `shadow_dungeon`. |
| `condition_reference_tile_vocab_size` | int | `44` | must equal `dataset.num_classes` | inference-based schema rule | Prevents semantic remapping bugs. |
| `condition_reference_embedding_dim` | int | `32` | `16..64` | inference-based, literature-aligned | Higher values add cost quickly with little expected benefit in this small schema. |
| `condition_reference_hidden_dim` | int | `64` | `32..128` | inference-based, literature-aligned | Small CNN hidden width is enough for `16x11` rooms. |

Interdependencies:

- `condition_reference_tile_vocab_size` must match `dataset.num_classes`.
- If `condition_use_reference_room_maps=false`, the remaining three parameters are inactive.

## Step 9. Failure Modes and Edge Cases

- If neighboring reference rooms are poor quality, the exemplar path can propagate local artifacts.
- If the tile vocabulary changes without updating `condition_reference_tile_vocab_size`, the reference encoder can silently misinterpret semantics. This is now blocked by config validation.
- Reference-room conditioning does not solve long-horizon progression failures. That remains the topology branch's job.

## Step 10. Scalability and Generalization Boundaries

- The new exemplar path scales linearly with room area and number of neighboring references, so it is safe at the current Zelda room size.
- It is not the limiting factor for larger graphs or longer dungeons; the denoiser and topology search remain the real scaling bottlenecks.
- Generalization remains bounded by the room-domain schema. This patch improves local fidelity, not out-of-domain transfer.

## Step 11. Comparison Against State of the Art

- The hybrid topology-plus-room design is still sensible and still more controllable than a single end-to-end diffusion model for this project.
- The room branch is more aligned with modern structured-generation practice after this patch, because it now uses domain-native exemplar conditioning instead of leaving that signal unused.
- The overall system is still not claimable as SOTA without matched-budget external benchmarking against layout-diffusion baselines for the room branch and graph-generation baselines for Block I.

## Step 12. Bias and Ethical Risk Analysis

- Reference-room conditioning can replicate aesthetic or structural artifacts from its local exemplars.
- This is a narrower, more controllable risk than introducing a large pretrained text encoder with opaque priors.
- The main architectural safety benefit remains the explicit topology branch, because progression constraints stay auditable.

## Step 13. Evidence-Based Decision Summary

### Core decisions

- Keep topology generation explicit.
  Supported by the 2025 Mario paper's long-level limitations and mixed-initiative workflow.
- Strengthen room-local semantic conditioning.
  Supported by AAAI 2024 single-example diffusion.
- Prefer a small learned reference encoder over a large generic prompt encoder.
  Supported by Prompt-Free Diffusion and the 2025 Mario paper.

## Step 14. Recommended Ablations

### A1. Reference-room conditioning off vs on

What it tests:

- Whether neighboring-room exemplars improve room coherence and local style consistency.

Metrics:

- room validity, structural artifact rate, boundary consistency, topology-conditioned control adherence.

YAML delta:

```yaml
diffusion:
  condition_use_reference_room_maps: false
masked_room:
  condition_use_reference_room_maps: false
```

### A2. Small vs larger reference encoder

What it tests:

- Whether the reference path follows the same "small is enough" pattern seen in the 2025 Mario paper.

YAML delta:

```yaml
diffusion:
  condition_use_reference_room_maps: true
  condition_reference_embedding_dim: 16
  condition_reference_hidden_dim: 32
```

```yaml
diffusion:
  condition_use_reference_room_maps: true
  condition_reference_embedding_dim: 64
  condition_reference_hidden_dim: 128
```

### A3. GPS vs GCN under reference conditioning

What it tests:

- Whether better graph conditioning still matters once local exemplars are available.

CLI:

```bash
python -m src.train_diffusion --config configs/zelda_hmolqd.yaml --condition-gnn-type gps --condition-use-reference-room-maps
python -m src.train_diffusion --config configs/zelda_hmolqd.yaml --condition-gnn-type gcn --condition-use-reference-room-maps
```

### A4. Masked-room null-local baseline vs reference-map local conditioning

What it tests:

- Whether masked-room performance was previously bottlenecked by missing local room context.

YAML delta:

```yaml
masked_room:
  condition_use_reference_room_maps: true
```

## Step 15. Priority Ranking

### Critical

- Remove hardcoded `context_dim=256` assumption from pipeline conditioning fallback.
  Status: fixed.

### High

- Add real exemplar/reference-room conditioning to Block III.
  Status: fixed.
- Give masked-room training access to real neighboring room exemplars.
  Status: fixed.

### Medium

- Replace or properly source the existing `style_id` pathway.
  Status: fixed for the repo's canonical sector-theme vocabulary. Numeric IDs and compound symbolic labels now resolve to stable style tokens end to end.
- Run matched-budget room-branch comparisons against structured diffusion baselines.
  Status: internal matched-budget room-branch benchmark harness is implemented; external baseline runs are still open.

### Reproducibility risks

- Hidden exemplar-conditioning behavior.
  Status: fixed by YAML/CLI exposure.
- Tile-vocabulary mismatch for the reference encoder.
  Status: fixed by config validation.

## Step 16. Immediate Implementation Applied

Implemented in code now:

1. Added `ReferenceRoomMapEncoder` to `src/core/condition_encoder.py`.
2. Added optional `reference_room_maps` inputs to `DualStreamConditionEncoder.forward(...)`.
3. Wired exemplar conditioning into `src/train_diffusion.py`.
4. Wired exemplar conditioning into `src/train_masked_room.py`.
5. Wired inference-time neighboring room grids into the pipeline in `src/pipeline/dungeon_pipeline.py`.
6. Promoted all new knobs into `src/config_system.py` and `configs/zelda_hmolqd.yaml`.
7. Fixed the pipeline's hardcoded `256`-dim conditioning fallback.
8. Added regression tests covering config, diffusion training, masked-room training, and the condition encoder behavior.
9. Wired explicit numeric `style_id` / `theme_id` metadata through room-level dataset samples, diffusion training, masked-room training, and inference-time room conditioning.
10. Added canonical sector-theme token resolution so symbolic labels like `fire-temple`, `ice_cavern`, and `shadow_dungeon` now map to stable style IDs instead of being dropped.
11. Added a dedicated internal room-branch benchmark harness for latent-vs-masked generation and reference-room conditioning ablations under matched topology/search budgets.

## Sources

- Text-to-Level Diffusion Models With Various Text Encoders for Super Mario Bros: https://arxiv.org/abs/2507.00184
- Prompt-Free Diffusion: Taking "Text" out of Text-to-Image Diffusion Models: https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_Prompt-Free_Diffusion_Taking_Text_out_of_Text-to-Image_Diffusion_Models_CVPR_2024_paper.pdf
- Procedural Level Generation with Diffusion Models from a Single Example: https://doi.org/10.1609/aaai.v38i9.28865
- MaskGIT: https://doi.org/10.1109/CVPR52688.2022.01103
- LayoutDM: https://openaccess.thecvf.com/content/CVPR2023/papers/Inoue_LayoutDM_Discrete_Diffusion_Model_for_Controllable_Layout_Generation_CVPR_2023_paper.pdf
- HouseDiffusion: https://openaccess.thecvf.com/content/CVPR2023/papers/Shabani_HouseDiffusion_Vector_Floorplan_Generation_via_a_Diffusion_Model_With_Discrete_CVPR_2023_paper.pdf
- GraphGPS: https://arxiv.org/abs/2205.12454
- GraphRNN: https://proceedings.mlr.press/v80/you18a.html
- DiGress: https://arxiv.org/abs/2209.14734
