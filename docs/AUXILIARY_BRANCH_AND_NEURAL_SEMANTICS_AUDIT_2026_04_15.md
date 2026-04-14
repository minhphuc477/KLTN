# Auxiliary Branch And Neural Semantics Audit 2026-04-15

Last updated: 2026-04-15

Scope:

- topology graph generation
- room generation
- auxiliary fast-sampler and masked-room branches
- decision on `hybrid vs fully neural semantics`

Canonical code touched in this pass:

- [`src/train_lcm.py`](../src/train_lcm.py)
- [`src/train_masked_room.py`](../src/train_masked_room.py)
- [`src/core/discrete_masked_model.py`](../src/core/discrete_masked_model.py)
- [`src/optimization/lcm_lora.py`](../src/optimization/lcm_lora.py)
- [`src/pipeline/room_topology_conditioning.py`](../src/pipeline/room_topology_conditioning.py)
- [`configs/zelda_hmolqd.yaml`](../configs/zelda_hmolqd.yaml)

This document is the concise research-backed answer to one question:

`Should the repo move to fully neural semantics now?`

Current answer: `no, not in production`.

The correct direction is:

1. keep the `graph-first hybrid` production contract
2. make the neural branches more topology-faithful
3. only promote a pure-neural semantic path if strict no-fallback ablations beat
   the hybrid path on the repo's own correctness metrics

## Step 1. Deep Research And Literature Review

### Most relevant sources

- [Graph2Plan, CVPR 2020](https://arxiv.org/abs/2004.13204)
- [HouseDiffusion, CVPR 2023](https://arxiv.org/abs/2211.13287)
- [Constrained Layout Generation with Factor Graphs, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Dupty_Constrained_Layout_Generation_with_Factor_Graphs_CVPR_2024_paper.pdf)
- [LayoutDiffusion, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_LayoutDiffusion_Improving_Graphic_Layout_Generation_by_Discrete_Diffusion_Probabilistic_Models_ICCV_2023_paper.pdf)
- [MaskGIT, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Chang_MaskGIT_Masked_Generative_Image_Transformer_CVPR_2022_paper.pdf)
- [Improved Techniques for Training Consistency Models, ICLR 2024](https://openreview.net/forum?id=WNzy9bRDvG)
- [Progressive Distillation for Fast Sampling of Diffusion Models, ICLR 2022](https://openreview.net/pdf?id=TIdIXIpzhoI)
- [VQ-VAE, NeurIPS 2017](https://arxiv.org/abs/1711.00937)
- [VQ-VAE-2, NeurIPS 2019](https://arxiv.org/abs/1906.00446)
- [DDIM, ICLR 2021](https://openreview.net/forum?id=St1giarCHLP)
- [Latent Diffusion Models, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf)
- [GraphRNN, ICML 2018](https://proceedings.mlr.press/v80/you18a.html)
- [DiGress, ICLR 2023](https://arxiv.org/abs/2209.14734)
- [PCGRL, AIIDE 2020](https://cdn.aaai.org/ojs/7416/7416-52-10717-1-2-20200923.pdf)
- [PCGML survey, IEEE TG 2018](https://arxiv.org/abs/1702.00539)

### What the literature actually supports

1. `Graph-first structured control is still the strongest bias`.
   Graph2Plan, HouseDiffusion, LayoutDiffusion, and the CVPR 2024 factor-graph
   paper all push control signals into generation explicitly instead of hoping
   the generator discovers them implicitly.

2. `Constraint-aware generation beats weak post-hoc fixing, but exact hard
   constraints still matter when the domain is small and semantics are sparse`.
   The factor-graph paper is especially relevant here: it argues that plain
   pairwise graph models miss higher-order constraints.

3. `Few-step students are only useful when the training objective matches the
   semantics that matter`.
   Progressive distillation and later consistency-model work support faster
   students, but they do not support using a speed student whose loss ignores
   sparse topology-critical semantics.

4. `Masked iterative generation is viable, but only when the semantic contract
   is explicit`.
   MaskGIT supports parallel masked decoding. It does not support greedy
   deterministic masked decoding with weak structure signals.

5. `Purely neural end-to-end PCG is not automatically better`.
   PCGRL and the PCGML survey both support the idea that explicit objectives and
   constraints are valuable, especially when training data is limited and the
   goal is playability rather than only distribution matching.

## Step 2. Assumptions Validation

### Valid assumptions

- topology should own progression semantics
- room geometry and mission semantics should not be fully conflated on this corpus
- teacher quality still caps the fast sampler
- exact graph markers are sparse enough that generic losses underweight them

### Fragile assumptions

- masked-room validation previously behaved like a real held-out validation set
  when it did not
- auxiliary checkpoints were always self-describing even without sidecar metadata
- pure-neural ablations could be judged fairly without topology-critical model
  selection metrics

### Hardcoded / config-surface assumptions promoted in this pass

- `masked_room.validation_fraction`
- `masked_room.validation_max_batches`
- `masked_room.best_checkpoint_metric`
- `fast_sampler.best_checkpoint_metric=val_topology_decode_ce_loss` as an allowed option

## Step 3. Logical Audit

### What remains logically correct

- `Block I -> conditioned room generator -> hybrid semantic enforcement`
- explicit graph semantics for start / key / item / boss / goal
- puzzle templates tied to edge semantics rather than a single generic puzzle family

### What was still logically weak

1. `Auxiliary branches were trained mostly on generic reconstruction quality`.
   That is misaligned with the real failure mode, which is sparse semantic drift.

2. `Masked-room branch claimed validation quality without a real held-out split`.
   That made checkpoint selection weaker than the diffusion and fast-sampler
   branches.

3. `Fast-sampler checkpoint selection could not target topology-critical decode
   CE even after that loss existed`.

## Step 4. Theory vs Implementation Check

### Gap found

- The repo narrative already said auxiliary branches should improve semantic
  placement.
- The implementation did not yet use a held-out split for masked-room, and the
  fast sampler could not choose checkpoints by topology-critical decode loss.

That is a true `theory vs implementation` gap, not just an opinion.

## Step 5. Gap And Bug Analysis

### Fixed in code now

1. `Masked-room validation split bug`.
   The branch now uses a deterministic held-out split via the same validation
   helper used elsewhere.

2. `Masked-room checkpoint selection gap`.
   The branch now exposes:
   - `masked_room.validation_fraction`
   - `masked_room.validation_max_batches`
   - `masked_room.best_checkpoint_metric`

3. `Fast-sampler topology-metric selection gap`.
   The branch now accepts `val_topology_decode_ce_loss` as a valid best-checkpoint metric.

4. `Checkpoint self-description gap`.
   Auxiliary checkpoint payloads now embed `topology_anchor_policy` directly, not
   only in sidecar metadata.

### Still open

- fast sampler still needs retraining under the new topology-focused loss
- masked-room still needs retraining under the real held-out validation regime
- pure-neural runtime semantics are still not proven better than hybrid

## Step 6. Redundancy / Unnecessary Work

What is still unnecessary today:

- treating fast-sampler as a production peer
- treating masked-room as a production peer
- claiming fully neural semantics is already better without no-fallback wins

What is still necessary today:

- deterministic graph-marker overlay
- semantic constrained decoding
- teacher fallback guards for auxiliary branches

## Step 7. Computational Complexity

The repo's own measured parameter counts already put the system in the
small-data caution zone:

- VQ-VAE: about `17.62M`
- diffusion teacher stack: about `70.08M`
- masked-room branch: about `12.45M`

See:

- [`CANONICAL_MODEL_RATIONALE_ABLATION_AND_COMPLEXITY_GUIDE.md`](CANONICAL_MODEL_RATIONALE_ABLATION_AND_COMPLEXITY_GUIDE.md)
- [`STATEFUL_PUZZLE_ARCHITECTURE_AUDIT_2026_04_09.md`](STATEFUL_PUZZLE_ARCHITECTURE_AUDIT_2026_04_09.md)

Implication:

- replacing the hybrid runtime with a larger pure-neural branch is not justified
  unless it clearly reduces overwrite, fallback, and repair load

## Step 8. Hyperparameter Sensitivity

### Newly important auxiliary-branch parameters

| Parameter | Type | Default | Valid range | Source | Notes |
|---|---|---:|---|---|---|
| `fast_sampler.topology_alignment_weight` | float | `0.25` | `>=0` | inference-based + repo evidence | Raises pressure on anchors, gates, doors, and traces. |
| `fast_sampler.topology_marker_weight` | float | `2.0` | `>=0` | inference-based | Keeps sparse markers from being washed out by floor tiles. |
| `fast_sampler.topology_trace_weight` | float | `0.75` | `>=0` | inference-based | Balances path trace against anchor exactness. |
| `fast_sampler.topology_focus_dilation` | int | `1` | `>=0` | inference-based | Stabilizes sparse CE targets. |
| `fast_sampler.best_checkpoint_metric` | str | `val_decode_ce_loss` | `{val_loss,val_decode_ce_loss,val_topology_decode_ce_loss,train_loss}` | inference-based | New topology-only option is for semantic-fidelity ablations. |
| `masked_room.topology_alignment_weight` | float | `0.25` | `>=0` | inference-based + repo evidence | Same rationale as fast sampler. |
| `masked_room.validation_fraction` | float | `0.1` | `0..0.5` | reproducibility requirement | Real held-out validation. |
| `masked_room.validation_max_batches` | int | `16` | `>=1` | reproducibility requirement | Keeps validation bounded. |
| `masked_room.best_checkpoint_metric` | str | `val_loss` | `{val_loss,val_topology_focus_loss,train_loss}` | inference-based | Topology-only option is for semantic-fidelity ablations. |

Safe operating rule:

- increase topology alignment weights gradually
- do not disable hybrid safeguards during branch retraining
- use topology-only selection metrics only for ablations, not as the first
  production default

## Step 9. Failure Modes

The current auxiliary branches still fail hardest on:

- sparse graph marker placement before overlay
- typed-gate exactness
- noisy obstacle interiors that trigger repair or teacher fallback
- cases where a room needs both spatial cleanliness and exact mission semantics

## Step 10. Scalability And Generalization Boundary

The current system generalizes acceptably only within:

- Zelda-like fixed-size room grids
- graph semantics already represented in the topology schema
- checkpoint families aligned with the same topology-anchor policy

It is not yet credible to claim robust transfer to:

- new room sizes
- new game mechanic vocabularies
- fully neural zero-repair generation

## Step 11. Comparison Against Other Publications

### Where the repo is strong

- stronger `mission correctness` contract than pure room-only generators
- richer graph-conditioned room semantics than older graph+GAN Zelda baselines
- explicit hybrid control that is closer to real dungeon-design constraints than
  unconstrained graph generators like GraphRNN or DiGress

### Where the repo is not yet proven better

- no matched-budget external benchmark yet proves it surpasses the best recent
  structured-layout papers
- auxiliary branches still do not beat the main hybrid diffusion path

Rigorous judgment:

- the current repo is `meaningfully novel as a graph-first hybrid Zelda system`
- it is `not yet empirically proven to surpass all relevant publications`

## Step 12. Bias / Ethical Risk

Main architectural risk here is not human demographic unfairness but
`evaluation bias`:

- over-crediting a neural branch because the hybrid runtime silently rescues it
- reporting “validation” from the training set
- comparing against external papers without matched constraints or budgets

This pass fixes one of those directly: masked-room now has a real held-out
validation split.

## Step 13. Evidence-Based Decision

### Should production move to fully neural semantics now?

`No.`

Reason:

- literature supports richer neural conditioning, not removing constraints
  prematurely
- local pure-neural ablations are still worse on overwrite and pre-overlay
  anchor error
- current production diffusion branch still depends on overlay because that
  overlay is carrying real correctness value

### Why keep the hybrid path for now?

Because the current hybrid path is better on the metrics that matter most:

- lower semantic overwrite
- lower fallback pressure
- better mission correctness

### Why still pursue stronger neural semantics?

Because the best recent layout and masked-generation papers support pushing more
structure into the generator itself, which can reduce repair load and narrow the
gap between the neural output and the final exported room.

That is different from saying `fully neural is already better`.

## Step 14. Recommended Ablations

Do not run these inside the report without fresh outputs. They are the next
defensible experiments.

### A. Fast sampler, topology-focused retrain

```powershell
python -m src.train_lcm `
  --config configs\zelda_hmolqd.yaml `
  --base-diffusion-checkpoint outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\checkpoints\diffusion\best_model.pth `
  --checkpoint-dir outputs\zelda_hmolqd_aux_topofocus_v1\checkpoints\fast_sampler `
  --topology-alignment-weight 0.25 `
  --topology-marker-weight 2.0 `
  --topology-trace-weight 0.75 `
  --topology-focus-dilation 1 `
  --best-checkpoint-metric val_decode_ce_loss
```

### B. Fast sampler, topology-metric checkpoint selection ablation

```powershell
python -m src.train_lcm `
  --config configs\zelda_hmolqd.yaml `
  --base-diffusion-checkpoint outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\checkpoints\diffusion\best_model.pth `
  --checkpoint-dir outputs\zelda_hmolqd_aux_topofocus_metric_v1\checkpoints\fast_sampler `
  --topology-alignment-weight 0.25 `
  --topology-marker-weight 2.0 `
  --topology-trace-weight 0.75 `
  --topology-focus-dilation 1 `
  --best-checkpoint-metric val_topology_decode_ce_loss
```

### C. Masked-room, held-out validation retrain

```powershell
python -m src.train_masked_room `
  --config configs\zelda_hmolqd.yaml `
  --checkpoint-dir outputs\zelda_hmolqd_aux_topofocus_v1\checkpoints\masked_room `
  --topology-alignment-weight 0.25 `
  --topology-marker-weight 2.0 `
  --topology-trace-weight 0.75 `
  --topology-focus-dilation 1 `
  --validation-fraction 0.1 `
  --validation-max-batches 16 `
  --best-checkpoint-metric val_loss
```

### D. Masked-room, topology-metric checkpoint selection ablation

```powershell
python -m src.train_masked_room `
  --config configs\zelda_hmolqd.yaml `
  --checkpoint-dir outputs\zelda_hmolqd_aux_topofocus_metric_v1\checkpoints\masked_room `
  --topology-alignment-weight 0.25 `
  --topology-marker-weight 2.0 `
  --topology-trace-weight 0.75 `
  --topology-focus-dilation 1 `
  --validation-fraction 0.1 `
  --validation-max-batches 16 `
  --best-checkpoint-metric val_topology_focus_loss
```

### E. Strict pure-neural semantic export check

```powershell
python main.py topology-audit-fixed-graph `
  --run-dir outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1 `
  --output-dir outputs\strict_pure_neural_recheck_v1 `
  --seeds 20260404 20260405 20260406 `
  --no-deterministic-graph-marker-overlay-enabled `
  --no-fast-sampler-teacher-fallback-enabled `
  --no-masked-room-teacher-fallback-enabled
```

Promotion rule:

- do not promote a pure-neural semantic path unless it beats hybrid on
  `avg_final_graph_marker_overwrite_rate`, `avg_final_pre_overlay_semantic_anchor_error`,
  and teacher fallback usage

## Step 15. Priority Ranking

### Critical

1. `Auxiliary branch selection/training must target topology-critical semantics`.
2. `Masked-room must use real held-out validation`.

### High

3. `Checkpoint payloads must carry topology policy metadata directly`.
4. `No-fallback pure-neural claims must be gated by strict ablations`.

### Medium

5. `Topology graph generation can be further neuralized later, but not at the
   expense of mission correctness`.

### Low

6. `Fully neural end-to-end semantics as a production target`.
   This remains a research goal, not the next production move.

## Step 16. Immediate Implementation Applied In This Pass

Implemented now:

1. `Masked-room real held-out validation split`
2. `Masked-room validation/checkpoint metric config surface`
3. `Fast-sampler topology-critical checkpoint metric option`
4. `Topology-anchor policy embedded directly into auxiliary checkpoint payloads`

Validated locally:

- `tests/test_train_lcm_checkpointing.py`
- `tests/test_discrete_masked_room_model.py`
- `tests/test_config_system.py`
- `tests/test_zelda_loader_graph_conditioning.py`

## Final Decision

The repo should stay `hybrid in production` today.

That is not because fully neural semantics are uninteresting. It is because:

- the literature supports stronger structured neural conditioning
- the local evidence does not yet support removing the hybrid safeguards
- the next credible path is to make auxiliary branches optimize for the
  semantics the hybrid runtime currently has to rescue

If a future `strict pure-neural, no-fallback` branch wins on the repo's own
semantic-placement metrics, then the production recommendation should change.
That has not happened yet.
