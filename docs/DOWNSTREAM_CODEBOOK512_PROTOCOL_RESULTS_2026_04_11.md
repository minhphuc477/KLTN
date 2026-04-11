# Downstream Codebook512 Protocol Results 2026-04-11

This note records the verification pass for the downstream retrain that used the `codebook512` VQ-VAE:

- diffusion
- fast sampler
- masked room

Run directory:

- [outputs/zelda_hmolqd_downstream_codebook512_v1](../outputs/zelda_hmolqd_downstream_codebook512_v1)

Primary compare export:

- [outputs/zelda_hmolqd_downstream_codebook512_v1_compare/summary.json](../outputs/zelda_hmolqd_downstream_codebook512_v1_compare/summary.json)

## Protocol Performed

1. Checked completed checkpoints and training logs
2. Verified checkpoint metadata against the actual frozen VQ-VAE
3. Ran manual rich-topology end-to-end compare across:
   - diffusion
   - fast sampler
   - masked room
4. Inspected pairwise room diffs
5. Inspected representative puzzle-room text exports

The fixed-graph multi-seed audit was started, but the full two-seed run exceeded the runtime window. The seed-20260404 diffusion artifact completed and was inspected; the broader branch judgment below relies primarily on the completed manual rich-topology compare.

## Research Context

This interpretation stays aligned with the broader literature already used in the architecture audits:

- VQ-VAE quality matters for downstream generation, but downstream control/structure quality also depends heavily on the conditional generator and runtime constraints.
- In structured generation systems, improving the tokenizer alone does **not** guarantee end-to-end gains if the generative/control stages remain the dominant bottleneck.

Relevant references:

- VQ-VAE: https://arxiv.org/abs/1711.00937
- VQ-VAE-2: https://arxiv.org/abs/1906.00446
- Graph2Plan: https://arxiv.org/abs/2004.13204
- HouseDiffusion: https://arxiv.org/abs/2211.13287
- LayoutDM: https://arxiv.org/abs/2303.08137

The repo-specific judgments below are evidence-based from local artifacts, not claimed directly by the papers.

## Key Findings

### 1. Diffusion remains the only branch that is clearly usable

Manual rich-topology compare:

- `generation_time_sec = 103.19`
- `total_tiles_repaired = 243`
- `repair_rate = 0.833`
- `avg_final_graph_marker_overwrite_rate = 0.1667`
- `avg_neural_graph_marker_exact_match_rate = 1.0`
- `avg_neural_semantic_anchor_error = 0.0`

Interpretation:

- semantics are still being honored cleanly
- the branch is not perfect, but it is coherent enough to remain the production baseline

### 2. Fast sampler regressed badly on this downstream run

Manual rich-topology compare:

- `generation_time_sec = 157.81`
- `total_tiles_repaired = 523`
- `repair_rate = 0.9167`
- `avg_final_graph_marker_overwrite_rate = 0.2917`
- `fast_sampler_teacher_fallback = 11`

This is the most important practical result.

The supposed speed branch is:

- slower than diffusion on this export
- structurally farther from diffusion
- heavily dependent on teacher rescue

Pairwise diff against diffusion:

- `12` rooms compared
- `168` changed tiles
- largest drift concentrated in rooms `2`, `9`, `3`, and `6`
- dominant error type: `structure_to_floor`

Interpretation:

- this is not a good fast branch yet
- the current `codebook512` downstream retrain did **not** improve fast-sampler usefulness

### 3. Masked-room is still not production-ready

Manual rich-topology compare:

- `generation_time_sec = 437.69`
- `total_tiles_repaired = 593`
- `repair_rate = 0.8333`
- `avg_final_graph_marker_overwrite_rate = 0.3333`
- `masked_room_teacher_fallback = 10`

Interpretation:

- the branch still depends heavily on teacher fallback
- it remains experimental/guarded rather than trustworthy as a primary generator

### 4. Puzzle rooms are more semantic than before, but still constructive

Representative diffusion puzzle rooms:

- [room_5.txt](../outputs/zelda_hmolqd_downstream_codebook512_v1_compare/diffusion_cfg3_logic0_steps50/rooms/room_5.txt)
- [room_9.txt](../outputs/zelda_hmolqd_downstream_codebook512_v1_compare/diffusion_cfg3_logic0_steps50/rooms/room_9.txt)

The stateful puzzle templates are visible, but they are still clearly constructive route/gate patterns, not fully learned Zelda-grade room logic.

That part is consistent with previous audits: the improvement is real, but it is still a hybrid system.

## Reproducibility Bug Found And Fixed

While verifying the run, one real bug was found:

- diffusion checkpoint metadata was still serializing the config-default VQ-VAE shape (`codebook_size=256`) instead of the resolved tokenizer loaded from the actual checkpoint path

That bug did **not** necessarily invalidate the training run itself, but it made the saved diffusion artifact misleading for later handoff and verification.

The fix was applied in:

- [src/train_diffusion.py](../src/train_diffusion.py)

Regression added in:

- [tests/test_config_system.py](../tests/test_config_system.py)

## Verdict

### Does `codebook512` help Block II?

Yes.

### Did it clearly improve the end-to-end downstream stack?

Not enough.

The current evidence says:

- Block II improved slightly in isolation
- diffusion still works
- fast sampler did not meaningfully benefit and may have regressed as a practical branch
- masked room is still fallback-heavy

So the overall downstream judgment is:

- **keep diffusion as the production branch**
- **do not treat this run as proof that `codebook512` is the new canonical tokenizer for the full stack**
- **do not trust fast sampler or masked room yet just because Block II improved**

## Recommendation

If we continue from here, the best next move is **not** another tokenizer experiment.

The best next moves are:

1. keep diffusion as the main branch
2. investigate why the fast sampler deteriorated under the new tokenizer contract
3. keep masked room as experimental / guarded
4. only promote `codebook512` to canonical if a repaired fast-sampler/distillation path catches back up

Current practical recommendation:

- for full-stack quality right now, the `codebook512` tokenizer is **promising but not yet a proven whole-system upgrade**

