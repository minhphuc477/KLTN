# Downstream Codebook512 Puzzle-Subtype Protocol Results 2026-04-15

This note records the post-train verification pass for the downstream retrain
that used:

- `codebook512` VQ-VAE
- puzzle-subtype topology channels
- final diffusion checkpoint completed on `2026-04-14`

Run directory:

- [outputs/zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1](../outputs/zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1)

Primary final exports:

- [manual compare v2 summary](../outputs/zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1/protocol_manual_compare_default_v2/summary.json)
- [manual compare v2 report](../outputs/zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1/protocol_manual_compare_default_v2/report.md)
- [fixed-graph audit v2 summary](../outputs/zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1/protocol_ablation_hybrid_default_v2/summary.json)

## Protocol Performed

1. Verified completed diffusion, fast-sampler, and masked-room checkpoints and logs.
2. Re-ran the manual rich-topology compare on the final diffusion weights.
3. Re-ran the fixed-graph multi-seed hybrid-default audit on the final diffusion weights.
4. Re-read the earlier ablation suite:
   - `hybrid_default`
   - `hybrid_weak_decode`
   - `neural_only_constrained_decode`
   - `strict_pure_neural`
   - `strict_pure_neural_no_fallback`
5. Audited the reporting code path and fixed one protocol bug:
   - per-room `final_post_overlay_semantic_anchor_avg_manhattan_error` already existed
   - dungeon-level summaries and export scripts were not surfacing it

Regression coverage for that bug now lives in:

- [tests/test_protocol_reporting.py](../tests/test_protocol_reporting.py)

## Research Context

This interpretation stays aligned with the literature already used in the
architecture audits:

- VQ-VAE remains the right discrete bottleneck for tile semantics under small
  datasets.
- latent diffusion remains the strongest teacher-style room generator in this
  stack class.
- masked iterative decoding is a reasonable research branch, but it only helps
  when the semantic contract is already strong.
- graph-first structured generation remains the correct high-level bias.

Relevant papers:

- van den Oord et al., `Neural Discrete Representation Learning`, NeurIPS 2017
- Razavi et al., `Generating Diverse High-Fidelity Images with VQ-VAE-2`, NeurIPS 2019
- Ho et al., `Denoising Diffusion Probabilistic Models`, NeurIPS 2020
- Song et al., `Denoising Diffusion Implicit Models`, ICLR 2021
- Rombach et al., `High-Resolution Image Synthesis with Latent Diffusion Models`, CVPR 2022
- Chang et al., `MaskGIT`, CVPR 2022
- Hu et al., `Graph2Plan`, CVPR 2020
- Shabani et al., `HouseDiffusion`, CVPR 2023
- Summerville et al., `Procedural Content Generation via Machine Learning`, IEEE TG 2018

The repo-specific judgments below are evidence-based from local artifacts, not
claimed directly by those papers.

## Key Findings

### 1. Diffusion remains the only production branch

Manual rich-topology compare v2:

- `generation_time_sec = 58.91`
- `repair_rate = 1.0000`
- `total_tiles_repaired = 368`
- `avg_final_graph_marker_overwrite_rate = 0.2917`
- `avg_neural_semantic_anchor_error = 0.0`
- `avg_final_pre_overlay_semantic_anchor_error = 7.875`
- `layout.primary_quality_metric = graph_edge_slot_adjacency_rate = 1.0`

Fixed-graph hybrid-default audit v2 over three seeds:

- `avg_repair_rate = 0.8611`
- `avg_total_tiles_repaired = 427.67`
- `avg_final_graph_marker_overwrite_rate = 0.2639`
- `avg_neural_semantic_anchor_error = 0.0`
- `unique_layout_count = 3`

Interpretation:

- the diffusion branch still gives the best overall structural reliability
- final layouts remain diverse across seeds
- the neural branch is still not placing all semantic anchors cleanly before
  overlay, so the hybrid contract still matters

### 2. Fast sampler is still a student, not a peer

Manual rich-topology compare v2:

- `generation_time_sec = 98.43`
- `repair_rate = 0.9167`
- `total_tiles_repaired = 494`
- `avg_final_graph_marker_overwrite_rate = 0.5000`
- `avg_final_pre_overlay_semantic_anchor_error = 13.5`
- `fast_sampler_teacher_fallback = 10`

Fixed-graph hybrid-default audit v2:

- `avg_repair_rate = 0.9444`
- `avg_total_tiles_repaired = 643.0`
- `avg_final_graph_marker_overwrite_rate = 0.2917`

Interpretation:

- the fast branch still depends heavily on teacher rescue
- it is not yet reliably cheaper in practice than the final diffusion export
- its semantic drift remains material

### 3. Masked-room improved in some semantic placement signals but is still guarded

Manual rich-topology compare v2:

- `generation_time_sec = 83.89`
- `repair_rate = 0.6667`
- `total_tiles_repaired = 396`
- `avg_final_graph_marker_overwrite_rate = 0.2500`
- `avg_final_pre_overlay_semantic_anchor_error = 6.75`
- `masked_room_teacher_fallback = 10`

Interpretation:

- some single-seed semantic-placement numbers look better than diffusion
- but the branch still relies on teacher fallback and does not yet behave like
  a trustworthy primary generator
- keep it as a guarded research branch

### 4. The ablations still support keeping the system hybrid

Earlier ablation summaries remain directionally consistent with the final run:

- `hybrid_weak_decode` produced large neural anchor errors (`21.375` to `22.5`)
  and much higher overwrite
- `strict_pure_neural` and `strict_pure_neural_no_fallback` stayed substantially
  worse on overwrite (`0.4167` to `0.6875`) with neural anchor error `22.5`
- `hybrid_default_v2` is materially better than the earlier hybrid-default pass
  on diffusion repair load and overwrite

Interpretation:

- pure-neural semantic placement is still not reliable enough
- deterministic overlay and fallback logic are still carrying real correctness
  value, not just legacy complexity

### 5. One protocol bug was found in the reporting layer

Bug:

- room-level code computed `final_post_overlay_semantic_anchor_avg_manhattan_error`
- dungeon-level summaries omitted the aggregate
- export summaries therefore could not prove post-overlay anchor error cleanly

Fix applied in:

- [src/pipeline/dungeon_pipeline.py](../src/pipeline/dungeon_pipeline.py)
- [scripts/run_fast_sampler_visual_audit.py](../scripts/run_fast_sampler_visual_audit.py)
- [scripts/export_semantic_anchor_end_to_end.py](../scripts/export_semantic_anchor_end_to_end.py)
- [scripts/run_fixed_graph_multi_seed_audit.py](../scripts/run_fixed_graph_multi_seed_audit.py)
- [scripts/export_manual_rich_topology_compare.py](../scripts/export_manual_rich_topology_compare.py)

The existing `v2` exports predate that reporting patch, so they should be
re-exported if the thesis/report needs the new aggregate field populated in the
artifact JSON.

## Verdict

### Does `codebook512 + puzzle-subtype conditioning` help the downstream stack?

Yes, but unevenly.

What improved:

- final diffusion export time dropped sharply versus the earlier run
- diffusion repair load improved
- hybrid-default multi-seed diffusion metrics improved over the earlier pass
- puzzle semantics are more explicit than the older generic puzzle-family path

What did not become true:

- fast sampler is still not a production-quality student
- masked-room is still not a trustworthy peer branch
- pure-neural semantic placement is still not strong enough to replace the
  hybrid overlay/repair contract

## Practical Recommendation

For the current repo state:

1. keep diffusion as the production branch
2. keep fast sampler and masked-room as research branches
3. do not spend the next pass on another tokenizer change first
4. spend the next pass on:
   - stronger teacher-to-student transfer
   - lower overlay dependence before final marker placement
   - richer learned puzzle semantics before resorting to symbolic rescue

Current practical judgment:

- `codebook512` is now a reasonable canonical tokenizer candidate for the
  diffusion branch
- it is still **not** proof that the full downstream stack has converged to the
  same quality level across all branchess
