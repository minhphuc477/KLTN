# Manual Rich Topology Comparison

This export uses one hand-authored topology with explicit semantic content and fixed room positions.
That makes branch comparison much more precise than comparing on changing Block I topologies.

## Runtime Overrides

- none

## Variants

- `diffusion_cfg3_logic0_steps50`: gen_time=103.19s, repair_rate=0.833, total_tiles_repaired=243
- `fast_cfg3_logic0_steps4`: gen_time=157.81s, repair_rate=0.917, total_tiles_repaired=523
- `masked_room_full`: gen_time=437.69s, repair_rate=0.833, total_tiles_repaired=593

## Pairwise Room Diff Audits

- `diffusion_vs_fast_sampler`: `outputs\zelda_hmolqd_downstream_codebook512_v1_compare\comparisons\diffusion_vs_fast_sampler\summary.json`
- `diffusion_vs_masked_room`: `outputs\zelda_hmolqd_downstream_codebook512_v1_compare\comparisons\diffusion_vs_masked_room\summary.json`
- `fast_sampler_vs_masked_room`: `outputs\zelda_hmolqd_downstream_codebook512_v1_compare\comparisons\fast_sampler_vs_masked_room\summary.json`

## Key Artifacts

- mission graph: `outputs\zelda_hmolqd_downstream_codebook512_v1_compare\mission_graph.json`
- mission graph PNG: `outputs\zelda_hmolqd_downstream_codebook512_v1_compare\mission_graph_layout.png`
- graph summary: `outputs\zelda_hmolqd_downstream_codebook512_v1_compare\graph_summary.json`
- overall summary: `outputs\zelda_hmolqd_downstream_codebook512_v1_compare\summary.json`
- dungeon alignment comparison: `outputs\zelda_hmolqd_downstream_codebook512_v1_compare\dungeon_alignment_comparison.png`
- rooms comparison: `outputs\zelda_hmolqd_downstream_codebook512_v1_compare\rooms_sheet_comparison.png`
