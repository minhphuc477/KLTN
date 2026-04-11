# Manual Rich Topology Comparison

This export uses one hand-authored topology with explicit semantic content and fixed room positions.
That makes branch comparison much more precise than comparing on changing Block I topologies.

## Runtime Overrides

- none

## Variants

- `diffusion_cfg3_logic0_steps50`: gen_time=52.99s, repair_rate=1.000, total_tiles_repaired=373
- `fast_cfg3_logic0_steps4`: gen_time=91.35s, repair_rate=1.000, total_tiles_repaired=590
- `masked_room_full`: gen_time=68.08s, repair_rate=0.833, total_tiles_repaired=578

## Pairwise Room Diff Audits

- `diffusion_vs_fast_sampler`: `outputs\zelda_hmolqd_downstream_codebook512_fastfix_v1_compare_seed20260404_fixed\comparisons\diffusion_vs_fast_sampler\summary.json`
- `diffusion_vs_masked_room`: `outputs\zelda_hmolqd_downstream_codebook512_fastfix_v1_compare_seed20260404_fixed\comparisons\diffusion_vs_masked_room\summary.json`
- `fast_sampler_vs_masked_room`: `outputs\zelda_hmolqd_downstream_codebook512_fastfix_v1_compare_seed20260404_fixed\comparisons\fast_sampler_vs_masked_room\summary.json`

## Key Artifacts

- mission graph: `outputs\zelda_hmolqd_downstream_codebook512_fastfix_v1_compare_seed20260404_fixed\mission_graph.json`
- mission graph PNG: `outputs\zelda_hmolqd_downstream_codebook512_fastfix_v1_compare_seed20260404_fixed\mission_graph_layout.png`
- graph summary: `outputs\zelda_hmolqd_downstream_codebook512_fastfix_v1_compare_seed20260404_fixed\graph_summary.json`
- overall summary: `outputs\zelda_hmolqd_downstream_codebook512_fastfix_v1_compare_seed20260404_fixed\summary.json`
- dungeon alignment comparison: `outputs\zelda_hmolqd_downstream_codebook512_fastfix_v1_compare_seed20260404_fixed\dungeon_alignment_comparison.png`
- rooms comparison: `outputs\zelda_hmolqd_downstream_codebook512_fastfix_v1_compare_seed20260404_fixed\rooms_sheet_comparison.png`
