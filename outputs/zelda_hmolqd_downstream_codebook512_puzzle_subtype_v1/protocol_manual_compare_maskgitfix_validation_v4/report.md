# Manual Rich Topology Comparison

This export uses one hand-authored topology with explicit semantic content and fixed room positions.
That makes branch comparison much more precise than comparing on changing Block I topologies.

## Runtime Overrides

- none

## Variants

- `diffusion_cfg3_logic0_steps50`: gen_time=22.35s, repair_rate=0.917, total_tiles_repaired=270, overwrite=0.167, post_overlay_anchor_error=0.000, graph_edge_slot_adjacency_rate=1.000, astar_solvable=True, softlock_safe=True, cbs_success=True, cbs_confusion_ratio=9.067
- `fast_cfg3_logic0_steps4`: gen_time=52.83s, repair_rate=1.000, total_tiles_repaired=275, overwrite=0.208, post_overlay_anchor_error=0.000, graph_edge_slot_adjacency_rate=1.000, astar_solvable=True, softlock_safe=True, cbs_success=True, cbs_confusion_ratio=8.956
- `masked_room_full`: gen_time=53.90s, repair_rate=0.833, total_tiles_repaired=243, overwrite=0.083, post_overlay_anchor_error=0.000, graph_edge_slot_adjacency_rate=1.000, astar_solvable=True, softlock_safe=True, cbs_success=True, cbs_confusion_ratio=8.956

## Pairwise Room Diff Audits

- `diffusion_vs_fast_sampler`: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_maskgitfix_validation_v4\comparisons\diffusion_vs_fast_sampler\summary.json`
- `diffusion_vs_masked_room`: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_maskgitfix_validation_v4\comparisons\diffusion_vs_masked_room\summary.json`
- `fast_sampler_vs_masked_room`: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_maskgitfix_validation_v4\comparisons\fast_sampler_vs_masked_room\summary.json`

## Key Artifacts

- mission graph: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_maskgitfix_validation_v4\mission_graph.json`
- mission graph PNG: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_maskgitfix_validation_v4\mission_graph_layout.png`
- graph summary: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_maskgitfix_validation_v4\graph_summary.json`
- overall summary: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_maskgitfix_validation_v4\summary.json`
- dungeon alignment comparison: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_maskgitfix_validation_v4\dungeon_alignment_comparison.png`
- rooms comparison: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_maskgitfix_validation_v4\rooms_sheet_comparison.png`
