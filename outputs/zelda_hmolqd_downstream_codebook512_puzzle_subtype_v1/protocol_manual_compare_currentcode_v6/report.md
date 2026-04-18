# Manual Rich Topology Comparison

This export uses one hand-authored topology with explicit semantic content and fixed room positions.
That makes branch comparison much more precise than comparing on changing Block I topologies.

## Runtime Overrides

- none

## Variants

- `diffusion_cfg3_logic0_steps50`: gen_time=117.42s, repair_rate=0.833, total_tiles_repaired=415, overwrite=0.250, post_overlay_anchor_error=0.000, graph_edge_slot_adjacency_rate=1.000, astar_solvable=True, softlock_safe=True, cbs_success=True, cbs_confusion_ratio=19.244
- `fast_cfg3_logic0_steps4`: gen_time=124.78s, repair_rate=0.833, total_tiles_repaired=468, overwrite=0.167, post_overlay_anchor_error=0.000, graph_edge_slot_adjacency_rate=1.000, astar_solvable=True, softlock_safe=True, cbs_success=True, cbs_confusion_ratio=19.244
- `masked_room_full`: gen_time=86.66s, repair_rate=0.750, total_tiles_repaired=381, overwrite=0.250, post_overlay_anchor_error=0.000, graph_edge_slot_adjacency_rate=1.000, astar_solvable=True, softlock_safe=True, cbs_success=True, cbs_confusion_ratio=19.244

## Pairwise Room Diff Audits

- `diffusion_vs_fast_sampler`: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_currentcode_v6\comparisons\diffusion_vs_fast_sampler\summary.json`
- `diffusion_vs_masked_room`: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_currentcode_v6\comparisons\diffusion_vs_masked_room\summary.json`
- `fast_sampler_vs_masked_room`: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_currentcode_v6\comparisons\fast_sampler_vs_masked_room\summary.json`

## Key Artifacts

- mission graph: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_currentcode_v6\mission_graph.json`
- mission graph PNG: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_currentcode_v6\mission_graph_layout.png`
- graph summary: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_currentcode_v6\graph_summary.json`
- overall summary: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_currentcode_v6\summary.json`
- search algorithm comparison: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_currentcode_v6\search_algorithm_comparison.json`
- dungeon alignment comparison: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_currentcode_v6\dungeon_alignment_comparison.png`
- rooms comparison: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_currentcode_v6\rooms_sheet_comparison.png`
