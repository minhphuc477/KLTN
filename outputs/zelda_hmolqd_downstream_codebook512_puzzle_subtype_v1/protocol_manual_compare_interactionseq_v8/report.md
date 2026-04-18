# Manual Rich Topology Comparison

This export uses one hand-authored topology with explicit semantic content and fixed room positions.
That makes branch comparison much more precise than comparing on changing Block I topologies.

## Runtime Overrides

- none

## Variants

- `diffusion_cfg3_logic0_steps50`: gen_time=41.37s, repair_rate=0.917, total_tiles_repaired=177, overwrite=0.250, post_overlay_anchor_error=0.000, graph_edge_slot_adjacency_rate=1.000, astar_solvable=True, softlock_safe=True, cbs_success=True, cbs_confusion_ratio=2.044
- `fast_cfg3_logic0_steps4`: gen_time=79.44s, repair_rate=0.833, total_tiles_repaired=438, overwrite=0.167, post_overlay_anchor_error=0.000, graph_edge_slot_adjacency_rate=1.000, astar_solvable=True, softlock_safe=True, cbs_success=False, cbs_confusion_ratio=inf
- `masked_room_full`: gen_time=58.50s, repair_rate=0.750, total_tiles_repaired=384, overwrite=0.417, post_overlay_anchor_error=0.000, graph_edge_slot_adjacency_rate=1.000, astar_solvable=True, softlock_safe=True, cbs_success=False, cbs_confusion_ratio=inf

## Pairwise Room Diff Audits

- `diffusion_vs_fast_sampler`: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_interactionseq_v8\comparisons\diffusion_vs_fast_sampler\summary.json`
- `diffusion_vs_masked_room`: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_interactionseq_v8\comparisons\diffusion_vs_masked_room\summary.json`
- `fast_sampler_vs_masked_room`: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_interactionseq_v8\comparisons\fast_sampler_vs_masked_room\summary.json`

## Key Artifacts

- mission graph: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_interactionseq_v8\mission_graph.json`
- mission graph PNG: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_interactionseq_v8\mission_graph_layout.png`
- graph summary: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_interactionseq_v8\graph_summary.json`
- overall summary: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_interactionseq_v8\summary.json`
- search algorithm comparison: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_interactionseq_v8\search_algorithm_comparison.json`
- dungeon alignment comparison: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_interactionseq_v8\dungeon_alignment_comparison.png`
- rooms comparison: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_interactionseq_v8\rooms_sheet_comparison.png`
