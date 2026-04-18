# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_interactionseq_v8\diffusion_cfg3_logic0_steps50`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_interactionseq_v8\fast_cfg3_logic0_steps4`

Compared rooms: 12
Total changed tiles: 163

## Worst Rooms

- room_1: changed_tiles=25, change_ratio=0.1420, change_types={'structure_to_floor': 25}
- room_3: changed_tiles=23, change_ratio=0.1307, change_types={'structure_to_floor': 23}
- room_7: changed_tiles=23, change_ratio=0.1307, change_types={'structure_to_floor': 23}
- room_5: changed_tiles=21, change_ratio=0.1193, change_types={'structure_to_floor': 20, 'W_to_B': 1}
- room_8: changed_tiles=20, change_ratio=0.1136, change_types={'structure_to_floor': 20}
- room_9: changed_tiles=20, change_ratio=0.1136, change_types={'floor_to_structure': 18, 'structure_to_floor': 2}
- room_4: changed_tiles=15, change_ratio=0.0852, change_types={'structure_to_floor': 12, 'floor_to_structure': 3}
- room_6: changed_tiles=8, change_ratio=0.0455, change_types={'floor_to_structure': 8}

## Aggregate Change Types

- structure_to_floor: 129
- floor_to_structure: 33
- W_to_B: 1
