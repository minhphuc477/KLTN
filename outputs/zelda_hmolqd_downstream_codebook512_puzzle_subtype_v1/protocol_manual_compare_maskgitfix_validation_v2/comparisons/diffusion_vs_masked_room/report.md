# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_maskgitfix_validation_v2\diffusion_cfg3_logic0_steps50`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_maskgitfix_validation_v2\masked_room_full`

Compared rooms: 12
Total changed tiles: 203

## Worst Rooms

- room_0: changed_tiles=35, change_ratio=0.1989, change_types={'floor_to_structure': 35}
- room_1: changed_tiles=30, change_ratio=0.1705, change_types={'floor_to_structure': 21, 'filled_to_void': 9}
- room_6: changed_tiles=30, change_ratio=0.1705, change_types={'filled_to_void': 18, 'structure_to_floor': 4, 'floor_to_structure': 8}
- room_3: changed_tiles=23, change_ratio=0.1307, change_types={'structure_to_floor': 23}
- room_7: changed_tiles=22, change_ratio=0.1250, change_types={'structure_to_floor': 22}
- room_9: changed_tiles=21, change_ratio=0.1193, change_types={'filled_to_void': 4, 'floor_to_structure': 17}
- room_5: changed_tiles=16, change_ratio=0.0909, change_types={'structure_to_floor': 13, 'B_to_W': 1, 'floor_to_structure': 1, 'W_to_B': 1}
- room_11: changed_tiles=10, change_ratio=0.0568, change_types={'floor_to_structure': 1, 'structure_to_floor': 9}

## Aggregate Change Types

- floor_to_structure: 88
- structure_to_floor: 75
- filled_to_void: 38
- B_to_W: 1
- W_to_B: 1
