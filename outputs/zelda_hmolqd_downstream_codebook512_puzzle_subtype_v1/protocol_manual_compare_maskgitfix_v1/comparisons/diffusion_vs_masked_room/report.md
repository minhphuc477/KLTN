# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_maskgitfix_v1\diffusion_cfg3_logic0_steps50`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_maskgitfix_v1\masked_room_full`

Compared rooms: 12
Total changed tiles: 179

## Worst Rooms

- room_0: changed_tiles=35, change_ratio=0.1989, change_types={'floor_to_structure': 35}
- room_1: changed_tiles=30, change_ratio=0.1705, change_types={'floor_to_structure': 20, 'filled_to_void': 9, 'B_to_W': 1}
- room_6: changed_tiles=27, change_ratio=0.1534, change_types={'filled_to_void': 22, 'floor_to_structure': 4, 'structure_to_floor': 1}
- room_9: changed_tiles=25, change_ratio=0.1420, change_types={'floor_to_structure': 18, 'filled_to_void': 6, 'B_to_W': 1}
- room_3: changed_tiles=23, change_ratio=0.1307, change_types={'structure_to_floor': 23}
- room_5: changed_tiles=12, change_ratio=0.0682, change_types={'structure_to_floor': 12}
- room_11: changed_tiles=10, change_ratio=0.0568, change_types={'floor_to_structure': 1, 'structure_to_floor': 9}
- room_2: changed_tiles=9, change_ratio=0.0511, change_types={'filled_to_void': 7, 'floor_to_structure': 2}

## Aggregate Change Types

- floor_to_structure: 83
- structure_to_floor: 49
- filled_to_void: 45
- B_to_W: 2
