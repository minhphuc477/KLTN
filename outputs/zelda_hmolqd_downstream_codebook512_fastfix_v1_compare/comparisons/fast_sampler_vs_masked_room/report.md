# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_fastfix_v1_compare\fast_cfg3_logic0_steps4`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_fastfix_v1_compare\masked_room_full`

Compared rooms: 12
Total changed tiles: 146

## Worst Rooms

- room_5: changed_tiles=38, change_ratio=0.2159, change_types={'structure_to_floor': 34, 'void_to_filled': 3, 'W_to_B': 1}
- room_0: changed_tiles=35, change_ratio=0.1989, change_types={'floor_to_structure': 35}
- room_1: changed_tiles=30, change_ratio=0.1705, change_types={'structure_to_floor': 27, 'void_to_filled': 3}
- room_2: changed_tiles=14, change_ratio=0.0795, change_types={'structure_to_floor': 1, 'floor_to_structure': 1, 'filled_to_void': 10, 'void_to_filled': 2}
- room_6: changed_tiles=13, change_ratio=0.0739, change_types={'floor_to_structure': 3, 'filled_to_void': 5, 'void_to_filled': 4, 'structure_to_floor': 1}
- room_11: changed_tiles=6, change_ratio=0.0341, change_types={'floor_to_structure': 4, 'structure_to_floor': 2}
- room_8: changed_tiles=5, change_ratio=0.0284, change_types={'structure_to_floor': 5}
- room_7: changed_tiles=3, change_ratio=0.0170, change_types={'structure_to_floor': 3}

## Aggregate Change Types

- structure_to_floor: 73
- floor_to_structure: 45
- filled_to_void: 15
- void_to_filled: 12
- W_to_B: 1
