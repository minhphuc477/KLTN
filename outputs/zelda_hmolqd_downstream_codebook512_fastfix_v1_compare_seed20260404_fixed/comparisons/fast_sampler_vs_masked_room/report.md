# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_fastfix_v1_compare_seed20260404_fixed\fast_cfg3_logic0_steps4`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_fastfix_v1_compare_seed20260404_fixed\masked_room_full`

Compared rooms: 12
Total changed tiles: 163

## Worst Rooms

- room_2: changed_tiles=42, change_ratio=0.2386, change_types={'floor_to_structure': 31, 'filled_to_void': 11}
- room_0: changed_tiles=35, change_ratio=0.1989, change_types={'floor_to_structure': 35}
- room_1: changed_tiles=32, change_ratio=0.1818, change_types={'structure_to_floor': 25, 'void_to_filled': 7}
- room_3: changed_tiles=29, change_ratio=0.1648, change_types={'structure_to_floor': 24, 'void_to_filled': 4, 'floor_to_structure': 1}
- room_6: changed_tiles=7, change_ratio=0.0398, change_types={'floor_to_structure': 2, 'void_to_filled': 2, 'filled_to_void': 1, 'W_to_B': 1, 'structure_to_floor': 1}
- room_5: changed_tiles=5, change_ratio=0.0284, change_types={'structure_to_floor': 5}
- room_4: changed_tiles=4, change_ratio=0.0227, change_types={'floor_to_structure': 3, 'structure_to_floor': 1}
- room_7: changed_tiles=3, change_ratio=0.0170, change_types={'structure_to_floor': 1, 'floor_to_structure': 2}

## Aggregate Change Types

- floor_to_structure: 78
- structure_to_floor: 59
- void_to_filled: 13
- filled_to_void: 12
- W_to_B: 1
