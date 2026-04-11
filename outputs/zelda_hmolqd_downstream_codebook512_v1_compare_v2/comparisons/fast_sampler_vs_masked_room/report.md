# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_v1_compare_v2\fast_cfg3_logic0_steps4`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_v1_compare_v2\masked_room_full`

Compared rooms: 12
Total changed tiles: 206

## Worst Rooms

- room_2: changed_tiles=47, change_ratio=0.2670, change_types={'floor_to_structure': 29, 'filled_to_void': 18}
- room_3: changed_tiles=40, change_ratio=0.2273, change_types={'structure_to_floor': 29, 'void_to_filled': 11}
- room_5: changed_tiles=36, change_ratio=0.2045, change_types={'structure_to_floor': 35, 'void_to_filled': 1}
- room_6: changed_tiles=35, change_ratio=0.1989, change_types={'floor_to_structure': 30, 'filled_to_void': 5}
- room_1: changed_tiles=29, change_ratio=0.1648, change_types={'structure_to_floor': 26, 'void_to_filled': 3}
- room_11: changed_tiles=10, change_ratio=0.0568, change_types={'floor_to_structure': 4, 'structure_to_floor': 6}
- room_8: changed_tiles=4, change_ratio=0.0227, change_types={'structure_to_floor': 4}
- room_7: changed_tiles=3, change_ratio=0.0170, change_types={'structure_to_floor': 2, 'floor_to_structure': 1}

## Aggregate Change Types

- structure_to_floor: 102
- floor_to_structure: 66
- filled_to_void: 23
- void_to_filled: 15
