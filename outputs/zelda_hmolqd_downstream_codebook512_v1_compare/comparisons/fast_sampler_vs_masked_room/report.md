# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_v1_compare\fast_cfg3_logic0_steps4`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_v1_compare\masked_room_full`

Compared rooms: 12
Total changed tiles: 128

## Worst Rooms

- room_5: changed_tiles=36, change_ratio=0.2045, change_types={'structure_to_floor': 34, 'void_to_filled': 2}
- room_6: changed_tiles=34, change_ratio=0.1932, change_types={'floor_to_structure': 29, 'filled_to_void': 5}
- room_1: changed_tiles=30, change_ratio=0.1705, change_types={'structure_to_floor': 27, 'void_to_filled': 3}
- room_11: changed_tiles=11, change_ratio=0.0625, change_types={'floor_to_structure': 4, 'structure_to_floor': 7}
- room_2: changed_tiles=8, change_ratio=0.0455, change_types={'structure_to_floor': 2, 'void_to_filled': 3, 'filled_to_void': 3}
- room_8: changed_tiles=4, change_ratio=0.0227, change_types={'structure_to_floor': 4}
- room_7: changed_tiles=3, change_ratio=0.0170, change_types={'structure_to_floor': 3}
- room_4: changed_tiles=2, change_ratio=0.0114, change_types={'floor_to_structure': 2}

## Aggregate Change Types

- structure_to_floor: 77
- floor_to_structure: 35
- void_to_filled: 8
- filled_to_void: 8
