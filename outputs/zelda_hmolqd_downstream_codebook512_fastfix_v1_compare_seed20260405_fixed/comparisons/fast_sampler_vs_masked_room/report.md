# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_fastfix_v1_compare_seed20260405_fixed\fast_cfg3_logic0_steps4`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_fastfix_v1_compare_seed20260405_fixed\masked_room_full`

Compared rooms: 12
Total changed tiles: 179

## Worst Rooms

- room_2: changed_tiles=46, change_ratio=0.2614, change_types={'structure_to_floor': 34, 'void_to_filled': 12}
- room_0: changed_tiles=35, change_ratio=0.1989, change_types={'structure_to_floor': 35}
- room_1: changed_tiles=32, change_ratio=0.1818, change_types={'floor_to_structure': 27, 'filled_to_void': 5}
- room_8: changed_tiles=20, change_ratio=0.1136, change_types={'floor_to_structure': 2, 'structure_to_floor': 16, 'void_to_filled': 2}
- room_4: changed_tiles=13, change_ratio=0.0739, change_types={'structure_to_floor': 10, 'floor_to_structure': 3}
- room_5: changed_tiles=11, change_ratio=0.0625, change_types={'floor_to_structure': 10, 'filled_to_void': 1}
- room_10: changed_tiles=8, change_ratio=0.0455, change_types={'void_to_filled': 2, 'structure_to_floor': 6}
- room_6: changed_tiles=6, change_ratio=0.0341, change_types={'void_to_filled': 1, 'filled_to_void': 4, 'floor_to_structure': 1}

## Aggregate Change Types

- structure_to_floor: 104
- floor_to_structure: 48
- void_to_filled: 17
- filled_to_void: 10
