# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_maskgitfix_validation_v2\fast_cfg3_logic0_steps4`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_maskgitfix_validation_v2\masked_room_full`

Compared rooms: 12
Total changed tiles: 181

## Worst Rooms

- room_1: changed_tiles=51, change_ratio=0.2898, change_types={'floor_to_structure': 42, 'filled_to_void': 9}
- room_0: changed_tiles=36, change_ratio=0.2045, change_types={'floor_to_structure': 34, 'structure_to_floor': 2}
- room_7: changed_tiles=21, change_ratio=0.1193, change_types={'structure_to_floor': 21}
- room_9: changed_tiles=19, change_ratio=0.1080, change_types={'structure_to_floor': 6, 'filled_to_void': 1, 'void_to_filled': 9, 'W_to_B': 1, 'floor_to_structure': 2}
- room_5: changed_tiles=15, change_ratio=0.0852, change_types={'structure_to_floor': 13, 'B_to_W': 1, 'floor_to_structure': 1}
- room_6: changed_tiles=14, change_ratio=0.0795, change_types={'floor_to_structure': 8, 'filled_to_void': 6}
- room_4: changed_tiles=12, change_ratio=0.0682, change_types={'floor_to_structure': 3, 'structure_to_floor': 9}
- room_2: changed_tiles=9, change_ratio=0.0511, change_types={'filled_to_void': 1, 'void_to_filled': 5, 'structure_to_floor': 1, 'floor_to_structure': 2}

## Aggregate Change Types

- floor_to_structure: 95
- structure_to_floor: 52
- filled_to_void: 17
- void_to_filled: 15
- B_to_W: 1
- W_to_B: 1
