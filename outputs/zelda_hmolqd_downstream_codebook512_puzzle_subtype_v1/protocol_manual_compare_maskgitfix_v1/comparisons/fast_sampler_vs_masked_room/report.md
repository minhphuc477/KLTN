# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_maskgitfix_v1\fast_cfg3_logic0_steps4`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_maskgitfix_v1\masked_room_full`

Compared rooms: 12
Total changed tiles: 131

## Worst Rooms

- room_0: changed_tiles=35, change_ratio=0.1989, change_types={'floor_to_structure': 35}
- room_2: changed_tiles=21, change_ratio=0.1193, change_types={'filled_to_void': 2, 'structure_to_floor': 8, 'void_to_filled': 8, 'floor_to_structure': 3}
- room_1: changed_tiles=20, change_ratio=0.1136, change_types={'floor_to_structure': 15, 'void_to_filled': 2, 'filled_to_void': 3}
- room_9: changed_tiles=16, change_ratio=0.0909, change_types={'void_to_filled': 7, 'filled_to_void': 1, 'structure_to_floor': 5, 'B_to_W': 1, 'floor_to_structure': 2}
- room_6: changed_tiles=15, change_ratio=0.0852, change_types={'floor_to_structure': 5, 'filled_to_void': 10}
- room_5: changed_tiles=12, change_ratio=0.0682, change_types={'structure_to_floor': 12}
- room_4: changed_tiles=8, change_ratio=0.0455, change_types={'floor_to_structure': 3, 'structure_to_floor': 5}
- room_11: changed_tiles=3, change_ratio=0.0170, change_types={'floor_to_structure': 3}

## Aggregate Change Types

- floor_to_structure: 66
- structure_to_floor: 30
- void_to_filled: 17
- filled_to_void: 17
- B_to_W: 1
