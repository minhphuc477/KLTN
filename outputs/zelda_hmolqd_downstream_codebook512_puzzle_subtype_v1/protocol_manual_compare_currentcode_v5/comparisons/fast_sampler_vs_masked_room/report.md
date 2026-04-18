# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_currentcode_v5\fast_cfg3_logic0_steps4`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_currentcode_v5\masked_room_full`

Compared rooms: 12
Total changed tiles: 165

## Worst Rooms

- room_2: changed_tiles=47, change_ratio=0.2670, change_types={'floor_to_structure': 34, 'filled_to_void': 13}
- room_0: changed_tiles=35, change_ratio=0.1989, change_types={'floor_to_structure': 35}
- room_5: changed_tiles=19, change_ratio=0.1080, change_types={'floor_to_structure': 9, 'structure_to_floor': 9, 'filled_to_void': 1}
- room_10: changed_tiles=18, change_ratio=0.1023, change_types={'floor_to_structure': 1, 'structure_to_floor': 17}
- room_4: changed_tiles=17, change_ratio=0.0966, change_types={'floor_to_structure': 4, 'structure_to_floor': 13}
- room_8: changed_tiles=8, change_ratio=0.0455, change_types={'floor_to_structure': 7, 'structure_to_floor': 1}
- room_1: changed_tiles=6, change_ratio=0.0341, change_types={'floor_to_structure': 2, 'void_to_filled': 2, 'structure_to_floor': 1, 'filled_to_void': 1}
- room_9: changed_tiles=6, change_ratio=0.0341, change_types={'structure_to_floor': 3, 'floor_to_structure': 3}

## Aggregate Change Types

- floor_to_structure: 104
- structure_to_floor: 44
- filled_to_void: 15
- void_to_filled: 2
