# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_maskgitfix_validation_v4\fast_cfg3_logic0_steps4`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_maskgitfix_validation_v4\masked_room_full`

Compared rooms: 12
Total changed tiles: 77

## Worst Rooms

- room_0: changed_tiles=35, change_ratio=0.1989, change_types={'floor_to_structure': 35}
- room_6: changed_tiles=23, change_ratio=0.1307, change_types={'structure_to_floor': 23}
- room_9: changed_tiles=7, change_ratio=0.0398, change_types={'structure_to_floor': 5, 'floor_to_structure': 2}
- room_5: changed_tiles=3, change_ratio=0.0170, change_types={'floor_to_structure': 3}
- room_4: changed_tiles=2, change_ratio=0.0114, change_types={'floor_to_structure': 1, 'structure_to_floor': 1}
- room_7: changed_tiles=2, change_ratio=0.0114, change_types={'structure_to_floor': 2}
- room_11: changed_tiles=2, change_ratio=0.0114, change_types={'floor_to_structure': 2}
- room_3: changed_tiles=1, change_ratio=0.0057, change_types={'filled_to_void': 1}

## Aggregate Change Types

- floor_to_structure: 44
- structure_to_floor: 31
- filled_to_void: 1
- void_to_filled: 1
