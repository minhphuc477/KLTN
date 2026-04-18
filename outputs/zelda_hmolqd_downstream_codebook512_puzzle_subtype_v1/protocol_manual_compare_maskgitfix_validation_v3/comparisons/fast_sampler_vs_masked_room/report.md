# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_maskgitfix_validation_v3\fast_cfg3_logic0_steps4`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_maskgitfix_validation_v3\masked_room_full`

Compared rooms: 12
Total changed tiles: 81

## Worst Rooms

- room_0: changed_tiles=35, change_ratio=0.1989, change_types={'floor_to_structure': 35}
- room_6: changed_tiles=21, change_ratio=0.1193, change_types={'structure_to_floor': 21}
- room_3: changed_tiles=4, change_ratio=0.0227, change_types={'structure_to_floor': 4}
- room_5: changed_tiles=4, change_ratio=0.0227, change_types={'floor_to_structure': 3, 'B_to_W': 1}
- room_9: changed_tiles=4, change_ratio=0.0227, change_types={'structure_to_floor': 4}
- room_7: changed_tiles=3, change_ratio=0.0170, change_types={'structure_to_floor': 3}
- room_1: changed_tiles=2, change_ratio=0.0114, change_types={'structure_to_floor': 2}
- room_2: changed_tiles=2, change_ratio=0.0114, change_types={'structure_to_floor': 2}

## Aggregate Change Types

- floor_to_structure: 42
- structure_to_floor: 37
- B_to_W: 1
- void_to_filled: 1
