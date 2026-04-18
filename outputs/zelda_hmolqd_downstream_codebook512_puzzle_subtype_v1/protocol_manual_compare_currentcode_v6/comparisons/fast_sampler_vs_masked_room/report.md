# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_currentcode_v6\fast_cfg3_logic0_steps4`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_currentcode_v6\masked_room_full`

Compared rooms: 12
Total changed tiles: 109

## Worst Rooms

- room_0: changed_tiles=35, change_ratio=0.1989, change_types={'floor_to_structure': 35}
- room_5: changed_tiles=18, change_ratio=0.1023, change_types={'floor_to_structure': 7, 'structure_to_floor': 11}
- room_10: changed_tiles=18, change_ratio=0.1023, change_types={'floor_to_structure': 1, 'structure_to_floor': 17}
- room_4: changed_tiles=17, change_ratio=0.0966, change_types={'floor_to_structure': 4, 'structure_to_floor': 13}
- room_9: changed_tiles=7, change_ratio=0.0398, change_types={'structure_to_floor': 3, 'floor_to_structure': 4}
- room_8: changed_tiles=5, change_ratio=0.0284, change_types={'floor_to_structure': 4, 'structure_to_floor': 1}
- room_11: changed_tiles=4, change_ratio=0.0227, change_types={'floor_to_structure': 4}
- room_3: changed_tiles=3, change_ratio=0.0170, change_types={'floor_to_structure': 3}

## Aggregate Change Types

- floor_to_structure: 64
- structure_to_floor: 45
