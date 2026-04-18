# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_currentcode_v6\diffusion_cfg3_logic0_steps50`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_currentcode_v6\masked_room_full`

Compared rooms: 12
Total changed tiles: 166

## Worst Rooms

- room_0: changed_tiles=35, change_ratio=0.1989, change_types={'floor_to_structure': 35}
- room_3: changed_tiles=27, change_ratio=0.1534, change_types={'structure_to_floor': 27}
- room_6: changed_tiles=18, change_ratio=0.1023, change_types={'structure_to_floor': 18}
- room_4: changed_tiles=17, change_ratio=0.0966, change_types={'floor_to_structure': 4, 'structure_to_floor': 13}
- room_5: changed_tiles=17, change_ratio=0.0966, change_types={'floor_to_structure': 7, 'structure_to_floor': 10}
- room_10: changed_tiles=17, change_ratio=0.0966, change_types={'floor_to_structure': 1, 'structure_to_floor': 16}
- room_9: changed_tiles=9, change_ratio=0.0511, change_types={'structure_to_floor': 5, 'floor_to_structure': 4}
- room_2: changed_tiles=7, change_ratio=0.0398, change_types={'structure_to_floor': 7}

## Aggregate Change Types

- structure_to_floor: 104
- floor_to_structure: 62
