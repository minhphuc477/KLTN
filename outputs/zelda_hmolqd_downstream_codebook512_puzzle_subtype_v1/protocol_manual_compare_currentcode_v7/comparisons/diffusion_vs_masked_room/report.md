# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_currentcode_v7\diffusion_cfg3_logic0_steps50`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_currentcode_v7\masked_room_full`

Compared rooms: 12
Total changed tiles: 170

## Worst Rooms

- room_6: changed_tiles=47, change_ratio=0.2670, change_types={'structure_to_floor': 47}
- room_0: changed_tiles=35, change_ratio=0.1989, change_types={'floor_to_structure': 35}
- room_5: changed_tiles=24, change_ratio=0.1364, change_types={'floor_to_structure': 7, 'structure_to_floor': 17}
- room_4: changed_tiles=17, change_ratio=0.0966, change_types={'floor_to_structure': 4, 'structure_to_floor': 13}
- room_11: changed_tiles=14, change_ratio=0.0795, change_types={'floor_to_structure': 3, 'structure_to_floor': 11}
- room_9: changed_tiles=12, change_ratio=0.0682, change_types={'structure_to_floor': 10, 'W_to_B': 1, 'floor_to_structure': 1}
- room_7: changed_tiles=11, change_ratio=0.0625, change_types={'floor_to_structure': 2, 'structure_to_floor': 9}
- room_10: changed_tiles=4, change_ratio=0.0227, change_types={'floor_to_structure': 1, 'structure_to_floor': 3}

## Aggregate Change Types

- structure_to_floor: 110
- floor_to_structure: 59
- W_to_B: 1
