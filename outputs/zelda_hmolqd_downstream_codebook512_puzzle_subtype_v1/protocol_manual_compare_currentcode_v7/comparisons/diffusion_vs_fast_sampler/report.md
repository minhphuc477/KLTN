# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_currentcode_v7\diffusion_cfg3_logic0_steps50`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_currentcode_v7\fast_cfg3_logic0_steps4`

Compared rooms: 12
Total changed tiles: 131

## Worst Rooms

- room_6: changed_tiles=47, change_ratio=0.2670, change_types={'structure_to_floor': 47}
- room_5: changed_tiles=19, change_ratio=0.1080, change_types={'structure_to_floor': 14, 'floor_to_structure': 5}
- room_9: changed_tiles=18, change_ratio=0.1023, change_types={'floor_to_structure': 4, 'structure_to_floor': 13, 'W_to_B': 1}
- room_10: changed_tiles=14, change_ratio=0.0795, change_types={'floor_to_structure': 14}
- room_11: changed_tiles=14, change_ratio=0.0795, change_types={'floor_to_structure': 1, 'structure_to_floor': 13}
- room_7: changed_tiles=11, change_ratio=0.0625, change_types={'floor_to_structure': 2, 'structure_to_floor': 9}
- room_8: changed_tiles=8, change_ratio=0.0455, change_types={'structure_to_floor': 6, 'floor_to_structure': 2}
- room_0: changed_tiles=0, change_ratio=0.0000, change_types={}

## Aggregate Change Types

- structure_to_floor: 102
- floor_to_structure: 28
- W_to_B: 1
