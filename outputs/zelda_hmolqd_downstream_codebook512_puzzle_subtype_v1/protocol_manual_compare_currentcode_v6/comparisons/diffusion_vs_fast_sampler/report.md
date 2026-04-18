# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_currentcode_v6\diffusion_cfg3_logic0_steps50`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_currentcode_v6\fast_cfg3_logic0_steps4`

Compared rooms: 12
Total changed tiles: 82

## Worst Rooms

- room_3: changed_tiles=30, change_ratio=0.1705, change_types={'structure_to_floor': 30}
- room_6: changed_tiles=18, change_ratio=0.1023, change_types={'structure_to_floor': 18}
- room_9: changed_tiles=8, change_ratio=0.0455, change_types={'floor_to_structure': 3, 'structure_to_floor': 5}
- room_2: changed_tiles=7, change_ratio=0.0398, change_types={'structure_to_floor': 7}
- room_7: changed_tiles=7, change_ratio=0.0398, change_types={'floor_to_structure': 2, 'structure_to_floor': 5}
- room_8: changed_tiles=6, change_ratio=0.0341, change_types={'floor_to_structure': 3, 'structure_to_floor': 3}
- room_11: changed_tiles=3, change_ratio=0.0170, change_types={'structure_to_floor': 3}
- room_5: changed_tiles=2, change_ratio=0.0114, change_types={'W_to_B': 1, 'floor_to_structure': 1}

## Aggregate Change Types

- structure_to_floor: 71
- floor_to_structure: 10
- W_to_B: 1
