# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_puzzle_structure_control_v1\protocol_manual_compare_puzzle_control_v3\diffusion_cfg3_logic0_steps50`
Candidate: `outputs\zelda_hmolqd_downstream_puzzle_structure_control_v1\protocol_manual_compare_puzzle_control_v3\masked_room_full`

Compared rooms: 12
Total changed tiles: 238

## Worst Rooms

- room_2: changed_tiles=36, change_ratio=0.2045, change_types={'floor_to_structure': 36}
- room_0: changed_tiles=35, change_ratio=0.1989, change_types={'floor_to_structure': 35}
- room_5: changed_tiles=33, change_ratio=0.1875, change_types={'structure_to_floor': 33}
- room_1: changed_tiles=30, change_ratio=0.1705, change_types={'structure_to_floor': 26, 'void_to_filled': 4}
- room_6: changed_tiles=27, change_ratio=0.1534, change_types={'floor_to_structure': 3, 'structure_to_floor': 13, 'filled_to_void': 11}
- room_7: changed_tiles=26, change_ratio=0.1477, change_types={'structure_to_floor': 26}
- room_10: changed_tiles=16, change_ratio=0.0909, change_types={'structure_to_floor': 16}
- room_4: changed_tiles=13, change_ratio=0.0739, change_types={'floor_to_structure': 3, 'structure_to_floor': 10}

## Aggregate Change Types

- structure_to_floor: 137
- floor_to_structure: 85
- filled_to_void: 11
- void_to_filled: 5
