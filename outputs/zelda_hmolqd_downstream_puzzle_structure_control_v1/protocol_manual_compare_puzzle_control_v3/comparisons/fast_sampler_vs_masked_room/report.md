# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_puzzle_structure_control_v1\protocol_manual_compare_puzzle_control_v3\fast_cfg3_logic0_steps4`
Candidate: `outputs\zelda_hmolqd_downstream_puzzle_structure_control_v1\protocol_manual_compare_puzzle_control_v3\masked_room_full`

Compared rooms: 12
Total changed tiles: 148

## Worst Rooms

- room_2: changed_tiles=36, change_ratio=0.2045, change_types={'floor_to_structure': 36}
- room_0: changed_tiles=35, change_ratio=0.1989, change_types={'floor_to_structure': 35}
- room_1: changed_tiles=31, change_ratio=0.1761, change_types={'structure_to_floor': 27, 'void_to_filled': 4}
- room_6: changed_tiles=15, change_ratio=0.0852, change_types={'structure_to_floor': 6, 'filled_to_void': 9}
- room_10: changed_tiles=11, change_ratio=0.0625, change_types={'floor_to_structure': 2, 'structure_to_floor': 9}
- room_11: changed_tiles=9, change_ratio=0.0511, change_types={'floor_to_structure': 4, 'structure_to_floor': 5}
- room_8: changed_tiles=8, change_ratio=0.0455, change_types={'floor_to_structure': 4, 'structure_to_floor': 4}
- room_9: changed_tiles=2, change_ratio=0.0114, change_types={'floor_to_structure': 2}

## Aggregate Change Types

- floor_to_structure: 84
- structure_to_floor: 51
- filled_to_void: 9
- void_to_filled: 4
