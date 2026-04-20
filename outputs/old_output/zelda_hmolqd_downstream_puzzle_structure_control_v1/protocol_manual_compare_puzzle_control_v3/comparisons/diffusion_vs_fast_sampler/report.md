# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_puzzle_structure_control_v1\protocol_manual_compare_puzzle_control_v3\diffusion_cfg3_logic0_steps50`
Candidate: `outputs\zelda_hmolqd_downstream_puzzle_structure_control_v1\protocol_manual_compare_puzzle_control_v3\fast_cfg3_logic0_steps4`

Compared rooms: 12
Total changed tiles: 126

## Worst Rooms

- room_5: changed_tiles=33, change_ratio=0.1875, change_types={'structure_to_floor': 33}
- room_7: changed_tiles=26, change_ratio=0.1477, change_types={'structure_to_floor': 26}
- room_6: changed_tiles=25, change_ratio=0.1420, change_types={'floor_to_structure': 8, 'structure_to_floor': 15, 'filled_to_void': 2}
- room_4: changed_tiles=12, change_ratio=0.0682, change_types={'floor_to_structure': 2, 'structure_to_floor': 10}
- room_10: changed_tiles=11, change_ratio=0.0625, change_types={'structure_to_floor': 10, 'floor_to_structure': 1}
- room_9: changed_tiles=9, change_ratio=0.0511, change_types={'floor_to_structure': 1, 'structure_to_floor': 7, 'void_to_filled': 1}
- room_8: changed_tiles=8, change_ratio=0.0455, change_types={'structure_to_floor': 4, 'floor_to_structure': 4}
- room_1: changed_tiles=1, change_ratio=0.0057, change_types={'floor_to_structure': 1}

## Aggregate Change Types

- structure_to_floor: 105
- floor_to_structure: 18
- filled_to_void: 2
- void_to_filled: 1
