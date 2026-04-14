# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_default\fast_cfg3_logic0_steps4`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_default\masked_room_full`

Compared rooms: 12
Total changed tiles: 38

## Worst Rooms

- room_2: changed_tiles=14, change_ratio=0.0795, change_types={'floor_to_structure': 14}
- room_1: changed_tiles=9, change_ratio=0.0511, change_types={'floor_to_structure': 2, 'void_to_filled': 5, 'filled_to_void': 1, 'structure_to_floor': 1}
- room_8: changed_tiles=5, change_ratio=0.0284, change_types={'floor_to_structure': 5}
- room_4: changed_tiles=3, change_ratio=0.0170, change_types={'floor_to_structure': 3}
- room_5: changed_tiles=3, change_ratio=0.0170, change_types={'structure_to_floor': 1, 'filled_to_void': 2}
- room_11: changed_tiles=3, change_ratio=0.0170, change_types={'floor_to_structure': 3}
- room_7: changed_tiles=1, change_ratio=0.0057, change_types={'structure_to_floor': 1}
- room_0: changed_tiles=0, change_ratio=0.0000, change_types={}

## Aggregate Change Types

- floor_to_structure: 27
- void_to_filled: 5
- filled_to_void: 3
- structure_to_floor: 3
