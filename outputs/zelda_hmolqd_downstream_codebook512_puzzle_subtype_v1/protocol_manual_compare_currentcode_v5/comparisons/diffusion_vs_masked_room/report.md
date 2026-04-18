# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_currentcode_v5\diffusion_cfg3_logic0_steps50`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_currentcode_v5\masked_room_full`

Compared rooms: 12
Total changed tiles: 202

## Worst Rooms

- room_2: changed_tiles=47, change_ratio=0.2670, change_types={'floor_to_structure': 34, 'filled_to_void': 13}
- room_6: changed_tiles=41, change_ratio=0.2330, change_types={'structure_to_floor': 41}
- room_0: changed_tiles=35, change_ratio=0.1989, change_types={'floor_to_structure': 35}
- room_5: changed_tiles=20, change_ratio=0.1136, change_types={'floor_to_structure': 9, 'filled_to_void': 1, 'structure_to_floor': 10}
- room_4: changed_tiles=17, change_ratio=0.0966, change_types={'floor_to_structure': 4, 'structure_to_floor': 13}
- room_10: changed_tiles=16, change_ratio=0.0909, change_types={'floor_to_structure': 1, 'structure_to_floor': 15}
- room_9: changed_tiles=10, change_ratio=0.0568, change_types={'structure_to_floor': 5, 'floor_to_structure': 3, 'void_to_filled': 2}
- room_1: changed_tiles=5, change_ratio=0.0284, change_types={'floor_to_structure': 2, 'void_to_filled': 2, 'filled_to_void': 1}

## Aggregate Change Types

- floor_to_structure: 98
- structure_to_floor: 85
- filled_to_void: 15
- void_to_filled: 4
