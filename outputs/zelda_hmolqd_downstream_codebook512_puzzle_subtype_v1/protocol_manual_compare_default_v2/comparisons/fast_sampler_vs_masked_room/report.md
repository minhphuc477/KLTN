# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_default_v2\fast_cfg3_logic0_steps4`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_default_v2\masked_room_full`

Compared rooms: 12
Total changed tiles: 121

## Worst Rooms

- room_0: changed_tiles=37, change_ratio=0.2102, change_types={'floor_to_structure': 37}
- room_1: changed_tiles=20, change_ratio=0.1136, change_types={'floor_to_structure': 14, 'filled_to_void': 2, 'structure_to_floor': 1, 'void_to_filled': 3}
- room_9: changed_tiles=19, change_ratio=0.1080, change_types={'void_to_filled': 8, 'floor_to_structure': 3, 'filled_to_void': 3, 'structure_to_floor': 5}
- room_2: changed_tiles=15, change_ratio=0.0852, change_types={'structure_to_floor': 11, 'void_to_filled': 3, 'floor_to_structure': 1}
- room_5: changed_tiles=10, change_ratio=0.0568, change_types={'floor_to_structure': 6, 'B_to_W': 1, 'structure_to_floor': 1, 'W_to_B': 2}
- room_4: changed_tiles=8, change_ratio=0.0455, change_types={'floor_to_structure': 3, 'structure_to_floor': 5}
- room_7: changed_tiles=5, change_ratio=0.0284, change_types={'floor_to_structure': 5}
- room_6: changed_tiles=3, change_ratio=0.0170, change_types={'structure_to_floor': 2, 'filled_to_void': 1}

## Aggregate Change Types

- floor_to_structure: 72
- structure_to_floor: 26
- void_to_filled: 14
- filled_to_void: 6
- W_to_B: 2
- B_to_W: 1
