# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_default_v2\diffusion_cfg3_logic0_steps50`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_default_v2\masked_room_full`

Compared rooms: 12
Total changed tiles: 180

## Worst Rooms

- room_0: changed_tiles=37, change_ratio=0.2102, change_types={'floor_to_structure': 37}
- room_9: changed_tiles=27, change_ratio=0.1534, change_types={'floor_to_structure': 18, 'filled_to_void': 8, 'B_to_W': 1}
- room_1: changed_tiles=26, change_ratio=0.1477, change_types={'floor_to_structure': 19, 'filled_to_void': 7}
- room_3: changed_tiles=23, change_ratio=0.1307, change_types={'structure_to_floor': 23}
- room_2: changed_tiles=18, change_ratio=0.1023, change_types={'filled_to_void': 10, 'structure_to_floor': 4, 'void_to_filled': 2, 'floor_to_structure': 2}
- room_6: changed_tiles=18, change_ratio=0.1023, change_types={'structure_to_floor': 17, 'filled_to_void': 1}
- room_11: changed_tiles=10, change_ratio=0.0568, change_types={'floor_to_structure': 1, 'structure_to_floor': 9}
- room_5: changed_tiles=7, change_ratio=0.0398, change_types={'floor_to_structure': 6, 'B_to_W': 1}

## Aggregate Change Types

- floor_to_structure: 92
- structure_to_floor: 58
- filled_to_void: 26
- void_to_filled: 2
- B_to_W: 2
