# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_default_v2\diffusion_cfg3_logic0_steps50`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_default_v2\fast_cfg3_logic0_steps4`

Compared rooms: 12
Total changed tiles: 157

## Worst Rooms

- room_9: changed_tiles=39, change_ratio=0.2216, change_types={'filled_to_void': 13, 'floor_to_structure': 23, 'B_to_W': 1, 'structure_to_floor': 2}
- room_2: changed_tiles=32, change_ratio=0.1818, change_types={'filled_to_void': 13, 'B_to_W': 1, 'void_to_filled': 2, 'floor_to_structure': 12, 'structure_to_floor': 4}
- room_3: changed_tiles=23, change_ratio=0.1307, change_types={'structure_to_floor': 23}
- room_6: changed_tiles=17, change_ratio=0.0966, change_types={'structure_to_floor': 16, 'floor_to_structure': 1}
- room_1: changed_tiles=14, change_ratio=0.0795, change_types={'floor_to_structure': 6, 'filled_to_void': 8}
- room_4: changed_tiles=14, change_ratio=0.0795, change_types={'structure_to_floor': 6, 'floor_to_structure': 8}
- room_11: changed_tiles=11, change_ratio=0.0625, change_types={'structure_to_floor': 11}
- room_5: changed_tiles=3, change_ratio=0.0170, change_types={'floor_to_structure': 1, 'B_to_W': 2}

## Aggregate Change Types

- structure_to_floor: 64
- floor_to_structure: 53
- filled_to_void: 34
- B_to_W: 4
- void_to_filled: 2
