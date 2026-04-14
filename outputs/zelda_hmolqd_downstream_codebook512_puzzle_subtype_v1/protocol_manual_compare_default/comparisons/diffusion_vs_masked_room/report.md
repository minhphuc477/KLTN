# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_default\diffusion_cfg3_logic0_steps50`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_default\masked_room_full`

Compared rooms: 12
Total changed tiles: 133

## Worst Rooms

- room_2: changed_tiles=26, change_ratio=0.1477, change_types={'structure_to_floor': 22, 'floor_to_structure': 3, 'void_to_filled': 1}
- room_6: changed_tiles=24, change_ratio=0.1364, change_types={'structure_to_floor': 24}
- room_3: changed_tiles=23, change_ratio=0.1307, change_types={'structure_to_floor': 23}
- room_1: changed_tiles=19, change_ratio=0.1080, change_types={'floor_to_structure': 7, 'filled_to_void': 9, 'structure_to_floor': 3}
- room_8: changed_tiles=17, change_ratio=0.0966, change_types={'void_to_filled': 1, 'structure_to_floor': 11, 'floor_to_structure': 4, 'B_to_W': 1}
- room_5: changed_tiles=12, change_ratio=0.0682, change_types={'floor_to_structure': 7, 'filled_to_void': 5}
- room_4: changed_tiles=5, change_ratio=0.0284, change_types={'structure_to_floor': 2, 'floor_to_structure': 3}
- room_11: changed_tiles=4, change_ratio=0.0227, change_types={'floor_to_structure': 4}

## Aggregate Change Types

- structure_to_floor: 85
- floor_to_structure: 31
- filled_to_void: 14
- void_to_filled: 2
- B_to_W: 1
