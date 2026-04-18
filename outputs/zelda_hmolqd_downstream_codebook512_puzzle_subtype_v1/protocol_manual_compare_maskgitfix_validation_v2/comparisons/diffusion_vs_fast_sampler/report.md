# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_maskgitfix_validation_v2\diffusion_cfg3_logic0_steps50`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_maskgitfix_validation_v2\fast_cfg3_logic0_steps4`

Compared rooms: 12
Total changed tiles: 156

## Worst Rooms

- room_9: changed_tiles=38, change_ratio=0.2159, change_types={'filled_to_void': 12, 'floor_to_structure': 23, 'B_to_W': 1, 'structure_to_floor': 2}
- room_3: changed_tiles=23, change_ratio=0.1307, change_types={'structure_to_floor': 23}
- room_6: changed_tiles=22, change_ratio=0.1250, change_types={'structure_to_floor': 7, 'filled_to_void': 12, 'floor_to_structure': 3}
- room_1: changed_tiles=21, change_ratio=0.1193, change_types={'structure_to_floor': 21}
- room_4: changed_tiles=18, change_ratio=0.1023, change_types={'structure_to_floor': 6, 'floor_to_structure': 12}
- room_2: changed_tiles=16, change_ratio=0.0909, change_types={'filled_to_void': 11, 'floor_to_structure': 3, 'structure_to_floor': 2}
- room_11: changed_tiles=11, change_ratio=0.0625, change_types={'structure_to_floor': 11}
- room_0: changed_tiles=3, change_ratio=0.0170, change_types={'floor_to_structure': 3}

## Aggregate Change Types

- structure_to_floor: 74
- floor_to_structure: 44
- filled_to_void: 36
- W_to_B: 1
- B_to_W: 1
