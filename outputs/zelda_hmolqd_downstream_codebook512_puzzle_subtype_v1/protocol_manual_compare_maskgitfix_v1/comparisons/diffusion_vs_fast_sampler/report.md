# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_maskgitfix_v1\diffusion_cfg3_logic0_steps50`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_maskgitfix_v1\fast_cfg3_logic0_steps4`

Compared rooms: 12
Total changed tiles: 148

## Worst Rooms

- room_9: changed_tiles=37, change_ratio=0.2102, change_types={'filled_to_void': 12, 'floor_to_structure': 23, 'structure_to_floor': 2}
- room_2: changed_tiles=28, change_ratio=0.1591, change_types={'filled_to_void': 13, 'floor_to_structure': 12, 'structure_to_floor': 3}
- room_3: changed_tiles=23, change_ratio=0.1307, change_types={'structure_to_floor': 23}
- room_6: changed_tiles=19, change_ratio=0.1080, change_types={'structure_to_floor': 4, 'filled_to_void': 12, 'floor_to_structure': 3}
- room_1: changed_tiles=15, change_ratio=0.0852, change_types={'floor_to_structure': 6, 'filled_to_void': 8, 'structure_to_floor': 1}
- room_4: changed_tiles=14, change_ratio=0.0795, change_types={'structure_to_floor': 6, 'floor_to_structure': 8}
- room_11: changed_tiles=11, change_ratio=0.0625, change_types={'structure_to_floor': 11}
- room_7: changed_tiles=1, change_ratio=0.0057, change_types={'structure_to_floor': 1}

## Aggregate Change Types

- floor_to_structure: 52
- structure_to_floor: 51
- filled_to_void: 45
