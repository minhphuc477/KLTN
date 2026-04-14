# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_default\diffusion_cfg3_logic0_steps50`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_default\fast_cfg3_logic0_steps4`

Compared rooms: 12
Total changed tiles: 146

## Worst Rooms

- room_2: changed_tiles=34, change_ratio=0.1932, change_types={'structure_to_floor': 33, 'void_to_filled': 1}
- room_6: changed_tiles=24, change_ratio=0.1364, change_types={'structure_to_floor': 24}
- room_3: changed_tiles=23, change_ratio=0.1307, change_types={'structure_to_floor': 23}
- room_1: changed_tiles=21, change_ratio=0.1193, change_types={'floor_to_structure': 5, 'filled_to_void': 13, 'structure_to_floor': 3}
- room_8: changed_tiles=17, change_ratio=0.0966, change_types={'void_to_filled': 1, 'structure_to_floor': 14, 'floor_to_structure': 2}
- room_5: changed_tiles=12, change_ratio=0.0682, change_types={'floor_to_structure': 9, 'filled_to_void': 3}
- room_4: changed_tiles=8, change_ratio=0.0455, change_types={'structure_to_floor': 5, 'floor_to_structure': 3}
- room_7: changed_tiles=4, change_ratio=0.0227, change_types={'floor_to_structure': 4}

## Aggregate Change Types

- structure_to_floor: 103
- floor_to_structure: 25
- filled_to_void: 16
- void_to_filled: 2
