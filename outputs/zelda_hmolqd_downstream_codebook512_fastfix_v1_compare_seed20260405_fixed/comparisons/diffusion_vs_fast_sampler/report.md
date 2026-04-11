# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_fastfix_v1_compare_seed20260405_fixed\diffusion_cfg3_logic0_steps50`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_fastfix_v1_compare_seed20260405_fixed\fast_cfg3_logic0_steps4`

Compared rooms: 12
Total changed tiles: 162

## Worst Rooms

- room_1: changed_tiles=26, change_ratio=0.1477, change_types={'structure_to_floor': 26}
- room_7: changed_tiles=26, change_ratio=0.1477, change_types={'floor_to_structure': 26}
- room_3: changed_tiles=24, change_ratio=0.1364, change_types={'structure_to_floor': 24}
- room_2: changed_tiles=21, change_ratio=0.1193, change_types={'filled_to_void': 12, 'floor_to_structure': 9}
- room_8: changed_tiles=20, change_ratio=0.1136, change_types={'structure_to_floor': 2, 'floor_to_structure': 16, 'filled_to_void': 2}
- room_11: changed_tiles=15, change_ratio=0.0852, change_types={'floor_to_structure': 4, 'structure_to_floor': 11}
- room_6: changed_tiles=11, change_ratio=0.0625, change_types={'filled_to_void': 6, 'void_to_filled': 3, 'floor_to_structure': 1, 'structure_to_floor': 1}
- room_4: changed_tiles=9, change_ratio=0.0511, change_types={'structure_to_floor': 2, 'floor_to_structure': 7}

## Aggregate Change Types

- floor_to_structure: 70
- structure_to_floor: 67
- filled_to_void: 22
- void_to_filled: 3
