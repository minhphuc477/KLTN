# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_fastfix_v1_compare_seed20260404_fixed\diffusion_cfg3_logic0_steps50`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_fastfix_v1_compare_seed20260404_fixed\masked_room_full`

Compared rooms: 12
Total changed tiles: 170

## Worst Rooms

- room_0: changed_tiles=35, change_ratio=0.1989, change_types={'floor_to_structure': 35}
- room_9: changed_tiles=27, change_ratio=0.1534, change_types={'structure_to_floor': 27}
- room_1: changed_tiles=25, change_ratio=0.1420, change_types={'structure_to_floor': 25}
- room_3: changed_tiles=22, change_ratio=0.1250, change_types={'structure_to_floor': 21, 'floor_to_structure': 1}
- room_5: changed_tiles=20, change_ratio=0.1136, change_types={'structure_to_floor': 20}
- room_2: changed_tiles=14, change_ratio=0.0795, change_types={'filled_to_void': 11, 'floor_to_structure': 3}
- room_11: changed_tiles=10, change_ratio=0.0568, change_types={'floor_to_structure': 2, 'structure_to_floor': 8}
- room_6: changed_tiles=9, change_ratio=0.0511, change_types={'floor_to_structure': 3, 'filled_to_void': 2, 'structure_to_floor': 4}

## Aggregate Change Types

- structure_to_floor: 110
- floor_to_structure: 46
- filled_to_void: 13
- W_to_B: 1
