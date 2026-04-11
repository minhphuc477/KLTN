# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_v1_compare_v2\diffusion_cfg3_logic0_steps50`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_v1_compare_v2\fast_cfg3_logic0_steps4`

Compared rooms: 12
Total changed tiles: 154

## Worst Rooms

- room_2: changed_tiles=44, change_ratio=0.2500, change_types={'structure_to_floor': 44}
- room_9: changed_tiles=25, change_ratio=0.1420, change_types={'structure_to_floor': 25}
- room_6: changed_tiles=21, change_ratio=0.1193, change_types={'structure_to_floor': 21}
- room_5: changed_tiles=16, change_ratio=0.0909, change_types={'floor_to_structure': 15, 'filled_to_void': 1}
- room_3: changed_tiles=14, change_ratio=0.0795, change_types={'floor_to_structure': 3, 'filled_to_void': 11}
- room_7: changed_tiles=12, change_ratio=0.0682, change_types={'structure_to_floor': 7, 'floor_to_structure': 5}
- room_11: changed_tiles=10, change_ratio=0.0568, change_types={'structure_to_floor': 1, 'floor_to_structure': 9}
- room_1: changed_tiles=6, change_ratio=0.0341, change_types={'floor_to_structure': 2, 'filled_to_void': 3, 'structure_to_floor': 1}

## Aggregate Change Types

- structure_to_floor: 102
- floor_to_structure: 37
- filled_to_void: 15
