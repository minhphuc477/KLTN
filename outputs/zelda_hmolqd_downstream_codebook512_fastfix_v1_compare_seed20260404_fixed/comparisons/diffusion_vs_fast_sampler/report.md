# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_fastfix_v1_compare_seed20260404_fixed\diffusion_cfg3_logic0_steps50`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_fastfix_v1_compare_seed20260404_fixed\fast_cfg3_logic0_steps4`

Compared rooms: 12
Total changed tiles: 130

## Worst Rooms

- room_2: changed_tiles=28, change_ratio=0.1591, change_types={'structure_to_floor': 28}
- room_9: changed_tiles=27, change_ratio=0.1534, change_types={'structure_to_floor': 27}
- room_1: changed_tiles=15, change_ratio=0.0852, change_types={'structure_to_floor': 4, 'filled_to_void': 7, 'floor_to_structure': 4}
- room_5: changed_tiles=15, change_ratio=0.0852, change_types={'structure_to_floor': 15}
- room_3: changed_tiles=13, change_ratio=0.0739, change_types={'floor_to_structure': 6, 'filled_to_void': 4, 'structure_to_floor': 3}
- room_11: changed_tiles=11, change_ratio=0.0625, change_types={'floor_to_structure': 1, 'structure_to_floor': 10}
- room_6: changed_tiles=10, change_ratio=0.0568, change_types={'floor_to_structure': 2, 'filled_to_void': 3, 'B_to_W': 1, 'structure_to_floor': 4}
- room_7: changed_tiles=6, change_ratio=0.0341, change_types={'structure_to_floor': 4, 'floor_to_structure': 2}

## Aggregate Change Types

- structure_to_floor: 99
- floor_to_structure: 16
- filled_to_void: 14
- B_to_W: 1
