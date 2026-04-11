# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_fastfix_v1_compare_seed20260405_fixed\diffusion_cfg3_logic0_steps50`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_fastfix_v1_compare_seed20260405_fixed\masked_room_full`

Compared rooms: 12
Total changed tiles: 161

## Worst Rooms

- room_0: changed_tiles=35, change_ratio=0.1989, change_types={'structure_to_floor': 35}
- room_2: changed_tiles=25, change_ratio=0.1420, change_types={'structure_to_floor': 25}
- room_7: changed_tiles=23, change_ratio=0.1307, change_types={'floor_to_structure': 23}
- room_3: changed_tiles=21, change_ratio=0.1193, change_types={'structure_to_floor': 20, 'floor_to_structure': 1}
- room_11: changed_tiles=15, change_ratio=0.0852, change_types={'floor_to_structure': 4, 'structure_to_floor': 11}
- room_5: changed_tiles=13, change_ratio=0.0739, change_types={'floor_to_structure': 11, 'filled_to_void': 1, 'structure_to_floor': 1}
- room_6: changed_tiles=11, change_ratio=0.0625, change_types={'void_to_filled': 1, 'filled_to_void': 7, 'floor_to_structure': 2, 'structure_to_floor': 1}
- room_1: changed_tiles=10, change_ratio=0.0568, change_types={'floor_to_structure': 3, 'structure_to_floor': 2, 'filled_to_void': 5}

## Aggregate Change Types

- structure_to_floor: 100
- floor_to_structure: 47
- filled_to_void: 13
- void_to_filled: 1
