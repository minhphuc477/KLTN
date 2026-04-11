# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_fastfix_v1_compare\diffusion_cfg3_logic0_steps50`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_fastfix_v1_compare\masked_room_full`

Compared rooms: 12
Total changed tiles: 189

## Worst Rooms

- room_0: changed_tiles=35, change_ratio=0.1989, change_types={'floor_to_structure': 35}
- room_2: changed_tiles=30, change_ratio=0.1705, change_types={'structure_to_floor': 5, 'filled_to_void': 20, 'floor_to_structure': 5}
- room_9: changed_tiles=30, change_ratio=0.1705, change_types={'structure_to_floor': 30}
- room_3: changed_tiles=27, change_ratio=0.1534, change_types={'structure_to_floor': 27}
- room_1: changed_tiles=24, change_ratio=0.1364, change_types={'structure_to_floor': 24}
- room_5: changed_tiles=20, change_ratio=0.1136, change_types={'structure_to_floor': 20}
- room_6: changed_tiles=9, change_ratio=0.0511, change_types={'filled_to_void': 5, 'floor_to_structure': 2, 'structure_to_floor': 2}
- room_7: changed_tiles=4, change_ratio=0.0227, change_types={'floor_to_structure': 4}

## Aggregate Change Types

- structure_to_floor: 111
- floor_to_structure: 53
- filled_to_void: 25
