# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_fastfix_v1_compare\diffusion_cfg3_logic0_steps50`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_fastfix_v1_compare\fast_cfg3_logic0_steps4`

Compared rooms: 12
Total changed tiles: 130

## Worst Rooms

- room_9: changed_tiles=30, change_ratio=0.1705, change_types={'structure_to_floor': 30}
- room_3: changed_tiles=27, change_ratio=0.1534, change_types={'structure_to_floor': 27}
- room_2: changed_tiles=22, change_ratio=0.1250, change_types={'structure_to_floor': 5, 'filled_to_void': 12, 'floor_to_structure': 5}
- room_5: changed_tiles=18, change_ratio=0.1023, change_types={'filled_to_void': 3, 'floor_to_structure': 14, 'B_to_W': 1}
- room_6: changed_tiles=11, change_ratio=0.0625, change_types={'structure_to_floor': 5, 'filled_to_void': 4, 'floor_to_structure': 2}
- room_7: changed_tiles=7, change_ratio=0.0398, change_types={'floor_to_structure': 7}
- room_1: changed_tiles=6, change_ratio=0.0341, change_types={'filled_to_void': 3, 'floor_to_structure': 3}
- room_4: changed_tiles=3, change_ratio=0.0170, change_types={'floor_to_structure': 2, 'structure_to_floor': 1}

## Aggregate Change Types

- structure_to_floor: 69
- floor_to_structure: 38
- filled_to_void: 22
- B_to_W: 1
