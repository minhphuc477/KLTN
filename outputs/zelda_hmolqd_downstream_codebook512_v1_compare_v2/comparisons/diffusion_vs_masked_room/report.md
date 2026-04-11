# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_v1_compare_v2\diffusion_cfg3_logic0_steps50`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_v1_compare_v2\masked_room_full`

Compared rooms: 12
Total changed tiles: 174

## Worst Rooms

- room_2: changed_tiles=41, change_ratio=0.2330, change_types={'structure_to_floor': 18, 'filled_to_void': 18, 'floor_to_structure': 5}
- room_3: changed_tiles=26, change_ratio=0.1477, change_types={'structure_to_floor': 26}
- room_1: changed_tiles=25, change_ratio=0.1420, change_types={'structure_to_floor': 25}
- room_9: changed_tiles=25, change_ratio=0.1420, change_types={'structure_to_floor': 25}
- room_5: changed_tiles=20, change_ratio=0.1136, change_types={'structure_to_floor': 20}
- room_6: changed_tiles=14, change_ratio=0.0795, change_types={'floor_to_structure': 9, 'filled_to_void': 5}
- room_7: changed_tiles=13, change_ratio=0.0739, change_types={'structure_to_floor': 8, 'floor_to_structure': 5}
- room_11: changed_tiles=6, change_ratio=0.0341, change_types={'floor_to_structure': 6}

## Aggregate Change Types

- structure_to_floor: 125
- floor_to_structure: 26
- filled_to_void: 23
