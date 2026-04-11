# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_v1_compare\diffusion_cfg3_logic0_steps50`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_v1_compare\masked_room_full`

Compared rooms: 12
Total changed tiles: 171

## Worst Rooms

- room_2: changed_tiles=44, change_ratio=0.2500, change_types={'structure_to_floor': 21, 'filled_to_void': 18, 'floor_to_structure': 5}
- room_9: changed_tiles=31, change_ratio=0.1761, change_types={'structure_to_floor': 31}
- room_3: changed_tiles=26, change_ratio=0.1477, change_types={'structure_to_floor': 26}
- room_1: changed_tiles=24, change_ratio=0.1364, change_types={'structure_to_floor': 24}
- room_5: changed_tiles=20, change_ratio=0.1136, change_types={'structure_to_floor': 20}
- room_6: changed_tiles=12, change_ratio=0.0682, change_types={'floor_to_structure': 7, 'filled_to_void': 5}
- room_7: changed_tiles=5, change_ratio=0.0284, change_types={'floor_to_structure': 4, 'structure_to_floor': 1}
- room_11: changed_tiles=5, change_ratio=0.0284, change_types={'floor_to_structure': 5}

## Aggregate Change Types

- structure_to_floor: 126
- filled_to_void: 23
- floor_to_structure: 22
