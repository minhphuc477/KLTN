# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_v1_compare\diffusion_cfg3_logic0_steps50`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_v1_compare\fast_cfg3_logic0_steps4`

Compared rooms: 12
Total changed tiles: 168

## Worst Rooms

- room_2: changed_tiles=43, change_ratio=0.2443, change_types={'filled_to_void': 18, 'structure_to_floor': 19, 'floor_to_structure': 6}
- room_9: changed_tiles=31, change_ratio=0.1761, change_types={'structure_to_floor': 31}
- room_3: changed_tiles=26, change_ratio=0.1477, change_types={'structure_to_floor': 26}
- room_6: changed_tiles=22, change_ratio=0.1250, change_types={'structure_to_floor': 22}
- room_5: changed_tiles=16, change_ratio=0.0909, change_types={'filled_to_void': 2, 'floor_to_structure': 14}
- room_11: changed_tiles=12, change_ratio=0.0682, change_types={'structure_to_floor': 2, 'floor_to_structure': 10}
- room_1: changed_tiles=6, change_ratio=0.0341, change_types={'floor_to_structure': 3, 'filled_to_void': 3}
- room_7: changed_tiles=6, change_ratio=0.0341, change_types={'floor_to_structure': 6}

## Aggregate Change Types

- structure_to_floor: 103
- floor_to_structure: 42
- filled_to_void: 23
