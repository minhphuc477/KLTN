# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_maskgitfix_validation_v4\diffusion_cfg3_logic0_steps50`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_maskgitfix_validation_v4\fast_cfg3_logic0_steps4`

Compared rooms: 12
Total changed tiles: 29

## Worst Rooms

- room_2: changed_tiles=5, change_ratio=0.0284, change_types={'structure_to_floor': 3, 'floor_to_structure': 2}
- room_6: changed_tiles=5, change_ratio=0.0284, change_types={'structure_to_floor': 3, 'floor_to_structure': 2}
- room_9: changed_tiles=4, change_ratio=0.0227, change_types={'floor_to_structure': 2, 'structure_to_floor': 2}
- room_1: changed_tiles=3, change_ratio=0.0170, change_types={'structure_to_floor': 3}
- room_8: changed_tiles=3, change_ratio=0.0170, change_types={'structure_to_floor': 1, 'floor_to_structure': 2}
- room_11: changed_tiles=3, change_ratio=0.0170, change_types={'floor_to_structure': 1, 'structure_to_floor': 2}
- room_4: changed_tiles=2, change_ratio=0.0114, change_types={'structure_to_floor': 1, 'floor_to_structure': 1}
- room_10: changed_tiles=2, change_ratio=0.0114, change_types={'floor_to_structure': 1, 'filled_to_void': 1}

## Aggregate Change Types

- structure_to_floor: 16
- floor_to_structure: 12
- filled_to_void: 1
