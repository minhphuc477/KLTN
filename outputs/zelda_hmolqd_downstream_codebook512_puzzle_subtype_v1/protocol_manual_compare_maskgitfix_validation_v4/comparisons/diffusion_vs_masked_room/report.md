# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_maskgitfix_validation_v4\diffusion_cfg3_logic0_steps50`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_maskgitfix_validation_v4\masked_room_full`

Compared rooms: 12
Total changed tiles: 78

## Worst Rooms

- room_0: changed_tiles=34, change_ratio=0.1932, change_types={'floor_to_structure': 34}
- room_6: changed_tiles=24, change_ratio=0.1364, change_types={'structure_to_floor': 24}
- room_2: changed_tiles=5, change_ratio=0.0284, change_types={'structure_to_floor': 3, 'floor_to_structure': 2}
- room_1: changed_tiles=3, change_ratio=0.0170, change_types={'structure_to_floor': 3}
- room_5: changed_tiles=3, change_ratio=0.0170, change_types={'floor_to_structure': 3}
- room_9: changed_tiles=3, change_ratio=0.0170, change_types={'structure_to_floor': 3}
- room_8: changed_tiles=2, change_ratio=0.0114, change_types={'floor_to_structure': 2}
- room_3: changed_tiles=1, change_ratio=0.0057, change_types={'filled_to_void': 1}

## Aggregate Change Types

- floor_to_structure: 43
- structure_to_floor: 34
- filled_to_void: 1
