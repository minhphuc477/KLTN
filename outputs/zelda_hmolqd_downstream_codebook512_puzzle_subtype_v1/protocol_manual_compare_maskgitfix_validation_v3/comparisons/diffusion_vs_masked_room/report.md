# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_maskgitfix_validation_v3\diffusion_cfg3_logic0_steps50`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_maskgitfix_validation_v3\masked_room_full`

Compared rooms: 12
Total changed tiles: 85

## Worst Rooms

- room_0: changed_tiles=35, change_ratio=0.1989, change_types={'floor_to_structure': 35}
- room_6: changed_tiles=25, change_ratio=0.1420, change_types={'structure_to_floor': 25}
- room_2: changed_tiles=5, change_ratio=0.0284, change_types={'structure_to_floor': 3, 'floor_to_structure': 2}
- room_3: changed_tiles=4, change_ratio=0.0227, change_types={'structure_to_floor': 4}
- room_5: changed_tiles=4, change_ratio=0.0227, change_types={'floor_to_structure': 3, 'B_to_W': 1}
- room_9: changed_tiles=4, change_ratio=0.0227, change_types={'structure_to_floor': 4}
- room_1: changed_tiles=3, change_ratio=0.0170, change_types={'structure_to_floor': 3}
- room_8: changed_tiles=2, change_ratio=0.0114, change_types={'floor_to_structure': 2}

## Aggregate Change Types

- floor_to_structure: 44
- structure_to_floor: 40
- B_to_W: 1
