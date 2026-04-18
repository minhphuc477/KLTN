# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_currentcode_v5\diffusion_cfg3_logic0_steps50`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_currentcode_v5\fast_cfg3_logic0_steps4`

Compared rooms: 12
Total changed tiles: 69

## Worst Rooms

- room_6: changed_tiles=45, change_ratio=0.2557, change_types={'structure_to_floor': 45}
- room_9: changed_tiles=10, change_ratio=0.0568, change_types={'floor_to_structure': 3, 'structure_to_floor': 5, 'void_to_filled': 2}
- room_8: changed_tiles=4, change_ratio=0.0227, change_types={'floor_to_structure': 1, 'structure_to_floor': 3}
- room_5: changed_tiles=3, change_ratio=0.0170, change_types={'floor_to_structure': 1, 'structure_to_floor': 2}
- room_7: changed_tiles=3, change_ratio=0.0170, change_types={'floor_to_structure': 2, 'structure_to_floor': 1}
- room_10: changed_tiles=2, change_ratio=0.0114, change_types={'floor_to_structure': 2}
- room_1: changed_tiles=1, change_ratio=0.0057, change_types={'floor_to_structure': 1}
- room_11: changed_tiles=1, change_ratio=0.0057, change_types={'structure_to_floor': 1}

## Aggregate Change Types

- structure_to_floor: 57
- floor_to_structure: 10
- void_to_filled: 2
