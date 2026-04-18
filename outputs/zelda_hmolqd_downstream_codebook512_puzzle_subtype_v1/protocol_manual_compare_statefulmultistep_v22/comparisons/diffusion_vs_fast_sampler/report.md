# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_statefulmultistep_v22\diffusion_cfg3_logic0_steps50`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_statefulmultistep_v22\fast_cfg3_logic0_steps4`

Compared rooms: 12
Total changed tiles: 68

## Worst Rooms

- room_11: changed_tiles=21, change_ratio=0.1193, change_types={'floor_to_structure': 13, 'structure_to_floor': 8}
- room_9: changed_tiles=18, change_ratio=0.1023, change_types={'structure_to_floor': 10, 'floor_to_structure': 8}
- room_10: changed_tiles=12, change_ratio=0.0682, change_types={'structure_to_floor': 12}
- room_8: changed_tiles=10, change_ratio=0.0568, change_types={'structure_to_floor': 2, 'floor_to_structure': 8}
- room_3: changed_tiles=5, change_ratio=0.0284, change_types={'structure_to_floor': 5}
- room_5: changed_tiles=1, change_ratio=0.0057, change_types={'structure_to_floor': 1}
- room_6: changed_tiles=1, change_ratio=0.0057, change_types={'structure_to_floor': 1}
- room_0: changed_tiles=0, change_ratio=0.0000, change_types={}

## Aggregate Change Types

- structure_to_floor: 39
- floor_to_structure: 29
