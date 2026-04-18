# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_statefulmultistep_v9\diffusion_cfg3_logic0_steps50`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_statefulmultistep_v9\fast_cfg3_logic0_steps4`

Compared rooms: 12
Total changed tiles: 46

## Worst Rooms

- room_9: changed_tiles=17, change_ratio=0.0966, change_types={'floor_to_structure': 10, 'structure_to_floor': 7}
- room_8: changed_tiles=13, change_ratio=0.0739, change_types={'structure_to_floor': 12, 'floor_to_structure': 1}
- room_10: changed_tiles=6, change_ratio=0.0341, change_types={'structure_to_floor': 6}
- room_5: changed_tiles=4, change_ratio=0.0227, change_types={'floor_to_structure': 4}
- room_3: changed_tiles=3, change_ratio=0.0170, change_types={'structure_to_floor': 1, 'floor_to_structure': 2}
- room_2: changed_tiles=1, change_ratio=0.0057, change_types={'floor_to_structure': 1}
- room_7: changed_tiles=1, change_ratio=0.0057, change_types={'structure_to_floor': 1}
- room_11: changed_tiles=1, change_ratio=0.0057, change_types={'floor_to_structure': 1}

## Aggregate Change Types

- structure_to_floor: 27
- floor_to_structure: 19
