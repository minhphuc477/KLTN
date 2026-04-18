# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_statefulmultistep_v17\fast_cfg3_logic0_steps4`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_statefulmultistep_v17\masked_room_full`

Compared rooms: 12
Total changed tiles: 91

## Worst Rooms

- room_1: changed_tiles=27, change_ratio=0.1534, change_types={'floor_to_structure': 27}
- room_2: changed_tiles=15, change_ratio=0.0852, change_types={'floor_to_structure': 12, 'structure_to_floor': 3}
- room_5: changed_tiles=14, change_ratio=0.0795, change_types={'floor_to_structure': 13, 'structure_to_floor': 1}
- room_6: changed_tiles=14, change_ratio=0.0795, change_types={'floor_to_structure': 11, 'structure_to_floor': 3}
- room_4: changed_tiles=7, change_ratio=0.0398, change_types={'structure_to_floor': 7}
- room_9: changed_tiles=7, change_ratio=0.0398, change_types={'floor_to_structure': 5, 'structure_to_floor': 2}
- room_7: changed_tiles=6, change_ratio=0.0341, change_types={'floor_to_structure': 6}
- room_11: changed_tiles=1, change_ratio=0.0057, change_types={'floor_to_structure': 1}

## Aggregate Change Types

- floor_to_structure: 75
- structure_to_floor: 16
