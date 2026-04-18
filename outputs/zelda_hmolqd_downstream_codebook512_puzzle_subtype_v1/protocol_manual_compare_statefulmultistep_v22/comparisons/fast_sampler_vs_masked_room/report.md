# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_statefulmultistep_v22\fast_cfg3_logic0_steps4`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_statefulmultistep_v22\masked_room_full`

Compared rooms: 12
Total changed tiles: 227

## Worst Rooms

- room_0: changed_tiles=35, change_ratio=0.1989, change_types={'floor_to_structure': 35}
- room_6: changed_tiles=32, change_ratio=0.1818, change_types={'floor_to_structure': 32}
- room_2: changed_tiles=31, change_ratio=0.1761, change_types={'floor_to_structure': 31}
- room_3: changed_tiles=28, change_ratio=0.1591, change_types={'structure_to_floor': 28}
- room_11: changed_tiles=26, change_ratio=0.1477, change_types={'structure_to_floor': 21, 'floor_to_structure': 5}
- room_7: changed_tiles=21, change_ratio=0.1193, change_types={'structure_to_floor': 21}
- room_1: changed_tiles=15, change_ratio=0.0852, change_types={'floor_to_structure': 15}
- room_9: changed_tiles=14, change_ratio=0.0795, change_types={'floor_to_structure': 7, 'structure_to_floor': 7}

## Aggregate Change Types

- floor_to_structure: 137
- structure_to_floor: 89
- B_to_W: 1
