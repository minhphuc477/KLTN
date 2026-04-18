# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_statefulmultistep_v9\fast_cfg3_logic0_steps4`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_statefulmultistep_v9\masked_room_full`

Compared rooms: 12
Total changed tiles: 133

## Worst Rooms

- room_6: changed_tiles=32, change_ratio=0.1818, change_types={'floor_to_structure': 32}
- room_5: changed_tiles=29, change_ratio=0.1648, change_types={'structure_to_floor': 29}
- room_7: changed_tiles=23, change_ratio=0.1307, change_types={'structure_to_floor': 23}
- room_8: changed_tiles=15, change_ratio=0.0852, change_types={'structure_to_floor': 3, 'floor_to_structure': 10, 'F_to_S': 1, 'S_to_W': 1}
- room_9: changed_tiles=15, change_ratio=0.0852, change_types={'structure_to_floor': 7, 'floor_to_structure': 8}
- room_4: changed_tiles=13, change_ratio=0.0739, change_types={'structure_to_floor': 10, 'floor_to_structure': 3}
- room_3: changed_tiles=3, change_ratio=0.0170, change_types={'floor_to_structure': 1, 'structure_to_floor': 2}
- room_2: changed_tiles=2, change_ratio=0.0114, change_types={'floor_to_structure': 1, 'structure_to_floor': 1}

## Aggregate Change Types

- structure_to_floor: 75
- floor_to_structure: 56
- F_to_S: 1
- S_to_W: 1
