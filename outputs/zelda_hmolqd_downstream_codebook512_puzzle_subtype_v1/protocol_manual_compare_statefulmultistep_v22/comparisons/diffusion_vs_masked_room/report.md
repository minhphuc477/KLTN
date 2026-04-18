# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_statefulmultistep_v22\diffusion_cfg3_logic0_steps50`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_statefulmultistep_v22\masked_room_full`

Compared rooms: 12
Total changed tiles: 219

## Worst Rooms

- room_0: changed_tiles=35, change_ratio=0.1989, change_types={'floor_to_structure': 35}
- room_3: changed_tiles=33, change_ratio=0.1875, change_types={'structure_to_floor': 33}
- room_2: changed_tiles=31, change_ratio=0.1761, change_types={'floor_to_structure': 31}
- room_6: changed_tiles=31, change_ratio=0.1761, change_types={'floor_to_structure': 31}
- room_7: changed_tiles=21, change_ratio=0.1193, change_types={'structure_to_floor': 21}
- room_11: changed_tiles=21, change_ratio=0.1193, change_types={'structure_to_floor': 16, 'floor_to_structure': 5}
- room_1: changed_tiles=15, change_ratio=0.0852, change_types={'floor_to_structure': 15}
- room_10: changed_tiles=10, change_ratio=0.0568, change_types={'structure_to_floor': 10}

## Aggregate Change Types

- floor_to_structure: 128
- structure_to_floor: 90
- B_to_W: 1
