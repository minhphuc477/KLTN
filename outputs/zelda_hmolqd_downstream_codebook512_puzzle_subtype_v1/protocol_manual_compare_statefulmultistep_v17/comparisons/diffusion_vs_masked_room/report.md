# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_statefulmultistep_v17\diffusion_cfg3_logic0_steps50`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_statefulmultistep_v17\masked_room_full`

Compared rooms: 12
Total changed tiles: 104

## Worst Rooms

- room_1: changed_tiles=27, change_ratio=0.1534, change_types={'floor_to_structure': 27}
- room_2: changed_tiles=17, change_ratio=0.0966, change_types={'floor_to_structure': 14, 'structure_to_floor': 3}
- room_5: changed_tiles=13, change_ratio=0.0739, change_types={'B_to_W': 2, 'structure_to_floor': 1, 'floor_to_structure': 10}
- room_6: changed_tiles=12, change_ratio=0.0682, change_types={'floor_to_structure': 6, 'structure_to_floor': 5, 'B_to_W': 1}
- room_9: changed_tiles=12, change_ratio=0.0682, change_types={'floor_to_structure': 5, 'structure_to_floor': 7}
- room_11: changed_tiles=10, change_ratio=0.0568, change_types={'structure_to_floor': 10}
- room_4: changed_tiles=7, change_ratio=0.0398, change_types={'structure_to_floor': 7}
- room_7: changed_tiles=6, change_ratio=0.0341, change_types={'floor_to_structure': 6}

## Aggregate Change Types

- floor_to_structure: 68
- structure_to_floor: 33
- B_to_W: 3
