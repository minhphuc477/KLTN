# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_statefulmultistep_v9\diffusion_cfg3_logic0_steps50`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_statefulmultistep_v9\masked_room_full`

Compared rooms: 12
Total changed tiles: 137

## Worst Rooms

- room_6: changed_tiles=32, change_ratio=0.1818, change_types={'floor_to_structure': 32}
- room_5: changed_tiles=25, change_ratio=0.1420, change_types={'structure_to_floor': 25}
- room_7: changed_tiles=24, change_ratio=0.1364, change_types={'structure_to_floor': 24}
- room_8: changed_tiles=18, change_ratio=0.1023, change_types={'structure_to_floor': 9, 'floor_to_structure': 5, 'F_to_S': 1, 'S_to_W': 1, 'B_to_W': 2}
- room_9: changed_tiles=16, change_ratio=0.0909, change_types={'floor_to_structure': 10, 'structure_to_floor': 6}
- room_4: changed_tiles=13, change_ratio=0.0739, change_types={'structure_to_floor': 10, 'floor_to_structure': 3}
- room_10: changed_tiles=7, change_ratio=0.0398, change_types={'floor_to_structure': 1, 'structure_to_floor': 6}
- room_2: changed_tiles=1, change_ratio=0.0057, change_types={'floor_to_structure': 1}

## Aggregate Change Types

- structure_to_floor: 80
- floor_to_structure: 53
- B_to_W: 2
- F_to_S: 1
- S_to_W: 1
