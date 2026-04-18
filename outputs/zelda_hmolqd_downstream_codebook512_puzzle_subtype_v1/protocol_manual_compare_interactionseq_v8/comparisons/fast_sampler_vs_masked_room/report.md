# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_interactionseq_v8\fast_cfg3_logic0_steps4`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_interactionseq_v8\masked_room_full`

Compared rooms: 12
Total changed tiles: 142

## Worst Rooms

- room_0: changed_tiles=35, change_ratio=0.1989, change_types={'structure_to_floor': 35}
- room_1: changed_tiles=35, change_ratio=0.1989, change_types={'floor_to_structure': 35}
- room_5: changed_tiles=27, change_ratio=0.1534, change_types={'floor_to_structure': 26, 'B_to_W': 1}
- room_6: changed_tiles=13, change_ratio=0.0739, change_types={'floor_to_structure': 8, 'structure_to_floor': 5}
- room_2: changed_tiles=6, change_ratio=0.0341, change_types={'floor_to_structure': 6}
- room_8: changed_tiles=6, change_ratio=0.0341, change_types={'floor_to_structure': 3, 'structure_to_floor': 3}
- room_10: changed_tiles=6, change_ratio=0.0341, change_types={'floor_to_structure': 6}
- room_3: changed_tiles=5, change_ratio=0.0284, change_types={'structure_to_floor': 5}

## Aggregate Change Types

- floor_to_structure: 93
- structure_to_floor: 48
- B_to_W: 1
