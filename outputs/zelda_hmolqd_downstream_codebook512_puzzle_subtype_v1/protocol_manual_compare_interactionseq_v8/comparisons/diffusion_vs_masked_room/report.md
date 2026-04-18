# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_interactionseq_v8\diffusion_cfg3_logic0_steps50`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_interactionseq_v8\masked_room_full`

Compared rooms: 12
Total changed tiles: 193

## Worst Rooms

- room_0: changed_tiles=35, change_ratio=0.1989, change_types={'structure_to_floor': 35}
- room_3: changed_tiles=28, change_ratio=0.1591, change_types={'structure_to_floor': 28}
- room_8: changed_tiles=24, change_ratio=0.1364, change_types={'structure_to_floor': 22, 'floor_to_structure': 2}
- room_9: changed_tiles=24, change_ratio=0.1364, change_types={'floor_to_structure': 22, 'structure_to_floor': 2}
- room_7: changed_tiles=23, change_ratio=0.1307, change_types={'structure_to_floor': 23}
- room_1: changed_tiles=12, change_ratio=0.0682, change_types={'floor_to_structure': 11, 'structure_to_floor': 1}
- room_4: changed_tiles=12, change_ratio=0.0682, change_types={'structure_to_floor': 8, 'floor_to_structure': 4}
- room_6: changed_tiles=11, change_ratio=0.0625, change_types={'floor_to_structure': 11}

## Aggregate Change Types

- structure_to_floor: 122
- floor_to_structure: 71
