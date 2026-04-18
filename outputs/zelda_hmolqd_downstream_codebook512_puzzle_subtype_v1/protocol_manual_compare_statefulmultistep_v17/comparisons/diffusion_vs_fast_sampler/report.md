# Room Variant Comparison

Baseline: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_statefulmultistep_v17\diffusion_cfg3_logic0_steps50`
Candidate: `outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1\protocol_manual_compare_statefulmultistep_v17\fast_cfg3_logic0_steps4`

Compared rooms: 12
Total changed tiles: 28

## Worst Rooms

- room_11: changed_tiles=11, change_ratio=0.0625, change_types={'structure_to_floor': 11}
- room_6: changed_tiles=7, change_ratio=0.0398, change_types={'structure_to_floor': 7}
- room_9: changed_tiles=5, change_ratio=0.0284, change_types={'structure_to_floor': 5}
- room_5: changed_tiles=3, change_ratio=0.0170, change_types={'structure_to_floor': 3}
- room_2: changed_tiles=2, change_ratio=0.0114, change_types={'floor_to_structure': 2}
- room_0: changed_tiles=0, change_ratio=0.0000, change_types={}
- room_1: changed_tiles=0, change_ratio=0.0000, change_types={}
- room_3: changed_tiles=0, change_ratio=0.0000, change_types={}

## Aggregate Change Types

- structure_to_floor: 26
- floor_to_structure: 2
