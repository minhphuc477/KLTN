# Ablation Study Report

## Configurations
- `FULL`: {'name': 'FULL', 'use_evolution': True, 'random_topology': False, 'use_wfc': True, 'pure_wfc': False, 'logic_guidance_scale': 1.0, 'latent_sampler': 'diffusion', 'categorical_codebook_size': None, 'use_tpe': True}

## Summary Metrics

config  n  success_rate  solvability_rate  confusion_ratio  confusion_index  path_optimal  tile_prior_kl  graph_edit_distance  generation_time_sec  novelty  reconstruction_error  constraint_valid_rate  room_repair_rate  tiles_repaired  topology_representable_edge_rate  topology_edge_connection_recall  topology_phantom_connection_rate  topology_preservation_score  diversity
  FULL  1           1.0               0.0              NaN              0.0           0.0      13.582112             0.191003            11.938713 0.161124                   0.0                    1.0               0.0             0.0                          0.222222                              1.0                               0.0                         0.65        0.0

## Paired Significance (vs FULL)

_No paired comparisons available_