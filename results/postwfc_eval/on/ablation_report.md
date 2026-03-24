# Ablation Study Report

## Configurations
- `FULL`: {'name': 'FULL', 'use_evolution': True, 'random_topology': False, 'use_wfc': True, 'pure_wfc': False, 'logic_guidance_scale': 1.0, 'latent_sampler': 'diffusion', 'categorical_codebook_size': None, 'use_tpe': True}

## Summary Metrics

config  n  success_rate  solvability_rate  confusion_ratio  confusion_index  path_optimal  tile_prior_kl  graph_edit_distance  generation_time_sec  novelty  reconstruction_error  constraint_valid_rate  room_repair_rate  tiles_repaired  topology_representable_edge_rate  topology_edge_connection_recall  topology_phantom_connection_rate  topology_preservation_score  diversity
  FULL  4           1.0               0.0              NaN              0.0           0.0      13.659483             0.228362            21.076299 0.068887                   0.0                    1.0               0.0             0.0                              0.25                              1.0                               0.0                       0.6625   0.099895

## Paired Significance (vs FULL)

_No paired comparisons available_