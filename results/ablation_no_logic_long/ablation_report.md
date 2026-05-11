# Ablation Study Report

## Configurations
- `NO_LOGIC`: {'name': 'NO_LOGIC', 'use_evolution': True, 'random_topology': False, 'use_wfc': True, 'pure_wfc': False, 'logic_guidance_scale': 0.0, 'latent_sampler': 'diffusion', 'categorical_codebook_size': None, 'use_tpe': True, 'disable_graph_node_cross_attention': False, 'topology_refinement_mode': 'gat2', 'room_generator_mode': 'latent_diffusion', 'use_reference_room_maps': None}

## Summary Metrics

  config  n  success_rate  solvability_rate  confusion_ratio  confusion_index  path_optimal  tile_prior_kl  graph_edit_distance  generation_time_sec  novelty  reconstruction_error  constraint_valid_rate  room_repair_rate  tiles_repaired  topology_representable_edge_rate  topology_edge_connection_recall  topology_phantom_connection_rate  topology_preservation_score  directed_representable_edge_rate  directed_edge_realization_rate  directed_directionality_leak_rate  directed_edge_preservation_score  diversity
NO_LOGIC  4           1.0               0.5         1.988409       524.452082      0.257715       1.176914              0.11384           496.595763 0.066915              0.210227                    1.0          0.902273          127.25                          0.645833                         0.104167                               0.0                       0.4375                          0.188259                             0.0                                0.0                               0.0   0.070089

## Paired Significance (vs FULL)

_No paired comparisons available_