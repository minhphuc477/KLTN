# Ablation Study Report

## Configurations
- `FULL`: {'name': 'FULL', 'use_evolution': True, 'random_topology': False, 'use_wfc': True, 'pure_wfc': False, 'logic_guidance_scale': 1.0, 'latent_sampler': 'diffusion', 'categorical_codebook_size': None, 'use_tpe': True, 'disable_graph_node_cross_attention': False, 'topology_refinement_mode': 'gat2', 'room_generator_mode': 'latent_diffusion', 'use_reference_room_maps': None}

## Summary Metrics

config  n  success_rate  solvability_rate  confusion_ratio  confusion_index  path_optimal  tile_prior_kl  graph_edit_distance  generation_time_sec  novelty  reconstruction_error  constraint_valid_rate  room_repair_rate  tiles_repaired  topology_representable_edge_rate  topology_edge_connection_recall  topology_phantom_connection_rate  topology_preservation_score  directed_representable_edge_rate  directed_edge_realization_rate  directed_directionality_leak_rate  directed_edge_preservation_score  diversity
  FULL  2           1.0               0.0              NaN       329.071336           0.0       1.493934             0.189772            64.897695 0.105893              0.153409                    1.0          0.850427           103.5                          0.604167                         0.410714                          0.142857                     0.542411                          0.302632                             0.2                                0.5                               0.0   0.343124

## Paired Significance (vs FULL)

_No paired comparisons available_