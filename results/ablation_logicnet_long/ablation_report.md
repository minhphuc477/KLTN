# Ablation Study Report

## Configurations
- `FULL`: {'name': 'FULL', 'use_evolution': True, 'random_topology': False, 'use_wfc': True, 'pure_wfc': False, 'logic_guidance_scale': 1.0, 'latent_sampler': 'diffusion', 'categorical_codebook_size': None, 'use_tpe': True, 'disable_graph_node_cross_attention': False, 'topology_refinement_mode': 'gat2', 'room_generator_mode': 'latent_diffusion', 'use_reference_room_maps': None}
- `NO_LOGIC`: {'name': 'NO_LOGIC', 'use_evolution': True, 'random_topology': False, 'use_wfc': True, 'pure_wfc': False, 'logic_guidance_scale': 0.0, 'latent_sampler': 'diffusion', 'categorical_codebook_size': None, 'use_tpe': True, 'disable_graph_node_cross_attention': False, 'topology_refinement_mode': 'gat2', 'room_generator_mode': 'latent_diffusion', 'use_reference_room_maps': None}

## Summary Metrics

  config  n  success_rate  solvability_rate  confusion_ratio  confusion_index  path_optimal  tile_prior_kl  graph_edit_distance  generation_time_sec  novelty  reconstruction_error  constraint_valid_rate  room_repair_rate  tiles_repaired  topology_representable_edge_rate  topology_edge_connection_recall  topology_phantom_connection_rate  topology_preservation_score  directed_representable_edge_rate  directed_edge_realization_rate  directed_directionality_leak_rate  directed_edge_preservation_score  diversity
    FULL  3           1.0          0.666667         1.953634       285.475244      0.346444       1.112153              0.13762          2005.122015 0.081239               0.17803                    1.0          0.969697       96.333333                          0.666667                         0.138889                               0.0                       0.4625                          0.180837                             0.0                                0.0                               0.0   0.071905
NO_LOGIC  0           0.0          0.000000              NaN              NaN      0.000000            NaN                  NaN                  NaN      NaN                   NaN                    NaN               NaN             NaN                               NaN                              NaN                               NaN                          NaN                               NaN                             NaN                                NaN                               NaN   0.000000

## Paired Significance (vs FULL)

_No paired comparisons available_