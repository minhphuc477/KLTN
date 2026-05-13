# Ablation Study Report

## Configurations
- `FULL`: {'name': 'FULL', 'use_evolution': True, 'random_topology': False, 'use_wfc': True, 'pure_wfc': False, 'logic_guidance_scale': 1.0, 'latent_sampler': 'diffusion', 'categorical_codebook_size': None, 'use_tpe': True, 'disable_graph_node_cross_attention': False, 'topology_refinement_mode': 'gat2', 'room_generator_mode': 'latent_diffusion', 'use_reference_room_maps': None}
- `NO_LOGIC`: {'name': 'NO_LOGIC', 'use_evolution': True, 'random_topology': False, 'use_wfc': True, 'pure_wfc': False, 'logic_guidance_scale': 0.0, 'latent_sampler': 'diffusion', 'categorical_codebook_size': None, 'use_tpe': True, 'disable_graph_node_cross_attention': False, 'topology_refinement_mode': 'gat2', 'room_generator_mode': 'latent_diffusion', 'use_reference_room_maps': None}

## Summary Metrics

  config  n  success_rate  solvability_rate  confusion_ratio  confusion_index  path_optimal  tile_prior_kl  graph_edit_distance  generation_time_sec  novelty  reconstruction_error  constraint_valid_rate  room_repair_rate  tiles_repaired  topology_representable_edge_rate  topology_edge_connection_recall  topology_phantom_connection_rate  topology_preservation_score  directed_representable_edge_rate  directed_edge_realization_rate  directed_directionality_leak_rate  directed_edge_preservation_score  diversity
    FULL  4           1.0              0.25         1.577778      1516.507594      0.158451       1.279065             0.066249           164.220486 0.026258                   1.0                    1.0          0.045455             1.0                          0.461976                         0.071429                            0.1875                     0.321282                          0.156568                             0.0                                0.0                               0.0   0.185945
NO_LOGIC  4           1.0              0.25         1.577778      1516.507594      0.158451       1.279065             0.066249           160.222265 0.026258                   1.0                    1.0          0.045455             1.0                          0.461976                         0.071429                            0.1875                     0.321282                          0.156568                             0.0                                0.0                               0.0   0.185945

## Paired Significance (vs FULL)

  config                            metric  n_pairs  delta_mean_cfg_minus_full  delta_ci_low  delta_ci_high  p_value  effect_size_d  p_value_bh_fdr  significant_fdr_0_05
NO_LOGIC                          solvable        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
NO_LOGIC                   confusion_ratio        1                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
NO_LOGIC                   confusion_index        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
NO_LOGIC                      path_optimal        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
NO_LOGIC                     tile_prior_kl        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
NO_LOGIC               graph_edit_distance        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
NO_LOGIC               generation_time_sec        4                  -3.998221     -7.039773      -0.956668 0.131217      -1.300164             1.0                 False
NO_LOGIC                           novelty        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
NO_LOGIC              reconstruction_error        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
NO_LOGIC                  constraint_valid        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
NO_LOGIC                  room_repair_rate        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
NO_LOGIC                    tiles_repaired        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
NO_LOGIC  topology_representable_edge_rate        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
NO_LOGIC   topology_edge_connection_recall        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
NO_LOGIC  topology_phantom_connection_rate        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
NO_LOGIC       topology_preservation_score        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
NO_LOGIC  directed_representable_edge_rate        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
NO_LOGIC    directed_edge_realization_rate        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
NO_LOGIC directed_directionality_leak_rate        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
NO_LOGIC  directed_edge_preservation_score        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False