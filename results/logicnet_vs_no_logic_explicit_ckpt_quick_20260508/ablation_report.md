# Ablation Study Report

## Configurations
- `FULL`: {'name': 'FULL', 'use_evolution': True, 'random_topology': False, 'use_wfc': True, 'pure_wfc': False, 'logic_guidance_scale': 1.0, 'latent_sampler': 'diffusion', 'categorical_codebook_size': None, 'use_tpe': True, 'disable_graph_node_cross_attention': False, 'topology_refinement_mode': 'gat2', 'room_generator_mode': 'latent_diffusion', 'use_reference_room_maps': None}
- `NO_LOGIC`: {'name': 'NO_LOGIC', 'use_evolution': True, 'random_topology': False, 'use_wfc': True, 'pure_wfc': False, 'logic_guidance_scale': 0.0, 'latent_sampler': 'diffusion', 'categorical_codebook_size': None, 'use_tpe': True, 'disable_graph_node_cross_attention': False, 'topology_refinement_mode': 'gat2', 'room_generator_mode': 'latent_diffusion', 'use_reference_room_maps': None}

## Summary Metrics

  config  n  success_rate  solvability_rate  confusion_ratio  confusion_index  path_optimal  tile_prior_kl  graph_edit_distance  generation_time_sec  novelty  reconstruction_error  constraint_valid_rate  room_repair_rate  tiles_repaired  topology_representable_edge_rate  topology_edge_connection_recall  topology_phantom_connection_rate  topology_preservation_score  directed_representable_edge_rate  directed_edge_realization_rate  directed_directionality_leak_rate  directed_edge_preservation_score  diversity
    FULL  2           1.0               0.5         1.531646        38.814679      0.326446       1.330074              0.11562            51.694362 0.102128              0.099432                    1.0              0.75           110.5                          0.427273                             0.45                          0.583333                     0.436439                              0.14                             0.0                                0.0                               0.0   0.334153
NO_LOGIC  2           1.0               0.5         1.531646        38.814679      0.326446       1.330079              0.11562            32.333815 0.102128              0.099432                    1.0              0.75           110.5                          0.427273                             0.45                          0.583333                     0.436439                              0.14                             0.0                                0.0                               0.0   0.334153

## Paired Significance (vs FULL)

  config                            metric  n_pairs  delta_mean_cfg_minus_full  delta_ci_low  delta_ci_high  p_value  effect_size_d  p_value_bh_fdr  significant_fdr_0_05
NO_LOGIC                          solvable        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
NO_LOGIC                   confusion_ratio        1                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
NO_LOGIC                   confusion_index        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
NO_LOGIC                      path_optimal        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
NO_LOGIC                     tile_prior_kl        2                   0.000005     -0.000470       0.000480 1.000000       0.010821             1.0                 False
NO_LOGIC               graph_edit_distance        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
NO_LOGIC               generation_time_sec        2                 -19.360546    -38.644681      -0.076412 0.502124      -1.003962             1.0                 False
NO_LOGIC                           novelty        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
NO_LOGIC              reconstruction_error        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
NO_LOGIC                  constraint_valid        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
NO_LOGIC                  room_repair_rate        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
NO_LOGIC                    tiles_repaired        2                   0.000000     -2.000000       2.000000 1.000000       0.000000             1.0                 False
NO_LOGIC  topology_representable_edge_rate        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
NO_LOGIC   topology_edge_connection_recall        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
NO_LOGIC  topology_phantom_connection_rate        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
NO_LOGIC       topology_preservation_score        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
NO_LOGIC  directed_representable_edge_rate        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
NO_LOGIC    directed_edge_realization_rate        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
NO_LOGIC directed_directionality_leak_rate        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
NO_LOGIC  directed_edge_preservation_score        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False