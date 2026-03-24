# Ablation Study Report

## Configurations
- `FULL`: {'name': 'FULL', 'use_evolution': True, 'random_topology': False, 'use_wfc': True, 'pure_wfc': False, 'logic_guidance_scale': 1.0, 'latent_sampler': 'diffusion', 'categorical_codebook_size': None, 'use_tpe': True, 'disable_graph_node_cross_attention': False}
- `COND_WEAK_GRAPH`: {'name': 'COND_WEAK_GRAPH', 'use_evolution': True, 'random_topology': False, 'use_wfc': True, 'pure_wfc': False, 'logic_guidance_scale': 1.0, 'latent_sampler': 'diffusion', 'categorical_codebook_size': None, 'use_tpe': False, 'disable_graph_node_cross_attention': True}

## Summary Metrics

         config  n  success_rate  solvability_rate  confusion_ratio  confusion_index  path_optimal  tile_prior_kl  graph_edit_distance  generation_time_sec  novelty  reconstruction_error  constraint_valid_rate  room_repair_rate  tiles_repaired  topology_representable_edge_rate  topology_edge_connection_recall  topology_phantom_connection_rate  topology_preservation_score  directed_representable_edge_rate  directed_edge_realization_rate  directed_directionality_leak_rate  directed_edge_preservation_score  diversity
           FULL  4           1.0               0.0              NaN              0.0           0.0      13.670064             0.209422            45.336892 0.095691                   0.0                    1.0               0.0             0.0                              0.25                              1.0                               0.0                       0.6625                           0.03125                             0.5                                0.0                               0.5    0.10588
COND_WEAK_GRAPH  4           1.0               0.0              NaN              0.0           0.0      13.670064             0.209422            16.940955 0.095691                   0.0                    1.0               0.0             0.0                              0.25                              1.0                               0.0                       0.6625                           0.03125                             0.5                                0.0                               0.5    0.10588

## Paired Significance (vs FULL)

         config                            metric  n_pairs  delta_mean_cfg_minus_full  delta_ci_low  delta_ci_high  p_value  effect_size_d  p_value_bh_fdr  significant_fdr_0_05
COND_WEAK_GRAPH                          solvable        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
COND_WEAK_GRAPH                   confusion_index        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
COND_WEAK_GRAPH                      path_optimal        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
COND_WEAK_GRAPH                     tile_prior_kl        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
COND_WEAK_GRAPH               graph_edit_distance        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
COND_WEAK_GRAPH               generation_time_sec        4                 -28.395938    -41.299805      -8.978512 0.131217      -1.704257             1.0                 False
COND_WEAK_GRAPH                           novelty        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
COND_WEAK_GRAPH              reconstruction_error        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
COND_WEAK_GRAPH                  constraint_valid        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
COND_WEAK_GRAPH                  room_repair_rate        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
COND_WEAK_GRAPH                    tiles_repaired        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
COND_WEAK_GRAPH  topology_representable_edge_rate        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
COND_WEAK_GRAPH   topology_edge_connection_recall        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
COND_WEAK_GRAPH  topology_phantom_connection_rate        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
COND_WEAK_GRAPH       topology_preservation_score        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
COND_WEAK_GRAPH  directed_representable_edge_rate        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
COND_WEAK_GRAPH    directed_edge_realization_rate        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
COND_WEAK_GRAPH directed_directionality_leak_rate        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
COND_WEAK_GRAPH  directed_edge_preservation_score        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False