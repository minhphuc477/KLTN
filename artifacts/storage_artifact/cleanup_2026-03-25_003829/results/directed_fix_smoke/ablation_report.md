# Ablation Study Report

## Configurations
- `FULL`: {'name': 'FULL', 'use_evolution': True, 'random_topology': False, 'use_wfc': True, 'pure_wfc': False, 'logic_guidance_scale': 1.0, 'latent_sampler': 'diffusion', 'categorical_codebook_size': None, 'use_tpe': True, 'disable_graph_node_cross_attention': False}
- `COND_NO_TPE`: {'name': 'COND_NO_TPE', 'use_evolution': True, 'random_topology': False, 'use_wfc': True, 'pure_wfc': False, 'logic_guidance_scale': 1.0, 'latent_sampler': 'diffusion', 'categorical_codebook_size': None, 'use_tpe': False, 'disable_graph_node_cross_attention': False}
- `COND_WEAK_GRAPH`: {'name': 'COND_WEAK_GRAPH', 'use_evolution': True, 'random_topology': False, 'use_wfc': True, 'pure_wfc': False, 'logic_guidance_scale': 1.0, 'latent_sampler': 'diffusion', 'categorical_codebook_size': None, 'use_tpe': False, 'disable_graph_node_cross_attention': True}

## Summary Metrics

         config  n  success_rate  solvability_rate  confusion_ratio  confusion_index  path_optimal  tile_prior_kl  graph_edit_distance  generation_time_sec  novelty  reconstruction_error  constraint_valid_rate  room_repair_rate  tiles_repaired  topology_representable_edge_rate  topology_edge_connection_recall  topology_phantom_connection_rate  topology_preservation_score  directed_representable_edge_rate  directed_edge_realization_rate  directed_directionality_leak_rate  directed_edge_preservation_score  diversity
           FULL  2           0.0               0.0              NaN              NaN           0.0            NaN                  NaN             0.625770      NaN                   NaN                    NaN               NaN             NaN                               NaN                              NaN                               NaN                          NaN                               NaN                             NaN                                NaN                               NaN        0.0
    COND_NO_TPE  2           0.0               0.0              NaN              NaN           0.0            NaN                  NaN             0.494031      NaN                   NaN                    NaN               NaN             NaN                               NaN                              NaN                               NaN                          NaN                               NaN                             NaN                                NaN                               NaN        0.0
COND_WEAK_GRAPH  2           0.0               0.0              NaN              NaN           0.0            NaN                  NaN             0.560740      NaN                   NaN                    NaN               NaN             NaN                               NaN                              NaN                               NaN                          NaN                               NaN                             NaN                                NaN                               NaN        0.0

## Paired Significance (vs FULL)

         config              metric  n_pairs  delta_mean_cfg_minus_full  delta_ci_low  delta_ci_high  p_value  effect_size_d  p_value_bh_fdr  significant_fdr_0_05
    COND_NO_TPE            solvable        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
    COND_NO_TPE        path_optimal        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
    COND_NO_TPE generation_time_sec        2                  -0.131738     -0.255965      -0.007511 0.511372      -1.060465             1.0                 False
COND_WEAK_GRAPH            solvable        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
COND_WEAK_GRAPH        path_optimal        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
COND_WEAK_GRAPH generation_time_sec        2                  -0.065029     -0.217869       0.087811 1.000000      -0.425473             1.0                 False