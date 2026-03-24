# Ablation Study Report

## Configurations
- `FULL`: {'name': 'FULL', 'use_evolution': True, 'random_topology': False, 'use_wfc': True, 'pure_wfc': False, 'logic_guidance_scale': 1.0, 'latent_sampler': 'diffusion', 'categorical_codebook_size': None, 'use_tpe': True, 'disable_graph_node_cross_attention': False, 'topology_refinement_mode': 'upgraded'}
- `TOPO_LIGHTWEIGHT`: {'name': 'TOPO_LIGHTWEIGHT', 'use_evolution': True, 'random_topology': False, 'use_wfc': True, 'pure_wfc': False, 'logic_guidance_scale': 1.0, 'latent_sampler': 'diffusion', 'categorical_codebook_size': None, 'use_tpe': True, 'disable_graph_node_cross_attention': False, 'topology_refinement_mode': 'lightweight'}

## Summary Metrics

          config  n  success_rate  solvability_rate  confusion_ratio  confusion_index  path_optimal  tile_prior_kl  graph_edit_distance  generation_time_sec  novelty  reconstruction_error  constraint_valid_rate  room_repair_rate  tiles_repaired  topology_representable_edge_rate  topology_edge_connection_recall  topology_phantom_connection_rate  topology_preservation_score  directed_representable_edge_rate  directed_edge_realization_rate  directed_directionality_leak_rate  directed_edge_preservation_score  diversity
            FULL  2           1.0               0.0              NaN              0.0           0.0       3.433838             0.238586            11.763523 0.131251              0.872159                    1.0          0.133929           169.0                          0.354167                              1.0                               0.5                     0.659375                          0.033333                             0.5                                0.0                               0.5   0.176757
TOPO_LIGHTWEIGHT  2           1.0               0.0              NaN              0.0           0.0       4.867655             0.238586             9.916807 0.131251              0.795455                    1.0          0.000000             0.0                          0.354167                              1.0                               0.5                     0.659375                          0.033333                             0.5                                0.0                               0.5   0.176757

## Paired Significance (vs FULL)

          config                            metric  n_pairs  delta_mean_cfg_minus_full  delta_ci_low  delta_ci_high  p_value  effect_size_d  p_value_bh_fdr  significant_fdr_0_05
TOPO_LIGHTWEIGHT                          solvable        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
TOPO_LIGHTWEIGHT                   confusion_index        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
TOPO_LIGHTWEIGHT                      path_optimal        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
TOPO_LIGHTWEIGHT                     tile_prior_kl        2                   1.433817      1.391685       1.475949 0.501125      34.031530             1.0                 False
TOPO_LIGHTWEIGHT               graph_edit_distance        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
TOPO_LIGHTWEIGHT               generation_time_sec        2                  -1.846716     -3.054021      -0.639411 0.511372      -1.529619             1.0                 False
TOPO_LIGHTWEIGHT                           novelty        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
TOPO_LIGHTWEIGHT              reconstruction_error        2                  -0.076705     -0.198864       0.045455 1.000000      -0.627907             1.0                 False
TOPO_LIGHTWEIGHT                  constraint_valid        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
TOPO_LIGHTWEIGHT                  room_repair_rate        2                  -0.133929     -0.142857      -0.125000 0.483379     -15.000000             1.0                 False
TOPO_LIGHTWEIGHT                    tiles_repaired        2                -169.000000   -169.000000    -169.000000 0.501125       0.000000             1.0                 False
TOPO_LIGHTWEIGHT  topology_representable_edge_rate        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
TOPO_LIGHTWEIGHT   topology_edge_connection_recall        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
TOPO_LIGHTWEIGHT  topology_phantom_connection_rate        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
TOPO_LIGHTWEIGHT       topology_preservation_score        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
TOPO_LIGHTWEIGHT  directed_representable_edge_rate        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
TOPO_LIGHTWEIGHT    directed_edge_realization_rate        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
TOPO_LIGHTWEIGHT directed_directionality_leak_rate        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
TOPO_LIGHTWEIGHT  directed_edge_preservation_score        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False