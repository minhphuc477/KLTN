# Ablation Study Report

## Configurations
- `FULL`: {'name': 'FULL', 'use_evolution': True, 'random_topology': False, 'use_wfc': True, 'pure_wfc': False, 'logic_guidance_scale': 1.0, 'latent_sampler': 'diffusion', 'categorical_codebook_size': None, 'use_tpe': True, 'disable_graph_node_cross_attention': False, 'topology_refinement_mode': 'gat2'}
- `TOPO_LIGHTWEIGHT`: {'name': 'TOPO_LIGHTWEIGHT', 'use_evolution': True, 'random_topology': False, 'use_wfc': True, 'pure_wfc': False, 'logic_guidance_scale': 1.0, 'latent_sampler': 'diffusion', 'categorical_codebook_size': None, 'use_tpe': True, 'disable_graph_node_cross_attention': False, 'topology_refinement_mode': 'lightweight'}

## Summary Metrics

          config  n  success_rate  solvability_rate  confusion_ratio  confusion_index  path_optimal  tile_prior_kl  graph_edit_distance  generation_time_sec  novelty  reconstruction_error  constraint_valid_rate  room_repair_rate  tiles_repaired  topology_representable_edge_rate  topology_edge_connection_recall  topology_phantom_connection_rate  topology_preservation_score  directed_representable_edge_rate  directed_edge_realization_rate  directed_directionality_leak_rate  directed_edge_preservation_score  diversity
            FULL  1           1.0               0.0              NaN              0.0           0.0       3.181022             0.268669            27.251385 0.187207              0.721591                    1.0          0.142857           172.0                             0.375                              1.0                               1.0                      0.61875                          0.066667                             1.0                                0.0                               1.0        0.0
TOPO_LIGHTWEIGHT  1           1.0               0.0              NaN              0.0           0.0       2.435912             0.268669            22.021210 0.187207              0.931818                    1.0          0.714286           706.0                             0.375                              1.0                               1.0                      0.61875                          0.066667                             1.0                                0.0                               1.0        0.0

## Paired Significance (vs FULL)

          config                            metric  n_pairs  delta_mean_cfg_minus_full  delta_ci_low  delta_ci_high  p_value  effect_size_d  p_value_bh_fdr  significant_fdr_0_05
TOPO_LIGHTWEIGHT                          solvable        1                   0.000000      0.000000       0.000000      1.0            0.0             1.0                 False
TOPO_LIGHTWEIGHT                   confusion_index        1                   0.000000      0.000000       0.000000      1.0            0.0             1.0                 False
TOPO_LIGHTWEIGHT                      path_optimal        1                   0.000000      0.000000       0.000000      1.0            0.0             1.0                 False
TOPO_LIGHTWEIGHT                     tile_prior_kl        1                  -0.745109     -0.745109      -0.745109      1.0            0.0             1.0                 False
TOPO_LIGHTWEIGHT               graph_edit_distance        1                   0.000000      0.000000       0.000000      1.0            0.0             1.0                 False
TOPO_LIGHTWEIGHT               generation_time_sec        1                  -5.230175     -5.230175      -5.230175      1.0            0.0             1.0                 False
TOPO_LIGHTWEIGHT                           novelty        1                   0.000000      0.000000       0.000000      1.0            0.0             1.0                 False
TOPO_LIGHTWEIGHT              reconstruction_error        1                   0.210227      0.210227       0.210227      1.0            0.0             1.0                 False
TOPO_LIGHTWEIGHT                  constraint_valid        1                   0.000000      0.000000       0.000000      1.0            0.0             1.0                 False
TOPO_LIGHTWEIGHT                  room_repair_rate        1                   0.571429      0.571429       0.571429      1.0            0.0             1.0                 False
TOPO_LIGHTWEIGHT                    tiles_repaired        1                 534.000000    534.000000     534.000000      1.0            0.0             1.0                 False
TOPO_LIGHTWEIGHT  topology_representable_edge_rate        1                   0.000000      0.000000       0.000000      1.0            0.0             1.0                 False
TOPO_LIGHTWEIGHT   topology_edge_connection_recall        1                   0.000000      0.000000       0.000000      1.0            0.0             1.0                 False
TOPO_LIGHTWEIGHT  topology_phantom_connection_rate        1                   0.000000      0.000000       0.000000      1.0            0.0             1.0                 False
TOPO_LIGHTWEIGHT       topology_preservation_score        1                   0.000000      0.000000       0.000000      1.0            0.0             1.0                 False
TOPO_LIGHTWEIGHT  directed_representable_edge_rate        1                   0.000000      0.000000       0.000000      1.0            0.0             1.0                 False
TOPO_LIGHTWEIGHT    directed_edge_realization_rate        1                   0.000000      0.000000       0.000000      1.0            0.0             1.0                 False
TOPO_LIGHTWEIGHT directed_directionality_leak_rate        1                   0.000000      0.000000       0.000000      1.0            0.0             1.0                 False
TOPO_LIGHTWEIGHT  directed_edge_preservation_score        1                   0.000000      0.000000       0.000000      1.0            0.0             1.0                 False