# Ablation Study Report

## Configurations
- `FULL`: {'name': 'FULL', 'use_evolution': True, 'random_topology': False, 'use_wfc': True, 'pure_wfc': False, 'logic_guidance_scale': 1.0, 'latent_sampler': 'diffusion', 'categorical_codebook_size': None, 'use_tpe': True, 'disable_graph_node_cross_attention': False}
- `COND_NO_TPE`: {'name': 'COND_NO_TPE', 'use_evolution': True, 'random_topology': False, 'use_wfc': True, 'pure_wfc': False, 'logic_guidance_scale': 1.0, 'latent_sampler': 'diffusion', 'categorical_codebook_size': None, 'use_tpe': False, 'disable_graph_node_cross_attention': False}
- `COND_WEAK_GRAPH`: {'name': 'COND_WEAK_GRAPH', 'use_evolution': True, 'random_topology': False, 'use_wfc': True, 'pure_wfc': False, 'logic_guidance_scale': 1.0, 'latent_sampler': 'diffusion', 'categorical_codebook_size': None, 'use_tpe': False, 'disable_graph_node_cross_attention': True}

## Summary Metrics

         config  n  success_rate  solvability_rate  confusion_ratio  confusion_index  path_optimal  tile_prior_kl  graph_edit_distance  generation_time_sec  novelty  reconstruction_error  constraint_valid_rate  room_repair_rate  tiles_repaired  topology_representable_edge_rate  topology_edge_connection_recall  topology_phantom_connection_rate  topology_preservation_score  directed_representable_edge_rate  directed_edge_realization_rate  directed_directionality_leak_rate  directed_edge_preservation_score  diversity
           FULL 12           1.0               0.0              NaN              0.0           0.0      13.539349             0.218213            20.891714 0.064285                   0.0                    1.0               0.0             0.0                          0.287037                              1.0                               0.0                     0.679167                          0.039782                        0.416667                           0.416667                               0.0   0.096478
    COND_NO_TPE 12           1.0               0.0              NaN              0.0           0.0      13.539349             0.218213            21.316287 0.064285                   0.0                    1.0               0.0             0.0                          0.287037                              1.0                               0.0                     0.679167                          0.039782                        0.416667                           0.416667                               0.0   0.096478
COND_WEAK_GRAPH 12           1.0               0.0              NaN              0.0           0.0      13.539349             0.218213            39.176158 0.064285                   0.0                    1.0               0.0             0.0                          0.287037                              1.0                               0.0                     0.679167                          0.039782                        0.416667                           0.416667                               0.0   0.096478

## Paired Significance (vs FULL)

         config                            metric  n_pairs  delta_mean_cfg_minus_full  delta_ci_low  delta_ci_high  p_value  effect_size_d  p_value_bh_fdr  significant_fdr_0_05
    COND_NO_TPE                          solvable       12                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
    COND_NO_TPE                   confusion_index       12                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
    COND_NO_TPE                      path_optimal       12                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
    COND_NO_TPE                     tile_prior_kl       12                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
    COND_NO_TPE               graph_edit_distance       12                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
    COND_NO_TPE               generation_time_sec       12                   0.424573     -0.369763       1.349613 0.405149       0.280618        1.000000                 False
    COND_NO_TPE                           novelty       12                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
    COND_NO_TPE              reconstruction_error       12                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
    COND_NO_TPE                  constraint_valid       12                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
    COND_NO_TPE                  room_repair_rate       12                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
    COND_NO_TPE                    tiles_repaired       12                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
    COND_NO_TPE  topology_representable_edge_rate       12                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
    COND_NO_TPE   topology_edge_connection_recall       12                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
    COND_NO_TPE  topology_phantom_connection_rate       12                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
    COND_NO_TPE       topology_preservation_score       12                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
    COND_NO_TPE  directed_representable_edge_rate       12                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
    COND_NO_TPE    directed_edge_realization_rate       12                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
    COND_NO_TPE directed_directionality_leak_rate       12                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
    COND_NO_TPE  directed_edge_preservation_score       12                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
COND_WEAK_GRAPH                          solvable       12                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
COND_WEAK_GRAPH                   confusion_index       12                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
COND_WEAK_GRAPH                      path_optimal       12                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
COND_WEAK_GRAPH                     tile_prior_kl       12                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
COND_WEAK_GRAPH               graph_edit_distance       12                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
COND_WEAK_GRAPH               generation_time_sec       12                  18.284444      9.183972      29.479043 0.001000       1.010865        0.037991                  True
COND_WEAK_GRAPH                           novelty       12                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
COND_WEAK_GRAPH              reconstruction_error       12                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
COND_WEAK_GRAPH                  constraint_valid       12                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
COND_WEAK_GRAPH                  room_repair_rate       12                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
COND_WEAK_GRAPH                    tiles_repaired       12                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
COND_WEAK_GRAPH  topology_representable_edge_rate       12                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
COND_WEAK_GRAPH   topology_edge_connection_recall       12                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
COND_WEAK_GRAPH  topology_phantom_connection_rate       12                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
COND_WEAK_GRAPH       topology_preservation_score       12                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
COND_WEAK_GRAPH  directed_representable_edge_rate       12                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
COND_WEAK_GRAPH    directed_edge_realization_rate       12                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
COND_WEAK_GRAPH directed_directionality_leak_rate       12                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
COND_WEAK_GRAPH  directed_edge_preservation_score       12                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False