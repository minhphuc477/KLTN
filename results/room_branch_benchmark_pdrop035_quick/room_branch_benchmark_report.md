# Room Branch Benchmark

## Configurations
- `LATENT_REF_ON`: {'name': 'LATENT_REF_ON', 'use_evolution': True, 'random_topology': False, 'use_wfc': True, 'pure_wfc': False, 'logic_guidance_scale': 1.0, 'latent_sampler': 'diffusion', 'categorical_codebook_size': None, 'use_tpe': True, 'disable_graph_node_cross_attention': False, 'topology_refinement_mode': 'gat2', 'room_generator_mode': 'latent_diffusion', 'use_reference_room_maps': True}
- `LATENT_REF_OFF`: {'name': 'LATENT_REF_OFF', 'use_evolution': True, 'random_topology': False, 'use_wfc': True, 'pure_wfc': False, 'logic_guidance_scale': 1.0, 'latent_sampler': 'diffusion', 'categorical_codebook_size': None, 'use_tpe': True, 'disable_graph_node_cross_attention': False, 'topology_refinement_mode': 'gat2', 'room_generator_mode': 'latent_diffusion', 'use_reference_room_maps': False}
- `MASKED_REF_ON`: {'name': 'MASKED_REF_ON', 'use_evolution': True, 'random_topology': False, 'use_wfc': True, 'pure_wfc': False, 'logic_guidance_scale': 1.0, 'latent_sampler': 'diffusion', 'categorical_codebook_size': None, 'use_tpe': True, 'disable_graph_node_cross_attention': False, 'topology_refinement_mode': 'gat2', 'room_generator_mode': 'discrete_masked', 'use_reference_room_maps': True}
- `MASKED_REF_OFF`: {'name': 'MASKED_REF_OFF', 'use_evolution': True, 'random_topology': False, 'use_wfc': True, 'pure_wfc': False, 'logic_guidance_scale': 1.0, 'latent_sampler': 'diffusion', 'categorical_codebook_size': None, 'use_tpe': True, 'disable_graph_node_cross_attention': False, 'topology_refinement_mode': 'gat2', 'room_generator_mode': 'discrete_masked', 'use_reference_room_maps': False}

## Summary

        config  n  success_rate  solvability_rate  confusion_ratio  confusion_index  path_optimal  tile_prior_kl  graph_edit_distance  generation_time_sec  novelty  reconstruction_error  constraint_valid_rate  room_repair_rate  tiles_repaired  topology_representable_edge_rate  topology_edge_connection_recall  topology_phantom_connection_rate  topology_preservation_score  directed_representable_edge_rate  directed_edge_realization_rate  directed_directionality_leak_rate  directed_edge_preservation_score  diversity
 LATENT_REF_ON  4           1.0              0.25              NaN         6.912458      0.000000       1.290505             0.106712            24.736082 0.079963              0.188920                    1.0          1.000000          168.00                          0.594863                         0.497222                            0.3125                     0.560189                          0.172835                        0.329167                              0.625                          0.133333   0.200219
LATENT_REF_OFF  4           1.0              0.25              NaN         6.912458      0.000000       1.290708             0.106712            21.337099 0.079963              0.188920                    1.0          1.000000          168.75                          0.594863                         0.497222                            0.3125                     0.560189                          0.172835                        0.329167                              0.625                          0.133333   0.200219
 MASKED_REF_ON  4           1.0              0.50              NaN         8.070151      0.000000       1.284026             0.106712            64.223791 0.079963              0.174716                    1.0          0.897436          172.25                          0.594863                         0.497222                            0.3125                     0.560189                          0.172835                        0.329167                              0.625                          0.133333   0.200219
MASKED_REF_OFF  4           1.0              0.50         1.826087         5.459537      0.136905       1.286147             0.106712            60.482607 0.079963              0.173295                    1.0          0.878205          174.00                          0.594863                         0.497222                            0.3125                     0.560189                          0.172835                        0.329167                              0.625                          0.133333   0.200219

## Significance vs LATENT_REF_ON

        config                            metric  n_pairs  delta_mean_cfg_minus_full  delta_ci_low  delta_ci_high  p_value  effect_size_d  p_value_bh_fdr  significant_fdr_0_05
LATENT_REF_OFF                          solvable        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
LATENT_REF_OFF                   confusion_index        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
LATENT_REF_OFF                      path_optimal        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
LATENT_REF_OFF                     tile_prior_kl        4                   0.000203     -0.000016       0.000421 0.509623       0.898622             1.0                 False
LATENT_REF_OFF               graph_edit_distance        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
LATENT_REF_OFF               generation_time_sec        4                  -3.398983     -9.534992       0.043084 0.379655      -0.632831             1.0                 False
LATENT_REF_OFF                           novelty        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
LATENT_REF_OFF              reconstruction_error        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
LATENT_REF_OFF                  constraint_valid        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
LATENT_REF_OFF                  room_repair_rate        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
LATENT_REF_OFF                    tiles_repaired        4                   0.750000      0.000000       1.500000 0.475631       0.904534             1.0                 False
LATENT_REF_OFF  topology_representable_edge_rate        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
LATENT_REF_OFF   topology_edge_connection_recall        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
LATENT_REF_OFF  topology_phantom_connection_rate        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
LATENT_REF_OFF       topology_preservation_score        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
LATENT_REF_OFF  directed_representable_edge_rate        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
LATENT_REF_OFF    directed_edge_realization_rate        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
LATENT_REF_OFF directed_directionality_leak_rate        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
LATENT_REF_OFF  directed_edge_preservation_score        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
MASKED_REF_OFF                          solvable        4                   0.250000      0.000000       0.750000 1.000000       0.577350             1.0                 False
MASKED_REF_OFF                   confusion_index        4                  -1.452920     -7.105163       2.759940 0.739565      -0.298374             1.0                 False
MASKED_REF_OFF                      path_optimal        4                   0.136905      0.000000       0.410714 1.000000       0.577350             1.0                 False
MASKED_REF_OFF                     tile_prior_kl        4                  -0.004358     -0.010491       0.004391 0.371407      -0.536432             1.0                 False
MASKED_REF_OFF               graph_edit_distance        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
MASKED_REF_OFF               generation_time_sec        4                  35.746525     25.929018      45.564032 0.120970       3.407392             1.0                 False
MASKED_REF_OFF                           novelty        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
MASKED_REF_OFF              reconstruction_error        4                  -0.015625     -0.046875       0.005682 0.755811      -0.553470             1.0                 False
MASKED_REF_OFF                  constraint_valid        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
MASKED_REF_OFF                  room_repair_rate        4                  -0.121795     -0.166667      -0.041667 0.249438      -1.727273             1.0                 False
MASKED_REF_OFF                    tiles_repaired        4                   6.000000    -60.000000      72.000000 1.000000       0.085289             1.0                 False
MASKED_REF_OFF  topology_representable_edge_rate        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
MASKED_REF_OFF   topology_edge_connection_recall        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
MASKED_REF_OFF  topology_phantom_connection_rate        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
MASKED_REF_OFF       topology_preservation_score        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
MASKED_REF_OFF  directed_representable_edge_rate        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
MASKED_REF_OFF    directed_edge_realization_rate        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
MASKED_REF_OFF directed_directionality_leak_rate        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
MASKED_REF_OFF  directed_edge_preservation_score        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
 MASKED_REF_ON                          solvable        4                   0.250000      0.000000       0.750000 1.000000       0.577350             1.0                 False
 MASKED_REF_ON                   confusion_index        4                   1.157694     -0.009025       2.759940 0.490627       0.766706             1.0                 False
 MASKED_REF_ON                      path_optimal        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
 MASKED_REF_ON                     tile_prior_kl        4                  -0.006479     -0.012119       0.001798 0.252687      -0.870583             1.0                 False
 MASKED_REF_ON               graph_edit_distance        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
 MASKED_REF_ON               generation_time_sec        4                  39.487709     32.992356      45.983061 0.120970       5.682788             1.0                 False
 MASKED_REF_ON                           novelty        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
 MASKED_REF_ON              reconstruction_error        4                  -0.014205     -0.046875       0.007102 0.755811      -0.497519             1.0                 False
 MASKED_REF_ON                  constraint_valid        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
 MASKED_REF_ON                  room_repair_rate        4                  -0.102564     -0.166667      -0.038462 0.249438      -1.472919             1.0                 False
 MASKED_REF_ON                    tiles_repaired        4                   4.250000    -55.500000      64.000000 0.863784       0.064863             1.0                 False
 MASKED_REF_ON  topology_representable_edge_rate        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
 MASKED_REF_ON   topology_edge_connection_recall        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
 MASKED_REF_ON  topology_phantom_connection_rate        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
 MASKED_REF_ON       topology_preservation_score        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
 MASKED_REF_ON  directed_representable_edge_rate        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
 MASKED_REF_ON    directed_edge_realization_rate        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
 MASKED_REF_ON directed_directionality_leak_rate        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
 MASKED_REF_ON  directed_edge_preservation_score        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False

## Notes

- This script closes the reproducibility gap for room-branch matched-budget comparisons inside the repo.
- External HouseDiffusion/LayoutDM-style baseline runs remain a separate experiment layer.