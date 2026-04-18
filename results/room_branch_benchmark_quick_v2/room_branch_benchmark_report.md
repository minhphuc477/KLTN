# Room Branch Benchmark

## Configurations
- `LATENT_REF_ON`: {'name': 'LATENT_REF_ON', 'use_evolution': True, 'random_topology': False, 'use_wfc': True, 'pure_wfc': False, 'logic_guidance_scale': 1.0, 'latent_sampler': 'diffusion', 'categorical_codebook_size': None, 'use_tpe': True, 'disable_graph_node_cross_attention': False, 'topology_refinement_mode': 'gat2', 'room_generator_mode': 'latent_diffusion', 'use_reference_room_maps': True}
- `LATENT_REF_OFF`: {'name': 'LATENT_REF_OFF', 'use_evolution': True, 'random_topology': False, 'use_wfc': True, 'pure_wfc': False, 'logic_guidance_scale': 1.0, 'latent_sampler': 'diffusion', 'categorical_codebook_size': None, 'use_tpe': True, 'disable_graph_node_cross_attention': False, 'topology_refinement_mode': 'gat2', 'room_generator_mode': 'latent_diffusion', 'use_reference_room_maps': False}
- `MASKED_REF_ON`: {'name': 'MASKED_REF_ON', 'use_evolution': True, 'random_topology': False, 'use_wfc': True, 'pure_wfc': False, 'logic_guidance_scale': 1.0, 'latent_sampler': 'diffusion', 'categorical_codebook_size': None, 'use_tpe': True, 'disable_graph_node_cross_attention': False, 'topology_refinement_mode': 'gat2', 'room_generator_mode': 'discrete_masked', 'use_reference_room_maps': True}
- `MASKED_REF_OFF`: {'name': 'MASKED_REF_OFF', 'use_evolution': True, 'random_topology': False, 'use_wfc': True, 'pure_wfc': False, 'logic_guidance_scale': 1.0, 'latent_sampler': 'diffusion', 'categorical_codebook_size': None, 'use_tpe': True, 'disable_graph_node_cross_attention': False, 'topology_refinement_mode': 'gat2', 'room_generator_mode': 'discrete_masked', 'use_reference_room_maps': False}

## Summary

        config  n  success_rate  solvability_rate  confusion_ratio  confusion_index  path_optimal  tile_prior_kl  graph_edit_distance  generation_time_sec  novelty  reconstruction_error  constraint_valid_rate  room_repair_rate  tiles_repaired  topology_representable_edge_rate  topology_edge_connection_recall  topology_phantom_connection_rate  topology_preservation_score  directed_representable_edge_rate  directed_edge_realization_rate  directed_directionality_leak_rate  directed_edge_preservation_score  diversity
 LATENT_REF_ON  4           1.0              0.25         1.956522         5.253548      0.127778       1.288194             0.106712            23.331603 0.079963              0.198864                    1.0          0.966880          159.75                          0.594863                         0.497222                            0.3125                     0.560189                          0.172835                        0.329167                              0.625                          0.133333   0.200219
LATENT_REF_OFF  4           1.0              0.25         1.956522         5.253548      0.127778       1.285849             0.106712            20.446906 0.079963              0.198864                    1.0          0.966880          159.75                          0.594863                         0.497222                            0.3125                     0.560189                          0.172835                        0.329167                              0.625                          0.133333   0.200219
 MASKED_REF_ON  4           1.0              0.25         1.695652         4.273061      0.147436       1.297491             0.106712            61.425199 0.079963              0.208807                    1.0          0.865919          159.75                          0.594863                         0.497222                            0.3125                     0.560189                          0.172835                        0.329167                              0.625                          0.133333   0.200219
MASKED_REF_OFF  4           1.0              0.25         1.695652         3.734762      0.147436       1.294670             0.106712            57.063682 0.079963              0.207386                    1.0          0.865919          162.25                          0.594863                         0.497222                            0.3125                     0.560189                          0.172835                        0.329167                              0.625                          0.133333   0.200219

## Significance vs LATENT_REF_ON

        config                            metric  n_pairs  delta_mean_cfg_minus_full  delta_ci_low  delta_ci_high  p_value  effect_size_d  p_value_bh_fdr  significant_fdr_0_05
LATENT_REF_OFF                          solvable        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
LATENT_REF_OFF                   confusion_ratio        1                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
LATENT_REF_OFF                   confusion_index        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
LATENT_REF_OFF                      path_optimal        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
LATENT_REF_OFF                     tile_prior_kl        4                  -0.002346     -0.003906      -0.000822 0.132467      -1.375594             1.0                 False
LATENT_REF_OFF               graph_edit_distance        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
LATENT_REF_OFF               generation_time_sec        4                  -2.884697     -8.602188       0.500483 0.511622      -0.586189             1.0                 False
LATENT_REF_OFF                           novelty        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
LATENT_REF_OFF              reconstruction_error        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
LATENT_REF_OFF                  constraint_valid        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
LATENT_REF_OFF                  room_repair_rate        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
LATENT_REF_OFF                    tiles_repaired        4                   0.000000     -2.000000       1.500000 1.000000       0.000000             1.0                 False
LATENT_REF_OFF  topology_representable_edge_rate        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
LATENT_REF_OFF   topology_edge_connection_recall        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
LATENT_REF_OFF  topology_phantom_connection_rate        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
LATENT_REF_OFF       topology_preservation_score        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
LATENT_REF_OFF  directed_representable_edge_rate        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
LATENT_REF_OFF    directed_edge_realization_rate        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
LATENT_REF_OFF directed_directionality_leak_rate        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
LATENT_REF_OFF  directed_edge_preservation_score        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
MASKED_REF_OFF                          solvable        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
MASKED_REF_OFF                   confusion_ratio        1                  -0.260870     -0.260870      -0.260870 1.000000       0.000000             1.0                 False
MASKED_REF_OFF                   confusion_index        4                  -1.518786     -2.920355      -0.117217 0.128718      -1.064336             1.0                 False
MASKED_REF_OFF                      path_optimal        4                   0.019658      0.000000       0.058974 1.000000       0.577350             1.0                 False
MASKED_REF_OFF                     tile_prior_kl        4                   0.006476      0.003008       0.012489 0.132467       1.224552             1.0                 False
MASKED_REF_OFF               graph_edit_distance        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
MASKED_REF_OFF               generation_time_sec        4                  33.732079     25.129928      42.334230 0.120970       3.902522             1.0                 False
MASKED_REF_OFF                           novelty        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
MASKED_REF_OFF              reconstruction_error        4                   0.008523     -0.073864       0.090909 1.000000       0.096976             1.0                 False
MASKED_REF_OFF                  constraint_valid        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
MASKED_REF_OFF                  room_repair_rate        4                  -0.100962     -0.160256      -0.038462 0.249438      -1.521278             1.0                 False
MASKED_REF_OFF                    tiles_repaired        4                   2.500000    -60.500000      64.500000 0.873282       0.038004             1.0                 False
MASKED_REF_OFF  topology_representable_edge_rate        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
MASKED_REF_OFF   topology_edge_connection_recall        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
MASKED_REF_OFF  topology_phantom_connection_rate        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
MASKED_REF_OFF       topology_preservation_score        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
MASKED_REF_OFF  directed_representable_edge_rate        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
MASKED_REF_OFF    directed_edge_realization_rate        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
MASKED_REF_OFF directed_directionality_leak_rate        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
MASKED_REF_OFF  directed_edge_preservation_score        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
 MASKED_REF_ON                          solvable        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
 MASKED_REF_ON                   confusion_ratio        1                  -0.260870     -0.260870      -0.260870 1.000000       0.000000             1.0                 False
 MASKED_REF_ON                   confusion_index        4                  -0.980487     -2.519273      -0.103614 0.128718      -0.733703             1.0                 False
 MASKED_REF_ON                      path_optimal        4                   0.019658      0.000000       0.058974 1.000000       0.577350             1.0                 False
 MASKED_REF_ON                     tile_prior_kl        4                   0.009296      0.003494       0.015796 0.132467       1.392130             1.0                 False
 MASKED_REF_ON               graph_edit_distance        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
 MASKED_REF_ON               generation_time_sec        4                  38.093596     28.662520      47.761565 0.120970       3.611768             1.0                 False
 MASKED_REF_ON                           novelty        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
 MASKED_REF_ON              reconstruction_error        4                   0.009943     -0.071023       0.090909 0.876781       0.113510             1.0                 False
 MASKED_REF_ON                  constraint_valid        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
 MASKED_REF_ON                  room_repair_rate        4                  -0.100962     -0.160256      -0.038462 0.249438      -1.521278             1.0                 False
 MASKED_REF_ON                    tiles_repaired        4                   0.000000    -64.000000      64.000000 1.000000       0.000000             1.0                 False
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