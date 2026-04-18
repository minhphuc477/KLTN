# Room Branch Benchmark

## Configurations
- `LATENT_REF_ON`: {'name': 'LATENT_REF_ON', 'use_evolution': True, 'random_topology': False, 'use_wfc': True, 'pure_wfc': False, 'logic_guidance_scale': 1.0, 'latent_sampler': 'diffusion', 'categorical_codebook_size': None, 'use_tpe': True, 'disable_graph_node_cross_attention': False, 'topology_refinement_mode': 'gat2', 'room_generator_mode': 'latent_diffusion', 'use_reference_room_maps': True}
- `LATENT_REF_OFF`: {'name': 'LATENT_REF_OFF', 'use_evolution': True, 'random_topology': False, 'use_wfc': True, 'pure_wfc': False, 'logic_guidance_scale': 1.0, 'latent_sampler': 'diffusion', 'categorical_codebook_size': None, 'use_tpe': True, 'disable_graph_node_cross_attention': False, 'topology_refinement_mode': 'gat2', 'room_generator_mode': 'latent_diffusion', 'use_reference_room_maps': False}
- `MASKED_REF_ON`: {'name': 'MASKED_REF_ON', 'use_evolution': True, 'random_topology': False, 'use_wfc': True, 'pure_wfc': False, 'logic_guidance_scale': 1.0, 'latent_sampler': 'diffusion', 'categorical_codebook_size': None, 'use_tpe': True, 'disable_graph_node_cross_attention': False, 'topology_refinement_mode': 'gat2', 'room_generator_mode': 'discrete_masked', 'use_reference_room_maps': True}
- `MASKED_REF_OFF`: {'name': 'MASKED_REF_OFF', 'use_evolution': True, 'random_topology': False, 'use_wfc': True, 'pure_wfc': False, 'logic_guidance_scale': 1.0, 'latent_sampler': 'diffusion', 'categorical_codebook_size': None, 'use_tpe': True, 'disable_graph_node_cross_attention': False, 'topology_refinement_mode': 'gat2', 'room_generator_mode': 'discrete_masked', 'use_reference_room_maps': False}

## Summary

        config  n  success_rate  solvability_rate  confusion_ratio  confusion_index  path_optimal  tile_prior_kl  graph_edit_distance  generation_time_sec  novelty  reconstruction_error  constraint_valid_rate  room_repair_rate  tiles_repaired  topology_representable_edge_rate  topology_edge_connection_recall  topology_phantom_connection_rate  topology_preservation_score  directed_representable_edge_rate  directed_edge_realization_rate  directed_directionality_leak_rate  directed_edge_preservation_score  diversity
 LATENT_REF_ON  4           0.0               0.0              NaN              NaN           0.0            NaN                  NaN             3.055928      NaN                   NaN                    NaN               NaN             NaN                               NaN                              NaN                               NaN                          NaN                               NaN                             NaN                                NaN                               NaN        0.0
LATENT_REF_OFF  4           0.0               0.0              NaN              NaN           0.0            NaN                  NaN             3.298676      NaN                   NaN                    NaN               NaN             NaN                               NaN                              NaN                               NaN                          NaN                               NaN                             NaN                                NaN                               NaN        0.0
 MASKED_REF_ON  4           0.0               0.0              NaN              NaN           0.0            NaN                  NaN             3.373319      NaN                   NaN                    NaN               NaN             NaN                               NaN                              NaN                               NaN                          NaN                               NaN                             NaN                                NaN                               NaN        0.0
MASKED_REF_OFF  4           0.0               0.0              NaN              NaN           0.0            NaN                  NaN             3.238616      NaN                   NaN                    NaN               NaN             NaN                               NaN                              NaN                               NaN                          NaN                               NaN                             NaN                                NaN                               NaN        0.0

## Significance vs LATENT_REF_ON

        config              metric  n_pairs  delta_mean_cfg_minus_full  delta_ci_low  delta_ci_high  p_value  effect_size_d  p_value_bh_fdr  significant_fdr_0_05
LATENT_REF_OFF            solvable        4                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
LATENT_REF_OFF        path_optimal        4                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
LATENT_REF_OFF generation_time_sec        4                   0.242748     -0.047558       0.458070 0.242689       0.897166        0.728068                 False
MASKED_REF_OFF            solvable        4                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
MASKED_REF_OFF        path_optimal        4                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
MASKED_REF_OFF generation_time_sec        4                   0.182687      0.030891       0.408825 0.120970       0.891006        0.728068                 False
 MASKED_REF_ON            solvable        4                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
 MASKED_REF_ON        path_optimal        4                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
 MASKED_REF_ON generation_time_sec        4                   0.317390      0.059037       0.567918 0.242689       1.260353        0.728068                 False

## Notes

- This script closes the reproducibility gap for room-branch matched-budget comparisons inside the repo.
- External HouseDiffusion/LayoutDM-style baseline runs remain a separate experiment layer.