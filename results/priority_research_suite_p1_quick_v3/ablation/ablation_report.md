# Ablation Study Report

## Configurations
- `FULL`: {'name': 'FULL', 'use_evolution': True, 'use_wfc': True, 'logic_guidance_scale': 1.0, 'latent_sampler': 'diffusion', 'categorical_codebook_size': None, 'use_tpe': True}
- `NO_EVOLUTION`: {'name': 'NO_EVOLUTION', 'use_evolution': False, 'use_wfc': True, 'logic_guidance_scale': 1.0, 'latent_sampler': 'diffusion', 'categorical_codebook_size': None, 'use_tpe': True}

## Summary Metrics

      config  n  success_rate  solvability_rate  confusion_ratio  confusion_index  path_optimal  tile_prior_kl  graph_edit_distance  generation_time_sec  novelty  reconstruction_error  constraint_valid_rate  room_repair_rate  tiles_repaired  diversity
        FULL  2           1.0               0.5              1.0         0.000000           0.5       4.998085             0.350463             8.781111 0.098860              0.750000                    0.0             0.000             0.0   0.257904
NO_EVOLUTION  1           1.0               0.0              NaN       181.932927           0.0       3.488550             0.278503           292.386624 0.130642              0.755682                    1.0             0.125           147.0   0.000000

## Paired Significance (vs FULL)

      config               metric  n_pairs  delta_mean_cfg_minus_full  delta_ci_low  delta_ci_high  p_value  effect_size_d  p_value_bh_fdr  significant_fdr_0_05
NO_EVOLUTION             solvable        1                  -1.000000     -1.000000      -1.000000      1.0            0.0             1.0                 False
NO_EVOLUTION      confusion_index        1                 181.932927    181.932927     181.932927      1.0            0.0             1.0                 False
NO_EVOLUTION         path_optimal        1                  -1.000000     -1.000000      -1.000000      1.0            0.0             1.0                 False
NO_EVOLUTION        tile_prior_kl        1                  -1.395411     -1.395411      -1.395411      1.0            0.0             1.0                 False
NO_EVOLUTION  graph_edit_distance        1                  -0.070000     -0.070000      -0.070000      1.0            0.0             1.0                 False
NO_EVOLUTION  generation_time_sec        1                 282.442065    282.442065     282.442065      1.0            0.0             1.0                 False
NO_EVOLUTION              novelty        1                   0.033374      0.033374       0.033374      1.0            0.0             1.0                 False
NO_EVOLUTION reconstruction_error        1                   0.034091      0.034091       0.034091      1.0            0.0             1.0                 False
NO_EVOLUTION     constraint_valid        1                   1.000000      1.000000       1.000000      1.0            0.0             1.0                 False
NO_EVOLUTION     room_repair_rate        1                   0.125000      0.125000       0.125000      1.0            0.0             1.0                 False
NO_EVOLUTION       tiles_repaired        1                 147.000000    147.000000     147.000000      1.0            0.0             1.0                 False