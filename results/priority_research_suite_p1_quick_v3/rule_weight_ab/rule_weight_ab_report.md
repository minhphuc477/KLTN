# Rule Weight A/B Report

- control: `CONTROL`
- treatment: `TREATMENT`
- seeds: 4
- calibrated_treatment: False

## Mean Metrics by Arm

      arm  n  overall_completeness  constraint_valid  path_exists  linearity  leniency  progression_complexity  topology_complexity  cycle_density  shortcut_density  gate_depth_ratio  path_depth_ratio  directionality_gap  repair_rate  total_repairs  generation_constraint_rejections  generation_time_sec  descriptor_gap
  CONTROL  4                   1.0               1.0          1.0   0.278273       1.0                0.409933             0.352438       0.277706               0.0          0.267857          0.250468                 0.0          0.0            0.0                               0.0            35.097649        0.179951
TREATMENT  4                   1.0               1.0          1.0   0.278273       1.0                0.409933             0.352438       0.277706               0.0          0.267857          0.250468                 0.0          0.0            0.0                               0.0            36.311393        0.179951

## Paired Significance (Treatment - Control)

                          metric  n_pairs  delta_mean_treatment_minus_control  delta_ci_low  delta_ci_high  p_value  effect_size_d  p_value_bh_fdr  significant_fdr_0_05
            overall_completeness        4                            0.000000       0.00000       0.000000 1.000000       0.000000             1.0                 False
                constraint_valid        4                            0.000000       0.00000       0.000000 1.000000       0.000000             1.0                 False
                     path_exists        4                            0.000000       0.00000       0.000000 1.000000       0.000000             1.0                 False
                       linearity        4                            0.000000       0.00000       0.000000 1.000000       0.000000             1.0                 False
                        leniency        4                            0.000000       0.00000       0.000000 1.000000       0.000000             1.0                 False
          progression_complexity        4                            0.000000       0.00000       0.000000 1.000000       0.000000             1.0                 False
             topology_complexity        4                            0.000000       0.00000       0.000000 1.000000       0.000000             1.0                 False
                   cycle_density        4                            0.000000       0.00000       0.000000 1.000000       0.000000             1.0                 False
                shortcut_density        4                            0.000000       0.00000       0.000000 1.000000       0.000000             1.0                 False
                gate_depth_ratio        4                            0.000000       0.00000       0.000000 1.000000       0.000000             1.0                 False
                path_depth_ratio        4                            0.000000       0.00000       0.000000 1.000000       0.000000             1.0                 False
              directionality_gap        4                            0.000000       0.00000       0.000000 1.000000       0.000000             1.0                 False
                     repair_rate        4                            0.000000       0.00000       0.000000 1.000000       0.000000             1.0                 False
                   total_repairs        4                            0.000000       0.00000       0.000000 1.000000       0.000000             1.0                 False
generation_constraint_rejections        4                            0.000000       0.00000       0.000000 1.000000       0.000000             1.0                 False
             generation_time_sec        4                            1.213744      -0.09803       2.368062 0.247938       0.913538             1.0                 False
                  descriptor_gap        4                            0.000000       0.00000       0.000000 1.000000       0.000000             1.0                 False