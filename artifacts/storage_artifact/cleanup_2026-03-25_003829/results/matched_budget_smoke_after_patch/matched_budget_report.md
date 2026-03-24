# Matched-Budget Block-I Benchmark

## Methods
- `RANDOM`
- `ES`
- `GA`
- `MAP_ELITES`
- `FULL`

## Settings

- `eval_budget`: 64
- `num_samples`: 2
- `room_budget`: [18, 33]
- `rule_space`: full
- `baseline_for_significance`: FULL

## Summary

    method  n  fitness  feasible_search_rate  feasible_operational_rate  overall_completeness  constraint_valid_rate  linearity  leniency  progression_complexity  topology_complexity  path_length  num_nodes  repair_rate  mean_generation_constraint_rejections  mean_candidate_repairs_applied  novelty_vs_reference  graph_edit_distance  generation_time_sec  evaluations_used  fidelity_js_divergence  expressive_overlap_reference  coverage_linearity_leniency  coverage_progression_topology
    RANDOM  2 0.564873                   0.0                        1.0                   1.0                    1.0   0.385452  0.800000                0.480579             0.300809          8.5       24.5          0.0                                    0.0                             0.0              0.053502             0.091169             1.168402              64.0                0.339022                      0.000000                        0.005                         0.0050
        ES  2 0.561660                   0.0                        1.0                   1.0                    1.0   0.290105  1.000000                0.427062             0.349674          6.5       26.0          0.0                                    0.5                             0.0              0.048475             0.052479             1.672744              64.0                0.404993                      0.055556                        0.005                         0.0050
        GA  2 0.588656                   0.0                        1.0                   1.0                    1.0   0.303571  0.785714                0.492289             0.395221          7.5       28.0          0.0                                    0.0                             0.0              0.053716             0.040102             1.527583              64.0                0.406275                      0.000000                        0.005                         0.0050
MAP_ELITES  2 0.523598                   0.0                        1.0                   1.0                    1.0   0.380952  0.833333                0.447672             0.352442          8.0       24.0          0.0                                    0.0                             0.0              0.043799             0.072173             2.070483              48.0                0.344557                      0.055556                        0.005                         0.0050
      FULL  2 0.554330                   0.0                        1.0                   1.0                    1.0   0.281481  0.791667                0.484254             0.362866          7.0       28.5          0.0                                    0.0                             0.0              0.094772             0.039107             1.598907              64.0                0.472111                      0.000000                        0.005                         0.0025

## Paired Significance

    method                           metric  n_pairs  delta_mean_cfg_minus_full  delta_ci_low  delta_ci_high  p_value  effect_size_d  p_value_bh_fdr  significant_fdr_0_05
    RANDOM                          fitness        2                   0.010543     -0.007153       0.028240 1.000000       0.595792             1.0                 False
    RANDOM                  feasible_search        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
    RANDOM             feasible_operational        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
    RANDOM             overall_completeness        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
    RANDOM                 constraint_valid        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
    RANDOM                        linearity        2                   0.103970      0.081159       0.126781 0.510122       4.557969             1.0                 False
    RANDOM                         leniency        2                   0.008333     -0.233333       0.250000 1.000000       0.034483             1.0                 False
    RANDOM           progression_complexity        2                  -0.003675     -0.008952       0.001601 1.000000      -0.696532             1.0                 False
    RANDOM              topology_complexity        2                  -0.062057     -0.084278      -0.039837 0.488128      -2.792811             1.0                 False
    RANDOM                      path_length        2                   1.500000      0.000000       3.000000 1.000000       1.000000             1.0                 False
    RANDOM                        num_nodes        2                  -4.000000     -7.000000      -1.000000 0.498375      -1.333333             1.0                 False
    RANDOM                   repair_applied        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
    RANDOM generation_constraint_rejections        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
    RANDOM        candidate_repairs_applied        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
    RANDOM             novelty_vs_reference        2                  -0.041270     -0.047879      -0.034661 0.499125      -6.244738             1.0                 False
    RANDOM              graph_edit_distance        2                   0.052062      0.023519       0.080605 0.506123       1.823956             1.0                 False
    RANDOM              generation_time_sec        2                  -0.430504     -0.448894      -0.412114 0.504624     -23.410063             1.0                 False
        ES                          fitness        2                   0.007331     -0.027860       0.042521 1.000000       0.208312             1.0                 False
        ES                  feasible_search        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
        ES             feasible_operational        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
        ES             overall_completeness        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
        ES                 constraint_valid        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
        ES                        linearity        2                   0.008623     -0.020434       0.037681 1.000000       0.296770             1.0                 False
        ES                         leniency        2                   0.208333      0.166667       0.250000 0.500625       5.000000             1.0                 False
        ES           progression_complexity        2                  -0.057192     -0.111780      -0.002604 0.503624      -1.047708             1.0                 False
        ES              topology_complexity        2                  -0.013192     -0.016187      -0.010197 0.488128      -4.404835             1.0                 False
        ES                      path_length        2                  -0.500000     -1.000000       0.000000 1.000000      -1.000000             1.0                 False
        ES                        num_nodes        2                  -2.500000     -7.000000       2.000000 1.000000      -0.555556             1.0                 False
        ES                   repair_applied        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
        ES generation_constraint_rejections        2                   0.500000      0.000000       1.000000 1.000000       1.000000             1.0                 False
        ES        candidate_repairs_applied        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
        ES             novelty_vs_reference        2                  -0.046298     -0.065582      -0.027013 0.499125      -2.400804             1.0                 False
        ES              graph_edit_distance        2                   0.013371     -0.013148       0.039891 1.000000       0.504210             1.0                 False
        ES              generation_time_sec        2                   0.073837     -0.102191       0.249865 1.000000       0.419462             1.0                 False
        GA                          fitness        2                   0.034326     -0.006655       0.075307 1.000000       0.837609             1.0                 False
        GA                  feasible_search        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
        GA             feasible_operational        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
        GA             overall_completeness        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
        GA                 constraint_valid        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
        GA                        linearity        2                   0.022090     -0.016667       0.060847 1.000000       0.569966             1.0                 False
        GA                         leniency        2                  -0.005952     -0.178571       0.166667 1.000000      -0.034483             1.0                 False
        GA           progression_complexity        2                   0.008036     -0.000514       0.016586 1.000000       0.939831             1.0                 False
        GA              topology_complexity        2                   0.032354      0.024966       0.039743 0.488128       4.379198             1.0                 False
        GA                      path_length        2                   0.500000     -1.000000       2.000000 1.000000       0.333333             1.0                 False
        GA                        num_nodes        2                  -0.500000     -2.000000       1.000000 1.000000      -0.333333             1.0                 False
        GA                   repair_applied        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
        GA generation_constraint_rejections        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
        GA        candidate_repairs_applied        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
        GA             novelty_vs_reference        2                  -0.041056     -0.049081      -0.033031 0.499125      -5.116047             1.0                 False
        GA              graph_edit_distance        2                   0.000995     -0.003741       0.005730 1.000000       0.210064             1.0                 False
        GA              generation_time_sec        2                  -0.071323     -0.105880      -0.036766 0.504624      -2.063929             1.0                 False
MAP_ELITES                          fitness        2                  -0.030731     -0.070238       0.008775 1.000000      -0.777876             1.0                 False
MAP_ELITES                  feasible_search        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
MAP_ELITES             feasible_operational        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
MAP_ELITES             overall_completeness        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
MAP_ELITES                 constraint_valid        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
MAP_ELITES                        linearity        2                   0.099471      0.037037       0.161905 0.510122       1.593220             1.0                 False
MAP_ELITES                         leniency        2                   0.041667     -0.166667       0.250000 1.000000       0.200000             1.0                 False
MAP_ELITES           progression_complexity        2                  -0.036582     -0.069996      -0.003168 0.503624      -1.094811             1.0                 False
MAP_ELITES              topology_complexity        2                  -0.010424     -0.031498       0.010649 1.000000      -0.494675             1.0                 False
MAP_ELITES                      path_length        2                   1.000000      1.000000       1.000000 0.492877       0.000000             1.0                 False
MAP_ELITES                        num_nodes        2                  -4.500000     -9.000000       0.000000 1.000000      -1.000000             1.0                 False
MAP_ELITES                   repair_applied        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
MAP_ELITES generation_constraint_rejections        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
MAP_ELITES        candidate_repairs_applied        2                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
MAP_ELITES             novelty_vs_reference        2                  -0.050973     -0.055438      -0.046509 0.499125     -11.418332             1.0                 False
MAP_ELITES              graph_edit_distance        2                   0.033065      0.029107       0.037024 0.506123       8.353620             1.0                 False
MAP_ELITES              generation_time_sec        2                   0.471576      0.061546       0.881607 0.504624       1.150100             1.0                 False