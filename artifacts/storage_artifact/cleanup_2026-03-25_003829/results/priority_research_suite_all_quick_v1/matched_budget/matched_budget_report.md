# Matched-Budget Block-I Benchmark

## Methods
- `RANDOM`
- `ES`
- `GA`
- `MAP_ELITES`
- `FULL`

## Settings

- `eval_budget`: 128
- `num_samples`: 4
- `room_budget`: [18, 33]
- `rule_space`: full
- `baseline_for_significance`: FULL

## Summary

    method  n  fitness  feasible_rate  overall_completeness  constraint_valid_rate  linearity  leniency  progression_complexity  topology_complexity  path_length  num_nodes  repair_rate  mean_generation_constraint_rejections  mean_candidate_repairs_applied  novelty_vs_reference  graph_edit_distance  generation_time_sec  evaluations_used  fidelity_js_divergence  expressive_overlap_reference  coverage_linearity_leniency  coverage_progression_topology
    RANDOM  4 0.553087            0.0                   1.0                    1.0   0.346059  0.900000                0.434153             0.322386         7.25      23.75          0.0                                    0.0                             0.0              0.054361             0.080699             2.221478             128.0                0.363691                      0.050000                         0.01                         0.0100
        ES  4 0.603660            0.0                   1.0                    1.0   0.335630  0.812500                0.439860             0.339844         7.75      26.00          0.0                                    0.0                             0.0              0.055887             0.054063             2.703914             120.0                0.359467                      0.050000                         0.01                         0.0075
        GA  4 0.594095            0.0                   1.0                    1.0   0.324199  0.900000                0.439783             0.339933         7.50      26.50          0.0                                    0.0                             0.0              0.053670             0.050850             2.452616             120.0                0.363946                      0.105263                         0.01                         0.0100
MAP_ELITES  4 0.549531            0.0                   1.0                    1.0   0.313335  0.837500                0.476634             0.367920         6.75      25.00          0.0                                    0.0                             0.0              0.055351             0.046518             3.004493             112.0                0.329346                      0.050000                         0.01                         0.0075
      FULL  4 0.587245            0.0                   1.0                    1.0   0.280739  0.816667                0.473504             0.356650         6.25      26.00          0.0                                    0.0                             0.0              0.051086             0.053223             2.707488             120.0                0.328380                      0.050000                         0.01                         0.0050

## Paired Significance

    method                           metric  n_pairs  delta_mean_cfg_minus_full  delta_ci_low  delta_ci_high  p_value  effect_size_d  p_value_bh_fdr  significant_fdr_0_05
    RANDOM                          fitness        4                  -0.034158     -0.050814      -0.017501 0.130467      -1.981334             1.0                 False
    RANDOM                         feasible        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
    RANDOM             overall_completeness        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
    RANDOM                 constraint_valid        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
    RANDOM                        linearity        4                   0.065320     -0.003388       0.126831 0.251937       0.928331             1.0                 False
    RANDOM                         leniency        4                   0.083333     -0.050000       0.300000 1.000000       0.450835             1.0                 False
    RANDOM           progression_complexity        4                  -0.039351     -0.081507       0.002804 0.385154      -0.888768             1.0                 False
    RANDOM              topology_complexity        4                  -0.034264     -0.082714       0.014187 0.508373      -0.695827             1.0                 False
    RANDOM                      path_length        4                   1.000000     -0.500000       2.500000 0.493627       0.707107             1.0                 False
    RANDOM                        num_nodes        4                  -2.250000     -4.500000       0.000000 0.501375      -0.904534             1.0                 False
    RANDOM                   repair_applied        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
    RANDOM generation_constraint_rejections        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
    RANDOM        candidate_repairs_applied        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
    RANDOM             novelty_vs_reference        4                   0.003276     -0.026184       0.034707 0.876531       0.099171             1.0                 False
    RANDOM              graph_edit_distance        4                   0.027476      0.002664       0.053025 0.256186       0.999695             1.0                 False
    RANDOM              generation_time_sec        4                  -0.486010     -0.675545      -0.296475 0.129718      -2.453168             1.0                 False
        ES                          fitness        4                   0.016415     -0.012310       0.046480 0.501375       0.513301             1.0                 False
        ES                         feasible        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
        ES             overall_completeness        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
        ES                 constraint_valid        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
        ES                        linearity        4                   0.054891     -0.012215       0.104111 0.251937       0.866221             1.0                 False
        ES                         leniency        4                  -0.004167     -0.354167       0.300000 1.000000      -0.012907             1.0                 False
        ES           progression_complexity        4                  -0.033644     -0.066116      -0.001173 0.249438      -1.010594             1.0                 False
        ES              topology_complexity        4                  -0.016806     -0.052010       0.022771 0.502874      -0.470269             1.0                 False
        ES                      path_length        4                   1.500000     -0.250000       2.750000 0.245439       1.000000             1.0                 False
        ES                        num_nodes        4                   0.000000     -2.000000       1.500000 1.000000       0.000000             1.0                 False
        ES                   repair_applied        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
        ES generation_constraint_rejections        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
        ES        candidate_repairs_applied        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
        ES             novelty_vs_reference        4                   0.004801     -0.018285       0.027887 0.876531       0.192508             1.0                 False
        ES              graph_edit_distance        4                   0.000841     -0.007825       0.007246 0.877531       0.103891             1.0                 False
        ES              generation_time_sec        4                  -0.003574     -0.090788       0.083640 1.000000      -0.039148             1.0                 False
        GA                          fitness        4                   0.006850     -0.027583       0.045317 0.627843       0.196036             1.0                 False
        GA                         feasible        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
        GA             overall_completeness        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
        GA                 constraint_valid        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
        GA                        linearity        4                   0.043460     -0.028700       0.127806 0.499875       0.516682             1.0                 False
        GA                         leniency        4                   0.083333      0.000000       0.250000 1.000000       0.577350             1.0                 False
        GA           progression_complexity        4                  -0.033722     -0.068533       0.001090 0.385154      -0.949960             1.0                 False
        GA              topology_complexity        4                  -0.016717     -0.052569       0.019134 0.508373      -0.460952             1.0                 False
        GA                      path_length        4                   1.250000     -1.000000       3.500000 0.622344       0.502519             1.0                 False
        GA                        num_nodes        4                   0.500000     -2.000000       2.500000 0.753562       0.229416             1.0                 False
        GA                   repair_applied        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
        GA generation_constraint_rejections        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
        GA        candidate_repairs_applied        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
        GA             novelty_vs_reference        4                   0.002584     -0.049293       0.036993 1.000000       0.055237             1.0                 False
        GA              graph_edit_distance        4                  -0.002372     -0.019018       0.014273 0.877531      -0.138958             1.0                 False
        GA              generation_time_sec        4                  -0.254872     -0.408105      -0.058998 0.247438      -1.468581             1.0                 False
MAP_ELITES                          fitness        4                  -0.037713     -0.109171       0.020735 0.507123      -0.581195             1.0                 False
MAP_ELITES                         feasible        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
MAP_ELITES             overall_completeness        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
MAP_ELITES                 constraint_valid        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
MAP_ELITES                        linearity        4                   0.032596      0.015009       0.056890 0.123219       1.439330             1.0                 False
MAP_ELITES                         leniency        4                   0.020833     -0.325000       0.366667 1.000000       0.059410             1.0                 False
MAP_ELITES           progression_complexity        4                   0.003130     -0.033969       0.023842 1.000000       0.096129             1.0                 False
MAP_ELITES              topology_complexity        4                   0.011269     -0.040094       0.064898 0.739315       0.196079             1.0                 False
MAP_ELITES                      path_length        4                   0.500000      0.000000       1.000000 0.499125       1.000000             1.0                 False
MAP_ELITES                        num_nodes        4                  -1.000000     -2.500000       1.000000 0.500875      -0.534522             1.0                 False
MAP_ELITES                   repair_applied        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
MAP_ELITES generation_constraint_rejections        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
MAP_ELITES        candidate_repairs_applied        4                   0.000000      0.000000       0.000000 1.000000       0.000000             1.0                 False
MAP_ELITES             novelty_vs_reference        4                   0.004265     -0.037097       0.045627 0.890777       0.100859             1.0                 False
MAP_ELITES              graph_edit_distance        4                  -0.006705     -0.040303       0.017945 0.871032      -0.215475             1.0                 False
MAP_ELITES              generation_time_sec        4                   0.297006      0.082955       0.543741 0.129718       1.245773             1.0                 False