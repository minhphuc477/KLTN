# Matched-Budget Block-I Benchmark

## Methods
- `RANDOM`
- `ES`
- `GA`
- `MAP_ELITES`
- `FULL`

## Settings

- `eval_budget`: 512
- `num_samples`: 64
- `room_budget`: [18, 33]
- `rule_space`: full
- `baseline_for_significance`: FULL

## Core Summary

    method  n  fitness  feasible_search_rate  feasible_operational_rate  overall_completeness  constraint_valid_rate  linearity  leniency  progression_complexity  topology_complexity  path_length  num_nodes  generation_time_sec  evaluations_used
    RANDOM 64 0.192353                   0.0                   0.812500              0.953125                    1.0   0.283992  0.761719                0.479441             0.321043     3.000000  15.609375             7.747655             512.0
        ES 64 0.317187                   0.0                   1.000000              1.000000                    1.0   0.368776  0.652641                0.476788             0.288632     8.390625  25.703125            12.271697             504.0
        GA 64 0.302126                   0.0                   1.000000              1.000000                    1.0   0.356061  0.572135                0.479509             0.297662     8.531250  27.203125            12.540650             504.0
MAP_ELITES 64 0.283089                   0.0                   1.000000              1.000000                    1.0   0.314248  0.599814                0.472601             0.295864     7.328125  27.250000            10.501494             484.0
      FULL 64 0.293156                   0.0                   0.984375              0.996094                    1.0   0.342695  0.634542                0.475939             0.293464     8.500000  28.296875            12.786271             504.0

## Topology Semantics

    method  key_gate_count  key_before_lock_rate  switch_gate_count  switch_before_gate_rate  battery_gate_count  battery_satisfaction_rate  path_redundancy  articulation_count  articulation_ratio  branch_count  branch_utility_rate  secret_component_count  secret_content_discoverability_rate  coverage_redundancy_articulation  coverage_branch_secret
    RANDOM        2.125000                   1.0           0.234375                 1.000000            0.046875                        1.0         0.041667            5.468750            0.344136      1.015625             1.000000                0.406250                             1.000000                            0.0375                  0.0025
        ES        3.078125                   1.0           0.390625                 0.994792            0.062500                        1.0         0.334635           11.718750            0.452878      2.625000             0.924219                0.406250                             1.000000                            0.1225                  0.0175
        GA        2.859375                   1.0           0.562500                 0.984375            0.062500                        1.0         0.343656           13.156250            0.480830      2.562500             0.956510                0.406250                             1.000000                            0.1100                  0.0125
MAP_ELITES        3.062500                   1.0           0.531250                 1.000000            0.031250                        1.0         0.301891           12.343750            0.447880      2.421875             0.952604                0.281250                             1.000000                            0.1175                  0.0100
      FULL        3.468750                   1.0           0.453125                 1.000000            0.015625                        1.0         0.308743           14.328125            0.504696      2.390625             0.963542                0.546875                             0.984375                            0.1100                  0.0075

## Paired Significance

    method                              metric  n_pairs  delta_mean_cfg_minus_full  delta_ci_low  delta_ci_high  p_value  effect_size_d  p_value_bh_fdr  significant_fdr_0_05
    RANDOM                             fitness       64                  -0.100802     -0.127329      -0.072401 0.000250      -0.894661        0.002181                  True
    RANDOM                     feasible_search       64                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
    RANDOM                feasible_operational       64                  -0.171875     -0.265625      -0.093750 0.001750      -0.455573        0.011997                  True
    RANDOM                overall_completeness       64                  -0.042969     -0.066406      -0.019531 0.001500      -0.455573        0.011074                  True
    RANDOM                    constraint_valid       64                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
    RANDOM                           linearity       64                  -0.058703     -0.090926      -0.026789 0.000750      -0.453875        0.005999                  True
    RANDOM                            leniency       64                   0.127176      0.029557       0.225596 0.017246       0.319849        0.087136                 False
    RANDOM              progression_complexity       64                   0.003502     -0.003347       0.011575 0.382154       0.115262        0.965443                 False
    RANDOM                 topology_complexity       64                   0.027579      0.015532       0.039994 0.000250       0.539737        0.002181                  True
    RANDOM                key_before_lock_rate       64                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
    RANDOM             switch_before_gate_rate       64                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
    RANDOM           battery_satisfaction_rate       64                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
    RANDOM                         path_length       64                  -5.500000     -6.172266      -4.812500 0.000250      -2.016737        0.002181                  True
    RANDOM                           num_nodes       64                 -12.687500    -14.703125     -10.577734 0.000250      -1.525023        0.002181                  True
    RANDOM                     path_redundancy       64                  -0.267076     -0.327635      -0.203506 0.000250      -1.050428        0.002181                  True
    RANDOM                  articulation_ratio       64                  -0.160560     -0.201652      -0.121341 0.000250      -0.990492        0.002181                  True
    RANDOM                 branch_utility_rate       64                   0.036458      0.005208       0.078125 0.066733       0.250160        0.256256                 False
    RANDOM secret_content_discoverability_rate       64                   0.015625      0.000000       0.046875 1.000000       0.125988        1.000000                 False
    RANDOM                      repair_applied       64                  -1.000000     -1.000000      -1.000000 0.000250       0.000000        0.002181                  True
    RANDOM    generation_constraint_rejections       64                   5.343750      4.484375       6.187500 0.000250       1.496148        0.002181                  True
    RANDOM           candidate_repairs_applied       64                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
    RANDOM                novelty_vs_reference       64                  -0.016934     -0.028157      -0.006304 0.004499      -0.380565        0.028793                  True
    RANDOM                 graph_edit_distance       64                   0.014460      0.003218       0.026702 0.016246       0.305255        0.086645                 False
    RANDOM                 generation_time_sec       64                  -5.038617     -5.252343      -4.822510 0.000250      -5.553045        0.002181                  True
        ES                             fitness       64                   0.024032     -0.010653       0.057407 0.158210       0.175648        0.542436                 False
        ES                     feasible_search       64                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
        ES                feasible_operational       64                   0.015625      0.000000       0.046875 1.000000       0.125988        1.000000                 False
        ES                overall_completeness       64                   0.003906      0.000000       0.011719 1.000000       0.125988        1.000000                 False
        ES                    constraint_valid       64                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
        ES                           linearity       64                   0.026080     -0.007172       0.060213 0.131967       0.196010        0.487263                 False
        ES                            leniency       64                   0.018099     -0.072379       0.113531 0.722569       0.046680        1.000000                 False
        ES              progression_complexity       64                   0.000850     -0.009741       0.010404 0.873282       0.020660        1.000000                 False
        ES                 topology_complexity       64                  -0.004832     -0.018959       0.009064 0.509373      -0.083100        1.000000                 False
        ES                key_before_lock_rate       64                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
        ES             switch_before_gate_rate       64                  -0.005208     -0.015625       0.000000 1.000000      -0.125988        1.000000                 False
        ES           battery_satisfaction_rate       64                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
        ES                         path_length       64                  -0.109375     -0.984766       0.765625 0.835791      -0.030515        1.000000                 False
        ES                           num_nodes       64                  -2.593750     -4.515625      -0.625000 0.009748      -0.327292        0.055045                 False
        ES                     path_redundancy       64                   0.025892     -0.047546       0.100886 0.499375       0.085800        1.000000                 False
        ES                  articulation_ratio       64                  -0.051818     -0.103660      -0.004967 0.038990      -0.263453        0.155961                 False
        ES                 branch_utility_rate       64                  -0.039323     -0.094805       0.021094 0.181705      -0.173841        0.581455                 False
        ES secret_content_discoverability_rate       64                   0.015625      0.000000       0.046875 1.000000       0.125988        1.000000                 False
        ES                      repair_applied       64                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
        ES    generation_constraint_rejections       64                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
        ES           candidate_repairs_applied       64                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
        ES                novelty_vs_reference       64                  -0.015318     -0.025884      -0.004011 0.007498      -0.354790        0.044989                  True
        ES                 graph_edit_distance       64                  -0.000472     -0.007287       0.006239 0.898775      -0.016297        1.000000                 False
        ES                 generation_time_sec       64                  -0.514574     -0.692344      -0.314490 0.000250      -0.667289        0.002181                  True
        GA                             fitness       64                   0.008970     -0.028348       0.046593 0.653837       0.058132        1.000000                 False
        GA                     feasible_search       64                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
        GA                feasible_operational       64                   0.015625      0.000000       0.046875 1.000000       0.125988        1.000000                 False
        GA                overall_completeness       64                   0.003906      0.000000       0.011719 1.000000       0.125988        1.000000                 False
        GA                    constraint_valid       64                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
        GA                           linearity       64                   0.013365     -0.019315       0.046661 0.450637       0.098541        1.000000                 False
        GA                            leniency       64                  -0.062407     -0.147588       0.024283 0.173207      -0.172477        0.573374                 False
        GA              progression_complexity       64                   0.003571     -0.003119       0.010741 0.348413       0.126234        0.903990                 False
        GA                 topology_complexity       64                   0.004197     -0.007033       0.015077 0.446638       0.096886        1.000000                 False
        GA                key_before_lock_rate       64                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
        GA             switch_before_gate_rate       64                  -0.015625     -0.039062       0.000000 0.494126      -0.179605        1.000000                 False
        GA           battery_satisfaction_rate       64                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
        GA                         path_length       64                   0.031250     -1.031641       1.000000 0.974506       0.007457        1.000000                 False
        GA                           num_nodes       64                  -1.093750     -2.781250       0.672266 0.231942      -0.153451        0.682738                 False
        GA                     path_redundancy       64                   0.034913     -0.022690       0.093076 0.250687       0.146131        0.707823                 False
        GA                  articulation_ratio       64                  -0.023866     -0.064457       0.018163 0.277931      -0.142013        0.762324                 False
        GA                 branch_utility_rate       64                  -0.007031     -0.049479       0.042995 0.787053      -0.037953        1.000000                 False
        GA secret_content_discoverability_rate       64                   0.015625      0.000000       0.046875 1.000000       0.125988        1.000000                 False
        GA                      repair_applied       64                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
        GA    generation_constraint_rejections       64                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
        GA           candidate_repairs_applied       64                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
        GA                novelty_vs_reference       64                  -0.011443     -0.021478      -0.001741 0.025994      -0.283177        0.113426                 False
        GA                 graph_edit_distance       64                  -0.004452     -0.011743       0.002599 0.234691      -0.152432        0.682738                 False
        GA                 generation_time_sec       64                  -0.245622     -0.442598      -0.038076 0.019495      -0.298736        0.093577                 False
MAP_ELITES                             fitness       64                  -0.010066     -0.042661       0.025284 0.569358      -0.072419        1.000000                 False
MAP_ELITES                     feasible_search       64                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
MAP_ELITES                feasible_operational       64                   0.015625      0.000000       0.046875 1.000000       0.125988        1.000000                 False
MAP_ELITES                overall_completeness       64                   0.003906      0.000000       0.011719 1.000000       0.125988        1.000000                 False
MAP_ELITES                    constraint_valid       64                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
MAP_ELITES                           linearity       64                  -0.028448     -0.066416       0.009772 0.157461      -0.184052        0.542436                 False
MAP_ELITES                            leniency       64                  -0.034728     -0.122787       0.054016 0.449638      -0.095574        1.000000                 False
MAP_ELITES              progression_complexity       64                  -0.003337     -0.016591       0.007662 0.603349      -0.068478        1.000000                 False
MAP_ELITES                 topology_complexity       64                   0.002400     -0.011157       0.016033 0.722819       0.043410        1.000000                 False
MAP_ELITES                key_before_lock_rate       64                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
MAP_ELITES             switch_before_gate_rate       64                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
MAP_ELITES           battery_satisfaction_rate       64                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
MAP_ELITES                         path_length       64                  -1.171875     -2.046875      -0.171875 0.022994      -0.298091        0.105117                 False
MAP_ELITES                           num_nodes       64                  -1.046875     -2.984766       0.937500 0.293427      -0.135030        0.782471                 False
MAP_ELITES                     path_redundancy       64                  -0.006852     -0.080463       0.072092 0.861285      -0.022187        1.000000                 False
MAP_ELITES                  articulation_ratio       64                  -0.056816     -0.108063      -0.006707 0.029993      -0.283033        0.125186                 False
MAP_ELITES                 branch_utility_rate       64                  -0.010937     -0.058333       0.044271 0.676081      -0.054102        1.000000                 False
MAP_ELITES secret_content_discoverability_rate       64                   0.015625      0.000000       0.046875 1.000000       0.125988        1.000000                 False
MAP_ELITES                      repair_applied       64                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
MAP_ELITES    generation_constraint_rejections       64                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
MAP_ELITES           candidate_repairs_applied       64                   0.000000      0.000000       0.000000 1.000000       0.000000        1.000000                 False
MAP_ELITES                novelty_vs_reference       64                   0.007372     -0.003569       0.019013 0.206198       0.160487        0.638550                 False
MAP_ELITES                 graph_edit_distance       64                  -0.000643     -0.006714       0.005601 0.848788      -0.025113        1.000000                 False
MAP_ELITES                 generation_time_sec       64                  -2.284778     -2.494769      -2.079754 0.000250      -2.569799        0.002181                  True