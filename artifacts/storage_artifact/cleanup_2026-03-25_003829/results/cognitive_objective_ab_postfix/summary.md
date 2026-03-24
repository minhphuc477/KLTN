# Cognitive Objective A/B

## Config

{
  "num_samples": 12,
  "seed": 42,
  "min_rooms": 8,
  "max_rooms": 16,
  "population_size": 24,
  "generations": 30,
  "rule_space": "full",
  "room_count_bias": 0.45,
  "persona": "balanced",
  "target_confusion_ratio": 1.8,
  "cognitive_weight": 0.08,
  "control_name": "COGNITIVE_OFF",
  "treatment_name": "COGNITIVE_ON"
}

## Means By Arm

          arm  generation_time_sec     nodes     edges  linearity  leniency  progression_complexity  topology_complexity  cycle_density  shortcut_density  gate_depth_ratio  path_depth_ratio  directionality_gap  constraint_valid  path_exists  cbs_fitness  cbs_confusion_ratio  cbs_path_efficiency  cbs_room_entropy
COGNITIVE_OFF            22.449936 25.833333 49.666667   0.288468  0.862500                0.458639             0.365075       0.269204               0.0          0.193750          0.259570                 0.0               0.0          1.0    -3.598069             1.637691             0.363988          0.382572
 COGNITIVE_ON            23.255255 26.083333 50.750000   0.307505  0.797222                0.468036             0.379528       0.286143               0.0          0.219907          0.279589                 0.0               0.0          1.0     0.976807             1.647149             0.614576          0.384700

## Paired Delta (treatment - control)

                metric  n_pairs  control_mean  treatment_mean  delta_mean_treatment_minus_control  delta_std
   generation_time_sec       12     22.449936       23.255255                            0.805318   2.142991
                 nodes       12     25.833333       26.083333                            0.250000   3.139135
                 edges       12     49.666667       50.750000                            1.083333   6.460887
             linearity       12      0.288468        0.307505                            0.019037   0.038374
              leniency       12      0.862500        0.797222                           -0.065278   0.116360
progression_complexity       12      0.458639        0.468036                            0.009397   0.037618
   topology_complexity       12      0.365075        0.379528                            0.014453   0.024940
         cycle_density       12      0.269204        0.286143                            0.016939   0.030485
      shortcut_density       12      0.000000        0.000000                            0.000000   0.000000
      gate_depth_ratio       12      0.193750        0.219907                            0.026157   0.099008
      path_depth_ratio       12      0.259570        0.279589                            0.020019   0.040169
    directionality_gap       12      0.000000        0.000000                            0.000000   0.000000
      constraint_valid       12      0.000000        0.000000                            0.000000   0.000000
           path_exists       12      1.000000        1.000000                            0.000000   0.000000
           cbs_fitness       12     -3.598069        0.976807                            4.574876   5.412594
   cbs_confusion_ratio       12      1.637691        1.647149                            0.009458   0.019294
   cbs_path_efficiency       12      0.363988        0.614576                            0.250588   0.326859
      cbs_room_entropy       12      0.382572        0.384700                            0.002128   0.040956
