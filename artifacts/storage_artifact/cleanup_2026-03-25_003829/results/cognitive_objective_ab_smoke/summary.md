# Cognitive Objective A/B

## Config

{
  "num_samples": 1,
  "seed": 42,
  "min_rooms": 6,
  "max_rooms": 8,
  "population_size": 8,
  "generations": 4,
  "rule_space": "full",
  "room_count_bias": 0.45,
  "persona": "balanced",
  "target_confusion_ratio": 1.8,
  "cognitive_weight": 0.08,
  "control_name": "COGNITIVE_OFF",
  "treatment_name": "COGNITIVE_ON"
}

## Means By Arm

          arm  generation_time_sec  nodes  edges  linearity  leniency  progression_complexity  topology_complexity  cycle_density  shortcut_density  gate_depth_ratio  path_depth_ratio  directionality_gap  constraint_valid  path_exists  cbs_fitness  cbs_confusion_ratio  cbs_path_efficiency  cbs_room_entropy
COGNITIVE_OFF             1.140683   31.0   62.0   0.258065       1.0                0.478072             0.381387       0.200000               0.0          0.285714          0.233333                 0.0               0.0          1.0   -10.000000             1.656452                0.000          0.336011
 COGNITIVE_ON             1.113496   30.0   62.0   0.233333       1.0                0.348185             0.343613       0.266667               0.0          0.166667          0.206897                 0.0               0.0          1.0     0.983381             1.670000                0.625          0.304322

## Paired Delta (treatment - control)

                metric  n_pairs  control_mean  treatment_mean  delta_mean_treatment_minus_control  delta_std
   generation_time_sec        1      1.140683        1.113496                           -0.027187        0.0
                 nodes        1     31.000000       30.000000                           -1.000000        0.0
                 edges        1     62.000000       62.000000                            0.000000        0.0
             linearity        1      0.258065        0.233333                           -0.024731        0.0
              leniency        1      1.000000        1.000000                            0.000000        0.0
progression_complexity        1      0.478072        0.348185                           -0.129887        0.0
   topology_complexity        1      0.381387        0.343613                           -0.037774        0.0
         cycle_density        1      0.200000        0.266667                            0.066667        0.0
      shortcut_density        1      0.000000        0.000000                            0.000000        0.0
      gate_depth_ratio        1      0.285714        0.166667                           -0.119048        0.0
      path_depth_ratio        1      0.233333        0.206897                           -0.026437        0.0
    directionality_gap        1      0.000000        0.000000                            0.000000        0.0
      constraint_valid        1      0.000000        0.000000                            0.000000        0.0
           path_exists        1      1.000000        1.000000                            0.000000        0.0
           cbs_fitness        1    -10.000000        0.983381                           10.983381        0.0
   cbs_confusion_ratio        1      1.656452        1.670000                            0.013548        0.0
   cbs_path_efficiency        1      0.000000        0.625000                            0.625000        0.0
      cbs_room_entropy        1      0.336011        0.304322                           -0.031689        0.0
