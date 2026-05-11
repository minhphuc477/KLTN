# Matched-Budget Block-I Benchmark

## Methods
- `MAP_ELITES`

## Settings

- `eval_budget`: 512
- `num_samples`: 1
- `room_budget`: [18, 33]
- `rule_space`: full
- `baseline_for_significance`: MAP_ELITES

## Core Summary

    method  n  fitness  feasible_search_rate  feasible_operational_rate  overall_completeness  constraint_valid_rate  linearity  leniency  progression_complexity  topology_complexity  path_length  num_nodes  generation_time_sec  evaluations_used
MAP_ELITES  1 0.195236                   0.0                        1.0                   1.0                    1.0   0.307692  0.666667                0.481026             0.320435          7.0       26.0            10.492692             484.0

## Topology Semantics

    method  key_gate_count  key_before_lock_rate  switch_gate_count  switch_before_gate_rate  battery_gate_count  battery_satisfaction_rate  path_redundancy  articulation_count  articulation_ratio  branch_count  branch_utility_rate  secret_component_count  secret_content_discoverability_rate  coverage_redundancy_articulation  coverage_branch_secret  map_elites_coverage  map_elites_qd_score  map_elites_mean_fitness  map_elites_feature_diversity  map_elites_num_elites
MAP_ELITES             2.0                   1.0                0.0                      1.0                 0.0                        1.0         0.285714                14.0            0.538462           3.0             0.333333                     1.0                                  1.0                            0.0025                  0.0025                 0.01                  1.0                      1.0                           0.0                    1.0

## Paired Significance

_No paired significance rows available_