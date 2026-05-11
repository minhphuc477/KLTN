# Matched-Budget Block-I Benchmark

## Methods
- `RANDOM`

## Settings

- `eval_budget`: 512
- `num_samples`: 128
- `room_budget`: [18, 33]
- `rule_space`: full
- `baseline_for_significance`: RANDOM

## Core Summary

method   n  fitness  feasible_search_rate  feasible_operational_rate  overall_completeness  constraint_valid_rate  linearity  leniency  progression_complexity  topology_complexity  path_length  num_nodes  generation_time_sec  evaluations_used
RANDOM 128 0.192895                   0.0                     0.8125              0.953125                    1.0   0.280407  0.777083                0.479709             0.324477          3.0  15.617188             8.676373             512.0

## Topology Semantics

method  key_gate_count  key_before_lock_rate  switch_gate_count  switch_before_gate_rate  battery_gate_count  battery_satisfaction_rate  path_redundancy  articulation_count  articulation_ratio  branch_count  branch_utility_rate  secret_component_count  secret_content_discoverability_rate  coverage_redundancy_articulation  coverage_branch_secret  map_elites_coverage  map_elites_qd_score  map_elites_mean_fitness  map_elites_feature_diversity  map_elites_num_elites
RANDOM        2.164062                   1.0             0.1875                 0.992188             0.03125                        1.0         0.036458             5.28125            0.336277      1.023438                  1.0                0.351562                                  1.0                             0.045                  0.0025                 0.05                  5.0                      1.0                      0.114637                    5.0

## Paired Significance

_No paired significance rows available_