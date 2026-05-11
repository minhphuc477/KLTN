# Matched-Budget Block-I Benchmark

## Methods
- `MAP_ELITES`

## Settings

- `eval_budget`: 1024
- `num_samples`: 96
- `room_budget`: [18, 33]
- `rule_space`: full
- `baseline_for_significance`: MAP_ELITES

## Core Summary

    method  n  fitness  feasible_search_rate  feasible_operational_rate  overall_completeness  constraint_valid_rate  linearity  leniency  progression_complexity  topology_complexity  path_length  num_nodes  generation_time_sec  evaluations_used
MAP_ELITES 96 0.302247                   0.0                        1.0                   1.0                    1.0   0.344862  0.506002                0.477799             0.299784       7.8125  25.739583            22.201781             992.0

## Topology Semantics

    method  key_gate_count  key_before_lock_rate  switch_gate_count  switch_before_gate_rate  battery_gate_count  battery_satisfaction_rate  path_redundancy  articulation_count  articulation_ratio  branch_count  branch_utility_rate  secret_component_count  secret_content_discoverability_rate  coverage_redundancy_articulation  coverage_branch_secret  map_elites_coverage  map_elites_qd_score  map_elites_mean_fitness  map_elites_feature_diversity  map_elites_num_elites
MAP_ELITES         2.71875                   1.0                0.5                 0.979167              0.0625                        1.0         0.334578           10.927083            0.423324      2.395833             0.934797                0.302083                                  1.0                             0.155                   0.015                 0.05                  5.0                      1.0                      0.113167                    5.0

## Paired Significance

_No paired significance rows available_