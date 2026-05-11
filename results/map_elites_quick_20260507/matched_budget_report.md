# Matched-Budget Block-I Benchmark

## Methods
- `MAP_ELITES`

## Settings

- `eval_budget`: 128
- `num_samples`: 3
- `room_budget`: [18, 33]
- `rule_space`: full
- `baseline_for_significance`: MAP_ELITES

## Core Summary

    method  n  fitness  feasible_search_rate  feasible_operational_rate  overall_completeness  constraint_valid_rate  linearity  leniency  progression_complexity  topology_complexity  path_length  num_nodes  generation_time_sec  evaluations_used
MAP_ELITES  3 0.159278                   0.0                        1.0                   1.0                    1.0   0.194176  0.777778                0.367949             0.269617          5.0  29.666667             3.813438             112.0

## Topology Semantics

    method  key_gate_count  key_before_lock_rate  switch_gate_count  switch_before_gate_rate  battery_gate_count  battery_satisfaction_rate  path_redundancy  articulation_count  articulation_ratio  branch_count  branch_utility_rate  secret_component_count  secret_content_discoverability_rate  coverage_redundancy_articulation  coverage_branch_secret
MAP_ELITES        2.666667                   1.0           1.666667                 0.666667            0.333333                        1.0         0.222222                17.0            0.587646      1.333333                  1.0                0.333333                                  1.0                            0.0075                  0.0025

## Paired Significance

_No paired significance rows available_