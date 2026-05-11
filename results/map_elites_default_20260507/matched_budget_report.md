# Matched-Budget Block-I Benchmark

## Methods
- `MAP_ELITES`

## Settings

- `eval_budget`: 512
- `num_samples`: 10
- `room_budget`: [18, 33]
- `rule_space`: full
- `baseline_for_significance`: MAP_ELITES

## Core Summary

    method  n  fitness  feasible_search_rate  feasible_operational_rate  overall_completeness  constraint_valid_rate  linearity  leniency  progression_complexity  topology_complexity  path_length  num_nodes  generation_time_sec  evaluations_used
MAP_ELITES 10 0.241586                   0.0                        1.0                   1.0                    1.0   0.262354  0.593333                0.446573             0.296478          6.1       27.2            11.676703             484.0

## Topology Semantics

    method  key_gate_count  key_before_lock_rate  switch_gate_count  switch_before_gate_rate  battery_gate_count  battery_satisfaction_rate  path_redundancy  articulation_count  articulation_ratio  branch_count  branch_utility_rate  secret_component_count  secret_content_discoverability_rate  coverage_redundancy_articulation  coverage_branch_secret
MAP_ELITES             2.5                   1.0                0.8                      0.8                 0.0                        1.0          0.22381                13.6            0.504539           2.5                0.975                     0.1                                  1.0                             0.025                   0.005

## Paired Significance

_No paired significance rows available_