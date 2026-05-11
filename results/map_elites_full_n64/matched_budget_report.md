# Matched-Budget Block-I Benchmark

## Methods
- `MAP_ELITES`

## Settings

- `eval_budget`: 512
- `num_samples`: 64
- `room_budget`: [18, 33]
- `rule_space`: full
- `baseline_for_significance`: MAP_ELITES

## Core Summary

    method  n  fitness  feasible_search_rate  feasible_operational_rate  overall_completeness  constraint_valid_rate  linearity  leniency  progression_complexity  topology_complexity  path_length  num_nodes  generation_time_sec  evaluations_used
MAP_ELITES 64  0.25068                   0.0                        1.0                   1.0                    1.0   0.315398  0.583408                0.466736             0.297864     7.796875   28.21875             11.70873             484.0

## Topology Semantics

    method  key_gate_count  key_before_lock_rate  switch_gate_count  switch_before_gate_rate  battery_gate_count  battery_satisfaction_rate  path_redundancy  articulation_count  articulation_ratio  branch_count  branch_utility_rate  secret_component_count  secret_content_discoverability_rate  coverage_redundancy_articulation  coverage_branch_secret
MAP_ELITES          3.0625                   1.0           0.578125                  0.96875            0.140625                        1.0         0.290178           13.484375            0.469101      2.453125             0.958333                0.328125                                  1.0                              0.12                    0.01

## Paired Significance

_No paired significance rows available_