# LogicNet Grid Pathfinder Ablation Protocol

This protocol compares four explicitly named LogicNet planner treatments:

- `cnn`: learned local planning baseline.
- `vin`: learnable recurrent planner baseline; this implementation is not
  claimed to reproduce every component of the original VIN architecture.
- `bellman_ford`: conservative differentiable relaxation, evaluated with both
  complete and deliberately truncated coverage.
- `perturb_and_map`: stochastic differentiable black-box planner ablation.

Use these partial configs:

- `configs/ablation_no_vin.yaml`
- `configs/ablation_vin.yaml`
- `configs/ablation_logic_bellman_full.yaml`
- `configs/ablation_logic_bellman_truncated.yaml`
- `configs/ablation_logic_perturb_and_map.yaml`

## Interpretation

This is an inductive-bias ablation, not an equal-parameter architecture
ablation.

- `cnn`: learned convolutional room pathfinder with trainable parameters.
- `bellman_ford`: fixed symbolic soft Bellman-Ford relaxation over the room
  grid; it has no trainable pathfinder weights and derives walkability from the
  semantic tile table.

The comparison should be reported as "learned local pathfinding proxy vs fixed
symbolic relaxation" rather than as two capacity-matched neural architectures.

## Controlled Variables

Run both conditions with the same:

- training/validation dungeon split
- VQ-VAE checkpoint
- diffusion architecture
- random seeds
- batch size and optimizer settings
- LogicNet loss weights
- validation sample budget
- guidance schedule

Only change:

```yaml
diffusion:
  logic_grid_pathfinder: cnn
```

versus:

```yaml
diffusion:
  logic_grid_pathfinder: bellman_ford
  logic_full_coverage: true
```

The truncated-depth treatment changes only:

```yaml
diffusion:
  logic_grid_pathfinder: bellman_ford
  logic_full_coverage: false
```

Do not interpret truncated Bellman failure as evidence against Bellman
planning; it is specifically a planning-horizon ablation.

## Primary Metrics

- Validation diffusion loss.
- Validation LogicNet loss.
- Solver pass rate on generated rooms.
- Tile-pattern JS divergence against the VGLC reference set.
- Guidance gradient quality:
  - mean guidance gradient norm
  - fraction of finite gradients
  - change in solvability proxy after a fixed number of guided denoising steps

## Reporting

Use paired seeds. Report mean, standard deviation, and paired differences across
seeds. Include the pathfinder parameter count in the table to avoid implying a
capacity-matched comparison.
