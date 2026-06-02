# LogicNet Grid Pathfinder Ablation Protocol

This protocol compares `logic_grid_pathfinder: cnn` and
`logic_grid_pathfinder: bellman_ford`.

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
```

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
