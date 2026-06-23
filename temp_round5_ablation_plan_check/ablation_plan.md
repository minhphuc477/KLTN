# Ablation Study Plan

Protocol: `fixed_seed_paired_ablation`

Estimate component necessity by changing one interpretable subsystem at a time where possible, using shared seeds and paired significance tests against FULL.

## Runtime Budget

- `num_rooms`: 8
- `target_curve`: [0.2, 0.4, 0.6, 0.8, 0.7, 0.5, 0.3, 0.2]
- `diffusion_steps`: 25
- `cbs_timeout`: 120000
- `evolution_population`: 24
- `evolution_generations`: 30

## Paired Statistics

- `baseline`: FULL
- `confidence_interval`: paired bootstrap over seed deltas
- `p_value`: random-sign permutation over paired seed deltas
- `multiple_comparison_control`: Benjamini-Hochberg FDR over exported p-values

## Experiments

### DIFFUSION_TOPO_ADDITIVE
- Tier: `block_iii`
- Component: diffusion topology conditioning
- Comparison: DIFFUSION_TOPO_ADDITIVE and DIFFUSION_TOPO_SPADE
- Isolates: conditioning injection style while keeping topology, sampler, and repair stack matched
- Interpretation: Tests whether SPADE-style affine topology modulation carries more useful structural signal than additive maps.

### DIFFUSION_TOPO_SPADE
- Tier: `block_iii`
- Component: diffusion topology conditioning
- Comparison: DIFFUSION_TOPO_ADDITIVE and DIFFUSION_TOPO_SPADE
- Isolates: conditioning injection style while keeping topology, sampler, and repair stack matched
- Interpretation: Tests whether SPADE-style affine topology modulation carries more useful structural signal than additive maps.

## Claim Boundaries

- RANDOM_TOPOLOGY is the strict topology null; NO_EVOLUTION is direct grammar generation.
- PURE_WFC bypasses neural room priors and is a heuristic-only baseline, not a repair ablation.
- Single-seed or quick-profile results are screening evidence; thesis claims should use paired multi-seed runs.