# Matched-Budget Topology Benchmark Protocol

This protocol compares Block-I topology generators under the same evaluation budget per seed.

## Methods
- `RANDOM`: random genome search
- `ES`: mutation-only evolutionary search (`crossover_rate=0`)
- `GA`: crossover + mutation evolutionary search
- `MAP_ELITES`: CVT-emitter search using the same runtime `EvolutionaryTopologyGenerator` backend (`search_strategy=cvt_emitter`)
- `FULL`: project default tuned evolutionary profile

## Metrics Reported
- feasibility / completeness: `feasible_rate`, `overall_completeness`, `constraint_valid_rate`
- structure: `linearity`, `leniency`, `progression_complexity`, `topology_complexity`, `path_length`, `num_nodes`
- generation-time robustness: `generation_constraint_rejections`, `candidate_repairs_applied`
- quality/diversity proxies: `novelty_vs_reference`, `graph_edit_distance`, `fidelity_js_divergence`, expressive coverages
- runtime: `generation_time_sec`, `evaluations_used`
- statistical testing: paired bootstrap CIs, paired sign-permutation p-values, BH-FDR correction

## Run Command
```bash
python scripts/run_matched_budget_topology_benchmark.py \
  --data-root "Data/The Legend of Zelda" \
  --methods RANDOM,ES,GA,MAP_ELITES,FULL \
  --num-samples 10 \
  --seed 42 \
  --eval-budget 512 \
  --output results/matched_budget
```

## Kaggle T4 x2 Preset
```bash
python scripts/run_matched_budget_topology_benchmark.py \
  --kaggle-t4x2 \
  --data-root "Data/The Legend of Zelda" \
  --output results/matched_budget_kaggle
```

Optional ablation run on Kaggle T4 x2:
```bash
python scripts/run_ablation_study.py \
  --kaggle-t4x2 \
  --output results/ablation_kaggle
```

## Output Files
- `matched_budget_raw.csv`
- `matched_budget_summary.csv`
- `matched_budget_significance.csv`
- `matched_budget_report.json`
- `matched_budget_report.md`

## External Benchmark Alignment (Next Step)
- PCG Benchmark framework repo (used in FDG 2025 paper): `https://github.com/amidos2006/pcg_benchmark`
- Benchmark paper: `https://arxiv.org/abs/2503.21474`

The current script gives matched-budget internal head-to-head evidence.  
Next, map this project's graph representation to `pcg_benchmark` Zelda content/control spaces for direct cross-publication comparability.

## Multi-GPU Training Note
- For two GPUs on one machine, PyTorch recommends `DistributedDataParallel` with one process per GPU:
  - `https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html`
  - `https://docs.pytorch.org/tutorials/beginner/dist_overview.html`
