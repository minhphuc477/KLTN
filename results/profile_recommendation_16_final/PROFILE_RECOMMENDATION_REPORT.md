# Profile Recommendation Report (16 samples)

## Run configuration
- Script: `scripts/recommend_realism_profile.py`
- Seed base: `42`
- Profiles compared: `gate_quality_heavy`, `engine_default`
- Budget: `num_generated=16`, `population_size=24`, `generations=30`, `bootstrap_samples=200`
- Weights:
  - `fidelity=0.40`
  - `overlap=0.20`
  - `diversity=0.20`
  - `node_gap=0.10`
  - `edge_gap=0.10`
  - `completeness=6.0`
  - `validity=8.0`

Source: `results/profile_recommendation_16_final/summary.json`

## Result
- **Winner:** `gate_quality_heavy`
- Objective score:
  - `gate_quality_heavy = 0.5302131916618117`
  - `engine_default = 0.5546155245534687`

## Why `gate_quality_heavy` won
- Better fidelity to references:
  - `fidelity_js_divergence = 0.24706531866736692` vs `0.36513954555366934`
- Better diversity:
  - `descriptor_diversity = 0.07664930371737827` vs `0.04091830353142155`
- Both profiles stayed fully feasible:
  - `overall_completeness = 1.0`
  - `constraint_valid_rate = 1.0`

## Trade-offs observed
- `engine_default` had better overlap and scale fit:
  - `expressive_overlap_reference = 0.09523809523809523` vs `0.041666666666666664`
  - Lower node/edge gap ratios than `gate_quality_heavy`
- `gate_quality_heavy` still ranked higher because fidelity + diversity weights dominated while feasibility remained perfect.

## Recommended action
- Keep benchmark default profile as `gate_quality_heavy` for research-facing runs.
- Keep `engine_default` as an explicit comparison baseline in reports.
