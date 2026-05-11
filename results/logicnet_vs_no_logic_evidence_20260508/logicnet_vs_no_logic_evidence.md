> SUPERSEDED: use `results/logicnet_vs_no_logic_explicit_ckpt_quick_20260508/corrected_logicnet_vs_no_logic_report.md` instead.
> Reason: this provisional report lacked checkpoint provenance and its positive LogicNet claim did not reproduce under explicit checkpoint pinning.

# LogicNet vs No Logic Evidence

Inputs:
- LogicNet: `results\ablation_full_long_matched\ablation_raw.csv` (`FULL`, `logic_guidance_scale=1.0`)
- No LogicNet: `results\ablation_no_logic_long\ablation_raw.csv` (`NO_LOGIC`, `logic_guidance_scale=0.0`)
- Paired seeds: 42, 43, 44, 45

## Key Result

On these 4 matched seeds, LogicNet did not improve solved-rate after repair: both settings solved 2/4 generated dungeons and both completed generation successfully. The useful evidence for LogicNet is instead in neural fidelity and repair burden: reconstruction error dropped from 0.2102 to 0.1591, and repaired tiles dropped from 127.25 to 95.50 on average.

## Metric Summary

| metric | direction | n_pairs | logic_mean | no_logic_mean | delta_logic_minus_no_logic | relative_delta_pct_of_no_logic | logic_wins | ties | logic_losses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| success | higher | 4 | 1 | 1 | 0 | 0 | 0 | 4 | 0 |
| solvable | higher | 4 | 0.5 | 0.5 | 0 | 0 | 0 | 4 | 0 |
| confusion_ratio | lower | 2 | 1.90069 | 1.98841 | -0.087719 | -4.41 | 1 | 1 | 0 |
| path_optimal | higher | 4 | 0.266705 | 0.257715 | 0.00899 | 3.49 | 1 | 3 | 0 |
| tile_prior_kl | lower | 4 | 1.16414 | 1.17691 | -0.012771 | -1.09 | 3 | 0 | 1 |
| reconstruction_error | lower | 4 | 0.159091 | 0.210227 | -0.051136 | -24.32 | 2 | 1 | 1 |
| room_repair_rate | lower | 4 | 0.879545 | 0.902273 | -0.022727 | -2.52 | 1 | 3 | 0 |
| tiles_repaired | lower | 4 | 95.5 | 127.25 | -31.75 | -24.95 | 3 | 0 | 1 |
| topology_preservation_score | higher | 4 | 0.4375 | 0.4375 | 0 | 0 | 0 | 4 | 0 |
| generation_time_sec | lower | 4 | 279.89 | 496.596 | -216.706 | -43.64 | 3 | 0 | 1 |

## Interpretation

- Use LogicNet if the thesis claim is that logical guidance reduces downstream symbolic repair and improves neural semantic fidelity before repair.
- Do not claim from this run that LogicNet improves end-to-end solvability; solved-rate was identical at 50 percent on the paired sample.
- Keep LogicNet optional at runtime if generation cost/complexity matters, because this evidence is directional and small-sample.
- The fairest next run is a single-process `FULL,NO_LOGIC` ablation with more seeds so significance reporting can compare both configs directly.

## Caveats

- n=4 paired seeds is directional evidence, not a high-power statistical result.
- generation_time_sec came from separate runner invocations and includes run-level overhead, so use it only as a rough signal.
- Solvability is evaluated after symbolic repair/WFC, so it may mask neural generator differences.

Generated files:
- `paired_metric_summary.csv`
- `paired_seed_deltas.csv`
- `paired_raw_join.csv`
- `logicnet_vs_no_logic_evidence.json`
