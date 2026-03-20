# OOD Scaling and Blinded Human-Eval Protocol

This protocol adds two thesis-grade evaluations beyond in-distribution solvability:

1. Out-of-distribution (OOD) scaling across room-count regimes.
2. Blinded human-rating packet generation (organization/readability/novelty/fairness).

## Script

- `scripts/run_ood_scaling_and_blinded_eval.py`

## What It Runs

- Methods (configurable):
  - `FULL_GA` (`rule_space=full`, `search_strategy=ga`)
  - `FULL_CVT` (`rule_space=full`, `search_strategy=cvt_emitter`)
  - `CORE_GA` (`rule_space=core`, `search_strategy=ga`)
- Regimes:
  - `in_dist` (reference-like room counts from VGLC quantiles)
  - `ood_small` (smaller-than-reference budgets)
  - `ood_large` (larger-than-reference budgets)
- Metrics:
  - completeness/validity
  - repair-rate and generation-time rejection diagnostics
  - descriptor means
  - novelty/fidelity overlap proxies
  - runtime

## Run Command

```bash
python scripts/run_ood_scaling_and_blinded_eval.py \
  --data-root "Data/The Legend of Zelda" \
  --methods FULL_GA,FULL_CVT \
  --num-samples 8 \
  --seed 42 \
  --population-size 24 \
  --generations 24 \
  --output results/ood_blinded_eval
```

## Outputs

- `ood_scaling_summary.csv`
- `ood_scaling_payload.json`
- `ood_scaling_report.json`
- `blinded/blinded_manifest.csv`
- `blinded/blinded_key.csv`
- `blinded/rating_sheet.csv`
- `blinded/images/*.png`

## Blinded Evaluation Workflow

1. Share only:
   - `blinded_manifest.csv`
   - `rating_sheet.csv`
   - `blinded/images/`
2. Collect ratings from evaluators.
3. Keep `blinded_key.csv` hidden until scoring is finalized.
4. Join ratings with `blinded_key.csv` for final analysis.

