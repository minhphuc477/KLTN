# Experimental Baselines

This directory contains isolated baselines for the paper comparison plan:

- `train_llm_baseline.py`: flattened-token GPT2-style autoregressive model.
- `train_dcgan_baseline.py`: lightweight discrete DCGAN using straight-through
  Gumbel-Softmax during generator training.
- `run_wfc_baseline.py`: local overlapping-pattern Wave Function Collapse
  baseline with no external WFC dependency.

Every script supports `--dry-run` and writes a JSON report under
`results/baselines/...`. Dry-runs intentionally skip P-CBS because P-CBS is a
search-heavy metric; use `--run-pcbs` on full runs when producing paper tables.

