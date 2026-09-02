# Experimental Baselines

This directory contains isolated **room-level diagnostic** baselines:

- `train_llm_baseline.py`: flattened-token GPT2-style autoregressive model.
- `train_dcgan_baseline.py`: lightweight discrete DCGAN using straight-through
  Gumbel-Softmax during generator training.
- `run_wfc_baseline.py`: local overlapping-pattern Wave Function Collapse
  baseline with no external WFC dependency.

These scripts are not confirmatory full-dungeon baselines. Their CLIs do not
materialize leave-one-dungeon-out folds, accept mission-graph conditions,
assemble complete dungeons, or emit `PublicationResultRecord` attempts. Do not
place their numbers in the primary N/G/G-L/G-LG table. A publishable external
baseline must use the same outer folds, mission-graph test bank, attempt budget,
stage contract, timeout accounting, and independent exact validator.

Every script supports `--dry-run` and writes a JSON diagnostic report under
`results/baselines/...`. P-CBS remains a secondary behavioral diagnostic and
must not be described as an independent hard-validity oracle.
