# Kaggle Research Evidence Runbook

This is the execution path for the missing evidence layer: train checkpoints,
run the ablations/baselines, and package result artifacts for Chapter 4 tables.

## One-Command Full Run

In a Kaggle notebook, use `%%bash` as the first line of the cell. A bare
`bash ...` line inside a Python cell causes `SyntaxError`.

```bash
%%bash
cd /kaggle/working/KLTN
bash kaggle/hmolqd_training_suite/run_kaggle_full_research_suite.sh
```

Kaggle API kernel equivalent:

```bash
%%bash
cd /kaggle/working/KLTN
python kaggle/hmolqd_training_suite/kaggle_kernel.py --mode full
```

Default behavior:

- trains `vqvae2` on the `stage_full` branch
- trains diffusion, fast sampler, and masked-room branches
- runs the evidence suite against the trained branch
- writes results under `/kaggle/working/hmolqd_training_suite/research`
- packages artifacts under `/kaggle/working/hmolqd_training_suite/artifacts`

## Evidence Suite Only

Use this after checkpoints already exist:

```bash
%%bash
cd /kaggle/working/KLTN
bash kaggle/hmolqd_training_suite/run_kaggle_research_suite.sh
```

Kaggle API kernel equivalent:

```bash
%%bash
cd /kaggle/working/KLTN
python kaggle/hmolqd_training_suite/kaggle_kernel.py --mode evidence
```

The runner infers:

- primary run: `$OUT_ROOT/downstream/${EVAL_TOKENIZER}_${EVAL_BRANCH}`
- config: `$OUT_ROOT/configs/${EVAL_TOKENIZER}_${EVAL_BRANCH}.yaml`
- VQ-VAE checkpoint: `$OUT_ROOT/tokenizers/${EVAL_TOKENIZER}/checkpoints/vqvae/vqvae_pretrained.pth`
- diffusion checkpoint: `$RUN_DIR/checkpoints/diffusion/best_model.pth`
- LogicNet checkpoint: `best_logic_model.pth` if present, else `best_model.pth`

## Core Outputs

The evidence suite creates:

```text
/kaggle/working/hmolqd_training_suite/research/
  research_suite_manifest.json
  steps.tsv
  logs/
  conditioning_logicnet_repair/
  fixed_graph/
  generated_graph_full_pipeline/
  ablation/
  random_baseline/
  matched_budget/
  pcg_benchmark_alignment/
  ood_blinded_eval/
  designer_controllability/
  pcbs_persona_map_sweep/
  pcbs_component_ablation/
  protocol_to_baselines/
  compute_sample_efficiency/
```

The final zip is:

```text
/kaggle/working/hmolqd_training_suite/artifacts/hmolqd_kaggle_research_artifacts.zip
```

## Fast Smoke Run

```bash
%%bash
cd /kaggle/working/KLTN
QUICK=1 \
TOKENIZERS="vqvae2" \
BRANCHES="stage_full" \
bash kaggle/hmolqd_training_suite/run_kaggle_full_research_suite.sh
```

Quick mode cuts seeds, generations, diffusion steps, and P-CBS timeouts. It is
for checking that the whole pipeline runs, not for thesis claims.

## Recommended Final Run

```bash
%%bash
cd /kaggle/working/KLTN
TOKENIZERS="vqvae2" \
BRANCHES="stage_full" \
PROFILE="auto" \
bash kaggle/hmolqd_training_suite/run_kaggle_full_research_suite.sh
```

For a broader model-selection run:

```bash
%%bash
cd /kaggle/working/KLTN
TOKENIZERS="vqvae vqvae2" \
BRANCHES="stage_full stage_tokens_only stage_trace_only stage_loss010 stage_loss050" \
bash kaggle/hmolqd_training_suite/run_kaggle_full_research_suite.sh
```

That run is expensive. Use it only when Kaggle timeout and storage budget are
sufficient.

## Main Evidence Switches

Set any of these to `0` to skip a section:

- `RUN_CONDITIONING_LOGICNET_REPAIR`
- `RUN_FIXED_GRAPH`
- `RUN_GENERATED_GRAPH`
- `RUN_ABLATION_STUDY`
- `RUN_RANDOM_BASELINE`
- `RUN_MATCHED_BUDGET`
- `RUN_PCG_BENCHMARK`
- `RUN_OOD_BLINDED`
- `RUN_DESIGNER_CONTROLLABILITY`
- `RUN_PCBS_SWEEP`
- `RUN_PCBS_COMPONENT_ABLATION`
- `RUN_PROTOCOL_COMPARE`
- `RUN_COMPUTE_CONSOLIDATION`

Useful budget controls:

- `CONDITIONING_SEEDS`, default `42,43,44`
- `FIXED_GRAPH_SEEDS`, default `20260404 20260405 20260406`
- `GENERATED_GRAPH_SEEDS`, default `20260514:20260518`
- `EVAL_NUM_SAMPLES`, default `8`
- `ABLATION_NUM_SAMPLES`, default `8`
- `POPULATION_SIZE`, default `24`
- `GENERATIONS`, default `24`
- `DIFFUSION_STEPS`, default `25`
- `TIMEOUT_ASTAR`, default `200000`
- `TIMEOUT_PCBS`, default `50000`

## Minimum Thesis Evidence

For the claims reviewed in the repo audit, the minimum useful completed artifact
set is:

1. `conditioning_logicnet_repair/conditioning_logicnet_repair_logic_deltas.csv`
2. `fixed_graph/summary.json`
3. `generated_graph_full_pipeline/full_pipeline_results.json`
4. `ablation/ablation_report.json`
5. `matched_budget/matched_budget_report.json`
6. `pcg_benchmark_alignment/pcg_benchmark_alignment_report.json`
7. `ood_blinded_eval/blinded/rating_sheet.csv`
8. `pcbs_persona_map_sweep/summary.json`
9. `compute_sample_efficiency/compute_sample_efficiency_report.md`

If an experiment fails, rerun with:

```bash
%%bash
cd /kaggle/working/KLTN
CONTINUE_ON_EVIDENCE_FAILURE=1 bash kaggle/hmolqd_training_suite/run_kaggle_research_suite.sh
```

Then inspect `research/steps.tsv` and `research/logs/*.log`.
