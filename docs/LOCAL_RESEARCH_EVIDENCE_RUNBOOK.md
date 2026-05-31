# Local Research Evidence Runbook

This runbook replaces the Kaggle workflow for local runs.

## 1. Environment Check

```bash
python -m pytest tests/test_config_system.py tests/test_neural_pipeline.py -q
python -m py_compile src/pipeline/dungeon_pipeline.py src/pipeline/runtime.py
```

## 2. Train Or Load Models

Use the project config as the source of truth:

```bash
python -m src.train_diffusion --config configs/zelda_hmolqd.yaml
python -m src.train_masked_room --config configs/zelda_hmolqd.yaml
```

For a smoke run:

```bash
python -m src.train_diffusion --config configs/zelda_hmolqd.yaml --quick
python -m src.train_masked_room --config configs/zelda_hmolqd.yaml --quick
```

## 3. Solvability And A* Evidence

Run the hard oracle and generated-output checks:

```bash
python scripts/validate_training_setup.py
python scripts/run_generated_graph_full_pipeline_eval.py --output-dir results/local_generated_graph_eval
python scripts/run_fixed_graph_multi_seed_audit.py --output-dir results/local_fixed_graph_audit
```

Core paper metrics to report:

- A* oracle solvability rate.
- Path length and states explored.
- Failure classes: syntax invalid, unreachable, resource lock, timeout.
- Generated-map pass rate over fixed seeds.

## 4. LogicNet Guidance Ablations

Run guidance scale and timing ablations:

```bash
python scripts/run_conditioning_logicnet_repair_ablation.py --execute --output-dir results/local_logicnet_ablation
```

Report at least:

- `guidance_scale=0` baseline.
- One or more positive guidance scales.
- Active-fraction or warm-up schedule if enabled.
- Solvability, diversity, and invalid-output rates.

## 5. P-CBS Persona Evidence

Run P-CBS against the same oracle-solvable maps:

```bash
python scripts/run_pcbs_persona_map_sweep.py --data-root "Data/The Legend of Zelda" --output-dir results/pcbs_persona_map_sweep --oracle-solved-only
python scripts/run_pcbs_component_ablation.py --data-root "Data/The Legend of Zelda" --output-dir results/pcbs_component_ablation
```

Report:

- Cognitive gap rate: oracle solves but P-CBS fails.
- Confusion index.
- Navigation entropy.
- Cognitive load.
- Puzzle stall fraction.
- Persona/component ablation deltas.

## 6. Telemetry-Driven Persona Calibration

Collect local playtest telemetry with `src.utils.playtest_telemetry.PlaytestTelemetryCollector`, then calibrate:

```bash
python scripts/calibrate_pcbs_personas_from_telemetry.py \
  --telemetry results/playtest \
  --pcbs-sweep-csv results/pcbs_persona_map_sweep/pcbs_persona_map_sweep.csv \
  --output-dir results/pcbs_telemetry_calibration
```

Outputs:

- `pcbs_telemetry_targets.json`
- `pcbs_baseline_metrics.json`
- `pcbs_persona_overrides.json`
- `pcbs_calibration_report.md`

Use `pcbs_persona_overrides.json` as the empirical calibration artifact for thesis/paper claims. The built-in persona constants should be described as priors; calibrated overrides are the telemetry-backed values.

## 7. Minimum Evidence Bundle For The Paper

Store these files under one timestamped result directory:

- Training config and checkpoint metadata.
- A* oracle validation CSV/JSON.
- LogicNet guidance ablation CSV/JSON.
- P-CBS persona sweep CSV/JSON/Markdown.
- P-CBS component ablation CSV/JSON/Markdown.
- Telemetry calibration report and overrides.

