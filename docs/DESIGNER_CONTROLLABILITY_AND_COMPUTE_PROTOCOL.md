# Designer Controllability And Compute Protocol

Last updated: 2026-05-23

This protocol adds two missing thesis-facing evidence layers:

1. designer-controllability target satisfaction
2. consolidated compute and sample-efficiency reporting

The scripts are intentionally safe to run before expensive experiments. The
controllability script writes a plan by default and only generates dungeons when
`--execute` is passed.

## Research Basis

- Pereira's dungeon work evaluates whether generated dungeons match designer
  inputs such as room count, keys, locks, and linearity, then validates them with
  player-facing evidence.
- Snodgrass and Ontanon's controllable PCG work frames controllability as
  satisfying desired constraints during or after sampling.
- The PCG Benchmark standardizes evaluation around quality, diversity, and
  controllability.
- PCGML surveys identify small data, parameter tuning, and limited training data
  as recurring issues, so sample-efficiency and compute reporting should be
  visible rather than left in scattered logs.

Sources:

- <https://repositorio.usp.br/item/002964434>
- <https://repositorio.usp.br/item/003032388>
- <https://www.ijcai.org/Abstract/16/116>
- <https://arxiv.org/abs/2503.21474>
- <https://arxiv.org/abs/1702.00539>
- <https://arxiv.org/abs/2404.18657>

## 1. Designer Controllability Proof

Script:

- `scripts/run_designer_controllability_proof.py`

What it tests:

- reference-centered target matching
- Pereira-style rows for small/easy, balanced key-lock, hard backtracking, and
  large stress dungeons
- one-axis sweeps for linearity, key/lock pressure, and room count
- methods:
  - `FULL_GA`
  - `FULL_CVT`
  - `CORE_GA`

Important interpretation:

- Raw `key_count` and `lock_count` are now passed directly to Block I fitness
  and violation scoring. The output table still reports both target and actual
  counts so the proof can show exact support or expose residual target drift.
- The target suite includes 100-room and 500-room stress rows to mirror the
  Pereira-style large-room discussion without pretending they are cheap smoke
  tests.

Plan-only command:

```powershell
python scripts\run_designer_controllability_proof.py `
  --output results\designer_controllability_proof
```

Full run command for later:

```powershell
python scripts\run_designer_controllability_proof.py `
  --execute `
  --output results\designer_controllability_proof `
  --methods FULL_GA,FULL_CVT,CORE_GA `
  --samples-per-target 8 `
  --population-size 32 `
  --generations 40 `
  --seed 42
```

Fast smoke command:

```powershell
python scripts\run_designer_controllability_proof.py `
  --execute `
  --quick `
  --output results\designer_controllability_smoke
```

Outputs:

- `designer_controllability_plan.json`
- `designer_controllability_plan.md`
- `designer_controllability_raw.csv`
- `designer_controllability_summary.csv`
- `designer_target_response_monotonicity.csv`
- `designer_controllability_payload.json`
- optional `designer_controllability_graphs.json`

Report-facing metrics:

- per-target normalized error
- per-target pass/fail under fixed tolerance
- macro pass rate across controlled metrics
- raw count errors for rooms, keys, locks, enemies, puzzles, and items
- descriptor errors for linearity, leniency, progression complexity, topology
  complexity, cycle density, shortcut density, gate density, and depth ratios
- monotonic target-response rows for linearity, room count, and gate pressure
- stress rows for `p_large_stress_100` and `p_large_stress_500`

## 2. Compute And Sample-Efficiency Consolidation

Script:

- `scripts/consolidate_compute_sample_efficiency.py`

What it does:

- scans existing `outputs/` and `results/` artifacts
- extracts runtime-like fields, sample-count fields, epoch/step fields,
  loss-like metrics, and success/pass metrics
- inventories JSON/CSV/checkpoint artifacts by run root
- reports missing runtime/sample/metric fields explicitly

Run command:

```powershell
python scripts\consolidate_compute_sample_efficiency.py `
  --roots outputs results `
  --output results\compute_sample_efficiency
```

Outputs:

- `metric_observations.csv`
- `artifact_inventory.csv`
- `compute_sample_efficiency_summary.csv`
- `compute_sample_efficiency_payload.json`
- `compute_sample_efficiency_report.md`

Interpretation rules:

- `observed_runtime_sec` is the largest runtime-like value found for a run; it
  is not guaranteed GPU time.
- `samples_per_sec` is only valid when both runtime and sample count are found.
- `best_loss_like_metric` is only comparable inside compatible families.
- `best_success_like_metric` is only comparable inside compatible families.

Minimum final-run metadata contract:

- `wall_time_sec` or `runtime_sec`
- generated/training sample count
- seed
- config snapshot path
- checkpoint path
- model parameter count where relevant
- best validation metric and the step/epoch at which it occurred
- generation/evaluation pass metric where relevant

## 3. How This Completes The Missing Evidence Layer

The repo already has fixed-seed ablations, PCG Benchmark alignment, OOD/blinded
packet generation, and P-CBS. These additions fill two narrower gaps:

- controllability is no longer only inferred from generic descriptors; it gets a
  target-satisfaction table.
- compute and sample efficiency are no longer scattered across logs and old
  run notes; they get one consolidated table with missing fields visible.
