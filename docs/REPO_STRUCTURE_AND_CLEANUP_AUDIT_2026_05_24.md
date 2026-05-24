# Repo Structure And Cleanup Audit

Last updated: 2026-05-24

## Folder Roles

| Folder | Role | Cleanup decision |
|---|---|---|
| `src/` | Canonical Python package and production/research implementation | Keep as source of truth |
| `scripts/` | Reproducible experiment, audit, and export commands | Keep; prefer new experiments here |
| `tests/` | Pytest coverage | Keep only importable pytest tests and fixtures |
| `docs/` | Current protocols, references, and audits | Keep current docs top-level; archive stale snapshots under `docs/archive/` |
| `configs/` | Validated YAML profiles | Keep |
| `bench/` | Standalone algorithm micro-benchmark harness | Keep; imported by benchmark scripts and GUI JPS path |
| `examples/` | Small demo artifacts and tutorial examples | Keep |
| `simulation/` | Backward-compatible shim package that re-exports `src.simulation` | Keep until legacy imports are no longer needed |
| `Data/`, `PAPERS/`, `REPORT_LATEX/` | Thesis inputs and paper assets | Keep |
| `outputs/`, `results/`, `exports/`, `artifacts/`, `tmp*` | Generated artifacts and local scratch data | Ignored by git; safe to clean between experiment runs |

## Intentional Overlaps

### `simulation/` and `src/simulation/`

`src/simulation/` is canonical. The root `simulation/` package contains shim
modules that re-export from `src.simulation` for backward compatibility. It is
not duplicate implementation logic.

### `src/core/logic_net.py` and `src/ml/logic_net.py`

`src/core/logic_net.py` is the production Block V LogicNet path used by
diffusion and the pipeline. `src/ml/logic_net.py` is a legacy differentiable
tortuosity/logic-loss module still covered by `tests/test_ml_components.py`.
Do not delete it until those tests and any legacy callers are migrated.

### `bench/` and `src/simulation/`

`bench/` contains standalone micro-benchmark solvers and timing harnesses.
Runtime gameplay/search code imports from `src.simulation`; benchmark-only code
may import `bench`.

## Cleanup Performed

- Removed misleading `test_*.py` scripts from `scripts/`:
  - `scripts/test_all_features.py` -> `scripts/validate_all_features.py`
  - `scripts/test_controllability.py` -> `scripts/validate_controllability.py`
- Promoted executable pytest coverage from `scripts/` to `tests/`:
  - `scripts/test_grammar_rules.py` -> `tests/test_grammar_rules.py`
  - `scripts/test_mathematical_rigor.py` -> `tests/test_mathematical_rigor.py`
- Moved manual debug probes out of `tests/` and into `scripts/debug/`.
- Moved manual validation probes out of `tests/`:
  - `tests/test_logicnet_pipeline_fix.py` -> `scripts/validate_logicnet_pipeline_fix.py`
  - `tests/test_training_setup_verification.py` -> `scripts/validate_training_setup.py`
- Converted empty pytest collection files into real regression tests:
  - `tests/test_wfc_integration.py`
  - `tests/test_simple_optimizer_check.py`
  - `tests/test_logicnet_registration_simple.py`
- Replaced hard-coded local absolute paths in active scripts/tests with
  repo-root discovery based on the current file path.
- Normalized touched validation/test output to ASCII so logs are readable in
  Windows terminals and CI.
- Removed tracked zero-byte tests:
  - `tests/test_logicnet_fix_simple.py`
  - `tests/test_logicnet_process_effect.py`
- Removed orphan tracked result artifact:
  - `tests/test_logicnet_process_effect_results.json`
- Aligned block numbering across architecture docs:
  - `Block V` = LogicNet guidance
  - `Block VI` = symbolic repair / overlay / stitching
  - `Block VII` = validation, P-CBS, and QD / MAP-Elites metrics
- Added the training hyperparameter/batch preflight checker:
  - `scripts/check_training_hyperparameters.py`
- Archived stale top-level documentation snapshots into `docs/archive/2026-q2/`
  and kept the root `docs/` folder limited to indexed current references,
  protocols, and stable runbooks.
- Replaced personal/local dataset defaults in standalone Zelda validation tools
  with repo-root-relative paths.

## Remaining Manual Cleanup

The tracked code folders are clean after this pass. The following ignored local
folders are generated cache/scratch artifacts only. Some were
removed during this pass; several old Windows temp folders returned access
denied and may require closing processes or elevated file ownership cleanup:

- `.pytest_tmp_codegen_audit`
- `.pytest_tmp_config`
- `pytest_basetemp_run`
- `artifacts/pytest_tmp`
- `outputs/pytest_tmp_codegen_audit2`
- `outputs/pytest_tmp_codegen_audit3`
- `outputs/pytest_tmp_codegen_audit4`
- `tmpapgwsida`
- `tmp_pytest_codex`

They are ignored by `.gitignore` and do not affect the tracked repo state.
