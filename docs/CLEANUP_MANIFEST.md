# Cleanup Manifest — KLTN

This manifest records cleanup actions taken and files recommended for removal or ignore in the `KLTN` repository.

## Actions taken ✅

1. Removed tracked generated artifacts (committed by accident):
   - All `__pycache__/` entries and `.pyc` files were removed from source control and added to `.gitignore`.
   - Removed generated output artifacts from `output/` (e.g. `processed_data.pkl`, `validation_report.json`) from the repository index and added `output/` and `*.pkl` to `.gitignore`.

2. `.gitignore` updated to ignore:
   - `__pycache__/`, `*.py[cod]`, `.pytest_cache/`, `output/`, `*.pkl`, `validation_report.json`, editor artifacts (`.vscode/`, `.DS_Store`, `*.swp`), and build artifacts (`dist/`, `build/`, `*.egg-info/`).

3. Fixed a Windows multiprocessing test issue in `tests/test_jps_no_hang.py` by moving the target to module scope and returning exceptions via `Queue` so the child process results are deterministically observable.

## Files removed from git index (kept locally)
- `Data/__pycache__/zelda_core.cpython-313.pyc`
- `__pycache__/gui_runner.cpython-313.pyc`
- `output/processed_data.pkl`, `output/validation_report.json`
- `simulation/__pycache__/...`, `src/__pycache__/...`, `src/visualization/__pycache__/...`, `tests/__pycache__/...` (multiple `.pyc` entries)

> These files are generated at runtime or by test collection and should not be tracked.

## Further recommendations (NOT yet applied)

- Move large binary assets out of the repo if they are not required for tests (e.g., certain images in `Data/Original/`). If they are required, consider adding a lightweight placeholder and instructing contributors to download the dataset via a script.
- Add a `scripts/clean` or `Makefile` target to remove generated artifacts locally (e.g., `__pycache__`, `*.pyc`, `output/*`) to keep working directories clean.
- Consider adding a pre-commit hook that runs `git ls-files --others --exclude-standard` checks and checks for committed bytecode.

## QA notes
- `tests/*jps*` pass locally after the test fix.
- Removing `.pyc` files is non-invasive and safe (they will be regenerated as needed).

---

If you want, I can open a small PR that contains the `.gitignore` update, the test fix, and the cleanup manifest so it can be reviewed and merged. Would you like me to proceed and push the branch + PR? 
