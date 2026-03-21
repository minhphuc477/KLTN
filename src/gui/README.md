# GUI Module Categories

This folder is modularized by interaction domain using smaller subfolders. Existing imports remain stable via top-level compatibility shim modules.

## Canonical Categories

- `controls`: user interaction handlers and control-surface adapters.
  - Includes shims for control panel flow, map controls, viewport/navigation, and runtime flags.
- `overlay`: UI overlays and heads-up rendering adapters.
  - Includes debug/help/status/toast/inventory marker overlays.
- `services`: orchestration and operational flow adapters.
  - Includes solver flow, topology flow, route I/O, path analysis/strategy, and watchdog operations.
- `common`: shared constants/catalog/widgets compatibility surface.
  - Includes constants/fallbacks/module catalog and common component shims.

## Domain Packages (canonical implementations)

- `control_panel`: panel layout/render/interaction canonical implementations.
- `solver`: solver scheduling/start/recovery/process canonical implementations.
- `rendering`: render helper, map overlay, status/debug/help canonical implementations.
- `runtime`: runtime flags/lifecycle/route/temp/toast canonical implementations.
- `map`: map loading/navigation/viewport/minimap canonical implementations.
- `gameplay`: path/inventory/step/auto-solve canonical implementations.
- `topology`: topology export/helpers/precheck/matching canonical implementations.
- `components`: shared widget/constant/fallback canonical implementations.
- `ai`: AI generation controls/pipeline/worker canonical implementations.

## Compatibility

- Legacy flat module names (for example `control_panel_logic.py`, `map_loading.py`, `solver_start_logic.py`, `render_helpers.py`, `topology_helpers.py`, and `widgets.py`) are kept as shims that re-export from categorized module paths.
- New code should prefer categorized imports, for example `src.gui.controls.control_panel_logic`, `src.gui.services.solver_start_logic`, `src.gui.overlay.status_display`, `src.gui.common.constants`, and domain-canonical imports like `src.gui.solver.start_logic`.

See `src/gui/common/module_catalog.py` for the machine-readable mapping.
