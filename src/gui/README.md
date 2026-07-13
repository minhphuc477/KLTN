# KLTN GUI Package Guide

This document is the working architecture reference for the GUI stack under `src/gui`.

It explains:
- how the GUI package is organized,
- which modules own each runtime responsibility,
- where orchestration lives after monolith extraction,
- how runtime flags and lifecycle flows work,
- how to extend the GUI safely without reintroducing monolithic code.

## 1) Purpose and Design Goals

The GUI layer powers ZAVE interactive validation, visualization, and solver experimentation.

Primary goals:
- Keep the main runner behavior stable while reducing monolithic methods.
- Split by interaction domain (map, solver, gameplay, rendering, topology, runtime).
- Keep one canonical import path per implementation.
- Preserve testability through dependency injection and thin wrappers.

## 2) Entry Points and Runtime Context

Main interactive entrypoint:
- `gui_runner.py` (repository root) defines `ZeldaGUI` and delegates most logic into `src/gui/*` modules.

Startup orchestration modules:
- `src/gui/app/gui_startup.py`: top-level boot process (`run_gui_main`).
- `src/gui/app/map_adapter_loader.py`: adapter-driven map loading.
- `src/gui/app/run_loop_pipeline.py`: main frame/event loop orchestration.

The GUI runner remains the integration surface, while feature logic is increasingly delegated to domain modules and orchestration bridges.

## 3) Folder Architecture

Current GUI subfolders:
- `ai`: AI dungeon generation controls, pipelines, and workers.
- `app`: startup, loop orchestration, and frame/event handlers.
- `common`: shared constants and the module catalog.
- `components`: widgets and component-level fallbacks.
- `control_panel`: control-panel behavior.
- `gameplay`: movement, inventory, paths, and actions.
- `map`: map loading, navigation, viewport, and minimap behavior.
- `orchestration`: cross-domain bridges grouped by domain.
- `rendering`: rendering and UI-overlay pipelines.
- `runtime`: display lifecycle, flags, routes, temporary files, toasts, and watchdogs.
- `solver`: solver scheduling, launch, recovery, requests, and comparison.
- `topology`: topology checks, matching, and export.

## 4) Canonical Import Policy

Use the domain implementations directly:
- `src.gui.orchestration.*`
- `src.gui.control_panel.*`
- `src.gui.solver.*`
- `src.gui.rendering.*`
- `src.gui.runtime.*`
- `src.gui.map.*`
- `src.gui.gameplay.*`
- `src.gui.topology.*`
- `src.gui.components.*`
- `src.gui.ai.*`

Import policy:
- New feature work and tests import canonical modules directly.
- Do not recreate forwarding-only packages; move shared behavior into a real domain module.

Machine-readable category index:
- `src/gui/common/module_catalog.py`

## 5) Orchestration Bridge Pattern

To keep `ZeldaGUI` API stable while reducing method body size, wrappers in `gui_runner.py` delegate to bridge modules with explicit dependency injection.

Bridge module examples:
- `src/gui/orchestration/solver/session_orchestration.py`
- `src/gui/orchestration/solver/launch_orchestration.py`
- `src/gui/orchestration/solver/comparison_orchestration.py`
- `src/gui/orchestration/solver/request_orchestration.py`
- `src/gui/orchestration/topology/orchestration.py`
- `src/gui/orchestration/topology/match_orchestration.py`
- `src/gui/orchestration/gameplay/action_orchestration.py`
- `src/gui/orchestration/gameplay/inventory_orchestration.py`
- `src/gui/orchestration/map/navigation_orchestration.py`
- `src/gui/orchestration/runtime/display_orchestration.py`
- `src/gui/orchestration/rendering/status_toast_orchestration.py`
- `src/gui/orchestration/rendering/panel_overlay_orchestration.py`
- `src/gui/orchestration/control_panel/animation_orchestration.py`

New code should target `src/gui/orchestration/*` for cross-domain flows.

Why this pattern is used:
- Preserves class method names and call contracts.
- Isolates dependency wiring (`pygame`, `time`, `logger`, helper functions).
- Improves testability and incremental refactoring safety.

## 6) Core Execution Flows

### 6.1 App Boot Flow

High-level flow:
1. Configure runtime/platform behavior (including display/bootstrap state).
2. Load maps through adapter or fallback test map.
3. Build `ZeldaGUI` instance.
4. Enter main loop.

Primary modules:
- `src/gui/app/gui_startup.py`
- `src/gui/app/map_adapter_loader.py`
- `src/gui/app/init_*.py`

### 6.2 Main Loop Flow

`src/gui/app/run_loop_pipeline.py` orchestrates per-frame responsibilities:
1. Poll events.
2. Dispatch keyboard/mouse/window handlers.
3. Update auto-step and continuous movement.
4. Update widgets/toasts/animations.
5. Handle async completion for preview/solver/AI generation.
6. Render frame and run display health checks.
7. Tick frame clock and exit on limits when testing.

### 6.3 Solver Lifecycle Flow

The solver path is split across canonical solver modules:
- Request/context assembly: `solver/request_helpers.py`, `orchestration/solver/request_orchestration.py`
- Orchestration entrypoints: `orchestration/solver/request_orchestration.py`, `orchestration/solver/session_orchestration.py`, `orchestration/solver/launch_orchestration.py`
- Startup policy and timeout/recovery: `solver/start_logic.py`, `solver/recovery.py`, `orchestration/solver/session_orchestration.py`
- Launch and fallback: `solver/launching.py`, `solver/worker_bootstrap.py`, `orchestration/solver/launch_orchestration.py`
- Sync debug route: `solver/sync_execution.py`
- Comparison and MAP-Elites hooks: `solver/comparison_runner.py`, `orchestration/solver/comparison_orchestration.py`

### 6.4 Rendering Flow

Rendering is layered and pipelined:
- Frame orchestrator: `rendering/render_frame_pipeline.py`
- Map and tile strata: `rendering/map_render_pipeline.py`
- Path and overlay strata: `rendering/path_overlay_pipeline.py`, `rendering/overlay_ui_pipeline.py`
- Post-map UI and sidebars: `rendering/post_map_ui_pipeline.py`, `rendering/sidebar_sections.py`
- Diagnostics/status/help/debug/toasts: `rendering/status_display.py`, `rendering/debug_overlay.py`, `rendering/help_overlay.py`, `orchestration/rendering/status_toast_orchestration.py`, `orchestration/rendering/panel_overlay_orchestration.py`

## 7) Runtime Flags and Environment Variables

Core GUI runtime flags are loaded by `src/gui/runtime/flags.py`:
- `KLTN_LOG_LEVEL`: set `DEBUG` for verbose logging.
- `KLTN_DEBUG_INPUT=1`: enable input diagnostics.
- `KLTN_SYNC_SOLVER=1`: force synchronous solver mode (debug-only, blocks UI).
- `KLTN_DEBUG_SOLVER_FLOW=1`: verbose solver flow logging.

Common loop/test controls used by app utilities:
- `KLTN_TEST_MODE`, `PYTEST_CURRENT_TEST`, `CI`: auto-bound run frames.
- `KLTN_RUN_MAX_FRAMES`: explicit frame cap in test mode.
- `KLTN_SOLVER_TIMEOUT`: override computed timeout budget.

Map precalc startup option:
- `KLTN_PRECALC_SOLVES=1` in adapter loader triggers background pre-solve attempts.

## 8) Testing Strategy for GUI Modules

The repository includes broad GUI unit coverage under `tests/test_gui_*.py` (around 60 files), including:
- control panel behavior,
- solver request/start/recovery/scheduling,
- route I/O and temp-file handling,
- topology helpers/export/match controls,
- rendering helpers and overlays,
- runtime diagnostics and focus/fullscreen behavior,
- gameplay movement, auto-step, inventory, and block-push systems.

Recommended focused regression slice during GUI refactors:
- `tests/test_gui_control_panel_animation.py`
- `tests/test_gui_auto_solve_execution.py`
- `tests/test_gui_inventory_display.py`
- `tests/test_gui_route_io.py`
- `tests/test_gui_solver_request_orchestration.py`

Suggested validation sequence:
1. Static diagnostics for touched files.
2. `python -m compileall` on touched GUI modules.
3. Focused GUI pytest slice.
4. Broader suite only when behavior contract changes.

## 9) Development Conventions

When adding or changing GUI behavior:
- Prefer extracting cohesive logic into canonical domain modules.
- Keep `gui_runner.py` methods as thin delegates where possible.
- Inject heavy dependencies (`pygame`, `time`, `os`, logger, helper functions) rather than importing implicitly inside helpers.
- Preserve behavior and method signatures before optimization.
- Avoid broad cross-domain refactors in one step; use domain batches.

When changing imports:
- Use the canonical domain path.
- Update callers and behavior tests in the same change when a module moves.

## 10) How to Add a New GUI Feature

Recommended workflow:
1. Choose canonical domain folder (`gameplay`, `solver`, `rendering`, etc.).
2. Add feature helper module there.
3. If existing GUI class API must stay stable, add or extend an orchestration bridge.
4. Delegate from `gui_runner.py` wrapper method.
5. Add targeted tests in `tests/test_gui_<feature>.py`.
6. Run focused regression slice and compile checks.

This keeps the GUI maintainable while preserving runtime compatibility.

## 11) Maintenance Notes

- `src/gui/common/module_catalog.py` should remain up to date as modules move or new categories are introduced.
- Forwarding-only compatibility packages were removed after all repository callers migrated.

## 12) Quick Reference

Use these as primary import targets in new code:
- `src.gui.orchestration.*`
- `src.gui.control_panel.*`
- `src.gui.map.*`
- `src.gui.gameplay.*`
- `src.gui.solver.*`
- `src.gui.rendering.*`
- `src.gui.runtime.*`
- `src.gui.topology.*`
- `src.gui.components.*`
- `src.gui.ai.*`

Shared constants and catalog metadata remain under `src.gui.common.*`; this is a
real shared domain, not a compatibility layer.
