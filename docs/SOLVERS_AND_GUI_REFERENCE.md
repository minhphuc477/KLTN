# Solver Algorithms and GUI Feature Reference

This document is the code-accurate reference for:
- Solver architecture and algorithm behavior
- Formula and heuristic logic
- GUI features (checkboxes, dropdowns, buttons, hotkeys)
- Current implementation status for each feature

Validated against:
- `gui_runner.py`
- `src/simulation/*`

Last updated: 2026-02-23.

## 1. Solver Architecture

The GUI solve flow is:

1. User selects solver/options in the control panel.
2. `_start_auto_solve()` synchronizes dropdown state (`_sync_solver_dropdown_settings`).
3. `_schedule_solver()` starts a background subprocess (`multiprocessing.Process`).
4. Child process runs `_run_solver_and_dump()` -> `_solve_in_subprocess(...)`.
5. Child writes result pickle to temp output file.
6. Main loop polls completion, loads pickle, then:
- starts animation immediately, or
- opens path preview overlay/dialog.
7. Main loop always cleans up temp files and solver state.

Important robustness details:
- Solver results are file-based (pickle), not pipe-based, to avoid large object transfer issues.
- Thread fallback exists if process spawn fails.
- Timeouts and forced termination are implemented in the main loop polling stage.

## 2. Full Game-State Model

All main solvers operate on full game state, not only `(row, col)`.

State includes:
- Position
- Small key count
- Bomb count
- Boss key possession
- Key item possession (ladder/item)
- Opened doors set
- Collected items set
- Pushed block state

Transitions are validated by environment logic (`_try_move_pure`), so door/key/bomb/item constraints are part of the search graph.

## 3. Solver Algorithms (GUI Solver Dropdown)

### 3.1 Algorithm Index Mapping

| Index | GUI Name | Primary Implementation |
|---|---|---|
| 0 | A* | `run_game_state_solver` -> `AStarGameStateSolver` -> `StateSpaceAStar(search_mode="astar")` |
| 1 | BFS | `run_game_state_solver` -> `BFSGameStateSolver` -> `StateSpaceAStar(search_mode="bfs")` |
| 2 | Dijkstra | `run_game_state_solver` -> `DijkstraGameStateSolver` -> `StateSpaceAStar(search_mode="dijkstra")` |
| 3 | Greedy | `run_game_state_solver` -> `GreedyGameStateSolver` -> `StateSpaceAStar(search_mode="greedy")` |
| 4 | D* Lite | `DStarLiteSolver` |
| 5 | DFS/IDDFS | `StateSpaceDFS(use_iddfs=True)` |
| 6 | Bidirectional A* | `BidirectionalAStar` |
| 7 | CBS (Balanced) | `CognitiveBoundedSearch(persona="balanced")` |
| 8 | CBS (Explorer) | `CognitiveBoundedSearch(persona="explorer")` |
| 9 | CBS (Cautious) | `CognitiveBoundedSearch(persona="cautious")` |
| 10 | CBS (Forgetful) | `CognitiveBoundedSearch(persona="forgetful")` |
| 11 | CBS (Speedrunner) | `CognitiveBoundedSearch(persona="speedrunner")` |
| 12 | CBS (Greedy) | `CognitiveBoundedSearch(persona="greedy")` |

### 3.2 Core Search Formulas

Inside `StateSpaceAStar`:

- A*: `f = g + h`
- BFS mode: `f = depth` (full game-state BFS)
- Dijkstra mode: `f = g`
- Greedy mode: `f = h`
- ARA* option: `f = g + w*h` (when `enable_ara=True`, `w=ara_weight`)

Priority options:
- `priority_tie_break`: adds locked-door-based tie-break term
- `priority_key_boost`: prefers states that collect keys
- Both are applied through priority tuple ordering in the open set

### 3.3 Heuristic Logic (StateSpaceAStar)

Baseline:
- Manhattan distance to goal

Enhancements:
- Graph BFS distance to goal room (when topology graph available)
- Locked-door requirement estimate
- Penalties if inventory is insufficient:
  - missing keys for locked doors
  - missing bombs for bomb doors
  - missing boss key for boss doors
  - missing ladder/item for element tiles
- Optional plan-guided correction from room-level abstract plan
- Persona modes (`balanced`, `speedrunner`, `completionist`) adjust weighting

### 3.4 Representation Modes

For A*/BFS/Dijkstra/Greedy wrappers, representation is forwarded to `StateSpaceAStar`:

- `hybrid`: room/graph front-end, then macro/tile fallback
- `tile`: tile-level state search only
- `graph`: graph-level only; fails if graph topology unavailable

Hierarchy strategy in `solve()`:
1. Room-level graph search
2. Macro-action POI search (A* mode)
3. Tile-level state search fallback

### 3.5 D* Lite

Implementation highlights:
- Maintains `g(s)` and `rhs(s)`
- Key: `[min(g,rhs)+h, min(g,rhs)]`
- Supports replan API (`replan`, `needs_replan`)
- If primary search fails, falls back to `StateSpaceAStar` for correctness

### 3.6 DFS/IDDFS

Implementation highlights:
- IDDFS enabled by default in GUI dispatch (`use_iddfs=True`)
- Depth-limited recursive DFS iterations with increasing depth bound
- Tracks DFS metrics: depth reached, backtracks, cycle detections

### 3.7 Bidirectional A*

Implementation highlights:
- Forward and backward frontiers
- Exact and approximate collision checks
- Backward predecessor logic handles inventory compatibility constraints
- Path reconstruction concatenates forward path and reversed backward path

### 3.8 CBS Personas

CBS uses bounded-rational scoring:
- `U = alpha*goal_progress + beta*info_gain - gamma*risk`

Cognitive components:
- Limited vision model
- Belief map with decay
- Capacity-bounded working memory
- Multi-heuristic decision system

Outputs include cognitive metrics:
- confusion index
- navigation entropy
- cognitive load
- aha latency
- replans/confusion events/backtrack loops

## 4. Other Solver Modules in `src/simulation`

These exist in codebase but are not all fully wired into main auto-solve path:

| Module | Purpose | GUI Integration Status |
|---|---|---|
| `multi_goal.py` | Multi-waypoint item collection ordering | Not wired to main solve pipeline |
| `parallel_astar.py` | Multiprocess hash-partitioned A* | Not wired end-to-end from GUI checkbox |
| `solver_comparison.py` | Algorithm metric comparison helper | Button path uses custom in-GUI comparison worker instead |
| `map_elites.py` | MAP-Elites archive and diversity metrics | Wired via `Run MAP-Elites` button + overlay toggle |

## 5. GUI Features

### 5.1 Checkboxes

| Flag | GUI Label | Behavior | Status |
|---|---|---|---|
| `solver_comparison` | Solver Comparison | Exposed in UI/presets; comparison is actually triggered by button | Not wired (checkbox-only) |
| `parallel_search` | Parallel Search | State exists and result poll exists | Not wired end-to-end |
| `multi_goal` | Multi-Goal Pathfinding | Flag exists only | Not wired |
| `ml_heuristic` | ML Heuristic | Used by quick grid/path planner heuristic | Partial (not main subprocess solver) |
| `dstar_lite` | D* Lite Replanning | Replan check exists in auto-step loop | Partial (activation state not fully wired) |
| `show_heatmap` | Show Heatmap Overlay | Renders search heatmap overlay | Implemented |
| `show_path` | Show Path Overlay | Toggle message exists | Partial (path rendering currently unconditional when path exists) |
| `show_map_elites` | Show MAP-Elites Overlay | Draws occupancy mini-overlay when MAP-Elites result exists | Implemented |
| `show_topology` | Show Topology Overlay | Renders graph nodes/edges over map | Implemented |
| `show_topology_legend` | Topology Legend (details) | Renders topology legend/details | Implemented |
| `show_minimap` | Show Minimap | Toggles minimap render and click-to-jump | Implemented |
| `diagonal_movement` | Diagonal Movement | Used in quick grid planner | Partial (not dominant in subprocess solver pipeline) |
| `use_jps` | Use Jump Point Search (JPS) | Enables JPS attempt in quick grid planner | Partial |
| `show_jps_overlay` | Show JPS Overlay | Draws jump segments/points if trace available | Implemented |
| `speedrun_mode` | Speedrun Mode | Set by presets | Not wired (behavioral logic) |
| `dynamic_difficulty` | Dynamic Difficulty | Flag exists only | Not wired |
| `force_grid` | Force Grid Solver | UI flag exists | Not wired to `force_grid_algorithm` runtime switch |
| `enable_prechecks` | Enable Prechecks | Runs connectivity + key/lock lower-bound checks before solve start | Implemented |
| `auto_prune_on_precheck` | Auto-Prune Dead-Ends on Precheck | Optional dead-end prune pass with undo snapshot capture | Implemented |
| `priority_tie_break` | Priority: Tie-Break by Locks | Alters open-set priority tuple | Implemented |
| `priority_key_boost` | Priority: Key-Pickup Boost | Alters open-set priority tuple | Implemented |
| `enable_ara` | Enable ARA* (weighted A*) | Enables weighted `f = g + w*h` | Implemented |
| `persist_dropdown_on_select` | Keep dropdown open after select | Passed to dropdown widgets when built | Partial (applies on widget build/rebuild) |

### 5.2 Dropdowns

| Dropdown | Options | Behavior | Status |
|---|---|---|---|
| Floor | Floor 1/2/3 | UI value only | Not wired |
| Zoom | 25%-200% | Applies map zoom and redraw | Implemented |
| ARA* weight | 1.0/1.25/1.5/2.0 | Used when `enable_ara` is enabled | Implemented |
| Difficulty | Easy/Medium/Hard/Expert | UI message only | Partial |
| Presets | Debugging/Fast Approx/Optimal/Speedrun/... | Batch-updates selected flags | Implemented |
| Solver | 13 algorithms | Controls subprocess dispatch algorithm index | Implemented |
| Search Space | Hybrid/Tile/Graph | Sets representation for game-state solver wrappers | Implemented |
| Apply Threshold | 0.70-0.90 | Used by topology match apply actions | Implemented |

### 5.3 Buttons

| Button | Behavior | Status |
|---|---|---|
| Start Auto-Solve | Launches background solver process | Implemented |
| Stop | Stops auto animation and clears path view state | Implemented |
| Generate Dungeon | Procedural generator (non-AI) | Implemented |
| AI Generate | Background neural pipeline generation | Implemented |
| Reset | Reloads current map and resets state | Implemented |
| Path Preview | Opens preview or triggers solve then preview | Implemented |
| Clear Path | Clears current planned/animated path | Implemented |
| Export Route | Saves JSON route to repo export folder | Implemented |
| Load Route | Loads latest route JSON from repo export folder | Implemented |
| Export Topology | Exports DOT topology file to repo export folder | Implemented |
| Compare Solvers | Runs async comparison and overlay | Implemented |
| Match Missing Nodes | Infers graph-room mappings | Implemented |
| Apply Tentative Matches | Applies staged mappings above threshold | Implemented |
| Undo Last Match | Restores previous room-node mapping snapshot | Implemented |
| Undo Prune | Restores prune snapshot after auto-prune | Implemented |
| Run MAP-Elites | Runs evaluator thread and stores archive/heatmap | Implemented |

### 5.4 User Hotkeys

Primary controls:
- `SPACE`: start auto-solve
- `R`: reset map
- `N` / `P`: next/previous map
- `H`: toggle heatmap
- `M`: toggle minimap
- `T`: toggle topology overlay
- `TAB`: toggle control panel collapse
- `F1`: help overlay
- `F11`: fullscreen
- `+` / `-` / mouse wheel: zoom
- `0`: reset zoom
- `C`: center on player
- `[` `]` or `,` `.`: speed down/up
- Arrow keys: manual movement

Debug/diagnostic keys also exist (`F7`, `F8`, `F12`, `Shift+F12`, `Ctrl+O`, etc.).

## 6. Exports and Artifacts (Repository-Local)

All GUI exports are written inside repo-local `exports/`:

- Routes: `exports/routes`
- Topology DOT: `exports/topology`
- MAP-Elites artifacts: `exports/artifacts`

Route export includes:
- start/goal
- path
- algorithm metadata
- solve time and explored nodes

## 7. Best-Practice Usage

1. Use `Solver=A*` + `Search Space=Hybrid` as baseline for solvability validation.
2. Enable `priority_tie_break` and `priority_key_boost` for lock-heavy dungeons.
3. Enable `enable_ara` and tune `ARA* weight` only when speed is more important than strict optimality.
4. Use CBS personas when you want human-like behavior metrics, not shortest-path optimality.
5. Treat `parallel_search`, `multi_goal`, and `dynamic_difficulty` as experimental/scaffolded until fully wired.

## 8. Notes on Legacy Docs

`docs/ZELDA_SOLVER_DOCUMENTATION.md` contains historical material.
Use this file as the canonical runtime reference for current code paths.
