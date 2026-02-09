
# Validation Pipeline — Upgraded Solver (Detailed)

This document provides a complete walkthrough of the validation pipeline for the upgraded Zelda solver. It contains the end-to-end flow, mathematical formulas, algorithms, pseudocode examples, tuning guidance, diagnostics to collect, and common pitfalls.

**Quick links**
- Solver & validator: [src/simulation/validator.py](src/simulation/validator.py)
- Dungeon adapter & stitcher: [src/data/zelda_core.py](src/data/zelda_core.py)
- Formula reference: [docs/solver_formulas.md](docs/solver_formulas.md)
- Short validation summary: [docs/validation.md](docs/validation.md)

## Overview and Objectives

- Purpose: deterministically verify solver correctness across the canonical dungeon dataset (two quests × nine dungeons), measure solver behavior (expanded states, runtime, macro and reverse-prune diagnostics), and provide actionable diagnostics to tune the solver for generated or edge-case dungeons.
- Guarantees: with exact precomputations and admissible heuristics the solver is sound — returned plans are legal when replayed on the forward model. Completeness depends on resource bounds and timeouts; with no timeout and exact model the search is complete.

## 1. Inputs and Preprocessing

### Map inputs
- Per-room tile maps and connectivity graphs (VGLC-processed) are loaded by `ZeldaDungeonAdapter`.
- Stitcher produces `sd` with:
  - `sd.global_grid`: full tile-level grid
  - `sd.graph`: room connectivity graph
  - `sd.room_positions`, `sd.room_to_node`, `sd.node_to_room`
  - `sd.start_global`, `sd.triforce_global` (start and goal tiles)

### POI extraction and per-room BFS
- POIs (Points of Interest) include doors, stairs, warps, keys, item tiles, and the goal tile. These are extracted per room and represented as tile coordinates plus room membership.
- For each room `r` and POI pair `(i, j)` run a BFS to compute:
  - concrete path `P_r(i→j)` (sequence of tiles)
  - distance `d_r(i,j) = |P_r(i→j)|`
- Store these in lookups for instant macro generation and exact local distances.

### Abstract graph construction
- Build `G_abs = (V_abs, E_abs)` where `V_abs` = POIs (optionally aggregated per room), and `E_abs` connects POIs that are intra-room or physically connected by warps/doors.
- Edge weight: `c_abs(i,j) = d_r(i,j) + δ_penalty(i,j)`; when `δ_penalty=0` and `d_r` exact, `c_abs` is a strict lower bound of tile-level cost.
- Precompute `D_abs(u)` for all `u` as shortest abstract distances to goal (Dijkstra/A*).

Why: precomputation gives exact lower bounds and enables macro-action generation and plan-guided heuristics.

## 2. Forward Model and State Representation

- State: `s = (p, R, I, B)` where
  - `p`: tile position
  - `R`: resource counts vector (bombs, arrows, etc.)
  - `I`: inventory/item bits (keys collected, items toggled)
  - `B`: environment bits (doors open/closed, switch states)
- Transition function `T(s, a) = s'` is deterministic and implemented by `ZeldaLogicEnv`.
- Bitset encoding: inventory and environment bits are packed into fixed-length bitsets (`GameStateBitset`) for fast hashing and dominance checks.

Important: the reverse reachability algorithm relies on exact forward semantics; ensure inventory/resource effects and deterministic teleports are implemented consistently.

## 3. Reverse Reachability — Deterministic Soft‑Lock Detection

### Purpose
- Detect states from which no deterministic action sequence can reach the goal given the forward semantics and resource constraints.

### Mathematical definition
- Let `G(s)` indicate goal states. Define backward reachable set `R` as the least fixed point:
  - `R^(0) = { s | G(s) }`
  - `R^(k+1) = R^(k) ∪ { s | ∃ a, T(s, a) ∈ R^(k) }`

Resource inversion: if forward action consumes `c` units (e.g., bombs), then when enumerating reverse predecessors of `s'` we treat predecessor resource `R_pre = R_post + c`.

### Practical algorithm (pseudocode)

```python
from collections import deque

def compute_reverse_reachability(env, resource_bounds):
    R = set()
    q = deque()
    # Seed with all goal states (coarse seeding can use POI-level goal states)
    for goal_state in env.enumerate_goal_states(resource_bounds):
        key = state_key(goal_state)
        R.add(key)
        q.append(goal_state)

    while q:
        s_prime = q.popleft()
        # enumerate valid inverse actions for s_prime
        for inv in env.inverse_actions(s_prime):
            s = apply_inverse(s_prime, inv)
            key = state_key(s)
            if key not in R:
                R.add(key)
                q.append(s)
    return R
```

### Optimizations
- Run a coarse abstract reverse reachability on `G_abs` first to prune whole POIs/rooms quickly.
- Bound resources (practical maximum bombs/keys) to reduce combinatorial explosion.
- Compress state space for reverse pass (coarsen `p` to room or POI) and refine later.

### Use during forward search
- During expansion, if successor `s` not in `R` (reverse set), prune it — a deterministic proof it cannot reach goal.

## 4. Macro‑Action A* (POI‑based Abstraction)

### Macro definition
- Given an abstract plan `π = [p0, p1, ..., pm]` over POIs, produce macros:
  - `M_k = P_r(p_k → p_{k+1})` (concrete tile path)
  - `cost(M_k) = |P_r(p_k, p_{k+1})|`

### Macro execution semantics
- When expanding a tile-level state `s` aligned with `p_k`, simulate `M_k` step by step under forward semantics:
  - For each tile in `P_r`, call `env.apply_move(s_sim, next_tile)` updating resources and bits.
  - If every step is valid, accept `s_sim` as the macro successor with `g' = g + cost(M_k)`.
  - If simulation fails at tile `t` (blocked or resource mismatch), abort and generate atomic successors from the failure tile.

### Pseudocode

```python
def attempt_macro(env, s, macro_path):
    s_sim = s.copy()
    for tile in macro_path:
        if not env.can_move(s_sim, tile):
            return False, s_sim  # failure at current tile
        s_sim = env.apply_move(s_sim, tile)
    return True, s_sim
```

### Heuristics and limits
- Only attempt macros when `s` is on-trajectory relative to `π` (e.g., nearest POI is `p_k`).
- Limit repeated macro attempts per state to avoid simulation overhead.

### Tradeoffs
- Pros: reduces branching factor and reaches distant goals faster.
- Cons: macro simulation cost and brittleness for dynamic events (e.g., consuming a resource earlier than expected changes reachability).

## 5. Plan‑Guided Heuristic `h(s)`

### Formula (exact admissible variant)
- `h_plan(s) = D_abs(u) + d_local(p, p_next)` where:
  - `u` is the current abstract node/POI nearest to `s`;
  - `D_abs(u)` is precomputed abstract distance to goal;
  - `d_local` is exact BFS tile distance from `p` to the next POI tile on `π`.
- Combined heuristic: `h(s) = max(h_base(s), h_plan(s))` to preserve conservative lower bounds.

### Weighted variant (non-admissible)
- `h_w(s) = α D_abs(u) + β d_local(p, p_next)` (α,β > 0). Use only when optimality not required.

### Implementation notes
- Precompute `D_abs(·)` and `d_local` tables per room for O(1) lookups.
- When macros are used, ensure `D_abs` includes macro costs to keep bounds consistent.

## 6. Dominance & Pareto Pruning

### Motivation
- Many states share same tile `p` but differ in resource vectors and inventory bits. Keep only non-dominated states per tile to reduce duplicates.

### Dominance relation (s1 dominates s2):
- `s1.position == s2.position`
- `s1.inventory_bits ⊇ s2.inventory_bits` (s1 has at least items of s2)
- `s1.resources >= s2.resources` component-wise (s1 has equal or more resources)
- `g(s1) ≤ g(s2)`

### Implementation
- Maintain a frontier of non-dominated representatives per `(position, env_bits)` bucket using compact integer tuples and bitsets for fast comparison.

## 7. Main Search Loop (Detailed)

1. Initialize: `open` with start state `s0`, `closed` empty, metrics counters zero.
2. Precompute `R` via reverse reachability.
3. While `open` not empty and not timed out:
   a. Pop `s` = argmin `f(s) = g(s) + h(s)`.
   b. If `env.is_goal(s)`: replay `trace` to verify and return success with diagnostics.
   c. If `s` aligns with abstract plan `π`, try macro(s):
       - If macro succeeds, process macro successor: dominance check, `R` membership, insert to open.
       - If macro fails, generate atomic successors from the failure tile.
   d. Else generate atomic successors (tile neighbors, interactions): for each succ:
       - Update `g`, compute `h`, dominance check, `R` membership; insert to open if passes.
4. If open exhausted or timeout: return failure with diagnostics.

Verification: on returning success, replay the returned action sequence deterministically in `ZeldaLogicEnv` to validate resource updates and final goal attainment.

## 8. Diagnostics and Metrics

Record per-run metrics:
- `status` (PASS/FAIL)
- `expanded_states` (total popped from open)
- `runtime_seconds`
- `macro_attempts`, `macro_successes`, `macro_failures`
- `reverse_pruned` (number of successors pruned by reverse reachability)
- `max_open_size`, `peak_memory` (optional)
- `plan_length` (tile steps)

Format: print a per-level table and optionally write a CSV under `results/` for later analysis.

## 9. Tuning Guidelines

- `timeout`: set high enough for full verification; for debugging, use smaller timeouts and inspect per-level diagnostics.
- Resource bounds for reverse pass: choose realistic maxima (e.g., bombs ≤ 10) to limit reverse state-space.
- `δ_penalty` on abstract edges: small positive values can bias safer routes; include these in `D_abs` if used by `h_plan` to avoid inconsistency.
- Macro attempt policy: attempt macros only when aligned and limit retries per state to reduce costly simulations.
- POI selection: include only meaningful POIs; too many POIs increase abstract graph cost and can reduce macro benefit.

## 10. Common Pitfalls and Edge Cases

- Incorrect reverse transition modeling -> false negatives (pruning valid forward states). Verify inversion logic with small exhaustive tests.
- Warps / virtual nodes produce many predecessors; use abstract reverse pass or coarsening to avoid explosion.
- Macros brittle if forward model triggers events mid-path (consumable pickup/usage earlier than abstract model expects); ensure robust fallback.
- Weighted heuristics speed search but sacrifice optimality and may hide logical issues — use admissible heuristics for validation.

## 11. Complexity and Guarantees

- Worst-case complexity is exponential in the size of the state space. The combination of macros, reverse pruning, and dominance pruning aims to reduce the effective branching factor.
- Soundness: returned solutions replayed on `ZeldaLogicEnv` ensure legality.
- Completeness: with exact models, no timeouts and full resource bounds, forward search is complete.

## 12. Reproducible Run Example (PowerShell)

```powershell
C:/path/to/.venv/Scripts/python.exe -c "
import sys,time,logging
sys.path.insert(0,'.')
logging.disable(logging.WARNING)
from src.data.zelda_core import ZeldaDungeonAdapter
from src.simulation.validator import ZeldaLogicEnv, StateSpaceAStar
adapter = ZeldaDungeonAdapter('Data/The Legend of Zelda')
for quest in (1,2):
  for i in range(1,10):
    sd = adapter.load_dungeon(i, variant=quest)
    sd = adapter.stitch_dungeon(sd)
    env = ZeldaLogicEnv(sd.global_grid, graph=sd.graph, room_to_node=sd.room_to_node, room_positions=sd.room_positions, node_to_room=sd.node_to_room)
    if sd.start_global: env.start_pos = sd.start_global
    if sd.triforce_global: env.goal_pos = sd.triforce_global
    env.state.position = env.start_pos or (0,0)
    solver = StateSpaceAStar(env, timeout=500000)
    t0=time.time(); res = solver.solve();
    success, plan, states, diag = res
    print(f"Q{quest}-D{i} {success} states={states} time={time.time()-t0:.3f} macros={diag.get('macro_stats')} reverse_pruned={diag.get('reverse_pruned')}")
"
```

## 13. Next steps and automation

- Add `scripts/run_validation.py` that runs a batch, writes CSV to `results/validation_<ts>.csv`, and uploads artifacts when needed.
- Add unit tests that verify reverse inversion rules on small handcrafted maps.
- Add a profiling mode to collect per-room expansion heatmaps for further tuning.

---
Generated by the engineering agent — full reference for the validation pipeline, implementation guidance, and tuning notes.

