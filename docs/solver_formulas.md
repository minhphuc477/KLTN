# Solver Upgrade — Compact Formula Sheet

This sheet summarizes the core formulas used in the upgrade:
- Reverse reachability set update for deterministic soft-lock detection
- Macro-action construction and cost
- Plan-guided heuristic used in tile-level A*

## Notation
- State: $s = (p, R, I, B)$ where:
  - $p$ = agent position (tile)
  - $R$ = resource vector (counts of bombs, keys, etc.)
  - $I$ = inventory/flags (bits for collected items / switches)
  - $B$ = binary map of door/wall open/closed or other environment bits
- Action: $a$ (deterministic); forward transition $T(s,a) = s'$
- Goal condition: $G(s)$ true when $s$ satisfies reaching triforce/goal
- POI = Point‑Of‑Interest (door, stair, warp, key tile, item, goal)

## 1) Reverse Reachability (Deterministic Soft‑lock Detection)

Define the backward‑reachable set $\mathcal{R}$ as the least fixed point starting from goal states.

Initialization:
$$
\mathcal{R}^{(0)} = \{ s \mid G(s) \}
$$

Iterative update (one backward BFS step):
$$
\mathcal{R}^{(k+1)} = \mathcal{R}^{(k)} \;\cup\; 
\{ s \;|\; \exists a:\; T(s,a) \in \mathcal{R}^{(k)} \}\n
$$

Important resource inversion rule: for any forward action $a$ that consumes $c$ units of a resource (e.g., bombs), the backward check requires that predecessor $s$ has resource counts equal to successor plus the consumed amount. Formally, if forward: $R' = R - c$, then when testing reverse transitions treat the predecessor as having $R = R' + c$.

Termination: iterate until $\mathcal{R}^{(k+1)} = \mathcal{R}^{(k)}$. A forward-expanded state $s$ can be safely pruned if $s \not\in \mathcal{R}$ (no deterministic sequence of actions leads to the goal).

Remarks:
- Correctness requires forward model determinism and exact modeling of resource increments/consumption.
- Time/space complexity: proportional to reachable state-space size under reverse transitions.

## 2) Macro‑Action A* (POI‑based Abstraction)

POI extraction per room yields a finite set $\mathcal{P}$ of POIs. Precompute intra-room shortest paths and distances:

- For each room $r$, for POIs $i,j\in\mathcal{P}_r$ compute
  $$d_r(i,j) = \text{BFS\tnormalfont-}\mathrm{dist}\big(\text{tile}(i),\text{tile}(j)\big)$$
  and concrete path $P_r(i\to j)$ (sequence of tiles).

Abstract graph $\mathcal{G}_{abs}=(V_{abs},E_{abs})$: nodes = POIs (or room-level aggregated nodes); edges between POIs in same room or connected by warps/doors.

Abstract edge weight (cost):
$$
c_{abs}(i,j) = d_r(i,j) + \delta_{penalty}(i,j)
$$
where $\delta_{penalty}$ is an optional small cost to bias edge selection (risk, preference). If $d_r$ is exact BFS length, $c_{abs}$ is a lower bound of tile-level cost.

Run A* / Dijkstra on $\mathcal{G}_{abs}$ to get abstract plan $\pi = [p_0, p_1, \dots, p_m]$ from start POI to goal POI.

Macro-action definition: for adjacent POIs $p_k,p_{k+1}$ in $\pi$ define macro-action
$$
M_{k} = P_{r}(p_k \to p_{k+1})
$$
(the concrete tile sequence). Macro-action cost:
$$
\mathrm{cost}(M_{k}) = |P_{r}(p_k\to p_{k+1})| = d_r(p_k,p_{k+1})
$$

Execution semantics:
- Simulate executing $M_k$ tile‑by‑tile under the full forward model (updating $R,I,B$). If execution completes without invalidation, accept the macro expansion as a single logical step with cost equal to its length; if invalidated (unexpected block, resource mismatch), abort at failure tile and fall back to atomic expansions from that tile.

Benefits:
- Reduces branching factor by replacing many single‑tile expansions with a small set of macro jumps.

## 3) Plan‑Guided Heuristic $h(s)$

Let $s=(p,R,I,B)$ and suppose the abstract plan $\pi$ is available and the current nearest abstract node (POI or room) is $u$.

Precomputed abstract distance to goal: $D_{abs}(u) = $ shortest cost on $\mathcal{G}_{abs}$ from $u$ to goal POI.
Local tile distance to next POI on plan: $d_{local}(p, p_{next}) = $ BFS tile distance from $p$ to the next POI tile.

Exact admissible plan heuristic:
$$
h_{plan}(s) = D_{abs}(u) + d_{local}(p, p_{next})
$$
If both $D_{abs}$ and $d_{local}$ are exact shortest distances, $h_{plan}$ is admissible for tile‑level cost.

Combine with baseline heuristic $h_{base}$ (e.g., obstacle‑aware BFS, Manhattan lower bound):
$$
h(s) = \max\big(h_{base}(s),\; h_{plan}(s)\big)
$$
To trade admissibility for speed, a weighted variant may be used:
$$
h_{w}(s) = \alpha\,D_{abs}(u) + \beta\,d_{local}(p,p_{next}),\quad \alpha,\beta>0
$$
(note: $\alpha,\beta\neq1$ may violate admissibility).

## 4) Compact Pipeline (operational)

1. Preprocessing
   - Extract POIs and build per‑room BFS tables $d_r(\cdot,\cdot)$ and concrete paths $P_r(\cdot\to\cdot)$.
   - Build abstract graph $\mathcal{G}_{abs}$ and compute $D_{abs}(\cdot)$ (Dijkstra / A*).
   - Optionally compute a coarse reverse reachability on abstract nodes for quick pruning.
2. Abstract planning
   - Run A*/Dijkstra on $\mathcal{G}_{abs}$ to obtain $\pi$.
3. Tile search (StateSpaceAStar)
   - Use open list ordered by $f(s)=g(s)+h(s)$ where $h$ is plan‑guided.
   - When expanding a state aligned with $\pi$, try macro‑action expansion $M_k$ first (simulate; commit or fallback).
   - After each expansion, consult exact reverse reachability $\mathcal{R}$; prune states with $s\not\in\mathcal{R}$.
4. Fallbacks & verification
   - If macros repeatedly fail, disable macro attempts and continue with atomic expansions.
   - Verify solution by replaying action sequence under the forward model.

## 5) Practical Notes
- To keep $h$ admissible, ensure precomputed costs are exact shortest path lengths; do not add optimistic negative penalties.
- Reverse reachability must exactly invert forward resource semantics; any modeling error can give false negatives.
- Macro-actions are heuristics in that they speed search but require robust fallbacks for correctness.

---
File generated by engineering agent — concise reference for implementation and documentation.
