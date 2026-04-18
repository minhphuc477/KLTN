# Playability Evaluation And CBS Research 2026-04-16

Last updated: 2026-04-16

Scope:

- how prior PCG papers evaluate `playability`
- whether `unlimited interaction` is the right design for this repo's CBS/CBD
- what claim is defensible for the current repo

## Short Answer

Do not collapse all validation into one unlimited-search CBS/CBD agent.

The research pattern is usually:

1. a `hard playability oracle`
2. one or more `behavioral or persona-based playtest agents`

Those roles are different.

For this repo, the correct split is still:

- hard oracle: `A* + graph progression validator + softlock checker`
- behavioral probe: `CognitiveBoundedSearch` / `CBS+`

If CBS/CBD is made `unlimited`, it becomes less human-like and starts to drift
toward a generic exhaustive planner. That may help as an oracle, but it weakens
the bounded-rational novelty claim.

## What Other Papers Usually Do

### 1. Solver-based hard playability checks

Several PCG papers use a solver or simulation-based metric as the hard
playability layer rather than a human-like agent.

- [PCGRL, AIIDE 2020](https://arxiv.org/abs/2001.09212)
  frames level generation around computable quality metrics and control signals.
  In practice, the Zelda-style setups in this line of work rely on explicit
  playability objectives rather than unconstrained “looks playable” judgments.
- [Towards Objective Metrics for Procedurally Generated Video Game Levels, 2022](https://arxiv.org/abs/2201.10334)
  explicitly uses `A*` behavior to define difficulty and diversity metrics,
  showing that solver traces are a valid evaluation basis when the goal is
  reproducible playability evidence.
- [The Procedural Content Generation Benchmark, FDG 2025](https://arxiv.org/abs/2503.21474)
  standardizes evaluation around `quality`, `diversity`, and `controllability`
  metrics, not one monolithic “human-like” agent.

Conclusion:

- prior work usually wants `deterministic, reproducible, cheap-enough`
  playability evidence for the hard pass/fail layer
- solver-based evidence is normal and defensible

### 2. Persona or human-like playtesting agents

Human-like playtesting agents do exist, but they are usually used as
`additional critics`, not as the only oracle.

- [Automated Playtesting with Procedural Personas through MCTS with Evolved Heuristics, 2018](https://arxiv.org/abs/1802.06881)
  uses procedural personas as synthetic playtesters to expose differences in
  level experience and style.
- [Playtesting: What is Beyond Personas, 2021/2022](https://arxiv.org/abs/2107.11965)
  argues that richer playtesting should explore different goals and alternative
  paths rather than reducing evaluation to a single fixed persona.
- [Generative Personas That Behave and Experience Like Humans, 2022](https://arxiv.org/abs/2209.00459)
  pushes further toward human-like behavior and experience modeling, again as a
  playtesting tool rather than the sole correctness oracle.

Conclusion:

- persona agents are valuable for `how` a dungeon is experienced
- they are not usually the only source of truth for `whether` a dungeon is
  mechanically solvable

## What This Means For Unlimited Interaction

### Unlimited interaction is good for the oracle layer

If the goal is “prove the dungeon is solvable if a sufficiently capable agent
tries long enough”, then more search budget is good.

That supports:

- larger `A*` / graph-progression budgets
- stronger inventory-state search
- explicit timeout accounting

### Unlimited interaction is bad for the bounded-rational persona layer

If the goal is “model how a human-like player behaves under memory,
uncertainty, and satisficing pressure”, then unlimited search is the wrong
bias.

It damages three things:

1. `cognitive realism`
   a human-like persona should forget, hesitate, and sometimes fail
2. `metric meaning`
   confusion, replans, and aha-latency matter because the agent is bounded
3. `novelty claim`
   an unlimited CBS/CBD stops being a clearly bounded-rational navigation model
   and starts looking like “another search procedure with custom heuristics”

## Defensible Claim For This Repo Right Now

The current repo can defensibly claim:

- `CognitiveBoundedSearch` is a repo-novel bounded-rational dungeon
  playtesting agent with persona controls and cognitive metrics
- it is useful as a `behavioral probe`
- it is integrated into export validation and comparison reports

The repo cannot yet defensibly claim:

- CBS/CBD is a new universally recognized search family like `A*`, `D* Lite`,
  or MAPF `CBS`
- CBS/CBD should replace the hard playability oracle

## How To Make The CBS/CBD Contribution Stronger

The most credible path is not “make it unlimited”.

The stronger path is:

1. keep `bounded CBS+` as the persona / behavioral layer
2. formalize its problem statement and bounds
3. compare it against classical search under matched budgets
4. use its traces to influence generation or repair

That last point is where the current repo can become more distinctive:

- validator trace -> topology-conditioned repair focus
- validator trace -> WFC forced initialization / protected corridor prior
- persona-conditioned difficulty / readability scoring

That is more novel than simply removing bounds.

## Recommended Validation Contract

Use this split in the paper and in code:

- `Mechanical solvability`
  - grid `A*`
  - graph progression validator
  - softlock checker
- `Human-like navigation quality`
  - `CBS+` personas
  - confusion / entropy / replans / aha-latency
- `Benchmark competitiveness`
  - matched-budget topology baselines
  - PCG Benchmark quality / diversity / controllability

## Bottom Line

If you want stronger evidence, increase the `oracle` budget.

If you want stronger novelty, improve the `bounded persona` model and integrate
its traces into repair/generation.

Do not merge those two goals into one unlimited CBS/CBD solver.
