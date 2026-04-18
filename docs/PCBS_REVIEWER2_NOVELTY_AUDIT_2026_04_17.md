# P-CBS Reviewer-2 Novelty Audit

Last updated: 2026-04-17

## Scope

This note is the thesis-facing `Reviewer 2` audit for
`Persona-Driven Cognitive Bounded Search (P-CBS)`.

It answers four questions:

1. what the closest prior art already did
2. what this repo's current `P-CBS` still does not justify claiming
3. what had to be changed in code to strengthen the claim
4. which metric table is the most convincing defense in the thesis

## Bottom-Line Verdict

`P-CBS` is **not** defensibly a brand-new universal search family on the level of
`A*`, `D* Lite`, or MAPF `Conflict-Based Search`.

It is defensibly a **moderately novel bounded-rational persona validator for
dungeon PCG**, especially when framed as a system contribution:

- persona-conditioned playtesting
- explicit cognitive penalties for revisitation and conditional uncertainty
- progression-affordance memory with inventory-triggered recall
- partial observability and bounded memory
- integration into the procedural-generation loop as a critic

The strongest safe claim is:

`P-CBS extends procedural-persona playtesting with explicit bounded cognitive
state and metacognitive resource control, allowing mechanically solvable levels
to be separated from cognitively frustrating ones.`

Do **not** claim:

- "P-CBS replaces the hard oracle"
- "P-CBS is the first human-like playtesting agent of any kind"
- "No prior work penalizes uncertainty or revisitation at all"

The first two are false. The third is too strong given adjacent literature on
human-like pathfinding and cognitive navigation.

## Closest Prior Art

### 1. Holmgard et al. procedural personas

Closest source:

- [Automated Playtesting with Procedural Personas through MCTS with Evolved Heuristics](https://antoniosliapis.com/papers/automated_playtesting_with_procedural_personas_through_mcts_with_evolved_heuristics.pdf)

What it already establishes:

- automated playtesting with archetypal player models
- persona-specific utility functions
- direct use inside game-content evaluation
- bounded play through computational limits in MCTS

Why this matters:

If the thesis says "we use personas with a utility function to test levels",
Holmgard already got there first.

What `P-CBS` adds beyond that:

- explicit local penalties for revisitation, puzzle complexity, and conditional
  uncertainty in a grid/inventory dungeon validator
- episodic memory over unresolved progression affordances such as locked doors,
  bomb walls, and puzzle anchors, with reactivation after inventory changes
- belief-map and working-memory state instead of only persona-biased tree policy
- direct use as a validator alongside a mechanical oracle in a neuro-symbolic
  PCG stack

Reviewer-2 judgment:

- `persona-driven playtesting`: not novel
- `bounded cognitive penalties inside the validator`: closer to novel
- `full system integration into H-MOLQD`: novel enough to report

### 2. Liapis / Holmgard persona critics and spatial metrics

Closest sources:

- [Procedural Personas as Critics for Dungeon Generation](https://antoniosliapis.com/papers/procedural_personas_as_critics_for_dungeon_generation.pdf)
- [Towards a Generic Method of Evaluating Game Levels](https://yannakakis.net/wp-content/uploads/2013/08/towards_a_generic_method_of_evaluating_game_levels.pdf)

What they already establish:

- simulation-based critics inside PCG
- levels can be optimized around persona behavior
- generic spatial metrics such as exploration/balance can drive generation

Why this matters:

If the thesis says "our persona agent feeds back into PCG evaluation", that is
also not new by itself.

What `P-CBS` adds:

- a richer cognitive-state model than the older critic formulations
- a clearer split between `mechanical solvability` and `experiential
  readability`
- cognitive metrics such as confusion, entropy, load, replans, and aha-latency

Reviewer-2 judgment:

The PCG-loop integration is not the novelty by itself. The novelty has to be
the **specific form** of cognitively bounded validation and the metrics it
produces.

### 3. PCGRL and dense reward shaping

Closest source:

- [PCGRL: Procedural Content Generation via Reinforcement Learning](https://arxiv.org/abs/2001.09212)

What it already establishes:

- iterative level construction as an MDP
- dense shaping toward computable design goals
- solver-backed Zelda/Sokoban playability constraints

Why this matters:

A reviewer can say your scalar utility is just another shaped reward.

Why that criticism is only partly right:

- `PCGRL` shapes a **generator**
- `P-CBS` shapes a **validator / synthetic playtester**
- `PCGRL` optimizes design actions
- `P-CBS` models bounded dungeon navigation under partial observability

Reviewer-2 judgment:

`P-CBS` is not just a rebranding of `PCGRL`, but the general idea of
hand-designed weighted objectives is not new either.

### 4. Multi-objective A* / scalarized search

Closest source:

- [A Practical Guide to Multi-Objective Reinforcement Learning and Planning](https://arxiv.org/abs/2103.09568)

Relevant point:

The guide explicitly notes that many multi-objective systems use a simple linear
combination of objectives, and that this can oversimplify the decision problem.

Why this matters:

The current `P-CBS` move utility is a scalarized weighted sum. That makes it
vulnerable to the criticism that it is only a handcrafted scalarization over
multiple terms.

But `P-CBS` is also **not** standard `MOA*`:

- it does not compute Pareto fronts
- it does not maintain nondominated frontier labels
- it is path-dependent through cognitive state
- it is not an optimal multiobjective planner

Reviewer-2 judgment:

Do not describe `P-CBS` as "a new multi-objective search algorithm". That claim
would be mathematically wrong.

## Where The Consensus Draft Is Strong

The supplied review is strongest on this claim:

`No exact prior art was identified that combines persona-driven automated
playtesting, explicit revisitation penalty, explicit conditional puzzle
uncertainty penalty, and dungeon-validator integration in one heuristic-search
controller.`

That is a plausible and defensible synthesis claim.

## Where The Consensus Draft Overclaims

The draft is too strong if it implies:

- nobody has penalized revisitation anywhere
- nobody has modeled uncertainty or memory in pathfinding
- no related algorithm penalizes puzzle complexity or confusion-like behavior

Adjacent prior art exists:

- [Towards a human-like approach to path finding](https://doi.org/10.1016/j.cag.2021.08.020)
- [Cognitive Path Planning With Spatial Memory Distortion](https://doi.org/10.1109/TVCG.2022.3163794)
- [The Effects of Human-like Modifications to Heuristic Action Evaluation in Video Game Pathfinding](https://doi.org/10.1145/3555858.3555888)

So the safe phrasing is:

`no exact integrated prior art`

not:

`no related prior art`

## Code-Side Gap That Had To Be Fixed

The repo had a real implementation weakness:

- `solve()` reset the environment but **did not reset** the belief map or
  working memory

That meant solver reuse could leak knowledge across runs. This is now fixed in
[cognitive_bounded_search.py](f:/KLTN/src/simulation/cognitive_bounded_search.py).

That fix matters for the thesis because any repeated benchmark using one solver
instance would otherwise contaminate the cognitive metrics.

## What Was Added To Strengthen The Novelty Claim

The core change is that `P-CBS` is no longer only a static weighted move scorer.

New implementation elements:

- explicit `deliberation_budget`
- `decision_pressure` estimate per decision point
- budget-gated short-horizon lookahead
- path-dependent `frustration` state
- progression-affordance memory with inventory-triggered reactivation
- benchmark outputs for:
  - `deliberation_events`
  - `budget_exhaustion_events`
  - `peak_frustration`
  - `final_deliberation_budget`
  - `affordance_reactivations`
  - `affordance_guided_steps`
  - `inventory_change_events`

Why this matters:

This moves the implementation closer to a **resource-rational validator** with
metacognitive control, which is more defensible than a plain weighted utility.

The new benchmark summary also exposes:

- `success_rate_given_oracle_solved`
- `cognitive_gap_rate_given_oracle_solved`

That second metric is the most convincing one for the thesis.

## Single Most Convincing Metric

Use:

`CognitiveGapRate = P(P-CBS fails | hard oracle solves)`

Interpretation:

- if it is near `0`, your persona agent behaves too much like the oracle
- if it is high on confusing maps and low on readable maps, then `P-CBS`
  captures bounded-rational difficulty rather than merely being a weak solver

This must be paired with cognitive diagnostics:

- `ConfusionIndex`
- `NavigationEntropy`
- `CognitiveLoad`
- `AhaLatency`
- `PeakFrustration`
- `BudgetExhaustionEvents`

## Report-Ready Table

| Table | Rows | Columns | Purpose |
|---|---|---|---|
| T1 | `A*`, `P-CBS balanced`, `P-CBS novice`, `P-CBS explorer`, `P-CBS cautious` | solvability, time, path length, oracle-conditioned success | mechanical vs bounded-rational performance |
| T2 | same personas | confusion index, navigation entropy, cognitive load, peak frustration, budget exhaustion | prove that the personas diverge in cognitively meaningful ways |
| T3 | map buckets: readable / moderate / confusing | oracle success, P-CBS success, cognitive gap, mean load | show that P-CBS fails selectively where humans plausibly struggle |
| T4 | ablations: no revisit penalty / no uncertainty penalty / no deliberation budget / full P-CBS | cognitive gap, confusion, load, success | isolate the contribution of each bounded-rational term |
| T5 | ablations: no affordance memory / full P-CBS | path length, confusion, budget exhaustion, frustration, aha-latency | prove the Zelda-specific progression-memory term matters beyond generic weighted search |

## Thesis-Safe Claim

Use this wording:

`P-CBS is introduced as a persona-driven bounded-rational validator for
procedural dungeon generation. Unlike exact playability solvers, P-CBS augments
dungeon traversal with explicit cognitive-state variables including bounded
working memory, revisitation aversion, conditional uncertainty, progression-affordance
memory, and a finite deliberation budget. This allows the evaluation pipeline to
distinguish mechanically solvable content from content that is likely to induce
confusion or frustration for different player archetypes.`

## Thesis-Unsafe Claim

Avoid this wording:

`P-CBS is a fundamentally new general search algorithm that replaces A*.`

That is not supported.

## Current Repo Status

The repo is now in a better position for the thesis:

- solver-state leakage fixed
- cognitive metrics extended
- benchmark summary exposes the right novelty-defense metric
- the novelty claim is stronger than before but still should be framed as a
  `bounded-rational persona validator`, not a universal planning breakthrough

## References

- [Automated Playtesting with Procedural Personas through MCTS with Evolved Heuristics](https://antoniosliapis.com/papers/automated_playtesting_with_procedural_personas_through_mcts_with_evolved_heuristics.pdf)
- [Procedural Personas as Critics for Dungeon Generation](https://antoniosliapis.com/papers/procedural_personas_as_critics_for_dungeon_generation.pdf)
- [Towards a Generic Method of Evaluating Game Levels](https://yannakakis.net/wp-content/uploads/2013/08/towards_a_generic_method_of_evaluating_game_levels.pdf)
- [PCGRL: Procedural Content Generation via Reinforcement Learning](https://arxiv.org/abs/2001.09212)
- [A Practical Guide to Multi-Objective Reinforcement Learning and Planning](https://arxiv.org/abs/2103.09568)
- [Rational deployment of multiple heuristics in optimal state-space search](https://doi.org/10.1016/j.artint.2017.11.001)
- [Towards a human-like approach to path finding](https://doi.org/10.1016/j.cag.2021.08.020)
- [Cognitive Path Planning With Spatial Memory Distortion](https://doi.org/10.1109/TVCG.2022.3163794)
- [The Effects of Human-like Modifications to Heuristic Action Evaluation in Video Game Pathfinding](https://doi.org/10.1145/3555858.3555888)
