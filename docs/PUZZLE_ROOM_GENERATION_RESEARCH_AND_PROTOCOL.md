# Puzzle Room Generation Research And Protocol

Updated: 2026-07-11

## Decision

Puzzle rooms are stateful planning problems. A room is not a valid puzzle
because it contains blocks, a gated door, or a visually interesting route. It
must have an executable sequence of legal state transitions from the entry to
the controlled exit.

The production architecture therefore uses this contract:

1. The mission graph specifies a gate family and required room semantics.
2. The room generator proposes geometry and semantic markers.
3. The constructive scaffold supplies a small set of graph-conditioned
   alternatives when the generated geometry is weak.
4. A room-local plan records the ordered interaction anchors and controlled
   puzzle doors.
5. After stitching, local anchors are globalized and the exact tile-state
   oracle executes the staged plan on the final artifact.

The final oracle, not a local proxy score or a neural loss, decides whether a
puzzle room is playable.

## Evidence Base

- Goldspinner separates static level scoring from simulation because visually
  plausible puzzle levels can still be dynamically unsolvable. This supports
  retaining an exact solver in the acceptance path:
  [Williams-King et al., 2012](https://ojs.aaai.org/index.php/AIIDE/article/view/12529).
- Data-Driven Sokoban Puzzle Generation constructs candidates through
  simulated play, guaranteeing solvability, and calibrates difficulty against
  player data. This supports solution-aware generation and an explicit
  distinction between solvability and perceived difficulty:
  [Kartal et al., 2016](https://ojs.aaai.org/index.php/AIIDE/article/view/12859).
- Tanagra combines planning with constraints so local geometry remains
  playable under designer changes. This supports the repository's
  graph-conditioned constructive scaffold, but not accepting a layout from
  geometry heuristics alone:
  [Smith, Whitehead, and Mateas, 2010](https://ojs.aaai.org/index.php/AIIDE/article/view/12379).
- Designer-authored action constraints can be compiled through graph grammars
  into target level content. This supports treating the mission graph as an
  executable interaction specification rather than only a visual label:
  [Linden, Lopes, and Bidarra, 2013](https://ojs.aaai.org/index.php/AIIDE/article/view/12592).
- Space-Time WFC demonstrates that including solutions can teach local game
  mechanics, but also reports slow generation and remaining global-constraint
  failures. It is therefore a useful solution-aware ablation, not a reason to
  replace the exact final oracle with local WFC compatibility:
  [Facey and Cooper, 2024](https://ojs.aaai.org/index.php/AIIDE/article/view/31863).
- Learned iterative repair can generate playable Zelda and Sokoban maps from
  small data, but it is a repair-style generator. It is an appropriate
  ablation or proposal mechanism, not evidence that a repaired output was
  natively produced by the neural model:
  [Siper, Khalifa, and Togelius, 2022](https://arxiv.org/abs/2202.10184).

## Implemented Contract Corrections

- Final assembly now passes globalized `room_puzzle_metadata` to the hard
  tile-state validator. Previously, the assembler created this metadata but
  discarded it before final validation, allowing staged puzzle doors to be
  evaluated with fallback semantics.
- The experimental advanced pipeline now compiles the same final-artifact
  metadata before it invokes its tile-state oracle.
- Room topology traces and training metadata prefer observed semantic marker
  positions in the actual grid over heuristic anchor guesses. Metadata records
  whether each stage anchor is observed or synthetic.
- The local planner state key includes completed puzzle stages. States with
  different switch or puzzle progress can no longer be merged as equivalent.
- Switch and toggle scaffolds now require a witnessed one-push move whose
  destination is the recorded puzzle marker. The reserved route represents
  the temporal interaction correctly: the initial block cell is not marked as
  permanently traversable, while the player-side staging cell and target are.
  A generic nearby pushable block no longer satisfies the scaffold contract.
- Room-local puzzle tracing uses the same state transition semantics as the
  validator. In particular, `push_block_to_switch` is complete only after a
  block reaches the marker, not when the player walks onto it. The local
  planner uses admissible A* rather than breadth-first enumeration of every
  movable-block configuration, and it stops the trace at the first failed
  prerequisite instead of fabricating later-stage paths.
- Structure-stripping augmentation is disabled by default. Removing block
  tiles while retaining the original ordered stage plan creates contradictory
  supervision. It can only be re-enabled after a counterfactual compiler
  rewrites the grid, plan, controlled doors, and solver proof together.

## What The Current System Does Not Yet Prove

- The optional learned stage-token and semantic-head path is implemented but
  has not been retrained on grounded puzzle labels.
- Scaffold contract and sequence metrics are local geometry proxies. They are
  diagnostic signals, not substitutes for the full final oracle.
- P-CBS measures a configured bounded-search model. It is not a calibrated
  human-subject difficulty model.
- A repair-assisted valid room is not evidence of standalone neural puzzle
  generation. Report pre-repair and post-repair rates separately.

## Required Ablations

Use fixed mission graphs and identical random seeds for each row. Report exact
stateful-oracle validity, not only tile cleanliness or LogicNet scores.

| ID | Variant | Purpose |
| --- | --- | --- |
| P0 | Graph markers only, no scaffold | Baseline for raw neural geometry. |
| P1 | Markers plus heuristic topology trace | Tests non-stateful topology guidance. |
| P2 | Constructive scaffold plus global puzzle-plan oracle | Tests symbolic plan compilation and validation. |
| P3 | P2 plus stage tokens and auxiliary semantic head | Tests learned ordered puzzle conditioning after retraining. |
| P4 | Solver-validated counterfactual puzzle-on/off data | Optional control ablation; do not run until the counterfactual compiler exists. |

For every variant, report:

- raw neural exact-plan validity before repair
- post-repair exact-plan validity
- stage-anchor grounding rate
- controlled-door completion rate
- solution length, state expansions, pushes, item pickups, and revisits
- puzzle family distribution and structural novelty
- repair rate and number of forced symbolic marker writes
- stage-trace completion, failed-stage index, and local planning state budget

Hyperparameter selection must not reward raw plan/stage counts or repair rate.
Those terms are directly gameable. The packaged stateful sweep ranks exact
oracle validity first, uses normalized contract/interaction/sequence success
rates, and treats repair dependence as a penalty. Plan and stage counts remain
descriptive outputs only.

## Acceptance Rule

A generated dungeon containing a `DOOR_PUZZLE` is accepted only when the
globalized puzzle metadata is supplied to `ZeldaValidator` and the exact
tile-state search reaches the goal without exhausting its state budget. A
budget-exhausted result is indeterminate, not solvable.
