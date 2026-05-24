# P-CBS Research And Publication Gap Audit

Last updated: 2026-05-24

## Research Position

The defensible role for `P-CBS` is a bounded-rational persona validator, not a
replacement for the hard oracle. This matches the PCG literature pattern:
solver/oracle metrics establish mechanical validity, while persona or
experience models explain how different players may experience valid content.

Relevant literature:

- Liapis et al., "Procedural Personas as Critics for Dungeon Generation"
  (<https://www.um.edu.mt/library/oar/handle/123456789/29721>)
- Yannakakis and Togelius, "Experience-Driven Procedural Content Generation"
  (<https://pure.itu.dk/en/publications/experience-driven-procedural-content-generation/>)
- Summerville et al., "Procedural Content Generation via Machine Learning"
  (<https://arxiv.org/abs/1702.00539>)
- Khalifa et al., "The Procedural Content Generation Benchmark"
  (<https://arxiv.org/abs/2503.21474>)

## Current Code Evidence

`src/simulation/cognitive_bounded_search.py` already contains the main pieces
needed for a publishable bounded-persona contribution:

- persona-specific utility weights
- partial-observation belief map
- finite working memory with decay
- decision pressure, deliberation budget, and frustration
- progression-affordance memory reactivated by inventory changes
- short-term focus persistence and focus switching

`src/evaluation/pcbs_validation.py` now keeps P-CBS separated from the hard
oracle and reports:

- `bounded_rationality_index`
- `readability_score`
- `cognitive_effort_index`
- `oracle_pcbs_path_delta`
- `pcbs_outcome_class`
- `pcbs_calibration_bucket`
- `pcbs_failure_driver`
- `pcbs_dominant_pressure`

`scripts/run_pcbs_component_ablation.py` now exports those fields for component
ablations, so the paper can show whether revisit penalty, uncertainty,
deliberation, affordance memory, and focus actually change behavior.

## Publication Claim Boundary

Defensible:

`P-CBS` is a repo-novel bounded-rational dungeon playtesting algorithm that
combines persona-conditioned utility, memory decay, partial observability,
deliberation budget, affordance recall, and focus persistence. It complements
`A*`/graph-progression/softlock validation by measuring readability and
bounded-rational effort.

Not defensible yet:

`P-CBS` should not be claimed as a new universally dominant search family or as
a replacement for exact reachability. It must remain a behavioral critic unless
validated against human traces.

## Missing Evidence Before Submission

- Full `A*` vs `P-CBS` table on the final generated branch with pre-repair and
  post-repair results separated.
- Matched-budget P-CBS component ablation across all relevant personas.
- Human or proxy calibration: even a small hand-labeled readability set would
  make the bounded-rational metrics much stronger.
- Statistical tests over paired seeds, not only aggregate means.
- Failure-driver table showing whether bounded failures are caused by state
  budget, confusion, puzzle stalls, affordance reactivation, or focus switching.

## Implemented In This Pass

- Added outcome classification and dominant-pressure reporting in
  `src/evaluation/pcbs_validation.py`.
- Added P-CBS outcome/failure fields to the conditioning/LogicNet/repair
  ablation rows.
- Added P-CBS outcome/failure fields to component-ablation CSV and Markdown
  summaries.
- Added quick-mode timeout caps to the component ablation script so smoke runs
  finish quickly.

