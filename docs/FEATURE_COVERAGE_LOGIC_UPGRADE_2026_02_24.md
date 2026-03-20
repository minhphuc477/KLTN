# Feature-Coverage Logic Upgrade (2026-02-24)

## Why this upgrade was needed

The previous optimization and conversion path focused heavily on `key/enemy` and generic gating.
VGLC Zelda graphs encode additional core mechanics that must be preserved and optimized:
- node labels: `p` (puzzle), `i/I` (items), `m` (mini-boss), etc.
- edge labels: `bombable`, `soft_locked`, `item_locked`, `switch`, `stair`, `key_locked`.

If these are collapsed to generic `PATH` or omitted from descriptor targets, topology quality drifts from source data and publication expectations.

## Data evidence (Block 0, VGLC)

From local audit over `18` DOT graphs (`531` nodes, `1167` edges):
- Node token counts: `e=329`, `p=215`, `i=79`, `k=69`, `I=24`, `m=40`.
- Edge token counts: `open=569`, `soft_locked=220`, `bombable=167`, `key_locked=151`, `item_locked=34`, `stair=24`, `switch=6`.

Artifact:
- `results/vglc_feature_distribution_2026_02_24.json`

This confirms puzzle/item and non-key gate semantics are first-class in the corpus and should not be treated as secondary noise.

## Publication rationale

- VGLC dataset paper: https://arxiv.org/abs/1606.07487
- Graph+GAN Zelda dungeon generation: https://arxiv.org/abs/2001.05065
- Zelda locked-door generation + player validation: https://www.sciencedirect.com/science/article/pii/S0957417421004504
- PCG Benchmark protocol (matched-budget comparison context): https://arxiv.org/abs/2503.21474

These works justify evaluating progression and structure beyond simple key/enemy binary signals.

## Implemented logic changes

### 1) Robust label-to-semantics conversion (critical fix)

File: `src/generation/evolutionary_director.py`

- Replaced brittle fallback node parsing with explicit token-based inference using `parse_node_label_tokens(...)`.
- Added composite-feature preservation during conversion:
  - enemy/key hints + puzzle/item hints (dynamic attrs for compatibility).
- Added explicit edge-constraint mapping from VGLC constraints into mission edge semantics:
  - `key_locked -> LOCKED` (+ fungible `requires_key_count=1` when needed)
  - `bombable/item_locked -> ITEM_GATE` (default `item_required=BOMB/ITEM`)
  - `soft_locked -> ONE_WAY`
  - `switch -> ON_OFF_GATE`
  - `stair -> STAIRS`
- Preserved raw VGLC constraint tokens in edge metadata (`metadata['vglc_constraints']`) for downstream metrics.

### 2) Feature-aware descriptors in benchmark extraction

File: `src/evaluation/benchmark_suite.py`

- Extended `GraphDescriptor` to track non-key/non-enemy features:
  - `puzzle_count`, `item_count`
  - `bombable_count`, `softlock_count`, `item_gate_count`, `switch_count`, `stair_count`
  - `gate_variety`
- Updated `extract_graph_descriptor(...)` formulas to include feature-complexity components instead of lock/key-only pressure.
- Extended summary and gap scoring so calibration can target these features.

### 3) Feature-aware evolutionary targeting

File: `src/generation/evolutionary_director.py`

- `TensionCurveEvaluator` now models and scores alignment for:
  - `puzzle_density`, `item_density`
  - `bombable_ratio`, `soft_lock_ratio`, `switch_ratio`, `stair_ratio`
  - `gate_variety`
- Descriptor fitness now mixes classic topology/progression objectives with this feature-coverage score.

### 4) Calibration policy extended to feature groups

File: `src/evaluation/benchmark_suite.py`

- Added rule groups for feature balancing:
  - `puzzle_features`, `item_features`, `softlock_features`, `switch_features`, `stair_features`
- Calibration now reacts to feature-count errors, not only 4D descriptors.

## Validation status

- Compile check passed for modified modules.
- Tests passed:
  - `tests/test_evaluation_benchmark_suite.py`
  - `tests/test_simulation_map_elites_runner.py`
- Smoke benchmark ran successfully and now exports expanded feature descriptors.

## Practical impact

The pipeline no longer treats progression quality as mostly key/enemy/lock.
It now preserves and optimizes puzzle-item-gate semantics that are actually present in VGLC and expected by prior Zelda-PCG publications.
