# Enemy/Key Modeling and Placeholder Audit (2026-02-24)

## Scope

This audit addresses two concerns:

1. Enemy/key semantics were mostly binary labels at graph-node level (`has_enemy`, `has_key`) with weak quantity control.
2. Need explicit verification for placeholder or empty runtime logic in `src/`.

## Data Research Summary

Generated from:
- `python scripts/audit_enemy_key_and_placeholders.py --pretty --output results/enemy_key_placeholder_audit_2026_02_24.json`

Block-0 VGLC data stats:
- Rooms parsed: `450`
- Graphs parsed: `18`
- Enemy tiles per room:
  - mean `1.0889`
  - p90 `4`
  - max `12`
  - nonzero rate `0.24`
- Key tiles per room:
  - mean `0.0` (keys are mostly graph-level semantics, not explicit room tiles in this corpus)
- Keys per graph:
  - mean `3.83`
  - p90 `6`
  - max `8`
- Enemy-labeled nodes per graph:
  - mean `19.28`
  - max `46`

Interpretation:
- Key progression is better represented at topology graph level than tile-level in this dataset.
- Enemy pressure is sparse across all rooms but concentrated in combat subsets.

## Best-Practice Changes Implemented

### 1) Count-aware node semantics

Added explicit quantity hints to mission nodes:
- `enemy_count_hint`
- `key_count_hint`

These are auto-normalized in graph construction:
- key-like nodes default to at least one key hint.
- enemy-like nodes get bounded hints based on node type and difficulty.
- defaults are now VGLC-aligned (regular combat mostly `1-2`, arena spikes up to `4`, boss `1-2`) with hard bounds (`enemy<=12`, `key<=4`).

Files:
- `src/generation/grammar.py`
- `src/generation/evolutionary_director.py`

### 2) Runtime propagation into generated graphs

Mission-to-NetworkX conversion now exports:
- `enemy_count_hint`, `key_count_hint`
- aliases: `enemy_count`, `key_count`
- booleans: `has_enemy`, `has_key`

NetworkX-to-Mission conversion now restores hint fields.

File:
- `src/generation/evolutionary_director.py`

### 3) Entity spawning uses quantity hints

Room semantics now include:
- `enemy_count_hint`
- `key_count_hint`

Spawner behavior:
- combat rooms respect `enemy_count_hint` when provided (bounded by spatial limits).
- key spawning supports multiple keys per room via `key_count_hint`.

File:
- `src/generation/entity_spawner.py`

### 4) Evaluation uses counts (not only binary presence)

Graph descriptor extraction now reads count hints (`*_count_hint` / `*_count`) and accumulates weighted key/enemy counts.

File:
- `src/evaluation/benchmark_suite.py`

## Placeholder / Empty Logic Audit

Audit result summary (`src/`):
- `noop_constructor`: `2`
- `exception_guard`: `11`
- `expected_base_contract`: `5`
- `expected_abstract`: `2`
- `potential_empty_logic`: `0`

One previously empty method (`ExplainabilityDebugOverlay._render_confidence_overlay`) was replaced with explicit fallback rendering (status banner) so the behavior is no longer silent/pass-through.

File:
- `src/utils/explainability_gui.py`

## Conclusion

The codebase now supports count-aware enemy/key semantics in generation, conversion, spawning, and evaluation paths, and the placeholder audit currently reports no unresolved potential empty logic in `src/`.
