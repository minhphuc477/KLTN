# PR Draft: pr/deterministic-mapping-normalizer

## Title
Deterministic mapping & canonical graph normalizer

## Summary
- Harden `Data/zelda_core._normalize_graph()` and `RoomGraphMatcher` so node labels and edge types are canonical and mapping is deterministic.
- Add mapping validation & deterministic fallback assignment.
- Add tests: `test_mapping_deterministic`, `test_infer_missing_mappings_*`, and mapping consistency diagnostics.

## Files (proposed)
- `Data/zelda_core.py` — extend `_normalize_graph()` and add `_validate_mapping()` improvements
- `tests/test_room_node_mapping.py` — add deterministic and consistency tests

## Acceptance
- Running mapping 50× with permuted IDs produces identical mapping
- Mapping consistency ratio >= 0.8 for grid tests

## Notes
Keep changes minimal and well-commented; prefer unit tests over heuristic fixes.
