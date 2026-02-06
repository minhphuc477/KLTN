# Work items for pr/deterministic-mapping-normalizer

- [ ] Audit current `_normalize_graph()` and ensure all edge labels are normalized to `edge_type` values.
- [ ] Add tests for label parsing robustness (e.g., '3_4', '(3,4)', 'r3c4') in `infer_missing_mappings`.
- [ ] Add deterministic sorting in all matching loops (use node signature and degree tie-breaks).
- [ ] Add `tests/test_mapping_deterministic_longrun.py` that runs 100 permutations.
- [ ] Add logging for mapping validation and a debug-mode to print mismatch diagnostics.
- [ ] Run full test-suite and address any failures.
