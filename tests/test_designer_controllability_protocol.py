from scripts.run_designer_controllability_proof import (
    _target_error_fields,
    build_target_response_rows,
    build_target_suite,
    method_list,
    summarize_rows,
)


def test_target_suite_contains_proxy_and_raw_count_controls():
    specs = build_target_suite()
    names = {spec.name for spec in specs}
    assert "p_balanced_keylock" in names
    assert "p_large_stress_100" in names
    assert "p_large_stress_500" in names

    balanced = next(spec for spec in specs if spec.name == "p_balanced_keylock")
    assert "linearity" in balanced.descriptor_targets
    assert "gating_density" in balanced.descriptor_targets
    assert "key_count" in balanced.descriptor_targets
    assert "lock_count" in balanced.descriptor_targets
    assert "key_count" in balanced.evaluation_targets
    assert "lock_count" in balanced.evaluation_targets
    assert balanced.min_rooms < balanced.max_rooms


def test_target_error_fields_uses_relative_error_for_counts():
    spec = next(spec for spec in build_target_suite() if spec.name == "p_balanced_keylock")
    actual = {
        "num_nodes": spec.merged_targets()["num_nodes"],
        "key_count": spec.merged_targets()["key_count"] + 1,
        "lock_count": spec.merged_targets()["lock_count"],
        "linearity": spec.merged_targets()["linearity"],
    }
    fields = _target_error_fields(actual, spec)
    assert fields["controlled_metric_count"] >= 4
    assert fields["norm_error_key_count"] == 1.0 / spec.merged_targets()["key_count"]
    assert "pass_linearity" in fields


def test_method_list_rejects_unknown_methods():
    assert [method.name for method in method_list("FULL_GA,CORE_GA")] == ["FULL_GA", "CORE_GA"]
    try:
        method_list("missing")
    except ValueError as exc:
        assert "Unknown method" in str(exc)
    else:
        raise AssertionError("method_list should reject unknown methods")


def test_target_response_rows_capture_monotonic_axes():
    spec_low = next(spec for spec in build_target_suite() if spec.name == "axis_size_12")
    spec_high = next(spec for spec in build_target_suite() if spec.name == "axis_size_24")
    rows = []
    for spec, actual_nodes in ((spec_low, 11), (spec_high, 23)):
        actual = {
            "num_nodes": actual_nodes,
            "num_edges": spec.merged_targets()["num_edges"],
            "path_length": spec.merged_targets()["path_length"],
        }
        row = {
            "target_family": spec.family,
            "target_name": spec.name,
            "method": "FULL_GA",
        }
        row.update(_target_error_fields(actual, spec))
        rows.append(row)

    summary = summarize_rows(rows)
    target_response = build_target_response_rows(summary)

    size_rows = [row for row in target_response if row["target_family"] == "axis_size"]
    assert size_rows
    assert all("target_mean" in row and "actual_mean" in row for row in size_rows)
