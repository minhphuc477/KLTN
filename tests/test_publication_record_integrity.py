"""Synthetic failure injection for publication evidence; no experiment artifacts."""

from dataclasses import replace
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from src.publication.analysis import aggregate_publication_records, paired_exact_validity_by_fold
from src.publication.records import PublicationResultRecord, StageResult
from src.publication.stage_bridge import build_publication_result_record


def _generated():
    grid = np.ones((4, 4), dtype=np.int32)
    return SimpleNamespace(
        rooms={0: SimpleNamespace(raw_neural_grid=grid, neural_grid=grid, room_grid=grid)},
        stitched_layout=SimpleNamespace(slot_positions={0: (0, 0)}),
        physical_mission_graph=nx.empty_graph(1),
        dungeon_grid=grid,
        metrics={},
    )


def _record():
    return PublicationResultRecord(
        run_id="run", git_commit="deadbeef", config_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64, protocol_sha256="c" * 64,
        outer_fold="holdout_dungeon_1", train_seed=42, sampling_seed=100,
        mission_graph_id="graph_1", model_id="g_graph", inference_variant="unguided",
        generation_status="completed",
        stages={name: StageResult(
            stage=name, status="valid", exact_valid=True, grid_path=f"{name}.npy",
            grid_sha256="d" * 64, shape=(4, 4),
        ) for name in ("raw", "constrained", "repaired", "compiled")},
    )


@pytest.mark.parametrize("failure", ["exception", "timeout", "incomplete", "unknown"])
def test_raw_validator_failure_retains_completed_attempt_and_every_grid(tmp_path, failure):
    def validator(grid, stage):
        if stage == "raw":
            if failure == "exception":
                raise RuntimeError("injected validator failure")
            return {"is_exact": True, "solvable": False, "termination_status": failure}
        return {"is_exact": True, "solvable": True, "termination_status": "solved"}

    metadata = _record().to_dict()
    for name in ("record_id", "schema_version", "stages", "generation_status",
                 "repair_metrics", "timing", "error_type"):
        metadata.pop(name)
    record = build_publication_result_record(
        _generated(), validator=validator, output_dir=tmp_path, **metadata,
    )
    expected = "timeout" if failure == "timeout" else "validator_error"
    assert record.generation_status == "completed"
    assert record.stages["raw"].status == expected
    assert record.stages["raw"].exact_valid is None
    assert len(list((tmp_path / "grids").glob("*.npy"))) == 4
    assert PublicationResultRecord.from_dict(record.to_dict()) == record
    raw = next(row for row in aggregate_publication_records([record]) if row["stage"] == "raw")
    assert raw["attempted_records"] == 1
    assert raw["exact_valid_rate_all_attempts"] == 0.0


def test_aggregation_rejects_duplicate_attempts_even_with_different_run_names():
    record = _record()
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_publication_records([record, replace(record, run_id="rerun")])


def test_primary_contrast_does_not_pool_guidance_conditions():
    records = []
    for fold in (1, 2):
        for model in ("g_graph", "g_l_local_logic"):
            record = replace(_record(), outer_fold=f"holdout_dungeon_{fold}", model_id=model)
            records.append(record)
            records.append(replace(record, inference_variant="logic_guided"))
    result = paired_exact_validity_by_fold(records, baseline_model_id="g_graph",
                                          comparison_model_id="g_l_local_logic",
                                          bootstrap_samples=10)
    assert result["inference_variant"] == "unguided"
    assert result["paired_record_count"] == 2


@pytest.mark.parametrize("field", ["protocol_sha256", "dataset_manifest_sha256", "git_commit"])
def test_paired_analysis_rejects_mixed_provenance(field):
    records = []
    for fold in (1, 2):
        baseline = replace(_record(), outer_fold=f"holdout_dungeon_{fold}")
        comparison = replace(baseline, model_id="g_l_local_logic", **{field: "e" * 64})
        records.extend([baseline, comparison])
    with pytest.raises(ValueError, match=field):
        paired_exact_validity_by_fold(
            records, baseline_model_id="g_graph", comparison_model_id="g_l_local_logic",
            bootstrap_samples=10,
        )


@pytest.mark.parametrize("seed", [True, 1.5, "42"])
def test_record_rejects_lossy_seed_coercion(seed):
    with pytest.raises(ValueError, match="seed"):
        replace(_record(), sampling_seed=seed)
    payload = _record().to_dict()
    payload["sampling_seed"] = seed
    with pytest.raises(ValueError, match="seed"):
        PublicationResultRecord.from_dict(payload)


@pytest.mark.parametrize("shape", [(4.5, 4), (True, 4), ("4", 4)])
def test_stage_rejects_lossy_shape_coercion(shape):
    with pytest.raises(ValueError, match="shape"):
        replace(_record().stages["raw"], shape=shape)
    payload = _record().stages["raw"].to_dict()
    payload["shape"] = list(shape)
    with pytest.raises(ValueError, match="shape"):
        StageResult.from_dict(payload)


def test_stage_bridge_refuses_to_overwrite_existing_attempt_grids(tmp_path):
    metadata = _record().to_dict()
    for name in ("record_id", "schema_version", "stages", "generation_status",
                 "repair_metrics", "timing", "error_type"):
        metadata.pop(name)
    kwargs = dict(
        validator=lambda grid, stage: {"is_exact": True, "solvable": True, "termination_status": "solved"},
        output_dir=tmp_path, **metadata,
    )
    build_publication_result_record(_generated(), **kwargs)
    previous = (tmp_path / "grids" / "raw.npy").read_bytes()
    with pytest.raises(FileExistsError):
        build_publication_result_record(_generated(), **kwargs)
    assert (tmp_path / "grids" / "raw.npy").read_bytes() == previous


def test_record_serialization_rejects_nonfinite_metadata():
    with pytest.raises(ValueError, match="JSON"):
        replace(_record(), timing={"generation_time_sec": float("nan")}).to_json()
