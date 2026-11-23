"""Failure-injection tests for the experiment executor; no trained models."""

import json
from dataclasses import replace
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from src.publication.attempt_runner import AttemptLedger, AttemptSpec, execute_attempts
from src.publication.analysis import aggregate_publication_records
from src.publication.dataset_provenance import file_sha256


def _spec(seed=1, model="g_graph", **overrides):
    metadata = dict(run_id="run", git_commit="test-source", config_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64, protocol_sha256="c" * 64,
        outer_fold="holdout_dungeon_1", train_seed=42, sampling_seed=seed,
        mission_graph_id="mission", model_id=model, inference_variant="unguided")
    metadata.update(overrides)
    return AttemptSpec(metadata, {"seed": seed}, {})


def _generate(_spec):
    grid = np.ones((4, 4), dtype=np.int32)
    return SimpleNamespace(
        rooms={0: SimpleNamespace(raw_neural_grid=grid, neural_grid=grid, room_grid=grid)},
        stitched_layout=SimpleNamespace(slot_positions={0: (0, 0)}),
        physical_mission_graph=nx.empty_graph(1), dungeon_grid=grid, metrics={},
    )


def _validator(_spec, _generated):
    return lambda grid, stage: {"is_exact": True, "solvable": True, "termination_status": "solved"}


def _run(ledger, generate=_generate, **kwargs):
    return execute_attempts(ledger, generate=generate, validator_factory=_validator,
                            seed_fn=lambda seed: None, **kwargs)


def test_batch_retains_generation_and_timeout_failures_and_continues(tmp_path):
    specs = [_spec(seed) for seed in (1, 2, 3)]
    ledger = AttemptLedger.create(tmp_path / "fresh", specs)
    called = []
    seeded = []

    def generate(spec):
        seed = spec.metadata["sampling_seed"]
        called.append(seed)
        if seed == 1:
            raise RuntimeError("injected generation failure")
        if seed == 2:
            raise TimeoutError("injected timeout")
        return _generate(spec)

    records = execute_attempts(ledger, generate=generate, validator_factory=_validator,
                              seed_fn=seeded.append)
    assert called == seeded == [1, 2, 3]
    assert [record.generation_status for record in records] == ["generation_error", "timeout", "completed"]
    rows = aggregate_publication_records(records)
    assert next(row for row in rows if row["stage"] == "raw")["exact_valid_rate_all_attempts"] == pytest.approx(1/3)
    assert all(len(record.stages) == 4 for record in records)


def test_absence_from_both_arms_blocks_analysis(tmp_path):
    ledger = AttemptLedger.create(tmp_path / "fresh", [_spec(model=model) for model in ("g_graph", "n_local")])
    with pytest.raises(ValueError, match="Incomplete"):
        ledger.complete_records()


def test_interruption_cannot_be_resumed_under_the_same_attempt_key(tmp_path):
    ledger = AttemptLedger.create(tmp_path / "fresh", [_spec()])
    def interrupt(spec):
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        _run(ledger, interrupt)
    with pytest.raises(ValueError, match="Incomplete"):
        ledger.complete_records()
    with pytest.raises(FileExistsError, match="reruns"):
        _run(ledger)


def test_completed_attempts_are_never_overwritten(tmp_path):
    ledger = AttemptLedger.create(tmp_path / "fresh", [_spec()])
    _run(ledger)
    before = (ledger.root / "attempts" / _spec().attempt_id / "record.json").read_bytes()
    with pytest.raises(FileExistsError):
        _run(ledger)
    assert (ledger.root / "attempts" / _spec().attempt_id / "record.json").read_bytes() == before


def test_plan_rejects_rerun_names_for_duplicate_attempts(tmp_path):
    with pytest.raises(ValueError, match="Duplicate"):
        AttemptLedger.create(tmp_path / "fresh", [_spec(), _spec(run_id="rerun")])
    assert not (tmp_path / "fresh").exists()


def test_manifest_tampering_is_detected_before_generation(tmp_path):
    ledger = AttemptLedger.create(tmp_path / "fresh", [_spec()])
    path = ledger.root / "manifest.json"
    payload = json.loads(path.read_text())
    payload["attempts"][0]["metadata"]["sampling_seed"] = 99
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="manifest changed"):
        _run(ledger)


def test_input_drift_aborts_instead_of_becoming_a_scored_generation_failure(tmp_path):
    artifact = tmp_path / "new_fixture.json"
    artifact.write_text("{}")
    spec = replace(_spec(), artifacts={"mission": {"path": str(artifact), "sha256": file_sha256(artifact)}})
    ledger = AttemptLedger.create(tmp_path / "fresh", [spec])
    artifact.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="missing or changed"):
        _run(ledger)
    with pytest.raises(ValueError, match="Incomplete"):
        ledger.complete_records()


@pytest.mark.parametrize("target", ["record", "grid", "unassembled_room"])
def test_changed_outputs_cannot_enter_analysis(tmp_path, target):
    ledger = AttemptLedger.create(tmp_path / "fresh", [_spec()])
    records = _run(ledger)
    if target == "grid":
        from pathlib import Path
        Path(records[0].stages["raw"].grid_path).write_bytes(b"changed")
    elif target == "unassembled_room":
        from pathlib import Path
        Path(records[0].repair_metrics["unassembled_rooms"][0]["artifacts"]["raw"]["path"]).write_bytes(b"changed")
    else:
        path = ledger.root / "attempts" / _spec().attempt_id / "record.json"
        payload = json.loads(path.read_text())
        payload["timing"] = {"changed": 1.0}
        path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="changed"):
        ledger.complete_records()


def test_unassembled_metrics_retain_boundary_edits_hidden_by_stitching(tmp_path):
    def generate(spec):
        result = _generate(spec)
        raw = np.ones((4, 4), dtype=np.int32)
        constrained = raw.copy()
        constrained[0, 0] = 2  # Stitching seals this corner in both stages.
        result.rooms[0] = SimpleNamespace(raw_neural_grid=raw, neural_grid=constrained, room_grid=constrained)
        return result
    ledger = AttemptLedger.create(tmp_path / "fresh", [_spec()])
    record = _run(ledger, generate)[0]
    assert record.repair_metrics["raw_to_constrained"]["changed_tile_count"] == 0
    assert record.repair_metrics["unassembled_rooms"][0]["raw_to_constrained"]["changed_tile_count"] == 1


def test_validator_exception_preserves_all_raw_stage_evidence(tmp_path):
    ledger = AttemptLedger.create(tmp_path / "fresh", [_spec()])
    def factory(spec, generated):
        def validate(grid, stage):
            raise RuntimeError("injected exact validator exception")
        return validate
    records = execute_attempts(ledger, generate=_generate, validator_factory=factory,
                              seed_fn=lambda seed: None)
    assert records[0].generation_status == "completed"
    assert all(stage.status == "validator_error" and stage.grid_path for stage in records[0].stages.values())


@pytest.mark.parametrize("during_setup", [False, True])
@pytest.mark.parametrize("exception,status", [(RuntimeError, "validator_error"), (TimeoutError, "timeout")])
def test_validator_setup_and_search_failures_retain_grids(tmp_path, during_setup, exception, status):
    ledger = AttemptLedger.create(tmp_path / "fresh", [_spec()])
    def factory(spec, generated):
        if during_setup:
            raise exception("injected setup failure")
        def validate(grid, stage):
            raise exception("injected search failure")
        return validate
    records = execute_attempts(ledger, generate=_generate, validator_factory=factory,
                              seed_fn=lambda seed: None)
    assert records[0].generation_status == "completed"
    assert all(stage.status == status and stage.grid_path for stage in records[0].stages.values())
