"""Sampling-matrix preflight using only fresh synthetic checkpoint bytes."""

import json

import pytest
import yaml

from experiments.paper_v1.plan_attempts import build_sampling_plan
from experiments.paper_v1.run_protocol import build_run_plan
from src.config_system import merge_config
from src.publication.attempt_runner import AttemptLedger, AttemptSpec
from src.publication.dataset_provenance import dataset_file_records, verify_dataset_manifest
from src.publication.protocol import build_leave_one_dungeon_out_folds, seal_protocol_manifest


def _inputs(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "room.txt").write_text("WWW\nWFW\nWWW\n", encoding="utf-8")
    (data / "dataset_manifest.json").write_text(json.dumps({"files": dataset_file_records(data)}), encoding="utf-8")
    base = merge_config()
    base["dataset"]["data_dir"] = str(data)
    manifest, configs = build_run_plan(base_config=base, output_root=tmp_path / "training",
        folds=(build_leave_one_dungeon_out_folds()[0],), training_seeds=(42,),
        dataset_provenance=verify_dataset_manifest(data))
    for path, config in configs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
    from pathlib import Path
    for job in manifest["jobs"]["tokenizer"] + manifest["jobs"]["diffusion"]:
        checkpoint = Path(job["expected_checkpoint"])
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"synthetic checkpoint fixture; never loaded")
    graph = tmp_path / "mission.json"
    graph.write_text('{"directed": true, "multigraph": false, "graph": {}, "nodes": [{"id": 0}], "links": []}', encoding="utf-8")
    from src.publication.execution_provenance import capture_execution_provenance
    from src.publication.evaluation_design import seal_evaluation_design
    execution = capture_execution_provenance()
    manifest["execution_provenance"] = execution
    manifest["evaluation_design"] = seal_evaluation_design(
        mission_bank={"holdout_dungeon_1": {"a": str(graph), "b": str(graph)}},
        sampling_seeds=(1, 2), generation_kwargs={"num_diffusion_steps": 2}, fold_ids=["holdout_dungeon_1"])
    manifest = seal_protocol_manifest(manifest)
    return dict(training_manifest=manifest, protocol_sha256=manifest["protocol_sha256"],
                mission_bank={"holdout_dungeon_1": {"a": str(graph), "b": str(graph)}},
                sampling_seeds=(1, 2), generation_kwargs={"num_diffusion_steps": 2}, git_commit=execution["git_commit"])


def test_sampling_plan_expands_every_arm_seed_and_mission(tmp_path):
    from src.publication.protocol import CONTROL_MODELS, PUBLICATION_VARIANTS

    plan = build_sampling_plan(**_inputs(tmp_path))
    specs = [AttemptSpec(**entry) for entry in plan["attempts"]]
    expected_count = (len(PUBLICATION_VARIANTS) + len(CONTROL_MODELS)) * 2 * 2
    assert len(specs) == expected_count
    ledger = AttemptLedger.create(tmp_path / "attempts", specs)
    assert len(ledger.specs()) == expected_count
    assert {spec.metadata["inference_variant"] for spec in specs} == {"unguided"}
    assert len({spec.artifacts["vqvae_checkpoint"]["path"] for spec in specs if "vqvae_checkpoint" in spec.artifacts}) == 1
    controls = [spec for spec in specs if spec.metadata["model_id"] in set(CONTROL_MODELS)]
    assert len(controls) == len(CONTROL_MODELS) * 2 * 2
    assert all(set(spec.artifacts) == {"resolved_config", "mission_graph"} for spec in controls)


def test_preflight_rejects_source_targets_before_any_training(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from src.publication.data_support import preflight_training_manifest
    inputs = _inputs(tmp_path)
    inputs["training_manifest"]["confirmatory_analysis"]["primary_endpoint"] = "raw_stage_exact_validity_over_all_requested_attempts"
    inputs["training_manifest"] = seal_protocol_manifest(inputs["training_manifest"])
    inspected = []
    def dataset(**kwargs):
        inspected.append(kwargs)
        return SimpleNamespace(rooms=[[[1, 2, 10, 42]]])
    monkeypatch.setattr("src.zelda_data.zelda_loader.ZeldaRoomDataset", dataset)
    with pytest.raises(ValueError, match="unsupported by the training targets"):
        preflight_training_manifest(inputs["training_manifest"])
    assert len(inspected) == 1
    assert inspected[0]["load_graphs"] is False
    assert inspected[0]["dungeon_ids"] == list(range(2, 10))


def test_compiled_endpoint_preflight_accepts_geometry_targets_and_checks_fold_once(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from src.publication.data_support import preflight_training_manifest
    inputs = _inputs(tmp_path)
    inspected = []
    def dataset(**kwargs):
        inspected.append(kwargs)
        return SimpleNamespace(rooms=[[[1, 2, 10, 42]]])
    monkeypatch.setattr("src.zelda_data.zelda_loader.ZeldaRoomDataset", dataset)
    overlaps = preflight_training_manifest(inputs["training_manifest"])
    assert len(inspected) == 2
    assert overlaps["holdout_dungeon_1"]["repeated_layout_count"] == 1
    assert inspected[1]["dungeon_ids"] == [1]


@pytest.mark.parametrize("seeds", [(), (True,), (1.5,), (-1,), (1, 1)])
def test_sampling_plan_rejects_invalid_seeds(tmp_path, seeds):
    inputs = _inputs(tmp_path)
    inputs["sampling_seeds"] = seeds
    with pytest.raises(ValueError, match="Sampling seeds"):
        build_sampling_plan(**inputs)


@pytest.mark.parametrize("mutation", ["omitted_arm", "omitted_fold", "unverified_data", "protocol_drift", "config_drift", "data_drift"])
def test_sampling_plan_rejects_incomplete_or_changed_inputs(tmp_path, mutation):
    from pathlib import Path
    inputs = _inputs(tmp_path)
    manifest = inputs["training_manifest"]
    if mutation == "omitted_arm":
        manifest["jobs"]["diffusion"].pop()
    elif mutation == "omitted_fold":
        inputs["mission_bank"] = {}
    elif mutation == "unverified_data":
        manifest["dataset_provenance"]["verification_status"] = "not_verified"
    elif mutation == "config_drift":
        path = Path(manifest["jobs"]["diffusion"][0]["config"])
        config = yaml.safe_load(path.read_text())
        config["runtime"]["seed"] += 1
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
    elif mutation == "data_drift":
        (tmp_path / "data" / "room.txt").write_text("changed source", encoding="utf-8")
    if mutation == "protocol_drift":
        manifest["training_seeds"] = [123]
    else:
        manifest = seal_protocol_manifest(manifest)
        inputs.update(training_manifest=manifest, protocol_sha256=manifest["protocol_sha256"])
    with pytest.raises(ValueError):
        build_sampling_plan(**inputs)


@pytest.mark.parametrize("change", ["seed", "mission", "settings", "git"])
def test_evaluation_cannot_be_chosen_after_training(tmp_path, change):
    from pathlib import Path
    inputs = _inputs(tmp_path)
    if change == "seed":
        inputs["sampling_seeds"] = (10, 20)
    elif change == "settings":
        inputs["generation_kwargs"]["num_diffusion_steps"] = 3
    elif change == "git":
        inputs["git_commit"] = "invented-commit"
    else:
        Path(inputs["mission_bank"]["holdout_dungeon_1"]["a"]).write_text('{"nodes": [], "links": []}')
    with pytest.raises(ValueError, match="frozen before training|Git commit|Evaluation mission is empty"):
        build_sampling_plan(**inputs)


def test_between_job_preflight_checks_tokenizer_config_too(tmp_path):
    from pathlib import Path
    from src.publication.data_support import verify_training_inputs
    inputs = _inputs(tmp_path)
    manifest = inputs["training_manifest"]
    path = Path(manifest["jobs"]["tokenizer"][0]["config"])
    config = yaml.safe_load(path.read_text())
    config["runtime"]["seed"] += 1
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="configuration changed"):
        verify_training_inputs(manifest)
