"""Synthetic checks of the real experiment adapter, without model checkpoints."""

import json
from dataclasses import replace
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from experiments.paper_v1.run_protocol import build_run_plan
from src.config_system import merge_config
from src.publication.attempt_runner import AttemptLedger, AttemptSpec
from src.publication.dataset_provenance import file_sha256
from src.publication.pipeline_attempt import CanonicalAttemptPipeline
from src.publication.protocol import build_leave_one_dungeon_out_folds, payload_sha256


def _spec(tmp_path):
    _, configs = build_run_plan(base_config=merge_config(), output_root=tmp_path / "plan",
        folds=(build_leave_one_dungeon_out_folds()[0],), training_seeds=(42,))
    config = next(config for _, config in configs if config["runtime"]["experiment_name"].find("g_graph") >= 0)
    paths = {}
    for name in ("resolved_config", "mission_graph", "vqvae_checkpoint", "diffusion_checkpoint"):
        path = tmp_path / name
        path.write_text("synthetic input only", encoding="utf-8")
        paths[name] = path
    paths["resolved_config"].write_text(json.dumps(config), encoding="utf-8")
    graph = nx.DiGraph()
    graph.add_node(0, is_start=True)
    graph.add_node(1, is_triforce=True)
    graph.add_edge(0, 1)
    paths["mission_graph"].write_text(json.dumps(nx.node_link_data(graph, edges="links")), encoding="utf-8")
    return AttemptSpec(
        dict(run_id="new", git_commit="source", config_sha256=payload_sha256(config),
             dataset_manifest_sha256="b" * 64, protocol_sha256="c" * 64,
             outer_fold="holdout_dungeon_1", train_seed=42, sampling_seed=12,
             mission_graph_id="graph", model_id="g_graph", inference_variant="unguided"),
        {"num_diffusion_steps": 2},
        {name: {"path": str(path), "sha256": file_sha256(path)} for name, path in paths.items()},
    )


def test_constructor_randomness_and_cache_hits_cannot_change_sampling_stream(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    calls = []
    monkeypatch.setattr("src.publication.pipeline_attempt.seed_everything",
                        lambda seed, **kwargs: np.random.seed(seed))
    def factory(**kwargs):
        calls.append(kwargs)
        np.random.random(100)
        return SimpleNamespace(generate_dungeon=lambda **kwargs: (np.random.random(), kwargs))
    adapter = CanonicalAttemptPipeline(factory)
    first, first_kwargs = adapter.generate(spec)
    second, second_kwargs = adapter.generate(spec)
    assert first == second
    assert len(calls) == 1
    assert calls[0]["strict_checkpoint_mode"] is True
    assert calls[0]["require_logic_net"] is False
    assert calls[0]["logic_net_checkpoint"] is None
    assert calls[0]["use_learned_refiner_rules"] is False
    assert first_kwargs["seed"] == second_kwargs["seed"] == 12
    assert first_kwargs["logic_guidance_scale"] == 0.0
    assert list(first_kwargs["mission_graph"].edges) == [(0, 1)]


@pytest.mark.parametrize("kwargs", [{"generate_topology": True}, {"logic_guidance_scale": 1.0}])
def test_canonical_adapter_rejects_undeclared_treatment_changes(tmp_path, kwargs):
    spec = replace(_spec(tmp_path), generation_kwargs=kwargs)
    adapter = CanonicalAttemptPipeline(lambda **kwargs: SimpleNamespace())
    with pytest.raises(ValueError):
        adapter.generate(spec)


def test_paired_models_cannot_change_sampling_settings(tmp_path):
    spec = _spec(tmp_path)
    other = replace(spec, metadata={**spec.metadata, "model_id": "n_local"},
                    generation_kwargs={"num_diffusion_steps": 7})
    with pytest.raises(ValueError, match="identical sampling"):
        AttemptLedger.create(tmp_path / "ledger", [spec, other])


@pytest.mark.parametrize("metadata,kwargs,match", [
    ({"outer_fold": "holdout_dungeon_2"}, {}, "Held-out dungeon"),
    ({"inference_variant": "logic_guided"}, {"logic_guidance_scale": 1.0}, "trained LogicNet"),
])
def test_adapter_rejects_wrong_fold_and_unsupported_guidance(tmp_path, metadata, kwargs, match):
    spec = _spec(tmp_path)
    spec = replace(spec, metadata={**spec.metadata, **metadata}, generation_kwargs=kwargs)
    adapter = CanonicalAttemptPipeline(lambda **kwargs: SimpleNamespace())
    with pytest.raises(ValueError, match=match):
        adapter.generate(spec)


@pytest.mark.parametrize("artifact", ["mission_graph", "vqvae_checkpoint"])
def test_paired_models_cannot_change_shared_inputs(tmp_path, artifact):
    spec = _spec(tmp_path)
    changed = tmp_path / "different_input"
    changed.write_text("another synthetic input")
    other = replace(spec, metadata={**spec.metadata, "model_id": "n_local"},
        artifacts={**spec.artifacts, artifact: {"path": str(changed), "sha256": file_sha256(changed)}})
    with pytest.raises(ValueError, match="Paired input drift"):
        AttemptLedger.create(tmp_path / "ledger", [spec, other])


def test_each_stage_gets_fresh_tile_and_spatial_validation(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    adapter = CanonicalAttemptPipeline(lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(adapter, "_load", lambda spec: SimpleNamespace())
    graph = nx.DiGraph([(0, 1)])
    generated = SimpleNamespace(physical_mission_graph=graph, puzzle_metadata={},
        stitched_layout=SimpleNamespace(slot_positions={0: (0, 0), 1: (0, 1)},
            room_offsets={0: (0, 0), 1: (0, 11)}, realization_metrics={}, connection_realizations={}))
    monkeypatch.setattr("src.publication.pipeline_attempt._validate_mission_graph_exact",
        lambda *args: {"termination_status": "solved", "solvable": True,
                       "all_rooms_reachable": True, "route_replay_status": "verified"})
    monkeypatch.setattr("src.publication.pipeline_attempt.validate_attached_global_state_contract", lambda *args, **kwargs: None)
    seen = []
    def tile(pipeline, grid, **kwargs):
        solved = bool(grid[1, 1] == 21)
        seen.append(grid.copy())
        assert kwargs["room_to_node"] == {(0, 0): 0, (0, 1): 1}
        return {"termination_status": "solved" if solved else "invalid", "solvable": solved,
                "is_exact": True, "is_valid_syntax": solved,
                "graph_tile_context_applied": True, "proven_unsolvable": not solved,
                "route_replay_status": "verified" if solved else "not_run"}
    monkeypatch.setattr("src.publication.pipeline_attempt._validate_dungeon", tile)
    monkeypatch.setattr("src.publication.pipeline_attempt.validate_connection_realizations",
                        lambda grid, connections: {"final_spatial_edge_records_broken": 0})
    validate = adapter.validator_factory(spec, generated)
    invalid = np.ones((16, 22), dtype=np.int32)
    valid = invalid.copy()
    valid[1, 1], valid[1, 20] = 21, 22
    raw = validate(invalid, "raw")
    compiled = validate(valid, "compiled")
    assert len(seen) == 2
    assert raw["accepted"] is False
    assert compiled["accepted"] is True
    assert compiled["termination_status"] == "solved"
    assert "contract" in compiled


@pytest.mark.parametrize("model", ["retrieval_training", "wfc_training", "pod_training", "ar_markov_training"])
def test_independent_control_uses_real_symbolic_pipeline_without_loading_weights(tmp_path, monkeypatch, model):
    from src.publication.control_attempt import ControlAttemptPipeline
    from src.publication.room_evidence import TrainingRoomReference
    spec = _spec(tmp_path)
    spec = replace(spec, metadata={**spec.metadata, "model_id": model},
                   artifacts={key: spec.artifacts[key] for key in ("resolved_config", "mission_graph")},
                   generation_kwargs={"apply_repair": False})
    seen = []
    def reference(config):
        seen.append(config["dataset"]["train_dungeons"])
        return TrainingRoomReference([np.ones((16, 11), dtype=np.int32)])
    monkeypatch.setattr(TrainingRoomReference, "from_config", reference)
    monkeypatch.setattr("src.publication.control_attempt.verify_dataset_manifest",
                        lambda root: {"content_sha256": "b" * 64})
    adapter = ControlAttemptPipeline()
    first = adapter.generate(spec)
    second = adapter.generate(spec)
    assert seen == [list(range(2, 10))]
    assert adapter._pipeline.vqvae is adapter._pipeline.diffusion is adapter._pipeline.condition_encoder is None
    assert first.metrics["symbolic_only"]
    assert first.metrics["graph_semantic_compilation_applied"]
    assert first.physical_mission_graph is not None
    np.testing.assert_array_equal(first.dungeon_grid, second.dungeon_grid)
    assert all(room.metrics["training_reference"]["exact_training_match"] for room in first.rooms.values())
    verdict = adapter.validator_factory(spec, first)(first.dungeon_grid, "compiled")
    assert verdict["termination_status"] != "validator_error"


def test_canonical_attempt_pipeline_supports_masked_room_branch(tmp_path, monkeypatch):
    _, configs = build_run_plan(
        base_config=merge_config(),
        output_root=tmp_path / "masked_plan",
        folds=(build_leave_one_dungeon_out_folds()[0],),
        training_seeds=(42,),
        model_branch="masked_room",
    )
    config = next(cfg for _, cfg in configs if "g_lg_full_logic" in cfg["runtime"]["experiment_name"])
    resolved_path = tmp_path / "resolved_config.json"
    resolved_path.write_text(json.dumps(config), encoding="utf-8")
    masked_ckpt = tmp_path / "masked_room_best.pth"
    masked_ckpt.write_text("synthetic masked room checkpoint", encoding="utf-8")
    mission_path = tmp_path / "mission_graph.json"
    graph = nx.DiGraph()
    graph.add_node(0, is_start=True)
    graph.add_node(1, is_triforce=True)
    graph.add_edge(0, 1)
    mission_path.write_text(json.dumps(nx.node_link_data(graph, edges="links")), encoding="utf-8")

    spec = AttemptSpec(
        dict(
            run_id="masked_run",
            git_commit="source",
            config_sha256=payload_sha256(config),
            dataset_manifest_sha256="b" * 64,
            protocol_sha256="c" * 64,
            outer_fold="holdout_dungeon_1",
            train_seed=42,
            sampling_seed=12,
            mission_graph_id="graph",
            model_id="g_lg_full_logic",
            inference_variant="unguided",
        ),
        {"num_diffusion_steps": 4},
        {
            "resolved_config": {"path": str(resolved_path), "sha256": file_sha256(resolved_path)},
            "mission_graph": {"path": str(mission_path), "sha256": file_sha256(mission_path)},
            "masked_room_checkpoint": {"path": str(masked_ckpt), "sha256": file_sha256(masked_ckpt)},
        },
    )
    calls = []
    monkeypatch.setattr("src.publication.pipeline_attempt.seed_everything", lambda *a, **k: None)

    def factory(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(generate_dungeon=lambda **gen_kwargs: SimpleNamespace(rooms={}, kwargs=gen_kwargs))

    adapter = CanonicalAttemptPipeline(factory)
    out = adapter.generate(spec)
    assert len(calls) == 1
    assert calls[0]["room_generator_mode"] == "discrete_masked"
    assert calls[0]["masked_room_checkpoint"] == str(masked_ckpt)
    assert calls[0]["condition_encoder_checkpoint"] == str(masked_ckpt)
    assert calls[0]["logic_net_checkpoint"] == str(masked_ckpt)
    assert calls[0]["strict_checkpoint_mode"] is True
    assert calls[0]["require_logic_net"] is True
    assert calls[0]["condition_gnn_type"] == "gps"
    assert calls[0]["condition_use_rrwp_edge_features"] is True
    assert out.kwargs["seed"] == 12
