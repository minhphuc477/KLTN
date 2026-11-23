from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import networkx as nx
import pytest
import torch

from src.publication.protocol import (
    PUBLICATION_VARIANTS,
    ModelVariant,
    apply_publication_variant,
    build_leave_one_dungeon_out_folds,
    build_protocol_manifest,
    payload_sha256,
)
from src.publication.records import (
    PublicationResultRecord,
    StageResult,
    changed_tile_metrics,
)
from src.publication.analysis import (
    aggregate_publication_records,
    holm_adjust_pvalues,
    paired_exact_validity_by_fold,
    paired_guidance_validity_by_fold,
)
from src.publication.stage_bridge import build_publication_result_record
from src.publication.dataset_provenance import (
    dataset_content_sha256,
    verify_dataset_manifest,
)
from src.train_diffusion import DiffusionTrainer
from src.training.diffusion_config import DiffusionTrainingConfig
from experiments.paper_v1.run_protocol import (
    assert_model_variant_config_parity,
    build_run_plan,
    materialize_protocol,
)
from src.config_system import merge_config
from src.pipeline.generation.room_processing import _compute_room_condition


def test_holm_adjustment_controls_preregistered_contrast_family() -> None:
    adjusted = holm_adjust_pvalues(
        {"N_vs_G": 0.01, "G_vs_GL": 0.03, "GL_vs_GLG": 0.04}
    )

    assert adjusted["N_vs_G"]["holm_adjusted_p_value"] == pytest.approx(0.03)
    assert adjusted["G_vs_GL"]["holm_adjusted_p_value"] == pytest.approx(0.06)
    assert adjusted["GL_vs_GLG"]["holm_adjusted_p_value"] == pytest.approx(0.06)
    assert adjusted["N_vs_G"]["reject_familywise_null"] is True
    assert adjusted["G_vs_GL"]["reject_familywise_null"] is False


def test_lodo_folds_hold_out_complete_dungeon_identities() -> None:
    folds = build_leave_one_dungeon_out_folds()

    assert len(folds) == 9
    assert {fold.test_dungeons[0] for fold in folds} == set(range(1, 10))
    for fold in folds:
        assert set(fold.train_dungeons).isdisjoint(fold.test_dungeons)
        assert set(fold.train_dungeons).union(fold.test_dungeons) == set(range(1, 10))
        assert fold.variants == (1, 2)


def test_publication_variants_are_separately_trainable_conditions() -> None:
    variants = {variant.model_id: variant for variant in PUBLICATION_VARIANTS}

    assert set(variants) == {
        "n_local",
        "g_graph",
        "g_l_local_logic",
        "g_lg_full_logic",
    }
    assert variants["n_local"].training_overrides()[
        "diffusion.condition_global_graph_enabled"
    ] is False
    assert variants["g_graph"].logic_net_enabled is False
    assert variants["g_l_local_logic"].training_overrides()[
        "diffusion.logic_global_reach_weight"
    ] == 0.0
    assert variants["g_l_local_logic"].training_overrides()[
        "diffusion.logic_lock_weight"
    ] == 0.0
    assert variants["g_lg_full_logic"].training_overrides()[
        "diffusion.logic_global_reach_weight"
    ] > 0.0
    assert variants["g_lg_full_logic"].training_overrides()[
        "diffusion.logic_lock_weight"
    ] > 0.0
    assert variants["g_l_local_logic"].tile_auxiliary_loss is False
    assert variants["g_lg_full_logic"].tile_auxiliary_loss is False
    for variant in variants.values():
        overrides = variant.training_overrides()
        assert overrides["diffusion.condition_strict_schema"] is True
        assert overrides["diffusion.condition_use_reference_room_maps"] is False
        assert overrides["diffusion.guidance_scale"] == 0.0
        assert overrides["generation.logic_guidance_scale"] == 0.0
        assert overrides["generation.enable_map_elites"] is False
        assert overrides["generation.use_neural_guided_repair"] is False
        assert overrides["generation.use_neural_repair_feedback"] is False
        assert overrides["generation.end_to_end_validation_mode"] == "report"


def test_masked_room_training_overrides_mirror_causal_factors() -> None:
    variants = {variant.model_id: variant for variant in PUBLICATION_VARIANTS}
    n_over = variants["n_local"].masked_room_training_overrides()
    g_over = variants["g_graph"].masked_room_training_overrides()
    gl_over = variants["g_l_local_logic"].masked_room_training_overrides()
    glg_over = variants["g_lg_full_logic"].masked_room_training_overrides()

    assert n_over["masked_room.condition_gnn_type"] == "gcn"
    assert n_over["masked_room.graph_conditioning_mode"] == "pooled"
    assert g_over["masked_room.condition_gnn_type"] == "gps"
    assert g_over["masked_room.logic_net_enabled"] is False
    assert gl_over["masked_room.logic_net_enabled"] is True
    assert gl_over["masked_room.alpha_logic"] == pytest.approx(0.1)
    assert gl_over["masked_room.logic_global_reach_weight"] == 0.0
    assert glg_over["masked_room.logic_global_reach_weight"] == pytest.approx(1.0)
    for over in (n_over, g_over, gl_over, glg_over):
        assert over["masked_room.puzzle_stage_semantics_loss_weight"] == 0.0
        assert over["generation.end_to_end_validation_mode"] == "report"


def test_build_run_plan_supports_masked_room_branch_without_vqvae_jobs(tmp_path: Path) -> None:
    manifest, configs = build_run_plan(
        base_config=merge_config(),
        output_root=tmp_path / "glmdm_plan",
        folds=(build_leave_one_dungeon_out_folds()[0],),
        training_seeds=(42,),
        model_branch="masked_room",
    )

    assert manifest["model_branch"] == "masked_room"
    assert manifest["tokenizer_run_count"] == 0
    assert manifest["diffusion_run_count"] == 4
    assert len(configs) == 4
    for job in manifest["jobs"]["diffusion"]:
        assert job["model_branch"] == "masked_room"
        assert job["depends_on"] == []
        assert "checkpoints\\masked_room\\masked_room_best.pth" in job["expected_checkpoint"] or "checkpoints/masked_room/masked_room_best.pth" in job["expected_checkpoint"]


def test_global_logic_variant_requires_graph_conditioning() -> None:
    with pytest.raises(ValueError, match="Global logic loss requires"):
        ModelVariant(
            model_id="invalid",
            description="invalid",
            global_graph_conditioning=False,
            local_logic_loss=False,
            global_logic_loss=True,
            tile_auxiliary_loss=False,
        )


def test_publication_variant_id_cannot_escape_config_directory() -> None:
    with pytest.raises(ValueError, match="model_id must use"):
        ModelVariant(
            model_id="../outside",
            description="invalid path-like identifier",
            global_graph_conditioning=False,
            local_logic_loss=False,
            global_logic_loss=False,
            tile_auxiliary_loss=False,
        )


def test_apply_publication_variant_isolates_fold_seed_and_output() -> None:
    fold = build_leave_one_dungeon_out_folds()[8]
    local_variant = PUBLICATION_VARIANTS[0]
    base = {
        "dataset": {"train_dungeons": [9], "test_dungeons": [1]},
        "runtime": {"seed": 999, "auto_resume": True},
        "diffusion": {"logic_net_enabled": True},
        "generation": {"enable_map_elites": True},
    }

    resolved = apply_publication_variant(
        base,
        fold=fold,
        variant=local_variant,
        training_seed=42,
    )

    assert resolved["dataset"]["train_dungeons"] == list(range(1, 9))
    assert resolved["dataset"]["test_dungeons"] == [9]
    assert resolved["dataset"]["variants"] == [1, 2]
    assert resolved["dataset"]["topology_supervision_mode"] == "runtime_aligned"
    assert resolved["runtime"]["seed"] == 42
    assert resolved["runtime"]["auto_resume"] is False
    assert resolved["runtime"]["resume"] is None
    assert resolved["diffusion"]["condition_global_graph_enabled"] is False
    assert resolved["diffusion"]["logic_net_enabled"] is False
    assert resolved["diffusion"]["best_checkpoint_metric"] == "val_diffusion_loss"
    assert resolved["diffusion"]["nonfinite_policy"] == "error"
    assert resolved["generation"]["enable_map_elites"] is False
    assert resolved["generation"]["use_neural_guided_repair"] is False
    assert resolved["generation"]["use_neural_repair_feedback"] is False
    # The input mapping is not mutated.
    assert base["runtime"]["seed"] == 999


def test_protocol_manifest_is_deterministic_and_complete() -> None:
    first = build_protocol_manifest()
    second = build_protocol_manifest()

    assert first == second
    assert first["run_count"] == 9 * 4 * 5
    assert len(first["protocol_sha256"]) == 64
    assert first["evidence_policy"]["allow_existing_checkpoints"] is False
    assert first["evidence_policy"]["require_raw_stage"] is True
    primary = first["inference_factors"]["primary_model_ablation"]
    secondary = first["inference_factors"]["secondary_logic_guidance"]
    assert primary["condition"] == "unguided"
    analysis = first["confirmatory_analysis"]
    assert analysis["primary_endpoint"] == "compiled_stage_exact_validity_over_all_requested_attempts"
    assert analysis["primary_stage"] == "compiled"
    assert "raw" in analysis["secondary_stages"]
    assert analysis["multiplicity_correction"].startswith("holm_step_down")
    assert analysis["familywise_alpha"] == pytest.approx(0.05)
    assert len(analysis["primary_contrast_family"]) == 3
    assert set(primary["eligible_model_ids"]) == {
        "n_local",
        "g_graph",
        "g_l_local_logic",
        "g_lg_full_logic",
    }
    assert secondary["eligible_model_ids"] == ["g_l_local_logic", "g_lg_full_logic"]
    claimed_digest = first["protocol_sha256"]
    unsigned = dict(first)
    unsigned.pop("protocol_sha256")
    assert payload_sha256(unsigned) == claimed_digest


def test_protocol_rejects_empty_or_duplicate_custom_folds() -> None:
    fold = build_leave_one_dungeon_out_folds()[0]

    with pytest.raises(ValueError, match="At least one outer fold"):
        build_protocol_manifest(folds=())
    with pytest.raises(ValueError, match="fold IDs must be unique"):
        build_protocol_manifest(folds=(fold, fold))


def _stage(name: str, status: str = "generated") -> StageResult:
    exact_valid = True if status == "valid" else False if status == "invalid" else None
    return StageResult(
        stage=name,
        status=status,
        grid_path=f"grids/{name}.npy",
        grid_sha256="a" * 64,
        shape=(16, 11),
        exact_valid=exact_valid,
        validation={},
    )


def _record(stages: dict[str, StageResult]) -> PublicationResultRecord:
    if set(stages) == {"raw"}:
        stages = {
            **stages,
            "constrained": _stage("constrained"),
            "repaired": _stage("repaired"),
            "compiled": _stage("compiled"),
        }
    return PublicationResultRecord(
        run_id="holdout_dungeon_9__g_graph__seed_42",
        git_commit="deadbeef",
        config_sha256="b" * 64,
        dataset_manifest_sha256="c" * 64,
        protocol_sha256="d" * 64,
        outer_fold="holdout_dungeon_9",
        train_seed=42,
        sampling_seed=100,
        mission_graph_id="dungeon_9_quest_1",
        model_id="g_graph",
        inference_variant="raw",
        generation_status="completed",
        stages=stages,
    )


def test_result_record_requires_raw_and_preserves_all_intervention_stages() -> None:
    with pytest.raises(ValueError, match="retain the raw stage"):
        _record({"constrained": _stage("constrained")})

    with pytest.raises(ValueError, match="preserve its constrained stage"):
        _record({"raw": _stage("raw"), "repaired": _stage("repaired")})

    with pytest.raises(ValueError, match="preserve its repaired stage"):
        _record({"raw": _stage("raw"), "compiled": _stage("compiled")})

    record = _record(
        {
            "raw": _stage("raw", "invalid"),
            "constrained": _stage("constrained", "invalid"),
            "repaired": _stage("repaired", "valid"),
            "compiled": _stage("compiled", "valid"),
        }
    )
    round_trip = PublicationResultRecord.from_dict(record.to_dict())

    assert round_trip == record
    assert list(round_trip.to_dict()["stages"]) == [
        "raw",
        "constrained",
        "repaired",
        "compiled",
    ]


def test_result_schema_rejects_contradictory_status_and_digest_metadata() -> None:
    with pytest.raises(ValueError, match="requires exact_valid=None"):
        StageResult(
            stage="raw",
            status="generated",
            grid_path="grids/raw.npy",
            grid_sha256="a" * 64,
            shape=(16, 11),
            exact_valid=True,
        )
    with pytest.raises(ValueError, match="requires grid_path"):
        StageResult(
            stage="raw",
            status="generated",
            grid_path=None,
            grid_sha256="a" * 64,
            shape=(16, 11),
            exact_valid=None,
        )
    with pytest.raises(ValueError, match="config_sha256"):
        PublicationResultRecord(
            **{
                **_record({"raw": _stage("raw")}).__dict__,
                "config_sha256": "not-a-digest",
            }
        )
    with pytest.raises(ValueError, match="emitted raw grid"):
        PublicationResultRecord(
            **{
                **_record({"raw": _stage("raw")}).__dict__,
                "stages": {
                    "raw": StageResult(
                        stage="raw",
                        status="validator_error",
                        grid_path=None,
                        grid_sha256=None,
                        shape=None,
                        exact_valid=None,
                    )
                },
            }
        )
    with pytest.raises(ValueError, match="retained grid_path requires shape"):
        StageResult(
            stage="raw",
            status="timeout",
            grid_path="grids/raw.npy",
            grid_sha256="a" * 64,
            shape=None,
            exact_valid=None,
        )


def test_changed_tile_metrics_are_exact_and_shape_safe() -> None:
    before = np.zeros((2, 3), dtype=np.int64)
    after = before.copy()
    after[0, 1] = 7

    metrics = changed_tile_metrics(before, after)

    assert metrics == {
        "changed_tile_count": 1,
        "changed_tile_fraction": pytest.approx(1.0 / 6.0),
        "any_change": True,
        "total_tile_count": 6,
    }
    with pytest.raises(ValueError, match="different shapes"):
        changed_tile_metrics(before, np.zeros((3, 2), dtype=np.int64))


def test_local_baseline_config_rejects_graph_or_logic_leakage() -> None:
    config = DiffusionTrainingConfig(
        condition_global_graph_enabled=False,
        graph_conditioning_mode="pooled",
        logic_net_enabled=False,
        graph_to_grid_edge_semantics=False,
        graph_spatial_alignment_weight=0.0,
    )
    assert config.condition_global_graph_enabled is False

    with pytest.raises(ValueError, match="graph_conditioning_mode='pooled'"):
        DiffusionTrainingConfig(
            condition_global_graph_enabled=False,
            graph_conditioning_mode="node_sequence",
            logic_net_enabled=False,
        )
    with pytest.raises(ValueError, match="logic_net_enabled=False"):
        DiffusionTrainingConfig(
            condition_global_graph_enabled=False,
            graph_conditioning_mode="pooled",
            logic_net_enabled=True,
        )


def test_local_conditioning_does_not_call_global_encoder() -> None:
    class LocalOnlyEncoder:
        def encode_local_only(self, **kwargs):
            assert set(kwargs) == {
                "neighbor_latents",
                "boundary_constraints",
                "position",
            }
            return torch.ones(1, 8)

    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.device = torch.device("cpu")
    trainer.config = SimpleNamespace(
        condition_global_graph_enabled=False,
        use_teacher_forced_neighbor_latents=False,
    )
    trainer.condition_encoder = LocalOnlyEncoder()

    condition = trainer._encode_graph_conditioning(
        {
            # Deliberately omit every global graph field.  A global encoder
            # call would fail, which makes this a leakage regression test.
            "boundary_constraints": torch.zeros(8),
            "room_position": torch.zeros(2),
        }
    )

    assert tuple(condition.shape) == (1, 8)


def test_local_inference_conditioning_does_not_consume_graph_fields() -> None:
    class LocalOnlyEncoder:
        output_dim = 8

        def encode_local_only(self, **kwargs):
            assert set(kwargs) == {
                "neighbor_latents",
                "boundary_constraints",
                "position",
            }
            return torch.ones(1, 8)

        def __call__(self, **kwargs):
            raise AssertionError("global conditioning must not run for n_local")

    pipeline = SimpleNamespace(
        device=torch.device("cpu"),
        condition_encoder=LocalOnlyEncoder(),
        room_generator_mode="latent_diffusion",
        diffusion_condition_global_graph_enabled=False,
        use_graph_node_cross_attention=True,
        strict_checkpoint_mode=True,
        diffusion_puzzle_structure_condition_enabled=False,
        masked_room_puzzle_structure_condition_enabled=False,
        diffusion_puzzle_stage_conditioning_enabled=False,
        masked_room_puzzle_stage_conditioning_enabled=False,
        diffusion_puzzle_stage_token_scale=0.2,
        masked_room_puzzle_stage_token_scale=0.2,
        _bump_diagnostic=lambda *_: None,
    )

    condition = _compute_room_condition(
        pipeline,
        neighbor_latents={},
        graph_context={"node_features": "deliberately-invalid-global-data"},
        boundary_constraints=torch.zeros(1, 8),
        position=torch.zeros(1, 2),
    )

    assert tuple(condition.shape) == (1, 8)


def test_run_plan_shares_only_fresh_fold_seed_tokenizer(tmp_path: Path) -> None:
    base = merge_config(yaml_path="configs/zelda_hmolqd.yaml")
    fold = build_leave_one_dungeon_out_folds()[8]
    manifest, configs = build_run_plan(
        base_config=base,
        output_root=tmp_path / "paper",
        training_seeds=(42,),
        folds=(fold,),
        variants=PUBLICATION_VARIANTS,
    )

    assert manifest["tokenizer_run_count"] == 1
    assert manifest["diffusion_run_count"] == 4
    claimed_digest = manifest["protocol_sha256"]
    unsigned = dict(manifest)
    unsigned.pop("protocol_sha256")
    assert payload_sha256(unsigned) == claimed_digest
    tokenizer_checkpoint = manifest["jobs"]["tokenizer"][0]["expected_checkpoint"]
    assert len(manifest["jobs"]["tokenizer"][0]["config_sha256"]) == 64
    assert all(len(job["config_sha256"]) == 64 for job in manifest["jobs"]["diffusion"])
    diffusion_configs = [config for path, config in configs if "diffusion" in path.parts]
    assert len(diffusion_configs) == 4
    for config in diffusion_configs:
        assert config["training"]["stage"] == "diffusion"
        assert config["diffusion"]["vqvae_checkpoint"] == tokenizer_checkpoint
        assert config["runtime"]["auto_resume"] is False
        assert config["runtime"]["resume"] is None
        assert config["diffusion"]["warm_start_checkpoint"] is None
        assert config["dataset"]["test_dungeons"] == [9]
        assert config["runtime"]["cudnn_benchmark"] is False
        assert config["runtime"]["cudnn_deterministic"] is True
        assert config["runtime"]["deterministic_algorithms"] is True


def test_model_variant_parity_guard_rejects_unplanned_confounds() -> None:
    reference = {
        "runtime": {"experiment_name": "a", "output_dir": "a"},
        "diffusion": {
            "checkpoint_dir": "a",
            "condition_global_graph_enabled": False,
            "learning_rate": 1e-4,
        },
        "generation": {"apply_repair": True},
    }
    candidate = {
        "runtime": {"experiment_name": "b", "output_dir": "b"},
        "diffusion": {
            "checkpoint_dir": "b",
            "condition_global_graph_enabled": True,
            "learning_rate": 2e-4,
        },
        "generation": {"apply_repair": True},
    }

    with pytest.raises(ValueError, match="learning_rate"):
        assert_model_variant_config_parity((reference, candidate))


def test_materializer_writes_safe_plan_and_refuses_nonempty_destination(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "paper's"
    fold = build_leave_one_dungeon_out_folds()[0]
    result = materialize_protocol(
        base_config_path="configs/zelda_hmolqd.yaml",
        output_root=output_root,
        training_seeds=(42,),
        folds=(fold,),
        variants=(PUBLICATION_VARIANTS[0],),
    )

    assert result.manifest_path.is_file()
    assert result.command_plan_path.is_file()
    assert result.tokenizer_config_count == 1
    assert result.diffusion_config_count == 1
    assert len(list((output_root / "configs").rglob("*.yaml"))) == 2
    assert "paper''s" in result.command_plan_path.read_text(encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        materialize_protocol(
            base_config_path="configs/zelda_hmolqd.yaml",
            output_root=output_root,
            training_seeds=(42,),
            folds=(fold,),
            variants=(PUBLICATION_VARIANTS[0],),
        )


def test_command_plan_stops_after_native_preflight_failure(tmp_path: Path) -> None:
    import shutil
    import subprocess

    shell = shutil.which("pwsh") or shutil.which("powershell")
    if shell is None:
        pytest.skip("PowerShell is required for the native exit-code regression")
    result = materialize_protocol(
        base_config_path="configs/zelda_hmolqd.yaml", output_root=tmp_path / "fresh",
        training_seeds=(42,), folds=(build_leave_one_dungeon_out_folds()[0],),
        variants=(PUBLICATION_VARIANTS[0],),
    )
    from experiments.paper_v1.run_protocol import _powershell_quote
    probe = tmp_path / "probe.ps1"
    probe.write_text(
        "$global:invocations = 0\n"
        "function python { $global:invocations += 1; $global:LASTEXITCODE = 7 }\n"
        f"try {{ . {_powershell_quote(result.command_plan_path)}; exit 3 }}\n"
        "catch { if ($global:invocations -eq 1 -and $_.Exception.Message -like '*preflight failed*') { exit 0 }; exit 4 }\n",
        encoding="utf-8",
    )
    completed = subprocess.run([shell, "-NoProfile", "-NonInteractive", "-File", str(probe)],
                               capture_output=True, text=True, timeout=30, check=False)
    assert completed.returncode == 0, completed.stderr + completed.stdout


def test_stage_bridge_retains_and_validates_all_generation_stages(tmp_path: Path) -> None:
    rooms = {
        0: SimpleNamespace(
            raw_neural_grid=np.zeros((4, 4), dtype=np.int32),
            neural_grid=np.ones((4, 4), dtype=np.int32),
            room_grid=np.full((4, 4), 2, dtype=np.int32),
        ),
        1: SimpleNamespace(
            raw_neural_grid=np.zeros((4, 4), dtype=np.int32),
            neural_grid=np.ones((4, 4), dtype=np.int32),
            room_grid=np.full((4, 4), 2, dtype=np.int32),
        ),
    }
    physical_graph = nx.Graph()
    physical_graph.add_edge(0, 1)
    generated = SimpleNamespace(
        rooms=rooms,
        stitched_layout=SimpleNamespace(slot_positions={0: (0, 0), 1: (0, 1)}),
        physical_mission_graph=physical_graph,
        dungeon_grid=np.full((4, 8), 3, dtype=np.int32),
        metrics={"generation_time_sec": 1.25, "not_timing": 7},
    )

    record = build_publication_result_record(
        generated,
        validator=lambda _grid, stage: {
            "is_exact": True,
            "solvable": stage != "raw",
            "termination_status": "goal_reached" if stage != "raw" else "frontier_exhausted",
        },
        output_dir=tmp_path,
        run_id="run",
        git_commit="deadbeef",
        config_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        protocol_sha256="c" * 64,
        outer_fold="holdout_dungeon_1",
        train_seed=42,
        sampling_seed=100,
        mission_graph_id="graph_1",
        model_id="g_graph",
        inference_variant="unguided",
    )

    assert record.stages["raw"].status == "invalid"
    assert record.stages["constrained"].status == "valid"
    assert record.stages["repaired"].status == "valid"
    assert record.stages["compiled"].status == "valid"
    assert record.repair_metrics["raw_to_constrained"]["changed_tile_count"] > 0
    assert record.repair_metrics["constrained_to_repaired"]["changed_tile_count"] > 0
    assert record.repair_metrics["repaired_to_compiled"]["changed_tile_count"] > 0
    assert set(path.name for path in (tmp_path / "grids").glob("*.npy")) == {
        "raw.npy",
        "constrained.npy",
        "repaired.npy",
        "compiled.npy",
    }


def test_publication_aggregation_keeps_failures_in_denominator() -> None:
    valid_record = _record({"raw": _stage("raw", "valid")})
    failed_record = replace(
        valid_record,
        sampling_seed=101,
        generation_status="generation_error",
        stages={
            "raw": StageResult(
                stage="raw",
                status="not_run",
                grid_path=None,
                exact_valid=None,
            )
        },
        error_type="RuntimeError",
    )

    rows = aggregate_publication_records([valid_record, failed_record])
    raw = next(row for row in rows if row["stage"] == "raw")

    assert raw["attempted_records"] == 2
    assert raw["valid_count"] == 1
    assert raw["indeterminate_or_missing_count"] == 1
    assert raw["exact_valid_rate_all_attempts"] == pytest.approx(0.5)
    assert raw["exact_valid_rate_evaluable_only"] == pytest.approx(1.0)


def test_paired_validity_aggregates_repeats_within_outer_fold() -> None:
    records = []
    for fold_index in (1, 2):
        for sample_seed in (10, 11):
            baseline = replace(
                _record({"raw": _stage("raw", "invalid")}),
                outer_fold=f"holdout_dungeon_{fold_index}",
                sampling_seed=sample_seed,
                mission_graph_id=f"graph_{fold_index}",
                model_id="n_local",
                inference_variant="unguided",
            )
            comparison = replace(
                baseline,
                model_id="g_graph",
                stages={**baseline.stages, "raw": _stage("raw", "valid")},
            )
            records.extend((baseline, comparison))

    result = paired_exact_validity_by_fold(
        records,
        baseline_model_id="n_local",
        comparison_model_id="g_graph",
        bootstrap_samples=200,
        seed=7,
    )

    assert result["analysis_status"] == "complete"
    assert result["paired_record_count"] == 4
    assert result["outer_fold_count"] == 2
    assert result["mean_fold_delta"] == pytest.approx(1.0)
    assert result["exact_random_sign_p_value"] == pytest.approx(0.5)
    assert result["paired_effect_size_dz"] is None


def test_paired_guidance_analysis_matches_conditions_within_model() -> None:
    records = []
    for fold_index in (1, 2):
        unguided = replace(
            _record({"raw": _stage("raw", "invalid")}),
            outer_fold=f"holdout_dungeon_{fold_index}",
            mission_graph_id=f"graph_{fold_index}",
            model_id="g_l_local_logic",
            inference_variant="unguided",
        )
        guided = replace(
            unguided,
            inference_variant="logic_guided",
            stages={**unguided.stages, "raw": _stage("raw", "valid")},
        )
        records.extend((unguided, guided))

    result = paired_guidance_validity_by_fold(
        records,
        model_id="g_l_local_logic",
        bootstrap_samples=200,
        seed=7,
    )

    assert result["analysis_status"] == "complete"
    assert result["paired_record_count"] == 2
    assert result["outer_fold_count"] == 2
    assert result["mean_fold_delta"] == pytest.approx(1.0)


def test_dataset_manifest_verification_is_content_addressed(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    source = data_root / "level.txt"
    source.write_text("abc\n", encoding="utf-8")
    files = [
        {
            "path": "level.txt",
            "size_bytes": source.stat().st_size,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    ]
    manifest = {
        "schema_version": "1.0",
        "generated_at_utc": "irrelevant-to-content-address",
        "files": files,
        "content_sha256": dataset_content_sha256(files),
    }
    (data_root / "dataset_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    verified = verify_dataset_manifest(data_root)
    assert verified["verification_status"] == "verified"
    assert verified["content_sha256"] == manifest["content_sha256"]

    source.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mismatched"):
        verify_dataset_manifest(data_root)


def test_publication_guidance_card_and_validator_enforce_gap_traceability() -> None:
    from scripts.validate_publication_readiness import publication_card_template, validate_card

    card_path = Path(__file__).resolve().parents[1] / "docs" / "publication_guidance_card.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))

    assert validate_card(card) == []
    assert {"RG1", "RG2", "RG3", "RG4"}.issubset(set(card["research_gaps"].keys()))

    missing_gaps = publication_card_template()
    missing_gaps.pop("research_gaps")
    assert any("research_gaps" in msg for msg in validate_card(missing_gaps))

    dangling_ref = publication_card_template()
    dangling_ref["problem_statements"]["PS1"]["research_gap"] = ["RG999"]
    assert any("unknown research_gap" in msg for msg in validate_card(dangling_ref))


def test_plan_attempts_supports_masked_room_branch_without_tokenizer_dependency(tmp_path: Path, monkeypatch) -> None:
    import yaml
    from experiments.paper_v1.plan_attempts import build_sampling_plan
    from src.publication.evaluation_design import seal_evaluation_design
    from src.publication.protocol import seal_protocol_manifest

    fold = build_leave_one_dungeon_out_folds()[0]
    output_root = tmp_path / "masked_plan"
    mission_path = tmp_path / "mission_0.json"
    mission_graph = nx.DiGraph()
    mission_graph.add_node(0, is_start=True)
    mission_graph.add_node(1, is_triforce=True)
    mission_graph.add_edge(0, 1)
    mission_path.write_text(json.dumps(nx.node_link_data(mission_graph, edges="links")), encoding="utf-8")

    manifest, configs = build_run_plan(
        base_config=merge_config(),
        output_root=output_root,
        folds=(fold,),
        training_seeds=(42,),
        model_branch="masked_room",
    )
    for path, config in configs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    for job in manifest["jobs"]["diffusion"]:
        ckpt = Path(job["expected_checkpoint"])
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text("synthetic masked_room checkpoint", encoding="utf-8")

    mission_bank = {fold.fold_id: {"m0": str(mission_path)}}
    gen_kwargs = {"num_diffusion_steps": 4, "logic_guidance_scale": 0.0}
    manifest["dataset_provenance"] = {"verification_status": "verified", "content_sha256": "a" * 64}
    manifest["execution_provenance"] = {"git_commit": "deadbeef"}
    manifest["evaluation_design"] = seal_evaluation_design(
        mission_bank=mission_bank,
        sampling_seeds=(11,),
        generation_kwargs=gen_kwargs,
        fold_ids=(fold.fold_id,),
    )
    manifest = seal_protocol_manifest(manifest)

    monkeypatch.setattr("experiments.paper_v1.plan_attempts.verify_execution_provenance", lambda *a, **k: None)
    monkeypatch.setattr(
        "experiments.paper_v1.plan_attempts.verify_dataset_manifest",
        lambda *a, **k: {"verification_status": "verified", "content_sha256": "a" * 64},
    )

    plan = build_sampling_plan(
        training_manifest=manifest,
        protocol_sha256=manifest["protocol_sha256"],
        mission_bank=mission_bank,
        sampling_seeds=(11,),
        generation_kwargs=gen_kwargs,
        git_commit="deadbeef",
    )
    neural_attempts = [a for a in plan["attempts"] if a["metadata"]["model_id"] in {v.model_id for v in PUBLICATION_VARIANTS}]
    assert len(neural_attempts) == 4
    for attempt in neural_attempts:
        assert "masked_room_checkpoint" in attempt["artifacts"]
        assert "vqvae_checkpoint" not in attempt["artifacts"]
        assert "diffusion_checkpoint" not in attempt["artifacts"]
