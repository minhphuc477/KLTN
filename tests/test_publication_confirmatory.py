"""Primary inference follows the sealed endpoint, not an analysis default."""

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from src.publication.attempt_runner import AttemptLedger, AttemptSpec, execute_attempts
from src.publication.confirmatory import analyze_primary_ledger
from src.publication.protocol import PUBLICATION_VARIANTS, build_protocol_manifest, build_leave_one_dungeon_out_folds, seal_protocol_manifest


def _ledger(tmp_path, *, omit_fold=False):
    protocol = build_protocol_manifest(training_seeds=(42,), folds=build_leave_one_dungeon_out_folds()[:2])
    specs = []
    for fold in protocol["folds"][:1 if omit_fold else 2]:
        for variant in PUBLICATION_VARIANTS:
            specs.append(AttemptSpec(dict(run_id="fresh", git_commit="synthetic-source", config_sha256="a" * 64,
                dataset_manifest_sha256="b" * 64, protocol_sha256=protocol["protocol_sha256"],
                outer_fold=fold["fold_id"], train_seed=42, sampling_seed=1, mission_graph_id="one",
                model_id=variant.model_id, inference_variant="unguided"), {}, {}))
    ledger = AttemptLedger.create(tmp_path / "ledger", specs)
    def generate(spec):
        grid = np.ones((4, 4), dtype=np.int32)
        return SimpleNamespace(rooms={0: SimpleNamespace(raw_neural_grid=grid, neural_grid=grid, room_grid=grid)},
            stitched_layout=SimpleNamespace(slot_positions={0: (0, 0)}), physical_mission_graph=nx.empty_graph(1),
            dungeon_grid=grid, metrics={})
    def factory(spec, generated):
        def validate(grid, stage):
            compiled_success = spec.metadata["model_id"] != "n_local"
            accepted = compiled_success if stage == "compiled" else not compiled_success
            return dict(is_exact=True, accepted=accepted, termination_status="solved" if accepted else "invalid")
        return validate
    execute_attempts(ledger, generate=generate, validator_factory=factory, seed_fn=lambda seed: None)
    return ledger, protocol


def test_primary_family_uses_compiled_evidence_even_when_raw_effect_is_opposite(tmp_path):
    ledger, protocol = _ledger(tmp_path)
    analysis = analyze_primary_ledger(ledger, protocol, protocol_sha256=protocol["protocol_sha256"], bootstrap_samples=20)
    assert analysis["analysis_status"] == "descriptive_only_dependent_folds"
    assert analysis["stage"] == "compiled"
    assert analysis["contrasts"]["g_graph_minus_n_local"]["mean_fold_delta"] == 1.0
    assert analysis["multiplicity"] == {}
    assert analysis["confirmatory_inference_permitted"] is False
    assert len(analysis["training_overlap"][0]["shared_training_dungeons"]) == 7
    assert all("exact_random_sign_p_value" not in item and "bootstrap_ci_95" not in item
               for item in analysis["contrasts"].values())


def test_complete_ledger_cannot_hide_a_whole_missing_protocol_fold(tmp_path):
    ledger, protocol = _ledger(tmp_path, omit_fold=True)
    with pytest.raises(ValueError, match="every declared fold"):
        analyze_primary_ledger(ledger, protocol, protocol_sha256=protocol["protocol_sha256"])


def test_complete_ledger_cannot_omit_same_sampling_seed_from_every_model(tmp_path):
    from dataclasses import replace
    ledger, protocol = _ledger(tmp_path)
    protocol["evaluation_design"] = {"independent_controls": [], "sampling_seeds": [1, 2],
        "missions": {fold["fold_id"]: {"one": {}} for fold in protocol["folds"]}}
    protocol = seal_protocol_manifest(protocol)
    records = [replace(record, protocol_sha256=protocol["protocol_sha256"]) for record in ledger.complete_records()]
    with pytest.raises(ValueError, match="complete pre-training evaluation matrix"):
        analyze_primary_ledger(SimpleNamespace(complete_records=lambda: records), protocol,
                               protocol_sha256=protocol["protocol_sha256"])


@pytest.mark.parametrize("defect", ["empty", "missing", "duplicate"])
def test_primary_analysis_rejects_malformed_sealed_job_matrix(tmp_path, defect):
    ledger, protocol = _ledger(tmp_path)
    records = ledger.complete_records()
    jobs = [dict(fold_id=r.outer_fold, training_seed=r.train_seed, model_id=r.model_id,
                 config_sha256=r.config_sha256) for r in records]
    protocol["jobs"] = {"diffusion": [] if defect == "empty" else jobs[:-1] if defect == "missing" else jobs + jobs[:1]}
    protocol = seal_protocol_manifest(protocol)
    from dataclasses import replace
    records = [replace(r, protocol_sha256=protocol["protocol_sha256"]) for r in records]
    with pytest.raises(ValueError, match="exactly once"):
        analyze_primary_ledger(SimpleNamespace(complete_records=lambda: records), protocol,
                               protocol_sha256=protocol["protocol_sha256"])
