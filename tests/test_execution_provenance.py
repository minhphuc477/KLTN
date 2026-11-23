"""Code edits, added files and late mission selection invalidate sealed runs."""

from types import SimpleNamespace

import pytest

from src.publication import execution_provenance as provenance
from src.publication.evaluation_design import seal_evaluation_design, verify_evaluation_design


@pytest.mark.parametrize("mutation", ["edit", "add", "delete", "environment", "commit"])
def test_execution_snapshot_detects_drift_including_uncommitted_code(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(provenance.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="a" * 40))
    monkeypatch.setattr(provenance, "environment_snapshot", lambda: {"python": "synthetic"})
    source = tmp_path / "src"
    source.mkdir()
    module = source / "model.py"
    module.write_text("value = 1\n")
    snapshot = provenance.capture_execution_provenance(tmp_path)
    provenance.verify_execution_provenance(snapshot, tmp_path)
    if mutation == "edit":
        module.write_text("value = 2\n")
    elif mutation == "add":
        (source / "new.py").write_text("value = 3\n")
    elif mutation == "delete":
        module.unlink()
    elif mutation == "environment":
        monkeypatch.setattr(provenance, "environment_snapshot", lambda: {"python": "different"})
    else:
        monkeypatch.setattr(provenance.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="b" * 40))
    with pytest.raises(ValueError):
        provenance.verify_execution_provenance(snapshot, tmp_path)


def test_evaluation_freeze_detects_changed_mission(tmp_path):
    mission = tmp_path / "mission.json"
    mission.write_text('{"directed": true, "multigraph": false, "nodes": [{"id": 0}], "links": []}')
    design = seal_evaluation_design(mission_bank={"fold": {"mission": mission}}, sampling_seeds=[4, 5],
                                    generation_kwargs={"num_diffusion_steps": 20}, fold_ids=["fold"])
    verify_evaluation_design(design, ["fold"])
    mission.write_text('{"directed": true, "multigraph": false, "nodes": [{"id": 0}, {"id": 1}], "links": []}')
    with pytest.raises(ValueError, match="changed after sealing"):
        verify_evaluation_design(design, ["fold"])


def test_file_sha256_stat_cache_tracks_large_file_mutations(tmp_path):
    import os
    from src.publication.dataset_provenance import file_sha256

    blob = tmp_path / "large_ckpt.bin"
    payload_a = b"A" * 1_050_000
    payload_b = b"B" * 1_050_000
    blob.write_bytes(payload_a)
    digest_a = file_sha256(blob)
    assert file_sha256(blob) == digest_a

    blob.write_bytes(payload_b)
    st = blob.stat()
    os.utime(blob, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    digest_b = file_sha256(blob)
    assert digest_b != digest_a
