import logging
from pathlib import Path

import pytest
import torch

from src.utils.checkpoint import (
    LATEST_RESUME_FILENAME,
    atomic_torch_save,
    checkpoint_directory_size_bytes,
    checkpoint_size_bytes,
    enforce_checkpoint_storage_budget,
    format_bytes,
    prune_checkpoints,
    resolve_resume_checkpoint,
    write_checkpoint_metadata,
)


def test_prune_checkpoints_keep_last_zero_removes_all(tmp_path: Path):
    for idx in range(3):
        ckpt = tmp_path / f"resume_epoch_{idx:04d}.pth"
        atomic_torch_save({"epoch": idx}, str(ckpt))
        (tmp_path / f"{ckpt.name}.meta.json").write_text("{}", encoding="utf-8")

    removed = prune_checkpoints(
        checkpoint_dir=str(tmp_path),
        pattern="resume_epoch_*.pth",
        keep_last=0,
    )

    assert len(removed) == 3
    assert list(tmp_path.glob("resume_epoch_*.pth")) == []
    assert list(tmp_path.glob("resume_epoch_*.pth.meta.json")) == []


def test_resolve_resume_checkpoint_uses_explicit_then_latest(tmp_path: Path):
    explicit = tmp_path / "manual_resume.pth"
    latest = tmp_path / LATEST_RESUME_FILENAME
    atomic_torch_save({"epoch": 2}, str(explicit))
    atomic_torch_save({"epoch": 5}, str(latest))

    assert resolve_resume_checkpoint(
        explicit_path=str(explicit),
        checkpoint_dir=str(tmp_path),
        auto_resume=True,
    ) == explicit
    assert resolve_resume_checkpoint(
        explicit_path=None,
        checkpoint_dir=str(tmp_path),
        auto_resume=True,
    ) == latest
    assert resolve_resume_checkpoint(
        explicit_path=None,
        checkpoint_dir=str(tmp_path),
        auto_resume=False,
    ) is None


def test_resolve_resume_checkpoint_rejects_missing_explicit_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        resolve_resume_checkpoint(
            explicit_path=str(tmp_path / "missing.pth"),
            checkpoint_dir=str(tmp_path),
            auto_resume=True,
        )


def test_checkpoint_metadata_records_file_size(tmp_path: Path):
    ckpt = tmp_path / "sample_resume.pth"
    atomic_torch_save({"epoch": 7, "value": torch.arange(8)}, str(ckpt))

    meta_path = write_checkpoint_metadata(str(ckpt), model_type="unit_test")
    metadata = meta_path.read_text(encoding="utf-8")

    assert "\"file_size_bytes\"" in metadata
    assert "\"file_size_human\"" in metadata
    assert checkpoint_size_bytes(ckpt) > 0
    assert checkpoint_directory_size_bytes(tmp_path) >= checkpoint_size_bytes(ckpt)
    assert format_bytes(checkpoint_size_bytes(ckpt)).endswith(("B", "KB", "MB", "GB", "TB"))


def test_enforce_checkpoint_storage_budget_removes_only_retained_resume_files(tmp_path: Path):
    latest = tmp_path / LATEST_RESUME_FILENAME
    periodic_old = tmp_path / "resume_epoch_0001.pth"
    periodic_new = tmp_path / "resume_epoch_0002.pth"
    best = tmp_path / "best_model.pth"

    atomic_torch_save({"payload": torch.arange(256)}, str(latest))
    atomic_torch_save({"payload": torch.arange(4096)}, str(periodic_old))
    atomic_torch_save({"payload": torch.arange(4096)}, str(periodic_new))
    atomic_torch_save({"payload": torch.arange(256)}, str(best))
    write_checkpoint_metadata(str(periodic_old), model_type="unit_test")
    write_checkpoint_metadata(str(periodic_new), model_type="unit_test")

    current_gb = checkpoint_directory_size_bytes(tmp_path) / float(1024 ** 3)
    result = enforce_checkpoint_storage_budget(
        logging.getLogger("checkpoint_test"),
        checkpoint_dir=tmp_path,
        budget_gb=current_gb * 0.5,
        warning_fraction=0.8,
        cleanup_enabled=True,
        cleanup_target_fraction=0.4,
        removable_patterns=("resume_epoch_*.pth",),
    )

    assert result["removed"]
    assert latest.exists()
    assert best.exists()
    assert checkpoint_directory_size_bytes(tmp_path) <= int(current_gb * 0.5 * (1024 ** 3))
