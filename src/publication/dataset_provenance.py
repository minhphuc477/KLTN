"""Content-addressed provenance for the Zelda publication corpus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from .protocol import canonical_json


DATASET_MANIFEST_NAME = "dataset_manifest.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_dataset_files(data_root: Path) -> list[Path]:
    """Return regular corpus files in stable relative-path order."""
    root = Path(data_root)
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.name != DATASET_MANIFEST_NAME
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def dataset_file_records(data_root: Path) -> list[Dict[str, Any]]:
    root = Path(data_root)
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": int(path.stat().st_size),
            "sha256": file_sha256(path),
        }
        for path in iter_dataset_files(root)
    ]


def dataset_content_sha256(files: Iterable[Mapping[str, Any]]) -> str:
    """Hash only stable file identity, not timestamps or machine-local roots."""
    normalized = [
        {
            "path": str(record["path"]),
            "size_bytes": int(record["size_bytes"]),
            "sha256": str(record["sha256"]).lower(),
        }
        for record in files
    ]
    normalized.sort(key=lambda record: record["path"])
    return hashlib.sha256(canonical_json({"files": normalized}).encode("utf-8")).hexdigest()


def verify_dataset_manifest(
    data_root: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Verify every current corpus file against its locked manifest."""
    root = Path(data_root).resolve()
    path = (
        root / DATASET_MANIFEST_NAME
        if manifest_path is None
        else Path(manifest_path).resolve()
    )
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    if not path.is_file():
        raise FileNotFoundError(f"Dataset manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_files = payload.get("files")
    if not isinstance(raw_files, list):
        raise ValueError("Dataset manifest files must be a list.")

    locked: dict[str, Mapping[str, Any]] = {}
    for record in raw_files:
        if not isinstance(record, Mapping):
            raise ValueError("Each dataset manifest file entry must be a mapping.")
        relative = Path(str(record.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"Unsafe dataset manifest path: {relative!s}")
        key = relative.as_posix()
        if key in locked:
            raise ValueError(f"Duplicate dataset manifest path: {key}")
        locked[key] = record

    current = {record["path"]: record for record in dataset_file_records(root)}
    missing = sorted(set(locked) - set(current))
    unexpected = sorted(set(current) - set(locked))
    mismatched = sorted(
        key
        for key in set(locked) & set(current)
        if int(locked[key].get("size_bytes", -1)) != current[key]["size_bytes"]
        or str(locked[key].get("sha256", "")).lower() != current[key]["sha256"]
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            "Dataset manifest verification failed: "
            f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}."
        )

    content_digest = dataset_content_sha256(current.values())
    claimed_digest = payload.get("content_sha256")
    if claimed_digest is not None and str(claimed_digest).lower() != content_digest:
        raise ValueError(
            "Dataset manifest content_sha256 does not match its verified file records."
        )
    return {
        "verification_status": "verified",
        "manifest_schema_version": str(payload.get("schema_version", "unknown")),
        "file_count": len(current),
        "content_sha256": content_digest,
    }


__all__ = [
    "DATASET_MANIFEST_NAME",
    "dataset_content_sha256",
    "dataset_file_records",
    "file_sha256",
    "iter_dataset_files",
    "verify_dataset_manifest",
]
