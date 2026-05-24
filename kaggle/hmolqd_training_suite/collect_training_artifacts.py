#!/usr/bin/env python
"""Collect Kaggle training metadata and optionally package key artifacts."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any


SUMMARY_NAMES = {
    "vqvae_run_summary.json",
    "vqvae_training_history.json",
    "training_hyperparameter_check.json",
    "resolved_config.yaml",
    "run_metadata.json",
}
CHECKPOINT_SUFFIXES = (".pth", ".pt")


def _json_or_none(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.relative_to(root)),
        "size_bytes": int(stat.st_size),
        "size_mb": round(stat.st_size / (1024.0 * 1024.0), 4),
    }


def build_manifest(run_root: Path, *, include_checkpoints: bool) -> dict[str, Any]:
    run_root = run_root.resolve()
    files = []
    summaries = []
    checkpoints = []
    for path in sorted(run_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(run_root)
        is_summary = path.name in SUMMARY_NAMES or path.name.endswith(".meta.json") or "metrics" in path.name
        is_checkpoint = path.suffix.lower() in CHECKPOINT_SUFFIXES
        if is_summary:
            record = _file_record(path, run_root)
            payload = _json_or_none(path)
            if payload is not None and path.name == "vqvae_run_summary.json":
                record["summary"] = payload
            summaries.append(record)
            files.append(record)
        elif is_checkpoint:
            record = _file_record(path, run_root)
            checkpoints.append(record)
            if include_checkpoints and (
                "best" in path.stem
                or "final" in path.stem
                or path.name in {"vqvae_pretrained.pth", "best_model.pth", "final_model.pth"}
            ):
                files.append(record)
        elif path.suffix.lower() in {".yaml", ".yml", ".json", ".csv", ".md", ".log"}:
            record = _file_record(path, run_root)
            files.append(record)

    return {
        "run_root": str(run_root),
        "include_checkpoints": bool(include_checkpoints),
        "summary_files": summaries,
        "checkpoints": checkpoints,
        "packaged_files": files,
        "total_checkpoint_size_mb": round(sum(item["size_bytes"] for item in checkpoints) / (1024.0 * 1024.0), 4),
    }


def write_zip(run_root: Path, manifest: dict[str, Any], zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
        zf.writestr("kaggle_training_manifest.json", manifest_bytes)
        for record in manifest["packaged_files"]:
            path = run_root / record["path"]
            if path.exists() and path.is_file():
                zf.write(path, arcname=record["path"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--zip-name", type=str, default="hmolqd_kaggle_artifacts.zip")
    parser.add_argument("--include-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-zip", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir is not None else run_root
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(run_root, include_checkpoints=bool(args.include_checkpoints))
    manifest_path = out_dir / "kaggle_training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")
    if not args.no_zip:
        zip_path = out_dir / args.zip_name
        write_zip(run_root, manifest, zip_path)
        print(f"Wrote artifact zip: {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
