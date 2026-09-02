"""Lock Zelda dataset provenance and train/test split into a JSON manifest."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

from src.publication.dataset_provenance import (
    dataset_content_sha256,
    dataset_file_records,
)


DEFAULT_DATA_ROOT = Path("Data") / "The Legend of Zelda"
DEFAULT_OUTPUT = Path("Data") / "The Legend of Zelda" / "dataset_manifest.json"


def _parse_ids(value: str) -> List[int]:
    ids: List[int] = []
    for part in str(value or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        ids.append(int(part))
    return ids


def build_manifest(
    data_root: Path,
    *,
    train_dungeon_ids: Iterable[int] = range(1, 9),
    test_dungeon_ids: Iterable[int] = (9,),
) -> dict:
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")

    files = dataset_file_records(data_root)

    return {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "VGLC The Legend of Zelda",
        "data_root": data_root.as_posix(),
        "split": {
            "train_dungeon_ids": [int(v) for v in train_dungeon_ids],
            "test_dungeon_ids": [int(v) for v in test_dungeon_ids],
        },
        "file_count": len(files),
        "content_sha256": dataset_content_sha256(files),
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-dungeons", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--test-dungeons", default="9")
    args = parser.parse_args()

    manifest = build_manifest(
        args.data_root,
        train_dungeon_ids=_parse_ids(args.train_dungeons),
        test_dungeon_ids=_parse_ids(args.test_dungeons),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "file_count": manifest["file_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
