#!/usr/bin/env python
"""Verify that a Kaggle H-MOLQD run produced all expected checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


TOKENIZER_REQUIRED = (
    "checkpoints/vqvae/vqvae_pretrained.pth",
)

BRANCH_REQUIRED = (
    "checkpoints/diffusion/best_model.pth",
    "checkpoints/diffusion/final_model.pth",
    "checkpoints/fast_sampler/fast_sampler_best.pth",
    "checkpoints/fast_sampler/fast_sampler_final.pth",
    "checkpoints/masked_room/masked_room_best.pth",
    "checkpoints/masked_room/masked_room_final.pth",
)

BRANCH_OPTIONAL = (
    "checkpoints/diffusion/best_logic_model.pth",
    "checkpoints/fast_sampler/fast_sampler_best_reselected.pth",
)


def _split_words(raw: str | Iterable[str]) -> list[str]:
    if isinstance(raw, str):
        return [part.strip() for part in raw.split() if part.strip()]
    result: list[str] = []
    for item in raw:
        result.extend(_split_words(str(item)))
    return result


def _record(path: Path, *, required: bool) -> dict[str, Any]:
    exists = path.is_file()
    size_bytes = path.stat().st_size if exists else 0
    return {
        "path": str(path),
        "required": bool(required),
        "exists": bool(exists),
        "size_bytes": int(size_bytes),
        "size_mb": round(size_bytes / (1024.0 * 1024.0), 4),
    }


def build_report(run_root: Path, tokenizers: list[str], branches: list[str]) -> dict[str, Any]:
    run_root = run_root.resolve()
    entries: list[dict[str, Any]] = []

    for tokenizer in tokenizers:
        tokenizer_root = run_root / "tokenizers" / tokenizer
        for rel_path in TOKENIZER_REQUIRED:
            entries.append(
                {
                    "kind": "tokenizer",
                    "tokenizer": tokenizer,
                    "branch": None,
                    **_record(tokenizer_root / rel_path, required=True),
                }
            )

    for tokenizer in tokenizers:
        for branch in branches:
            run_name = f"{tokenizer}_{branch}"
            branch_root = run_root / "downstream" / run_name
            for rel_path in BRANCH_REQUIRED:
                entries.append(
                    {
                        "kind": "branch",
                        "tokenizer": tokenizer,
                        "branch": branch,
                        **_record(branch_root / rel_path, required=True),
                    }
                )
            for rel_path in BRANCH_OPTIONAL:
                entries.append(
                    {
                        "kind": "branch",
                        "tokenizer": tokenizer,
                        "branch": branch,
                        **_record(branch_root / rel_path, required=False),
                    }
                )

    missing_required = [entry for entry in entries if entry["required"] and not entry["exists"]]
    present_required = [entry for entry in entries if entry["required"] and entry["exists"]]
    return {
        "run_root": str(run_root),
        "tokenizers": tokenizers,
        "branches": branches,
        "required_count": len(present_required) + len(missing_required),
        "present_required_count": len(present_required),
        "missing_required_count": len(missing_required),
        "missing_required": missing_required,
        "entries": entries,
        "complete": not missing_required,
    }


def write_tsv(report: dict[str, Any], output: Path) -> None:
    rows = ["kind\ttokenizer\tbranch\trequired\texists\tsize_mb\tpath"]
    for entry in report["entries"]:
        rows.append(
            "\t".join(
                [
                    str(entry["kind"]),
                    str(entry["tokenizer"]),
                    "" if entry["branch"] is None else str(entry["branch"]),
                    str(entry["required"]).lower(),
                    str(entry["exists"]).lower(),
                    str(entry["size_mb"]),
                    str(entry["path"]),
                ]
            )
        )
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--tokenizers", nargs="+", required=True)
    parser.add_argument("--branches", nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, default=None)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when any required checkpoint is missing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tokenizers = _split_words(args.tokenizers)
    branches = _split_words(args.branches)
    report = build_report(args.run_root, tokenizers, branches)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.output_tsv is not None:
        args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
        write_tsv(report, args.output_tsv)

    print(
        "[checkpoint-audit] required={present}/{total} missing={missing} complete={complete}".format(
            present=report["present_required_count"],
            total=report["required_count"],
            missing=report["missing_required_count"],
            complete=str(report["complete"]).lower(),
        )
    )
    if report["missing_required"]:
        print("[checkpoint-audit] missing required checkpoints:", file=sys.stderr)
        for entry in report["missing_required"]:
            print(f"  - {entry['path']}", file=sys.stderr)
    return 1 if args.strict and report["missing_required"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
