"""
Block-0 dataset/process audit for local VGLC Zelda data.

Usage:
    python scripts/audit_block0_data.py --pretty
    python scripts/audit_block0_data.py --output results/block0_audit.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.benchmark_suite import audit_block0_dataset


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Block-0 VGLC inputs and parsed graph/room stats.")
    parser.add_argument("--data-root", type=Path, default=Path("Data") / "The Legend of Zelda")
    parser.add_argument("--max-rooms", type=int, default=5000)
    parser.add_argument("--graph-limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--pretty", action="store_true")
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    payload = audit_block0_dataset(
        data_root=args.data_root,
        max_rooms=max(1, int(args.max_rooms)),
        graph_limit=args.graph_limit,
    )
    text = json.dumps(payload, indent=2 if bool(args.pretty) else None)
    print(text)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
