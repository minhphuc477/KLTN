"""
Evaluate the hard A* oracle against P-CBS on one dungeon instance.

This script is report-oriented: it emits JSON and a markdown table that can be
ported directly into the thesis or paper draft.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.definitions import CHAR_TO_SEMANTIC
from src.evaluation.pcbs_validation import (
    build_ieee_markdown_table,
    evaluate_astar_vs_pcbs,
)
from src.zelda_data.zelda_core import ZeldaDungeonAdapter


def _load_grid_from_txt(path: Path) -> np.ndarray:
    lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Grid text file is empty: {path}")
    width = max(len(line) for line in lines)
    grid = np.zeros((len(lines), width), dtype=np.int32)
    for row, line in enumerate(lines):
        for col, char in enumerate(line):
            grid[row, col] = int(CHAR_TO_SEMANTIC.get(char, CHAR_TO_SEMANTIC["-"]))
    return grid


def _resolve_source(args: argparse.Namespace) -> tuple[Any, str]:
    if args.grid_txt:
        grid_path = Path(args.grid_txt)
        return _load_grid_from_txt(grid_path), grid_path.stem

    adapter = ZeldaDungeonAdapter(args.data_root)
    dungeon = adapter.load_dungeon(int(args.dungeon_num), variant=int(args.variant))
    stitched = adapter.stitch_dungeon(dungeon)
    return stitched, f"D{int(args.dungeon_num)}_v{int(args.variant)}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare A* and P-CBS on one dungeon")
    parser.add_argument("--grid-txt", type=str, default="", help="Optional VGLC-style grid text file")
    parser.add_argument("--data-root", type=str, default="Data/The Legend of Zelda", help="Dataset root for dungeon loading")
    parser.add_argument("--dungeon-num", type=int, default=1, help="Zelda dungeon number when --grid-txt is not used")
    parser.add_argument("--variant", type=int, default=1, help="Dungeon variant when --grid-txt is not used")
    parser.add_argument("--persona", type=str, default="novice", help="P-CBS persona")
    parser.add_argument("--timeout-astar", type=int, default=200000, help="A* timeout")
    parser.add_argument("--timeout-pcbs", type=int, default=50000, help="P-CBS timeout")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="results/pcbs_vs_astar", help="Output directory")
    args = parser.parse_args()

    source, map_name = _resolve_source(args)
    result = evaluate_astar_vs_pcbs(
        source,
        persona=args.persona,
        timeout_astar=args.timeout_astar,
        timeout_pcbs=args.timeout_pcbs,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{map_name}_{args.persona}_pcbs_vs_astar.json"
    md_path = output_dir / f"{map_name}_{args.persona}_pcbs_vs_astar.md"

    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md_path.write_text(build_ieee_markdown_table(result, map_name=map_name), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
