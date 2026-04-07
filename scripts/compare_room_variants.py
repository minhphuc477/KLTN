"""
Compare two exported room-variant directories room-by-room.

Typical use:
python scripts/compare_room_variants.py ^
  --baseline outputs/.../diffusion_cfg3_logic0_steps50 ^
  --candidate outputs/.../fast_cfg3_logic0_steps4 ^
  --output-dir outputs/.../room_diff_audit
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_room_text(path: Path) -> List[str]:
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.strip() != ""]


def _char_color(ch: str) -> Tuple[int, int, int]:
    palette = {
        "-": (20, 20, 24),
        "F": (210, 197, 156),
        "W": (24, 24, 28),
        "B": (98, 78, 58),
        "D": (246, 214, 52),
        "M": (200, 44, 44),
        "S": (68, 175, 98),
        "P": (168, 92, 198),
        "O": (255, 166, 71),
        "I": (110, 203, 220),
        "K": (255, 166, 71),
    }
    return palette.get(ch, (130, 130, 130))


def _save_char_grid(lines: List[str], path: Path, tile_px: int = 16) -> None:
    h = len(lines)
    w = len(lines[0]) if h else 0
    canvas = np.zeros((h * tile_px, w * tile_px, 3), dtype=np.uint8)
    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            canvas[r * tile_px:(r + 1) * tile_px, c * tile_px:(c + 1) * tile_px] = _char_color(ch)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)


def _save_diff_heatmap(baseline: List[str], candidate: List[str], path: Path, tile_px: int = 16) -> None:
    h = len(baseline)
    w = len(baseline[0]) if h else 0
    canvas = np.zeros((h * tile_px, w * tile_px, 3), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            same = baseline[r][c] == candidate[r][c]
            color = (70, 160, 90) if same else (220, 70, 70)
            canvas[r * tile_px:(r + 1) * tile_px, c * tile_px:(c + 1) * tile_px] = color
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)


def _joined_room_signature(lines: List[str]) -> str:
    return "\n".join(lines)


def _categorize_change(src: str, dst: str) -> str:
    if src == dst:
        return "same"
    if src in {"W", "B"} and dst == "F":
        return "structure_to_floor"
    if src == "F" and dst in {"W", "B"}:
        return "floor_to_structure"
    if src == "-" and dst != "-":
        return "void_to_filled"
    if src != "-" and dst == "-":
        return "filled_to_void"
    return f"{src}_to_{dst}"


def compare_variant_dirs(baseline_dir: Path, candidate_dir: Path, output_dir: Path) -> Path:
    baseline_rooms = baseline_dir / "rooms"
    candidate_rooms = candidate_dir / "rooms"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_room_dir = output_dir / "rooms"
    per_room_dir.mkdir(parents=True, exist_ok=True)

    room_ids = sorted(
        int(path.stem.split("_")[-1])
        for path in baseline_rooms.glob("room_*.txt")
        if (candidate_rooms / path.name).exists()
    )
    results: List[Dict[str, object]] = []

    total_changed = 0
    aggregate_change_types: Counter[str] = Counter()

    for room_id in room_ids:
        baseline_lines = _read_room_text(baseline_rooms / f"room_{room_id}.txt")
        candidate_lines = _read_room_text(candidate_rooms / f"room_{room_id}.txt")
        if len(baseline_lines) != len(candidate_lines):
            raise ValueError(f"Room {room_id} line count mismatch.")
        if baseline_lines and any(len(a) != len(b) for a, b in zip(baseline_lines, candidate_lines)):
            raise ValueError(f"Room {room_id} width mismatch.")

        changed = 0
        change_types: Counter[str] = Counter()
        changed_positions: List[Tuple[int, int, str, str]] = []
        for r, (base_row, cand_row) in enumerate(zip(baseline_lines, candidate_lines)):
            for c, (src, dst) in enumerate(zip(base_row, cand_row)):
                if src != dst:
                    changed += 1
                    kind = _categorize_change(src, dst)
                    change_types[kind] += 1
                    aggregate_change_types[kind] += 1
                    changed_positions.append((r, c, src, dst))

        total_changed += changed
        room_out = per_room_dir / f"room_{room_id}"
        room_out.mkdir(parents=True, exist_ok=True)
        (room_out / "baseline.txt").write_text(_joined_room_signature(baseline_lines) + "\n", encoding="utf-8")
        (room_out / "candidate.txt").write_text(_joined_room_signature(candidate_lines) + "\n", encoding="utf-8")
        (room_out / "changed_positions.json").write_text(
            json.dumps(
                [
                    {"row": int(r), "col": int(c), "baseline": src, "candidate": dst}
                    for r, c, src, dst in changed_positions
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        _save_char_grid(baseline_lines, room_out / "baseline.png")
        _save_char_grid(candidate_lines, room_out / "candidate.png")
        _save_diff_heatmap(baseline_lines, candidate_lines, room_out / "diff_heatmap.png")

        results.append(
            {
                "room_id": int(room_id),
                "changed_tiles": int(changed),
                "change_ratio": float(changed / (len(baseline_lines) * len(baseline_lines[0])) if baseline_lines else 0.0),
                "change_types": dict(change_types),
            }
        )

    results.sort(key=lambda item: int(item["changed_tiles"]), reverse=True)
    summary = {
        "baseline_dir": str(baseline_dir),
        "candidate_dir": str(candidate_dir),
        "num_rooms_compared": len(results),
        "total_changed_tiles": int(total_changed),
        "aggregate_change_types": dict(aggregate_change_types),
        "rooms": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Room Variant Comparison",
        "",
        f"Baseline: `{baseline_dir}`",
        f"Candidate: `{candidate_dir}`",
        "",
        f"Compared rooms: {len(results)}",
        f"Total changed tiles: {int(total_changed)}",
        "",
        "## Worst Rooms",
        "",
    ]
    for item in results[: min(8, len(results))]:
        lines.append(
            f"- room_{int(item['room_id'])}: changed_tiles={int(item['changed_tiles'])}, "
            f"change_ratio={float(item['change_ratio']):.4f}, change_types={item['change_types']}"
        )
    lines.extend(
        [
            "",
            "## Aggregate Change Types",
            "",
            *(f"- {k}: {v}" for k, v in aggregate_change_types.most_common()),
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two exported room variant directories.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = compare_variant_dirs(args.baseline, args.candidate, args.output_dir)
    print(json.dumps({"output": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
