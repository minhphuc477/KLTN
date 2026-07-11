"""A/B benchmark for StateSpaceAStar priority modes using VGLC dungeons."""

import csv
import copy
import time
from pathlib import Path

from src.evaluation.pcbs_validation import extract_validation_env_kwargs, prepare_dungeon_grid_for_validation
from src.evaluation.search_benchmark_utils import path_transition_count
from src.simulation import StateSpaceAStar
from src.simulation.validator import ZeldaLogicEnv
from src.zelda_data.zelda_core import ZeldaDungeonAdapter

DATA_ROOT = Path("Data/The Legend of Zelda")
OUT_DIR = Path("artifacts")
DUNGEONS = list(range(1, 10))
VARIANT = 1


def main() -> None:
    """Run all priority modes and write one timestamped CSV report."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    adapter = ZeldaDungeonAdapter(str(DATA_ROOT))
    runs = []

    for dungeon_num in DUNGEONS:
        vglc = DATA_ROOT / "Processed" / f"tloz{dungeon_num}_{VARIANT}.txt"
        dot = DATA_ROOT / "Graph Processed" / f"LoZ_{dungeon_num}.dot"
        if not vglc.exists() or not dot.exists():
            print(f"Skipping dungeon {dungeon_num} (missing data)")
            continue

        dungeon = adapter.load_dungeon(dungeon_num, variant=VARIANT)
        stitched = adapter.stitch_dungeon(dungeon)
        grid = prepare_dungeon_grid_for_validation(stitched).grid
        env_kwargs = extract_validation_env_kwargs(stitched)

        modes = (
            ("baseline", {}),
            ("priority_tie_break", {"tie_break": True}),
            ("priority_key_boost", {"key_boost": True}),
        )
        for mode_name, options in modes:
            env = ZeldaLogicEnv(
                semantic_grid=grid.copy(),
                render_mode=False,
                **copy.deepcopy(env_kwargs),
            )
            solver = StateSpaceAStar(env, timeout=200000, priority_options=options)
            started = time.perf_counter()
            success, path, states = solver.solve()
            elapsed = time.perf_counter() - started
            path_len = path_transition_count(path)
            runs.append(
                {
                    "dungeon": dungeon_num,
                    "mode": mode_name,
                    "success": success,
                    "states": states,
                    "time_s": elapsed,
                    "path_len": path_len,
                }
            )
            print(
                f"D{dungeon_num} {mode_name}: success={success}, states={states}, "
                f"time={elapsed:.2f}s, path_len={path_len}"
            )

    output_path = OUT_DIR / f"ab_benchmark_{int(time.time())}.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dungeon", "mode", "success", "states", "time_s", "path_len"],
        )
        writer.writeheader()
        writer.writerows(runs)

    print("\n## A/B Benchmark Summary")
    for dungeon_num in sorted({row["dungeon"] for row in runs}):
        print(f"\n### Dungeon {dungeon_num}")
        for row in (item for item in runs if item["dungeon"] == dungeon_num):
            print(
                f"- {row['mode']}: success={row['success']}, states={row['states']}, "
                f"time={row['time_s']:.2f}s, path_len={row['path_len']}"
            )
    print(f"\nCSV results: {output_path}")


if __name__ == "__main__":
    main()
