"""A/B benchmark for StateSpaceAStar priority modes using VGLC dungeons.

Outputs CSV report to artifacts/ab_benchmark_<ts>.csv and prints a markdown summary.
"""
import time
import csv
from pathlib import Path
from src.data.zelda_core import ZeldaDungeonAdapter
from src.simulation import StateSpaceAStar

DATA_ROOT = Path('Data/The Legend of Zelda')
OUT_DIR = Path('artifacts')
OUT_DIR.mkdir(parents=True, exist_ok=True)

DUNGEONS = list(range(1, 10))  # tloz1_1 .. tloz9_1 (quest 1)
VARIANT = 1

runs = []
for d in DUNGEONS:
    vglc = DATA_ROOT / 'Processed' / f'tloz{d}_{VARIANT}.txt'
    dot = DATA_ROOT / 'Graph Processed' / f'LoZ_{d}.dot'
    if not vglc.exists() or not dot.exists():
        print(f"Skipping dungeon {d} (missing data)")
        continue
    adapter = ZeldaDungeonAdapter(str(DATA_ROOT))
    dungeon = adapter.load_dungeon(d, variant=VARIANT)
    stitched = adapter.stitch_dungeon(dungeon)

    # Create env via GUI loader pattern
    from gui_runner import ZeldaGUI
    gui = ZeldaGUI(maps=[stitched], map_names=[f"D{d}"])
    env = gui.env

    for mode_name, opts in [('baseline', {}), ('priority_tie_break', {'tie_break': True}), ('priority_key_boost', {'key_boost': True})]:
        s = StateSpaceAStar(env, timeout=200000, priority_options=opts)
        t0 = time.time()
        ok, path, states = s.solve()
        elapsed = time.time() - t0
        runs.append({'dungeon': d, 'mode': mode_name, 'success': ok, 'states': states, 'time_s': elapsed, 'path_len': len(path)})
        print(f"D{d} {mode_name}: success={ok}, states={states}, time={elapsed:.2f}s, path_len={len(path)}")

# Write CSV
from datetime import datetime
fname = OUT_DIR / f"ab_benchmark_{int(time.time())}.csv"
with open(fname, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['dungeon','mode','success','states','time_s','path_len'])
    w.writeheader()
    for r in runs:
        w.writerow(r)

# Print markdown summary
print('\n## A/B Benchmark Summary')
for d in sorted(set(r['dungeon'] for r in runs)):
    print(f"\n### Dungeon {d}")
    for r in [x for x in runs if x['dungeon']==d]:
        print(f"- {r['mode']}: success={r['success']}, states={r['states']}, time={r['time_s']:.2f}s, path_len={r['path_len']}")

print(f"\nCSV results: {fname}")
