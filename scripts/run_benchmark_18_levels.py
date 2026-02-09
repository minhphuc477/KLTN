#!/usr/bin/env python3
"""
Full 18-Level Nintendo Zelda CBS+ Benchmark
============================================

Runs A* and CBS+ (4 personas) on ALL 18 Nintendo Zelda dungeons
(dungeons 1-9, quests 1 & 2).

Outputs:
- results/cbs_benchmark_full_18.csv
- results/figures/fig_confusion_ratio_all_levels.png
- results/figures/fig_greedy_vs_balanced_full.png
- results/figures/fig_persona_comparison.png

Usage:
    python scripts/run_benchmark_18_levels.py
    python scripts/run_benchmark_18_levels.py --quick  # Only D1
"""

import sys
import os
import csv
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.zelda_core import ZeldaDungeonAdapter
from src.simulation.validator import ZeldaLogicEnv, StateSpaceAStar
from src.simulation.cognitive_bounded_search import (
    CognitiveBoundedSearch, CBSMetrics, AgentPersona
)

logger = logging.getLogger(__name__)

# =============================================================================
# BENCHMARK
# =============================================================================

PERSONAS = ['balanced', 'greedy', 'explorer', 'forgetful']

HEADERS = [
    'map_id', 'dungeon', 'variant', 'solver', 'persona',
    'success', 'path_length', 'states_explored',
    'PER', 'confusion_ratio',
    'confusion_index', 'room_entropy', 'replans',
    'confusion_events', 'backtrack_loops',
    'cognitive_load', 'aha_latency',
    'belief_entropy_final', 'navigation_entropy',
    'time_s',
]


def run_full_18(
    data_dir: str = None,
    personas: List[str] = None,
    timeout_astar: int = 200000,
    timeout_cbs: int = 5000,
    seed: int = 42,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Run benchmark on all 18 Nintendo Zelda dungeons."""
    if data_dir is None:
        data_dir = str(PROJECT_ROOT / 'Data' / 'The Legend of Zelda')
    if personas is None:
        personas = PERSONAS

    adapter = ZeldaDungeonAdapter(data_dir)
    rows: List[Dict[str, Any]] = []
    astar_cache: Dict[str, int] = {}  # map_id -> path_length

    for dungeon_num in range(1, 10):
        for variant in [1, 2]:
            map_id = f'D{dungeon_num}_v{variant}'
            try:
                dungeon = adapter.load_dungeon(dungeon_num, variant=variant)
                stitched = adapter.stitch_dungeon(dungeon)
                grid = stitched.global_grid.copy()
                start = stitched.start_global
                goal = stitched.triforce_global

                if start is None or goal is None:
                    if verbose:
                        print(f'  {map_id}: missing start/goal — skipping')
                    continue

                manhattan = abs(start[0] - goal[0]) + abs(start[1] - goal[1])

                # ---- A* --------------------------------------------------
                t0 = time.time()
                env_a = ZeldaLogicEnv(semantic_grid=grid)
                solver_a = StateSpaceAStar(env_a, timeout=timeout_astar)
                success_a, path_a, states_a = solver_a.solve()
                dt_a = time.time() - t0
                pl_a = len(path_a)
                per_a = manhattan / max(1, pl_a) if success_a else 0.0
                astar_cache[map_id] = pl_a if success_a else 0

                rows.append({
                    'map_id': map_id, 'dungeon': dungeon_num, 'variant': variant,
                    'solver': 'A*', 'persona': 'optimal',
                    'success': int(success_a), 'path_length': pl_a,
                    'states_explored': states_a, 'PER': round(per_a, 4),
                    'confusion_ratio': 1.0,
                    'confusion_index': 0.0, 'room_entropy': 0.0,
                    'replans': 0, 'confusion_events': 0, 'backtrack_loops': 0,
                    'cognitive_load': 0.0, 'aha_latency': 0,
                    'belief_entropy_final': 0.0, 'navigation_entropy': 0.0,
                    'time_s': round(dt_a, 3),
                })
                if verbose:
                    print(f'  {map_id} A*: ok={success_a}  path={pl_a}  states={states_a}  {dt_a:.1f}s')

                # ---- CBS+ per persona ------------------------------------
                for persona_name in personas:
                    t0 = time.time()
                    env_c = ZeldaLogicEnv(semantic_grid=grid.copy())
                    cbs = CognitiveBoundedSearch(
                        env_c, persona=persona_name,
                        timeout=timeout_cbs, seed=seed,
                    )
                    success_c, path_c, states_c, metrics = cbs.solve()
                    dt_c = time.time() - t0
                    pl_c = len(path_c)
                    per_c = manhattan / max(1, pl_c) if success_c else 0.0
                    cr = pl_c / max(1, pl_a) if success_a and pl_a > 0 else float('inf')

                    rows.append({
                        'map_id': map_id, 'dungeon': dungeon_num, 'variant': variant,
                        'solver': 'CBS+', 'persona': persona_name,
                        'success': int(success_c), 'path_length': pl_c,
                        'states_explored': states_c, 'PER': round(per_c, 4),
                        'confusion_ratio': round(cr, 4) if cr != float('inf') else -1,
                        'confusion_index': round(getattr(metrics, 'confusion_index', 0.0), 4),
                        'room_entropy': round(getattr(metrics, 'room_entropy', 0.0), 4),
                        'replans': getattr(metrics, 'replans', 0),
                        'confusion_events': getattr(metrics, 'confusion_events', 0),
                        'backtrack_loops': getattr(metrics, 'backtrack_loops', 0),
                        'cognitive_load': round(getattr(metrics, 'cognitive_load', 0.0), 4),
                        'aha_latency': getattr(metrics, 'aha_latency', 0),
                        'belief_entropy_final': round(getattr(metrics, 'belief_entropy_final', 0.0), 4),
                        'navigation_entropy': round(getattr(metrics, 'navigation_entropy', 0.0), 4),
                        'time_s': round(dt_c, 3),
                    })
                    if verbose:
                        print(f'  {map_id} CBS+/{persona_name}: ok={success_c}  path={pl_c}  '
                              f'CR={cr:.2f}  CI={metrics.confusion_index:.3f}  {dt_c:.1f}s')

            except Exception as exc:
                if verbose:
                    print(f'  {map_id}: ERROR — {exc}')
                import traceback; traceback.print_exc()

    return rows


# =============================================================================
# VISUALIZATION
# =============================================================================

def generate_figures(csv_path: str, fig_dir: str = 'results/figures'):
    """Generate publication-quality figures from benchmark CSV."""
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f'Cannot generate figures (missing dependency): {e}')
        return

    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)

    # ---- Fig 1: Confusion Ratio — all levels, balanced persona -----------
    _fig_cr_all_levels(df, fig_dir)
    # ---- Fig 2: Greedy vs Balanced ---------------------------------------
    _fig_greedy_vs_balanced(df, fig_dir)
    # ---- Fig 3: All personas compared ------------------------------------
    _fig_persona_comparison(df, fig_dir)

    print(f'Figures saved to {fig_dir}/')


def _fig_cr_all_levels(df, fig_dir):
    import pandas as pd, matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    balanced = df[(df['solver'] == 'CBS+') & (df['persona'] == 'balanced') & (df['success'] == 1)]
    if balanced.empty:
        print('  No balanced CBS+ results — skipping fig_confusion_ratio_all_levels')
        return

    # Group by dungeon for error bars across variants
    grouped = balanced.groupby('dungeon')['confusion_ratio']
    means = grouped.mean()
    stds = grouped.std().fillna(0)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(means))
    ax.bar(x, means, yerr=stds, capsize=4, color='#4C72B0', edgecolor='black', linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f'D{d}' for d in means.index])
    ax.set_xlabel('Dungeon')
    ax.set_ylabel('Confusion Ratio (CBS / A*)')
    ax.set_title('Confusion Ratio Across All Nintendo Zelda Dungeons (Balanced Persona)')
    ax.axhline(1.0, color='red', ls='--', lw=0.8, label='Optimal (A*)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/fig_confusion_ratio_all_levels.png', dpi=150)
    plt.close(fig)
    print(f'  Saved fig_confusion_ratio_all_levels.png')


def _fig_greedy_vs_balanced(df, fig_dir):
    import pandas as pd, matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cbs = df[(df['solver'] == 'CBS+') & (df['success'] == 1)]
    greedy = cbs[cbs['persona'] == 'greedy'].set_index('map_id')['confusion_ratio']
    balanced = cbs[cbs['persona'] == 'balanced'].set_index('map_id')['confusion_ratio']
    common = sorted(set(greedy.index) & set(balanced.index))
    if not common:
        print('  No common greedy+balanced results — skipping fig_greedy_vs_balanced_full')
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(common))
    w = 0.35
    ax.bar(x - w/2, [balanced[m] for m in common], w, label='Balanced', color='#4C72B0')
    ax.bar(x + w/2, [greedy[m] for m in common], w, label='Greedy', color='#DD8452')
    ax.set_xticks(x)
    ax.set_xticklabels(common, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Level')
    ax.set_ylabel('Confusion Ratio')
    ax.set_title('Greedy vs Balanced Persona — Confusion Ratio per Level')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/fig_greedy_vs_balanced_full.png', dpi=150)
    plt.close(fig)
    print(f'  Saved fig_greedy_vs_balanced_full.png')


def _fig_persona_comparison(df, fig_dir):
    import pandas as pd, matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cbs = df[(df['solver'] == 'CBS+') & (df['success'] == 1)]
    if cbs.empty:
        print('  No CBS+ results — skipping fig_persona_comparison')
        return

    personas_present = sorted(cbs['persona'].unique())
    dungeons = sorted(cbs['dungeon'].unique())
    n_personas = len(personas_present)
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860']

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(dungeons))
    w = 0.8 / max(n_personas, 1)

    for i, persona in enumerate(personas_present):
        pdata = cbs[cbs['persona'] == persona]
        means = []
        for d in dungeons:
            vals = pdata[pdata['dungeon'] == d]['confusion_ratio']
            means.append(vals.mean() if len(vals) > 0 else 0)
        offset = (i - n_personas / 2 + 0.5) * w
        ax.bar(x + offset, means, w, label=persona.capitalize(),
               color=colors[i % len(colors)], edgecolor='black', linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([f'D{d}' for d in dungeons])
    ax.set_xlabel('Dungeon')
    ax.set_ylabel('Confusion Ratio')
    ax.set_title('All Personas Compared — Confusion Ratio Across Dungeons')
    ax.axhline(1.0, color='red', ls='--', lw=0.8)
    ax.legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/fig_persona_comparison.png', dpi=150)
    plt.close(fig)
    print(f'  Saved fig_persona_comparison.png')


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Full 18-level CBS+ Benchmark')
    parser.add_argument('--quick', action='store_true', help='Only dungeon 1')
    parser.add_argument('--timeout-astar', type=int, default=200000)
    parser.add_argument('--timeout-cbs', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='results/cbs_benchmark_full_18.csv')
    parser.add_argument('--no-figures', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s | %(levelname)s | %(message)s')

    print('='*70)
    print('CBS+ Benchmark — 18 Nintendo Zelda Dungeons')
    print('='*70)

    rows = run_full_18(
        personas=PERSONAS,
        timeout_astar=args.timeout_astar,
        timeout_cbs=args.timeout_cbs,
        seed=args.seed,
        verbose=True,
    )

    # Save CSV
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(args.output, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=HEADERS)
            w.writeheader()
            for r in rows:
                w.writerow({h: r.get(h, '') for h in HEADERS})
        print(f'\nWrote {len(rows)} rows to {args.output}')

    # Figures
    if not args.no_figures and rows:
        generate_figures(args.output)

    # Summary
    if rows:
        _print_summary(rows)

    return rows


def _print_summary(rows):
    astar = [r for r in rows if r['solver'] == 'A*']
    cbs = [r for r in rows if r['solver'] == 'CBS+']
    print('\n' + '='*70)
    print('SUMMARY')
    print('='*70)
    print(f'Levels tested: {len(astar)}')
    print(f'A* success: {sum(1 for r in astar if r["success"])} / {len(astar)}')
    for persona in PERSONAS:
        pr = [r for r in cbs if r['persona'] == persona]
        ok = [r for r in pr if r['success']]
        avg_cr = np.mean([r['confusion_ratio'] for r in ok]) if ok else 0
        print(f'  {persona:12s}: {len(ok)}/{len(pr)} solved, avg CR={avg_cr:.2f}')
    print('='*70)


if __name__ == '__main__':
    main()
