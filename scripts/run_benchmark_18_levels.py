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
import csv
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.zelda_data.zelda_core import ZeldaDungeonAdapter
from src.simulation.validator import ZeldaLogicEnv
from src.simulation.cognitive_bounded_search import (
    CognitiveBoundedSearch
)
from src.evaluation.search_benchmark_utils import (
    confusion_ratio_vs_oracle,
    finite_mean,
    path_efficiency_ratio,
    run_astar_oracle,
)
from src.evaluation.pcbs_validation import prepare_dungeon_grid_for_validation

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
    'time_s', 'solver_status', 'failure_reason',
    'oracle_status', 'oracle_failure_reason',
]


def run_full_18(
    data_dir: str = None,
    personas: List[str] = None,
    dungeon_nums: List[int] = None,
    variants: List[int] = None,
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
    if dungeon_nums is None:
        dungeon_nums = list(range(1, 10))
    if variants is None:
        variants = [1, 2]

    adapter = ZeldaDungeonAdapter(data_dir)
    rows: List[Dict[str, Any]] = []

    for dungeon_num in dungeon_nums:
        for variant in variants:
            map_id = f'D{dungeon_num}_v{variant}'
            try:
                dungeon = adapter.load_dungeon(dungeon_num, variant=variant)
                stitched = adapter.stitch_dungeon(dungeon)
                prepared = prepare_dungeon_grid_for_validation(stitched)
                grid = prepared.grid
                start = prepared.start
                goal = prepared.goal

                manhattan = abs(start[0] - goal[0]) + abs(start[1] - goal[1])

                # ---- A* --------------------------------------------------
                env_a = ZeldaLogicEnv(semantic_grid=grid)
                astar_payload = run_astar_oracle(env_a, timeout=timeout_astar)
                dt_a = float(astar_payload['time_ms']) / 1000.0
                success_a = bool(astar_payload['success'])
                pl_a = int(astar_payload['path_length'])
                states_a = int(astar_payload['states_explored'])
                per_a = path_efficiency_ratio(pl_a, manhattan)
                rows.append({
                    'map_id': map_id, 'dungeon': dungeon_num, 'variant': variant,
                    'solver': 'A*', 'persona': 'optimal',
                    'success': int(success_a), 'path_length': pl_a,
                    'states_explored': states_a, 'PER': round(per_a, 4),
                    'confusion_ratio': 1.0 if success_a else float('nan'),
                    'confusion_index': 0.0, 'room_entropy': 0.0,
                    'replans': 0, 'confusion_events': 0, 'backtrack_loops': 0,
                    'cognitive_load': 0.0, 'aha_latency': 0,
                    'belief_entropy_final': 0.0, 'navigation_entropy': 0.0,
                    'time_s': round(dt_a, 3),
                    'solver_status': str(astar_payload['status']),
                    'failure_reason': str(astar_payload['failure_reason']),
                    'oracle_status': str(astar_payload['status']),
                    'oracle_failure_reason': str(astar_payload['failure_reason']),
                })
                if verbose:
                    print(
                        f"  {map_id} A*: status={astar_payload['status']}  "
                        f"path={pl_a}  states={states_a}  {dt_a:.1f}s"
                    )

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
                    per_c = path_efficiency_ratio(pl_c, manhattan)
                    cr = confusion_ratio_vs_oracle(
                        pl_a,
                        pl_c,
                        oracle_status=str(astar_payload['status']),
                        candidate_success=bool(success_c),
                    )
                    cbs_status = (
                        'solved'
                        if success_c
                        else ('timeout' if int(states_c or 0) >= int(timeout_cbs) else 'failed')
                    )
                    cbs_failure_reason = (
                        ''
                        if success_c
                        else (
                            f"Timeout: explored {int(states_c or 0):,} states "
                            f"(limit: {int(timeout_cbs):,})"
                            if int(states_c or 0) >= int(timeout_cbs)
                            else 'Bounded-rational search failed before reaching goal'
                        )
                    )

                    rows.append({
                        'map_id': map_id, 'dungeon': dungeon_num, 'variant': variant,
                        'solver': 'CBS+', 'persona': persona_name,
                        'success': int(success_c), 'path_length': pl_c,
                        'states_explored': states_c, 'PER': round(per_c, 4),
                        'confusion_ratio': round(cr, 4) if np.isfinite(cr) else float('nan'),
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
                        'solver_status': cbs_status,
                        'failure_reason': cbs_failure_reason,
                        'oracle_status': str(astar_payload['status']),
                        'oracle_failure_reason': str(astar_payload['failure_reason']),
                    })
                    if verbose:
                        cr_text = f"{float(cr):.2f}" if np.isfinite(cr) else "n/a"
                        print(f'  {map_id} CBS+/{persona_name}: ok={success_c}  path={pl_c}  '
                              f'CR={cr_text}  CI={metrics.confusion_index:.3f}  {dt_c:.1f}s')

            except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
                if verbose:
                    print(f'  {map_id}: ERROR â€” {exc}')
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
    except ImportError as e:
        print(f'Cannot generate figures (missing dependency): {e}')
        return

    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)

    # ---- Fig 1: Confusion Ratio â€” all levels, balanced persona -----------
    _fig_cr_all_levels(df, fig_dir)
    # ---- Fig 2: Greedy vs Balanced ---------------------------------------
    _fig_greedy_vs_balanced(df, fig_dir)
    # ---- Fig 3: All personas compared ------------------------------------
    _fig_persona_comparison(df, fig_dir)

    print(f'Figures saved to {fig_dir}/')


def _fig_cr_all_levels(df, fig_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    balanced = df[
        (df['solver'] == 'CBS+')
        & (df['persona'] == 'balanced')
        & (df['success'] == 1)
        & (df['oracle_status'] == 'solved')
    ]
    if balanced.empty:
        print('  No balanced CBS+ results â€” skipping fig_confusion_ratio_all_levels')
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
    print('  Saved fig_confusion_ratio_all_levels.png')


def _fig_greedy_vs_balanced(df, fig_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cbs = df[(df['solver'] == 'CBS+') & (df['success'] == 1) & (df['oracle_status'] == 'solved')]
    greedy = cbs[cbs['persona'] == 'greedy'].set_index('map_id')['confusion_ratio']
    balanced = cbs[cbs['persona'] == 'balanced'].set_index('map_id')['confusion_ratio']
    common = sorted(set(greedy.index) & set(balanced.index))
    if not common:
        print('  No common greedy+balanced results â€” skipping fig_greedy_vs_balanced_full')
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
    ax.set_title('Greedy vs Balanced Persona â€” Confusion Ratio per Level')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/fig_greedy_vs_balanced_full.png', dpi=150)
    plt.close(fig)
    print('  Saved fig_greedy_vs_balanced_full.png')


def _fig_persona_comparison(df, fig_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cbs = df[(df['solver'] == 'CBS+') & (df['success'] == 1) & (df['oracle_status'] == 'solved')]
    if cbs.empty:
        print('  No CBS+ results â€” skipping fig_persona_comparison')
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
    ax.set_title('All Personas Compared â€” Confusion Ratio Across Dungeons')
    ax.axhline(1.0, color='red', ls='--', lw=0.8)
    ax.legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/fig_persona_comparison.png', dpi=150)
    plt.close(fig)
    print('  Saved fig_persona_comparison.png')


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
    print('CBS+ Benchmark â€” 18 Nintendo Zelda Dungeons')
    print('='*70)

    rows = run_full_18(
        personas=PERSONAS,
        dungeon_nums=[1] if args.quick else list(range(1, 10)),
        variants=[1, 2],
        timeout_astar=args.timeout_astar,
        timeout_cbs=args.timeout_cbs,
        seed=args.seed,
        verbose=bool(args.verbose),
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
        _write_summary_json(rows, args.output)

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
    print(
        f"A* statuses: solved={sum(1 for r in astar if r['solver_status'] == 'solved')} "
        f"timeout={sum(1 for r in astar if r['solver_status'] == 'timeout')} "
        f"no_path={sum(1 for r in astar if r['solver_status'] == 'no_path')} "
        f"invalid={sum(1 for r in astar if r['solver_status'] == 'invalid_map')}"
    )
    for persona in PERSONAS:
        pr = [r for r in cbs if r['persona'] == persona]
        ok = [r for r in pr if r['success']]
        ok_with_oracle = [r for r in ok if r.get('oracle_status') == 'solved']
        oracle_solved_total = sum(1 for r in pr if r.get('oracle_status') == 'solved')
        avg_cr = finite_mean([r['confusion_ratio'] for r in ok_with_oracle])
        print(
            f'  {persona:12s}: {len(ok)}/{len(pr)} solved, '
            f'conditioned_oracle={len(ok_with_oracle)}/{oracle_solved_total} '
            f'avg CR={avg_cr:.2f}'
        )
    print('='*70)


def _write_summary_json(rows: List[Dict[str, Any]], output_csv: str) -> None:
    astar = [r for r in rows if r['solver'] == 'A*']
    cbs = [r for r in rows if r['solver'] == 'CBS+']
    payload = {
        'total_maps': len(astar),
        'astar': {
            'solved': sum(1 for r in astar if r['solver_status'] == 'solved'),
            'timeout': sum(1 for r in astar if r['solver_status'] == 'timeout'),
            'no_path': sum(1 for r in astar if r['solver_status'] == 'no_path'),
            'invalid_map': sum(1 for r in astar if r['solver_status'] == 'invalid_map'),
            'avg_states': float(np.mean([r['states_explored'] for r in astar])) if astar else 0.0,
        },
        'cbs': {},
    }
    for persona in PERSONAS:
        persona_rows = [r for r in cbs if r['persona'] == persona]
        solved_rows = [r for r in persona_rows if r['success']]
        solved_with_oracle = [r for r in solved_rows if r.get('oracle_status') == 'solved']
        oracle_solved_total = sum(1 for r in persona_rows if r.get('oracle_status') == 'solved')
        payload['cbs'][persona] = {
            'success_rate': float(len(solved_rows) / max(1, len(persona_rows))),
            'success_rate_given_oracle_solved': float(
                len(solved_with_oracle) / max(1, oracle_solved_total)
            ),
            'avg_confusion_ratio': float(finite_mean([r.get('confusion_ratio') for r in solved_with_oracle])),
            'avg_confusion_index': float(np.mean([r['confusion_index'] for r in solved_rows])) if solved_rows else 0.0,
        }
    out_path = Path(output_csv).with_name(f"{Path(output_csv).stem}_summary.json")
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

