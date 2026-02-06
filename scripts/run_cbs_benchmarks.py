"""
Run CBS+ and A* benchmarks on Zelda dungeons (levels 1-9).

This script:
1. Loads Zelda dungeons 1-9 from Data/The Legend of Zelda/
2. Runs both StateSpaceAStar and CBS+ on each level
3. Computes all metrics (success, path_length, steps, PER, entropy, replans)
4. Saves results to results/cbs_benchmark.csv
5. Computes "Confusion Ratio" = CBS_steps / A*_steps for each level

Usage:
    python scripts/run_cbs_benchmarks.py
    python scripts/run_cbs_benchmarks.py --levels 1,2,3 --personas balanced,explorer
    python scripts/run_cbs_benchmarks.py --quick  # Only level 1
"""
import os
import sys
import csv
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.zelda_core import ZeldaDungeonAdapter
from src.simulation.validator import StateSpaceAStar, ZeldaLogicEnv
from src.simulation.cognitive_bounded_search import (
    CognitiveBoundedSearch, AgentPersona, CBSMetrics
)


def run_bench(
    dungeon_nums: List[int] = None,
    variants: Tuple[int, ...] = (1, 2),
    personas: List[str] = None,
    out_csv: str = 'results/cbs_benchmark.csv',
    timeout_astar: int = 200000,
    timeout_cbs: int = 50000,
    seed: int = 42,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run benchmark comparing A* and CBS+ on Zelda dungeons.
    
    Args:
        dungeon_nums: List of dungeon numbers (1-9), default all
        variants: Tuple of variant numbers to test
        personas: List of persona names to test, default ['balanced']
        out_csv: Output CSV path
        timeout_astar: A* solver timeout
        timeout_cbs: CBS+ solver timeout
        seed: Random seed for reproducibility
        verbose: Print progress
        
    Returns:
        List of result dictionaries
    """
    if dungeon_nums is None:
        dungeon_nums = list(range(1, 10))
    if personas is None:
        personas = ['balanced']
    
    # Create output directories
    Path('results/figures').mkdir(parents=True, exist_ok=True)
    Path('results').mkdir(parents=True, exist_ok=True)

    adapter = ZeldaDungeonAdapter('Data/The Legend of Zelda')

    # Extended headers for CBS+ metrics
    headers = [
        'map_id', 'solver', 'persona', 'success', 'path_length', 'steps', 
        'states_explored', 'PER', 'entropy_final', 'room_entropy', 'replans', 'confusion_events',
        'backtrack_loops', 'keys_collected', 'doors_opened', 'confusion_index',
        'cognitive_load', 'aha_latency', 'info_gain_total', 'timeout_flag',
        'confusion_ratio'
    ]

    rows = []
    astar_results = {}  # Cache A* results for confusion ratio

    for dungeon_num in dungeon_nums:
        for variant in variants:
            map_id = f'D{dungeon_num}_v{variant}'
            try:
                dungeon = adapter.load_dungeon(dungeon_num, variant=variant)
                stitched = adapter.stitch_dungeon(dungeon)
                grid = stitched.global_grid.copy()
                start = stitched.start_global
                goal = stitched.triforce_global
                
                if start is None or goal is None:
                    if verbose:
                        print(f'{map_id}: missing start/goal, skipping')
                    continue

                # Mark start/goal on grid
                grid[start[0], start[1]] = stitched.global_grid[start[0], start[1]]
                grid[goal[0], goal[1]] = stitched.global_grid[goal[0], goal[1]]
                
                manhattan = abs(start[0] - goal[0]) + abs(start[1] - goal[1])

                # A* solver (run once per map)
                env_astar = ZeldaLogicEnv(semantic_grid=grid)
                solver_a = StateSpaceAStar(env_astar, timeout=timeout_astar)
                success_a, path_a, states_a = solver_a.solve()
                path_len_a = len(path_a)
                per_a = path_len_a / max(1, manhattan)
                
                # Store A* result for confusion ratio calculation
                astar_results[map_id] = {
                    'success': success_a,
                    'path_length': path_len_a,
                    'steps': path_len_a,
                    'states': states_a
                }
                
                # A* row
                rows.append({
                    'map_id': map_id,
                    'solver': 'A*',
                    'persona': 'optimal',
                    'success': int(success_a),
                    'path_length': path_len_a,
                    'steps': path_len_a,
                    'states_explored': states_a,
                    'PER': round(per_a, 3),
                    'entropy_final': 0.0,
                    'room_entropy': 0.0,  # A* has no room entropy (optimal)
                    'replans': 0,
                    'confusion_events': 0,
                    'backtrack_loops': 0,
                    'keys_collected': 0,
                    'doors_opened': 0,
                    'confusion_index': 0.0,
                    'cognitive_load': 0.0,
                    'aha_latency': 0,
                    'info_gain_total': 0.0,
                    'timeout_flag': int(states_a >= timeout_astar),
                    'confusion_ratio': 1.0  # A* is baseline
                })
                
                if verbose:
                    print(f'{map_id} A*: success={success_a} path={path_len_a} states={states_a}')

                # CBS+ solvers (run for each persona)
                for persona_name in personas:
                    # ZeldaLogicEnv finds start/goal from grid tiles
                    env_cbs = ZeldaLogicEnv(semantic_grid=grid)
                    cbs = CognitiveBoundedSearch(
                        env_cbs, 
                        persona=persona_name, 
                        timeout=timeout_cbs, 
                        seed=seed
                    )
                    success_c, path_c, states_c, metrics = cbs.solve()
                    path_len_c = len(path_c)
                    per_c = path_len_c / max(1, manhattan)
                    
                    # Confusion Ratio = CBS_steps / A*_steps
                    confusion_ratio = path_len_c / max(1, path_len_a) if success_a else float('inf')
                    
                    rows.append({
                        'map_id': map_id,
                        'solver': 'CBS+',
                        'persona': persona_name,
                        'success': int(success_c),
                        'path_length': path_len_c,
                        'steps': path_len_c,
                        'states_explored': states_c,
                        'PER': round(per_c, 3),
                        'entropy_final': round(metrics.belief_entropy_final, 3),
                        'room_entropy': round(getattr(metrics, 'room_entropy', 0.0), 3),  # Navigational entropy
                        'replans': metrics.replans,
                        'confusion_events': metrics.confusion_events,
                        'backtrack_loops': metrics.backtrack_loops,
                        'keys_collected': 0,  # Not tracked yet
                        'doors_opened': 0,  # Not tracked yet
                        'confusion_index': round(metrics.confusion_index, 3),
                        'cognitive_load': round(metrics.cognitive_load, 3),
                        'aha_latency': metrics.aha_latency,
                        'info_gain_total': round(metrics.exploration_efficiency, 3),
                        'timeout_flag': int(states_c >= timeout_cbs),
                        'confusion_ratio': round(confusion_ratio, 3)
                    })
                    
                    if verbose:
                        print(f'{map_id} CBS+ ({persona_name}): success={success_c} path={path_len_c} '
                              f'states={states_c} confusion_ratio={confusion_ratio:.2f}')

            except Exception as e:
                if verbose:
                    print(f'{map_id}: ERROR {e}')
                import traceback
                traceback.print_exc()

    # Write CSV
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    if verbose:
        print(f'\nWrote {len(rows)} results to {out_csv}')
    
    return rows


def compute_summary_statistics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute summary statistics from benchmark results.
    
    Returns:
        Dictionary with summary statistics
    """
    astar_rows = [r for r in rows if r['solver'] == 'A*']
    cbs_rows = [r for r in rows if r['solver'] == 'CBS+']
    
    summary = {
        'total_maps': len(astar_rows),
        'astar': {
            'success_rate': sum(r['success'] for r in astar_rows) / max(1, len(astar_rows)),
            'avg_path_length': np.mean([r['path_length'] for r in astar_rows if r['success']]) if any(r['success'] for r in astar_rows) else 0,
            'avg_states': np.mean([r['states_explored'] for r in astar_rows]) if astar_rows else 0,
        },
        'cbs': {}
    }
    
    # Group CBS by persona
    personas = set(r['persona'] for r in cbs_rows)
    for persona in personas:
        persona_rows = [r for r in cbs_rows if r['persona'] == persona]
        successful = [r for r in persona_rows if r['success']]
        
        summary['cbs'][persona] = {
            'success_rate': len(successful) / max(1, len(persona_rows)),
            'avg_path_length': np.mean([r['path_length'] for r in successful]) if successful else 0,
            'avg_confusion_ratio': np.mean([r['confusion_ratio'] for r in successful]) if successful else 0,
            'avg_confusion_index': np.mean([r['confusion_index'] for r in successful]) if successful else 0,
            'avg_replans': np.mean([r['replans'] for r in successful]) if successful else 0,
            'avg_backtrack_loops': np.mean([r['backtrack_loops'] for r in successful]) if successful else 0,
        }
    
    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    """Print formatted summary statistics."""
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    print(f"Total maps tested: {summary['total_maps']}")
    print(f"\nA* (Optimal):")
    print(f"  Success rate: {summary['astar']['success_rate']*100:.1f}%")
    print(f"  Avg path length: {summary['astar']['avg_path_length']:.1f}")
    print(f"  Avg states explored: {summary['astar']['avg_states']:.0f}")
    
    print(f"\nCBS+ Results by Persona:")
    for persona, stats in summary['cbs'].items():
        print(f"\n  {persona.upper()}:")
        print(f"    Success rate: {stats['success_rate']*100:.1f}%")
        print(f"    Avg path length: {stats['avg_path_length']:.1f}")
        print(f"    Avg confusion ratio: {stats['avg_confusion_ratio']:.2f}x")
        print(f"    Avg confusion index: {stats['avg_confusion_index']:.3f}")
        print(f"    Avg replans: {stats['avg_replans']:.1f}")
        print(f"    Avg backtrack loops: {stats['avg_backtrack_loops']:.1f}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Run CBS+ benchmarks on Zelda dungeons')
    parser.add_argument('--levels', type=str, default='1,2,3,4,5,6,7,8,9',
                        help='Comma-separated dungeon numbers (default: 1-9)')
    parser.add_argument('--variants', type=str, default='1,2',
                        help='Comma-separated variant numbers (default: 1,2)')
    parser.add_argument('--personas', type=str, default='balanced',
                        help='Comma-separated personas (default: balanced)')
    parser.add_argument('--all-personas', action='store_true',
                        help='Run all personas')
    parser.add_argument('--output', type=str, default='results/cbs_benchmark.csv',
                        help='Output CSV path')
    parser.add_argument('--timeout-astar', type=int, default=200000,
                        help='A* solver timeout')
    parser.add_argument('--timeout-cbs', type=int, default=50000,
                        help='CBS+ solver timeout')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test (level 1 only)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Parse arguments
    if args.quick:
        dungeon_nums = [1]
        variants = (1,)
    else:
        dungeon_nums = [int(x) for x in args.levels.split(',')]
        variants = tuple(int(x) for x in args.variants.split(','))
    
    if args.all_personas:
        personas = [p.value for p in AgentPersona]
    else:
        personas = [p.strip() for p in args.personas.split(',')]
    
    # Run benchmark
    rows = run_bench(
        dungeon_nums=dungeon_nums,
        variants=variants,
        personas=personas,
        out_csv=args.output,
        timeout_astar=args.timeout_astar,
        timeout_cbs=args.timeout_cbs,
        seed=args.seed,
        verbose=not args.quiet
    )
    
    # Compute and print summary
    if rows:
        summary = compute_summary_statistics(rows)
        print_summary(summary)


if __name__ == '__main__':
    main()
