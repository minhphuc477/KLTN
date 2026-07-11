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
import sys
import copy
import csv
import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any
import math
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.zelda_data.zelda_core import ZeldaDungeonAdapter
from src.simulation.validator import ZeldaLogicEnv
from src.simulation.cognitive_bounded_search import (
    CognitiveBoundedSearch, AgentPersona
)
from src.evaluation.search_benchmark_utils import (
    confusion_ratio_vs_oracle,
    finite_mean,
    path_efficiency_ratio,
    run_astar_oracle,
)
from src.evaluation.pcbs_validation import extract_validation_env_kwargs, prepare_dungeon_grid_for_validation


def _json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats with None for valid JSON output."""
    if isinstance(value, dict):
        return {key: _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_json_safe(inner) for inner in value]
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    if math.isfinite(numeric):
        return value
    return None


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
        'solver_status', 'failure_reason', 'time_ms',
        'deliberation_events', 'budget_exhaustion_events', 'peak_frustration', 'final_deliberation_budget',
        'affordance_reactivations', 'affordance_guided_steps', 'inventory_change_events',
        'focus_switches', 'focus_guided_steps',
        'oracle_status', 'oracle_failure_reason', 'confusion_ratio'
    ]

    rows = []
    for dungeon_num in dungeon_nums:
        for variant in variants:
            map_id = f'D{dungeon_num}_v{variant}'
            try:
                dungeon = adapter.load_dungeon(dungeon_num, variant=variant)
                stitched = adapter.stitch_dungeon(dungeon)
                prepared = prepare_dungeon_grid_for_validation(stitched)
                grid = prepared.grid
                env_kwargs = extract_validation_env_kwargs(stitched)
                start = prepared.start
                goal = prepared.goal
                
                manhattan = abs(start[0] - goal[0]) + abs(start[1] - goal[1])

                # A* solver (run once per map)
                env_astar = ZeldaLogicEnv(semantic_grid=grid, **copy.deepcopy(env_kwargs))
                astar_payload = run_astar_oracle(env_astar, timeout=timeout_astar)
                success_a = bool(astar_payload['success'])
                path_len_a = int(astar_payload['path_length'])
                states_a = int(astar_payload['states_explored'])
                per_a = path_efficiency_ratio(path_len_a, manhattan)
                
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
                    'timeout_flag': int(astar_payload['timeout_flag']),
                    'solver_status': str(astar_payload['status']),
                    'failure_reason': str(astar_payload['failure_reason']),
                    'time_ms': round(float(astar_payload['time_ms']), 3),
                    'deliberation_events': 0,
                    'budget_exhaustion_events': 0,
                    'peak_frustration': 0.0,
                    'final_deliberation_budget': 0.0,
                    'affordance_reactivations': 0,
                    'affordance_guided_steps': 0,
                    'inventory_change_events': 0,
                    'focus_switches': 0,
                    'focus_guided_steps': 0,
                    'oracle_status': str(astar_payload['status']),
                    'oracle_failure_reason': str(astar_payload['failure_reason']),
                    'confusion_ratio': 0.0 if success_a else float('nan'),
                })
                
                if verbose:
                    print(
                        f"{map_id} A*: status={astar_payload['status']} "
                        f"path={path_len_a} states={states_a}"
                    )

                # CBS+ solvers (run for each persona)
                for persona_name in personas:
                    # ZeldaLogicEnv finds start/goal from grid tiles
                    env_cbs = ZeldaLogicEnv(semantic_grid=grid, **copy.deepcopy(env_kwargs))
                    t0 = time.perf_counter()
                    cbs = CognitiveBoundedSearch(
                        env_cbs, 
                        persona=persona_name, 
                        timeout=timeout_cbs, 
                        seed=seed
                    )
                    success_c, path_c, states_c, metrics = cbs.solve()
                    cbs_time_ms = (time.perf_counter() - t0) * 1000.0
                    path_len_c = max(0, len(path_c) - 1)
                    per_c = path_efficiency_ratio(path_len_c, manhattan)
                    
                    # Confusion ratio is excess path overhead relative to A*.
                    confusion_ratio = confusion_ratio_vs_oracle(
                        path_len_a,
                        path_len_c,
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
                        'solver_status': cbs_status,
                        'failure_reason': cbs_failure_reason,
                        'time_ms': round(cbs_time_ms, 3),
                    'deliberation_events': int(getattr(metrics, 'deliberation_events', 0) or 0),
                    'budget_exhaustion_events': int(getattr(metrics, 'budget_exhaustion_events', 0) or 0),
                    'peak_frustration': round(float(getattr(metrics, 'peak_frustration', 0.0) or 0.0), 3),
                    'final_deliberation_budget': round(float(getattr(metrics, 'final_deliberation_budget', 0.0) or 0.0), 3),
                    'affordance_reactivations': int(getattr(metrics, 'affordance_reactivations', 0) or 0),
                    'affordance_guided_steps': int(getattr(metrics, 'affordance_guided_steps', 0) or 0),
                    'inventory_change_events': int(getattr(metrics, 'inventory_change_events', 0) or 0),
                    'focus_switches': int(getattr(metrics, 'focus_switches', 0) or 0),
                    'focus_guided_steps': int(getattr(metrics, 'focus_guided_steps', 0) or 0),
                    'oracle_status': str(astar_payload['status']),
                    'oracle_failure_reason': str(astar_payload['failure_reason']),
                    'confusion_ratio': round(confusion_ratio, 3) if np.isfinite(confusion_ratio) else float('nan'),
                    })
                    
                    if verbose:
                        ratio_text = (
                            f"{float(confusion_ratio):.2f}"
                            if np.isfinite(confusion_ratio)
                            else "n/a"
                        )
                        print(
                            f'{map_id} CBS+ ({persona_name}): status={cbs_status} path={path_len_c} '
                            f'states={states_c} confusion_ratio={ratio_text}'
                        )

            except (AttributeError, RuntimeError, ValueError, TypeError) as e:
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
            'solved_maps': sum(1 for r in astar_rows if str(r.get('solver_status')) == 'solved'),
            'timeout_maps': sum(1 for r in astar_rows if str(r.get('solver_status')) == 'timeout'),
            'no_path_maps': sum(1 for r in astar_rows if str(r.get('solver_status')) == 'no_path'),
            'invalid_maps': sum(1 for r in astar_rows if str(r.get('solver_status')) == 'invalid_map'),
        },
        'cbs': {}
    }
    
    # Group CBS by persona
    personas = set(r['persona'] for r in cbs_rows)
    for persona in personas:
        persona_rows = [r for r in cbs_rows if r['persona'] == persona]
        successful = [r for r in persona_rows if r['success']]
        oracle_solved_rows = [r for r in persona_rows if str(r.get('oracle_status')) == 'solved']
        oracle_conditioned_success = [r for r in oracle_solved_rows if r['success']]
        success_rate_given_oracle = (
            len(oracle_conditioned_success) / len(oracle_solved_rows)
            if oracle_solved_rows else float('nan')
        )
        cognitive_gap_rate = (
            1.0 - success_rate_given_oracle
            if oracle_solved_rows else float('nan')
        )
        
        summary['cbs'][persona] = {
            'success_rate': len(successful) / max(1, len(persona_rows)),
            'success_rate_given_oracle_solved': success_rate_given_oracle,
            'cognitive_gap_rate_given_oracle_solved': cognitive_gap_rate,
            'avg_path_length': np.mean([r['path_length'] for r in successful]) if successful else 0,
            'avg_confusion_ratio': finite_mean([r.get('confusion_ratio') for r in successful]),
            'avg_confusion_index': np.mean([r['confusion_index'] for r in persona_rows]) if persona_rows else 0,
            'avg_replans': np.mean([r['replans'] for r in persona_rows]) if persona_rows else 0,
            'avg_backtrack_loops': np.mean([r['backtrack_loops'] for r in persona_rows]) if persona_rows else 0,
            'avg_deliberation_events': np.mean([r['deliberation_events'] for r in persona_rows]) if persona_rows else 0,
            'avg_budget_exhaustion_events': np.mean([r['budget_exhaustion_events'] for r in persona_rows]) if persona_rows else 0,
            'avg_peak_frustration': np.mean([r['peak_frustration'] for r in persona_rows]) if persona_rows else 0,
            'avg_affordance_reactivations': np.mean([r['affordance_reactivations'] for r in persona_rows]) if persona_rows else 0,
            'avg_affordance_guided_steps': np.mean([r['affordance_guided_steps'] for r in persona_rows]) if persona_rows else 0,
            'avg_inventory_change_events': np.mean([r['inventory_change_events'] for r in persona_rows]) if persona_rows else 0,
            'avg_focus_switches': np.mean([r['focus_switches'] for r in persona_rows]) if persona_rows else 0,
            'avg_focus_guided_steps': np.mean([r['focus_guided_steps'] for r in persona_rows]) if persona_rows else 0,
            'oracle_solved_maps': len(oracle_solved_rows),
        }
    
    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    """Print formatted summary statistics."""
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    print(f"Total maps tested: {summary['total_maps']}")
    print("\nA* (Optimal):")
    print(f"  Success rate: {summary['astar']['success_rate']*100:.1f}%")
    print(f"  Avg path length: {summary['astar']['avg_path_length']:.1f}")
    print(f"  Avg states explored: {summary['astar']['avg_states']:.0f}")
    print(
        f"  Oracle statuses: solved={summary['astar']['solved_maps']} "
        f"timeout={summary['astar']['timeout_maps']} "
        f"no_path={summary['astar']['no_path_maps']} "
        f"invalid={summary['astar']['invalid_maps']}"
    )
    
    print("\nCBS+ Results by Persona:")
    for persona, stats in summary['cbs'].items():
        oracle_success_text = (
            f"{stats['success_rate_given_oracle_solved']*100:.1f}%"
            if np.isfinite(stats['success_rate_given_oracle_solved'])
            else "n/a"
        )
        cognitive_gap_text = (
            f"{stats['cognitive_gap_rate_given_oracle_solved']*100:.1f}%"
            if np.isfinite(stats['cognitive_gap_rate_given_oracle_solved'])
            else "n/a"
        )
        print(f"\n  {persona.upper()}:")
        print(f"    Success rate: {stats['success_rate']*100:.1f}%")
        print(f"    Success rate | oracle solved: {oracle_success_text}")
        print(f"    Cognitive gap | oracle solved: {cognitive_gap_text}")
        print(f"    Avg path length: {stats['avg_path_length']:.1f}")
        print(f"    Avg path overhead: {stats['avg_confusion_ratio']:.2%}")
        print(f"    Avg confusion index: {stats['avg_confusion_index']:.3f}")
        print(f"    Avg replans: {stats['avg_replans']:.1f}")
        print(f"    Avg backtrack loops: {stats['avg_backtrack_loops']:.1f}")
        print(f"    Avg deliberation events: {stats['avg_deliberation_events']:.1f}")
        print(f"    Avg budget exhaustions: {stats['avg_budget_exhaustion_events']:.1f}")
        print(f"    Avg peak frustration: {stats['avg_peak_frustration']:.3f}")
        print(f"    Avg affordance reactivations: {stats['avg_affordance_reactivations']:.1f}")
        print(f"    Avg affordance-guided steps: {stats['avg_affordance_guided_steps']:.1f}")
        print(f"    Avg inventory changes: {stats['avg_inventory_change_events']:.1f}")
        print(f"    Avg focus switches: {stats['avg_focus_switches']:.1f}")
        print(f"    Avg focus-guided steps: {stats['avg_focus_guided_steps']:.1f}")
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
        summary_path = Path(args.output).with_name(f"{Path(args.output).stem}_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(_json_safe(summary), indent=2), encoding='utf-8')
        print_summary(summary)


if __name__ == '__main__':
    main()

