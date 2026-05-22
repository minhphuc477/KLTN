"""Process-safe solver worker functions extracted from gui_runner.py."""

from __future__ import annotations

import logging
import os
import pickle

from src.gui.components.constants import GUI_ALGORITHM_NAMES
from src.gui.solver.utils import convert_diagonal_to_4dir as _convert_diagonal_to_4dir


def _solve_in_subprocess(grid, start_pos, goal_pos, algorithm_idx, feature_flags, priority_options,
                         graph=None, room_to_node=None, room_positions=None, node_to_room=None):
    """Compute a path for a grid in a separate process and return a picklable dict."""
    try:
        from src.simulation.validator import ZeldaLogicEnv, StateSpaceAStar, SolverOptions

        grid_arr = grid
        try:
            import numpy as _np
            if not isinstance(grid_arr, _np.ndarray):
                grid_arr = _np.array(grid, dtype=_np.int64)
        except (AttributeError, RuntimeError, ValueError, TypeError):
            grid_arr = grid

        priority_options = dict(priority_options or {})
        strict_original_mode = bool(feature_flags.get('strict_original_mode', False))
        raw_profile = str(priority_options.get('rules_profile', '') or '').strip().lower()
        if raw_profile in {'strict_original', 'original', 'nes'}:
            strict_original_mode = True
        if strict_original_mode:
            priority_options['rules_profile'] = 'strict_original'
        elif raw_profile in {'extended'}:
            priority_options['rules_profile'] = 'extended'
        else:
            priority_options['rules_profile'] = 'vglc_strict'
        if strict_original_mode:
            priority_options['allow_diagonals'] = False

        solver_options = SolverOptions(rules_profile=priority_options['rules_profile'])

        env = ZeldaLogicEnv(
            grid_arr,
            render_mode=False,
            graph=graph,
            room_to_node=room_to_node,
            room_positions=room_positions,
            node_to_room=node_to_room,
            solver_options=solver_options,
        )

        result = {
            'success': False,
            'path': None,
            'teleports': 0,
            'solver_result': None,
            'message': None,
        }

        alg_name = (
            GUI_ALGORITHM_NAMES[algorithm_idx]
            if isinstance(algorithm_idx, int) and 0 <= algorithm_idx < len(GUI_ALGORITHM_NAMES)
            else f'Unknown({algorithm_idx})'
        )

        logger = logging.getLogger(__name__)
        logger.info('===================================================')
        logger.info('SOLVER DISPATCH: algorithm_idx=%s -> %s', algorithm_idx, alg_name)
        logger.info('Start: %s, Goal: %s', start_pos, goal_pos)
        logger.info('===================================================')

        cbs_personas = {
            7: 'balanced',
            8: 'explorer',
            9: 'cautious',
            10: 'forgetful',
            11: 'speedrunner',
            12: 'greedy',
            13: 'completionist',
            14: 'novice',
        }

        def _int_option(name, default):
            raw = priority_options.get(name, os.environ.get(f'KLTN_{name.upper()}', default))
            try:
                return int(raw)
            except (AttributeError, RuntimeError, ValueError, TypeError):
                return int(default)

        def _optional_int_option(name, default):
            raw = priority_options.get(name, os.environ.get(f'KLTN_{name.upper()}', default))
            if raw is None:
                return None
            raw_text = str(raw).strip().lower()
            if raw_text in {'', 'none', 'random'}:
                return None
            try:
                return int(raw)
            except (AttributeError, RuntimeError, ValueError, TypeError):
                return int(default) if default is not None else None

        try:
            if algorithm_idx in cbs_personas:
                from src.simulation.cognitive_bounded_search import PersonaDrivenCognitiveBoundedSearch
                from src.gui.solver.pcbs_route import compress_pcbs_route_for_replay

                persona = cbs_personas[algorithm_idx]
                default_seed_by_persona = {
                    'balanced': 123,
                    'explorer': 123,
                    'cautious': 0,
                    'forgetful': 123,
                    'speedrunner': 123,
                    'greedy': 123,
                    'completionist': 42,
                    'novice': 42,
                }
                pcbs_timeout = _int_option('pcbs_timeout', priority_options.get('timeout', 25000))
                pcbs_seed = _optional_int_option(
                    'pcbs_seed',
                    priority_options.get('seed', default_seed_by_persona.get(persona, 123)),
                )
                route_mode = str(
                    priority_options.get('pcbs_route_mode', os.environ.get('KLTN_PCBS_ROUTE_MODE', 'solution'))
                    or 'solution'
                ).strip().lower()
                pcbs_representation = str(priority_options.get('representation', 'hybrid') or 'hybrid').strip().lower()
                logger.info(
                    'Using P-CBS with persona=%s timeout=%s seed=%s route_mode=%s representation=%s',
                    persona,
                    pcbs_timeout,
                    pcbs_seed,
                    route_mode,
                    pcbs_representation,
                )
                cbs = PersonaDrivenCognitiveBoundedSearch(
                    env,
                    persona=persona,
                    timeout=pcbs_timeout,
                    seed=pcbs_seed,
                    representation=pcbs_representation,
                )
                ok, path, states, metrics = cbs.solve()

                if ok:
                    route_stats = {
                        'raw_trajectory_len': len(path) if path else 0,
                        'display_path_len': len(path) if path else 0,
                        'loops_removed': 0,
                        'compressed': False,
                    }
                    replay_path = path
                    if route_mode not in {'trace', 'trajectory', 'raw'}:
                        replay_path, route_stats = compress_pcbs_route_for_replay(
                            grid=grid_arr,
                            path=path,
                            solver_options=solver_options,
                        )
                        if route_stats.get('compression_error'):
                            logger.warning('P-CBS route compression skipped: %s', route_stats['compression_error'])
                    display_path = _convert_diagonal_to_4dir(replay_path, grid=grid_arr) if replay_path else replay_path
                    cbs_metrics = {
                        'confusion_index': round(metrics.confusion_index, 3),
                        'navigation_entropy': round(metrics.navigation_entropy, 3),
                        'cognitive_load': round(metrics.cognitive_load, 3),
                        'aha_latency': metrics.aha_latency,
                        'unique_tiles': metrics.unique_tiles_visited,
                        'unique_rooms': metrics.unique_rooms_visited,
                        'room_entropy': round(metrics.room_entropy, 3),
                        'total_steps': metrics.total_steps,
                        'peak_memory': metrics.peak_memory_usage,
                        'replans': metrics.replans,
                        'confusion_events': metrics.confusion_events,
                        'backtrack_loops': metrics.backtrack_loops,
                    }
                    result.update({
                        'success': True,
                        'path': display_path,
                        'teleports': 0,
                        'solver_result': {
                            'nodes': states,
                            'original_path_len': len(path) if path else 0,
                            'trajectory_len': len(path) if path else 0,
                            'display_path_len': len(display_path) if display_path else 0,
                            'algorithm': 'P-CBS',
                            'cbs_metrics': cbs_metrics,
                            'persona': persona,
                            'representation': pcbs_representation,
                            'pcbs_timeout': pcbs_timeout,
                            'pcbs_seed': pcbs_seed,
                            'pcbs_route_mode': route_mode,
                            'pcbs_route_compressed': bool(route_stats.get('compressed', False)),
                            'pcbs_loops_removed': int(route_stats.get('loops_removed', 0) or 0),
                        },
                    })
                else:
                    result['message'] = f'P-CBS ({persona}) found no solution (explored {states} states)'

            elif algorithm_idx in {0, 1, 2, 3, 4, 5, 6}:
                from src.simulation import GameStateSearchConfig, SearchRepresentation, run_game_state_solver

                try:
                    rep_mode = SearchRepresentation.parse(priority_options.get('representation', 'hybrid'))
                    config = GameStateSearchConfig(
                        timeout=int(priority_options.get('timeout', 100000)),
                        tie_break=bool(priority_options.get('tie_break', False)),
                        key_boost=bool(priority_options.get('key_boost', False)),
                        enable_ara=bool(priority_options.get('enable_ara', False)),
                        ara_weight=float(priority_options.get('ara_weight', 1.0)),
                        allow_diagonals=bool(priority_options.get('allow_diagonals', False)),
                        rules_profile=str(priority_options.get('rules_profile', 'vglc_strict')),
                        representation=rep_mode,
                        max_depth=int(priority_options.get('max_depth', 500)),
                        use_iddfs=bool(priority_options.get('use_iddfs', True)),
                    )
                except (AttributeError, RuntimeError, ValueError, TypeError):
                    config = GameStateSearchConfig()

                search_result = run_game_state_solver(env, algorithm_idx, config)
                ok = bool(search_result.success)
                path = list(search_result.path or [])
                nodes = int(search_result.states_explored or 0)
                algo_label = search_result.algorithm

                if ok:
                    display_path = _convert_diagonal_to_4dir(path, grid=grid_arr) if path else path
                    result.update({
                        'success': True,
                        'path': display_path,
                        'teleports': 0,
                        'solver_result': {
                            'nodes': nodes,
                            'states_explored': nodes,
                            'original_path_len': len(path) if path else 0,
                            'algorithm': algo_label,
                            'representation': config.representation.value,
                            'rules_profile': str(config.rules_profile),
                            **dict(search_result.metadata or {}),
                        },
                    })
                else:
                    result['message'] = f'{algo_label} found no solution (explored {nodes} states)'

            else:
                ssa = StateSpaceAStar(env, priority_options=priority_options)
                ok, path, nodes = ssa.solve()
                if ok:
                    display_path = _convert_diagonal_to_4dir(path, grid=grid_arr) if path else path
                    result.update({
                        'success': True,
                        'path': display_path,
                        'teleports': 0,
                        'solver_result': {
                            'nodes': nodes,
                            'original_path_len': len(path) if path else 0,
                            'algorithm': 'A* (fallback)',
                        },
                    })
                else:
                    result['message'] = f'Fallback A* found no solution (explored {nodes} states)'
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            logger.exception('Solver exception: %s', exc)
            result['message'] = f'Solver error: {exc}'

        logger.info('SOLVER RESULT: success=%s, path_len=%d', result['success'], len(result['path']) if result['path'] else 0)
        return result
    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
        return {'success': False, 'path': None, 'teleports': 0, 'solver_result': None, 'message': f'Child failed: {exc}'}


def solve_in_subprocess(grid, start_pos, goal_pos, algorithm_idx, feature_flags, priority_options,
                        graph=None, room_to_node=None, room_positions=None, node_to_room=None):
    """Public API wrapper for subprocess-safe solve helper."""
    return _solve_in_subprocess(
        grid,
        start_pos,
        goal_pos,
        algorithm_idx,
        feature_flags,
        priority_options,
        graph=graph,
        room_to_node=room_to_node,
        room_positions=room_positions,
        node_to_room=node_to_room,
    )


def run_solver_and_dump(grid_or_path, start_pos, goal_pos, algorithm_idx, feature_flags, priority_options, out_path,
                        graph=None, room_to_node=None, room_positions=None, node_to_room=None):
    """Run solver and pickle full result to disk (Windows multiprocessing safe)."""
    import sys

    def _log(msg):
        try:
            sys.stderr.write(f'[SOLVER_SUBPROCESS] {msg}\n')
            sys.stderr.flush()
        except (AttributeError, RuntimeError, ValueError, TypeError):
            pass

    _log(f'Started: start={start_pos}, goal={goal_pos}, alg={algorithm_idx}, out={out_path}')

    grid = grid_or_path
    try:
        if isinstance(grid_or_path, str) and os.path.exists(grid_or_path):
            _log(f'Loading grid from file: {grid_or_path}')
            try:
                import numpy as _np
                grid = _np.load(grid_or_path, allow_pickle=False)
                _log(f'Grid loaded: shape={grid.shape}')
            except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
                _log(f'numpy load failed: {exc}, trying pickle')
                try:
                    with open(grid_or_path, 'rb') as gf:
                        grid = pickle.load(gf)
                    _log('Grid loaded via pickle')
                except (AttributeError, RuntimeError, ValueError, TypeError) as exc2:
                    _log(f'pickle load failed: {exc2}')
                    grid = grid_or_path
    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
        _log(f'Grid load exception: {exc}')
        grid = grid_or_path

    _log('Calling solve_in_subprocess...')
    res = solve_in_subprocess(
        grid,
        start_pos,
        goal_pos,
        algorithm_idx,
        feature_flags,
        priority_options,
        graph=graph,
        room_to_node=room_to_node,
        room_positions=room_positions,
        node_to_room=node_to_room,
    )

    path_len = len(res.get('path', []) or []) if res else 0
    _log(f'Solver returned: success={res.get("success") if res else None}, path_len={path_len}')

    try:
        with open(out_path, 'wb') as f:
            pickle.dump(res, f)
        _log(f'Result written to {out_path}')
    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
        _log(f'Failed to write result: {exc}')
        try:
            with open(out_path, 'wb') as f:
                pickle.dump({'success': False, 'message': f'failed to write output: {exc}'}, f)
        except (AttributeError, RuntimeError, ValueError, TypeError):
            pass


def run_preview_and_dump(grid_or_path, start_pos, goal_pos, algorithm_idx, feature_flags, priority_options, out_path,
                         graph=None, room_to_node=None, room_positions=None, node_to_room=None):
    """Run quick preview solver and write compact result to disk."""
    try:
        grid = grid_or_path
        try:
            if isinstance(grid_or_path, str) and os.path.exists(grid_or_path):
                import numpy as _np
                grid = _np.load(grid_or_path, allow_pickle=False)
        except (AttributeError, RuntimeError, ValueError, TypeError):
            pass

        res = solve_in_subprocess(
            grid,
            start_pos,
            goal_pos,
            algorithm_idx,
            feature_flags,
            {**priority_options, 'ara_weight': priority_options.get('ara_weight', 1.0)},
            graph=graph,
            room_to_node=room_to_node,
            room_positions=room_positions,
            node_to_room=node_to_room,
        )
        out = {
            'success': res.get('success', False),
            'path': res.get('path'),
            'solver_result': res.get('solver_result', {}),
            'message': res.get('message'),
        }
        try:
            with open(out_path, 'wb') as f:
                pickle.dump(out, f)
        except (AttributeError, RuntimeError, ValueError, TypeError):
            try:
                with open(out_path, 'wb') as f:
                    pickle.dump({'success': False, 'message': 'failed to write preview output'}, f)
            except (AttributeError, RuntimeError, ValueError, TypeError):
                pass
    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
        try:
            with open(out_path, 'wb') as f:
                pickle.dump({'success': False, 'message': str(exc)}, f)
        except (AttributeError, RuntimeError, ValueError, TypeError):
            pass



