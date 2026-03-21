"""Process-safe solver worker functions extracted from gui_runner.py."""

from __future__ import annotations

import logging
import os
import pickle

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
        except Exception:
            grid_arr = grid

        priority_options = dict(priority_options or {})
        strict_original_mode = bool(feature_flags.get('strict_original_mode', False))
        raw_profile = str(priority_options.get('rules_profile', '') or '').strip().lower()
        if raw_profile in {'strict_original', 'original', 'nes'}:
            strict_original_mode = True
        priority_options['rules_profile'] = 'strict_original' if strict_original_mode else 'extended'
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

        algorithm_names = [
            'A*', 'BFS', 'Dijkstra', 'Greedy', 'D* Lite',
            'DFS/IDDFS', 'Bidirectional A*',
            'CBS (Balanced)', 'CBS (Explorer)', 'CBS (Cautious)',
            'CBS (Forgetful)', 'CBS (Speedrunner)', 'CBS (Greedy)'
        ]
        alg_name = algorithm_names[algorithm_idx] if algorithm_idx < len(algorithm_names) else f'Unknown({algorithm_idx})'

        logger = logging.getLogger(__name__)
        logger.info('═══════════════════════════════════════════════════')
        logger.info('SOLVER DISPATCH: algorithm_idx=%s -> %s', algorithm_idx, alg_name)
        logger.info('Start: %s, Goal: %s', start_pos, goal_pos)
        logger.info('═══════════════════════════════════════════════════')

        cbs_personas = {
            7: 'balanced',
            8: 'explorer',
            9: 'cautious',
            10: 'forgetful',
            11: 'speedrunner',
            12: 'greedy',
        }

        try:
            if algorithm_idx in cbs_personas:
                from src.simulation.cognitive_bounded_search import CognitiveBoundedSearch

                persona = cbs_personas[algorithm_idx]
                logger.info('Using CBS with persona=%s', persona)
                cbs = CognitiveBoundedSearch(env, persona=persona, timeout=100000)
                ok, path, states, metrics = cbs.solve()

                if ok:
                    display_path = _convert_diagonal_to_4dir(path, grid=grid_arr) if path else path
                    cbs_metrics = {
                        'confusion_index': round(metrics.confusion_index, 3),
                        'navigation_entropy': round(metrics.navigation_entropy, 3),
                        'cognitive_load': round(metrics.cognitive_load, 3),
                        'aha_latency': metrics.aha_latency,
                        'unique_tiles': metrics.unique_tiles_visited,
                        'total_steps': metrics.total_steps,
                        'peak_memory': metrics.peak_memory_usage,
                        'replans': metrics.replans,
                        'confusion_events': metrics.confusion_events,
                    }
                    result.update({
                        'success': True,
                        'path': display_path,
                        'teleports': 0,
                        'solver_result': {
                            'nodes': states,
                            'original_path_len': len(path) if path else 0,
                            'cbs_metrics': cbs_metrics,
                            'persona': persona,
                        },
                    })
                else:
                    result['message'] = f'CBS ({persona}) found no solution (explored {states} states)'

            elif algorithm_idx in {0, 1, 2, 3}:
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
                        rules_profile=str(priority_options.get('rules_profile', 'extended')),
                        representation=rep_mode,
                    )
                except Exception:
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
                        },
                    })
                else:
                    result['message'] = f'{algo_label} found no solution (explored {nodes} states)'

            elif algorithm_idx == 4:
                from src.simulation.dstar_lite import DStarLiteSolver

                dstar = DStarLiteSolver(env, heuristic_mode='balanced')
                start_state = env.state.copy()
                ok, path, nodes = dstar.solve(start_state)

                if ok:
                    display_path = _convert_diagonal_to_4dir(path, grid=grid_arr) if path else path
                    algo_label = 'D* Lite (fallback: A*)' if getattr(dstar, 'used_fallback', False) else 'D* Lite'
                    result.update({
                        'success': True,
                        'path': display_path,
                        'teleports': 0,
                        'solver_result': {
                            'nodes': nodes,
                            'original_path_len': len(path) if path else 0,
                            'algorithm': algo_label,
                            'replans': dstar.replans_count,
                        },
                    })
                else:
                    result['message'] = f'D* Lite found no solution (explored {nodes} states)'

            elif algorithm_idx == 5:
                from src.simulation.state_space_dfs import StateSpaceDFS

                dfs = StateSpaceDFS(
                    env,
                    timeout=100000,
                    max_depth=500,
                    allow_diagonals=priority_options.get('allow_diagonals', False),
                    use_iddfs=True,
                )
                ok, path, nodes = dfs.solve()

                if ok:
                    display_path = _convert_diagonal_to_4dir(path, grid=grid_arr) if path else path
                    result.update({
                        'success': True,
                        'path': display_path,
                        'teleports': 0,
                        'solver_result': {
                            'nodes': nodes,
                            'original_path_len': len(path) if path else 0,
                            'algorithm': 'DFS/IDDFS',
                            'max_depth': dfs.metrics.max_depth_reached,
                            'backtracks': dfs.metrics.backtrack_count,
                        },
                    })
                else:
                    result['message'] = f'DFS/IDDFS found no solution (explored {nodes} states)'

            elif algorithm_idx == 6:
                from src.simulation.bidirectional_astar import BidirectionalAStar

                bidir = BidirectionalAStar(
                    env,
                    timeout=100000,
                    allow_diagonals=priority_options.get('allow_diagonals', False),
                    heuristic_mode='balanced',
                )
                ok, path, nodes = bidir.solve()

                if ok:
                    display_path = _convert_diagonal_to_4dir(path, grid=grid_arr) if path else path
                    algo_label = 'Bidirectional A* (fallback: A*)' if getattr(bidir, 'used_fallback', False) else 'Bidirectional A*'
                    result.update({
                        'success': True,
                        'path': display_path,
                        'teleports': 0,
                        'solver_result': {
                            'nodes': nodes,
                            'original_path_len': len(path) if path else 0,
                            'algorithm': algo_label,
                            'meeting_point': bidir.meeting_point,
                            'collision_checks': bidir.collision_checks,
                        },
                    })
                else:
                    result['message'] = f'Bidirectional A* found no solution (explored {nodes} states)'

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
        except Exception as exc:
            logger.exception('Solver exception: %s', exc)
            result['message'] = f'Solver error: {exc}'

        logger.info('SOLVER RESULT: success=%s, path_len=%d', result['success'], len(result['path']) if result['path'] else 0)
        return result
    except Exception as exc:
        return {'success': False, 'path': None, 'teleports': 0, 'solver_result': None, 'message': f'Child failed: {exc}'}


def _run_solver_and_dump(grid_or_path, start_pos, goal_pos, algorithm_idx, feature_flags, priority_options, out_path,
                         graph=None, room_to_node=None, room_positions=None, node_to_room=None):
    """Run solver and pickle result to disk (Windows multiprocessing safe)."""
    import sys

    def _log(msg):
        try:
            sys.stderr.write(f'[SOLVER_SUBPROCESS] {msg}\n')
            sys.stderr.flush()
        except Exception:
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
            except Exception as exc:
                _log(f'numpy load failed: {exc}, trying pickle')
                try:
                    with open(grid_or_path, 'rb') as gf:
                        grid = pickle.load(gf)
                    _log('Grid loaded via pickle')
                except Exception as exc2:
                    _log(f'pickle load failed: {exc2}')
                    grid = grid_or_path
    except Exception as exc:
        _log(f'Grid load exception: {exc}')
        grid = grid_or_path

    _log('Calling _solve_in_subprocess...')
    res = _solve_in_subprocess(
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
    except Exception as exc:
        _log(f'Failed to write result: {exc}')
        try:
            with open(out_path, 'wb') as f:
                pickle.dump({'success': False, 'message': f'failed to write output: {exc}'}, f)
        except Exception:
            pass


def _run_preview_and_dump(grid_or_path, start_pos, goal_pos, algorithm_idx, feature_flags, priority_options, out_path,
                          graph=None, room_to_node=None, room_positions=None, node_to_room=None):
    """Run quick preview solver and write compact result to disk."""
    try:
        grid = grid_or_path
        try:
            if isinstance(grid_or_path, str) and os.path.exists(grid_or_path):
                import numpy as _np
                grid = _np.load(grid_or_path, allow_pickle=False)
        except Exception:
            pass

        res = _solve_in_subprocess(
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
        except Exception:
            try:
                with open(out_path, 'wb') as f:
                    pickle.dump({'success': False, 'message': 'failed to write preview output'}, f)
            except Exception:
                pass
    except Exception as exc:
        try:
            with open(out_path, 'wb') as f:
                pickle.dump({'success': False, 'message': str(exc)}, f)
        except Exception:
            pass
