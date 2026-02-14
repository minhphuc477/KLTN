"""
GUI Runner for ZAVE (Zelda AI Validation Environment)
====================================================

Interactive visual interface for validating Zelda dungeon maps.

Features:
- Real-time visualization of map and agent
- Manual play mode (arrow keys)
- Auto-solve mode (A* pathfinding)
- Map loading from processed data
- Smooth delta-time animations
- Heatmap overlay for A* search visualization
- Modern semi-transparent HUD

Controls:
- Arrow Keys: Move Link
- SPACE: Run A* solver (auto-solve)
- R: Reset map
- N: Next map (if multiple loaded)
- P: Previous map
- H: Toggle heatmap overlay
- ESC: Quit


"""

import sys
import os
import time
import math
import logging
import threading
import numpy as np
from typing import Tuple, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# Allow debug mode via env var KLTN_LOG_LEVEL=DEBUG for interactive troubleshooting
if os.environ.get('KLTN_LOG_LEVEL', '').upper() == 'DEBUG':
    logger.setLevel(logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)

# Allow targeted input diagnostics via env var KLTN_DEBUG_INPUT=1
DEBUG_INPUT_ACTIVE = os.environ.get('KLTN_DEBUG_INPUT', '') == '1'
if DEBUG_INPUT_ACTIVE:
    logger.info('INPUT_DIAG: KLTN_DEBUG_INPUT is active (diagnostic input dumps enabled)')

# DEBUG: Synchronous solver mode to bypass multiprocessing issues
# Set KLTN_SYNC_SOLVER=1 only for debugging (will block UI during solving)
# ASYNC mode by default for responsive UI during long solves
DEBUG_SYNC_SOLVER = os.environ.get('KLTN_SYNC_SOLVER', '0') == '1'
if DEBUG_SYNC_SOLVER:
    logger.info('Solver running in SYNC mode - UI will freeze during solving (direct execution, no pickle files)')
else:
    logger.info('Solver running in ASYNC mode - animated solving enabled')

# DEBUG: Verbose solver flow logging
DEBUG_SOLVER_FLOW = os.environ.get('KLTN_DEBUG_SOLVER_FLOW', '0') == '1'
if DEBUG_SOLVER_FLOW:
    logger.setLevel(logging.DEBUG)
    logger.warning('DEBUG: KLTN_DEBUG_SOLVER_FLOW=1 - Verbose solver logging enabled')

# Import simulation components (use new canonical paths)
from src.simulation.validator import (
    ZeldaLogicEnv, 
    ZeldaValidator, 
    StateSpaceAStar,
    SanityChecker,
    create_test_map,
    SEMANTIC_PALETTE,
    Action,
    GameState,
    ACTION_DELTAS,
    PUSHABLE_IDS,
    WALKABLE_IDS
)

# Local matcher for topology repair (use new canonical path)
from src.data.zelda_core import RoomGraphMatcher

# Try to import Pygame
# NOTE: Importing pygame does NOT create a window - windows are only created
# when pygame.display.set_mode() is called. The ZeldaGUI class is only
# instantiated in main(), which is protected by if __name__ == "__main__".
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    logger.warning("Pygame not installed. Run 'pip install pygame' for GUI support.")

# Try to import new visualization system
try:
    from src.visualization.renderer import ZeldaRenderer, ThemeConfig, Vector2
    from src.visualization.effects import (
        EffectManager, PopEffect, FlashEffect, RippleEffect,
        ItemCollectionEffect, ItemUsageEffect, ItemMarkerEffect
    )
    from src.visualization.hud import ModernHUD, HUDTheme
    from src.visualization.path_preview import PathPreviewDialog
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logger.warning("New visualization system not available, using fallback rendering.")

# Try to import GUI widgets
try:
    from src.gui.widgets import (
        CheckboxWidget, DropdownWidget, ButtonWidget,
        WidgetManager, WidgetTheme
    )
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False
    logger.warning("GUI widgets not available, using keyboard controls only.")

# --- Subprocess-based solver helper ---
# This helper runs inside a separate process to avoid blocking the main thread
# with heavy CPU-bound pathfinding work (which would starve the GUI due to the GIL).
import pickle
import tempfile
import multiprocessing

def _convert_diagonal_to_4dir(path, grid=None):
    """Convert a path with diagonal moves to 4-directional movement.
    
    Each diagonal move (e.g., NE) is split into two orthogonal moves.
    This preserves pathfinding speed while showing standard grid-based animation.
    
    CRITICAL FIX: When grid is provided, we validate intermediate positions
    to avoid routing through water/walls. We try vertical-first, then
    horizontal-first, and pick whichever doesn't go through obstacles.
    
    Args:
        path: List of (row, col) tuples
        grid: Optional numpy array of tile IDs - used to validate intermediate positions
    
    Returns:
        List of (row, col) tuples with only orthogonal (4-dir) moves
    """
    if not path or len(path) < 2:
        return path
    
    # Define obstacle tile IDs that intermediate positions should avoid
    # Import here to avoid circular imports
    try:
        from src.simulation.validator import SEMANTIC_PALETTE, BLOCKING_IDS, WATER_IDS
        obstacle_ids = BLOCKING_IDS | WATER_IDS
    except ImportError:
        # Fallback: water=40, wall=2, void=0
        obstacle_ids = {0, 2, 40}
    
    def is_walkable(pos):
        """Check if a position is walkable (not an obstacle)."""
        if grid is None:
            return True  # No grid provided, assume walkable
        r, c = pos
        if r < 0 or r >= grid.shape[0] or c < 0 or c >= grid.shape[1]:
            return False
        tile_id = int(grid[r, c])
        return tile_id not in obstacle_ids
    
    converted = [path[0]]  # Start position
    
    for i in range(len(path) - 1):
        curr = path[i]
        next_pos = path[i + 1]
        dr = next_pos[0] - curr[0]
        dc = next_pos[1] - curr[1]
        
        # If diagonal (both dr and dc non-zero), split into two moves
        if dr != 0 and dc != 0:
            # Try both intermediate options and pick the walkable one
            # Option 1: Move vertically first (change row, keep col)
            vert_first = (curr[0] + dr, curr[1])
            # Option 2: Move horizontally first (keep row, change col)
            horz_first = (curr[0], curr[1] + dc)
            
            # Pick the intermediate that avoids obstacles
            if is_walkable(vert_first):
                intermediate = vert_first
            elif is_walkable(horz_first):
                intermediate = horz_first
            else:
                # Both blocked - this shouldn't happen if solver is correct
                # Fall back to vertical-first and log warning
                intermediate = vert_first
                logger.warning(
                    'Diagonal conversion: both intermediates blocked at %s->%s (vert=%s, horz=%s)',
                    curr, next_pos, vert_first, horz_first
                )
            
            converted.append(intermediate)
            converted.append(next_pos)
        else:
            # Orthogonal move - keep as is
            converted.append(next_pos)
    
    return converted

def _solve_in_subprocess(grid, start_pos, goal_pos, algorithm_idx, feature_flags, priority_options,
                         graph=None, room_to_node=None, room_positions=None, node_to_room=None):
    """Compute a path for a grid in a separate process and return a picklable dict.

    Arguments: 
        grid: may be an ndarray-like or nested lists
        graph: Optional NetworkX DiGraph for room connectivity (enables stair traversal)
        room_to_node: Optional mapping of room positions to graph nodes
        room_positions: Optional mapping of room positions to pixel offsets
        node_to_room: Optional mapping of graph nodes to room positions (includes virtual nodes)
    
    The function re-creates a ZeldaLogicEnv locally inside the child process and runs 
    the same solver logic used on the main thread.
    """
    try:
        # Re-import heavy modules inside child process
        from src.simulation.validator import ZeldaLogicEnv, StateSpaceAStar, SEMANTIC_PALETTE, GameState
        from src.data.zelda_core import DungeonSolver, ValidationMode
        import time

        # If a NumPy-like file path was passed earlier, caller will have loaded it; ensure grid is ndarray
        grid_arr = grid
        try:
            import numpy as _np
            if not isinstance(grid_arr, _np.ndarray):
                grid_arr = _np.array(grid, dtype=_np.int64)
        except Exception:
            # If numpy not available or conversion failed, keep original
            grid_arr = grid

        # CRITICAL: Pass graph connectivity data to enable stair traversal
        env = ZeldaLogicEnv(grid_arr, render_mode=False, graph=graph, 
                            room_to_node=room_to_node, room_positions=room_positions,
                            node_to_room=node_to_room)

        result = {
            'success': False,
            'path': None,
            'teleports': 0,
            'solver_result': None,
            'message': None,
        }

        # Algorithm names for debug logging
        algorithm_names = ["A*", "BFS", "Dijkstra", "Greedy", "D* Lite",
                          "DFS/IDDFS", "Bidirectional A*",
                          "CBS (Balanced)", "CBS (Explorer)", "CBS (Cautious)",
                          "CBS (Forgetful)", "CBS (Speedrunner)", "CBS (Greedy)"]
        alg_name = algorithm_names[algorithm_idx] if algorithm_idx < len(algorithm_names) else f"Unknown({algorithm_idx})"
        
        # CRITICAL DEBUG: Log which algorithm is being used
        import logging
        logger = logging.getLogger(__name__)
        logger.info('═══════════════════════════════════════════════════')
        logger.info(f'🔍 SOLVER DISPATCH: algorithm_idx={algorithm_idx} → {alg_name}')
        logger.info(f'   Start: {start_pos}, Goal: {goal_pos}')
        logger.info('═══════════════════════════════════════════════════')

        # Determine if CBS solver is selected (indices 7-12)
        cbs_personas = {
            7: 'balanced',
            8: 'explorer',
            9: 'cautious',
            10: 'forgetful',
            11: 'speedrunner',
            12: 'greedy'
        }
        
        # Single solver attempt - PROPER DISPATCH TO DIFFERENT ALGORITHMS
        try:
            if algorithm_idx in cbs_personas:
                # Use CBS solver with selected persona
                from src.simulation.cognitive_bounded_search import CognitiveBoundedSearch
                
                persona = cbs_personas[algorithm_idx]
                logger.info(f'✓ Using CBS with persona={persona}')
                cbs = CognitiveBoundedSearch(env, persona=persona, timeout=100000)
                ok, path, states, metrics = cbs.solve()
                
                if ok:
                    # Convert diagonal path to 4-directional for standard animation display
                    display_path = _convert_diagonal_to_4dir(path, grid=grid_arr) if path else path
                    
                    # Extract CBS metrics for display
                    cbs_metrics = {
                        'confusion_index': round(metrics.confusion_index, 3),
                        'navigation_entropy': round(metrics.navigation_entropy, 3),
                        'cognitive_load': round(metrics.cognitive_load, 3),
                        'aha_latency': metrics.aha_latency,
                        'unique_tiles': metrics.unique_tiles_visited,
                        'total_steps': metrics.total_steps,
                        'peak_memory': metrics.peak_memory_usage,
                        'replans': metrics.replans,
                        'confusion_events': metrics.confusion_events
                    }
                    
                    logger.info(f'✓ CBS succeeded: path_len={len(display_path)}, states={states}')
                    result.update({
                        'success': True, 
                        'path': display_path,
                        'teleports': 0, 
                        'solver_result': {
                            'nodes': states, 
                            'original_path_len': len(path) if path else 0,
                            'cbs_metrics': cbs_metrics,
                            'persona': persona
                        }
                    })
                else:
                    logger.warning(f'✗ CBS failed: explored {states} states')
                    result['message'] = f'CBS ({persona}) found no solution (explored {states} states)'
            
            elif algorithm_idx == 0:
                # A* - Use StateSpaceAStar (existing implementation)
                logger.info('✓ Using A* (StateSpaceAStar)')
                ssa = StateSpaceAStar(env, priority_options=priority_options)
                ok, path, nodes = ssa.solve()
                if ok:
                    display_path = _convert_diagonal_to_4dir(path, grid=grid_arr) if path else path
                    logger.info(f'✓ A* succeeded: path_len={len(display_path)}, nodes={nodes}')
                    result.update({
                        'success': True, 
                        'path': display_path,
                        'teleports': 0, 
                        'solver_result': {'nodes': nodes, 'original_path_len': len(path) if path else 0, 'algorithm': 'A*'}
                    })
                else:
                    logger.warning(f'✗ A* failed: explored {nodes} states')
                    result['message'] = f'A* found no solution (explored {nodes} states)'
            
            elif algorithm_idx == 1:
                # BFS - Use StateSpaceAStar with BFS search mode (game-state-aware)
                logger.info('✓ Using BFS (Breadth-First Search via StateSpaceAStar)')
                ssa = StateSpaceAStar(env, priority_options=priority_options, search_mode='bfs')
                ok, path, nodes = ssa.solve()
                if ok:
                    display_path = _convert_diagonal_to_4dir(path, grid=grid_arr) if path else path
                    logger.info(f'✓ BFS succeeded: path_len={len(display_path)}, nodes={nodes}')
                    result.update({
                        'success': True,
                        'path': display_path,
                        'teleports': 0,
                        'solver_result': {'nodes': nodes, 'original_path_len': len(path) if path else 0, 'algorithm': 'BFS'}
                    })
                else:
                    logger.warning(f'✗ BFS failed: explored {nodes} states')
                    result['message'] = f'BFS found no solution (explored {nodes} states)'
            
            elif algorithm_idx == 2:
                # Dijkstra - Use StateSpaceAStar with Dijkstra search mode (no heuristic)
                logger.info('✓ Using Dijkstra (Uniform Cost Search via StateSpaceAStar)')
                ssa = StateSpaceAStar(env, priority_options=priority_options, search_mode='dijkstra')
                ok, path, nodes = ssa.solve()
                if ok:
                    display_path = _convert_diagonal_to_4dir(path, grid=grid_arr) if path else path
                    logger.info(f'✓ Dijkstra succeeded: path_len={len(display_path)}, nodes={nodes}')
                    result.update({
                        'success': True,
                        'path': display_path,
                        'teleports': 0,
                        'solver_result': {'nodes': nodes, 'original_path_len': len(path) if path else 0, 'algorithm': 'Dijkstra'}
                    })
                else:
                    logger.warning(f'✗ Dijkstra failed: explored {nodes} states')
                    result['message'] = f'Dijkstra found no solution (explored {nodes} states)'
            
            elif algorithm_idx == 3:
                # Greedy Best-First - Use StateSpaceAStar with Greedy search mode (heuristic only)
                logger.info('✓ Using Greedy Best-First Search (via StateSpaceAStar)')
                ssa = StateSpaceAStar(env, priority_options=priority_options, search_mode='greedy')
                ok, path, nodes = ssa.solve()
                if ok:
                    display_path = _convert_diagonal_to_4dir(path, grid=grid_arr) if path else path
                    logger.info(f'✓ Greedy succeeded: path_len={len(display_path)}, nodes={nodes}')
                    result.update({
                        'success': True,
                        'path': display_path,
                        'teleports': 0,
                        'solver_result': {'nodes': nodes, 'original_path_len': len(path) if path else 0, 'algorithm': 'Greedy'}
                    })
                else:
                    logger.warning(f'✗ Greedy failed: explored {nodes} states')
                    result['message'] = f'Greedy found no solution (explored {nodes} states)'
            
            elif algorithm_idx == 4:
                # D* Lite - Use incremental replanning search
                logger.info('✓ Using D* Lite (incremental replanning)')
                from src.simulation.dstar_lite import DStarLiteSolver
                
                dstar = DStarLiteSolver(env, heuristic_mode="balanced")
                # Create initial state
                start_state = env.state.copy()
                ok, path, nodes = dstar.solve(start_state)
                
                if ok:
                    display_path = _convert_diagonal_to_4dir(path, grid=grid_arr) if path else path
                    logger.info(f'✓ D* Lite succeeded: path_len={len(display_path)}, nodes={nodes}')
                    result.update({
                        'success': True,
                        'path': display_path,
                        'teleports': 0,
                        'solver_result': {
                            'nodes': nodes,
                            'original_path_len': len(path) if path else 0,
                            'algorithm': 'D* Lite',
                            'replans': dstar.replans_count
                        }
                    })
                else:
                    logger.warning(f'✗ D* Lite failed: explored {nodes} states')
                    result['message'] = f'D* Lite found no solution (explored {nodes} states)'
            
            elif algorithm_idx == 5:
                # DFS/IDDFS - Depth-first search with iterative deepening
                logger.info('✓ Using DFS/IDDFS (iterative deepening depth-first search)')
                from src.simulation.state_space_dfs import StateSpaceDFS
                
                dfs = StateSpaceDFS(
                    env,
                    timeout=100000,
                    max_depth=500,
                    allow_diagonals=priority_options.get('allow_diagonals', False),
                    use_iddfs=True  # Use IDDFS for better completeness
                )
                ok, path, nodes = dfs.solve()
                
                if ok:
                    display_path = _convert_diagonal_to_4dir(path, grid=grid_arr) if path else path
                    logger.info(f'✓ DFS/IDDFS succeeded: path_len={len(display_path)}, nodes={nodes}')
                    result.update({
                        'success': True,
                        'path': display_path,
                        'teleports': 0,
                        'solver_result': {
                            'nodes': nodes,
                            'original_path_len': len(path) if path else 0,
                            'algorithm': 'DFS/IDDFS',
                            'max_depth': dfs.metrics.max_depth_reached,
                            'backtracks': dfs.metrics.backtrack_count
                        }
                    })
                else:
                    logger.warning(f'✗ DFS/IDDFS failed: explored {nodes} states')
                    result['message'] = f'DFS/IDDFS found no solution (explored {nodes} states)'
            
            elif algorithm_idx == 6:
                # Bidirectional A* - Meet-in-the-middle search
                logger.info('✓ Using Bidirectional A* (meet-in-the-middle search)')
                from src.simulation.bidirectional_astar import BidirectionalAStar
                
                bidir = BidirectionalAStar(
                    env,
                    timeout=100000,
                    allow_diagonals=priority_options.get('allow_diagonals', False),
                    heuristic_mode="balanced"
                )
                ok, path, nodes = bidir.solve()
                
                if ok:
                    display_path = _convert_diagonal_to_4dir(path, grid=grid_arr) if path else path
                    logger.info(f'✓ Bidirectional A* succeeded: path_len={len(display_path)}, nodes={nodes}')
                    result.update({
                        'success': True,
                        'path': display_path,
                        'teleports': 0,
                        'solver_result': {
                            'nodes': nodes,
                            'original_path_len': len(path) if path else 0,
                            'algorithm': 'Bidirectional A*',
                            'meeting_point': bidir.meeting_point,
                            'collision_checks': bidir.collision_checks
                        }
                    })
                else:
                    logger.warning(f'✗ Bidirectional A* failed: explored {nodes} states')
                    result['message'] = f'Bidirectional A* found no solution (explored {nodes} states)'
            
            else:
                # Unknown algorithm - fallback to A*
                logger.warning(f'⚠ Unknown algorithm_idx={algorithm_idx}, falling back to A*')
                ssa = StateSpaceAStar(env, priority_options=priority_options)
                ok, path, nodes = ssa.solve()
                if ok:
                    display_path = _convert_diagonal_to_4dir(path, grid=grid_arr) if path else path
                    result.update({
                        'success': True, 
                        'path': display_path,
                        'teleports': 0, 
                        'solver_result': {'nodes': nodes, 'original_path_len': len(path) if path else 0, 'algorithm': 'A* (fallback)'}
                    })
                else:
                    result['message'] = f'Fallback A* found no solution (explored {nodes} states)'
                    
        except Exception as e:
            logger.exception(f'✗ Solver exception: {e}')
            result['message'] = f'Solver error: {e}'

        logger.info(f'🔍 SOLVER RESULT: success={result["success"]}, path_len={len(result["path"]) if result["path"] else 0}')
        return result
    except Exception as e:
        return {'success': False, 'path': None, 'teleports': 0, 'solver_result': None, 'message': f'Child failed: {e}'}


def _run_solver_and_dump(grid_or_path, start_pos, goal_pos, algorithm_idx, feature_flags, priority_options, out_path,
                         graph=None, room_to_node=None, room_positions=None, node_to_room=None):
    """Top-level helper to run solver and pickle the result to disk.

    This must be module-level so it is picklable by multiprocessing on Windows.
    `grid_or_path` may be a nested list (legacy) or a filesystem path to a .npy file.
    
    Args:
        graph: Optional NetworkX DiGraph for room connectivity
        room_to_node: Optional mapping of room positions to graph nodes
        room_positions: Optional mapping of room positions to pixel offsets
        node_to_room: Optional mapping of graph nodes to room positions (includes virtual nodes)
    """
    import sys
    # Subprocess logging - write to stderr since stdout may not be visible
    def _log(msg):
        try:
            sys.stderr.write(f'[SOLVER_SUBPROCESS] {msg}\n')
            sys.stderr.flush()
        except Exception:
            pass
    
    _log(f'Started: start={start_pos}, goal={goal_pos}, alg={algorithm_idx}, out={out_path}')
    
    # Load grid from file if a path was provided (avoids expensive pickling of large arrays)
    grid = grid_or_path
    try:
        if isinstance(grid_or_path, str) and os.path.exists(grid_or_path):
            _log(f'Loading grid from file: {grid_or_path}')
            try:
                import numpy as _np
                grid = _np.load(grid_or_path, allow_pickle=False)
                _log(f'Grid loaded: shape={grid.shape}')
            except Exception as e:
                _log(f'numpy load failed: {e}, trying pickle')
                try:
                    # Last effort: use pickle
                    with open(grid_or_path, 'rb') as gf:
                        grid = pickle.load(gf)
                    _log('Grid loaded via pickle')
                except Exception as e2:
                    _log(f'pickle load failed: {e2}')
                    grid = grid_or_path
    except Exception as e:
        _log(f'Grid load exception: {e}')
        grid = grid_or_path

    _log('Calling _solve_in_subprocess...')
    res = _solve_in_subprocess(grid, start_pos, goal_pos, algorithm_idx, feature_flags, priority_options,
                               graph=graph, room_to_node=room_to_node, room_positions=room_positions,
                               node_to_room=node_to_room)
    
    path_len = len(res.get('path', []) or []) if res else 0
    _log(f'Solver returned: success={res.get("success") if res else None}, path_len={path_len}')
    
    try:
        with open(out_path, 'wb') as f:
            pickle.dump(res, f)
        _log(f'Result written to {out_path}')
    except Exception as e:
        _log(f'Failed to write result: {e}')
        # Best-effort logging - child process may not have same stdout; write minimal failure to file
        try:
            with open(out_path, 'wb') as f:
                pickle.dump({'success': False, 'message': f'failed to write output: {e}'}, f)
        except Exception:
            pass


def _run_preview_and_dump(grid_or_path, start_pos, goal_pos, algorithm_idx, feature_flags, priority_options, out_path,
                          graph=None, room_to_node=None, room_positions=None, node_to_room=None):
    """Lightweight preview runner that writes a short preview result quickly.

    Runs in a separate process to avoid blocking the GUI. Attempts a fast StateSpaceAStar
    with a small timeout or returns failure quickly.
    """
    try:
        # Try to reuse the same solver machinery but with aggressive timeout to ensure quick return
        grid = grid_or_path
        try:
            if isinstance(grid_or_path, str) and os.path.exists(grid_or_path):
                import numpy as _np
                grid = _np.load(grid_or_path, allow_pickle=False)
        except Exception:
            pass

        # Use the shared helper with graph data for stair traversal
        res = _solve_in_subprocess(grid, start_pos, goal_pos, algorithm_idx, feature_flags, 
                                   {**priority_options, 'ara_weight': priority_options.get('ara_weight', 1.0)},
                                   graph=graph, room_to_node=room_to_node, room_positions=room_positions,
                                   node_to_room=node_to_room)
        # If result is heavy, we can trim it to only include path and nodes
        out = {'success': res.get('success', False), 'path': res.get('path'), 'solver_result': res.get('solver_result', {}), 'message': res.get('message')}
        try:
            with open(out_path, 'wb') as f:
                pickle.dump(out, f)
        except Exception:
            try:
                with open(out_path, 'wb') as f:
                    pickle.dump({'success': False, 'message': 'failed to write preview output'})
            except Exception:
                pass
    except Exception as e:
        try:
            with open(out_path, 'wb') as f:
                pickle.dump({'success': False, 'message': str(e)})
        except Exception:
            pass



class ToastNotification:
    """Floating toast message with auto-dismiss and fade animations."""
    def __init__(self, message: str, duration: float = 3.0, toast_type: str = 'info'):
        self.message = message
        self.duration = duration
        self.toast_type = toast_type  # 'info', 'success', 'error', 'warning'
        self.created_at = time.time()
        
        # Colors by type
        self.colors = {
            'info': (100, 200, 255),
            'success': (100, 255, 150),
            'error': (255, 100, 100),
            'warning': (255, 200, 100)
        }
    
    def is_expired(self) -> bool:
        """Check if toast should be removed."""
        return time.time() - self.created_at > self.duration
    
    def get_alpha(self) -> int:
        """Calculate current alpha for fade in/out animation and clamp it into valid range."""
        elapsed = time.time() - self.created_at
        alpha = 240
        # Fade in (0.3s)
        if elapsed < 0.3:
            alpha = int((elapsed / 0.3) * 240)
        # Fade out (last 0.5s)
        elif elapsed > self.duration - 0.5:
            remaining = self.duration - elapsed
            alpha = int((remaining / 0.5) * 240)
        # Ensure alpha is valid integer in [0,255]
        try:
            alpha = int(alpha)
        except Exception:
            alpha = 0
        alpha = max(0, min(255, alpha))
        return alpha
    
    def render(self, surface: pygame.Surface, center_x: int, y: int):
        """Render toast notification at specified position."""
        alpha = self.get_alpha()
        font = pygame.font.Font(None, 20)
        text_surf = font.render(self.message, True, (255, 255, 255))
        
        padding = 15
        toast_w = text_surf.get_width() + padding * 2
        toast_h = text_surf.get_height() + padding * 2
        
        toast_surf = pygame.Surface((toast_w, toast_h), pygame.SRCALPHA)
        
        # Background with border (coerce color components to ints; handle alpha issues robustly)
        bg_rect = pygame.Rect(0, 0, toast_w, toast_h)
        try:
            bg_color = (int(50), int(60), int(80), int(alpha))
            pygame.draw.rect(toast_surf, bg_color, bg_rect, border_radius=8)
        except Exception:
            logger.exception("Failed drawing toast background with color %r; falling back to opaque color. alpha=%r", (50,60,80,alpha), alpha)
            pygame.draw.rect(toast_surf, (50, 60, 80), bg_rect, border_radius=8)

        # Colored border by type (coerce to ints and fallback)
        try:
            col = self.colors.get(self.toast_type, (200, 200, 200))
            border_color = (int(col[0]), int(col[1]), int(col[2]), int(alpha))
            pygame.draw.rect(toast_surf, border_color, bg_rect, 2, border_radius=8)
        except Exception:
            logger.exception("Failed drawing toast border with color %r; falling back to opaque.", (self.toast_type, col, alpha))
            pygame.draw.rect(toast_surf, (200, 200, 200), bg_rect, 2, border_radius=8)

        # Text with alpha
        text_with_alpha = text_surf.copy()
        try:
            text_with_alpha.set_alpha(int(alpha))
        except Exception:
            logger.exception("Failed to set text alpha: %r", alpha)
        toast_surf.blit(text_with_alpha, (padding, padding))
        
        # Render centered
        surface.blit(toast_surf, (center_x - toast_w // 2, y))


class ZeldaGUI:
    """
    Interactive GUI for Zelda dungeon validation.
    
    Features:
    - Resizable window (drag corners/edges)
    - Zoom in/out with +/- keys or mouse wheel
    - Pan with middle mouse or WASD when zoomed
    - Fullscreen toggle with F11
    - Smooth delta-time based animations
    - Heatmap overlay for A* search (toggle with H)
    """
    
    # Zoom levels available
    ZOOM_LEVELS = [16, 24, 32, 48, 64]
    DEFAULT_ZOOM_IDX = 2  # 32px default
    
    # Minimum window size
    MIN_WIDTH = 400
    MIN_HEIGHT = 300
    
    def __init__(self, maps: list = None, map_names: list = None):
        """
        Initialize GUI.
        
        Args:
            maps: List of semantic grids to visualize
            map_names: List of names for each map
        """
        if not PYGAME_AVAILABLE:
            raise ImportError("Pygame is required for GUI. Install with: pip install pygame")
        
        self.maps = maps if maps else [create_test_map()]
        self.map_names = map_names if map_names else [f"Map {i+1}" for i in range(len(self.maps))]
        self.current_map_idx = 0
        
        # Attempt to enable Windows DPI awareness *before* initializing Pygame so mouse coords match pixels
        try:
            import ctypes
            # Prefer the per-monitor v2 context if available (Windows 10+)
            try:
                DPI_AWARE_CONTEXT_PER_MONITOR_AWARE_V2 = -4
                ctypes.windll.user32.SetProcessDpiAwarenessContext(DPI_AWARE_CONTEXT_PER_MONITOR_AWARE_V2)
                logger.debug('SetProcessDpiAwarenessContext(PER_MONITOR_AWARE_V2) succeeded')
            except Exception:
                try:
                    # Try SetProcessDpiAwareness (Windows 8.1+ via shcore)
                    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
                    logger.debug('SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE) succeeded')
                except Exception:
                    try:
                        # Fallback to legacy API
                        ctypes.windll.user32.SetProcessDPIAware()
                        logger.debug('SetProcessDPIAware() succeeded')
                    except Exception:
                        logger.debug('Could not set process DPI awareness')
        except Exception:
            logger.debug('DPI awareness calls not supported on this platform')

        # Initialize Pygame
        try:
            pygame.init()
        except Exception as e:
            logger.exception("Failed to initialize Pygame")
            raise

        # Wrap pygame.mouse.set_cursor to be tolerant on platforms where system cursors are unsupported
        try:
            _orig_set_cursor = pygame.mouse.set_cursor
            def _wrapped_set_cursor(cursor):
                try:
                    _orig_set_cursor(cursor)
                except Exception:
                    logger.debug('set_cursor failed or unsupported in this environment', exc_info=True)
            pygame.mouse.set_cursor = _wrapped_set_cursor
        except Exception:
            # If we can't patch, just ignore - cursor changes may still work
            logger.debug('Could not wrap pygame.mouse.set_cursor; continuing')
        
        # Display settings
        self.zoom_idx = self.DEFAULT_ZOOM_IDX
        self.TILE_SIZE = self.ZOOM_LEVELS[self.zoom_idx]
        self.HUD_HEIGHT = 10  # Minimal bottom margin (status/message moved to sidebar)
        self.SIDEBAR_WIDTH = 220  # Wider for dungeon names
        
        # Get screen info for smart sizing
        display_info = pygame.display.Info()
        max_screen_w = display_info.current_w - 100
        max_screen_h = display_info.current_h - 100
        
        # Calculate initial window size (fit largest map)
        # Handle both raw grids and StitchedDungeon objects
        max_map_h = max(m.global_grid.shape[0] if hasattr(m, 'global_grid') else m.shape[0] for m in self.maps)
        max_map_w = max(m.global_grid.shape[1] if hasattr(m, 'global_grid') else m.shape[1] for m in self.maps)
        
        # Smart sizing: fit map with some padding, but don't exceed screen
        ideal_w = max_map_w * self.TILE_SIZE + self.SIDEBAR_WIDTH
        ideal_h = max_map_h * self.TILE_SIZE + self.HUD_HEIGHT
        
        self.screen_w = min(ideal_w, max_screen_w)
        self.screen_h = min(ideal_h, max_screen_h)
        
        # Ensure minimum size
        self.screen_w = max(self.screen_w, self.MIN_WIDTH)
        self.screen_h = max(self.screen_h, self.MIN_HEIGHT)
        
        # Create resizable window
        self.screen = pygame.display.set_mode(
            (self.screen_w, self.screen_h), 
            pygame.RESIZABLE
        )
        pygame.display.set_caption("ZAVE: Zelda AI Validation Environment")
        # Remember previous window size so we can restore it after exiting fullscreen
        self._prev_window_size = (self.screen_w, self.screen_h)

        # Ensure mouse events are not grabbed and cursor is visible on startup
        try:
            pygame.event.set_grab(False)
        except Exception:
            logger.debug('Could not clear event grab at startup')
        try:
            pygame.mouse.set_visible(True)
        except Exception:
            logger.debug('Could not ensure mouse cursor visible at startup')

        # Try to raise and focus the window on Windows so clicks are accepted
        try:
            if os.name == 'nt':
                try:
                    # Prefer ctypes to avoid adding pywin32 dependency
                    import ctypes
                    user32 = ctypes.windll.user32
                    hwnd = pygame.display.get_wm_info().get('window')
                    if hwnd:
                        logger.debug('Attempting to bring window to foreground (hwnd=%s)', hwnd)
                        SW_SHOW = 5
                        user32.ShowWindow(hwnd, SW_SHOW)
                        user32.SetForegroundWindow(hwnd)
                        pygame.event.pump()
                        pygame.mouse.set_visible(True)
                        pygame.event.set_grab(False)
                        logger.debug('Set focus to window via Win32 API')
                except Exception:
                    logger.debug('Windows focus helper failed', exc_info=True)
        except Exception:
            logger.debug('Focus bring-to-front helper encountered an error', exc_info=True)

        # Track last ungrab attempt to avoid spamming
        self._last_ungrab_attempt = 0.0
        
        # Display health & recovery settings
        self._display_check_interval = float(os.environ.get('KLTN_DISPLAY_CHECK_INTERVAL', '1.0'))
        self._display_check_last = time.time()
        self._display_recovery_attempts = 0
        self._display_recovery_attempts_limit = int(os.environ.get('KLTN_DISPLAY_RECOVER_LIMIT', '3'))

        # View offset for panning
        self.view_offset_x = 0
        self.view_offset_y = 0
        self.dragging = False
        self.drag_start = (0, 0)
        
        # Fullscreen state
        self.fullscreen = False
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 14, bold=True)
        self.big_font = pygame.font.SysFont('Arial', 20, bold=True)
        self.small_font = pygame.font.SysFont('Arial', 12)
        
        # Debug helpers for control panel visualisation and hit-padding
        # Can be enabled via env KLTN_DEBUG_CONTROL_PANEL=1 or toggled at runtime (F8)
        self.debug_control_panel = os.environ.get('KLTN_DEBUG_CONTROL_PANEL', '0') == '1'
        self.debug_panel_click_padding = int(os.environ.get('KLTN_DEBUG_PANEL_PADDING', '40')) if self.debug_control_panel else 0

        # Delta-time tracking for smooth animations
        self.last_frame_time = time.time()
        self.delta_time = 0.0
        # Display health check timing (throttled to avoid per-frame work)
        self._display_check_last = 0.0
        self._display_check_interval = float(os.environ.get('KLTN_DISPLAY_CHECK_INTERVAL', '1.0'))

        # Start a watchdog thread to detect UI freezes and dump stack/screenshot for debugging
        # Disabled by default - enable via KLTN_ENABLE_WATCHDOG=1 for troubleshooting
        try:
            import faulthandler
            self._watchdog_enabled = os.environ.get('KLTN_ENABLE_WATCHDOG', '0') == '1'
            self._watchdog_threshold = float(os.environ.get('KLTN_WATCHDOG_THRESHOLD', '1.25'))
            self._watchdog_last_dump = 0.0
            self._watchdog_dump_limit = int(os.environ.get('KLTN_WATCHDOG_DUMP_LIMIT', '3'))
            self._watchdog_dumps = 0
            # Path requested by watchdog for the main thread to save a screenshot (thread-safe)
            self._watchdog_request_screenshot = None
            if self._watchdog_enabled:
                def _watchdog_start():
                    try:
                        t = threading.Thread(target=self._watchdog_loop, daemon=True)
                        t.start()
                        self._watchdog_thread = t
                        logger.debug('Watchdog thread started (threshold=%s s)', self._watchdog_threshold)
                    except Exception:
                        logger.exception('Failed to start watchdog thread')
                _watchdog_start()
        except Exception:
            # If faulthandler or threading not available, skip watchdog
            self._watchdog_enabled = False

        # Track consecutive empty render frames and threshold beyond which we force a display reinit
        self._consecutive_empty_frames = 0
        try:
            self._empty_frame_recovery_threshold = int(os.environ.get('KLTN_EMPTY_FRAME_RECOVERY', '8'))
        except Exception:
            self._empty_frame_recovery_threshold = 8
        
        # New visualization system
        if VISUALIZATION_AVAILABLE:
            self.renderer = ZeldaRenderer(self.TILE_SIZE)
            self.effects = EffectManager()
            self.modern_hud = ModernHUD()
        else:
            self.renderer = None
            self.effects = None
            self.modern_hud = None

        # State for match/undo stack
        self.match_undo_stack = []

        # Heatmap state for A* visualization
        self.show_heatmap = False
        self.search_heatmap = {}  # position -> visit count
        
        # Load assets (fallback for when new system unavailable)
        self._load_assets()
        
        # Initialize environment
        self.env = None
        self.solver = None
        self.auto_path = []
        # ===== DEBUG TEST PATH =====
        # Set KLTN_DEBUG_TEST_PATH=1 to enable red debug path overlay
        if os.environ.get('KLTN_DEBUG_TEST_PATH') == '1':
            self._test_path = [(5, 5), (5, 6), (5, 7), (5, 8), (6, 8), (7, 8), (8, 8), (8, 9), (8, 10)]
            print(f"[DEBUG_INIT] _test_path ENABLED with {len(self._test_path)} points for visual testing")
        else:
            self._test_path = None
        # ===========================
        self.auto_step_idx = 0
        self.auto_mode = False
        self.auto_step_timer = 0.0  # Timer for controlling animation speed
        self.auto_step_interval = 0.15  # Base interval between steps (seconds)
        self.message = "Press SPACE to auto-solve, Arrow keys to move"
        self.message_time = time.time()  # Track when message was set
        self.message_duration = 3.0  # How long to show messages (seconds)
        self.error_message = None
        self.error_time = 0
        self.status_message = "Ready"
        self.show_help = False  # Toggle help overlay
        
        # State-space solver tracking (inventory/edge info)
        self.solver_result = None  # Stores keys_available, keys_used, edge_types etc.
        self.current_keys_held = 0  # Keys currently held during auto-solve
        self.current_keys_used = 0  # Keys used so far during auto-solve
        self.current_edge_types = []  # Edge types traversed so far
        self.door_unlock_times = {}  # Track when doors are unlocked for visual feedback
        
        # Path preview dialog (Feature 5)
        self.path_preview_dialog = None  # PathPreviewDialog instance when showing preview
        self.path_preview_mode = False  # True when showing path preview
        # If True, show a blocking modal dialog. If False, show non-modal overlay + sidebar summary.
        # Default: False to avoid blocking the map view (user prefers sidebar preview).
        self.preview_modal_enabled = False
        # When True the map will show the path overlay and a small sidebar preview box (non-modal)
        self.preview_overlay_visible = False
        # If True, automatically start animation after solver completes (skip preview confirmation)
        # Default: True for immediate animation on SPACE press
        # Set KLTN_AUTO_START_SOLVER=0 to require confirmation
        self.auto_start_solver = os.environ.get('KLTN_AUTO_START_SOLVER', '1') != '0'

        # Topology overlay and DOT export
        self.show_topology = False
        self.topology_export_path = None
        # Topology legend & semantics (for overlays/tooltips)
        self.show_topology_legend = False
        self.topology_semantics = {
            "nodes": {
                "e": ["room", "enemy"],
                "S": ["room", "switch"],
                "b": ["room", "boss"],
                "k": ["room", "key"],
                "K": ["room", "boss key"],
                "I": ["room", "key item"],
                "p": ["room", "puzzle"],
                "s": ["room", "start"],
                "t": ["room", "triforce"]
            },
            "edges": {
                "S": ["door", "switch locked"],
                "b": ["door", "bombable"],
                "k": ["door", "key locked"],
                "K": ["door", "boss key locked"],
                "I": ["door", "key item locked"],
                "l": ["door", "soft locked"],
                "s": ["visible", "impassable"]
            }
        }

        # Solver metrics and comparison results
        self.last_solver_metrics = None  # dict: {name,nodes,time_ms,path_len}
        self.solver_comparison_results = None  # list of dicts
        self.show_solver_comparison_overlay = False

        # === CRITICAL: Solver subprocess state (must be initialized!) ===
        # These variables track the background solver process and must exist before
        # any solver-related code runs (including _schedule_solver, _start_auto_solve)
        self.solver_running = False      # True while solver subprocess is active
        self.solver_proc = None          # multiprocessing.Process handle
        self.solver_done = True          # True when no solver pending (initially done)
        self.solver_outfile = None       # Temp file for solver pickle output
        self.solver_gridfile = None      # Temp file for grid numpy array
        self._pending_solver_trigger = False  # Flag to trigger solver on next frame (for algorithm changes)
        
        # Preview subprocess state (separate from main solver)
        self.preview_proc = None         # multiprocessing.Process handle for preview
        self.preview_outfile = None      # Temp file for preview pickle output
        self.preview_gridfile = None     # Temp file for preview grid
        self.preview_done = True         # True when no preview pending
        self.preview_result = None       # Cached result from preview worker
        self.preview_thread = None       # Threading fallback for preview

        # Presets
        self.presets = ['Debugging', 'Fast Approx', 'Optimal', 'Speedrun']
        self.current_preset_idx = 0

        # D* Lite integration
        self.dstar_solver = None
        self.dstar_active = False

        # Parallel search state
        self.parallel_search_thread = None
        self.parallel_search_done = False
        self.parallel_search_result = None
        
        # Smooth agent animation state
        self.agent_visual_pos = None  # Vector2 for smooth movement
        self.agent_target_pos = None  # Grid position target
        
        # === BLOCK PUSH ANIMATION SYSTEM ===
        # List of active block push animations
        # Each entry: {'from_pos': (r,c), 'to_pos': (r,c), 'start_time': float, 'duration': int}
        self.block_push_animations = []
        self.block_push_duration = 200  # milliseconds for block slide animation
        
        # Speed control system
        self.speed_levels = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
        self.speed_index = 2  # Start at 1.0x
        self.speed_multiplier = self.speed_levels[self.speed_index]
        
        # Game metrics
        self.step_count = 0  # Total steps taken
        self.item_pickup_times = {}  # Track when items were picked up for animation
        
        # Item totals for "X/Y collected" display
        self.total_keys = 0  # Total keys in dungeon
        self.total_bombs = 0  # Total bomb items
        self.total_boss_keys = 0  # Total boss keys
        self.keys_collected = 0  # Keys collected so far
        self.bombs_collected = 0  # Bombs collected
        self.boss_keys_collected = 0  # Boss keys collected
        
        # Toast notification system
        self.toast_notifications = []  # List of ToastNotification objects
        # Debug overlay & logging
        self.debug_overlay_enabled = False
        self.debug_click_log = []  # List of (pos, time, handled_widget_name)
        
        # Continuous movement (hold key to move)
        self.keys_held = {pygame.K_UP: False, pygame.K_DOWN: False, pygame.K_LEFT: False, pygame.K_RIGHT: False}
        self.move_timer = 0.0  # Timer for continuous movement
        self.move_delay = 0.15  # Delay between moves (seconds)
        
        # Minimap settings
        self.show_minimap = True  # Toggle minimap display
        self.minimap_size = 150  # Pixel size of minimap
        self.minimap_clickable = True  # Allow clicking minimap to navigate
        
        # === NEW: Item tracking for enhanced visualization ===
        self.collected_items = []  # List of (pos, item_type, timestamp)
        self.collected_positions = set()  # Set of (row, col) for O(1) lookup during rendering
        self.item_type_map = {}  # pos -> item_type (key, bomb, boss_key, triforce)
        self.used_items = []       # List of (pos, item_type, target_pos, timestamp)
        self.item_markers = {}     # Dict: position -> ItemMarkerEffect
        self.collection_effects = []  # Active collection effects
        self.usage_effects = []    # Active usage effects
        
        # === PATH ITEMS PREVIEW - Track items along auto-solve path ===
        self.path_items_summary = {}  # {item_type: count} - items along path
        self.path_item_positions = {}  # {item_type: [(row, col), ...]} - positions of items on path
        
        # === Toast Notification System ===
        self.toast_notifications = []  # List of ToastNotification objects
        
        # === NEW: GUI Control Panel ===
        self.control_panel_enabled = WIDGETS_AVAILABLE
        self.widget_manager = None
        self.control_panel_width = 360  # Logical expanded width (increased)
        self.control_panel_width_current = float(self.control_panel_width)  # Animated visual width
        self.control_panel_collapsed = False  # Track collapsed state
        self.control_panel_rect = None
        self.collapse_button_rect = None  # Rectangle for collapse button

        # Animation state for smooth collapse/expand
        self.control_panel_animating = False
        self.control_panel_anim_start = 0.0
        self.control_panel_anim_from = float(self.control_panel_width)
        self.control_panel_anim_to = float(self.control_panel_width)
        self.control_panel_anim_duration = 0.22
        self.control_panel_target_collapsed = False
        self.control_panel_x = None  # Custom X position (None = default right side)
        self.control_panel_y = None  # Custom Y position (None = default below minimap)
        self.dragging_panel = False
        self.drag_panel_offset = (0, 0)
        self.resizing_panel = False
        self.resize_edge = None  # 'left', 'right', 'top', 'bottom'

        # Control panel scroll state (for small screens)
        self.control_panel_scroll = 0
        self.control_panel_scroll_step = 20
        self.control_panel_can_scroll = False
        self.control_panel_scroll_max = 0
        self.control_panel_scroll_track_rect = None
        self.control_panel_scroll_thumb_rect = None
        self.control_panel_scroll_dragging = False
        self.control_panel_scroll_drag_offset = 0
        self.control_panel_content_height = 0
        # Debug toggle to draw layout markers and print metrics
        self.debug_control_panel = False

        # Inventory refresh flag (used when updates originate from worker threads)
        self.inventory_needs_refresh = False

        # Scroll inertia/momentum
        self.control_panel_scroll_velocity = 0.0  # pixels per second
        self.control_panel_scroll_damping = 6.0   # damping factor (higher = faster stop)
        # Ignore clicks during active scroll or shortly after to avoid accidental toggles
        self.control_panel_ignore_click_until = 0.0

        self.min_panel_width = 250
        self.max_panel_width = 500
        self.min_panel_height = 300
        
        # Feature toggles (controlled by checkboxes)
        self.feature_flags = {
            'solver_comparison': False,
            'parallel_search': False,
            'multi_goal': False,
            'ml_heuristic': False,
            'dstar_lite': False,
            'show_heatmap': False,
            'show_topology_legend': False,
            'show_minimap': True,
            'show_path': True,  # Show solver path overlay (always visible when path exists)
            'show_topology': False,  # Show topology graph overlay
            'diagonal_movement': False,
            'speedrun_mode': False,
            'dynamic_difficulty': False,
            'force_grid': False,
            'enable_prechecks': False,
            'auto_prune_on_precheck': False,
            'priority_tie_break': False,
            'priority_key_boost': False,
            'enable_ara': False,
            'use_jps': False,
            'show_jps_overlay': False,
            # MAP-Elites visualization toggle - when enabled the last MAP-Elites
            # heatmap generated by the evaluator will be rendered as an overlay
            'show_map_elites': False,
        }
        # Toggle to force using selected grid algorithm even when graph info exists
        self.force_grid_algorithm = False
        
        # Dropdown selections
        self.current_floor = 1
        self.zoom_level_idx = 3  # 100%
        self.difficulty_idx = 1  # Medium
        self.algorithm_idx = 0   # A*
        
        self._load_current_map()
        self._center_view()  # Center the map in view
        
        # Initialize control panel after map loaded
        if self.control_panel_enabled:
            self._init_control_panel()

        # Draw an initial frame to ensure window contents are painted promptly
        try:
            self._render()
            pygame.display.flip()
        except Exception:
            pass

    
    def _load_assets(self):
        """Load tile images - using colored squares for reliability."""
        self.images = {}
        
        # Color definitions for tile rendering
        color_map = {
            SEMANTIC_PALETTE['VOID']: (20, 20, 20),
            SEMANTIC_PALETTE['FLOOR']: (200, 180, 140),
            SEMANTIC_PALETTE['WALL']: (60, 60, 140),
            SEMANTIC_PALETTE['BLOCK']: (139, 90, 43),
            SEMANTIC_PALETTE['DOOR_OPEN']: (100, 80, 60),
            SEMANTIC_PALETTE['DOOR_LOCKED']: (139, 69, 19),
            SEMANTIC_PALETTE['DOOR_BOMB']: (80, 80, 80),
            SEMANTIC_PALETTE['DOOR_BOSS']: (180, 40, 40),
            SEMANTIC_PALETTE['DOOR_PUZZLE']: (140, 80, 180),
            SEMANTIC_PALETTE['DOOR_SOFT']: (100, 100, 60),
            SEMANTIC_PALETTE['ENEMY']: (200, 50, 50),
            SEMANTIC_PALETTE['START']: (80, 180, 80),
            SEMANTIC_PALETTE['TRIFORCE']: (255, 215, 0),
            SEMANTIC_PALETTE['BOSS']: (150, 20, 20),
            SEMANTIC_PALETTE['KEY_SMALL']: (255, 200, 50),
            SEMANTIC_PALETTE['KEY_BOSS']: (200, 100, 50),
            SEMANTIC_PALETTE['KEY_ITEM']: (100, 200, 255),
            SEMANTIC_PALETTE['ITEM_MINOR']: (200, 200, 200),
            SEMANTIC_PALETTE['ELEMENT']: (50, 80, 180),
            SEMANTIC_PALETTE['ELEMENT_FLOOR']: (80, 100, 160),
            SEMANTIC_PALETTE['STAIR']: (120, 100, 80),
            SEMANTIC_PALETTE['PUZZLE']: (180, 100, 180),
        }
        
        # Create colored square tiles for each semantic ID
        for tile_id, color in color_map.items():
            surf = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE))
            surf.fill(color)
            
            # Add visual indicators for special tiles
            if tile_id == SEMANTIC_PALETTE['DOOR_LOCKED']:
                # Draw keyhole indicator
                pygame.draw.circle(surf, (255, 200, 50), 
                                 (self.TILE_SIZE//2, self.TILE_SIZE//2 - 4), 4)
                pygame.draw.rect(surf, (255, 200, 50),
                               (self.TILE_SIZE//2 - 2, self.TILE_SIZE//2, 4, 8))
            elif tile_id == SEMANTIC_PALETTE['DOOR_BOMB']:
                # Draw crack pattern
                pygame.draw.line(surf, (40, 40, 40), (8, 8), (24, 24), 2)
                pygame.draw.line(surf, (40, 40, 40), (24, 8), (8, 24), 2)
            elif tile_id == SEMANTIC_PALETTE['KEY_SMALL']:
                # Draw key with glow effect for better visibility
                # Outer glow (yellow)
                pygame.draw.circle(surf, (255, 255, 100), (16, 10), 9)
                # Key head (circle)
                pygame.draw.circle(surf, (255, 215, 0), (16, 10), 6)
                # Key shaft
                pygame.draw.rect(surf, (255, 215, 0), (14, 10, 4, 16))
                # Key teeth
                pygame.draw.rect(surf, (255, 215, 0), (14, 22, 2, 3))
                pygame.draw.rect(surf, (255, 215, 0), (16, 24, 2, 2))
                # Inner shine
                pygame.draw.circle(surf, (255, 255, 200), (17, 9), 2)
                pygame.draw.circle(surf, (255, 255, 100, 150), (16, 10), 9)
                # Key head (circle)
                pygame.draw.circle(surf, (255, 215, 0), (16, 10), 6)
                # Key shaft
                pygame.draw.rect(surf, (255, 215, 0), (14, 10, 4, 16))
                # Key teeth
                pygame.draw.rect(surf, (255, 215, 0), (14, 22, 2, 3))
                pygame.draw.rect(surf, (255, 215, 0), (16, 24, 2, 2))
                # Inner shine
                pygame.draw.circle(surf, (255, 255, 200), (17, 9), 2)
            elif tile_id == SEMANTIC_PALETTE['TRIFORCE']:
                # Draw golden triangle
                points = [(16, 4), (4, 28), (28, 28)]
                pygame.draw.polygon(surf, (255, 255, 200), points)
                pygame.draw.polygon(surf, (200, 180, 0), points, 2)
            elif tile_id == SEMANTIC_PALETTE['ENEMY']:
                # Draw enemy indicator (red circle with eyes)
                pygame.draw.circle(surf, (255, 100, 100), (16, 16), 10)
                pygame.draw.circle(surf, (0, 0, 0), (12, 12), 3)
                pygame.draw.circle(surf, (0, 0, 0), (20, 12), 3)
            elif tile_id == SEMANTIC_PALETTE['START']:
                # Draw stair pattern
                pygame.draw.rect(surf, (60, 140, 60), (4, 4, 24, 24))
                for i in range(4):
                    pygame.draw.line(surf, (40, 100, 40), (8, 8+i*6), (24, 8+i*6), 2)
            elif tile_id == SEMANTIC_PALETTE['STAIR']:
                # Draw stair steps
                for i in range(4):
                    pygame.draw.rect(surf, (100, 80, 60), (4+i*4, 20-i*4, 20-i*4, 4))
            elif tile_id == SEMANTIC_PALETTE['WALL']:
                # Add brick pattern to walls
                pygame.draw.rect(surf, (50, 50, 120), (2, 2, 28, 28), 2)
                pygame.draw.line(surf, (70, 70, 150), (0, 16), (32, 16), 1)
                pygame.draw.line(surf, (70, 70, 150), (16, 0), (16, 32), 1)
            elif tile_id == SEMANTIC_PALETTE['BLOCK']:
                # Add block texture
                pygame.draw.rect(surf, (100, 60, 30), (2, 2, 28, 28), 2)
            elif tile_id == SEMANTIC_PALETTE['DOOR_OPEN']:
                # Draw open doorway
                pygame.draw.rect(surf, (40, 30, 20), (8, 0, 16, 32))
            elif tile_id == SEMANTIC_PALETTE['ELEMENT']:
                # Water/lava pattern
                for i in range(4):
                    pygame.draw.arc(surf, (80, 120, 200), (i*8, 8, 16, 16), 0, 3.14, 2)
                    pygame.draw.arc(surf, (80, 120, 200), (i*8, 16, 16, 16), 3.14, 6.28, 2)
            
            # Convert surface to display format with alpha for robust blitting
            try:
                self.images[tile_id] = surf.convert_alpha()
            except Exception:
                # Fallback to raw surface if convert_alpha fails
                self.images[tile_id] = surf
        
        # Create Link sprite
        self.link_img = self._create_link_sprite()

        # Create a small stair sprite (glowing marker) for visual emphasis
        try:
            # Force stair sprite to full tile size and use a bright, high-contrast overlay
            sprite_size = self.TILE_SIZE
            self.stair_sprite = pygame.Surface((sprite_size, sprite_size), pygame.SRCALPHA)
            self.stair_sprite.fill((0, 0, 0, 0))

            # Full-tile translucent fill (warm gold)
            pygame.draw.rect(self.stair_sprite, (255, 220, 100, 180), (0, 0, sprite_size, sprite_size))
            # Strong border for clear visibility
            pygame.draw.rect(self.stair_sprite, (255, 200, 50), (1, 1, sprite_size-2, sprite_size-2), 4)

            # Center triangle to indicate stair direction
            pts = [(sprite_size//2, sprite_size//6), (sprite_size//6, sprite_size*5//6), (sprite_size*5//6, sprite_size*5//6)]
            pygame.draw.polygon(self.stair_sprite, (255, 245, 180), pts)
            pygame.draw.polygon(self.stair_sprite, (255, 200, 50), pts, 2)

            # Slight inner highlight circle
            pygame.draw.circle(self.stair_sprite, (255, 255, 220, 64), (sprite_size//2, sprite_size//2), max(6, sprite_size//6))

            # Convert stair sprite for robust blitting
            try:
                self.stair_sprite = self.stair_sprite.convert_alpha()
            except Exception:
                pass
            self.stair_anim_phase = 0.0
        except Exception:
            self.stair_sprite = None
            self.stair_anim_phase = 0.0
    
    def _create_link_sprite(self) -> pygame.Surface:
        """Create a detailed Link sprite using pygame drawing."""
        link_img = pygame.Surface((self.TILE_SIZE - 4, self.TILE_SIZE - 4), pygame.SRCALPHA)
        
        # Transparent background
        link_img.fill((0, 0, 0, 0))
        
        # Body colors
        green = (0, 168, 0)
        skin = (252, 216, 168)
        brown = (136, 112, 0)
        dark_green = (0, 120, 0)
        
        # Draw Link's body (green tunic)
        pygame.draw.rect(link_img, green, (8, 12, 12, 12))  # Torso
        pygame.draw.rect(link_img, dark_green, (6, 18, 4, 8))  # Left arm
        pygame.draw.rect(link_img, dark_green, (18, 18, 4, 8))  # Right arm
        
        # Draw head
        pygame.draw.rect(link_img, skin, (8, 2, 12, 10))  # Face
        pygame.draw.circle(link_img, (0, 0, 0), (11, 6), 2)  # Left eye
        pygame.draw.circle(link_img, (0, 0, 0), (17, 6), 2)  # Right eye
        
        # Draw hair/cap (brown)
        pygame.draw.rect(link_img, brown, (6, 0, 16, 4))  # Hair top
        pygame.draw.rect(link_img, brown, (4, 2, 4, 6))  # Hair left
        pygame.draw.rect(link_img, brown, (20, 2, 4, 6))  # Hair right
        
        # Draw shield (brown rectangle on left side)
        pygame.draw.rect(link_img, brown, (2, 14, 6, 10))
        pygame.draw.rect(link_img, (200, 150, 50), (3, 15, 4, 8))  # Shield front
        
        # Draw sword (on right side)
        pygame.draw.rect(link_img, (180, 180, 180), (22, 12, 4, 14))  # Blade
        pygame.draw.rect(link_img, brown, (22, 10, 4, 4))  # Hilt
        
        try:
            return link_img.convert_alpha()
        except Exception:
            return link_img
    
    def _init_control_panel(self):
        """Initialize the GUI control panel with widgets."""
        if not WIDGETS_AVAILABLE:
            return
        
        self.widget_manager = WidgetManager()
        self._update_control_panel_positions()
    
    def _update_control_panel_positions(self):
        """Update control panel and widget positions (called on resize)."""
        if not WIDGETS_AVAILABLE or not self.widget_manager:
            return
        
        # Control panel position
        # Use custom position if set, otherwise default to right side
        collapsed_width = 40
        # Clamp panel width so it does not extend past the sidebar area and cover the map
        # Allow a wider panel but keep a safe cap based on screen and sidebar so it doesn't overlap map badly
        max_allowed_panel_width = max(collapsed_width, min(self.control_panel_width_current, self.max_panel_width, max(120, min(self.SIDEBAR_WIDTH * 2, max(120, self.screen_w - self.SIDEBAR_WIDTH - 40)))))
        panel_width = int(max_allowed_panel_width)
        # Debug: log when we had to clamp panel width to avoid map overlap
        try:
            original_width = int(max(collapsed_width, min(self.control_panel_width_current, self.max_panel_width)))
            if panel_width != original_width:
                logger.debug('Control panel width clamped from %d to %d due to SIDEBAR_WIDTH=%d', original_width, panel_width, self.SIDEBAR_WIDTH)
        except Exception:
            pass

        # Compute default dock position and allow custom drag; then clamp to sidebar area
        sidebar_x = self.screen_w - self.SIDEBAR_WIDTH
        if self.control_panel_x is not None and self.control_panel_y is not None:
            # Use custom dragged position
            panel_x = self.control_panel_x
            panel_y = self.control_panel_y
        else:
            # Default position (docked to right side, left of the sidebar)
            panel_x = sidebar_x - panel_width - 10
            panel_y = self.minimap_size + 20 if self.show_minimap else 10

        # Clamp panel inside usable area (keep it left of the sidebar)
        min_x = 10
        max_x = max(min_x, sidebar_x - panel_width - 10)
        panel_x = max(min_x, min(panel_x, max_x))

        # Calculate panel height (align with _render_control_panel logic)
        max_available_height = self.screen_h - panel_y - self.HUD_HEIGHT - 20
        min_panel_height = 120
        if max_available_height < min_panel_height:
            panel_height = min_panel_height
        else:
            panel_height = min(max_available_height, 700)

        self.control_panel_rect = pygame.Rect(
            panel_x, panel_y,
            panel_width,
            panel_height
        )

        # Collapse/Expand button (always visible) - top-right
        button_size = 32
        self.collapse_button_rect = pygame.Rect(
            panel_x + panel_width - 34, panel_y + 4,
            button_size, button_size
        )
        
        # Don't update widgets if effectively collapsed (small width)
        if panel_width <= collapsed_width + 8:
            return
        
        # Only rebuild widgets if none exist yet
        widgets_exist = hasattr(self, 'widget_manager') and self.widget_manager and len(self.widget_manager.widgets) > 0
        if widgets_exist:
            # Just update widget positions instead of rebuilding
            self._reposition_widgets(panel_x, panel_y)
            return
        
        # Clear existing widgets if reinitializing
        if hasattr(self, 'widget_manager') and self.widget_manager:
            # Preserve existing dropdown state (so rebuilding UI doesn't lose open/selected state)
            saved_dropdown_state = self.widget_manager.snapshot_dropdown_state()
            self.widget_manager.widgets.clear()
        
        # === CONSISTENT LAYOUT CONSTANTS ===
        margin_left = 12
        margin_top = 48  # Space for collapse button + "FEATURES" title
        checkbox_spacing = 26
        # Increase dropdown spacing to make room for labels
        dropdown_spacing = 44
        section_gap = 18
        
        # Start position (below "FEATURES" title)
        y_offset = panel_y + margin_top
        x_offset = panel_x + margin_left
        
        # === CHECKBOXES SECTION ===
        checkbox_labels = [
            ('solver_comparison', 'Solver Comparison'),
            ('parallel_search', 'Parallel Search'),
            ('multi_goal', 'Multi-Goal Pathfinding'),
            ('ml_heuristic', 'ML Heuristic'),
            ('dstar_lite', 'D* Lite Replanning'),
            ('show_heatmap', 'Show Heatmap Overlay'),
            ('show_path', 'Show Path Overlay'),  # Path visualization toggle
            ('show_map_elites', 'Show MAP-Elites Overlay'),
            ('show_topology', 'Show Topology Overlay'),
            ('show_topology_legend', 'Topology Legend (details)'),
            ('show_minimap', 'Show Minimap'),
            ('diagonal_movement', 'Diagonal Movement'),
            ('use_jps', 'Use Jump Point Search (JPS)'),
            ('show_jps_overlay', 'Show JPS Overlay'),
            ('speedrun_mode', 'Speedrun Mode'),
            ('dynamic_difficulty', 'Dynamic Difficulty'),
            ('force_grid', 'Force Grid Solver'),
            ('enable_prechecks', 'Enable Prechecks (fast checks before solve)'),
            ('auto_prune_on_precheck', 'Auto-Prune Dead-Ends on Precheck'),
            ('priority_tie_break', 'Priority: Tie-Break by Locks'),
            ('priority_key_boost', 'Priority: Key-Pickup Boost'),
            ('enable_ara', 'Enable ARA* (weighted A*)'),
            ('persist_dropdown_on_select', 'Keep dropdown open after select'),
        ]
        
        for flag_name, label in checkbox_labels:
            checkbox = CheckboxWidget(
                (x_offset, y_offset),
                label,
                checked=self.feature_flags.get(flag_name, False)
            )
            checkbox.flag_name = flag_name
            self.widget_manager.add_widget(checkbox)
            y_offset += checkbox_spacing
        
        # Section gap before dropdowns
        y_offset += section_gap
        
        # === DROPDOWNS SECTION ===
        # Floor selector
        floor_dropdown = DropdownWidget(
            (x_offset, y_offset),
            "Floor",
            ["Floor 1", "Floor 2", "Floor 3"],
            selected=self.current_floor - 1,
            keep_open_on_select=self.feature_flags.get('persist_dropdown_on_select', False)
        )
        floor_dropdown.control_name = 'floor'
        self.widget_manager.add_widget(floor_dropdown)
        y_offset += dropdown_spacing
        
        # Zoom level
        zoom_dropdown = DropdownWidget(
            (x_offset, y_offset),
            "Zoom",
            ["25%", "50%", "75%", "100%", "150%", "200%"],
            selected=self.zoom_level_idx,
            keep_open_on_select=self.feature_flags.get('persist_dropdown_on_select', False)
        )
        zoom_dropdown.control_name = 'zoom'
        self.widget_manager.add_widget(zoom_dropdown)
        y_offset += dropdown_spacing

        # ARA* weight (for weighted A* when enabled)
        ara_dropdown = DropdownWidget(
            (x_offset, y_offset),
            "ARA* weight",
            ["1.0", "1.25", "1.5", "2.0"],
            selected=0,
            keep_open_on_select=self.feature_flags.get('persist_dropdown_on_select', False)
        )
        ara_dropdown.control_name = 'ara_weight'
        self.widget_manager.add_widget(ara_dropdown)
        y_offset += dropdown_spacing
        
        # Difficulty
        difficulty_dropdown = DropdownWidget(
            (x_offset, y_offset),
            "Difficulty",
            ["Easy", "Medium", "Hard", "Expert"],
            selected=self.difficulty_idx,
            keep_open_on_select=self.feature_flags.get('persist_dropdown_on_select', False)
        )
        difficulty_dropdown.control_name = 'difficulty'
        self.widget_manager.add_widget(difficulty_dropdown)
        y_offset += dropdown_spacing
        
        # Presets
        presets_dropdown = DropdownWidget(
            (x_offset, y_offset),
            "Presets",
            self.presets,
            selected=self.current_preset_idx,
            keep_open_on_select=self.feature_flags.get('persist_dropdown_on_select', False)
        )
        presets_dropdown.control_name = 'presets'
        self.widget_manager.add_widget(presets_dropdown)
        y_offset += dropdown_spacing

        # Algorithm
        algorithm_dropdown = DropdownWidget(
            (x_offset, y_offset),
            "Solver",
            ["A*", "BFS", "Dijkstra", "Greedy", "D* Lite",
             "DFS/IDDFS", "Bidirectional A*",
             "CBS (Balanced)", "CBS (Explorer)", "CBS (Cautious)", 
             "CBS (Forgetful)", "CBS (Speedrunner)", "CBS (Greedy)"],
            selected=self.algorithm_idx,
            keep_open_on_select=self.feature_flags.get('persist_dropdown_on_select', False)
        )
        algorithm_dropdown.control_name = 'algorithm'
        self.widget_manager.add_widget(algorithm_dropdown)
        y_offset += dropdown_spacing

        # Match apply threshold (for tentative proposals)
        threshold_dropdown = DropdownWidget(
            (x_offset, y_offset),
            "Apply Threshold",
            ["0.70", "0.75", "0.80", "0.85", "0.90"],
            selected=3,
            keep_open_on_select=self.feature_flags.get('persist_dropdown_on_select', False)
        )
        threshold_dropdown.control_name = 'match_threshold'
        self.widget_manager.add_widget(threshold_dropdown)
        # Keep a float copy
        self.match_apply_threshold = float(threshold_dropdown.options[threshold_dropdown.selected])
        y_offset += dropdown_spacing
        
        # Section gap before buttons
        y_offset += section_gap

        # After adding dropdowns, restore any previous dropdown snapshot
        try:
            self.widget_manager.apply_dropdown_state(saved_dropdown_state)
        except Exception:
            pass
        
        # === BUTTONS SECTION ===
        button_width = 125
        button_height = 30
        buttons_per_row = 2
        button_h_spacing = 8
        button_v_spacing = 8
        
        # Primary action buttons
        primary_buttons = [
            ("Start Auto-Solve", self._start_auto_solve),
            ("Stop", self._stop_auto_solve),
            ("Generate Dungeon", self._generate_dungeon),
            ("AI Generate", self._generate_ai_dungeon),
            ("Reset", self._reset_map),
        ]
        
        # Secondary action buttons
        secondary_buttons = [
            ("Path Preview", self._show_path_preview),
            ("Clear Path", self._clear_path),
            ("Export Route", self._export_route),
            ("Load Route", self._load_route),
            ("Export Topology", self._export_topology),
            ("Compare Solvers", self._run_solver_comparison),
            ("Match Missing Nodes", self._match_missing_nodes),
            ("Apply Tentative Matches", self._apply_tentative_matches),
            ("Undo Last Match", self._undo_last_match),
            ("Undo Prune", self._undo_prune),
            ("Run MAP-Elites", self._start_map_elites),
        ]
        
        # Render primary buttons in 2x2 grid
        button_start_y = y_offset
        for i, (label, callback) in enumerate(primary_buttons):
            row = i // buttons_per_row
            col = i % buttons_per_row
            button_x = x_offset + col * (button_width + button_h_spacing)
            button_y = button_start_y + row * (button_height + button_v_spacing)
            
            button = ButtonWidget(
                (button_x, button_y),
                label,
                callback,
                width=button_width,
                height=button_height
            )
            self.widget_manager.add_widget(button)
        
        # Update y_offset for secondary buttons
        y_offset = button_start_y + (len(primary_buttons) // buttons_per_row) * (button_height + button_v_spacing) + 12
        
        # Render secondary buttons in 2x2 grid
        for i, (label, callback) in enumerate(secondary_buttons):
            row = i // buttons_per_row
            col = i % buttons_per_row
            button_x = x_offset + col * (button_width + button_h_spacing)
            button_y = y_offset + row * (button_height + button_v_spacing)
            
            button = ButtonWidget(
                (button_x, button_y),
                label,
                callback,
                width=button_width,
                height=button_height
            )
            self.widget_manager.add_widget(button)
        
        # === POST-BUILD: make sure widgets are positioned and compute content height ===
        # Reposition widgets once so final positions (matching _reposition_widgets) are used
        try:
            self._reposition_widgets(panel_x, panel_y)
        except Exception:
            pass
        # Compute top-most and bottom-most widget to determine content height
        max_widget_bottom = 0
        min_widget_top = 10**9
        for w in self.widget_manager.widgets:
            # Determine widget top (prefer full_rect if present)
            if hasattr(w, 'full_rect') and getattr(w, 'full_rect') is not None:
                top = int(getattr(w, 'full_rect').top)
            elif hasattr(w, 'rect') and getattr(w, 'rect') is not None:
                top = int(getattr(w, 'rect').top)
            else:
                top = panel_y

            # Determine widget bottom (consider dropdown_rect if present, as menu may expand)
            bottoms = []
            if hasattr(w, 'dropdown_rect') and getattr(w, 'dropdown_rect') is not None:
                try:
                    bottoms.append(int(getattr(w, 'dropdown_rect').bottom))
                except Exception:
                    pass
            if hasattr(w, 'full_rect') and getattr(w, 'full_rect') is not None:
                bottoms.append(int(getattr(w, 'full_rect').bottom))
            if hasattr(w, 'rect') and getattr(w, 'rect') is not None:
                bottoms.append(int(getattr(w, 'rect').bottom))

            bottom = max(bottoms) if bottoms else top

            min_widget_top = min(min_widget_top, top)
            max_widget_bottom = max(max_widget_bottom, bottom)

        # Desired content height (space required to show everything)
        extra_top = max(0, panel_y - min_widget_top)
        content_height = max_widget_bottom - panel_y + 12 + extra_top if max_widget_bottom > 0 else min_panel_height
        self.control_panel_content_height = content_height

        # If content exceeds available height, enable scrolling instead of enlarging
        if content_height > max_available_height:
            # Keep panel_height limited and enable scrolling
            panel_height = min(max_available_height, 700)
            self.control_panel_can_scroll = True
            self.control_panel_scroll = getattr(self, 'control_panel_scroll', 0)
            self.control_panel_scroll = max(0, min(self.control_panel_scroll, content_height - panel_height))
            # Update scroll max for external usage
            self.control_panel_scroll_max = max(0, content_height - panel_height)
        else:
            # Grow to fit content when possible
            panel_height = min(max_available_height, max(content_height, min_panel_height))
            self.control_panel_can_scroll = False
            self.control_panel_scroll = 0
            self.control_panel_scroll_max = 0

        # Update control panel rect height and collapse button Y
        self.control_panel_rect.height = panel_height
        self.collapse_button_rect.y = panel_y + 4
        # Reposition widgets in case layout changed
        self._reposition_widgets(panel_x, panel_y)
    
    def _reposition_widgets(self, panel_x: int, panel_y: int):
        """Reposition existing widgets when panel is dragged (without rebuilding)."""
        if not self.widget_manager or not self.widget_manager.widgets:
            return

        # Define consistent spacing
        margin_left = 12
        # Compute dynamic top margin from collapse button size to align items under header
        button_size = 28
        button_margin = 6
        margin_top = button_margin + max(button_size, self.font.get_height()) + 8
        item_spacing = 44  # Vertical spacing between checkboxes (increased for label breathing room)
        section_gap = 20  # Gap between sections

        # Starting Y position (below "FEATURES" title)
        current_y = panel_y + margin_top

        # Counters for different widget types
        checkbox_idx = 0
        dropdown_idx = 0
        button_idx = 0

        # Button layout configuration
        button_width = 125
        button_height = 30
        buttons_per_row = 2
        button_h_spacing = 8
        button_v_spacing = 8

        for widget in self.widget_manager.widgets:
            if isinstance(widget, CheckboxWidget):
                # Position checkboxes in a vertical list
                widget.pos = (panel_x + margin_left, current_y + checkbox_idx * item_spacing)
                checkbox_idx += 1

            elif isinstance(widget, DropdownWidget):
                # Position dropdowns after checkboxes with section gap
                if checkbox_idx > 0 and dropdown_idx == 0:
                    current_y += checkbox_idx * item_spacing + section_gap

                widget.pos = (panel_x + margin_left, current_y + dropdown_idx * item_spacing)
                dropdown_idx += 1

            elif isinstance(widget, ButtonWidget):
                # Position buttons in a grid after dropdowns
                if dropdown_idx > 0 and button_idx == 0:
                    current_y += dropdown_idx * item_spacing + section_gap

                row = button_idx // buttons_per_row
                col = button_idx % buttons_per_row
                widget.pos = (
                    panel_x + margin_left + col * (button_width + button_h_spacing),
                    current_y + row * (button_height + button_v_spacing)
                )
                button_idx += 1

    def _dump_control_panel_widget_state(self, mouse_pos: tuple):
        """Debug helper: log each widget rects and whether mouse/sc_pos hit them.

        This is defensive and avoids using any variables that may not be available in
        other layout helper contexts.
        """
        try:
            scroll_offset = getattr(self, 'control_panel_scroll', 0) if getattr(self, 'control_panel_can_scroll', False) else 0
            panel_rect = getattr(self, 'control_panel_rect', None)
            panel_top = panel_rect.y if panel_rect is not None else None
            header_height = self.font.get_height() + 12
            sc_pos = (mouse_pos[0], mouse_pos[1] + scroll_offset)
            if DEBUG_INPUT_ACTIVE:
                logger.info('DUMP_WIDGETS: mouse_pos=%s sc_pos=%s scroll=%s panel_top=%s header_h=%s', mouse_pos, sc_pos, scroll_offset, panel_top, header_height)
            else:
                logger.debug('DUMP_WIDGETS: mouse_pos=%s sc_pos=%s scroll=%s panel_top=%s header_h=%s', mouse_pos, sc_pos, scroll_offset, panel_top, header_height)
            for w in self.widget_manager.widgets:
                fr = getattr(w, 'full_rect', getattr(w, 'rect', None))
                rr = getattr(w, 'rect', None)
                dropdown_r = getattr(w, 'dropdown_rect', None)
                try:
                    fr_tuple = (fr.x, fr.y, fr.width, fr.height) if fr is not None else None
                except Exception:
                    fr_tuple = None
                try:
                    r_tuple = (rr.x, rr.y, rr.width, rr.height) if rr is not None else None
                except Exception:
                    r_tuple = None
                try:
                    dr_tuple = (dropdown_r.x, dropdown_r.y, dropdown_r.width, dropdown_r.height) if dropdown_r is not None else None
                except Exception:
                    dr_tuple = None
                contains_mouse = bool(fr and fr.collidepoint(mouse_pos))
                contains_sc = bool(fr and fr.collidepoint(sc_pos))
                rect_contains_mouse = bool(rr and rr.collidepoint(mouse_pos))
                rect_contains_sc = bool(rr and rr.collidepoint(sc_pos))
                label = getattr(w, 'label', getattr(w, 'control_name', None) or w.__class__.__name__)
                if DEBUG_INPUT_ACTIVE:
                    logger.info('WIDGET: name=%r full_rect=%s rect=%s dropdown=%s contains_mouse=%s contains_sc=%s rect_contains_mouse=%s rect_contains_sc=%s type=%s', label, fr_tuple, r_tuple, dr_tuple, contains_mouse, contains_sc, rect_contains_mouse, rect_contains_sc, w.__class__.__name__)
                else:
                    logger.debug('WIDGET: name=%r full_rect=%s rect=%s dropdown=%s contains_mouse=%s contains_sc=%s rect_contains_mouse=%s rect_contains_sc=%s type=%s', label, fr_tuple, r_tuple, dr_tuple, contains_mouse, contains_sc, rect_contains_mouse, rect_contains_sc, w.__class__.__name__)
        except Exception:
            logger.exception('Failed to dump control panel widget state')
        
    
    def _update_inventory_and_hud(self):
        """Reconcile counters and update the modern HUD (if present).

        This centralizes synchronization so any pickup/usage path calls the same routine.
        If called from a non-main thread, set a flag so the main thread performs the UI update
        (pygame surfaces & rendering should be touched only from the main thread).
        """
        # Instrumentation: log entry and key metrics for debugging real-time update issues
        try:
            thread_name = None
            import threading
            thread_name = threading.current_thread().name
        except Exception:
            thread_name = 'unknown'
        logger.debug("_update_inventory_and_hud: entry (thread=%s, inventory_needs_refresh=%s)",
                     thread_name, getattr(self, 'inventory_needs_refresh', False))

        try:
            # Always recompute counters (safe operation)
            before_keys = getattr(self, 'keys_collected', None)
            env_keys = getattr(getattr(self, 'env', None), 'state', None)
            env_keys_val = getattr(env_keys, 'keys', None) if env_keys is not None else None
            logger.debug("Counters before sync: env.keys=%s, keys_collected=%s", env_keys_val, before_keys)
            self._sync_inventory_counters()
            after_keys = getattr(self, 'keys_collected', None)
            logger.debug("Counters after sync: keys_collected=%s", after_keys)
        except Exception:
            logger.exception("_update_inventory_and_hud: failed while syncing counters")

        # Defensive consistency check
        try:
            if getattr(self.env, 'state', None):
                if getattr(self.env.state, 'keys', 0) < 0:
                    logger.warning('Inventory inconsistency: env.state.keys < 0')
        except Exception:
            pass

        # If we're not on the main thread, defer the HUD update to the main loop
        try:
            import threading
            if threading.current_thread() is not threading.main_thread():
                # Mark for refresh; main loop will call _update_inventory_and_hud() again
                self.inventory_needs_refresh = True
                logger.debug("_update_inventory_and_hud: deferred to main thread (set inventory_needs_refresh=True)")
                return
        except Exception:
            pass

        # Update modern HUD if available (main-thread only)
        try:
            # If we were previously deferring the update, clear the flag now since we're on the main thread
            if getattr(self, 'inventory_needs_refresh', False):
                logger.debug("_update_inventory_and_hud: clearing deferred flag (main thread)")
                try:
                    self.inventory_needs_refresh = False
                except Exception:
                    pass

            if getattr(self, 'modern_hud', None):
                # Record HUD state before update for diagnostics
                try:
                    hud_before = getattr(self.modern_hud, 'last', None) if hasattr(self.modern_hud, 'last') else None
                except Exception:
                    hud_before = None

                self.modern_hud.update_game_state(
                    keys=getattr(self.env.state, 'keys', 0),
                    bombs=getattr(self.env.state, 'bomb_count', 0),
                    has_boss_key=getattr(self.env.state, 'has_boss_key', False),
                    position=getattr(self.env.state, 'position', (0, 0)),
                    steps=getattr(self, 'step_count', 0),
                    message=getattr(self, 'message', '')
                )
                # Also mirror counts onto HUD fields if present (include usage counters)
                try:
                    if hasattr(self.modern_hud, 'keys_collected'):
                        self.modern_hud.keys_collected = getattr(self, 'keys_collected', 0)
                    if hasattr(self.modern_hud, 'bombs_collected'):
                        self.modern_hud.bombs_collected = getattr(self, 'bombs_collected', 0)
                    if hasattr(self.modern_hud, 'boss_keys_collected'):
                        self.modern_hud.boss_keys_collected = getattr(self, 'boss_keys_collected', 0)

                    # Usage counters (keys_used, bombs_used, boss_keys_used)
                    if hasattr(self.modern_hud, 'keys_used'):
                        self.modern_hud.keys_used = getattr(self, 'keys_used', 0)
                    if hasattr(self.modern_hud, 'bombs_used'):
                        self.modern_hud.bombs_used = getattr(self, 'bombs_used', 0)
                    if hasattr(self.modern_hud, 'boss_keys_used'):
                        self.modern_hud.boss_keys_used = getattr(self, 'boss_keys_used', 0)

                    # If a nested inventory panel exists, mirror values there too
                    if hasattr(self.modern_hud, 'inventory'):
                        try:
                            self.modern_hud.inventory.keys_collected = getattr(self, 'keys_collected', 0)
                            self.modern_hud.inventory.bombs_collected = getattr(self, 'bombs_collected', 0)
                            self.modern_hud.inventory.boss_keys_collected = getattr(self, 'boss_keys_collected', 0)
                            self.modern_hud.inventory.keys_used = getattr(self, 'keys_used', 0)
                            self.modern_hud.inventory.bombs_used = getattr(self, 'bombs_used', 0)
                            self.modern_hud.inventory.boss_keys_used = getattr(self, 'boss_keys_used', 0)
                        except Exception:
                            logger.exception("Failed setting nested inventory attributes")
                except Exception:
                    logger.exception("Failed setting HUD count attributes")

                try:
                    # Capture HUD state after update
                    hud_after = getattr(self.modern_hud, 'last', None) if hasattr(self.modern_hud, 'last') else None
                    logger.debug("HUD updated: before=%r after=%r env.keys=%s keys_collected=%s",
                                 hud_before, hud_after, getattr(self.env.state, 'keys', None), getattr(self, 'keys_collected', None))
                except Exception:
                    logger.exception("Failed to log HUD post-update state")
        except Exception as e:
            logger.warning(f"Failed to update modern HUD: {e}")

    def _remove_from_path_items(self, pos, item_type):
        """Remove a collected item from path_item_positions and update summary.
        
        Args:
            pos: (row, col) position of collected item
            item_type: 'keys', 'boss_keys', 'ladders', 'bombs', etc.
        """
        try:
            path_positions = getattr(self, 'path_item_positions', {})
            path_summary = getattr(self, 'path_items_summary', {})
            
            if item_type in path_positions and pos in path_positions[item_type]:
                path_positions[item_type].remove(pos)
                if item_type in path_summary and path_summary[item_type] > 0:
                    path_summary[item_type] -= 1
                logger.debug('Removed %s at %s from path items preview', item_type, pos)
        except Exception as e:
            logger.warning('Failed to remove %s at %s from path items: %s', item_type, pos, e)

    def _track_item_collection(self, old_state, new_state):
        """Detect when items are collected by comparing states."""
        # DEBUG: Log state comparison
        logger.debug("TRACK_COLLECTION: old_keys=%d, new_keys=%d, old_bomb=%s, new_bomb=%s, old_boss=%s, new_boss=%s",
                     old_state.keys, new_state.keys,
                     old_state.has_bomb, new_state.has_bomb,
                     getattr(old_state, 'has_boss_key', False), getattr(new_state, 'has_boss_key', False))
        
        # Check for key collection
        if new_state.keys > old_state.keys:
            keys_collected = new_state.keys - old_state.keys
            pos = new_state.position
            timestamp = time.time()
            
            logger.info("KEY_COLLECTED: pos=%s, count=%d, total_collected_now=%d", pos, keys_collected, self.keys_collected + keys_collected)
            
            self.collected_items.append((pos, 'key', timestamp))
            self.collected_positions.add(pos)  # Track position for rendering
            self.keys_collected += keys_collected
            # Mark pickup time for flashing UI
            self.item_pickup_times['key'] = timestamp
            # Ensure item type map contains this pos
            try:
                self.item_type_map[pos] = self.item_type_map.get(pos, 'key')
            except Exception:
                pass
            
            # Remove marker if exists
            if pos in self.item_markers and self.item_markers[pos].item_type == 'key':
                del self.item_markers[pos]
            
            # === REMOVE FROM PATH ITEMS PREVIEW ===
            self._remove_from_path_items(pos, 'keys')
            
            # Add collection effect
            if self.effects:
                effect = ItemCollectionEffect(pos, 'key', 'KEY', 
                                             f'Key collected at ({pos[0]}, {pos[1]})!')
                self.effects.add_effect(effect)
                self.collection_effects.append(effect)
                # Add pop effect for immediate visual feedback
                self.effects.add_effect(PopEffect(pos, (255, 215, 0)))  # Gold pop for key
            
            # Show toast notification
            self._show_toast(f"Key collected! Now have {self.keys_collected}/{self.total_keys}", 
                           duration=2.5, toast_type='success')
            try:
                logger.info(f"Detected key collection at {pos} (keys_collected={self.keys_collected}, env.keys={getattr(self.env.state,'keys',None)})")
            except Exception:
                pass
            try:
                self.last_pickup_msg = f"Picked up key at {pos}"
            except Exception:
                pass
            # Ensure HUD and counters are up to date
            try:
                self._update_inventory_and_hud()
            except Exception:
                pass
        
        # Check for bomb collection
        if new_state.has_bomb and not old_state.has_bomb:
            pos = new_state.position
            timestamp = time.time()
            
            self.collected_items.append((pos, 'bomb', timestamp))
            self.collected_positions.add(pos)  # Track position for rendering
            self.bombs_collected += 1
            self.item_pickup_times['bomb'] = timestamp
            try:
                self.item_type_map[pos] = self.item_type_map.get(pos, 'bomb')
            except Exception:
                pass
            
            # === REMOVE FROM PATH ITEMS PREVIEW ===
            self._remove_from_path_items(pos, 'bombs')
            
            # Remove marker
            if pos in self.item_markers and self.item_markers[pos].item_type == 'bomb':
                del self.item_markers[pos]
            
            # Add collection effect
            if self.effects:
                effect = ItemCollectionEffect(pos, 'bomb', 'BOMB',
                                             f'Bomb collected at ({pos[0]}, {pos[1]})!')
                self.effects.add_effect(effect)
                self.collection_effects.append(effect)
                # Add pop effect for immediate visual feedback
                self.effects.add_effect(PopEffect(pos, (255, 107, 53)))  # Orange pop for bomb
            
            # Show toast notification
            self._show_toast("Bomb acquired! Can now blow up weak walls", 
                           duration=3.0, toast_type='success')
            try:
                logger.info(f"Detected bomb collection at {pos} (bombs_collected={self.bombs_collected}, env.has_bomb={getattr(self.env.state,'has_bomb',None)})")
            except Exception:
                pass
            try:
                self._update_inventory_and_hud()
            except Exception:
                pass
        
        # Check for boss key collection
        if new_state.has_boss_key and not old_state.has_boss_key:
            pos = new_state.position
            timestamp = time.time()
            
            self.collected_items.append((pos, 'boss_key', timestamp))
            self.collected_positions.add(pos)  # Track position for rendering
            self.boss_keys_collected += 1
            
            # Remove marker
            if pos in self.item_markers and self.item_markers[pos].item_type == 'boss_key':
                del self.item_markers[pos]
            
            # === REMOVE FROM PATH ITEMS PREVIEW ===
            self._remove_from_path_items(pos, 'boss_keys')
            
            # Add collection effect
            if self.effects:
                effect = ItemCollectionEffect(pos, 'boss_key', 'BOSS KEY',
                                             f'Boss Key collected at ({pos[0]}, {pos[1]})!')
                self.effects.add_effect(effect)
                self.collection_effects.append(effect)
                # Add pop effect for immediate visual feedback
                self.effects.add_effect(PopEffect(pos, (176, 66, 255)))  # Purple pop for boss key
            
            # Show toast notification
            self._show_toast("Boss Key acquired! Can now face the boss", 
                           duration=3.0, toast_type='success')
            try:
                self._update_inventory_and_hud()
            except Exception:
                pass
    
    def _track_item_usage(self, old_state, new_state):
        """Detect when items are used (doors opened, walls bombed)."""
        # Defensive attribute initialization for lightweight runner tests
        if not hasattr(self, 'used_items'):
            self.used_items = []
        if not hasattr(self, 'usage_effects'):
            self.usage_effects = []

        # Check if key was used
        if new_state.keys < old_state.keys:
            keys_used = old_state.keys - new_state.keys
            pos = new_state.position
            timestamp = time.time()

            try:
                self.used_items.append((pos, 'key', pos, timestamp))
            except Exception:
                logger.exception("Failed appending to used_items")
            # Track used counter for visualization
            self.keys_used = getattr(self, 'keys_used', 0) + keys_used
            try:
                logger.info(f"Key used at {pos} (keys_used={self.keys_used}, env.keys={getattr(self.env.state,'keys',None)})")
                self.last_use_msg = f"Used key at {pos}"
            except Exception:
                pass
            try:
                self._update_inventory_and_hud()
            except Exception:
                pass

            # Add usage effect
            if self.effects:
                effect = ItemUsageEffect(old_state.position, pos, 'key')
                self.effects.add_effect(effect)
                self.usage_effects.append(effect)
            # Show toast for visibility in auto-solve
            self._show_toast(f"Key used! ({self.keys_used} used)", duration=1.8, toast_type='info')
        
        # Check if bomb was used
        if old_state.has_bomb and not new_state.has_bomb:
            pos = new_state.position
            timestamp = time.time()
            
            self.used_items.append((pos, 'bomb', pos, timestamp))
            # Track bombs used
            self.bombs_used = getattr(self, 'bombs_used', 0) + 1
            try:
                logger.info(f"Bomb used at {pos} (bombs_used={self.bombs_used}, env.has_bomb={getattr(self.env.state,'has_bomb',None)})")
            except Exception:
                pass
            
            # Add explosion effect
            if self.effects:
                effect = ItemUsageEffect(old_state.position, pos, 'bomb')
                self.effects.add_effect(effect)
                self.usage_effects.append(effect)
            self._show_toast(f"Bomb used! ({self.bombs_used} used)", duration=1.8, toast_type='info')
            try:
                self._update_inventory_and_hud()
            except Exception:
                pass
        
        # Check if boss key was used
        if old_state.has_boss_key and not new_state.has_boss_key:
            pos = new_state.position
            timestamp = time.time()
            
            self.used_items.append((pos, 'boss_key', pos, timestamp))
            # Track boss key used
            self.boss_keys_used = getattr(self, 'boss_keys_used', 0) + 1
            
            # Add usage effect
            if self.effects:
                effect = ItemUsageEffect(old_state.position, pos, 'boss_key')
                self.effects.add_effect(effect)
                self.usage_effects.append(effect)
            self._show_toast(f"Boss key used! ({self.boss_keys_used} used)", duration=2.5, toast_type='info')
            try:
                self._update_inventory_and_hud()
            except Exception:
                pass
    
    def _scan_and_mark_items(self):
        """Scan the map for all items and create markers.
        
        This populates item_type_map with all item positions so that
        _sync_inventory_counters() can correctly count collected items.
        """
        self.item_markers.clear()
        self.item_type_map.clear()
        self.total_keys = 0
        self.total_bombs = 0
        self.total_boss_keys = 0

        if not self.env:
            return
        
        h, w = self.env.height, self.env.width
        
        for r in range(h):
            for c in range(w):
                tile_id = self.env.grid[r, c]
                pos = (r, c)
                
                # Check if already collected
                if pos in getattr(self.env.state, 'collected_items', set()):
                    continue
                
                # Create marker for items and populate item_type_map
                if tile_id == SEMANTIC_PALETTE['KEY_SMALL']:
                    self.total_keys += 1
                    marker = ItemMarkerEffect(pos, 'key', 'K')
                    self.item_markers[pos] = marker
                    self.item_type_map[pos] = 'key'
                    if self.effects:
                        self.effects.add_effect(marker)
                
                elif tile_id == SEMANTIC_PALETTE.get('ITEM_BOMB', -1) or tile_id == SEMANTIC_PALETTE.get('ITEM_MINOR', -1):
                    self.total_bombs += 1
                    marker = ItemMarkerEffect(pos, 'bomb', 'B')
                    self.item_markers[pos] = marker
                    self.item_type_map[pos] = 'bomb'
                    if self.effects:
                        self.effects.add_effect(marker)
                
                elif tile_id == SEMANTIC_PALETTE['KEY_BOSS']:
                    self.total_boss_keys += 1
                    marker = ItemMarkerEffect(pos, 'boss_key', 'BK')
                    self.item_markers[pos] = marker
                    self.item_type_map[pos] = 'boss_key'
                    if self.effects:
                        self.effects.add_effect(marker)
                
                elif tile_id == SEMANTIC_PALETTE['TRIFORCE']:
                    marker = ItemMarkerEffect(pos, 'triforce', 'TRI')
                    self.item_markers[pos] = marker
                    self.item_type_map[pos] = 'triforce'
                    if self.effects:
                        self.effects.add_effect(marker)
        
        logger.debug("_scan_and_mark_items: found %d keys, %d bombs, %d boss_keys, item_type_map has %d entries",
                     self.total_keys, self.total_bombs, self.total_boss_keys, len(self.item_type_map))

    def _apply_pickup_at(self, pos: Tuple[int, int]) -> bool:
        """Apply pickup logic at a position for teleport landings or external mutations.

        This mutates self.env.state to include the collected item and updates
        visual markers/effects and pickup timers. Returns True if an item was
        collected at the position.
        """
        if not self.env:
            return False
        try:
            r, c = pos
            if r < 0 or c < 0 or r >= self.env.height or c >= self.env.width:
                return False
            if pos in getattr(self.env.state, 'collected_items', set()):
                return False
            tile_id = int(self.env.grid[r, c])
            if tile_id not in (SEMANTIC_PALETTE['KEY_SMALL'], SEMANTIC_PALETTE.get('ITEM_BOMB', -1), SEMANTIC_PALETTE['KEY_BOSS'], SEMANTIC_PALETTE['TRIFORCE']):
                return False

            # Mutate state to add collected position and update inventories
            try:
                collected = set(getattr(self.env.state, 'collected_items', set()) or set())
                collected = collected | {pos}
                self.env.state.collected_items = collected
            except Exception:
                pass

            if tile_id == SEMANTIC_PALETTE['KEY_SMALL']:
                try:
                    self.env.state.keys = getattr(self.env.state, 'keys', 0) + 1
                except Exception:
                    pass
                # CRITICAL: Modify grid to make item visually disappear
                try:
                    self.env.grid[r, c] = SEMANTIC_PALETTE['FLOOR']
                except Exception:
                    logger.warning("Failed to update grid for collected key")
                self.collected_items.append((pos, 'key', time.time()))
                self.collected_positions.add(pos)  # Track position for rendering
                self.item_pickup_times['key'] = time.time()
                self.keys_collected = getattr(self, 'keys_collected', 0) + 1
                # Remove marker if present
                if pos in self.item_markers and self.item_markers[pos].item_type == 'key':
                    del self.item_markers[pos]
                if self.effects:
                    eff = ItemCollectionEffect(pos, 'key', 'KEY', f'Key collected at ({pos[0]}, {pos[1]})!')
                    self.effects.add_effect(eff)
                    self.collection_effects.append(eff)
                self._show_toast(f"Key collected! Now have {self.keys_collected}/{self.total_keys}", duration=2.5, toast_type='success')
                self.item_type_map[pos] = 'key'
                return True

            if tile_id == SEMANTIC_PALETTE.get('ITEM_BOMB', -1):
                try:
                    self.env.state.bomb_count += 4  # Add 4 consumable bombs
                except Exception:
                    pass
                # CRITICAL: Modify grid to make item visually disappear
                try:
                    self.env.grid[r, c] = SEMANTIC_PALETTE['FLOOR']
                except Exception:
                    logger.warning("Failed to update grid for collected bomb")
                self.collected_items.append((pos, 'bomb', time.time()))
                self.collected_positions.add(pos)  # Track position for rendering
                self.item_pickup_times['bomb'] = time.time()
                self.bombs_collected = getattr(self, 'bombs_collected', 0) + 1
                if pos in self.item_markers and self.item_markers[pos].item_type == 'bomb':
                    del self.item_markers[pos]
                if self.effects:
                    eff = ItemCollectionEffect(pos, 'bomb', 'BOMB', f'Bomb collected at ({pos[0]}, {pos[1]})!')
                    self.effects.add_effect(eff)
                    self.collection_effects.append(eff)
                self._show_toast("Bomb acquired! Can now blow up weak walls", duration=3.0, toast_type='success')
                self.item_type_map[pos] = 'bomb'
                return True

            if tile_id == SEMANTIC_PALETTE['KEY_BOSS']:
                try:
                    self.env.state.has_boss_key = True
                except Exception:
                    pass
                # CRITICAL: Modify grid to make item visually disappear
                try:
                    self.env.grid[r, c] = SEMANTIC_PALETTE['FLOOR']
                except Exception:
                    logger.warning("Failed to update grid for collected boss key")
                self.collected_items.append((pos, 'boss_key', time.time()))
                self.collected_positions.add(pos)  # Track position for rendering
                self.item_pickup_times['boss_key'] = time.time()
                self.boss_keys_collected = getattr(self, 'boss_keys_collected', 0) + 1
                if pos in self.item_markers and self.item_markers[pos].item_type == 'boss_key':
                    del self.item_markers[pos]
                if self.effects:
                    eff = ItemCollectionEffect(pos, 'boss_key', 'BOSS KEY', f'Boss Key collected at ({pos[0]}, {pos[1]})!')
                    self.effects.add_effect(eff)
                    self.collection_effects.append(eff)
                self._show_toast("Boss Key acquired! Can now face the boss", duration=3.0, toast_type='success')
                self.item_type_map[pos] = 'boss_key'
                return True

            if tile_id == SEMANTIC_PALETTE['TRIFORCE']:
                # Mark triforce collected as special event but do not increment counts
                try:
                    self.env.state.collected_items = getattr(self.env.state, 'collected_items', set()) | {pos}
                except Exception:
                    pass
                # NOTE: Triforce usually stays visible (goal marker), but mark collected in state
                if self.effects:
                    eff = ItemCollectionEffect(pos, 'triforce', 'TRI', f'Triforce at ({pos[0]}, {pos[1]})!')
                    self.effects.add_effect(eff)
                self._show_toast("Triforce found!", duration=3.0, toast_type='success')
                self.item_type_map[pos] = 'triforce'
                return True

        except Exception as e:
            logger.warning(f"_apply_pickup_at failed: {e}")
        return False
    
    def _render_item_legend(self, surface):
        """Render legend showing item counts and path items preview."""
        if not self.env:
            return
        # Sync counters in case changes happened outside _track_* (defensive)
        self._sync_inventory_counters()

        # Get path items summary for preview
        path_summary = getattr(self, 'path_items_summary', {})
        has_path_items = any(v > 0 for v in path_summary.values()) if path_summary else False
        
        # Calculate legend height based on content
        base_lines = 3  # Keys, Bombs, Boss Keys
        path_item_lines = 1 if has_path_items else 0  # Path items preview line
        total_lines = base_lines + path_item_lines
        
        legend_x = 10
        legend_height = 20 + (total_lines * 20)
        legend_y = self.screen_h - legend_height - 40
        
        # Background
        legend_bg = pygame.Surface((350, legend_height), pygame.SRCALPHA)
        legend_bg.fill((30, 30, 40, 220))
        surface.blit(legend_bg, (legend_x, legend_y))
        
        # Border
        pygame.draw.rect(surface, (70, 70, 100), (legend_x, legend_y, 350, legend_height), 2)
        
        # Title
        title_text = "Inventory"
        title_surf = self.small_font.render(title_text, True, (100, 200, 255))
        surface.blit(title_surf, (legend_x + 10, legend_y + 4))
        
        # Inventory text
        y_offset = legend_y + 24
        
        legend_text = [
            f"[K] Keys: {self.env.state.keys} held | {self.keys_collected}/{self.total_keys} collected | {getattr(self,'keys_used',0)} used",
            f"[B] Bombs: {getattr(self.env.state, 'bomb_count', 0)} held | {self.bombs_collected}/{self.total_bombs} collected | {getattr(self,'bombs_used',0)} used",
            f"[BK] Boss Key: {'Yes' if getattr(self.env.state, 'has_boss_key', False) else 'No'} | {self.boss_keys_collected}/{self.total_boss_keys} collected"
        ]
        
        for text in legend_text:
            text_surf = self.small_font.render(text, True, (255, 255, 200))
            surface.blit(text_surf, (legend_x + 10, y_offset))
            y_offset += 18
        
        # Path items preview (items ahead on path)
        if has_path_items:
            y_offset += 4  # Small gap
            
            # Build path items preview text
            path_parts = []
            if path_summary.get('keys', 0) > 0:
                remaining = path_summary['keys']
                path_parts.append(f"{remaining}[K]")
            if path_summary.get('boss_keys', 0) > 0:
                remaining = path_summary['boss_keys']
                path_parts.append(f"{remaining}[BK]")
            if path_summary.get('ladders', 0) > 0:
                remaining = path_summary['ladders']
                path_parts.append(f"{remaining}[L]")
            if path_summary.get('doors_locked', 0) > 0:
                path_parts.append(f"{path_summary['doors_locked']}[D]")
            if path_summary.get('doors_bomb', 0) > 0:
                path_parts.append(f"{path_summary['doors_bomb']}[Bx]")
            if path_summary.get('doors_boss', 0) > 0:
                path_parts.append(f"{path_summary['doors_boss']}[BD]")
            
            if path_parts:
                path_text = f"▶ Path ahead: {' '.join(path_parts)}"
                path_surf = self.small_font.render(path_text, True, (100, 255, 150))
                surface.blit(path_surf, (legend_x + 10, y_offset))

    def _sync_inventory_counters(self):
        """Reconcile counters from collected_items and env.state to ensure UI accuracy.

        Uses multiple sources for robustness:
        1. self.collected_items list (primary - actively maintained by _track_item_collection)
        2. self.env.state.collected_items + item_type_map (backup)
        
        This ensures real-time updates work correctly during auto-solve.
        """
        # Initialize defaults
        self.keys_collected = getattr(self, 'keys_collected', 0)
        self.bombs_collected = getattr(self, 'bombs_collected', 0)
        self.boss_keys_collected = getattr(self, 'boss_keys_collected', 0)

        # PRIMARY: Count from self.collected_items list (actively maintained during auto-solve)
        kc_list = 0
        bc_list = 0
        bkc_list = 0
        if self.collected_items:
            for _pos, item_type, _ts in self.collected_items:
                if item_type == 'key':
                    kc_list += 1
                elif item_type == 'bomb':
                    bc_list += 1
                elif item_type == 'boss_key':
                    bkc_list += 1

        # BACKUP: Count from env.state.collected_items using item_type_map
        kc_map = 0
        bc_map = 0
        bkc_map = 0
        try:
            collected_set = set(getattr(self.env.state, 'collected_items', set()) or set())
            for pos in collected_set:
                it = self.item_type_map.get(pos)
                if it == 'key':
                    kc_map += 1
                elif it == 'bomb':
                    bc_map += 1
                elif it == 'boss_key':
                    bkc_map += 1
        except Exception:
            pass

        # Use the maximum of both sources (handles edge cases where one source might miss items)
        self.keys_collected = max(kc_list, kc_map)
        self.bombs_collected = max(bc_list, bc_map)
        self.boss_keys_collected = max(bkc_list, bkc_map)

        # Ensure used counters exist
        self.keys_used = getattr(self, 'keys_used', 0)
        self.bombs_used = getattr(self, 'bombs_used', 0)
        self.boss_keys_used = getattr(self, 'boss_keys_used', 0)

    def _scan_items_along_path(self, path=None):
        """Scan a path and identify all collectible items along it.
        
        This function analyzes the path positions and finds:
        - KEY_SMALL (30): Regular keys
        - KEY_BOSS (31): Boss keys  
        - KEY_ITEM (32): Ladder/special item
        - ITEM_MINOR (33): Bombs and other minor items
        - DOOR_LOCKED (11): Where keys will be used
        - DOOR_BOMB (12): Where bombs will be used
        - DOOR_BOSS (14): Where boss key will be used
        
        Results stored in:
        - self.path_items_summary: {item_type: count}
        - self.path_item_positions: {item_type: [(row, col), ...]}
        
        Returns:
            dict: Summary of items found along path
        """
        if path is None:
            path = getattr(self, 'auto_path', [])
        
        if not path or len(path) == 0:
            self.path_items_summary = {}
            self.path_item_positions = {}
            return {}
        
        # Tile IDs for items and doors
        KEY_SMALL = SEMANTIC_PALETTE.get('KEY_SMALL', 30)
        KEY_BOSS = SEMANTIC_PALETTE.get('KEY_BOSS', 31)
        KEY_ITEM = SEMANTIC_PALETTE.get('KEY_ITEM', 32)
        ITEM_MINOR = SEMANTIC_PALETTE.get('ITEM_MINOR', 33)
        DOOR_LOCKED = SEMANTIC_PALETTE.get('DOOR_LOCKED', 11)
        DOOR_BOMB = SEMANTIC_PALETTE.get('DOOR_BOMB', 12)
        DOOR_BOSS = SEMANTIC_PALETTE.get('DOOR_BOSS', 14)
        TRIFORCE = SEMANTIC_PALETTE.get('TRIFORCE', 22)
        
        # Initialize tracking
        summary = {
            'keys': 0,
            'boss_keys': 0,
            'ladders': 0,
            'bombs': 0,
            'doors_locked': 0,
            'doors_bomb': 0,
            'doors_boss': 0,
            'triforce': 0
        }
        positions = {
            'keys': [],
            'boss_keys': [],
            'ladders': [],
            'bombs': [],
            'doors_locked': [],
            'doors_bomb': [],
            'doors_boss': [],
            'triforce': []
        }
        
        # Get grid from environment
        if not self.env or not hasattr(self.env, 'grid'):
            self.path_items_summary = summary
            self.path_item_positions = positions
            return summary
        
        grid = self.env.grid
        already_collected = getattr(self, 'collected_positions', set())
        
        # Scan each position in path
        for pos in path:
            r, c = pos
            if r < 0 or r >= grid.shape[0] or c < 0 or c >= grid.shape[1]:
                continue
            
            tile_id = int(grid[r, c])
            
            # Skip already collected items
            if pos in already_collected:
                continue
            
            if tile_id == KEY_SMALL:
                summary['keys'] += 1
                positions['keys'].append(pos)
            elif tile_id == KEY_BOSS:
                summary['boss_keys'] += 1
                positions['boss_keys'].append(pos)
            elif tile_id == KEY_ITEM:
                summary['ladders'] += 1
                positions['ladders'].append(pos)
            elif tile_id == ITEM_MINOR:
                summary['bombs'] += 1
                positions['bombs'].append(pos)
            elif tile_id == DOOR_LOCKED:
                summary['doors_locked'] += 1
                positions['doors_locked'].append(pos)
            elif tile_id == DOOR_BOMB:
                summary['doors_bomb'] += 1
                positions['doors_bomb'].append(pos)
            elif tile_id == DOOR_BOSS:
                summary['doors_boss'] += 1
                positions['doors_boss'].append(pos)
            elif tile_id == TRIFORCE:
                summary['triforce'] += 1
                positions['triforce'].append(pos)
        
        self.path_items_summary = summary
        self.path_item_positions = positions
        
        # Log summary for debugging
        items_found = []
        if summary['keys'] > 0:
            items_found.append(f"{summary['keys']} key(s)")
        if summary['boss_keys'] > 0:
            items_found.append(f"{summary['boss_keys']} boss key(s)")
        if summary['ladders'] > 0:
            items_found.append(f"{summary['ladders']} ladder(s)")
        if summary['bombs'] > 0:
            items_found.append(f"{summary['bombs']} bomb(s)")
        if summary['doors_locked'] > 0:
            items_found.append(f"{summary['doors_locked']} locked door(s)")
        if summary['doors_bomb'] > 0:
            items_found.append(f"{summary['doors_bomb']} bomb door(s)")
        if summary['doors_boss'] > 0:
            items_found.append(f"{summary['doors_boss']} boss door(s)")
        if summary['triforce'] > 0:
            items_found.append(f"{summary['triforce']} triforce")
        
        if items_found:
            logger.info('PATH ITEMS: %s', ', '.join(items_found))
        else:
            logger.info('PATH ITEMS: No collectible items along path')
        
        return summary

    def _get_path_items_display_text(self):
        """Generate a display string summarizing items along the path.
        
        Returns:
            str: Human-readable summary like "Path: 3 keys, 2 doors, 1 boss key"
        """
        summary = getattr(self, 'path_items_summary', {})
        if not summary:
            return ""
        
        parts = []
        if summary.get('keys', 0) > 0:
            parts.append(f"{summary['keys']}[K]")
        if summary.get('boss_keys', 0) > 0:
            parts.append(f"{summary['boss_keys']}[BK]")
        if summary.get('ladders', 0) > 0:
            parts.append(f"{summary['ladders']}[L]")
        if summary.get('bombs', 0) > 0:
            parts.append(f"{summary['bombs']}[B]")
        if summary.get('doors_locked', 0) > 0:
            parts.append(f"{summary['doors_locked']}[D]")
        if summary.get('doors_bomb', 0) > 0:
            parts.append(f"{summary['doors_bomb']}[Bx]")
        if summary.get('doors_boss', 0) > 0:
            parts.append(f"{summary['doors_boss']}[BD]")
        if summary.get('triforce', 0) > 0:
            parts.append(f"{summary['triforce']}[T]")
        
        return " ".join(parts) if parts else ""
    
    def _render_error_banner(self, surface):
        """Render error message banner at top of screen with fade effect."""
        if hasattr(self, 'error_message') and self.error_message:
            elapsed = time.time() - self.error_time
            if elapsed < 5.0:  # Show for 5 seconds
                # Calculate fade (0.0 to 1.0)
                alpha = 1.0 if elapsed < 4.0 else (5.0 - elapsed)
                
                # Draw red banner at top
                banner_height = 45
                banner_rect = pygame.Rect(0, 0, self.screen_w, banner_height)
                banner_surface = pygame.Surface((self.screen_w, banner_height), pygame.SRCALPHA)
                banner_surface.fill((200, 0, 0, int(220 * alpha)))
                surface.blit(banner_surface, (0, 0))
                
                # Draw error icon and text
                font = pygame.font.Font(None, 28)
                text = f"[!] {self.error_message}"
                text_surf = font.render(text, True, (255, 255, 255))
                text_surf.set_alpha(int(255 * alpha))
                text_rect = text_surf.get_rect(center=(self.screen_w // 2, banner_height // 2))
                surface.blit(text_surf, text_rect)
                
                # Draw border
                pygame.draw.rect(surface, (150, 0, 0), banner_rect, 2)
            else:
                self.error_message = None
    
    def _render_solver_status_banner(self, surface):
        """Render solver status banner showing current algorithm and progress."""
        # Only show when solver is running
        if not getattr(self, 'solver_running', False):
            return
        
        algorithm_names = ["A*", "BFS", "Dijkstra", "Greedy", "D* Lite",
                         "DFS/IDDFS", "Bidirectional A*",
                         "CBS (Balanced)", "CBS (Explorer)", "CBS (Cautious)",
                         "CBS (Forgetful)", "CBS (Speedrunner)", "CBS (Greedy)"]
        
        # CRITICAL FIX: Use solver_algorithm_idx (saved when solver started) instead of algorithm_idx
        # This ensures banner shows the algorithm the CURRENTLY RUNNING solver is using,
        # even if user changes the dropdown while solver is running
        alg_idx = getattr(self, 'solver_algorithm_idx', getattr(self, 'algorithm_idx', 0))
        alg_name = algorithm_names[alg_idx] if alg_idx < len(algorithm_names) else f"Algorithm {alg_idx}"
        
        # DEBUG: Log banner rendering to diagnose CBS→A* display bug
        logger.debug('BANNER: Rendering solver banner with solver_algorithm_idx=%d, alg_name=%s', alg_idx, alg_name)
        
        # Draw yellow banner below error banner area
        banner_height = 50
        banner_y = 50  # Below error banner
        banner_rect = pygame.Rect(0, banner_y, self.screen_w, banner_height)
        banner_surface = pygame.Surface((self.screen_w, banner_height), pygame.SRCALPHA)
        
        # Animated pulse effect
        pulse = (math.sin(time.time() * 3) + 1) / 2  # 0 to 1
        alpha = int(180 + 75 * pulse)
        banner_surface.fill((200, 150, 0, alpha))
        surface.blit(banner_surface, (0, banner_y))
        
        # Draw status text
        font = pygame.font.Font(None, 32)
        text = f"🔍 Computing path with {alg_name}..."
        text_surf = font.render(text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=(self.screen_w // 2, banner_y + banner_height // 2))
        surface.blit(text_surf, text_rect)
        
        # Draw border
        pygame.draw.rect(surface, (255, 200, 0), banner_rect, 2)
    
    def _render_status_bar(self, surface):
        """Render status bar at bottom of screen."""
        bar_height = 30
        bar_y = self.screen_h - bar_height
        bar_rect = pygame.Rect(0, bar_y, self.screen_w, bar_height)
        
        # Background
        bar_surface = pygame.Surface((self.screen_w, bar_height), pygame.SRCALPHA)
        bar_surface.fill((40, 40, 50, 200))
        surface.blit(bar_surface, (0, bar_y))
        
        # Status text
        font = pygame.font.Font(None, 20)
        
        # Left: Status
        status_text = f"Status: {self.status_message}"
        status_surf = font.render(status_text, True, (180, 220, 255))
        surface.blit(status_surf, (10, bar_y + 7))
        
        # Center: Current action/message + ALGORITHM INDICATOR
        algorithm_names = ["A*", "BFS", "Dijkstra", "Greedy", "D* Lite",
                         "DFS/IDDFS", "Bidirectional A*",
                         "CBS (Balanced)", "CBS (Explorer)", "CBS (Cautious)",
                         "CBS (Forgetful)", "CBS (Speedrunner)", "CBS (Greedy)"]
        alg_idx = getattr(self, 'algorithm_idx', 0)
        alg_name = algorithm_names[alg_idx] if alg_idx < len(algorithm_names) else f"Alg{alg_idx}"
        
        # Algorithm indicator (ALWAYS SHOW THIS)
        alg_indicator = f"[{alg_name}]"
        alg_color = (255, 215, 0) if self.auto_mode else (150, 180, 255)  # Gold when running, blue when idle
        alg_surf = font.render(alg_indicator, True, alg_color)
        alg_rect = alg_surf.get_rect(center=(self.screen_w // 2, bar_y + bar_height // 2))
        surface.blit(alg_surf, alg_rect)
        
        if self.auto_mode:
            progress = f"Step {self.auto_step_idx + 1}/{len(self.auto_path)}"
            progress_surf = font.render(progress, True, (100, 255, 100))
            progress_rect = progress_surf.get_rect(centerx=alg_rect.right + 60, centery=bar_y + bar_height // 2)
            surface.blit(progress_surf, progress_rect)
            
            # Show path items if available (right of progress)
            items_text = self._get_path_items_display_text()
            if items_text:
                items_surf = font.render(items_text, True, (255, 220, 100))
                surface.blit(items_surf, (progress_rect.right + 20, bar_y + 7))
        elif getattr(self, 'auto_path', None) and len(self.auto_path) > 0:
            # Show path preview info when path exists but not animating
            items_text = self._get_path_items_display_text()
            if items_text:
                preview_text = f"Path ready ({len(self.auto_path)} steps) | {items_text}"
            else:
                preview_text = f"Path ready ({len(self.auto_path)} steps) - Press ENTER to start"
            preview_surf = font.render(preview_text, True, (100, 200, 255))
            preview_rect = preview_surf.get_rect(center=(self.screen_w // 2, bar_y + bar_height // 2))
            surface.blit(preview_surf, preview_rect)
        
        # Right: FPS
        fps_text = f"FPS: {int(self.clock.get_fps())}"
        fps_surf = font.render(fps_text, True, (255, 255, 180))
        fps_rect = fps_surf.get_rect(right=self.screen_w - 10, centery=bar_y + bar_height // 2)
        surface.blit(fps_surf, fps_rect)
        
        # Border
        pygame.draw.rect(surface, (60, 60, 80), bar_rect, 1)
    
    def _render_control_panel(self, surface):
        """Render the control panel with all GUI widgets and metrics."""
        if not self.control_panel_enabled or not self.widget_manager:
            return
        logger.debug(f"_render_control_panel: width_current={self.control_panel_width_current}, collapsed={self.control_panel_collapsed}, animating={getattr(self, 'control_panel_animating', False)}")

        # Collapsed threshold (always defined so later checks won't fail)
        collapsed_width = 40

        # Ensure control panel rects/positions are up-to-date (keeps animation offsets consistent)
        try:
            self._update_control_panel_positions()
        except Exception:
            pass

        # If update set a rect, use its coordinates for rendering; otherwise fallback to local computation
        if getattr(self, 'control_panel_rect', None):
            panel_rect = self.control_panel_rect
            panel_x, panel_y, panel_width, panel_height = panel_rect.x, panel_rect.y, panel_rect.width, panel_rect.height
        else:
            # Use animated width for current visual state
            panel_width = int(max(collapsed_width, min(self.control_panel_width_current, self.max_panel_width)))
            panel_width = max(collapsed_width, min(panel_width, self.max_panel_width))

            # Compute default dock position and allow custom drag
            sidebar_x = self.screen_w - self.SIDEBAR_WIDTH
            if self.control_panel_x is not None and self.control_panel_y is not None:
                panel_x = self.control_panel_x
                panel_y = self.control_panel_y
            else:
                panel_x = sidebar_x - panel_width - 10
                panel_y = 10

            min_x = 10
            max_x = max(min_x, sidebar_x - panel_width - 10)
            panel_x = max(min_x, min(panel_x, max_x))
            panel_y = max(10, min(panel_y, self.screen_h - 150))

            max_available_height = self.screen_h - panel_y - self.HUD_HEIGHT - 20
            min_panel_height = 120
            if max_available_height < min_panel_height:
                panel_height = min_panel_height
            else:
                panel_height = min(max_available_height, 700)

            if panel_width <= 0 or panel_height <= 0:
                return
            panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
            self.control_panel_rect = panel_rect


        # Create panel surface (use SRCALPHA so we can fade during animation)
        panel_surf = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        # Draw background onto panel surface
        bg_rect = pygame.Rect(0, 0, panel_width, panel_height)
        pygame.draw.rect(panel_surf, (40, 45, 60, 255), bg_rect, border_radius=8)
        pygame.draw.rect(panel_surf, (60, 60, 80, 255), bg_rect, 2, border_radius=8)

        # If animating, compute alpha for fade-in/out based on visual progress
        alpha = 255
        if getattr(self, 'control_panel_animating', False):
            a_from = self.control_panel_anim_from
            a_to = self.control_panel_anim_to
            denom = (a_to - a_from) if abs(a_to - a_from) > 1e-6 else 1.0
            progress = max(0.0, min(1.0, (self.control_panel_width_current - a_from) / denom))
            ease = progress * progress * (3 - 2 * progress)
            # If we're expanding, fade from 0->255; if collapsing, fade out
            alpha = int(255 * ease)

        # Blit background (we will blit widgets separately with alpha applied to whole panel)
        # Panel surface will be blitted at the end after widget rendering to ensure alpha applies to entire block
        

        # Update rect for mouse interaction (ensure using shifted coordinates from _update_control_panel_positions)
        # Note: _update_control_panel_positions has already computed self.control_panel_rect
        # and self.collapse_button_rect with any slide offset applied.
        # If they haven't been computed yet, fallback to unshifted rects.
        if not getattr(self, 'control_panel_rect', None):
            self.control_panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        if not getattr(self, 'collapse_button_rect', None):
            self.collapse_button_rect = pygame.Rect(
                panel_x + panel_width - 28 - 6,
                panel_y + 6,
                28,
                28
            )
        
        # Button will be drawn after panel surface is blitted so it remains on top
        # Track mouse hover for collapse button appearance
        mouse_pos = pygame.mouse.get_pos()
        is_hovering = self.collapse_button_rect.collidepoint(mouse_pos)
        
        # If visual width is small (collapsed visual), only show the collapse button (it will be drawn later)
        if panel_width <= collapsed_width + 8:
            # Draw only collapse button on main surface
            if is_hovering:
                button_color = (80, 120, 180)
                border_color = (150, 200, 255)
            else:
                button_color = (60, 80, 120)
                border_color = (100, 150, 200)
            pygame.draw.rect(surface, button_color, self.collapse_button_rect, border_radius=4)
            pygame.draw.rect(surface, border_color, self.collapse_button_rect, 2, border_radius=4)
            if getattr(self, 'control_panel_animating', False):
                button_text = ">" if self.control_panel_target_collapsed else "<"
            else:
                button_text = "<" if not self.control_panel_collapsed else ">"
            button_surf = self.font.render(button_text, True, (200, 220, 255))
            button_rect = button_surf.get_rect(center=self.collapse_button_rect.center)
            surface.blit(button_surf, button_rect)
            return
        
        # Update widget positions to match panel location (including any slide offset computed previously)
        # This sets widgets to their correct absolute screen coordinates for click detection
        self._reposition_widgets(panel_x, panel_y)
        
        # Backup widget positions AFTER repositioning so we restore to correct coords
        # (not stale positions from before the reposition)
        backups = [(w, getattr(w, 'pos', None)) for w in self.widget_manager.widgets]
        
        # Render widgets individually into the panel surface with per-widget alpha
        # We'll draw the header on top later so scrolling content never overlaps it

        content_alpha = 255
        if getattr(self, 'control_panel_animating', False):
            # Use the same progress/ease as alpha computed earlier
            a_from = self.control_panel_anim_from
            a_to = self.control_panel_anim_to
            denom = (a_to - a_from) if abs(a_to - a_from) > 1e-6 else 1.0
            progress = max(0.0, min(1.0, (self.control_panel_width_current - a_from) / denom))
            ease = progress * progress * (3 - 2 * progress)
            # Snap alpha near the end to avoid near-invisible final frames
            if ease >= 0.98:
                content_alpha = 255
            elif ease <= 0.02:
                content_alpha = 0
            else:
                content_alpha = int(255 * ease)
        
        for widget, orig_pos in backups:
            try:
                # Compute widget local position relative to panel and apply scroll offset.
                # Use the widget's full_rect (includes label area) so labels are not clipped.
                if orig_pos is None:
                    continue
                full_rect = getattr(widget, 'full_rect', widget.rect)
                dropdown_rect = getattr(widget, 'dropdown_rect', None)
                scroll_offset = getattr(self, 'control_panel_scroll', 0) if getattr(self, 'control_panel_can_scroll', False) else 0

                local_x = full_rect.x - panel_x
                local_y = full_rect.y - panel_y - scroll_offset

                # Create a temp surface sized to include the label area (full_rect) so title is visible.
                # When dropdown is open, we still render only the main control here; the expanded menu
                # will be drawn later on the main surface by `render_menu` to avoid panel clipping.
                target_w = min(panel_width - 24, max(full_rect.width, dropdown_rect.width if dropdown_rect is not None else 0, widget.rect.width))
                target_h = full_rect.height

                # Skip drawing if widget is outside the visible panel area (vertical clipping)
                # Ensure widgets do not draw into the header area at the top
                header_height = self.font.get_height() + 12
                if local_y + target_h < header_height or local_y > panel_height:
                    continue

                temp_surf = pygame.Surface((max(1, target_w), max(1, target_h)), pygame.SRCALPHA)

                # Temporarily set widget.pos so the main control sits below any label area
                # Compute label offset: full_rect.height includes label area above the control.
                label_offset = max(0, full_rect.height - widget.rect.height)
                widget.pos = (0, int(label_offset))
                widget.render(temp_surf)

                # Apply per-widget alpha
                if content_alpha < 255:
                    temp_surf.set_alpha(content_alpha)

                # Blit into panel surface
                blit_x = local_x
                blit_y = local_y
                panel_surf.blit(temp_surf, (blit_x, blit_y))

            except Exception as e:
                logger.warning(f"Per-widget render failed: {e}")

        # Draw scrollbar if content exceeds visible area
        if getattr(self, 'control_panel_can_scroll', False):
            track_w = 10
            track_margin = 8
            track_local_x = panel_width - track_w - track_margin
            track_local_y = 16
            track_h = panel_height - 32
            track_rect = pygame.Rect(track_local_x, track_local_y, track_w, track_h)
            # Visuals: subtle track + thumb
            pygame.draw.rect(panel_surf, (60, 65, 80, 200), track_rect, border_radius=6)
            # Compute visible content height excluding header
            header_height = self.font.get_height() + 12
            visible_h = max(10, panel_height - header_height - 16)
            content_h = getattr(self, 'control_panel_content_height', visible_h)
            max_scroll = max(content_h - visible_h, 0)
            thumb_h = max(int((visible_h / content_h) * track_h) if content_h > 0 else track_h, 20)
            if max_scroll > 0:
                thumb_y_local = track_local_y + int((getattr(self, 'control_panel_scroll', 0) / max_scroll) * (track_h - thumb_h))
            else:
                thumb_y_local = track_local_y
            thumb_rect = pygame.Rect(track_local_x + 1, thumb_y_local, track_w - 2, thumb_h)
            pygame.draw.rect(panel_surf, (100, 130, 180, 220), thumb_rect, border_radius=6)
            # Store global rects for mouse interaction handling
            self.control_panel_scroll_track_rect = pygame.Rect(panel_x + track_rect.x, panel_y + track_rect.y, track_rect.width, track_rect.height)
            self.control_panel_scroll_thumb_rect = pygame.Rect(panel_x + thumb_rect.x, panel_y + thumb_rect.y, thumb_rect.width, thumb_rect.height)
            self.control_panel_scroll_max = max_scroll
        else:
            # Clear any previous scroll rects
            self.control_panel_scroll_track_rect = None
            self.control_panel_scroll_thumb_rect = None
            self.control_panel_scroll_max = 0

        # Draw header (fixed) on top of scrolled content so it never overlaps
        header_height = self.font.get_height() + 12
        header_rect = pygame.Rect(0, 0, panel_width, header_height)
        header_surf = pygame.Surface((panel_width, header_height), pygame.SRCALPHA)
        # Slightly darker strip to separate header from scroll area
        pygame.draw.rect(header_surf, (35, 40, 55, 230), pygame.Rect(0, 0, panel_width, header_height), border_radius=0)
        # Draw FEATURES title centered vertically in the header
        features_title = self.font.render("FEATURES", True, (100, 200, 100))
        header_surf.blit(features_title, (12, (header_height - self.font.get_height()) // 2))
        panel_surf.blit(header_surf, (0, 0))

        # Apply overall panel alpha (for expand/collapse animation)
        if alpha < 255:
            panel_surf.set_alpha(alpha)
        # Blit panel surface onto main surface (panel bg + widgets)
        surface.blit(panel_surf, (panel_x, panel_y))
        # Restore panel surface alpha to full (avoid side-effects)
        if alpha < 255:
            panel_surf.set_alpha(255)

        # Restore widget global positions
        for widget, orig_pos in backups:
            try:
                if orig_pos is not None:
                    widget.pos = orig_pos
            except Exception:
                pass

        # Ensure collapse button colors are always assigned
        if is_hovering:
            button_color = (80, 120, 180)
            border_color = (150, 200, 255)
        else:
            button_color = (60, 80, 120)
            border_color = (100, 150, 200)
        pygame.draw.rect(surface, button_color, self.collapse_button_rect, border_radius=4)
        pygame.draw.rect(surface, border_color, self.collapse_button_rect, 2, border_radius=4)
        if getattr(self, 'control_panel_animating', False):
            button_text = ">" if self.control_panel_target_collapsed else "<"
        else:
            button_text = "<" if not self.control_panel_collapsed else ">"
        button_surf = self.font.render(button_text, True, (200, 220, 255))
        button_rect = button_surf.get_rect(center=self.collapse_button_rect.center)
        surface.blit(button_surf, button_rect)

        # Render dropdown menus on top of everything so they do not get clipped by the panel
        try:
            for widget in self.widget_manager.widgets:
                if getattr(widget, '__class__', None).__name__ == 'DropdownWidget' and getattr(widget, 'is_open', False):
                    try:
                        # Use the same content alpha used for widget fading during animation
                        scroll_offset = self.control_panel_scroll if getattr(self, 'control_panel_can_scroll', False) else 0
                        widget.render_menu(surface, alpha=content_alpha, scroll_offset=scroll_offset, panel_rect=self.control_panel_rect)
                    except Exception as e:
                        logger.warning(f"Dropdown menu render failed: {e}")
        except Exception:
            pass

        # Render tooltips if mouse over a widget
        self._render_tooltips(surface, mouse_pos)    
    def _render_tooltips(self, surface, mouse_pos):
        """Render tooltips for widgets under mouse cursor."""
        if not self.widget_manager:
            return
        
        # Tooltip definitions for each control
        tooltips = {
            'show_heatmap': 'Toggle A* search heatmap visualization',
            'show_minimap': 'Toggle minimap display in top-right corner',
            'show_path': 'Show/hide the solution path preview',
            'smooth_camera': 'Enable smooth camera transitions',
            'show_grid': 'Toggle grid overlay on map',
            'zoom': 'Adjust map zoom level (also use +/- keys)',
            'difficulty': 'Select map difficulty level',
            'algorithm': 'Choose pathfinding algorithm for auto-solve',
            'ml_heuristic': 'Use experimental ML-style heuristic (may be non-admissible)',
            'parallel_search': 'Run multiple strategies in parallel and pick fastest result',
            'solver_comparison': 'Run a comparison of available solvers and report metrics',
            'dstar_lite': 'Enable D* Lite incremental replanning (if implemented)',
            'show_topology': 'Draw room nodes & edges from topology graph on the map',
        }
        
        # Check which widget is under mouse
        for widget in self.widget_manager.widgets:
            # When panel is scrolled, translate mouse into scrolled coords for hit tests
            test_pos = mouse_pos
            header_height = self.font.get_height() + 12
            if getattr(self, 'control_panel_can_scroll', False) and getattr(self, 'control_panel_rect', None) and self.control_panel_rect.collidepoint(mouse_pos):
                # Only translate if mouse is below the fixed header area
                panel_top = self.control_panel_rect.y
                if mouse_pos[1] > panel_top + header_height:
                    test_pos = (mouse_pos[0], mouse_pos[1] + getattr(self, 'control_panel_scroll', 0))
            if hasattr(widget, 'rect') and widget.rect.collidepoint(test_pos) and widget.rect.y >= self.control_panel_rect.y + header_height:
                # Get tooltip text
                tooltip_text = None
                if hasattr(widget, 'flag_name') and widget.flag_name in tooltips:
                    tooltip_text = tooltips[widget.flag_name]
                elif hasattr(widget, 'control_name') and widget.control_name in tooltips:
                    tooltip_text = tooltips[widget.control_name]
                elif isinstance(widget, ButtonWidget) and hasattr(widget, 'label'):
                    # Button tooltips
                    button_tooltips = {
                        'Start Auto-Solve': 'Begin automatic pathfinding solution (SPACE)',
                        'Stop': 'Stop the current auto-solve operation',
                        'Generate Dungeon': 'Create a new random dungeon map (BSP)',
                        'AI Generate': 'Generate dungeon using trained latent diffusion AI model',
                        'Reset': 'Reset current map to initial state (R key)',
                        'Path Preview': 'Preview the complete solution path',
                        'Clear Path': 'Clear the displayed path overlay',
                        'Export Route': 'Save current path to file',
                        'Load Route': 'Load path from file',
                    }
                    tooltip_text = button_tooltips.get(widget.label)
                
                if tooltip_text:
                    self._draw_tooltip(surface, mouse_pos, tooltip_text)
                break
    
    def _draw_tooltip(self, surface, pos, text):
        """Draw a tooltip box at the specified position."""
        font = pygame.font.Font(None, 18)
        padding = 8
        
        # Render text
        text_surf = font.render(text, True, (255, 255, 255))
        text_rect = text_surf.get_rect()
        
        # Calculate tooltip position (offset from mouse)
        tooltip_x = pos[0] + 15
        tooltip_y = pos[1] + 15
        
        # Keep tooltip on screen
        if tooltip_x + text_rect.width + padding * 2 > self.screen_w:
            tooltip_x = pos[0] - text_rect.width - padding * 2 - 15
        if tooltip_y + text_rect.height + padding * 2 > self.screen_h:
            tooltip_y = pos[1] - text_rect.height - padding * 2 - 15
        
        # Draw background
        bg_rect = pygame.Rect(
            tooltip_x - padding,
            tooltip_y - padding,
            text_rect.width + padding * 2,
            text_rect.height + padding * 2
        )
        bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
        bg_surface.fill((50, 50, 60, 240))
        surface.blit(bg_surface, bg_rect.topleft)
        
        # Draw border
        pygame.draw.rect(surface, (100, 150, 200), bg_rect, 2)
        
        # Draw text
        surface.blit(text_surf, (tooltip_x, tooltip_y))
    
    def _handle_control_panel_click(self, pos, button, event_type='down'):
        """Handle mouse clicks on control panel widgets."""
        if not self.control_panel_enabled or not self.widget_manager:
            return False
        
        if event_type == 'down':
            # Compute padded hit rect if debug padding is active
            panel_rect = getattr(self, 'control_panel_rect', None)
            padding = getattr(self, 'debug_panel_click_padding', 0) if getattr(self, 'debug_control_panel', False) else 0
            panel_hit_rect = (pygame.Rect(panel_rect.x - padding, panel_rect.y, panel_rect.width + padding, panel_rect.height) if panel_rect and padding else panel_rect)

            # If the panel is currently being scrolled (dragging or inertia), don't forward clicks to widgets
            # Only swallow clicks that occur within the panel bounds; otherwise allow clicks to fall through
            if (getattr(self, 'control_panel_scroll_dragging', False) or time.time() < getattr(self, 'control_panel_ignore_click_until', 0.0)):
                if panel_hit_rect and panel_hit_rect.collidepoint(pos):
                    logger.debug('Ignored click on control panel due to active scroll/ignore window (dragging=%s ignore_until=%s) and pos inside panel_hit_rect', getattr(self,'control_panel_scroll_dragging',False), getattr(self,'control_panel_ignore_click_until',0.0))
                    return True
            # Translate mouse pos into scrolled widget coordinates when panel is scrolled
            if getattr(self, 'control_panel_can_scroll', False) and panel_hit_rect and panel_hit_rect.collidepoint(pos):
                header_height = 45
                panel_top = panel_rect.y if panel_rect is not None else 0
                # Only translate clicks that occur below the fixed header area
                if pos[1] > panel_top + header_height:
                    sc_pos = (pos[0], pos[1] + getattr(self, 'control_panel_scroll', 0))
                else:
                    sc_pos = pos
            else:
                sc_pos = pos

            # If click is outside the panel bounds, avoid doing heavy layout checks and widget dumps.
            # However, if any dropdowns are open we should still dispatch to the widget manager so they
            # can be closed by an outside click. Otherwise, return False so clicks fall through quickly.
            try:
                if not (panel_hit_rect and panel_hit_rect.collidepoint(pos)):
                    # Close any open dropdowns if present by dispatching the click; this will also
                    # return True if a dropdown consumed the event (closing or interacting).
                    if any(isinstance(w, DropdownWidget) and getattr(w, 'is_open', False) for w in self.widget_manager.widgets):
                        return self.widget_manager.handle_mouse_down(pos, button)
                    return False
            except Exception:
                # If something unexpected happens, do not block clicks; let caller handle as usual
                logger.exception('Error while checking outside-panel click handling')


            # Debug: log transformation and scroll state
            logger.debug('Control panel click: pos=%s sc_pos=%s scroll=%s header_h=%s', pos, sc_pos, getattr(self, 'control_panel_scroll', 0), 45)

            # Check whether any widget's rect/full_rect currently contains the sc_pos. If none do,
            # we may have stale widget positions (layout drift); attempt a reposition and re-evaluate once.
            try:
                any_contains = False
                for w in self.widget_manager.widgets:
                    fr = getattr(w, 'full_rect', getattr(w, 'rect', None))
                    rr = getattr(w, 'rect', None)
                    if (fr and fr.collidepoint(sc_pos)) or (rr and rr.collidepoint(sc_pos)):
                        any_contains = True
                        break
                if not any_contains:
                    if DEBUG_INPUT_ACTIVE:
                        logger.info('No widget claims sc_pos=%s: attempting _reposition_widgets to refresh layout', sc_pos)
                    else:
                        logger.debug('No widget claims sc_pos=%s: attempting _reposition_widgets to refresh layout', sc_pos)
                    try:
                        # Use current panel coords to reposition
                        panel_rect = getattr(self, 'control_panel_rect', None)
                        if panel_rect:
                            self._reposition_widgets(panel_rect.x, panel_rect.y)
                            # Recompute any_contains after reposition
                            for w in self.widget_manager.widgets:
                                fr = getattr(w, 'full_rect', getattr(w, 'rect', None))
                                rr = getattr(w, 'rect', None)
                                if (fr and fr.collidepoint(sc_pos)) or (rr and rr.collidepoint(sc_pos)):
                                    any_contains = True
                                    break
                    except Exception:
                        logger.exception('Reposition attempt failed')
            except Exception:
                logger.exception('Failure while checking widget rects before dispatch')

            handled = self.widget_manager.handle_mouse_down(sc_pos, button)

            logger.debug('Control panel click handled=%s at pos=%s sc_pos=%s any_contains=%s', handled, pos, sc_pos, any_contains)
            if not handled:
                # Dump per-widget hit info to help debug click routing (only when debug input diagnostics enabled)
                if DEBUG_INPUT_ACTIVE:
                    try:
                        self._dump_control_panel_widget_state(pos)
                    except Exception:
                        logger.exception('Failed to dump widget hit tests after unhandled click')

                # If unhandled and the panel is scrollable, attempt to auto-scroll the panel to bring
                # the nearest off-screen widget into view and retry the click once.
                try:
                    panel_rect = getattr(self, 'control_panel_rect', None)
                    if panel_rect and self.widget_manager and getattr(self, 'control_panel_can_scroll', False):
                        header_height = 45
                        click_y_local = sc_pos[1] - panel_rect.y - header_height
                        nearest = None
                        nearest_dist = None
                        for w in self.widget_manager.widgets:
                            fr = getattr(w, 'full_rect', getattr(w, 'rect', None))
                            if fr is None:
                                continue
                            widget_top_rel = fr.y - panel_rect.y
                            widget_bottom_rel = fr.bottom - panel_rect.y
                            if widget_top_rel < header_height or widget_bottom_rel > panel_rect.height:
                                center = (widget_top_rel + widget_bottom_rel) / 2.0
                                dist = abs(center - click_y_local)
                                if nearest is None or dist < nearest_dist:
                                    nearest = w
                                    nearest_dist = dist
                        if nearest is not None:
                            prev_scroll = getattr(self, 'control_panel_scroll', 0)
                            # Use full_rect if available, otherwise fall back to rect
                            widget_rect = getattr(nearest, 'full_rect', nearest.rect)
                            target_scroll = max(0, min(getattr(self, 'control_panel_scroll_max', 0), widget_rect.y - panel_rect.y - header_height))
                            if abs(target_scroll - prev_scroll) > 1:
                                self.control_panel_scroll = target_scroll
                                self.control_panel_ignore_click_until = time.time() + 0.12
                                logger.info('Control panel auto-scrolled to reveal widget (scroll=%s)', self.control_panel_scroll)
                                # Re-dispatch the original click against the widgets using the new scroll
                                new_sc_pos = (pos[0], pos[1] + self.control_panel_scroll)
                                try:
                                    handled = self.widget_manager.handle_mouse_down(new_sc_pos, button)
                                    logger.debug('After auto-scroll, re-dispatch handled=%s', handled)
                                except Exception:
                                    logger.exception('Re-dispatch after auto-scroll failed')
                except Exception:
                    logger.exception('Auto-scroll retry failed')

            if handled:
                logger.debug('Control panel click handled by widget manager at pos=%r (button=%r)', pos, button)
                # Update feature flags from checkboxes
                for widget in self.widget_manager.widgets:
                    if isinstance(widget, CheckboxWidget) and hasattr(widget, 'flag_name'):
                        old_value = self.feature_flags.get(widget.flag_name, False)
                        self.feature_flags[widget.flag_name] = widget.checked
                        logger.info('Feature flag set: %s=%s', widget.flag_name, widget.checked)
                        
                        # Apply flags immediately with visual feedback
                        if widget.flag_name == 'show_heatmap' and old_value != widget.checked:
                            self.show_heatmap = widget.checked
                            if self.renderer:
                                self.renderer.show_heatmap = widget.checked
                            self._set_message(f"Heatmap: {'ON' if widget.checked else 'OFF'}")
                        elif widget.flag_name == 'show_path' and old_value != widget.checked:
                            # Path overlay toggle
                            self._set_message(f"Path overlay: {'ON' if widget.checked else 'OFF'}", 1.5)
                        elif widget.flag_name == 'show_minimap':
                            self.show_minimap = widget.checked
                            self._set_message(f"Minimap: {'ON' if widget.checked else 'OFF'}")
                        elif widget.flag_name == 'show_topology' and old_value != widget.checked:
                            # Enable/disable topology overlay
                            self.show_topology = widget.checked
                            if widget.checked:
                                current = self.maps[self.current_map_idx]
                                if not hasattr(current, 'graph') or not current.graph:
                                    self._set_message('Topology not available for this map', 3.0)
                                else:
                                    self._set_message('Topology overlay: ON', 2.0)
                            else:
                                self._set_message('Topology overlay: OFF', 1.2)
                        elif widget.flag_name == 'show_topology_legend' and old_value != widget.checked:
                            self.show_topology_legend = widget.checked
                            self._set_message(f"Topology legend: {'ON' if widget.checked else 'OFF'}", 1.8)
                        elif hasattr(widget, 'control_name') and widget.control_name == 'zoom':
                            old_zoom_idx = self.zoom_level_idx
                            self.zoom_level_idx = widget.selected
                            if old_zoom_idx != self.zoom_level_idx:
                                # Map zoom_level_idx to actual zoom: [25%, 50%, 75%, 100%, 150%, 200%]
                                zoom_map = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 4}  # Map to ZOOM_LEVELS indices
                                if self.zoom_level_idx < len(zoom_map):
                                    new_zoom_idx = zoom_map.get(self.zoom_level_idx, 2)
                                    if new_zoom_idx != self.zoom_idx:
                                        self.zoom_idx = new_zoom_idx
                                        self.TILE_SIZE = self.ZOOM_LEVELS[self.zoom_idx]
                                        self._load_assets()
                                        self._center_view()
                                        zoom_labels = ["25%", "50%", "75%", "100%", "150%", "200%"]
                                        self.message = f"Zoom: {zoom_labels[self.zoom_level_idx]}"
                        elif hasattr(widget, 'control_name') and widget.control_name == 'difficulty':
                            self.difficulty_idx = widget.selected
                            difficulty_names = ["Easy", "Medium", "Hard", "Expert"]
                            self.message = f"Difficulty: {difficulty_names[self.difficulty_idx]}"
                        elif hasattr(widget, 'control_name') and widget.control_name == 'algorithm':
                            old_algorithm_idx = self.algorithm_idx
                            self.algorithm_idx = widget.selected
                            if old_algorithm_idx != self.algorithm_idx:
                                algorithm_names = ["A*", "BFS", "Dijkstra", "Greedy", "D* Lite",
                                                 "DFS/IDDFS", "Bidirectional A*",
                                                 "CBS (Balanced)", "CBS (Explorer)", "CBS (Cautious)",
                                                 "CBS (Forgetful)", "CBS (Speedrunner)", "CBS (Greedy)"]
                                self.message = f"Solver: {algorithm_names[self.algorithm_idx]}"
                                # DEBUG: Log dropdown selection to diagnose CBS→A* display bug
                                logger.info('DROPDOWN: Algorithm changed from %d(%s) to %d(%s)',
                                           old_algorithm_idx, algorithm_names[old_algorithm_idx] if old_algorithm_idx < len(algorithm_names) else 'Unknown',
                                           self.algorithm_idx, algorithm_names[self.algorithm_idx])
                                
                                # CRITICAL FIX: If solver is currently running with old algorithm, stop it
                                # This prevents the banner from showing the wrong algorithm name
                                if getattr(self, 'solver_running', False):
                                    logger.info('DROPDOWN: Stopping solver running with old algorithm %s', 
                                               algorithm_names[old_algorithm_idx] if old_algorithm_idx < len(algorithm_names) else 'Unknown')
                                    # Terminate the solver process if it exists
                                    if hasattr(self, 'solver_proc') and self.solver_proc:
                                        try:
                                            self.solver_proc.terminate()
                                            logger.info('DROPDOWN: Terminated solver process')
                                        except Exception as e:
                                            logger.warning('DROPDOWN: Failed to terminate solver process: %s', e)
                                    # Clear preview workers
                                    if hasattr(self, 'preview_thread') and self.preview_thread:
                                        self.preview_thread = None
                                    if hasattr(self, 'preview_proc') and self.preview_proc:
                                        try:
                                            self.preview_proc.terminate()
                                        except Exception:
                                            pass
                                        self.preview_proc = None
                                    # Clear all solver state using centralized helper
                                    self._clear_solver_state(reason=f"algorithm changed to {algorithm_names[self.algorithm_idx]}")
                                    self._set_message(f"🔄 Switched to {algorithm_names[self.algorithm_idx]} (press SPACE to solve)", 2.5)
                                    # Don't auto-restart solver here - let user press SPACE when ready
                                    continue  # Skip the existing path recompute logic below
                                
                                # CRITICAL FIX: If a path was already computed, automatically recompute with new algorithm
                                # This ensures changing the solver dropdown immediately shows the new algorithm's path
                                had_existing_path = bool(self.auto_path)
                                if had_existing_path:
                                    logger.info('═══════════════════════════════════════════════════')
                                    logger.info('🔄 ALGORITHM CHANGED: %s → %s',
                                               algorithm_names[old_algorithm_idx], algorithm_names[self.algorithm_idx])
                                    logger.info('   Triggering automatic resolve to show new path')
                                    logger.info('═══════════════════════════════════════════════════')
                                    # Clear old path first
                                    self.auto_path = []
                                    self.auto_mode = False
                                    # Trigger new solve with updated algorithm_idx
                                    # Use a short delay to ensure the UI updates before solver starts
                                    self._set_message(f"🔄 Recomputing with {algorithm_names[self.algorithm_idx]}...", 2.0)
                                    # Start solver on next frame to avoid blocking the UI event handler
                                    # We'll use a simple flag that the main loop checks
                                    self._pending_solver_trigger = True
                        elif hasattr(widget, 'control_name') and widget.control_name == 'presets':
                            old = self.current_preset_idx
                            self.current_preset_idx = widget.selected
                            if old != self.current_preset_idx:
                                p = self.presets[self.current_preset_idx]
                                # Apply simple presets
                                if p == 'Debugging':
                                    self.feature_flags['show_heatmap'] = True
                                    self.feature_flags['solver_comparison'] = False
                                    self.feature_flags['ml_heuristic'] = False
                                elif p == 'Fast Approx':
                                    self.feature_flags['ml_heuristic'] = True
                                    self.feature_flags['parallel_search'] = True
                                elif p == 'Optimal':
                                    self.feature_flags['ml_heuristic'] = False
                                    self.feature_flags['parallel_search'] = False
                                elif p == 'Speedrun':
                                    self.feature_flags['speedrun_mode'] = True
                                    self.feature_flags['ml_heuristic'] = True
                                self._set_message(f"Preset applied: {p}")

            return handled
        elif event_type == 'up':
            # Translate pos like we do for mouse-down when scrolled
            if getattr(self, 'control_panel_can_scroll', False) and getattr(self, 'control_panel_rect', None) and self.control_panel_rect.collidepoint(pos):
                sc_pos = (pos[0], pos[1] + getattr(self, 'control_panel_scroll', 0))
            else:
                sc_pos = pos
            return self.widget_manager.handle_mouse_up(sc_pos, button)
        return False
    
    # Button callbacks
    def _stop_auto_solve(self):
        """Stop auto-solve and clear visual state."""
        self.auto_mode = False
        self.auto_path = []  # Clear path display
        self.auto_step_idx = 0
        self.block_push_animations = []  # Clear block push animations
        self.message = "Auto-solve stopped"
    
    def _generate_dungeon(self):
        """Generate a new random dungeon using the procedural generator."""
        try:
            from src.generation.dungeon_generator import DungeonGenerator, Difficulty
            import random
            
            # Generate random seed for reproducibility display
            seed = random.randint(0, 999999)
            
            # Create generator with medium difficulty, reasonable size
            generator = DungeonGenerator(
                width=40,
                height=40,
                difficulty=Difficulty.MEDIUM,
                seed=seed
            )
            
            # Generate the dungeon grid
            grid = generator.generate()
            
            # Add the generated dungeon to the map list
            dungeon_name = f"Generated #{seed}"
            self.maps.append(grid)
            self.map_names.append(dungeon_name)
            
            # Switch to the new map
            self.current_map_idx = len(self.maps) - 1
            self._load_current_map()
            self._center_view()
            
            # Clear any existing effects and reset state
            if self.effects:
                self.effects.clear()
            self.step_count = 0
            self.auto_path = []
            self.auto_mode = False
            
            self._set_message(f"Generated dungeon (seed: {seed}, {len(generator.rooms)} rooms)")
            logger.info(f"Generated dungeon: seed={seed}, rooms={len(generator.rooms)}, keys={len(generator.key_positions)}")
            
        except ImportError as e:
            logger.warning(f"Dungeon generator not available: {e}")
            self._set_message("Dungeon generator module not found")
        except Exception as e:
            logger.exception(f"Failed to generate dungeon: {e}")
            self._set_message(f"Generation failed: {str(e)}")

    def _generate_ai_dungeon(self):
        """Generate a comprehensive dungeon using the full H-MOLQD pipeline.

        Pipeline:
            1. MissionGrammar   → mission graph (rooms + lock/key ordering)
            2. ConditionEncoder → graph-aware conditioning vector
            3. LatentDiffusion  → denoised latent (DDIM 50 steps)
            4. VQ-VAE decode    → tile logits → argmax → semantic grid
            5. SymbolicRefiner  → WFC repair for structural validity
        """
        try:
            import torch
            import numpy as np
            import random as _random
            from pathlib import Path
            from src.core.latent_diffusion import create_latent_diffusion
            from src.core.vqvae import create_vqvae
            from src.core.condition_encoder import create_condition_encoder
            from src.core.logic_net import LogicNet as _LogicNet
            from src.generation.grammar import (
                MissionGrammar,
                Difficulty as GrammarDifficulty,
                graph_to_gnn_input,
            )

            checkpoint_path = Path(__file__).parent / "checkpoints" / "final_model.pth"

            if not checkpoint_path.exists():
                self._set_message("No AI checkpoint found – train first!")
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                return

            self._set_message("Generating mission graph…")
            logger.info(f"AI Generation: Loading checkpoint from {checkpoint_path}")

            device = torch.device("cpu")

            # ======================================================
            # STEP 1: Generate mission graph via MissionGrammar
            # ======================================================
            seed = _random.randint(0, 999999)
            grammar = MissionGrammar(seed=seed)
            mission_graph = grammar.generate(
                difficulty=GrammarDifficulty.MEDIUM,
                num_rooms=_random.randint(5, 10),
                max_keys=2,
            )
            num_nodes = len(mission_graph.nodes)
            num_edges = len(mission_graph.edges)
            logger.info(f"  Mission graph: {num_nodes} nodes, {num_edges} edges, seed={seed}")

            # Convert to GNN tensors for condition encoder
            gnn_input = graph_to_gnn_input(mission_graph, current_node_idx=0)
            node_features_raw = gnn_input['node_features']   # [N, D_node]
            edge_index = gnn_input['edge_index']              # [2, E]
            tpe = gnn_input['tpe']                            # [N, 8]

            self._set_message("Loading AI model…")

            # ======================================================
            # STEP 2: Build & load model components
            # ======================================================
            vqvae = create_vqvae(num_classes=44, latent_dim=64)
            diffusion = create_latent_diffusion(latent_dim=64, context_dim=256)
            cond_encoder = create_condition_encoder(latent_dim=64, output_dim=256)

            # Wire LogicNet so checkpoint keys match
            diffusion.guidance.logic_net = _LogicNet(latent_dim=64, num_classes=44)

            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

            if "ema_diffusion_state_dict" in ckpt:
                diffusion.load_state_dict(ckpt["ema_diffusion_state_dict"])
            elif "diffusion_state_dict" in ckpt:
                diffusion.load_state_dict(ckpt["diffusion_state_dict"])
            if "vqvae_state_dict" in ckpt:
                vqvae.load_state_dict(ckpt["vqvae_state_dict"])
                logger.info("  Loaded VQ-VAE from main checkpoint")
            else:
                # Fallback: load separately pretrained VQ-VAE
                vqvae_path = Path("checkpoints/vqvae_pretrained.pth")
                if vqvae_path.exists():
                    vqvae_ckpt = torch.load(vqvae_path, map_location=device, weights_only=False)
                    vqvae.load_state_dict(vqvae_ckpt["model_state_dict"])
                    logger.info(f"  Loaded VQ-VAE from {vqvae_path}")
                else:
                    logger.warning("  ⚠ No VQ-VAE weights found — decode will produce noise!")
            if "condition_encoder_state_dict" in ckpt:
                cond_encoder.load_state_dict(ckpt["condition_encoder_state_dict"])

            vqvae.eval()
            diffusion.eval()
            cond_encoder.eval()

            self._set_message("Running diffusion sampling…")

            # ======================================================
            # STEP 3: Encode graph → conditioning vector
            # ======================================================
            # The condition encoder's global stream expects node_feature_dim=5.
            # MissionGrammar produces 12-dim features (8 type + 2 pos + 2 extra).
            # We project to 5 dims matching training's synthetic graph format:
            #   [is_start, has_enemy, has_key, has_boss_key, has_triforce]
            from src.generation.grammar import NodeType
            node_feat_5 = torch.zeros(num_nodes, 5, device=device)
            sorted_ids = sorted(mission_graph.nodes.keys())
            for i, nid in enumerate(sorted_ids):
                nt = mission_graph.nodes[nid].node_type
                if nt == NodeType.START:
                    node_feat_5[i, 0] = 1.0   # is_start
                elif nt == NodeType.ENEMY:
                    node_feat_5[i, 1] = 1.0   # has_enemy
                elif nt == NodeType.KEY:
                    node_feat_5[i, 2] = 1.0   # has_key
                elif nt == NodeType.LOCK:
                    node_feat_5[i, 3] = 1.0   # has_boss_key (lock)
                elif nt == NodeType.GOAL:
                    node_feat_5[i, 4] = 1.0   # has_triforce

            with torch.no_grad():
                # Encode graph topology → [N, 256] node embeddings
                c_global = cond_encoder.encode_global_only(
                    node_feat_5, edge_index
                )
                # Mean-pool to [1, 256] — same as training does
                conditioning = c_global.mean(dim=0, keepdim=True)

            # ======================================================
            # STEP 4: Diffusion sampling → VQ-VAE decode
            # ======================================================
            # Training data: stitched dungeons padded to a max size.
            # VQ-VAE downsamples by ~4×, so ~11×16 room → 3×4 latent.
            # For a full dungeon we use a larger latent to get a bigger grid.
            # Scale latent proportional to the graph complexity:
            #   base ~3×4 for single room, scale up for multi-room graphs
            scale = max(1, int(num_nodes ** 0.5))
            lat_h = 3 * scale
            lat_w = 4 * scale
            logger.info(f"  Latent shape: (1, 64, {lat_h}, {lat_w}) for {num_nodes}-node graph")

            with torch.no_grad():
                z = diffusion.ddim_sample(
                    context=conditioning,
                    shape=(1, 64, lat_h, lat_w),
                    num_steps=50,
                )
                # Decode through VQ-VAE → [1, 44, H, W] logits
                # Pass target_size so transposed convolutions output the right dims
                target_h = lat_h * 4   # VQ-VAE has 4× downsampling
                target_w = lat_w * 4
                recon = vqvae.decode(z, target_size=(target_h, target_w))
                # argmax → tile IDs (H, W)
                tile_grid = recon.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)

            h, w = tile_grid.shape
            logger.info(f"  Raw generation: {h}×{w}, unique_tiles={len(np.unique(tile_grid))}")

            # ======================================================
            # STEP 5: Symbolic refinement (WFC repair)
            # ======================================================
            self._set_message("Refining dungeon structure…")
            try:
                from src.core.symbolic_refiner import create_symbolic_refiner
                refiner = create_symbolic_refiner(max_repair_attempts=3)
                start_pos = (2, 2)
                goal_pos = (h - 3, w - 3)

                # Find actual start/triforce positions if they exist
                from src.core.definitions import SEMANTIC_PALETTE as _SP
                start_positions = np.argwhere(tile_grid == _SP['START'])
                goal_positions = np.argwhere(tile_grid == _SP['TRIFORCE'])
                if len(start_positions) > 0:
                    start_pos = tuple(start_positions[0])
                if len(goal_positions) > 0:
                    goal_pos = tuple(goal_positions[0])

                repaired_grid, success = refiner.repair_room(
                    tile_grid, start_pos, goal_pos
                )
                if success:
                    tile_grid = repaired_grid.astype(np.int32)
                    logger.info("  Symbolic refinement: SUCCESS")
                else:
                    logger.info("  Symbolic refinement: no repair needed or failed")
            except Exception as e:
                logger.warning(f"  Symbolic refinement skipped: {e}")

            # ======================================================
            # STEP 6: Ensure START and TRIFORCE exist
            # ======================================================
            from src.core.definitions import SEMANTIC_PALETTE as _SP
            if not np.any(tile_grid == _SP['START']):
                # Place START on the first walkable floor tile near top-left
                floor_positions = np.argwhere(tile_grid == _SP['FLOOR'])
                if len(floor_positions) > 0:
                    sp = floor_positions[0]
                    tile_grid[sp[0], sp[1]] = _SP['START']
                    logger.info(f"  Placed START at ({sp[0]}, {sp[1]})")

            if not np.any(tile_grid == _SP['TRIFORCE']):
                # Place TRIFORCE on the last walkable floor tile near bottom-right
                floor_positions = np.argwhere(tile_grid == _SP['FLOOR'])
                if len(floor_positions) > 0:
                    gp = floor_positions[-1]
                    tile_grid[gp[0], gp[1]] = _SP['TRIFORCE']
                    logger.info(f"  Placed TRIFORCE at ({gp[0]}, {gp[1]})")

            # ======================================================
            # STEP 7: Add to GUI map list
            # ======================================================
            h, w = tile_grid.shape
            dungeon_name = f"AI #{seed} ({num_nodes}rm {h}×{w})"

            self.maps.append(tile_grid)
            self.map_names.append(dungeon_name)

            # Switch to the new map
            self.current_map_idx = len(self.maps) - 1
            self._load_current_map()
            self._center_view()

            # Clear any existing effects and reset state
            if self.effects:
                self.effects.clear()
            self.step_count = 0
            self.auto_path = []
            self.auto_mode = False

            self._set_message(
                f"AI dungeon generated: {num_nodes} rooms, {h}×{w} tiles, seed={seed}"
            )
            logger.info(
                f"AI dungeon complete: seed={seed}, graph={num_nodes}N/{num_edges}E, "
                f"grid={h}×{w}, unique_tiles={len(np.unique(tile_grid))}"
            )

        except Exception as e:
            logger.exception(f"AI generation failed: {e}")
            self._set_message(f"AI generation failed: {str(e)}")

    def _reset_map(self):
        """Reset the current map."""
        self._load_current_map()
        self._center_view()
        if self.effects:
            self.effects.clear()
        self.step_count = 0
        self.message = "Map Reset"
    
    def _show_path_preview(self):
        """Show path preview dialog (placeholder)."""
        self.message = "Path preview not available"
    
    def _clear_path(self):
        """Clear the current path."""
        self.auto_path = []
        self.auto_mode = False
        self.message = "Path cleared"
    
    def _export_route(self):
        """Export the current route (placeholder)."""
        self.message = "Route export not implemented yet"
    
    def _load_route(self):
        """Load a saved route (placeholder)."""
        self.message = "Route loading not implemented yet"

    def load_visual_assets(self, templates_dir: str = None, link_sprite_path: str = None):
        """Optional: override GUI assets with extracted visual tiles/sprites.

        Usage (copy-paste into startup code):
            gui = ZeldaGUI(maps)
            gui.load_visual_assets('data/tileset.png', 'data/link_sprite.png')

        Behaviour:
        - If `templates_dir` is a folder of tile images, create pygame surfaces from them
          and assign to `self.images` keyed by semantic id (best-effort).
        - If `link_sprite_path` is provided, attempt to cut a Link sprite and replace `self.link_img`.
        """
        try:
            from src.data_processing.visual_extractor import extract_grid
            from PIL import Image
        except Exception:
            return

        if templates_dir and os.path.isdir(templates_dir):
            # load any PNGs found and map into fallback image slots
            for fn in sorted(os.listdir(templates_dir)):
                if not fn.lower().endswith('.png'):
                    continue
                name = os.path.splitext(fn)[0]
                try:
                    im = Image.open(os.path.join(templates_dir, fn)).convert('RGBA')
                    surf = pygame.image.fromstring(im.tobytes(), im.size, im.mode)
                    # best-effort: if filename contains 'floor','wall','door','key' map to semantic ids
                    ln = name.lower()
                    if 'floor' in ln or 'f' == ln:
                        self.images[SEMANTIC_PALETTE['FLOOR']] = pygame.transform.scale(surf, (self.TILE_SIZE, self.TILE_SIZE))
                    if 'wall' in ln:
                        self.images[SEMANTIC_PALETTE['WALL']] = pygame.transform.scale(surf, (self.TILE_SIZE, self.TILE_SIZE))
                    if 'door' in ln:
                        self.images[SEMANTIC_PALETTE['DOOR_OPEN']] = pygame.transform.scale(surf, (self.TILE_SIZE, self.TILE_SIZE))
                    if 'key' in ln:
                        self.images[SEMANTIC_PALETTE['KEY']] = pygame.transform.scale(surf, (self.TILE_SIZE, self.TILE_SIZE))
                except Exception:
                    continue

        if link_sprite_path and os.path.exists(link_sprite_path):
            try:
                im = Image.open(link_sprite_path).convert('RGBA')
                im = im.resize((self.TILE_SIZE - 4, self.TILE_SIZE - 4), Image.NEAREST)
                self.link_img = pygame.image.fromstring(im.tobytes(), im.size, im.mode)
            except Exception as e:
                logger.warning(f"Failed to load link sprite from {link_sprite_path}: {e}")
        
        logger.info(f"Loaded {len([k for k in self.images if k in (1,2,10,12)])} visual assets")
        return True

    def load_visual_map(self, image_path: str, templates_dir: str | None = None):
        """Public API: create a GUI map from a screenshot and switch to it.

        - `image_path` can be a full screenshot (HUD allowed).
        - `templates_dir` is passed to the visual extractor (tileset or folder).

        This method is intentionally permissive and returns a bool for success
        so automated tests can call it without a file dialog.
        """
        try:
            from src.data_processing.visual_integration import visual_extract_to_room, make_stitched_for_single_room, infer_inventory_from_room
        except Exception:
            return False

        try:
            ids, conf = visual_extract_to_room(image_path, templates_dir or '')
            # convert single-room semantic -> environment grid expected by GUI
            stitched = make_stitched_for_single_room(ids)
            self.maps = [stitched.global_grid]
            self.current_map_idx = 0
            self._load_current_map()
            self.message = f"Loaded visual map: {image_path}"
            return True
        except Exception as e:
            self.message = f"Visual load failed: {e}"
            return False

    def _place_items_from_graph(self, grid: np.ndarray, graph, room_positions: dict, room_to_node: dict):
        """Place items (keys, boss keys, etc.) from graph node attributes into the grid.
        
        The VGLC data adapter stores items as graph node attributes (has_key=True, etc.)
        but doesn't place them in the semantic grid. This function materializes those
        items into the grid so the inventory system can track them.
        
        Args:
            grid: Numpy array of semantic tile IDs (modified in place)
            graph: NetworkX graph with node attributes
            room_positions: Dict mapping room position -> (row_offset, col_offset) in global grid
            room_to_node: Dict mapping room position -> graph node ID
        """
        if not graph or not room_positions or not room_to_node:
            return
        
        # Reverse mapping: node_id -> room_position
        node_to_room = {v: k for k, v in room_to_node.items()}
        
        # Room dimensions (standard VGLC Zelda)
        ROOM_HEIGHT = 16
        ROOM_WIDTH = 11
        
        items_placed = 0
        
        for node_id in graph.nodes():
            attrs = graph.nodes[node_id]
            room_pos = node_to_room.get(node_id)
            if room_pos is None or room_pos not in room_positions:
                continue
            
            r_off, c_off = room_positions[room_pos]
            
            # Find a valid floor position in the room interior (avoid walls)
            # We'll search for a FLOOR tile in the interior region
            def find_floor_position(offset_r: int = 0, offset_c: int = 0):
                """Find a floor tile position in the room, offset from center."""
                # Room interior is roughly rows 2-13 (avoid top/bottom walls)
                # and columns 2-8 (avoid left/right walls)
                center_r = r_off + ROOM_HEIGHT // 2 + offset_r
                center_c = c_off + ROOM_WIDTH // 2 + offset_c
                
                # Search in expanding rectangles from center
                for dr in range(-5, 6):
                    for dc in range(-3, 4):
                        r = center_r + dr
                        c = center_c + dc
                        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                            if grid[r, c] == SEMANTIC_PALETTE['FLOOR']:
                                return (r, c)
                return None
            
            # Place key if room has_key
            if attrs.get('has_key', False):
                pos = find_floor_position(offset_r=-2, offset_c=0)
                if pos:
                    grid[pos[0], pos[1]] = SEMANTIC_PALETTE['KEY_SMALL']
                    items_placed += 1
                    logger.debug("Placed KEY at %s (room %s, node %d)", pos, room_pos, node_id)
            
            # Place boss key if room has boss (boss key is usually near boss)
            # Note: has_boss_key isn't in standard graph format, boss key might need different handling
            if attrs.get('has_item', False):
                # KEY_ITEM could be ladder or other special item
                pos = find_floor_position(offset_r=0, offset_c=-2)
                if pos:
                    grid[pos[0], pos[1]] = SEMANTIC_PALETTE['KEY_ITEM']
                    items_placed += 1
                    logger.debug("Placed KEY_ITEM at %s (room %s, node %d)", pos, room_pos, node_id)
        
        if items_placed > 0:
            logger.info("_place_items_from_graph: placed %d items into grid", items_placed)

    def _load_current_map(self):
        """Load and initialize the current map."""
        current_dungeon = self.maps[self.current_map_idx]
        # Extract grid and graph info from StitchedDungeon
        if hasattr(current_dungeon, 'global_grid'):
            # It's a StitchedDungeon object
            grid = current_dungeon.global_grid.copy()  # Copy so we can modify it
            graph = current_dungeon.graph
            room_to_node = current_dungeon.room_to_node
            room_positions = current_dungeon.room_positions
            node_to_room = getattr(current_dungeon, 'node_to_room', None)
            
            # === PLACE ITEMS FROM GRAPH NODE ATTRIBUTES INTO GRID ===
            # Keys are stored in graph nodes (has_key=True) but not in the grid
            if graph and room_positions and room_to_node:
                self._place_items_from_graph(grid, graph, room_positions, room_to_node)
        else:
            # Legacy: just a grid array
            grid = current_dungeon
            graph = None
            room_to_node = None
            room_positions = None
            node_to_room = None
        
        self.env = ZeldaLogicEnv(grid, render_mode=False, graph=graph, 
                                  room_to_node=room_to_node, room_positions=room_positions,
                                  node_to_room=node_to_room)
        # Defer heavy solver initialization until actually needed (pressing SPACE)
        self.solver = None
        self.auto_path = []
        self.auto_step_idx = 0
        self.auto_mode = False
        
        # Clear any active block push animations
        self.block_push_animations = []
        
        # DEBUG: Add test path to verify rendering works (will be overwritten by solver)
        # Set KLTN_DEBUG_TEST_PATH=1 environment variable to enable
        if os.environ.get('KLTN_DEBUG_TEST_PATH') == '1':
            self._test_path = [(5, 5), (5, 6), (5, 7), (5, 8), (6, 8), (7, 8), (8, 8), (8, 9), (8, 10)]
            print(f"[LOAD_MAP] _test_path ENABLED: {len(self._test_path)} points at {self._test_path[0]} to {self._test_path[-1]}")
        else:
            self._test_path = None
        
        # Clear solver result when loading new map
        self.solver_result = None
        self.current_keys_held = 0
        self.current_keys_used = 0
        self.current_edge_types = []
        
        # Count total items in dungeon for "X/Y collected" display
        self.total_keys = len(self.env._find_all_positions(SEMANTIC_PALETTE['KEY_SMALL']))
        # Count total bomb items in dungeon for consumable tracking
        bomb_items = self.env._find_all_positions(SEMANTIC_PALETTE['ITEM_MINOR'])
        key_items = self.env._find_all_positions(SEMANTIC_PALETTE['KEY_ITEM'])
        self.total_bombs = len(bomb_items) + len(key_items)  # Each gives 4 bombs
        self.total_boss_keys = len(self.env._find_all_positions(SEMANTIC_PALETTE['KEY_BOSS']))
        self.keys_collected = 0
        self.bombs_collected = 0
        self.boss_keys_collected = 0
        # Reset usage counters
        self.keys_used = 0
        self.bombs_used = 0
        self.boss_keys_used = 0
        # Reset collected/used logs
        self.collected_items = []
        self.collected_positions = set()  # Reset collected positions for new map
        self.used_items = []
        
        # Reset path items preview (new feature)
        self.path_items_summary = {}
        self.path_item_positions = {}
        
        # Clear search heatmap
        self.search_heatmap = {}
        
        # Initialize renderer agent position
        if self.renderer and self.env.start_pos:
            self.renderer.set_agent_position(
                self.env.start_pos[0],
                self.env.start_pos[1],
                immediate=True
            )
        
        # Run sanity check
        checker = SanityChecker(grid)
        is_valid, errors = checker.check_all()
        
        if not is_valid:
            self.message = f"Map Error: {errors[0]}"
        else:
            self.message = f"Map {self.current_map_idx + 1}/{len(self.maps)} - Press SPACE to solve"
        
        # Auto-fit zoom to show entire map
        self._auto_fit_zoom()
        # Re-center view after zoom changes to avoid large offsets that place the
        # viewport outside of the map (fixes black/empty window on start)
        try:
            self._center_view()
        except Exception:
            pass
        # Log map + view state to help debug black/empty screen issues
        try:
            logger.info("Map loaded: %dx%d, TILE_SIZE=%d, view_offset=(%d,%d), images=%d",
                        self.env.width if self.env else 0,
                        self.env.height if self.env else 0,
                        self.TILE_SIZE,
                        getattr(self, 'view_offset_x', 0),
                        getattr(self, 'view_offset_y', 0),
                        len(getattr(self, 'images', {})))
        except Exception:
            pass

        # Kick off optional preview/scan work on main thread (must not run in watchdog)
        try:
            self._start_preview_for_current_map()
        except Exception:
            logger.exception('Failed to start preview for current map')
    
    def _center_view(self):
        """Center the current map in the view."""
        if self.env is None:
            return
        map_w = self.env.width * self.TILE_SIZE
        map_h = self.env.height * self.TILE_SIZE
        view_w = self.screen_w - self.SIDEBAR_WIDTH
        view_h = self.screen_h - self.HUD_HEIGHT
        # Allow centering even when map is smaller than view (negative offset)
        self.view_offset_x = (map_w - view_w) // 2
        self.view_offset_y = (map_h - view_h) // 2
        self._clamp_view_offset()
    
    def _auto_fit_zoom(self):
        """Automatically set zoom level to fit the entire map in view."""
        if self.env is None:
            return
        
        view_w = self.screen_w - self.SIDEBAR_WIDTH - 20  # padding
        view_h = self.screen_h - self.HUD_HEIGHT - 20
        
        map_h = self.env.height
        map_w = self.env.width
        
        # Find the largest zoom level that fits
        best_zoom_idx = 0
        for idx, tile_size in enumerate(self.ZOOM_LEVELS):
            if map_w * tile_size <= view_w and map_h * tile_size <= view_h:
                best_zoom_idx = idx
            else:
                break
        
        # Apply the zoom
        if best_zoom_idx != self.zoom_idx:
            self.zoom_idx = best_zoom_idx
            self.TILE_SIZE = self.ZOOM_LEVELS[self.zoom_idx]
            self._load_assets()
            # Update renderer tile size
            if self.renderer:
                self.renderer.set_tile_size(self.TILE_SIZE)
        
        self._center_view()
        
        # Initialize agent position in renderer
        if self.renderer and self.env and self.env.start_pos:
            self.renderer.set_agent_position(
                self.env.start_pos[0], 
                self.env.start_pos[1], 
                immediate=True
            )
    
    def _change_zoom(self, delta: int, center: tuple | None = None):
        """Change zoom level by delta steps.

        If `center` is provided (screen coordinates), the view will be adjusted so
        that the map tile under the `center` pixel remains under the cursor after
        the zoom. If `center` is None, the view is centered as before.
        """
        old_idx = self.zoom_idx
        new_idx = max(0, min(len(self.ZOOM_LEVELS) - 1, self.zoom_idx + delta))
        if new_idx == old_idx:
            return

        # Compute tile-space coordinates under center (in tile units)
        if center is None:
            center_x = (self.screen_w - self.SIDEBAR_WIDTH) // 2
            center_y = (self.screen_h - self.HUD_HEIGHT) // 2
        else:
            center_x, center_y = center
            # Clamp center to map area
            center_x = max(0, min(center_x, self.screen_w - self.SIDEBAR_WIDTH))
            center_y = max(0, min(center_y, self.screen_h - self.HUD_HEIGHT))

        old_tile = self.TILE_SIZE
        # Tile coordinates (floating) corresponding to center pixel
        tile_x = (self.view_offset_x + center_x) / float(old_tile) if old_tile else 0.0
        tile_y = (self.view_offset_y + center_y) / float(old_tile) if old_tile else 0.0

        # Apply new zoom
        self.zoom_idx = new_idx
        self.TILE_SIZE = self.ZOOM_LEVELS[self.zoom_idx]
        self._load_assets()  # Reload assets at new size
        # Update renderer tile size
        if self.renderer:
            self.renderer.set_tile_size(self.TILE_SIZE)

        # Compute new offsets so (tile_x, tile_y) stays under center pixel
        new_view_offset_x = int(tile_x * self.TILE_SIZE - center_x)
        new_view_offset_y = int(tile_y * self.TILE_SIZE - center_y)
        self.view_offset_x = new_view_offset_x
        self.view_offset_y = new_view_offset_y
        # Clamp to valid ranges
        self._clamp_view_offset()

        self.message = f"Zoom: {self.TILE_SIZE}px"
    
    def _safe_set_mode(self, size, flags=0, allow_fallback=True):
        """Robust wrapper around pygame.display.set_mode.

        Attempts set_mode and, on failure or invalid surface (size 0), performs
        a display reinit and retries. If all attempts fail and allow_fallback is
        True, falls back to a windowed 800x600 surface to avoid leaving the
        application with a null/zero-sized display.
        Returns the created screen surface (or None on fatal failure).
        """
        try:
            screen = pygame.display.set_mode(size, flags)
        except Exception:
            logger.exception('set_mode(%s, flags=%s) failed; attempting display reinit', size, flags)
            try:
                pygame.display.quit()
                pygame.display.init()
                screen = pygame.display.set_mode(size, flags)
            except Exception:
                logger.exception('Reinit + set_mode failed')
                screen = None

        # Validate surface
        try:
            if screen is None:
                raise RuntimeError('No screen')
            w, h = screen.get_size()
            if w == 0 or h == 0:
                raise RuntimeError(f'Invalid screen size {w}x{h}')
            return screen
        except Exception:
            logger.exception('Created screen is invalid')
            if not allow_fallback:
                return None
            try:
                # Final fallback to a safe windowed mode
                logger.warning('Falling back to windowed 800x600')
                screen = pygame.display.set_mode((800, 600), pygame.RESIZABLE)
                return screen
            except Exception:
                logger.exception('Fallback windowed mode failed')
                return None

    def _attempt_display_reinit(self):
        """Attempt to fully reinitialize the SDL display and restore mode."""
        try:
            pygame.display.quit()
            pygame.display.init()
            # Recreate the preferred mode based on current fullscreen state
            if getattr(self, 'fullscreen', False):
                try:
                    disp = pygame.display.Info()
                    new_size = (int(disp.current_w), int(disp.current_h))
                except Exception:
                    new_size = (800, 600)
                flags = pygame.FULLSCREEN | getattr(pygame, 'HWSURFACE', 0) | getattr(pygame, 'DOUBLEBUF', 0)
                screen = self._safe_set_mode(new_size, flags)
            else:
                prev = getattr(self, '_prev_window_size', (800, 600))
                screen = self._safe_set_mode(prev, pygame.RESIZABLE)
            if screen:
                self.screen = screen
                try:
                    self.screen_w, self.screen_h = self.screen.get_size()
                except Exception:
                    self.screen_w, self.screen_h = (800, 600)
                # Reload assets as surfaces may need recreation
                try:
                    self._load_assets()
                except Exception:
                    logger.exception('Failed to reload assets after display reinit')
                return True
        except Exception:
            logger.exception('Display reinit failed')
        return False

    def _handle_watchdog_screenshot(self) -> bool:
        """Save the requested watchdog screenshot on the main thread and clear the request.

        Returns True if a screenshot was saved, False otherwise. Always clears the
        request to avoid repeated attempts.
        """
        shot = getattr(self, '_watchdog_request_screenshot', None)
        if not shot:
            return False
        try:
            surf = pygame.display.get_surface()
            if not surf:
                logger.warning('Watchdog requested screenshot but no display surface available')
                try:
                    self._watchdog_request_screenshot = None
                except Exception:
                    pass
                return False
            try:
                pygame.image.save(surf, shot)
                logger.warning('Watchdog screenshot saved by main thread: %s', shot)
                self._show_toast(f'Watchdog screenshot: {os.path.basename(shot)}', 3.0, 'info')
                return True
            except Exception:
                logger.exception('Failed to save watchdog screenshot on main thread')
                return False
        finally:
            try:
                self._watchdog_request_screenshot = None
            except Exception:
                pass

    def report_ui_state(self) -> dict:
        """Return diagnostic information about GUI state for troubleshooting (callable from REPL)."""
        try:
            info = {
                'fullscreen': self.fullscreen,
                'screen_w': getattr(self, 'screen_w', None),
                'screen_h': getattr(self, 'screen_h', None),
                'preview_overlay_visible': getattr(self, 'preview_overlay_visible', False),
                'preview_modal_enabled': getattr(self, 'preview_modal_enabled', False),
                'control_panel_enabled': getattr(self, 'control_panel_enabled', False),
                'control_panel_rect': getattr(self, 'control_panel_rect', None),
                'control_panel_collapsed': getattr(self, 'control_panel_collapsed', False),
                'solver_running': getattr(self, 'solver_running', False),
                'solver_proc_alive': getattr(getattr(self, 'solver_proc', None), 'is_alive', lambda : False)(),
                'solver_comparison_thread_alive': getattr(getattr(self, 'solver_comparison_thread', None), 'is_alive', lambda : False)(),
                'debug_click_log_len': len(getattr(self, 'debug_click_log', [])),
            }
        except Exception:
            logger.exception('Failed to build UI state report')
            return {}
        return info

    def _ensure_display_alive(self, force=False):
        """Check display health and attempt recovery if needed.

        If the display surface is None or has zero size, try to restore it.
        This method is intentionally conservative and returns False only when
        no recovery was possible.
        """
        try:
            surf = pygame.display.get_surface()
            if surf is None:
                logger.warning('Display surface is None; attempting recovery')
                recovered = self._attempt_display_reinit()
                if recovered:
                    self._show_toast('Recovered display', 3.0, 'warning')
                    logger.info('Display recovered successfully')
                    # reset attempt counter
                    self._display_recovery_attempts = 0
                    return True
                else:
                    self._show_toast('Display unavailable; fallback failed', 5.0, 'error')
                    return False

            w, h = surf.get_size()
            if w == 0 or h == 0:
                logger.warning('Display surface has invalid size %sx%s; attempting recovery', w, h)
                recovered = self._attempt_display_reinit()
                if recovered:
                    self._show_toast('Recovered display', 3.0, 'warning')
                    self._display_recovery_attempts = 0
                    return True
                else:
                    self._show_toast('Display invalid; returning to windowed mode', 5.0, 'error')
                    return False
            # Healthy
            # Reset attempt counter when healthy
            self._display_recovery_attempts = 0
            return True
        except Exception:
            logger.exception('Error while checking display health')
            return False

    def _force_focus(self) -> bool:
        """Try to force the window to the foreground on Windows.

        Uses a conservative Win32 sequence (AttachThreadInput + SetForegroundWindow + temporary TOPMOST) to
        work around Windows' foreground activation blocking. Returns True on success.
        No-op on non-Windows platforms.
        """
        if os.name != 'nt':
            return False
        try:
            import ctypes
            from ctypes import wintypes
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            hwnd = pygame.display.get_wm_info().get('window')
            if not hwnd:
                logger.debug('No hwnd available for focus')
                return False
            # Get foreground window and thread ids
            fg = user32.GetForegroundWindow()
            pid = wintypes.DWORD()
            fg_tid = user32.GetWindowThreadProcessId(fg, ctypes.byref(pid))
            cur_tid = kernel32.GetCurrentThreadId()
            # Attach input so we can set foreground
            attached = False
            try:
                attached = bool(user32.AttachThreadInput(fg_tid, cur_tid, True))
            except Exception:
                attached = False
            # Bring window to front and set foreground
            SW_SHOW = 5
            user32.ShowWindow(hwnd, SW_SHOW)
            try:
                user32.SetForegroundWindow(hwnd)
            except Exception:
                logger.debug('SetForegroundWindow failed; continuing')
            try:
                user32.BringWindowToTop(hwnd)
            except Exception:
                pass
            # Temporary topmost toggle to help in edge cases
            try:
                SWP_NOSIZE = 0x0001
                SWP_NOMOVE = 0x0002
                HWND_TOPMOST = -1
                HWND_NOTOPMOST = -2
                user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)
                user32.SetWindowPos(hwnd, HWND_NOTOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)
            except Exception:
                pass
            # Detach thread input if we attached
            try:
                if attached:
                    user32.AttachThreadInput(fg_tid, cur_tid, False)
            except Exception:
                pass
            pygame.event.pump()
            logger.debug('Attempted Win32 force-focus sequence')
            return True
        except Exception:
            logger.exception('Win32 focus helper failed')
            return False

    def _toggle_fullscreen(self):
        """Toggle fullscreen mode with robust handling.

        Uses `pygame.display.Info()` to obtain a valid fullscreen size and
        preserves the previous windowed size for restore. Ensures event pump
        and asset/layout reinitialization to avoid dark screens or unresponsiveness.
        """
        self.fullscreen = not self.fullscreen
        try:
            if self.fullscreen:
                # Save current window size so we can restore it later
                try:
                    self._prev_window_size = (self.screen_w, self.screen_h)
                except Exception:
                    self._prev_window_size = getattr(self, '_prev_window_size', (800, 600))

                # Use display info to pick a safe fullscreen size and use double buffered HW surface if available
                try:
                    disp = pygame.display.Info()
                    new_size = (int(disp.current_w), int(disp.current_h))
                except Exception:
                    new_size = (0, 0)

                flags = pygame.FULLSCREEN | getattr(pygame, 'HWSURFACE', 0) | getattr(pygame, 'DOUBLEBUF', 0)
                screen = self._safe_set_mode(new_size, flags)
                if not screen:
                    # If we couldn't set fullscreen, revert and notify user
                    self.fullscreen = False
                    self._set_message('Failed to enter fullscreen; reverted to windowed', 4.0)
                    self._show_toast('Fullscreen failed; using windowed mode', 4.0, 'warning')
                else:
                    self.screen = screen
                    try:
                        self.screen_w, self.screen_h = self.screen.get_size()
                    except Exception:
                        try:
                            self.screen_w, self.screen_h = new_size
                        except Exception:
                            self.screen_w, self.screen_h = (800, 600)
# Ensure event pump, grab input, and immediate redraw
                    try:
                        try:
                            default_grab = '0' if __import__('platform').system().lower().startswith('win') else '1'
                            if os.environ.get('KLTN_FULLSCREEN_GRAB', default_grab) == '1':
                                pygame.event.set_grab(True)
                            else:
                                logger.debug('KLTN_FULLSCREEN_GRAB=0 or platform indicates no grab; skipping event grab')
                        except Exception:
                            logger.debug('Could not set event grab for fullscreen')
                        pygame.event.pump()
                        self._load_assets()
                        self._render()
                        pygame.display.flip()
                        self._show_toast('Entered fullscreen', 2.5, 'success')
                    except Exception:
                        logger.exception('Post-fullscreen redraw failed')

            else:
                # Restore previous window size (if known) when exiting fullscreen
                prev = getattr(self, '_prev_window_size', (800, 600))
                screen = self._safe_set_mode(prev, pygame.RESIZABLE)
                if not screen:
                    logger.exception('Failed to restore windowed mode; falling back to 800x600')
                    self.screen = pygame.display.set_mode((800, 600), pygame.RESIZABLE)
                    self.screen_w, self.screen_h = self.screen.get_size()
                    self._show_toast('Restored to windowed (fallback)', 3.0, 'warning')
                else:
                    self.screen = screen
                    self.screen_w, self.screen_h = self.screen.get_size()
                    try:
                        try:
                            pygame.event.set_grab(False)
                        except Exception:
                            logger.debug('Could not clear event grab on exiting fullscreen')
                        pygame.event.pump()
                        self._load_assets()
                        self._render()
                        pygame.display.flip()
                        self._show_toast('Exited fullscreen', 2.0, 'info')
                    except Exception:
                        logger.exception('Post-windowed redraw failed')
        except Exception:
            logger.exception('Unhandled exception in _toggle_fullscreen')
            # Best-effort revert
            try:
                prev = getattr(self, '_prev_window_size', (800, 600))
                self.screen = pygame.display.set_mode(prev, pygame.RESIZABLE)
                self.screen_w, self.screen_h = self.screen.get_size()
            except Exception:
                logger.exception('Failed to revert to previous window mode after fullscreen error')
                try:
                    self.screen = pygame.display.set_mode((800, 600), pygame.RESIZABLE)
                    self.screen_w, self.screen_h = self.screen.get_size()
                except Exception:
                    self.screen_w, self.screen_h = (800, 600)

            # Give SDL a chance to process pending events and stabilize the display
            try:
                pygame.event.pump()
            except Exception:
                pass

            # Recalculate view and reload any assets that depend on display format
            try:
                # Convert surfaces to match new display format and re-layout widgets
                self._load_assets()
            except Exception:
                logger.exception('Failed to reload assets after fullscreen toggle')
            try:
                if self.control_panel_enabled:
                    self._update_control_panel_positions()
            except Exception:
                pass

            # Center view and render one immediate frame to avoid a dark or stale display
            try:
                self._center_view()
                self._render()
                pygame.display.flip()
            except Exception:
                logger.exception('Failed to render after fullscreen toggle')

            self._set_message(f'Fullscreen: {"ON" if self.fullscreen else "OFF"}', 1.5)

    # ------------------ Control Panel Animation ------------------
    def _start_toggle_panel_animation(self, target_collapsed: bool):
        """Begin animated transition to collapsed or expanded state."""
        collapsed_width = 40
        # Compute target width
        target_width = float(collapsed_width) if target_collapsed else float(self.control_panel_width)
        # Initialize animation state
        self.control_panel_anim_from = float(self.control_panel_width_current)
        self.control_panel_anim_to = float(target_width)
        self.control_panel_anim_start = time.time()
        self.control_panel_anim_duration = 0.22
        self.control_panel_target_collapsed = target_collapsed
        self.control_panel_animating = True

    def _update_control_panel_animation(self):
        """Update animation state; should be called each frame."""
        if not getattr(self, 'control_panel_animating', False):
            return
        # Keep widget positions up to date during animation to maintain correct hitboxes
        try:
            self._update_control_panel_positions()
        except Exception:
            pass
        elapsed = time.time() - self.control_panel_anim_start
        t = min(1.0, elapsed / max(1e-6, self.control_panel_anim_duration))
        # Smoothstep easing
        ease = t * t * (3 - 2 * t)
        self.control_panel_width_current = self.control_panel_anim_from + (self.control_panel_anim_to - self.control_panel_anim_from) * ease
        # If finished
        if t >= 1.0:
            self.control_panel_animating = False
            self.control_panel_width_current = self.control_panel_anim_to
            # Apply final collapsed flag
            self.control_panel_collapsed = bool(self.control_panel_target_collapsed)
            # Ensure widgets are (re)initialized / repositioned after animation completes
            try:
                if not self.control_panel_collapsed:
                    self._update_control_panel_positions()
                self._set_message(f"Panel: {'collapsed' if self.control_panel_collapsed else 'expanded'}")
            except Exception:
                # Avoid raising during animation cleanup
                pass

    def _update_control_panel_scroll(self):
        """Per-frame update that applies inertia (momentum) and clamps scroll."""
        if not getattr(self, 'control_panel_can_scroll', False):
            return
        # If user is actively dragging the thumb, don't apply inertia
        if getattr(self, 'control_panel_scroll_dragging', False):
            return
        vel = getattr(self, 'control_panel_scroll_velocity', 0.0)
        # Nothing to do if no velocity
        if abs(vel) < 1.0:
            self.control_panel_scroll_velocity = 0.0
            return
        # Advance scroll by velocity (pixels per second) scaled by delta_time
        prev = self.control_panel_scroll
        self.control_panel_scroll = max(0, min(getattr(self, 'control_panel_scroll_max', 0), self.control_panel_scroll + vel * max(1e-6, self.delta_time)))
        # If hit bounds, zero velocity
        if self.control_panel_scroll <= 0 or self.control_panel_scroll >= getattr(self, 'control_panel_scroll_max', 0):
            self.control_panel_scroll_velocity = 0.0
        else:
            # Apply simple linear damping per second
            damping = getattr(self, 'control_panel_scroll_damping', 6.0)
            self.control_panel_scroll_velocity *= max(0.0, 1.0 - damping * self.delta_time)
            if abs(self.control_panel_scroll_velocity) < 1.0:
                self.control_panel_scroll_velocity = 0.0
        # Set a short ignore window to avoid accidental toggles while momentum is active
        if abs(self.control_panel_scroll - prev) > 0.5:
            self.control_panel_ignore_click_until = time.time() + 0.12

    def run(self, max_frames: Optional[int] = None):
        """Main game loop with delta-time support.

        When running under tests (env var KLTN_TEST_MODE or under pytest), a small
        default max_frames is used to avoid infinite loops. Callers can override
        with the optional max_frames parameter.
        """
        # If running in test mode and no explicit max_frames provided, set a small limit
        if max_frames is None and (os.environ.get('KLTN_TEST_MODE') or os.environ.get('PYTEST_CURRENT_TEST') or os.environ.get('CI')):
            try:
                max_frames = int(os.environ.get('KLTN_RUN_MAX_FRAMES', '10'))
            except Exception:
                max_frames = 10

        # Heartbeat logging variables for responsiveness debugging
        heartbeat_last = time.time()
        heartbeat_interval = 0.5  # seconds (more frequent for debugging)

        running = True
        frame_count = 0
        
        while running:
            # Calculate delta time for smooth animations
            current_time = time.time()
            self.delta_time = current_time - self.last_frame_time
            self.last_frame_time = current_time
            
            # Handle events (measure event.get duration to catch blocking behavior)
            try:
                _evt_start = time.time()
                _events = pygame.event.get()
                _evt_dur = time.time() - _evt_start
                if _evt_dur > 0.05:
                    logger.debug('Slow event.get() detected: %.3fs', _evt_dur)
            except Exception:
                logger.exception('pygame.event.get() raised')
                _events = []

            # Periodic fallback: if window is not focused, try to clear grab/ensure cursor visible
            try:
                # Only for windowed mode; in fullscreen platforms we may intentionally grab
                if not self.fullscreen:
                    focused = pygame.mouse.get_focused()
                    if not focused and time.time() - getattr(self, '_last_ungrab_attempt', 0.0) > 2.0:
                        logger.debug('Window lacks input focus; attempting to clear event grab and show cursor')
                        try:
                            pygame.event.set_grab(False)
                        except Exception:
                            logger.debug('Failed to clear event grab during fallback')
                        try:
                            pygame.mouse.set_visible(True)
                        except Exception:
                            logger.debug('Failed to set mouse visible during fallback')
                        self._last_ungrab_attempt = time.time()
            except Exception:
                logger.exception('Error during input focus fallback')
            for event in _events:
                # Sanity-fix: if preview overlay is active but we have no dialog or planned path, it can block input forever; clear it.
                if getattr(self, 'preview_overlay_visible', False) and not (getattr(self, 'path_preview_dialog', None) or getattr(self, 'auto_path', None)):
                    try:
                        logger.warning('Clearing stale preview_overlay_visible (no dialog/path present) to restore input')
                        self.preview_overlay_visible = False
                        self.path_preview_dialog = None
                        try:
                            self._set_message('Cleared stale preview overlay', 1.5)
                        except Exception:
                            pass
                    except Exception:
                        logger.exception('Failed to clear stale preview overlay')

                # Handle window focus events (improves input responsiveness on Windows)
                if event.type == getattr(pygame, 'WINDOWFOCUSGAINED', None):
                    logger.debug('WINDOWFOCUSGAINED: clearing event grab and showing mouse cursor')
                    try:
                        pygame.event.set_grab(False)
                    except Exception:
                        logger.debug('Could not clear event grab on focus gained')
                    try:
                        pygame.mouse.set_visible(True)
                    except Exception:
                        logger.debug('Could not set mouse visible on focus gained')
                    try:
                        self._set_message('Window focused', 1.5)
                    except Exception:
                        pass
                    continue
                if event.type == getattr(pygame, 'WINDOWFOCUSLOST', None):
                    logger.debug('WINDOWFOCUSLOST: pausing input interactions')
                    try:
                        self._set_message('Window lost focus', 1.5)
                    except Exception:
                        pass
                    continue

                # Global KEYDOWN diagnostics
                if event.type == pygame.KEYDOWN:
                    try:
                        # Track last key event for debug overlay
                        self._last_key_event = {'key': event.key, 'mods': pygame.key.get_mods(), 'time': time.time()}
                    except Exception:
                        pass
                    logger.debug('KEYDOWN key=%s mods=%s', event.key, pygame.key.get_mods())
                    # Ctrl+O to force hide any overlays (debug quick-fix when overlay blocks interaction)
                    if event.key == pygame.K_o and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                        try:
                            if getattr(self, 'preview_overlay_visible', False) or getattr(self, 'show_solver_comparison_overlay', False):
                                self.preview_overlay_visible = False
                                self.show_solver_comparison_overlay = False
                                self.path_preview_dialog = None
                                self._show_toast('Overlays hidden (Ctrl+O)', 2.0, 'success')
                                self._set_message('Overlays hidden', 2.0)
                            else:
                                self._show_toast('No overlays active', 1.5, 'info')
                        except Exception:
                            logger.exception('Failed to toggle overlays')
                        continue

                # Developer debug overlay toggle
                if event.type == pygame.KEYDOWN and event.key == pygame.K_F12:
                    self.debug_overlay_enabled = not getattr(self, 'debug_overlay_enabled', False)
                    if self.debug_overlay_enabled:
                        self._set_message('Debug overlay ON (F12 to toggle)')
                    else:
                        self._set_message('Debug overlay OFF')
                    continue

                # Debug key: F forces clearing input grab / shows cursor (helpful when clicks are ignored)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                    try:
                        pygame.event.set_grab(False)
                    except Exception:
                        logger.debug('Failed to clear event grab via F key')
                    try:
                        pygame.mouse.set_visible(True)
                    except Exception:
                        logger.debug('Failed to set mouse visible via F key')
                    try:
                        self._show_toast('Forced focus/ungrab (F)', 2.0, 'info')
                        self._set_message('Forced focus/ungrab (F)')
                    except Exception:
                        pass
                    continue
                # Shift-F12 toggles control-panel layout debug overlay (shows widget bounds & metrics)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_F12 and (pygame.key.get_mods() & pygame.KMOD_SHIFT):
                    self.debug_control_panel = not getattr(self, 'debug_control_panel', False)
                    if self.debug_control_panel:
                        self._set_message('Control panel debug ON (Shift+F12)')
                    else:
                        self._set_message('Control panel debug OFF')
                    continue
                # Page Up / Page Down to scroll control panel when visible and hovered
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_PAGEUP, pygame.K_PAGEDOWN):
                    if self.control_panel_enabled and getattr(self, 'control_panel_can_scroll', False) and getattr(self, 'control_panel_rect', None) and self.control_panel_rect.collidepoint(pygame.mouse.get_pos()) and not self.control_panel_collapsed:
                        # Page amount: visible content height (excluding header area)
                        page_amount = max(1, self.control_panel_rect.height - 32)
                        if event.key == pygame.K_PAGEUP:
                            self.control_panel_scroll = max(0, int(self.control_panel_scroll - page_amount))
                        else:
                            self.control_panel_scroll = min(getattr(self, 'control_panel_scroll_max', 0), int(self.control_panel_scroll + page_amount))
                        # Stop any momentum when keyboard-driven
                        self.control_panel_scroll_velocity = 0.0
                        self.control_panel_ignore_click_until = time.time() + 0.12
                        continue
                if event.type == pygame.KEYDOWN and event.key == pygame.K_F11 and (pygame.key.get_mods() & pygame.KMOD_SHIFT):
                    # Shift-F11 clears debug click log
                    self.debug_click_log = []
                    self._set_message('Debug log cleared')
                    continue
                # Handle path preview dialog input first (if active)
                # If a non-modal overlay is active, handle its quick interactions (start/dismiss)
                if getattr(self, 'preview_overlay_visible', False) and (self.path_preview_dialog or getattr(self, 'auto_path', None)):
                    # Keyboard shortcuts for non-modal overlay
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            # Dismiss overlay (keep planned path stored but hide overlay)
                            self.preview_overlay_visible = False
                            self.path_preview_dialog = None
                            self.message = "Path preview dismissed"
                            continue
                        if event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                            # Start auto-solve from overlay
                            self._execute_auto_solve_from_preview()
                            continue
                    # Mouse click on sidebar buttons handled further down when button rects exist

                # Quick key: T toggles topology overlay
                if event.type == pygame.KEYDOWN and event.key == pygame.K_t:
                    self.show_topology = not getattr(self, 'show_topology', False)
                    # Reflect the panel checkbox if present
                    for w in (self.widget_manager.widgets if self.widget_manager else []):
                        if isinstance(w, CheckboxWidget) and getattr(w, 'flag_name', '') == 'show_topology':
                            w.checked = self.show_topology
                    if self.show_topology:
                        # Notify and warn if no graph
                        cur = self.maps[self.current_map_idx]
                        if not hasattr(cur, 'graph') or not cur.graph:
                            self._set_message('Topology not available for this map', 3.0)
                        else:
                            self._set_message('Topology overlay: ON', 2.0)
                    else:
                        self._set_message('Topology overlay: OFF', 1.2)
                    continue
                if self.path_preview_mode and self.path_preview_dialog:
                    result = self.path_preview_dialog.handle_input(event)
                    if result == 'start':
                        # User confirmed - start auto-solve
                        self._execute_auto_solve_from_preview()
                        continue
                    elif result == 'cancel':
                        # User cancelled - switch to non-modal overlay (keep planned path visible)
                        self.path_preview_mode = False
                        # Keep the dialog instance for overlay rendering if available
                        self.preview_overlay_visible = True
                        self.message = "Path preview closed; overlay visible in sidebar/map (Enter to start or Esc to dismiss)"
                        continue

                # Dismiss solver comparison overlay with Esc
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE and getattr(self, 'show_solver_comparison_overlay', False):
                    self.show_solver_comparison_overlay = False
                    self._set_message('Solver comparison closed', 1.2)
                    continue
                
                # If non-modal overlay visible and clicked in sidebar buttons, handle them
                if getattr(self, 'preview_overlay_visible', False) and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mouse_pos = event.pos
                    if getattr(self, 'sidebar_start_button_rect', None) and self.sidebar_start_button_rect.collidepoint(mouse_pos):
                        self._execute_auto_solve_from_preview()
                        continue
                    if getattr(self, 'sidebar_dismiss_button_rect', None) and self.sidebar_dismiss_button_rect.collidepoint(mouse_pos):
                        self.preview_overlay_visible = False
                        self.path_preview_dialog = None
                        self.message = "Path preview dismissed"
                        continue

                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.VIDEORESIZE:
                    # Handle window resize
                    self.screen_w = max(event.w, self.MIN_WIDTH)
                    self.screen_h = max(event.h, self.MIN_HEIGHT)
                    if not self.fullscreen:
                        # Use safe wrapper to avoid producing a zero-sized or invalid surface
                        screen = self._safe_set_mode((self.screen_w, self.screen_h), pygame.RESIZABLE)
                        if not screen:
                            logger.warning('VIDEORESIZE: _safe_set_mode failed; attempting display reinit')
                            # Attempt a display reinit when set_mode fails
                            try:
                                self._attempt_display_reinit()
                            except Exception:
                                logger.exception('VIDEORESIZE: display reinit failed')
                        else:
                            self.screen = screen
                            try:
                                self.screen_w, self.screen_h = self.screen.get_size()
                            except Exception:
                                pass
                        # Refresh assets/layout and force an immediate present
                        try:
                            self._load_assets()
                            self._render()
                            try:
                                pygame.display.flip()
                            except Exception:
                                logger.exception('Flip failed after VIDEORESIZE')
                        except Exception:
                            logger.exception('Failed to refresh UI after VIDEORESIZE')
                    # Update control panel widget positions
                    if self.control_panel_enabled:
                        self._update_control_panel_positions()
                
                elif event.type == pygame.MOUSEWHEEL:
                    mouse_pos = pygame.mouse.get_pos()
                    # If mouse is over control panel and scrolling is enabled, apply momentum to panel
                    panel_rect = getattr(self, 'control_panel_rect', None)
                    padding = getattr(self, 'debug_panel_click_padding', 0) if getattr(self, 'debug_control_panel', False) else 0
                    panel_hit_rect = (pygame.Rect(panel_rect.x - padding, panel_rect.y, panel_rect.width + padding, panel_rect.height) if panel_rect and padding else panel_rect)
                    if self.control_panel_enabled and getattr(self, 'control_panel_can_scroll', False) and panel_hit_rect and panel_hit_rect.collidepoint(mouse_pos) and not self.control_panel_collapsed:
                        # Use wheel to add velocity (pixels per second)
                        wheel_power = getattr(self, 'control_panel_scroll_step', 20) * 12
                        # Negative event.y means scroll down? We want positive y to scroll up (decrease coord)
                        self.control_panel_scroll_velocity += -event.y * wheel_power
                        # Clamp velocity to reasonable bounds
                        max_v = 2000
                        self.control_panel_scroll_velocity = max(-max_v, min(max_v, self.control_panel_scroll_velocity))
                        # Ignore immediate clicks while momentum is active
                        self.control_panel_ignore_click_until = time.time() + 0.12
                    else:
                        # Zoom with mouse wheel when not over panel
                        # Only perform mouse-centered zoom when the mouse is over the main map area
                        sidebar_x = self.screen_w - self.SIDEBAR_WIDTH
                        if mouse_pos[0] < sidebar_x:
                            self._change_zoom(event.y, center=mouse_pos)
                        else:
                            # Falling back to center zoom if wheel over sidebar
                            self._change_zoom(event.y)
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = getattr(event, 'pos', pygame.mouse.get_pos())
                    # Diagnostic: include focus/grab/buttons to help when clicks are ignored
                    try:
                        focused = pygame.mouse.get_focused()
                        grabbed = pygame.event.get_grab()
                        pressed = pygame.mouse.get_pressed()
                    except Exception:
                        focused = False
                        grabbed = False
                        pressed = None
                    logger.debug('MOUSEBUTTONDOWN at %s (button=%s) fullscreen=%s focused=%s grabbed=%s pressed=%s', mouse_pos, getattr(event,'button',None), self.fullscreen, focused, grabbed, pressed)

                    # Input diagnostic: record overlay/panel state and where click occurred
                    try:
                        in_sidebar = mouse_pos[0] >= (self.screen_w - self.SIDEBAR_WIDTH)
                        in_control_panel = bool(getattr(self, 'control_panel_rect', None) and self.control_panel_rect.collidepoint(mouse_pos))
                        sidebar_x = self.screen_w - self.SIDEBAR_WIDTH
                        diag = {
                            'preview_overlay_visible': getattr(self, 'preview_overlay_visible', False),
                            'preview_modal_enabled': getattr(self, 'preview_modal_enabled', False),
                            'show_solver_comparison_overlay': getattr(self, 'show_solver_comparison_overlay', False),
                            'control_panel_active': getattr(self, 'control_panel_enabled', False),
                            'in_sidebar': in_sidebar,
                            'in_control_panel': in_control_panel,
                            'control_panel_ignore_until': getattr(self, 'control_panel_ignore_click_until', 0.0),
                            # Extra diagnostics to detect accidental panel cover / mis-positioning
                            'control_panel_rect': (None if getattr(self, 'control_panel_rect', None) is None else tuple(self.control_panel_rect)),
                            'control_panel_collapsed': getattr(self, 'control_panel_collapsed', False),
                            'sidebar_x': sidebar_x,
                            'mouse_pos': mouse_pos,
                        }
                        if DEBUG_INPUT_ACTIVE:
                            logger.info('INPUT_DIAG: %s', diag)
                        else:
                            logger.debug('INPUT_DIAG: %s', diag)

                        # If debug injection enabled, forcibly clear overlays/ignore flags so click can pass through (useful for repro)
                        if DEBUG_INPUT_ACTIVE:
                            logger.info('INPUT_DIAG: KLTN_DEBUG_INPUT active - clearing overlays and ignore flags for this click')
                            try:
                                self.preview_overlay_visible = False
                                self.show_solver_comparison_overlay = False
                                self.path_preview_dialog = None
                                self.control_panel_ignore_click_until = 0.0
                                self.control_panel_scroll_dragging = False
                                self._show_toast('Debug: overlays/ignore cleared', 1.2, 'info')
                            except Exception:
                                logger.exception('INPUT_DIAG: failed to clear debug state')
                    except Exception:
                        logger.exception('INPUT_DIAG: failure while computing diagnostics')

                    # If preview overlay is visible but not modal, allow clicking the main map area to dismiss it
                    try:
                        sidebar_x = self.screen_w - self.SIDEBAR_WIDTH
                        if getattr(self, 'preview_overlay_visible', False) and not getattr(self, 'preview_modal_enabled', False) and mouse_pos[0] < sidebar_x:
                            logger.info('Click on map detected while non-modal preview overlay active: dismissing overlay')
                            try:
                                self.preview_overlay_visible = False
                                self.path_preview_dialog = None
                                self._show_toast('Preview dismissed (click)', 1.5, 'info')
                                self._set_message('Preview dismissed', 1.5)
                            except Exception:
                                pass
                            # Do not consume this event; allow subsequent checks to process the click
                        
                    except Exception:
                        logger.exception('Error while checking/dismissing non-modal preview overlay')

                    # Track last mouse event for overlay/debug
                    try:
                        self._last_mouse_event = {'type': 'down', 'pos': mouse_pos, 'button': getattr(event,'button',None), 'time': time.time()}
                    except Exception:
                        self._last_mouse_event = None

                    # If window not focused, attempt to force focus (Windows)
                    if not focused:
                        logger.info('Window does not have input focus; attempting to force focus')
                        try:
                            self._force_focus()
                        except Exception:
                            logger.exception('Force focus attempt failed')
                        continue

                    # Record click to debug log
                    if getattr(self, 'debug_click_log', None) is not None:
                        self.debug_click_log.insert(0, (mouse_pos, time.time()))
                        # Keep bounded history
                        if len(self.debug_click_log) > 50:
                            self.debug_click_log.pop()
                    
                    # Handle collapse button click first (animated)
                    if self.control_panel_enabled and self.collapse_button_rect and self.collapse_button_rect.collidepoint(mouse_pos):
                        # Ignore input if animation already running
                        if not getattr(self, 'control_panel_animating', False):
                            target_collapsed = not self.control_panel_collapsed
                            self._start_toggle_panel_animation(target_collapsed)
                        continue
                    
                    # Check if starting to drag panel (click on title bar area)
                    if self.control_panel_enabled and self.control_panel_rect and not self.control_panel_collapsed:
                        # Check if clicking on scrollbar thumb to start drag
                        if event.button == 1 and getattr(self, 'control_panel_scroll_thumb_rect', None) and self.control_panel_scroll_thumb_rect.collidepoint(mouse_pos):
                            self.control_panel_scroll_dragging = True
                            self.control_panel_scroll_drag_offset = mouse_pos[1] - self.control_panel_scroll_thumb_rect.y
                            continue
                        # Clicking on track -> page to that location
                        if event.button == 1 and getattr(self, 'control_panel_scroll_track_rect', None) and self.control_panel_scroll_track_rect.collidepoint(mouse_pos):
                            tr = self.control_panel_scroll_track_rect
                            rel = mouse_pos[1] - tr.y
                            max_move = tr.height - getattr(self, 'control_panel_scroll_thumb_rect', pygame.Rect(0,0,0,20)).height
                            ratio = max(0.0, min(1.0, rel / tr.height))
                            self.control_panel_scroll = int(ratio * getattr(self, 'control_panel_scroll_max', 0))
                            continue

                        title_bar_height = 45
                        title_bar_rect = pygame.Rect(
                            self.control_panel_rect.x,
                            self.control_panel_rect.y,
                            self.control_panel_rect.width,
                            title_bar_height
                        )
                        if title_bar_rect.collidepoint(mouse_pos) and not self.collapse_button_rect.collidepoint(mouse_pos):
                            self.dragging_panel = True
                            self.drag_panel_offset = (mouse_pos[0] - self.control_panel_rect.x, mouse_pos[1] - self.control_panel_rect.y)
                            continue
                        
                        # Check if starting to resize panel (near edges)
                        edge_threshold = 8
                        mx, my = mouse_pos
                        rect = self.control_panel_rect
                        
                        if abs(mx - rect.left) < edge_threshold and rect.top <= my <= rect.bottom:
                            self.resizing_panel = True
                            self.resize_edge = 'left'
                            continue
                        elif abs(mx - rect.right) < edge_threshold and rect.top <= my <= rect.bottom:
                            self.resizing_panel = True
                            self.resize_edge = 'right'
                            continue
                        elif abs(my - rect.bottom) < edge_threshold and rect.left <= mx <= rect.right:
                            self.resizing_panel = True
                            self.resize_edge = 'bottom'
                            continue
                    
                    # Handle control panel clicks
                    if self.control_panel_enabled and self._handle_control_panel_click(mouse_pos, event.button, 'down'):
                        continue  # Control panel handled the click
                    
                    if event.button == 1:  # Left click - check minimap and start map drag if on map
                        if self._handle_minimap_click(mouse_pos):
                            pass  # Minimap click handled
                        else:
                            # Start map drag with left button when clicking on the main map area (not on sidebar or panel)
                            sidebar_x = self.screen_w - self.SIDEBAR_WIDTH
                            if mouse_pos[0] < sidebar_x and not (self.control_panel_enabled and getattr(self, 'control_panel_rect', None) and self.control_panel_rect.collidepoint(mouse_pos)):
                                self.dragging = True
                                self.dragging_button = 1
                                self.drag_start = event.pos
                    elif event.button == 2:  # Middle mouse
                        self.dragging = True
                        self.dragging_button = 2
                        self.drag_start = event.pos
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    mouse_pos = getattr(event, 'pos', pygame.mouse.get_pos())
                    try:
                        focused = pygame.mouse.get_focused()
                        grabbed = pygame.event.get_grab()
                        pressed = pygame.mouse.get_pressed()
                    except Exception:
                        focused = False
                        grabbed = False
                        pressed = None
                    logger.debug('MOUSEBUTTONUP at %s (button=%s) focused=%s grabbed=%s pressed=%s', mouse_pos, getattr(event,'button',None), focused, grabbed, pressed)

                    # Track last mouse event
                    try:
                        self._last_mouse_event = {'type': 'up', 'pos': mouse_pos, 'button': getattr(event,'button',None), 'time': time.time()}
                    except Exception:
                        pass

                    # Stop dragging/resizing panel
                    if self.dragging_panel:
                        self.dragging_panel = False
                    if self.resizing_panel:
                        self.resizing_panel = False
                        self.resize_edge = None
                    # Stop scrollbar drag if active
                    if getattr(self, 'control_panel_scroll_dragging', False):
                        self.control_panel_scroll_dragging = False
                    # Handle control panel clicks
                    if self.control_panel_enabled and self._handle_control_panel_click(mouse_pos, event.button, 'up'):
                        continue
                    
                    if event.button == 2 or (hasattr(self, 'dragging_button') and event.button == getattr(self, 'dragging_button')):
                        self.dragging = False
                        self.dragging_button = None
                
                elif event.type == pygame.MOUSEMOTION:
                    mouse_pos = event.pos
                    try:
                        focused = pygame.mouse.get_focused()
                        grabbed = pygame.event.get_grab()
                        buttons = pygame.mouse.get_pressed()
                    except Exception:
                        focused = False
                        grabbed = False
                        buttons = None
                    # Throttle very high-frequency mouse motion logging and summarize suppressed events
                    now = time.time()
                    last_log = getattr(self, '_last_mouse_log_time', 0.0)
                    suppressed = getattr(self, '_mouse_motion_suppressed', 0)
                    throttle = 0.05  # seconds
                    if (now - last_log) > throttle:
                        # Log current event and report how many events were suppressed since last log
                        logger.debug('MOUSEMOTION at %s rel=%s buttons=%s focused=%s grabbed=%s suppressed=%d', mouse_pos, getattr(event,'rel',None), buttons, focused, grabbed, suppressed)
                        self._last_mouse_log_time = now
                        self._mouse_motion_suppressed = 0
                        self._last_mouse_summary_time = getattr(self, '_last_mouse_summary_time', 0.0)
                    else:
                        # Increment suppressed counter; do not call logger to avoid flooding
                        self._mouse_motion_suppressed = suppressed + 1
                        # Occasionally emit a short summary to indicate continued motion (once per second)
                        last_summary = getattr(self, '_last_mouse_summary_time', 0.0)
                        if (now - last_summary) > 1.0 and self._mouse_motion_suppressed % 20 == 0:
                            logger.debug('MOUSEMOTION: still receiving motion events; suppressed=%d so far', self._mouse_motion_suppressed)
                            self._last_mouse_summary_time = now

                    # Track last mouse motion event (kept regardless of logging)
                    try:
                        self._last_mouse_event = {'type': 'motion', 'pos': mouse_pos, 'rel': getattr(event,'rel',None), 'time': time.time()}
                    except Exception:
                        pass
                    
                    # Handle panel dragging
                    if self.dragging_panel:
                        self.control_panel_x = mouse_pos[0] - self.drag_panel_offset[0]
                        self.control_panel_y = mouse_pos[1] - self.drag_panel_offset[1]
                        # Clamp to screen bounds
                        self.control_panel_x = max(0, min(self.control_panel_x, self.screen_w - self.control_panel_width))
                        self.control_panel_y = max(0, min(self.control_panel_y, self.screen_h - 100))
                        # Update widget positions to follow panel
                        self._reposition_widgets(self.control_panel_x, self.control_panel_y)
                    
                    # Handle panel resizing
                    elif self.resizing_panel and self.control_panel_rect:
                        if self.resize_edge == 'left':
                            old_right = self.control_panel_rect.right
                            new_x = mouse_pos[0]
                            new_width = old_right - new_x
                            if self.min_panel_width <= new_width <= self.max_panel_width:
                                self.control_panel_width = new_width
                                self.control_panel_x = new_x
                        elif self.resize_edge == 'right':
                            new_width = mouse_pos[0] - self.control_panel_rect.x
                            if self.min_panel_width <= new_width <= self.max_panel_width:
                                self.control_panel_width = new_width
                        elif self.resize_edge == 'bottom':
                            new_height = mouse_pos[1] - self.control_panel_rect.y
                            if self.min_panel_height <= new_height <= self.screen_h - self.control_panel_rect.y - 20:
                                pass  # Height is auto-calculated, just update visual feedback
                    
                    # Handle scrollbar thumb dragging
                    elif getattr(self, 'control_panel_scroll_dragging', False) and getattr(self, 'control_panel_can_scroll', False) and getattr(self, 'control_panel_scroll_track_rect', None):
                        track_rect = self.control_panel_scroll_track_rect
                        thumb_rect = getattr(self, 'control_panel_scroll_thumb_rect', None)
                        if thumb_rect is None:
                            continue
                        # Compute local mouse position inside track
                        rel_y = mouse_pos[1] - track_rect.y
                        max_move = track_rect.height - thumb_rect.height
                        new_thumb_top = max(0, min(rel_y - getattr(self, 'control_panel_scroll_drag_offset', 0), max_move))
                        if max_move > 0:
                            ratio = new_thumb_top / max_move
                            self.control_panel_scroll = int(ratio * getattr(self, 'control_panel_scroll_max', 0))
                            # Clamp
                            self.control_panel_scroll = max(0, min(self.control_panel_scroll, getattr(self, 'control_panel_scroll_max', 0)))
                    
                    # Handle map dragging
                    elif self.dragging:
                        dx = self.drag_start[0] - event.pos[0]
                        dy = self.drag_start[1] - event.pos[1]
                        self.view_offset_x += dx
                        self.view_offset_y += dy
                        self.drag_start = event.pos
                        self._clamp_view_offset()
                    # Update cursor for resize edges (when not dragging)
                    elif self.control_panel_enabled and self.control_panel_rect and not self.control_panel_collapsed:
                        edge_threshold = 8
                        mx, my = mouse_pos
                        rect = self.control_panel_rect
                        
                        if abs(mx - rect.left) < edge_threshold and rect.top <= my <= rect.bottom:
                            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_SIZEWE)
                        elif abs(mx - rect.right) < edge_threshold and rect.top <= my <= rect.bottom:
                            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_SIZEWE)
                        elif abs(my - rect.bottom) < edge_threshold and rect.left <= mx <= rect.right:
                            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_SIZENS)
                        else:
                            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)

                elif event.type == pygame.KEYUP:
                    # Ensure we stop continuous movement when keys are released
                    try:
                        if event.key in getattr(self, 'keys_held', {}):
                            self.keys_held[event.key] = False
                    except Exception:
                        logger.debug('Failed to handle KEYUP for %r', getattr(event, 'key', None))

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.fullscreen:
                            self._toggle_fullscreen()
                        else:
                            running = False
                    
                    elif event.key == pygame.K_F11:
                        self._toggle_fullscreen()
                    
                    elif event.key == pygame.K_h:
                        # Toggle heatmap overlay (H key)
                        if not self.show_help:  # Don't toggle if help shown
                            self.show_heatmap = not self.show_heatmap
                            self.feature_flags['show_heatmap'] = self.show_heatmap
                            if self.renderer:
                                self.renderer.show_heatmap = self.show_heatmap
                            # Update checkbox widget if available
                            if self.widget_manager:
                                for widget in self.widget_manager.widgets:
                                    if isinstance(widget, CheckboxWidget) and hasattr(widget, 'flag_name') and widget.flag_name == 'show_heatmap':
                                        widget.checked = self.show_heatmap
                            self.message = f"Heatmap: {'ON' if self.show_heatmap else 'OFF'}"
                    
                    elif event.key == pygame.K_F1:
                        self.show_help = not self.show_help
                    
                    elif event.key == pygame.K_TAB:
                        # Toggle control panel with Tab key
                        if self.control_panel_enabled:
                            # Animate toggle for Tab key as well
                            if not getattr(self, 'control_panel_animating', False):
                                target_collapsed = not self.control_panel_collapsed
                                self._start_toggle_panel_animation(target_collapsed)

                    elif event.key == pygame.K_F7:
                        # Diagnostic hotkey: dump widget hit-test state at current mouse
                        try:
                            pos = pygame.mouse.get_pos()
                            logger.info('DIAG DUMP (F7): mouse_pos=%s control_panel_rect=%s scroll=%s', pos, getattr(self,'control_panel_rect',None), getattr(self,'control_panel_scroll',0))
                            try:
                                self._dump_control_panel_widget_state(pos)
                            except Exception:
                                logger.exception('F7: _dump_control_panel_widget_state failed')
                        except Exception:
                            logger.exception('F7 diagnostic failed')

                    elif event.key == pygame.K_F8:
                        # Toggle debug overlay for control panel and hit-padding
                        try:
                            self.debug_control_panel = not getattr(self, 'debug_control_panel', False)
                            self.debug_panel_click_padding = int(os.environ.get('KLTN_DEBUG_PANEL_PADDING', '40')) if self.debug_control_panel else 0
                            self._show_toast(f"Debug control panel {'ON' if self.debug_control_panel else 'OFF'}", 1.6, 'info')
                            logger.info('Toggled debug_control_panel=%s padding=%s', self.debug_control_panel, self.debug_panel_click_padding)
                        except Exception:
                            logger.exception('Failed to toggle debug control panel')
                    
                    elif event.key == pygame.K_m:
                        # Toggle minimap
                        self.show_minimap = not self.show_minimap
                        self.feature_flags['show_minimap'] = self.show_minimap
                        # Update checkbox widget if available
                        if self.widget_manager:
                            for widget in self.widget_manager.widgets:
                                if isinstance(widget, CheckboxWidget) and hasattr(widget, 'flag_name') and widget.flag_name == 'show_minimap':
                                    widget.checked = self.show_minimap
                        self.message = f"Minimap: {'ON' if self.show_minimap else 'OFF'}"
                    
                    elif event.key == pygame.K_RIGHTBRACKET or event.key == pygame.K_PERIOD:
                        # Increase speed
                        self.speed_index = min(len(self.speed_levels) - 1, self.speed_index + 1)
                        self.speed_multiplier = self.speed_levels[self.speed_index]
                        self.message = f"Speed: {self.speed_multiplier}x"
                    
                    elif event.key == pygame.K_LEFTBRACKET or event.key == pygame.K_COMMA:
                        # Decrease speed
                        self.speed_index = max(0, self.speed_index - 1)
                        self.speed_multiplier = self.speed_levels[self.speed_index]
                        self.message = f"Speed: {self.speed_multiplier}x"
                    
                    elif event.key == pygame.K_SPACE:
                        self._start_auto_solve()
                    
                    elif event.key == pygame.K_r:
                        self._load_current_map()
                        self._center_view()
                        # Clear effects
                        if self.effects:
                            self.effects.clear()
                        # Reset step count
                        self.step_count = 0
                        self.message = "Map Reset"
                    
                    elif event.key == pygame.K_n:
                        self._next_map()
                    
                    elif event.key == pygame.K_p:
                        self._prev_map()

                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        self._change_zoom(1)

                    elif event.key == pygame.K_MINUS:
                        self._change_zoom(-1)

                    elif event.key == pygame.K_0:
                        # Reset zoom to default
                        self.zoom_idx = self.DEFAULT_ZOOM_IDX
                        self.TILE_SIZE = self.ZOOM_LEVELS[self.zoom_idx]
                        self._load_assets()
                        self._center_view()
                        self.message = "Zoom reset to default"

                    elif event.key == pygame.K_f:
                        # Auto-fit zoom to show entire map
                        self._auto_fit_zoom()
                        self.message = f"Auto-fit: {self.TILE_SIZE}px"

                    elif event.key == pygame.K_c:
                        # Center view on player
                        self._center_on_player()

                    elif event.key == pygame.K_l:
                        ok = self.load_visual_map(os.path.join(os.getcwd(), 'screenshot.png'))
                        if not ok:
                            self.message = "Failed to load ./screenshot.png"

                    # Track key holds for continuous movement
                    elif event.key in self.keys_held and not self.auto_mode:
                        self.keys_held[event.key] = True
                        self.move_timer = 0.0  # Reset timer for immediate first move
                        
                    elif not self.auto_mode:
                        # Manual movement - check for diagonal combos first
                        keys = pygame.key.get_pressed()
                        action = None

                        # Check diagonal combinations (two arrow keys pressed)
                        if keys[pygame.K_UP] and keys[pygame.K_LEFT]:
                            action = Action.UP_LEFT
                        elif keys[pygame.K_UP] and keys[pygame.K_RIGHT]:
                            action = Action.UP_RIGHT
                        elif keys[pygame.K_DOWN] and keys[pygame.K_LEFT]:
                            action = Action.DOWN_LEFT
                        elif keys[pygame.K_DOWN] and keys[pygame.K_RIGHT]:
                            action = Action.DOWN_RIGHT
                        # Cardinal directions (single key)
                        elif keys[pygame.K_UP]:
                            action = Action.UP
                        elif keys[pygame.K_DOWN]:
                            action = Action.DOWN
                        elif keys[pygame.K_LEFT]:
                            action = Action.LEFT
                        elif keys[pygame.K_RIGHT]:
                            action = Action.RIGHT

                        if action is not None:
                            self._manual_step(action)
                            self._center_on_player()

            # Auto-solve stepping with timer-based animation
            # DEBUG: Log auto_mode status periodically (every 60 frames)
            if frame_count % 60 == 0:
                logger.debug('DEBUG_ANIM: frame=%d auto_mode=%s env.done=%s auto_path_len=%d auto_step_idx=%s',
                             frame_count,
                             getattr(self, 'auto_mode', None),
                             getattr(self.env, 'done', None) if self.env else None,
                             len(getattr(self, 'auto_path', []) or []),
                             getattr(self, 'auto_step_idx', None))
            
            # BUGFIX: The hotfix below was causing animation failure by resetting env mid-animation
            # The env.reset() call in _execute_auto_solve() already handles initial reset correctly
            # This hotfix was resetting player position back to start, breaking the animation
            # REMOVED: Hotfix that called env.reset() when auto_mode=True and env.done=True
            
            if self.auto_mode and not self.env.done:
                # Accumulate time and step when interval reached
                self.auto_step_timer += self.delta_time
                # Calculate effective interval (faster with higher speed multiplier)
                effective_interval = self.auto_step_interval / max(0.1, self.speed_multiplier)
                
                # Timer diagnostics every 30 frames
                if frame_count % 30 == 0:
                    logger.debug('TIMER_DIAG: frame=%d timer=%.3f threshold=%.3f delta=%.3f step=%d/%d',
                                frame_count, self.auto_step_timer, effective_interval, 
                                self.delta_time, self.auto_step_idx, len(self.auto_path) if self.auto_path else 0)
                
                if self.auto_step_timer >= effective_interval:
                    self.auto_step_timer = 0.0
                    logger.debug('DEBUG_ANIM: Calling _auto_step() at frame=%d, step_idx=%d/%d',
                                 frame_count, self.auto_step_idx, len(self.auto_path) if self.auto_path else 0)
                    self._auto_step()
                    self._center_on_player()
            
            # Update widget manager with mouse position
            if self.widget_manager:
                mouse_pos = pygame.mouse.get_pos()
                self.widget_manager.update(mouse_pos, self.delta_time)
            
            # Handle continuous movement (hold key to move) with diagonal support
            if not self.auto_mode and any(self.keys_held.values()):
                self.move_timer += self.delta_time
                if self.move_timer >= self.move_delay:
                    self.move_timer = 0.0
                    # Check for diagonal combinations FIRST (two keys held)
                    if self.keys_held[pygame.K_UP] and self.keys_held[pygame.K_LEFT]:
                        self._manual_step(Action.UP_LEFT)
                    elif self.keys_held[pygame.K_UP] and self.keys_held[pygame.K_RIGHT]:
                        self._manual_step(Action.UP_RIGHT)
                    elif self.keys_held[pygame.K_DOWN] and self.keys_held[pygame.K_LEFT]:
                        self._manual_step(Action.DOWN_LEFT)
                    elif self.keys_held[pygame.K_DOWN] and self.keys_held[pygame.K_RIGHT]:
                        self._manual_step(Action.DOWN_RIGHT)
                    # Cardinal directions (single key)
                    elif self.keys_held[pygame.K_UP]:
                        self._manual_step(Action.UP)
                    elif self.keys_held[pygame.K_DOWN]:
                        self._manual_step(Action.DOWN)
                    elif self.keys_held[pygame.K_LEFT]:
                        self._manual_step(Action.LEFT)
                    elif self.keys_held[pygame.K_RIGHT]:
                        self._manual_step(Action.RIGHT)
            
            # Update toast notifications
            self._update_toasts()

            # Periodic heartbeat to confirm main loop alive
            try:
                now = time.time()
                if now - heartbeat_last > heartbeat_interval:
                    heartbeat_last = now
                    logger.debug("GUI heartbeat - frame=%d auto_mode=%s solver_running=%s", frame_count, getattr(self,'auto_mode',False), getattr(self,'solver_running',False))
            except Exception:
                pass

            # Check if algorithm change triggered a pending solver (deferred to avoid blocking event handler)
            if getattr(self, '_pending_solver_trigger', False):
                self._pending_solver_trigger = False
                algorithm_names = ["A*", "BFS", "Dijkstra", "Greedy", "D* Lite",
                                 "DFS/IDDFS", "Bidirectional A*",
                                 "CBS (Balanced)", "CBS (Explorer)", "CBS (Cautious)",
                                 "CBS (Forgetful)", "CBS (Speedrunner)", "CBS (Greedy)"]
                alg_name = algorithm_names[self.algorithm_idx] if self.algorithm_idx < len(algorithm_names) else f"Algorithm {self.algorithm_idx}"
                logger.info('🔄 Processing pending solver trigger: Starting %s solver...', alg_name)
                self._start_auto_solve()

            # Update animated control panel state (if active)
            self._update_control_panel_animation()

            # Update control panel scroll inertia (momentum)
            self._update_control_panel_scroll()

            # If parallel search ran in background, handle result on main thread
            if getattr(self, 'parallel_search_done', False) and getattr(self, 'parallel_search_result', None):
                best = self.parallel_search_result
                # Convert alg index to name
                alg_names = ['A*','BFS','Dijkstra','Greedy']
                name = alg_names[best['alg']] if best['alg'] < len(alg_names) else f"Alg{best['alg']}"
                self._set_message(f"Parallel best: {name} ({best['nodes']} nodes, {best['time_ms']:.0f}ms)")
                self.parallel_search_done = False
                self.parallel_search_result = None
                # Use found path
                _handle_found_path = None
                try:
                    # Reuse the same handling as in _start_auto_solve: set auto_path and show preview
                    self.auto_path = best['path']
                    self.preview_overlay_visible = True
                    logger.debug('Parallel search: setting preview_overlay_visible=True (parallel best path)')
                    self.path_preview_dialog = PathPreviewDialog(path=self.auto_path, env=self.env, solver_result={}, speed_multiplier=self.speed_multiplier)
                    self._set_message('Parallel result ready (sidebar preview)')
                except Exception as e:
                    logger.warning(f"Failed to display parallel search preview: {e}")

            # If a preview process finished, read its output and apply it
            if getattr(self, 'preview_proc', None) and not getattr(self, 'preview_done', False):
                p = getattr(self, 'preview_proc')
                if not p.is_alive():
                    out = getattr(self, 'preview_outfile', None)
                    res = None
                    try:
                        if out and os.path.exists(out):
                            with open(out, 'rb') as f:
                                res = pickle.load(f)
                    except Exception as e:
                        logger.exception('Failed to read preview output: %s', e)
                    finally:
                        try:
                            p.join(timeout=0.1)
                        except Exception:
                            pass
                        try:
                            if out and os.path.exists(out):
                                os.remove(out)
                        except Exception:
                            pass
                        try:
                            gf = getattr(self, 'preview_gridfile', None)
                            if gf and os.path.exists(gf):
                                os.remove(gf)
                        except Exception:
                            pass
                        self.preview_proc = None
                        self.preview_outfile = None
                        self.preview_gridfile = None
                        self.preview_done = True

                    if res:
                        try:
                            if res.get('success') and res.get('path'):
                                self.auto_path = res.get('path')
                                self.preview_overlay_visible = True
                                logger.debug('Preview result: setting preview_overlay_visible=True (preview has path)')
                                try:
                                    # Use (x or {}) pattern to handle None values
                                    solver_result_preview = (res.get('solver_result') or {}) if res else {}
                                    self.path_preview_dialog = PathPreviewDialog(path=self.auto_path, env=self.env, solver_result=solver_result_preview, speed_multiplier=self.speed_multiplier)
                                except Exception:
                                    self.path_preview_dialog = None
                                self._set_message('Preview ready (sidebar)')
                            else:
                                msg = res.get('message') or 'Preview finished with no path'
                                self._set_message(msg)
                        except Exception as e:
                            logger.exception('Failed to apply preview output on main thread: %s', e)
                    else:
                        self._set_message('Preview finished (no output)')
                    self.preview_done = True
                if getattr(self, 'solver_running', False):
                    # small ping in status message to reassure user
                    self.status_message = 'Solving...'
                else:
                    self.status_message = 'Ready'

            # If a solver subprocess (or thread fallback) finished, read its output and apply result on the main thread
            if not getattr(self, 'solver_done', False):
                proc = getattr(self, 'solver_proc', None)
                proc_alive = False
                solver_starting = getattr(self, 'solver_starting', False)
                
                # CRITICAL: Check for solver timeout (30 second default for complex maps)
                solver_timeout = float(os.environ.get('KLTN_SOLVER_TIMEOUT', '30.0'))
                solver_start_time = getattr(self, 'solver_start_time', None)
                timed_out = False
                if solver_start_time and (time.time() - solver_start_time) > solver_timeout:
                    timed_out = True
                    logger.error('SOLVER: TIMEOUT after %.1fs - forcefully terminating', solver_timeout)
                    if proc:
                        try:
                            proc.terminate()
                            proc.join(timeout=0.5)
                        except Exception as e:
                            logger.exception('SOLVER: Failed to terminate timed-out process: %s', e)
                    proc_alive = False
                
                if not timed_out:
                    try:
                        proc_alive = proc.is_alive() if proc else False
                    except Exception as e:
                        logger.exception('SOLVER: proc.is_alive() raised exception: %s', e)
                        proc_alive = False

                # Startup grace: avoid treating proc None as completion while spawn thread is still starting
                startup_grace = float(os.environ.get('KLTN_SOLVER_STARTUP_GRACE', '1.5'))
                solver_age = (time.time() - solver_start_time) if solver_start_time else 0.0
                out = getattr(self, 'solver_outfile', None)
                out_exists = os.path.exists(out) if out else False
                if solver_starting and proc is None and not out_exists and not timed_out and solver_age < startup_grace:
                    logger.debug('SOLVER: Waiting for process start (age=%.2fs < %.2fs grace)', solver_age, startup_grace)
                elif proc is None or not proc_alive:
                    # CRITICAL: Wrap ENTIRE completion block in try/finally to guarantee solver_running cleanup
                    try:
                        if proc is not None:
                            logger.info('SOLVER: Subprocess done, proc.is_alive()=False, exitcode=%s', proc.exitcode)
                        else:
                            logger.info('SOLVER: No subprocess handle (thread fallback or spawn failure)')

                        # Try to load the outfile
                        out = getattr(self, 'solver_outfile', None)
                        logger.info('SOLVER: Reading result from %s, exists=%s', out, os.path.exists(out) if out else 'N/A')
                        res = None
                        try:
                            if out and os.path.exists(out):
                                with open(out, 'rb') as f:
                                    res = pickle.load(f)
                                path_len = len(res.get('path', []) or []) if res else 0
                                # CRITICAL FIX: Use (x or {}) pattern to handle None values
                                # res.get('solver_result', {}) returns None if key exists with None value
                                solver_result_safe = (res.get('solver_result') or {}) if res else {}
                                logger.info('SOLVER: Result loaded, path_len=%d, success=%s, keys=%s',
                                            path_len,
                                            res.get('success') if res else None,
                                            solver_result_safe.get('keys_used', 'N/A'))
                            else:
                                logger.warning('SOLVER: Output file missing or path is None: %s', out)
                        except Exception as e:
                            logger.exception('SOLVER: Failed to read solver output: %s', e)

                        # Apply results on main thread
                        if res:
                            try:
                                if res.get('success') and res.get('path'):
                                    self.auto_path = res.get('path')
                                    # Use (x or {}) pattern to handle None values
                                    solver_result = (res.get('solver_result') or {}) if res else {}
                                    
                                    # CRITICAL: Verify path doesn't go through water
                                    water_violations = []
                                    if self.auto_path and len(self.auto_path) > 0:
                                        grid = self.env.grid
                                        for i, (r, c) in enumerate(self.auto_path):
                                            tile_id = int(grid[r, c])
                                            if tile_id == 40:  # ELEMENT (water)
                                                water_violations.append((i, r, c, tile_id))
                                    
                                    if water_violations:
                                        print(f"\n{'='*60}")
                                        print(f"ERROR: PATH GOES THROUGH WATER!")
                                        print(f"Found {len(water_violations)} water tiles in path:")
                                        for step, r, c, tid in water_violations[:5]:
                                            print(f"  Step {step}: position ({r}, {c}) = tile ID {tid} (WATER)")
                                        print(f"{'='*60}\n")
                                        logger.error(f"PATH ERROR: {len(water_violations)} water tiles in path!")
                                    else:
                                        print(f"\n{'='*60}")
                                        print(f"PATH VERIFIED: No water tiles")
                                        print(f"Path length: {len(self.auto_path)} steps")
                                        print(f"{'='*60}\n")
                                    
                                    # Print path sample to verify water avoidance
                                    print(f"\n{'='*60}")
                                    print(f"PATH LOADED: {len(self.auto_path)} steps")
                                    if len(self.auto_path) > 10:
                                        print(f"First 10 steps: {self.auto_path[:10]}")
                                    print(f"{'='*60}\n")
                                    
                                    logger.info('SOLVER: Path applied! auto_path len=%d, first=%s, last=%s',
                                                len(self.auto_path),
                                                self.auto_path[0] if self.auto_path else None,
                                                self.auto_path[-1] if self.auto_path else None)
                                    
                                    # Auto-start mode: skip preview and start animation immediately
                                    if getattr(self, 'auto_start_solver', False):
                                        logger.info('SOLVER: auto_start_solver=True, starting animation immediately')
                                        self._execute_auto_solve(self.auto_path, solver_result, teleports=0)
                                        self._set_message(f'Auto-solve started! Path: {len(self.auto_path)} steps')
                                        logger.info('SOLVER: Animation started, auto_mode=%s, auto_step_idx=%s',
                                                    getattr(self, 'auto_mode', None),
                                                    getattr(self, 'auto_step_idx', None))
                                    else:
                                        # Show preview for user confirmation
                                        logger.info('SOLVER: auto_start_solver=False, showing preview dialog')
                                        self.preview_overlay_visible = True
                                        self.path_preview_dialog = PathPreviewDialog(path=self.auto_path, env=self.env, solver_result=solver_result, speed_multiplier=self.speed_multiplier)
                                        self._set_message('Solver finished (press ENTER to start or ESC to dismiss)')
                                else:
                                    msg = res.get('message') or 'Solver finished with no path'
                                    logger.warning('SOLVER: No valid path in result: %s', msg)
                                    self._set_message(msg)
                            except Exception as e:
                                logger.exception('SOLVER: Failed to apply result on main thread: %s', e)
                                self._set_message('Solver error (see logs)')
                        else:
                            if timed_out:
                                logger.error('SOLVER: No result - subprocess timed out')
                                self._set_message(f'Solver timed out after {int(solver_timeout)}s')
                            else:
                                logger.warning('SOLVER: No result loaded (res is None), subprocess may have crashed')
                                self._set_message('Solver finished (no output)')
                    finally:
                        # CRITICAL: ALWAYS clean up process and files in finally block
                        # This MUST run even if result loading/application crashes
                        logger.info('SOLVER: Entering cleanup finally block')
                        try:
                            if proc:
                                proc.join(timeout=0.1)
                        except Exception as e:
                            logger.exception('SOLVER: proc.join() failed: %s', e)
                        try:
                            out = getattr(self, 'solver_outfile', None)
                            if out and os.path.exists(out):
                                os.remove(out)
                        except Exception as e:
                            logger.exception('SOLVER: Failed to remove output file: %s', e)
                        try:
                            gf = getattr(self, 'solver_gridfile', None)
                            if gf and os.path.exists(gf):
                                os.remove(gf)
                        except Exception as e:
                            logger.exception('SOLVER: Failed to remove grid file: %s', e)
                        
                        # CRITICAL: Clear all solver state atomically using centralized helper
                        self._clear_solver_state(reason="solver completed/failed")

            # Render
            self._render()

            # Present frame to the display (ensure visual updates after resize/fullscreen changes)
            try:
                pygame.display.flip()
            except Exception:
                logger.exception('pygame.display.flip() failed; attempting pygame.display.update() and fallback')
                try:
                    pygame.display.update()
                except Exception:
                    logger.exception('pygame.display.update() also failed')
                    # Try a reinit if flip/update both fail and display seems unhealthy
                    try:
                        if not self._ensure_display_alive():
                            logger.warning('Display not healthy after flip/update; attempted recovery')
                    except Exception:
                        logger.exception('Attempted display recovery after flip/update failures')

            # If watchdog requested a screenshot, perform it on the main thread (thread-safe)
            try:
                # Let a dedicated helper perform the watchdog screenshot save and clear the request
                try:
                    self._handle_watchdog_screenshot()
                except Exception:
                    logger.exception('Error during watchdog screenshot handling')
            except Exception:
                # Be defensive: avoid crashing the main loop due to watchdog handling
                logger.exception('Error handling watchdog screenshot request')

            # Periodic display health check (throttled)
            try:
                now = time.time()
                if now - getattr(self, '_display_check_last', 0.0) >= getattr(self, '_display_check_interval', 1.0):
                    self._display_check_last = now
                    ok = self._ensure_display_alive()
                    if not ok:
                        # If recovery failed, show a persistent message to the user
                        self._set_message('Display recovery attempted; see logs', 6.0)
            except Exception:
                logger.exception('Error during display health check')

            # Increment frame counter and check test-mode limit
            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                logger.debug("Exiting run loop due to max_frames=%r", max_frames)
                running = False

            # Cap framerate - use higher FPS during auto_mode for smoother animations
            self.clock.tick(60 if self.auto_mode else 30)
        
        pygame.quit()

    def _next_map(self):
        """Move to the next map and stop auto-solve if running."""
        try:
            if getattr(self, 'auto_mode', False):
                self._stop_auto('map change (next)')
        except Exception:
            pass
        try:
            self.current_map_idx = (self.current_map_idx + 1) % len(self.maps)
            self._load_current_map()
            self._center_view()
            if self.effects:
                self.effects.clear()
            self.step_count = 0
        except Exception:
            logger.exception("_next_map failed")

    def _prev_map(self):
        """Move to the previous map and stop auto-solve if running."""
        try:
            if getattr(self, 'auto_mode', False):
                self._stop_auto('map change (prev)')
        except Exception:
            pass
        try:
            self.current_map_idx = (self.current_map_idx - 1) % len(self.maps)
            self._load_current_map()
            self._center_view()
            if self.effects:
                self.effects.clear()
            self.step_count = 0
        except Exception:
            logger.exception("_prev_map failed")
    
    def _clamp_view_offset(self):
        """Clamp view offset to valid range.

        When the dungeon/map is smaller than the viewport, allow negative offsets so
        the user can pan the small map freely inside the window (showing empty
        margins) while still preventing arbitrary unrestricted panning.
        """
        if self.env is None:
            return
        map_w = self.env.width * self.TILE_SIZE
        map_h = self.env.height * self.TILE_SIZE
        view_w = self.screen_w - self.SIDEBAR_WIDTH
        view_h = self.screen_h - self.HUD_HEIGHT

        # Allow negative minimum when map is smaller than view so the map can be
        # shifted inside the viewport. When map is larger, behave as before.
        min_offset_x = min(0, map_w - view_w)
        max_offset_x = max(0, map_w - view_w)
        min_offset_y = min(0, map_h - view_h)
        max_offset_y = max(0, map_h - view_h)

        self.view_offset_x = min(max(self.view_offset_x, min_offset_x), max_offset_x)
        self.view_offset_y = min(max(self.view_offset_y, min_offset_y), max_offset_y)
    
    def _center_on_player(self):
        """Center the view on the player position."""
        if self.env is None:
            return
        r, c = self.env.state.position
        player_x = c * self.TILE_SIZE
        player_y = r * self.TILE_SIZE
        
        view_w = self.screen_w - self.SIDEBAR_WIDTH
        view_h = self.screen_h - self.HUD_HEIGHT
        
        self.view_offset_x = player_x - view_w // 2
        self.view_offset_y = player_y - view_h // 2
        self._clamp_view_offset()
    
    def _start_preview_for_current_map(self):
        """Perform light-weight post-map-load work on main thread:

        - Scan/map annotation (fast)
        - Kick off a lightweight preview search (process preferred, fallback to thread)
        - Ensure rendering/screenshot work happens on the main thread (no pygame calls from watcher)
        """
        # If env is finished state for some reason, reload to ensure consistent setup
        if getattr(self, 'env', None) and getattr(self.env, 'done', False):
            try:
                self._load_current_map()
            except Exception:
                logger.exception('Failed to reload current map during preview startup')

        # Scan and mark items (measure duration, should be fast)
        try:
            _scan_start = time.time()
            self._scan_and_mark_items()
            _scan_dur = time.time() - _scan_start
            if _scan_dur > 0.05:
                logger.debug('Item scan took %.3fs', _scan_dur)
        except Exception:
            logger.exception('Item scanning failed')

        # Update status and render once to reflect initial state
        algorithm_names = ["A*", "BFS", "Dijkstra", "Greedy", "D* Lite",
                          "DFS/IDDFS", "Bidirectional A*",
                          "CBS (Balanced)", "CBS (Explorer)", "CBS (Cautious)",
                          "CBS (Forgetful)", "CBS (Speedrunner)", "CBS (Greedy)"]
        solver_name = algorithm_names[self.algorithm_idx] if hasattr(self, 'algorithm_idx') else 'A*'
        self.message = f"Solving ({solver_name})..."
        try:
            self._render()
            pygame.display.flip()
        except Exception:
            # Rendering should only be attempted on main thread; swallow errors here
            logger.debug('Render/flip failed during preview startup (may be uninitialized yet)')

        logger.info('Starting solver: algorithm=%s, solver_running=%s', solver_name, getattr(self,'solver_running',False))

        # Allow disabling automatic preview startup for debugging
        if os.environ.get('KLTN_DISABLE_PREVIEW') == '1':
            logger.info('Automatic preview startup disabled via KLTN_DISABLE_PREVIEW=1')
            return

        # Start a non-blocking preview worker to avoid freezing the UI.
        if getattr(self, 'preview_thread', None) and getattr(self, 'preview_thread').is_alive():
            self._set_message('Preview already running...')
            return

        def _preview_worker():
            """Runs quick solvers off the main thread and stores a preview result for main thread to apply."""
            try:
                current_dungeon = self.maps[self.current_map_idx]
                # Prefer graph solver when available (fast on small graphs)
                if hasattr(current_dungeon, 'graph') and current_dungeon.graph and not getattr(self, 'force_grid_algorithm', False):
                    try:
                        from src.data.zelda_core import DungeonSolver, ValidationMode
                        solver = DungeonSolver()
                        result = solver.solve(current_dungeon, mode=ValidationMode.FULL)
                        if result.get('solvable', False):
                            # Try lightweight grid path derived from graph
                            self.solver_result = result
                            success, path, teleports = self._smart_grid_path()
                            if success:
                                self.preview_result = {'path': path, 'solver_result': result, 'teleports': teleports}
                                return

                            # Try graph-guided teleport path
                            success2, path2, teleports2 = self._graph_guided_path()
                            if success2:
                                self.preview_result = {'path': path2, 'solver_result': result, 'teleports': teleports2}
                                return
                    except Exception:
                        # Log but do not raise - worker should not crash main thread
                        logger.debug('Graph-based quick solve failed in worker', exc_info=True)

                # Otherwise try fast grid-based solver directly
                try:
                    success, path, teleports = self._smart_grid_path()
                    if success:
                        self.preview_result = {'path': path, 'solver_result': {}, 'teleports': teleports}
                        return
                except Exception:
                    logger.debug('Grid quick solve failed in worker', exc_info=True)
            finally:
                # indicate worker finished even if no preview found
                self.preview_thread = None

            # No quick preview found; schedule heavy solver if allowed
            if os.environ.get('KLTN_ALLOW_HEAVY', '1') == '1':
                try:
                    # Launch heavy solver asynchronously
                    self._schedule_solver()
                except Exception:
                    logger.exception('Failed to schedule heavy solver from preview worker')
            else:
                self._set_message('No preview found; heavy solver disabled', 3.0)

        # Start preview using a separate process to avoid GIL starvation on heavy maps.
        # Create the grid file and spawn preview process *asynchronously* to avoid blocking the main thread
        def _spawn_preview_process_async():
            try:
                import numpy as _np
                cur = self.maps[self.current_map_idx]
                grid_data = cur.global_grid if hasattr(cur, 'global_grid') else cur
                # Extract graph connectivity for stair traversal
                graph = getattr(cur, 'graph', None)
                room_to_node = getattr(cur, 'room_to_node', None)
                room_positions = getattr(cur, 'room_positions', None)
                
                fd, grid_file = tempfile.mkstemp(prefix='zave_preview_', suffix='.npy')
                os.close(fd)
                _np.save(grid_file, _np.array(grid_data, dtype=_np.int64))
                preview_out = os.path.join(tempfile.gettempdir(), f'zave_preview_out_{int(time.time())}_{os.getpid()}.pkl')
                proc = multiprocessing.Process(
                    target=_run_preview_and_dump, 
                    args=(grid_file, tuple(self.env.start_pos), tuple(self.env.goal_pos), getattr(self, 'algorithm_idx', 0), dict(self.feature_flags), {}, preview_out),
                    kwargs={'graph': graph, 'room_to_node': room_to_node, 'room_positions': room_positions},
                    daemon=True)
                logger.debug('Starting preview process for map %s -> outfile=%s gridfile=%s', self.current_map_idx, preview_out, grid_file)
                proc.start()
                logger.debug('Preview process started pid=%s alive=%s', getattr(proc,'pid',None), proc.is_alive())
                self.preview_proc = proc
                self.preview_outfile = preview_out
                self.preview_gridfile = grid_file
                self.preview_result = None
                self.preview_done = False
                self._set_message('Preview search started (background)')
            except Exception:
                logger.exception('Failed to spawn preview process; falling back to threaded worker')
                # Fallback to threaded worker (less ideal) if process spawn fails
                self.preview_result = None
                self.preview_thread = threading.Thread(target=_preview_worker, daemon=True)
                self.preview_thread.start()
                self._set_message('Preview search started (thread)')

        # Spawn the preview starting sequence on a short-lived background thread so the main
        # UI thread never blocks on file IO or process creation (can be slow on Windows).
        th = threading.Thread(target=_spawn_preview_process_async, daemon=True)
        th.start()

    def _clear_solver_state(self, reason="cleanup"):
        """Helper to centralize solver state cleanup and ensure consistency.
        
        Args:
            reason: Description of why solver is being cleared (for logging)
        """
        logger.info('SOLVER_CLEANUP: Clearing solver state (%s)', reason)
        self.solver_running = False
        self.solver_done = True
        self.solver_proc = None
        self.solver_outfile = None
        self.solver_gridfile = None
        self.solver_start_time = None
        self.solver_starting = False
        # CRITICAL: Clear the saved algorithm so banner doesn't show stale info
        if hasattr(self, 'solver_algorithm_idx'):
            delattr(self, 'solver_algorithm_idx')
        logger.debug('SOLVER_CLEANUP: State cleared')

    def _start_auto_solve(self):
        """Start auto-solve mode using state-space solver with inventory tracking.

        This schedules the heavy solver in a background process/thread using
        the existing `_schedule_solver()` helper. Non-blocking and safe to call
        from the main loop or event handlers.
        """
        algorithm_names = ["A*", "BFS", "Dijkstra", "Greedy", "D* Lite",
                         "DFS/IDDFS", "Bidirectional A*",
                         "CBS (Balanced)", "CBS (Explorer)", "CBS (Cautious)",
                         "CBS (Forgetful)", "CBS (Speedrunner)", "CBS (Greedy)"]
        
        # CRITICAL FIX: Read algorithm_idx directly from dropdown widget at solve-time
        # This is the DEFINITIVE source of truth for which algorithm the user selected
        alg_idx = None  # Will be set by widget lookup
        if hasattr(self, 'widget_manager') and self.widget_manager:
            for widget in self.widget_manager.widgets:
                if hasattr(widget, 'control_name') and widget.control_name == 'algorithm':
                    alg_idx = widget.selected
                    logger.info('SOLVER_FIX: Read algorithm_idx=%d (%s) directly from dropdown widget.selected',
                               alg_idx, algorithm_names[alg_idx] if alg_idx < len(algorithm_names) else 'Unknown')
                    break
        
        # Fallback to self.algorithm_idx if widget not found (shouldn't happen in normal operation)
        if alg_idx is None:
            alg_idx = getattr(self, 'algorithm_idx', 0)
            logger.warning('SOLVER_FIX: Widget lookup failed, falling back to self.algorithm_idx=%d', alg_idx)
        
        alg_name = algorithm_names[alg_idx] if alg_idx < len(algorithm_names) else f"Algorithm {alg_idx}"
        
        logger.info('═══════════════════════════════════════════════════')
        logger.info('DEBUG_SOLVER: _start_auto_solve() called')
        logger.info(f'  Algorithm: {alg_name} (idx={alg_idx})')
        logger.info('DEBUG_SOLVER: solver_running=%s, auto_mode=%s, auto_start_solver=%s',
                    getattr(self, 'solver_running', None),
                    getattr(self, 'auto_mode', None),
                    getattr(self, 'auto_start_solver', None))
        logger.info('═══════════════════════════════════════════════════')
        
        if getattr(self, 'solver_running', False):
            self._set_message('Solver already running', 1.5)
            logger.warning('DEBUG_SOLVER: Solver already running - returning early')
            # CRITICAL DIAGNOSTIC: Log full solver state to help diagnose stuck state
            logger.warning('DEBUG_SOLVER: Solver state dump:')
            logger.warning('  solver_running=%s', getattr(self, 'solver_running', None))
            logger.warning('  solver_done=%s', getattr(self, 'solver_done', None))
            logger.warning('  solver_proc=%s (alive=%s)', 
                          getattr(self, 'solver_proc', None),
                          getattr(getattr(self, 'solver_proc', None), 'is_alive', lambda: 'N/A')())
            logger.warning('  solver_outfile=%s (exists=%s)', 
                          getattr(self, 'solver_outfile', None),
                          os.path.exists(getattr(self, 'solver_outfile', '') or ''))
            logger.warning('  solver_start_time=%s (age=%.1fs)', 
                          getattr(self, 'solver_start_time', None),
                          (time.time() - getattr(self, 'solver_start_time', time.time())))
            
            # RECOVERY: If solver_running=True but proc is dead and done, force cleanup
            proc = getattr(self, 'solver_proc', None)
            done = getattr(self, 'solver_done', False)
            start_time = getattr(self, 'solver_start_time', None)
            try:
                proc_alive = proc.is_alive() if proc else False
            except Exception:
                proc_alive = False
            
            # Calculate age and check timeout (default 30s, configurable via env var)
            solver_age = (time.time() - start_time) if start_time else 0
            solver_timeout = float(os.environ.get('KLTN_SOLVER_TIMEOUT', '30'))
            
            # FORCE CLEANUP if:
            # 1. Process dead but not marked done, OR
            # 2. Process alive but timed out (hung), OR
            # 3. No process handle at all (spawn failed)
            needs_recovery = False
            recovery_reason = None
            
            if not proc:
                needs_recovery = True
                recovery_reason = 'No process handle (spawn failed or already cleaned)'
            elif not proc_alive and not done:
                needs_recovery = True
                recovery_reason = f'Process dead (exitcode={getattr(proc, "exitcode", "N/A")}) but not marked done'
            elif proc_alive and solver_age > solver_timeout:
                needs_recovery = True
                recovery_reason = f'Process alive but timed out ({solver_age:.1f}s > {solver_timeout}s)'
                # Terminate hung process
                try:
                    logger.warning('DEBUG_SOLVER: Terminating hung process pid=%s', getattr(proc, 'pid', 'N/A'))
                    proc.terminate()
                    proc.join(timeout=1.0)
                    if proc.is_alive():
                        logger.error('DEBUG_SOLVER: Process still alive after terminate, trying kill')
                        proc.kill()
                        proc.join(timeout=0.5)
                except Exception as e:
                    logger.exception('DEBUG_SOLVER: Failed to terminate hung process: %s', e)
            
            if needs_recovery:
                logger.error('DEBUG_SOLVER: RECOVERY TRIGGERED - %s', recovery_reason)
                logger.error('DEBUG_SOLVER: Force-cleaning stuck solver state')
                self.solver_running = False
                # CRITICAL FIX: DO NOT set solver_done=True during recovery!
                # We want the main loop to poll and clean up properly, not skip polling entirely.
                # self.solver_done = True  # ← BUG: This prevents retry from working!
                self.solver_proc = None
                self.solver_outfile = None
                self.solver_gridfile = None
                self.solver_start_time = None
                self.solver_starting = False
                # Clear solver_algorithm_idx to avoid stale banner display
                if hasattr(self, 'solver_algorithm_idx'):
                    delattr(self, 'solver_algorithm_idx')
                self._set_message(f'Recovered: {recovery_reason[:40]}')
                logger.info('DEBUG_SOLVER: Recovery complete - retrying solver start')
                # Don't return - allow immediate retry after recovery
            else:
                # Still running legitimately - don't allow retry
                logger.warning('DEBUG_SOLVER: Solver legitimately running (age=%.1fs < %.1fs timeout)', solver_age, solver_timeout)
                return
        # Stop any running preview process to avoid orphaned workers and stale output
        try:
            p = getattr(self, 'preview_proc', None)
            if p and p.is_alive():
                logger.info('DEBUG_SOLVER: Terminating existing preview process pid=%s', getattr(p, 'pid', None))
                try:
                    p.terminate()
                except Exception:
                    logger.exception('DEBUG_SOLVER: Failed to terminate preview process')
                try:
                    p.join(timeout=0.2)
                except Exception:
                    pass
        except Exception:
            logger.exception('DEBUG_SOLVER: Error while stopping preview process')
        try:
            out = getattr(self, 'preview_outfile', None)
            if out and os.path.exists(out):
                os.remove(out)
        except Exception:
            pass
        try:
            gf = getattr(self, 'preview_gridfile', None)
            if gf and os.path.exists(gf):
                os.remove(gf)
        except Exception:
            pass
        # Clear any previous preview state
        self.preview_done = False
        self.preview_proc = None
        self.preview_outfile = None
        self.preview_gridfile = None
        # Reset solver_done so we can detect when it completes
        self.solver_done = False
        # CRITICAL: Clear previous path so old path doesn't show while new solver runs
        self.auto_path = []
        self.auto_mode = False
        self.auto_step_idx = 0
        # Clear visual state from previous solve
        self.block_push_animations = []  # Clear any pending block animations
        self.door_unlock_times = {}  # Clear door unlock visual state
        # Clear collected items state (will be reset again in _execute_auto_solve)
        self.collected_items = []
        self.collected_positions = set()
        logger.info('DEBUG_SOLVER: Cleared previous state, solver_done=False')
        
        # DEBUG: Synchronous solver for debugging multiprocessing issues
        if DEBUG_SYNC_SOLVER:
            logger.warning('DEBUG_SOLVER: Running solver SYNCHRONOUSLY (blocking)')
            self._run_solver_sync()
            return
        
        # Indicate starting
        self._set_message('Starting solver in background...', 2.0)
        try:# CRITICAL FIX: Pass algorithm_idx explicitly to _schedule_solver() to eliminate race conditions
            self._schedule_solver(algorithm_idx=alg_idx)
            self._schedule_solver()
            logger.info('DEBUG_SOLVER: _schedule_solver() completed without exception')
        except Exception:
            logger.exception('Failed to schedule solver')
            self._set_message('Failed to start solver', 3.0)

    def _run_solver_sync(self):
        """DEBUG: Run solver synchronously in main thread to bypass multiprocessing issues.
        
        This blocks the UI but helps diagnose whether the issue is in multiprocessing
        or in the solver/animation logic itself.
        """
        logger.warning('DEBUG_SYNC: Starting synchronous solver (UI will freeze)')
        self._set_message('Running solver synchronously (debug)...', 5.0)
        
        # Get grid and positions
        cur = self.maps[self.current_map_idx]
        if hasattr(cur, 'global_grid'):
            grid_arr = cur.global_grid
            # Extract graph connectivity data for stair traversal
            graph = getattr(cur, 'graph', None)
            room_to_node = getattr(cur, 'room_to_node', None)
            room_positions = getattr(cur, 'room_positions', None)
            node_to_room = getattr(cur, 'node_to_room', None)
        else:
            grid_arr = cur
            graph = None
            room_to_node = None
            room_positions = None
            node_to_room = None
        
        if not self.env or not getattr(self.env, 'start_pos', None) or not getattr(self.env, 'goal_pos', None):
            self._set_message('Start/goal not defined for this map')
            logger.error('DEBUG_SYNC: No start/goal defined')
            return
        
        start = tuple(self.env.start_pos)
        goal = tuple(self.env.goal_pos)
        alg_idx = getattr(self, 'algorithm_idx', 0)
        flags = dict(self.feature_flags)
        priority_options = {
            'tie_break': self.feature_flags.get('priority_tie_break', False),
            'key_boost': self.feature_flags.get('priority_key_boost', False),
            'enable_ara': self.feature_flags.get('enable_ara', False),
        }
        
        logger.info('DEBUG_SYNC: Calling _solve_in_subprocess with start=%s, goal=%s', start, goal)
        
        try:
            # Call solver directly in main thread with graph data for stair traversal
            result = _solve_in_subprocess(grid_arr, start, goal, alg_idx, flags, priority_options,
                                          graph=graph, room_to_node=room_to_node, room_positions=room_positions,
                                          node_to_room=node_to_room)
            
            logger.info('DEBUG_SYNC: Solver returned: success=%s, path_len=%d',
                        result.get('success'),
                        len(result.get('path', []) or []))
            
            if result.get('success') and result.get('path'):
                self.auto_path = result['path']
                solver_result = result.get('solver_result', {})
                
                logger.info('DEBUG_SYNC: Path loaded successfully, first=%s, last=%s',
                            self.auto_path[0] if self.auto_path else None,
                            self.auto_path[-1] if self.auto_path else None)
                
                # Always execute auto-solve in sync mode (we want to test animation)
                logger.info('DEBUG_SYNC: Calling _execute_auto_solve()')
                self._execute_auto_solve(self.auto_path, solver_result, teleports=0)
                self._set_message(f'DEBUG: Solver done! Path: {len(self.auto_path)} steps. auto_mode={self.auto_mode}')
                
                logger.info('DEBUG_SYNC: After execute: auto_mode=%s, auto_step_idx=%s',
                            self.auto_mode, self.auto_step_idx)
            else:
                msg = result.get('message') or 'No path found'
                logger.warning('DEBUG_SYNC: Solver failed: %s', msg)
                self._set_message(f'DEBUG: No path - {msg}')
                
        except Exception as e:
            logger.exception('DEBUG_SYNC: Solver exception')
            self._set_message(f'DEBUG: Solver error - {e}')

    def _watchdog_loop(self):
        """Background watchdog that writes stack traces and a screenshot when the main loop stalls.

        Controlled by environment vars:
        - KLTN_ENABLE_WATCHDOG (default '1') enable watchdog
        - KLTN_WATCHDOG_THRESHOLD (seconds, default 1.25)
        - KLTN_WATCHDOG_DUMP_LIMIT (how many dumps to write, default 3)
        - KLTN_WATCHDOG_TERMINATE_SOLVER (if '1' will terminate solver proc when dumping)
        """
        try:
            import faulthandler
        except Exception:
            logger.debug('faulthandler not available; watchdog disabled')
            return

        while getattr(self, '_watchdog_enabled', False):
            try:
                time.sleep(0.5)
                last = getattr(self, 'last_frame_time', 0)
                now = time.time()
                if now - last > getattr(self, '_watchdog_threshold', 1.25):
                    # Avoid spamming dumps
                    if self._watchdog_dumps >= getattr(self, '_watchdog_dump_limit', 3):
                        continue
                    self._watchdog_dumps += 1
                    self._watchdog_last_dump = now

                    ts = int(now)
                    pid = os.getpid()
                    tmpdir = tempfile.gettempdir()
                    trace_path = os.path.join(tmpdir, f'zave_watchdog_trace_{pid}_{ts}.txt')
                    try:
                        with open(trace_path, 'w') as f:
                            f.write(f'Watchdog dump: time={now} last_frame={last}\n')
                            faulthandler.dump_traceback(file=f)
                        logger.warning('Watchdog detected stall; stack dump written: %s', trace_path)
                    except Exception:
                        logger.exception('Failed writing watchdog stack dump')

                    # Request main thread to save a screenshot (pygame display calls must be done on the main thread)
                    try:
                        shot_path = os.path.join(tmpdir, f'zave_watchdog_shot_{pid}_{ts}.png')
                        # Signal main thread to save the screenshot instead of calling pygame from this thread
                        try:
                            self._watchdog_request_screenshot = shot_path
                            logger.warning('Watchdog requested screenshot: %s', shot_path)
                        except Exception:
                            logger.exception('Failed to set watchdog screenshot request')
                    except Exception:
                        logger.exception('Watchdog screenshot request failed')

                    # Optionally terminate solver process if configured (to recover UI)
                    try:
                        if os.environ.get('KLTN_WATCHDOG_TERMINATE_SOLVER') == '1':
                            proc = getattr(self, 'solver_proc', None)
                            if proc and proc.is_alive():
                                logger.warning('Watchdog terminating solver process pid=%s', getattr(proc,'pid',None))
                                try:
                                    proc.terminate()
                                except Exception:
                                    logger.exception('Failed to terminate solver process')
                    except Exception:
                        logger.exception('Watchdog failed to check/terminate solver process')

                    # Set a user-visible message
                    try:
                        self._set_message(f'Watchdog: dumped trace ({os.path.basename(trace_path)})', 5.0)
                    except Exception:
                        pass
            except Exception:
                logger.exception('Uncaught exception in watchdog loop')
        logger.debug('Watchdog loop exiting')
        return
        

    
    def _schedule_solver(self, algorithm_idx=None):
        """Start solver in background worker process/thread.
        
        Args:
            algorithm_idx: Algorithm index to use (if None, read from self.algorithm_idx)
        """
        if getattr(self, 'solver_running', False):
            self._set_message('Solver already running...')
            logger.warning('SOLVER: _schedule_solver blocked - solver_running already True')
            return
        
        # CRITICAL: Set solver_running=True IMMEDIATELY to prevent race condition
        # This must happen BEFORE spawning any thread
        self.solver_running = True
        self.solver_done = False
        self.solver_start_time = time.time()  # Track start time for timeout detection
        self.solver_starting = True
        
        # CRITICAL FIX: Use explicit algorithm_idx parameter if provided
        # This eliminates race conditions between dropdown widget state and self.algorithm_idx
        if algorithm_idx is not None:
            current_alg_idx = algorithm_idx
            # Also sync self.algorithm_idx for consistency with other parts of the code
            self.algorithm_idx = algorithm_idx
            logger.info('SOLVER_FIX: Using explicit algorithm_idx=%d passed to _schedule_solver()', algorithm_idx)
        else:
            current_alg_idx = getattr(self, 'algorithm_idx', None)
            logger.info('SOLVER: Using self.algorithm_idx=%s (no explicit arg provided)', current_alg_idx)
        
        # DEBUG: Log algorithm_idx when solver starts to diagnose CBS→A* display bug
        logger.info('SOLVER: Acquired solver lock, solver_running=True, solver_done=False, start_time=%.3f, algorithm_idx=%s', 
                   self.solver_start_time, current_alg_idx)
        
        # CRITICAL FIX: Save the algorithm_idx that THIS solver is using
        # This ensures the banner shows the correct algorithm even if user changes dropdown mid-solve
        self.solver_algorithm_idx = current_alg_idx if current_alg_idx is not None else 0
        
        self._auto_recenter_done = False

        cur = self.maps[self.current_map_idx]
        if hasattr(cur, 'global_grid'):
            grid_arr = cur.global_grid
            # Extract graph connectivity data for stair traversal
            graph = getattr(cur, 'graph', None)
            room_to_node = getattr(cur, 'room_to_node', None)
            room_positions = getattr(cur, 'room_positions', None)
            node_to_room = getattr(cur, 'node_to_room', None)
        else:
            grid_arr = cur
            graph = None
            room_to_node = None
            room_positions = None
            node_to_room = None
        # Ensure start/goal defined
        if not self.env or not getattr(self.env, 'start_pos', None) or not getattr(self.env, 'goal_pos', None):
            self._set_message('Start/goal not defined for this map')
            # CRITICAL: Clear solver_running immediately on early exit
            self._clear_solver_state(reason="missing start/goal")
            logger.warning('SOLVER: Missing start/goal - cleared solver state')
            return
        start = tuple(self.env.start_pos)
        goal = tuple(self.env.goal_pos)
        alg_idx = getattr(self, 'algorithm_idx', 0)
        flags = dict(self.feature_flags)
        priority_options = {
            'tie_break': self.feature_flags.get('priority_tie_break', False),
            'key_boost': self.feature_flags.get('priority_key_boost', False),
            'enable_ara': self.feature_flags.get('enable_ara', False),
            'allow_diagonals': True,  # Enable for fast pathfinding (30× speedup), converted to 4-dir for display
        }

        # Create temp file path for child to write result with pickle
        temp_dir = tempfile.gettempdir()
        out_file = os.path.join(temp_dir, f'zave_solver_out_{int(time.time())}_{os.getpid()}.pkl')
        logger.info('SOLVER: Starting subprocess, pickle_path=%s', out_file)
        logger.info('SOLVER: start=%s, goal=%s, algorithm_idx=%s', start, goal, alg_idx)

        # Save grid to .npy to avoid costly pickling in Process spawn
        grid_file = None
        try:
            import numpy as _np
            fd, grid_file = tempfile.mkstemp(prefix='zave_grid_', suffix='.npy')
            os.close(fd)
            _np.save(grid_file, _np.array(grid_arr, dtype=_np.int64))
        except Exception:
            # Fallback: try pickling to file (last resort)
            try:
                fd, grid_file = tempfile.mkstemp(prefix='zave_grid_', suffix='.pkl')
                os.close(fd)
                with open(grid_file, 'wb') as gf:
                    pickle.dump(grid_arr, gf)
            except Exception:
                grid_file = None
        
        # CRITICAL: Set outfile/gridfile IMMEDIATELY so run loop can find them
        # This must happen BEFORE spawning the thread
        self.solver_outfile = out_file
        self.solver_gridfile = grid_file
        logger.info('SOLVER: File handles stored: outfile=%s, gridfile=%s', out_file, grid_file)
        
        def _start_proc():
            try:
                # Test mode: start a lightweight sleep target instead of full solver to validate process spawn
                if os.environ.get('KLTN_SOLVER_TEST') == '1':
                    import time as _time
                    proc = multiprocessing.Process(target=_time.sleep, args=(2,), daemon=True)
                    proc.start()
                    self.solver_proc = proc
                    self.solver_starting = False
                    # Note: solver_outfile and solver_gridfile already set before _start_proc
                    self._set_message('Test solver process started (sleep)')
                    return

                # Prefer passing a grid file path to avoid large pickle overhead
                grid_arg = grid_file if grid_file else grid_arr
                logger.info('SOLVER: Creating subprocess with gridfile=%s, outfile=%s', grid_file, out_file)
                proc = multiprocessing.Process(target=_run_solver_and_dump, 
                                               args=(grid_arg, start, goal, alg_idx, flags, priority_options, out_file),
                                               kwargs={'graph': graph, 'room_to_node': room_to_node, 'room_positions': room_positions, 'node_to_room': node_to_room},
                                               daemon=True)
                proc.start()
                logger.info('SOLVER: Subprocess started pid=%s, is_alive=%s', getattr(proc,'pid',None), proc.is_alive())
                self.solver_proc = proc
                self.solver_starting = False
                # Note: solver_outfile and solver_gridfile already set before _start_proc
                self._set_message('Solver started in background')
                logger.info('SOLVER: Process handle stored, waiting for completion in run loop')
            except Exception as e:
                logger.exception('SOLVER: Failed to start solver process: %s', e)
                # Fallback to thread-based run (less ideal)
                # Note: solver_running already True, keep it True during fallback
                logger.info('SOLVER: Falling back to thread-based solver')
                
                # CRITICAL FIX: Clear solver_starting IMMEDIATELY so main loop knows thread has started
                # If we don't do this, main loop will wait in startup grace period and miss the result
                self.solver_starting = False
                
                def _thread_fallback():
                    try:
                        # Use grid_arr (not grid which doesn't exist in this scope)
                        # Pass graph connectivity for stair traversal
                        res = _solve_in_subprocess(grid_arr, start, goal, alg_idx, flags, priority_options,
                                                   graph=graph, room_to_node=room_to_node, room_positions=room_positions,
                                                   node_to_room=node_to_room)
                        logger.info('SOLVER: Thread fallback completed, success=%s', res.get('success') if res else None)
                        # write fallback output
                        try:
                            with open(out_file, 'wb') as f:
                                pickle.dump(res, f)
                            logger.info('SOLVER: Thread fallback wrote result to %s', out_file)
                        except Exception as write_err:
                            logger.exception('SOLVER: Thread fallback failed to write output: %s', write_err)
                    except Exception as solve_err:
                        logger.exception('SOLVER: Thread fallback solver exception: %s', solve_err)
                    finally:
                        # CRITICAL FIX: Only clear solver_running, NOT solver_done!
                        # Main loop needs solver_done=False to detect completion and read results.
                        # Setting solver_done=True here would skip result polling entirely.
                        self.solver_running = False
                        self.solver_start_time = None
                        self.solver_starting = False
                        # Clear solver_algorithm_idx to avoid stale banner display
                        if hasattr(self, 'solver_algorithm_idx'):
                            delattr(self, 'solver_algorithm_idx')
                        logger.info('SOLVER: Thread fallback finished, solver_running=False (main loop will poll results)')
                # Store outfile so run loop can read it
                self.solver_outfile = out_file
                self.solver_gridfile = grid_file
                try:
                    th = threading.Thread(target=_thread_fallback, daemon=True)
                    th.start()
                    self._set_message('Solver started in background (thread fallback)')
                except Exception as thread_err:
                    logger.exception('SOLVER: Failed to start thread fallback: %s', thread_err)
                    # CRITICAL: If thread spawn fails, clear solver state BUT NOT solver_done!
                    # Setting solver_done=True would prevent the completion block from cleaning up temp files.
                    self.solver_running = False
                    self.solver_proc = None
                    self.solver_outfile = None
                    # Clear solver_algorithm_idx to avoid stale banner display
                    if hasattr(self, 'solver_algorithm_idx'):
                        delattr(self, 'solver_algorithm_idx')
                    self.solver_gridfile = None
                    self.solver_start_time = None
                    self.solver_starting = False
                    self._set_message('Failed to start solver')
                    logger.error('SOLVER: Complete failure - all solver mechanisms exhausted')
        # Start the process on a worker thread to avoid blocking the caller
        try:
            t = threading.Thread(target=_start_proc, daemon=True)
            t.start()
        except Exception:
            _start_proc()

    def _execute_auto_solve(self, path, solver_result, teleports=0):
        """
        Execute auto-solve immediately without preview (fallback).
        
        Args:
            path: Planned path
            solver_result: Solver metadata (may include CBS metrics)
            teleports: Number of teleport/warp moves
        """
        # Validate path before starting animation
        if not path or len(path) == 0:
            logger.error('EXECUTE: Refusing to start animation with empty path')
            self._show_error('No valid path to animate')
            return
        
        logger.info('EXECUTE: path=%d steps, setting auto_mode=True', len(path) if path else 0)
        logger.info('EXECUTE: Before state: auto_mode=%s, auto_step_idx=%s',
                    getattr(self, 'auto_mode', None),
                    getattr(self, 'auto_step_idx', None))
        
        self.auto_path = path
        self.auto_step_idx = 0
        self.auto_mode = True
        self.auto_step_timer = 0.0  # Reset animation timer
        # Reset usage counters for visual run
        self.keys_used = 0
        self.bombs_used = 0
        self.boss_keys_used = 0
        self.used_items = []
        # CRITICAL FIX: Reset collected items tracking when starting new auto-solve
        # This ensures items show as uncollected at start of animation
        self.collected_items = []
        self.collected_positions = set()
        self.keys_collected = 0
        self.bombs_collected = 0
        self.boss_keys_collected = 0
        self.item_type_map = {}  # Reset item type mapping
        self.item_pickup_times = {}  # Reset pickup flash timers
        self.env.reset()
        
        # Store CBS metrics if available
        if solver_result and 'cbs_metrics' in solver_result:
            self.last_solver_metrics = {
                'name': f"CBS ({solver_result.get('persona', 'unknown')})",
                'nodes': solver_result.get('nodes', 0),
                'path_len': len(path),
                'cbs': solver_result['cbs_metrics']
            }
        
        # === SCAN ITEMS ALONG PATH for visualization ===
        try:
            self._scan_items_along_path(path)
            items_text = self._get_path_items_display_text()
            if items_text:
                logger.info('EXECUTE: Path items preview: %s', items_text)
        except Exception as e:
            logger.warning('EXECUTE: Failed to scan path items: %s', e)
            items_text = ""
        
        logger.info('EXECUTE: After state: auto_mode=%s, auto_step_idx=%s, auto_path_len=%d',
                    self.auto_mode,
                    self.auto_step_idx,
                    len(self.auto_path) if self.auto_path else 0)
        logger.info('EXECUTE: Animation ready, first_step=%s, last_step=%s',
                    self.auto_path[0] if self.auto_path else None,
                    self.auto_path[-1] if self.auto_path else None)
        
        # Build informative message with path items preview
        keys_used = solver_result.get('keys_used', 0) if solver_result else 0
        keys_avail = solver_result.get('keys_available', 0) if solver_result else 0
        key_info = f"Keys: {keys_avail}->{keys_avail - keys_used}" if keys_used > 0 else ""
        
        # Include items along path in message
        items_display = self._get_path_items_display_text()
        
        # Check if CBS metrics are available
        if solver_result and 'cbs_metrics' in solver_result:
            cbs = solver_result['cbs_metrics']
            persona = solver_result.get('persona', 'unknown')
            base_msg = f"CBS ({persona.title()}): {len(path)} steps"
            metrics_msg = f"Confusion: {cbs['confusion_index']:.2f} | Cognitive Load: {cbs['cognitive_load']:.2f}"
            self.message = f"{base_msg} | {metrics_msg}"
            
            # Show detailed toast notification for CBS completion
            toast_msg = f"CBS ({persona.title()}) completed | Confusion: {cbs['confusion_index']:.2f} | Entropy: {cbs['navigation_entropy']:.2f}"
            self._show_toast(toast_msg, duration=4.0, toast_type='success')
        elif teleports > 0:
            self.message = f"Path: {len(path)} ({teleports} warps) {key_info}"
        else:
            base_msg = f"Path: {len(path)} steps"
            if items_display:
                self.message = f"{base_msg} | Items: {items_display}"
            elif key_info:
                self.message = f"{base_msg} {key_info}"
            else:
                self.message = base_msg
    
    def _execute_auto_solve_from_preview(self):
        """
        Start auto-solve after user confirms path preview.
        """
        self.auto_step_idx = 0
        self.auto_mode = True
        self.auto_step_timer = 0.0  # Reset animation timer
        # Reset usage counters for visual run
        self.keys_used = 0
        self.bombs_used = 0
        self.boss_keys_used = 0
        self.used_items = []
        # CRITICAL FIX: Reset collected items tracking when starting from preview
        # This ensures items show as uncollected at start of animation
        self.collected_items = []
        self.collected_positions = set()
        self.keys_collected = 0
        self.bombs_collected = 0
        self.boss_keys_collected = 0
        self.item_type_map = {}  # Reset item type mapping
        self.item_pickup_times = {}  # Reset pickup flash timers
        self.env.reset()
        
        # === SCAN ITEMS ALONG PATH for visualization ===
        try:
            self._scan_items_along_path(self.auto_path)
            items_text = self._get_path_items_display_text()
        except Exception as e:
            logger.warning('EXECUTE_PREVIEW: Failed to scan path items: %s', e)
            items_text = ""
        
        # Dismiss preview / overlay
        self.path_preview_mode = False
        preview_dialog = self.path_preview_dialog
        self.path_preview_dialog = None
        self.preview_overlay_visible = False
        
        # Use stored path and show message with items preview
        if preview_dialog:
            base_msg = f"Auto-solve started! Path: {len(self.auto_path)} steps"
            if items_text:
                self.message = f"{base_msg} | Items: {items_text}"
            else:
                self.message = base_msg
        else:
            self.message = "Auto-solve started!"
    
    def _smart_grid_path(self):
        """
        Smart pathfinding that prioritizes walking and only warps via STAIRs.
        Returns (success, path, teleport_count).
        """
        from collections import deque
        import networkx as nx
        
        current_dungeon = self.maps[self.current_map_idx]
        # Support both StitchedDungeon objects (with .global_grid) and raw numpy grids
        if hasattr(current_dungeon, 'global_grid'):
            grid = current_dungeon.global_grid
            room_positions = getattr(current_dungeon, 'room_positions', {})
            room_to_node = getattr(current_dungeon, 'room_to_node', {})
            graph = getattr(current_dungeon, 'graph', None)
        else:
            grid = current_dungeon
            room_positions = {}
            room_to_node = {}
            graph = None
        H, W = grid.shape
        
        # Constants
        ROOM_HEIGHT = 16
        ROOM_WIDTH = 11
        FLOOR = 1
        WALL = 2
        BLOCK = 3
        VOID = 0
        DOOR_OPEN = 10
        DOOR_LOCKED = 11
        START = 21
        TRIFORCE = 22
        KEY = 30
        STAIR = 42
        
        WALKABLE = {FLOOR, DOOR_OPEN, KEY, START, TRIFORCE, STAIR, DOOR_LOCKED}
        
        start = self.env.start_pos
        goal = self.env.goal_pos
        
        # Guard: if start or goal missing (tests may simulate missing), bail out gracefully
        if start is None or goal is None:
            return False, [], 0

        # Reset and optionally collect search heatmap data
        if getattr(self, 'show_heatmap', False):
            self.search_heatmap = {}

        # Helper: get room for a position
        def get_room(pos):
            y, x = pos
            for room_pos, (ry, rx) in room_positions.items():
                if ry <= y < ry + ROOM_HEIGHT and rx <= x < rx + ROOM_WIDTH:
                    return room_pos
            return None

        # Helper: find all STAIR tiles in a room
        def get_stairs_in_room(room_pos):
            if room_pos not in room_positions:
                return []
            ry, rx = room_positions[room_pos]
            stairs = []
            for dy in range(ROOM_HEIGHT):
                for dx in range(ROOM_WIDTH):
                    y, x = ry + dy, rx + dx
                    if 0 <= y < H and 0 <= x < W and grid[y, x] == STAIR:
                        stairs.append((y, x))
            return stairs

        # Helper: find entry point in room
        def find_entry(room_pos):
            if room_pos not in room_positions:
                return None
            ry, rx = room_positions[room_pos]
            # Prefer stair, then center, then any walkable
            for dy in range(ROOM_HEIGHT):
                for dx in range(ROOM_WIDTH):
                    y, x = ry + dy, rx + dx
                    if 0 <= y < H and 0 <= x < W and grid[y, x] == STAIR:
                        return (y, x)
            cy, cx = ry + ROOM_HEIGHT // 2, rx + ROOM_WIDTH // 2
            if 0 <= cy < H and 0 <= cx < W and grid[cy, cx] in WALKABLE:
                return (cy, cx)
            for dy in range(ROOM_HEIGHT):
                for dx in range(ROOM_WIDTH):
                    y, x = ry + dy, rx + dx
                    if 0 <= y < H and 0 <= x < W and grid[y, x] in WALKABLE:
                        return (y, x)
            return None
        
        # Get graph info for stair connections (if available)
        node_to_room = {v: k for k, v in room_to_node.items()}
        # 'graph' variable already set earlier based on current_dungeon
        
        # Find stair destinations based on graph connectivity
        def get_stair_destinations(pos):
            """Find where stairs lead to based on graph."""
            if grid[pos[0], pos[1]] != STAIR:
                return []
            
            if not graph:
                return []
            
            current_room = get_room(pos)
            if not current_room:
                return []
            
            current_node = room_to_node.get(current_room)
            if current_node is None:
                return []
            
            destinations = []
            # Check graph neighbors
            for neighbor_node in graph.neighbors(current_node):
                neighbor_room = node_to_room.get(neighbor_node)
                if neighbor_room and neighbor_room in current_dungeon.room_positions:
                    # Find stairs in neighbor room, or entry point
                    neighbor_stairs = get_stairs_in_room(neighbor_room)
                    if neighbor_stairs:
                        destinations.extend(neighbor_stairs)
                    else:
                        entry = find_entry(neighbor_room)
                        if entry:
                            destinations.append(entry)
            
            return destinations
        
        # Choose search algorithm based on UI selection
        alg = getattr(self, 'algorithm_idx', 0)
        # 0: A*, 1: BFS, 2: Dijkstra, 3: Greedy, 4: D* Lite (fallback to A*)
        # 5-10: CBS variants (use subprocess solver, not quick grid path)
        # Note: D* Lite not yet implemented — selecting it currently uses A* fallback.
        
        # CRITICAL FIX: CBS algorithms (5-10) require the full subprocess solver
        # They cannot be implemented as simple grid search, so skip quick path
        # and return to trigger the heavy solver immediately
        cbs_algorithms = {5, 6, 7, 8, 9, 10}
        if alg in cbs_algorithms:
            algorithm_names = ["A*", "BFS", "Dijkstra", "Greedy", "D* Lite",
                             "DFS/IDDFS", "Bidirectional A*",
                             "CBS (Balanced)", "CBS (Explorer)", "CBS (Cautious)",
                             "CBS (Forgetful)", "CBS (Speedrunner)", "CBS (Greedy)"]
            alg_name = algorithm_names[alg] if alg < len(algorithm_names) else f"Algorithm {alg}"
            logger.info(f"CBS algorithm selected ({alg_name}): Skipping quick grid path, will use full solver")
            # Return failure to trigger fallback to full subprocess solver with CBS support
            return False, [], 0
        
        if alg == 4:
            logger.info("D* Lite selected but not implemented; using A* fallback")
            self._set_message("D* Lite selected: using A* fallback (not implemented)", 2.5)

        def heuristic(a, b):
            # Use ML heuristic if enabled and model available, else Manhattan (or octile for diagonal)
            if self.feature_flags.get('ml_heuristic', False) and getattr(self, 'ml_model', None):
                try:
                    return self._ml_heuristic(a, b)
                except Exception as e:
                    logger.warning(f"ML heuristic failed, falling back to Manhattan/Octile: {e}")
            if self.feature_flags.get('diagonal_movement', False):
                # Use octile distance when diagonal moves are allowed
                return math.hypot(a[0]-b[0], a[1]-b[1])
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        # Placeholder ML heuristic loader
        if not hasattr(self, 'ml_model'):
            self.ml_model = None
        def _ml_heuristic_stub(a, b):
            # For now, a simple biased Manhattan (simulates ML model outputs)
            return 0.9 * (abs(a[0] - b[0]) + abs(a[1] - b[1]))
        if getattr(self, 'ml_model', None) is None:
            self._ml_heuristic = _ml_heuristic_stub
        else:
            # If you add a real model loader, assign self._ml_heuristic accordingly
            pass
        max_iterations = 200000
        counter = 0
        # Track iterations for diagnostics (nodes expanded)
        iterations = 0
        self.last_search_iterations = 0

        if alg == 1:
            # BFS with stair teleportation (existing behavior)
            initial = (start, frozenset())
            visited = {start}
            queue = deque([(start, [start], 0)])  # pos, path, teleport_count
            iterations = 0
            while queue and iterations < max_iterations:
                iterations += 1
                pos, path, teleports = queue.popleft()
                y, x = pos
                if getattr(self, 'show_heatmap', False):
                    self.search_heatmap[pos] = self.search_heatmap.get(pos, 0) + 1
                if pos == goal:
                    self.last_search_iterations = iterations
                    return True, path, teleports
                # 4-directional walking
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and grid[ny, nx] in WALKABLE:
                        npos = (ny, nx)
                        if npos not in visited:
                            visited.add(npos)
                            queue.append((npos, path + [npos], teleports))
                # Stair teleportation
                if grid[y, x] == STAIR:
                    for dest in get_stair_destinations(pos):
                        if dest not in visited:
                            visited.add(dest)
                            queue.append((dest, path + [dest], teleports + 1))
            # No grid path found within BFS limits
            self.last_search_iterations = iterations
            return self._graph_guided_path()
        else:
            # Implement priority-search-based algorithms (A*, Dijkstra, Greedy)
            import heapq
            # Node state: (pos, visited_stairs_frozenset)
            start_state = (start, frozenset())
            # Priority queue entries: (priority, g_cost, counter, pos, visited_stairs, teleports, path)
            start_h = heuristic(start, goal)
            if alg == 0 or alg == 4:
                start_f = start_h
            elif alg == 2:
                start_f = 0
            elif alg == 3:
                start_f = start_h
            else:
                start_f = start_h
            heap = []
            heapq.heappush(heap, (start_f, 0, counter, start, frozenset(), 0, [start]))
            counter += 1
            best = {}  # (pos, stairs) -> best_g
            iterations = 0
            # Read GUI flags for diagonal movement and JPS
            allow_diag = self.feature_flags.get('diagonal_movement', False)
            use_jps = self.feature_flags.get('use_jps', False)
            while heap and iterations < max_iterations:
                iterations += 1
                f, g, _cnt, pos, stairs, teleports, path = heapq.heappop(heap)
                if getattr(self, 'show_heatmap', False):
                    self.search_heatmap[pos] = self.search_heatmap.get(pos, 0) + 1
                if pos == goal:
                    self.last_search_iterations = iterations
                    return True, path, teleports
                key = (pos, stairs)
                if key in best and g > best[key]:
                    continue
                best[key] = g

                y, x = pos
                # Determine neighbor generator based on diagonal option
                if allow_diag:
                    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1,-1), (-1,1), (1,-1), (1,1)]
                else:
                    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]

                for dy, dx in deltas:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and grid[ny, nx] in WALKABLE:
                        npos = (ny, nx)
                        step_cost = math.hypot(dy, dx)
                        new_g = g + step_cost
                        new_stairs = stairs
                        new_teleports = teleports
                        # compute priority
                        h = heuristic(npos, goal)
                        if alg == 2:
                            new_f = new_g
                        elif alg == 3:
                            new_f = h
                        else:
                            new_f = new_g + h
                        nkey = (npos, new_stairs)
                        if nkey in best and new_g >= best[nkey]:
                            continue
                        heapq.heappush(heap, (new_f, new_g, counter, npos, new_stairs, new_teleports, path + [npos]))
                        counter += 1
                # Stair teleportation
                if grid[y, x] == STAIR:
                    for dest in get_stair_destinations(pos):
                        npos = dest
                        if npos in stairs:
                            continue
                        new_stairs = set(stairs)
                        new_stairs.add(npos)
                        new_stairs = frozenset(new_stairs)
                        new_g = g + 1
                        new_teleports = teleports + 1
                        h = heuristic(npos, goal)
                        if alg == 2:
                            new_f = new_g
                        elif alg == 3:
                            new_f = h
                        else:
                            new_f = new_g + h
                        nkey = (npos, new_stairs)
                        if nkey in best and new_g >= best[nkey]:
                            continue
                        heapq.heappush(heap, (new_f, new_g, counter, npos, new_stairs, new_teleports, path + [npos]))
                        counter += 1
                # Optional JPS path expansion: attempt to use JPS local planner between node and neighbors
                if use_jps:
                    try:
                        from bench.grid_solvers import jps as _jps
                        # Try to link current pos to goal or neighbors via JPS and capture trace
                        if allow_diag:
                            _res = _jps(grid.tolist() if hasattr(grid, 'tolist') else grid, pos, goal, allow_diagonal=True, trace=True)
                        else:
                            _res = _jps(grid.tolist() if hasattr(grid, 'tolist') else grid, pos, goal, allow_diagonal=False, trace=True)
                        if _res:
                            if len(_res) == 3:
                                jp_path, jp_nodes, jp_trace = _res
                            else:
                                jp_path, jp_nodes = _res
                                jp_trace = None
                            # Store latest trace for overlay
                            self.last_jps_trace = jp_trace
                            if jp_path and len(jp_path) > 1:
                                # Use the last jump target from the returned path
                                target = jp_path[-1]
                                new_g = g + euclidean(pos, target)
                                h = heuristic(target, goal)
                                heapq.heappush(heap, (new_g + h, new_g, counter, target, stairs, teleports, path + [target]))
                                counter += 1
                    except Exception:
                        self.last_jps_trace = None
                        pass
            # No path found within priority-search limits
            self.last_search_iterations = iterations
            return self._graph_guided_path()    
    def _graph_guided_path(self):
        """Fallback: follow graph path with teleportation when needed."""
        import networkx as nx
        from collections import deque
        
        current_dungeon = self.maps[self.current_map_idx]
        # Support both StitchedDungeon objects (with .global_grid) and raw numpy grids
        if hasattr(current_dungeon, 'global_grid'):
            grid = current_dungeon.global_grid
            room_positions = getattr(current_dungeon, 'room_positions', {})
            room_to_node = getattr(current_dungeon, 'room_to_node', {})
            graph = getattr(current_dungeon, 'graph', None)
        else:
            grid = current_dungeon
            room_positions = {}
            room_to_node = {}
            graph = None
        H, W = grid.shape
        
        ROOM_HEIGHT = 16
        ROOM_WIDTH = 11
        FLOOR = 1
        DOOR_OPEN = 10
        DOOR_LOCKED = 11
        START = 21
        TRIFORCE = 22
        KEY = 30
        STAIR = 42
        WALKABLE = {FLOOR, DOOR_OPEN, KEY, START, TRIFORCE, STAIR, DOOR_LOCKED}
        
        start = self.env.start_pos
        goal = self.env.goal_pos
        
        def get_room(pos):
            y, x = pos
            for room_pos, (ry, rx) in current_dungeon.room_positions.items():
                if ry <= y < ry + ROOM_HEIGHT and rx <= x < rx + ROOM_WIDTH:
                    return room_pos
            return None
        
        def find_entry(room_pos):
            if room_pos not in current_dungeon.room_positions:
                return None
            ry, rx = current_dungeon.room_positions[room_pos]
            for dy in range(ROOM_HEIGHT):
                for dx in range(ROOM_WIDTH):
                    y, x = ry + dy, rx + dx
                    if 0 <= y < H and 0 <= x < W and grid[y, x] == STAIR:
                        return (y, x)
            cy, cx = ry + ROOM_HEIGHT // 2, rx + ROOM_WIDTH // 2
            if 0 <= cy < H and 0 <= cx < W and grid[cy, cx] in WALKABLE:
                return (cy, cx)
            for dy in range(ROOM_HEIGHT):
                for dx in range(ROOM_WIDTH):
                    y, x = ry + dy, rx + dx
                    if 0 <= y < H and 0 <= x < W and grid[y, x] in WALKABLE:
                        return (y, x)
            return None
        
        def local_bfs(from_pos, to_pos, max_steps=5000):
            if from_pos == to_pos:
                return [from_pos]
            visited = {from_pos}
            queue = deque([(from_pos, [from_pos])])
            while queue and len(visited) < max_steps:
                pos, path = queue.popleft()
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = pos[0] + dy, pos[1] + dx
                    if 0 <= ny < H and 0 <= nx < W and (ny, nx) not in visited:
                        if grid[ny, nx] in WALKABLE:
                            if (ny, nx) == to_pos:
                                return path + [(ny, nx)]
                            visited.add((ny, nx))
                            queue.append(((ny, nx), path + [(ny, nx)]))
            return None
        
        # Get graph path
        room_to_node = getattr(current_dungeon, 'room_to_node', {})
        node_to_room = {v: k for k, v in room_to_node.items()}
        graph = current_dungeon.graph
        
        start_room = get_room(start)
        goal_room = get_room(goal)
        start_node = room_to_node.get(start_room)
        goal_node = room_to_node.get(goal_room)
        
        if start_node is None or goal_node is None:
            return False, [], 0
        
        try:
            graph_path = nx.shortest_path(graph, start_node, goal_node)
        except nx.NetworkXNoPath:
            return False, [], 0
        
        # Build room sequence
        room_sequence = []
        for node in graph_path:
            room = node_to_room.get(node)
            if room and room in current_dungeon.room_positions:
                room_sequence.append(room)
        
        # Build path with teleportation
        full_path = [start]
        current_pos = start
        teleports = 0
        
        for target_room in room_sequence:
            target_pos = find_entry(target_room)
            if not target_pos or current_pos == target_pos:
                continue
            
            segment = local_bfs(current_pos, target_pos)
            if segment:
                full_path.extend(segment[1:])
                current_pos = segment[-1]
            else:
                full_path.append(target_pos)
                current_pos = target_pos
                teleports += 1
        
        # Final to goal
        if current_pos != goal:
            segment = local_bfs(current_pos, goal)
            if segment:
                full_path.extend(segment[1:])
            else:
                full_path.append(goal)
                teleports += 1
        
        if full_path[-1] != goal:
            full_path.append(goal)
        
        return True, full_path, teleports

    def _hybrid_graph_grid_path(self):
        """
        Hybrid pathfinding: use graph to find room sequence, 
        then BFS within each room and teleport between disconnected clusters.
        """
        import networkx as nx
        from collections import deque
        
        current_dungeon = self.maps[self.current_map_idx]
        grid = current_dungeon.global_grid
        H, W = grid.shape
        
        # Constants
        ROOM_HEIGHT = 16
        ROOM_WIDTH = 11
        FLOOR = 1
        WALL = 2
        BLOCK = 3
        VOID = 0
        DOOR_OPEN = 10
        DOOR_LOCKED = 11
        START = 21
        TRIFORCE = 22
        KEY = 30
        STAIR = 42
        
        WALKABLE = {FLOOR, DOOR_OPEN, KEY, START, TRIFORCE, STAIR, DOOR_LOCKED}
        
        start = self.env.start_pos
        goal = self.env.goal_pos
        
        if not start or not goal:
            return False, []
        
        # Helper: find room containing position
        def get_room(pos):
            y, x = pos
            for room_pos, (ry, rx) in current_dungeon.room_positions.items():
                if ry <= y < ry + ROOM_HEIGHT and rx <= x < rx + ROOM_WIDTH:
                    return room_pos
            return None
        
        # Helper: find passable position in room (prefer stairs, then center)
        def find_entry_point(room_pos):
            if room_pos not in current_dungeon.room_positions:
                return None
            ry, rx = current_dungeon.room_positions[room_pos]
            
            # First try to find a stair
            for dy in range(ROOM_HEIGHT):
                for dx in range(ROOM_WIDTH):
                    y, x = ry + dy, rx + dx
                    if 0 <= y < H and 0 <= x < W and grid[y, x] == STAIR:
                        return (y, x)
            
            # Then try center
            cy, cx = ry + ROOM_HEIGHT // 2, rx + ROOM_WIDTH // 2
            if 0 <= cy < H and 0 <= cx < W and grid[cy, cx] in WALKABLE:
                return (cy, cx)
            
            # Then any walkable tile
            for dy in range(ROOM_HEIGHT):
                for dx in range(ROOM_WIDTH):
                    y, x = ry + dy, rx + dx
                    if 0 <= y < H and 0 <= x < W and grid[y, x] in WALKABLE:
                        return (y, x)
            return None
        
        # Helper: BFS within connected tiles (allows room-to-room through doors)
        def local_bfs(from_pos, to_pos, max_steps=5000):
            if from_pos == to_pos:
                return [from_pos]
            
            visited = {from_pos}
            queue = deque([(from_pos, [from_pos])])
            
            while queue and len(visited) < max_steps:
                pos, path = queue.popleft()
                y, x = pos
                
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and (ny, nx) not in visited:
                        tile = grid[ny, nx]
                        if tile in WALKABLE:
                            if (ny, nx) == to_pos:
                                return path + [(ny, nx)]
                            visited.add((ny, nx))
                            queue.append(((ny, nx), path + [(ny, nx)]))
            return None
        
        # Step 1: Get graph path from start room to goal room
        start_room = get_room(start)
        goal_room = get_room(goal)
        
        if not start_room or not goal_room:
            return False, []
        
        room_to_node = getattr(current_dungeon, 'room_to_node', {})
        node_to_room = {v: k for k, v in room_to_node.items()}
        
        start_node = room_to_node.get(start_room)
        goal_node = room_to_node.get(goal_room)
        
        if start_node is None or goal_node is None:
            # Fallback to direct BFS
            path = local_bfs(start, goal, max_steps=50000)
            return (True, path) if path else (False, [])
        
        # Get graph path
        graph = current_dungeon.graph
        try:
            graph_path = nx.shortest_path(graph, start_node, goal_node)
        except nx.NetworkXNoPath:
            return False, []
        
        # Step 2: Build room sequence (skip unmapped nodes)
        room_sequence = []
        for node in graph_path:
            room = node_to_room.get(node)
            if room and room in current_dungeon.room_positions:
                room_sequence.append(room)
        
        if not room_sequence:
            return False, []
        
        # Step 3: Build full path by connecting rooms
        full_path = [start]
        current_pos = start
        
        for target_room in room_sequence:
            target_pos = find_entry_point(target_room)
            if not target_pos:
                continue
            
            # Skip if we're already at this target
            if current_pos == target_pos:
                continue
            
            # Try BFS to target
            path_segment = local_bfs(current_pos, target_pos)
            
            if path_segment:
                # Connected physically - add path
                full_path.extend(path_segment[1:])
                current_pos = path_segment[-1]
            else:
                # Not connected physically - teleport!
                full_path.append(target_pos)
                current_pos = target_pos
        
        # Final segment to goal
        if current_pos != goal:
            final_segment = local_bfs(current_pos, goal)
            if final_segment:
                full_path.extend(final_segment[1:])
            else:
                # Teleport to goal as last resort
                full_path.append(goal)
        
        # Validate path ends at goal
        if full_path[-1] != goal:
            full_path.append(goal)
        
        return True, full_path
    
    def _stop_auto(self, reason: str = None):
        """Stop auto-solve mode with consistent logging and cleanup."""
        try:
            logger.debug("_stop_auto called: %s", reason)
            self.auto_mode = False
            # Keep path visible for 'path complete' (victory) but clear for other reasons
            if reason != 'path complete':
                try:
                    self.auto_path = []
                    self.auto_step_idx = 0
                except Exception:
                    pass
            # Optional: set a status message
            try:
                if reason == 'path complete':
                    self._set_message("Solution complete! Victory!")
                else:
                    self._set_message(f"Auto-solve stopped: {reason}")
            except Exception:
                pass
        except Exception:
            logger.exception("_stop_auto failed: %s", reason)

    def _auto_step(self):
        """Execute one step of auto-solve with comprehensive error handling."""
        # Helper to consistently stop auto-mode with a log (local wrapper)
        def _stop_auto_local(reason: str = None):
            try:
                logger.debug("_stop_auto_local calling _stop_auto: %s", reason)
                self._stop_auto(reason)
            except Exception:
                logger.exception("_stop_auto_local failed: %s", reason)

        try:
            # Entry instrumentation
            try:
                logger.debug("_auto_step entry: auto_mode=%s auto_step_idx=%s path_len=%s", getattr(self, 'auto_mode', None), getattr(self, 'auto_step_idx', None), len(getattr(self, 'auto_path', []) if getattr(self, 'auto_path', None) else []))
            except Exception:
                logger.debug("_auto_step entry: failed to read entry state")

            # Validate auto-mode state
            if not self.auto_mode:
                logger.debug("_auto_step: auto_mode disabled; returning")
                return
            
            if not hasattr(self, 'auto_path') or not self.auto_path:
                logger.warning("_auto_step: No solution path available (path empty or missing)")
                self._show_error("No solution path available")
                self.auto_mode = False
                return
            
            if self.auto_step_idx >= len(self.auto_path) - 1:
                logger.info(f"_auto_step: Path complete at index {self.auto_step_idx}/{len(self.auto_path)-1}")
                _stop_auto_local('path complete')
                self._set_message("Solution complete!")
                self.status_message = "Completed"
                return

            # D* Lite replanning check (if active)
            if self.feature_flags.get('dstar_lite', False) and getattr(self, 'dstar_active', False) and getattr(self, 'dstar_solver', None):
                try:
                    current_state = self.env.get_state() if hasattr(self.env, 'get_state') else self.env.state
                    if self.dstar_solver.needs_replan(current_state):
                        success, new_path, updated = self.dstar_solver.replan(current_state)
                        if success and new_path:
                            # Align auto_step_idx to current position in new path
                            curpos = self.env.state.position
                            try:
                                idx = new_path.index(curpos)
                            except ValueError:
                                idx = 0
                            self.auto_path = new_path
                            self.auto_step_idx = idx
                            self._set_message(f"D* Lite replanned ({updated} updates)")
                except Exception as e:
                    logger.warning(f"D* Lite replanning failed: {e}")

            
            # Validate environment
            if self.env is None:
                self._show_error("Environment not initialized")
                _stop_auto_local('env none')
                return
            
            if not hasattr(self.env, 'state') or self.env.state is None:
                self._show_error("Invalid environment state")
                _stop_auto_local('env.state invalid')
                return

            # Advance to next step and compute direction
            self.auto_step_idx += 1
            target = self.auto_path[self.auto_step_idx]
            current = self.env.state.position
            dr = target[0] - current[0]
            dc = target[1] - current[1]

            # If any background thread requested inventory refresh, handle it now
            if getattr(self, 'inventory_needs_refresh', False):
                try:
                    self._update_inventory_and_hud()
                except Exception:
                    pass
                finally:
                    self.inventory_needs_refresh = False

            # Teleport - directly set position (non-adjacent move)
            if abs(dr) > 1 or abs(dc) > 1:
                old_state = GameState(
                    position=self.env.state.position,
                    keys=self.env.state.keys,
                    bomb_count=self.env.state.bomb_count,
                    has_boss_key=self.env.state.has_boss_key,
                    opened_doors=self.env.state.opened_doors.copy() if hasattr(self.env.state.opened_doors, 'copy') else set(self.env.state.opened_doors),
                    collected_items=self.env.state.collected_items.copy() if hasattr(self.env.state.collected_items, 'copy') else set(self.env.state.collected_items)
                )

                self.env.state.position = target
                self._set_message(f"Teleport! {current} -> {target}")
                self.status_message = "Teleporting..."

                # Apply pickup at teleport destination (some teleports land on items)
                try:
                    self._apply_pickup_at(target)
                except Exception as e:
                    logger.warning(f"Pickup application failed on teleport: {e}")

                # Track item changes (with error handling)
                try:
                    self._track_item_collection(old_state, self.env.state)
                    self._track_item_usage(old_state, self.env.state)
                    # Ensure counters are immediately reconciled for UI
                    try:
                        self._sync_inventory_counters()
                    except Exception:
                        pass
                except Exception as e:
                    logger.warning(f"Item tracking failed: {e}")

                # Update visual position (instant for teleport)
                if self.renderer:
                    try:
                        self.renderer.set_agent_position(target[0], target[1], immediate=True)
                    except Exception as e:
                        logger.warning(f"Renderer update failed: {e}")

                if self.effects:
                    try:
                        # Add ripple effect at teleport destination (grid coordinates)
                        self.effects.add_effect(RippleEffect(target, (100, 200, 255)))
                    except Exception as e:
                        logger.warning(f"Effect creation failed: {e}")

                # Check if reached goal
                if target == self.env.goal_pos:
                    self.env.won = True
                    self.env.done = True
                    self.auto_mode = False
                    self._set_message("AUTO-SOLVE: Victory!")
                    self.status_message = "Victory!"
                return

            # Normal move - capture old state
            old_state = GameState(
                position=self.env.state.position,
                keys=self.env.state.keys,
                bomb_count=self.env.state.bomb_count,
                has_boss_key=self.env.state.has_boss_key,
                opened_doors=self.env.state.opened_doors.copy() if hasattr(self.env.state.opened_doors, 'copy') else set(self.env.state.opened_doors),
                collected_items=self.env.state.collected_items.copy() if hasattr(self.env.state.collected_items, 'copy') else set(self.env.state.collected_items)
            )

            if dr == -1:
                action = Action.UP
            elif dr == 1:
                action = Action.DOWN
            elif dc == -1:
                action = Action.LEFT
            else:
                action = Action.RIGHT
            
            # Check for block push BEFORE env.step (for animation purposes)
            try:
                self._check_and_start_block_push(current, target, action)
            except Exception as e:
                logger.warning("Block push check failed: %s", e)
            
            logger.debug("_auto_step: performing env.step action=%r (int=%s) target=%s current=%s", action, int(action), target, current)
            state, reward, done, info = self.env.step(int(action))
            logger.debug("_auto_step: env.step returned info=%r, new_pos=%s, env.keys=%s", info, getattr(self.env.state, 'position', None), getattr(self.env.state, 'keys', None))
            
            # Get new position immediately after step
            new_pos = self.env.state.position
            
            # Increment step counter
            self.step_count += 1
            
            # Track item collection and usage (single call, not duplicated)
            self._track_item_collection(old_state, self.env.state)
            self._track_item_usage(old_state, self.env.state)
            
            # Update modern HUD during auto-solve with collected counts
            if self.modern_hud:
                self.modern_hud.update_game_state(
                    keys=self.env.state.keys,
                    bombs=self.env.state.bomb_count,
                    has_boss_key=self.env.state.has_boss_key,
                    position=new_pos,
                    steps=self.step_count,
                    message=getattr(self, 'message', '')
                )
                # Sync counters and update inventory display with collection counts
                self._sync_inventory_counters()
                if hasattr(self.modern_hud, 'keys_collected'):
                    self.modern_hud.keys_collected = self.keys_collected
                    self.modern_hud.bombs_collected = self.bombs_collected
                    self.modern_hud.boss_keys_collected = self.boss_keys_collected
                # Update usage counters for modern HUD if supported
                if hasattr(self.modern_hud, 'keys_used'):
                    self.modern_hud.keys_used = getattr(self, 'keys_used', 0)
                if hasattr(self.modern_hud, 'bombs_used'):
                    self.modern_hud.bombs_used = getattr(self, 'bombs_used', 0)
                if hasattr(self.modern_hud, 'boss_keys_used'):
                    self.modern_hud.boss_keys_used = getattr(self, 'boss_keys_used', 0)
            
            # Update visual position (smooth)
            if self.renderer:
                self.renderer.set_agent_position(new_pos[0], new_pos[1], immediate=False)
            
            # Check if done (NOT indented under renderer)
            if done:
                self.auto_mode = False
                if self.env.won:
                    self._set_message("AUTO-SOLVE: Victory!")
                    self.status_message = "Victory!"
                    if self.effects:
                        try:
                            # Victory flash effect at goal position
                            goal_pos = self.env.goal_pos
                            self.effects.add_effect(FlashEffect(goal_pos, (255, 215, 0), 0.5))
                        except Exception as e:
                            logger.warning(f"Victory effect failed: {e}")
                else:
                    self._set_message(f"AUTO-SOLVE: Failed - {info.get('msg', '')}")
                    self.status_message = "Failed"
        
        except KeyError as e:
            self._show_error(f"State access error: {str(e)}")
            _stop_auto_local('KeyError')
        except IndexError as e:
            self._show_error(f"Path index error: {str(e)}")
            _stop_auto_local('IndexError')
        except AttributeError as e:
            logger.exception("Auto-step AttributeError caught: %s", e)
            self._show_error(f"Invalid state attribute: {str(e)}")
            _stop_auto_local('AttributeError')
        except Exception as e:
            self._show_error(f"Auto-solve error: {str(e)}")
            self.auto_mode = False
            import traceback
            traceback.print_exc()
    
    def _show_error(self, message: str):
        """Display error message to user with visual feedback."""
        logger.error(message)
        self.error_message = message
        self.error_time = time.time()
        self.status_message = "Error"
    
    def _show_message(self, message: str, duration: float = 3.0):
        """Display informational message to user."""
        logger.info(message)
        self.message = message
        self.message_time = time.time()
        self.message_duration = duration
        self.status_message = "Info"

    # --- Topology helpers ---
    def _export_topology(self):
        """Export current map topology to a DOT file (if available)."""
        current = self.maps[self.current_map_idx]
        graph = getattr(current, 'graph', None)
        room_positions = getattr(current, 'room_positions', None)
        if graph is None:
            self._set_message('No topology graph available for this map', 3.0)
            return
        # Try to use networkx pydot writer
        try:
            import networkx as nx
            try:
                fname = f"topology_map_{self.current_map_idx+1}.dot"
                nx.nx_pydot.write_dot(graph, fname)
                # Add node positions as comments if available
                if room_positions:
                    with open(fname, 'a') as f:
                        f.write('\n// room positions\n')
                        for room, (ry, rx) in room_positions.items():
                            f.write(f"// {room}: {ry},{rx}\n")
                self._set_message(f"Topology exported to {fname}")
                self.topology_export_path = fname
            except Exception as e:
                # Fallback to manual DOT generation
                fname = f"topology_map_{self.current_map_idx+1}.dot"
                with open(fname, 'w', encoding='utf-8') as f:
                    f.write('graph topology {\n')
                    for n in graph.nodes():
                        f.write(f'  "{n}";\n')
                    for u, v in graph.edges():
                        f.write(f'  "{u}" -- "{v}";\n')
                    f.write('}\n')
                self._set_message(f"Topology exported to {fname} (manual)\n{e}")
                self.topology_export_path = fname
        except ImportError:
            self._set_message('NetworkX not available - cannot export DOT automatically', 4.0)

    def _render_topology_overlay(self, surface: pygame.Surface):
        """Draw room nodes and edges on the map area with high-visibility styling.

        Uses `room_to_node` mapping to place graph node ids on the stitched room positions.
        """
        current = self.maps[self.current_map_idx]
        if not hasattr(current, 'graph') or not current.graph:
            return
        graph = current.graph
        room_positions = getattr(current, 'room_positions', {})
        room_to_node = getattr(current, 'room_to_node', {})
        node_to_room = {v: k for k, v in room_to_node.items()} if room_to_node else {}

        # Prepare an alpha surface for glow effects
        try:
            overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        except Exception:
            overlay = None

        # Precompute pixel positions for nodes (by node id)
        node_pos = {}
        unmatched_nodes = 0
        unmatched_node_ids = []
        for node in graph.nodes():
            room_pos = node_to_room.get(node)
            if room_pos is None:
                unmatched_nodes += 1
                unmatched_node_ids.append(node)
                continue
            rp = room_positions.get(room_pos)
            if not rp:
                unmatched_nodes += 1
                unmatched_node_ids.append(node)
                continue
            ry, rx = rp
            cx = rx * self.TILE_SIZE + self.TILE_SIZE * 0.5 - self.view_offset_x
            cy = ry * self.TILE_SIZE + self.TILE_SIZE * 0.5 - self.view_offset_y
            node_pos[node] = (cx, cy)

        # --- DRAW EDGES ---
        # Edge colors based on type (if edge has 'type' attribute)
        edge_colors = {
            'open': (100, 255, 100, 180),        # Green - normal passage
            'key_locked': (255, 220, 100, 200),  # Yellow - key door
            'bombable': (255, 150, 50, 200),     # Orange - bomb wall
            'soft_locked': (180, 100, 255, 180), # Purple - one-way
            'stair': (100, 200, 255, 200),       # Cyan - teleport/stair
        }
        default_edge_color = (150, 150, 200, 150)

        target_surface = overlay if overlay else surface
        for u, v, data in graph.edges(data=True):
            if u not in node_pos or v not in node_pos:
                continue
            x1, y1 = node_pos[u]
            x2, y2 = node_pos[v]
            # Get edge type for coloring
            edge_type = data.get('type', 'open') if data else 'open'
            color = edge_colors.get(edge_type, default_edge_color)
            # Draw edge line
            try:
                pygame.draw.line(target_surface, color[:3], (int(x1), int(y1)), (int(x2), int(y2)), 3)
            except Exception:
                pygame.draw.line(surface, color[:3], (int(x1), int(y1)), (int(x2), int(y2)), 3)

        # --- DRAW NODES ---
        node_radius = max(8, self.TILE_SIZE // 3)
        font = pygame.font.SysFont('Arial', 12, bold=True)
        for node, (cx, cy) in node_pos.items():
            # Draw glow/outline
            try:
                pygame.draw.circle(target_surface, (255, 255, 255, 100), (int(cx), int(cy)), node_radius + 3)
            except Exception:
                pass
            # Draw filled circle
            pygame.draw.circle(target_surface, (80, 120, 200), (int(cx), int(cy)), node_radius)
            pygame.draw.circle(target_surface, (150, 200, 255), (int(cx), int(cy)), node_radius, 2)
            # Draw node label
            try:
                label = font.render(str(node), True, (255, 255, 255))
                lx = int(cx - label.get_width() / 2)
                ly = int(cy - label.get_height() / 2)
                target_surface.blit(label, (lx, ly))
            except Exception:
                pass

        # Blit overlay if we used one
        if overlay:
            surface.blit(overlay, (0, 0))

        # Show unmatched node warning if any
        if unmatched_nodes > 0:
            try:
                warn_font = pygame.font.SysFont('Arial', 14, bold=True)
                warn_text = warn_font.render(f'{unmatched_nodes} unmatched nodes', True, (255, 150, 100))
                surface.blit(warn_text, (10, 10))
            except Exception:
                pass

    def _match_missing_nodes(self):
        """Attempt to infer and stage mapping proposals for unmatched nodes.

        Uses RoomGraphMatcher.infer_missing_mappings to generate proposals with confidences.
        High-confidence proposals (>= configured threshold) are applied automatically.
        Lower confidence proposals are kept as 'tentative' in `current.match_proposals` for manual apply.
        """
        current = self.maps[self.current_map_idx]
        graph = getattr(current, 'graph', None)
        rooms = getattr(current, 'rooms', None)
        room_positions = getattr(current, 'room_positions', None)
        room_to_node = dict(getattr(current, 'room_to_node', {}) or {})

        if graph is None or rooms is None:
            self._set_message('No topology available for this map', 3.0)
            return

        matcher = RoomGraphMatcher()
        proposed_r2n, proposed_n2r, confidences = matcher.infer_missing_mappings(
            rooms, graph, room_positions=room_positions, room_to_node=room_to_node)

        if not proposed_r2n:
            self._set_message('No proposals for missing nodes', 3.0)
            return

        # Save proposals for later apply/preview
        current.match_proposals = (proposed_r2n, proposed_n2r, confidences)

        # Determine configured threshold
        threshold_widget = next((w for w in self.widget_manager.widgets if getattr(w, 'control_name', None) == 'match_threshold'), None)
        try:
            threshold_val = float(threshold_widget.options[threshold_widget.selected]) if threshold_widget else getattr(self, 'match_apply_threshold', 0.85)
        except Exception:
            threshold_val = getattr(self, 'match_apply_threshold', 0.85)

        # Split by confidence
        confident = {n: r for r, n in proposed_r2n.items() if confidences.get(n, 0) >= threshold_val}
        tentative = {n: r for r, n in proposed_r2n.items() if confidences.get(n, 0) < threshold_val}

        applied = 0
        if confident:
            # Save undo snapshot
            snapshot = dict(getattr(current, 'room_to_node', {}) or {})
            applied_nodes = list(confident.items())
            self.match_undo_stack.append(snapshot)

            # Apply confident matches
            for room_pos, node_id in confident.items():
                current.room_to_node[room_pos] = node_id
                current.rooms[room_pos].graph_node_id = node_id
                applied += 1

            self._set_message(f'Applied {applied} confident matches. {len(tentative)} tentative remain', 4.0)
            logger.info('Applied matches: %s', applied_nodes)

        else:
            self._set_message(f'No matches above threshold ({threshold_val}). {len(proposed_r2n)} proposals available', 5.0)
            logger.info('Match proposals (none auto-applied):')
            for node, room in proposed_n2r.items():
                logger.info('  Node %s -> Room %s (conf=%.2f)', node, room, confidences.get(node, 0.0))

    def _undo_last_match(self):
        """Undo last applied match snapshot, if any."""
        current = self.maps[self.current_map_idx]
        if not self.match_undo_stack:
            self._set_message('Nothing to undo', 2.0)
            return

        snapshot = self.match_undo_stack.pop()
        # Restore room_to_node and room.graph_node_id
        current.room_to_node = dict(snapshot)
        for rpos, room in current.rooms.items():
            room.graph_node_id = current.room_to_node.get(rpos)

        # Clear staged proposals (they may be invalid now)
        if hasattr(current, 'match_proposals'):
            del current.match_proposals

        self._set_message('Undo: restored previous mapping', 3.0)
        logger.info('Undo applied: restored previous mapping')

    def _undo_prune(self):
        """Undo the last applied prune snapshot, if any."""
        if not hasattr(self, '_precheck_snapshot') or not self._precheck_snapshot:
            self._set_message('No prune snapshot to undo', 2.0)
            return
        current = self.maps[self.current_map_idx]
        snap = self._precheck_snapshot
        current.rooms = dict(snap['rooms'])
        current.room_to_node = dict(snap.get('room_to_node', {}))
        for pos, room in current.rooms.items():
            room.graph_node_id = current.room_to_node.get(pos)
        # Clear snapshot
        self._precheck_snapshot = None
        self._set_message('Undo: restored rooms before pruning', 3.0)
        logger.info('Undo prune: restored previous rooms and mapping')

    def _apply_tentative_matches(self):
        """Apply staged tentative matches above the configured threshold."""
        current = self.maps[self.current_map_idx]
        if not hasattr(current, 'match_proposals'):
            self._set_message('No staged proposals to apply', 2.0)
            return

        proposed_r2n, proposed_n2r, confidences = current.match_proposals

        threshold_widget = next((w for w in self.widget_manager.widgets if getattr(w, 'control_name', None) == 'match_threshold'), None)
        try:
            threshold_val = float(threshold_widget.options[threshold_widget.selected]) if threshold_widget else getattr(self, 'match_apply_threshold', 0.85)
        except Exception:
            threshold_val = getattr(self, 'match_apply_threshold', 0.85)

        to_apply = {n: r for r, n in proposed_r2n.items() if confidences.get(n, 0) >= threshold_val}

        if not to_apply:
            self._set_message('No proposals meet threshold', 2.0)
            return

        # Snapshot and apply
        snapshot = dict(getattr(current, 'room_to_node', {}) or {})
        self.match_undo_stack.append(snapshot)
        applied = 0
        for room_pos, node_id in to_apply.items():
            current.room_to_node[room_pos] = node_id
            current.rooms[room_pos].graph_node_id = node_id
            applied += 1

        # Remove applied from staged proposals
        for node_id in list(to_apply.values()):
            proposed_n2r.pop(node_id, None)
        for room_pos in list(to_apply.keys()):
            proposed_r2n.pop(room_pos, None)

        if not proposed_r2n:
            del current.match_proposals

        self._set_message(f'Applied {applied} tentative matches', 3.0)
        logger.info('Applied tentative matches: %d', applied)

        # Color maps for inline markers
        marker_color_map = {
            's': (80, 220, 80), 'k': (255, 215, 0), 'K': (200, 160, 0),
            'b': (220, 40, 40), 'e': (200, 60, 60), 'I': (160, 40, 200),
            'p': (80, 160, 255), 't': (255, 200, 60)
        }
        edge_color_map = {'S': (40, 200, 255), 'b': (220, 40, 40), 'k': (255, 180, 60), 'l': (160, 80, 200), 's': (100, 100, 100)}

        # Draw edges (shadow + bright center line) between placed nodes and inline labels
        for u, v in graph.edges():
            pu = node_pos.get(u)
            pv = node_pos.get(v)
            if not pu or not pv:
                continue
            try:
                if overlay:
                    pygame.draw.line(overlay, (20, 40, 60, 220), pu, pv, max(6, int(self.TILE_SIZE * 0.2)))
                    pygame.draw.line(overlay, (40, 200, 255, 220), pu, pv, max(3, int(self.TILE_SIZE * 0.12)))
                else:
                    pygame.draw.line(surface, (20, 40, 60), pu, pv, max(6, int(self.TILE_SIZE * 0.2)))
                    pygame.draw.line(surface, (40, 200, 255), pu, pv, max(3, int(self.TILE_SIZE * 0.12)))

                # Inline edge label: midpoint marker with letter (if edge has label)
                try:
                    ed = graph.get_edge_data(u, v) or {}
                    label_char = ed.get('label') or ed.get('edge_type', '')
                    if label_char:
                        # If label is long (edge_type like 'key_locked'), prefer the first letter key if present
                        if len(label_char) > 1 and label_char in edge_color_map:
                            ch = label_char
                        else:
                            # Some DOT edges use single-char labels like 'k', 'S', 'l'
                            ch = label_char if len(label_char) == 1 else None
                        if ch:
                            mx = (pu[0] + pv[0]) / 2
                            my = (pu[1] + pv[1]) / 2
                            bg = edge_color_map.get(ch, (80, 80, 120))
                            # Draw a small rectangle with the letter
                            rect_w = max(12, int(self.TILE_SIZE * 0.6))
                            rect_h = max(12, int(self.TILE_SIZE * 0.4))
                            rx = int(mx - rect_w / 2)
                            ry = int(my - rect_h / 2)
                            pygame.draw.rect(surface, bg, (rx, ry, rect_w, rect_h))
                            pygame.draw.rect(surface, (10, 10, 10), (rx, ry, rect_w, rect_h), 1)
                            tf = pygame.font.SysFont('Arial', max(10, int(rect_h * 0.6)), bold=True)
                            label_s = tf.render(str(ch), True, (255, 255, 255))
                            surface.blit(label_s, (rx + rect_w//2 - label_s.get_width()//2, ry + rect_h//2 - label_s.get_height()//2))
                except Exception:
                    pass
            except Exception:
                pass

        # Draw nodes with halo, fill, outline, and labeled badge
        hovered_node = None
        hovered_edge = None
        mouse_pos = pygame.mouse.get_pos() if pygame.get_init() else (0, 0)
        for node, (cx, cy) in node_pos.items():
            radius = max(6, int(self.TILE_SIZE * 0.35))
            try:
                if overlay:
                    pygame.draw.circle(overlay, (40, 200, 255, 90), (int(cx), int(cy)), int(radius * 1.6))
                pygame.draw.circle(surface, (255, 160, 60), (int(cx), int(cy)), radius)
                pygame.draw.circle(surface, (15, 15, 15), (int(cx), int(cy)), radius, 2)

                # Node label (node id)
                font = pygame.font.SysFont('Arial', max(12, int(self.TILE_SIZE // 5)), bold=True)
                label = str(node)
                txt = font.render(label, True, (255, 255, 255))
                outline = font.render(label, True, (10, 10, 10))
                ox = int(cx) - txt.get_width() // 2
                oy = int(cy) - txt.get_height() // 2
                surface.blit(outline, (ox - 1, oy))
                surface.blit(outline, (ox + 1, oy))
                surface.blit(outline, (ox, oy - 1))
                surface.blit(outline, (ox, oy + 1))
                surface.blit(txt, (ox, oy))

                # Semantic inline marker (small colored circle with char)
                try:
                    nattr = graph.nodes[node] if hasattr(graph, 'nodes') else {}
                    label_char = nattr.get('label') if isinstance(nattr, dict) else None
                    # Prefer single-character label, else derive from flags
                    if not label_char or len(label_char) != 1:
                        if nattr.get('is_start'):
                            label_char = 's'
                        elif nattr.get('is_triforce'):
                            label_char = 't'
                        elif nattr.get('is_boss'):
                            label_char = 'b'
                        elif nattr.get('has_key'):
                            label_char = 'k'
                        elif nattr.get('has_item'):
                            label_char = 'I'
                        elif nattr.get('has_enemy'):
                            label_char = 'e'
                        elif nattr.get('has_puzzle'):
                            label_char = 'p'
                        else:
                            label_char = None
                    if label_char:
                        mc = marker_color_map.get(label_char, (120, 120, 120))
                        mx = int(cx - radius - 8)
                        my = int(cy - radius - 8)
                        # Draw a distinctive Triforce icon for 't' (three golden triangles)
                        if label_char == 't':
                            tri_size = max(10, int(radius * 0.9))
                            # center of triforce is (mx,my)
                            cx0, cy0 = mx, my
                            # top triangle
                            top = [
                                (cx0, cy0 - tri_size // 2 - 2),
                                (cx0 - tri_size // 2, cy0 + tri_size // 2),
                                (cx0 + tri_size // 2, cy0 + tri_size // 2)
                            ]
                            # left triangle (shifted left-bottom)
                            left = [
                                (cx0 - tri_size // 2, cy0 + tri_size // 2),
                                (cx0 - tri_size, cy0 + tri_size // 2 + tri_size // 2),
                                (cx0, cy0 + tri_size // 2 + tri_size // 2)
                            ]
                            # right triangle
                            right = [
                                (cx0 + tri_size // 2, cy0 + tri_size // 2),
                                (cx0, cy0 + tri_size // 2 + tri_size // 2),
                                (cx0 + tri_size, cy0 + tri_size // 2 + tri_size // 2)
                            ]
                            pygame.draw.polygon(surface, (255, 215, 0), top)
                            pygame.draw.polygon(surface, (255, 215, 0), left)
                            pygame.draw.polygon(surface, (255, 215, 0), right)
                            pygame.draw.polygon(surface, (10, 10, 10), top, 1)
                            pygame.draw.polygon(surface, (10, 10, 10), left, 1)
                            pygame.draw.polygon(surface, (10, 10, 10), right, 1)
                        else:
                            pygame.draw.circle(surface, mc, (mx, my), max(6, int(radius * 0.5)))
                            tf = pygame.font.SysFont('Arial', max(10, int(radius * 0.6)), bold=True)
                            chs = tf.render(label_char, True, (255, 255, 255))
                            surface.blit(chs, (mx - chs.get_width()//2, my - chs.get_height()//2))
                except Exception:
                    pass

                # Degree badge
                try:
                    degree = graph.degree[node] if hasattr(graph, 'degree') else sum(1 for a, b in graph.edges() if a == node or b == node)
                except Exception:
                    degree = 0
                badge_radius = max(6, int(radius * 0.5))
                bx = int(cx + radius * 0.7)
                by = int(cy - radius * 0.7)
                pygame.draw.circle(surface, (255, 80, 80), (bx, by), badge_radius)
                pygame.draw.circle(surface, (20, 20, 20), (bx, by), badge_radius, 1)
                bfont = pygame.font.SysFont('Arial', max(10, int(badge_radius * 1.0)), bold=True)
                btxt = bfont.render(str(degree), True, (255, 255, 255))
                surface.blit(btxt, (bx - btxt.get_width() // 2, by - btxt.get_height() // 2))

                # Hover detection (node takes precedence)
                dx = mouse_pos[0] - cx
                dy = mouse_pos[1] - cy
                if dx*dx + dy*dy <= (radius * 1.4) ** 2:
                    hovered_node = node
            except Exception:
                pass

        # Edge hover detection (if no node hovered)
        if not hovered_node:
            def point_segment_distance(p, a, b):
                # p,a,b are (x,y)
                px, py = p
                ax, ay = a
                bx, by = b
                dx = bx - ax
                dy = by - ay
                if dx == 0 and dy == 0:
                    return ((px-ax)**2 + (py-ay)**2) ** 0.5
                t = ((px-ax) * dx + (py-ay) * dy) / (dx*dx + dy*dy)
                t = max(0.0, min(1.0, t))
                projx = ax + t * dx
                projy = ay + t * dy
                return ((px-projx)**2 + (py-projy)**2) ** 0.5

            for u, v in graph.edges():
                pu = node_pos.get(u)
                pv = node_pos.get(v)
                if not pu or not pv:
                    continue
                dist = point_segment_distance(mouse_pos, pu, pv)
                if dist <= max(8, int(self.TILE_SIZE * 0.12)):
                    hovered_edge = (u, v, graph.get_edge_data(u, v).get('label', ''))
                    break

        # Blit the overlay after drawing all glow elements
        if overlay:
            try:
                surface.blit(overlay, (0, 0))
            except Exception:
                pass

        # Small legend to explain the overlay (non-intrusive)
        try:
            # Base legend
            lines = [
                "Topology legend:",
                "Nodes = graph nodes (orange)",
                "Badge = neighbor count",
                "Edges = connectivity (cyan)"
            ]
            if unmatched_nodes:
                # Show a short list of node ids for debugging
                ids_preview = ', '.join(str(n) for n in (unmatched_node_ids[:8] if 'unmatched_node_ids' in locals() else []))
                lines.append(f"Unplaced nodes: {unmatched_nodes} [{ids_preview}]")
                # Also log for diagnostics
                try:
                    logger.info(f"Topology unplaced nodes: {unmatched_node_ids}")
                except Exception:
                    pass

            # If the user toggled the detailed legend, include the JSON-style mapping
            if getattr(self, 'show_topology_legend', False):
                lines.append("")
                lines.append("Node semantics:")
                for k, v in self.topology_semantics.get('nodes', {}).items():
                    lines.append(f"{k}: {', '.join(v)}")
                lines.append("")
                lines.append("Edge semantics:")
                for k, v in self.topology_semantics.get('edges', {}).items():
                    lines.append(f"{k}: {', '.join(v)}")

            lf = pygame.font.SysFont('Arial', 12)
            lw = 360
            lh = 6 + 18 * len(lines)
            # Clamp legend size
            lw = min(lw, self.screen_w - 20)
            lh = min(lh, self.screen_h - 20)

            legend_surf = pygame.Surface((lw, lh), pygame.SRCALPHA)
            legend_surf.fill((10, 10, 10, 200))

            # Color maps for small markers
            marker_color_map = {
                's': (80, 220, 80), 'k': (255, 215, 0), 'K': (200, 160, 0),
                'b': (220, 40, 40), 'e': (200, 60, 60), 'I': (160, 40, 200),
                'p': (80, 160, 255), 't': (255, 200, 60)
            }
            edge_color_map = {'S': (40, 200, 255), 'b': (220, 40, 40), 'k': (255, 180, 60), 'l': (160, 80, 200), 's': (100, 100, 100)}

            y = 6
            for i, l in enumerate(lines):
                # If mapping line like 'k: room, key' try to draw a small color marker
                m = None
                if ':' in l and len(l) >= 3:
                    key = l.split(':', 1)[0].strip()
                    if key in marker_color_map:
                        m = ('node', key, marker_color_map[key])
                    elif key in edge_color_map:
                        m = ('edge', key, edge_color_map[key])
                if m:
                    # Draw marker circle or square then text
                    if m[0] == 'node':
                        # Draw Triforce icon for 't' in legend, otherwise circle
                        if m[1] == 't':
                            cx_l = 8 + 8
                            cy_l = y + 9
                            ts = 8
                            top = [(cx_l, cy_l - ts//2 - 1), (cx_l - ts//2, cy_l + ts//2), (cx_l + ts//2, cy_l + ts//2)]
                            left = [(cx_l - ts//2, cy_l + ts//2), (cx_l - ts, cy_l + ts//2 + ts//2), (cx_l, cy_l + ts//2 + ts//2)]
                            right = [(cx_l + ts//2, cy_l + ts//2), (cx_l, cy_l + ts//2 + ts//2), (cx_l + ts, cy_l + ts//2 + ts//2)]
                            pygame.draw.polygon(legend_surf, (255, 215, 0), top)
                            pygame.draw.polygon(legend_surf, (255, 215, 0), left)
                            pygame.draw.polygon(legend_surf, (255, 215, 0), right)
                            pygame.draw.polygon(legend_surf, (10, 10, 10), top, 1)
                            pygame.draw.polygon(legend_surf, (10, 10, 10), left, 1)
                            pygame.draw.polygon(legend_surf, (10, 10, 10), right, 1)
                        else:
                            pygame.draw.circle(legend_surf, m[2], (8 + 6, y + 9), 6)
                    else:
                        pygame.draw.rect(legend_surf, m[2], (8, y + 3, 12, 12))
                    txt = lf.render(l, True, (230, 230, 230))
                    legend_surf.blit(txt, (28, y))
                else:
                    txt = lf.render(l, True, (230, 230, 230))
                    legend_surf.blit(txt, (8, y))
                y += 18

            surface.blit(legend_surf, (10, 10))
        except Exception:
            pass

        # Draw hover tooltip for node/edge if applicable
        try:
            if hovered_node is not None or hovered_edge is not None:
                tip_lines = []
                if hovered_node is not None:
                    node = hovered_node
                    tip_lines.append(f"Node: {node}")
                    nattr = graph.nodes[node]
                    # Prefer single-char label if available
                    label_char = nattr.get('label') or nattr.get('label', '')
                    # If label is like 's' or 'e' use semantics mapping
                    if label_char and label_char in self.topology_semantics.get('nodes', {}):
                        tip_lines.append(f"Type: {', '.join(self.topology_semantics['nodes'][label_char])}")
                    else:
                        # Fallback: infer flags set on node
                        flags = []
                        for k in ['is_start', 'is_triforce', 'is_boss', 'has_key', 'has_item', 'has_enemy', 'has_puzzle']:
                            if nattr.get(k):
                                flags.append(k)
                        if flags:
                            tip_lines.append("Flags: " + ", ".join(flags))
                else:
                    u, v, label_char = hovered_edge
                    tip_lines.append(f"Edge: {u} -> {v}")
                    if label_char and label_char in self.topology_semantics.get('edges', {}):
                        tip_lines.append(f"Type: {', '.join(self.topology_semantics['edges'][label_char])}")
                    else:
                        # Edge attributes fallback
                        ed = graph.get_edge_data(u, v) or {}
                        if 'edge_type' in ed:
                            tip_lines.append(f"Type: {ed['edge_type']}")

                # Render tooltip box near mouse
                mxx, myy = mouse_pos
                pad = 6
                tf = pygame.font.SysFont('Arial', 12)
                tw = max(tf.render(t, True, (255,255,255)).get_width() for t in tip_lines) + pad * 2
                th = (len(tip_lines) * 18) + pad * 2
                tx = mxx + 12
                ty = myy + 12
                # Clamp inside screen
                if tx + tw > self.screen_w - 10:
                    tx = self.screen_w - tw - 10
                if ty + th > self.screen_h - 10:
                    ty = self.screen_h - th - 10
                tip_surf = pygame.Surface((tw, th), pygame.SRCALPHA)
                tip_surf.fill((12, 12, 12, 220))
                pygame.draw.rect(tip_surf, (200,200,200), (0, 0, tw, th), 1)
                for i, t in enumerate(tip_lines):
                    t_s = tf.render(t, True, (230, 230, 230))
                    tip_surf.blit(t_s, (pad, pad + i * 18))
                surface.blit(tip_surf, (tx, ty))
        except Exception:
            pass

    # --- Solver comparison helpers ---
    def _set_last_solver_metrics(self, name, nodes, time_ms, path_len):
        self.last_solver_metrics = {
            'name': name,
            'nodes': nodes,
            'time_ms': time_ms,
            'path_len': path_len
        }

    def _run_solver_comparison(self):
        """Start an asynchronous solver comparison worker to avoid blocking the GUI."""
        # If a comparison is already running, notify user
        if getattr(self, 'solver_comparison_thread', None) and self.solver_comparison_thread.is_alive():
            self._show_toast('Solver comparison already running', 2.5, 'warning')
            return

        def _worker():
            results = []
            alg_names = ['A*', 'BFS', 'Dijkstra', 'Greedy', 'D* Lite', 'StateSpace',
                        'CBS (Balanced)', 'CBS (Explorer)', 'CBS (Cautious)']
            for idx, name in enumerate(alg_names):
                start_t = time.time()
                # D* Lite special-case: use D* Lite implementation
                if name == 'D* Lite':
                    try:
                        from src.simulation.dstar_lite import DStarLiteSolver
                        start_state = GameState(position=self.env.start_pos, opened_doors=self.env.state.opened_doors.copy() if hasattr(self.env, 'state') else set())
                        ds = DStarLiteSolver(self.env)
                        success, path, nodes = ds.solve(start_state)
                        elapsed = (time.time() - start_t) * 1000
                        results.append({'name': name, 'success': success, 'path_len': len(path), 'nodes': nodes, 'time_ms': elapsed})
                        if success:
                            self._set_last_solver_metrics(name, nodes, elapsed, len(path))
                        continue
                    except Exception as e:
                        results.append({'name': name, 'success': False, 'path_len': 0, 'nodes': 0, 'time_ms': 0, 'error': str(e)})
                        continue
                if name == 'StateSpace':
                    try:
                        start_state = self.env.get_state() if hasattr(self.env, 'get_state') else GameState(position=self.env.start_pos)
                        try:
                            from src.simulation.validator import StateSpaceAStar
                            temp_solver = self.solver if self.solver is not None else StateSpaceAStar(self.env)
                        except Exception:
                            temp_solver = self.solver
                        if temp_solver is None:
                            raise RuntimeError('State-space solver unavailable')
                        success, path, states = temp_solver.solve()
                        elapsed = (time.time() - start_t) * 1000
                        nodes = getattr(temp_solver, 'last_states_explored', getattr(temp_solver, 'last_states', 0))
                        results.append({'name': name, 'success': success, 'path_len': len(path), 'nodes': nodes, 'time_ms': elapsed})
                        if success and not self.last_solver_metrics:
                            self._set_last_solver_metrics(name, nodes, elapsed, len(path))
                        continue
                    except Exception as e:
                        results.append({'name': name, 'success': False, 'path_len': 0, 'nodes': 0, 'time_ms': 0, 'error': str(e)})
                        continue
                # CBS special-case: use CognitiveBoundedSearch implementation
                if 'CBS' in name:
                    try:
                        from src.simulation.cognitive_bounded_search import CognitiveBoundedSearch
                        # Extract persona from name like "CBS (Balanced)"
                        persona_map = {
                            'CBS (Balanced)': 'balanced',
                            'CBS (Explorer)': 'explorer',
                            'CBS (Cautious)': 'cautious',
                            'CBS (Forgetful)': 'forgetful',
                            'CBS (Speedrunner)': 'speedrunner',
                            'CBS (Greedy)': 'greedy'
                        }
                        persona = persona_map.get(name, 'balanced')
                        cbs = CognitiveBoundedSearch(self.env, persona=persona, timeout=100000)
                        ok, path, states, metrics = cbs.solve()
                        elapsed = (time.time() - start_t) * 1000
                        if ok:
                            results.append({
                                'name': name,
                                'success': True,
                                'path_len': len(path),
                                'nodes': states,
                                'time_ms': elapsed,
                                'confusion': round(metrics.confusion_index, 3),
                                'cog_load': round(metrics.cognitive_load, 3)
                            })
                            if not self.last_solver_metrics:
                                self._set_last_solver_metrics(name, states, elapsed, len(path))
                        else:
                            results.append({
                                'name': name,
                                'success': False,
                                'path_len': 0,
                                'nodes': states,
                                'time_ms': elapsed,
                                'confusion': 0,
                                'cog_load': 0
                            })
                        continue
                    except Exception as e:
                        results.append({'name': name, 'success': False, 'path_len': 0, 'nodes': 0, 'time_ms': 0, 'error': str(e), 'confusion': 0, 'cog_load': 0})
                        continue
                # For other algorithms use existing mechanism (set algorithm_idx temporarily)
                saved_alg = getattr(self, 'algorithm_idx', 0)
                saved_preview = self.preview_overlay_visible
                saved_modal = self.preview_modal_enabled
                saved_path = list(self.auto_path) if self.auto_path else []
                try:
                    self.preview_overlay_visible = False
                    self.preview_modal_enabled = False
                    self.algorithm_idx = idx  # FIXED: No longer force CBS indices to 0
                    t0 = time.time()
                    success, path, teleports = self._smart_grid_path()
                    elapsed = (time.time() - t0) * 1000
                    nodes = getattr(self, 'last_search_iterations', 0)
                    results.append({'name': name, 'success': success, 'path_len': len(path), 'nodes': nodes, 'time_ms': elapsed})
                    if success and not self.last_solver_metrics:
                        self._set_last_solver_metrics(name, nodes, elapsed, len(path))
                finally:
                    self.algorithm_idx = saved_alg
                    self.preview_overlay_visible = saved_preview
                    self.preview_modal_enabled = saved_modal
                    self.auto_path = saved_path
            # Store results and notify main thread (non-blocking)
            self.solver_comparison_results = results
            self.show_solver_comparison_overlay = True
            self._set_message('Solver comparison complete', 3.0)
            self._show_toast('Solver comparison finished', 3.0, 'success')

        import threading
        self.solver_comparison_thread = threading.Thread(target=_worker, daemon=True)
        self.solver_comparison_thread.start()
        self._show_toast('Solver comparison started (background)', 2.0, 'info')

    def _start_map_elites(self, n_samples: int = 200, resolution: int = 20):
        """Start a background MAP-Elites evaluation on the currently loaded maps.

        Runs on a background thread so the GUI stays responsive. Results are stored
        in `self.map_elites_result` and a toast is shown when complete.
        """
        import threading
        if getattr(self, 'map_elites_thread', None) and self.map_elites_thread.is_alive():
            self._show_toast('MAP-Elites already running', 3.0, 'warning')
            return
        maps_copy = list(getattr(self, 'maps', []) or [])
        if not maps_copy:
            self._show_toast('No maps available for MAP-Elites', 3.0, 'warning')
            return
        self._show_toast(f'Starting MAP-Elites ({n_samples} samples, res={resolution})', 3.0)
        self.map_elites_thread = threading.Thread(target=self._map_elites_worker, args=(maps_copy, n_samples, resolution), daemon=True)
        self.map_elites_thread.start()

    def _map_elites_worker(self, maps, n_samples: int, resolution: int):
        """Background worker implementing MAP-Elites on a set of pre-loaded maps.

        This function uses the lightweight `src.simulation.map_elites` helper and the
        built-in `DungeonSolver` for validation.
        """
        try:
            from src.simulation.map_elites import run_map_elites_on_maps, plot_heatmap
            evaluator, occ = run_map_elites_on_maps(maps, resolution=resolution)
            # Try to produce a plot if plotting backend available
            try:
                import os
                out_dir = getattr(self, 'artifacts_dir', '.') or '.'
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f'map_elites_heatmap_res{resolution}.png')
                if plot_heatmap is not None:
                    plot_heatmap(occ, out_path)
                    self.map_elites_heatmap_path = out_path
                else:
                    self.map_elites_heatmap_path = None
            except Exception:
                self.map_elites_heatmap_path = None
            self.map_elites_result = evaluator
            self._show_toast('MAP-Elites completed', 4.0, 'success')
        except Exception as e:
            logger.exception('MAP-Elites worker failed: %s', e)
            self._show_toast(f'MAP-Elites failed: {e}', 4.0, 'error')

    def _render_solver_comparison_overlay(self, surface: pygame.Surface):
        """Render a small sidebar table with solver comparison results."""
        if not getattr(self, 'solver_comparison_results', None):
            return
        sidebar_x = self.screen_w - self.SIDEBAR_WIDTH
        box_w = self.SIDEBAR_WIDTH - 20
        # Check if any CBS results exist to adjust layout
        has_cbs = any('CBS' in r['name'] for r in self.solver_comparison_results)
        row_height = 22 if has_cbs else 18
        box_h = min(300, 24 + row_height * len(self.solver_comparison_results) + 20)
        box_y = 220
        box_rect = pygame.Rect(sidebar_x + 10, box_y, box_w, box_h)
        pygame.draw.rect(surface, (38, 38, 55), box_rect)
        pygame.draw.rect(surface, (100, 150, 255), box_rect, 1)
        font = pygame.font.SysFont('Arial', 11, bold=True)
        if has_cbs:
            header = font.render('Solver Comparison', True, (200, 200, 255))
        else:
            header = font.render('Solver   Success   Len   Nodes   ms', True, (200, 200, 255))
        surface.blit(header, (box_rect.x + 6, box_rect.y + 6))
        y = box_rect.y + 28
        small = pygame.font.SysFont('Arial', 10)
        for r in self.solver_comparison_results:
            if 'CBS' in r['name'] and 'confusion' in r:
                # CBS result with metrics (2-line display)
                line1 = f"{r['name'][:15]:15} {str(r.get('success',False))[:5]:5} Len:{r.get('path_len',0):<4}"
                line2 = f"  Confusion:{r.get('confusion',0):.2f} Load:{r.get('cog_load',0):.2f} {int(r.get('time_ms',0))}ms"
                color = (200, 255, 200) if r.get('success') else (255, 150, 150)
                surface.blit(small.render(line1, True, color), (box_rect.x + 6, y))
                surface.blit(small.render(line2, True, (180, 180, 255)), (box_rect.x + 6, y + 11))
                y += row_height
            else:
                # Standard result (1-line display)
                text = f"{r['name'][:7]:7}   {str(r.get('success',False))[:5]:5}   {r.get('path_len',0):3}   {r.get('nodes',0):6}   {int(r.get('time_ms',0)):4}"
                color = (200, 200, 200)
                surface.blit(small.render(text, True, color), (box_rect.x + 6, y))
                y += row_height
        # Dismiss hint
        hint = small.render('Press Esc to close', True, (150,150,150))
        surface.blit(hint, (box_rect.x + 6, box_rect.y + box_rect.h - 18))
    
    def _set_message(self, message: str, duration: float = 3.0):
        """Set status message with timestamp for auto-hide."""
        self.message = message
        self.message_time = time.time()
        self.message_duration = duration
    
    def _show_toast(self, message: str, duration: float = 3.0, toast_type: str = 'info'):
        """Show a floating toast notification."""
        if hasattr(self, 'toast_notifications'):
            self.toast_notifications.append(ToastNotification(message, duration, toast_type))
    
    def _format_cbs_metrics_tooltip(self, cbs_metrics: dict) -> str:
        """Format CBS metrics for detailed tooltip display."""
        lines = [
            f"Confusion Index: {cbs_metrics['confusion_index']:.3f}",
            f"Navigation Entropy: {cbs_metrics['navigation_entropy']:.3f}",
            f"Cognitive Load: {cbs_metrics['cognitive_load']:.3f}",
            f"Aha Latency: {cbs_metrics['aha_latency']} steps",
            f"Unique Tiles: {cbs_metrics['unique_tiles']}",
            f"Peak Memory: {cbs_metrics['peak_memory']} items",
            f"Replans: {cbs_metrics['replans']}",
            f"Confusion Events: {cbs_metrics['confusion_events']}"
        ]
        return "\n".join(lines)
    
    def _update_toasts(self):
        """Update and remove expired toasts."""
        if hasattr(self, 'toast_notifications'):
            self.toast_notifications = [t for t in self.toast_notifications if not t.is_expired()]
    
    def _render_toasts(self, surface: pygame.Surface):
        """Render all active toast notifications."""
        if not hasattr(self, 'toast_notifications'):
            return
        base_y = self.screen_h - 140  # Above bottom panel
        for i, toast in enumerate(self.toast_notifications):
            toast.render(surface, self.screen_w // 2, base_y - i * 70)
    
    # ========================================
    # BLOCK PUSH ANIMATION SYSTEM
    # ========================================
    
    def _start_block_push_animation(self, block_from: Tuple[int, int], block_to: Tuple[int, int]):
        """Start animating a block being pushed from one position to another.
        
        Args:
            block_from: Original block position (row, col)
            block_to: Destination position (row, col)
        """
        self.block_push_animations.append({
            'from_pos': block_from,
            'to_pos': block_to,
            'start_time': pygame.time.get_ticks(),
            'duration': self.block_push_duration,
            'progress': 0.0
        })
        logger.debug('BLOCK_PUSH: Started animation from %s to %s', block_from, block_to)
    
    def _update_block_push_animations(self):
        """Update all active block push animations and complete finished ones."""
        if not hasattr(self, 'block_push_animations') or not self.block_push_animations:
            return
        
        current_time = pygame.time.get_ticks()
        still_active = []
        
        for anim in self.block_push_animations:
            elapsed = current_time - anim['start_time']
            progress = min(1.0, elapsed / anim['duration'])
            anim['progress'] = progress
            
            if progress < 1.0:
                # Animation still in progress
                still_active.append(anim)
            else:
                # Animation complete - update the grid
                from_pos = anim['from_pos']
                to_pos = anim['to_pos']
                
                # Move block in grid: clear original position, place block at destination
                try:
                    # Get the BLOCK tile ID
                    block_id = SEMANTIC_PALETTE['BLOCK']
                    floor_id = SEMANTIC_PALETTE['FLOOR']
                    
                    # Update grid state
                    self.env.grid[from_pos[0], from_pos[1]] = floor_id
                    self.env.grid[to_pos[0], to_pos[1]] = block_id
                    
                    logger.debug('BLOCK_PUSH: Animation complete, grid updated: %s->%s', from_pos, to_pos)
                    
                    # Add visual effect at destination
                    if self.effects:
                        self.effects.add_effect(PopEffect(to_pos, (139, 90, 43)))  # Brown pop
                except Exception as e:
                    logger.warning('BLOCK_PUSH: Failed to update grid after animation: %s', e)
        
        self.block_push_animations = still_active
    
    def _render_block_push_animations(self, surface: pygame.Surface):
        """Render blocks that are currently being pushed with smooth interpolation.
        
        Args:
            surface: The pygame surface to draw on (map_surface)
        """
        if not hasattr(self, 'block_push_animations') or not self.block_push_animations:
            return
        
        for anim in self.block_push_animations:
            progress = anim.get('progress', 0.0)
            
            # Ease-out function for smooth deceleration: 1 - (1 - t)^2
            eased = 1.0 - (1.0 - progress) ** 2
            
            from_r, from_c = anim['from_pos']
            to_r, to_c = anim['to_pos']
            
            # Interpolate grid position
            cur_r = from_r + (to_r - from_r) * eased
            cur_c = from_c + (to_c - from_c) * eased
            
            # Convert to screen coordinates
            screen_x = int(cur_c * self.TILE_SIZE - self.view_offset_x)
            screen_y = int(cur_r * self.TILE_SIZE - self.view_offset_y)
            
            # Get block tile surface
            block_id = SEMANTIC_PALETTE['BLOCK']
            if self.renderer and hasattr(self.renderer, 'sprite_manager'):
                block_surf = self.renderer.sprite_manager.get_tile(block_id, self.TILE_SIZE)
            else:
                # Fallback: use images dict
                block_surf = self.images.get(block_id)
            
            if block_surf:
                surface.blit(block_surf, (screen_x, screen_y))
                
                # Add subtle shadow/highlight during animation for depth effect
                if progress < 1.0:
                    highlight = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
                    alpha = int(60 * (1.0 - progress))  # Fade out as animation completes
                    highlight.fill((255, 255, 200, alpha))
                    surface.blit(highlight, (screen_x, screen_y))
    
    def _get_animating_block_positions(self) -> set:
        """Get set of block positions currently being animated (to skip normal rendering)."""
        if not hasattr(self, 'block_push_animations') or not self.block_push_animations:
            return set()
        
        positions = set()
        for anim in self.block_push_animations:
            positions.add(anim['from_pos'])
            # Don't add to_pos - we want to show empty floor there during animation
        return positions
    
    def _check_and_start_block_push(self, player_pos: Tuple[int, int], target_pos: Tuple[int, int], 
                                     action: Action) -> bool:
        """Check if moving to target_pos would push a block and start animation if so.
        
        Args:
            player_pos: Current player position (row, col)
            target_pos: Position player is trying to move to (row, col)
            action: The movement action being taken
            
        Returns:
            True if a block push was initiated, False otherwise
        """
        if not hasattr(self, 'env') or self.env is None:
            return False
        
        # Check if target is a pushable block
        target_tile = self.env.grid[target_pos[0], target_pos[1]]
        if target_tile not in PUSHABLE_IDS:
            return False
        
        # Calculate push direction (same as movement direction)
        dr = target_pos[0] - player_pos[0]
        dc = target_pos[1] - player_pos[1]
        
        # Calculate where block would be pushed to
        push_dest_r = target_pos[0] + dr
        push_dest_c = target_pos[1] + dc
        
        # Check bounds
        if not (0 <= push_dest_r < self.env.height and 0 <= push_dest_c < self.env.width):
            return False
        
        # Check if destination is walkable (can receive the block)
        dest_tile = self.env.grid[push_dest_r, push_dest_c]
        if dest_tile not in WALKABLE_IDS:
            return False
        
        # Valid block push! Start animation
        self._start_block_push_animation(target_pos, (push_dest_r, push_dest_c))
        
        # Show toast notification
        self._show_toast(f'Block pushed!', 1.5, 'info')
        
        return True

    def _show_warning(self, message: str):
        """Display warning message to user."""
        logger.warning(message)
        self.message = f"[!] {message}"
    
    def _manual_step(self, action: Action):
        """Execute manual step."""
        if self.env.done:
            return
        
        old_pos = self.env.state.position
        old_keys = self.env.state.keys
        old_bombs = self.env.state.bomb_count
        old_boss_key = self.env.state.has_boss_key
        
        # Calculate target position for block push check
        dr, dc = ACTION_DELTAS.get(action, (0, 0))
        target_pos = (old_pos[0] + dr, old_pos[1] + dc)
        
        # Check if this move will push a block (before env.step modifies state)
        if (0 <= target_pos[0] < self.env.height and 0 <= target_pos[1] < self.env.width):
            self._check_and_start_block_push(old_pos, target_pos, action)
        
        state, reward, done, info = self.env.step(int(action))
        new_pos = self.env.state.position
        
        # Increment step counter
        self.step_count += 1
        
        # Check for item pickups and add visual feedback
        if self.env.state.keys > old_keys:
            # Key picked up!
            keys_gained = self.env.state.keys - old_keys
            self.keys_collected += keys_gained
            if self.effects:
                self.effects.add_effect(PopEffect(new_pos, (255, 215, 0)))  # Gold flash
            self.item_pickup_times['key'] = time.time()
            self.message = f"Key collected! ({self.keys_collected}/{self.total_keys}, {self.env.state.keys} held)"
        
        if self.env.state.bomb_count > old_bombs:
            # Bomb acquired!
            self.bombs_collected += 1
            if self.effects:
                self.effects.add_effect(PopEffect(new_pos, (200, 80, 80)))  # Red flash
            self.item_pickup_times['bomb'] = time.time()
            self.message = f"Bombs acquired! ({self.env.state.bomb_count} held)"
        
        if self.env.state.has_boss_key and not old_boss_key:
            # Boss key found!
            self.boss_keys_collected += 1
            if self.effects:
                self.effects.add_effect(FlashEffect(new_pos, (180, 40, 180), 0.5))  # Purple flash
            self.item_pickup_times['boss_key'] = time.time()
            self.message = f"BOSS KEY acquired! ({self.boss_keys_collected}/{self.total_boss_keys})"

        # Detect and track item usage (keys/bombs/boss keys)
        try:
            self._track_item_usage(old_state, self.env.state)
        except Exception:
            pass

        # Update modern HUD with current game state
        if self.modern_hud:
            self.modern_hud.update_game_state(
                keys=self.env.state.keys,
                bombs=1 if self.env.state.has_bomb else 0,
                has_boss_key=self.env.state.has_boss_key,
                position=new_pos,
                steps=self.step_count,
                message=self.message
            )
            # Update usage counters for HUD where supported
            if hasattr(self.modern_hud, 'keys_used'):
                self.modern_hud.keys_used = getattr(self, 'keys_used', 0)
            if hasattr(self.modern_hud, 'bombs_used'):
                self.modern_hud.bombs_used = getattr(self, 'bombs_used', 0)
            if hasattr(self.modern_hud, 'boss_keys_used'):
                self.modern_hud.boss_keys_used = getattr(self, 'boss_keys_used', 0)
        
        # Update visual position (smooth animation)
        if self.renderer and new_pos != old_pos:
            self.renderer.set_agent_position(new_pos[0], new_pos[1], immediate=False)
            # Add pop effect at new position (grid coordinates)
            if self.effects:
                self.effects.add_effect(PopEffect(new_pos, (100, 255, 100)))
        
        if done:
            if self.env.won:
                self.message = "YOU WIN!"
                if self.effects:
                    goal_pos = self.env.goal_pos
                    self.effects.add_effect(FlashEffect(goal_pos, (255, 215, 0), 0.5))
            else:
                self.message = f"Game Over: {info.get('msg', '')}"
        else:
            msg = info.get('msg', '')
            if msg:
                self.message = msg
    
    def _render_path_GUARANTEED(self, surface):
        """GUARANTEED path rendering - draws path no matter what.
        
        This method provides bulletproof path visualization that works
        regardless of auto_mode, preview state, or feature flags.
        Call this AFTER tiles are drawn but BEFORE HUD elements.
        """
        # Get auto_path safely
        path = getattr(self, 'auto_path', None)
        test_path = getattr(self, '_test_path', None)
        
        if not path or len(path) < 1:
            # Also try test path for debugging (only if KLTN_DEBUG_TEST_PATH=1)
            path = test_path
            if not path or len(path) < 1:
                return
            is_test = True
        else:
            is_test = False
        
        # Validate path structure: ensure all entries are valid (row, col) tuples
        try:
            for i, point in enumerate(path):
                if not isinstance(point, (tuple, list)) or len(point) != 2:
                    logger.warning(f"Invalid path point at index {i}: {point}")
                    return
                if not all(isinstance(coord, (int, float)) for coord in point):
                    logger.warning(f"Invalid path coordinates at index {i}: {point}")
                    return
        except Exception as e:
            logger.warning(f"Path validation failed: {e}")
            return
        
        path_len = len(path)
        TILE_SIZE = self.TILE_SIZE
        vx = self.view_offset_x
        vy = self.view_offset_y
        
        # Colors for path rendering
        if is_test:
            line_color = (255, 0, 0)  # RED for test path
            outline_color = (128, 0, 0)
            start_color = (255, 100, 100)
            end_color = (255, 50, 50)
        else:
            line_color = (0, 255, 255)  # CYAN for real path
            outline_color = (0, 0, 0)
            start_color = (0, 255, 0)  # GREEN start
            end_color = (255, 215, 0)  # GOLD end
        
        # Draw THICK BRIGHT lines for each segment
        if path_len > 1:
            for i in range(path_len - 1):
                # Path uses (row, col) format where row=y, col=x
                r1, c1 = path[i]
                r2, c2 = path[i + 1]
                
                # Convert to screen coordinates (center of each tile)
                sx1 = int(c1 * TILE_SIZE - vx + TILE_SIZE // 2)
                sy1 = int(r1 * TILE_SIZE - vy + TILE_SIZE // 2)
                sx2 = int(c2 * TILE_SIZE - vx + TILE_SIZE // 2)
                sy2 = int(r2 * TILE_SIZE - vy + TILE_SIZE // 2)
                
                # Draw BLACK outline for visibility (thick)
                pygame.draw.line(surface, outline_color, (sx1, sy1), (sx2, sy2), 7)
                # Draw BRIGHT path line
                pygame.draw.line(surface, line_color, (sx1, sy1), (sx2, sy2), 5)
        
        # Draw start circle (GREEN/RED)
        sr, sc = path[0]
        cx = int(sc * TILE_SIZE - vx + TILE_SIZE // 2)
        cy = int(sr * TILE_SIZE - vy + TILE_SIZE // 2)
        pygame.draw.circle(surface, (0, 0, 0), (cx, cy), 10)  # Black outline
        pygame.draw.circle(surface, start_color, (cx, cy), 8)
        
        # Draw end circle (GOLD/RED)
        er, ec = path[-1]
        ecx = int(ec * TILE_SIZE - vx + TILE_SIZE // 2)
        ecy = int(er * TILE_SIZE - vy + TILE_SIZE // 2)
        pygame.draw.circle(surface, (0, 0, 0), (ecx, ecy), 10)  # Black outline
        pygame.draw.circle(surface, end_color, (ecx, ecy), 8)
        
        # === DRAW ITEM MARKERS ALONG PATH ===
        # This highlights items that will be collected during auto-solve
        path_item_positions = getattr(self, 'path_item_positions', {})
        collected_positions = getattr(self, 'collected_positions', set())
        current_step = getattr(self, 'auto_step_idx', 0)
        
        # Item marker colors and symbols
        item_colors = {
            'keys': (255, 215, 0),      # Gold
            'boss_keys': (255, 100, 50), # Orange-red
            'ladders': (100, 200, 255),  # Light blue
            'bombs': (150, 150, 150),    # Gray
            'doors_locked': (139, 69, 19),  # Brown
            'doors_bomb': (80, 80, 80),     # Dark gray
            'doors_boss': (180, 40, 40),    # Dark red
            'triforce': (255, 255, 100)  # Bright yellow
        }
        
        # Pulsing animation for items
        pulse = (math.sin(time.time() * 4) + 1) / 2  # 0-1 oscillation
        
        for item_type, positions in path_item_positions.items():
            if not positions:
                continue
            
            color = item_colors.get(item_type, (255, 255, 255))
            
            for pos in positions:
                # Skip already collected items
                if pos in collected_positions:
                    continue
                
                # Check if this item is ahead of current position in path
                try:
                    item_path_idx = path.index(pos) if pos in path else -1
                except ValueError:
                    item_path_idx = -1
                
                # Only highlight items not yet reached
                if item_path_idx >= 0 and item_path_idx <= current_step:
                    continue
                
                r, c = pos
                ix = int(c * TILE_SIZE - vx + TILE_SIZE // 2)
                iy = int(r * TILE_SIZE - vy + TILE_SIZE // 2)
                
                # Draw pulsing marker ring
                ring_size = int(12 + 4 * pulse)
                ring_alpha = int(180 + 75 * pulse)
                
                # Draw outer ring (black outline)
                pygame.draw.circle(surface, (0, 0, 0), (ix, iy), ring_size + 2, 3)
                # Draw colored ring
                pygame.draw.circle(surface, color, (ix, iy), ring_size, 3)
                
                # Draw item type indicator (small inner circle or icon)
                inner_color = (255, 255, 255)  # White inner
                pygame.draw.circle(surface, inner_color, (ix, iy), 4)
        
        # Debug: Log rendering info every 2 seconds
        if not hasattr(self, '_guaranteed_path_log_time'):
            self._guaranteed_path_log_time = 0
        now = time.time()
        if now - self._guaranteed_path_log_time > 2.0:
            self._guaranteed_path_log_time = now
            items_count = sum(len(v) for v in path_item_positions.values()) if path_item_positions else 0
            logger.debug('GUARANTEED_PATH: Rendered %d segments, %d item markers, start=(%d,%d)->screen(%d,%d), end=(%d,%d)->screen(%d,%d), is_test=%s',
                        path_len - 1, items_count, path[0][0], path[0][1], cx, cy, path[-1][0], path[-1][1], ecx, ecy, is_test)

    def _render(self):
        """Render the current state using new visualization system or fallback."""
        # Clear screen
        self.screen.fill((25, 25, 35))
        
        h, w = self.env.height, self.env.width
        # Compute view area and ensure valid integer sizes (avoid zero/negative surfaces)
        view_w = max(1, int(self.screen_w - self.SIDEBAR_WIDTH))
        view_h = max(1, int(self.screen_h - self.HUD_HEIGHT))
        
        # Create map surface for the main view area (use convert for faster blits)
        try:
            map_surface = pygame.Surface((view_w, view_h)).convert()
        except Exception:
            # Fallback to plain surface if convert is unsupported
            map_surface = pygame.Surface((view_w, view_h))
        map_surface.fill((20, 20, 30))


        
        tiles_drawn = 0
        
        # Apply speed multiplier to animation updates
        effective_dt = self.delta_time * self.speed_multiplier
        
        # Update new renderer if available
        if self.renderer:
            self.renderer.update(effective_dt)
        if self.effects:
            self.effects.update(effective_dt)
        
        # Update block push animations
        self._update_block_push_animations()
        
        # If a background thread requested an inventory refresh, perform it here (main thread)
        if getattr(self, 'inventory_needs_refresh', False):
            try:
                logger.debug("Processing deferred inventory refresh on main thread")
                self._update_inventory_and_hud()
            except Exception:
                pass
            finally:
                self.inventory_needs_refresh = False

        # Update modern HUD with current game state every frame (real-time)
        if self.modern_hud and self.env:
            self.modern_hud.update_game_state(
                keys=self.env.state.keys,
                bombs=1 if self.env.state.has_bomb else 0,
                has_boss_key=self.env.state.has_boss_key,
                position=self.env.state.position,
                steps=self.step_count,
                message=self.message
            )
            if hasattr(self.modern_hud, 'inventory'):
                self.modern_hud.inventory.keys_collected = self.keys_collected
                self.modern_hud.inventory.bombs_collected = self.bombs_collected
                self.modern_hud.inventory.boss_keys_collected = self.boss_keys_collected
                self.modern_hud.inventory.keys_used = getattr(self, 'keys_used', 0)
                self.modern_hud.inventory.bombs_used = getattr(self, 'bombs_used', 0)
                self.modern_hud.inventory.boss_keys_used = getattr(self, 'boss_keys_used', 0)
            # Backwards compatibility: also set direct attributes if present
            if hasattr(self.modern_hud, 'keys_collected'):
                self.modern_hud.keys_collected = self.keys_collected
                self.modern_hud.bombs_collected = self.bombs_collected
                self.modern_hud.boss_keys_collected = self.boss_keys_collected
            if hasattr(self.modern_hud, 'keys_used'):
                self.modern_hud.keys_used = getattr(self, 'keys_used', 0)
            if hasattr(self.modern_hud, 'bombs_used'):
                self.modern_hud.bombs_used = getattr(self, 'bombs_used', 0)
            if hasattr(self.modern_hud, 'boss_keys_used'):
                self.modern_hud.boss_keys_used = getattr(self, 'boss_keys_used', 0)
        
        # Draw grid (only visible tiles for performance)
        start_c = max(0, int(self.view_offset_x) // self.TILE_SIZE)
        start_r = max(0, int(self.view_offset_y) // self.TILE_SIZE)
        end_c = min(w, start_c + (view_w // self.TILE_SIZE) + 2)
        end_r = min(h, start_r + (view_h // self.TILE_SIZE) + 2)
        # Diagnostic logging to help explain why nothing is drawn (throttled to once per second)
        if not hasattr(self, '_last_render_log_time'):
            self._last_render_log_time = 0.0
        
        current_time = time.time()
        if current_time - self._last_render_log_time >= 1.0:  # Log max once per second
            self._last_render_log_time = current_time
            try:
                logger.debug("Draw ranges r=%d..%d c=%d..%d (map h=%d w=%d)", start_r, end_r, start_c, end_c, h, w)
                if start_r < h and start_c < w:
                    sample_tile = int(self.env.grid[start_r, start_c])
                    logger.debug("Render sample tile: tile=%r, images_contains=%r", 
                                sample_tile, sample_tile in getattr(self, 'images', {}))
            except Exception:
                logger.exception("Failed to log draw ranges")
        
        # Quick visible fallback directly on `self.screen` if drawing would be empty
        if (end_r <= start_r) or (end_c <= start_c):
            try:
                # Draw large visible red box on screen with a label so user can see tests
                pygame.draw.rect(self.screen, (200, 0, 0), (self.screen_w//2 - 120, self.screen_h//2 - 40, 240, 80))
                f = self.big_font if hasattr(self, 'big_font') else pygame.font.SysFont('Arial', 20, True)
                txt = f.render('RENDER RANGE EMPTY - CHECK OFFSETS', True, (255, 255, 255))
                self.screen.blit(txt, (self.screen_w//2 - txt.get_width()//2, self.screen_h//2 - txt.get_height()//2))
            except Exception:
                pass
        
        # Pre-fetch collected items for efficient lookup during rendering
        # Combine env state collected_items with GUI's collected_positions for robustness
        env_collected = getattr(self.env.state, 'collected_items', set()) or set()
        gui_collected = getattr(self, 'collected_positions', set()) or set()
        collected_items = env_collected | gui_collected  # Union of both sets
        # Define collectible tile IDs that should be hidden if in collected_items
        COLLECTIBLE_TILE_IDS = (
            SEMANTIC_PALETTE.get('KEY_SMALL', -1),
            SEMANTIC_PALETTE.get('KEY_BOSS', -1),
            SEMANTIC_PALETTE.get('ITEM_BOMB', -1),
            SEMANTIC_PALETTE.get('KEY_ITEM', -1),
            SEMANTIC_PALETTE.get('ITEM_MINOR', -1),
        )
        
        # Get positions of blocks currently being animated (to skip their normal rendering)
        animating_block_positions = self._get_animating_block_positions()
        
        # Use new renderer for map tiles if available
        if self.renderer:
            for r in range(start_r, end_r):
                for c in range(start_c, end_c):
                    tile_id = self.env.grid[r, c]
                    screen_x = c * self.TILE_SIZE - self.view_offset_x
                    screen_y = r * self.TILE_SIZE - self.view_offset_y
                    
                    # FALLBACK: If position is in collected_items and it's a collectible tile,
                    # render as FLOOR instead (defensive in case grid wasn't updated)
                    if (r, c) in collected_items and tile_id in COLLECTIBLE_TILE_IDS:
                        tile_id = SEMANTIC_PALETTE['FLOOR']
                    
                    # Skip blocks being animated - render FLOOR underneath instead
                    if (r, c) in animating_block_positions and tile_id == SEMANTIC_PALETTE['BLOCK']:
                        tile_id = SEMANTIC_PALETTE['FLOOR']
                    
                    # Use sprite manager (with procedural fallback)
                    tile_surface = self.renderer.sprite_manager.get_tile(tile_id, self.TILE_SIZE)
                    map_surface.blit(tile_surface, (screen_x, screen_y))
                    tiles_drawn += 1
                    # Draw stair sprite overlay if tile is stair
                    if tile_id == SEMANTIC_PALETTE['STAIR'] and getattr(self, 'stair_sprite', None):
                        try:
                            alpha = int(140 + 90 * math.sin(time.time() * 3.0))
                            s = self.stair_sprite.copy()
                            s.set_alpha(max(20, alpha))
                            sx = screen_x + (self.TILE_SIZE - s.get_width()) // 2
                            sy = screen_y + (self.TILE_SIZE - s.get_height()) // 2
                            map_surface.blit(s, (sx, sy))
                        except Exception:
                            pass
        else:
            # Fallback rendering
            for r in range(start_r, end_r):
                for c in range(start_c, end_c):
                    tile_id = self.env.grid[r, c]
                    
                    # FALLBACK: If position is in collected_items and it's a collectible tile,
                    # render as FLOOR instead (defensive in case grid wasn't updated)
                    if (r, c) in collected_items and tile_id in COLLECTIBLE_TILE_IDS:
                        tile_id = SEMANTIC_PALETTE['FLOOR']
                    
                    # Skip blocks being animated - render FLOOR underneath instead
                    if (r, c) in animating_block_positions and tile_id == SEMANTIC_PALETTE['BLOCK']:
                        tile_id = SEMANTIC_PALETTE['FLOOR']
                    
                    img = self.images.get(tile_id, self.images.get(SEMANTIC_PALETTE['FLOOR']))
                    screen_x = c * self.TILE_SIZE - self.view_offset_x
                    screen_y = r * self.TILE_SIZE - self.view_offset_y
                    map_surface.blit(img, (screen_x, screen_y))
                    tiles_drawn += 1
                    # Draw stair sprite overlay for fallback tiles
                    if tile_id == SEMANTIC_PALETTE['STAIR'] and getattr(self, 'stair_sprite', None):
                        try:
                            alpha = int(140 + 90 * math.sin(time.time() * 3.0))
                            s = self.stair_sprite.copy()
                            s.set_alpha(max(20, alpha))
                            sx = screen_x + (self.TILE_SIZE - s.get_width()) // 2
                            sy = screen_y + (self.TILE_SIZE - s.get_height()) // 2
                            map_surface.blit(s, (sx, sy))
                        except Exception:
                            pass
        
        # === RENDER ANIMATED BLOCKS ===
        # Draw blocks that are currently being pushed with smooth interpolation
        try:
            self._render_block_push_animations(map_surface)
        except Exception as e:
            logger.warning('Failed to render block push animations: %s', e)
        
        # Draw heatmap overlay if enabled and we have search data
        if self.show_heatmap and self.search_heatmap:
            max_visits = max(self.search_heatmap.values()) if self.search_heatmap else 1
            for (r, c), visits in self.search_heatmap.items():
                if start_r <= r < end_r and start_c <= c < end_c:
                    # Normalize intensity 0.0 - 1.0
                    intensity = visits / max_visits
                    # Blue (cold) to Red (hot) gradient
                    red = int(255 * intensity)
                    blue = int(255 * (1 - intensity))
                    heat_color = (red, 0, blue, 100)
                    
                    heat_surf = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
                    heat_surf.fill(heat_color)
                    screen_x = c * self.TILE_SIZE - self.view_offset_x
                    screen_y = r * self.TILE_SIZE - self.view_offset_y
                    map_surface.blit(heat_surf, (screen_x, screen_y))

        # Draw JPS overlay (jump points & segments) for debugging/teaching
        if self.feature_flags.get('show_jps_overlay', False) and getattr(self, 'last_jps_trace', None):
            trace = self.last_jps_trace
            jumps = trace.get('jumps', []) if trace else []
            segments = trace.get('segments', []) if trace else []
            # Draw segments (semi-transparent lines)
            for a, b in segments:
                ar, ac = a; br, bc = b
                if start_r <= ar < end_r and start_c <= ac < end_c and start_r <= br < end_r and start_c <= bc < end_c:
                    ax = ac * self.TILE_SIZE + self.TILE_SIZE // 2 - self.view_offset_x
                    ay = ar * self.TILE_SIZE + self.TILE_SIZE // 2 - self.view_offset_y
                    bx = bc * self.TILE_SIZE + self.TILE_SIZE // 2 - self.view_offset_x
                    by = br * self.TILE_SIZE + self.TILE_SIZE // 2 - self.view_offset_y
                    pygame.draw.line(map_surface, (255, 200, 80, 150), (ax, ay), (bx, by), 2)

        # Small MAP-Elites overlay (draw 20x20 occupancy mini-grid in sidebar) if available
        if self.feature_flags.get('show_map_elites', False) and getattr(self, 'map_elites_result', None):
            try:
                occ = self.map_elites_result.occupancy_grid()
                res = occ.shape[0]
                mini_w = 120
                mini_h = 120
                cell_w = max(2, mini_w // res)
                cell_h = max(2, mini_h // res)
                mini_surf = pygame.Surface((cell_w * res, cell_h * res))
                mini_surf.fill((30, 30, 40))
                for yy in range(res):
                    for xx in range(res):
                        if occ[yy, xx]:
                            pygame.draw.rect(mini_surf, (100, 200, 255), (xx * cell_w, (res - 1 - yy) * cell_h, cell_w, cell_h))
                # Blit to top-right sidebar area
                sidebar_x = self.screen_w - self.SIDEBAR_WIDTH + 10
                blit_x = sidebar_x + 8
                blit_y = 120
                map_surface.blit(mini_surf, (blit_x - self.view_offset_x, blit_y - self.view_offset_y))
            except Exception:
                pass
            trace = self.last_jps_trace
            jumps = trace.get('jumps', []) if trace else []
            segments = trace.get('segments', []) if trace else []
            # Draw segments (semi-transparent lines)
            for a, b in segments:
                ar, ac = a; br, bc = b
                if start_r <= ar < end_r and start_c <= ac < end_c and start_r <= br < end_r and start_c <= bc < end_c:
                    ax = ac * self.TILE_SIZE + self.TILE_SIZE // 2 - self.view_offset_x
                    ay = ar * self.TILE_SIZE + self.TILE_SIZE // 2 - self.view_offset_y
                    bx = bc * self.TILE_SIZE + self.TILE_SIZE // 2 - self.view_offset_x
                    by = br * self.TILE_SIZE + self.TILE_SIZE // 2 - self.view_offset_y
                    try:
                        pygame.draw.line(map_surface, (255, 180, 0, 180), (ax, ay), (bx, by), 2)
                    except Exception:
                        # Pygame may not handle alpha in draw.line on some setups; fallback
                        pygame.draw.line(map_surface, (255, 180, 0), (ax, ay), (bx, by), 2)
            # Draw jump points
            for jr, jc in jumps:
                if start_r <= jr < end_r and start_c <= jc < end_c:
                    sx = jc * self.TILE_SIZE - self.view_offset_x
                    sy = jr * self.TILE_SIZE - self.view_offset_y
                    dot = pygame.Surface((8, 8), pygame.SRCALPHA)
                    dot.fill((255, 100, 0, 200))
                    map_surface.blit(dot, (sx + self.TILE_SIZE//2 - 4, sy + self.TILE_SIZE//2 - 4))
        
        # Draw solution path whenever a path exists
        show_path = self.auto_path and len(self.auto_path) > 0
        # DEBUG: Log path rendering decision (throttled to avoid spam)
        if not hasattr(self, '_path_render_log_counter'):
            self._path_render_log_counter = 0
        self._path_render_log_counter += 1
        if self._path_render_log_counter % 120 == 1:  # Log every 120 frames (~2 seconds at 60fps)
            logger.debug('DEBUG_RENDER: show_path=%s, auto_path=%s, len=%d, auto_mode=%s, preview_visible=%s',
                         show_path,
                         bool(self.auto_path),
                         len(self.auto_path) if self.auto_path else 0,
                         self.auto_mode,
                         getattr(self, 'preview_overlay_visible', False))
            if self.auto_path and len(self.auto_path) > 0:
                logger.debug('DEBUG_RENDER: Path first=%s, last=%s, step_idx=%d, view_offset=(%d,%d)',
                             self.auto_path[0], self.auto_path[-1],
                             getattr(self, 'auto_step_idx', 0),
                             getattr(self, 'view_offset_x', 0),
                             getattr(self, 'view_offset_y', 0))
        if show_path:
            logger.debug(f"Drawing path overlay: {len(self.auto_path)} points, auto_mode={self.auto_mode}, step_idx={self.auto_step_idx}")
            # FIRST: Draw the FULL planned path as a line (cyan/light blue, behind visited tiles)
            if len(self.auto_path) > 1:
                for i in range(len(self.auto_path) - 1):
                    r1, c1 = self.auto_path[i]
                    r2, c2 = self.auto_path[i + 1]
                    # Convert to screen coordinates (center of each tile)
                    # Note: positions are (row, col) where row=y, col=x
                    x1 = int(c1 * self.TILE_SIZE - self.view_offset_x + self.TILE_SIZE // 2)
                    y1 = int(r1 * self.TILE_SIZE - self.view_offset_y + self.TILE_SIZE // 2)
                    x2 = int(c2 * self.TILE_SIZE - self.view_offset_x + self.TILE_SIZE // 2)
                    y2 = int(r2 * self.TILE_SIZE - self.view_offset_y + self.TILE_SIZE // 2)
                    # Draw future path (cyan) vs visited path (green)
                    if i >= self.auto_step_idx:
                        # Future path - bright cyan with outline for visibility
                        pygame.draw.line(map_surface, (0, 0, 0), (x1, y1), (x2, y2), 5)  # Black outline
                        pygame.draw.line(map_surface, (0, 255, 255), (x1, y1), (x2, y2), 3)  # Cyan fill
                    else:
                        # Visited path - bright green with outline
                        pygame.draw.line(map_surface, (0, 0, 0), (x1, y1), (x2, y2), 6)  # Black outline
                        pygame.draw.line(map_surface, (0, 255, 0), (x1, y1), (x2, y2), 4)  # Green fill
            
            # THIRD: Draw start and end markers for clear visibility
            if len(self.auto_path) >= 1:
                # Start marker (green circle)
                start_r, start_c = self.auto_path[0]
                start_x = int(start_c * self.TILE_SIZE - self.view_offset_x + self.TILE_SIZE // 2)
                start_y = int(start_r * self.TILE_SIZE - self.view_offset_y + self.TILE_SIZE // 2)
                pygame.draw.circle(map_surface, (0, 0, 0), (start_x, start_y), 10)  # Black outline
                pygame.draw.circle(map_surface, (0, 255, 100), (start_x, start_y), 8)  # Green fill
                
                # End/goal marker (red/gold circle)
                end_r, end_c = self.auto_path[-1]
                end_x = int(end_c * self.TILE_SIZE - self.view_offset_x + self.TILE_SIZE // 2)
                end_y = int(end_r * self.TILE_SIZE - self.view_offset_y + self.TILE_SIZE // 2)
                pygame.draw.circle(map_surface, (0, 0, 0), (end_x, end_y), 10)  # Black outline
                pygame.draw.circle(map_surface, (255, 215, 0), (end_x, end_y), 8)  # Gold fill
            
            # FOURTH: Draw tile highlights for visited positions (when animating)
            if self.auto_mode and self.auto_step_idx > 0:
                for i, pos in enumerate(self.auto_path[:self.auto_step_idx + 1]):
                    pr, pc = pos
                    path_surf = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
                    
                    # Check if this position is a recently unlocked door
                    current_time = time.time()
                    is_recent_unlock = pos in self.door_unlock_times and (current_time - self.door_unlock_times[pos]) < 2.0
                    
                    if is_recent_unlock:
                        # Flash effect for recently unlocked doors (yellow/gold)
                        flash_alpha = (math.sin(current_time * 8) + 1) / 2  # 0 to 1
                        alpha = int(150 + 105 * flash_alpha)
                        path_surf.fill((255, 215, 0, alpha))  # Gold
                    else:
                        # Use green with slight gradient based on progress
                        alpha = 40 + int(20 * (i / max(1, len(self.auto_path))))
                        path_surf.fill((0, 255, 0, alpha))
                    
                    screen_x = pc * self.TILE_SIZE - self.view_offset_x
                    screen_y = pr * self.TILE_SIZE - self.view_offset_y
                    map_surface.blit(path_surf, (screen_x, screen_y))
        
        # === GUARANTEED PATH RENDERING ===
        # This ensures the path is ALWAYS visible when auto_path has data,
        # regardless of auto_mode, preview state, or feature flags.
        try:
            self._render_path_GUARANTEED(map_surface)
        except Exception as e:
            logger.warning('_render_path_GUARANTEED failed: %s', e)
        
        # Draw Link (use smooth animation if renderer available)
        if self.renderer and self.renderer.agent_visual_pos:
            # Smooth animated position
            visual_pos = self.renderer.agent_visual_pos
            link_x = int(visual_pos.x * self.TILE_SIZE - self.view_offset_x + 2)
            link_y = int(visual_pos.y * self.TILE_SIZE - self.view_offset_y + 2)
        else:
            # Direct grid position
            pr, pc = self.env.state.position
            link_x = pc * self.TILE_SIZE - self.view_offset_x + 2
            link_y = pr * self.TILE_SIZE - self.view_offset_y + 2
        map_surface.blit(self.link_img, (link_x, link_y))
        
        # Render visual effects on map surface
        if self.effects:
            self.effects.render(map_surface, self.TILE_SIZE, (self.view_offset_x, self.view_offset_y))
        
        # If nothing was drawn (e.g., view region outside map or sizes miscalculated), attempt an auto-fix and show diagnostics
        if tiles_drawn == 0:
            try:
                # Try auto-fit + center once to recover from bad offsets
                if not getattr(self, '_auto_recenter_done', False):
                    logger.info('No tiles drawn — attempting auto-fit zoom + center')
                    try:
                        self._auto_fit_zoom()
                        self._center_view()
                    except Exception:
                        pass
                    self._auto_recenter_done = True

                diag_font = pygame.font.SysFont('Arial', 18, bold=True)
                diag_text = diag_font.render('No map tiles visible - check zoom/offset', True, (255, 100, 100))
                tx = max(10, (view_w - diag_text.get_width()) // 2)
                ty = max(10, (view_h - diag_text.get_height()) // 2)
                # Draw a semi-opaque box behind message for visibility
                box = pygame.Surface((diag_text.get_width() + 20, diag_text.get_height() + 18), pygame.SRCALPHA)
                box.fill((30, 10, 10, 200))
                map_surface.blit(box, (tx - 10, ty - 9))
                map_surface.blit(diag_text, (tx, ty))

                # Additional diagnostic lines useful for debugging
                small = pygame.font.SysFont('Arial', 12)
                try:
                    map_w = self.env.width if self.env is not None else 0
                    map_h = self.env.height if self.env is not None else 0
                except Exception:
                    map_w = map_h = 0
                diag2 = small.render(f'Tile: {self.TILE_SIZE}px  ViewOffset: ({self.view_offset_x},{self.view_offset_y})', True, (220, 220, 220))
                diag3 = small.render(f'Map: {map_w}x{map_h}  View: {view_w}x{view_h}', True, (200, 200, 200))
                map_surface.blit(diag2, (10, ty + diag_text.get_height() + 8))
                map_surface.blit(diag3, (10, ty + diag_text.get_height() + 24))
            except Exception:
                pass

            # Track consecutive empty frames and try to recover display if persistent
            try:
                self._consecutive_empty_frames = getattr(self, '_consecutive_empty_frames', 0) + 1
                if self._consecutive_empty_frames >= getattr(self, '_empty_frame_recovery_threshold', 8):
                    logger.warning('Detected %d consecutive empty frames — attempting display reinit', self._consecutive_empty_frames)
                    try:
                        recovered = self._attempt_display_reinit()
                        if recovered:
                            self._show_toast('Recovered display after blank frames', 3.0, 'success')
                            logger.info('Recovered display after empty-frame sequence')
                        else:
                            self._show_toast('Display recovery failed', 4.0, 'error')
                    except Exception:
                        logger.exception('Error during forced display reinit')
                    finally:
                        self._consecutive_empty_frames = 0
            except Exception:
                logger.exception('Failed handling consecutive empty frames counter')
        else:
            # Reset counter when frames are healthy
            try:
                self._consecutive_empty_frames = 0
            except Exception:
                pass

        # Blit map surface to screen
        self.screen.blit(map_surface, (0, 0))

        # Debug overlay removed - was causing yellow/magenta square in corner

        # Draw translucent overlays that may capture clicks so users can see what's on top
        try:
            # Preview overlay (non-modal) indicator
            if getattr(self, 'preview_overlay_visible', False):
                try:
                    logger.debug('Rendering preview overlay (will capture clicks)')
                    ov = pygame.Surface((view_w, view_h), pygame.SRCALPHA)
                    ov.fill((40, 30, 40, 130))
                    self.screen.blit(ov, (0, 0))
                    label = self.big_font.render('PATH PREVIEW (overlay) - captures clicks', True, (255, 220, 120))
                    self.screen.blit(label, (20, view_h//2 - 20))
                except Exception:
                    pass
            # Solver comparison modal
            if getattr(self, 'show_solver_comparison_overlay', False):
                try:
                    logger.debug('Rendering solver comparison modal (captures clicks)')
                    ov2 = pygame.Surface((view_w, view_h), pygame.SRCALPHA)
                    ov2.fill((20, 20, 20, 180))
                    self.screen.blit(ov2, (0, 0))
                    label2 = self.big_font.render('SOLVER COMPARISON - modal', True, (200, 200, 255))
                    self.screen.blit(label2, (20, view_h//2 - 20))
                except Exception:
                    pass
        except Exception:
            pass
        
        # Get current position for display (use actual grid position)
        pr, pc = self.env.state.position
        
        # Draw sidebar background
        sidebar_x = self.screen_w - self.SIDEBAR_WIDTH
        pygame.draw.rect(self.screen, (35, 35, 50), (sidebar_x, 0, self.SIDEBAR_WIDTH, self.screen_h))
        pygame.draw.line(self.screen, (60, 60, 80), (sidebar_x, 0), (sidebar_x, self.screen_h), 2)
        
        # Sidebar content
        y_pos = 10
        
        # Title
        title = self.big_font.render("ZAVE", True, (100, 200, 255))
        self.screen.blit(title, (sidebar_x + 10, y_pos))
        y_pos += 28
        
        # Dungeon name
        if self.current_map_idx < len(self.map_names):
            name = self.map_names[self.current_map_idx]
        else:
            name = f"Map {self.current_map_idx + 1}"
        name_surf = self.font.render(name, True, (255, 220, 100))
        self.screen.blit(name_surf, (sidebar_x + 10, y_pos))
        y_pos += 20
        
        # Map number
        map_num = f"({self.current_map_idx + 1}/{len(self.maps)})"
        num_surf = self.small_font.render(map_num, True, (150, 150, 150))
        self.screen.blit(num_surf, (sidebar_x + 10, y_pos))
        y_pos += 18
        
        size_info = f"Size: {w}x{h}"
        size_surf = self.small_font.render(size_info, True, (150, 150, 150))
        self.screen.blit(size_surf, (sidebar_x + 10, y_pos))
        y_pos += 20
        
        # Divider
        pygame.draw.line(self.screen, (60, 60, 80), (sidebar_x + 10, y_pos), (self.screen_w - 10, y_pos))
        y_pos += 10
        
        # Inventory section
        inv_title = self.font.render("Inventory", True, (255, 200, 100))
        self.screen.blit(inv_title, (sidebar_x + 10, y_pos))
        y_pos += 22
        
        # Check if items were recently picked up (within last 1 second) for highlight effect
        current_time = time.time()
        key_highlight = 'key' in self.item_pickup_times and (current_time - self.item_pickup_times['key']) < 1.0
        bomb_highlight = 'bomb' in self.item_pickup_times and (current_time - self.item_pickup_times['bomb']) < 1.0
        boss_key_highlight = 'boss_key' in self.item_pickup_times and (current_time - self.item_pickup_times['boss_key']) < 1.0
        
        # Sync counts before rendering to be safe
        self._sync_inventory_counters()
        
        # DEBUG: Log inventory values being rendered (every 60 frames to avoid spam)
        if hasattr(self, '_inv_render_frame_count'):
            self._inv_render_frame_count = getattr(self, '_inv_render_frame_count', 0) + 1
        else:
            self._inv_render_frame_count = 0
        if self._inv_render_frame_count % 60 == 0:
            logger.debug("INVENTORY_RENDER: keys_collected=%d, total_keys=%d, env.state.keys=%d, bombs_collected=%d, has_bomb=%s, boss_keys_collected=%d, has_boss_key=%s, collected_items_len=%d",
                         self.keys_collected, self.total_keys, self.env.state.keys,
                         self.bombs_collected, self.env.state.has_bomb,
                         self.boss_keys_collected, getattr(self.env.state, 'has_boss_key', False),
                         len(self.collected_items))
        
        # Keys with flash animation and X/Y collected format
        if self.total_keys > 0:
            keys_text = f"Keys: {self.keys_collected}/{self.total_keys} ({self.env.state.keys} held)"
        else:
            keys_text = f"Keys: {self.env.state.keys}"
        # Small last pickup/use hints
        if getattr(self, 'last_pickup_msg', None):
            hint = self.small_font.render(self.last_pickup_msg, True, (200, 200, 200))
            self.screen.blit(hint, (sidebar_x + 15, y_pos))
            y_pos += 16
        if getattr(self, 'last_use_msg', None):
            hint2 = self.small_font.render(self.last_use_msg, True, (200, 200, 200))
            self.screen.blit(hint2, (sidebar_x + 15, y_pos))
            y_pos += 16
        if key_highlight:
            # Flash between yellow and white
            flash_alpha = (math.sin(current_time * 15) + 1) / 2  # 0 to 1
            keys_color = (255, int(220 + 35 * flash_alpha), int(100 + 155 * flash_alpha))
        else:
            keys_color = (255, 220, 100)
        keys_surf = self.small_font.render(keys_text, True, keys_color)
        self.screen.blit(keys_surf, (sidebar_x + 15, y_pos))
        y_pos += 18
        
        # Bombs with highlight and collected count
        if self.total_bombs > 0:
            bomb_text = f"Bombs: {self.bombs_collected}/{self.total_bombs} {'[Y]' if self.env.state.has_bomb else '[N]'}"
        else:
            bomb_text = f"Bomb: {'[Y]' if self.env.state.has_bomb else '[N]'}"
        if bomb_highlight:
            flash_alpha = (math.sin(current_time * 15) + 1) / 2
            bomb_color = (int(200 + 55 * flash_alpha), int(80 + 175 * flash_alpha), int(80 + 175 * flash_alpha))
        else:
            bomb_color = (100, 255, 100) if self.env.state.has_bomb else (150, 150, 150)
        bomb_surf = self.small_font.render(bomb_text, True, bomb_color)
        self.screen.blit(bomb_surf, (sidebar_x + 15, y_pos))
        y_pos += 18
        
        # Boss key with highlight and collected status
        if self.total_boss_keys > 0:
            boss_key_text = f"Boss Key: {self.boss_keys_collected}/{self.total_boss_keys} {'[Y]' if self.env.state.has_boss_key else '[N]'}"
        else:
            boss_key_text = f"Boss Key: {'[Y]' if self.env.state.has_boss_key else '[N]'}"
        if boss_key_highlight:
            flash_alpha = (math.sin(current_time * 15) + 1) / 2
            boss_color = (int(180 + 75 * flash_alpha), int(40 + 215 * flash_alpha), int(180 + 75 * flash_alpha))
        else:
            boss_color = (255, 150, 100) if self.env.state.has_boss_key else (150, 150, 150)
        boss_surf = self.small_font.render(boss_key_text, True, boss_color)
        self.screen.blit(boss_surf, (sidebar_x + 15, y_pos))
        y_pos += 18
        
        # State-space solver info (if available)
        if self.solver_result:
            y_pos += 5
            pygame.draw.line(self.screen, (60, 60, 80), (sidebar_x + 10, y_pos), (self.screen_w - 10, y_pos))
            y_pos += 8
            
            solver_title = self.font.render("Path Analysis", True, (100, 200, 255))
            self.screen.blit(solver_title, (sidebar_x + 10, y_pos))
            y_pos += 20
            
            # Keys info from solver
            keys_avail = self.solver_result.get('keys_available', 0)
            keys_used = self.solver_result.get('keys_used', 0)
            key_info = f"Keys: {keys_avail} found, {keys_used} used"
            key_color = (255, 220, 100) if keys_used > 0 else (150, 200, 150)
            key_surf = self.small_font.render(key_info, True, key_color)
            self.screen.blit(key_surf, (sidebar_x + 15, y_pos))
            y_pos += 16
            
            # Edge types breakdown
            edge_types = self.solver_result.get('edge_types', [])
            if edge_types:
                type_counts = {}
                for et in edge_types:
                    type_counts[et] = type_counts.get(et, 0) + 1
                
                # Display each edge type with color
                edge_colors = {
                    'open': (100, 255, 100),       # Green - normal door
                    'key_locked': (255, 220, 100), # Yellow - key door
                    'bombable': (255, 150, 50),    # Orange - bomb wall
                    'soft_locked': (180, 100, 255),# Purple - one-way
                    'stair': (100, 200, 255),      # Cyan - teleport
                }
                
                for etype, count in type_counts.items():
                    color = edge_colors.get(etype, (150, 150, 150))
                    type_name = etype.replace('_', ' ').title()
                    et_text = f"  {type_name}: {count}"
                    et_surf = self.small_font.render(et_text, True, color)
                    self.screen.blit(et_surf, (sidebar_x + 15, y_pos))
                    y_pos += 14
        y_pos += 7
        
        # Divider
        pygame.draw.line(self.screen, (60, 60, 80), (sidebar_x + 10, y_pos), (self.screen_w - 10, y_pos))
        y_pos += 10
        
        # === STATUS SECTION ===
        status_title = self.font.render("STATUS", True, (180, 220, 255))
        self.screen.blit(status_title, (sidebar_x + 10, y_pos))
        y_pos += 22
        
        # Map name and position
        if self.env:
            map_name = self.map_names[self.current_map_idx] if self.current_map_idx < len(self.map_names) else f"Map {self.current_map_idx + 1}"
            pos = self.env.state.position
            status_text = f"{map_name}"
            status_surf = self.small_font.render(status_text, True, (200, 220, 255))
            self.screen.blit(status_surf, (sidebar_x + 15, y_pos))
            y_pos += 16
            
            pos_text = f"Pos: ({pos[0]}, {pos[1]})"
            pos_surf = self.small_font.render(pos_text, True, (150, 150, 150))
            self.screen.blit(pos_surf, (sidebar_x + 15, y_pos))
            y_pos += 16
            
            # Status message
            ready_text = "Auto-solving" if self.auto_mode else "Ready"
            ready_color = (100, 255, 100) if self.auto_mode else (200, 200, 200)
            ready_surf = self.small_font.render(ready_text, True, ready_color)
            self.screen.blit(ready_surf, (sidebar_x + 15, y_pos))
            y_pos += 18

            # D* Lite indicator (show when enabled or active)
            try:
                if getattr(self, 'dstar_active', False) and getattr(self, 'dstar_solver', None):
                    replans = getattr(self.dstar_solver, 'replans_count', 0)
                    ds_text = f"D* Lite: ACTIVE ({replans} replans)"
                    ds_surf = self.small_font.render(ds_text, True, (100, 220, 255))
                    self.screen.blit(ds_surf, (sidebar_x + 15, y_pos))
                    y_pos += 16
                elif self.feature_flags.get('dstar_lite', False):
                    ds_text = "D* Lite: enabled"
                    ds_surf = self.small_font.render(ds_text, True, (180, 180, 255))
                    self.screen.blit(ds_surf, (sidebar_x + 15, y_pos))
                    y_pos += 16
            except Exception:
                pass

            # Stair debug: count stairs and show sprite status
            try:
                stair_positions = list(map(tuple, self.env._find_all_positions(SEMANTIC_PALETTE['STAIR']))) if hasattr(self.env, '_find_all_positions') else []
                stair_count = len(stair_positions)
                stair_text = f"Stairs: {stair_count} | Sprite: {'Full' if getattr(self, 'stair_sprite', None) else 'No'}"
                stair_surf = self.small_font.render(stair_text, True, (200, 200, 150))
                self.screen.blit(stair_surf, (sidebar_x + 15, y_pos))
                y_pos += 16
            except Exception:
                pass
        
        # Divider
        pygame.draw.line(self.screen, (60, 60, 80), (sidebar_x + 10, y_pos), (self.screen_w - 10, y_pos))
        y_pos += 10
        
        # === MESSAGE SECTION ===
        message_title = self.font.render("MESSAGE", True, (100, 255, 150))
        self.screen.blit(message_title, (sidebar_x + 10, y_pos))
        y_pos += 22
        
        # Show current message with fade effect
        if self.message and (time.time() - self.message_time) < self.message_duration:
            elapsed = time.time() - self.message_time
            remaining = self.message_duration - elapsed
            alpha = min(1.0, remaining / 0.5) if remaining < 0.5 else 1.0
            
            # Word wrap message to fit sidebar
            max_chars = 28
            words = self.message.split()
            lines = []
            current_line = ""
            
            for word in words:
                test_line = f"{current_line} {word}".strip()
                if len(test_line) <= max_chars:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)
            
            # Render each line
            for line in lines[:3]:  # Max 3 lines
                msg_color = tuple(int(c * alpha) for c in (150, 255, 200))
                msg_surf = self.small_font.render(line, True, msg_color)
                self.screen.blit(msg_surf, (sidebar_x + 15, y_pos))
                y_pos += 16
        else:
            # Show default message
            default_msg = "Press SPACE to solve"
            msg_surf = self.small_font.render(default_msg, True, (120, 120, 120))
            self.screen.blit(msg_surf, (sidebar_x + 15, y_pos))
            y_pos += 16
        
        y_pos += 8
        
        # Divider
        pygame.draw.line(self.screen, (60, 60, 80), (sidebar_x + 10, y_pos), (self.screen_w - 10, y_pos))
        y_pos += 10
        
        # === METRICS SECTION ===
        metrics_title = self.font.render("Metrics", True, (150, 200, 255))
        self.screen.blit(metrics_title, (sidebar_x + 10, y_pos))
        y_pos += 22
        
        # Steps
        steps_text = f"Steps: {self.step_count}"
        steps_surf = self.small_font.render(steps_text, True, (200, 200, 200))
        self.screen.blit(steps_surf, (sidebar_x + 15, y_pos))
        y_pos += 16
        
        # Speed
        speed_color = (100, 255, 100) if self.speed_multiplier == 1.0 else (255, 200, 100)
        speed_text = f"Speed: {self.speed_multiplier}x"
        speed_surf = self.small_font.render(speed_text, True, speed_color)
        self.screen.blit(speed_surf, (sidebar_x + 15, y_pos))
        y_pos += 16
        
        # Zoom
        zoom_text = f"Zoom: {self.TILE_SIZE}px"
        zoom_surf = self.small_font.render(zoom_text, True, (150, 150, 150))
        self.screen.blit(zoom_surf, (sidebar_x + 15, y_pos))
        y_pos += 16
        
        # FPS
        fps = int(self.clock.get_fps())
        fps_color = (100, 255, 100) if fps >= 25 else (255, 150, 150)
        fps_text = f"FPS: {fps}"
        fps_surf = self.small_font.render(fps_text, True, fps_color)
        self.screen.blit(fps_surf, (sidebar_x + 15, y_pos))
        y_pos += 18
        
        # Divider
        pygame.draw.line(self.screen, (60, 60, 80), (sidebar_x + 10, y_pos), (self.screen_w - 10, y_pos))
        y_pos += 10
        
        # Controls section
        ctrl_title = self.font.render("Controls", True, (100, 200, 100))
        self.screen.blit(ctrl_title, (sidebar_x + 10, y_pos))
        y_pos += 20
        
        controls = [
            "Arrows Move",
            "SPACE Auto-solve",
            "R     Reset map",
            "N/P   Next/Prev",
            "+/-   Zoom",
            "0     Reset zoom",
            "C     Center view",
            "M     Minimap",
            "[/]   Speed+/-",
            "F11   Fullscreen",
            "F1    Help",
            "H     Heatmap",
            "ESC   Quit",
        ]
        
        for ctrl in controls:
            ctrl_surf = self.small_font.render(ctrl, True, (120, 120, 120))
            self.screen.blit(ctrl_surf, (sidebar_x + 15, y_pos))
            y_pos += 15
        
        # Draw HUD at bottom
        hud_y = self.screen_h - self.HUD_HEIGHT
        pygame.draw.rect(self.screen, (30, 30, 45), (0, hud_y, self.screen_w - self.SIDEBAR_WIDTH, self.HUD_HEIGHT))
        pygame.draw.line(self.screen, (60, 60, 80), (0, hud_y), (self.screen_w - self.SIDEBAR_WIDTH, hud_y), 2)
        
        # Status message
        msg_color = (255, 255, 100) if self.env.won else (200, 200, 200)
        msg_surf = self.font.render(self.message, True, msg_color)
        self.screen.blit(msg_surf, (10, hud_y + 10))
        
        # Position info
        pos_text = f"Position: ({pr}, {pc})"
        pos_surf = self.small_font.render(pos_text, True, (150, 150, 150))
        self.screen.blit(pos_surf, (10, hud_y + 35))
        
        # Win state
        if self.env.won:
            win_text = "*** VICTORY! ***"
            win_surf = self.big_font.render(win_text, True, (255, 215, 0))
            self.screen.blit(win_surf, (10, hud_y + 55))
        
        # Render minimap if enabled
        if self.show_minimap:
            self._render_minimap()
        
        # Help overlay
        if self.show_help:
            self._render_help_overlay()
        
        # Path preview dialog (Feature 5) - render on top of everything
        if self.path_preview_mode and self.path_preview_dialog:
            # Render path overlay on map
            try:
                self.path_preview_dialog.render_path_overlay(
                    self.screen,
                    self.TILE_SIZE,
                    self.view_offset_x,
                    self.view_offset_y,
                    self.SIDEBAR_WIDTH,
                    self.HUD_HEIGHT
                )
            except Exception as e:
                logger.warning(f"Failed to render path overlay: {e}")
            
            # Render dialog box
            try:
                self.path_preview_dialog.render(self.screen)
            except Exception as e:
                logger.warning(f"Failed to render path preview dialog: {e}")
        elif getattr(self, 'preview_overlay_visible', False) and getattr(self, 'path_preview_dialog', None):
            # Non-modal overlay: render only the path overlay (no blocking dialog)
            try:
                self.path_preview_dialog.render_path_overlay(
                    self.screen,
                    self.TILE_SIZE,
                    self.view_offset_x,
                    self.view_offset_y,
                    self.SIDEBAR_WIDTH,
                    self.HUD_HEIGHT
                )
            except Exception as e:
                logger.warning(f"Failed to render path overlay (non-modal): {e}")

            # Minimal sidebar preview box with start/dismiss buttons
            try:
                sidebar_x = self.screen_w - self.SIDEBAR_WIDTH
                box_h = 80
                box_y = 120  # fixed area near top of sidebar (below header area)
                box_rect = pygame.Rect(sidebar_x + 10, box_y, self.SIDEBAR_WIDTH - 20, box_h)
                pygame.draw.rect(self.screen, (40, 40, 60), box_rect)
                pygame.draw.rect(self.screen, (100, 150, 255), box_rect, 2)

                # Text details
                font = pygame.font.SysFont('Arial', 14, bold=True)
                small = pygame.font.SysFont('Arial', 12)
                path_len = len(self.auto_path) if getattr(self, 'auto_path', None) else 0
                text1 = font.render(f"Preview: {path_len} steps", True, (200, 200, 255))
                self.screen.blit(text1, (box_rect.x + 8, box_rect.y + 8))

                # Keys info (if available)
                keys_used = getattr(self, 'solver_result', {}).get('keys_used', 0) if getattr(self, 'solver_result', None) else 0
                keys_avail = getattr(self, 'solver_result', {}).get('keys_available', 0) if getattr(self, 'solver_result', None) else 0
                keys_text = f"Keys: {keys_used} / {keys_avail}" if keys_avail > 0 else "Keys: None"
                self.screen.blit(small.render(keys_text, True, (200, 200, 200)), (box_rect.x + 8, box_rect.y + 34))

                # Start & Dismiss buttons
                start_rect = pygame.Rect(box_rect.x + 8, box_rect.y + 48, 140, 24)
                dismiss_rect = pygame.Rect(box_rect.x + 156, box_rect.y + 48, 60, 24)
                pygame.draw.rect(self.screen, (40, 140, 40), start_rect)
                pygame.draw.rect(self.screen, (140, 40, 40), dismiss_rect)
                pygame.draw.rect(self.screen, (100, 255, 100), start_rect, 1)
                pygame.draw.rect(self.screen, (255, 100, 100), dismiss_rect, 1)
                self.sidebar_start_button_rect = start_rect
                self.sidebar_dismiss_button_rect = dismiss_rect

                start_text = small.render("Start Auto-Solve", True, (255, 255, 255))
                dismiss_text = small.render("Dismiss", True, (255, 255, 255))
                self.screen.blit(start_text, (start_rect.x + 8, start_rect.y + 4))
                self.screen.blit(dismiss_text, (dismiss_rect.x + 6, dismiss_rect.y + 4))

            except Exception as e:
                logger.warning(f"Failed to render sidebar preview box: {e}")
        else:
            # Ensure stale sidebar button rects are cleared
            self.sidebar_start_button_rect = None
            self.sidebar_dismiss_button_rect = None

        # Render topology overlay (if enabled via checkbox or feature_flags)
        # Sync feature_flags to instance variable
        if self.feature_flags.get('show_topology', False):
            self.show_topology = True
        if getattr(self, 'show_topology', False):
            try:
                logger.debug("Rendering topology overlay")
                self._render_topology_overlay(self.screen)
            except Exception as e:
                logger.warning(f"Topology overlay failed: {e}")

        # Render solver comparison overlay (if available)
        if getattr(self, 'show_solver_comparison_overlay', False):
            try:
                self._render_solver_comparison_overlay(self.screen)
            except Exception as e:
                logger.warning(f"Solver comparison overlay failed: {e}")
        
        # Render control panel
        if self.control_panel_enabled:
            self._render_control_panel(self.screen)

        # Render developer debug overlay (toggle with F12)
        if getattr(self, 'debug_overlay_enabled', False):
            try:
                self._render_debug_overlay(self.screen)
            except Exception as e:
                logger.warning(f"Debug overlay render failed: {e}")
        
        # Render item legend
        if self.auto_mode:
            self._render_item_legend(self.screen)
        
        # Render error banner (on top of everything)
        self._render_error_banner(self.screen)
        
        # Render solver status banner (shows algorithm being used)
        self._render_solver_status_banner(self.screen)
        
        # Render toast notifications (on top of everything)
        self._render_toasts(self.screen)
        
        # NOTE: pygame.display.flip() is called by the main run() loop after _render()
        # Do NOT call flip() here to avoid double-buffer swap issues

    def _render_debug_overlay(self, surface: pygame.Surface):
        """Render debug overlay with mouse coords, widget rects, and recent clicks.
        Toggle with F12. Shift-F11 clears click log.
        """
        try:
            font = pygame.font.SysFont('Arial', 12)
        except Exception:
            return

        # Background box
        box_w = 380
        box_h = 24 + 16 * min(10, len(self.widget_manager.widgets) if self.widget_manager else 0)
        box_surf = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        box_surf.fill((20, 20, 30, 220))
        surface.blit(box_surf, (10, 10))

        x = 14
        y = 14
        mouse_pos = pygame.mouse.get_pos()
        surface.blit(font.render(f"Mouse: {mouse_pos}", True, (220, 220, 220)), (x, y))
        y += 16
        panel_rect = getattr(self, 'control_panel_rect', None)
        surface.blit(font.render(f"Panel rect: {panel_rect}", True, (220, 220, 220)), (x, y))
        y += 16
        surface.blit(font.render(f"Collapse btn: {getattr(self,'collapse_button_rect',None)}", True, (220, 220, 220)), (x, y))
        y += 18

        # Focus & last-event diagnostics
        focused = pygame.mouse.get_focused()
        grabbed = pygame.event.get_grab()
        now = time.time()
        surface.blit(font.render(f"Focused: {focused}   Grabbed: {grabbed}", True, (200, 240, 200)), (x, y))
        y += 16
        last = getattr(self, '_last_mouse_event', None)
        if last:
            age = int((now - last.get('time', now)) * 1000)
            ltxt = f"Last mouse: {last.get('type')} pos={last.get('pos')} btn={last.get('button', '')} age={age}ms"
            surface.blit(font.render(ltxt, True, (200, 240, 200)), (x, y))
            y += 16
        lastk = getattr(self, '_last_key_event', None)
        if lastk:
            k_age = int((now - lastk.get('time', now)) * 1000)
            try:
                kname = pygame.key.name(lastk.get('key'))
            except Exception:
                kname = str(lastk.get('key'))
            surface.blit(font.render(f"Last key: {kname} age={k_age}ms mods={lastk.get('mods')}", True, (200, 240, 200)), (x, y))
            y += 16

        # List first few widgets
        if self.widget_manager:
            for w in self.widget_manager.widgets[:8]:
                info = f"{getattr(w,'control_name',w.__class__.__name__)} rect={w.rect} open={getattr(w,'is_open',False)} state={w.state}"
                surface.blit(font.render(info, True, (200, 200, 255)), (x, y))
                y += 14

        # Draw outlines for panel, collapse button, and open dropdown menus
        if panel_rect:
            try:
                pygame.draw.rect(surface, (200, 80, 80), panel_rect, 2)
            except Exception:
                pass
        if getattr(self, 'collapse_button_rect', None):
            try:
                pygame.draw.rect(surface, (80, 200, 120), self.collapse_button_rect, 2)
            except Exception:
                pass

        # Recent clicks
        cx = 14
        cy = box_h + 30
        surface.blit(font.render("Recent clicks (latest first):", True, (200, 200, 180)), (cx, cy))
        cy += 14
        for pos, ts in (self.debug_click_log[:8] if getattr(self,'debug_click_log',None) else []):
            surface.blit(font.render(f"{pos} @ {int(ts)}", True, (220, 220, 180)), (cx, cy))
            cy += 12

    def _render_unified_bottom_panel(self):
        """Render unified bottom HUD panel - STATUS and MESSAGE only (inventory moved to sidebar)."""
        # Panel dimensions - reduced height since no inventory
        panel_height = 80
        panel_y = self.screen_h - panel_height - 5
        panel_x = 5
        panel_width = self.screen_w - self.SIDEBAR_WIDTH - 15
        
        # Create rounded background
        panel_surf = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_bg = pygame.Rect(0, 0, panel_width, panel_height)
        pygame.draw.rect(panel_surf, (35, 35, 50, 230), panel_bg, border_radius=8)
        pygame.draw.rect(panel_surf, (60, 60, 80), panel_bg, 2, border_radius=8)
        
        self.screen.blit(panel_surf, (panel_x, panel_y))
        
        # Calculate section widths (only status and message)
        padding = 15
        total_inner_width = panel_width - (padding * 2)
        
        status_width = int(total_inner_width * 0.50)  # 50% for status
        message_width = total_inner_width - status_width  # 50% for message
        
        # Section X positions
        status_x = panel_x + padding
        message_x = status_x + status_width + 20
        
        content_y = panel_y + 10
        section_height = panel_height - 20
        
        # Draw vertical divider
        divider_color = (60, 60, 80)
        pygame.draw.line(self.screen, divider_color,
                        (message_x - 10, content_y),
                        (message_x - 10, content_y + section_height), 2)
        
        # Render sections
        self._render_status_section(status_x, content_y, status_width, section_height)
        self._render_message_section(message_x, content_y, message_width, section_height)
    
    def _render_message_section(self, x: int, y: int, width: int, height: int):
        """Render message/status section in bottom panel."""
        # Title
        title_surf = self.font.render("MESSAGE", True, (100, 255, 150))
        self.screen.blit(title_surf, (x, y))
        y += 22
        
        # Current message with appropriate color
        msg_color = (255, 255, 100) if (self.env and self.env.won) else (200, 200, 200)
        
        # Wrap long messages
        if len(self.message) > 35:
            msg_lines = [self.message[i:i+35] for i in range(0, len(self.message), 35)]
            for line in msg_lines[:2]:  # Max 2 lines
                msg_surf = self.small_font.render(line, True, msg_color)
                self.screen.blit(msg_surf, (x, y))
                y += 16
        else:
            msg_surf = self.small_font.render(self.message, True, msg_color)
            self.screen.blit(msg_surf, (x, y))
    
    def _render_progress_bar(self, surface: pygame.Surface, x: int, y: int, width: int, height: int, 
                             filled: int, total: int, color_filled: tuple, color_empty: tuple):
        """Render a segmented progress bar with filled/empty indicators."""
        if total == 0:
            return
        
        segments = min(total, 10)  # Max 10 segments for visual clarity
        segment_width = width // max(1, segments)
        items_per_segment = total / segments
        
        for i in range(segments):
            segment_x = x + i * segment_width
            segment_rect = pygame.Rect(segment_x + 1, y, segment_width - 2, height)
            
            # Determine if this segment should be filled
            segment_threshold = (i + 1) * items_per_segment
            is_filled = filled >= segment_threshold
            
            if is_filled:
                # Filled segment with gradient effect
                pygame.draw.rect(surface, color_filled, segment_rect, border_radius=2)
                # Lighter border
                highlight = tuple(min(c + 40, 255) for c in color_filled[:3])
                pygame.draw.rect(surface, highlight, segment_rect, 1, border_radius=2)
            else:
                # Empty segment
                pygame.draw.rect(surface, color_empty, segment_rect, border_radius=2)
                pygame.draw.rect(surface, (60, 60, 80), segment_rect, 1, border_radius=2)
    
    def _render_inventory_section(self, x: int, y: int, width: int, height: int):
        """Render inventory section with progress bars and icons."""
        # CRITICAL: Sync inventory counters before rendering to ensure real-time display
        try:
            self._sync_inventory_counters()
        except Exception:
            pass
        
        # DEBUG: Log inventory values every frame for diagnostics
        logger.debug("INVENTORY_RENDER: keys_collected=%d, total_keys=%d, env.state.keys=%d, "
                    "bombs_collected=%d, has_bomb=%s, boss_keys_collected=%d, has_boss_key=%s, "
                    "collected_items_len=%d",
                    self.keys_collected, self.total_keys,
                    getattr(self.env.state, 'keys', 0) if self.env and self.env.state else 0,
                    self.bombs_collected, 
                    getattr(self.env.state, 'has_bomb', False) if self.env and self.env.state else False,
                    self.boss_keys_collected,
                    getattr(self.env.state, 'has_boss_key', False) if self.env and self.env.state else False,
                    len(self.collected_items) if hasattr(self, 'collected_items') else 0)
        
        # Title
        title_surf = self.font.render("INVENTORY", True, (100, 200, 255))
        self.screen.blit(title_surf, (x, y))
        self.screen.blit(title_surf, (x, y))
        
        y_offset = y + 25
        line_height = 24
        bar_width = width - 10
        bar_height = 8
        
        if not self.env:
            return
        
        # Keys with progress bar
        held_keys = self.env.state.keys if hasattr(self.env.state, 'keys') else 0
        keys_color = (255, 215, 0)  # Gold
        
        # Flash animation if recently picked up
        current_time = time.time()
        for pos, pickup_time in list(self.item_pickup_times.items()):
            if current_time - pickup_time < 0.5:
                keys_color = (255, 255, 150)
                break
        
        # Text: "K: 4/7 (3 held)"
        keys_text = f"K: {self.keys_collected}/{self.total_keys}"
        if held_keys > 0:
            keys_text += f" ({held_keys} held)"
        keys_surf = self.small_font.render(keys_text, True, keys_color)
        self.screen.blit(keys_surf, (x, y_offset))
        
        # Progress bar
        if self.total_keys > 0:
            self._render_progress_bar(self.screen, x, y_offset + 16, bar_width, bar_height,
                                     self.keys_collected, self.total_keys,
                                     keys_color, (40, 40, 50))
        y_offset += line_height + 10
        
        # Bombs
        has_bomb = hasattr(self.env.state, 'has_bomb') and self.env.state.has_bomb
        bombs_color = (255, 107, 53) if has_bomb else (100, 100, 100)  # Orange
        bombs_status = "[YES]" if has_bomb else "[NO]"
        
        if self.total_bombs > 0:
            bombs_text = f"B: {bombs_status} Bomb"
            bombs_surf = self.small_font.render(bombs_text, True, bombs_color)
            self.screen.blit(bombs_surf, (x, y_offset))
            y_offset += line_height
        
        # Boss Key with progress bar
        has_boss_key = hasattr(self.env.state, 'has_boss_key') and self.env.state.has_boss_key
        boss_key_color = (176, 66, 255) if has_boss_key else (100, 100, 100)  # Purple
        
        if self.total_boss_keys > 0:
            boss_key_text = f"Boss Key: {self.boss_keys_collected}/{self.total_boss_keys}"
            if has_boss_key:
                boss_key_text += " [Y]"
            boss_key_surf = self.small_font.render(boss_key_text, True, boss_key_color)
            self.screen.blit(boss_key_surf, (x, y_offset))
            
            # Progress bar
            self._render_progress_bar(self.screen, x, y_offset + 16, bar_width, bar_height,
                                     self.boss_keys_collected, self.total_boss_keys,
                                     boss_key_color, (40, 40, 50))
    
    def _render_metrics_section(self, x: int, y: int, width: int, height: int):
        """Render metrics section (steps, speed, zoom, env)."""
        # Title
        title_surf = self.font.render("METRICS", True, (150, 200, 255))
        self.screen.blit(title_surf, (x, y))
        
        y_offset = y + 25
        line_height = 20
        
        # Steps
        steps_surf = self.small_font.render(f"Steps: {self.step_count}", True, (200, 200, 200))
        self.screen.blit(steps_surf, (x, y_offset))
        y_offset += line_height
        
        # Speed
        speed_color = (100, 255, 100) if self.speed_multiplier == 1.0 else (255, 200, 100)
        speed_surf = self.small_font.render(f"Speed: {self.speed_multiplier}x", True, speed_color)
        self.screen.blit(speed_surf, (x, y_offset))
        y_offset += line_height
        
        # Zoom
        zoom_surf = self.small_font.render(f"Zoom: {self.TILE_SIZE}px", True, (150, 150, 150))
        self.screen.blit(zoom_surf, (x, y_offset))
        y_offset += line_height
        
        # Env Steps
        env_steps = self.env.step_count if self.env and hasattr(self.env, 'step_count') else 0
        env_surf = self.small_font.render(f"Env: {env_steps}", True, (150, 150, 150))
        self.screen.blit(env_surf, (x, y_offset))
    
    def _render_controls_section(self, x: int, y: int, width: int, height: int):
        """Render controls section in two-column layout."""
        # Title
        title_surf = self.font.render("CONTROLS", True, (100, 200, 100))
        self.screen.blit(title_surf, (x, y))
        
        y_offset = y + 25
        line_height = 16
        col_width = width // 2
        
        # Two-column layout
        controls_left = [
            ("ARROWS", "Move"),
            ("SPACE", "Solve"),
            ("R", "Reset"),
            ("N/P", "Maps"),
            ("[/]", "Speed"),
        ]
        
        controls_right = [
            ("M", "Minimap"),
            ("H", "Heatmap"),
            ("+/-", "Zoom"),
            ("F11", "Full"),
            ("ESC", "Quit"),
        ]
        
        text_color = (120, 120, 120)
        
        # Left column
        for key, desc in controls_left:
            control_surf = self.small_font.render(f"{key:4s} {desc}", True, text_color)
            self.screen.blit(control_surf, (x, y_offset))
            y_offset += line_height
        
        # Right column
        y_offset = y + 25
        for key, desc in controls_right:
            control_surf = self.small_font.render(f"{key:4s} {desc}", True, text_color)
            self.screen.blit(control_surf, (x + col_width, y_offset))
            y_offset += line_height
    
    def _render_status_section(self, x: int, y: int, width: int, height: int):
        """Render status section with game state information."""
        # Title
        title_surf = self.font.render("STATUS", True, (180, 220, 255))
        self.screen.blit(title_surf, (x, y))
        
        y_offset = y + 25
        line_height = 18
        
        # Victory or current status
        if self.env and self.env.won:
            status_text = "*** VICTORY! ***"
            status_color = (255, 215, 0)
            status_surf = self.big_font.render(status_text, True, status_color)
            self.screen.blit(status_surf, (x, y_offset))
        else:
            # Current map/dungeon
            if self.current_map_idx < len(self.map_names):
                map_name = self.map_names[self.current_map_idx]
                map_text = f"Map: {map_name[:15]}"
                map_surf = self.small_font.render(map_text, True, (150, 200, 255))
                self.screen.blit(map_surf, (x, y_offset))
                y_offset += line_height
            
            # Position
            if self.env and hasattr(self.env.state, 'position'):
                pos = self.env.state.position
                pos_text = f"Pos: ({pos[0]}, {pos[1]})"
                pos_surf = self.small_font.render(pos_text, True, (150, 150, 150))
                self.screen.blit(pos_surf, (x, y_offset))
                y_offset += line_height
            
            # Auto-solve progress
            if self.auto_mode and self.auto_path:
                progress_text = f"Auto: {self.auto_step_idx}/{len(self.auto_path)}"
                progress_surf = self.small_font.render(progress_text, True, (100, 255, 150))
                self.screen.blit(progress_surf, (x, y_offset))
                y_offset += line_height
            
            # Status message
            if self.status_message:
                status_surf = self.small_font.render(self.status_message[:20], True, (180, 220, 255))
                self.screen.blit(status_surf, (x, y_offset))
    
    def _render_minimap(self):
        """Render small dungeon overview map in bottom-right corner."""
        if not self.env:
            return
        
        # Minimap positioning (bottom-right, above HUD)
        minimap_margin = 20
        minimap_x = self.screen_w - self.SIDEBAR_WIDTH - self.minimap_size - minimap_margin
        minimap_y = self.screen_h - self.HUD_HEIGHT - self.minimap_size - minimap_margin
        
        # Create semi-transparent minimap surface
        minimap = pygame.Surface((self.minimap_size, self.minimap_size), pygame.SRCALPHA)
        pygame.draw.rect(minimap, (40, 40, 60, 220), minimap.get_rect(), border_radius=8)
        
        # Draw title
        title_font = pygame.font.SysFont('Arial', 10, bold=True)
        title_surf = title_font.render("Dungeon Map", True, (180, 180, 200))
        minimap.blit(title_surf, (5, 3))
        
        # Calculate scaling factor to fit dungeon in minimap
        map_h, map_w = self.env.height, self.env.width
        content_area = self.minimap_size - 30  # Leave room for title and padding
        scale_x = content_area / map_w
        scale_y = content_area / map_h
        scale = min(scale_x, scale_y)
        
        # Calculate offset to center the minimap content
        scaled_w = int(map_w * scale)
        scaled_h = int(map_h * scale)
        offset_x = (self.minimap_size - scaled_w) // 2
        offset_y = 18 + (self.minimap_size - 18 - scaled_h) // 2
        
        # Draw simplified dungeon layout
        # Use different colors for different tile types
        for r in range(map_h):
            for c in range(map_w):
                tile_id = self.env.grid[r, c]
                
                # Determine tile color based on semantic type
                if tile_id == SEMANTIC_PALETTE['VOID']:
                    continue  # Skip void tiles (transparent)
                elif tile_id == SEMANTIC_PALETTE['WALL'] or tile_id == SEMANTIC_PALETTE['BLOCK']:
                    color = (60, 60, 80)  # Dark gray for walls
                elif tile_id == SEMANTIC_PALETTE['START']:
                    color = (80, 180, 80)  # Green for start
                elif tile_id == SEMANTIC_PALETTE['TRIFORCE']:
                    color = (255, 215, 0)  # Gold for goal
                elif tile_id in [SEMANTIC_PALETTE['KEY_SMALL'], SEMANTIC_PALETTE['KEY_BOSS']]:
                    color = (255, 200, 50)  # Yellow for keys
                elif tile_id in [SEMANTIC_PALETTE['DOOR_LOCKED'], SEMANTIC_PALETTE['DOOR_BOMB'], SEMANTIC_PALETTE['DOOR_BOSS']]:
                    color = (180, 100, 50)  # Brown for locked doors
                elif tile_id == SEMANTIC_PALETTE['STAIR']:
                    color = (100, 150, 255)  # Blue for stairs
                elif tile_id == SEMANTIC_PALETTE['ENEMY']:
                    color = (200, 50, 50)  # Red for enemies
                else:
                    color = (100, 120, 140)  # Light gray for floors
                
                # Draw mini-tile
                mini_x = offset_x + int(c * scale)
                mini_y = offset_y + int(r * scale)
                mini_w = max(1, int(scale))
                mini_h = max(1, int(scale))
                pygame.draw.rect(minimap, color, (mini_x, mini_y, mini_w, mini_h))
        
        # Draw current player position (bright dot)
        pr, pc = self.env.state.position
        player_x = offset_x + int(pc * scale)
        player_y = offset_y + int(pr * scale)
        player_size = max(2, int(scale * 1.5))
        pygame.draw.circle(minimap, (255, 100, 100), (player_x, player_y), player_size)
        # Add white outline for visibility
        pygame.draw.circle(minimap, (255, 255, 255), (player_x, player_y), player_size + 1, 1)
        
        # Highlight uncollected items with pulsing effect
        current_time = time.time()
        pulse = (math.sin(current_time * 3) + 1) / 2  # 0 to 1
        
        # Draw uncollected keys (yellow pulsing dots)
        for pos in self.env._find_all_positions(SEMANTIC_PALETTE['KEY_SMALL']):
            if pos not in self.env.state.collected_items:
                r, c = pos
                mini_x = offset_x + int(c * scale)
                mini_y = offset_y + int(r * scale)
                size = int(2 + pulse * 2)
                pygame.draw.circle(minimap, (255, 255, 0), (mini_x, mini_y), size)
        
        # Draw uncollected boss keys (orange pulsing dots)
        for pos in self.env._find_all_positions(SEMANTIC_PALETTE['KEY_BOSS']):
            if pos not in self.env.state.collected_items:
                r, c = pos
                mini_x = offset_x + int(c * scale)
                mini_y = offset_y + int(r * scale)
                size = int(2 + pulse * 2)
                pygame.draw.circle(minimap, (255, 150, 0), (mini_x, mini_y), size)
        
        # Draw border
        pygame.draw.rect(minimap, (70, 70, 100), minimap.get_rect(), 2, border_radius=8)
        
        # Blit minimap to screen
        self.screen.blit(minimap, (minimap_x, minimap_y))
    
    def _handle_minimap_click(self, mouse_pos: Tuple[int, int]) -> bool:
        """Handle mouse click on minimap to jump to that location."""
        if not self.show_minimap or not self.env:
            return False
        
        # Calculate minimap position
        minimap_margin = 20
        minimap_x = self.screen_w - self.SIDEBAR_WIDTH - self.minimap_size - minimap_margin
        minimap_y = self.screen_h - self.HUD_HEIGHT - self.minimap_size - minimap_margin
        
        # Check if click is within minimap bounds
        mx, my = mouse_pos
        if not (minimap_x <= mx <= minimap_x + self.minimap_size and
                minimap_y <= my <= minimap_y + self.minimap_size):
            return False
        
        # Convert mouse position to map coordinates
        map_h, map_w = self.env.height, self.env.width
        content_area = self.minimap_size - 30
        scale_x = content_area / map_w
        scale_y = content_area / map_h
        scale = min(scale_x, scale_y)
        
        scaled_w = int(map_w * scale)
        scaled_h = int(map_h * scale)
        offset_x = (self.minimap_size - scaled_w) // 2
        offset_y = 18 + (self.minimap_size - 18 - scaled_h) // 2
        
        # Calculate clicked tile
        local_x = mx - minimap_x - offset_x
        local_y = my - minimap_y - offset_y
        
        if local_x < 0 or local_y < 0:
            return True
        
        tile_c = int(local_x / scale)
        tile_r = int(local_y / scale)
        
        if 0 <= tile_r < map_h and 0 <= tile_c < map_w:
            # Center view on clicked tile
            self.view_offset_x = int(tile_c * self.TILE_SIZE - (self.screen_w - self.SIDEBAR_WIDTH) / 2)
            self.view_offset_y = int(tile_r * self.TILE_SIZE - (self.screen_h - self.HUD_HEIGHT) / 2)
            self._clamp_view_offset()
            self.message = f"Jumped to ({tile_r}, {tile_c})"
        
        return True
    
    def _render_help_overlay(self):
        """Render help overlay."""
        overlay = pygame.Surface((self.screen_w, self.screen_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))
        
        help_lines = [
            "ZAVE - Zelda AI Validation Environment",
            "",
            "Movement:",
            "  Arrow Keys - Move Link",
            "  Mouse Wheel - Zoom in/out",
            "  Middle Mouse Drag - Pan view",
            "",
            "Actions:",
            "  SPACE - Run A* auto-solver",
            "  R - Reset current map",
            "  N - Next map",
            "  P - Previous map",
            "",
            "View:",
            "  +/- or Wheel - Zoom in/out",
            "  0 - Reset zoom to default",
            "  C - Center view on player",
            "  M - Toggle minimap",
            "  H - Toggle A* heatmap",
            "  TAB - Toggle control panel",
            "  F11 - Toggle fullscreen",
            "",
            "Speed Control:",
            "  [ or , - Decrease speed",
            "  ] or . - Increase speed",
            "  (Speeds: 0.25x, 0.5x, 1x, 2x, 5x, 10x)",
            "",
            "Press F1 or ESC to close this help",
        ]
        
        y = 50
        for line in help_lines:
            if line.startswith("ZAVE"):
                surf = self.big_font.render(line, True, (100, 200, 255))
            elif line.endswith(":") and not line.startswith(" "):
                surf = self.font.render(line, True, (255, 200, 100))
            else:
                surf = self.small_font.render(line, True, (200, 200, 200))
            self.screen.blit(surf, (50, y))
            y += 22 if line else 10


def load_maps_from_adapter():
    """Load processed maps from data adapter using new zelda_core - ALL 18 variants."""
    try:
        from src.data.zelda_core import ZeldaDungeonAdapter, DungeonSolver
        from pathlib import Path
        
        data_root = Path(__file__).parent / "Data" / "The Legend of Zelda"
        
        if not data_root.exists():
            print(f"Data folder not found: {data_root}")
            return None, None
        
        adapter = ZeldaDungeonAdapter(str(data_root))
        solver = DungeonSolver()
        
        maps = []  # Store full StitchedDungeon objects
        map_names = []  # Track dungeon names
        print("Loading all 18 dungeon variants (9 dungeons x 2 variants)...")
        
        for dungeon_num in range(1, 10):
            for variant in [1, 2]:
                try:
                    dungeon = adapter.load_dungeon(dungeon_num, variant=variant)
                    stitched = adapter.stitch_dungeon(dungeon)

                    # Store the full stitched dungeon (includes graph and room mappings)
                    maps.append(stitched)

                    # Store name
                    quest_name = "Quest 1" if variant == 1 else "Quest 2"
                    map_names.append(f"Dungeon {dungeon_num} ({quest_name})")

                    # Fast startup: do not block on expensive solvability checks here.
                    print(f"  D{dungeon_num}-{variant}: Loaded - {stitched.global_grid.shape}")
                except Exception as e:
                    print(f"  D{dungeon_num}-{variant}: Error - {e}")

        # If requested, perform precalculation asynchronously so startup is not blocked
        if os.environ.get('KLTN_PRECALC_SOLVES', '0') == '1':
            try:
                import threading
                def _precalc_worker():
                    print('Starting background precalc solves for loaded maps...')
                    for idx, m in enumerate(maps):
                        try:
                            r = solver.solve(m)
                            status = '[OK]' if r.get('solvable') else '[X]'
                            print(f"  [precalc] Map {idx+1}: {status}")
                        except Exception as e:
                            print(f"  [precalc] Map {idx+1}: Error - {e}")
                threading.Thread(target=_precalc_worker, daemon=True).start()
            except Exception:
                print('Precalc worker failed to start')
        
        return maps if maps else None, map_names if map_names else None
        
    except Exception as e:
        print(f"Error loading maps: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """Main entry point."""
    print("=== ZAVE GUI Runner ===\n")
    
    if not PYGAME_AVAILABLE:
        print("Pygame is not installed. Please run: pip install pygame")
        return
    
    # Try to load processed maps
    maps, map_names = load_maps_from_adapter()
    
    if maps:
        print(f"Loaded {len(maps)} maps from data adapter")
    else:
        print("Using test map")
        maps = [create_test_map()]
        map_names = ["Test Map"]
    
    # Start GUI
    gui = ZeldaGUI(maps, map_names)
    gui.run()


if __name__ == "__main__":
    # Required for multiprocessing on Windows (freeze_support)
    multiprocessing.freeze_support()
    main()
