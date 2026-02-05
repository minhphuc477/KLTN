"""
FORENSIC INVESTIGATION: Solver Failure Analysis
================================================
This script diagnoses why the solver fails to find a path for D1-1.

Checks performed:
1. Start/goal validity (in bounds, not walls)
2. Goal detection logic
3. State expansion correctness
4. Dominance pruning aggressiveness
5. Timeout sufficiency
6. Heuristic admissibility
"""

import os
import sys
import logging
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import required modules
from Data.zelda_core import SEMANTIC_PALETTE, ID_TO_NAME, ZeldaDungeonAdapter
from simulation.validator import ZeldaLogicEnv, StateSpaceAStar, GameState

def diagnose_map(dungeon_num: int = 1):
    """Run comprehensive forensic analysis on a specific map."""
    
    map_name = f"D{dungeon_num}-1"
    logger.info("=" * 80)
    logger.info(f"FORENSIC INVESTIGATION: {map_name}")
    logger.info("=" * 80)
    
    # Load map using adapter
    data_root = os.path.join(os.path.dirname(__file__), 'Data', 'The Legend of Zelda')
    adapter = ZeldaDungeonAdapter(data_root=data_root)
    dungeon = adapter.load_dungeon(dungeon_num, variant=1)
    
    # Stitch dungeon to get global grid
    stitched = adapter.stitch_dungeon(dungeon)
    
    grid = stitched.global_grid
    graph = stitched.graph
    
    logger.info(f"\n1. MAP STRUCTURE:")
    logger.info(f"   Grid shape: {grid.shape}")
    logger.info(f"   Has graph: {graph is not None}")
    logger.info(f"   Graph nodes: {len(graph.nodes()) if graph else 0}")
    
    # Find start and goal positions
    start_positions = np.argwhere(grid == SEMANTIC_PALETTE['START'])
    goal_positions = np.argwhere(grid == SEMANTIC_PALETTE['TRIFORCE'])
    
    logger.info(f"\n2. START/GOAL POSITIONS:")
    logger.info(f"   Start positions found: {len(start_positions)}")
    if len(start_positions) > 0:
        start_pos = tuple(start_positions[0])
        logger.info(f"   Start: {start_pos} -> tile={grid[start_pos]} ({ID_TO_NAME.get(grid[start_pos], 'UNKNOWN')})")
    else:
        logger.error("   NO START POSITION FOUND!")
        start_pos = None
    
    logger.info(f"   Goal positions found: {len(goal_positions)}")
    if len(goal_positions) > 0:
        goal_pos = tuple(goal_positions[0])
        logger.info(f"   Goal: {goal_pos} -> tile={grid[goal_pos]} ({ID_TO_NAME.get(grid[goal_pos], 'UNKNOWN')})")
    else:
        logger.error("   NO GOAL POSITION FOUND!")
        goal_pos = None
    
    if start_pos is None or goal_pos is None:
        logger.error("FATAL: Missing start or goal position!")
        return
    
    # Initialize environment
    env = ZeldaLogicEnv(semantic_grid=grid, graph=graph)
    
    logger.info(f"\n3. ENVIRONMENT INITIALIZATION:")
    logger.info(f"   Env start_pos: {env.start_pos}")
    logger.info(f"   Env goal_pos: {env.goal_pos}")
    logger.info(f"   Match expected: {env.start_pos == start_pos and env.goal_pos == goal_pos}")
    
    # Check immediate neighbors of start
    logger.info(f"\n4. START POSITION NEIGHBORS:")
    start_r, start_c = env.start_pos
    deltas = [(-1, 0, 'UP'), (1, 0, 'DOWN'), (0, -1, 'LEFT'), (0, 1, 'RIGHT')]
    for dr, dc, direction in deltas:
        nr, nc = start_r + dr, start_c + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
            tile_id = grid[nr, nc]
            tile_name = ID_TO_NAME.get(tile_id, f'ID_{tile_id}')
            logger.info(f"   {direction:6s}: ({nr:2d}, {nc:2d}) -> {tile_name:15s} (walkable: {tile_id in [1, 10, 21, 22]})")
        else:
            logger.info(f"   {direction:6s}: OUT OF BOUNDS")
    
    # Check immediate neighbors of goal
    logger.info(f"\n5. GOAL POSITION NEIGHBORS:")
    goal_r, goal_c = env.goal_pos
    for dr, dc, direction in deltas:
        nr, nc = goal_r + dr, goal_c + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
            tile_id = grid[nr, nc]
            tile_name = ID_TO_NAME.get(tile_id, f'ID_{tile_id}')
            logger.info(f"   {direction:6s}: ({nr:2d}, {nc:2d}) -> {tile_name:15s} (walkable: {tile_id in [1, 10, 21, 22]})")
        else:
            logger.info(f"   {direction:6s}: OUT OF BOUNDS")
    
    # Count tile types
    logger.info(f"\n6. TILE TYPE DISTRIBUTION:")
    unique, counts = np.unique(grid, return_counts=True)
    for tile_id, count in sorted(zip(unique, counts), key=lambda x: -x[1])[:15]:
        tile_name = ID_TO_NAME.get(int(tile_id), f'ID_{int(tile_id)}')
        logger.info(f"   {tile_name:20s}: {count:4d} tiles")
    
    # Run solver with VERBOSE logging
    logger.info(f"\n7. RUNNING SOLVER (15K timeout):")
    logger.info("   Creating StateSpaceAStar solver...")
    
    # Test with different timeouts
    for timeout in [1000, 5000, 15000, 30000]:
        logger.info(f"\n   --- Testing with timeout={timeout} ---")
        solver = StateSpaceAStar(env, timeout=timeout, heuristic_mode="balanced")
        
        # Run solver
        success, path, states_explored = solver.solve()
        
        logger.info(f"   Result: success={success}, path_length={len(path)}, states_explored={states_explored}")
        
        if success:
            logger.info(f"   PATH FOUND!")
            logger.info(f"   Path (first 10): {path[:10]}")
            logger.info(f"   Path (last 10): {path[-10:]}")
            break
        else:
            logger.warning(f"   NO PATH FOUND (explored {states_explored}/{timeout} states)")
            
            # Check if we hit timeout
            if states_explored >= timeout:
                logger.warning(f"   TIMEOUT HIT - need to increase limit or optimize")
            else:
                logger.error(f"   SEARCH EXHAUSTED - no valid path exists OR logic bug!")
    
    # Additional diagnostics if failed
    if not success:
        logger.info(f"\n8. DETAILED DIAGNOSTICS (Dominance Pruning Analysis):")
        
        # Run solver again with manual state tracking
        logger.info("   Re-running solver with instrumentation...")
        
        # Check if dominance pruning is too aggressive
        env.reset()
        solver = StateSpaceAStar(env, timeout=1000, heuristic_mode="balanced")
        
        # Manually trace first few states
        logger.info("   Initial state:")
        start_state = env.state
        logger.info(f"     Position: {start_state.position}")
        logger.info(f"     Keys: {start_state.keys}")
        logger.info(f"     Has bomb: {start_state.has_bomb}")
        logger.info(f"     Has boss_key: {start_state.has_boss_key}")
        logger.info(f"     Has item: {start_state.has_item}")
        logger.info(f"     Opened doors: {len(start_state.opened_doors)}")
        logger.info(f"     Collected items: {len(start_state.collected_items)}")
        
        # Test heuristic
        h = solver._heuristic(start_state)
        logger.info(f"     Heuristic: {h}")
        
        # Check if goal is directly reachable (Manhattan distance)
        manhattan = abs(start_pos[0] - goal_pos[0]) + abs(start_pos[1] - goal_pos[1])
        logger.info(f"     Manhattan distance to goal: {manhattan}")
        
        logger.info(f"\n9. HYPOTHESIS TESTING:")
        logger.info(f"   H1 (Dominance too aggressive): Need to check pruned states")
        logger.info(f"   H2 (Timeout too low): {states_explored} states explored")
        logger.info(f"   H3 (Goal detection bug): Goal at {goal_pos}, looking for exact match")
        logger.info(f"   H4 (Invalid neighbors): Check _try_move_pure logic")
        logger.info(f"   H5 (Inadmissible heuristic): h={h} vs manhattan={manhattan}")
        
        # Check if goal is in closed set (means we visited but didn't recognize)
        logger.info(f"\n10. CRITICAL CHECK - Goal Recognition:")
        logger.info(f"    Goal position: {env.goal_pos}")
        logger.info(f"    Goal tile ID: {grid[env.goal_pos]} ({ID_TO_NAME.get(grid[env.goal_pos], 'UNKNOWN')})")
        logger.info(f"    Expected tile: TRIFORCE (ID={SEMANTIC_PALETTE['TRIFORCE']})")


def main():
    """Run forensic analysis."""
    dungeon_num = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    diagnose_map(dungeon_num)


if __name__ == "__main__":
    main()
