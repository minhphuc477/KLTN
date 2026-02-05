"""
EMERGENCY DIAGNOSTIC: Real-time State Exploration Logger
=========================================================
This script instruments the solver to show WHAT states are being explored.
"""

import os
import sys
import numpy as np
from collections import defaultdict

from Data.zelda_core import ZeldaDungeonAdapter, SEMANTIC_PALETTE, ID_TO_NAME
from simulation.validator import ZeldaLogicEnv, StateSpaceAStar

def create_instrumented_solver(env, timeout=5000):
    """Create solver with logging hooks."""
    solver = StateSpaceAStar(env, timeout=timeout, heuristic_mode="balanced")
    
    # Track exploration pattern
    exploration_log = {
        'by_position': defaultdict(int),
        'unique_states': 0,
        'dominated_pruned': 0,
        'closed_duplicates': 0,
        'samples': []
    }
    
    # Wrap the solve method
    original_solve = solver.solve
    
    def instrumented_solve():
        import heapq
        
        # Copy initial setup from original solve
        open_set = []
        closed_set = set()
        solver._best_at_pos = {}
        
        initial_state = env.state.copy()
        initial_hash = hash(initial_state)
        start_pos = initial_state.position
        goal_pos = env.goal_pos
        
        # Heuristic
        h = solver._heuristic(initial_state)
        g = 0
        f = g + h
        
        counter = 0
        heapq.heappush(open_set, (f, counter, initial_hash, initial_state, [start_pos]))
        counter += 1
        
        states_explored = 0
        dominated_states_pruned = 0
        
        print(f"\n[INSTRUMENTED SOLVER START]")
        print(f"   Start: {start_pos}, Goal: {goal_pos}")
        print(f"   Initial heuristic: {h}")
        print(f"   Timeout: {timeout}\n")
        
        # Log every N states
        LOG_INTERVAL = 500
        
        grid = env.grid
        height, width = grid.shape
        
        # Cardinal deltas
        cardinal_deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        CARDINAL_COST = 1.0
        
        while open_set and states_explored < timeout:
            entry = heapq.heappop(open_set)
            
            if len(entry) == 5:
                _, _, state_hash, current_state, path = entry
            else:
                priority, state_hash, current_state, path = entry
            
            if state_hash in closed_set:
                exploration_log['closed_duplicates'] += 1
                continue
            
            # Dominance pruning check (copy from actual solver)
            is_dominated = False
            if current_state.position in solver._best_at_pos:
                best = solver._best_at_pos[current_state.position]
                if (current_state.keys <= best.keys and 
                    int(current_state.has_bomb) <= int(best.has_bomb) and
                    int(current_state.has_boss_key) <= int(best.has_boss_key) and
                    int(current_state.has_item) <= int(best.has_item) and
                    current_state.opened_doors.issubset(best.opened_doors) and
                    current_state.collected_items.issubset(best.collected_items)):
                    if (current_state.keys < best.keys or 
                        int(current_state.has_bomb) < int(best.has_bomb) or
                        int(current_state.has_boss_key) < int(best.has_boss_key) or
                        int(current_state.has_item) < int(best.has_item) or
                        len(current_state.opened_doors) < len(best.opened_doors) or
                        len(current_state.collected_items) < len(best.collected_items)):
                        is_dominated = True
            
            if is_dominated:
                dominated_states_pruned += 1
                exploration_log['dominated_pruned'] += 1
                continue
            
            # Update best state
            if current_state.position not in solver._best_at_pos:
                solver._best_at_pos[current_state.position] = current_state
            else:
                best = solver._best_at_pos[current_state.position]
                if (current_state.keys >= best.keys and
                    int(current_state.has_bomb) >= int(best.has_bomb) and
                    int(current_state.has_boss_key) >= int(best.has_boss_key) and
                    int(current_state.has_item) >= int(best.has_item) and
                    current_state.opened_doors.issuperset(best.opened_doors) and
                    current_state.collected_items.issuperset(best.collected_items)):
                    if (current_state.keys > best.keys or
                        len(current_state.opened_doors) > len(best.opened_doors) or
                        len(current_state.collected_items) > len(best.collected_items) or
                        int(current_state.has_bomb) > int(best.has_bomb) or
                        int(current_state.has_boss_key) > int(best.has_boss_key) or
                        int(current_state.has_item) > int(best.has_item)):
                        solver._best_at_pos[current_state.position] = current_state
            
            closed_set.add(state_hash)
            states_explored += 1
            exploration_log['unique_states'] += 1
            exploration_log['by_position'][current_state.position] += 1
            
            # Log progress
            if states_explored % LOG_INTERVAL == 0 or states_explored < 10:
                curr_h = solver._heuristic(current_state)
                print(f"   [{states_explored:5d}] pos={current_state.position}, "
                      f"keys={current_state.keys}, h={curr_h:.1f}, "
                      f"tile={ID_TO_NAME.get(grid[current_state.position], 'UNK')}")
                
                # Sample for detailed analysis
                if len(exploration_log['samples']) < 20:
                    exploration_log['samples'].append({
                        'step': states_explored,
                        'pos': current_state.position,
                        'keys': current_state.keys,
                        'h': curr_h
                    })
            
            # Check win
            if current_state.position == goal_pos:
                print(f"\n[SUCCESS] GOAL REACHED at state {states_explored}!")
                return True, path, states_explored
            
            # Explore neighbors
            curr_r, curr_c = current_state.position
            neighbors_generated = 0
            
            for dr, dc in cardinal_deltas:
                new_r, new_c = curr_r + dr, curr_c + dc
                
                if not (0 <= new_r < height and 0 <= new_c < width):
                    continue
                
                target_pos = (new_r, new_c)
                target_tile = grid[new_r, new_c]
                
                # Try move (simplified - check basic walkability)
                if target_tile in {SEMANTIC_PALETTE['VOID'], SEMANTIC_PALETTE['WALL']}:
                    continue
                
                # Create new state (simplified - no full logic)
                new_state = current_state.copy()
                new_state.position = target_pos
                new_hash = hash(new_state)
                
                if new_hash not in closed_set:
                    new_g = len(path)
                    new_h = solver._heuristic(new_state)
                    new_f = new_g + new_h
                    new_path = path + [target_pos]
                    
                    heapq.heappush(open_set, (new_f, counter, new_hash, new_state, new_path))
                    counter += 1
                    neighbors_generated += 1
        
        print(f"\n[FAILURE] NO PATH FOUND (timeout reached)")
        print(f"   States explored: {states_explored}")
        print(f"   Dominated pruned: {dominated_states_pruned}")
        print(f"   Unique positions: {len(solver._best_at_pos)}")
        
        return False, [], states_explored
    
    solver.solve = instrumented_solve
    return solver, exploration_log


def main():
    """Run emergency diagnostics."""
    print("="*80)
    print("EMERGENCY DIAGNOSTIC: D1-1 Solver Instrumentation")
    print("="*80)
    
    # Load D1-1
    data_root = os.path.join(os.path.dirname(__file__), 'Data', 'The Legend of Zelda')
    adapter = ZeldaDungeonAdapter(data_root=data_root)
    dungeon = adapter.load_dungeon(1, variant=1)
    stitched = adapter.stitch_dungeon(dungeon)
    
    grid = stitched.global_grid
    graph = stitched.graph
    
    print(f"\nMap: D1-1")
    print(f"Grid: {grid.shape}")
    print(f"Graph nodes: {len(graph.nodes()) if graph else 0}")
    
    # Create environment
    env = ZeldaLogicEnv(semantic_grid=grid, graph=graph)
    
    print(f"Start: {env.start_pos}")
    print(f"Goal: {env.goal_pos}")
    print(f"Manhattan distance: {abs(env.start_pos[0] - env.goal_pos[0]) + abs(env.start_pos[1] - env.goal_pos[1])}")
    
    # Run instrumented solver
    solver, log = create_instrumented_solver(env, timeout=5000)
    success, path, states = solver.solve()
    
    print(f"\n{'='*80}")
    print("ANALYSIS:")
    print(f"{'='*80}")
    print(f"Result: {'SUCCESS' if success else 'FAILURE'}")
    print(f"States explored: {states}")
    print(f"Dominated pruned: {log['dominated_pruned']}")
    print(f"Unique positions visited: {len(log['by_position'])}")
    
    # Show most visited positions
    print(f"\nTop 20 most visited positions:")
    sorted_positions = sorted(log['by_position'].items(), key=lambda x: -x[1])[:20]
    for pos, count in sorted_positions:
        tile_id = grid[pos]
        tile_name = ID_TO_NAME.get(tile_id, f'ID_{tile_id}')
        print(f"  {pos}: {count} visits, tile={tile_name}")
    
    print(f"\n{'='*80}")
    print("CONCLUSION:")
    print(f"{'='*80}")
    
    if not success:
        print("[X] Solver is STUCK - hitting timeout every time")
        print("\nPossible causes:")
        print("  1. Map is genuinely unsolvable (path blocked)")
        print("  2. Solver logic has a bug (not generating correct neighbors)")
        print("  3. Heuristic is misleading the search")
        print("  4. State space is too large (need better pruning)")
        
        # Check if we're even moving toward goal
        samples = log['samples']
        if samples:
            print(f"\nSample trajectory:")
            for s in samples[:10]:
                print(f"  Step {s['step']:5d}: pos={s['pos']}, keys={s['keys']}, h={s['h']:.1f}")


if __name__ == "__main__":
    main()
