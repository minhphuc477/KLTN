"""Debug script to investigate A* solver failures."""

import numpy as np
import sys
sys.path.insert(0, r"C:\Users\MPhuc\Desktop\KLTN\Data")

from adapter import IntelligentDataAdapter, SEMANTIC_PALETTE
from stitcher import DungeonStitcher
from simulation.validator import ZeldaValidator

def debug_dungeon(dungeon_id: str):
    """Debug a specific dungeon's solver behavior."""
    print(f"\n{'='*70}")
    print(f"DEBUGGING: {dungeon_id}")
    print(f"{'='*70}")
    
    # Process all dungeons
    data_root = r"C:\Users\MPhuc\Desktop\KLTN\Data\The Legend of Zelda"
    adapter = IntelligentDataAdapter(data_root)
    dungeons = adapter.process_all_dungeons()
    
    if dungeon_id not in dungeons:
        print(f"ERROR: {dungeon_id} not found!")
        return None
    
    dungeon_data = dungeons[dungeon_id]
    
    print(f"Rooms: {len(dungeon_data.rooms)}")
    print(f"Graph: {dungeon_data.graph.number_of_nodes()} nodes, {dungeon_data.graph.number_of_edges()} edges")
    
    # Find which room contains the START
    start_room = None
    for room_id, room in dungeon_data.rooms.items():
        if room.grid is not None and SEMANTIC_PALETTE['START'] in room.grid:
            start_room = room_id
            print(f"START found in room: {room_id}")
            break
    
    if start_room:
        # Find neighbors of start room in graph
        start_node = int(start_room) if start_room.isdigit() else start_room
        try:
            neighbors = list(dungeon_data.graph.neighbors(start_node))
            print(f"START room neighbors in graph: {neighbors}")
        except:
            print(f"Could not find graph node for {start_room}")
    
    print(f"\nGraph edges (first 20):")
    for i, (u, v, data) in enumerate(dungeon_data.graph.edges(data=True)):
        if i >= 20:
            break
        print(f"  {u} -> {v}: {data}")
    
    # Stitch dungeon
    stitcher = DungeonStitcher(dungeon_data)
    stitched = stitcher.stitch()
    
    print(f"Stitched grid: {stitched.global_grid.shape}")
    print(f"Start: {stitched.start_pos}")
    print(f"Goal: {stitched.goal_pos}")
    
    # DEBUG: Check room layout positions
    print(f"\nRoom layout positions:")
    for room_id, (r, c) in sorted(stitcher.layout.items()):
        print(f"  Room {room_id}: layout_pos=({r}, {c})")
    
    # Check if rooms 7 and 8 are adjacent in layout
    if 7 in stitcher.layout and 8 in stitcher.layout:
        r7, c7 = stitcher.layout[7]
        r8, c8 = stitcher.layout[8]
        print(f"\nRoom 7 layout: ({r7}, {c7})")
        print(f"Room 8 layout: ({r8}, {c8})")
        print(f"Adjacent? {(r7 == r8 and abs(c7 - c8) == 1) or (c7 == c8 and abs(r7 - r8) == 1)}")
    
    # Check tile at START position
    if stitched.start_pos:
        sr, sc = stitched.start_pos
        start_tile = stitched.global_grid[sr, sc]
        print(f"Tile at START: {start_tile} (expected {SEMANTIC_PALETTE['START']}={SEMANTIC_PALETTE['START']})")
        
        # Check neighbors of START
        print("\nNeighbors of START:")
        deltas = [(-1, 0, "UP"), (1, 0, "DOWN"), (0, -1, "LEFT"), (0, 1, "RIGHT")]
        for dr, dc, direction in deltas:
            nr, nc = sr + dr, sc + dc
            if 0 <= nr < stitched.global_grid.shape[0] and 0 <= nc < stitched.global_grid.shape[1]:
                neighbor_tile = stitched.global_grid[nr, nc]
                tile_name = [k for k, v in SEMANTIC_PALETTE.items() if v == neighbor_tile]
                print(f"  {direction}: tile={neighbor_tile} ({tile_name[0] if tile_name else 'UNKNOWN'})")
    
    # Check tile at GOAL position
    if stitched.goal_pos:
        gr, gc = stitched.goal_pos
        goal_tile = stitched.global_grid[gr, gc]
        print(f"\nTile at GOAL: {goal_tile} (expected {SEMANTIC_PALETTE['TRIFORCE']}={SEMANTIC_PALETTE['TRIFORCE']})")
    
    # Check grid statistics
    unique_tiles, counts = np.unique(stitched.global_grid, return_counts=True)
    print("\nGrid composition:")
    for tile_id, count in zip(unique_tiles, counts):
        tile_name = [k for k, v in SEMANTIC_PALETTE.items() if v == tile_id]
        print(f"  {tile_id:2d} ({tile_name[0] if tile_name else 'UNKNOWN'}): {count:5d} tiles")
    
    # Try to validate
    print("\nAttempting validation with debug...")
    
    # Create validator with modified A* for debugging
    from simulation.validator import ZeldaLogicEnv, StateSpaceAStar, GameState
    import heapq
    
    env = ZeldaLogicEnv(stitched.global_grid, render_mode=False)
    
    print(f"Environment initialized:")
    print(f"  Start pos: {env.start_pos}")
    print(f"  Goal pos: {env.goal_pos}")
    print(f"  Grid shape: {env.original_grid.shape}")
    
    # Run A* with inline debugging
    solver = StateSpaceAStar(env, timeout=100000, heuristic_mode="balanced")
    
    env.reset()
    
    if env.goal_pos is None:
        print("ERROR: No goal position!")
        return stitched
        
    if env.start_pos is None:
        print("ERROR: No start position!")
        return stitched
    
    grid = env.original_grid
    height, width = grid.shape
    
    start_state = env.state.copy()
    start_h = solver._heuristic(start_state)
    
    print(f"\nStart state:")
    print(f"  Position: {start_state.position}")
    print(f"  Keys: {start_state.keys}")
    print(f"  Boss key: {start_state.has_boss_key}")
    print(f"  Heuristic: {start_h}")
    
    open_set = [(start_h, 0, hash(start_state), start_state, [start_state.position])]
    heapq.heapify(open_set)
    
    closed_set = set()
    g_scores = {hash(start_state): 0}
    
    states_explored = 0
    counter = 1
    
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    print(f"\nStarting A* search...")
    print(f"Goal position: {env.goal_pos}")
    
    while open_set and states_explored < 20:  # Limit to 20 for debugging
        f_score, _, state_hash, current_state, path = heapq.heappop(open_set)
        
        if state_hash in closed_set:
            continue
        
        closed_set.add(state_hash)
        states_explored += 1
        
        print(f"\nState {states_explored}:")
        print(f"  Position: {current_state.position}")
        print(f"  f_score: {f_score:.2f}")
        print(f"  Path length: {len(path)}")
        
        # Check win condition
        if current_state.position == env.goal_pos:
            print(f"FOUND GOAL! Path length: {len(path)}")
            break
        
        # Explore neighbors
        curr_r, curr_c = current_state.position
        valid_moves = 0
        
        for dr, dc in deltas:
            new_r, new_c = curr_r + dr, curr_c + dc
            
            # Bounds check
            if not (0 <= new_r < height and 0 <= new_c < width):
                continue
            
            target_pos = (new_r, new_c)
            target_tile = grid[new_r, new_c]
            
            # Determine if move is possible
            can_move, new_state = solver._try_move_pure(current_state, target_pos, target_tile)
            
            if not can_move:
                print(f"    {target_pos}: tile={target_tile} BLOCKED")
                continue
            
            new_hash = hash(new_state)
            
            if new_hash in closed_set:
                continue
            
            move_cost = solver._get_movement_cost(target_tile, target_pos, current_state)
            g_score = g_scores[state_hash] + move_cost
            
            if new_hash in g_scores and g_score >= g_scores[new_hash]:
                continue
            
            g_scores[new_hash] = g_score
            h_score = solver._heuristic(new_state)
            f_score_new = g_score + h_score
            
            valid_moves += 1
            print(f"    {target_pos}: tile={target_tile} VALID g={g_score:.1f} h={h_score:.1f} f={f_score_new:.1f}")
            
            new_path = path + [new_state.position]
            heapq.heappush(open_set, (f_score_new, counter, new_hash, new_state, new_path))
            counter += 1
        
        print(f"  Valid moves: {valid_moves}")
        print(f"  Open set size: {len(open_set)}")
    
    print(f"\nExplored {states_explored} states")
    print(f"Open set size at end: {len(open_set)}")
    
    # Original validation
    validator = ZeldaValidator()
    result = validator.validate_single(stitched.global_grid, render=False, persona_mode="balanced")
    
    print(f"\nResult:")
    print(f"  Solvable: {result.is_solvable}")
    print(f"  Valid Syntax: {result.is_valid_syntax}")
    print(f"  Error: {result.error_message}")
    print(f"  Logical Errors: {result.logical_errors}")
    
    return stitched

if __name__ == "__main__":
    # Debug the first failing dungeon
    stitched = debug_dungeon("zelda_1_quest1")
