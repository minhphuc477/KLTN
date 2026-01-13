"""
Test solver using VGLC-First Adapter.
"""
import numpy as np
from collections import deque
from vglc_first_adapter import VGLCFirstAdapter, SEMANTIC_PALETTE


def find_walkable_positions(grid: np.ndarray):
    """Find all walkable (floor-like) positions in a room."""
    walkable = set()
    walkable_ids = {
        SEMANTIC_PALETTE['FLOOR'],
        SEMANTIC_PALETTE['DOOR_OPEN'],
        SEMANTIC_PALETTE['START'],
        SEMANTIC_PALETTE['TRIFORCE'],
        SEMANTIC_PALETTE['STAIR'],
        SEMANTIC_PALETTE['KEY_SMALL'],
        SEMANTIC_PALETTE['ITEM_MINOR'],
    }
    
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if grid[r, c] in walkable_ids:
                walkable.add((r, c))
    
    return walkable


def stitch_dungeon(dungeon):
    """Stitch dungeon rooms into a single grid using VGLC positions."""
    # Find bounds
    positions = [r.position for r in dungeon.rooms.values()]
    min_row = min(p[0] for p in positions)
    max_row = max(p[0] for p in positions)
    min_col = min(p[1] for p in positions)
    max_col = max(p[1] for p in positions)
    
    # Normalize
    room_height = 16
    room_width = 11
    
    rows = max_row - min_row + 1
    cols = max_col - min_col + 1
    
    # Create full grid
    full_h = rows * room_height
    full_w = cols * room_width
    full_grid = np.zeros((full_h, full_w), dtype=np.int64)
    
    for room in dungeon.rooms.values():
        # Get normalized position
        r = room.position[0] - min_row
        c = room.position[1] - min_col
        
        start_row = r * room_height
        start_col = c * room_width
        
        # Copy room grid
        full_grid[start_row:start_row+room_height, start_col:start_col+room_width] = room.grid
    
    # CRITICAL: Connect doors between adjacent rooms
    # For Zelda rooms: doors are at row 1 (N), row 14 (S), col 1 (W), col 9 (E)
    # Wall rows are: 0, 15 (top/bottom) and cols 0, 10 (left/right)
    
    for room in dungeon.rooms.values():
        r, c = room.position
        norm_r = r - min_row
        norm_c = c - min_col
        
        room_start_row = norm_r * room_height
        room_start_col = norm_c * room_width
        
        # Check SOUTH neighbor - punch through row 15 of this room and row 0 of south room
        south_pos = (r + 1, c)
        south_id = f"{south_pos[0]}_{south_pos[1]}"
        if south_id in dungeon.rooms:
            # Find door columns in this room's row 14
            for dc in range(room_width):
                row_14 = room_start_row + 14
                col_pos = room_start_col + dc
                
                if full_grid[row_14, col_pos] == SEMANTIC_PALETTE['DOOR_OPEN']:
                    # Punch through row 15 (last row of this room) and row 0 (first of south)
                    full_grid[room_start_row + 15, col_pos] = SEMANTIC_PALETTE['FLOOR']
                    full_grid[room_start_row + 16, col_pos] = SEMANTIC_PALETTE['FLOOR']  # Row 0 of south
        
        # Check NORTH neighbor - punch through row 0 of this room and row 15 of north room
        north_pos = (r - 1, c)
        north_id = f"{north_pos[0]}_{north_pos[1]}"
        if north_id in dungeon.rooms:
            # Find door columns in this room's row 1
            for dc in range(room_width):
                row_1 = room_start_row + 1
                col_pos = room_start_col + dc
                
                if full_grid[row_1, col_pos] == SEMANTIC_PALETTE['DOOR_OPEN']:
                    # Punch through row 0 (first row of this room) - north room's row 15 will be handled
                    full_grid[room_start_row, col_pos] = SEMANTIC_PALETTE['FLOOR']
                    if room_start_row > 0:
                        full_grid[room_start_row - 1, col_pos] = SEMANTIC_PALETTE['FLOOR']  # Row 15 of north
        
        # Check EAST neighbor - punch through col 10 of this room and col 0 of east room
        east_pos = (r, c + 1)
        east_id = f"{east_pos[0]}_{east_pos[1]}"
        if east_id in dungeon.rooms:
            # Find door rows in this room's col 9
            for dr in range(room_height):
                col_9 = room_start_col + 9
                row_pos = room_start_row + dr
                
                if full_grid[row_pos, col_9] == SEMANTIC_PALETTE['DOOR_OPEN']:
                    # Punch through col 10 (last col of this room) and col 0 (first of east)
                    full_grid[row_pos, room_start_col + 10] = SEMANTIC_PALETTE['FLOOR']
                    full_grid[row_pos, room_start_col + 11] = SEMANTIC_PALETTE['FLOOR']  # Col 0 of east
        
        # Check WEST neighbor - punch through col 0 of this room and col 10 of west room
        west_pos = (r, c - 1)
        west_id = f"{west_pos[0]}_{west_pos[1]}"
        if west_id in dungeon.rooms:
            # Find door rows in this room's col 1
            for dr in range(room_height):
                col_1 = room_start_col + 1
                row_pos = room_start_row + dr
                
                if full_grid[row_pos, col_1] == SEMANTIC_PALETTE['DOOR_OPEN']:
                    # Punch through col 0 (first col of this room)
                    full_grid[row_pos, room_start_col] = SEMANTIC_PALETTE['FLOOR']
                    if room_start_col > 0:
                        full_grid[row_pos, room_start_col - 1] = SEMANTIC_PALETTE['FLOOR']  # Col 10 of west
    
    return full_grid, min_row, min_col


def find_path_bfs(grid: np.ndarray, start: tuple, goals: set):
    """BFS pathfinding from start to any goal."""
    walkable_ids = {
        SEMANTIC_PALETTE['FLOOR'],
        SEMANTIC_PALETTE['DOOR_OPEN'],
        SEMANTIC_PALETTE['START'],
        SEMANTIC_PALETTE['TRIFORCE'],
        SEMANTIC_PALETTE['STAIR'],
        SEMANTIC_PALETTE['KEY_SMALL'],
        SEMANTIC_PALETTE['ITEM_MINOR'],
    }
    
    h, w = grid.shape
    visited = set()
    queue = deque([(start, [start])])
    visited.add(start)
    
    while queue:
        pos, path = queue.popleft()
        
        if pos in goals:
            return path
        
        # 4-directional movement
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                if grid[nr, nc] in walkable_ids:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [(nr, nc)]))
    
    return None


def analyze_dungeon_solvability(dungeon):
    """Analyze if dungeon is solvable."""
    print(f"\n{'='*60}")
    print(f"ANALYZING DUNGEON: {dungeon.dungeon_id}")
    print(f"{'='*60}")
    
    # Stitch dungeon
    full_grid, min_row, min_col = stitch_dungeon(dungeon)
    print(f"Full dungeon size: {full_grid.shape}")
    
    # Find START room with multiple strategies
    # Strategy 1: STAIR room with doors
    start_room = None
    for rid, room in dungeon.rooms.items():
        if SEMANTIC_PALETTE['STAIR'] in room.grid:
            door_count = sum(room.doors.values()) if hasattr(room, 'doors') else 0
            if door_count > 0:
                start_room = room
                break
    
    # Strategy 2: Room with most doors (central hub)
    if not start_room:
        max_doors = 0
        for rid, room in dungeon.rooms.items():
            door_count = sum(room.doors.values()) if hasattr(room, 'doors') else 0
            if door_count > max_doors:
                max_doors = door_count
                start_room = room
    
    if not start_room:
        print("ERROR: No suitable START room found!")
        return False
    
    # Find a walkable floor position in the START room
    room_height, room_width = 16, 11
    norm_r = start_room.position[0] - min_row
    norm_c = start_room.position[1] - min_col
    room_start_row = norm_r * room_height
    room_start_col = norm_c * room_width
    
    # Find any floor tile in the room
    start_pos = None
    for local_r in range(room_height):
        for local_c in range(room_width):
            global_r = room_start_row + local_r
            global_c = room_start_col + local_c
            if full_grid[global_r, global_c] == SEMANTIC_PALETTE['FLOOR']:
                start_pos = (global_r, global_c)
                break
        if start_pos:
            break
    
    if not start_pos:
        print("ERROR: No floor tile in START room!")
        return False
    
    print(f"START position (floor in start room): {start_pos}")
    print(f"  Tile value at start: {full_grid[start_pos]}")
    print(f"  START room: {start_room.room_id} at {start_room.position}")
    
    # Find TRIFORCE positions
    triforce_locs = np.where(full_grid == SEMANTIC_PALETTE['TRIFORCE'])
    if len(triforce_locs[0]) == 0:
        print("ERROR: No TRIFORCE found!")
        return False
    
    triforce_positions = set(zip(triforce_locs[0], triforce_locs[1]))
    print(f"TRIFORCE positions: {triforce_positions}")
    
    # Count walkable tiles
    walkable = find_walkable_positions(full_grid)
    print(f"Total walkable tiles: {len(walkable)}")
    
    # BFS from START
    path = find_path_bfs(full_grid, start_pos, triforce_positions)
    
    if path:
        print(f"\n✓ PATH FOUND! Length: {len(path)} steps")
        
        # Show path summary
        room_height, room_width = 16, 11
        rooms_visited = set()
        for pos in path:
            room_r = pos[0] // room_height + min_row
            room_c = pos[1] // room_width + min_col
            rooms_visited.add((room_r, room_c))
        
        print(f"Rooms traversed: {len(rooms_visited)}")
        print(f"Room path: {sorted(rooms_visited)}")
        return True
    else:
        print("\n✗ NO PATH FOUND!")
        
        # Debug: Find reachable from START
        walkable_ids = {
            SEMANTIC_PALETTE['FLOOR'],
            SEMANTIC_PALETTE['DOOR_OPEN'],
            SEMANTIC_PALETTE['START'],
            SEMANTIC_PALETTE['TRIFORCE'],
            SEMANTIC_PALETTE['STAIR'],
        }
        
        visited = set()
        queue = deque([start_pos])
        visited.add(start_pos)
        
        while queue:
            pos = queue.popleft()
            h, w = full_grid.shape
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = pos[0] + dr, pos[1] + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                    if full_grid[nr, nc] in walkable_ids:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        
        print(f"Reachable from START: {len(visited)} tiles")
        
        # Show which rooms are reachable
        room_height, room_width = 16, 11
        reachable_rooms = set()
        for pos in visited:
            room_r = pos[0] // room_height + min_row
            room_c = pos[1] // room_width + min_col
            reachable_rooms.add((room_r, room_c))
        
        print(f"Reachable rooms: {len(reachable_rooms)}")
        print(f"Room coordinates: {sorted(reachable_rooms)}")
        
        return False


def main():
    adapter = VGLCFirstAdapter('Data')
    
    results = []
    
    import os
    
    processed_dir = 'Data/The Legend of Zelda/Processed'
    graph_dir = 'Data/The Legend of Zelda/Graph Processed'
    
    # Test file pairings:
    # tloz1_1.txt -> LoZ_1.dot (Quest 1, Dungeon 1)
    # tloz1_2.txt -> LoZ_2.dot (Quest 1, Dungeon 2) - but we only have 2 Quest 1 files
    # Actually the naming seems to be: tlozX_Y.txt where X is dungeon, Y is sub-level?
    
    # Let me just try all permutations and see what works
    test_pairs = [
        ('tloz1_1.txt', 'LoZ_1.dot', 'Dungeon 1'),
        ('tloz1_2.txt', 'LoZ_2.dot', 'Dungeon 2'),
        ('tloz2_1.txt', 'LoZ_3.dot', 'Dungeon 3'),  # Maybe tloz2 = dungeon 2 level 1?
        ('tloz2_2.txt', 'LoZ_4.dot', 'Dungeon 4'),
        ('tloz3_1.txt', 'LoZ_5.dot', 'Dungeon 5'),
        ('tloz3_2.txt', 'LoZ_6.dot', 'Dungeon 6'),
        ('tloz4_1.txt', 'LoZ_7.dot', 'Dungeon 7'),
        ('tloz4_2.txt', 'LoZ_8.dot', 'Dungeon 8'),
        ('tloz5_1.txt', 'LoZ_9.dot', 'Dungeon 9'),
    ]
    
    # Actually let me just test the known working pair and try to understand the pattern
    # by examining file sizes
    print("File sizes:")
    for f in sorted(os.listdir(processed_dir)):
        if f.endswith('.txt') and f != 'README.txt':
            fpath = os.path.join(processed_dir, f)
            print(f"  {f}: {os.path.getsize(fpath)} bytes")
    
    print("\nTesting known pair: tloz1_1.txt + LoZ_1.dot")
    map_file = os.path.join(processed_dir, 'tloz1_1.txt')
    graph_file = os.path.join(graph_dir, 'LoZ_1.dot')
    
    if os.path.exists(map_file) and os.path.exists(graph_file):
        dungeon = adapter.process_dungeon(map_file, graph_file, 'test1')
        success = analyze_dungeon_solvability(dungeon)
        results.append(('D1', success))
    
    print("\nTesting: tloz1_2.txt + LoZ_2.dot")
    map_file = os.path.join(processed_dir, 'tloz1_2.txt')
    graph_file = os.path.join(graph_dir, 'LoZ_2.dot')
    
    if os.path.exists(map_file) and os.path.exists(graph_file):
        dungeon = adapter.process_dungeon(map_file, graph_file, 'test2')
        success = analyze_dungeon_solvability(dungeon)
        results.append(('D2', success))
    
    # Try matching by dungeon number in filename
    print("\n\nTesting dungeons 3-9 with tlozX_1.txt matching LoZ_X.dot:")
    for dnum in range(3, 10):
        map_file = os.path.join(processed_dir, f'tloz{dnum}_1.txt')
        graph_file = os.path.join(graph_dir, f'LoZ_{dnum}.dot')
        
        if not os.path.exists(map_file):
            print(f"\nSkipping dungeon {dnum}: {map_file} not found")
            continue
        if not os.path.exists(graph_file):
            print(f"\nSkipping dungeon {dnum}: {graph_file} not found")
            continue
        
        try:
            dungeon = adapter.process_dungeon(map_file, graph_file, f'D{dnum}')
            success = analyze_dungeon_solvability(dungeon)
            results.append((f'D{dnum}', success))
        except Exception as e:
            print(f"\nError processing dungeon {dnum}: {e}")
            import traceback
            traceback.print_exc()
            results.append((f'D{dnum}', False))
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    solvable = sum(1 for _, s in results if s)
    total = len(results)
    
    for dname, success in results:
        status = "✓ SOLVABLE" if success else "✗ NOT SOLVABLE"
        print(f"{dname}: {status}")
    
    print(f"\nTotal: {solvable}/{total} solvable ({100*solvable/total:.1f}%)" if total > 0 else "\nNo results")
    
    return solvable == total if total > 0 else False


if __name__ == '__main__':
    main()
