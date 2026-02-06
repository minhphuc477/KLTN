"""
Worker utilities for process-based parallel search.
Can be used by main GUI process to run grid algorithms in separate processes.
"""
import time
import numpy as np
from typing import Tuple, List
from simulation.validator import SEMANTIC_PALETTE
from heapq import heappush, heappop

FLOOR = SEMANTIC_PALETTE['FLOOR']
DOOR_OPEN = SEMANTIC_PALETTE['DOOR_OPEN']
DOOR_LOCKED = SEMANTIC_PALETTE['DOOR_LOCKED']
KEY = SEMANTIC_PALETTE.get('KEY_SMALL', 30)
START = SEMANTIC_PALETTE.get('START', 21)
TRIFORCE = SEMANTIC_PALETTE.get('TRIFORCE', 22)
STAIR = SEMANTIC_PALETTE.get('STAIR', 42)

WALKABLE = {FLOOR, DOOR_OPEN, KEY, START, TRIFORCE, STAIR, DOOR_LOCKED}


def manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


def run_grid_algorithm(grid_list: List[List[int]], start: Tuple[int,int], goal: Tuple[int,int], alg: int, allow_teleports: bool = True, topology: dict = None):
    """Run a grid algorithm on the provided grid. Supports stair teleports and optional room-guided behavior.
    alg: 0=A*, 1=BFS, 2=Dijkstra, 3=Greedy
    Returns: dict with success, path(list), nodes, time_ms
    """
    # Detect stair tiles for teleport support
    stairs = []
    for r, row in enumerate(grid_list):
        for c, val in enumerate(row):
            if val == STAIR:
                stairs.append((r, c))

    # If topology provided, use it to restrict teleport destinations
    room_to_stairs = {}
    adj = {}
    if topology:
        room_to_stairs = topology.get('room_to_stairs', {})
        adj = topology.get('adj', {})
        # Build stairs list from topology if provided (override automatic)
        all_stairs = []
        for sts in room_to_stairs.values():
            all_stairs.extend(sts)
        if all_stairs:
            stairs = list(all_stairs)

    # Helper: get room id for a stair position
    stair_to_room = {}
    for room, sts in room_to_stairs.items():
        for s in sts:
            stair_to_room[tuple(s)] = room
    grid = np.array(grid_list, dtype=int)
    H, W = grid.shape
    t0 = time.time()
    if alg == 1:
        # BFS with optional teleportation via stairs
        from collections import deque
        q = deque([ (start, [start]) ])
        visited = {start}
        nodes = 0
        while q:
            pos, path = q.popleft()
            nodes += 1
            if pos == goal:
                return {'success': True, 'path': path, 'nodes': nodes, 'time_ms': (time.time()-t0)*1000}
            # neighbors
            for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = pos[0]+dy, pos[1]+dx
                if 0<=ny<H and 0<=nx<W and grid[ny,nx] in WALKABLE and (ny,nx) not in visited:
                    visited.add((ny,nx))
                    q.append(((ny,nx), path + [(ny,nx)]))
            # Teleportation step: from stair to allowed stairs
            if allow_teleports and pos in stairs:
                # If topology available, restrict teleports to neighbor rooms
                if topology and tuple(pos) in stair_to_room:
                    room = stair_to_room[tuple(pos)]
                    neighbors = adj.get(room, [])
                    dests = []
                    for nroom in neighbors:
                        dests.extend(room_to_stairs.get(nroom, []))
                else:
                    dests = [d for d in stairs if d != pos]
                for dest in dests:
                    dtuple = tuple(dest)
                    if dtuple not in visited:
                        visited.add(dtuple)
                        q.append((dtuple, path + [dtuple]))
        return {'success': False, 'path': [], 'nodes': nodes, 'time_ms': (time.time()-t0)*1000}
    else:
        # Priority-based
        counter = 0
        heap = []
        start_h = manhattan(start, goal)
        if alg == 0 or alg == 3:
            start_f = start_h
        elif alg == 2:
            start_f = 0
        else:
            start_f = start_h
        heappush(heap, (start_f, 0, counter, start, [start]))
        counter += 1
        best = {start: 0}
        nodes = 0
        max_iter = 200000
        iters = 0
        while heap and iters < max_iter:
            iters += 1
            f, g, _c, pos, path = heappop(heap)
            nodes += 1
            if pos == goal:
                return {'success': True, 'path': path, 'nodes': nodes, 'time_ms': (time.time()-t0)*1000}
            y, x = pos
            # 4-directional neighbors
            for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = y+dy, x+dx
                if 0<=ny<H and 0<=nx<W and grid[ny,nx] in WALKABLE:
                    npos = (ny, nx)
                    new_g = g + 1
                    h = manhattan(npos, goal)
                    if alg == 2:
                        new_f = new_g
                    elif alg == 3:
                        new_f = h
                    else:
                        new_f = new_g + h
                    if npos in best and new_g >= best[npos]:
                        continue
                    best[npos] = new_g
                    heappush(heap, (new_f, new_g, counter, npos, path + [npos]))
                    counter += 1
            # Teleportation: if on a stair, consider teleporting to allowed stairs
            if allow_teleports and pos in stairs:
                if topology and tuple(pos) in stair_to_room:
                    room = stair_to_room[tuple(pos)]
                    neighbors = adj.get(room, [])
                    dests = []
                    for nroom in neighbors:
                        dests.extend(room_to_stairs.get(nroom, []))
                else:
                    dests = [d for d in stairs if d != pos]
                for dest in dests:
                    npos = tuple(dest)
                    if npos == pos:
                        continue
                    new_g = g + 1  # teleport cost of 1
                    h = manhattan(npos, goal)
                    if alg == 2:
                        new_f = new_g
                    elif alg == 3:
                        new_f = h
                    else:
                        new_f = new_g + h
                    if npos in best and new_g >= best[npos]:
                        continue
                    best[npos] = new_g
                    heappush(heap, (new_f, new_g, counter, npos, path + [npos]))
                    counter += 1
        return {'success': False, 'path': [], 'nodes': nodes, 'time_ms': (time.time()-t0)*1000}


# CLI support for subprocess invocation
if __name__ == '__main__':
    import argparse, json, sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid-file', help='JSON file with grid as list of lists')
    parser.add_argument('--alg', type=int, default=0)
    parser.add_argument('--start', help='r,c')
    parser.add_argument('--goal', help='r,c')
    parser.add_argument('--out', help='output JSON file', required=False)
    args = parser.parse_args()
    grid = json.load(open(args.grid_file, 'r'))
    start = tuple(map(int, args.start.split(',')))
    goal = tuple(map(int, args.goal.split(',')))
    res = run_grid_algorithm(grid, start, goal, args.alg)
    out = args.out or 'parallel_result.json'
    json.dump(res, open(out, 'w'))
    print(out)
