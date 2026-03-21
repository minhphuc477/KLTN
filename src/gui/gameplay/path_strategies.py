"""Pathfinding strategy helpers extracted from GUI runner."""

import math
from collections import deque


def graph_guided_path(gui):
    """Fallback strategy: follow graph path with teleportation when needed."""
    import networkx as nx

    current_dungeon = gui.maps[gui.current_map_idx]
    if hasattr(current_dungeon, "global_grid"):
        grid = current_dungeon.global_grid
        room_positions = getattr(current_dungeon, "room_positions", {})
        room_to_node = getattr(current_dungeon, "room_to_node", {})
        graph = getattr(current_dungeon, "graph", None)
    else:
        grid = current_dungeon
        room_positions = {}
        room_to_node = {}
        graph = None
    h, w = grid.shape

    room_height = 16
    room_width = 11
    floor = 1
    door_open = 10
    door_locked = 11
    start_tile = 21
    triforce = 22
    key = 30
    stair = 42
    walkable = {floor, door_open, key, start_tile, triforce, stair, door_locked}

    start = gui.env.start_pos
    goal = gui.env.goal_pos
    if start is None or goal is None:
        return False, [], 0

    def get_room(pos):
        y, x = pos
        for room_pos, (ry, rx) in room_positions.items():
            if ry <= y < ry + room_height and rx <= x < rx + room_width:
                return room_pos
        return None

    def find_entry(room_pos):
        if room_pos not in room_positions:
            return None
        ry, rx = room_positions[room_pos]
        for dy in range(room_height):
            for dx in range(room_width):
                y, x = ry + dy, rx + dx
                if 0 <= y < h and 0 <= x < w and grid[y, x] == stair:
                    return (y, x)
        cy, cx = ry + room_height // 2, rx + room_width // 2
        if 0 <= cy < h and 0 <= cx < w and grid[cy, cx] in walkable:
            return (cy, cx)
        for dy in range(room_height):
            for dx in range(room_width):
                y, x = ry + dy, rx + dx
                if 0 <= y < h and 0 <= x < w and grid[y, x] in walkable:
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
                if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in visited:
                    if grid[ny, nx] in walkable:
                        if (ny, nx) == to_pos:
                            return path + [(ny, nx)]
                        visited.add((ny, nx))
                        queue.append(((ny, nx), path + [(ny, nx)]))
        return None

    if not room_positions or graph is None:
        direct = local_bfs(start, goal, max_steps=50000)
        if direct:
            return True, direct, 0
        return False, [], 0

    node_to_room = {v: k for k, v in room_to_node.items()}
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

    room_sequence = []
    for node in graph_path:
        room = node_to_room.get(node)
        if room and room in room_positions:
            room_sequence.append(room)

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


def hybrid_graph_grid_path(gui):
    """Hybrid strategy: graph room sequence and BFS segments between room entry points."""
    import networkx as nx

    current_dungeon = gui.maps[gui.current_map_idx]
    if hasattr(current_dungeon, "global_grid"):
        grid = current_dungeon.global_grid
        room_positions = getattr(current_dungeon, "room_positions", {})
        room_to_node = getattr(current_dungeon, "room_to_node", {})
        graph = getattr(current_dungeon, "graph", None)
    else:
        grid = current_dungeon
        room_positions = {}
        room_to_node = {}
        graph = None
    h, w = grid.shape

    room_height = 16
    room_width = 11
    floor = 1
    door_open = 10
    door_locked = 11
    start_tile = 21
    triforce = 22
    key = 30
    stair = 42
    walkable = {floor, door_open, key, start_tile, triforce, stair, door_locked}

    start = gui.env.start_pos
    goal = gui.env.goal_pos

    if start is None or goal is None:
        return False, [], 0

    def get_room(pos):
        y, x = pos
        for room_pos, (ry, rx) in room_positions.items():
            if ry <= y < ry + room_height and rx <= x < rx + room_width:
                return room_pos
        return None

    def find_entry_point(room_pos):
        if room_pos not in room_positions:
            return None
        ry, rx = room_positions[room_pos]

        for dy in range(room_height):
            for dx in range(room_width):
                y, x = ry + dy, rx + dx
                if 0 <= y < h and 0 <= x < w and grid[y, x] == stair:
                    return (y, x)

        cy, cx = ry + room_height // 2, rx + room_width // 2
        if 0 <= cy < h and 0 <= cx < w and grid[cy, cx] in walkable:
            return (cy, cx)

        for dy in range(room_height):
            for dx in range(room_width):
                y, x = ry + dy, rx + dx
                if 0 <= y < h and 0 <= x < w and grid[y, x] in walkable:
                    return (y, x)
        return None

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
                if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in visited:
                    tile = grid[ny, nx]
                    if tile in walkable:
                        if (ny, nx) == to_pos:
                            return path + [(ny, nx)]
                        visited.add((ny, nx))
                        queue.append(((ny, nx), path + [(ny, nx)]))
        return None

    if not room_positions or graph is None:
        direct = local_bfs(start, goal, max_steps=50000)
        return (True, direct, 0) if direct else (False, [], 0)

    start_room = get_room(start)
    goal_room = get_room(goal)

    if not start_room or not goal_room:
        return False, [], 0

    node_to_room = {v: k for k, v in room_to_node.items()}

    start_node = room_to_node.get(start_room)
    goal_node = room_to_node.get(goal_room)

    if start_node is None or goal_node is None:
        path = local_bfs(start, goal, max_steps=50000)
        return (True, path, 0) if path else (False, [], 0)

    try:
        graph_path = nx.shortest_path(graph, start_node, goal_node)
    except nx.NetworkXNoPath:
        return False, [], 0

    room_sequence = []
    for node in graph_path:
        room = node_to_room.get(node)
        if room and room in room_positions:
            room_sequence.append(room)

    if not room_sequence:
        return False, [], 0

    full_path = [start]
    current_pos = start
    teleports = 0

    for target_room in room_sequence:
        target_pos = find_entry_point(target_room)
        if not target_pos:
            continue

        if current_pos == target_pos:
            continue

        path_segment = local_bfs(current_pos, target_pos)

        if path_segment:
            full_path.extend(path_segment[1:])
            current_pos = path_segment[-1]
        else:
            full_path.append(target_pos)
            current_pos = target_pos
            teleports += 1

    if current_pos != goal:
        final_segment = local_bfs(current_pos, goal)
        if final_segment:
            full_path.extend(final_segment[1:])
        else:
            full_path.append(goal)
            teleports += 1

    if full_path[-1] != goal:
        full_path.append(goal)

    return True, full_path, teleports


def smart_grid_path(
    gui,
    logger,
    convert_diagonal_to_4dir,
    semantic_palette,
    np_module,
    path_cls,
    os_module,
):
    """Smart grid search with optional stair teleportation and algorithm-specific behavior."""
    current_dungeon = gui.maps[gui.current_map_idx]
    if hasattr(current_dungeon, "global_grid"):
        grid = current_dungeon.global_grid
        room_positions = getattr(current_dungeon, "room_positions", {})
        room_to_node = getattr(current_dungeon, "room_to_node", {})
        graph = getattr(current_dungeon, "graph", None)
    else:
        grid = current_dungeon
        room_positions = {}
        room_to_node = {}
        graph = None
    h, w = grid.shape

    room_height = 16
    room_width = 11
    floor = 1
    door_open = 10
    door_locked = 11
    start_tile = 21
    triforce = 22
    key = 30
    stair = 42

    walkable = {floor, door_open, key, start_tile, triforce, stair, door_locked}

    start = gui.env.start_pos
    goal = gui.env.goal_pos

    if start is None or goal is None:
        return False, [], 0

    if getattr(gui, "show_heatmap", False):
        gui.search_heatmap = {}

    def get_room(pos):
        y, x = pos
        for room_pos, (ry, rx) in room_positions.items():
            if ry <= y < ry + room_height and rx <= x < rx + room_width:
                return room_pos
        return None

    def get_stairs_in_room(room_pos):
        if room_pos not in room_positions:
            return []
        ry, rx = room_positions[room_pos]
        stairs = []
        for dy in range(room_height):
            for dx in range(room_width):
                y, x = ry + dy, rx + dx
                if 0 <= y < h and 0 <= x < w and grid[y, x] == stair:
                    stairs.append((y, x))
        return stairs

    def find_entry(room_pos):
        if room_pos not in room_positions:
            return None
        ry, rx = room_positions[room_pos]
        for dy in range(room_height):
            for dx in range(room_width):
                y, x = ry + dy, rx + dx
                if 0 <= y < h and 0 <= x < w and grid[y, x] == stair:
                    return (y, x)
        cy, cx = ry + room_height // 2, rx + room_width // 2
        if 0 <= cy < h and 0 <= cx < w and grid[cy, cx] in walkable:
            return (cy, cx)
        for dy in range(room_height):
            for dx in range(room_width):
                y, x = ry + dy, rx + dx
                if 0 <= y < h and 0 <= x < w and grid[y, x] in walkable:
                    return (y, x)
        return None

    node_to_room = {v: k for k, v in room_to_node.items()}

    def get_stair_destinations(pos):
        if grid[pos[0], pos[1]] != stair:
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
        for neighbor_node in graph.neighbors(current_node):
            neighbor_room = node_to_room.get(neighbor_node)
            if neighbor_room and neighbor_room in room_positions:
                neighbor_stairs = get_stairs_in_room(neighbor_room)
                if neighbor_stairs:
                    destinations.extend(neighbor_stairs)
                else:
                    entry = find_entry(neighbor_room)
                    if entry:
                        destinations.append(entry)

        return destinations

    alg = getattr(gui, "algorithm_idx", 0)

    full_solver_algorithms = {5, 6, 7, 8, 9, 10, 11, 12}
    if alg in full_solver_algorithms:
        alg_name_fn = getattr(gui, "_algorithm_name", lambda idx: f"Algorithm {idx}")
        alg_name = alg_name_fn(alg)
        logger.info(f"{alg_name} selected: skipping quick grid path, using full solver")
        return False, [], 0

    if alg == 4:
        logger.info("D* Lite selected - using incremental replanning")
        try:
            from src.simulation.dstar_lite import DStarLiteSolver

            start_state = gui.env.state.copy()
            solver = DStarLiteSolver(gui.env, heuristic_mode="balanced")
            success, path, nodes_explored = solver.solve(start_state)

            if success and path:
                display_path = convert_diagonal_to_4dir(path, grid=grid)
                algo_label = "D* Lite (fallback: A*)" if getattr(solver, "used_fallback", False) else "D* Lite"
                logger.info(f"{algo_label} succeeded: path_len={len(display_path)}, nodes={nodes_explored}")
                return True, display_path, 0
            logger.warning("D* Lite failed, falling back to A*")
        except Exception as e:
            logger.error(f"D* Lite error: {e}, falling back to A*")

    def heuristic(a, b):
        if gui.feature_flags.get("ml_heuristic", False) and getattr(gui, "_ml_heuristic", None):
            try:
                return gui._ml_heuristic(a, b)
            except Exception as e:
                logger.warning(f"ML heuristic failed, falling back to Manhattan/Octile: {e}")
        if gui.feature_flags.get("diagonal_movement", False):
            return math.hypot(a[0] - b[0], a[1] - b[1])
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    if not getattr(gui, "_ml_heuristic_ready", False):
        gui._ml_heuristic_ready = True
        gui.ml_model = None

        model_paths = []
        base_root = path_cls(getattr(gui, "repo_root", path_cls.cwd()))
        env_path = os_module.environ.get("KLTN_ML_HEURISTIC_MODEL", "").strip()
        if env_path:
            model_paths.append(path_cls(env_path))
        model_paths.extend([
            base_root / "checkpoints" / "heuristic_net.pth",
            base_root / "models" / "heuristic_net.pth",
        ])

        try:
            from src.ml.heuristic_learning import HeuristicTrainer

            for model_path in model_paths:
                if not model_path or not model_path.exists():
                    continue
                try:
                    gui.ml_model = HeuristicTrainer.load_model(str(model_path))
                    logger.info(f"Loaded ML heuristic model: {model_path}")
                    break
                except Exception as model_err:
                    logger.warning(f"Failed loading ML heuristic model {model_path}: {model_err}")
        except Exception as import_err:
            logger.debug(f"ML heuristic module unavailable: {import_err}")

        if gui.ml_model is None and gui.feature_flags.get("ml_heuristic", False):
            logger.info("ML heuristic enabled but no trained model found; using Manhattan fallback.")

        def _ml_heuristic_runtime(a, b):
            base = float(abs(a[0] - b[0]) + abs(a[1] - b[1]))
            model = getattr(gui, "ml_model", None)
            if model is None:
                return base
            try:
                state = getattr(gui, "game_state", None)
                if state is None and getattr(gui, "env", None) is not None:
                    state = getattr(gui.env, "state", None)

                keys = float(getattr(state, "keys", 0)) if state is not None else 0.0
                bomb_count = float(getattr(state, "bomb_count", 0)) if state is not None else 0.0
                has_bomb = 1.0 if bomb_count > 0 else 0.0
                has_boss_key = float(getattr(state, "has_boss_key", False)) if state is not None else 0.0
                has_item = float(getattr(state, "has_item", False)) if state is not None else 0.0

                total_cells = max(1, h * w)
                locked_ratio = float(np_module.sum(grid == semantic_palette["DOOR_LOCKED"])) / total_cells
                exploration_progress = 1.0 - (base / max(1.0, float(h + w)))

                features = np_module.array(
                    [
                        float(a[1]) / max(1.0, float(w - 1)),
                        float(a[0]) / max(1.0, float(h - 1)),
                        min(keys / 5.0, 1.0),
                        has_bomb,
                        has_boss_key,
                        has_item,
                        base / max(1.0, float(h + w)),
                        locked_ratio,
                        min(keys / 10.0, 1.0),
                        max(0.0, min(1.0, exploration_progress)),
                    ],
                    dtype=np_module.float32,
                )

                pred = float(model.predict_cost(features))
                if not np_module.isfinite(pred):
                    return base

                return max(0.0, min(pred, base))
            except Exception as runtime_err:
                logger.debug(f"ML heuristic runtime fallback: {runtime_err}")
                return base

        gui._ml_heuristic = _ml_heuristic_runtime

    max_iterations = 200000
    counter = 0
    iterations = 0
    gui.last_search_iterations = 0

    if alg == 1:
        visited = {start}
        queue = deque([(start, [start], 0)])
        iterations = 0
        while queue and iterations < max_iterations:
            iterations += 1
            pos, path, teleports = queue.popleft()
            y, x = pos
            if getattr(gui, "show_heatmap", False):
                gui.search_heatmap[pos] = gui.search_heatmap.get(pos, 0) + 1
            if pos == goal:
                gui.last_search_iterations = iterations
                return True, convert_diagonal_to_4dir(path, grid=grid), teleports
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] in walkable:
                    npos = (ny, nx)
                    if npos not in visited:
                        visited.add(npos)
                        queue.append((npos, path + [npos], teleports))
            if grid[y, x] == stair:
                for dest in get_stair_destinations(pos):
                    if dest not in visited:
                        visited.add(dest)
                        queue.append((dest, path + [dest], teleports + 1))
        gui.last_search_iterations = iterations
        return graph_guided_path(gui)

    import heapq

    def euclidean(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

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
    best = {}
    iterations = 0
    allow_diag = gui.feature_flags.get("diagonal_movement", False)
    use_jps = gui.feature_flags.get("use_jps", False)
    while heap and iterations < max_iterations:
        iterations += 1
        _f, g, _cnt, pos, stairs, teleports, path = heapq.heappop(heap)
        if getattr(gui, "show_heatmap", False):
            gui.search_heatmap[pos] = gui.search_heatmap.get(pos, 0) + 1
        if pos == goal:
            gui.last_search_iterations = iterations
            return True, path, teleports
        key_state = (pos, stairs)
        if key_state in best and g > best[key_state]:
            continue
        best[key_state] = g

        y, x = pos
        if allow_diag:
            deltas = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        else:
            deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dy, dx in deltas:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] in walkable:
                npos = (ny, nx)
                step_cost = math.hypot(dy, dx)
                new_g = g + step_cost
                new_stairs = stairs
                new_teleports = teleports
                h_score = heuristic(npos, goal)
                if alg == 2:
                    new_f = new_g
                elif alg == 3:
                    new_f = h_score
                else:
                    new_f = new_g + h_score
                nkey = (npos, new_stairs)
                if nkey in best and new_g >= best[nkey]:
                    continue
                heapq.heappush(heap, (new_f, new_g, counter, npos, new_stairs, new_teleports, path + [npos]))
                counter += 1

        if grid[y, x] == stair:
            for dest in get_stair_destinations(pos):
                npos = dest
                if npos in stairs:
                    continue
                new_stairs = set(stairs)
                new_stairs.add(npos)
                new_stairs = frozenset(new_stairs)
                new_g = g + 1
                new_teleports = teleports + 1
                h_score = heuristic(npos, goal)
                if alg == 2:
                    new_f = new_g
                elif alg == 3:
                    new_f = h_score
                else:
                    new_f = new_g + h_score
                nkey = (npos, new_stairs)
                if nkey in best and new_g >= best[nkey]:
                    continue
                heapq.heappush(heap, (new_f, new_g, counter, npos, new_stairs, new_teleports, path + [npos]))
                counter += 1

        if use_jps:
            try:
                from bench.grid_solvers import jps as _jps

                if allow_diag:
                    res = _jps(grid.tolist() if hasattr(grid, "tolist") else grid, pos, goal, allow_diagonal=True, trace=True)
                else:
                    res = _jps(grid.tolist() if hasattr(grid, "tolist") else grid, pos, goal, allow_diagonal=False, trace=True)
                if res:
                    if len(res) == 3:
                        jp_path, _jp_nodes, jp_trace = res
                    else:
                        jp_path, _jp_nodes = res
                        jp_trace = None
                    gui.last_jps_trace = jp_trace
                    if jp_path and len(jp_path) > 1:
                        target = jp_path[-1]
                        new_g = g + euclidean(pos, target)
                        h_score = heuristic(target, goal)
                        heapq.heappush(heap, (new_g + h_score, new_g, counter, target, stairs, teleports, path + [target]))
                        counter += 1
            except Exception:
                gui.last_jps_trace = None

    gui.last_search_iterations = iterations
    return graph_guided_path(gui)