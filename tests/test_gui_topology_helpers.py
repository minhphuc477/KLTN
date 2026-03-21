from src.gui.services.topology_helpers import (
    build_room_adjacency_from_graph,
    capture_precheck_snapshot,
    local_bfs_4dir,
    min_locked_between,
    node_has_critical_content,
    node_has_small_key,
    room_entry_point,
    room_for_global_position,
    topology_has_path,
    update_env_topology_view,
    walkable_grid_reachable,
)


class _FakeGraph:
    def __init__(self):
        self.nodes = {
            1: {"label": "s", "is_start": True},
            2: {"label": "k"},
            3: {"label": ""},
        }
        self._edges = [(1, 2), (2, 3)]

    def __contains__(self, node_id):
        return node_id in self.nodes

    def edges(self):
        return list(self._edges)

    def to_undirected(self):
        return self

    def get_edge_data(self, u, v):
        if (u, v) in self._edges or (v, u) in self._edges:
            if (u, v) == (1, 2) or (v, u) == (1, 2):
                return {"edge_type": "key_locked"}
            return {"edge_type": "open"}
        return {}

    def successors(self, u):
        return [b for a, b in self._edges if a == u]

    def predecessors(self, u):
        return [a for a, b in self._edges if b == u]


def test_room_for_global_position_maps_correct_room():
    room_positions = {(0, 0): (0, 0), (0, 1): (0, 11)}
    assert room_for_global_position((5, 5), room_positions) == (0, 0)
    assert room_for_global_position((5, 15), room_positions) == (0, 1)
    assert room_for_global_position((99, 99), room_positions) is None


def test_node_has_small_key_detection():
    assert node_has_small_key({"is_key": True}) is True
    assert node_has_small_key({"label": "k"}) is True
    assert node_has_small_key({"label": "key"}) is True
    assert node_has_small_key({"is_boss_key": True, "label": "k"}) is False


def test_node_has_critical_content_and_adjacency_building():
    graph = _FakeGraph()
    assert node_has_critical_content(graph, 1) is True
    assert node_has_critical_content(graph, 2) is True
    assert node_has_critical_content(graph, 3) is False

    room_to_node = {(0, 0): 1, (0, 1): 2, (0, 2): 3}
    node_to_room = {1: (0, 0), 2: (0, 1), 3: (0, 2)}
    adjacency = build_room_adjacency_from_graph(graph, room_to_node, node_to_room)

    assert (0, 1) in adjacency[(0, 0)]
    assert (0, 0) in adjacency[(0, 1)]
    assert (0, 2) in adjacency[(0, 1)]


def test_topology_path_and_min_locked_between():
    graph = _FakeGraph()
    assert topology_has_path(graph, 1, 3) is True
    assert min_locked_between(graph, 1, 3) == 1


def test_walkable_grid_reachable_fallback():
    grid = [
        [1, 1, 1],
        [1, 2, 1],
        [1, 1, 1],
    ]
    action_deltas = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
    assert walkable_grid_reachable(grid, (0, 0), (2, 2), {2}, action_deltas) is True


def test_capture_snapshot_and_update_env_topology_view():
    current = type("Current", (), {})()
    current.graph = _FakeGraph()
    current.room_to_node = {(0, 0): 1}
    current.node_to_room = {1: (0, 0)}
    current.rooms = {(0, 0): {"dummy": 1}}
    current.room_positions = {(0, 0): (0, 0)}

    snap = capture_precheck_snapshot(current, reason="test")
    assert snap["reason"] == "test"
    assert snap["room_to_node"] == {(0, 0): 1}

    env = type("Env", (), {})()
    update_env_topology_view(env, current)
    assert env.graph is current.graph
    assert env.room_to_node == current.room_to_node
    assert env.node_to_room == current.node_to_room


def test_room_entry_point_prefers_stair_then_center_then_walkable():
    grid = [
        [1, 1, 1, 1],
        [1, 42, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ]
    room_positions = {(0, 0): (0, 0)}
    walkable = {1, 42}
    assert room_entry_point(grid, room_positions, (0, 0), walkable, stair_tile=42, room_h=4, room_w=4) == (1, 1)


def test_local_bfs_4dir_returns_path_when_reachable():
    grid = [
        [1, 1, 1],
        [1, 2, 1],
        [1, 1, 1],
    ]
    walkable = {1}
    path = local_bfs_4dir(grid, walkable, (0, 0), (2, 2), max_steps=100)
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)

