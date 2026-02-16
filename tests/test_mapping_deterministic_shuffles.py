import random
import networkx as nx
from Data.zelda_core import RoomGraphMatcher, Room
import numpy as np


def make_room(pos):
    return Room(position=pos, char_grid=np.zeros((2, 2)), semantic_grid=np.zeros((2, 2)), doors={'N':False,'S':False,'E':False,'W':False}, has_stair=False)


def test_mapping_deterministic_longrun():
    # build a 3x3 grid of rooms and graph mirroring it
    rooms = {}
    for r in range(3):
        for c in range(3):
            doors = {'N': False, 'S': False, 'E': False, 'W': False}
            rooms[(r, c)] = make_room((r, c),)
    # connect doors
    for r in range(3):
        for c in range(3):
            if r < 2:
                rooms[(r, c)].doors['S'] = True
                rooms[(r + 1, c)].doors['N'] = True
            if c < 2:
                rooms[(r, c)].doors['E'] = True
                rooms[(r, c + 1)].doors['W'] = True

    base_nodes = list(range(100, 109))
    G = nx.DiGraph()
    G.add_nodes_from(base_nodes)
    pos_to_node = {}
    idx = 0
    for r in range(3):
        for c in range(3):
            pos_to_node[(r, c)] = base_nodes[idx]
            idx += 1
    for (r,c), nid in pos_to_node.items():
        if r < 2:
            G.add_edge(nid, pos_to_node[(r+1,c)])
            G.add_edge(pos_to_node[(r+1,c)], nid)
        if c < 2:
            G.add_edge(nid, pos_to_node[(r,c+1)])
            G.add_edge(pos_to_node[(r,c+1)], nid)

    G.nodes[pos_to_node[(0,0)]]['is_start'] = True

    matcher = RoomGraphMatcher()

    canonical = None
    for seed in range(100):
        rng = random.Random(seed)
        perm = base_nodes.copy()
        rng.shuffle(perm)
        mapping = {old: new for old, new in zip(base_nodes, perm)}
        H = nx.relabel_nodes(G, mapping)
        d = matcher.match(rooms.copy(), H)
        room_map = {pos: r.graph_node_id for pos, r in d.rooms.items()}
        # Normalize mapping into structural ranks
        nodes_sorted = sorted(list(H.nodes()), key=lambda x: matcher._node_signature(H, x))
        rank = {n: i for i, n in enumerate(nodes_sorted)}
        normalized = {pos: rank[n] for pos, n in room_map.items()}
        if canonical is None:
            canonical = normalized
        else:
            assert normalized == canonical, f'Mapping varies on seed {seed}'
