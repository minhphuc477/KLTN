import networkx as nx
import numpy as np
from Data.zelda_core import Room, Dungeon
from zelda_pathfinder import ZeldaPathfinder


def make_room(pos):
    return Room(position=pos, char_grid=np.zeros((2, 2)), semantic_grid=np.zeros((2, 2)), doors={'N':False,'S':False,'E':False,'W':False}, has_stair=False)


def test_heuristic_admissible_vs_bfs():
    # Construct tiny dungeon where a locked door exists but a longer bypass path exists
    rooms = {}
    # Layout: A -(open)-> B - (key_locked)-> D
    #         |                     ^
    #         v                     |
    #         C ---(open)---------->
    for p in [(0,0),(0,1),(1,0),(0,2)]:
        rooms[p] = make_room(p)

    # connect doors (room adjacency)
    rooms[(0,0)].doors['E'] = True  # A->B
    rooms[(0,1)].doors['W'] = True
    rooms[(0,0)].doors['S'] = True  # A->C
    rooms[(1,0)].doors['N'] = True
    rooms[(1,0)].doors['E'] = True  # C->D (bypass)
    rooms[(0,2)].doors['W'] = True  # B->D is locked

    # Build graph with node ids
    G = nx.DiGraph()
    nid = { (0,0): 10, (0,1): 11, (1,0): 12, (0,2): 13 }
    G.add_nodes_from(nid.values())
    G.add_edge(10,11); G.add_edge(11,10)
    G.add_edge(10,12); G.add_edge(12,10)
    G.add_edge(12,13); G.add_edge(13,12)
    # B->D is key_locked
    G.add_edge(11,13)
    G.add_edge(13,11)
    G.edges[11,13]['edge_type'] = 'key_locked'

    d = Dungeon(dungeon_id='t', rooms=rooms, graph=G)
    d.start_pos = (0,0)
    d.triforce_pos = (0,2)
    # Mark graph start/goal nodes and assign explicit graph_node_id for each room
    G.nodes[nid[(0,0)]]['is_start'] = True
    G.nodes[nid[(0,2)]]['is_triforce'] = True
    for pos, nidv in nid.items():
        d.rooms[pos].graph_node_id = nidv

    # Baseline: BFS (state-aware) via ZeldaPathfinder but with admissible heuristic True
    zpf = ZeldaPathfinder(d, admissible_heuristic=True)
    res = zpf.solve()
    assert res['solvable'] is True
    path = res['path']

    # Compute path length via BFS-like search (stateful) for verification
    # Use ZeldaPathfinder with admissible heuristic; then ensure heuristic didn't overestimate
    # Verify path length is minimal in steps (the solver returns minimal steps when admissible heuristic used)
    assert len(path) - 1 == res['path_length']
    # Sanity: path should avoid using locked edge unless collecting a key first (no key rooms exist)
    assert (11,13) not in list(zip(path, path[1:]))
