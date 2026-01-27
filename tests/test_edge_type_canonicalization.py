import networkx as nx
from Data.zelda_core import RoomGraphMatcher


def test_edge_type_canonicalization():
    G = nx.DiGraph()
    # Add edges with various label forms
    G.add_node(1); G.add_node(2); G.add_node(3); G.add_node(4)
    G.add_edge(1,2, label='k')
    G.add_edge(2,3, label='K')
    G.add_edge(3,4, label='b')
    G.add_edge(4,1, label='S')
    # Pre-set edge_type variants
    G.add_edge(1,3, edge_type='k')
    G.add_edge(2,4, edge_type='B')

    matcher = RoomGraphMatcher()
    matcher._normalize_graph(G)

    assert G.edges[1,2]['edge_type'] == 'key_locked'
    assert G.edges[2,3]['edge_type'] == 'boss_locked'
    assert G.edges[3,4]['edge_type'] == 'bombable'
    # 'S' should map to 'switch_locked'
    assert G.edges[4,1]['edge_type'] == 'switch_locked'
    assert G.edges[1,3]['edge_type'] == 'key_locked'
    assert G.edges[2,4]['edge_type'] == 'bombable'
