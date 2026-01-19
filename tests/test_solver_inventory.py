import networkx as nx
from graph_solver import solve_with_inventory


def test_solve_with_inventory_key_opens_door():
    G = nx.DiGraph()
    G.add_node('start')
    G.add_node('kroom')
    G.add_node('locked')
    G.add_node('goal')

    G.add_edge('start', 'kroom')
    G.add_edge('kroom', 'locked')
    G.add_edge('locked', 'goal', edge_type='key_locked')

    items = {'kroom': 'small_key'}

    def goal_test(node, inv):
        return node == 'goal'

    path, actions, ok = solve_with_inventory(G, 'start', goal_test, items)
    assert ok is True
    assert 'collect:small_key' in ' '.join(actions)
    assert any(a.startswith('pass:key_locked') for a in actions)

    # Without the key available it should fail
    path2, actions2, ok2 = solve_with_inventory(G, 'start', goal_test, {})
    assert ok2 is False
