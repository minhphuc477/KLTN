"""Check edges for specific nodes."""
import networkx as nx
G = nx.drawing.nx_pydot.read_dot('Data/The Legend of Zelda/Graph Processed/LoZ_1.dot')

for node in ['5', '6', '11', '13', '17']:
    print(f'Edges involving node {node}:')
    for u, v, data in G.edges(data=True):
        if u == node or v == node:
            label = data.get('label', '').strip('"')
            print(f'  {u} -> {v}: "{label}"')
    print()
