"""Analyze graph connectivity to find path from START to TRIFORCE."""
import networkx as nx

# Parse the graph
G = nx.drawing.nx_pydot.read_dot('Data/The Legend of Zelda/Graph Processed/LoZ_1.dot')

# Find node 11 (triforce) neighbors
print('Node 11 (triforce) connections:')
for u, v, data in G.edges(data=True):
    if u == '11' or v == '11':
        label = data.get('label', '').strip('"')
        print(f'  {u} -> {v} (edge label: "{label}")')

# Find node 7 (start) neighbors  
print()
print('Node 7 (start) connections:')
for u, v, data in G.edges(data=True):
    if u == '7' or v == '7':
        label = data.get('label', '').strip('"')
        print(f'  {u} -> {v} (edge label: "{label}")')

# Show connectivity path from START to TRIFORCE
print()
print('Path from node 7 (start) to node 11 (triforce):')
try:
    # Use undirected version for path finding
    UG = G.to_undirected()
    path = nx.shortest_path(UG, '7', '11')
    print(f'  Shortest path: {" -> ".join(path)}')
    
    # Show labels on the path
    print('  Node labels on path:')
    for node in path:
        label = G.nodes[node].get('label', '').strip('"')
        print(f'    {node}: "{label}"')
except Exception as e:
    print(f'  No path found: {e}')

# Node 15 is boss (b), Node 11 is triforce (t)
# In Zelda, triforce is accessed after defeating boss
print()
print('Node 15 (boss) connections:')
for u, v, data in G.edges(data=True):
    if u == '15' or v == '15':
        label = data.get('label', '').strip('"')
        print(f'  {u} -> {v} (edge label: "{label}")')
