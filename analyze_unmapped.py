"""Analyze unmapped nodes in D7-1."""
from src.data.zelda_core import ZeldaDungeonAdapter

adapter = ZeldaDungeonAdapter('Data/The Legend of Zelda')
dungeon = adapter.load_dungeon(7, variant=1)

print('='*70)
print('D7-1 LOADED DUNGEON ANALYSIS')  
print('='*70)
print()
print(f'Rooms in dungeon: {len(dungeon.rooms)}')
print(f'Graph nodes: {len(dungeon.graph.nodes())}')
print()

# Check which nodes are mapped
mapped_nodes = set()
for pos, room in dungeon.rooms.items():
    if room.graph_node_id is not None:
        mapped_nodes.add(room.graph_node_id)
        
print(f'Nodes with rooms: {len(mapped_nodes)}')
print(f'Mapped: {sorted(mapped_nodes)}')

all_nodes = set(dungeon.graph.nodes())
unmapped = all_nodes - mapped_nodes
print(f'Unmapped nodes: {sorted(unmapped)}')

# Check what labels these unmapped nodes have
print()
print('Unmapped node details:')
for n in sorted(unmapped):
    attrs = dungeon.graph.nodes[n]
    label = attrs.get('label', '')
    is_virtual = attrs.get('is_virtual', False)
    virtual_parent = attrs.get('virtual_parent')
    print(f'  Node {n}: label="{label}", is_virtual={is_virtual}, virtual_parent={virtual_parent}')
    
# Check what neighbors the unmapped nodes have
print()
print('Unmapped node neighbors:')
for n in sorted(unmapped):
    successors = list(dungeon.graph.successors(n))
    predecessors = list(dungeon.graph.predecessors(n))
    all_neighbors = set(successors + predecessors)
    print(f'  Node {n}: neighbors={sorted(all_neighbors)}')
    for neighbor in sorted(all_neighbors):
        if neighbor in mapped_nodes:
            # Find which room maps to this neighbor
            for pos, room in dungeon.rooms.items():
                if room.graph_node_id == neighbor:
                    print(f'    -> Neighbor {neighbor} is in room {pos}')
                    break
