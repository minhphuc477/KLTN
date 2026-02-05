"""Debug script to investigate solver issues"""
from Data.zelda_core import ZeldaDungeonAdapter, SEMANTIC_PALETTE
from simulation.validator import ZeldaLogicEnv, StateSpaceAStar, WALKABLE_IDS, BLOCKING_IDS, PUSHABLE_IDS, WATER_IDS
from pathlib import Path
import numpy as np

data_root = Path('Data/The Legend of Zelda')
adapter = ZeldaDungeonAdapter(str(data_root))

dungeon = adapter.load_dungeon(1, variant=1)
stitched = adapter.stitch_dungeon(dungeon)
grid = stitched.global_grid

print('=== KEYS IN GRID ===')
key_small_count = np.sum(grid == SEMANTIC_PALETTE['KEY_SMALL'])
key_boss_count = np.sum(grid == SEMANTIC_PALETTE['KEY_BOSS'])
key_item_count = np.sum(grid == SEMANTIC_PALETTE['KEY_ITEM'])
print(f'KEY_SMALL (30) in grid: {key_small_count}')
print(f'KEY_BOSS (31) in grid: {key_boss_count}')
print(f'KEY_ITEM (32) in grid: {key_item_count}')
print()

print('=== GRAPH NODE ITEMS ===')
for node, attrs in dungeon.graph.nodes(data=True):
    label = attrs.get('label', '')
    has_key = attrs.get('has_key')
    has_item = attrs.get('has_item')
    print(f'Node {node}: label="{label}", has_key={has_key}, has_item={has_item}')
    
print()
# Count items
key_nodes = [n for n, a in dungeon.graph.nodes(data=True) if a.get('has_key')]
item_nodes = [n for n, a in dungeon.graph.nodes(data=True) if a.get('has_item')]
print(f'Nodes with keys: {len(key_nodes)} - {key_nodes}')
print(f'Nodes with items: {len(item_nodes)} - {item_nodes}')

# Now check which rooms map to which nodes
print()
print('=== ROOM TO NODE MAPPING ===')
stitched = adapter.stitch_dungeon(dungeon)
for room_pos, node_id in sorted(stitched.room_to_node.items()):
    node_attrs = dungeon.graph.nodes[node_id] if node_id in dungeon.graph.nodes else {}
    label = node_attrs.get('label', '')
    print(f'Room {room_pos} -> Node {node_id} (label="{label}")')

print()
print('=== CHECKING KEY_ITEM PLACEMENT ===')
# The graph says nodes 1, 2, 9, 13 have items (I or i)
# These should have KEY_ITEM tiles in the grid
grid = stitched.global_grid

# Check each room that should have items
for node_id in item_nodes:
    # Find room for this node
    room_pos = None
    for rp, nid in stitched.room_to_node.items():
        if nid == node_id:
            room_pos = rp
            break
    
    if room_pos is None:
        print(f'Node {node_id}: NO ROOM MAPPED')
        continue
    
    if room_pos not in stitched.room_positions:
        print(f'Node {node_id}: Room {room_pos} not in room_positions')
        continue
    
    r_off, c_off = stitched.room_positions[room_pos]
    room_slice = grid[r_off:r_off+16, c_off:c_off+11]
    
    # Check for KEY_ITEM tiles
    key_item_count = np.sum(room_slice == SEMANTIC_PALETTE['KEY_ITEM'])
    item_minor_count = np.sum(room_slice == SEMANTIC_PALETTE['ITEM_MINOR'])
    
    label = dungeon.graph.nodes[node_id].get('label', '')
    print(f'Node {node_id} (label="{label}"): Room {room_pos}')
    print(f'  KEY_ITEM (32) count: {key_item_count}')
    print(f'  ITEM_MINOR (33) count: {item_minor_count}')
    
    # Print unique tiles in this room
    unique, counts = np.unique(room_slice, return_counts=True)
    print(f'  Room tiles: {dict(zip(unique, counts))}')
