"""Diagnostic audit script for VGLC data processing pipeline."""
import sys
import os
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def main():
    # Test 1: Analyze raw data dimensions
    print('=== RAW DATA ANALYSIS ===')
    for fname in ['tloz1_1.txt', 'tloz2_1.txt']:
        fpath = f'Data/The Legend of Zelda/Processed/{fname}'
        with open(fpath) as f:
            lines = [l.rstrip('\n') for l in f]
        max_w = max(len(l) for l in lines)
        print(f'{fname}: {len(lines)} lines, max width: {max_w}')
        print(f'  Room rows: {len(lines)//16}, Room cols: {max_w//11}')

    # Test 2: Check DOT file structure
    print('\n=== DOT FILE ANALYSIS ===')
    for dot_file in ['LoZ_1.dot', 'LoZ_2.dot']:
        fpath = f'Data/The Legend of Zelda/Graph Processed/{dot_file}'
        with open(fpath) as f:
            content = f.read()
        nodes = re.findall(r'^\s*(\d+)\s*\[', content, re.MULTILINE)
        edges = re.findall(r'(\d+)\s*->\s*(\d+)', content)
        
        start_nodes = []
        for m in re.finditer(r'^\s*(\d+)\s*\[([^\]]*)\]', content, re.MULTILINE):
            nid = int(m.group(1))
            attrs = m.group(2)
            lm = re.search(r'label\s*=\s*"([^"]*)"', attrs)
            label = lm.group(1) if lm else ''
            parts = [p.strip() for p in label.split(',')]
            if 's' in parts:
                start_nodes.append((nid, label))
        
        print(f'{dot_file}: {len(nodes)} nodes, {len(edges)} directed edges')
        print(f'  Start nodes: {start_nodes}')

    # Test 3: Run extractors
    print('\n=== EXTRACTOR TEST ===')
    from src.data.zelda_core import GridBasedRoomExtractor, VGLCParser, DOTParser

    for fname in ['tloz1_1.txt', 'tloz2_1.txt']:
        fpath = f'Data/The Legend of Zelda/Processed/{fname}'
        extractor = GridBasedRoomExtractor()
        rooms_grid = extractor.extract(fpath)
        parser = VGLCParser()
        rooms_vglc = parser.parse(fpath)
        print(f'{fname}:')
        print(f'  GridBasedRoomExtractor: {len(rooms_grid)} rooms')
        print(f'    Positions: {sorted([pos for pos, _ in rooms_grid])}')
        print(f'  VGLCParser: {len(rooms_vglc)} rooms')
        print(f'    Positions: {sorted(rooms_vglc.keys())}')

    # Test 4: DOT parser
    print('\n=== DOT PARSER TEST ===')
    dot_parser = DOTParser()
    for dot_file in ['LoZ_1.dot', 'LoZ_2.dot']:
        fpath = f'Data/The Legend of Zelda/Graph Processed/{dot_file}'
        g = dot_parser.parse(fpath)
        start_n = [n for n, d in g.nodes(data=True) if d.get('is_start')]
        trif_n = [n for n, d in g.nodes(data=True) if d.get('is_triforce')]
        print(f'{dot_file}: {len(g.nodes())} nodes, {len(g.edges())} edges')
        print(f'  Start: {start_n}, Triforce: {trif_n}')
        for n, d in sorted(g.nodes(data=True)):
            print(f'  Node {n}: label="{d.get("label","")}" start={d.get("is_start")} trif={d.get("is_triforce")}')

    # Test 5: Full pipeline
    print('\n=== FULL PIPELINE TEST ===')
    from src.data.zelda_core import ZeldaDungeonAdapter
    adapter = ZeldaDungeonAdapter('Data/The Legend of Zelda')
    
    for dnum in [1, 2]:
        try:
            dungeon = adapter.load_dungeon(dnum, 1)
            print(f'\nDungeon {dnum}:')
            print(f'  Rooms: {len(dungeon.rooms)}')
            print(f'  Graph nodes: {len(dungeon.graph.nodes())}')
            print(f'  Graph edges: {len(dungeon.graph.edges())}')
            print(f'  Start pos: {dungeon.start_pos}')
            print(f'  Triforce pos: {dungeon.triforce_pos}')
            
            # Check which rooms have node assignments
            assigned = sum(1 for r in dungeon.rooms.values() if r.graph_node_id is not None)
            print(f'  Rooms with node assignment: {assigned}/{len(dungeon.rooms)}')
            
            # Check start node handling
            start_nodes = [n for n, d in dungeon.graph.nodes(data=True) if d.get('is_start')]
            print(f'  Start graph nodes: {start_nodes}')
            
            # Check if start node has a room
            for sn in start_nodes:
                has_room = any(r.graph_node_id == sn for r in dungeon.rooms.values())
                print(f'  Start node {sn} has room assigned: {has_room}')
                if has_room:
                    room = [r for r in dungeon.rooms.values() if r.graph_node_id == sn][0]
                    print(f'    Room position: {room.position}, has_stair: {room.has_stair}')
                    
            # Check door detection on corridor rooms
            print(f'  Room details (doors):')
            for pos, room in sorted(dungeon.rooms.items()):
                door_count = sum(room.doors.values())
                node_id = room.graph_node_id
                print(f'    {pos}: doors={room.doors} (count={door_count}), node={node_id}, stair={room.has_stair}')
        except Exception as e:
            print(f'\nDungeon {dnum}: ERROR - {e}')
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()
