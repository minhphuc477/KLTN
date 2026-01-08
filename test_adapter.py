"""Test adapter to trace room/graph matching."""
import sys
import os
import re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import importlib.util
spec = importlib.util.spec_from_file_location('adapter', 'data/adapter.py')
adapter_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter_mod)

IntelligentDataAdapter = adapter_mod.IntelligentDataAdapter
SEMANTIC_PALETTE = adapter_mod.SEMANTIC_PALETTE
NODE_CONTENT_MAP = adapter_mod.NODE_CONTENT_MAP

print("=== NODE_CONTENT_MAP ===")
print(NODE_CONTENT_MAP)

# Load adapter
adapter = IntelligentDataAdapter('Data/The Legend of Zelda')

# Test graph parsing directly
filepath = 'Data/The Legend of Zelda/Graph Processed/LoZ_1.dot'
print(f"\n=== Reading file: {filepath} ===")

with open(filepath, 'r') as f:
    content = f.read()

print(f"Content length: {len(content)}")
print(f"First 300 chars: {repr(content[:300])}")

# Test the regex pattern
node_pattern = r'(\d+)\s*\[label="([^"]*)"\]'
matches = list(re.finditer(node_pattern, content))
print(f"\nRegex pattern: {node_pattern}")
print(f"Matches found: {len(matches)}")

print("\n=== Nodes with 's' or 't' in label ===")
for m in matches:
    node_id = int(m.group(1))
    label = m.group(2)
    if 's' in label or 't' in label or node_id in [7, 11]:
        print(f"  Node {node_id}: label='{label}'")
        
        # Process like the adapter does
        contents = [c.strip() for c in label.split(',') if c.strip()]
        content_types = []
        for c in contents:
            if c in NODE_CONTENT_MAP:
                content_types.append(NODE_CONTENT_MAP[c])
        
        is_start = 'start' in content_types
        has_triforce = 'triforce' in content_types
        print(f"         contents={contents}, types={content_types}")
        print(f"         is_start={is_start}, has_triforce={has_triforce}")

# Now test the actual adapter method
print("\n=== adapter.parse_dot_graph() result ===")
graph = adapter.parse_dot_graph(filepath)
for node, attrs in graph.nodes(data=True):
    if node in [7, 11] or attrs.get('is_start') or attrs.get('has_triforce'):
        print(f"  Node {node}: {attrs}")

