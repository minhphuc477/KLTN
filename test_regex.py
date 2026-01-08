"""Test regex parsing for DOT file."""
import re

# Read raw DOT file
with open('Data/The Legend of Zelda/Graph Processed/LoZ_1.dot', 'r') as f:
    content = f.read()

print("Content sample:")
print(repr(content[:500]))
print()

NODE_CONTENT_MAP = {
    'e': 'enemy',
    's': 'start',
    't': 'triforce',
    'b': 'boss',
    'k': 'key',
    'K': 'boss_key',
    'I': 'key_item',
    'i': 'item',
    'p': 'puzzle',
}

# Test simple regex
node_pattern = r'(\d+)\s*\[label="([^"]*)"\]'
matches = list(re.finditer(node_pattern, content))
print(f"Pattern: {node_pattern}")
print(f"Matches found: {len(matches)}")
print()

print("=== Nodes with START or TRIFORCE ===")
for match in matches:
    node_id = int(match.group(1))
    label = match.group(2)
    
    contents = [c.strip() for c in label.split(',') if c.strip()]
    content_types = []
    for c in contents:
        if c in NODE_CONTENT_MAP:
            content_types.append(NODE_CONTENT_MAP[c])
    
    is_start = 'start' in content_types
    has_triforce = 'triforce' in content_types
    
    if is_start or has_triforce:
        print(f"Node {node_id}: label='{label}' -> types={content_types}")
        print(f"         is_start={is_start}, has_triforce={has_triforce}")

