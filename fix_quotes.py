#!/usr/bin/env python3
"""Fix escaped quotes in grammar.py"""

import re

# Read file 
with open('src/generation/grammar.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix escaped quotes: \\" -> "
# Also fix escaped triple quotes
content = content.replace('\\"', '"')
content = content.replace(r'\"\"\"', '"""')

# Write back
with open('src/generation/grammar.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Fixed escaped quotes in grammar.py")
