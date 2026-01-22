#!/usr/bin/env python3
"""Analyze a spritesheet to recommend tile size, margins, spacing, and report duplicates/empty tiles."""
from pathlib import Path
from PIL import Image
import hashlib
import sys

def analyze(path: Path):
    img = Image.open(path).convert('RGBA')
    w, h = img.size
    print(f'Size: {w}x{h}')

    # Candidate tile sizes
    candidates = [8, 16, 32]

    for tile in candidates:
        cols = w // tile
        rows = h // tile
        rem_w = w % tile
        rem_h = h % tile
        print(f'Candidate tile {tile} -> cols={cols}, rows={rows}, rem_w={rem_w}, rem_h={rem_h}')

        # Count non-empty tiles and duplicates
        nonempty = 0
        empties = 0
        hashes = {}
        for r in range(rows):
            for c in range(cols):
                x = c * tile
                y = r * tile
                region = img.crop((x, y, x + tile, y + tile))
                data = region.tobytes()
                # Check if fully transparent
                if all(b == 0 for b in data[3::4]):
                    empties += 1
                else:
                    nonempty += 1
                hsh = hashlib.sha1(data).hexdigest()
                hashes.setdefault(hsh, []).append((r, c))
        unique = sum(1 for k, v in hashes.items() if any(True for _ in v))
        dup_counts = [len(v) for v in hashes.values() if len(v) > 1]
        print(f'  nonempty={nonempty}, empty={empties}, unique_tiles={unique}, duplicates_count={sum(dup_counts)}')
        # Show top duplicate groups
        dup_examples = [v for v in hashes.values() if len(v) > 1][:5]
        if dup_examples:
            print('  Sample duplicate groups (first 5):')
            for group in dup_examples:
                print('   ', group[:8])
        print('')

    # Try to detect spacing/margin by looking for full transparent columns/rows
    pixels = img.load()
    # Check vertical lines
    empty_cols = []
    for x in range(w):
        col_empty = True
        for y in range(h):
            if pixels[x, y][3] != 0:
                col_empty = False
                break
        if col_empty:
            empty_cols.append(x)
    empty_rows = []
    for y in range(h):
        row_empty = True
        for x in range(w):
            if pixels[x, y][3] != 0:
                row_empty = False
                break
        if row_empty:
            empty_rows.append(y)
    print(f'Empty columns count: {len(empty_cols)}, sample indexes: {empty_cols[:10]}')
    print(f'Empty rows count: {len(empty_rows)}, sample indexes: {empty_rows[:10]}')

    print('\nRecommendation:')
    # Simple heuristic: choose tile size with minimal remainder and reasonable unique count
    best = None
    for tile in candidates:
        cols = w // tile
        rows = h // tile
        rem = (w % tile) + (h % tile)
        if best is None or rem < best[0]:
            best = (rem, tile)
    print(f'  Suggested tile size: {best[1]} (minimal remainder {best[0]})')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: analyze_spritesheet.py <image>')
        sys.exit(1)
    analyze(Path(sys.argv[1]))
