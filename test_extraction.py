"""
Test script to verify VGLC room extraction is working correctly.
This tests the PRECISE GridBasedRoomExtractor (forensically verified).
"""

import sys
sys.path.insert(0, r"C:\Users\MPhuc\Desktop\KLTN\Data")

import numpy as np
from pathlib import Path
from adapter import GridBasedRoomExtractor

def visualize_room(room_grid: np.ndarray, position: tuple) -> None:
    """Print a room grid for visual inspection."""
    print(f"\n=== Room at {position} ({room_grid.shape[0]}x{room_grid.shape[1]}) ===")
    for row in room_grid:
        print(''.join(row))

def test_single_file():
    """Deep test on tloz1_1.txt with detailed format analysis."""
    filepath = r"C:\Users\MPhuc\Desktop\KLTN\Data\The Legend of Zelda\Processed\tloz1_1.txt"
    
    print("="*70)
    print("DEEP TEST: tloz1_1.txt")
    print("="*70)
    
    # Load raw file
    with open(filepath, 'r') as f:
        lines = [line.rstrip('\n') for line in f]
    
    print(f"Total lines: {len(lines)}")
    max_width = max(len(line) for line in lines) if lines else 0
    print(f"Max line width: {max_width}")
    print(f"Expected grid: {len(lines)//16} rows x {max_width//11} cols")
    
    # Format analysis
    print("\n--- RAW FORMAT ANALYSIS (Line 16) ---")
    if len(lines) > 16:
        line16 = lines[16]
        print(f"Full line ({len(line16)} chars): {line16}")
        if len(line16) >= 66:
            print(f"  Cols 0-10:   '{line16[0:11]}'")
            print(f"  Cols 11-21:  '{line16[11:22]}'")
            print(f"  Cols 22-32:  '{line16[22:33]}'")
            print(f"  Cols 33-43:  '{line16[33:44]}'")
            print(f"  Cols 44-54:  '{line16[44:55]}'")
            print(f"  Cols 55-65:  '{line16[55:66]}'")
    
    # Test extractor
    print("\n--- EXTRACTION RESULTS ---")
    extractor = GridBasedRoomExtractor()
    rooms = extractor.extract(filepath)
    
    print(f"Rooms extracted: {len(rooms)}")
    
    # Show room positions
    print("\nRoom positions:")
    for (row, col), grid in rooms:
        print(f"  ({row}, {col}) - {grid.shape}")
    
    # Show sample rooms
    for i, ((row, col), grid) in enumerate(rooms[:4]):
        visualize_room(grid, (row, col))
    
    # Validation
    print("\n--- VALIDATION ---")
    issues = []
    for (row, col), grid in rooms:
        # Check dimensions
        if grid.shape != (16, 11):
            issues.append(f"Room ({row},{col}): wrong shape {grid.shape}")
        
        # Check walls
        wall_count = np.sum(grid == 'W')
        if wall_count < 30:
            issues.append(f"Room ({row},{col}): too few walls ({wall_count})")
        
        # Check floor
        floor_count = np.sum(grid == 'F')
        if floor_count < 10:
            issues.append(f"Room ({row},{col}): too few floors ({floor_count})")
        
        # Check for dashes in interior
        interior = grid[2:-2, 2:-2]
        dash_count = np.sum(interior == '-')
        if dash_count > 10:
            issues.append(f"Room ({row},{col}): has {dash_count} dashes in interior (extraction error)")
    
    if issues:
        print("ISSUES FOUND:")
        for issue in issues[:10]:
            print(f"  - {issue}")
    else:
        print("ALL ROOMS VALID!")
    
    return len(rooms), len(issues)


def test_all_files():
    """Test room extraction on all VGLC files."""
    data_root = Path(r"C:\Users\MPhuc\Desktop\KLTN\Data\The Legend of Zelda\Processed")
    
    extractor = GridBasedRoomExtractor()
    
    print("\n" + "="*70)
    print("FULL VGLC EXTRACTION TEST")
    print("="*70)
    
    results = []
    
    for txt_file in sorted(data_root.glob("tloz*.txt")):
        rooms = extractor.extract(str(txt_file))
        
        # Count tiles
        total_start = sum(np.sum(g == 'S') for _, g in rooms)
        total_triforce = sum(np.sum(g == 'T') for _, g in rooms)
        total_doors = sum(np.sum(g == 'D') for _, g in rooms)
        
        status = "OK"
        if len(rooms) < 5:
            status = "FEW_ROOMS"
        elif total_doors == 0:
            status = "NO_DOORS"
        
        results.append((txt_file.name, len(rooms), total_start, total_triforce, total_doors, status))
        print(f"  {txt_file.name}: {len(rooms):2d} rooms, S={total_start}, T={total_triforce}, D={total_doors:2d} [{status}]")
    
    print("\nSummary:")
    print(f"  Files tested: {len(results)}")
    print(f"  Total rooms: {sum(r[1] for r in results)}")
    ok_count = sum(1 for r in results if r[5] == "OK")
    print(f"  Status OK: {ok_count}/{len(results)}")


if __name__ == "__main__":
    test_single_file()
    test_all_files()
