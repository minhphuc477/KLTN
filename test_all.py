"""
Comprehensive Test Suite for KLTN Zelda Dungeon Validator
==========================================================

This script tests:
1. Core dungeon loading and parsing
2. Room extraction and detection
3. Dungeon stitching and door connections
4. Solvability (START -> TRIFORCE pathfinding)
5. GUI integration (if pygame is available)

Run with: python test_all.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from Data.zelda_core import (
    ZeldaDungeonAdapter, 
    DungeonSolver, 
    SEMANTIC_PALETTE,
    ROOM_HEIGHT,
    ROOM_WIDTH
)

def test_dungeon_loading():
    """Test loading all dungeons - both variants."""
    print("\n" + "="*60)
    print("TEST 1: Dungeon Loading (18 variants total)")
    print("="*60)
    
    DATA_ROOT = Path(__file__).parent / "Data" / "The Legend of Zelda"
    adapter = ZeldaDungeonAdapter(str(DATA_ROOT))
    
    results = []
    for d in range(1, 10):
        for v in [1, 2]:
            try:
                dungeon = adapter.load_dungeon(d, variant=v)
                room_count = len(dungeon.rooms)
                has_start = dungeon.start_pos is not None
                has_triforce = dungeon.triforce_pos is not None
                
                status = "✓" if has_start and has_triforce else "⚠"
                results.append((d, v, room_count, has_start, has_triforce))
                print(f"  D{d}-{v}: {status} {room_count} rooms | START: {has_start} | TRIFORCE: {has_triforce}")
            except Exception as e:
                print(f"  D{d}-{v}: ✗ ERROR - {e}")
            results.append((d, 0, False, False))
    
    # Summary
    all_loaded = all(r[1] > 0 for r in results)
    all_have_start = all(r[2] for r in results)
    all_have_triforce = all(r[3] for r in results)
    
    print(f"\n  Summary:")
    print(f"    All dungeons loaded: {'✓' if all_loaded else '✗'}")
    print(f"    All have START: {'✓' if all_have_start else '✗'}")
    print(f"    All have TRIFORCE: {'✓' if all_have_triforce else '✗'}")
    
    return all_loaded and all_have_start and all_have_triforce


def test_room_extraction():
    """Test room extraction details."""
    print("\n" + "="*60)
    print("TEST 2: Room Extraction Details")
    print("="*60)
    
    DATA_ROOT = Path(__file__).parent / "Data" / "The Legend of Zelda"
    adapter = ZeldaDungeonAdapter(str(DATA_ROOT))
    
    # Test first dungeon in detail
    dungeon = adapter.load_dungeon(1)
    
    print(f"  Dungeon 1:")
    print(f"    Total rooms: {len(dungeon.rooms)}")
    
    # Check room dimensions
    all_correct_size = True
    for pos, room in dungeon.rooms.items():
        h, w = room.semantic_grid.shape
        if h != ROOM_HEIGHT or w != ROOM_WIDTH:
            print(f"    ⚠ Room {pos}: wrong size {(h,w)} (expected {(ROOM_HEIGHT, ROOM_WIDTH)})")
            all_correct_size = False
    
    if all_correct_size:
        print(f"    ✓ All rooms are {ROOM_HEIGHT}×{ROOM_WIDTH}")
    
    # Check door detection
    rooms_with_doors = sum(1 for room in dungeon.rooms.values() if any(room.doors.values()))
    print(f"    Rooms with doors: {rooms_with_doors}/{len(dungeon.rooms)}")
    
    # Check START room
    if dungeon.start_pos:
        start_room = dungeon.rooms[dungeon.start_pos]
        start_door_count = sum(start_room.doors.values())
        print(f"    START room: {dungeon.start_pos} (doors: {start_door_count})")
    
    # Check TRIFORCE room
    if dungeon.triforce_pos:
        triforce_room = dungeon.rooms[dungeon.triforce_pos]
        triforce_door_count = sum(triforce_room.doors.values())
        print(f"    TRIFORCE room: {dungeon.triforce_pos} (doors: {triforce_door_count})")
    
    return all_correct_size


def test_stitching():
    """Test dungeon stitching."""
    print("\n" + "="*60)
    print("TEST 3: Dungeon Stitching")
    print("="*60)
    
    DATA_ROOT = Path(__file__).parent / "Data" / "The Legend of Zelda"
    adapter = ZeldaDungeonAdapter(str(DATA_ROOT))
    
    results = []
    for d in range(1, 10):
        try:
            dungeon = adapter.load_dungeon(d)
            stitched = adapter.stitch_dungeon(dungeon)
            
            grid_shape = stitched.global_grid.shape
            has_start_global = stitched.start_global is not None
            has_triforce_global = stitched.triforce_global is not None
            
            # Count triforce markers in grid
            triforce_count = (stitched.global_grid == SEMANTIC_PALETTE['TRIFORCE']).sum()
            
            status = "✓" if has_start_global and has_triforce_global else "⚠"
            results.append((d, grid_shape, has_start_global, has_triforce_global, triforce_count))
            print(f"  D{d}: {status} Grid: {grid_shape} | START: {has_start_global} | TRIFORCE: {triforce_count}")
        except Exception as e:
            print(f"  D{d}: ✗ ERROR - {e}")
            results.append((d, (0,0), False, False, 0))
    
    all_stitched = all(r[1][0] > 0 for r in results)
    all_markers = all(r[2] and r[3] for r in results)
    
    print(f"\n  Summary:")
    print(f"    All dungeons stitched: {'✓' if all_stitched else '✗'}")
    print(f"    All have markers: {'✓' if all_markers else '✗'}")
    
    return all_stitched and all_markers


def test_solvability():
    """Test dungeon solvability - all 18 variants."""
    print("\n" + "="*60)
    print("TEST 4: Dungeon Solvability (BFS Pathfinding - 18 variants)")
    print("="*60)
    
    DATA_ROOT = Path(__file__).parent / "Data" / "The Legend of Zelda"
    adapter = ZeldaDungeonAdapter(str(DATA_ROOT))
    solver = DungeonSolver()
    
    results = []
    for d in range(1, 10):
        for v in [1, 2]:
            try:
                dungeon = adapter.load_dungeon(d, variant=v)
                stitched = adapter.stitch_dungeon(dungeon)
                result = solver.solve(stitched)
                
                solvable = result['solvable']
                path_len = result.get('path_length', 0)
                rooms = result.get('rooms_traversed', 0)
                
                status = "✓ SOLVABLE" if solvable else "✗ NOT SOLVABLE"
                results.append((d, v, solvable, path_len, rooms))
                
                if solvable:
                    print(f"  D{d}-{v}: {status} ({path_len} steps, {rooms} rooms)")
                else:
                    reason = result.get('reason', 'Unknown')
                    reachable = result.get('reachable_tiles', 0)
                    print(f"  D{d}-{v}: {status} - {reason} ({reachable} tiles reachable)")
            except Exception as e:
                print(f"  D{d}-{v}: ✗ ERROR - {e}")
                results.append((d, v, False, 0, 0))
    
    solvable_count = sum(1 for r in results if r[2])
    total_steps = sum(r[3] for r in results if r[2])
    avg_steps = total_steps / solvable_count if solvable_count > 0 else 0
    
    print(f"\n  Summary:")
    print(f"    Solvable: {solvable_count}/18 ({100*solvable_count/18:.1f}%)")
    print(f"    Average path length: {avg_steps:.1f} steps")
    
    return solvable_count == 18


def test_gui_integration():
    """Test GUI integration."""
    print("\n" + "="*60)
    print("TEST 5: GUI Integration")
    print("="*60)
    
    try:
        import pygame
        print(f"  ✓ Pygame {pygame.version.ver} installed")
        
        # Test map loading
        from gui_runner import load_maps_from_adapter
        maps = load_maps_from_adapter()
        
        if maps:
            print(f"  ✓ Loaded {len(maps)} maps for GUI")
            
            # Check map validity
            all_valid = True
            for i, m in enumerate(maps):
                if m.shape[0] < ROOM_HEIGHT or m.shape[1] < ROOM_WIDTH:
                    print(f"    ⚠ Map {i+1}: suspiciously small {m.shape}")
                    all_valid = False
            
            if all_valid:
                print(f"  ✓ All maps have valid dimensions")
            
            print(f"\n  To run GUI: python gui_runner.py")
            print(f"  Controls:")
            print(f"    - Arrow Keys: Move Link")
            print(f"    - SPACE: Auto-solve with A*")
            print(f"    - R: Reset map")
            print(f"    - N/P: Next/Previous dungeon")
            print(f"    - ESC: Quit")
            
            return True
        else:
            print(f"  ✗ Failed to load maps")
            return False
            
    except ImportError:
        print(f"  ⚠ Pygame not installed")
        print(f"  Install with: pip install pygame")
        return False
    except Exception as e:
        print(f"  ✗ GUI test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("KLTN ZELDA DUNGEON VALIDATOR - COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    tests = [
        ("Dungeon Loading", test_dungeon_loading),
        ("Room Extraction", test_room_extraction),
        ("Dungeon Stitching", test_stitching),
        ("Solvability", test_solvability),
        ("GUI Integration", test_gui_integration),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  ✗ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print(f"\n  Overall: {passed_count}/{total_count} tests passed ({100*passed_count/total_count:.1f}%)")
    
    if passed_count == total_count:
        print("\n  🎉 ALL TESTS PASSED! System is fully operational.")
    else:
        print("\n  ⚠️  Some tests failed. Check output above for details.")
    
    print("="*60)


if __name__ == "__main__":
    main()
