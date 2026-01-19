"""Test the new state-space solver with all validation modes."""
from pathlib import Path
from Data.zelda_core import (
    ZeldaDungeonAdapter, 
    DungeonSolver, 
    ValidationMode,
    StateSpaceGraphSolver
)

def test_state_space_solver():
    print("=" * 80)
    print("STATE-SPACE SOLVER TEST")
    print("Testing all 18 dungeons with STRICT, REALISTIC, and FULL modes")
    print("=" * 80)
    print()
    
    data_root = Path("Data/The Legend of Zelda")
    adapter = ZeldaDungeonAdapter(str(data_root))
    solver = DungeonSolver()
    
    results = {
        'strict': [],
        'realistic': [],
        'full': []
    }
    
    for quest in [1, 2]:
        for level in range(1, 10):
            name = f"D{level}-{quest}"
            
            try:
                dungeon = adapter.load_dungeon(level, quest)
                stitched = adapter.stitch_dungeon(dungeon)
                
                # Test each mode
                strict = solver.solve(stitched, mode=ValidationMode.STRICT)
                realistic = solver.solve(stitched, mode=ValidationMode.REALISTIC)
                full = solver.solve(stitched, mode=ValidationMode.FULL)
                
                results['strict'].append((name, strict['solvable']))
                results['realistic'].append((name, realistic['solvable']))
                results['full'].append((name, full['solvable']))
                
                # Format output
                s = "✓" if strict['solvable'] else "✗"
                r = "✓" if realistic['solvable'] else "✗"
                f = "✓" if full['solvable'] else "✗"
                
                # Get details from full mode
                if full['solvable']:
                    edge_types = full.get('edge_types', [])
                    keys_avail = full.get('keys_available', 0)
                    keys_used = full.get('keys_used', 0)
                    
                    # Count edge types
                    type_counts = {}
                    for et in edge_types:
                        type_counts[et] = type_counts.get(et, 0) + 1
                    
                    detail = f"Keys: {keys_avail} found, {keys_used} used | "
                    detail += " ".join([f"{k}:{v}" for k, v in type_counts.items()])
                else:
                    detail = full.get('reason', 'Unknown')
                
                print(f"{name}: STRICT={s} REAL={r} FULL={f}  |  {detail}")
                
            except Exception as e:
                print(f"{name}: ERROR - {e}")
                import traceback
                traceback.print_exc()
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    strict_pass = sum(1 for _, ok in results['strict'] if ok)
    realistic_pass = sum(1 for _, ok in results['realistic'] if ok)
    full_pass = sum(1 for _, ok in results['full'] if ok)
    
    print(f"STRICT mode (normal doors only):        {strict_pass}/18")
    print(f"REALISTIC mode (+ soft-lock + stairs):  {realistic_pass}/18")
    print(f"FULL mode (with key/bomb tracking):     {full_pass}/18")
    
    print()
    print("Legend:")
    print("  STRICT    = Only traverse normal doors (what's visible in tile data)")
    print("  REALISTIC = Normal + soft-locked (one-way) + stairs")
    print("  FULL      = All edges with key/bomb inventory tracking")
    
if __name__ == "__main__":
    test_state_space_solver()
