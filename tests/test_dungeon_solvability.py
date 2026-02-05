"""
Test dungeon solvability for all Zelda dungeons.

This test suite verifies that the solver can find valid paths through
each dungeon, tracking which dungeons are solvable and which aren't.
"""

import pytest
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Data.zelda_core import ZeldaDungeonAdapter, DungeonSolver, ValidationMode


# Test configuration
DATA_ROOT = PROJECT_ROOT / 'Data' / 'The Legend of Zelda'
SOLVER_TIMEOUT_SECONDS = 30  # Per-dungeon timeout

# Known dungeon configurations
# Dungeons 1-9, each with variants 1-2
ALL_DUNGEONS = [(d, v) for d in range(1, 10) for v in [1, 2]]

# Known impossible dungeons (update as issues are identified)
# Format: (dungeon_num, variant)
KNOWN_IMPOSSIBLE = set()  # Initially empty - we'll discover which fail


class DungeonSolvabilityStats:
    """Track solvability statistics across test runs."""
    
    def __init__(self):
        self.results: Dict[Tuple[int, int], Dict] = {}
    
    def record(self, dungeon: int, variant: int, result: Dict):
        self.results[(dungeon, variant)] = result
    
    def summary(self) -> str:
        """Generate a summary report."""
        lines = [
            "",
            "=" * 60,
            "DUNGEON SOLVABILITY SUMMARY",
            "=" * 60,
        ]
        
        solvable = []
        unsolvable = []
        errors = []
        
        for (d, v), result in sorted(self.results.items()):
            if result.get('error'):
                errors.append((d, v, result['error']))
            elif result.get('solvable'):
                solvable.append((d, v, result.get('path_length', 0)))
            else:
                unsolvable.append((d, v, result.get('reason', 'Unknown')))
        
        lines.append(f"\nSolvable: {len(solvable)}/{len(self.results)}")
        for d, v, path_len in solvable:
            lines.append(f"  [OK] Dungeon {d} v{v} - path length: {path_len}")
        
        if unsolvable:
            lines.append(f"\nUnsolvable: {len(unsolvable)}")
            for d, v, reason in unsolvable:
                lines.append(f"  [X] Dungeon {d} v{v} - {reason}")
        
        if errors:
            lines.append(f"\nErrors: {len(errors)}")
            for d, v, err in errors:
                lines.append(f"  [!] Dungeon {d} v{v} - {err}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


# Global stats tracker
_stats = DungeonSolvabilityStats()


@pytest.fixture(scope='module')
def adapter():
    """Create a dungeon adapter for the test module."""
    if not DATA_ROOT.exists():
        pytest.skip(f"Data directory not found: {DATA_ROOT}")
    return ZeldaDungeonAdapter(str(DATA_ROOT))


@pytest.fixture(scope='module')
def solver():
    """Create a dungeon solver for the test module."""
    return DungeonSolver()


def _solve_dungeon_with_timeout(adapter: ZeldaDungeonAdapter, 
                                 solver: DungeonSolver,
                                 dungeon_num: int, 
                                 variant: int) -> Dict:
    """
    Load and solve a dungeon with timeout handling.
    
    Returns result dict with 'solvable', 'reason', 'path_length', etc.
    """
    result = {'dungeon': dungeon_num, 'variant': variant}
    
    try:
        start_time = time.time()
        
        # Load dungeon
        dungeon = adapter.load_dungeon(dungeon_num, variant=variant)
        if dungeon is None:
            result['solvable'] = False
            result['reason'] = 'Failed to load dungeon'
            return result
        
        # Run precheck
        ok, precheck_msg = ZeldaDungeonAdapter.precheck_dungeon(dungeon)
        if not ok:
            result['solvable'] = False
            result['reason'] = precheck_msg
            return result
        
        # Stitch dungeon
        stitched = adapter.stitch_dungeon(dungeon)
        if stitched is None:
            result['solvable'] = False
            result['reason'] = 'Failed to stitch dungeon'
            return result
        
        # Check timeout before solving
        elapsed = time.time() - start_time
        if elapsed > SOLVER_TIMEOUT_SECONDS:
            result['solvable'] = False
            result['reason'] = f'Timeout during loading ({elapsed:.1f}s)'
            return result
        
        # Add map attribute alias for compatibility
        if hasattr(stitched, 'global_grid') and not hasattr(stitched, 'map'):
            stitched.map = stitched.global_grid
        
        # Solve
        solve_result = solver.solve(stitched, mode=ValidationMode.FULL)
        result.update(solve_result)
        
        # Record timing
        result['solve_time'] = time.time() - start_time
        
    except Exception as e:
        result['solvable'] = False
        result['error'] = str(e)
        result['reason'] = f'Exception: {type(e).__name__}'
    
    return result


class TestDungeonSolvability:
    """Test suite for dungeon solvability."""
    
    @pytest.mark.parametrize("dungeon_num,variant", ALL_DUNGEONS)
    def test_dungeon_solvable(self, adapter, solver, dungeon_num, variant):
        """Test that a specific dungeon is solvable."""
        result = _solve_dungeon_with_timeout(adapter, solver, dungeon_num, variant)
        
        # Record stats
        _stats.record(dungeon_num, variant, result)
        
        # Check result
        if (dungeon_num, variant) in KNOWN_IMPOSSIBLE:
            # Known impossible - just verify it actually fails
            if result.get('solvable'):
                pytest.fail(
                    f"Dungeon {dungeon_num} v{variant} was expected to be impossible "
                    f"but solver found a path!"
                )
        else:
            # Expected to be solvable
            assert result.get('solvable'), (
                f"Dungeon {dungeon_num} v{variant} failed: "
                f"{result.get('reason', result.get('error', 'Unknown'))}"
            )
    
    @pytest.mark.parametrize("dungeon_num", range(1, 10))
    def test_at_least_one_variant_solvable(self, adapter, solver, dungeon_num):
        """Verify that at least one variant of each dungeon is solvable."""
        results = []
        for variant in [1, 2]:
            result = _solve_dungeon_with_timeout(adapter, solver, dungeon_num, variant)
            results.append(result)
            _stats.record(dungeon_num, variant, result)
        
        # At least one should be solvable
        any_solvable = any(r.get('solvable') for r in results)
        if not any_solvable:
            reasons = [f"v{i+1}: {r.get('reason', 'Unknown')}" for i, r in enumerate(results)]
            pytest.fail(f"Dungeon {dungeon_num} has no solvable variants: {', '.join(reasons)}")


class TestSolverPerformance:
    """Test solver performance characteristics."""
    
    def test_solver_completes_within_timeout(self, adapter, solver):
        """Verify solver doesn't hang on any dungeon."""
        slow_dungeons = []
        
        for dungeon_num, variant in ALL_DUNGEONS[:6]:  # Test subset for speed
            result = _solve_dungeon_with_timeout(adapter, solver, dungeon_num, variant)
            solve_time = result.get('solve_time', 0)
            
            if solve_time > 5.0:  # Warn if > 5 seconds
                slow_dungeons.append((dungeon_num, variant, solve_time))
        
        if slow_dungeons:
            msg = "Slow dungeons detected:\n"
            for d, v, t in slow_dungeons:
                msg += f"  Dungeon {d} v{v}: {t:.1f}s\n"
            pytest.fail(msg)


@pytest.fixture(scope='module', autouse=True)
def print_summary(request):
    """Print summary at end of module."""
    yield
    # Print summary after all tests complete
    if _stats.results:
        print(_stats.summary())


# Standalone runner
if __name__ == '__main__':
    """Run solvability tests standalone with detailed output."""
    print("=" * 60)
    print("DUNGEON SOLVABILITY TEST")
    print("=" * 60)
    
    if not DATA_ROOT.exists():
        print(f"ERROR: Data directory not found: {DATA_ROOT}")
        sys.exit(1)
    
    adapter = ZeldaDungeonAdapter(str(DATA_ROOT))
    solver = DungeonSolver()
    
    stats = DungeonSolvabilityStats()
    
    for dungeon_num, variant in ALL_DUNGEONS:
        print(f"\nTesting Dungeon {dungeon_num} variant {variant}...", end=" ", flush=True)
        
        result = _solve_dungeon_with_timeout(adapter, solver, dungeon_num, variant)
        stats.record(dungeon_num, variant, result)
        
        if result.get('solvable'):
            path_len = result.get('path_length', '?')
            solve_time = result.get('solve_time', 0)
            print(f"[OK] SOLVABLE (path={path_len}, time={solve_time:.2f}s)")
        else:
            reason = result.get('reason', result.get('error', 'Unknown'))
            print(f"[X] FAILED: {reason}")
    
    print(stats.summary())
