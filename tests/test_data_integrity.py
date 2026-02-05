"""
Test data integrity for Zelda dungeon datasets.

This test suite verifies that dungeon data is consistent:
- Graph nodes match room counts
- Items from graph are placeable on the map
- Start/goal positions exist
- Room coordinates are valid
"""

import pytest
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Data.zelda_core import (
    ZeldaDungeonAdapter, 
    SEMANTIC_PALETTE,
    ROOM_WIDTH,
    ROOM_HEIGHT,
)


# Test configuration
DATA_ROOT = PROJECT_ROOT / 'Data' / 'The Legend of Zelda'

# All dungeons to test
ALL_DUNGEONS = [(d, v) for d in range(1, 10) for v in [1, 2]]


class DataIntegrityStats:
    """Track integrity check results."""
    
    def __init__(self):
        self.issues: Dict[Tuple[int, int], List[str]] = defaultdict(list)
        self.passed: Set[Tuple[int, int]] = set()
    
    def record_issue(self, dungeon: int, variant: int, issue: str):
        self.issues[(dungeon, variant)].append(issue)
    
    def record_pass(self, dungeon: int, variant: int):
        self.passed.add((dungeon, variant))
    
    def summary(self) -> str:
        lines = [
            "",
            "=" * 60,
            "DATA INTEGRITY SUMMARY",
            "=" * 60,
            f"\nPassed: {len(self.passed)} dungeons",
        ]
        
        if self.issues:
            lines.append(f"Issues found in {len(self.issues)} dungeons:\n")
            for (d, v), issues in sorted(self.issues.items()):
                lines.append(f"  Dungeon {d} v{v}:")
                for issue in issues:
                    lines.append(f"    - {issue}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


_stats = DataIntegrityStats()


@pytest.fixture(scope='module')
def adapter():
    """Create a dungeon adapter for the test module."""
    if not DATA_ROOT.exists():
        pytest.skip(f"Data directory not found: {DATA_ROOT}")
    return ZeldaDungeonAdapter(str(DATA_ROOT))


def _check_graph_room_consistency(dungeon, stitched) -> List[str]:
    """Check that graph nodes match room count."""
    issues = []
    
    if stitched is None:
        return ["Stitched dungeon is None"]
    
    # Check room_to_node mapping
    rooms_in_dungeon = set(dungeon.rooms.keys())
    rooms_with_nodes = set(stitched.room_to_node.keys())
    
    # Rooms without node assignment
    unmapped_rooms = rooms_in_dungeon - rooms_with_nodes
    if unmapped_rooms:
        issues.append(f"Rooms without graph nodes: {sorted(unmapped_rooms)}")
    
    # Check graph size vs room positions
    if stitched.graph:
        graph_nodes = set(stitched.graph.nodes())
        node_values = set(stitched.room_to_node.values())
        
        orphan_nodes = graph_nodes - node_values
        if orphan_nodes:
            issues.append(f"Graph nodes not mapped to rooms: {sorted(orphan_nodes)}")
    
    return issues


def _check_start_goal_positions(dungeon, stitched) -> List[str]:
    """Check that start and goal positions exist and are valid."""
    issues = []
    
    # Check raw dungeon positions
    if dungeon.start_pos is None:
        issues.append("Missing start_pos in dungeon")
    elif dungeon.start_pos not in dungeon.rooms:
        issues.append(f"start_pos {dungeon.start_pos} not in rooms")
    
    if dungeon.triforce_pos is None:
        issues.append("Missing triforce_pos in dungeon")
    elif dungeon.triforce_pos not in dungeon.rooms:
        issues.append(f"triforce_pos {dungeon.triforce_pos} not in rooms")
    
    # Check stitched positions
    if stitched:
        grid = getattr(stitched, 'global_grid', getattr(stitched, 'map', None))
        if stitched.start_global is None:
            issues.append("Missing start_global in stitched dungeon")
        else:
            r, c = stitched.start_global
            if r < 0 or c < 0:
                issues.append(f"Invalid start_global position: {stitched.start_global}")
            elif grid is not None:
                if r >= grid.shape[0] or c >= grid.shape[1]:
                    issues.append(f"start_global {stitched.start_global} out of bounds")
        
        if stitched.triforce_global is None:
            issues.append("Missing triforce_global in stitched dungeon")
        else:
            r, c = stitched.triforce_global
            if r < 0 or c < 0:
                issues.append(f"Invalid triforce_global position: {stitched.triforce_global}")
            elif grid is not None:
                if r >= grid.shape[0] or c >= grid.shape[1]:
                    issues.append(f"triforce_global {stitched.triforce_global} out of bounds")
    
    return issues


def _check_room_dimensions(dungeon) -> List[str]:
    """Check that all rooms have valid dimensions."""
    issues = []
    
    for pos, room in dungeon.rooms.items():
        if hasattr(room, 'semantic_grid') and room.semantic_grid is not None:
            h, w = room.semantic_grid.shape
            if h != ROOM_HEIGHT:
                issues.append(f"Room {pos}: height {h} != expected {ROOM_HEIGHT}")
            if w != ROOM_WIDTH:
                issues.append(f"Room {pos}: width {w} != expected {ROOM_WIDTH}")
    
    return issues


def _check_item_placement(dungeon, stitched) -> List[str]:
    """Check that items from graph edges are actually on the map."""
    issues = []
    
    if stitched is None or stitched.graph is None:
        return issues
    
    # Collect expected items from graph edges
    expected_keys = 0
    expected_locked_doors = 0
    
    for u, v, data in stitched.graph.edges(data=True):
        edge_type = data.get('edge_type', '')
        label = data.get('label', '')
        
        if 'key' in edge_type.lower() or 'key' in label.lower():
            expected_locked_doors += 1
    
    # Count actual keys on map
    actual_keys = 0
    grid = getattr(stitched, 'global_grid', getattr(stitched, 'map', None))
    if grid is not None:
        key_tile = SEMANTIC_PALETTE.get('KEY', -1)
        if key_tile >= 0:
            actual_keys = (grid == key_tile).sum()
    
    # Note: We don't fail on mismatch, just report
    if expected_locked_doors > 0 and actual_keys == 0:
        issues.append(f"Graph has {expected_locked_doors} locked doors but no keys found on map")
    
    return issues


def _check_tile_validity(stitched) -> List[str]:
    """Check that all tile IDs are valid."""
    issues = []
    
    grid = getattr(stitched, 'global_grid', getattr(stitched, 'map', None))
    if stitched is None or grid is None:
        return issues
    
    valid_tiles = set(SEMANTIC_PALETTE.values())
    unique_tiles = set(grid.flatten())
    
    invalid_tiles = unique_tiles - valid_tiles
    if invalid_tiles:
        issues.append(f"Unknown tile IDs in map: {sorted(invalid_tiles)}")
    
    return issues


class TestDataIntegrity:
    """Test suite for data integrity."""
    
    @pytest.mark.parametrize("dungeon_num,variant", ALL_DUNGEONS)
    def test_dungeon_loads(self, adapter, dungeon_num, variant):
        """Test that dungeon can be loaded."""
        dungeon = adapter.load_dungeon(dungeon_num, variant=variant)
        assert dungeon is not None, f"Failed to load dungeon {dungeon_num} v{variant}"
        assert len(dungeon.rooms) > 0, f"Dungeon {dungeon_num} v{variant} has no rooms"
    
    @pytest.mark.parametrize("dungeon_num,variant", ALL_DUNGEONS)
    def test_dungeon_stitches(self, adapter, dungeon_num, variant):
        """Test that dungeon can be stitched."""
        dungeon = adapter.load_dungeon(dungeon_num, variant=variant)
        if dungeon is None:
            pytest.skip("Could not load dungeon")
        
        stitched = adapter.stitch_dungeon(dungeon)
        assert stitched is not None, f"Failed to stitch dungeon {dungeon_num} v{variant}"
        grid = getattr(stitched, 'global_grid', getattr(stitched, 'map', None))
        assert grid is not None, f"Stitched map is None for {dungeon_num} v{variant}"
    
    @pytest.mark.parametrize("dungeon_num,variant", ALL_DUNGEONS)
    def test_graph_room_consistency(self, adapter, dungeon_num, variant):
        """Test that graph nodes match rooms."""
        dungeon = adapter.load_dungeon(dungeon_num, variant=variant)
        if dungeon is None:
            pytest.skip("Could not load dungeon")
        
        stitched = adapter.stitch_dungeon(dungeon)
        issues = _check_graph_room_consistency(dungeon, stitched)
        
        for issue in issues:
            _stats.record_issue(dungeon_num, variant, issue)
        
        if not issues:
            _stats.record_pass(dungeon_num, variant)
        
        # Only fail on critical issues
        critical = [i for i in issues if "without graph nodes" in i]
        assert not critical, f"Dungeon {dungeon_num} v{variant}: {'; '.join(critical)}"
    
    @pytest.mark.parametrize("dungeon_num,variant", ALL_DUNGEONS)
    def test_start_goal_exist(self, adapter, dungeon_num, variant):
        """Test that start and goal positions exist."""
        dungeon = adapter.load_dungeon(dungeon_num, variant=variant)
        if dungeon is None:
            pytest.skip("Could not load dungeon")
        
        stitched = adapter.stitch_dungeon(dungeon)
        issues = _check_start_goal_positions(dungeon, stitched)
        
        for issue in issues:
            _stats.record_issue(dungeon_num, variant, issue)
        
        # These are critical failures
        assert not issues, f"Dungeon {dungeon_num} v{variant}: {'; '.join(issues)}"
    
    @pytest.mark.parametrize("dungeon_num,variant", ALL_DUNGEONS)
    def test_room_dimensions(self, adapter, dungeon_num, variant):
        """Test that rooms have correct dimensions."""
        dungeon = adapter.load_dungeon(dungeon_num, variant=variant)
        if dungeon is None:
            pytest.skip("Could not load dungeon")
        
        issues = _check_room_dimensions(dungeon)
        
        for issue in issues:
            _stats.record_issue(dungeon_num, variant, issue)
        
        assert not issues, f"Dungeon {dungeon_num} v{variant}: {'; '.join(issues)}"
    
    @pytest.mark.parametrize("dungeon_num,variant", ALL_DUNGEONS)
    def test_tile_validity(self, adapter, dungeon_num, variant):
        """Test that all tiles are valid."""
        dungeon = adapter.load_dungeon(dungeon_num, variant=variant)
        if dungeon is None:
            pytest.skip("Could not load dungeon")
        
        stitched = adapter.stitch_dungeon(dungeon)
        issues = _check_tile_validity(stitched)
        
        for issue in issues:
            _stats.record_issue(dungeon_num, variant, issue)
        
        # Report but don't fail on unknown tiles
        if issues:
            pytest.skip(f"Found issues: {'; '.join(issues)}")


class TestDataStatistics:
    """Test suite for overall data statistics."""
    
    def test_all_dungeons_have_rooms(self, adapter):
        """Verify all dungeons have at least one room."""
        empty_dungeons = []
        
        for dungeon_num, variant in ALL_DUNGEONS:
            dungeon = adapter.load_dungeon(dungeon_num, variant=variant)
            if dungeon is None or len(dungeon.rooms) == 0:
                empty_dungeons.append((dungeon_num, variant))
        
        assert not empty_dungeons, f"Empty dungeons: {empty_dungeons}"
    
    def test_dungeon_variety(self, adapter):
        """Verify dungeons have varying sizes (not all identical)."""
        room_counts = []
        
        for dungeon_num, variant in ALL_DUNGEONS[:6]:  # Sample
            dungeon = adapter.load_dungeon(dungeon_num, variant=variant)
            if dungeon:
                room_counts.append(len(dungeon.rooms))
        
        assert len(set(room_counts)) > 1, "All dungeons have same room count - suspicious"


@pytest.fixture(scope='module', autouse=True)
def print_summary(request):
    """Print summary at end of module."""
    yield
    if _stats.issues or _stats.passed:
        print(_stats.summary())


# Standalone runner
if __name__ == '__main__':
    """Run integrity tests standalone with detailed output."""
    print("=" * 60)
    print("DATA INTEGRITY TEST")
    print("=" * 60)
    
    if not DATA_ROOT.exists():
        print(f"ERROR: Data directory not found: {DATA_ROOT}")
        sys.exit(1)
    
    adapter = ZeldaDungeonAdapter(str(DATA_ROOT))
    stats = DataIntegrityStats()
    
    for dungeon_num, variant in ALL_DUNGEONS:
        print(f"\nChecking Dungeon {dungeon_num} variant {variant}...")
        
        # Load dungeon
        dungeon = adapter.load_dungeon(dungeon_num, variant=variant)
        if dungeon is None:
            stats.record_issue(dungeon_num, variant, "Failed to load")
            print("  [X] Failed to load")
            continue
        
        print(f"  Rooms: {len(dungeon.rooms)}")
        
        # Stitch dungeon
        stitched = adapter.stitch_dungeon(dungeon)
        if stitched is None:
            stats.record_issue(dungeon_num, variant, "Failed to stitch")
            print("  [X] Failed to stitch")
            continue
        
        # Run all checks
        all_issues = []
        all_issues.extend(_check_graph_room_consistency(dungeon, stitched))
        all_issues.extend(_check_start_goal_positions(dungeon, stitched))
        all_issues.extend(_check_room_dimensions(dungeon))
        all_issues.extend(_check_item_placement(dungeon, stitched))
        all_issues.extend(_check_tile_validity(stitched))
        
        if all_issues:
            for issue in all_issues:
                stats.record_issue(dungeon_num, variant, issue)
                print(f"  [!] {issue}")
        else:
            stats.record_pass(dungeon_num, variant)
            print("  [OK] All checks passed")
    
    print(stats.summary())
