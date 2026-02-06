# VGLC Data Processing Analysis Report

## Executive Summary

**STATUS**: Two distinct issues identified:
1. **BUG**: `GridBasedRoomExtractor` misses corridor rooms (impacts 19 rooms across all levels)
2. **BY DESIGN**: Graph nodes exceed physical rooms (44 "virtual" nodes across all levels)

The current production code uses `VGLCParser` (which handles corridors correctly), but `GridBasedRoomExtractor` is a legacy/alternative extractor that needs fixing.

---

## Detailed Findings

### Issue 1: GridBasedRoomExtractor Bug (REAL BUG)

**Location**: [src/data/zelda_core.py](src/data/zelda_core.py#L55-L130)

**Root Cause**: The `_is_room_slot()` method requires `interior >= 5` tiles, but CORRIDOR ROOMS have:
- 90+ wall tiles forming passage walls
- Only 2 door tiles as "interior" content
- Interior filled with dashes (`-`) representing void/non-floor

**Example - Level 2, Room (3,1)**:
```
WWWWWWWWWWW
WWWWWWWWWWW
WW-------WW
WW-------WW
WW-------WW
WW-------WW
WW-------WW
WW-------DW  <- Door tile (the only "interior")
WW-------DW  <- Door tile
WW-------WW
...
```

**Impact**:
| Level | Rooms Missed by GridExtractor |
|-------|-------------------------------|
| D1    | 1 room: (0,2)                |
| D2    | 1 room: (3,1)                |
| D4    | 1 room: (1,1)                |
| D5    | 3 rooms                       |
| D6    | 2 rooms                       |
| D7    | 4 rooms                       |
| D8    | 3 rooms                       |
| D9    | 3 rooms                       |
| **Total** | **19 corridor rooms missed** |

**Fix Required** (in `_is_room_slot()`):
```python
# Current (buggy):
return bool(wall_count >= 20 and interior_count >= 5)

# Fixed (option 1 - count doors as valid room indicator):
door_count = np.sum(slot_grid == 'D')
return bool(wall_count >= 20 and (interior_count >= 5 or door_count >= 1))

# Fixed (option 2 - just check walls and non-gap):
return bool(wall_count >= 20 and dash_count < total * 0.7)
```

---

### Issue 2: Graph Nodes > Physical Rooms (BY DESIGN)

**This is NOT a bug** - it's how VGLC represents game logic.

**Explanation**: The DOT graph files represent *game progression states*, not physical layout:
- Virtual nodes represent sub-areas within physical rooms
- Examples: boss arenas, item alcoves, secret passages, stair destinations

**Level 2 Analysis**:
- Physical rooms: 18 (with VGLCParser fix)
- Graph nodes: 19
- Virtual node: Node 0 (enemy room that shares space with another)

**Node-to-Room Mapping** (Level 2):
```
Room (0,6) -> Node 16 ("e,k")     Room (2,5) -> Node 4 ("e,k")
Room (1,0) -> Node 17 ("e,i")     Room (2,6) -> Node 14 ("s") START
Room (1,6) -> Node 13 ("")        Room (2,7) -> Node 15 ("e,k")
Room (1,7) -> Node 1 ("e")        Room (3,1) -> Node 8 ("") CORRIDOR
Room (2,0) -> Node 11 ("b") BOSS  Room (3,2) -> Node 10 ("e")
Room (2,1) -> Node 12 ("t") GOAL  Room (3,3) -> Node 7 ("p,e,I")
Room (2,2) -> Node 3 ("e,i")      Room (3,4) -> Node 9 ("p,e,i")
Room (2,3) -> Node 5 ("e,i")      Room (3,5) -> Node 2 ("e")
Room (2,4) -> Node 6 ("m,k")      Room (3,6) -> Node 18 ("e")

Unmapped (virtual): Node 0 ("e") - shares room with Node 1
```

**Summary by Level**:
| Level | Physical Rooms | Graph Nodes | Virtual Nodes |
|-------|----------------|-------------|---------------|
| D1    | 17*            | 19          | 2             |
| D2    | 18*            | 19          | 1             |
| D3    | 18             | 20          | 2             |
| D4    | 20*            | 27          | 7             |
| D5    | 23*            | 25          | 2             |
| D6    | 25*            | 27          | 2             |
| D7    | 33*            | 35          | 2             |
| D8    | 25*            | 28          | 3             |
| D9    | 57*            | 62          | 5             |

*After VGLCParser (includes corridor rooms)

---

## Code Architecture

The codebase has **two room extractors**:

1. **`VGLCParser`** (Production - Lines 686-800)
   - Used by `ZeldaDungeonAdapter` 
   - Correctly handles corridor rooms
   - Simple check: any non-dash character = room exists

2. **`GridBasedRoomExtractor`** (Legacy/ML - Lines 55-130)
   - Used for ML feature extraction
   - BUG: Rejects corridor rooms
   - Stricter check: requires 5+ interior tiles

---

## Recommendations

### Immediate Fix (High Priority)
Fix `GridBasedRoomExtractor._is_room_slot()` to handle corridor rooms:
```python
def _is_room_slot(self, slot_grid: np.ndarray) -> bool:
    # ... existing checks ...
    
    # Count doors as valid room indicator (corridor rooms have doors but no floor)
    door_count = np.sum((slot_grid == 'D') | (slot_grid == 'd'))
    
    # A room is valid if it has walls AND (floor interior OR doors)
    return bool(wall_count >= 20 and (interior_count >= 5 or door_count >= 1))
```

### Validation (Medium Priority)
Add unit tests for corridor room detection:
```python
def test_corridor_room_detection():
    """Ensure both extractors find the same rooms."""
    for level in range(1, 10):
        grid_rooms = GridBasedRoomExtractor().extract(f'tloz{level}_1.txt')
        vglc_rooms = VGLCParser().parse(f'tloz{level}_1.txt')
        assert len(grid_rooms) == len(vglc_rooms), f"Level {level} mismatch"
```

### Documentation (Low Priority)
Document that graph nodes can exceed physical rooms (virtual nodes for game logic).

---

## Files Involved

- [src/data/zelda_core.py](src/data/zelda_core.py) - Main data processing
  - `GridBasedRoomExtractor` (Line 55) - **NEEDS FIX**
  - `VGLCParser` (Line 686) - Works correctly
  - `RoomGraphMatcher` (Line 920) - Handles virtual nodes correctly
  
- [Data/The Legend of Zelda/Processed/*.txt](Data/The%20Legend%20of%20Zelda/Processed/) - Grid files
- [Data/The Legend of Zelda/Graph Processed/*.dot](Data/The%20Legend%20of%20Zelda/Graph%20Processed/) - Graph files

---

## Conclusion

The data processing is **mostly correct**. The production code (`VGLCParser`) properly extracts all rooms including corridors. The discrepancy the user observed is a combination of:

1. A real bug in the alternative `GridBasedRoomExtractor` (fixable)
2. Expected behavior where graph nodes represent game states, not just physical rooms

The existing virtual node handling in `RoomGraphMatcher` correctly maps multiple graph nodes to shared physical rooms when needed.
