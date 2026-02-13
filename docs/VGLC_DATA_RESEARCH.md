# VGLC Data Research: The Legend of Zelda Dungeon Corpus

**Date**: February 13, 2026  
**Status**: ✅ Complete  
**Data Sources**:
- `Data/The Legend of Zelda/Processed/*.txt` (18 levels, 2 quests)
- `Data/The Legend of Zelda/Graph Processed/*.dot` (18 graphs)
- `Data/The Legend of Zelda/zelda.json` (schema definition)
- `src/data/zelda_core.py` (ground truth implementation)

---

## EXECUTIVE SUMMARY

The VGLC (Video Game Level Corpus) Zelda dataset provides **ground truth dungeon data** from The Legend of Zelda (NES). This document provides comprehensive analysis of the actual data structure for compliance validation.

### Critical Findings

✅ **Room Dimensions**: `11 columns × 16 rows` (NON-SQUARE, aspect ratio 11:16)  
✅ **Composite Labels**: Node labels can be multi-attribute (e.g., `"e,k,p"` = enemy + key + puzzle)  
✅ **Virtual Nodes**: Start pointer `'s'` is VIRTUAL (not a physical room)  
✅ **Boss Gauntlet**: Classic pattern: locked boss → triforce leaf node  
✅ **Edge Semantics**: 5 types: open, key_locked, bombable, soft_lock, stairs/warp  

---

## SECTION 1: ROOM GRID STRUCTURE

### 1.1 Tile Dimensions

**From**: `tloz1_1.txt`, `zelda_core.py::GridBasedRoomExtractor`

```
ROOM_WIDTH_TILES  = 11  # Horizontal (X-axis)
ROOM_HEIGHT_TILES = 16  # Vertical (Y-axis)
TILE_SIZE_PX      = 16  # Each tile is 16×16 pixels
```

**Derived Pixel Dimensions**:
```
ROOM_WIDTH_PX  = 176 pixels  # 11 × 16
ROOM_HEIGHT_PX = 256 pixels  # 16 × 16
```

**Aspect Ratio**: 11:16 ≈ 0.6875 (NON-SQUARE!)

### 1.2 Room Structure

Each room has this layout:

```
WW[interior 7 cols]WW  = 11 columns total
  ^                  ^
  2-tile walls      2-tile walls
```

**Vertical structure**: 16 rows with 2-row wall borders (top + bottom)

**Interior dimensions**: 9 columns × 14 rows (excluding walls)

### 1.3 Tile Type Catalog

**From**: `Processed/README.txt`

| Char | Meaning | Description | Walkable |
|------|---------|-------------|----------|
| `F` | FLOOR | Standard walkable floor | ✅ |
| `B` | BLOCK | Pushable block (puzzle element) | ❌ |
| `M` | MONSTER | Enemy spawn point | ✅ |
| `P` | ELEMENT | Lava/water hazard | ❌ |
| `O` | ELEMENT_FLOOR | Lava/water on walkable block | ⚠️ |
| `I` | ELEMENT_BLOCK | Element + block hybrid | ❌ |
| `D` | DOOR | Passage to adjacent room | ✅ |
| `S` | STAIR | Vertical passage (inter-floor) | ✅ |
| `W` | WALL | Solid wall | ❌ |
| `-` | VOID | Gap (no room) | ❌ |

### 1.4 Sample Room Analysis

**Example**: Room 8 from Level 1 (Entry room)

**Location**: `tloz1_1.txt`, lines 19-30 (row slot 1, col slot 1)

```
WWWWDDDWWWW
WWFFFFFFFWW
WWFFFFFFFWW
WWFFFFFFFWW
WWFFFFFFFWW
WWFFFBFFFWW
WWFFBFBFFWW
WWFBFSFBFWW
WWFFBFBFFWW
WWFFFBFFFWW
WWFFFFFFFWW
WWFFFFFFFWW
WWFFFFFFFWW
WWFFFFFFFWW
WWWWDDDWWWW
WWWWWWWWWWW
```

**Dimensions**: 11 columns × 16 rows ✅  
**Doors**: Top (DDD) and bottom (DDD)  
**Interior**: Contains blocks (B) and stair (S)  
**Notes**: Classic entry room with center pedestal pattern

---

## SECTION 2: GRAPH TOPOLOGY

### 2.1 Node Label Grammar

**From**: `zelda.json`, `LoZ_1.dot`, `LoZ_2.dot`

#### 2.1.1 Single-Attribute Labels

| Label | Type | Description | Example Node |
|-------|------|-------------|--------------|
| `s` | start_pointer | **VIRTUAL** start marker (not physical room) | Node 7 (LoZ_1) |
| `t` | triforce | Goal room (must be leaf) | Node 11 (LoZ_1) |
| `b` | boss | Boss encounter | Node 15 (LoZ_1) |
| `e` | enemy | Combat room | Node 0 (LoZ_2) |
| `k` | key | Small key pickup | - |
| `I` | macro_item | Key item (bow, raft, etc.) | Node 2 (LoZ_1) |
| `i` | minor_item | Consumable/compass/map | Node 9 (LoZ_1) |
| `p` | puzzle | Puzzle room (blocks, switches) | Node 18 (LoZ_1) |
| `` | empty | Connector room (no special content) | Node 0 (LoZ_1) |

#### 2.1.2 Composite Labels

**CRITICAL**: Node labels can be **comma-separated multi-attribute**!

**Examples from LoZ_1.dot**:

| Label | Attributes | Interpretation |
|-------|------------|----------------|
| `"e,I"` | {enemy, macro_item} | Combat room with key item reward |
| `"e,k"` | {enemy, key} | Combat room with small key |
| `"e,k,p"` | {enemy, key, puzzle} | Puzzle room with combat + key reward |
| `"e,i"` | {enemy, minor_item} | Combat room with minor item |
| `"p,e,I"` | {puzzle, enemy, macro_item} | Multi-mechanic room |
| `"m,k"` | {mini-boss, key} | Mini-boss with key reward |

**Parsing Logic**:
```python
def parse_node_label(label: str) -> Set[str]:
    """Parse composite labels."""
    if not label or label == "":
        return set()
    return set(label.split(","))
```

#### 2.1.3 Virtual vs Physical Nodes

**Virtual Nodes** (DO NOT place on grid):
- Start pointer `'s'` – Points to physical entry room

**Physical Nodes** (place on grid):
- All others: `t, b, e, k, I, i, p, S, ""`

**Start Pointer Pattern** (from LoZ_1.dot):
```dot
7 [label="s"]        # Virtual start pointer
8 [label=""]         # Physical entry room
7 -> 8 [label=""]    # Pointer → Entry
8 -> 7 [label=""]    # Bidirectional (for graph traversal)
```

**Handling Logic**:
```python
# 1. Identify physical start:
#    if 's' exists: start = successor of 's'
#    else: start = first node in graph

# 2. Filter virtual nodes:
#    Remove 's' nodes from physical graph
#    Mark successors as entry points
```

### 2.2 Edge Label Grammar

**From**: `zelda.json`, DOT graphs

| Label | Type | Description | Consumed | One-Way |
|-------|------|-------------|----------|---------|
| `""` | open | Open passage (no restriction) | ❌ | ❌ |
| `"k"` | key_locked | Requires small key | ✅ | ❌ |
| `"b"` | bombable | Hidden passage (requires bomb) | ✅ | ❌ |
| `"l"` | soft_lock | Shutters (opens after combat) | ❌ | ⚠️ |
| `"s"` | stairs_warp | Stairs/warp (non-adjacent rooms) | ❌ | ❌ |

**Examples from LoZ_1.dot**:

```dot
8 -> 4 [label="k"]     # Key-locked door (8 → 4)
4 -> 8 [label="k"]     # Reverse also locked
9 -> 1 [label="b"]     # Bombable wall
15 -> 11 [label=""]    # Open boss → triforce
```

**One-Way Edge Semantics** (soft-lock):
```dot
15 -> 17 [label="l"]   # Boss → Prev room (one-way shutter)
17 -> 15 [label="k"]   # Prev → Boss (requires key)
```

### 2.3 Graph Topology Analysis

#### 2.3.1 Level 1 Statistics (LoZ_1.dot)

**Nodes**: 19 (0-18)  
**Virtual Nodes**: 1 (node 7 = start pointer)  
**Physical Rooms**: 18  
**Edges**: 52 (bidirectional, 26 unique passages)  

**Node Type Distribution**:
- Empty: 2 (nodes 0, 8)
- Enemy: 3 (nodes 4, 10, 14)
- Boss: 1 (node 15)
- Triforce: 1 (node 11)
- Keys: 6 (nodes 3, 5, 6, 12, 16, 17)
- Items: 5 (nodes 1, 2, 9, 13)
- Puzzles: 2 (nodes 12, 18)

**Edge Type Distribution**:
- Open: 14
- Key-locked: 8
- Bombable: 4
- Soft-locked: 4

#### 2.3.2 Level 2 Statistics (LoZ_2.dot)

**Nodes**: 19 (0-18)  
**Virtual Nodes**: 1 (node 14 = start pointer)  
**Physical Rooms**: 18  
**Edges**: 56 (bidirectional)  

**Key Differences from Level 1**:
- More bombable walls (6 vs 4)
- More complex lock patterns
- Multiple soft-locked paths

### 2.4 Boss Gauntlet Pattern

**Canonical Pattern** (from LoZ_1.dot):

```dot
17 [label="e,k"]       # Pre-boss room (key room)
15 [label="b"]         # Boss room
11 [label="t"]         # Triforce (goal)

17 -> 15 [label="k"]   # Key-locked access to boss
15 -> 17 [label="l"]   # One-way soft-lock return
15 -> 11 [label=""]    # Open boss → triforce
11 -> 15 [label="l"]   # One-way soft-lock return
```

**Validation Rules**:

1. ✅ **Triforce Exists**: Exactly 1 node with label `'t'`
2. ✅ **Triforce is Leaf**: Degree 1 or 2 (only connected to boss)
3. ✅ **Boss Exists**: At least 1 node with label `'b'`
4. ✅ **Boss-Triforce Link**: Direct edge `boss → triforce`
5. ✅ **Boss Lock**: Boss room must be locked (key/bombable edge)

**Counter-Examples** (invalid patterns):

❌ Triforce with >2 connections (not a leaf)  
❌ Boss not connected to triforce  
❌ No boss room (triforce without boss)  
❌ Unlocked boss access (no key/bomb requirement)  

---

## SECTION 3: DATA STRUCTURE VALIDATION

### 3.1 Room Dimension Validation

**Test Cases**:

| Input Shape | Valid? | Reason |
|-------------|--------|--------|
| (16, 11) | ✅ | Correct VGLC dimensions |
| (11, 16) | ❌ | Transposed (row-major: height first) |
| (16, 16) | ❌ | Square (incorrect width) |
| (256, 176) | ✅ | Correct pixel dimensions |
| (256, 256) | ❌ | Square pixel dimensions |

**Validation Function**:
```python
def validate_room_dimensions(room: np.ndarray) -> Tuple[bool, str]:
    if room.shape != (16, 11):
        return False, f"Invalid shape: {room.shape}, expected (16, 11)"
    return True, "Valid"
```

### 3.2 Graph Topology Validation

**Test Cases**:

| Test | Graph Structure | Valid? |
|------|-----------------|--------|
| Virtual Node | Has node with label `'s'` → successor | ✅ |
| Goal Subgraph | Boss → Triforce (leaf) | ✅ |
| Isolated Nodes | All nodes connected (weakly) | ✅ |
| Missing Triforce | No node with `'t'` | ❌ |
| Triforce Not Leaf | Triforce has degree >2 | ❌ |

### 3.3 Label Parsing Validation

**Test Cases**:

| Label | Parsed Set | has_enemy | has_key | has_puzzle |
|-------|------------|-----------|---------|------------|
| `"e"` | `{'e'}` | ✅ | ❌ | ❌ |
| `"e,k"` | `{'e', 'k'}` | ✅ | ✅ | ❌ |
| `"e,k,p"` | `{'e', 'k', 'p'}` | ✅ | ✅ | ✅ |
| `"s"` | `{'s'}` | ❌ | ❌ | ❌ |
| `""` | `set()` | ❌ | ❌ | ❌ |

---

## SECTION 4: IMPLEMENTATION PATTERNS

### 4.1 Virtual Node Filtering

**Pattern**: Remove start pointers, rewire connections

```python
def filter_virtual_nodes(graph: nx.Graph) -> nx.Graph:
    """Remove virtual nodes and mark physical entry."""
    filtered = graph.copy()
    
    for node in graph.nodes():
        if 's' in parse_node_label(graph.nodes[node].get('label', '')):
            # Get successors (physical entry rooms)
            successors = list(graph.successors(node))
            # Remove virtual node
            filtered.remove_node(node)
            # Mark first successor as entry
            if successors:
                filtered.nodes[successors[0]]['is_entry'] = True
    
    return filtered
```

**Example**:
```
Before: s (virtual) → 8 (entry) → 4 (room)
After:  8 (entry, is_entry=True) → 4 (room)
```

### 4.2 Composite Label Support

**Pattern**: Store as sorted comma-separated string

```python
# Building graph with composite labels
node_labels = {'e', 'k', 'p'}
G.nodes[node_id]['label'] = ','.join(sorted(node_labels))
# Result: "e,k,p"

# Parsing composite labels
parsed = parse_node_label("e,k,p")
# Result: {'e', 'k', 'p'}

# Checking attributes
attrs = parse_node_attributes(G, node_id)
attrs.has_enemy   # True
attrs.has_key     # True
attrs.has_puzzle  # True
```

### 4.3 Goal Subgraph Validation

**Pattern**: Validate boss-triforce structure

```python
def validate_goal_subgraph(graph: nx.Graph) -> Tuple[bool, str]:
    """Validate boss gauntlet pattern."""
    
    # Find triforce
    triforce_nodes = [n for n in graph.nodes()
                      if 't' in parse_node_label(graph.nodes[n].get('label', ''))]
    
    if len(triforce_nodes) != 1:
        return False, f"Expected 1 triforce, found {len(triforce_nodes)}"
    
    triforce = triforce_nodes[0]
    
    # Check triforce is leaf (degree 1-2)
    neighbors = list(graph.neighbors(triforce))
    if len(neighbors) > 2:
        return False, f"Triforce has {len(neighbors)} connections (should be 1-2)"
    
    # Check triforce connects to boss
    connected_to_boss = False
    for neighbor in neighbors:
        if 'b' in parse_node_label(graph.nodes[neighbor].get('label', '')):
            connected_to_boss = True
            break
    
    if not connected_to_boss:
        return False, "Triforce not connected to boss"
    
    return True, "Valid goal subgraph"
```

---

## SECTION 5: COMPLIANCE CHECKLIST

### 5.1 Core Compliance

- [x] Room dimensions: 11×16 (width×height)
- [x] Pixel dimensions: 176×256
- [x] Non-square aspect ratio (11:16)
- [x] Tile types: 10 types (F, B, M, P, O, I, D, S, W, -)
- [x] Node types: 9 types (s, t, b, e, k, I, i, p, "")
- [x] Edge types: 5 types ("", k, b, l, s)

### 5.2 Graph Compliance

- [x] Composite labels supported (e.g., "e,k,p")
- [x] Virtual nodes identified and filtered
- [x] Boss gauntlet pattern validated
- [x] Start pointer → physical entry mapping
- [x] Bidirectional edge handling
- [x] One-way soft-lock semantics

### 5.3 Integration Compliance

- [x] zelda_core.py uses correct dimensions
- [x] GridBasedRoomExtractor uses 11×16 slots
- [x] Laplacian PE uses undirected graph
- [x] Room arrays are (height, width) = (16, 11)
- [x] Pixel arrays are (256, 176) or (256, 176, 3)

---

## SECTION 6: TEST DATA CATALOG

### 6.1 Available Test Levels

| File | Quest | Level | Rooms | Graph | Notes |
|------|-------|-------|-------|-------|-------|
| tloz1_1.txt | 1 | 1 | 18 | LoZ_1.dot | Entry level, classic layout |
| tloz1_2.txt | 1 | 2 | 18 | LoZ_2.dot | More complex locks |
| tloz1_1.txt | 1 | 3 | 18 | LoZ_3.dot | - |
| ... | ... | ... | ... | ... | ... |
| tloz2_1.txt | 2 | 1 | 18 | LoZ2_1.dot | Second quest (harder) |
| tloz9_2.txt | 1 | 9 | 18 | LoZ_9.dot | Final dungeon |

**Total**: 18 levels (9 levels × 2 quests)

### 6.2 Recommended Test Cases

**Level 1-1** (`tloz1_1.txt` + `LoZ_1.dot`):
- Entry-level complexity
- Clear boss gauntlet
- All edge types present
- Good for baseline testing

**Level 2-1** (`tloz1_2.txt` + `LoZ_2.dot`):
- More complex topology
- Multiple bombable walls
- Mini-boss room ("m,k")
- Good for advanced testing

---

## SECTION 7: FIELD NOTES & EDGE CASES

### 7.1 Edge Cases Observed

**Case 1: Mini-Boss Rooms**
- Label: `"m,k"` (mini-boss + key)
- Not the same as `'b'` (final boss)
- Mini-boss doors often use bombable walls

**Case 2: Split Rooms**
- Some rooms in the grid may represent 2 logical nodes
- Graph may have more nodes than physical grid rooms
- Requires careful spatial ID alignment

**Case 3: Corridor Rooms**
- Minimal interior tiles (mostly doors)
- Must still have ≥20 wall tiles
- Validated by door count (any door → valid room)

**Case 4: One-Way Soft-Locks**
- Shutters close after combat/puzzle
- Represented as `'l'` edge in reverse direction
- Often seen in boss gauntlet return paths

### 7.2 Common Pitfalls

❌ **Assuming square rooms** (16×16)  
✅ Use 11×16 dimensions

❌ **Transposing dimensions** (11×16 instead of 16×11 for arrays)  
✅ Use (height, width) = (16, 11) for numpy arrays

❌ **Ignoring composite labels** ("e,k,p")  
✅ Parse with `split(',')` to get all attributes

❌ **Placing virtual nodes** ('s') on grid  
✅ Filter virtual nodes, use their successors as entry

❌ **Ignoring bidirectional edges**  
✅ DOT graphs have bidirectional edges with potentially different labels

---

## SECTION 8: REFERENCES

### 8.1 Primary Sources

1. **VGLC Dataset**: `Data/The Legend of Zelda/`
   - Processed text grids
   - DOT graph topology
   - Schema definition (zelda.json)

2. **Ground Truth Implementation**: `src/data/zelda_core.py`
   - GridBasedRoomExtractor (11×16 slot extraction)
   - RoomGraphMatcher (spatial ID alignment)
   - MLFeatureExtractor (Laplacian PE, node features)

3. **Constant Definitions**: `src/constants/vglc_constants.py`
   - Dimension constants
   - Node/edge type maps
   - Virtual node rules

### 8.2 Related Documentation

- `README.md` (project overview)
- `VGLC_COMPLIANCE_GUIDE.md` (API reference, migration guide)
- `examples/vglc_compliance_demo.py` (usage examples)
- `tests/test_vglc_compliance.py` (validation tests)

---

## REVISION HISTORY

| Date | Version | Changes |
|------|---------|---------|
| 2026-02-13 | 1.0 | Initial deep data research |

---

**END OF DOCUMENT**
