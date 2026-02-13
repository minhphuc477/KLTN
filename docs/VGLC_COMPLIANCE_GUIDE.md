# VGLC Compliance Guide

**Version**: 1.0  
**Date**: February 13, 2026  
**Status**: Complete  

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Quick Start](#2-quick-start)
3. [Core Concepts](#3-core-concepts)
4. [API Reference](#4-api-reference)
5. [Usage Examples](#5-usage-examples)
6. [Migration Guide](#6-migration-guide)
7. [Best Practices](#7-best-practices)
8. [Troubleshooting](#8-troubleshooting)
9. [Appendix](#9-appendix)

---

## 1. Introduction

### 1.1 What is VGLC?

VGLC (Video Game Level Corpus) is a dataset of video game levels represented as:
- **Grid data**: 2D tile arrays (text files)
- **Graph topology**: Mission graphs (DOT files)
- **Schema**: Node/edge type definitions (JSON)

This guide focuses on **The Legend of Zelda** dungeon data from VGLC.

### 1.2 Purpose of This Guide

This guide provides:
- Complete API documentation for `src/data/vglc_utils.py`
- Usage examples for all compliance features
- Migration guide for existing code
- Best practices for VGLC-compliant generation

### 1.3 Who Should Use This Guide

- **Developers** integrating VGLC data into AI systems
- **Researchers** working with procedural content generation
- **Engineers** building dungeon generation pipelines
- **Data Scientists** analyzing VGLC datasets

---

## 2. Quick Start

### 2.1 Installation

No additional installation required if you have the base project dependencies.

**Required imports**:
```python
from src.data.vglc_utils import (
    parse_node_label,
    filter_virtual_nodes,
    validate_topology,
    validate_room_dimensions,
)
from src.constants.vglc_constants import ROOM_SHAPE
```

### 2.2 Basic Example

```python
import networkx as nx
import numpy as np

# Create a simple VGLC dungeon
G = nx.DiGraph()
G.add_node(0, label="s")       # Virtual start pointer
G.add_node(1, label="e,k")     # Enemy + key
G.add_node(2, label="b")       # Boss
G.add_node(3, label="t")       # Triforce

G.add_edge(0, 1)
G.add_edge(1, 2, label="k")    # Key-locked
G.add_edge(2, 3)

# Filter virtual nodes
G_physical = filter_virtual_nodes(G)

# Validate
report = validate_topology(G)
print(f"Valid: {report.is_valid}")

# Create room with correct dimensions
room = np.zeros(ROOM_SHAPE, dtype=int)  # (16, 11)
valid, msg = validate_room_dimensions(room)
print(f"Room valid: {valid}")
```

### 2.3 Running the Demo

```bash
cd c:\\Users\\MPhuc\\Desktop\\KLTN
python examples/vglc_compliance_demo.py
```

### 2.4 Running Tests

```bash
pytest tests/test_vglc_compliance.py -v
```

---

## 3. Core Concepts

### 3.1 VGLC Room Dimensions

**CRITICAL**: Rooms are **NON-SQUARE**!

```python
ROOM_WIDTH_TILES = 11   # Horizontal
ROOM_HEIGHT_TILES = 16  # Vertical
TILE_SIZE_PX = 16

# Derived dimensions
ROOM_WIDTH_PX = 176     # 11 * 16
ROOM_HEIGHT_PX = 256    # 16 * 16

# Numpy shape (row-major: height first)
ROOM_SHAPE = (16, 11)   # Use this for np.zeros(ROOM_SHAPE)
```

**Aspect Ratio**: 11:16 ≈ 0.6875 (taller than wide)

### 3.2 Composite Node Labels

Node labels can contain **multiple attributes** separated by commas:

| Label | Attributes | Meaning |
|-------|------------|---------|
| `"e"` | `{enemy}` | Enemy room |
| `"e,k"` | `{enemy, key}` | Enemy room with key |
| `"e,k,p"` | `{enemy, key, puzzle}` | Enemy + key + puzzle |
| `"p,e,I"` | `{puzzle, enemy, macro_item}` | Multi-mechanic room |

**Parsing**:
```python
label = "e,k,p"
parsed = parse_node_label(label)
# Result: {'e', 'k', 'p'}
```

### 3.3 Virtual Nodes

**Virtual nodes** are meta-nodes used for graph structure but **not placed on the physical grid**.

**Example**: Start pointer `'s'`

```
Original Graph:
  s (virtual) → 8 (entry room) → 9 (room) → ...

Physical Graph (after filtering):
  8 (entry, is_entry=True) → 9 → ...
```

**Key Operations**:
```python
# Detect virtual nodes
virtual_nodes = get_virtual_nodes(graph)

# Filter virtual nodes
physical_graph = filter_virtual_nodes(graph)

# Get physical start
start_node = get_physical_start_node(graph)
```

### 3.4 Boss Gauntlet Pattern

Classic Zelda dungeon pattern:

```
... → Pre-Boss Room → [KEY_LOCK] → Boss Room → Triforce (leaf)
```

**Rules**:
1. Triforce node (`'t'`) must exist
2. Triforce must be a **leaf** (no outgoing edges in directed graphs, degree ≤2 in undirected)
3. Triforce connects to Boss node (`'b'`)
4. Boss exists

**Validation**:
```python
valid, msg = validate_goal_subgraph(graph)
if not valid:
    print(f"Goal subgraph invalid: {msg}")
```

### 3.5 Edge Types

| Label | Type | Description | Consumes Resource |
|-------|------|-------------|-------------------|
| `""` | open | Open passage | ❌ |
| `"k"` | key_locked | Requires key | ✅ |
| `"b"` | bombable | Hidden passage | ✅ |
| `"l"` | soft_lock | One-way shutters | ❌ |
| `"s"` | stairs_warp | Non-adjacent connection | ❌ |

---

## 4. API Reference

### 4.1 Graph Parsing Functions

All functions are in `src.data.vglc_utils`.

#### `parse_node_label(label: str) -> Set[str]`

Parse composite node labels.

```python
parsed = parse_node_label("e,k,p")
# Result: {'e', 'k', 'p'}
```

#### `parse_node_attributes(graph, node_id) -> NodeAttributes`

Extract node attributes with boolean flags.

```python
attrs = parse_node_attributes(G, 1)
print(attrs.has_enemy)   # True
print(attrs.has_key)     # True
```

#### `parse_edge_attributes(graph, source, target) -> EdgeAttributes`

Extract edge attributes with boolean flags.

```python
edge_attrs = parse_edge_attributes(G, 1, 2)
print(edge_attrs.is_key_locked)      # True
print(edge_attrs.consumes_resource)  # True
```

### 4.2 Virtual Node Functions

#### `is_virtual_node(graph, node_id) -> bool`

Check if node is virtual (start pointer).

####  `get_virtual_nodes(graph) -> List[int]`

Get all virtual node IDs.

#### `filter_virtual_nodes(graph) -> nx.Graph`

Remove virtual nodes and rewire connections.

#### `get_physical_start_node(graph) -> Optional[int]`

Get physical start room (successor of `'s'` pointer or marked entry).

### 4.3 Validation Functions

#### `validate_goal_subgraph(graph) -> Tuple[bool, str]`

Validate boss gauntlet pattern.

```python
valid, msg = validate_goal_subgraph(G)
if not valid:
    print(f"Invalid: {msg}")
```

#### `validate_topology(graph) -> TopologyReport`

Comprehensive graph validation with detailed report.

```python
report = validate_topology(G)
print(f"Valid: {report.is_valid}")
print(f"Boss: {report.num_boss}, Triforce: {report.num_triforce}")

if not report.is_valid:
    print(report.summary())  # Full diagnostic report
```

#### `validate_room_dimensions(room_array) -> Tuple[bool, str]`

Validate room dimensions (16×11).

```python
room = np.zeros((16, 11))
valid, msg = validate_room_dimensions(room)
assert valid
```

#### `validate_pixel_dimensions(image_array) -> Tuple[bool, str]`

Validate pixel dimensions (256×176).

```python
image = np.zeros((256, 176, 3), dtype=np.uint8)
valid, msg = validate_pixel_dimensions(image)
assert valid
```

### 4.4 Utility Functions

#### `analyze_vglc_graph(graph, verbose=True) -> TopologyReport`

Analyze and optionally print validation report.

#### `convert_to_physical_graph(graph, validate=True) -> Tuple[nx.Graph, int]`

Convert to physical graph and get start node.

---

## 5. Usage Examples

### 5.1 Creating VGLC-Compliant Graphs

```python
import networkx as nx

# Create dungeon
G = nx.DiGraph()
G.add_node(0, label="s")       # Virtual start
G.add_node(1, label="e,k")     # Entry + key
G.add_node(2, label="b")       # Boss
G.add_node(3, label="t")       # Triforce

G.add_edge(0, 1, label="")
G.add_edge(1, 2, label="k")
G.add_edge(2, 3, label="l")

# Validate
report = validate_topology(G)
if report.is_valid:
    print("✅ Valid VGLC dungeon!")
```

### 5.2 Filtering Virtual Nodes

```python
# Before filtering
start = get_physical_start_node(G)
print(f"Physical start: {start}")

# Filter
G_physical = filter_virtual_nodes(G)
print(f"Physical nodes: {G_physical.number_of_nodes()}")
```

### 5.3 Generating Rooms

```python
import numpy as np
from src.constants.vglc_constants import ROOM_SHAPE

# Create room with correct dimensions
room = np.zeros(ROOM_SHAPE, dtype=int)  # (16, 11)

# Add walls
room[0, :] = 1   # Top
room[-1, :] = 1  # Bottom
room[:, 0] = 1   # Left 
room[:, -1] = 1  # Right

# Validate
valid, msg = validate_room_dimensions(room)
assert valid
```

---

## 6. Migration Guide

### 6.1 Update Room Dimensions

**Before**:
```python
ROOM_SIZE = 16  # ❌ WRONG - assumes square
room = np.zeros((ROOM_SIZE, ROOM_SIZE))
```

**After**:
```python
from src.constants.vglc_constants import ROOM_SHAPE
room = np.zeros(ROOM_SHAPE, dtype=int)  # ✅ (16, 11)
```

### 6.2 Update Label Parsing

**Before**:
```python
# ❌ Old - single attribute only
label = graph.nodes[node_id]['label']
is_enemy = (label == 'e')
```

**After**:
```python
# ✅ New - handles composite labels
attrs = parse_node_attributes(graph, node_id)
is_enemy = attrs.has_enemy  # Works for "e", "e,k", "e,k,p"
```

### 6.3 Update Virtual Node Handling

**Before**:
```python
# ❌ Old - ignored virtual nodes
start = 0  # Hardcoded
```

**After**:
```python
# ✅ New - proper detection
start = get_physical_start_node(graph)
graph_physical = filter_virtual_nodes(graph)
```

---

## 7. Best Practices

### 7.1 Always Validate Dimensions

```python
# After generating any room
room = generate_room(...)
valid, msg = validate_room_dimensions(room)
if not valid:
    raise ValueError(f"Invalid room: {msg}")
```

### 7.2 Filter Virtual Nodes Early

```python
def process_dungeon(mission_graph):
    # Filter FIRST
    physical_graph = filter_virtual_nodes(mission_graph)
    start_node = get_physical_start_node(mission_graph)
    
    # Then proceed
    rooms = generate_rooms(physical_graph, start_node)
    return rooms
```

### 7.3 Use TopologyReport for Debugging

```python
report = validate_topology(graph)
if not report.is_valid:
    print(report.summary())  # Full diagnostic
    
    # Fix specific issues
    if not report.has_start:
        print("ERROR: No start node!")
    if not report.goal_subgraph_valid:
        print(f"ERROR: {report.goal_subgraph_message}")
```

---

## 8. Troubleshooting

### 8.1 "Wrong dimensions" Error

**Solution**: Use `ROOM_SHAPE` constant
```python
# ✅ Correct
from src.constants.vglc_constants import ROOM_SHAPE
room = np.zeros(ROOM_SHAPE)  # (16, 11)
```

### 8.2 "No physical start node found"

**Solutions**:
1. Add virtual start pointer:
```python
G.add_node(0, label="s")
G.add_edge(0, 1)  # Point to entry
```

2. Or mark entry explicitly:
```python
G.nodes[1]['is_entry'] = True
```

### 8.3 "Virtual nodes present" Error

**Solution**: Filter before validation
```python
# ✅ Correct
physical_graph = filter_virtual_nodes(graph)
report = validate_topology(physical_graph)
```

### 8.4 "Triforce not connected to boss"

**Solution**: Ensure boss→triforce edge
```python
G.add_node(4, label="b")  # Boss
G.add_node(5, label="t")  # Triforce
G.add_edge(4, 5)          # Connect!
```

---

## 9. Appendix

### 9.1 Node Type Reference

| Code | Type | Virtual |
|------|------|---------|
| `s` | start_pointer | ✅ |
| `t` | triforce | ❌ |
| `b` | boss | ❌ |
| `e` | enemy | ❌ |
| `k` | key | ❌ |
| `I` | macro_item | ❌ |
| `i` | minor_item | ❌ |
| `p` | puzzle | ❌ |

### 9.2 Edge Type Reference

| Code | Type | Consumes |
|------|------|----------|
| `` | open | ❌ |
| `k` | key_locked | ✅ |
| `b` | bombable | ✅ |
| `l` | soft_lock | ❌ |
| `s` | stairs_warp | ❌ |

### 9.3 Related Files

- **Implementation**: [src/data/vglc_utils.py](../src/data/vglc_utils.py)
- **Constants**: [src/constants/vglc_constants.py](../src/constants/vglc_constants.py)
- **Tests**: [tests/test_vglc_compliance.py](../tests/test_vglc_compliance.py)
- **Demo**: [examples/vglc_compliance_demo.py](../examples/vglc_compliance_demo.py)
- **Research**: [VGLC_DATA_RESEARCH.md](./VGLC_DATA_RESEARCH.md)

---

**END OF GUIDE**

*For questions, see project documentation or open GitHub issue.*
