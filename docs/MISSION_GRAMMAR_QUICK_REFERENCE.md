# Quick Reference: Mission Grammar Production Rules

## New Production Rules (User's Request)

### ✅ Implemented Rules

| # | Rule Name | Status | Lines | Weight | Purpose |
|---|-----------|--------|-------|--------|---------|
| A | **MergeRule** | ✅ Existing | 1068-1144 | 0.5 | Create shortcuts/cycles between distant nodes |
| B | **AddSwitchRule** | ✅ Existing (InsertSwitchRule) | 1147-1194 | 0.4 | Add switches controlling ON/OFF gates |
| C | **AddItemGateRule** | ✅ NEW | 1255-1343 | 0.4 | Item-based progression (BOMB, HOOKSHOT, etc.) |
| D | **AddBossGauntletRule** | ✅ Existing (AddBossGauntlet) | 1197-1252 | 1.0 | BIG_KEY → BOSS_DOOR → GOAL hierarchy |
| E | **CreateHubRule** | ✅ NEW | 1346-1412 | 0.3 | Central hub with 4 connections |
| F | **AddStairsRule** | ✅ NEW | 1415-1485 | 0.25 | Multi-floor support with stairs |
| G | **AddSecretRule** | ✅ NEW | 1488-1548 | 0.35 | Hidden rooms with rewards |
| H | **AddTeleportRule** | ✅ NEW | 1551-1608 | 0.2 | Warp connections between distant nodes |
| I | **PruneGraphRule** | ✅ NEW | 1611-1707 | 0.15 | Cleanup redundant EMPTY node chains |

## API Quick Reference

### New Enums

```python
# NodeType additions
NodeType.BOSS           # Boss encounter room
NodeType.STAIRS_UP      # Stairs ascending
NodeType.STAIRS_DOWN    # Stairs descending
NodeType.SECRET         # Secret/hidden room

# EdgeType additions
EdgeType.ITEM_GATE      # Requires specific item
EdgeType.STATE_BLOCK    # Switch-controlled gate
EdgeType.WARP           # Teleportation
EdgeType.STAIRS         # Vertical connection
```

### New Node Fields

```python
MissionNode(
    position=(x, y, z),        # Now 3D (z = floor)
    required_item="BOMB",      # For ITEM_GATE edges
    item_type="HOOKSHOT",      # For ITEM nodes
    switch_id=42,              # For switch control
    is_hub=True,               # Hub marker
    is_secret=True,            # Secret marker
)
```

### New Edge Fields

```python
MissionEdge(
    item_required="BOMB",      # Item gate requirement
    switch_id=42,              # Switch controlling this edge
    metadata={...},            # Additional properties
)
```

### New Helper Methods

```python
# Distance and connectivity
graph.get_shortest_path_length(node_a, node_b) → int
graph.get_manhattan_distance(node_a, node_b) → int
graph.get_reachable_nodes(start, excluded_edges, excluded_nodes) → Set[int]

# Node queries
graph.get_node_degree(node_id) → int
graph.get_nodes_with_degree_less_than(max_degree) → List[MissionNode]
```

## Usage Examples

### Example 1: Generate with All Rules

```python
from src.generation.grammar import MissionGrammar, Difficulty

grammar = MissionGrammar(seed=42)
graph = grammar.generate(
    difficulty=Difficulty.HARD,
    num_rooms=15,
    max_keys=3,
)

print(f"Generated {len(graph.nodes)} nodes, {len(graph.edges)} edges")
```

### Example 2: Apply Specific Rule

```python
from src.generation.grammar import AddItemGateRule

rule = AddItemGateRule()
context = {'rng': random.Random(123), 'difficulty': 0.7}

if rule.can_apply(graph, context):
    graph = rule.apply(graph, context)
    print("Item gate added!")
```

### Example 3: Check Multi-Floor Nodes

```python
for node in graph.nodes.values():
    if node.position[2] > 0:  # Floor > 0
        print(f"Node {node.id} on floor {node.position[2]}")
```

### Example 4: Find Hubs

```python
hubs = [n for n in graph.nodes.values() if n.is_hub]
print(f"Found {len(hubs)} hub nodes")
```

### Example 5: Trace Item Requirements

```python
for edge in graph.edges:
    if edge.edge_type == EdgeType.ITEM_GATE:
        print(f"Edge {edge.source}→{edge.target} requires {edge.item_required}")
```

## Rule Application Order

Rules are selected probabilistically based on weights during generation.
Recommended conceptual ordering (not enforced):

1. **Structural** (applied first)
   - StartRule
   - BranchRule
   - CreateHubRule

2. **Progression** (mid-generation)
   - InsertLockKeyRule
   - AddItemGateRule
   - InsertChallengeRule

3. **Advanced Topology** (mid-to-late)
   - MergeRule
   - InsertSwitchRule
   - AddStairsRule

4. **Boss Hierarchy** (late)
   - AddBossGauntlet

5. **Optional Content** (late)
   - AddSecretRule
   - AddTeleportRule

6. **Cleanup** (last)
   - PruneGraphRule

## Common Patterns

### Pattern 1: Multi-Floor Hub

```python
# CreateHubRule creates hub on floor 0
# AddStairsRule adds stairs from hub to floor 1
# Result: Multi-floor hub with vertical/horizontal branching
```

### Pattern 2: Item Progression Chain

```python
# AddItemGateRule (BOMB) blocks path to region A
# AddItemGateRule (HOOKSHOT) blocks path to BOMB
# Result: Must get HOOKSHOT → BOMB → region A
```

### Pattern 3: Secret Boss Key

```python
# AddSecretRule creates hidden room
# AddBossGauntlet places BIG_KEY in secret
# Result: Must find secret to access boss
```

### Pattern 4: Backtracking Mitigation

```python
# AddBossGauntlet forces backtracking (BIG_KEY far from BOSS_DOOR)
# MergeRule adds shortcut near BOSS_DOOR
# Result: Less painful backtracking
```

## Testing Quick Commands

```bash
# Run comprehensive tests
python scripts/test_grammar_rules.py

# Quick smoke test
python -c "from src.generation.grammar import MissionGrammar; g=MissionGrammar(seed=1); graph=g.generate(); print(f'{len(graph.nodes)} nodes')"

# Test specific rule
python -c "from src.generation.grammar import AddItemGateRule, MissionGrammar; g=MissionGrammar(seed=1); graph=g.generate(); r=AddItemGateRule(); print(r.can_apply(graph, {'rng': __import__('random').Random()}))"
```

## Known Issues & Limits

1. **PruneGraphRule Preconditions**
   - Rarely applicable (needs 3+ EMPTY chain)
   - Random generators may not create such chains
   - Solution: Generate larger graphs (20+ rooms)

2. **Rule Weight Tuning**
   - Current weights are heuristic-based
   - May need adjustment for desired content density
   - Test with different difficulty levels

3. **3D Layout Algorithm**
   - Currently preserves floor assignments
   - No vertical compaction/optimization
   - Future: 3D force-directed layout

4. **Item Type Hardcoded**
   - AddItemGateRule uses fixed item list
   - Future: Load from config or item database
   - Current: ["BOMB", "HOOKSHOT", "BOW", "FIRE_ROD", "ICE_ROD"]

## Performance Notes

- **Small Graphs (< 10 nodes):** < 1ms generation time
- **Medium Graphs (10-20 nodes):** < 5ms generation time
- **Large Graphs (20-50 nodes):** < 20ms generation time
- **Validation (lock-key):** O(V+E) BFS per lock
- **Bottleneck:** Prune rule's chain detection (BFS per EMPTY node)

## Maintainability

### Adding New Rules

1. Extend `ProductionRule` base class
2. Implement `can_apply()` and `apply()`
3. Add to `MissionGrammar.__init__()` rules list
4. Set appropriate weight (0.1-1.0)
5. Add test case in `test_grammar_rules.py`

### Adding New Node Types

1. Add enum to `NodeType`
2. Update `to_feature_vector()` if needed
3. Add handling in relevant rules
4. Update documentation

### Adding New Edge Types

1. Add enum to `EdgeType`
2. Update `add_edge()` if new parameters needed
3. Add handling in layout/rendering code
4. Update documentation

## Contact & Support

For issues or questions about the mission grammar system:
- Check `docs/MISSION_GRAMMAR_IMPLEMENTATION_REPORT.md` for full details
- Run test suite: `python scripts/test_grammar_rules.py`
- Review examples in `__main__` section of `src/generation/grammar.py`
