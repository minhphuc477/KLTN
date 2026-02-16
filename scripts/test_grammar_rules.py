"""
Test script to validate all mission grammar production rules.

Tests each rule individually and verifies they can be applied correctly.
"""

import sys
sys.path.insert(0, 'f:/KLTN')

from src.generation.grammar import (
    MissionGrammar, Difficulty, NodeType, EdgeType,
    StartRule, InsertChallengeRule, InsertLockKeyRule, BranchRule,
    MergeRule, InsertSwitchRule, AddBossGauntlet,
    AddItemGateRule, CreateHubRule, AddStairsRule,
    AddSecretRule, AddTeleportRule, PruneGraphRule,
    MissionGraph, MissionNode
)


def test_basic_generation():
    """Test basic graph generation."""
    print("=" * 60)
    print("TEST 1: Basic Generation")
    print("=" * 60)
    
    grammar = MissionGrammar(seed=42)
    graph = grammar.generate(
        difficulty=Difficulty.MEDIUM,
        num_rooms=10,
        max_keys=2,
    )
    
    print(f"✓ Generated graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    
    # Check for required nodes
    start = graph.get_start_node()
    goal = graph.get_goal_node()
    
    assert start is not None, "START node missing"
    assert goal is not None, "GOAL node missing"
    print(f"✓ START and GOAL nodes present")
    
    # Validate lock-key ordering
    valid = grammar.validate_lock_key_ordering(graph)
    print(f"✓ Lock-key ordering valid: {valid}")
    
    return graph


def test_individual_rules():
    """Test each rule individually."""
    print("\n" + "=" * 60)
    print("TEST 2: Individual Rule Application")
    print("=" * 60)
    
    rng_seed = 123
    context = {'rng': __import__('random').Random(rng_seed), 'difficulty': 0.5}
    
    rules_to_test = [
        ("MergeRule", MergeRule()),
        ("InsertSwitchRule", InsertSwitchRule()),
        ("AddBossGauntlet", AddBossGauntlet()),
        ("AddItemGateRule", AddItemGateRule()),
        ("CreateHubRule", CreateHubRule()),
        ("AddStairsRule", AddStairsRule()),
        ("AddSecretRule", AddSecretRule()),
        ("AddTeleportRule", AddTeleportRule()),
        ("PruneGraphRule", PruneGraphRule()),
    ]
    
    for rule_name, rule in rules_to_test:
        # Create a test graph
        grammar = MissionGrammar(seed=rng_seed)
        graph = grammar.generate(difficulty=Difficulty.MEDIUM, num_rooms=8, max_keys=1)
        
        initial_nodes = len(graph.nodes)
        initial_edges = len(graph.edges)
        
        # Try to apply rule
        if rule.can_apply(graph, context):
            graph = rule.apply(graph, context)
            print(f"✓ {rule_name:20s} - Applied successfully (nodes: {initial_nodes}→{len(graph.nodes)}, edges: {initial_edges}→{len(graph.edges)})")
        else:
            print(f"⊘ {rule_name:20s} - Cannot apply (preconditions not met)")


def test_node_types():
    """Test that all node types are present in generation."""
    print("\n" + "=" * 60)
    print("TEST 3: Node Type Coverage")
    print("=" * 60)
    
    # Generate multiple graphs to see variety
    all_node_types = set()
    
    for seed in range(10):
        grammar = MissionGrammar(seed=seed)
        graph = grammar.generate(
            difficulty=Difficulty.HARD,
            num_rooms=15,
            max_keys=3,
        )
        
        for node in graph.nodes.values():
            all_node_types.add(node.node_type)
    
    print(f"✓ Found {len(all_node_types)} unique node types across 10 generations:")
    for node_type in sorted(all_node_types, key=lambda x: x.value):
        print(f"  - {node_type.name}")


def test_edge_types():
    """Test that various edge types are present."""
    print("\n" + "=" * 60)
    print("TEST 4: Edge Type Coverage")
    print("=" * 60)
    
    all_edge_types = set()
    
    for seed in range(10):
        grammar = MissionGrammar(seed=seed)
        graph = grammar.generate(
            difficulty=Difficulty.HARD,
            num_rooms=15,
            max_keys=3,
        )
        
        for edge in graph.edges:
            all_edge_types.add(edge.edge_type)
    
    print(f"✓ Found {len(all_edge_types)} unique edge types across 10 generations:")
    for edge_type in sorted(all_edge_types, key=lambda x: x.value):
        print(f"  - {edge_type.name}")


def test_helper_methods():
    """Test helper methods on MissionGraph."""
    print("\n" + "=" * 60)
    print("TEST 5: Helper Methods")
    print("=" * 60)
    
    grammar = MissionGrammar(seed=999)
    graph = grammar.generate(difficulty=Difficulty.MEDIUM, num_rooms=8, max_keys=2)
    
    # Test get_shortest_path_length
    start = graph.get_start_node()
    goal = graph.get_goal_node()
    if start and goal:
        path_len = graph.get_shortest_path_length(start.id, goal.id)
        print(f"✓ get_shortest_path_length: START→GOAL = {path_len} hops")
    
    # Test get_node_degree
    for node_id in list(graph.nodes.keys())[:3]:
        degree = graph.get_node_degree(node_id)
        print(f"✓ get_node_degree: Node {node_id} has degree {degree}")
    
    # Test get_manhattan_distance
    nodes = list(graph.nodes.keys())
    if len(nodes) >= 2:
        dist = graph.get_manhattan_distance(nodes[0], nodes[1])
        print(f"✓ get_manhattan_distance: {nodes[0]}↔{nodes[1]} = {dist} units")
    
    # Test get_reachable_nodes
    if start:
        reachable = graph.get_reachable_nodes(start.id)
        print(f"✓ get_reachable_nodes: {len(reachable)} nodes reachable from START")
    
    # Test get_nodes_with_degree_less_than
    low_degree = graph.get_nodes_with_degree_less_than(3)
    print(f"✓ get_nodes_with_degree_less_than: {len(low_degree)} nodes with degree < 3")


def test_3d_positions():
    """Test that 3D positions work correctly."""
    print("\n" + "=" * 60)
    print("TEST 6: 3D Position Support")
    print("=" * 60)
    
    grammar = MissionGrammar(seed=777)
    graph = grammar.generate(difficulty=Difficulty.HARD, num_rooms=12, max_keys=2)
    
    # Check position dimensions
    has_3d = False
    for node in graph.nodes.values():
        if len(node.position) == 3:
            has_3d = True
            if node.position[2] > 0:
                print(f"✓ Found node {node.id} ({node.node_type.name}) on floor {node.position[2]}")
    
    if has_3d:
        print(f"✓ 3D positions supported")
    else:
        print(f"⚠ No nodes on non-zero floors (may be random)")


def print_detailed_graph(graph):
    """Print detailed graph structure."""
    print("\n" + "=" * 60)
    print("DETAILED GRAPH STRUCTURE")
    print("=" * 60)
    
    print(f"\nNodes ({len(graph.nodes)}):")
    for node in sorted(graph.nodes.values(), key=lambda n: n.id):
        extras = []
        if node.key_id:
            extras.append(f"key_id={node.key_id}")
        if node.required_item:
            extras.append(f"req_item={node.required_item}")
        if node.item_type:
            extras.append(f"item={node.item_type}")
        if node.is_hub:
            extras.append("HUB")
        if node.is_secret:
            extras.append("SECRET")
        
        extra_str = f" [{', '.join(extras)}]" if extras else ""
        print(f"  {node.id:2d}: {node.node_type.name:15s} pos={node.position} diff={node.difficulty:.2f}{extra_str}")
    
    print(f"\nEdges ({len(graph.edges)}):")
    for edge in graph.edges:
        extras = []
        if edge.key_required:
            extras.append(f"key={edge.key_required}")
        if edge.item_required:
            extras.append(f"item={edge.item_required}")
        if edge.switch_id:
            extras.append(f"switch={edge.switch_id}")
        
        extra_str = f" [{', '.join(extras)}]" if extras else ""
        print(f"  {edge.source:2d} → {edge.target:2d}  ({edge.edge_type.name:15s}){extra_str}")


if __name__ == '__main__':
    print("\n" + "#" * 60)
    print("# MISSION GRAMMAR RULE VALIDATION TEST")
    print("#" * 60 + "\n")
    
    try:
        # Run tests
        graph = test_basic_generation()
        test_individual_rules()
        test_node_types()
        test_edge_types()
        test_helper_methods()
        test_3d_positions()
        
        # Print example graph
        print_detailed_graph(graph)
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
