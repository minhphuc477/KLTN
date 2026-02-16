"""
Wave 3 Rules - Quick Usage Examples
====================================

Demonstrates all 7 pedagogical & quality control rules with concrete examples.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.generation.grammar import (
    MissionGrammar, MissionGraph, MissionNode, MissionEdge,
    NodeType, EdgeType, Difficulty,
    validate_skill_chains, validate_battery_reachability,
    validate_resource_loops
)


def example_1_skill_chain():
    """Example: Tutorial sequence after getting bow."""
    print("\n=== EXAMPLE 1: Skill Chain (Tutorial Sequence) ===")
    
    grammar = MissionGrammar(seed=42)
    graph = grammar.generate(Difficulty.MEDIUM, num_rooms=10, max_keys=1)
    
    # Find tutorial sequences
    tutorials = [n for n in graph.nodes.values() if n.is_tutorial]
    
    if tutorials:
        for tutorial in tutorials:
            print(f"\n📚 Tutorial Room {tutorial.id}:")
            print(f"   Type: {tutorial.node_type.name}")
            print(f"   Difficulty: {tutorial.difficulty_rating}")
            print(f"   Tension: {tutorial.tension_value}")
            
            # Find what follows
            successors = graph.get_successors(tutorial.id, depth=2)
            print(f"   Followed by: {[f'{n.node_type.name}' for n in successors[:2]]}")
    else:
        print("No tutorial chains generated (probabilistic)")


def example_2_pacing_breaker():
    """Example: Sanctuary after combat gauntlet."""
    print("\n=== EXAMPLE 2: Pacing Breaker (Sanctuary) ===")
    
    grammar = MissionGrammar(seed=123)
    graph = grammar.generate(Difficulty.HARD, num_rooms=12, max_keys=2)
    
    # Find sanctuaries
    sanctuaries = [n for n in graph.nodes.values() if n.is_sanctuary]
    
    if sanctuaries:
        for sanctuary in sanctuaries:
            print(f"\n🕊️ Sanctuary Room {sanctuary.id}:")
            print(f"   Type: {sanctuary.node_type.name}")
            print(f"   Tension: {sanctuary.tension_value} (calm)")
            print(f"   Position: {sanctuary.position}")
    else:
        print("No sanctuaries generated (probabilistic)")
    
    # Show tension chains
    chains = graph.detect_high_tension_chains(min_length=2)
    if chains:
        print(f"\n⚔️ Found {len(chains)} high-tension chain(s):")
        for i, chain in enumerate(chains):
            types = [graph.nodes[nid].node_type.name for nid in chain]
            print(f"   Chain {i+1}: {' → '.join(types)}")


def example_3_resource_farm():
    """Example: Bomb farm near bomb wall."""
    print("\n=== EXAMPLE 3: Resource Loop (Farming Spot) ===")
    
    grammar = MissionGrammar(seed=456)
    graph = grammar.generate(Difficulty.MEDIUM, num_rooms=15, max_keys=2)
    
    # Find resource farms
    farms = [n for n in graph.nodes.values() if n.node_type == NodeType.RESOURCE_FARM]
    
    if farms:
        for farm in farms:
            print(f"\n💣 Resource Farm {farm.id}:")
            print(f"   Drops: {farm.drops_resource}")
            print(f"   Position: {farm.position}")
            
            # Find what gate it supports
            gates = [e for e in graph.edges 
                    if e.item_required == farm.drops_resource]
            if gates:
                print(f"   Supports {len(gates)} gate(s)")
    else:
        print("No resource farms generated (no item gates)")


def example_4_gatekeeper():
    """Example: Mini-boss guarding hookshot."""
    print("\n=== EXAMPLE 4: Gatekeeper (Mini-Boss Guardian) ===")
    
    grammar = MissionGrammar(seed=789)
    graph = grammar.generate(Difficulty.HARD, num_rooms =12, max_keys=2)
    
    # Find mini-bosses
    minibosses = [n for n in graph.nodes.values() if n.is_mini_boss]
    
    if minibosses:
        for boss in minibosses:
            print(f"\n🐉 Mini-Boss {boss.id}:")
            print(f"   Difficulty: {boss.difficulty_rating}")
            print(f"   Room Size: {boss.room_size}")
            print(f"   Tension: {boss.tension_value}")
            
            # Find what it guards
            guarded = [e.target for e in graph.edges 
                      if e.source == boss.id and e.edge_type == EdgeType.SHUTTER]
            if guarded:
                for item_id in guarded:
                    if item_id in graph.nodes:
                        item = graph.nodes[item_id]
                        print(f"   Guards: {item.node_type.name} (node {item_id})")
    else:
        print("No mini-bosses generated (probabilistic)")


def example_5_battery_pattern():
    """Example: 3 switches controlling 1 door."""
    print("\n=== EXAMPLE 5: Multi-Lock (Battery Pattern) ===")
    
    grammar = MissionGrammar(seed=321)
    graph = grammar.generate(Difficulty.EXPERT, num_rooms=15, max_keys=2)
    
    # Find battery edges
    batteries = [e for e in graph.edges if e.battery_id is not None]
    
    if batteries:
        for battery in batteries:
            print(f"\n🔋 Battery Lock (ID {battery.battery_id}):")
            print(f"   Locks: {battery.source} → {battery.target}")
            print(f"   Requires {len(battery.switches_required)} switches:")
            
            for switch_id in battery.switches_required:
                if switch_id in graph.nodes:
                    switch = graph.nodes[switch_id]
                    print(f"      - Switch {switch_id} at {switch.position}")
    else:
        print("No battery patterns generated (need large dungeon with hubs)")


def example_6_item_shortcut():
    """Example: Hookshot shortcut back to start."""
    print("\n=== EXAMPLE 6: Item Shortcut (Backtracking Reward) ===")
    
    grammar = MissionGrammar(seed=654)
    graph = grammar.generate(Difficulty.MEDIUM, num_rooms=15, max_keys=2)
    
    # Find item-gated shortcuts
    shortcuts = [e for e in graph.edges 
                if e.edge_type == EdgeType.ITEM_GATE 
                and e.preferred_direction == "backward"]
    
    if shortcuts:
        for shortcut in shortcuts:
            print(f"\n🪝 Item Shortcut:")
            print(f"   From: {shortcut.source} → To: {shortcut.target}")
            print(f"   Requires: {shortcut.item_required}")
            print(f"   Saves: {shortcut.path_savings} hops")
    else:
        print("No shortcuts generated (need distant items)")


def example_7_pruning():
    """Example: Before/after dead-end pruning."""
    print("\n=== EXAMPLE 7: Dead-End Pruning (Quality Control) ===")
    
    grammar = MissionGrammar(seed=987)
    graph = grammar.generate(Difficulty.MEDIUM, num_rooms=12, max_keys=1)
    
    # Count dead ends
    dead_ends = []
    valuable_dead_ends = []
    
    valuable_types = {
        NodeType.KEY, NodeType.ITEM, NodeType.BOSS, NodeType.MINI_BOSS,
        NodeType.SWITCH, NodeType.GOAL, NodeType.TREASURE
    }
    
    for node in graph.nodes.values():
        degree = graph.get_node_degree(node.id)
        if degree == 1:
            if node.node_type in valuable_types or node.is_secret:
                valuable_dead_ends.append(node)
            else:
                dead_ends.append(node)
    
    print(f"\nDead-End Analysis:")
    print(f"   Total dead-ends: {len(dead_ends) + len(valuable_dead_ends)}")
    print(f"   Valuable (kept): {len(valuable_dead_ends)}")
    print(f"   Useless (pruned): {len(dead_ends)}")
    
    if valuable_dead_ends:
        print(f"\n   Kept dead-ends:")
        for node in valuable_dead_ends[:3]:
            print(f"      - {node.node_type.name} at node {node.id}")


def full_generation_example():
    """Complete dungeon with all Wave 3 features."""
    print("\n" + "="*60)
    print("FULL GENERATION: Large Dungeon with All Wave 3 Rules")
    print("="*60)
    
    grammar = MissionGrammar(seed=42)
    graph = grammar.generate(Difficulty.MEDIUM, num_rooms=25, max_keys=3)  # Larger dungeon
    
    print(f"\nGenerated Dungeon:")
    print(f"  Total Nodes: {len(graph.nodes)}")
    print(f"  Total Edges: {len(graph.edges)}")
    print(f"  Connected: {graph.is_graph_connected()}")
    
    # Count Wave 3 features
    tutorials = sum(1 for n in graph.nodes.values() if n.is_tutorial)
    sanctuaries = sum(1 for n in graph.nodes.values() if n.is_sanctuary)
    farms = sum(1 for n in graph.nodes.values() if n.node_type == NodeType.RESOURCE_FARM)
    minibosses = sum(1 for n in graph.nodes.values() if n.is_mini_boss)
    batteries = sum(1 for e in graph.edges if e.battery_id is not None)
    shortcuts = sum(1 for e in graph.edges 
                   if e.edge_type == EdgeType.ITEM_GATE and e.preferred_direction == "backward")
    
    print(f"\nWave 3 Features:")
    print(f"  [Tutorial] Tutorial Chains: {tutorials}")
    print(f"  [Pacing] Sanctuaries: {sanctuaries}")
    print(f"  [Safety] Resource Farms: {farms}")
    print(f"  [Quality] Mini-Bosses: {minibosses}")
    print(f"  [Quality] Battery Locks: {batteries}")
    print(f"  [Quality] Item Shortcuts: {shortcuts}")
    
    # Node type distribution
    print(f"\nNode Type Distribution:")
    type_counts = {}
    for node in graph.nodes.values():
        type_name = node.node_type.name
        type_counts[type_name] = type_counts.get(type_name, 0) + 1
    
    for node_type, count in sorted(type_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {node_type}: {count}")
    
    # Validation
    print(f"\nValidation:")
    print(f"  Lock-Key Valid: {grammar.validate_lock_key_ordering(graph)}")
    print(f"  Skill Chains Valid: {validate_skill_chains(graph)}")
    print(f"  Battery Reachability: {validate_battery_reachability(graph)}")
    print(f"  Resource Loops Valid: {validate_resource_loops(graph)}")


if __name__ == '__main__':
    # Run all examples
    example_1_skill_chain()
    example_2_pacing_breaker()
    example_3_resource_farm()
    example_4_gatekeeper()
    example_5_battery_pattern()
    example_6_item_shortcut()
    example_7_pruning()
    
    full_generation_example()
    
    print("\n" + "="*60)
    print("✅ All Wave 3 Examples Complete!")
    print("="*60)
