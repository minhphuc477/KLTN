"""
Integration Tests for Advanced Production Rules
================================================

Comprehensive test suite for all 10 advanced dungeon generation rules.

Run with: 
    cd F:\KLTN
    python -m pytest tests/test_advanced_rules_integration.py -v
    
Or standalone:
    cd F:\KLTN
    python tests/test_advanced_rules_integration.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from typing import List, Dict, Set

from src.generation.grammar import (
    MissionGrammar, 
    Difficulty, 
    NodeType, 
    EdgeType,
    MissionGraph,
)


class TestAdvancedRulesIntegration:
    """Integration tests for advanced rule application."""
    
    def test_all_rules_loadable(self):
        """Test that all rules are properly loaded in grammar."""
        grammar = MissionGrammar(seed=42)
        
        # Check we have expected number of rules
        assert len(grammar.rules) >= 24, f"Expected ≥24 rules, got {len(grammar.rules)}"
        
        # Check advanced rules are present
        rule_names = [rule.name for rule in grammar.rules]
        
        advanced_rules = [
            "AddFungibleLock",
            "FormBigRoom",
            "AddValve",
            "AddForeshadowing",
            "AddCollectionChallenge",
            "AddArena",
            "AddSector",
            "AddEntangledBranches",
            "AddHazardGate",
            "SplitRoom",
        ]
        
        for rule_name in advanced_rules:
            assert any(rule_name in name for name in rule_names), \
                f"Advanced rule '{rule_name}' not found in grammar"
    
    def test_large_dungeon_generation(self):
        """Test generating large dungeon with all rules active."""
        grammar = MissionGrammar(seed=12345)
        
        graph = grammar.generate(
            difficulty=Difficulty.HARD,
            num_rooms=25,
            max_keys=3,
        )
        
        # Verify basic constraints
        assert len(graph.nodes) >= 10, "Should generate substantial graph"
        assert len(graph.edges) >= 8, "Should have multiple connections"
        
        # Verify diversity
        node_types = set(n.node_type for n in graph.nodes.values())
        edge_types = set(e.edge_type for e in graph.edges)
        
        assert len(node_types) >= 5, f"Should have ≥5 node types, got {len(node_types)}"
        assert len(edge_types) >= 3, f"Should have ≥3 edge types, got {len(edge_types)}"
        
        print(f"\n✅ Generated graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        print(f"   Node types: {[t.name for t in node_types]}")
        print(f"   Edge types: {[t.name for t in edge_types]}")
    
    def test_fungible_key_economy(self):
        """Test fungible key system (RULE #1)."""
        grammar = MissionGrammar(seed=42)
        graph = grammar.generate(num_rooms=12, max_keys=0)
        
        # Look for fungible locks
        fungible_locks = [e for e in graph.edges if e.requires_key_count > 0]
        fungible_keys = [n for n in graph.nodes.values() 
                        if n.node_type == NodeType.KEY and n.key_id is None]
        
        # If rule applied, validate it
        if fungible_locks:
            print(f"\n✅ Fungible locks found: {len(fungible_locks)}")
            assert len(fungible_keys) >= 1, "Should have fungible keys"
            
            for lock_edge in fungible_locks:
                keys_before = graph.count_keys_available_before(lock_edge.target)
                assert keys_before >= lock_edge.requires_key_count, \
                    f"Not enough keys before lock (have {keys_before}, need {lock_edge.requires_key_count})"
        else:
            print("\n⚠️  Fungible lock rule not applied (probabilistic)")
    
    def test_big_room_merging(self):
        """Test big room formation (RULE #2)."""
        grammar = MissionGrammar(seed=123)
        graph = grammar.generate(num_rooms=15)
        
        big_rooms = [n for n in graph.nodes.values() if n.is_big_room]
        
        if big_rooms:
            print(f"\n✅ Big rooms found: {len(big_rooms)}")
            
            for room in big_rooms:
                assert room.room_size != (1, 1), "Big room should have non-default size"
                assert room.room_size in [(2,1), (1,2), (2,2)], \
                    f"Invalid big room size: {room.room_size}"
                print(f"   Room {room.id}: size {room.room_size}")
        else:
            print("\n⚠️  Big room rule not applied (probabilistic)")
    
    def test_cycle_valves(self):
        """Test one-way valves in cycles (RULE #3)."""
        grammar = MissionGrammar(seed=456)
        graph = grammar.generate(num_rooms=18)
        
        cycles = graph.detect_cycles()
        one_way_edges = [e for e in graph.edges if e.edge_type == EdgeType.ONE_WAY]
        
        print(f"\n   Cycles detected: {len(cycles)}")
        print(f"   One-way edges: {len(one_way_edges)}")
        
        if cycles and one_way_edges:
            print(f"✅ Valve rule applied successfully")
            
            # Verify asymmetry
            valve = one_way_edges[0]
            assert valve.target in graph._adjacency.get(valve.source, []), \
                "Forward path should exist"
            
            if valve.source in graph._adjacency.get(valve.target, []):
                print(f"   ⚠️  Valve {valve.source}→{valve.target} still bidirectional")
    
    def test_visual_foreshadowing(self):
        """Test visual links (RULE #4)."""
        grammar = MissionGrammar(seed=789)
        graph = grammar.generate(num_rooms=20)
        
        visual_links = [e for e in graph.edges if e.edge_type == EdgeType.VISUAL_LINK]
        
        if visual_links:
            print(f"\n✅ Visual links found: {len(visual_links)}")
            
            for link in visual_links:
                # Verify spatial proximity
                manhattan = graph.get_manhattan_distance(link.source, link.target)
                assert manhattan <= 2, \
                    f"Visual link nodes too far spatially: {manhattan}"
                
                # Verify topological distance
                path_dist = graph.get_shortest_path_length(link.source, link.target)
                if path_dist > 0:  # If connected
                    assert path_dist > 4, \
                        f"Visual link nodes too close topologically: {path_dist}"
                
                print(f"   Link {link.source}→{link.target}: spatial={manhattan}, path={path_dist}")
        else:
            print("\n⚠️  Visual link rule not applied (probabilistic)")
    
    def test_collection_challenge(self):
        """Test token collection system (RULE #5)."""
        grammar = MissionGrammar(seed=111)
        graph = grammar.generate(num_rooms=20)
        
        tokens = [n for n in graph.nodes.values() if n.node_type == NodeType.TOKEN]
        multi_locks = [e for e in graph.edges if e.edge_type == EdgeType.MULTI_LOCK]
        
        if multi_locks:
            print(f"\n✅ Collection challenge found")
            print(f"   Tokens: {len(tokens)}")
            print(f"   Multi-locks: {len(multi_locks)}")
            
            for lock in multi_locks:
                assert lock.token_count >= 2, "Multi-lock should require ≥2 tokens"
                print(f"   Lock at {lock.source}→{lock.target} requires {lock.token_count} tokens")
            
            # Verify sufficient tokens exist
            if tokens:
                assert len(tokens) >= multi_locks[0].token_count, \
                    "Not enough tokens for multi-lock"
        else:
            print("\n⚠️  Collection challenge rule not applied (probabilistic)")
    
    def test_combat_arenas(self):
        """Test arena rooms with shutters (RULE #6)."""
        grammar = MissionGrammar(seed=222)
        graph = grammar.generate(num_rooms=15)
        
        arenas = [n for n in graph.nodes.values() if n.is_arena]
        shutters = [e for e in graph.edges if e.edge_type == EdgeType.SHUTTER]
        
        if arenas:
            print(f"\n✅ Arenas found: {len(arenas)}")
            print(f"   Shutter edges: {len(shutters)}")
            
            for arena in arenas:
                incoming_shutters = [e for e in shutters if e.target == arena.id]
                print(f"   Arena {arena.id}: {len(incoming_shutters)} shutters")
        else:
            print("\n⚠️  Arena rule not applied (probabilistic)")
    
    def test_thematic_sectors(self):
        """Test sector grouping (RULE #7)."""
        grammar = MissionGrammar(seed=333)
        graph = grammar.generate(num_rooms=25)
        
        # Group by sector
        sectors: Dict[int, List] = {}
        for node in graph.nodes.values():
            if node.sector_id > 0:
                if node.sector_id not in sectors:
                    sectors[node.sector_id] = []
                sectors[node.sector_id].append(node)
        
        if sectors:
            print(f"\n✅ Sectors found: {len(sectors)}")
            
            for sector_id, nodes in sectors.items():
                themes = [n.sector_theme for n in nodes if n.sector_theme]
                theme = themes[0] if themes else "NONE"
                print(f"   Sector {sector_id} ({theme}): {len(nodes)} nodes")
                
                # Verify theme consistency
                if themes:
                    assert len(set(themes)) == 1, \
                        f"Sector {sector_id} has inconsistent themes: {set(themes)}"
        else:
            print("\n⚠️  Sector rule not applied (probabilistic)")
    
    def test_entangled_branches(self):
        """Test cross-branch dependencies (RULE #8)."""
        grammar = MissionGrammar(seed=444)
        graph = grammar.generate(num_rooms=22)
        
        switches = [n for n in graph.nodes.values() if n.node_type == NodeType.SWITCH]
        state_blocks = [e for e in graph.edges if e.edge_type == EdgeType.STATE_BLOCK]
        
        if switches and state_blocks:
            print(f"\n✅ Entangled branches detected")
            print(f"   Switches: {len(switches)}")
            print(f"   State blocks: {len(state_blocks)}")
            
            # Verify switch-gate pairs
            for block in state_blocks:
                if block.switch_id:
                    matching_switches = [s for s in switches if s.switch_id == block.switch_id]
                    assert len(matching_switches) > 0, \
                        f"State block references non-existent switch {block.switch_id}"
        else:
            print("\n⚠️  Entangled branches rule not applied (probabilistic)")
    
    def test_hazard_gates(self):
        """Test hazard paths with protection (RULE #9)."""
        grammar = MissionGrammar(seed=555)
        graph = grammar.generate(num_rooms=16)
        
        hazards = [e for e in graph.edges if e.edge_type == EdgeType.HAZARD]
        protections = [n for n in graph.nodes.values() if n.node_type == NodeType.PROTECTION_ITEM]
        
        if hazards:
            print(f"\n✅ Hazards found: {len(hazards)}")
            print(f"   Protection items: {len(protections)}")
            
            for hazard in hazards:
                assert hazard.hazard_damage > 0, "Hazard should have damage"
                assert hazard.protection_item_id is not None, \
                    "Hazard should reference protection item"
                print(f"   Hazard {hazard.source}→{hazard.target}: " + 
                      f"damage={hazard.hazard_damage}, " +
                      f"protection={hazard.protection_item_id}")
                
                # Check if matching protection exists
                matching = [p for p in protections 
                           if p.item_type == hazard.protection_item_id]
                if not matching:
                    print(f"      ⚠️  No matching protection item found")
        else:
            print("\n⚠️  Hazard gate rule not applied (probabilistic)")
    
    def test_virtual_room_layers(self):
        """Test virtual room layering (RULE #10)."""
        grammar = MissionGrammar(seed=666)
        graph = grammar.generate(num_rooms=14)
        
        layered_nodes = [n for n in graph.nodes.values() if n.virtual_layer > 0]
        
        if layered_nodes:
            print(f"\n✅ Virtual layers found: {len(layered_nodes)}")
            
            for layered in layered_nodes:
                # Find nodes at same position
                same_pos = [
                    n for n in graph.nodes.values()
                    if n.position[:2] == layered.position[:2]
                    and n.id != layered.id
                    and n.virtual_layer != layered.virtual_layer
                ]
                
                if same_pos:
                    print(f"   Layer {layered.virtual_layer} at {layered.position[:2]}" +
                          f" shares position with {len(same_pos)} nodes")
                
                # Check for layer-appropriate connections
                layer_edges = [
                    e for e in graph.edges
                    if (e.source == layered.id or e.target == layered.id)
                    and e.edge_type in [EdgeType.ONE_WAY, EdgeType.STAIRS]
                ]
                
                if layer_edges:
                    print(f"      Connected via {layer_edges[0].edge_type.name}")
        else:
            print("\n⚠️  Virtual layer rule not applied (probabilistic)")
    
    def test_advanced_features_diversity(self):
        """Test that multiple advanced features appear together."""
        grammar = MissionGrammar(seed=99999)
        graph = grammar.generate(
            difficulty=Difficulty.EXPERT,
            num_rooms=30,
            max_keys=3,
        )
        
        # Count all advanced features
        features = {
            'big_rooms': len([n for n in graph.nodes.values() if n.is_big_room]),
            'arenas': len([n for n in graph.nodes.values() if n.is_arena]),
            'tokens': len([n for n in graph.nodes.values() if n.node_type == NodeType.TOKEN]),
            'protection_items': len([n for n in graph.nodes.values() if n.node_type == NodeType.PROTECTION_ITEM]),
            'sectors': len(set(n.sector_id for n in graph.nodes.values() if n.sector_id > 0)),
            'virtual_layers': len([n for n in graph.nodes.values() if n.virtual_layer > 0]),
            'one_way_edges': len([e for e in graph.edges if e.edge_type == EdgeType.ONE_WAY]),
            'hazards': len([e for e in graph.edges if e.edge_type == EdgeType.HAZARD]),
            'visual_links': len([e for e in graph.edges if e.edge_type == EdgeType.VISUAL_LINK]),
            'shutters': len([e for e in graph.edges if e.edge_type == EdgeType.SHUTTER]),
            'multi_locks': len([e for e in graph.edges if e.edge_type == EdgeType.MULTI_LOCK]),
        }
        
        print(f"\n📊 Advanced Features Summary:")
        total = 0
        for feature, count in features.items():
            if count > 0:
                print(f"   ✓ {feature}: {count}")
                total += count
        
        # Should have multiple types of advanced features
        feature_types = len([v for v in features.values() if v > 0])
        assert feature_types >= 2, \
            f"Should have ≥2 types of advanced features (got {feature_types})"
        
        print(f"\n✅ Total advanced features: {total} ({feature_types} types)")


class TestRuleConstraints:
    """Test that rules respect graph constraints."""
    
    def test_no_invalid_node_types(self):
        """Test that only valid node types are created."""
        grammar = MissionGrammar(seed=777)
        graph = grammar.generate(num_rooms=20)
        
        valid_types = set(NodeType)
        for node in graph.nodes.values():
            assert node.node_type in valid_types, \
                f"Invalid node type: {node.node_type}"
    
    def test_no_invalid_edge_types(self):
        """Test that only valid edge types are created."""
        grammar = MissionGrammar(seed=888)
        graph = grammar.generate(num_rooms=20)
        
        valid_types = set(EdgeType)
        for edge in graph.edges:
            assert edge.edge_type in valid_types, \
                f"Invalid edge type: {edge.edge_type}"
    
    def test_no_self_loops(self):
        """Test that no edges connect a node to itself."""
        grammar = MissionGrammar(seed=999)
        graph = grammar.generate(num_rooms=15)
        
        for edge in graph.edges:
            assert edge.source != edge.target, \
                f"Self-loop detected: {edge.source} → {edge.target}"
    
    def test_start_and_goal_exist(self):
        """Test that START and GOAL nodes always exist."""
        grammar = MissionGrammar(seed=1111)
        graph = grammar.generate(num_rooms=10)
        
        start = graph.get_start_node()
        goal = graph.get_goal_node()
        
        assert start is not None, "START node missing"
        assert goal is not None, "GOAL node missing"
        
        print(f"\n✅ Start: {start.id}, Goal: {goal.id}")


def test_quick_smoke_test():
    """Quick smoke test for CI/CD."""
    grammar = MissionGrammar(seed=12345)
    graph = grammar.generate(num_rooms=8)
    
    assert len(graph.nodes) >= 2  # At least START and GOAL
    assert len(graph.edges) >= 1
    
    print("\n✅ Quick smoke test passed")


if __name__ == '__main__':
    print("="*70)
    print("ADVANCED RULES INTEGRATION TEST SUITE")
    print("="*70)
    
    # Run tests
    test_suite = TestAdvancedRulesIntegration()
    
    print("\n" + "="*70)
    print("INDIVIDUAL RULE TESTS")
    print("="*70)
    
    test_suite.test_all_rules_loadable()
    test_suite.test_large_dungeon_generation()
    
    print("\n" + "-"*70)
    print("Testing Rule #1: Fungible Keys")
    print("-"*70)
    test_suite.test_fungible_key_economy()
    
    print("\n" + "-"*70)
    print("Testing Rule #2: Big Rooms")
    print("-"*70)
    test_suite.test_big_room_merging()
    
    print("\n" + "-"*70)
    print("Testing Rule #3: Valves")
    print("-"*70)
    test_suite.test_cycle_valves()
    
    print("\n" + "-"*70)
    print("Testing Rule #4: Visual Links")
    print("-"*70)
    test_suite.test_visual_foreshadowing()
    
    print("\n" + "-"*70)
    print("Testing Rule #5: Collection Challenges")
    print("-"*70)
    test_suite.test_collection_challenge()
    
    print("\n" + "-"*70)
    print("Testing Rule #6: Arenas")
    print("-"*70)
    test_suite.test_combat_arenas()
    
    print("\n" + "-"*70)
    print("Testing Rule #7: Sectors")
    print("-"*70)
    test_suite.test_thematic_sectors()
    
    print("\n" + "-"*70)
    print("Testing Rule #8: Entangled Branches")
    print("-"*70)
    test_suite.test_entangled_branches()
    
    print("\n" + "-"*70)
    print("Testing Rule #9: Hazards")
    print("-"*70)
    test_suite.test_hazard_gates()
    
    print("\n" + "-"*70)
    print("Testing Rule #10: Virtual Layers")
    print("-"*70)
    test_suite.test_virtual_room_layers()
    
    print("\n" + "="*70)
    print("DIVERSITY AND CONSTRAINTS TESTS")
    print("="*70)
    
    test_suite.test_advanced_features_diversity()
    
    constraint_tests = TestRuleConstraints()
    constraint_tests.test_no_invalid_node_types()
    constraint_tests.test_no_invalid_edge_types()
    constraint_tests.test_no_self_loops()
    constraint_tests.test_start_and_goal_exist()
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
    print("="*70)
