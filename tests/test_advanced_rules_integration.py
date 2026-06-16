"""
Integration Tests for Advanced Production Rules
================================================

Comprehensive test suite for all 10 advanced dungeon generation rules.

Run with: 
    cd <repo root>
    python -m pytest tests/test_advanced_rules_integration.py -v
    
Or standalone:
    cd <repo root>
    python tests/test_advanced_rules_integration.py
"""
# pyright: reportPrivateUsage=false

import sys
import random
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict

import pytest

from src.generation.grammar import (
    AddBossGauntlet,
    AddSecretRule,
    AddValveRule,
    CreateHubRule,
    InsertChallengeRule,
    InsertSwitchRule,
    MissionEdge,
    MissionGraph,
    MissionNode,
    MissionGrammar, 
    Difficulty, 
    NodeType, 
    EdgeType,
)


def _positions_are_unique(graph: MissionGraph) -> bool:
    positions = [
        (int(node.position[0]), int(node.position[1]), int(node.position[2]) if len(node.position) > 2 else 0)
        for node in graph.nodes.values()
    ]
    return len(positions) == len(set(positions))


class TestAdvancedRulesIntegration:
    """Integration tests for advanced rule application."""
    
    def test_all_rules_loadable(self):
        """Test that all rules are properly loaded in grammar."""
        grammar = MissionGrammar(seed=42)
        
        # Check we have expected number of rules
        assert len(grammar.rules) >= 24, f"Expected >=24 rules, got {len(grammar.rules)}"
        
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

    def test_add_boss_gauntlet_creates_strict_boss_chain(self):
        """Boss gauntlet should build BOSS_DOOR -> BOSS -> GOAL, not door -> goal directly."""
        graph = MissionGraph()
        graph.add_node(MissionNode(id=0, node_type=NodeType.START, position=(0, 0, 0), difficulty=0.0))
        graph.add_node(MissionNode(id=1, node_type=NodeType.ENEMY, position=(1, 0, 0), difficulty=0.4))
        graph.add_node(MissionNode(id=2, node_type=NodeType.GOAL, position=(3, 0, 0), difficulty=1.0))
        graph.add_edge(0, 1, EdgeType.PATH)
        graph.add_edge(1, 2, EdgeType.PATH)

        rule = AddBossGauntlet()
        updated = rule.apply(graph, {"rng": None})
        updated.sanitize()

        boss_doors = updated.get_nodes_by_type(NodeType.BOSS_DOOR)
        bosses = updated.get_nodes_by_type(NodeType.BOSS)
        big_keys = updated.get_nodes_by_type(NodeType.BIG_KEY)

        assert len(boss_doors) == 1
        assert len(bosses) == 1
        assert len(big_keys) >= 1
        assert any(
            edge.source == boss_doors[0].id and edge.target == bosses[0].id and edge.edge_type == EdgeType.PATH
            for edge in updated.edges
        )
        assert any(
            edge.source == bosses[0].id and edge.target == 2 and edge.edge_type == EdgeType.PATH
            for edge in updated.edges
        )
        assert not any(edge.source == boss_doors[0].id and edge.target == 2 for edge in updated.edges)

        grammar = MissionGrammar(seed=42)
        assert grammar.validate_goal_gauntlet(updated)
        assert _positions_are_unique(updated)

    def test_insert_challenge_adjacent_edge_does_not_collide_with_source_room(self):
        graph = MissionGraph()
        graph.add_node(MissionNode(id=0, node_type=NodeType.START, position=(0, 0, 0), difficulty=0.0))
        graph.add_node(MissionNode(id=1, node_type=NodeType.GOAL, position=(0, 1, 0), difficulty=1.0))
        graph.add_edge(0, 1, EdgeType.PATH)

        updated = InsertChallengeRule(NodeType.ENEMY).apply(
            graph,
            {"rng": random.Random(3), "layout_bounds": (-4, 4, -4, 4)},
        )

        assert len(updated.nodes) == 3
        assert _positions_are_unique(updated)
        inserted = next(node for node in updated.nodes.values() if node.node_type == NodeType.ENEMY)
        assert inserted.position != (0, 0, 0)
        assert inserted.position != (0, 1, 0)

    def test_insert_switch_uses_free_position_instead_of_absolute_override(self):
        graph = MissionGraph()
        graph.add_node(MissionNode(id=0, node_type=NodeType.START, position=(0, 0, 0), difficulty=0.0))
        graph.add_node(MissionNode(id=1, node_type=NodeType.ENEMY, position=(0, 1, 0), difficulty=0.3))
        graph.add_node(MissionNode(id=2, node_type=NodeType.PUZZLE, position=(1, 0, 0), difficulty=0.4))
        graph.add_node(MissionNode(id=3, node_type=NodeType.GOAL, position=(1, 1, 0), difficulty=1.0))
        graph.add_edge(0, 1, EdgeType.PATH)
        graph.add_edge(1, 3, EdgeType.PATH)
        graph.add_edge(0, 2, EdgeType.PATH)

        updated = InsertSwitchRule().apply(
            graph,
            {"rng": random.Random(7), "layout_bounds": (-4, 4, -4, 4)},
        )

        assert any(node.node_type == NodeType.SWITCH for node in updated.nodes.values())
        assert _positions_are_unique(updated)

    def test_create_hub_spokes_use_free_positions(self):
        graph = MissionGraph()
        graph.add_node(MissionNode(id=0, node_type=NodeType.START, position=(0, 0, 0), difficulty=0.0))
        graph.add_node(MissionNode(id=1, node_type=NodeType.EMPTY, position=(0, 1, 0), difficulty=0.2))
        graph.add_node(MissionNode(id=2, node_type=NodeType.GOAL, position=(0, 2, 0), difficulty=1.0))
        graph.add_edge(0, 1, EdgeType.PATH)
        graph.add_edge(1, 2, EdgeType.PATH)

        updated = CreateHubRule().apply(
            graph,
            {"rng": random.Random(11), "layout_bounds": (-8, 8, -8, 8)},
        )

        assert _positions_are_unique(updated)

    def test_validate_goal_gauntlet_rejects_boss_door_cycle(self):
        """Validation should reject BOSS -> BOSS_DOOR -> BOSS cycles masquerading as a gauntlet."""
        grammar = MissionGrammar(seed=42)
        graph = MissionGraph()
        graph.add_node(MissionNode(id=0, node_type=NodeType.START, position=(0, 0, 0), difficulty=0.0))
        graph.add_node(MissionNode(id=1, node_type=NodeType.BOSS, position=(1, 0, 0), difficulty=0.9))
        graph.add_node(MissionNode(id=2, node_type=NodeType.GOAL, position=(2, 0, 0), difficulty=1.0))
        graph.add_node(MissionNode(id=3, node_type=NodeType.BOSS_DOOR, position=(1, -1, 0), difficulty=0.8, key_id=3))
        graph.add_node(MissionNode(id=4, node_type=NodeType.BIG_KEY, position=(0, 1, 0), difficulty=0.4, key_id=3))
        graph.add_edge(0, 1, EdgeType.PATH)
        graph.add_edge(1, 2, EdgeType.PATH)
        graph.add_edge(1, 3, EdgeType.BOSS_LOCKED, key_required=3)
        graph.add_edge(3, 1, EdgeType.PATH)
        graph.add_edge(0, 4, EdgeType.PATH)
        graph.sanitize()

        assert not grammar.validate_goal_gauntlet(graph)

    def test_repair_goal_gauntlet_normalizes_missing_boss(self):
        """Repair should create a strict approach -> BOSS_DOOR -> BOSS -> GOAL chain without self-loops."""
        grammar = MissionGrammar(seed=42)
        graph = MissionGraph()
        graph.add_node(MissionNode(id=0, node_type=NodeType.START, position=(0, 0, 0), difficulty=0.0))
        graph.add_node(MissionNode(id=1, node_type=NodeType.ENEMY, position=(1, 0, 0), difficulty=0.4))
        graph.add_node(MissionNode(id=2, node_type=NodeType.BOSS_DOOR, position=(2, 0, 0), difficulty=0.9, key_id=2))
        graph.add_node(MissionNode(id=3, node_type=NodeType.GOAL, position=(3, 0, 0), difficulty=1.0))
        graph.add_edge(0, 1, EdgeType.PATH)
        graph.add_edge(1, 2, EdgeType.BOSS_LOCKED, key_required=2)
        graph.add_edge(2, 3, EdgeType.PATH)
        graph.sanitize()

        repaired = grammar._repair_goal_gauntlet(graph)
        repaired.sanitize()

        assert grammar.validate_goal_gauntlet(repaired)
        assert not any(edge.source == edge.target for edge in repaired.edges)
        boss_doors = repaired.get_nodes_by_type(NodeType.BOSS_DOOR)
        bosses = repaired.get_nodes_by_type(NodeType.BOSS)
        goals = repaired.get_nodes_by_type(NodeType.GOAL)
        assert len(boss_doors) == 1
        assert len(bosses) == 1
        assert len(goals) == 1
        assert any(edge.source == boss_doors[0].id and edge.target == bosses[0].id for edge in repaired.edges)
        assert any(edge.source == bosses[0].id and edge.target == goals[0].id for edge in repaired.edges)

    def test_repair_goal_gauntlet_normalizes_existing_boss_without_door(self):
        """Repair should not create a BOSS -> BOSS_DOOR -> BOSS cycle when only BOSS -> GOAL exists."""
        grammar = MissionGrammar(seed=42)
        graph = MissionGraph()
        graph.add_node(MissionNode(id=0, node_type=NodeType.START, position=(0, 0, 0), difficulty=0.0))
        graph.add_node(MissionNode(id=1, node_type=NodeType.BOSS, position=(1, 0, 0), difficulty=0.9))
        graph.add_node(MissionNode(id=2, node_type=NodeType.GOAL, position=(2, 0, 0), difficulty=1.0))
        graph.add_edge(0, 1, EdgeType.PATH)
        graph.add_edge(1, 2, EdgeType.PATH)
        graph.sanitize()

        repaired = grammar._repair_goal_gauntlet(graph)
        repaired.sanitize()

        assert grammar.validate_goal_gauntlet(repaired)
        assert not any(
            edge.source == 1 and getattr(repaired.nodes.get(edge.target), "node_type", None) == NodeType.BOSS_DOOR
            for edge in repaired.edges
        )
        assert not any(edge.source == edge.target for edge in repaired.edges)
    
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
        
        assert len(node_types) >= 5, f"Should have >=5 node types, got {len(node_types)}"
        assert len(edge_types) >= 3, f"Should have >=3 edge types, got {len(edge_types)}"
        
        print(f"\nGenerated graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
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
            print(f"\nFungible locks found: {len(fungible_locks)}")
            assert len(fungible_keys) >= 1, "Should have fungible keys"
            
            for lock_edge in fungible_locks:
                keys_before = graph.count_keys_available_before(lock_edge.target)
                assert keys_before >= lock_edge.requires_key_count, \
                    f"Not enough keys before lock (have {keys_before}, need {lock_edge.requires_key_count})"
        else:
            pytest.skip("Fungible lock rule not applied for this deterministic seed")
    
    def test_fungible_lock_does_not_count_big_key_as_small_key(self):
        """Boss keys should not satisfy requires_key_count small-key locks."""
        grammar = MissionGrammar(seed=42)
        graph = MissionGraph()
        graph.generation_stats["require_goal_gauntlet"] = True

        graph.add_node(MissionNode(id=0, node_type=NodeType.START, position=(0, 0, 0), difficulty=0.0))
        graph.add_node(MissionNode(id=1, node_type=NodeType.BIG_KEY, position=(1, 0, 0), difficulty=0.4, key_id=2))
        graph.add_node(MissionNode(id=2, node_type=NodeType.BOSS_DOOR, position=(2, 0, 0), difficulty=0.9, key_id=2))
        graph.add_node(MissionNode(id=3, node_type=NodeType.BOSS, position=(3, 0, 0), difficulty=0.95))
        graph.add_node(MissionNode(id=4, node_type=NodeType.GOAL, position=(4, 0, 0), difficulty=1.0))
        graph.add_node(MissionNode(id=5, node_type=NodeType.ENEMY, position=(1, 1, 0), difficulty=0.4))
        graph.add_node(MissionNode(id=6, node_type=NodeType.ENEMY, position=(2, 1, 0), difficulty=0.5))

        graph.add_edge(0, 1, EdgeType.PATH)
        graph.add_edge(0, 5, EdgeType.PATH)
        graph.add_edge(5, 2, EdgeType.BOSS_LOCKED, key_required=2)
        graph.add_edge(2, 3, EdgeType.PATH)
        graph.add_edge(3, 4, EdgeType.PATH)

        fungible_lock = MissionEdge(source=5, target=6, edge_type=EdgeType.LOCKED)
        fungible_lock.requires_key_count = 1
        graph.edges.append(fungible_lock)
        graph.sanitize()

        assert not grammar.validate_progression_constraints(graph)

    def test_big_room_merging(self):
        """Test big room formation (RULE #2)."""
        grammar = MissionGrammar(seed=123)
        graph = grammar.generate(num_rooms=15)
        
        big_rooms = [n for n in graph.nodes.values() if n.is_big_room]
        
        if big_rooms:
            print(f"\nBig rooms found: {len(big_rooms)}")
            
            for room in big_rooms:
                assert room.room_size != (1, 1), "Big room should have non-default size"
                assert room.room_size in [(2,1), (1,2), (2,2)], \
                    f"Invalid big room size: {room.room_size}"
                print(f"   Room {room.id}: size {room.room_size}")
        else:
            pytest.skip("Big room rule not applied for this deterministic seed")
    
    def test_cycle_valves(self):
        """Test one-way valves in cycles (RULE #3)."""
        grammar = MissionGrammar(seed=456)
        graph = grammar.generate(num_rooms=18)
        
        cycles = graph.detect_cycles()
        one_way_edges = [e for e in graph.edges if e.edge_type == EdgeType.ONE_WAY]
        
        print(f"\n   Cycles detected: {len(cycles)}")
        print(f"   One-way edges: {len(one_way_edges)}")
        
        if cycles and one_way_edges:
            print("Valve rule applied successfully")
            
            # Verify asymmetry
            valve = one_way_edges[0]
            assert valve.target in graph._adjacency.get(valve.source, []), \
                "Forward path should exist"
            
            if valve.source in graph._adjacency.get(valve.target, []):
                raise AssertionError(f"Valve {valve.source}->{valve.target} still bidirectional")
        else:
            pytest.skip("Valve rule not applied for this deterministic seed")
    
    def test_visual_foreshadowing(self):
        """Test visual links (RULE #4)."""
        grammar = MissionGrammar(seed=789)
        graph = grammar.generate(num_rooms=20)
        
        visual_links = [e for e in graph.edges if e.edge_type == EdgeType.VISUAL_LINK]
        
        if visual_links:
            print(f"\nVisual links found: {len(visual_links)}")
            
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
                
                print(f"   Link {link.source}->{link.target}: spatial={manhattan}, path={path_dist}")
        else:
            pytest.skip("Visual link rule not applied for this deterministic seed")
    
    def test_collection_challenge(self):
        """Test token collection system (RULE #5)."""
        grammar = MissionGrammar(seed=111)
        graph = grammar.generate(num_rooms=20)
        
        tokens = [n for n in graph.nodes.values() if n.node_type == NodeType.TOKEN]
        multi_locks = [e for e in graph.edges if e.edge_type == EdgeType.MULTI_LOCK]
        
        if multi_locks:
            print("\nCollection challenge found")
            print(f"   Tokens: {len(tokens)}")
            print(f"   Multi-locks: {len(multi_locks)}")
            
            for lock in multi_locks:
                assert lock.token_count >= 2, "Multi-lock should require >=2 tokens"
                print(f"   Lock at {lock.source}->{lock.target} requires {lock.token_count} tokens")
            
            # Verify sufficient tokens exist
            if tokens:
                assert len(tokens) >= multi_locks[0].token_count, \
                    "Not enough tokens for multi-lock"
        else:
            pytest.skip("Collection challenge rule not applied for this deterministic seed")
    
    def test_combat_arenas(self):
        """Test arena rooms with shutters (RULE #6)."""
        grammar = MissionGrammar(seed=222)
        graph = grammar.generate(num_rooms=15)
        
        arenas = [n for n in graph.nodes.values() if n.is_arena]
        shutters = [e for e in graph.edges if e.edge_type == EdgeType.SHUTTER]
        
        if arenas:
            print(f"\nArenas found: {len(arenas)}")
            print(f"   Shutter edges: {len(shutters)}")
            
            for arena in arenas:
                incoming_shutters = [e for e in shutters if e.target == arena.id]
                print(f"   Arena {arena.id}: {len(incoming_shutters)} shutters")
        else:
            pytest.skip("Arena rule not applied for this deterministic seed")
    
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
            print(f"\nSectors found: {len(sectors)}")
            
            for sector_id, nodes in sectors.items():
                themes = [n.sector_theme for n in nodes if n.sector_theme]
                theme = themes[0] if themes else "NONE"
                print(f"   Sector {sector_id} ({theme}): {len(nodes)} nodes")
                
                # Verify theme consistency
                if themes:
                    assert len(set(themes)) == 1, \
                        f"Sector {sector_id} has inconsistent themes: {set(themes)}"
        else:
            pytest.skip("Sector rule not applied for this deterministic seed")
    
    def test_entangled_branches(self):
        """Test cross-branch dependencies (RULE #8)."""
        grammar = MissionGrammar(seed=444)
        graph = grammar.generate(num_rooms=22)
        
        switches = [n for n in graph.nodes.values() if n.node_type == NodeType.SWITCH]
        state_blocks = [e for e in graph.edges if e.edge_type == EdgeType.STATE_BLOCK]
        
        if switches and state_blocks:
            print("\nEntangled branches detected")
            print(f"   Switches: {len(switches)}")
            print(f"   State blocks: {len(state_blocks)}")
            
            # Verify switch-gate pairs
            for block in state_blocks:
                if block.switch_id:
                    matching_switches = [s for s in switches if s.switch_id == block.switch_id]
                    assert len(matching_switches) > 0, \
                        f"State block references non-existent switch {block.switch_id}"
        else:
            pytest.skip("Entangled branches rule not applied for this deterministic seed")
    
    def test_hazard_gates(self):
        """Test hazard paths with protection (RULE #9)."""
        grammar = MissionGrammar(seed=555)
        graph = grammar.generate(num_rooms=16)
        
        hazards = [e for e in graph.edges if e.edge_type == EdgeType.HAZARD]
        protections = [n for n in graph.nodes.values() if n.node_type == NodeType.PROTECTION_ITEM]
        
        if hazards:
            print(f"\nHazards found: {len(hazards)}")
            print(f"   Protection items: {len(protections)}")
            
            for hazard in hazards:
                assert hazard.hazard_damage > 0, "Hazard should have damage"
                assert hazard.protection_item_id is not None, \
                    "Hazard should reference protection item"
                print(f"   Hazard {hazard.source}->{hazard.target}: " + 
                      f"damage={hazard.hazard_damage}, " +
                      f"protection={hazard.protection_item_id}")
                
                # Check if matching protection exists
                matching = [p for p in protections 
                           if p.item_type == hazard.protection_item_id]
                assert matching, "No matching protection item found"
        else:
            pytest.skip("Hazard gate rule not applied for this deterministic seed")
    
    def test_virtual_room_layers(self):
        """Test virtual room layering (RULE #10)."""
        grammar = MissionGrammar(seed=666)
        graph = grammar.generate(num_rooms=14)
        
        layered_nodes = [n for n in graph.nodes.values() if n.virtual_layer > 0]
        
        if layered_nodes:
            print(f"\nVirtual layers found: {len(layered_nodes)}")
            
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
            pytest.skip("Virtual layer rule not applied for this deterministic seed")
    
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
        
        print("\n📊 Advanced Features Summary:")
        total = 0
        for feature, count in features.items():
            if count > 0:
                print(f"   ✓ {feature}: {count}")
                total += count
        
        # Should have multiple types of advanced features
        feature_types = len([v for v in features.values() if v > 0])
        assert feature_types >= 2, \
            f"Should have >=2 types of advanced features (got {feature_types})"
        
        print(f"\nTotal advanced features: {total} ({feature_types} types)")


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
                f"Self-loop detected: {edge.source} -> {edge.target}"
    
    def test_add_valve_rule_preserves_string_node_ids(self):
        """Valve insertion should not require integer-coercible node IDs."""
        graph = MissionGraph()
        graph.add_node(MissionNode(id="start", node_type=NodeType.START, position=(0, 0, 0)))
        graph.add_node(MissionNode(id="a", node_type=NodeType.EMPTY, position=(1, 0, 0)))
        graph.add_node(MissionNode(id="b", node_type=NodeType.EMPTY, position=(2, 0, 0)))
        graph.add_node(MissionNode(id="c", node_type=NodeType.EMPTY, position=(1, 1, 0)))
        graph.add_node(MissionNode(id="goal", node_type=NodeType.GOAL, position=(3, 0, 0)))

        graph.add_edge("start", "a", EdgeType.PATH)
        graph.add_edge("a", "b", EdgeType.PATH)
        graph.add_edge("b", "goal", EdgeType.PATH)
        graph.add_edge("b", "c", EdgeType.PATH)
        graph.edges.append(MissionEdge(source="c", target="a", edge_type=EdgeType.PATH))
        graph._adjacency["c"].append("a")
        graph._adjacency["a"].append("c")

        updated = AddValveRule().apply(graph, {"rng": random.Random(0)})

        one_way_edges = [edge for edge in updated.edges if edge.edge_type == EdgeType.ONE_WAY]
        assert one_way_edges, "Expected AddValveRule to convert one cycle edge to ONE_WAY"
        valve = one_way_edges[0]
        assert valve.target in updated._adjacency.get(valve.source, [])
        assert valve.source not in updated._adjacency.get(valve.target, [])

    def test_add_valve_rule_rejects_gated_return_loop(self):
        graph = MissionGraph()
        graph.add_node(MissionNode(id=0, node_type=NodeType.START, position=(0, 0, 0)))
        graph.add_node(MissionNode(id=1, node_type=NodeType.EMPTY, position=(1, 0, 0)))
        graph.add_node(MissionNode(id=2, node_type=NodeType.EMPTY, position=(2, 0, 0)))
        graph.add_node(MissionNode(id=3, node_type=NodeType.EMPTY, position=(2, 1, 0)))
        graph.add_node(MissionNode(id=4, node_type=NodeType.GOAL, position=(3, 0, 0)))

        graph.add_edge(0, 1, EdgeType.PATH)
        graph.add_edge(1, 2, EdgeType.PATH)
        graph.add_edge(2, 4, EdgeType.PATH)
        graph.add_edge(2, 3, EdgeType.PATH)
        graph.edges.append(MissionEdge(source=3, target=1, edge_type=EdgeType.ITEM_GATE, item_required="BOMB"))
        graph._adjacency[3].append(1)

        updated = AddValveRule().apply(graph, {"rng": random.Random(0)})

        assert not any(edge.edge_type == EdgeType.ONE_WAY for edge in updated.edges)

    def test_add_secret_rule_uses_bounded_non_colliding_relative_positions(self):
        graph = MissionGraph()
        graph.add_node(MissionNode(id=0, node_type=NodeType.START, position=(0, 0, 0)))
        graph.add_node(MissionNode(id=1, node_type=NodeType.EMPTY, position=(5, 5, 0)))
        graph.add_node(MissionNode(id=2, node_type=NodeType.GOAL, position=(8, 8, 0)))
        graph.add_edge(0, 1, EdgeType.PATH)
        graph.add_edge(1, 2, EdgeType.PATH)
        occupied_offsets = [(-1, 2), (0, 2), (1, 2), (-1, 3), (0, 3), (1, 3), (2, 1), (-2, 1)]
        next_id = 3
        for dr, dc in occupied_offsets:
            graph.add_node(
                MissionNode(
                    id=next_id,
                    node_type=NodeType.SECRET,
                    position=(5 + dr, 5 + dc, 0),
                    is_secret=True,
                )
            )
            next_id += 1

        updated = AddSecretRule().apply(
            graph,
            {"rng": random.Random(1), "layout_bounds": (0, 8, 0, 8), "difficulty": 0.5},
        )
        positions = [node.position for node in updated.nodes.values()]

        assert len(positions) == len(set(positions))
        for row, col, _floor in positions:
            assert 0 <= row <= 8
            assert 0 <= col <= 8

    def test_start_and_goal_exist(self):
        """Test that START and GOAL nodes always exist."""
        grammar = MissionGrammar(seed=1111)
        graph = grammar.generate(num_rooms=10)
        
        start = graph.get_start_node()
        goal = graph.get_goal_node()
        
        assert start is not None, "START node missing"
        assert goal is not None, "GOAL node missing"
        
        print(f"\nStart: {start.id}, Goal: {goal.id}")


def test_quick_smoke_test():
    """Quick smoke test for CI/CD."""
    grammar = MissionGrammar(seed=12345)
    graph = grammar.generate(num_rooms=8)
    
    assert len(graph.nodes) >= 2  # At least START and GOAL
    assert len(graph.edges) >= 1
    
    print("\nQuick smoke test passed")


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
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("="*70)
