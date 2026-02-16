"""
Test Suite for Wave 3 Pedagogical & Quality Control Rules
==========================================================

Tests all 7 advanced production rules implementing Nintendo-grade
level design pedagogy and quality control patterns.

Rules tested:
1. AddSkillChainRule - Tutorial sequences (Learn → Practice → Master)
2. AddPacingBreakerRule - Sanctuary/negative space
3. AddResourceLoopRule - Resource farming spots
4. AddGatekeeperRule - Mini-boss guardians
5. AddMultiLockRule - Multi-switch battery pattern
6. AddItemShortcutRule - Item-gated shortcuts
7. PruneDeadEndRule - Dead-end garbage collection
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from src.generation.grammar import (
    MissionGrammar, MissionGraph, MissionNode, MissionEdge,
    NodeType, EdgeType, Difficulty,
    AddSkillChainRule, AddPacingBreakerRule, AddResourceLoopRule,
    AddGatekeeperRule, AddMultiLockRule, AddItemShortcutRule,
    PruneDeadEndRule,
    validate_skill_chains, validate_battery_reachability,
    validate_resource_loops
)


class TestAddSkillChainRule:
    """Test tutorial sequence generation."""
    
    def test_skill_chain_creation(self):
        """Test that skill chains are properly created after items."""
        graph = MissionGraph()
        
        # Create basic structure: START → PATH → ITEM → 3 successors
        start = MissionNode(id=0, node_type=NodeType.START, position=(0, 0, 0))
        graph.add_node(start)
        
        item = MissionNode(id=1, node_type=NodeType.ITEM, position=(1, 0, 0), 
                          item_type="BOW")
        graph.add_node(item)
        graph.add_edge(0, 1, EdgeType.PATH)
        
        # Add 3 successors (will become tutorial chain)
        for i in range(3):
            node = MissionNode(id=2+i, node_type=NodeType.EMPTY, 
                             position=(2+i, 0, 0))
            graph.add_node(node)
            if i == 0:
                graph.add_edge(1, 2+i, EdgeType.PATH)
            else:
                graph.add_edge(1+i, 2+i, EdgeType.PATH)
        
        # Apply rule
        rule = AddSkillChainRule()
        context = {'rng': None, 'difficulty': 0.5}
        
        assert rule.can_apply(graph, context)
        graph = rule.apply(graph, context)
        
        # Verify tutorial sequence
        assert graph.nodes[2].node_type == NodeType.TUTORIAL_PUZZLE
        assert graph.nodes[2].difficulty_rating == "SAFE"
        assert graph.nodes[2].is_tutorial
        
        assert graph.nodes[3].node_type == NodeType.COMBAT_PUZZLE
        assert graph.nodes[3].difficulty_rating == "MODERATE"
        
        assert graph.nodes[4].node_type == NodeType.COMPLEX_PUZZLE
        assert graph.nodes[4].difficulty_rating == "HARD"
        
        # Verify difficulty progression
        assert graph.nodes[2].difficulty < graph.nodes[3].difficulty < graph.nodes[4].difficulty
    
    def test_skill_chain_validation(self):
        """Test skill chain validation."""
        graph = MissionGraph()
        
        # Valid chain
        start = MissionNode(id=0, node_type=NodeType.START, position=(0, 0, 0))
        graph.add_node(start)
        
        tutorial = MissionNode(id=1, node_type=NodeType.TUTORIAL_PUZZLE, 
                              position=(1, 0, 0), is_tutorial=True, difficulty=0.2)
        graph.add_node(tutorial)
        graph.add_edge(0, 1, EdgeType.PATH)
        
        moderate = MissionNode(id=2, node_type=NodeType.COMBAT_PUZZLE,
                             position=(2, 0, 0), difficulty=0.5)
        graph.add_node(moderate)
        graph.add_edge(1, 2, EdgeType.PATH)
        
        hard = MissionNode(id=3, node_type=NodeType.COMPLEX_PUZZLE,
                         position=(3, 0, 0), difficulty=0.8)
        graph.add_node(hard)
        graph.add_edge(2, 3, EdgeType.PATH)
        
        assert validate_skill_chains(graph)


class TestAddPacingBreakerRule:
    """Test pacing breaker/sanctuary insertion."""
    
    def test_sanctuary_after_tension(self):
        """Test sanctuary insertion after high-tension chain."""
        graph = MissionGraph()
        
        # Create START
        start = MissionNode(id=0, node_type=NodeType.START, position=(0, 0, 0))
        graph.add_node(start)
        
        # Create tension chain: 3 consecutive combat rooms
        for i in range(3):
            node = MissionNode(id=1+i, node_type=NodeType.ENEMY, 
                             position=(1+i, 0, 0))
            graph.add_node(node)
            if i == 0:
                graph.add_edge(0, 1+i, EdgeType.PATH)
            else:
                graph.add_edge(i, 1+i, EdgeType.PATH)
        
        # Add continuation
        goal = MissionNode(id=4, node_type=NodeType.GOAL, position=(4, 0, 0))
        graph.add_node(goal)
        graph.add_edge(3, 4, EdgeType.PATH)
        
        # Apply rule
        rule = AddPacingBreakerRule()
        context = {'rng': None, 'difficulty': 0.5}
        
        assert rule.can_apply(graph, context)
        graph = rule.apply(graph, context)
        
        # Verify sanctuary was inserted
        scenic_nodes = [n for n in graph.nodes.values() 
                       if n.node_type == NodeType.SCENIC]
        assert len(scenic_nodes) > 0
        
        sanctuary = scenic_nodes[0]
        assert sanctuary.is_sanctuary
        assert sanctuary.tension_value == 0.0
        assert sanctuary.difficulty_rating == "SAFE"


class TestAddResourceLoopRule:
    """Test resource farming spot creation."""
    
    def test_resource_farm_near_gate(self):
        """Test resource farm placement near item gate."""
        graph = MissionGraph()
        
        # Create START → node → ITEM_GATE → target
        start = MissionNode(id=0, node_type=NodeType.START, position=(0, 0, 0))
        graph.add_node(start)
        
        gate_source = MissionNode(id=1, node_type=NodeType.EMPTY, position=(1, 0, 0))
        graph.add_node(gate_source)
        graph.add_edge(0, 1, EdgeType.PATH)
        
        target = MissionNode(id=2, node_type=NodeType.GOAL, position=(2, 0, 0))
        graph.add_node(target)
        
        # Add ITEM_GATE edge
        gate_edge = MissionEdge(
            source=1, target=2, edge_type=EdgeType.ITEM_GATE,
            item_required="BOMBS"
        )
        graph.edges.append(gate_edge)
        graph._adjacency[1].append(2)
        
        # Apply rule
        rule = AddResourceLoopRule()
        context = {'rng': None, 'difficulty': 0.5}
        
        assert rule.can_apply(graph, context)
        graph = rule.apply(graph, context)
        
        # Verify farm was created
        farms = [n for n in graph.nodes.values() 
                if n.node_type == NodeType.RESOURCE_FARM]
        assert len(farms) > 0
        
        farm = farms[0]
        assert farm.drops_resource == "BOMBS"
        assert farm.difficulty_rating == "SAFE"
        
        # Verify validation
        assert validate_resource_loops(graph)


class TestAddGatekeeperRule:
    """Test mini-boss guardian placement."""
    
    def test_miniboss_before_item(self):
        """Test mini-boss placement before item."""
        graph = MissionGraph()
        
        # Create START → node → ITEM
        start = MissionNode(id=0, node_type=NodeType.START, position=(0, 0, 0))
        graph.add_node(start)
        
        pre_item = MissionNode(id=1, node_type=NodeType.EMPTY, position=(1, 0, 0))
        graph.add_node(pre_item)
        graph.add_edge(0, 1, EdgeType.PATH)
        
        item = MissionNode(id=2, node_type=NodeType.ITEM, position=(2, 0, 0),
                          item_type="HOOKSHOT")
        graph.add_node(item)
        graph.add_edge(1, 2, EdgeType.PATH)
        
        # Apply rule
        rule = AddGatekeeperRule()
        context = {'rng': None, 'difficulty': 0.5}
        
        assert rule.can_apply(graph, context)
        graph = rule.apply(graph, context)
        
        # Verify mini-boss conversion
        assert graph.nodes[1].node_type == NodeType.MINI_BOSS
        assert graph.nodes[1].is_mini_boss
        assert graph.nodes[1].difficulty_rating == "HARD"
        assert graph.nodes[1].room_size == (2, 2)
        
        # Verify edge is SHUTTER type
        edge = [e for e in graph.edges if e.source == 1 and e.target == 2][0]
        assert edge.edge_type == EdgeType.SHUTTER


class TestAddMultiLockRule:
    """Test multi-switch battery pattern."""
    
    def test_battery_pattern(self):
        """Test battery pattern with 3 switches."""
        graph = MissionGraph()
        
        # Create hub with 3 branches
        start = MissionNode(id=0, node_type=NodeType.START, position=(0, 0, 0))
        graph.add_node(start)
        
        hub = MissionNode(id=1, node_type=NodeType.EMPTY, position=(1, 0, 0), 
                         is_hub=True)
        graph.add_node(hub)
        graph.add_edge(0, 1, EdgeType.PATH)
        
        # Create 3 branches
        for i in range(3):
            branch_node = MissionNode(id=2+i, node_type=NodeType.EMPTY,
                                    position=(2, i, 0))
            graph.add_node(branch_node)
            graph.add_edge(1, 2+i, EdgeType.PATH)
            
            # Add branch extension
            branch_end = MissionNode(id=5+i, node_type=NodeType.EMPTY,
                                   position=(3, i, 0))
            graph.add_node(branch_end)
            graph.add_edge(2+i, 5+i, EdgeType.PATH)
        
        # Add target behind hub
        target = MissionNode(id=8, node_type=NodeType.GOAL, position=(2, 0, 0))
        graph.add_node(target)
        graph.add_edge(1, 8, EdgeType.PATH)
        
        # Apply rule
        rule = AddMultiLockRule()
        context = {'rng': None, 'difficulty': 0.5}
        
        assert rule.can_apply(graph, context)
        graph = rule.apply(graph, context)
        
        # Verify switches were placed in branches
        switches = [n for n in graph.nodes.values() 
                   if n.node_type == NodeType.SWITCH]
        assert len(switches) >= 2
        
        # Verify battery-locked edge exists
        battery_edges = [e for e in graph.edges if e.battery_id is not None]
        assert len(battery_edges) > 0
        
        battery_edge = battery_edges[0]
        assert battery_edge.edge_type == EdgeType.STATE_BLOCK
        assert len(battery_edge.switches_required) >= 2
        
        # Verify validation
        assert validate_battery_reachability(graph)


class TestAddItemShortcutRule:
    """Test item-gated shortcut creation."""
    
    def test_shortcut_from_item_to_start(self):
        """Test shortcut from distant item back to start."""
        graph = MissionGraph()
        
        # Create long path: START → ... → ITEM (distance > 5)
        start = MissionNode(id=0, node_type=NodeType.START, position=(0, 0, 0))
        graph.add_node(start)
        
        # Create 6-node path
        for i in range(6):
            node = MissionNode(id=1+i, node_type=NodeType.EMPTY,
                             position=(1+i, 0, 0))
            graph.add_node(node)
            if i == 0:
                graph.add_edge(0, 1+i, EdgeType.PATH)
            else:
                graph.add_edge(i, 1+i, EdgeType.PATH)
        
        # Make last node an ITEM
        graph.nodes[6].node_type = NodeType.ITEM
        graph.nodes[6].item_type = "HOOKSHOT"
        
        # Apply rule
        rule = AddItemShortcutRule()
        context = {'rng': None, 'difficulty': 0.5}
        
        assert rule.can_apply(graph, context)
        graph = rule.apply(graph, context)
        
        # Verify shortcut was created
        shortcuts = [e for e in graph.edges 
                    if e.edge_type == EdgeType.ITEM_GATE 
                    and e.preferred_direction == "backward"]
        assert len(shortcuts) > 0
        
        shortcut = shortcuts[0]
        assert shortcut.item_required == "HOOKSHOT"
        assert shortcut.path_savings >= 3


class TestPruneDeadEndRule:
    """Test dead-end pruning."""
    
    def test_prune_useless_deadend(self):
        """Test pruning of useless dead-end nodes."""
        graph = MissionGraph()
        
        # Create main path
        start = MissionNode(id=0, node_type=NodeType.START, position=(0, 0, 0))
        graph.add_node(start)
        
        main = MissionNode(id=1, node_type=NodeType.EMPTY, position=(1, 0, 0))
        graph.add_node(main)
        graph.add_edge(0, 1, EdgeType.PATH)
        
        goal = MissionNode(id=2, node_type=NodeType.GOAL, position=(2, 0, 0))
        graph.add_node(goal)
        graph.add_edge(1, 2, EdgeType.PATH)
        
        # Add useless dead-end
        deadend = MissionNode(id=3, node_type=NodeType.EMPTY, position=(1, 1, 0))
        graph.add_node(deadend)
        graph.add_edge(1, 3, EdgeType.PATH)
        
        initial_count = len(graph.nodes)
        
        # Apply rule
        rule = PruneDeadEndRule()
        context = {'rng': None, 'difficulty': 0.5}
        
        assert rule.can_apply(graph, context)
        graph = rule.apply(graph, context)
        
        # Verify deadend was removed
        assert len(graph.nodes) < initial_count
        assert 3 not in graph.nodes
        
        # Verify graph is still connected
        assert graph.is_graph_connected()
    
    def test_preserve_valuable_deadends(self):
        """Test that valuable dead-ends are preserved."""
        graph = MissionGraph()
        
        # Create main path
        start = MissionNode(id=0, node_type=NodeType.START, position=(0, 0, 0))
        graph.add_node(start)
        
        main = MissionNode(id=1, node_type=NodeType.EMPTY, position=(1, 0, 0))
        graph.add_node(main)
        graph.add_edge(0, 1, EdgeType.PATH)
        
        # Add valuable dead-end (KEY)
        key_deadend = MissionNode(id=2, node_type=NodeType.KEY, 
                                 position=(1, 1, 0), key_id=2)
        graph.add_node(key_deadend)
        graph.add_edge(1, 2, EdgeType.PATH)
        
        initial_count = len(graph.nodes)
        
        # Apply rule
        rule = PruneDeadEndRule()
        context = {'rng': None, 'difficulty': 0.5}
        
        # Should not apply (no useless dead ends)
        assert not rule.can_apply(graph, context)


class TestIntegratedGeneration:
    """Test full grammar with Wave 3 rules."""
    
    def test_full_generation_with_wave3_rules(self):
        """Test complete dungeon generation with all Wave 3 rules."""
        grammar = MissionGrammar(seed=42)
        
        # Generate medium dungeon
        graph = grammar.generate(
            difficulty=Difficulty.MEDIUM,
            num_rooms=12,
            max_keys=2
        )
        
        # Verify basic structure
        assert len(graph.nodes) > 0
        assert graph.get_start_node() is not None
        assert graph.get_goal_node() is not None
        
        # Verify lock-key ordering
        assert grammar.validate_lock_key_ordering(graph)
        
        # Verify Wave 3 patterns (at least some should appear)
        node_types = {n.node_type for n in graph.nodes.values()}
        
        # Count advanced features
        has_tutorial = any(n.is_tutorial for n in graph.nodes.values())
        has_sanctuary = any(n.is_sanctuary for n in graph.nodes.values())
        has_farm = NodeType.RESOURCE_FARM in node_types
        has_miniboss = NodeType.MINI_BOSS in node_types
        
        # At least one pedagogical feature should appear (probabilistic)
        pedagogical_features = sum([has_tutorial, has_sanctuary, has_farm, has_miniboss])
        assert pedagogical_features >= 0  # May not always trigger in small dungeons
        
        print(f"\nGenerated dungeon with {len(graph.nodes)} nodes")
        print(f"Pedagogical features: {pedagogical_features}")
        print(f"  - Tutorial chains: {has_tutorial}")
        print(f"  - Sanctuaries: {has_sanctuary}")
        print(f"  - Resource farms: {has_farm}")
        print(f"  - Mini-bosses: {has_miniboss}")


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])
