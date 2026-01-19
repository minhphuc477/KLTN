"""
Test solver inventory functionality.

Tests the state-space solver's ability to handle key collection
and door unlocking mechanics.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
import pytest

# Import from canonical source
from Data.zelda_core import StateSpaceGraphSolver, InventoryState, ValidationMode


def test_solve_with_inventory_key_opens_door():
    """Test that solver can collect keys and open locked doors."""
    G = nx.DiGraph()
    
    # Create nodes with labels
    G.add_node(0, label='s', is_start=True)  # start
    G.add_node(1, label='k', has_key=True)   # key room
    G.add_node(2, label='')                   # locked room
    G.add_node(3, label='t', is_triforce=True)  # goal
    
    # Add edges
    G.add_edge(0, 1, label='')    # start -> key room (open)
    G.add_edge(1, 2, label='')    # key room -> before locked (open)
    G.add_edge(2, 3, label='k')   # before locked -> goal (key_locked)
    
    # Also add reverse edges for bidirectional traversal
    G.add_edge(1, 0, label='')
    G.add_edge(2, 1, label='')
    G.add_edge(3, 2, label='k')
    
    solver = StateSpaceGraphSolver(G, mode=ValidationMode.FULL)
    result = solver.solve(0, 3)
    
    assert result['solvable'] is True
    assert 1 in result['path']  # Should visit key room
    assert result['keys_used'] == 1  # Should use one key


def test_solve_without_key_fails():
    """Test that solver fails when key is required but not available."""
    G = nx.DiGraph()
    
    # Create nodes WITHOUT key
    G.add_node(0, label='s', is_start=True)
    G.add_node(1, label='')  # No key here
    G.add_node(2, label='t', is_triforce=True)
    
    # Add edges - direct path requires key
    G.add_edge(0, 1, label='')
    G.add_edge(1, 2, label='k')  # key_locked
    G.add_edge(1, 0, label='')
    G.add_edge(2, 1, label='k')
    
    solver = StateSpaceGraphSolver(G, mode=ValidationMode.FULL)
    result = solver.solve(0, 2)
    
    assert result['solvable'] is False


def test_strict_mode_ignores_locked():
    """Test STRICT mode ignores locked doors."""
    G = nx.DiGraph()
    
    G.add_node(0, label='s', is_start=True)
    G.add_node(1, label='k', has_key=True)
    G.add_node(2, label='t', is_triforce=True)
    
    G.add_edge(0, 1, label='')
    G.add_edge(1, 2, label='k')  # key_locked - should be blocked in STRICT
    
    solver = StateSpaceGraphSolver(G, mode=ValidationMode.STRICT)
    result = solver.solve(0, 2)
    
    assert result['solvable'] is False  # STRICT mode can't pass locked doors


def test_bombable_wall():
    """Test that bombable walls can be opened."""
    G = nx.DiGraph()
    
    G.add_node(0, label='s', is_start=True)
    G.add_node(1, label='t', is_triforce=True)
    
    G.add_edge(0, 1, label='b')  # bombable
    G.add_edge(1, 0, label='b')
    
    solver = StateSpaceGraphSolver(G, mode=ValidationMode.FULL)
    result = solver.solve(0, 1)
    
    assert result['solvable'] is True  # Bombs are assumed infinite in FULL mode