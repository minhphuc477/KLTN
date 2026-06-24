"""
VGLC Graph Utilities
====================

Utility functions for VGLC-compliant graph topology handling.

This module provides:
- Virtual node filtering (remove meta-nodes before layout)
- Composite node label parsing
- Edge type parsing
- Boss-Goal subgraph validation
- Physical start node identification

All functions are designed to work with NetworkX graphs that follow
the VGLC DOT format specification.

Usage:
    import networkx as nx
    from src.utils.graph_utils import filter_virtual_nodes, validate_goal_subgraph
    
    # Remove virtual nodes before layout
    G_physical, virtual_info = filter_virtual_nodes(mission_graph)
    
    # Validate Boss-Goal pattern
    is_valid, errors = validate_goal_subgraph(mission_graph)
    if not is_valid:
        print(f"Validation failed: {errors}")
"""

import logging
from typing import Dict, List, Set, Tuple, Optional, Any

import networkx as nx

from src.constants.vglc_constants import (
    VIRTUAL_NODE_TYPES,
    GOAL_NODE_MAX_DEGREE,
    GOAL_CONNECTS_TO_BOSS,
    BOSS_REQUIRED_FOR_GOAL,
    MIN_PATH_LENGTH_START_TO_GOAL,
    NODE_TYPE_MAP,
    parse_composite_node_label,
    parse_edge_label,
)

logger = logging.getLogger(__name__)


_NODE_SEMANTIC_ALIASES = {
    "s": "start_pointer",
    "start_pointer": "start_pointer",
    "start": "start",
    "goal": "triforce",
    "triforce": "triforce",
    "t": "triforce",
    "boss": "boss",
    "b": "boss",
    "boss_door": "boss_door",
    "boss-door": "boss_door",
    "bossdoor": "boss_door",
    "big_key": "boss_key",
    "bigkey": "boss_key",
    "boss_key": "boss_key",
    "bosskey": "boss_key",
    "k": "key",
    "key": "key",
    "enemy": "enemy",
    "e": "enemy",
    "mini_boss": "mini_boss",
    "miniboss": "mini_boss",
    "m": "mini_boss",
    "switch": "switch",
    "s1": "switch",
    "puzzle": "puzzle",
    "p": "puzzle",
    "item": "macro_item",
    "macro_item": "macro_item",
    "minor_item": "minor_item",
    "treasure": "minor_item",
    "i": "minor_item",
    "i_macro": "macro_item",
}


def _canonicalize_node_semantics(*raw_values: Any) -> Set[str]:
    """Normalize legacy DOT labels and exported graph attrs to shared semantic tags."""
    semantics: Set[str] = set()
    for raw in raw_values:
        if raw is None:
            continue
        if isinstance(raw, (list, tuple, set)):
            for item in raw:
                semantics.update(_canonicalize_node_semantics(item))
            continue
        text = str(raw).strip()
        if not text:
            continue
        for token in text.split(","):
            token = token.strip()
            if not token:
                continue
            if token in NODE_TYPE_MAP:
                semantics.add(NODE_TYPE_MAP[token])
                continue
            lowered = token.lower()
            semantics.add(_NODE_SEMANTIC_ALIASES.get(lowered, lowered))
    return semantics


def _node_semantics(G: nx.Graph, node: Any) -> Set[str]:
    """Return canonical semantic tags for a node across legacy and exported attrs."""
    data = G.nodes[node]
    semantics = _canonicalize_node_semantics(
        data.get("label", ""),
        data.get("type", ""),
    )
    if bool(data.get("is_goal")) or bool(data.get("is_triforce")):
        semantics.add("triforce")
    if bool(data.get("is_boss")):
        semantics.add("boss")
    if bool(data.get("is_start")):
        semantics.add("start")
    return semantics


def _has_semantic(G: nx.Graph, node: Any, semantic: str) -> bool:
    return semantic in _node_semantics(G, node)


def _find_nodes_with_semantic(G: nx.Graph, semantic: str) -> List[Any]:
    return [node for node in G.nodes() if _has_semantic(G, node, semantic)]


# ==========================================
# VIRTUAL NODE FILTERING
# ==========================================

def filter_virtual_nodes(G: nx.Graph) -> Tuple[nx.Graph, Dict[str, Any]]:
    """
    Remove virtual nodes (e.g., start pointer 's') from graph.
    
    Virtual nodes are meta-nodes used for graph structure but should NOT
    be placed on the physical dungeon grid. The most common virtual node
    is 's' (start pointer), which points to the actual physical start room.
    
    Args:
        G: NetworkX graph with node attributes (must have 'label' attribute)
        
    Returns:
        Tuple of:
        - G_physical: Graph with only physical nodes
        - virtual_info: Dict with keys:
            - 'removed_nodes': List of removed node IDs
            - 'virtual_types': Dict mapping removed nodes to their types
            - 'physical_start': ID of physical start room (successor of 's')
            - 'successors': Dict mapping virtual nodes to their successors
            
    Example:
        >>> G = nx.DiGraph()
        >>> G.add_node(0, label='s')        # Virtual start pointer
        >>> G.add_node(1, label='e')        # Physical enemy room
        >>> G.add_edge(0, 1)
        >>> G_physical, info = filter_virtual_nodes(G)
        >>> 0 not in G_physical.nodes()
        True
        >>> info['physical_start']
        1
    """
    G_physical = G.copy()
    virtual_info = {
        'removed_nodes': [],
        'virtual_types': {},
        'physical_start': None,
        'successors': {},
    }
    
    # Find and remove virtual nodes
    for node in list(G_physical.nodes()):
        node_data = G_physical.nodes[node]
        node_label = node_data.get('label', '')
        
        # Parse composite labels (e.g., "e,k,p")
        node_types = parse_composite_node_label(node_label)
        
        # Get raw codes for virtual check
        label_codes = set(code.strip() for code in str(node_label).split(','))
        
        # Check if any type is virtual
        if label_codes & VIRTUAL_NODE_TYPES:
            # Record node info
            virtual_info['removed_nodes'].append(node)
            virtual_info['virtual_types'][node] = list(node_types)
            
            # Get successors (physical rooms this virtual node points to)
            if G_physical.is_directed():
                successors = list(G_physical.successors(node))
            else:
                successors = list(G_physical.neighbors(node))
            
            if successors:
                virtual_info['successors'][node] = successors
                
                # If this is start pointer, mark physical start
                if 's' in label_codes or 'start_pointer' in node_types:
                    virtual_info['physical_start'] = successors[0]
                    logger.info(
                        f"Virtual start pointer (node {node}) points to "
                        f"physical start room {successors[0]}"
                    )
            
            # Remove virtual node from graph
            G_physical.remove_node(node)
            logger.debug(f"Removed virtual node {node} (label='{node_label}')")
    
    if virtual_info['removed_nodes']:
        logger.info(
            f"Filtered {len(virtual_info['removed_nodes'])} virtual nodes: "
            f"{virtual_info['removed_nodes']}"
        )
    
    return G_physical, virtual_info


# ==========================================
# NODE/EDGE LABEL PARSING
# ==========================================

def get_node_types(G: nx.Graph, node: int) -> Set[str]:
    """
    Get the set of types for a node (handles composite labels).
    
    Args:
        G: NetworkX graph
        node: Node ID
        
    Returns:
        Set of node type names (e.g., {'enemy', 'key', 'puzzle'})
        
    Example:
        >>> G = nx.Graph()
        >>> G.add_node(1, label='e,k,p')
        >>> get_node_types(G, 1)
        {'enemy', 'key', 'puzzle'}
    """
    node_label = G.nodes[node].get('label', '')
    return parse_composite_node_label(node_label)


def has_node_type(G: nx.Graph, node: int, type_name: str) -> bool:
    """
    Check if a node has a specific type (works with composite labels).
    
    Args:
        G: NetworkX graph
        node: Node ID
        type_name: Type to check (e.g., 'enemy', 'boss', 'triforce')
        
    Returns:
        True if node has this type
        
    Example:
        >>> G = nx.Graph()
        >>> G.add_node(1, label='e,k')
        >>> has_node_type(G, 1, 'enemy')
        True
        >>> has_node_type(G, 1, 'puzzle')
        False
    """
    node_types = get_node_types(G, node)
    return type_name in node_types


def find_nodes_by_type(G: nx.Graph, type_name: str) -> List[int]:
    """
    Find all nodes that have a specific type.
    
    Args:
        G: NetworkX graph
        type_name: Type to search for (e.g., 'triforce', 'boss')
        
    Returns:
        List of node IDs that have this type
        
    Example:
        >>> G = nx.Graph()
        >>> G.add_node(1, label='e')
        >>> G.add_node(2, label='e,k')
        >>> G.add_node(3, label='b')
        >>> find_nodes_by_type(G, 'enemy')
        [1, 2]
    """
    matching_nodes = []
    for node in G.nodes():
        if has_node_type(G, node, type_name):
            matching_nodes.append(node)
    return matching_nodes


def get_edge_type(G: nx.Graph, source: int, target: int) -> str:
    """
    Get the connection type for an edge.
    
    Args:
        G: NetworkX graph
        source: Source node ID
        target: Target node ID
        
    Returns:
        Edge type name (e.g., 'key_locked', 'open')
        
    Example:
        >>> G = nx.Graph()
        >>> G.add_edge(1, 2, label='k')
        >>> get_edge_type(G, 1, 2)
        'key_locked'
    """
    edge_data = G.get_edge_data(source, target, default={})
    edge_label = edge_data.get('label', '')
    return parse_edge_label(edge_label)


# ==========================================
# PHYSICAL START NODE IDENTIFICATION
# ==========================================

def get_physical_start_node(G: nx.Graph) -> Optional[int]:
    """
    Get the physical start node (the actual room where gameplay begins).
    
    This function handles three cases:
    1. If virtual 's' node exists, return its successor (physical start)
    2. If no virtual node, look for node with 'start' type
    3. Fallback: return node with highest centrality (likely entry point)
    
    Args:
        G: NetworkX graph
        
    Returns:
        Node ID of physical start room, or None if graph is empty
        
    Example:
        >>> G = nx.DiGraph()
        >>> G.add_node(0, label='s')   # Virtual pointer
        >>> G.add_node(1, label='e')   # Physical start
        >>> G.add_edge(0, 1)
        >>> get_physical_start_node(G)
        1
    """
    # Check for virtual start pointer
    for node in G.nodes():
        node_label = G.nodes[node].get('label', '')
        label_codes = set(code.strip() for code in str(node_label).split(','))
        
        if 's' in label_codes:
            # This is virtual start pointer - get its successor
            if G.is_directed():
                successors = list(G.successors(node))
            else:
                successors = list(G.neighbors(node))
            
            if successors:
                logger.info(f"Found physical start via virtual pointer: {successors[0]}")
                return successors[0]
    
    # No virtual pointer - look for 'start' type node
    start_nodes = find_nodes_by_type(G, 'start')
    if start_nodes:
        logger.info(f"Found physical start node: {start_nodes[0]}")
        return start_nodes[0]
    
    # Fallback: node with highest degree centrality
    if G.number_of_nodes() > 0:
        centrality = nx.degree_centrality(G)
        central_node = max(centrality, key=centrality.get)
        logger.warning(
            f"No start node found; using highest centrality node: {central_node}"
        )
        return central_node
    
    logger.error("Cannot determine physical start: graph is empty")
    return None


# ==========================================
# BOSS-GOAL SUBGRAPH VALIDATION
# ==========================================

def validate_goal_subgraph(G: nx.Graph) -> Tuple[bool, List[str]]:
    """
    Validate the Boss-Goal subgraph pattern.
    
    VGLC Constraint: Goal (triforce) must follow this pattern:
    - Goal is a LEAF node (degree 1)
    - Goal connects ONLY to Boss
    - Boss exists in the graph
    - No cycles through Goal
    - Goal room has no enemies (handled by room generation)
    
    Canonical structure:
    [Pre-Boss] --[k/b]--> [Boss:b] --[l/open]--> [Goal:t]
    
    Args:
        G: NetworkX graph with node 'label' attributes
        
    Returns:
        Tuple of:
        - is_valid: True if pattern is correct
        - errors: List of error messages (empty if valid)
        
    Example:
        >>> G = nx.DiGraph()
        >>> G.add_node(1, label='b')  # Boss
        >>> G.add_node(2, label='t')  # Goal
        >>> G.add_edge(1, 2, label='l')
        >>> is_valid, errors = validate_goal_subgraph(G)
        >>> is_valid
        True
    """
    return _validate_goal_subgraph_strict(G)


def _validate_goal_subgraph_strict(G: nx.Graph) -> Tuple[bool, List[str]]:
    """Strict export-side boss gauntlet validation aligned with generation-time checks."""
    errors: List[str] = []

    goal_nodes = _find_nodes_with_semantic(G, "triforce")
    boss_nodes = _find_nodes_with_semantic(G, "boss")
    boss_door_nodes = _find_nodes_with_semantic(G, "boss_door")
    boss_key_nodes = _find_nodes_with_semantic(G, "boss_key")

    if not goal_nodes:
        errors.append("No goal (triforce) node found in graph")
        return False, errors
    if len(goal_nodes) != 1:
        errors.append(f"Expected exactly one goal (triforce) node, found {len(goal_nodes)}")
        return False, errors

    if BOSS_REQUIRED_FOR_GOAL and not boss_nodes:
        errors.append("No boss node found (required for goal)")
        return False, errors
    if len(boss_nodes) != 1:
        errors.append(f"Expected exactly one boss node, found {len(boss_nodes)}")
        return False, errors

    goal = goal_nodes[0]
    boss = boss_nodes[0]

    goal_degree = G.degree(goal)
    if goal_degree > GOAL_NODE_MAX_DEGREE:
        errors.append(f"Goal node {goal} has degree {goal_degree} (should be <= {GOAL_NODE_MAX_DEGREE})")

    if G.is_directed():
        goal_neighbors = list(dict.fromkeys(list(G.predecessors(goal)) + list(G.successors(goal))))
        goal_outgoing = list(dict.fromkeys(G.successors(goal)))
    else:
        goal_neighbors = list(dict.fromkeys(G.neighbors(goal)))
        goal_outgoing = []

    if GOAL_CONNECTS_TO_BOSS and goal_neighbors != [boss]:
        errors.append(
            f"Goal node {goal} must connect only to boss {boss} (neighbors: {goal_neighbors})"
        )

    if goal_outgoing:
        errors.append(
            f"Goal node {goal} has outgoing edges {goal_outgoing} (should be terminal node)"
        )

    # Preserve the legacy/simple boss -> goal case when there is no explicit boss door.
    if boss_door_nodes:
        if len(boss_door_nodes) != 1:
            errors.append(f"Expected exactly one boss door node, found {len(boss_door_nodes)}")
        else:
            boss_door = boss_door_nodes[0]
            if G.is_directed():
                boss_predecessors = list(dict.fromkeys(G.predecessors(boss)))
                boss_successors = list(dict.fromkeys(G.successors(boss)))
                boss_door_predecessors = list(dict.fromkeys(G.predecessors(boss_door)))
                boss_door_successors = list(dict.fromkeys(G.successors(boss_door)))

                if boss_predecessors != [boss_door]:
                    errors.append(
                        f"Boss node {boss} must have only boss door {boss_door} as predecessor (found {boss_predecessors})"
                    )
                if boss_successors != [goal]:
                    errors.append(
                        f"Boss node {boss} must have only goal {goal} as successor (found {boss_successors})"
                    )
                if boss_door_successors != [boss]:
                    errors.append(
                        f"Boss door node {boss_door} must have only boss {boss} as successor (found {boss_door_successors})"
                    )
                if not boss_door_predecessors or any(
                    predecessor in {boss_door, boss, goal}
                    for predecessor in boss_door_predecessors
                ):
                    errors.append(
                        f"Boss door node {boss_door} has invalid predecessors {boss_door_predecessors}"
                    )
            else:
                boss_neighbors = list(dict.fromkeys(G.neighbors(boss)))
                boss_door_neighbors = list(dict.fromkeys(G.neighbors(boss_door)))
                if sorted(boss_neighbors) != sorted([goal, boss_door]):
                    errors.append(
                        f"Boss node {boss} must connect only to goal {goal} and boss door {boss_door} (neighbors: {boss_neighbors})"
                    )
                if boss not in boss_door_neighbors:
                    errors.append(
                        f"Boss door node {boss_door} must connect to boss {boss} (neighbors: {boss_door_neighbors})"
                    )

            boss_door_key_id = G.nodes[boss_door].get("key_id")
            if boss_door_key_id is None:
                errors.append(f"Boss door node {boss_door} is missing key_id")
            else:
                has_matching_big_key = any(
                    G.nodes[node].get("key_id") == boss_door_key_id
                    for node in boss_key_nodes
                )
                if not has_matching_big_key:
                    errors.append(
                        f"No BIG_KEY / boss_key provider found for boss door {boss_door} (key_id={boss_door_key_id})"
                    )

    is_valid = len(errors) == 0
    if is_valid:
        logger.info("Boss-Goal subgraph validation PASSED")
    else:
        logger.warning(f"Boss-Goal subgraph validation FAILED: {errors}")
    return is_valid, errors


# ==========================================
# GRAPH TOPOLOGY VALIDATION
# ==========================================

def validate_graph_topology(G: nx.Graph) -> Tuple[bool, List[str]]:
    """
    Comprehensive validation of graph topology (all VGLC constraints).
    
    Checks:
    - Graph is connected
    - Start node exists and is reachable
    - Goal node exists and is leaf
    - Boss-Goal pattern is correct
    - Minimum path length constraint
    - No virtual nodes in graph (should be filtered first)
    
    Args:
        G: NetworkX graph
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check graph is not empty
    if G.number_of_nodes() == 0:
        errors.append("Graph is empty (no nodes)")
        return False, errors
    
    # Check no virtual nodes remain
    for node in G.nodes():
        node_label = G.nodes[node].get('label', '')
        label_codes = set(code.strip() for code in str(node_label).split(','))
        if label_codes & VIRTUAL_NODE_TYPES:
            errors.append(
                f"Virtual node {node} (label='{node_label}') found in graph "
                f"(should be filtered before layout)"
            )
    
    # Check connectivity
    if not nx.is_connected(G.to_undirected()):
        errors.append("Graph is not connected (has isolated components)")
    
    # Get start and goal
    start = get_physical_start_node(G)
    goal_nodes = find_nodes_by_type(G, 'triforce')
    
    if start is None:
        errors.append("No start node found or determined")
    
    if not goal_nodes:
        errors.append("No goal (triforce) node found")
    
    # Check path length from start to goal
    if start is not None and goal_nodes:
        goal = goal_nodes[0]
        try:
            path_length = nx.shortest_path_length(G, start, goal)
            
            if path_length < MIN_PATH_LENGTH_START_TO_GOAL:
                errors.append(
                    f"Path from start to goal is too short: {path_length} rooms "
                    f"(minimum: {MIN_PATH_LENGTH_START_TO_GOAL})"
                )
        except nx.NetworkXNoPath:
            errors.append(f"No path exists from start (node {start}) to goal (node {goal})")
    
    # Validate Boss-Goal subgraph
    goal_valid, goal_errors = validate_goal_subgraph(G)
    if not goal_valid:
        errors.extend(goal_errors)
    
    is_valid = len(errors) == 0
    return is_valid, errors


# ==========================================
# EXPORTS
# ==========================================

__all__ = [
    # Virtual node handling
    'filter_virtual_nodes',
    'get_physical_start_node',
    
    # Node/edge parsing
    'get_node_types',
    'has_node_type',
    'find_nodes_by_type',
    'get_edge_type',
    
    # Validation
    'validate_goal_subgraph',
    'validate_graph_topology',
]
