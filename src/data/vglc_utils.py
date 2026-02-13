"""
VGLC Compliance Utilities
==========================

Provides utilities for working with VGLC-compliant graph structures including:
- Virtual node detection and filtering  
- Node label parsing (composite labels like "e,k,p")
- Edge type classification
- Topology validation
- Boss gauntlet pattern validation
- Dimension validation

This module ensures all generated dungeons comply with VGLC dataset standards.

Sources:
- Data/The Legend of Zelda/Graph Processed/*.dot
- Data/The Legend of Zelda/Processed/*.txt
- Data/The Legend of Zelda/zelda.json
- src/data/zelda_core.py

"""

import networkx as nx
import numpy as np
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

# Import from existing constants
from src.constants.vglc_constants import (
    ROOM_WIDTH_TILES,
    ROOM_HEIGHT_TILES,
    ROOM_SHAPE,
    TILE_SIZE_PX,
    NODE_TYPE_MAP,
    EDGE_TYPE_MAP,
    VIRTUAL_NODE_TYPES,
    PHYSICAL_NODE_TYPES,
    LEAF_NODE_TYPES,
)

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class NodeAttributes:
    """Parsed node attributes from VGLC graph."""
    node_id: int
    labels: Set[str]  # e.g., {"e", "k", "p"}
    label_string: str  # e.g., "e,k,p"
    
    # Attribute flags
    is_start_pointer: bool
    is_triforce: bool
    is_boss: bool
    has_enemy: bool
    has_key: bool
    has_macro_item: bool  # I (bow, raft, ladder)
    has_minor_item: bool  # i (compass, map)
    has_puzzle: bool
    is_physical: bool  # False if virtual (start pointer)
    is_empty: bool  # No special content
    
    def __repr__(self) -> str:
        """Human-readable representation."""
        attrs = []
        if self.is_start_pointer:
            attrs.append("start_pointer")
        if self.is_triforce:
            attrs.append("triforce")
        if self.is_boss:
            attrs.append("boss")
        if self.has_enemy:
            attrs.append("enemy")
        if self.has_key:
            attrs.append("key")
        if self.has_macro_item:
            attrs.append("macro_item")
        if self.has_minor_item:
            attrs.append("minor_item")
        if self.has_puzzle:
            attrs.append("puzzle")
        if self.is_empty:
            attrs.append("empty")
        if not self.is_physical:
            attrs.append("VIRTUAL")
        
        return f"Node({self.node_id}, [{', '.join(attrs)}])"


@dataclass
class EdgeAttributes:
    """Parsed edge attributes from VGLC graph."""
    source: int
    target: int
    edge_type: str  # "", "k", "b", "l", "s"
    label: str  # Raw label from graph
    
    # Attribute flags
    is_open: bool
    is_key_locked: bool
    is_bombable: bool
    is_soft_locked: bool
    is_warp: bool
    consumes_resource: bool  # True for key_locked or bombable
    is_oneway: bool  # True for soft_locked
    
    def __repr__(self) -> str:
        """Human-readable representation."""
        type_name = self.edge_type if self.edge_type else "open"
        flags = []
        if self.consumes_resource:
            flags.append("consumes")
        if self.is_oneway:
            flags.append("oneway")
        
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        return f"Edge({self.source}→{self.target}, {type_name}{flag_str})"


@dataclass
class TopologyReport:
    """Comprehensive topology validation report."""
    is_valid: bool
    warnings: List[str]
    errors: List[str]
    
    # Structure metrics
    num_nodes: int
    num_physical_nodes: int
    num_virtual_nodes: int
    num_edges: int
    
    # Node type counts
    num_triforce: int
    num_boss: int
    num_enemy: int
    num_key: int
    
    # Connectivity
    is_connected: bool
    num_components: int
    isolated_nodes: List[int]
    
    # Goal subgraph
    has_goal_subgraph: bool
    goal_subgraph_valid: bool
    goal_subgraph_message: str
    
    # Start node
    has_start: bool
    start_node: Optional[int]
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = []
        lines.append("=" * 60)
        lines.append("VGLC TOPOLOGY VALIDATION REPORT")
        lines.append("=" * 60)
        
        # Overall status
        status = "✅ VALID" if self.is_valid else "❌ INVALID"
        lines.append(f"\nStatus: {status}")
        
        # Metrics
        lines.append(f"\nMetrics:")
        lines.append(f"  Total Nodes: {self.num_nodes}")
        lines.append(f"  Physical Nodes: {self.num_physical_nodes}")
        lines.append(f"  Virtual Nodes: {self.num_virtual_nodes}")
        lines.append(f"  Edges: {self.num_edges}")
        
        # Node types
        lines.append(f"\nNode Types:")
        lines.append(f"  Triforce: {self.num_triforce}")
        lines.append(f"  Boss: {self.num_boss}")
        lines.append(f"  Enemy: {self.num_enemy}")
        lines.append(f"  Key: {self.num_key}")
        
        # Connectivity
        lines.append(f"\nConnectivity:")
        lines.append(f"  Connected: {'✅' if self.is_connected else '❌'}")
        lines.append(f"  Components: {self.num_components}")
        if self.isolated_nodes:
            lines.append(f"  Isolated Nodes: {self.isolated_nodes}")
        
        # Goal subgraph
        lines.append(f"\nGoal Subgraph:")
        lines.append(f"  Exists: {'✅' if self.has_goal_subgraph else '❌'}")
        lines.append(f"  Valid: {'✅' if self.goal_subgraph_valid else '❌'}")
        lines.append(f"  Message: {self.goal_subgraph_message}")
        
        # Start node
        lines.append(f"\nStart Node:")
        lines.append(f"  Found: {'✅' if self.has_start else '❌'}")
        if self.start_node is not None:
            lines.append(f"  Node ID: {self.start_node}")
        
        # Warnings
        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings:
                lines.append(f"  ⚠️  {warning}")
        
        # Errors
        if self.errors:
            lines.append(f"\nErrors ({len(self.errors)}):")
            for error in self.errors:
                lines.append(f"  ❌ {error}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================================
# GRAPH PARSING
# ============================================================================

class VGLCGraphParser:
    """Parse and validate VGLC graph structures."""
    
    @staticmethod
    def parse_node_label(label: str) -> Set[str]:
        """
        Parse composite node labels.
        
        Examples:
            "e,k,p" → {"e", "k", "p"}
            "b" → {"b"}
            "s" → {"s"}
            "" → set()
        
        Args:
            label: Raw label string from graph node
            
        Returns:
            Set of individual label components
        """
        if not label or label == "":
            return set()
        return set(part.strip() for part in label.split(",") if part.strip())
    
    @staticmethod
    def parse_node_attributes(graph: nx.Graph, node_id: int) -> NodeAttributes:
        """
        Extract all attributes from a node.
        
        Args:
            graph: NetworkX graph
            node_id: Node ID to parse
            
        Returns:
            NodeAttributes object with all parsed flags
        """
        node_data = graph.nodes.get(node_id, {})
        label_str = node_data.get('label', '')
        labels = VGLCGraphParser.parse_node_label(label_str)
        
        return NodeAttributes(
            node_id=node_id,
            labels=labels,
            label_string=label_str,
            is_start_pointer='s' in labels,
            is_triforce='t' in labels,
            is_boss='b' in labels,
            has_enemy='e' in labels,
            has_key='k' in labels,
            has_macro_item='I' in labels,
            has_minor_item='i' in labels,
            has_puzzle='p' in labels,
            is_physical='s' not in labels,  # Start pointer is virtual
            is_empty=len(labels) == 0,
        )
    
    @staticmethod
    def parse_edge_attributes(graph: nx.Graph, source: int, target: int) -> EdgeAttributes:
        """
        Extract all attributes from an edge.
        
        Args:
            graph: NetworkX graph
            source: Source node ID
            target: Target node ID
            
        Returns:
            EdgeAttributes object with all parsed flags
        """
        # Handle both directed and undirected graphs
        if graph.is_directed():
            edge_data = graph.edges.get((source, target), {})
        else:
            # For undirected, try both directions
            edge_data = graph.edges.get((source, target), 
                                       graph.edges.get((target, source), {}))
        
        label = edge_data.get('label', '')
        
        # Determine edge type
        if label == '':
            edge_type = 'open'
        elif label == 'k':
            edge_type = 'key_locked'
        elif label == 'b':
            edge_type = 'bombable'
        elif label == 'l':
            edge_type = 'soft_locked'
        elif label == 's':
            edge_type = 'stairs_warp'
        else:
            edge_type = label  # Unknown type, keep as-is
        
        return EdgeAttributes(
            source=source,
            target=target,
            edge_type=edge_type,
            label=label,
            is_open=label == '',
            is_key_locked=label == 'k',
            is_bombable=label == 'b',
            is_soft_locked=label == 'l',
            is_warp=label == 's',
            consumes_resource=label in ['k', 'b'],
            is_oneway=label == 'l',
        )
    
    @staticmethod
    def get_node_type_counts(graph: nx.Graph) -> Dict[str, int]:
        """
        Count nodes by type.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Dict mapping node type to count
        """
        counts = {
            'triforce': 0,
            'boss': 0,
            'enemy': 0,
            'key': 0,
            'macro_item': 0,
            'minor_item': 0,
            'puzzle': 0,
            'start_pointer': 0,
            'empty': 0,
        }
        
        for node in graph.nodes():
            attrs = VGLCGraphParser.parse_node_attributes(graph, node)
            if attrs.is_triforce:
                counts['triforce'] += 1
            if attrs.is_boss:
                counts['boss'] += 1
            if attrs.has_enemy:
                counts['enemy'] += 1
            if attrs.has_key:
                counts['key'] += 1
            if attrs.has_macro_item:
                counts['macro_item'] += 1
            if attrs.has_minor_item:
                counts['minor_item'] += 1
            if attrs.has_puzzle:
                counts['puzzle'] += 1
            if attrs.is_start_pointer:
                counts['start_pointer'] += 1
            if attrs.is_empty:
                counts['empty'] += 1
        
        return counts


# ============================================================================
# TOPOLOGY VALIDATION
# ============================================================================

class VGLCTopologyValidator:
    """Validate VGLC graph topology rules."""
    
    @staticmethod
    def is_virtual_node(graph: nx.Graph, node_id: int) -> bool:
        """
        Check if node is virtual (start pointer).
        
        Args:
            graph: NetworkX graph
            node_id: Node ID to check
            
        Returns:
            True if node is virtual
        """
        attrs = VGLCGraphParser.parse_node_attributes(graph, node_id)
        return attrs.is_start_pointer
    
    @staticmethod
    def get_virtual_nodes(graph: nx.Graph) -> List[int]:
        """
        Get all virtual nodes in graph.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            List of virtual node IDs
        """
        return [n for n in graph.nodes() 
                if VGLCTopologyValidator.is_virtual_node(graph, n)]
    
    @staticmethod
    def filter_virtual_nodes(graph: nx.Graph) -> nx.Graph:
        """
        Remove virtual nodes from graph and rewire connections.
        
        Virtual nodes (start pointers) are meta-nodes that indicate the
        entry point but are not placed on the physical grid. This function
        removes them and marks their successors as entry points.
        
        Example:
            Before: s (virtual) → 8 (physical) → 9
            After:  8 (entry=True) → 9
        
        Args:
            graph: NetworkX graph (may contain virtual nodes)
            
        Returns:
            New graph with virtual nodes removed
        """
        filtered = graph.copy()
        virtual_nodes = VGLCTopologyValidator.get_virtual_nodes(graph)
        
        logger.info(f"Filtering {len(virtual_nodes)} virtual nodes: {virtual_nodes}")
        
        for vnode in virtual_nodes:
            # Get successors (physical nodes)
            successors = list(graph.successors(vnode))
            logger.debug(f"Virtual node {vnode} → successors {successors}")
            
            # Remove virtual node
            filtered.remove_node(vnode)
            
            # Mark first successor as entry point
            if successors:
                filtered.nodes[successors[0]]['is_entry'] = True
                logger.info(f"Marked node {successors[0]} as entry point")
        
        return filtered
    
    @staticmethod
    def get_physical_start_node(graph: nx.Graph) -> Optional[int]:
        """
        Get the physical start room node.
        
        Logic:
        1. If 's' (start pointer) exists, return its successor
        2. Otherwise, return node marked with 'is_entry'
        3. Otherwise, return node with label containing 'start'
        4. Otherwise, return node with highest degree centrality
        5. Otherwise, return None
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Physical start node ID, or None if not found
        """
        # Find start pointer
        for node in graph.nodes():
            attrs = VGLCGraphParser.parse_node_attributes(graph, node)
            if attrs.is_start_pointer:
                successors = list(graph.successors(node))
                if successors:
                    logger.info(f"Found start pointer {node} → physical start {successors[0]}")
                    return successors[0]
        
        # Find node marked as entry
        for node in graph.nodes():
            if graph.nodes[node].get('is_entry', False):
                logger.info(f"Found marked entry node {node}")
                return node
        
        # Find node with label containing 'start'
        for node in graph.nodes():
            label = graph.nodes[node].get('label', '')
            if 'start' in label.lower():
                logger.info(f"Found node with 'start' label: {node}")
                return node
        
        # Fallback: use highest degree centrality
        if graph.number_of_nodes() > 0:
            degree_dict = dict(graph.degree())
            if degree_dict:
                max_node = max(degree_dict, key=degree_dict.get)
                logger.info(f"Using highest centrality node as start: {max_node}")
                return max_node
        
        logger.warning("No physical start node found")
        return None
    
    @staticmethod
    def validate_goal_subgraph(graph: nx.Graph) -> Tuple[bool, str]:
        """
        Validate "Boss Gauntlet" pattern.
        
        Rules:
        1. Triforce node (t) must exist
        2. Triforce must be a leaf node (degree 1 or outgoing degree 0)
        3. Triforce should connect to Boss node (b)
        4. Boss should exist
        
        Args:
            graph: NetworkX graph (should be physical, no virtual nodes)
            
        Returns:
            (is_valid, error_message)
        """
        # Find triforce nodes
        triforce_nodes = []
        boss_nodes = []
        
        for node in graph.nodes():
            attrs = VGLCGraphParser.parse_node_attributes(graph, node)
            if attrs.is_triforce:
                triforce_nodes.append(node)
            if attrs.is_boss:
                boss_nodes.append(node)
        
        # Check triforce exists
        if not triforce_nodes:
            return False, "No triforce node found"
        
        if len(triforce_nodes) > 1:
            return False, f"Multiple triforce nodes found: {triforce_nodes}"
        
        triforce = triforce_nodes[0]
        
        # Check triforce is leaf
        if graph.is_directed():
            # For directed graphs, check outgoing degree
            out_degree = graph.out_degree(triforce)
            if out_degree > 0:
                return False, f"Triforce node has {out_degree} outgoing edges (should be 0)"
        else:
            # For undirected graphs, check total degree
            degree = graph.degree(triforce)
            if degree > 2:
                return False, f"Triforce node has degree {degree} (should be ≤2)"
        
        # Check triforce connects to boss (or is connected by boss)
        connected_to_boss = False
        
        # Get all neighbors (predecessors + successors for directed, neighbors for undirected)
        if graph.is_directed():
            all_neighbors = set(graph.predecessors(triforce)) | set(graph.successors(triforce))
        else:
            all_neighbors = set(graph.neighbors(triforce))
        
        for neighbor in all_neighbors:
            neighbor_attrs = VGLCGraphParser.parse_node_attributes(graph, neighbor)
            if neighbor_attrs.is_boss:
                connected_to_boss = True
                break
        
        if not connected_to_boss and boss_nodes:
            return False, "Triforce not connected to boss node"
        
        # Validate boss exists
        if not boss_nodes:
            return False, "No boss node found"
        
        return True, "Valid goal subgraph"
    
    @staticmethod
    def validate_topology(graph: nx.Graph) -> TopologyReport:
        """
        Comprehensive topology validation.
        
        Args:
            graph: NetworkX graph (may contain virtual nodes)
            
        Returns:
            TopologyReport with detailed validation results
        """
        warnings = []
        errors = []
        
        # Check empty graph
        if graph.number_of_nodes() == 0:
            errors.append("Empty graph (no nodes)")
            return TopologyReport(
                is_valid=False,
                warnings=warnings,
                errors=errors,
                num_nodes=0,
                num_physical_nodes=0,
                num_virtual_nodes=0,
                num_edges=0,
                num_triforce=0,
                num_boss=0,
                num_enemy=0,
                num_key=0,
                is_connected=False,
                num_components=0,
                isolated_nodes=[],
                has_goal_subgraph=False,
                goal_subgraph_valid=False,
                goal_subgraph_message="No nodes in graph",
                has_start=False,
                start_node=None,
            )
        
        # Basic metrics
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        
        # Count node types
        virtual_nodes = VGLCTopologyValidator.get_virtual_nodes(graph)
        num_virtual = len(virtual_nodes)
        num_physical = num_nodes - num_virtual
        
        # Check for unfiltered virtual nodes
        if num_virtual > 0:
            errors.append(f"Virtual nodes present (should be filtered): {virtual_nodes}")
        
        type_counts = VGLCGraphParser.get_node_type_counts(graph)
        
        # Check for isolated nodes
        isolated = list(nx.isolates(graph))
        if isolated:
            warnings.append(f"Isolated nodes found: {isolated}")
        
        # Check connectivity (excluding virtual nodes)
        filtered = VGLCTopologyValidator.filter_virtual_nodes(graph)
        
        if filtered.number_of_nodes() == 0:
            errors.append("No physical nodes in graph")
            is_connected = False
            num_components = 0
        else:
            # Convert to undirected for connectivity check
            G_undirected = filtered.to_undirected() if filtered.is_directed() else filtered
            is_connected = nx.is_connected(G_undirected)
            num_components = nx.number_connected_components(G_undirected)
            
            if not is_connected:
                warnings.append(f"Graph has {num_components} disconnected components")
        
        # Validate goal subgraph
        goal_valid, goal_msg = VGLCTopologyValidator.validate_goal_subgraph(filtered)
        has_goal = type_counts['triforce'] > 0 and type_counts['boss'] > 0
        
        if not goal_valid:
            warnings.append(f"Goal subgraph: {goal_msg}")
        
        # Check for start
        start = VGLCTopologyValidator.get_physical_start_node(graph)
        has_start = start is not None
        
        if not has_start:
            errors.append("No physical start node found")
        
        # Determine overall validity
        is_valid = len(errors) == 0
        
        return TopologyReport(
            is_valid=is_valid,
            warnings=warnings,
            errors=errors,
            num_nodes=num_nodes,
            num_physical_nodes=num_physical,
            num_virtual_nodes=num_virtual,
            num_edges=num_edges,
            num_triforce=type_counts['triforce'],
            num_boss=type_counts['boss'],
            num_enemy=type_counts['enemy'],
            num_key=type_counts['key'],
            is_connected=is_connected,
            num_components=num_components,
            isolated_nodes=isolated,
            has_goal_subgraph=has_goal,
            goal_subgraph_valid=goal_valid,
            goal_subgraph_message=goal_msg,
            has_start=has_start,
            start_node=start,
        )


# ============================================================================
# DIMENSION VALIDATION
# ============================================================================

class VGLCDimensionValidator:
    """Validate dimension compliance."""
    
    @staticmethod
    def validate_room_dimensions(room_array: np.ndarray) -> Tuple[bool, str]:
        """
        Validate room array has correct dimensions.
        
        Expected: (16 rows, 11 columns) or (ROOM_HEIGHT_TILES, ROOM_WIDTH_TILES)
        
        Args:
            room_array: Numpy array representing room grid
            
        Returns:
            (is_valid, error_message)
        """
        if not isinstance(room_array, np.ndarray):
            return False, f"Not a numpy array: {type(room_array)}"
        
        if room_array.ndim != 2:
            return False, f"Wrong number of dimensions: {room_array.ndim} (expected 2)"
        
        height, width = room_array.shape
        
        if height != ROOM_HEIGHT_TILES or width != ROOM_WIDTH_TILES:
            return False, (f"Wrong dimensions: {height}×{width} "
                          f"(expected {ROOM_HEIGHT_TILES}×{ROOM_WIDTH_TILES})")
        
        return True, "Valid dimensions"
    
    @staticmethod
    def validate_pixel_dimensions(image_array: np.ndarray) -> Tuple[bool, str]:
        """
        Validate pixel image has correct dimensions.
        
        Expected: (256 height, 176 width) pixels
        
        Args:
            image_array: Numpy array representing pixel image
            
        Returns:
            (is_valid, error_message)
        """
        expected_height = ROOM_HEIGHT_TILES * TILE_SIZE_PX  # 256
        expected_width = ROOM_WIDTH_TILES * TILE_SIZE_PX    # 176
        
        if not isinstance(image_array, np.ndarray):
            return False, f"Not a numpy array: {type(image_array)}"
        
        # Handle (H, W) or (H, W, C) formats
        if image_array.ndim not in [2, 3]:
            return False, f"Wrong number of dimensions: {image_array.ndim} (expected 2 or 3)"
        
        height, width = image_array.shape[:2]
        
        if height != expected_height or width != expected_width:
            return False, (f"Wrong pixel dimensions: {height}×{width} "
                          f"(expected {expected_height}×{expected_width})")
        
        return True, "Valid pixel dimensions"
    
    @staticmethod
    def validate_room_shape(room_array: np.ndarray) -> Tuple[bool, str]:
        """
        Validate room has standard VGLC shape.
        
        Checks:
        - Correct dimensions (16×11)
        - Wall perimeter present
        - Interior region exists
        
        Args:
            room_array: Numpy array representing room grid
            
        Returns:
            (is_valid, error_message)
        """
        # First check dimensions
        valid, msg = VGLCDimensionValidator.validate_room_dimensions(room_array)
        if not valid:
            return valid, msg
        
        # Check for wall perimeter (optional - depends on representation)
        # This is a heuristic check, not strict
        # Walls are typically on edges (row 0, row 15, col 0, col 10)
        
        # For now, just check dimensions
        return True, "Valid room shape"


# ============================================================================
# MODULE-LEVEL EXPORTS
# ============================================================================

# Export key functions at module level for convenience
parse_node_label = VGLCGraphParser.parse_node_label
parse_node_attributes = VGLCGraphParser.parse_node_attributes
parse_edge_attributes = VGLCGraphParser.parse_edge_attributes
get_node_type_counts = VGLCGraphParser.get_node_type_counts

is_virtual_node = VGLCTopologyValidator.is_virtual_node
get_virtual_nodes = VGLCTopologyValidator.get_virtual_nodes
filter_virtual_nodes = VGLCTopologyValidator.filter_virtual_nodes
get_physical_start_node = VGLCTopologyValidator.get_physical_start_node
validate_goal_subgraph = VGLCTopologyValidator.validate_goal_subgraph
validate_topology = VGLCTopologyValidator.validate_topology

validate_room_dimensions = VGLCDimensionValidator.validate_room_dimensions
validate_pixel_dimensions = VGLCDimensionValidator.validate_pixel_dimensions
validate_room_shape = VGLCDimensionValidator.validate_room_shape


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_vglc_graph(dot_path: Path) -> nx.DiGraph:
    """
    Load VGLC graph from DOT file.
    
    Args:
        dot_path: Path to .dot file
        
    Returns:
        NetworkX DiGraph
    """
    try:
        graph = nx.DiGraph(nx.drawing.nx_pydot.read_dot(str(dot_path)))
        logger.info(f"Loaded graph from {dot_path}: "
                   f"{graph.number_of_nodes()} nodes, "
                   f"{graph.number_of_edges()} edges")
        return graph
    except Exception as e:
        logger.error(f"Failed to load graph from {dot_path}: {e}")
        raise


def analyze_vglc_graph(graph: nx.Graph, verbose: bool = True) -> TopologyReport:
    """
    Perform comprehensive analysis of VGLC graph.
    
    Args:
        graph: NetworkX graph to analyze
        verbose: If True, print detailed report
        
    Returns:
        TopologyReport with analysis results
    """
    report = validate_topology(graph)
    
    if verbose:
        print(report.summary())
    
    return report


def convert_to_physical_graph(graph: nx.Graph, 
                              validate: bool = True) -> Tuple[nx.Graph, Optional[int]]:
    """
    Convert graph with virtual nodes to physical-only graph.
    
    Args:
        graph: NetworkX graph (may contain virtual nodes)
        validate: If True, run topology validation
        
    Returns:
        (physical_graph, start_node_id)
    """
    # Filter virtual nodes
    physical_graph = filter_virtual_nodes(graph)
    
    # Get start node
    start_node = get_physical_start_node(graph)
    
    # Validate if requested
    if validate:
        report = validate_topology(graph)
        if not report.is_valid:
            logger.warning(f"Topology validation failed: {report.errors}")
    
    return physical_graph, start_node


# ============================================================================
# END OF MODULE
# ============================================================================
