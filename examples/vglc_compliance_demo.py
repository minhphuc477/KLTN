"""
VGLC Compliance Demonstration
==============================

Shows how to:
1. Create VGLC-compliant graphs
2. Validate topology
3. Handle virtual nodes
4. Generate rooms with correct dimensions
5. Parse composite labels
6. Analyze real VGLC data

"""

import networkx as nx
import numpy as np
from pathlib import Path

from src.data.vglc_utils import (
    # Parsing
    parse_node_label,
    parse_node_attributes,
    parse_edge_attributes,
    # Validation
    filter_virtual_nodes,
    get_physical_start_node,
    validate_goal_subgraph,
    validate_topology,
    validate_room_dimensions,
    validate_pixel_dimensions,
    # Utilities
    analyze_vglc_graph,
    convert_to_physical_graph,
)

from src.constants.vglc_constants import (
    ROOM_WIDTH_TILES,
    ROOM_HEIGHT_TILES,
    ROOM_WIDTH_PX,
    ROOM_HEIGHT_PX,
    ROOM_SHAPE,
)


def demo_basic_compliance():
    """Demonstrate basic VGLC compliance checks."""
    print("=" * 70)
    print("DEMO 1: Basic VGLC Compliance")
    print("=" * 70)
    
    # Create a simple dungeon graph with virtual start pointer
    G = nx.DiGraph()
    G.add_node(0, label="s")       # Virtual start pointer
    G.add_node(1, label="e,k")     # Enemy room with key (composite label!)
    G.add_node(2, label="p")       # Puzzle room
    G.add_node(3, label="b")       # Boss
    G.add_node(4, label="t")       # Triforce (goal)
    
    G.add_edge(0, 1, label="")     # Virtual → Physical start (open)
    G.add_edge(1, 2, label="")     # Open door
    G.add_edge(2, 3, label="k")    # Key-locked door
    G.add_edge(3, 4, label="l")    # Soft-locked (shutters after boss)
    
    print(f"\nOriginal graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Identify physical start
    start = get_physical_start_node(G)
    print(f"Physical start node: {start}")
    
    # Filter virtual nodes
    G_physical = filter_virtual_nodes(G)
    print(f"Physical graph: {G_physical.number_of_nodes()} nodes (virtual removed)")
    
    # Validate topology
    print("\n" + "-" * 70)
    print("Topology Validation:")
    print("-" * 70)
    report = validate_topology(G)
    
    print(f"Valid: {report.is_valid}")
    print(f"Physical nodes: {report.num_physical_nodes}")
    print(f"Virtual nodes: {report.num_virtual_nodes}")
    print(f"Boss: {report.num_boss}, Triforce: {report.num_triforce}")
    print(f"Connected: {report.is_connected}")
    print(f"Goal subgraph valid: {report.goal_subgraph_valid}")
    
    if report.warnings:
        print(f"\nWarnings: {report.warnings}")


def demo_goal_validation():
    """Demonstrate goal subgraph validation."""
    print("\n" + "=" * 70)
    print("DEMO 2: Goal Subgraph Validation")
    print("=" * 70)
    
    # Valid goal subgraph (Boss → Triforce)
    print("\n✅ Valid Pattern: Boss → Triforce")
    G_valid = nx.DiGraph()
    G_valid.add_node(1, label="e")
    G_valid.add_node(2, label="b")    # Boss
    G_valid.add_node(3, label="t")    # Triforce (leaf)
    G_valid.add_edge(1, 2, label="k") # Key-locked boss door
    G_valid.add_edge(2, 3, label="")  # Open boss → triforce
    
    valid, msg = validate_goal_subgraph(G_valid)
    print(f"   Result: {valid}")
    print(f"   Message: {msg}")
    
    # Invalid: no boss
    print("\n❌ Invalid Pattern: Missing Boss")
    G_invalid = nx.DiGraph()
    G_invalid.add_node(1, label="e")
    G_invalid.add_node(2, label="t")  # Triforce without boss
    G_invalid.add_edge(1, 2)
    
    valid, msg = validate_goal_subgraph(G_invalid)
    print(f"   Result: {valid}")
    print(f"   Message: {msg}")
    
    # Invalid: triforce not leaf (has outgoing edge)
    print("\n❌ Invalid Pattern: Triforce Not Leaf")
    G_invalid2 = nx.DiGraph()
    G_invalid2.add_node(1, label="b")
    G_invalid2.add_node(2, label="t")
    G_invalid2.add_node(3, label="e")
    G_invalid2.add_edge(1, 2)
    G_invalid2.add_edge(2, 3)  # Triforce has outgoing edge!
    
    valid, msg = validate_goal_subgraph(G_invalid2)
    print(f"   Result: {valid}")
    print(f"   Message: {msg}")


def demo_dimension_validation():
    """Demonstrate dimension validation."""
    print("\n" + "=" * 70)
    print("DEMO 3: Dimension Validation")
    print("=" * 70)
    
    print(f"\nVGLC Room Dimensions:")
    print(f"  Width: {ROOM_WIDTH_TILES} tiles ({ROOM_WIDTH_PX} pixels)")
    print(f"  Height: {ROOM_HEIGHT_TILES} tiles ({ROOM_HEIGHT_PX} pixels)")
    print(f"  Shape (numpy): {ROOM_SHAPE}")
    print(f"  Aspect Ratio: {ROOM_WIDTH_TILES}:{ROOM_HEIGHT_TILES}")
    
    # Valid room (11×16)
    print("\n✅ Valid Room (11×16):")
    room_valid = np.zeros(ROOM_SHAPE, dtype=int)
    valid, msg = validate_room_dimensions(room_valid)
    print(f"   Shape: {room_valid.shape}")
    print(f"   Valid: {valid}")
    print(f"   Message: {msg}")
    
    # Invalid room (square 16×16)
    print("\n❌ Invalid Room (16×16 - SQUARE):")
    room_invalid = np.zeros((16, 16), dtype=int)
    valid, msg = validate_room_dimensions(room_invalid)
    print(f"   Shape: {room_invalid.shape}")
    print(f"   Valid: {valid}")
    print(f"   Message: {msg}")
    
    # Valid pixel image (176×256)
    print("\n✅ Valid Pixel Image (176×256):")
    image_valid = np.zeros((ROOM_HEIGHT_PX, ROOM_WIDTH_PX, 3), dtype=np.uint8)
    valid, msg = validate_pixel_dimensions(image_valid)
    print(f"   Shape: {image_valid.shape}")
    print(f"   Valid: {valid}")
    print(f"   Message: {msg}")
    
    # Invalid pixel image (square 256×256)
    print("\n❌ Invalid Pixel Image (256×256 - SQUARE):")
    image_invalid = np.zeros((256, 256, 3), dtype=np.uint8)
    valid, msg = validate_pixel_dimensions(image_invalid)
    print(f"   Shape: {image_invalid.shape}")
    print(f"   Valid: {valid}")
    print(f"   Message: {msg}")


def demo_composite_labels():
    """Demonstrate composite label parsing."""
    print("\n" + "=" * 70)
    print("DEMO 4: Composite Label Parsing")
    print("=" * 70)
    
    labels = [
        ("e", "Enemy"),
        ("e,k", "Enemy + Key"),
        ("e,k,p", "Enemy + Key + Puzzle"),
        ("b", "Boss"),
        ("t", "Triforce"),
        ("s", "Start Pointer (Virtual)"),
        ("e,I", "Enemy + Macro Item"),
        ("p,e,I", "Puzzle + Enemy + Macro Item"),
    ]
    
    G = nx.DiGraph()
    
    for node_id, (label, description) in enumerate(labels, start=1):
        G.add_node(node_id, label=label)
        
        parsed = parse_node_label(label)
        attrs = parse_node_attributes(G, node_id)
        
        print(f"\n{description} ('{label}'):")
        print(f"   Parsed: {parsed}")
        print(f"   Has Enemy: {attrs.has_enemy}")
        print(f"   Has Key: {attrs.has_key}")
        print(f"   Has Puzzle: {attrs.has_puzzle}")
        print(f"   Is Boss: {attrs.is_boss}")
        print(f"   Is Triforce: {attrs.is_triforce}")
        print(f"   Is Virtual: {not attrs.is_physical}")


def demo_realistic_dungeon():
    """Demonstrate realistic VGLC dungeon structure."""
    print("\n" + "=" * 70)
    print("DEMO 5: Realistic VGLC Dungeon")
    print("=" * 70)
    
    # Simplified version of actual LoZ Level 1 structure
    G = nx.DiGraph()
    
    # Rooms
    G.add_node(0, label="s")        # Virtual start pointer
    G.add_node(1, label="")         # Entry room
    G.add_node(2, label="e,k")      # Enemy + key
    G.add_node(3, label="p")        # Puzzle room
    G.add_node(4, label="e,I")      # Enemy + macro item (bow)
    G.add_node(5, label="e,k")      # Enemy + key
    G.add_node(6, label="e")        # Pre-boss combat
    G.add_node(7, label="b")        # Boss (Aquamentus)
    G.add_node(8, label="t")        # Triforce piece
    
    # Connections (realistic edge types)
    G.add_edge(0, 1, label="")      # Virtual → Entry
    G.add_edge(1, 2, label="")      # Open
    G.add_edge(2, 3, label="")      # Open
    G.add_edge(3, 4, label="b")     # Bombable (secret)
    G.add_edge(3, 5, label="k")     # Key-locked
    G.add_edge(5, 6, label="")      # Open
    G.add_edge(6, 7, label="k")     # Boss key door
    G.add_edge(7, 8, label="l")     # Soft-lock (shutters after boss)
    
    print("\n📊 Dungeon Statistics:")
    print(f"   Total Nodes: {G.number_of_nodes()}")
    print(f"   Edges: {G.number_of_edges()}")
    
    # Get physical start
    start = get_physical_start_node(G)
    print(f"   Physical Start: Node {start}")
    
    # Convert to physical graph
    G_physical, start_node = convert_to_physical_graph(G, validate=True)
    print(f"   Physical Nodes: {G_physical.number_of_nodes()}")
    
    # Validate
    print("\n📋 Validation Report:")
    report = validate_topology(G)
    print(f"   Valid: {report.is_valid}")
    print(f"   Boss: {report.num_boss}")
    print(f"   Triforce: {report.num_triforce}")
    print(f"   Goal Subgraph Valid: {report.goal_subgraph_valid}")
    
    # Print full report
    if not report.is_valid:
        print("\n" + report.summary())


def demo_real_vglc_data():
    """Demonstrate loading real VGLC data (if available)."""
    print("\n" + "=" * 70)
    print("DEMO 6: Real VGLC Data Analysis")
    print("=" * 70)
    
    # Check if VGLC data exists
    level1_text = Path("Data/The Legend of Zelda/Processed/tloz1_1.txt")
    level1_graph = Path("Data/The Legend of Zelda/Graph Processed/LoZ_1.dot")
    
    if level1_text.exists():
        print("\n✅ Found VGLC Level 1 text data")
        with open(level1_text, 'r') as f:
            lines = f.readlines()
        print(f"   Text file: {len(lines)} lines")
        print(f"   File size: {level1_text.stat().st_size} bytes")
    else:
        print("\n⚠️  VGLC text data not found")
    
    if level1_graph.exists():
        print("\n✅ Found VGLC Level 1 graph data")
        try:
            import pydot
            G = nx.DiGraph(nx.drawing.nx_pydot.read_dot(str(level1_graph)))
            
            print(f"\n📊 Level 1 Graph Statistics:")
            print(f"   Nodes: {G.number_of_nodes()}")
            print(f"   Edges: {G.number_of_edges()}")
            
            # Analyze
            report = validate_topology(G)
            print(f"\n   Physical Nodes: {report.num_physical_nodes}")
            print(f"   Virtual Nodes: {report.num_virtual_nodes}")
            print(f"   Boss Rooms: {report.num_boss}")
            print(f"   Triforce Rooms: {report.num_triforce}")
            print(f"   Enemy Rooms: {report.num_enemy}")
            print(f"   Key Rooms: {report.num_key}")
            
        except ImportError:
            print("   (pydot not available - install with: pip install pydot)")
    else:
        print("\n⚠️  VGLC graph data not found")


def main():
    """Run all demos."""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  VGLC COMPLIANCE DEMONSTRATION".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    
    demo_basic_compliance()
    demo_goal_validation()
    demo_dimension_validation()
    demo_composite_labels()
    demo_realistic_dungeon()
    demo_real_vglc_data()
    
    print("\n" + "=" * 70)
    print("✅ VGLC Compliance Demo Complete!")
    print("=" * 70)
    print("\nFor more information, see:")
    print("  - docs/VGLC_DATA_RESEARCH.md")
    print("  - docs/VGLC_COMPLIANCE_GUIDE.md")
    print("  - src/data/vglc_utils.py")
    print("  - tests/test_vglc_compliance.py")
    print()


if __name__ == "__main__":
    main()
