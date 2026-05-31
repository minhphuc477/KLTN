"""Reporting helpers for evolutionary topology generation."""

from __future__ import annotations

from ._shared import *
from .converters import mission_graph_to_networkx

def visualize_evolution_stats(evo_generator: EvolutionaryTopologyGenerator) -> None:
    """Print evolution statistics summary."""
    stats = evo_generator.get_statistics()
    
    print("\n" + "=" * 60)
    print("EVOLUTIONARY SEARCH STATISTICS")
    print("=" * 60)
    
    print("\nGenerations Run: {}".format(stats['generations_run']))
    print("Final Best Fitness: {:.4f}".format(stats['final_best_fitness']))
    print("Converged: {}".format(stats['converged']))
    
    if stats['best_fitness_history']:
        print("\nFitness Progression:")
        print("  Initial: {:.4f}".format(stats['best_fitness_history'][0]))
        print("  Gen 25%: {:.4f}".format(stats['best_fitness_history'][len(stats['best_fitness_history'])//4]))
        print("  Gen 50%: {:.4f}".format(stats['best_fitness_history'][len(stats['best_fitness_history'])//2]))
        print("  Gen 75%: {:.4f}".format(stats['best_fitness_history'][3*len(stats['best_fitness_history'])//4]))
        print("  Final: {:.4f}".format(stats['best_fitness_history'][-1]))
    
    if stats['diversity_history']:
        print("\nDiversity:")
        print("  Initial: {:.4f}".format(stats['diversity_history'][0]))
        print("  Final: {:.4f}".format(stats['diversity_history'][-1]))


def print_graph_summary(G: nx.Graph) -> None:
    """Print summary of generated graph."""
    print("\n" + "=" * 60)
    print("GENERATED GRAPH SUMMARY")
    print("=" * 60)
    
    print("\nTopology:")
    print("  Nodes: {}".format(G.number_of_nodes()))
    print("  Edges: {}".format(G.number_of_edges()))
    
    # Count node types
    node_types = defaultdict(int)
    for node_id in G.nodes():
        node_type = G.nodes[node_id].get('type', 'UNKNOWN')
        node_types[node_type] += 1
    
    print("\nNode Types:")
    for node_type, count in sorted(node_types.items()):
        print("  {}: {}".format(node_type, count))
    
    # Check connectivity
    if nx.is_connected(G):
        print("\nConnectivity: CONNECTED")
        
        # Find START and GOAL
        start_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'START']
        goal_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'GOAL']
        
        if start_nodes and goal_nodes:
            path_length = nx.shortest_path_length(G, start_nodes[0], goal_nodes[0])
            print("  Shortest path (START â†’ GOAL): {} nodes".format(path_length))
    else:
        print("\nConnectivity: DISCONNECTED")
    
    print("\nNodes (first 10):")
    for node_id in list(G.nodes())[:10]:
        data = G.nodes[node_id]
        print(
            "  {}: {} (difficulty={:.2f})".format(
                node_id,
                data.get('type', 'UNKNOWN'),
                data.get('difficulty', 0.0),
            )
        )
