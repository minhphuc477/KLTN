"""
Example: Using the Evolutionary Topology Director
=================================================

This script demonstrates how to use the EvolutionaryTopologyGenerator
to create custom dungeon topologies based on target difficulty curves.

Usage:
    python examples/evolutionary_generation_demo.py
"""

import sys
import logging
from pathlib import Path

# Setup Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
from src.generation.evolutionary_director import (
    EvolutionaryTopologyGenerator,
    mission_graph_to_networkx,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def plot_curve_comparison(target, extracted, title="Tension Curve"):
    """Plot target vs extracted tension curves."""
    plt.figure(figsize=(10, 5))
    x = range(len(target))
    
    plt.plot(x, target, 'b-o', label='Target', linewidth=2, markersize=8)
    plt.plot(x, extracted, 'r--s', label='Generated', linewidth=2, markersize=6)
    
    plt.xlabel('Progression Point')
    plt.ylabel('Difficulty/Tension')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.1)
    
    # Calculate MSE
    mse = np.mean((np.array(target) - np.array(extracted)) ** 2)
    plt.text(0.02, 0.98, f'MSE: {mse:.4f}', 
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return plt


def plot_evolution_stats(generator, title="Evolution Progress"):
    """Plot evolution statistics."""
    stats = generator.get_statistics()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Fitness over time
    ax1 = axes[0]
    generations = range(len(stats['best_fitness_history']))
    ax1.plot(generations, stats['best_fitness_history'], 'b-', 
             label='Best Fitness', linewidth=2)
    ax1.plot(generations, stats['avg_fitness_history'], 'g--', 
             label='Avg Fitness', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness')
    ax1.set_title('Fitness Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    # Diversity over time
    ax2 = axes[1]
    ax2.plot(generations, stats['diversity_history'], 'r-', linewidth=2)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Diversity (Normalized Hamming Distance)')
    ax2.set_title('Population Diversity')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return plt


# ============================================================================
# EXAMPLE 1: Linear Rising Difficulty
# ============================================================================

def example_linear_progression():
    """Generate dungeon with smooth difficulty increase."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Linear Rising Difficulty (Tutorial → Boss)")
    print("="*70 + "\n")
    
    # Define target: smooth progression from easy to hard
    target_curve = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    
    print(f"Target Curve: {target_curve}")
    print("Expected behavior: Tutorial → Gradual Challenge → Boss Fight\n")
    
    # Create generator
    generator = EvolutionaryTopologyGenerator(
        target_curve=target_curve,
        population_size=40,
        generations=100,
        mutation_rate=0.15,
        crossover_rate=0.7,
        genome_length=20,
        seed=42,
    )
    
    # Evolve graph
    print("Starting evolution...")
    best_graph = generator.evolve()
    
    # Get statistics
    stats = generator.get_statistics()
    print(f"\nFinal Fitness: {stats['final_best_fitness']:.4f}")
    print(f"Generations: {stats['generations_run']}")
    print(f"Converged: {stats['converged']}")
    
    # Analyze graph
    print(f"\nGraph Properties:")
    print(f"  Nodes: {best_graph.number_of_nodes()}")
    print(f"  Edges: {best_graph.number_of_edges()}")
    
    # Count node types
    from collections import Counter
    node_types = Counter(best_graph.nodes[n]['type'] for n in best_graph.nodes())
    print(f"\nNode Type Distribution:")
    for ntype, count in sorted(node_types.items()):
        print(f"  {ntype}: {count}")
    
    return generator, best_graph


# ============================================================================
# EXAMPLE 2: Wave Pattern (Multiple Boss Fights)
# ============================================================================

def example_wave_pattern():
    """Generate dungeon with alternating difficulty (mini-boss pattern)."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Wave Pattern (Multiple Peaks)")
    print("="*70 + "\n")
    
    # Define target: waves of increasing difficulty
    target_curve = [0.2, 0.5, 0.3, 0.7, 0.4, 0.9, 0.5, 1.0]
    
    print(f"Target Curve: {target_curve}")
    print("Expected behavior: Mini-Boss → Recovery → Mini-Boss → Final Boss\n")
    
    # Create generator with higher diversity settings
    generator = EvolutionaryTopologyGenerator(
        target_curve=target_curve,
        population_size=50,
        generations=150,
        mutation_rate=0.2,  # Higher mutation for more exploration
        crossover_rate=0.6,
        genome_length=25,
        seed=123,
    )
    
    # Evolve graph
    print("Starting evolution...")
    best_graph = generator.evolve()
    
    # Get statistics
    stats = generator.get_statistics()
    print(f"\nFinal Fitness: {stats['final_best_fitness']:.4f}")
    print(f"Generations: {stats['generations_run']}")
    
    # Analyze complexity
    print(f"\nGraph Complexity:")
    print(f"  Total Nodes: {best_graph.number_of_nodes()}")
    print(f"  Average Degree: {2 * best_graph.number_of_edges() / best_graph.number_of_nodes():.2f}")
    
    return generator, best_graph


# ============================================================================
# EXAMPLE 3: Metroidvania Style (Backtracking)
# ============================================================================

def example_metroidvania():
    """Generate non-linear dungeon with backtracking."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Metroidvania Style (Non-Linear Exploration)")
    print("="*70 + "\n")
    
    # Define target: gradual build-up with plateau for exploration
    target_curve = [0.1, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 1.0]
    
    print(f"Target Curve: {target_curve}")
    print("Expected behavior: Exploration zones with gradual power-ups\n")
    
    # Customize transition matrix to favor branching
    custom_transitions = {
        "Start": {
            "InsertChallenge_ENEMY": 0.2,
            "InsertChallenge_PUZZLE": 0.2,
            "Branch": 0.5,  # Higher branching
            "InsertLockKey": 0.1
        },
        "Branch": {
            "InsertChallenge_ENEMY": 0.3,
            "InsertChallenge_PUZZLE": 0.3,
            "InsertLockKey": 0.2,
            "Branch": 0.2  # Can branch again
        },
        "InsertLockKey": {
            "Branch": 0.4,  # Lock-key often leads to branching
            "InsertChallenge_ENEMY": 0.3,
            "InsertChallenge_PUZZLE": 0.3
        },
        "InsertChallenge_ENEMY": {
            "InsertChallenge_ENEMY": 0.3,
            "Branch": 0.3,
            "InsertLockKey": 0.2,
            "InsertChallenge_PUZZLE": 0.2
        },
        "InsertChallenge_PUZZLE": {
            "Branch": 0.4,
            "InsertLockKey": 0.3,
            "InsertChallenge_ENEMY": 0.3
        },
    }
    
    # Create generator with custom transitions
    generator = EvolutionaryTopologyGenerator(
        target_curve=target_curve,
        zelda_transition_matrix=custom_transitions,
        population_size=60,
        generations=150,
        mutation_rate=0.18,
        crossover_rate=0.75,
        genome_length=30,  # Longer genome for more complex structures
        seed=456,
    )
    
    # Evolve graph
    print("Starting evolution...")
    best_graph = generator.evolve()
    
    # Analyze branching structure
    stats = generator.get_statistics()
    print(f"\nFinal Fitness: {stats['final_best_fitness']:.4f}")
    print(f"Branching Analysis:")
    
    # Count nodes by degree
    degree_dist = {}
    for node in best_graph.nodes():
        degree = best_graph.degree(node)
        degree_dist[degree] = degree_dist.get(degree, 0) + 1
    
    print(f"  Degree Distribution:")
    for degree in sorted(degree_dist.keys()):
        print(f"    Degree {degree}: {degree_dist[degree]} nodes")
    
    return generator, best_graph


# ============================================================================
# EXAMPLE 4: Quick Prototyping (Fast Iteration)
# ============================================================================

def example_quick_prototype():
    """Generate multiple dungeons quickly for testing."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Quick Prototyping (Generate 5 variants)")
    print("="*70 + "\n")
    
    target_curve = [0.2, 0.5, 0.8, 1.0]
    
    variants = []
    
    for i in range(5):
        print(f"\nVariant {i+1}:")
        
        generator = EvolutionaryTopologyGenerator(
            target_curve=target_curve,
            population_size=20,  # Small population
            generations=30,      # Few generations
            mutation_rate=0.2,
            crossover_rate=0.7,
            genome_length=12,    # Short genome
            seed=1000 + i,
        )
        
        graph = generator.evolve()
        stats = generator.get_statistics()
        
        print(f"  Fitness: {stats['final_best_fitness']:.4f}")
        print(f"  Nodes: {graph.number_of_nodes()}")
        print(f"  Generations: {stats['generations_run']}")
        
        variants.append((generator, graph))
    
    print("\nAll variants generated successfully!")
    return variants


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("EVOLUTIONARY TOPOLOGY DIRECTOR - USAGE EXAMPLES")
    print("="*70)
    
    # Run examples
    gen1, graph1 = example_linear_progression()
    gen2, graph2 = example_wave_pattern()
    gen3, graph3 = example_metroidvania()
    variants = example_quick_prototype()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\n✓ All examples completed successfully!")
    print("\nKey Takeaways:")
    print("  1. Different target curves produce different dungeon structures")
    print("  2. Custom transition matrices enable specific design patterns")
    print("  3. Population size and generations trade speed vs quality")
    print("  4. Even random initialization often produces good results")
    print("\nNext Steps:")
    print("  - Integrate with 2D layout generation (Block II)")
    print("  - Add more sophisticated fitness functions")
    print("  - Implement constraint injection (required items, etc.)")
    print("  - Export graphs for use in game engine")
    
    print("\n🎮 Ready to generate infinite dungeon variations!\n")
    
    # Optional: Save graphs for later use
    import pickle
    output_dir = PROJECT_ROOT / "results" / "generated_graphs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    graphs_to_save = {
        "linear_progression": graph1,
        "wave_pattern": graph2,
        "metroidvania": graph3,
    }
    
    for name, graph in graphs_to_save.items():
        filepath = output_dir / f"{name}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(graph, f)
        print(f"Saved {name} to {filepath}")


if __name__ == "__main__":
    main()
