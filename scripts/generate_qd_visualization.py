"""Generate QD visualizations for presentation."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # Load data
    n64_path = Path('results/map_elites_full_n64/matched_budget_raw.csv')
    n96_path = Path('results/map_elites_large_20260509/matched_budget_raw.csv')
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('MAP-Elites Quality-Diversity Analysis', fontsize=16, fontweight='bold')
    
    # N64 data
    if n64_path.exists():
        df64 = pd.read_csv(n64_path)
        
        # Plot 1: Linearity vs Leniency (N64)
        ax = axes[0, 0]
        scatter = ax.scatter(df64['linearity'], df64['leniency'], c=df64['fitness'], cmap='viridis', s=50)
        ax.set_xlabel('Linearity')
        ax.set_ylabel('Leniency')
        ax.set_title('N64: Linearity vs Leniency (n=64)')
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Fitness')
        
        # Plot 2: Progression vs Topology Complexity (N64)
        ax = axes[0, 1]
        scatter = ax.scatter(df64['progression_complexity'], df64['topology_complexity'], c=df64['fitness'], cmap='plasma', s=50)
        ax.set_xlabel('Progression Complexity')
        ax.set_ylabel('Topology Complexity')
        ax.set_title('N64: Complexity Space')
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Fitness')
        
        # Plot 3: Fitness distribution (N64)
        ax = axes[0, 2]
        ax.hist(df64['fitness'], bins=15, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Fitness')
        ax.set_ylabel('Count')
        mean_fitness = df64['fitness'].mean()
        ax.set_title(f'N64: Fitness Distribution (mean={mean_fitness:.3f})')
        ax.axvline(mean_fitness, color='red', linestyle='--', label='Mean')
        ax.legend()
    
    # N96 data
    if n96_path.exists():
        df96 = pd.read_csv(n96_path)
        
        # Plot 4: Linearity vs Leniency (N96)
        ax = axes[1, 0]
        scatter = ax.scatter(df96['linearity'], df96['leniency'], c=df96['fitness'], cmap='viridis', s=50)
        ax.set_xlabel('Linearity')
        ax.set_ylabel('Leniency')
        ax.set_title('N96: Linearity vs Leniency (n=96)')
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Fitness')
        
        # Plot 5: Progression vs Topology Complexity (N96)
        ax = axes[1, 1]
        scatter = ax.scatter(df96['progression_complexity'], df96['topology_complexity'], c=df96['fitness'], cmap='plasma', s=50)
        ax.set_xlabel('Progression Complexity')
        ax.set_ylabel('Topology Complexity')
        ax.set_title('N96: Complexity Space')
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Fitness')
        
        # Plot 6: Fitness distribution (N96)
        ax = axes[1, 2]
        ax.hist(df96['fitness'], bins=15, color='coral', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Fitness')
        ax.set_ylabel('Count')
        mean_fitness = df96['fitness'].mean()
        ax.set_title(f'N96: Fitness Distribution (mean={mean_fitness:.3f})')
        ax.axvline(mean_fitness, color='red', linestyle='--', label='Mean')
        ax.legend()
    
    plt.tight_layout()
    out_path = Path('results/presentation_assets_20260508/map_elites_qd_comprehensive.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Generated: {out_path}')

if __name__ == '__main__':
    main()
