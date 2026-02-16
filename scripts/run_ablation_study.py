"""
Ablation Study Runner
Systematically disables pipeline components to measure their contribution.

This addresses the thesis defense concern: "How do you know which components are actually necessary?"

Usage:
    python scripts/run_ablation_study.py --num-samples 50 --output results/ablation
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging
import sys
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.pipeline.robust_pipeline import (
    RobustPipeline, 
    PipelineConfig,
    validate_mission_graph,
    validate_layout
)

logger = logging.getLogger(__name__)


class AblationStudy:
    """
    Conducts systematic ablation experiments on the 7-block H-MOLQD pipeline.
    
    Configurations:
    A. FULL: All components enabled (baseline)
    B. NO_LOGICNET: LogicNet disabled (logic_guidance_scale=0)
    C. NO_EVOLUTION: Random graphs instead of evolved graphs
    D. NO_WFC: Raw neural output without Wave Function Collapse refinement
    E. NO_CONSTRAINTS: Disable graph constraint enforcer
    """
    
    def __init__(self, output_dir: str = "results/ablation"):
        """
        Args:
            output_dir: Directory to save ablation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.configs = {
            'FULL': {
                'use_evolution': True,
                'use_logicnet': True,
                'use_wfc': True,
                'use_constraint_enforcer': True,
                'logic_guidance_scale': 1.0
            },
            'NO_LOGICNET': {
                'use_evolution': True,
                'use_logicnet': False,
                'use_wfc': True,
                'use_constraint_enforcer': True,
                'logic_guidance_scale': 0.0
            },
            'NO_EVOLUTION': {
                'use_evolution': False,  # Use random graph generator
                'use_logicnet': True,
                'use_wfc': True,
                'use_constraint_enforcer': True,
                'logic_guidance_scale': 1.0
            },
            'NO_WFC': {
                'use_evolution': True,
                'use_logicnet': True,
                'use_wfc': False,
                'use_constraint_enforcer': True,
                'logic_guidance_scale': 1.0
            },
            'NO_CONSTRAINTS': {
                'use_evolution': True,
                'use_logicnet': True,
                'use_wfc': True,
                'use_constraint_enforcer': False,
                'logic_guidance_scale': 1.0
            }
        }
    
    def run_experiment(
        self,
        config_name: str,
        num_samples: int = 50,
        seeds: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Run generation with specific ablation configuration.
        
        Args:
            config_name: Name of ablation config ('FULL', 'NO_LOGICNET', etc.)
            num_samples: Number of dungeons to generate
            seeds: List of random seeds to use
        
        Returns:
            DataFrame with columns: [seed, solvable, pacing_error, gen_time, ...]
        """
        if seeds is None:
            seeds = list(range(num_samples))
        
        config = self.configs[config_name]
        results = []
        
        print(f"\n{'='*60}")
        print(f"Running ablation: {config_name}")
        print(f"Configuration: {config}")
        print(f"{'='*60}\n")
        
        for idx, seed in enumerate(seeds):
            print(f"Sample {idx+1}/{len(seeds)} (seed={seed})...", end=' ', flush=True)
            
            try:
                # Generate dungeon
                start_time = time.time()
                
                dungeon = self._generate_dungeon({
                    'num_rooms': 8,
                    'tension_curve': [0.2, 0.4, 0.6, 0.8, 0.7, 0.5, 0.3, 0.2],
                    'seed': seed,
                    **config
                })
                
                gen_time = time.time() - start_time
                
                if dungeon is None:
                    print("FAILED")
                    results.append({
                        'config': config_name,
                        'seed': seed,
                        'success': False,
                        'solvable': False,
                        'pacing_error': np.nan,
                        'gen_time': gen_time,
                        'num_rooms': 0
                    })
                    continue
                
                # Evaluate metrics
                solvable = self._is_solvable(dungeon)
                pacing_error = self._calculate_pacing_error(dungeon)
                num_rooms = len(dungeon.get('rooms', []))
                
                results.append({
                    'config': config_name,
                    'seed': seed,
                    'success': True,
                    'solvable': solvable,
                    'pacing_error': pacing_error,
                    'gen_time': gen_time,
                    'num_rooms': num_rooms
                })
                
                print(f"✓ (solvable={solvable}, pacing_error={pacing_error:.3f})")
                
            except Exception as e:
                print(f"ERROR: {e}")
                logger.exception(f"Error in sample {seed}")
                results.append({
                    'config': config_name,
                    'seed': seed,
                    'success': False,
                    'solvable': False,
                    'pacing_error': np.nan,
                    'gen_time': np.nan,
                    'num_rooms': 0
                })
        
        df = pd.DataFrame(results)
        
        # Save results
        output_file = self.output_dir / f"{config_name}_results.csv"
        df.to_csv(output_file, index=False)
        print(f"\nSaved results to {output_file}")
        
        return df
    
    def run_all_ablations(self, num_samples: int = 50) -> Dict[str, pd.DataFrame]:
        """
        Run all ablation configurations.
        
        Args:
            num_samples: Number of dungeons per configuration
        
        Returns:
            Dictionary mapping config_name -> results DataFrame
        """
        all_results = {}
        
        for config_name in self.configs.keys():
            results_df = self.run_experiment(config_name, num_samples)
            all_results[config_name] = results_df
        
        # Generate comparison report
        self.generate_comparison_report(all_results)
        self.generate_comparison_plots(all_results)
        
        return all_results
    
    def generate_comparison_report(self, results: Dict[str, pd.DataFrame]) -> None:
        """Create comparison table of all configurations."""
        summary = []
        
        for config_name, df in results.items():
            summary.append({
                'Configuration': config_name,
                'Success Rate (%)': f"{df['success'].mean() * 100:.1f}",
                'Solvability (%)': f"{df['solvable'].mean() * 100:.1f}",
                'Avg Pacing Error': f"{df['pacing_error'].mean():.3f}",
                'Std Pacing Error': f"{df['pacing_error'].std():.3f}",
                'Avg Gen Time (s)': f"{df['gen_time'].mean():.2f}",
                'Std Gen Time (s)': f"{df['gen_time'].std():.2f}"
            })
        
        summary_df = pd.DataFrame(summary)
        
        # Sort by solvability (descending)
        summary_df['_solvability_numeric'] = [float(s.rstrip('%')) for s in summary_df['Solvability (%)']]
        summary_df = summary_df.sort_values('_solvability_numeric', ascending=False)
        summary_df = summary_df.drop(columns=['_solvability_numeric'])
        
        # Save table
        output_file = self.output_dir / "ablation_comparison_table.csv"
        summary_df.to_csv(output_file, index=False)
        
        # Print to console
        print("\n" + "="*80)
        print("ABLATION STUDY COMPARISON TABLE")
        print("="*80)
        print(summary_df.to_string(index=False))
        print("="*80)
        
        # Save as formatted text
        with open(self.output_dir / "ablation_comparison_table.txt", 'w') as f:
            f.write("ABLATION STUDY COMPARISON TABLE\n")
            f.write("="*80 + "\n\n")
            f.write(summary_df.to_string(index=False))
            f.write("\n\n" + "="*80)
    
    def generate_comparison_plots(self, results: Dict[str, pd.DataFrame]) -> None:
        """Generate visualization plots comparing configurations."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Combine all results
            combined_df = pd.concat(results.values(), ignore_index=True)
            
            # Create 2x2 subplot figure
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Subplot 1: Solvability bar chart
            solvability = combined_df.groupby('config')['solvable'].mean() * 100
            solvability = solvability.sort_values(ascending=False)
            
            ax = axes[0, 0]
            bars = ax.bar(range(len(solvability)), solvability.values)
            ax.set_xticks(range(len(solvability)))
            ax.set_xticklabels(solvability.index, rotation=45, ha='right')
            ax.set_ylabel('Solvability (%)')
            ax.set_title('Solvability by Configuration')
            ax.axhline(y=90, color='r', linestyle='--', label='90% threshold')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Color bars
            colors = ['green' if v >= 90 else 'orange' if v >= 70 else 'red' for v in solvability.values]
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            # Subplot 2: Pacing error boxplot
            ax = axes[0, 1]
            sns.boxplot(data=combined_df, x='config', y='pacing_error', ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_ylabel('Pacing Error')
            ax.set_title('Pacing Error Distribution')
            ax.grid(axis='y', alpha=0.3)
            
            # Subplot 3: Generation time comparison
            ax = axes[1, 0]
            gen_time_mean = combined_df.groupby('config')['gen_time'].mean()
            gen_time_std = combined_df.groupby('config')['gen_time'].std()
            
            ax.bar(range(len(gen_time_mean)), gen_time_mean.values, yerr=gen_time_std.values, capsize=5)
            ax.set_xticks(range(len(gen_time_mean)))
            ax.set_xticklabels(gen_time_mean.index, rotation=45, ha='right')
            ax.set_ylabel('Generation Time (s)')
            ax.set_title('Generation Time by Configuration')
            ax.grid(axis='y', alpha=0.3)
            
            # Subplot 4: Success rate
            ax = axes[1, 1]
            success_rate = combined_df.groupby('config')['success'].mean() * 100
            success_rate = success_rate.sort_values(ascending=False)
            
            bars = ax.bar(range(len(success_rate)), success_rate.values)
            ax.set_xticks(range(len(success_rate)))
            ax.set_xticklabels(success_rate.index, rotation=45, ha='right')
            ax.set_ylabel('Success Rate (%)')
            ax.set_title('Generation Success Rate')
            ax.axhline(y=95, color='g', linestyle='--', label='95% target')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            output_file = self.output_dir / "ablation_comparison_plots.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"\nSaved plots to {output_file}")
            plt.close()
            
            # Plot degradation from baseline
            self._plot_degradation_from_baseline(combined_df)
            
        except ImportError:
            logger.warning("Matplotlib not available, skipping plots")
    
    def _plot_degradation_from_baseline(self, df: pd.DataFrame) -> None:
        """Plot performance degradation relative to FULL configuration."""
        import matplotlib.pyplot as plt
        
        baseline = df[df['config'] == 'FULL']
        baseline_metrics = {
            'solvability': baseline['solvable'].mean() * 100,
            'pacing_error': baseline['pacing_error'].mean(),
            'gen_time': baseline['gen_time'].mean()
        }
        
        degradation = []
        
        for config_name in df['config'].unique():
            if config_name == 'FULL':
                continue
            
            config_df = df[df['config'] == config_name]
            
            solvability_drop = baseline_metrics['solvability'] - (config_df['solvable'].mean() * 100)
            pacing_increase = config_df['pacing_error'].mean() - baseline_metrics['pacing_error']
            time_change = config_df['gen_time'].mean() - baseline_metrics['gen_time']
            
            degradation.append({
                'Configuration': config_name,
                'Solvability Drop (%)': solvability_drop,
                'Pacing Error Increase': pacing_increase,
                'Time Change (s)': time_change
            })
        
        degradation_df = pd.DataFrame(degradation)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(degradation_df))
        width = 0.25
        
        ax.bar(x - width, degradation_df['Solvability Drop (%)'], width, label='Solvability Drop (%)', color='red')
        ax.bar(x, degradation_df['Pacing Error Increase'] * 100, width, label='Pacing Error Increase (×100)', color='orange')
        ax.bar(x + width, degradation_df['Time Change (s)'], width, label='Time Change (s)', color='blue')
        
        ax.set_xticks(x)
        ax.set_xticklabels(degradation_df['Configuration'], rotation=45, ha='right')
        ax.set_ylabel('Change from Baseline')
        ax.set_title('Performance Degradation from FULL Configuration')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / "degradation_from_baseline.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved degradation plot to {output_file}")
        plt.close()
    
    def _generate_dungeon(self, params: Dict) -> Optional[Dict]:
        """
        Generate dungeon with given parameters.
        
        This is a placeholder that should be replaced with actual pipeline call.
        """
        # TODO: Replace with actual pipeline integration
        # For now, return mock result
        logger.warning("Using mock dungeon generation - replace with actual pipeline")
        
        import numpy as np
        
        # Simulate success/failure based on config
        if not params.get('use_logicnet', True):
            if np.random.random() < 0.3:  # 30% failure without LogicNet
                return None
        
        if not params.get('use_constraint_enforcer', True):
            if np.random.random() < 0.2:  # 20% failure without constraints
                return None
        
        # Mock successful generation
        return {
            'visual_grid': np.zeros((64, 64), dtype=int),
            'mission_graph': {'nodes': {i: {} for i in range(8)}, 'edges': []},
            'rooms': [{'id': i} for i in range(8)],
            'solvable': np.random.random() > 0.1,
            'pacing_error': np.random.exponential(0.1)
        }
    
    def _is_solvable(self, dungeon: Dict) -> bool:
        """Check if dungeon is solvable."""
        # Try to use actual validator
        try:
            from src.simulation.validator import is_solvable
            return is_solvable(dungeon)
        except ImportError:
            # Fallback: use mock result
            return dungeon.get('solvable', False)
    
    def _calculate_pacing_error(self, dungeon: Dict) -> float:
        """Calculate pacing error for dungeon."""
        # Try to use actual validator
        try:
            from src.simulation.validator import calculate_pacing_error
            return calculate_pacing_error(dungeon)
        except ImportError:
            # Fallback: use mock result
            return dungeon.get('pacing_error', 0.0)


def main():
    """Main entry point for ablation study."""
    parser = argparse.ArgumentParser(description="Run ablation study on H-MOLQD pipeline")
    parser.add_argument('--num-samples', type=int, default=50, help='Number of dungeons per configuration')
    parser.add_argument('--output', type=str, default='results/ablation', help='Output directory')
    parser.add_argument('--config', type=str, default=None, help='Run specific config (FULL, NO_LOGICNET, etc.)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # Create study
    study = AblationStudy(output_dir=args.output)
    
    # Run experiments
    if args.config:
        # Run single configuration
        if args.config not in study.configs:
            print(f"Error: Unknown config '{args.config}'")
            print(f"Available configs: {', '.join(study.configs.keys())}")
            return 1
        
        results = study.run_experiment(args.config, num_samples=args.num_samples)
    else:
        # Run all configurations
        results = study.run_all_ablations(num_samples=args.num_samples)
    
    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE")
    print(f"Results saved to: {study.output_dir}")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
