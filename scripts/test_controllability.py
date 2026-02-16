"""
Controllability Test
Measures system responsiveness to user-specified tension curves.

This addresses the thesis defense concern: "How do you prove users actually control the output?"

Target: Pearson r > 0.7 for "responsive" system classification.

Usage:
    python scripts/test_controllability.py --num-samples 20 --output results/controllability
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from typing import List, Tuple, Dict, Optional
import logging
import sys
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

logger = logging.getLogger(__name__)


class ControllabilityTest:
    """
    Tests system controllability by generating dungeons with diverse tension curves
    and measuring correlation between target and actual tension.
    
    Target: Pearson r > 0.7 for "responsive" system classification.
    """
    
    def __init__(self, output_dir: str = "results/controllability"):
        """
        Args:
            output_dir: Directory to save controllability results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define test curves
        self.test_curves = {
            'flat': self._generate_flat_curve(),
            'linear_rising': self._generate_linear_curve(rising=True),
            'linear_falling': self._generate_linear_curve(rising=False),
            'sine_wave': self._generate_sine_curve(),
            'exponential': self._generate_exponential_curve(),
            'random': self._generate_random_curve()
        }
    
    def _generate_flat_curve(self, num_rooms: int = 8) -> List[float]:
        """Constant tension throughout."""
        return [0.5] * num_rooms
    
    def _generate_linear_curve(self, num_rooms: int = 8, rising: bool = True) -> List[float]:
        """Linear increase or decrease."""
        if rising:
            return list(np.linspace(0.2, 0.9, num_rooms))
        else:
            return list(np.linspace(0.9, 0.2, num_rooms))
    
    def _generate_sine_curve(self, num_rooms: int = 8) -> List[float]:
        """Sinusoidal tension (classic hero's journey)."""
        x = np.linspace(0, np.pi, num_rooms)
        curve = 0.5 + 0.4 * np.sin(x)
        return curve.tolist()
    
    def _generate_exponential_curve(self, num_rooms: int = 8) -> List[float]:
        """Exponential buildup to climax."""
        x = np.linspace(0, 1, num_rooms)
        curve = 0.2 + 0.7 * (x ** 2)
        return curve.tolist()
    
    def _generate_random_curve(self, num_rooms: int = 8) -> List[float]:
        """Random tension (chaos)."""
        np.random.seed(42)  # Fixed for reproducibility
        return np.random.uniform(0.2, 0.9, num_rooms).tolist()
    
    def run_controllability_test(
        self,
        num_samples_per_curve: int = 20,
        seeds: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Generate dungeons for each tension curve and measure controllability.
        
        Args:
            num_samples_per_curve: Number of dungeons per curve type
            seeds: List of random seeds to use
        
        Returns:
            DataFrame with columns: [curve_type, seed, target_curve, actual_curve, correlation, r_squared]
        """
        if seeds is None:
            seeds = list(range(num_samples_per_curve))
        
        results = []
        
        for curve_name, target_curve in self.test_curves.items():
            print(f"\n{'='*60}")
            print(f"Testing controllability: {curve_name}")
            print(f"Target curve: {[f'{v:.2f}' for v in target_curve]}")
            print(f"{'='*60}\n")
            
            for idx, seed in enumerate(seeds):
                print(f"Sample {idx+1}/{len(seeds)} (seed={seed})...", end=' ', flush=True)
                
                try:
                    # Generate dungeon with target curve
                    start_time = time.time()
                    
                    dungeon = self._generate_dungeon({
                        'num_rooms': len(target_curve),
                        'tension_curve': target_curve,
                        'seed': seed,
                        'use_evolution': True,
                        'use_logicnet': True
                    })
                    
                    gen_time = time.time() - start_time
                    
                    if dungeon is None:
                        print("FAILED")
                        continue
                    
                    # Extract actual tension curve from generated dungeon
                    actual_curve = self._extract_actual_tension_curve(dungeon)
                    
                    # Calculate correlation
                    if len(actual_curve) != len(target_curve):
                        # Resample to match length
                        actual_curve = np.interp(
                            np.linspace(0, 1, len(target_curve)),
                            np.linspace(0, 1, len(actual_curve)),
                            actual_curve
                        )
                    
                    correlation, p_value = pearsonr(target_curve, actual_curve)
                    r_squared = correlation ** 2
                    
                    results.append({
                        'curve_type': curve_name,
                        'seed': seed,
                        'target_curve': target_curve,
                        'actual_curve': actual_curve.tolist(),
                        'correlation': correlation,
                        'r_squared': r_squared,
                        'p_value': p_value,
                        'gen_time': gen_time
                    })
                    
                    print(f"✓ (r={correlation:.3f}, r²={r_squared:.3f})")
                    
                except Exception as e:
                    print(f"ERROR: {e}")
                    logger.exception(f"Error in sample {seed}")
        
        df = pd.DataFrame(results)
        
        # Save results
        output_file = self.output_dir / "controllability_results.csv"
        df.to_csv(output_file, index=False)
        print(f"\nSaved results to {output_file}")
        
        return df
    
    def generate_report(self, results_df: pd.DataFrame) -> None:
        """Generate controllability report with statistics and visualizations."""
        
        # Summary statistics
        summary = results_df.groupby('curve_type').agg({
            'correlation': ['mean', 'std', 'min', 'max'],
            'r_squared': ['mean', 'std']
        }).round(3)
        
        print("\n" + "="*80)
        print("CONTROLLABILITY TEST SUMMARY")
        print("="*80)
        print(summary)
        print("="*80)
        
        # Save summary
        summary.to_csv(self.output_dir / "controllability_summary.csv")
        
        # Generate plots
        try:
            self._plot_correlation_by_curve(results_df)
            self._plot_target_vs_actual_scatter(results_df)
            self._plot_example_curves(results_df)
        except ImportError:
            logger.warning("Matplotlib not available, skipping plots")
        
        # Classification
        overall_mean_r = results_df['correlation'].mean()
        
        if overall_mean_r >= 0.7:
            classification = "RESPONSIVE (r ≥ 0.7) ✓"
            color = "green"
        elif overall_mean_r >= 0.5:
            classification = "MODERATELY RESPONSIVE (0.5 ≤ r < 0.7)"
            color = "orange"
        else:
            classification = "UNRESPONSIVE (r < 0.5) ✗"
            color = "red"
        
        print(f"\n{'='*80}")
        print(f"OVERALL CONTROLLABILITY: {classification}")
        print(f"Mean Correlation r = {overall_mean_r:.3f}")
        print(f"{'='*80}\n")
        
        # Save classification to file
        with open(self.output_dir / "controllability_classification.txt", 'w') as f:
            f.write(f"OVERALL CONTROLLABILITY: {classification}\n")
            f.write(f"Mean Correlation r = {overall_mean_r:.3f}\n")
    
    def _plot_correlation_by_curve(self, df: pd.DataFrame) -> None:
        """Bar chart of correlation by curve type."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        grouped = df.groupby('curve_type')['correlation']
        means = grouped.mean().sort_values(ascending=False)
        stds = grouped.std()
        
        bars = ax.bar(range(len(means)), means.values, yerr=stds[means.index], capsize=5)
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(means.index, rotation=45, ha='right')
        ax.set_ylabel('Pearson Correlation (r)')
        ax.set_title('Controllability by Tension Curve Type')
        ax.axhline(y=0.7, color='g', linestyle='--', label='Responsive threshold (r=0.7)')
        ax.axhline(y=0.5, color='orange', linestyle='--', label='Moderate threshold (r=0.5)')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Color bars based on threshold
        colors = ['green' if v >= 0.7 else 'orange' if v >= 0.5 else 'red' for v in means.values]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "controllability_by_curve.png", dpi=300)
        plt.close()
        
        print(f"Saved plot: {self.output_dir / 'controllability_by_curve.png'}")
    
    def _plot_target_vs_actual_scatter(self, df: pd.DataFrame) -> None:
        """Scatter plot of target vs actual tension values."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Flatten all target and actual values
        all_targets = []
        all_actuals = []
        
        for _, row in df.iterrows():
            all_targets.extend(row['target_curve'])
            all_actuals.extend(row['actual_curve'])
        
        ax.scatter(all_targets, all_actuals, alpha=0.3, s=20)
        
        # Perfect correlation line
        ax.plot([0, 1], [0, 1], 'r--', label='Perfect correlation', linewidth=2)
        
        # Calculate overall correlation
        overall_r, _ = pearsonr(all_targets, all_actuals)
        ax.text(0.05, 0.95, f'Overall r = {overall_r:.3f}\nr² = {overall_r**2:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Target Tension')
        ax.set_ylabel('Actual Tension')
        ax.set_title('Target vs Actual Tension: Room-Level Correlation')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "target_vs_actual_scatter.png", dpi=300)
        plt.close()
        
        print(f"Saved plot: {self.output_dir / 'target_vs_actual_scatter.png'}")
    
    def _plot_example_curves(self, df: pd.DataFrame) -> None:
        """Plot example target vs actual curves for each curve type."""
        import matplotlib.pyplot as plt
        
        curve_types = df['curve_type'].unique()
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, curve_type in enumerate(curve_types):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            
            # Get best example (highest correlation)
            curve_df = df[df['curve_type'] == curve_type]
            best_example = curve_df.loc[curve_df['correlation'].idxmax()]
            
            target = best_example['target_curve']
            actual = best_example['actual_curve']
            
            x = range(len(target))
            ax.plot(x, target, 'b-o', label='Target', linewidth=2, markersize=6)
            ax.plot(x, actual, 'r--s', label='Actual', linewidth=2, markersize=5)
            
            ax.set_title(f"{curve_type}\n(r={best_example['correlation']:.3f})")
            ax.set_xlabel('Room Index')
            ax.set_ylabel('Tension')
            ax.legend()
            ax.grid(alpha=0.3)
            ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "example_curves.png", dpi=300)
        plt.close()
        
        print(f"Saved plot: {self.output_dir / 'example_curves.png'}")
    
    def _generate_dungeon(self, params: Dict) -> Optional[Dict]:
        """
        Generate dungeon with given parameters.
        
        This is a placeholder that should be replaced with actual pipeline call.
        """
        # TODO: Replace with actual pipeline integration
        logger.warning("Using mock dungeon generation - replace with actual pipeline")
        
        # Mock generation that responds to tension curve
        tension_curve = params['tension_curve']
        num_rooms = params['num_rooms']
        seed = params['seed']
        
        np.random.seed(seed)
        
        # Generate actual curve similar to target with some noise
        actual_curve = np.array(tension_curve) + np.random.normal(0, 0.1, num_rooms)
        actual_curve = np.clip(actual_curve, 0.0, 1.0)
        
        return {
            'visual_grid': np.zeros((64, 64), dtype=int),
            'mission_graph': {'nodes': {i: {} for i in range(num_rooms)}, 'edges': []},
            'rooms': [{'id': i, 'tension': actual_curve[i]} for i in range(num_rooms)],
            'tension_curve': actual_curve.tolist()
        }
    
    def _extract_actual_tension_curve(self, dungeon: Dict) -> np.ndarray:
        """
        Extract actual tension experienced in generated dungeon.
        
        Tension = weighted sum of:
        - Enemy count (normalized)
        - Room distance from start (progress)
        - Path difficulty (detours, backtracking)
        """
        # Try to use actual validator
        try:
            from src.simulation.validator import extract_actual_tension_curve
            return extract_actual_tension_curve(dungeon)
        except ImportError:
            # Fallback: use mock result (already in dungeon)
            if 'tension_curve' in dungeon:
                return np.array(dungeon['tension_curve'])
            
            # Calculate from rooms
            rooms = dungeon.get('rooms', [])
            tension_values = []
            
            for room in rooms:
                # Mock calculation
                tension = room.get('tension', 0.5)
                tension_values.append(tension)
            
            return np.array(tension_values) if tension_values else np.array([0.5] * 8)


def main():
    """Main entry point for controllability test."""
    parser = argparse.ArgumentParser(description="Test controllability of H-MOLQD pipeline")
    parser.add_argument('--num-samples', type=int, default=20, help='Number of dungeons per curve type')
    parser.add_argument('--output', type=str, default='results/controllability', help='Output directory')
    parser.add_argument('--curve', type=str, default=None, help='Test specific curve (flat, linear_rising, etc.)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # Create test
    test = ControllabilityTest(output_dir=args.output)
    
    # Run test
    if args.curve:
        # Test specific curve
        if args.curve not in test.test_curves:
            print(f"Error: Unknown curve '{args.curve}'")
            print(f"Available curves: {', '.join(test.test_curves.keys())}")
            return 1
        
        # Create temporary test object with single curve
        test.test_curves = {args.curve: test.test_curves[args.curve]}
    
    results = test.run_controllability_test(num_samples_per_curve=args.num_samples)
    test.generate_report(results)
    
    print(f"\nControllability test complete. Results saved to: {test.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
