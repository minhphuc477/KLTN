"""
Controllability Test
Measures system responsiveness to user-specified tension curves.

This addresses the thesis defense concern: "How do you prove users actually control the output?"

The script is checkpoint-backed by default. It refuses to publish-looking
numbers from random-weight generation or a surrogate. ``--allow-surrogate-smoke``
is available only to exercise the report schema without trained artifacts.

Usage:
    python scripts/validate_controllability.py --num-samples 20 --output results/controllability \
        --vqvae-checkpoint outputs/.../vqvae_pretrained.pth \
        --diffusion-checkpoint outputs/.../best_model.pth
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from typing import List, Tuple, Dict, Optional
import logging
import sys
import time
import math

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


class ControllabilityTest:
    """
    Tests system controllability by generating dungeons with diverse tension curves
    and measuring correlation between target and actual tension.
    
    The statistical protocol belongs in ``run_designer_controllability_proof``.
    This legacy convenience wrapper deliberately refuses to masquerade
    random-weight or surrogate output as model evidence.
    """
    
    def __init__(
        self,
        output_dir: str = "results/controllability",
        *,
        vqvae_checkpoint: Optional[str] = None,
        diffusion_checkpoint: Optional[str] = None,
        logic_net_checkpoint: Optional[str] = None,
        device: str = "auto",
        guidance_scale: float = 3.0,
        logic_guidance_scale: float = 0.0,
        num_diffusion_steps: int = 25,
        responsive_correlation_threshold: float = 0.7,
        allow_surrogate_smoke: bool = False,
    ):
        """
        Args:
            output_dir: Directory to save controllability results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._pipeline = None
        self._pipeline_init_attempted = False
        self.vqvae_checkpoint = Path(vqvae_checkpoint) if vqvae_checkpoint else None
        self.diffusion_checkpoint = Path(diffusion_checkpoint) if diffusion_checkpoint else None
        self.logic_net_checkpoint = Path(logic_net_checkpoint) if logic_net_checkpoint else None
        self.device = str(device)
        self.guidance_scale = float(guidance_scale)
        self.logic_guidance_scale = float(logic_guidance_scale)
        self.num_diffusion_steps = int(max(1, num_diffusion_steps))
        self.responsive_correlation_threshold = float(responsive_correlation_threshold)
        self.allow_surrogate_smoke = bool(allow_surrogate_smoke)
        self.evidence_mode = "surrogate_smoke" if self.allow_surrogate_smoke else "checkpoint_backed"
        
        # Define test curves
        self.test_curves = {
            'flat': self._generate_flat_curve(),
            'linear_rising': self._generate_linear_curve(rising=True),
            'linear_falling': self._generate_linear_curve(rising=False),
            'sine_wave': self._generate_sine_curve(),
            'exponential': self._generate_exponential_curve(),
            'random': self._generate_random_curve()
        }

    def _get_generation_pipeline(self):
        """Lazily initialize and cache the neural-symbolic pipeline."""
        if self._pipeline_init_attempted:
            return self._pipeline

        self._pipeline_init_attempted = True
        missing = [
            name
            for name, path in (
                ("vqvae_checkpoint", self.vqvae_checkpoint),
                ("diffusion_checkpoint", self.diffusion_checkpoint),
            )
            if path is None or not path.is_file()
        ]
        if missing:
            if self.allow_surrogate_smoke:
                logger.warning(
                    "Using explicitly requested surrogate smoke mode; missing trained artifacts: %s",
                    ", ".join(missing),
                )
                self._pipeline = None
                return None
            raise FileNotFoundError(
                "Controllability evidence requires trained checkpoints. Missing or unreadable: "
                + ", ".join(missing)
            )
        try:
            from src.pipeline.dungeon_pipeline import NeuralSymbolicDungeonPipeline
            self._pipeline = NeuralSymbolicDungeonPipeline.from_legacy_kwargs(
                vqvae_checkpoint=str(self.vqvae_checkpoint),
                diffusion_checkpoint=str(self.diffusion_checkpoint),
                logic_net_checkpoint=(str(self.logic_net_checkpoint) if self.logic_net_checkpoint else None),
                device=self.device,
                enable_logging=False,
                strict_checkpoint_mode=True,
            )
            logger.info("Initialized checkpoint-backed pipeline for controllability generation")
        except (ImportError, RuntimeError, ValueError, TypeError, OSError) as exc:
            if not self.allow_surrogate_smoke:
                raise RuntimeError(
                    "Could not initialize checkpoint-backed controllability pipeline"
                ) from exc
            self._pipeline = None
            logger.warning(
                "Pipeline unavailable; using explicitly requested surrogate smoke mode",
                exc_info=True,
            )
        return self._pipeline
    
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

    @staticmethod
    def _safe_pearson(x: List[float], y: List[float]) -> Tuple[float, float]:
        """Compute Pearson correlation safely for constant/degenerate inputs."""
        if len(x) < 2 or len(y) < 2:
            return float('nan'), float('nan')
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if np.std(x_arr) == 0 or np.std(y_arr) == 0:
            return float('nan'), float('nan')
        return pearsonr(x_arr, y_arr)
    
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
            DataFrame with paired target and observed semantic-tension curves.
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
                        'use_logicnet': self.logic_net_checkpoint is not None,
                    })
                    
                    gen_time = time.time() - start_time
                    
                    if dungeon is None:
                        print("FAILED")
                        continue
                    
                    # Measure generated room semantics, never the requested target.
                    actual_curve = self._extract_actual_tension_curve(dungeon)
                    
                    # Calculate correlation
                    if len(actual_curve) != len(target_curve):
                        # Resample to match length
                        actual_curve = np.interp(
                            np.linspace(0, 1, len(target_curve)),
                            np.linspace(0, 1, len(actual_curve)),
                            actual_curve
                        )
                    
                    correlation, p_value = self._safe_pearson(target_curve, actual_curve)
                    r_squared = correlation ** 2 if not math.isnan(correlation) else float('nan')
                    
                    results.append({
                        'curve_type': curve_name,
                        'seed': seed,
                        'target_curve': target_curve,
                        'actual_curve': actual_curve.tolist(),
                        'actual_curve_source': str(dungeon.get('actual_curve_source', 'unknown')),
                        'correlation': correlation,
                        'r_squared': r_squared,
                        'p_value': p_value,
                        'gen_time': gen_time,
                        'evidence_mode': self.evidence_mode,
                    })
                    
                    print(f"OK (r={correlation:.3f}, r^2={r_squared:.3f})")
                    
                except (AttributeError, RuntimeError, ValueError, TypeError) as e:
                    print(f"ERROR: {e}")
                    logger.exception(f"Error in sample {seed}")
        
        df = pd.DataFrame(results)
        
        # Save results
        output_file = self.output_dir / "controllability_results.csv"
        df.to_csv(output_file, index=False)
        provenance = {
            "evidence_mode": self.evidence_mode,
            "vqvae_checkpoint": str(self.vqvae_checkpoint) if self.vqvae_checkpoint else None,
            "diffusion_checkpoint": str(self.diffusion_checkpoint) if self.diffusion_checkpoint else None,
            "logic_net_checkpoint": str(self.logic_net_checkpoint) if self.logic_net_checkpoint else None,
            "guidance_scale": self.guidance_scale,
            "logic_guidance_scale": self.logic_guidance_scale,
            "num_diffusion_steps": self.num_diffusion_steps,
            "responsive_correlation_threshold": self.responsive_correlation_threshold,
        }
        (self.output_dir / "controllability_evidence.json").write_text(
            json.dumps(provenance, indent=2), encoding="utf-8"
        )
        print(f"\nSaved results to {output_file}")
        
        return df
    
    def generate_report(self, results_df: pd.DataFrame) -> None:
        """Generate controllability report with statistics and visualizations."""
        if results_df is None or results_df.empty:
            print("No controllability results available to report.")
            return
        
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
        overall_mean_r = results_df['correlation'].dropna().mean()
        if math.isnan(overall_mean_r):
            overall_mean_r = 0.0
        
        threshold = self.responsive_correlation_threshold
        moderate_threshold = max(0.0, threshold - 0.2)
        if overall_mean_r >= threshold:
            classification = f"RESPONSIVE (r >= {threshold:.2f})"
        elif overall_mean_r >= moderate_threshold:
            classification = f"MODERATELY RESPONSIVE ({moderate_threshold:.2f} <= r < {threshold:.2f})"
        else:
            classification = f"UNRESPONSIVE (r < {moderate_threshold:.2f})"
        if self.evidence_mode != "checkpoint_backed":
            classification = "SMOKE ONLY - NOT MODEL EVIDENCE: " + classification
        
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
        
        _fig, ax = plt.subplots(figsize=(10, 6))
        
        grouped = df.groupby('curve_type')['correlation']
        means = grouped.mean().sort_values(ascending=False)
        stds = grouped.std()
        
        bars = ax.bar(range(len(means)), means.values, yerr=stds[means.index], capsize=5)
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(means.index, rotation=45, ha='right')
        ax.set_ylabel('Pearson Correlation (r)')
        ax.set_title('Controllability by Tension Curve Type')
        threshold = self.responsive_correlation_threshold
        moderate_threshold = max(0.0, threshold - 0.2)
        ax.axhline(y=threshold, color='g', linestyle='--', label=f'Responsive threshold (r={threshold:.2f})')
        ax.axhline(y=moderate_threshold, color='orange', linestyle='--', label=f'Moderate threshold (r={moderate_threshold:.2f})')
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
        
        _fig, ax = plt.subplots(figsize=(8, 8))
        
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
        overall_r, _ = self._safe_pearson(all_targets, all_actuals)
        overall_r2 = overall_r ** 2 if not math.isnan(overall_r) else float('nan')
        ax.text(0.05, 0.95, f'Overall r = {overall_r:.3f}\nr^2 = {overall_r2:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Target Tension')
        ax.set_ylabel('Observed Semantic Tension')
        ax.set_title('Target vs Observed Semantic Tension: Room-Level Correlation')
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
        
        _fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, curve_type in enumerate(curve_types):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            
            # Get best example (highest finite correlation). Fall back to first row.
            curve_df = df[df['curve_type'] == curve_type]
            finite = curve_df[curve_df['correlation'].notna()]
            best_example = finite.loc[finite['correlation'].idxmax()] if not finite.empty else curve_df.iloc[0]
            
            target = best_example['target_curve']
            actual = best_example['actual_curve']
            
            x = range(len(target))
            ax.plot(x, target, 'b-o', label='Target', linewidth=2, markersize=6)
            ax.plot(x, actual, 'r--s', label='Observed semantic proxy', linewidth=2, markersize=5)
            
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
        Generate a checkpoint-backed dungeon, or an explicitly labeled smoke surrogate.
        """
        pipeline = self._get_generation_pipeline()
        if pipeline is not None:
            try:
                # Use evolutionary topology generation guided by target tension curve.
                res = pipeline.generate_dungeon(
                    mission_graph=None,
                    generate_topology=True,
                    target_curve=params.get('tension_curve'),
                    num_rooms=int(params.get('num_rooms', 8)),
                    seed=int(params.get('seed', 0)),
                    guidance_scale=self.guidance_scale,
                    logic_guidance_scale=self.logic_guidance_scale,
                    num_diffusion_steps=self.num_diffusion_steps,
                    apply_repair=True,
                )

                out = {
                    'visual_grid': getattr(res, 'dungeon_grid', np.zeros((64, 64), dtype=int)),
                    'mission_graph': {'nodes': {}, 'edges': []},
                    'rooms': [],
                    'actual_curve_source': 'room_semantic_proxy',
                }
                try:
                    mission_graph = getattr(res, 'mission_graph', None)
                    if mission_graph is not None:
                        out['mission_graph'] = {
                            'nodes': dict(mission_graph.nodes(data=True)),
                            'edges': list(mission_graph.edges()),
                        }
                    for rid, room in getattr(res, 'rooms', {}).items():
                        room_grid = getattr(room, 'room_grid', None)
                        if room_grid is None:
                            raise RuntimeError(f"Generated room {rid!r} has no semantic room grid")
                        out['rooms'].append({'id': rid, 'grid': np.asarray(room_grid, dtype=np.int32)})
                    out['room_order'] = self._resolve_room_order(
                        mission_graph,
                        [room['id'] for room in out['rooms']],
                    )
                except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    if not self.allow_surrogate_smoke:
                        raise RuntimeError("Could not extract generated room semantics") from exc
                    logger.debug('Could not fully unpack pipeline result', exc_info=True)
                return out
            except (AttributeError, RuntimeError, ValueError, TypeError, OSError) as exc:
                if not self.allow_surrogate_smoke:
                    raise RuntimeError("Checkpoint-backed generation failed") from exc
                logger.debug("Pipeline generation failed; using requested surrogate smoke mode", exc_info=True)

        if not self.allow_surrogate_smoke:
            raise RuntimeError("No checkpoint-backed generation pipeline is available")

        # Deterministic schema smoke path. Its outputs are never model evidence.
        tension_curve = params.get('tension_curve', [0.5] * params.get('num_rooms', 8))
        num_rooms = int(params.get('num_rooms', len(tension_curve)))
        seed = int(params.get('seed', 0))

        np.random.seed(seed)
        actual_curve = np.array(tension_curve) + np.random.normal(0, 0.1, num_rooms)
        actual_curve = np.clip(actual_curve, 0.0, 1.0)

        return {
            'visual_grid': np.zeros((64, 64), dtype=int),
            'mission_graph': {'nodes': {i: {} for i in range(num_rooms)}, 'edges': []},
            'rooms': [{'id': i, 'synthetic_tension': float(actual_curve[i])} for i in range(num_rooms)],
            'room_order': list(range(num_rooms)),
            'actual_curve_source': 'surrogate_smoke',
        }

    @staticmethod
    def _resolve_room_order(mission_graph: object, room_ids: List[object]) -> List[object]:
        """Return a deterministic start-to-goal room order when topology exposes one."""
        available = set(room_ids)
        try:
            import networkx as nx

            if not isinstance(mission_graph, nx.Graph):
                raise TypeError("mission graph is not a NetworkX graph")
            start_nodes = [
                node_id
                for node_id, attrs in mission_graph.nodes(data=True)
                if str(attrs.get('type', '')).upper() == 'START'
                or bool(attrs.get('is_start', False))
            ]
            goal_nodes = [
                node_id
                for node_id, attrs in mission_graph.nodes(data=True)
                if str(attrs.get('type', '')).upper() in {'GOAL', 'TRIFORCE'}
                or bool(attrs.get('is_goal', False))
            ]
            if start_nodes and goal_nodes:
                start = min(start_nodes, key=str)
                goal = min(goal_nodes, key=str)
                try:
                    path = nx.shortest_path(mission_graph, start, goal)
                except nx.NetworkXNoPath:
                    path = nx.shortest_path(mission_graph.to_undirected(), start, goal)
                ordered = [node_id for node_id in path if node_id in available]
                if ordered:
                    remaining = sorted(available.difference(ordered), key=str)
                    return ordered + remaining
        except (ImportError, AttributeError, TypeError, ValueError):
            pass
        return sorted(available, key=str)

    @staticmethod
    def _room_semantic_tension(room_grid: np.ndarray) -> float:
        """Compute a transparent room-local challenge proxy from final tile IDs."""
        from src.core.definitions import SEMANTIC_PALETTE

        grid = np.asarray(room_grid, dtype=np.int32)
        floor = max(1, int(np.sum(grid == int(SEMANTIC_PALETTE['FLOOR']))))
        weighted_tiles = (
            ("ENEMY", 1.0),
            ("BOSS", 3.0),
            ("ELEMENT", 1.5),
            ("DOOR_LOCKED", 0.6),
            ("DOOR_BOMB", 0.6),
            ("DOOR_BOSS", 0.8),
            ("DOOR_PUZZLE", 0.6),
            ("PUZZLE", 0.5),
            ("BLOCK", 0.25),
        )
        load = 0.0
        for tile_name, weight in weighted_tiles:
            tile_id = SEMANTIC_PALETTE.get(tile_name)
            if tile_id is not None:
                load += float(weight) * float(np.sum(grid == int(tile_id)))
        return float(load / floor)
    
    def _extract_actual_tension_curve(self, dungeon: Dict) -> np.ndarray:
        """
        Extract an observed semantic-tension curve from generated room grids.
        
        This is a transparent structural proxy, not a human-experience metric.
        It intentionally never reads the requested tension target.
        """
        rooms_by_id = {
            room.get('id'): room
            for room in list(dungeon.get('rooms', []) or [])
            if isinstance(room, dict) and 'id' in room
        }
        order = list(dungeon.get('room_order', []) or [])
        if str(dungeon.get('actual_curve_source', '')) == 'surrogate_smoke':
            if not self.allow_surrogate_smoke:
                raise RuntimeError('Surrogate curve is not allowed in checkpoint-backed evidence mode')
            values = [
                float(rooms_by_id[room_id]['synthetic_tension'])
                for room_id in order
                if room_id in rooms_by_id and 'synthetic_tension' in rooms_by_id[room_id]
            ]
            return np.asarray(values, dtype=np.float64)

        values = [
            self._room_semantic_tension(np.asarray(rooms_by_id[room_id]['grid']))
            for room_id in order
            if room_id in rooms_by_id and 'grid' in rooms_by_id[room_id]
        ]
        if not values:
            raise RuntimeError('No generated room grids available for observed semantic tension')
        curve = np.asarray(values, dtype=np.float64)
        maximum = float(curve.max(initial=0.0))
        return curve / maximum if maximum > 0.0 else curve


def main():
    """Main entry point for controllability test."""
    parser = argparse.ArgumentParser(description="Test controllability of H-MOLQD pipeline")
    parser.add_argument('--num-samples', type=int, default=20, help='Number of dungeons per curve type')
    parser.add_argument('--output', type=str, default='results/controllability', help='Output directory')
    parser.add_argument('--curve', type=str, default=None, help='Test specific curve (flat, linear_rising, etc.)')
    parser.add_argument('--vqvae-checkpoint', type=str, default=None, help='Trained VQ-VAE checkpoint required for evidence mode.')
    parser.add_argument('--diffusion-checkpoint', type=str, default=None, help='Trained diffusion checkpoint required for evidence mode.')
    parser.add_argument('--logic-net-checkpoint', type=str, default=None, help='Optional trained LogicNet checkpoint.')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--guidance-scale', type=float, default=3.0)
    parser.add_argument('--logic-guidance-scale', type=float, default=0.0)
    parser.add_argument('--num-diffusion-steps', type=int, default=25)
    parser.add_argument('--responsive-correlation-threshold', type=float, default=0.7)
    parser.add_argument(
        '--allow-surrogate-smoke',
        action='store_true',
        help='Permit deterministic surrogate output for schema smoke tests only; output is labeled non-evidence.',
    )
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    if not bool(args.allow_surrogate_smoke):
        missing = [
            flag
            for flag, value in (
                ("--vqvae-checkpoint", args.vqvae_checkpoint),
                ("--diffusion-checkpoint", args.diffusion_checkpoint),
            )
            if not value or not Path(value).is_file()
        ]
        if missing:
            parser.error(
                "Scientific controllability execution requires readable trained checkpoints: "
                + ", ".join(missing)
                + ". Use --allow-surrogate-smoke only for non-evidence schema smoke tests."
            )
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # Create test
    test = ControllabilityTest(
        output_dir=args.output,
        vqvae_checkpoint=args.vqvae_checkpoint,
        diffusion_checkpoint=args.diffusion_checkpoint,
        logic_net_checkpoint=args.logic_net_checkpoint,
        device=args.device,
        guidance_scale=args.guidance_scale,
        logic_guidance_scale=args.logic_guidance_scale,
        num_diffusion_steps=args.num_diffusion_steps,
        responsive_correlation_threshold=args.responsive_correlation_threshold,
        allow_surrogate_smoke=bool(args.allow_surrogate_smoke),
    )
    
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
