"""
Generation and Evaluation Pipeline for KLTN PCG
===============================================

Generate dungeon maps and evaluate them using:
1. LogicNet for quick solvability approximation
2. External validator for ground-truth verification
3. WFC/Symbolic repair for fixing invalid maps

Usage:
    python -m src.generate --checkpoint checkpoints/best_model.pth --num-samples 100
    
    # Quick test
    python -m src.generate --quick
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.logic_net import LogicNet
from src.utils.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


# =============================================================================
# IMPORT VALIDATOR (with fallback)
# =============================================================================

from src.simulation import (
    ZeldaLogicEnv,
    StateSpaceAStar,
    SEMANTIC_PALETTE,
)
VALIDATOR_AVAILABLE = True
logger.info("External validator available")


# =============================================================================
# DUNGEON VALIDATOR
# =============================================================================

class DungeonValidator:
    """
    Validates dungeon solvability using multiple methods.
    
    Methods:
    1. LogicNet: Fast differentiable approximation
    2. A* Search: Ground-truth verification (if available)
    
    Args:
        use_external: Use external A* validator if available
        logic_iterations: Number of LogicNet iterations
    """
    
    def __init__(
        self,
        use_external: bool = True,
        logic_iterations: int = 30,
    ):
        self.use_external = use_external and VALIDATOR_AVAILABLE
        self.logic_net = LogicNet(num_iterations=logic_iterations)
    
    def check_solvability(
        self,
        dungeon_map: torch.Tensor,
        start: Optional[Tuple[int, int]] = None,
        goal: Optional[Tuple[int, int]] = None,
        use_ground_truth: bool = False,
    ) -> bool:
        """
        Check if dungeon is solvable.
        
        Args:
            dungeon_map: (1, H, W) or (H, W) tensor
            start: Start position (row, col)
            goal: Goal position (row, col)
            use_ground_truth: Use A* instead of LogicNet
            
        Returns:
            True if solvable
        """
        # Convert to numpy for external validator
        if isinstance(dungeon_map, torch.Tensor):
            grid = dungeon_map.detach().cpu().squeeze().numpy()
        else:
            grid = dungeon_map
        
        H, W = grid.shape
        
        # Find start/goal if not provided
        if start is None:
            start = self._find_tile(grid, 'START', default=(2, 2))
        if goal is None:
            goal = self._find_tile(grid, 'TRIFORCE', default=(H-3, W-3))
        
        # Use external validator if requested and available
        if use_ground_truth and self.use_external:
            return self._check_with_astar(grid, start, goal)
        
        # Use LogicNet for fast approximation
        return self._check_with_logic_net(dungeon_map, start, goal)
    
    def _check_with_logic_net(
        self,
        dungeon_map: torch.Tensor,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        threshold: float = 0.5,
    ) -> bool:
        """Check solvability using LogicNet."""
        # Ensure tensor is on CPU for LogicNet
        if dungeon_map.is_cuda:
            dungeon_map = dungeon_map.cpu()
        
        if dungeon_map.dim() == 2:
            dungeon_map = dungeon_map.unsqueeze(0).unsqueeze(0)
        elif dungeon_map.dim() == 3:
            dungeon_map = dungeon_map.unsqueeze(0)
        
        # Convert to walkability (simple threshold)
        walkability = (dungeon_map > 0).float()
        
        with torch.no_grad():
            score = self.logic_net(walkability, [start], [goal])
        
        return score.item() > threshold
    
    def _check_with_astar(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> bool:
        """Check solvability using A* search."""
        if not VALIDATOR_AVAILABLE:
            return self._check_with_logic_net(
                torch.tensor(grid).float(), start, goal
            )
        
        try:
            # Create environment
            env = ZeldaLogicEnv(grid.astype(np.int32))
            solver = StateSpaceAStar(env, timeout=5000)
            success, _, _ = solver.solve()
            return success
        except Exception as e:
            logger.warning(f"A* validation failed: {e}")
            return False
    
    def _find_tile(
        self,
        grid: np.ndarray,
        tile_name: str,
        default: Tuple[int, int],
    ) -> Tuple[int, int]:
        """Find tile position in grid."""
        if SEMANTIC_PALETTE is not None:
            tile_id = SEMANTIC_PALETTE.get(tile_name, -1)
            positions = np.where(grid == tile_id)
            if len(positions[0]) > 0:
                return (int(positions[0][0]), int(positions[1][0]))
        return default


# =============================================================================
# WFC REPAIR (Simplified)
# =============================================================================

class WFCRepair:
    """
    Wave Function Collapse based repair for invalid dungeons.
    
    Fixes common solvability issues:
    1. Disconnected regions
    2. Missing path from start to goal
    3. Key/door balance issues
    
    This is a simplified version - full WFC would use proper
    constraint propagation.
    """
    
    def __init__(self, max_iterations: int = 100):
        self.max_iterations = max_iterations
    
    def repair(
        self,
        dungeon_map: torch.Tensor,
        start: Optional[Tuple[int, int]] = None,
        goal: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Attempt to repair an invalid dungeon.
        
        Args:
            dungeon_map: (1, H, W) or (H, W) tensor
            start: Start position
            goal: Goal position
            
        Returns:
            Repaired dungeon map
        """
        # Convert to numpy for manipulation
        if isinstance(dungeon_map, torch.Tensor):
            grid = dungeon_map.squeeze().numpy().copy()
        else:
            grid = np.array(dungeon_map, copy=True)
        
        H, W = grid.shape
        
        # Default positions
        if start is None:
            start = (2, 2)
        if goal is None:
            goal = (H - 3, W - 3)
        
        # Repair strategies
        grid = self._ensure_border_walls(grid)
        grid = self._carve_path(grid, start, goal)
        grid = self._ensure_start_goal(grid, start, goal)
        
        return torch.tensor(grid, dtype=torch.float32).unsqueeze(0)
    
    def _ensure_border_walls(self, grid: np.ndarray) -> np.ndarray:
        """Ensure grid has wall borders."""
        grid[0, :] = 0  # Top
        grid[-1, :] = 0  # Bottom
        grid[:, 0] = 0  # Left
        grid[:, -1] = 0  # Right
        return grid
    
    def _carve_path(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> np.ndarray:
        """Carve a simple path from start to goal."""
        sr, sc = start
        gr, gc = goal
        
        # Simple L-shaped path
        r, c = sr, sc
        
        # Move vertically first
        while r != gr:
            grid[r, c] = max(grid[r, c], 0.5)  # Make walkable
            r += 1 if gr > r else -1
        
        # Then horizontally
        while c != gc:
            grid[r, c] = max(grid[r, c], 0.5)
            c += 1 if gc > c else -1
        
        # Mark goal
        grid[gr, gc] = max(grid[gr, gc], 0.5)
        
        return grid
    
    def _ensure_start_goal(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> np.ndarray:
        """Ensure start and goal are walkable."""
        sr, sc = start
        gr, gc = goal
        
        grid[sr, sc] = 1.0
        grid[gr, gc] = 1.0
        
        return grid


# =============================================================================
# GENERATION PIPELINE
# =============================================================================

def generate_and_evaluate(
    model: nn.Module,
    num_samples: int = 100,
    device: Optional[torch.device] = None,
    use_repair: bool = True,
    use_ground_truth: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Generate dungeons and evaluate solvability.
    
    Args:
        model: Generator model
        num_samples: Number of samples to generate
        device: Device for generation
        use_repair: Apply WFC repair to invalid maps
        use_ground_truth: Use A* for validation (slower but accurate)
        verbose: Print per-sample results
        
    Returns:
        Dictionary with results:
        - valid_maps: List of valid dungeon tensors
        - success_rate: Fraction of valid dungeons
        - repaired_count: Number of maps that needed repair
        - metrics: Additional metrics
    """
    model.eval()
    
    if device is None:
        device = next(model.parameters()).device
    
    validator = DungeonValidator()
    wfc_repair = WFCRepair() if use_repair else None
    
    valid_maps = []
    repaired_count = 0
    initial_valid = 0
    
    logger.info(f"Generating {num_samples} dungeon samples...")
    
    with torch.no_grad():
        for i in range(num_samples):
            # Generate
            coarse_map = model.sample(num_samples=1, device=device)
            
            # Check initial solvability
            is_solvable = validator.check_solvability(
                coarse_map,
                use_ground_truth=use_ground_truth,
            )
            
            if is_solvable:
                initial_valid += 1
                valid_maps.append(coarse_map.cpu())
                if verbose:
                    logger.info(f"Sample {i+1}: Valid")
            elif use_repair and wfc_repair is not None:
                # Attempt repair
                repaired_map = wfc_repair.repair(coarse_map)
                
                if validator.check_solvability(
                    repaired_map,
                    use_ground_truth=use_ground_truth,
                ):
                    valid_maps.append(repaired_map.cpu())
                    repaired_count += 1
                    if verbose:
                        logger.info(f"Sample {i+1}: Repaired")
                else:
                    if verbose:
                        logger.info(f"Sample {i+1}: Failed (repair unsuccessful)")
            else:
                if verbose:
                    logger.info(f"Sample {i+1}: Invalid")
    
    # Calculate metrics
    total_valid = len(valid_maps)
    success_rate = total_valid / num_samples
    
    results = {
        'valid_maps': valid_maps,
        'success_rate': success_rate,
        'total_valid': total_valid,
        'initial_valid': initial_valid,
        'repaired_count': repaired_count,
        'num_samples': num_samples,
        'metrics': {
            'initial_success_rate': initial_valid / num_samples,
            'repair_success_rate': repaired_count / max(num_samples - initial_valid, 1),
        }
    }
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Generation Results:")
    logger.info(f"  Total Samples: {num_samples}")
    logger.info(f"  Initially Valid: {initial_valid} ({100*initial_valid/num_samples:.1f}%)")
    logger.info(f"  Repaired: {repaired_count}")
    logger.info(f"  Final Success Rate: {total_valid}/{num_samples} ({100*success_rate:.1f}%)")
    logger.info(f"{'='*50}")
    
    return results


def save_generated_maps(
    maps: List[torch.Tensor],
    output_dir: str = "./generated_dungeons",
    format: str = "npy",
) -> List[Path]:
    """
    Save generated maps to files.
    
    Args:
        maps: List of dungeon tensors
        output_dir: Output directory
        format: 'npy' or 'txt'
        
    Returns:
        List of saved file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    for i, map_tensor in enumerate(maps):
        grid = map_tensor.squeeze().numpy()
        
        if format == 'npy':
            filepath = output_path / f"dungeon_{i:04d}.npy"
            np.save(filepath, grid)
        else:
            filepath = output_path / f"dungeon_{i:04d}.txt"
            with open(filepath, 'w') as f:
                for row in grid:
                    line = ''.join(['F' if v > 0.5 else 'W' for v in row])
                    f.write(line + '\n')
        
        saved_files.append(filepath)
    
    logger.info(f"Saved {len(saved_files)} maps to {output_dir}")
    return saved_files


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate and Evaluate KLTN PCG Dungeons',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to model checkpoint (if None, uses random model)'
    )
    parser.add_argument(
        '--num-samples', type=int, default=100,
        help='Number of samples to generate'
    )
    parser.add_argument(
        '--output-dir', type=str, default='./generated_dungeons',
        help='Directory to save generated maps'
    )
    parser.add_argument(
        '--use-repair', action='store_true', default=True,
        help='Apply WFC repair to invalid maps'
    )
    parser.add_argument(
        '--no-repair', action='store_true',
        help='Disable WFC repair'
    )
    parser.add_argument(
        '--ground-truth', action='store_true',
        help='Use A* for ground-truth validation (slower)'
    )
    parser.add_argument(
        '--save', action='store_true',
        help='Save valid maps to files'
    )
    parser.add_argument(
        '--format', type=str, default='npy',
        choices=['npy', 'txt'],
        help='Output format for saved maps'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick test (10 samples)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S',
    )
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Load or create model
    # Import here to avoid circular imports
    from src.train import SimpleDungeonGenerator
    
    model = SimpleDungeonGenerator(latent_dim=64).to(device)
    
    if args.checkpoint and Path(args.checkpoint).exists():
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from {args.checkpoint}")
    else:
        logger.warning("No checkpoint provided, using random initialized model")
    
    # Generate
    num_samples = 10 if args.quick else args.num_samples
    use_repair = not args.no_repair and args.use_repair
    
    results = generate_and_evaluate(
        model,
        num_samples=num_samples,
        device=device,
        use_repair=use_repair,
        use_ground_truth=args.ground_truth,
        verbose=args.verbose,
    )
    
    # Save if requested
    if args.save and results['valid_maps']:
        save_generated_maps(
            results['valid_maps'],
            output_dir=args.output_dir,
            format=args.format,
        )
    
    return results


if __name__ == '__main__':
    main()
