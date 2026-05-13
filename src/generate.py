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

import sys
import argparse
import logging
import random
from collections import deque
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Use Block V LogicNet (not legacy ml.logic_net)
from src.core.logic_net import LogicNet
from src.config_system import load_resolved_config_for_artifact
from src.core.symbolic_refiner import create_symbolic_refiner
from src.core.definitions import (
    SEMANTIC_PALETTE as CORE_SEMANTIC_PALETTE,
    SEMANTIC_TO_CHAR,
    semantic_grid_to_vglc_lines,
)
from src.gui.ai.generation_pipeline import (
    generate_dungeon_with_pipeline,
    generate_mission_graph,
    load_canonical_generation_pipeline,
)
from src.pipeline.dungeon_pipeline import NeuralSymbolicDungeonPipeline, pipeline_kwargs_from_resolved_config

logger = logging.getLogger(__name__)


# =============================================================================
# IMPORT VALIDATOR (with fallback)
# =============================================================================

try:
    from src.simulation import (
        ZeldaLogicEnv,
        StateSpaceAStar,
        SEMANTIC_PALETTE,
        WALKABLE_IDS,
    )
    VALIDATOR_AVAILABLE = True
    logger.info("External validator available")
except ImportError:
    VALIDATOR_AVAILABLE = False
    SEMANTIC_PALETTE = CORE_SEMANTIC_PALETTE
    WALKABLE_IDS = {
        SEMANTIC_PALETTE["FLOOR"],
        SEMANTIC_PALETTE["DOOR_OPEN"],
        SEMANTIC_PALETTE["DOOR_SOFT"],
        SEMANTIC_PALETTE["START"],
        SEMANTIC_PALETTE["TRIFORCE"],
        SEMANTIC_PALETTE["KEY_SMALL"],
        SEMANTIC_PALETTE["KEY_BOSS"],
        SEMANTIC_PALETTE["KEY_ITEM"],
        SEMANTIC_PALETTE["ITEM_MINOR"],
        SEMANTIC_PALETTE["ELEMENT_FLOOR"],
        SEMANTIC_PALETTE["STAIR"],
        SEMANTIC_PALETTE["ENEMY"],
        SEMANTIC_PALETTE["BOSS"],
        SEMANTIC_PALETTE["PUZZLE"],
    }
    logger.warning("External validator not available, using LogicNet approximation only")


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
        # Use Block V LogicNet for differentiable solvability
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

        # Canonical generation returns stitched semantic ID grids, not VQ latents.
        # Route those through a direct grid-space reachability check instead of
        # feeding them into the room-level LogicNet latent head.
        if self._is_semantic_grid_input(dungeon_map):
            return self._check_with_grid_bfs(grid, start, goal)

        # Use LogicNet only for latent / tile-probability tensors.
        return self._check_with_logic_net(dungeon_map, start, goal)

    def _is_semantic_grid_input(self, dungeon_map: Any) -> bool:
        """Return True when the input looks like a stitched semantic grid."""
        if isinstance(dungeon_map, torch.Tensor):
            if dungeon_map.dim() == 2:
                return True
            if dungeon_map.dim() == 3:
                return int(dungeon_map.shape[0]) == 1
            if dungeon_map.dim() == 4:
                return int(dungeon_map.shape[1]) == 1
            return False

        array = np.asarray(dungeon_map)
        return array.ndim == 2
    
    def _check_with_logic_net(
        self,
        dungeon_map: torch.Tensor,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        threshold: float = 0.5,
    ) -> bool:
        """Check solvability using Block V LogicNet."""
        # Accept tensor-like inputs for robustness in caller code paths.
        if not isinstance(dungeon_map, torch.Tensor):
            dungeon_map = torch.as_tensor(dungeon_map, dtype=torch.float32)

        if dungeon_map.is_cuda:
            dungeon_map = dungeon_map.cpu()

        if dungeon_map.dim() == 2:
            dungeon_map = dungeon_map.unsqueeze(0).unsqueeze(0)
        elif dungeon_map.dim() == 3:
            dungeon_map = dungeon_map.unsqueeze(0)
        elif dungeon_map.dim() != 4:
            raise ValueError(
                f"Expected dungeon_map to have 2, 3, or 4 dimensions, got {dungeon_map.dim()}"
            )

        dungeon_map = dungeon_map.float()
        
        with torch.no_grad():
            # Block V LogicNet: forward(z, graph_data) -> (loss, info)
            loss, _ = self.logic_net(dungeon_map)
            # Lower loss = more solvable
            solvability = 1.0 - loss.item()
        
        return solvability > threshold
    
    def _check_with_astar(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> bool:
        """Check solvability using A* search."""
        if not VALIDATOR_AVAILABLE:
            return self._check_with_grid_bfs(grid, start, goal)
        
        try:
            # Create environment
            env = ZeldaLogicEnv(grid.astype(np.int32))
            solver = StateSpaceAStar(env, timeout=5000)
            success, _, _ = solver.solve()
            return success
        except (AttributeError, RuntimeError, ValueError, TypeError) as e:
            logger.warning(f"A* validation failed: {e}")
            return False

    def _check_with_grid_bfs(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> bool:
        """Fast semantic-grid approximation for stitched dungeon validation."""
        grid = np.rint(np.asarray(grid)).astype(np.int64, copy=False)
        if grid.ndim != 2:
            raise ValueError(f"Expected 2D grid for BFS validation, got shape={tuple(grid.shape)}.")

        height, width = grid.shape
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
        if not (0 <= start[0] < height and 0 <= start[1] < width):
            return False
        if not (0 <= goal[0] < height and 0 <= goal[1] < width):
            return False

        walkable_ids = {int(v) for v in WALKABLE_IDS}
        if int(grid[start]) not in walkable_ids:
            walkable_ids.add(int(grid[start]))
        if int(grid[goal]) not in walkable_ids:
            walkable_ids.add(int(grid[goal]))

        queue = deque([start])
        visited = {start}

        while queue:
            row, col = queue.popleft()
            if (row, col) == goal:
                return True
            for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                next_row = row + d_row
                next_col = col + d_col
                if not (0 <= next_row < height and 0 <= next_col < width):
                    continue
                next_pos = (next_row, next_col)
                if next_pos in visited:
                    continue
                if int(grid[next_row, next_col]) not in walkable_ids:
                    continue
                visited.add(next_pos)
                queue.append(next_pos)

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
# WFC REPAIR (Uses Block VI SymbolicRefiner)
# =============================================================================

class WFCRepair:
    """
    Wrapper around Block VI SymbolicRefiner for dungeon repair.
    
    Uses full WFC with constraint propagation, learned tile stats,
    and A*-based path analysis.
    """
    
    def __init__(self, max_iterations: int = 100):
        self.refiner = create_symbolic_refiner(
            max_repair_attempts=5,
        )
    
    def repair(
        self,
        dungeon_map: torch.Tensor,
        start: Optional[Tuple[int, int]] = None,
        goal: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Attempt to repair an invalid dungeon using Block VI SymbolicRefiner.
        """
        if isinstance(dungeon_map, torch.Tensor):
            grid = dungeon_map.cpu().squeeze().numpy().copy().astype(int)
        else:
            grid = np.array(dungeon_map, copy=True).astype(int)
        
        H, W = grid.shape
        if start is None:
            start = (2, 2)
        if goal is None:
            goal = (H - 3, W - 3)
        
        repaired_grid, _success = self.refiner.repair_room(grid, start, goal)
        
        return torch.tensor(repaired_grid, dtype=torch.float32).unsqueeze(0)


# =============================================================================
# GENERATION PIPELINE
# =============================================================================


class CanonicalDungeonGenerator(nn.Module):
    """
    Wrap the canonical room-wise neural-symbolic pipeline in a sample() interface.

    This keeps the legacy CLI/evaluation helpers working while ensuring generation
    actually follows the documented mission-graph -> per-room -> symbolic-repair
    stack instead of constructing a separate hardcoded inference model.
    """

    def __init__(
        self,
        pipeline: NeuralSymbolicDungeonPipeline,
        *,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.pipeline = pipeline
        self.base_seed = None if seed is None else int(seed)
        self.samples_generated = 0
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)

    def sample(self, num_samples: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
        if device is None:
            device = self._device_anchor.device

        rooms: List[torch.Tensor] = []
        for _ in range(max(1, int(num_samples))):
            sample_seed = (
                None
                if self.base_seed is None
                else int(self.base_seed + self.samples_generated)
            )
            mission_data = generate_mission_graph(random, seed=sample_seed)
            result = generate_dungeon_with_pipeline(
                self.pipeline,
                mission_data["mission_graph"],
                seed=sample_seed,
                logger=logger,
            )
            grid = torch.as_tensor(result.dungeon_grid, dtype=torch.float32, device=device)
            if grid.dim() != 2:
                raise ValueError(
                    "Canonical pipeline must return a 2D semantic grid for CLI generation, "
                    f"got shape={tuple(grid.shape)}."
                )
            rooms.append(grid.unsqueeze(0).unsqueeze(0))
            self.samples_generated += 1

        return torch.cat(rooms, dim=0)


def load_generation_pipeline(
    checkpoint_path: Optional[str],
    *,
    device: torch.device,
    strict_checkpoint_mode: bool = False,
) -> NeuralSymbolicDungeonPipeline:
    """Load the canonical inference stack, or create a random-init canonical stack."""
    if checkpoint_path:
        checkpoint = Path(checkpoint_path)
        if checkpoint.exists():
            return load_canonical_generation_pipeline(
                checkpoint_path=checkpoint,
                device=device,
                logger=logger,
                strict_checkpoint_mode=bool(strict_checkpoint_mode),
            )
        if strict_checkpoint_mode:
            raise FileNotFoundError(f"Generation checkpoint not found at {checkpoint_path!r}.")
        logger.warning(
            "Checkpoint %s not found; falling back to randomly initialized canonical pipeline.",
            checkpoint,
        )
    else:
        logger.warning("No checkpoint provided; using randomly initialized canonical pipeline.")

    resolved_config = load_resolved_config_for_artifact(checkpoint_path) if checkpoint_path else None
    pipeline_kwargs = (
        pipeline_kwargs_from_resolved_config(resolved_config)
        if isinstance(resolved_config, dict)
        else {}
    )
    return NeuralSymbolicDungeonPipeline(
        device=str(device),
        enable_logging=False,
        strict_checkpoint_mode=bool(strict_checkpoint_mode),
        **pipeline_kwargs,
    )


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
        try:
            device = next(model.parameters()).device
        except StopIteration:
            first_buffer = next(model.buffers(), None)
            device = first_buffer.device if first_buffer is not None else torch.device("cpu")
    
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
    logger.info("Generation Results:")
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
        grid = map_tensor.detach().cpu().squeeze().numpy()

        if format == 'npy':
            filepath = output_path / f"dungeon_{i:04d}.npy"
            np.save(filepath, grid)
        else:
            filepath = output_path / f"dungeon_{i:04d}.txt"
            if grid.ndim != 2:
                raise ValueError(
                    f"TXT export expects 2D semantic grid, got shape={tuple(grid.shape)}"
                )
            grid_int = np.rint(grid).astype(np.int32, copy=False)
            if not np.allclose(grid, grid_int, atol=1e-4):
                raise ValueError(
                    "TXT export expects semantic ID grids with values close to integers; "
                    f"got non-integer data range [{float(np.min(grid)):.4f}, {float(np.max(grid)):.4f}]."
                )
            unknown_ids = sorted(
                int(v) for v in np.unique(grid_int) if int(v) not in SEMANTIC_TO_CHAR
            )
            if unknown_ids:
                logger.warning(
                    "TXT export encountered unknown semantic IDs %s; writing as '-'",
                    unknown_ids,
                )
            lines = semantic_grid_to_vglc_lines(grid_int)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
                f.write('\n')
        
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
        '--seed', type=int, default=None,
        help='Base seed for reproducible mission-graph and room generation.'
    )
    parser.add_argument(
        '--strict-checkpoint-mode', action='store_true',
        help='Fail instead of falling back when checkpoint metadata or files are missing.'
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
    
    pipeline = load_generation_pipeline(
        args.checkpoint,
        device=device,
        strict_checkpoint_mode=bool(args.strict_checkpoint_mode),
    )
    model = CanonicalDungeonGenerator(pipeline, seed=args.seed).to(device)
    
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
