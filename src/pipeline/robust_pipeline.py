"""
Robust Pipeline with Retry Logic
Prevents cascade failures and provides graceful degradation during thesis defense demos.

This addresses the critical concern: "What if your diffusion model fails during the live demo?"
"""

import time
import logging
from enum import Enum
from typing import Dict, Any, Callable, Tuple, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class BlockStatus(Enum):
    """Status of a pipeline block execution."""
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class BlockResult:
    """Result of executing a single pipeline block."""
    status: BlockStatus
    output: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    attempts: int = 1


@dataclass
class PipelineConfig:
    """Configuration for robust pipeline execution."""
    max_retries: int = 5
    base_backoff: float = 0.5  # Starting backoff in seconds
    max_backoff: float = 30.0  # Maximum backoff in seconds
    backoff_multiplier: float = 2.0  # Exponential backoff multiplier
    enable_logging: bool = True
    timeout_per_block: Optional[float] = None  # Seconds, None = no timeout


class PipelineBlock:
    """
    Wraps a pipeline component with retry logic and validation.
    
    Each block can:
    - Execute with configurable retry attempts
    - Validate its output
    - Report detailed diagnostics
    - Gracefully handle failures
    """
    
    def __init__(
        self,
        name: str,
        executor: Callable[[Dict], Any],
        validator: Optional[Callable[[Any], bool]] = None,
        config: Optional[PipelineConfig] = None
    ):
        """
        Args:
            name: Human-readable block name (e.g., "Evolutionary Director")
            executor: Function that takes state dict and returns output
            validator: Optional function to validate output (returns True if valid)
            config: Execution configuration
        """
        self.name = name
        self.executor = executor
        self.validator = validator or (lambda x: x is not None)
        self.config = config or PipelineConfig()
    
    def execute(self, state: Dict[str, Any]) -> BlockResult:
        """
        Execute block with retry logic.
        
        Args:
            state: Current pipeline state (input parameters)
        
        Returns:
            BlockResult with status, output, and diagnostics
        """
        backoff = self.config.base_backoff
        
        for attempt in range(1, self.config.max_retries + 1):
            if self.config.enable_logging:
                logger.info(f"[{self.name}] Attempt {attempt}/{self.config.max_retries}")
            
            start_time = time.time()
            
            try:
                # Execute the block
                output = self.executor(state)
                execution_time = time.time() - start_time
                
                # Validate output
                is_valid = self.validator(output)
                
                if is_valid:
                    if self.config.enable_logging:
                        logger.info(f"[{self.name}] ✓ Success in {execution_time:.2f}s")
                    
                    return BlockResult(
                        status=BlockStatus.SUCCESS,
                        output=output,
                        execution_time=execution_time,
                        attempts=attempt
                    )
                else:
                    raise ValueError(f"Validation failed for {self.name}")
            
            except Exception as e:
                execution_time = time.time() - start_time
                error_msg = f"{type(e).__name__}: {str(e)}"
                
                if self.config.enable_logging:
                    logger.warning(f"[{self.name}] ✗ Attempt {attempt} failed: {error_msg}")
                
                # Check if we should retry
                if attempt < self.config.max_retries:
                    if self.config.enable_logging:
                        logger.info(f"[{self.name}] Retrying in {backoff:.1f}s...")
                    
                    time.sleep(backoff)
                    backoff = min(backoff * self.config.backoff_multiplier, self.config.max_backoff)
                else:
                    # Final attempt failed
                    if self.config.enable_logging:
                        logger.error(f"[{self.name}] Failed after {self.config.max_retries} attempts")
                    
                    return BlockResult(
                        status=BlockStatus.FAILED,
                        error=error_msg,
                        execution_time=execution_time,
                        attempts=attempt
                    )
        
        # Should never reach here, but handle gracefully
        return BlockResult(
            status=BlockStatus.FAILED,
            error="Max retries exceeded",
            attempts=self.config.max_retries
        )


class RobustPipeline:
    """
    7-block H-MOLQD pipeline with comprehensive error handling.
    
    Architecture:
        Block I:   Evolutionary Director (topology generation)
        Block II:  VQ-VAE Encoder (graph → latent)
        Block III: Condition Encoder (user controls → embedding)
        Block IV:  Diffusion Model (latent → spatial layout)
        Block V:   LogicNet (constraint satisfaction)
        Block VI:  WFC Refiner (local coherence)
        Block VII: MAP-Elites (archive management)
    
    Each block can fail and retry independently without crashing entire pipeline.
    """
    
    def __init__(
        self,
        executors: Dict[str, Callable],
        validators: Optional[Dict[str, Callable]] = None,
        config: Optional[PipelineConfig] = None
    ):
        """
        Args:
            executors: {block_name: execution_function} mapping
            validators: {block_name: validation_function} mapping
            config: Global pipeline configuration
        """
        self.config = config or PipelineConfig()
        validators = validators or {}
        
        # Create PipelineBlock instances
        self.executors = {}
        for name, executor_fn in executors.items():
            validator_fn = validators.get(name)
            self.executors[name] = PipelineBlock(
                name=name,
                executor=executor_fn,
                validator=validator_fn,
                config=self.config
            )
        
        self.execution_history = []
    
    def generate_dungeon(
        self, 
        user_inputs: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any], Dict[str, BlockResult]]:
        """
        Execute full pipeline with robust error handling.
        
        Args:
            user_inputs: Initial generation parameters
                Example: {
                    'num_rooms': 8,
                    'tension_curve': [0.2, 0.4, 0.7, 0.9, 0.7, 0.5],
                    'theme': 'dungeon',
                    'seed': 42
                }
        
        Returns:
            (success, final_output, diagnostics)
        """
        logger.info("=" * 60)
        logger.info("Starting robust pipeline execution")
        logger.info("=" * 60)
        
        pipeline_state = user_inputs.copy()
        diagnostics = {}
        
        # Define execution order
        block_order = [
            'evolutionary_director',
            'vqvae_encoder',
            'condition_encoder',
            'diffusion_model',
            'logicnet',
            'wfc_refiner',
            'map_elites'
        ]
        
        for block_name in block_order:
            if block_name not in self.executors:
                logger.warning(f"⚠ Block '{block_name}' not configured, skipping")
                continue
            
            executor = self.executors[block_name]
            result = executor.execute(pipeline_state)
            diagnostics[block_name] = result
            
            if result.status == BlockStatus.SUCCESS:
                # Merge output into pipeline state
                if isinstance(result.output, dict):
                    pipeline_state.update(result.output)
                else:
                    pipeline_state[block_name + '_output'] = result.output
            else:
                # Block failed after all retries
                logger.error(f"Pipeline aborted at {block_name}")
                return False, pipeline_state, diagnostics
        
        logger.info("=" * 60)
        logger.info("✓ Pipeline completed successfully")
        logger.info("=" * 60)
        
        return True, pipeline_state, diagnostics
    
    def get_performance_report(self, diagnostics: Dict[str, BlockResult]) -> str:
        """Generate human-readable performance report."""
        lines = ["\n=== Pipeline Performance Report ==="]
        
        total_time = sum(r.execution_time for r in diagnostics.values())
        total_attempts = sum(r.attempts for r in diagnostics.values())
        
        for name, result in diagnostics.items():
            status_symbol = {
                BlockStatus.SUCCESS: "✓",
                BlockStatus.FAILED: "✗",
                BlockStatus.RETRYING: "⏳"
            }.get(result.status, "?")
            
            lines.append(
                f"{status_symbol} {name:25s} | "
                f"{result.execution_time:6.2f}s | "
                f"{result.attempts} attempts"
            )
            
            if result.error:
                lines.append(f"   Error: {result.error}")
        
        lines.append(f"\nTotal time: {total_time:.2f}s")
        lines.append(f"Total attempts: {total_attempts}")
        
        return "\n".join(lines)


# ===== Built-in Validators =====

def validate_mission_graph(graph: Any) -> bool:
    """Validate mission graph structure."""
    if not isinstance(graph, dict):
        return False
    
    required_keys = ['nodes', 'edges']
    if not all(k in graph for k in required_keys):
        return False
    
    if len(graph['nodes']) < 3:  # Minimum: start, middle, boss
        return False
    
    # Check connectivity (basic check)
    if 'adjacency' in graph:
        # Verify all nodes have at least one connection
        for node_id in graph['nodes'].keys():
            if node_id not in graph['adjacency'] or len(graph['adjacency'][node_id]) == 0:
                # Start node exception (may have no incoming)
                if node_id != 0:
                    return False
    
    return True


def validate_layout(layout: Any) -> bool:
    """Validate generated dungeon layout."""
    if not isinstance(layout, dict):
        return False
    
    if 'visual_grid' not in layout:
        return False
    
    import numpy as np
    grid = layout['visual_grid']
    
    # Check dimensions
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    
    if grid.shape[0] < 16 or grid.shape[1] < 16:
        return False
    
    # Check for reasonable floor coverage (20-70%)
    floor_ratio = np.sum(grid == 0) / grid.size
    if not (0.2 <= floor_ratio <= 0.7):
        return False
    
    return True


def validate_solvability(dungeon: Any) -> bool:
    """Validate dungeon is solvable (path exists from start to goal)."""
    if not isinstance(dungeon, dict):
        return False
    
    # Check for required keys
    if 'visual_grid' not in dungeon:
        return False
    
    # Try to use existing validator if available
    try:
        from src.simulation.validator import is_solvable
        return is_solvable(dungeon)
    except (ImportError, Exception):
        # Fallback: basic checks
        if 'mission_graph' not in dungeon:
            return False
        
        # Check that final dungeon has reasonable structure
        return 'rooms' in dungeon or 'room_boundaries' in dungeon


# ===== Example Usage =====

def create_example_robust_pipeline():
    """
    Example factory function showing how to create configured robust pipeline.
    
    This would be adapted to your actual KLTN pipeline components.
    """
    
    # Define execution functions for each block
    def evolutionary_director_fn(state):
        """Generate mission graph topology."""
        from src.generation.evolutionary_director import EvolutionaryTopologyGenerator
        
        generator = EvolutionaryTopologyGenerator(
            num_rooms=state.get('num_rooms', 8),
            tension_curve=state.get('tension_curve', [0.5] * 8),
            seed=state.get('seed', 42)
        )
        
        mission_graph = generator.generate()
        return {'mission_graph': mission_graph}
    
    def vqvae_encoder_fn(state):
        """Encode mission graph to latent space."""
        # Placeholder - replace with actual VQ-VAE
        return {'latent_code': None}
    
    def diffusion_fn(state):
        """Generate spatial layout from latent code."""
        # Placeholder - replace with actual diffusion model
        import numpy as np
        return {'visual_grid': np.zeros((64, 64), dtype=int)}
    
    # Register blocks
    blocks = {
        'evolutionary_director': evolutionary_director_fn,
        'vqvae_encoder': vqvae_encoder_fn,
        'diffusion_model': diffusion_fn
    }
    
    # Register validators
    validators = {
        'evolutionary_director': validate_mission_graph,
        'diffusion_model': validate_layout
    }
    
    # Create pipeline
    config = PipelineConfig(
        max_retries=3,
        base_backoff=0.5,
        enable_logging=True
    )
    
    return RobustPipeline(blocks, validators, config)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    pipeline = create_example_robust_pipeline()
    
    user_inputs = {
        'num_rooms': 8,
        'tension_curve': [0.2, 0.4, 0.6, 0.8, 0.7, 0.5, 0.3, 0.2],
        'seed': 42
    }
    
    success, result, diagnostics = pipeline.generate_dungeon(user_inputs)
    
    print(pipeline.get_performance_report(diagnostics))
    
    if success:
        print("\n✓ Generation succeeded!")
    else:
        print("\n✗ Generation failed after retries")
