"""
Robust Pipeline with Retry Logic
Prevents cascade failures and provides graceful degradation during thesis defense demos.

This addresses the critical concern: "What if your diffusion model fails during the live demo?"
"""

import time
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from enum import Enum
from typing import Dict, Any, Callable, Tuple, Optional, List
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
    attempt_errors: List[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    """Configuration for robust pipeline execution."""
    max_retries: int = 5
    base_backoff: float = 0.5  # Starting backoff in seconds
    max_backoff: float = 30.0  # Maximum backoff in seconds
    backoff_multiplier: float = 2.0  # Exponential backoff multiplier
    enable_logging: bool = True
    timeout_per_block: Optional[float] = None  # Seconds, None = no timeout
    human_playability_filter_enabled: bool = False
    human_playability_persona: str = "balanced"
    human_playability_astar_budget: int = 100_000
    human_playability_pcbs_budget: int = 5_000
    human_playability_max_normalized_confusion: float = 1.0


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
        attempt_errors: List[str] = []
        
        for attempt in range(1, self.config.max_retries + 1):
            if self.config.enable_logging:
                logger.info(f"[{self.name}] Attempt {attempt}/{self.config.max_retries}")
            
            start_time = time.time()
            
            try:
                # Execute the block
                if self.config.timeout_per_block is not None and self.config.timeout_per_block > 0:
                    pool = ThreadPoolExecutor(max_workers=1)
                    try:
                        future = pool.submit(self.executor, state)
                        output = future.result(timeout=self.config.timeout_per_block)
                    except FutureTimeoutError:
                        future.cancel()
                        pool.shutdown(wait=False, cancel_futures=True)
                        raise
                    except Exception:
                        pool.shutdown(wait=True, cancel_futures=True)
                        raise
                    else:
                        pool.shutdown(wait=True)
                else:
                    output = self.executor(state)
                execution_time = time.time() - start_time
                
                # Validate output
                is_valid = self.validator(output)
                
                if is_valid:
                    if self.config.enable_logging:
                        logger.info(f"[{self.name}] [OK] Success in {execution_time:.2f}s")
                    
                    return BlockResult(
                        status=BlockStatus.SUCCESS,
                        output=output,
                        execution_time=execution_time,
                        attempts=attempt
                    )
                else:
                    raise ValueError(f"Validation failed for {self.name}")
            
            except FutureTimeoutError:
                execution_time = time.time() - start_time
                error_msg = (
                    f"TimeoutError: {self.name} exceeded timeout_per_block="
                    f"{self.config.timeout_per_block}s"
                )
                attempt_errors.append(error_msg)

                if self.config.enable_logging:
                    logger.warning(f"[{self.name}] [FAIL] Attempt {attempt} failed: {error_msg}")

                if attempt < self.config.max_retries:
                    if self.config.enable_logging:
                        logger.info(f"[{self.name}] Retrying in {backoff:.1f}s...")

                    time.sleep(backoff)
                    backoff = min(backoff * self.config.backoff_multiplier, self.config.max_backoff)
                else:
                    if self.config.enable_logging:
                        logger.error(f"[{self.name}] Failed after {self.config.max_retries} attempts")

                    return BlockResult(
                        status=BlockStatus.FAILED,
                        error=error_msg,
                        execution_time=execution_time,
                        attempts=attempt,
                        attempt_errors=list(attempt_errors),
                    )
            except (AttributeError, RuntimeError, ValueError, TypeError) as e:
                execution_time = time.time() - start_time
                error_msg = f"{type(e).__name__}: {str(e)}"
                attempt_errors.append(error_msg)
                
                if self.config.enable_logging:
                    logger.warning(f"[{self.name}] [FAIL] Attempt {attempt} failed: {error_msg}")

                # Shape/configuration contract failures are deterministic for
                # a fixed block input. Retrying the same state only hides the
                # root cause and can exhaust an entire generation batch.
                if getattr(e, "retryable", True) is False:
                    if self.config.enable_logging:
                        logger.error(
                            "[%s] Non-retryable contract failure; aborting after attempt %d",
                            self.name,
                            attempt,
                        )
                    return BlockResult(
                        status=BlockStatus.FAILED,
                        error=error_msg,
                        execution_time=execution_time,
                        attempts=attempt,
                        attempt_errors=list(attempt_errors),
                    )
                
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
                        attempts=attempt,
                        attempt_errors=list(attempt_errors),
                    )
        
        # Should never reach here, but handle gracefully
        return BlockResult(
            status=BlockStatus.FAILED,
            error="Max retries exceeded",
            attempts=self.config.max_retries,
            attempt_errors=list(attempt_errors),
        )


class RobustPipeline:
    """
    7-block H-MOLQD pipeline with comprehensive error handling.
    
    Architecture:
        Block I:   Evolutionary Director (topology generation)
        Block II:  VQ-VAE Encoder (graph -> latent)
        Block III: Condition Encoder (user controls -> embedding)
        Block IV:  Diffusion Model (latent -> spatial layout)
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
                logger.warning(f"[WARN] Block '{block_name}' not configured, skipping")
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

            if (
                block_name == "wfc_refiner"
                and self.config.human_playability_filter_enabled
            ):
                started = time.time()
                accepted, metrics = evaluate_human_playability(
                    pipeline_state,
                    persona=self.config.human_playability_persona,
                    astar_budget=self.config.human_playability_astar_budget,
                    pcbs_budget=self.config.human_playability_pcbs_budget,
                    max_normalized_confusion=(
                        self.config.human_playability_max_normalized_confusion
                    ),
                )
                diagnostics["human_playability"] = BlockResult(
                    status=BlockStatus.SUCCESS if accepted else BlockStatus.FAILED,
                    output=metrics,
                    error=None if accepted else "P-CBS acceptance policy rejected the dungeon",
                    execution_time=time.time() - started,
                )
                pipeline_state["human_playability_metrics"] = metrics
                if not accepted:
                    logger.info(
                        "Pipeline rejected the exact-solvable dungeon under the "
                        "opt-in P-CBS acceptance policy"
                    )
                    return False, pipeline_state, diagnostics
        
        logger.info("=" * 60)
        logger.info("[OK] Pipeline completed successfully")
        logger.info("=" * 60)
        
        return True, pipeline_state, diagnostics
    
    def get_performance_report(self, diagnostics: Dict[str, BlockResult]) -> str:
        """Generate human-readable performance report."""
        lines = ["\n=== Pipeline Performance Report ==="]
        
        total_time = sum(r.execution_time for r in diagnostics.values())
        total_attempts = sum(r.attempts for r in diagnostics.values())
        
        for name, result in diagnostics.items():
            status_symbol = {
                BlockStatus.SUCCESS: "[OK]",
                BlockStatus.FAILED: "[FAIL]",
                BlockStatus.RETRYING: "[...]"
            }.get(result.status, "?")
            
            lines.append(
                f"{status_symbol} {name:25s} | "
                f"{result.execution_time:6.2f}s | "
                f"{result.attempts} attempts"
            )
            
            if result.error:
                lines.append(f"   Error: {result.error}")
            if result.attempt_errors:
                recent = result.attempt_errors[-3:]
                for idx, err in enumerate(recent, start=max(1, len(result.attempt_errors) - len(recent) + 1)):
                    lines.append(f"   Attempt {idx}: {err}")
        
        lines.append(f"\nTotal time: {total_time:.2f}s")
        lines.append(f"Total attempts: {total_attempts}")
        
        return "\n".join(lines)


# ===== Built-in Validators =====

def validate_mission_graph(graph: Any) -> bool:
    """Validate mission graph structure and resource-aware progression."""
    if not isinstance(graph, dict):
        return False
    
    # Allow wrapped block outputs: {'mission_graph': {...}}
    if 'mission_graph' in graph and 'nodes' not in graph:
        graph = graph.get('mission_graph', {})
        if not isinstance(graph, dict):
            return False
    
    required_keys = ['nodes', 'edges']
    if not all(k in graph for k in required_keys):
        return False
    
    if len(graph['nodes']) < 3:  # Minimum: start, middle, boss
        return False
    
    # Check connectivity (undirected reachability)
    if 'adjacency' in graph:
        adjacency = graph['adjacency']
        nodes = list(graph['nodes'].keys())
        if len(nodes) > 1:
            # Build undirected adjacency to avoid false negatives on sink nodes.
            undirected = {n: set() for n in nodes}
            for src, nbrs in adjacency.items():
                if src not in undirected:
                    undirected[src] = set()
                for dst in nbrs:
                    undirected.setdefault(dst, set())
                    undirected[src].add(dst)
                    undirected[dst].add(src)
            
            # Every node must have at least one connection in non-trivial graphs.
            if any(len(undirected.get(n, set())) == 0 for n in nodes):
                return False
            
            # Graph should be single connected component.
            start = nodes[0]
            stack = [start]
            seen = {start}
            while stack:
                cur = stack.pop()
                for nb in undirected.get(cur, set()):
                    if nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
            if len(seen) != len(nodes):
                return False
    
    try:
        import networkx as nx
        from src.evaluation.validator import ExternalValidator

        directed = nx.DiGraph()
        directed.add_nodes_from(
            (node_id, dict(attrs))
            for node_id, attrs in dict(graph["nodes"]).items()
        )
        for edge in graph["edges"]:
            if not isinstance(edge, (list, tuple)) or len(edge) < 2:
                return False
            attrs = dict(edge[2]) if len(edge) >= 3 and isinstance(edge[2], dict) else {}
            directed.add_edge(edge[0], edge[1], **attrs)
        result = ExternalValidator(mode="full").validate(directed)
        return bool(result.is_solvable)
    except (ImportError, AttributeError, RuntimeError, ValueError, TypeError):
        logger.exception("Mission-graph validation failed")
        return False


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
    
    # Check for reasonable floor coverage (20-70%).
    # Canonical floor id is SEMANTIC_PALETTE['FLOOR'] (typically 1), but we
    # also accept legacy grids that use 0 for floor.
    floor_id = 1
    try:
        from src.core.definitions import SEMANTIC_PALETTE
        floor_id = int(SEMANTIC_PALETTE.get('FLOOR', 1))
    except Exception:
        floor_id = 1
    floor_count = int(np.sum(grid == floor_id))
    if floor_id != 0:
        floor_count += int(np.sum(grid == 0))
    floor_ratio = floor_count / grid.size
    if not (0.2 <= floor_ratio <= 0.7):
        return False
    
    return True


def validate_solvability(dungeon: Any) -> bool:
    """Validate the exact semantic grid with the canonical tile-state oracle."""
    if not isinstance(dungeon, dict):
        return False
    
    # Check for required keys
    if 'visual_grid' not in dungeon:
        return False
    
    try:
        import numpy as np
        from src.simulation.validator import ZeldaValidator

        grid = np.asarray(dungeon["visual_grid"])
        if grid.ndim != 2:
            return False
        return bool(ZeldaValidator().validate_single(grid).is_solvable)
    except (ImportError, AttributeError, RuntimeError, ValueError, TypeError):
        logger.exception("Tile-state validation failed")
        return False


def _extract_visual_grid(payload: Any) -> Any:
    """Find the final semantic grid in a pipeline-state or block wrapper."""
    if isinstance(payload, dict):
        if "visual_grid" in payload:
            return payload["visual_grid"]
        for key in ("wfc_refiner", "wfc_refiner_output", "layout", "dungeon"):
            if key in payload:
                grid = _extract_visual_grid(payload[key])
                if grid is not None:
                    return grid
    return None


def evaluate_human_playability(
    dungeon: Any,
    *,
    persona: str = "balanced",
    astar_budget: int = 100_000,
    pcbs_budget: int = 5_000,
    max_normalized_confusion: float = 1.0,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Apply an opt-in bounded-player acceptance policy after exact validation.

    P-CBS failure is reported as a human-model rejection, not as mathematical
    unsolvability. Thresholds therefore belong in calibrated ablations.
    """
    import math
    import numpy as np

    from src.evaluation.cbs_fitness import compute_cbs_fitness

    grid = _extract_visual_grid(dungeon)
    if grid is None:
        return False, {"failure_reason": "missing visual_grid"}
    metrics = compute_cbs_fitness(
        np.asarray(grid),
        persona=str(persona),
        astar_timeout=max(1, int(astar_budget)),
        cbs_timeout=max(1, int(pcbs_budget)),
    )
    normalized = float(metrics.get("normalized_confusion_ratio", float("inf")))
    accepted = (
        bool(metrics.get("solvable_astar", False))
        and bool(metrics.get("solvable_cbs", False))
        and math.isfinite(normalized)
        and normalized <= float(max_normalized_confusion)
    )
    metrics = dict(metrics)
    metrics["accepted_by_human_playability_policy"] = bool(accepted)
    metrics["max_normalized_confusion"] = float(max_normalized_confusion)
    return bool(accepted), metrics


# ===== Example Usage =====

def create_example_robust_pipeline():
    """
    Example factory function showing how to create configured robust pipeline.
    
    This would be adapted to your actual KLTN pipeline components.
    """
    
    # Define execution functions for each block
    def evolutionary_director_fn(state):
        """Generate mission graph topology."""
        import networkx as nx
        from src.generation.evolutionary_director import EvolutionaryTopologyGenerator
        
        num_rooms = int(state.get('num_rooms', 8))
        tension_curve = list(state.get('tension_curve', [0.5] * max(1, num_rooms)))
        seed = int(state.get('seed', 42))
        
        generator = EvolutionaryTopologyGenerator(
            target_curve=tension_curve,
            population_size=max(16, num_rooms * 4),
            generations=max(20, min(80, num_rooms * 8)),
            max_nodes=max(num_rooms, 5),
            seed=seed,
        )
        graph = generator.evolve()
        
        # Convert to deterministic directed payload for downstream blocks/validators.
        if isinstance(graph, nx.DiGraph):
            directed = graph.copy()
        else:
            directed = nx.DiGraph()
            directed.add_nodes_from(graph.nodes(data=True))
            directed.add_edges_from((u, v, {'edge_type': 'open'}) for u, v in graph.edges())
        
        ordered_nodes = sorted(directed.nodes(), key=lambda n: (type(n).__name__, str(n)))
        remap = {old: idx for idx, old in enumerate(ordered_nodes)}
        directed = nx.relabel_nodes(directed, remap, copy=True)
        
        # Guarantee a minimum playable graph size for downstream validation/demo.
        target_nodes = max(3, num_rooms)
        if directed.number_of_nodes() == 0:
            directed.add_node(0)
        while directed.number_of_nodes() < target_nodes:
            new_id = directed.number_of_nodes()
            directed.add_node(new_id)
            directed.add_edge(new_id - 1, new_id, edge_type='open')
        
        # Ensure weak connectivity.
        components = list(nx.weakly_connected_components(directed))
        if len(components) > 1:
            for i in range(len(components) - 1):
                src = min(components[i])
                dst = min(components[i + 1])
                directed.add_edge(src, dst, edge_type='open')
        
        for node_id, attrs in directed.nodes(data=True):
            attrs.setdefault('is_start', node_id == 0)
            attrs.setdefault('is_boss', node_id == directed.number_of_nodes() - 1)
            attrs.setdefault('is_goal', node_id == directed.number_of_nodes() - 1)
            attrs.setdefault('tension', float(tension_curve[min(node_id, len(tension_curve) - 1)]))
        
        payload = {
            'nodes': {n: dict(directed.nodes[n]) for n in directed.nodes()},
            'edges': [(u, v, dict(directed.get_edge_data(u, v, default={}))) for u, v in directed.edges()],
            'adjacency': {n: list(directed.successors(n)) for n in directed.nodes()},
        }
        return {'mission_graph': payload}
    
    def vqvae_encoder_fn(state):
        """Encode mission graph to latent space."""
        import numpy as np
        
        mission_graph = state.get('mission_graph', {})
        nodes = mission_graph.get('nodes', {})
        adjacency = mission_graph.get('adjacency', {})
        if not nodes:
            raise ValueError("Missing mission_graph nodes for encoder")
        
        ordered = sorted(nodes.keys())
        latent = np.zeros((len(ordered), 4), dtype=np.float32)
        for i, node_id in enumerate(ordered):
            attrs = nodes.get(node_id, {})
            latent[i, 0] = float(attrs.get('tension', 0.5))
            latent[i, 1] = float(attrs.get('is_start', False))
            latent[i, 2] = float(attrs.get('is_boss', False))
            latent[i, 3] = float(len(adjacency.get(node_id, []))) / max(1, len(ordered) - 1)
        
        return {'latent_code': latent}
    
    def diffusion_fn(state):
        """Generate spatial layout from latent code."""
        import numpy as np
        
        seed = int(state.get('seed', 42))
        rng = np.random.default_rng(seed)
        
        # Build a deterministic pseudo-layout with controlled floor ratio.
        height, width = 64, 64
        grid = np.ones((height, width), dtype=np.int64)  # 1 = wall-like
        interior = np.zeros((height - 2, width - 2), dtype=np.int64)  # 0 = floor-like
        
        # Add structured obstacles so floor coverage is realistically constrained.
        obstacle_mask = rng.random(interior.shape) < 0.35
        interior[obstacle_mask] = 1
        grid[1:-1, 1:-1] = interior
        
        return {'visual_grid': grid}
    
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
        print("\n[OK] Generation succeeded!")
    else:
        print("\n[FAIL] Generation failed after retries")
