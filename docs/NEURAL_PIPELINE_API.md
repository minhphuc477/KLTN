# Neural-Symbolic Dungeon Pipeline - Complete API Reference

**Module**: `src.pipeline.dungeon_pipeline`  
**Version**: 1.0.0  
**Status**: Production Ready ✅

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Core Classes](#core-classes)
4. [Pipeline API](#pipeline-api)
5. [Data Structures](#data-structures)
6. [Usage Examples](#usage-examples)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The **NeuralSymbolicDungeonPipeline** is the master orchestrator for the complete 7-block H-MOLQD (Hierarchical Multi-Objective Latent Quality-Diversity) architecture. It integrates:

- **Block I**: Data Adapter (`zelda_core.py`)
- **Block II**: Semantic VQ-VAE (`vqvae.py`)
- **Block III**: Dual-Stream Condition Encoder (`condition_encoder.py`)
- **Block IV**: Latent Diffusion with Guidance (`latent_diffusion.py`)
- **Block V**: LogicNet (`logic_net.py`)
- **Block VI**: Symbolic Refiner (`symbolic_refiner.py`)
- **Block VII**: MAP-Elites Validator (`map_elites.py`)

### Architecture Flow

```
Mission Graph → Graph Context Preparation
                ↓
For each room in topological order:
    1. Get neighbor latents (already generated rooms)
    2. Encode dual-stream context (local + global)
    3. Sample latent with diffusion + LogicNet guidance
    4. Decode to discrete tiles via VQ-VAE
    5. Apply symbolic WFC repair (if enabled)
    6. Store result
                ↓
Stitch rooms → Complete dungeon
                ↓
Validate with MAP-Elites → Quality-diversity metrics
```

---

## Quick Start

### Installation

```bash
# Install dependencies
pip install torch numpy networkx

# Optional (for visualization)
pip install matplotlib seaborn
```

### Basic Usage

```python
from src.pipeline import create_pipeline
import networkx as nx

# Initialize pipeline
pipeline = create_pipeline(
    checkpoint_dir="./checkpoints",
    device='cuda'
)

# Create mission graph
G = nx.DiGraph()
G.add_nodes_from([0, 1, 2])
G.add_edges_from([(0, 1), (1, 2)])

# Generate dungeon
result = pipeline.generate_dungeon(
    mission_graph=G,
    seed=42
)

# Access results
print(f"Generated {len(result.rooms)} rooms")
print(f"Dungeon shape: {result.dungeon_grid.shape}")
print(f"Repair rate: {result.metrics['repair_rate']:.1%}")

# Save dungeon
import numpy as np
np.save("dungeon.npy", result.dungeon_grid)
```

---

## Core Classes

### NeuralSymbolicDungeonPipeline

**Main pipeline orchestrator integrating all 7 blocks.**

#### Constructor

```python
NeuralSymbolicDungeonPipeline(
    vqvae_checkpoint: Optional[str] = None,
    diffusion_checkpoint: Optional[str] = None,
    logic_net_checkpoint: Optional[str] = None,
    condition_encoder_checkpoint: Optional[str] = None,
    device: str = 'auto',
    use_learned_refiner_rules: bool = True,
    map_elites_resolution: int = 20,
    enable_logging: bool = True,
)
```

**Parameters**:

- `vqvae_checkpoint` (str, optional): Path to VQ-VAE checkpoint file (.pth)
- `diffusion_checkpoint` (str, optional): Path to diffusion model checkpoint
- `logic_net_checkpoint` (str, optional): Path to LogicNet checkpoint
- `condition_encoder_checkpoint` (str, optional): Path to condition encoder checkpoint
- `device` (str): Device to run on ('cuda', 'cpu', or 'auto')
- `use_learned_refiner_rules` (bool): Use learned tile statistics for WFC repair
- `map_elites_resolution` (int): MAP-Elites grid resolution (default: 20×20)
- `enable_logging` (bool): Enable detailed logging

**Attributes**:

- `vqvae` (SemanticVQVAE): VQ-VAE model (Block II)
- `condition_encoder` (DualStreamConditionEncoder): Context encoder (Block III)
- `diffusion` (LatentDiffusionModel): Diffusion model (Block IV)
- `logic_net` (LogicNet): Solvability loss model (Block V)
- `refiner` (SymbolicRefiner): WFC repair module (Block VI)
- `map_elites` (MAPElitesEvaluator): Quality-diversity evaluator (Block VII)
- `device` (torch.device): Device being used

**Methods**:

- [`generate_room()`](#generate_room) - Generate single room
- [`generate_dungeon()`](#generate_dungeon) - Generate complete dungeon

---

## Pipeline API

### generate_room()

**Generate a single room using the complete 7-block pipeline.**

```python
@torch.no_grad()
def generate_room(
    self,
    neighbor_latents: Dict[str, Optional[torch.Tensor]],
    graph_context: Dict[str, Any],
    room_id: int,
    boundary_constraints: Optional[torch.Tensor] = None,
    position: Optional[torch.Tensor] = None,
    guidance_scale: float = 7.5,
    logic_guidance_scale: float = 1.0,
    num_diffusion_steps: int = 50,
    use_ddim: bool = True,
    apply_repair: bool = True,
    start_goal_coords: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
    seed: Optional[int] = None,
) -> RoomGenerationResult
```

**Parameters**:

- `neighbor_latents` (Dict[str, Optional[Tensor]]): Neighboring room latents
  - Keys: 'N', 'S', 'E', 'W' (cardinal directions)
  - Values: (1, 64, 4, 3) latent tensors or None
  
- `graph_context` (Dict[str, Any]): Graph neural network context
  - `node_features`: (num_nodes, feature_dim) node embeddings
  - `edge_index`: (2, num_edges) edge connectivity
  - `tpe`: (num_nodes, 8) topological positional encoding
  - `current_node_idx`: Index of current room in graph

- `room_id` (int): Unique room identifier

- `boundary_constraints` (Tensor, optional): (1, 8) door configuration mask
  - Default: zeros (no doors)

- `position` (Tensor, optional): (1, 2) grid position [row, col]
  - Default: zeros

- `guidance_scale` (float): Classifier-free guidance scale
  - Higher = stronger conditioning (typical: 5.0-10.0)
  - Default: 7.5

- `logic_guidance_scale` (float): LogicNet gradient guidance scale
  - Controls solvability guidance strength
  - 0.0 = disabled, 1.0-2.0 = moderate, 5.0+ = strong
  - Default: 1.0

- `num_diffusion_steps` (int): Number of DDIM/DDPM sampling steps
  - More steps = higher quality but slower
  - Typical: 50-200
  - Default: 50

- `use_ddim` (bool): Use DDIM (deterministic) vs DDPM (stochastic)
  - DDIM is faster and deterministic
  - Default: True

- `apply_repair` (bool): Apply symbolic WFC repair
  - Fixes unsolvable paths
  - Default: True

- `start_goal_coords` (Tuple, optional): ((start_r, start_c), (goal_r, goal_c))
  - Required if `apply_repair=True`
  - Defines entrance/exit for pathfinding

- `seed` (int, optional): Random seed for reproducibility

**Returns**:

`RoomGenerationResult` with:
- `room_grid` (np.ndarray): (16, 11) final discrete tile IDs
- `latent` (torch.Tensor): (1, 64, 4, 3) VQ-VAE latent
- `neural_grid` (np.ndarray): (16, 11) tiles before repair
- `was_repaired` (bool): Whether repair was applied
- `repair_mask` (np.ndarray, optional): (16, 11) changed tiles
- `metrics` (dict): Generation statistics

**Example**:

```python
# Prepare inputs
neighbor_latents = {
    'N': previous_room_latent,  # (1, 64, 4, 3)
    'S': None,
    'E': None,
    'W': None,
}

graph_context = {
    'node_features': torch.randn(5, 6),
    'edge_index': torch.tensor([[0,1], [1,2]]).t(),
    'tpe': torch.randn(5, 8),
    'current_node_idx': 2,
}

# Generate room
result = pipeline.generate_room(
    neighbor_latents=neighbor_latents,
    graph_context=graph_context,
    room_id=2,
    start_goal_coords=((8, 0), (8, 10)),
    seed=42
)

# Use result
print(f"Room shape: {result.room_grid.shape}")
print(f"Was repaired: {result.was_repaired}")
```

---

### generate_dungeon()

**Generate a complete multi-room dungeon using graph-guided generation.**

```python
@torch.no_grad()
def generate_dungeon(
    self,
    mission_graph: nx.Graph,
    guidance_scale: float = 7.5,
    logic_guidance_scale: float = 1.0,
    num_diffusion_steps: int = 50,
    apply_repair: bool = True,
    seed: Optional[int] = None,
    enable_map_elites: bool = True,
) -> DungeonGenerationResult
```

**Parameters**:

- `mission_graph` (nx.Graph): NetworkX directed graph
  - Nodes represent rooms
  - Edges represent doors/connections
  - Node attributes (optional): room_type, items, etc.

- `guidance_scale` (float): Classifier-free guidance scale
  - Default: 7.5

- `logic_guidance_scale` (float): LogicNet gradient guidance scale
  - Default: 1.0

- `num_diffusion_steps` (int): Diffusion steps per room
  - Default: 50

- `apply_repair` (bool): Apply symbolic repair to all rooms
  - Default: True

- `seed` (int, optional): Random seed for reproducibility

- `enable_map_elites` (bool): Compute MAP-Elites metrics
  - Default: True

**Returns**:

`DungeonGenerationResult` with:
- `dungeon_grid` (np.ndarray): Complete stitched dungeon
- `rooms` (Dict[int, RoomGenerationResult]): Individual room results
- `mission_graph` (nx.Graph): Original graph
- `metrics` (dict): Overall statistics
- `map_elites_score` (dict, optional): Quality-diversity metrics
- `generation_time` (float): Time in seconds

**Example**:

```python
import networkx as nx

# Create mission graph
G = nx.DiGraph()
G.add_nodes_from([0, 1, 2, 3])
G.add_edges_from([(0, 1), (1, 2), (2, 3)])

# Generate dungeon
result = pipeline.generate_dungeon(
    mission_graph=G,
    guidance_scale=7.5,
    logic_guidance_scale=1.0,
    num_diffusion_steps=50,
    seed=42
)

# Access results
print(f"Rooms: {len(result.rooms)}")
print(f"Shape: {result.dungeon_grid.shape}")
print(f"Time: {result.generation_time:.2f}s")

# Per-room analysis
for room_id, room in result.rooms.items():
    print(f"Room {room_id}: repaired={room.was_repaired}")
```

---

## Data Structures

### RoomGenerationResult

**Result of generating a single room.**

```python
@dataclass
class RoomGenerationResult:
    room_id: int                    # Room identifier
    room_grid: np.ndarray          # (16, 11) final discrete tiles
    latent: torch.Tensor           # (1, 64, 4, 3) VQ-VAE latent
    neural_grid: np.ndarray        # (16, 11) before repair
    was_repaired: bool             # Whether repair was needed
    repair_mask: Optional[np.ndarray]  # (16, 11) bool mask
    metrics: Dict[str, float]      # Statistics
```

**Metrics Keys**:
- `room_id`: Room identifier
- `neural_grid_entropy`: Entropy of neural output
- `was_repaired`: Boolean flag
- `tiles_changed`: Number of tiles modified by repair

---

### DungeonGenerationResult

**Result of generating a complete dungeon.**

```python
@dataclass
class DungeonGenerationResult:
    dungeon_grid: np.ndarray       # (H, W) stitched dungeon
    rooms: Dict[int, RoomGenerationResult]  # Per-room results
    mission_graph: nx.Graph        # Original graph
    metrics: Dict[str, Any]        # Overall statistics
    map_elites_score: Optional[Dict[str, float]]  # QD metrics
    generation_time: float         # Seconds
```

**Metrics Keys**:
- `num_rooms`: Number of rooms generated
- `total_tiles_repaired`: Total tiles modified
- `repair_rate`: Fraction of rooms repaired [0, 1]
- `dungeon_shape`: (height, width) tuple
- `generation_time_sec`: Generation time

**MAP-Elites Score Keys** (if enabled):
- `linearity`: Path directness [0, 1]
- `leniency`: Item density [0, 1]
- `path_length`: Optimal path length (tiles)

---

## Usage Examples

### Example 1: Single Room Generation

```python
from src.pipeline import create_pipeline
import torch

pipeline = create_pipeline(device='cuda')

# Just north neighbor
neighbor_latents = {
    'N': torch.randn(1, 64, 4, 3, device='cuda'),
    'S': None, 'E': None, 'W': None,
}

# Minimal graph context
graph_context = {
    'node_features': torch.randn(1, 6, device='cuda'),
    'edge_index': torch.zeros(2, 0, dtype=torch.long, device='cuda'),
    'tpe': torch.randn(1, 8, device='cuda'),
    'current_node_idx': 0,
}

result = pipeline.generate_room(
    neighbor_latents=neighbor_latents,
    graph_context=graph_context,
    room_id=0,
    seed=42
)

print(result.room_grid.shape)  # (16, 11)
```

### Example 2: Linear Dungeon

```python
import networkx as nx

# Create linear graph
G = nx.DiGraph()
G.add_nodes_from(range(5))
G.add_edges_from([(i, i+1) for i in range(4)])

result = pipeline.generate_dungeon(G, seed=42)

# Save
import numpy as np
np.save("linear_dungeon.npy", result.dungeon_grid)
```

### Example 3: Branching Dungeon

```python
# Create branching graph
G = nx.DiGraph()
G.add_edges_from([
    (0, 1), (0, 2),    # Start branches
    (1, 3), (2, 4),    # Sub-branches
    (3, 5), (4, 5)     # Converge to goal
])

result = pipeline.generate_dungeon(
    G,
    logic_guidance_scale=2.0,  # Strong solvability guidance
    num_diffusion_steps=100,    # High quality
    seed=42
)
```

### Example 4: Batch Generation

```python
# Generate multiple dungeons
dungeons = []

for seed in range(10):
    result = pipeline.generate_dungeon(
        mission_graph=my_graph,
        seed=seed
    )
    dungeons.append(result)

# Analyze diversity
repair_rates = [d.metrics['repair_rate'] for d in dungeons]
print(f"Mean repair rate: {np.mean(repair_rates):.1%}")
```

### Example 5: Custom Checkpoint Loading

```python
pipeline = NeuralSymbolicDungeonPipeline(
    vqvae_checkpoint="models/vqvae_epoch100.pth",
    diffusion_checkpoint="models/diffusion_best.pth",
    logic_net_checkpoint="models/logic_net_final.pth",
    device='cuda',
    use_learned_refiner_rules=True,
)
```

---

## Best Practices

### Performance Optimization

1. **Use CUDA when available**:
   ```python
   pipeline = create_pipeline(device='cuda')
   ```

2. **Reduce diffusion steps for speed**:
   ```python
   result = pipeline.generate_dungeon(
       mission_graph=G,
       num_diffusion_steps=25  # Faster, slight quality loss
   )
   ```

3. **Disable repair for pure neural output**:
   ```python
   result = pipeline.generate_dungeon(
       mission_graph=G,
       apply_repair=False  # Faster, may be unsolvable
   )
   ```

4. **Disable MAP-Elites for faster iteration**:
   ```python
   result = pipeline.generate_dungeon(
       mission_graph=G,
       enable_map_elites=False
   )
   ```

### Quality Optimization

1. **Increase diffusion steps**:
   ```python
   result = pipeline.generate_dungeon(
       mission_graph=G,
       num_diffusion_steps=200  # Higher quality
   )
   ```

2. **Tune guidance scales**:
   ```python
   # Strong conditioning
   result = pipeline.generate_dungeon(
       mission_graph=G,
       guidance_scale=10.0,        # Strong CFG
       logic_guidance_scale=2.0    # Strong solvability
   )
   ```

3. **Use learned refiner rules**:
   ```python
   pipeline = create_pipeline(
       use_learned_refiner_rules=True  # Data-driven WFC
   )
   ```

### Reproducibility

Always set seed for deterministic generation:

```python
result = pipeline.generate_dungeon(
    mission_graph=G,
    seed=42  # Reproducible
)
```

### Memory Management

For large dungeons or limited VRAM:

```python
# Generate rooms sequentially (already done internally)
# Or use CPU for very large graphs
pipeline = create_pipeline(device='cpu')
```

---

## Troubleshooting

### Common Issues

**Issue**: "CUDA out of memory"

**Solution**:
```python
# Use CPU
pipeline = create_pipeline(device='cpu')

# Or reduce batch size (if applicable)
```

---

**Issue**: "Checkpoint not found"

**Solution**:
```python
# Pipeline works without checkpoints (random init)
pipeline = create_pipeline(checkpoint_dir="./nonexistent")

# Or verify path
from pathlib import Path
assert Path("./checkpoints/vqvae_best.pth").exists()
```

---

**Issue**: "Low quality output"

**Solution**:
```python
# Increase diffusion steps
result = pipeline.generate_dungeon(
    mission_graph=G,
    num_diffusion_steps=200,
    logic_guidance_scale=2.0
)
```

---

**Issue**: "Generation too slow"

**Solution**:
```python
# Reduce steps
result = pipeline.generate_dungeon(
    mission_graph=G,
    num_diffusion_steps=25,
    enable_map_elites=False
)
```

---

**Issue**: "Rooms not connected"

**Solution**:
```python
# This is expected for simple stitching
# Use DungeonStitcher from zelda_core for proper layout
from src.data.zelda_core import DungeonStitcher
stitcher = DungeonStitcher()
# Implement custom stitching logic based on graph
```

---

## Advanced Topics

### Custom Graph Contexts

For more sophisticated graph encoding:

```python
# Prepare detailed node features
node_features = compute_node_embeddings(mission_graph)  # Custom function
edge_index = extract_graph_edges(mission_graph)
tpe = compute_laplacian_eigenvectors(mission_graph)

graph_context = {
    'node_features': node_features,
    'edge_index': edge_index,
    'tpe': tpe,
    'current_node_idx': room_idx,
}
```

### Custom Repair Logic

Override repair behavior:

```python
class CustomPipeline(NeuralSymbolicDungeonPipeline):
    def _extract_room_start_goal(self, graph, room_id):
        # Custom logic to determine entrance/exit
        return ((custom_start_r, custom_start_c), 
                (custom_goal_r, custom_goal_c))
```

### Extending the Pipeline

Add custom post-processing:

```python
class ExtendedPipeline(NeuralSymbolicDungeonPipeline):
    def generate_dungeon(self, *args, **kwargs):
        result = super().generate_dungeon(*args, **kwargs)
        
        # Custom post-processing
        result.dungeon_grid = self.apply_custom_filter(result.dungeon_grid)
        
        return result
```

---

## API Changelog

### Version 1.0.0 (2026-02-13)
- Initial release
- Complete 7-block integration
- Single room and multi-room generation
- MAP-Elites evaluation
- Comprehensive error handling

---

## References

- **Architecture**: See `docs/NEURAL_PIPELINE_RESEARCH.md`
- **Block Details**: See `docs/BLOCK_IO_REFERENCE.md`
- **Training**: See `docs/TRAINING_GUIDE.md`
- **Examples**: See `examples/neural_generation_demo.py`

---

**Last Updated**: February 13, 2026  
**Maintainer**: KLTN Team
