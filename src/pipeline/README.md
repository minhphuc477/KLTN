# Neural-Symbolic Dungeon Pipeline

**Complete 7-block H-MOLQD pipeline for Legend of Zelda dungeon generation.**

---

## Quick Start

```python
from src.pipeline import create_pipeline
import networkx as nx

# Initialize pipeline
pipeline = create_pipeline(checkpoint_dir="./checkpoints")

# Create mission graph
G = nx.DiGraph()
G.add_nodes_from([0, 1, 2, 3])
G.add_edges_from([(0, 1), (1, 2), (2, 3)])

# Generate dungeon
result = pipeline.generate_dungeon(G, seed=42)

# Access results
print(f"Generated {len(result.rooms)} rooms")
print(f"Repair rate: {result.metrics['repair_rate']:.1%}")
print(f"Time: {result.generation_time:.2f}s")

# Save
import numpy as np
np.save("dungeon.npy", result.dungeon_grid)
```

---

## Features

- ✅ **Single room generation** with full neural-symbolic pipeline
- ✅ **Multi-room dungeons** with graph-guided generation
- ✅ **LogicNet guidance** for solvability optimization
- ✅ **Symbolic WFC repair** for broken paths
- ✅ **MAP-Elites evaluation** for quality-diversity metrics
- ✅ **Checkpoint management** for trained models
- ✅ **Reproducible generation** with seeds
- ✅ **CPU/CUDA support** for flexible deployment

---

## Architecture

**7 Integrated Blocks:**

1. **Data Adapter** (`zelda_core.py`) - VGLC parsing and graph alignment
2. **VQ-VAE** (`vqvae.py`) - Discrete latent representation
3. **Condition Encoder** (`condition_encoder.py`) - Local + global context fusion
4. **Latent Diffusion** (`latent_diffusion.py`) - Guided generation
5. **LogicNet** (`logic_net.py`) - Differentiable solvability loss
6. **Symbolic Refiner** (`symbolic_refiner.py`) - WFC-based repair
7. **MAP-Elites** (`map_elites.py`) - Quality-diversity evaluation

---

## Installation

```bash
# Core dependencies
pip install torch numpy networkx

# Optional (for visualization)
pip install matplotlib seaborn
```

---

## Usage Examples

### Example 1: Single Room

```python
from src.pipeline import create_pipeline
import torch

pipeline = create_pipeline(device='cuda')

neighbor_latents = {
    'N': torch.randn(1, 64, 4, 3, device='cuda'),
    'S': None, 'E': None, 'W': None,
}

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

print(f"Room shape: {result.room_grid.shape}")  # (16, 11)
```

### Example 2: Linear Dungeon

```python
import networkx as nx

# Create 5-room linear graph
G = nx.DiGraph()
G.add_nodes_from(range(5))
G.add_edges_from([(i, i+1) for i in range(4)])

# Generate
result = pipeline.generate_dungeon(G, seed=42)

# Save
np.save("linear_dungeon.npy", result.dungeon_grid)
```

### Example 3: High-Quality Generation

```python
result = pipeline.generate_dungeon(
    mission_graph=G,
    guidance_scale=10.0,           # Strong conditional guidance
    logic_guidance_scale=2.0,      # Strong solvability guidance
    num_diffusion_steps=200,       # High quality
    apply_repair=True,
    seed=42,
)
```

---

## Demo

```bash
# Run complete demo
python examples/neural_generation_demo.py

# With custom parameters
python examples/neural_generation_demo.py \
    --num-rooms 8 \
    --guidance-scale 10.0 \
    --logic-guidance-scale 2.0 \
    --seed 42 \
    --output-dir ./outputs

# Specific demo
python examples/neural_generation_demo.py --demo 2
```

---

## Testing

```bash
# Run all tests
pytest tests/test_neural_pipeline.py -v

# Quick smoke test
python tests/test_neural_pipeline.py
```

---

## Documentation

- **API Reference**: [`docs/NEURAL_PIPELINE_API.md`](../docs/NEURAL_PIPELINE_API.md)
- **Research Report**: [`docs/NEURAL_PIPELINE_RESEARCH.md`](../docs/NEURAL_PIPELINE_RESEARCH.md)
- **Implementation**: [`docs/NEURAL_PIPELINE_IMPLEMENTATION.md`](../docs/NEURAL_PIPELINE_IMPLEMENTATION.md)

---

## Performance

**Benchmarks (RTX 3080)**:

- Single room: ~700ms
- 16-room dungeon: ~12s
- Memory: ~2GB VRAM

**Optimization**:

```python
# Fast (lower quality)
result = pipeline.generate_dungeon(G, num_diffusion_steps=25)

# High quality (slower)
result = pipeline.generate_dungeon(G, num_diffusion_steps=200)
```

---

## API

### Main Classes

```python
NeuralSymbolicDungeonPipeline(
    vqvae_checkpoint=None,
    diffusion_checkpoint=None,
    logic_net_checkpoint=None,
    device='auto',
)
```

### Main Methods

```python
# Single room
result = pipeline.generate_room(
    neighbor_latents,
    graph_context,
    room_id,
    seed=42
)

# Full dungeon
result = pipeline.generate_dungeon(
    mission_graph,
    seed=42
)
```

### Result Structure

```python
# Room result
result.room_grid          # (16, 11) discrete tiles
result.latent            # (1, 64, 4, 3) VQ-VAE latent
result.was_repaired      # bool
result.metrics           # dict

# Dungeon result
result.dungeon_grid      # (H, W) stitched dungeon
result.rooms             # Dict[int, RoomGenerationResult]
result.metrics           # dict
result.map_elites_score  # dict (optional)
result.generation_time   # float
```

---

## Troubleshooting

**Q: CUDA out of memory?**
A: Use CPU mode: `pipeline = create_pipeline(device='cpu')`

**Q: Low quality output?**
A: Increase diffusion steps: `num_diffusion_steps=200`

**Q: Slow generation?**
A: Reduce steps: `num_diffusion_steps=25`

**Q: Missing checkpoints?**
A: Pipeline works without checkpoints (random init)

---

## Citation

```bibtex
@misc{kltn2026neural,
  title={Neural-Symbolic Dungeon Generation with H-MOLQD},
  author={Le Tran Minh Phuc},
  year={2026},
  howpublished={\url{https://github.com/minhphuc477/KLTN}}
}
```

---

## License

See main project LICENSE file.

---

**Status**: Production Ready ✅
**Last Updated**: February 13, 2026
