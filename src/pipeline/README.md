# Neural-Symbolic Dungeon Pipeline

Minimal usage guide for `src.pipeline`.

## Quick Start

```python
from src.pipeline import create_pipeline
import networkx as nx

pipeline = create_pipeline(checkpoint_dir="./checkpoints")

G = nx.DiGraph()
G.add_nodes_from([0, 1, 2, 3])
G.add_edges_from([(0, 1), (1, 2), (2, 3)])

result = pipeline.generate_dungeon(G, seed=42)
print(len(result.rooms), result.generation_time)
```

## Core APIs

```python
# generate full dungeon from graph
result = pipeline.generate_dungeon(mission_graph=G, seed=42)

# auto-generate topology + dungeon
result = pipeline.generate_dungeon(generate_topology=True, num_rooms=8, seed=42)
```

## Testing

```bash
pytest tests/test_neural_pipeline.py -v
```

## Documentation

- **Documentation Index**: [`docs/INDEX.md`](../../docs/INDEX.md)
- **Architecture & Benchmarks**: [`docs/SOTA_COMPARISON_AND_BENCHMARKS.md`](../../docs/SOTA_COMPARISON_AND_BENCHMARKS.md)
- **Block-by-Block Audit**: [`docs/BLOCK_BY_BLOCK_ARCHITECTURE_AND_IMPLEMENTATION_AUDIT.md`](../../docs/BLOCK_BY_BLOCK_ARCHITECTURE_AND_IMPLEMENTATION_AUDIT.md)
