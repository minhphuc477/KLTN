# KLTN: Neural-Symbolic Dungeon Generation

**Research implementation of advanced procedural content generation for Legend of Zelda dungeons using evolutionary algorithms and neural-symbolic AI.**

## Overview

KLTN implements a complete research pipeline for generating Legend of Zelda-like dungeon topologies and layouts using cutting-edge AI techniques. The system combines evolutionary search, neural networks, and symbolic reasoning to create playable, solvable dungeons that match target difficulty curves.

## Key Features

- **Evolutionary Topology Generation**: Search-based procedural content generation using genetic algorithms to evolve dungeon graphs that match tension curves
- **Neural-Symbolic Pipeline**: Complete 7-block H-MOLQD architecture integrating VQ-VAE, latent diffusion, LogicNet guidance, and symbolic repair - with full Block I integration for automatic topology generation
- **VGLC Compliance**: Full compliance with Video Game Level Corpus standards for Zelda dungeon structure validation, including composite node labels, Boss-Goal subgraph validation, and centralized graph utilities
- **Interactive GUI**: Real-time visualization and validation environment for dungeon exploration with route export/loading and multi-algorithm pathfinding
- **Comprehensive Testing**: 250+ test functions covering all major components and VGLC compliance requirements

## Architecture

### Block I: Evolutionary Topology Director (Fully Integrated)
- Implements evolutionary search over graph grammars
- Generates dungeon topologies matching target difficulty curves
- Uses (μ+λ)-ES with tournament selection and biased mutation
- Produces NetworkX graphs with VGLC-compliant node and edge attributes
- **Now integrated into main pipeline**: Call `generate_dungeon(generate_topology=True)` to automatically evolve topology

### Block II-VII: Neural-Symbolic Pipeline (H-MOLQD)
- **Block II**: Semantic VQ-VAE for discrete latent representation
- **Block III**: Dual-stream condition encoder for local/global context fusion
- **Block IV**: Latent diffusion with classifier-free guidance
- **Block V**: LogicNet for differentiable solvability constraints
- **Block VI**: Symbolic WaveFunctionCollapse repair for broken paths
- **Block VII**: MAP-Elites quality-diversity evaluation

All 7 blocks are now fully integrated and work seamlessly together.

See [ARCHITECTURE_DIAGRAMS.md](ARCHITECTURE_DIAGRAMS.md) for detailed ASCII diagrams of the system flow.

## Installation

### Requirements
```bash
# Core dependencies
pip install torch>=2.0.0 numpy>=1.24.0 networkx>=3.0 scipy>=1.10.0

# Optional (for full pipeline)
pip install torch-geometric torch-scatter torch-sparse
```

### Quick Setup
```bash
# Clone repository
git clone https://github.com/minhphuc477/KLTN.git
cd KLTN

# Install dependencies
pip install -r requirements-hmolqd.txt
pip install -r requirements-visual.txt

# Run basic validation
python main.py --dungeon 1 --variant 1
```

## Usage

### Basic Dungeon Validation
```bash
# Validate single dungeon
python main.py --dungeon 1 --variant 1

# Validate all dungeons with GUI
python main.py --all --gui

# Export processed data
python main.py --dungeon 1 --export dungeon_data.npz
```

### Interactive GUI
```bash
# Launch visualization environment
python gui_runner.py

# Controls:
# Arrow keys: Manual movement
# SPACE: Auto-solve with A*
# R: Reset dungeon
# H: Toggle heatmap overlay
# ESC: Exit
```

### Neural Pipeline Generation
```python
from src.pipeline import create_pipeline
import networkx as nx

# Initialize pipeline
pipeline = create_pipeline(checkpoint_dir="./checkpoints")

# Option 1: Generate with automatic topology evolution (all 7 blocks)
result = pipeline.generate_dungeon(
    generate_topology=True,
    target_curve=[0.2, 0.5, 0.8, 1.0],
    num_rooms=8,
    seed=42
)
print(f"Generated {len(result.rooms)} rooms in {result.generation_time:.2f}s")

# Option 2: Provide pre-made mission graph (Blocks II-VII only)
G = nx.DiGraph()
G.add_nodes_from([0, 1, 2, 3])
G.add_edges_from([(0, 1), (1, 2), (2, 3)])
result = pipeline.generate_dungeon(G, seed=42)
print(f"Generated {len(result.rooms)} rooms in {result.generation_time:.2f}s")
```

### Evolutionary Topology Search
```python
from src.generation.evolutionary_director import EvolutionaryTopologyGenerator

# Initialize evolutionary search
gen = EvolutionaryTopologyGenerator(
    target_curve=[0.2, 0.5, 0.8, 1.0],
    population_size=50,
    generations=100,
    seed=42
)

# Evolve optimal topology
dungeon_graph = gen.evolve()
print(f"Evolved topology with {dungeon_graph.number_of_nodes()} rooms")
```

## Project Structure

```
KLTN/
├── src/
│   ├── generation/          # Evolutionary topology director
│   ├── pipeline/            # Neural-symbolic pipeline (7 blocks)
│   ├── core/                # Neural network components
│   ├── data/                # VGLC data processing and compliance
│   ├── simulation/          # Dungeon validation and solving
│   ├── utils/               # Graph utilities and training helpers
│   └── visualization/       # Plotting and analysis tools
├── tests/                   # Comprehensive test suite
├── examples/                # Usage examples and demos
├── docs/                    # Research documentation
├── Data/                    # Zelda dungeon dataset
├── checkpoints/             # Trained model weights
└── results/                 # Experiment outputs and analysis
```

## Research Components

### Evolutionary Search
- **Algorithm**: (μ+λ)-Evolution Strategy with tournament selection
- **Representation**: Grammar rule sequences (genotype) → NetworkX graphs (phenotype)
- **Fitness**: Tension curve matching + solvability constraints
- **Mutation**: Biased operators following Zelda design patterns

### Neural Architecture
- **VQ-VAE**: Semantic latent space for 16×11 room layouts
- **Diffusion**: Guided latent diffusion with LogicNet constraints
- **LogicNet**: Differentiable pathfinding for solvability optimization
- **Symbolic Repair**: WaveFunctionCollapse for constraint satisfaction

### VGLC Compliance
- **Dimensions**: Strict 11×16 room layouts (non-square)
- **Topology**: Boss-Goal subgraph patterns with virtual node filtering
- **Labels**: Composite node types (e.g., "enemy,key,puzzle")
- **Validation**: 36 comprehensive compliance tests

## Testing

```bash
# Run full test suite
pytest tests/ -v

# Run specific component tests
pytest tests/test_vglc_compliance.py -v
pytest tests/test_neural_pipeline.py -v

# Quick validation
python scripts/quick_validation_test.py --verbose
```

## Documentation

- **API Reference**: `docs/NEURAL_PIPELINE_API.md`
- **Research Report**: `docs/NEURAL_PIPELINE_RESEARCH.md`
- **Implementation**: `docs/NEURAL_PIPELINE_IMPLEMENTATION.md`
- **VGLC Analysis**: `VGLC_DATA_ANALYSIS_REPORT.md`

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{kltn2026neural,
  title={Neural-Symbolic Dungeon Generation with H-MOLQD},
  author={Le Tran Minh Phuc},
  year={2026},
  howpublished={\url{https://github.com/minhphuc477/KLTN}}
}
```

## License

See LICENSE file for details.

## Contact

For questions or collaboration opportunities, please open an issue on GitHub.