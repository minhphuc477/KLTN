# KLTN: Neural-Symbolic Dungeon Generation

**Research implementation of advanced procedural content generation for Legend of Zelda dungeons using evolutionary algorithms and neural-symbolic AI.**

## Overview

KLTN implements a complete research pipeline for generating Legend of Zelda-like dungeon topologies and layouts using cutting-edge AI techniques. The system combines evolutionary search, neural networks, and symbolic reasoning to create playable, solvable dungeons that match target difficulty curves.

## Key Features

- **Evolutionary Topology Generation**: Search-based procedural content generation using genetic algorithms to evolve dungeon graphs that match tension curves
- **Neural-Symbolic Pipeline**: Complete 7-block H-MOLQD architecture integrating VQ-VAE, latent diffusion, LogicNet guidance, and symbolic repair - with full Block I integration for automatic topology generation
- **VGLC Compliance**: Full compliance with Video Game Level Corpus standards for Zelda dungeon structure validation, including composite node labels, Boss-Goal subgraph validation, and centralized graph utilities
- **Interactive GUI**: Real-time visualization and validation environment for dungeon exploration with route export/loading and multi-algorithm pathfinding
- **Comprehensive Testing**: 490+ test functions covering major components and VGLC compliance requirements

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

# Deterministic run (also supported via KLTN_SEED environment variable)
python main.py --dungeon 1 --variant 1 --seed 42

# Validate all dungeons with GUI
python main.py --all --gui

# Export processed data
python main.py --dungeon 1 --export dungeon_data.npz
```

### Interactive GUI
```bash
# Launch visualization environment
python gui_runner.py

# Optional deterministic GUI AI generation seed
set KLTN_AI_SEED=42

# Optional checkpoint override for GUI AI generation
set KLTN_CHECKPOINT_PATH=checkpoints\final_model.pth

# Optional hardening: require checkpoint metadata sidecars for GUI AI generation
set KLTN_STRICT_CHECKPOINTS=1

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

# Optional hardening: fail fast when checkpoints/metadata are incompatible
strict_pipeline = create_pipeline(
  checkpoint_dir="./checkpoints",
  strict_checkpoint_mode=True,
)

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
│   ├── zelda_data/          # VGLC data processing and compliance
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

## Testing

```bash
# Run full test suite
python -m pytest tests/ -v

# Run specific component tests
python -m pytest tests/test_vglc_compliance.py -v
python -m pytest tests/test_neural_pipeline.py -v

# Focused validation
python -m pytest tests/test_topology_generation_fixes.py -q
python -m pytest tests/test_vglc_compliance.py -q
```

## Training (staged)

```bash
# Full staged training (VQ-VAE then diffusion)
python -m src.train --stage all --data-dir "Data/The Legend of Zelda"

# Only VQ-VAE pretraining
python -m src.train --stage vqvae --epochs-vqvae 300

# VQ-VAE with CoordConv + differentiable illegal adjacency penalty
python -m src.train_vqvae --data-dir "Data/The Legend of Zelda" --seed 42 --use-coordconv --mrf-penalty-weight 0.05

# Only diffusion (with pretrained VQ-VAE)
python -m src.train --stage diffusion --vqvae-checkpoint checkpoints/vqvae_pretrained.pth

# Diffusion with graph-node token conditioning for U-Net cross-attention
python -m src.train_diffusion --data-dir "Data/The Legend of Zelda" --graph-conditioning-mode node_sequence --condition-gnn-type gcn

# Training now writes sidecar metadata (*.meta.json) next to checkpoints,
# which can be validated when strict checkpoint mode is enabled in the pipeline.
```

## Topology tuning workflow

```bash
# Sweep realism coefficients and export ranked results
python scripts/sweep_block_i_realism_tuning.py --num-generated 12 --seed 42

# Matched-budget topology benchmark with preflight config validation
python scripts/run_matched_budget_topology_benchmark.py --num-samples 12 --seed 42

# Benchmark defaults to the recommended gate_quality_heavy realism profile
python -m src.evaluation.benchmark_suite --num-generated 12

# Run benchmark with an explicit override profile
python -m src.evaluation.benchmark_suite --num-generated 12 --realism-tuning-json '{"adapt_edge_density_gain":0.62,"adapt_edge_budget_gain":0.44}'

# Run benchmark with named profile (recommended)
python -m src.evaluation.benchmark_suite --num-generated 12 --realism-profile gate_quality_heavy

# Run weighted side-by-side profile recommendation (writes ranked summary)
python scripts/recommend_realism_profile.py --num-generated 16 --output-dir results/profile_recommendation

# Windows-safe alternative: put overrides in a JSON file
python -m src.evaluation.benchmark_suite --num-generated 12 --realism-tuning-file results/genome_len_override_moderate.json

# Sequence-break analysis (critical-path gate bypass diagnostics)
python scripts/analyze_sequence_breaks.py --num-samples 8 --output results/sequence_break_analysis.json

# Per-rule marginal fitness credit (leave-one-out ablation on best genome)
python scripts/analyze_rule_marginal_credit.py --output results/rule_marginal_credit.json

# Paired cognitive objective A/B (off vs on) with per-seed deltas
python scripts/run_cognitive_objective_ab.py --num-samples 12 --seed 42 --output results/cognitive_objective_ab

# One-command P0/P1/P2(+others) consolidated suite
python scripts/run_priority_research_suite.py --output-dir results/priority_research_suite

# Quick preview without execution (safe for checking selected steps)
python scripts/run_priority_research_suite.py --steps sequence_break_analysis,rule_marginal_credit --dry-run

# Run only one priority bucket (p0 | p1 | p2 | others), bounded quick profile
python scripts/run_priority_research_suite.py --priority p2 --quick --step-timeout-sec 300 --output-dir results/priority_research_suite_p2_quick

# Heavier research profile (includes 100-sample feature distribution when selected)
python scripts/run_priority_research_suite.py --priority p2 --full-research --output-dir results/priority_research_suite_p2_full
```

## Defense Evidence Workflow

Use this checklist to pre-empt common reviewer attacks with reproducible artifacts.

```bash
# Attack 1 (Topology realism): verify cycle and soft-gate rules exist
python scripts/test_grammar_rules.py

# Attack 2 (Evidence): run fixed-seed ablations and benchmark outputs
python scripts/run_ablation_study.py --num-samples 50 --output results/ablation
python -m src.evaluation.benchmark_suite --num-generated 100

# Attack 2 (Expressive range): export distribution summaries/plots
python scripts/analyze_block_i_feature_distribution.py --num-generated 1000 --output-dir results/feature_distribution

# Attack 3 (Renderable/exportable output): GUI exports route JSON and topology DOT
python gui_runner.py
```

Expected evidence artifacts:
- `results/ablation/` (component necessity and significance)
- `results/feature_distribution/` (expressive range CSV/plots)
- `exports/routes/*.json` and `exports/topology/*.dot` (presentation-ready exports)

## Documentation

- **Start Here (single docs entrypoint)**: `docs/INDEX.md`
- **Docs Folder Landing**: `docs/README.md`
- **Architecture & Benchmarks**: `docs/SOTA_COMPARISON_AND_BENCHMARKS.md`
- **Block-by-Block Audit**: `docs/BLOCK_BY_BLOCK_ARCHITECTURE_AND_IMPLEMENTATION_AUDIT.md`
- **Paper Blueprint + Room Block**: `docs/IEEE_TOG_BLUEPRINT_AND_ROOM_GENERATION.md`
- **Topology Rubric**: `docs/TOPOLOGY_STACK_EVALUATION_RUBRIC_2026_03_08.md`
- **Solver + GUI Reference**: `docs/SOLVERS_AND_GUI_REFERENCE.md`
- **VGLC Compliance Guide**: `docs/VGLC_COMPLIANCE_GUIDE.md`

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{kltn2026neural,
  title={Neural-Symbolic Dungeon Generation with H-MOLQD},
  year={2026},
  howpublished={\url{https://github.com/minhphuc477/KLTN}}
}
```

## License

See LICENSE file for details.

## Contact

For questions or collaboration opportunities, please open an issue on GitHub.
