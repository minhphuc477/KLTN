# KLTN Architecture Diagrams

This file is a compact, ASCII-safe overview of the repository architecture.
For the canonical implementation walk-through, use
[`docs/CURRENT_ARCHITECTURE.md`](docs/CURRENT_ARCHITECTURE.md).

## Overall System Architecture

```text
User entrypoints
  main.py / gui_runner.py / training scripts
            |
            v
  create_pipeline(...) -> NeuralSymbolicDungeonPipeline
            |
            +--> Block I: Evolutionary topology generation
            |      - mission grammar
            |      - search / scoring / topology validation
            |
            +--> Blocks II-VII: Room generation and repair
                   - VQ-VAE latent representation
                   - condition encoding
                   - latent diffusion
                   - LogicNet guidance
                   - symbolic repair
                   - quality-diversity evaluation
```

## Block I: Topology Search

```text
Target curve / room count / seed
            |
            v
EvolutionaryTopologyGenerator
  -> graph grammar execution
  -> descriptor + tension scoring
  -> topology validation
  -> best mission graph
```

## Blocks II-VII: Neural-Symbolic Room Stack

```text
Mission graph + room constraints
            |
            v
DualStreamConditionEncoder
  -> local room context
  -> global mission graph context
  -> graph-to-grid spatial conditioning
            |
            v
LatentDiffusionModel
  -> CFG / DDIM / DDPM
  -> optional graph-aware topology refinement
  -> LogicNet guidance
            |
            v
SemanticVQVAE decode
            |
            v
SymbolicRefiner / WFC repair
            |
            v
Playable room layouts + diagnostics
```

## Graph-To-Grid Conditioning Detail

```text
Graph nodes
  + node positions
  + topology features
  + edge_index
        |
        v
GraphToGridCrossAttention
  -> node structural encoding
  -> optional graph message passing prepass
  -> per-grid-position cross-attention
        |
        v
SpatialGraphConditioner
  -> room topology bias
  -> gated graph conditioning
  -> conditioned U-Net features
```

## Main Data Assets

```text
Data/The Legend of Zelda/     raw and processed level data
checkpoints/                  trained weights and metadata
artifacts/                    experiment outputs and reports
docs/                         current reference docs
docs/archive/                 dated snapshot notes
```

## Suggested Reading Order

1. [`README.md`](README.md)
2. [`docs/INDEX.md`](docs/INDEX.md)
3. [`docs/CURRENT_ARCHITECTURE.md`](docs/CURRENT_ARCHITECTURE.md)
4. [`docs/BLOCK_BY_BLOCK_ARCHITECTURE_AND_IMPLEMENTATION_AUDIT.md`](docs/BLOCK_BY_BLOCK_ARCHITECTURE_AND_IMPLEMENTATION_AUDIT.md)
