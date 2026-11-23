# KLTN Architecture Diagrams

This file is a compact, ASCII-safe overview of the repository architecture.
For the canonical implementation walk-through, use
[`docs/CURRENT_ARCHITECTURE.md`](docs/CURRENT_ARCHITECTURE.md).

## Overall System Architecture: G-LMDM (Primary Proposed Model)

In the proposed **G-LMDM** architecture, **Block II (VQ-VAE) is completely eliminated**.
The generative process operates directly on discrete $16\times 11$ tile tokens with absorbing-state diffusion.

```text
User entrypoints (main.py / gui_runner.py / scripts)
            |
            v
  create_pipeline(model_type="masked_room") -> NeuralSymbolicDungeonPipeline
            |
            +--> Stage 1 [Block I]: Evolutionary Topology Generator
            |      - Mission Graph Grammar execution
            |      - (mu + lambda)-ES / MAP-Elites QD search
            |      - Tension curve & leniency/linearity optimization
            |      - Output: Directed Mission Graph G = (V, E, X_V, X_E)
            |
            +--> Stage 2 [Block III]: Multi-Scale Conditioning & Boundary Clamping
            |      - Global: GraphGPS + RRWP (Relative Random Walk Probabilities)
            |      - Local: Incident doorway mask & boundary tile clamping at t=0
            |      - GraphToGridCrossAttention (projects graph nodes to 16x11 grid)
            |
            +--> Stage 3 [G-LMDM Generator]: Direct Discrete Masked Token Transformer (MaskGIT-style)
            |      - Iterative absorbing-state masked generative process
            |      - Direct prediction of 16x11 categorical tile logits
            |      - Zero VQ-VAE / Zero continuous latent space z_q / Zero Gaussian noise
            |      - Strict parameter budget: 84.8K denoiser (3.58M train / 3.36M infer)
            |
            +--> Stage 4 [Block V]: Direct 16x11 Differentiable LogicNet Regularization
            |      - Soft Bellman-Ford value iteration directly on 16x11 walkability logits
            |      - Penalizes corridor cut-sets between doorway anchors during training
            |
            +--> Stage 5 [Block VI]: Global Dungeon Assembly & Attributed Symbolic Repair
            |      - Strict / relaxed spatial room placement on dungeon grid
            |      - Boundary passage realization & door alignment
            |      - Schema 2.0 4-stage attribution: raw -> constrained -> repaired -> compiled
            |
            +--> Stage 6 [Block VII]: Dual-Tier Independent Verification
                   - Exact StateSpaceAStar Oracle (Pareto dominance, formal proofs)
                   - Bounded-Rational P-CBS Proxy (8 procedural personas, cognitive diagnostics)
```

## Comparative Baseline Architecture: H-MOLQD (Continuous Comparator Path)

Retained strictly for experimental comparison against G-LMDM to test Hypothesis $H_2$ (Benchmark Blocks $B_2$–$B_3$):

```text
Mission Graph G
      |
      v
DualStreamConditionEncoder [Block III]
      |
      v
Continuous Latent Diffusion [Block IV] (UNet [1,2] 47.13M total, or DiT+Flow 22.44M total)
      |  (denoises continuous latent representation z in R^{4x3x64})
      v
SemanticVQVAE / SemanticVQVAE2 [Block II] (17.37M params)
      |  (lossy continuous latent decode: argmax z_q -> 16x11 grid)
      v
Attributed Symbolic Repair & Verification [Blocks VI - VII]
```

## Graph-To-Grid Conditioning Detail (Stage 2 / Block III)

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
Data/The Legend of Zelda/     raw and processed VGLC dungeon level data
configs/                      canonical YAML config and ablation overrides
src/                          core models, generation, simulation, pipeline
scripts/                      training, audit, benchmark, and export CLIs
docs/                         active architecture, solver, grammar, and protocol docs
checkpoints/                  optional external/trained model weights
results/                      generated evaluation outputs (untracked)
```

## Suggested Reading Order

1. [`README.md`](README.md)
2. [`docs/INDEX.md`](docs/INDEX.md)
3. [`docs/CURRENT_ARCHITECTURE.md`](docs/CURRENT_ARCHITECTURE.md)
4. [`docs/CANONICAL_MODEL_RATIONALE_ABLATION_AND_COMPLEXITY_GUIDE.md`](docs/CANONICAL_MODEL_RATIONALE_ABLATION_AND_COMPLEXITY_GUIDE.md)
