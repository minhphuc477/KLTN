# KLTN Architecture Diagrams

## Overall System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           KLTN: Neural-Symbolic Dungeon Generation           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   Mission Graph │    │   Evolutionary  │    │   Neural-Sym.   │         │
│  │   (NetworkX)    │───▶│   Topology Dir. │───▶│   Pipeline      │         │
│  │                 │    │   (Block I)     │    │   (Blocks II-VII)│         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│                                                                             │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   VGLC Dataset  │    │   Validation    │    │   GUI Runner    │         │
│  │   (Ground Truth)│───▶│   & Testing     │───▶│   (Visualization)│         │
│  │                 │    │                 │    │                 │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Evolutionary Search Flow (Block I)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Evolutionary Topology Director                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐                                                        │
│  │   Target Curve  │                                                        │
│  │   [0.2,0.5,0.8] │                                                        │
│  └─────────────────┘                                                        │
│                                                                             │
│         │                                                                   │
│         ▼                                                                   │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   Population    │    │   Evaluation    │    │   Selection     │         │
│  │   Genotypes     │───▶│   (Fitness)     │───▶│   (Tournament)   │         │
│  │   List[int]     │    │   Curve Match   │    │   (μ+λ)-ES      │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│                                                                             │
│         │                                                                   │
│         ▼                                                                   │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   Crossover     │    │   Mutation      │    │   Next Gen      │         │
│  │   (1-point)     │───▶│   (Biased)      │───▶│   Population     │         │
│  │                 │    │   Zelda Rules   │    │                 │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│                                                                             │
│         │                                                                   │
│         ▼                                                                   │
│                                                                             │
│  ┌─────────────────┐                                                        │
│  │   Best Graph    │                                                        │
│  │   NetworkX      │                                                        │
│  │   (VGLC Comp.)  │                                                        │
│  └─────────────────┘                                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Genotype: List[int] (rule sequence)
Phenotype: NetworkX Graph (dungeon topology)
Fitness: Float 0.0-1.0 (curve matching + solvability)
```

## Neural-Symbolic Pipeline (Blocks II-VII)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 H-MOLQD Neural-Symbolic Pipeline                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐                                                        │
│  │   Mission Graph │                                                        │
│  │   (NetworkX)    │                                                        │
│  └─────────────────┘                                                        │
│                                                                             │
│         │                                                                   │
│         ▼                                                                   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                           Block II: VQ-VAE                              │ │
│  │                    Semantic Latent Encoding                             │ │
│  │                                                                         │ │
│  │  Room Grid (16×11) ──▶ Encoder ──▶ Quantize ──▶ Latent (64×4×3)        │ │
│  │  Discrete Tiles        ↓              ↓              ↓                  │ │
│  │                        Decoder ◀─── Dequantize ◀─── Reconstruction      │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│         │                                                                   │
│         ▼                                                                   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         Block III: Condition Encoder                    │ │
│  │                   Dual-Stream Context Conditioning                       │ │
│  │                                                                         │ │
│  │  Neighbor Latents ──┐                                                   │ │
│  │  (N/S/E/W)          │                                                   │ │
│  │                     ├──▶ Local Context ──┐                              │ │
│  │  Graph Context      │                    │                              │ │
│  │  (Node/Edge/TPE)    └────────────────────┼──▶ Fusion ──▶ Condition (256)│ │
│  │                                          │                              │ │
│  │                                          └────────────▶ Global Context  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│         │                                                                   │
│         ▼                                                                   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         Block IV: Latent Diffusion                      │ │
│  │                   Guided Generation with LogicNet                        │ │
│  │                                                                         │ │
│  │  Noise Latent ──▶ Diffusion Steps ──▶ Guidance ──▶ Denoised Latent     │ │
│  │  (64×4×3)         (1000 steps)        (CFG + Logic)   (64×4×3)          │ │
│  │                                                                         │ │
│  │  Condition (256) ──────────────────────────────────────────────────────▶ │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│         │                                                                   │
│         ▼                                                                   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                          Block V: LogicNet                              │ │
│  │                  Differentiable Pathfinding Guidance                    │ │
│  │                                                                         │ │
│  │  Latent ──▶ VQ-VAE Decode ──▶ Tile Probs ──▶ Pathfinding ──▶ Loss       │ │
│  │  (64×4×3)   (44×16×11)       (44×16×11)   (A* Search)    (Scalar)       │ │
│  │                                                                         │ │
│  │  Gradient ◀───────────────────────────────────────────────────────────── │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│         │                                                                   │
│         ▼                                                                   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         Block VI: Symbolic Refiner                      │ │
│  │                    WaveFunctionCollapse Repair                           │ │
│  │                                                                         │ │
│  │  Neural Grid ──▶ Constraint Check ──▶ WFC Repair ──▶ Final Grid         │ │
│  │  (16×11)         (Path Validation)    (Tile Rules)    (16×11)           │ │
│  │                                                                         │ │
│  │  Start/Goal ───────────────────────────────────────────────────────────▶ │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│         │                                                                   │
│         ▼                                                                   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                        Block VII: MAP-Elites                           │ │
│  │                   Quality-Diversity Evaluation                          │ │
│  │                                                                         │ │
│  │  Dungeon ──▶ Solvability ──▶ Metrics ──▶ Archive ──▶ QD Score           │ │
│  │  (Grid)      (A* Search)    (Leniency)   (Grid)      (Dict)              │ │
│  │                           (Linearity)                                    │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│         │                                                                   │
│         ▼                                                                   │
│                                                                             │
│  ┌─────────────────┐                                                        │
│  │   Final Dungeon │                                                        │
│  │   (Stitched)    │                                                        │
│  │   VGLC Comp.    │                                                        │
│  └─────────────────┘                                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow and Integration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        KLTN Data Flow Architecture                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐                                                        │
│  │   VGLC Dataset  │                                                        │
│  │   (Ground Truth │                                                        │
│  │    Zelda Data)  │                                                        │
│  └─────────────────┘                                                        │
│                                                                             │
│         │                                                                   │
│         ▼                                                                   │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   Data Adapter  │    │   Graph Parser  │    │   Validation    │         │
│  │   (zelda_core)  │───▶│   (vglc_utils)  │───▶│   (36 tests)    │         │
│  │   Load/Stitch   │    │   Virtual Nodes  │    │   Compliance    │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│                                                                             │
│         │                                                                   │
│         ▼                                                                   │
│                                                                             │
│  ┌─────────────────┐                                                        │
│  │   Mission Graph │                                                        │
│  │   NetworkX      │                                                        │
│  │   (Topology)    │                                                        │
│  └─────────────────┘                                                        │
│                                                                             │
│         ├─────────────────────┬─────────────────────────────────────────────┤ │
│         │                     │                                             │
│         ▼                     ▼                                             │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐                                │
│  │   Evolutionary  │    │   Neural-Sym.   │                                │
│  │   Search        │    │   Pipeline      │                                │
│  │   (Block I)     │    │   (Blocks II-VII│                                │
│  │   Genotype→     │    │   Graph→Room    │                                │
│  │   Phenotype     │    │   Generation    │                                │
│  └─────────────────┘    └─────────────────┘                                │
│                                                                             │
│         │                     │                                             │
│         └─────────────────────┼─────────────────────────────────────────────┤ │
│                               │                                             │
│         ┌─────────────────────▼─────────────────────────────────────────────┐ │
│         │                    Dungeon Generation                             │ │
│         │                    (Stitched Layout)                             │ │
│         └───────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│         │                                                                   │
│         ▼                                                                   │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   Validation    │    │   GUI Runner    │    │   Export        │         │
│  │   (Solver)      │───▶│   (Visualization│───▶│   (NPZ/JSON)    │         │
│  │   Solvability   │    │    Heatmaps)    │    │   Results       │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Interaction Details

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Component Interaction Flow                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐                                                        │
│  │   User Input    │                                                        │
│  │   (Config/Seed) │                                                        │
│  └─────────────────┘                                                        │
│                                                                             │
│         │                                                                   │
│         ▼                                                                   │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   Config Parser │    │   Seed Manager  │    │   Device Setup  │         │
│  │   (JSON/YAML)   │───▶│   (Reproducible)│───▶│   (CPU/CUDA)    │         │
│  │                 │    │                 │    │                 │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│                                                                             │
│         │                                                                   │
│         ▼                                                                   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                        Pipeline Orchestrator                            │ │
│  │                                                                         │ │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │ │
│  │  │  Topology   │    │   Room Gen  │    │   Stitching │    │ Validation  │ │ │
│  │  │  Director   │───▶│   (Per Room)│───▶│   (Layout)  │───▶│   (Solver)  │ │ │
│  │  │             │    │             │    │             │    │             │ │ │
│  │  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│         │                                                                   │
│         ▼                                                                   │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   Metrics       │    │   Checkpoint   │    │   Results       │         │
│  │   Collection    │───▶│   Management   │───▶│   Export        │         │
│  │   (QD Scores)   │    │   (Model Save) │    │   (Analysis)    │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│                                                                             │
│         │                                                                   │
│         ▼                                                                   │
│                                                                             │
│  ┌─────────────────┐                                                        │
│  │   Final Output  │                                                        │
│  │   (Dungeon +    │                                                        │
│  │    Statistics)  │                                                        │
│  └─────────────────┘                                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Data Structures

```
RoomGenerationResult:
├── room_id: int
├── room_grid: np.ndarray (16, 11) - discrete tiles
├── latent: torch.Tensor (1, 64, 4, 3) - VQ-VAE latent
├── neural_grid: np.ndarray (16, 11) - before repair
├── was_repaired: bool
├── repair_mask: Optional[np.ndarray] (16, 11)
└── metrics: Dict[str, float]

DungeonGenerationResult:
├── dungeon_grid: np.ndarray (H, W) - stitched layout
├── rooms: Dict[int, RoomGenerationResult]
├── mission_graph: nx.Graph - topology
├── metrics: Dict[str, Any] - generation stats
├── map_elites_score: Optional[Dict[str, float]]
└── generation_time: float
```

## Evolutionary Algorithm Details

```
Individual:
├── genotype: List[int] - rule sequence
├── phenotype: nx.Graph - dungeon topology
├── fitness: float - curve matching score
└── age: int - generations survived

Evolution Process:
├── Population Size: μ + λ (typically 50 + 100)
├── Selection: Tournament (size 3)
├── Crossover: 1-point with probability 0.8
├── Mutation: Biased (Zelda transition matrix)
├── Elitism: Best μ individuals survive
└── Termination: Max generations or fitness plateau
```

## Neural Network Specifications

```
SemanticVQVAE:
├── Input: (44, 16, 11) - semantic tile channels
├── Latent: (64, 4, 3) - compressed representation
├── Codebook: 512 vectors
├── Reconstruction: MSE + VQ loss
└── Architecture: ResNet-style encoder/decoder

LatentDiffusionModel:
├── Latent Dim: 64
├── Condition Dim: 256
├── Timesteps: 1000
├── Noise Schedule: Cosine
├── Guidance: Classifier-free + LogicNet
└── Sampling: DDIM (50-200 steps)

LogicNet:
├── Latent Input: (64, 4, 3)
├── Tile Output: (44, 16, 11)
├── Iterations: 20 differentiable A*
├── Loss: Pathfinding constraint violation
└── Gradient: Backprop through search
```

## VGLC Compliance Layers

```
Validation Hierarchy:
├── Room Dimensions: 11×16 (non-square)
├── Virtual Nodes: Filtered before layout
├── Composite Labels: enemy,key,puzzle patterns
├── Edge Types: key_locked, bombable, open
├── Boss-Goal Pattern: Pre-Boss → Boss → Goal
├── Path Length: Min 3 rooms (start→goal)
└── Topology: Connected, no cycles in goal path
```