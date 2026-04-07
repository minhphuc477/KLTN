# Current Architecture

This document captures the current end-to-end architecture in code.
The original dated snapshot is archived at
`docs/archive/2026-q1/CURRENT_ARCHITECTURE_FULL_DRAWING_2026_03_25.md`.
Architecture positioning, comparison boundaries, attention-layer inventory,
and remaining runtime/fallback hard-coded knobs are documented separately in
`docs/ARCHITECTURE_POSITIONING_AND_ATTENTION_NOTES_2026_04_02.md`.

Canonical training now runs through `python main.py train --config configs/zelda_hmolqd.yaml ...`.
`python -m src.train ...` remains available only as a thin compatibility wrapper over the same validated path.
`python -m src.train_vqvae --config configs/zelda_hmolqd.yaml ...` now resolves VQ-VAE stage settings from that same validated config contract.
`python -m src.train_lcm --config configs/zelda_hmolqd.yaml ...` now does the same for the fast-sampler stage.
`python -m src.generate ...` now samples through the canonical `NeuralSymbolicDungeonPipeline` instead of a separate hardcoded inference stack.
Canonical GUI/script generation now also discovers the nearest `resolved_config.yaml/json` snapshot and reuses the validated runtime generation defaults from that config, rather than silently falling back to private helper constants.
Active graph/topology paths now avoid Python's salted `hash(...)` for edge deduplication, MAP-Elites graph caching, and graph-derived per-room seeds.
Graph-guided validation also now normalizes tuple room keys with deterministic dense IDs instead of salted hash-derived room IDs.
The reusable condition-encoder factory still defaults to `gnn_type=gcn` for compatibility, while the canonical diffusion YAML now uses `gnn_type=gps` and the masked-room branch keeps `gnn_type=gcn`.

## 1) Full System Architecture

```mermaid
flowchart TB
    %% =========================
    %% Entry points
    %% =========================
    U1[main.py train canonical config path]
    U2[gui_runner.py interactive path]
    U3[scripts/train_full_and_export_png.py run/export path]

    %% =========================
    %% Data / Assets
    %% =========================
    D0[Data/The Legend of Zelda Processed txt and dot]
    D1[checkpoints directory]

    %% =========================
    %% Core pipeline facade
    %% =========================
    CP[create_pipeline in src/pipeline/dungeon_pipeline.py]
    NP[NeuralSymbolicDungeonPipeline]

    U2 --> CP
    U3 --> CP
    CP --> NP
    D1 --> CP

    %% =========================
    %% Block 0
    %% =========================
    subgraph B0
        B0_label["Block 0 Data Adapter and Stitching Utilities"]
        B0A[VGLCParser and adapters in src/data_processing/data_adapter.py]
        B0B[DungeonStitcher in src/zelda_data/zelda_core.py]
    end

    D0 --> B0A

    %% =========================
    %% Block I
    %% =========================
    subgraph B1
        B1_label["Block I Evolutionary Topology Director"]
        B1A[generate_dungeon generate_topology flag]
        B1B[EvolutionaryTopologyGenerator]
        B1C[GraphGrammarExecutor]
        B1D[Tension and descriptor evaluator]
        B1E[validate_graph_topology]
        B1F[filter_virtual_nodes and get_physical_start_node]

        B1A --> B1B --> B1C --> B1D
        B1B --> B1E
        B1B --> B1F
    end

    NP --> B1A

    %% =========================
    %% Block II
    %% =========================
    subgraph B2
        B2_label["Block II Semantic VQ-VAE"]
        B2A[SemanticVQVAE encode decode]
        B2B[num_classes 44 codebook 256 latent_dim 64]
    end

    NP --> B2A
    B2A --> B2B

    %% =========================
    %% Block III
    %% =========================
    subgraph B3
        B3_label["Block III Dual-Stream Condition Encoder"]
        B3A[DualStreamConditionEncoder]
        B3B[LocalStreamEncoder boundary and neighbors]
        B3C[GlobalStreamEncoder graph GNN plus current-room distance]
        B3D[GraphToGridCrossAttention with distance-aware bias]
        B3F[SpatialGraphConditioner]
        B3E[RoomTopologyConditioner]
    end

    NP --> B3A
    B3A --> B3B
    B3A --> B3C
    B3A --> B3D
    B3A --> B3F

    %% =========================
    %% Block IV
    %% =========================
    subgraph B4
        B4_label["Block IV Latent Diffusion"]
        B4A[LatentDiffusionModel]
        B4B[CFG cfg_scale plus conditional dropout plus Min-SNR]
        B4C[DDIM or DDPM sampling]
        B4D[Topology refinement mode none lightweight gat2]
        B4E[Latent boundary masking and inpaint]
    end

    NP --> B4A
    B4A --> B4B
    B4A --> B4C
    B4A --> B4D
    B4A --> B4E

    %% =========================
    %% Block V
    %% =========================
    subgraph B5
        B5_label["Block V LogicNet Guidance"]
        B5A[LogicNet room-grid plus topology-trace plus anchor losses]
        B5B[GradientGuidance inside diffusion]
    end

    NP --> B5A
    B4A --> B5B
    B5A --> B5B

    %% =========================
    %% Block VI
    %% =========================
    subgraph B6
        B6_label["Block VI Symbolic Refiner"]
        B6A[SymbolicRefiner]
        B6B[Weighted Bayesian WFC]
        B6C[repair_room_with_feedback and local inpaint callbacks]
        B6D[max_repair_attempts 5 margin 2]
    end

    NP --> B6A
    B6A --> B6B
    B6A --> B6C
    B6A --> B6D

    %% =========================
    %% Block VII
    %% =========================
    subgraph B7
        B7_label["Block VII MAP-Elites Quality Diversity"]
        B7A[MAPElitesEvaluator]
        B7B[descriptor_mode hybrid tie_breaker quality_score in pipeline]
        B7C[advanced CVT archive optional]
    end

    NP --> B7A
    B7A --> B7B
    B7A --> B7C

    %% =========================
    %% Outputs
    %% =========================
    O1[RoomGenerationResult per room]
    O2[DungeonGenerationResult stitched grid rooms metrics]
    O3[Exporter outputs npy png txt ids txt vglc]

    B2A --> O1
    B3A --> O1
    B4A --> O1
    B5A --> O1
    B6A --> O1
    B7A --> O2
    B0B --> O2
    O2 --> O3

    U3 --> O3
```

## 2) Runtime Execution Flow (Single Dungeon Generation)

```mermaid
sequenceDiagram
    autonumber
    participant Caller as Caller script or GUI
    participant Facade as create_pipeline
    participant Pipe as NeuralSymbolicDungeonPipeline
    participant Evo as EvolutionaryTopologyGenerator
    participant Cond as ConditionEncoder
    participant Diff as LatentDiffusionModel
    participant Ref as SymbolicRefiner
    participant ME as MAPElitesEvaluator

    Caller->>Facade: create_pipeline checkpoint_dir
    Facade->>Pipe: init with checkpoint paths

    Caller->>Pipe: generate_dungeon params

    alt mission_graph is None and generate_topology true
        Pipe->>Evo: evolve target_curve and search params
        Evo-->>Pipe: mission_graph
        Pipe->>Pipe: validate_graph_topology
    end

    Pipe->>Pipe: filter_virtual_nodes mission_graph_physical
    Pipe->>Pipe: prepare_graph_context tensors

    loop for each room_id in deterministic generation order
        Pipe->>Cond: encode local plus global context
        Pipe->>Diff: sample latent DDIM or DDPM with CFG and optional LogicNet room-topology guidance
        Diff-->>Pipe: z_latent
        Pipe->>Pipe: decode via VQ-VAE logits to neural_grid
        alt apply_repair true
            Pipe->>Ref: repair_room_with_feedback
            Ref-->>Pipe: repaired grid and diagnostics
        end
        Pipe->>Pipe: store RoomGenerationResult and latent cache
    end

    Pipe->>Pipe: stitch rooms into dungeon grid
    alt enable_map_elites true
        Pipe->>ME: add_dungeon and compute descriptors
        ME-->>Pipe: archive and score
    end

    Pipe-->>Caller: DungeonGenerationResult
```

## 3) Strict Mode and Fallback Behavior

```mermaid
flowchart LR
    A[Pipeline strict_checkpoint_mode false by default]
    B[Checkpoint schema mismatch]
    C[Warn and continue with partial load]
    D[Condition encoding failure]
    E[Fallback to zero condition]
    F[Topology validation failure]
    G[Warn and continue]

    H[Pipeline strict_checkpoint_mode true]
    I[Checkpoint schema mismatch]
    J[Raise error fail-fast]
    K[Condition encoding failure]
    L[Raise error fail-fast]
    M[Topology validation failure]
    N[Raise error fail-fast]

    A --> B --> C
    A --> D --> E
    A --> F --> G

    H --> I --> J
    H --> K --> L
    H --> M --> N
```

## 4) Canonical Training and Schema Lock

- Canonical experiment surface: `main.py train` plus the validated YAML/CLI config system in `src/config_system.py`.
- Legacy compatibility entrypoint: `src/train.py` forwards to `main.py train` so there is no second divergent training surface anymore.
- Standalone VQ-VAE compatibility entrypoint: `src/train_vqvae.py` now accepts `--config` and resolves dataset/runtime/VQ-VAE stage settings from the same validated YAML contract used by `main.py train`.
- Standalone fast-sampler compatibility entrypoint: `src/train_lcm.py` now accepts `--config` and resolves dataset/runtime/fast-sampler settings from the same validated YAML contract used by `main.py train`.
- GUI AI generation now routes through `NeuralSymbolicDungeonPipeline.generate_dungeon(...)` rather than a separate pooled-graph shortcut, so the interactive path matches the documented room-wise Block I-VII pipeline.
- Offline generation/evaluation via `src/generate.py` now also routes through the canonical mission-graph-conditioned room-wise pipeline instead of building a separate hardcoded VQ-VAE/diffusion/condition-encoder stack.
- Symbolic repair budgets are now explicit pipeline constructor controls: `symbolic_max_repair_attempts`, `symbolic_repair_margin`, and `symbolic_adjacency_threshold`.
- Explicit dataset lock: `dataset.schema_profile=zelda_v1` makes the current `16x11`, `44`-class, `6/8/8` graph schema contract visible in config and metadata instead of hiding it in validators.
- Current canonical YAML: `configs/zelda_hmolqd.yaml` now uses the reduced small-data-balanced profile (`diffusion.model_channels=96`, `diffusion.condition_hidden_dim=192`, `diffusion.condition_num_gnn_layers=2`) plus a further-downsized masked-room branch (`model_channels=64`, `hidden_dim=48`, `unet_channel_mult=[1,2]`, `unet_num_res_blocks=1`, `unet_num_heads=4`) so the auxiliary discrete branch stays within the Zelda corpus small-data guardrails.
- Runtime seed propagation: standalone diffusion, masked-room, VQ-VAE, and fast-sampler entrypoints now all carry `runtime.seed` through their resolved config objects and apply shared seeding explicitly.
- Stable seed derivation: research/benchmark scripts now use deterministic BLAKE2-based seed offsets instead of Python `hash(...)` when deriving per-method or per-room seeds.
- Runtime guardrails: diffusion and masked-room training now log trainable parameter counts and warn when the configured model looks oversized relative to the available Zelda sample count.
- Composite diffusion checkpoints now reconstruct bundled diffusion, condition-encoder, and LogicNet submodules from embedded config values instead of silently assuming default widths.
- Canonical diffusion training now requires a trained VQ-VAE checkpoint and resolves it from either the just-finished Stage 1 artifact, `diffusion.vqvae_checkpoint`, or `vqvae.checkpoint_dir/vqvae_pretrained.pth`; it no longer has a valid canonical path that silently trains against a random VQ-VAE.
- Random-init pipeline fallbacks also now inherit latent width, context width, and class count from already-bound components where possible, reducing consistency drift when only part of the neural stack is loaded.

## 5) Effective Hyperparameter Layers (Current)

```mermaid
flowchart TB
    HP0[Script-level defaults in train_full_and_export_png]
    HP1[Pipeline-level defaults in generate_dungeon and generate_room]
    HP2[Block model defaults VQ-VAE Condition Encoder Diffusion LogicNet Refiner]
    HP3[Topology search defaults EvolutionaryTopologyGenerator]
    HP4[WFC internal defaults WeightedBayesianWFCConfig]

    HP0 --> HP1 --> HP2
    HP1 --> HP3
    HP1 --> HP4
```

## 6) Component Index (Code Locations)

- Pipeline facade and orchestration: src/pipeline/dungeon_pipeline.py
- Topology evolution: src/generation/evolutionary_director.py
- VQ-VAE: src/core/vqvae.py
- Condition encoder: src/core/condition_encoder.py
- Diffusion: src/core/latent_diffusion.py
- LogicNet: src/core/logic_net.py
- Symbolic refiner and repair: src/core/symbolic_refiner.py
- Weighted Bayesian WFC: src/generation/weighted_bayesian_wfc.py
- MAP-Elites: src/simulation/map_elites.py
- Zelda data and stitching: src/zelda_data/zelda_core.py
- VGLC parsing and adapter: src/data_processing/data_adapter.py
- Export runner: scripts/train_full_and_export_png.py
