# H-MOLQD Block I/O Reference

> **Complete input/output specification for all 7 blocks and key algorithms.**
> Generated after integration audit and verification (all blocks tested end-to-end).
> Verified against source code (all class names, signatures, and shapes confirmed).

---

## Architecture Overview

```
VGLC Files ──→ [Block I] ──→ Rooms + Graph
                                 │
                   ┌─────────────┼─────────────┐
                   ↓             ↓             ↓
             [Block II]    [Block III]    [Block V]
             VQ-VAE        CondEncoder   LogicNet
                │             │             │
                ↓             ↓             ↓
             Latent z      Context c    Logic Loss
                │             │
                └──────┬──────┘
                       ↓
                  [Block IV]
              Latent Diffusion
                       │
                       ↓
                Generated z_gen
                       │
               ┌───────┴───────┐
               ↓               ↓
          [Block II]      [Block VII]
           Decode        SymbolicRefiner
               │               │
               └───────┬───────┘
                       ↓
               Refined Room Grid
                       │
                       ↓
                  [Block VI]
                  MAP-Elites
                       │
                       ↓
              Diverse Elite Archive
```

---

## Block I — Data Adapter

**File:** `src/data_processing/data_adapter.py` (1238 lines)
**Purpose:** Parse VGLC text files into structured tensors and mission graphs.

### Key Classes

| Class | Line | Purpose |
|-------|------|---------|
| `VGLCParser` | — | Parses VGLC `.txt` files → character grids |
| `GraphvizParser` | 375 | Parses `.dot` topology files → `nx.DiGraph` |
| `PhaseAligner` | 506 | Auto-aligns room boundaries for proper stitching |
| `GraphFingerprinter` | 626 | Matches rooms to graph nodes via content matching |
| `MLFeatureExtractor` | 845 | Computes TPE, node features, P-matrix |
| `IntelligentDataAdapter` | 1004 | **Main orchestrator** — loads, parses, aligns all data |

> **Note:** The main entry point is `IntelligentDataAdapter`, not `DataAdapter`.

#### `VGLCParser`
| | Description |
|---|---|
| **Input** | VGLC `.txt` file path (Zelda dungeon layout, H=11 rows × W=16 columns per room) |
| **Output** | `np.ndarray` character grid |
| **Key Methods** | `load_grid(filepath) → np.ndarray`, `extract_rooms(grid) → Dict[(r,c), np.ndarray]` |

#### `IntelligentDataAdapter` (main entry point)
| | Description |
|---|---|
| **Input** | `data_dir: str` — directory containing VGLC text + `.dot` graph files |
| **Output** | `List[DungeonTensor]` — fully aligned rooms + graphs |
| **Key Methods** | `process_all() → List[DungeonTensor]`, `process_single(txt_path, dot_path) → DungeonTensor` |

### Data Structures

#### `RoomTensor`
```
Fields:
  room_id       : int               — Unique ID (row*100 + col)
  position      : (int, int)        — Grid position (row, col)
  semantic_grid : np.ndarray [H=11, W=16] int   — Tile IDs (0–43)
  char_grid     : np.ndarray [H=11, W=16] str   — Original characters
  graph_node_id : Optional[int]     — Matched graph node
  contents      : List[str]         — Items/entities in room
  doors         : Dict[str, str]    — {direction: door_type}
  features      : Dict[str, Any]    — Additional features

Methods:
  to_tensor(num_classes=44) → np.ndarray [H, W, C]  — One-hot encoded
```

#### `DungeonTensor`
```
Fields:
  dungeon_id     : str
  rooms          : Dict[(row,col), RoomTensor]
  graph          : nx.DiGraph           — Mission graph
  layout_grid    : np.ndarray [R, C]    — Room positions
  global_semantic: Optional[np.ndarray] — Stitched semantic grid (if computed)
  tpe_vectors    : np.ndarray [N, 8]    — Topological Positional Encoding
  p_matrix       : np.ndarray [N, N, 3] — Dependency matrix
  node_features  : np.ndarray [N, 5]    — Node feature vectors
  node_to_room   : Dict[int, (row,col)]

Methods:
  get_room_tensors(num_classes=44) → np.ndarray [N, H, W, C]
  num_rooms → int
  num_nodes → int
```

### Tile Vocabulary (44 classes)

Defined in `src/core/definitions.py` via `TileID` IntEnum and `SEMANTIC_PALETTE`.

| ID | Meaning | ID | Meaning |
|----|---------|-----|---------|
| 0 | Void (outside map) | 1 | Floor (walkable) |
| 2 | Wall (impassable) | 3 | Block (pushable) |
| 10 | Door (open) | 11 | Door (locked) |
| 12 | Door (bomb) | 13 | Door (puzzle) |
| 14 | Door (boss) | 15 | Door (soft/one-way) |
| 20 | Enemy | 21 | Start position |
| 22 | Triforce/Goal | 23 | Boss enemy |
| 30 | Key (small) | 31 | Key (boss) |
| 32 | Key (item) | 33 | Item (minor) |
| 40 | Element (hazard) | 41 | Element floor |
| 42 | Stair/warp | 43 | Puzzle element |

---

## Block II — Semantic VQ-VAE

**File:** `src/core/vqvae.py` (858 lines)
**Purpose:** Compress room layouts into discrete latent codes.

### `SemanticVQVAE`

| Method | Input | Output |
|--------|-------|--------|
| `encode(x)` | `x: [B, C=44, H=11, W=16]` | `(z_q: [B, D, H', W'], indices: [B, H', W'])` |
| `decode(z_q, target_size)` | `z_q: [B, D, H', W']`, `target_size: (H, W)` (opt) | `recon: [B, C=44, H, W]` |
| `decode_indices(indices)` | `indices: [B, H', W']` | `recon: [B, C=44, H, W]` |
| `forward(x)` | `x: [B, C=44, H=11, W=16]` | `(recon, indices, losses_dict)` |
| `compute_loss(x, recon, vq_losses)` | tensors + dict | `Dict` w/ `total_loss`, `recon_loss`, `vq_loss`, `commit_loss`, `perplexity` |
| `get_codebook_usage()` | — | `Dict` w/ `total`, `used`, `utilization`, `dead_codes` |
| `reset_dead_codes()` | — | None (resets unused codebook entries) |

> **Important:** `indices` shape is `[B, H', W']` (spatial), NOT `[B, H'×W']` (flattened).

### Constructor
```python
SemanticVQVAE(
    num_classes: int = 44,          # Tile vocabulary size (= input channels)
    codebook_size: int = 512,       # K — number of codebook entries
    latent_dim: int = 64,           # D — latent channel dimension
    hidden_dim: int = 128,          # Encoder/decoder hidden channels
    num_res_blocks: int = 2,        # ResBlocks per encoder/decoder stage
    commitment_cost: float = 0.25,  # β for commitment loss
    rare_tile_weight: float = 5.0,  # Loss multiplier for rare tiles
    use_ema: bool = True,           # Use EMA codebook updates
)
```

### Factory
```python
create_vqvae(
    num_classes: int = 44,
    codebook_size: int = 512,
    latent_dim: int = 64,
    **kwargs,  # Forwarded to SemanticVQVAE constructor
) → SemanticVQVAE
```

### Key Parameters
- **Codebook size K:** 512 (default). Each spatial position maps to one of K codes.
- **Latent dim D:** 64. Channel dimension of latent space.
- **Spatial reduction:** Input `[B, 44, 11, 16]` → Latent `[B, 64, H', W']` where `H' ≈ H/4, W' ≈ W/4`.

### Internal Components
| Component | Purpose |
|-----------|---------|
| `Encoder` | `in_channels=num_classes(44)` → ResBlocks → Downsample (channel_mult `(1,2,4)`) |
| `VectorQuantizer` | Codebook lookup, EMA updates, dead code reset (Phase 1B) |
| `Decoder` | Upsample → ResBlocks → `out_channels=num_classes(44)` (channel_mult `(4,2,1)`) |
| `ResBlock` | Residual block with GroupNorm + SiLU |

### Data Format Note

The data loader (`ZeldaDungeonDataset`) returns `[1, H, W]` normalized tile IDs.
The training pipeline converts this to `[C=44, H, W]` one-hot before calling `encode()`:
```python
# In DiffusionTrainer.encode_to_latent():
tile_ids = (x.squeeze(1) * 43).round().long().clamp(0, 43)
x_onehot = F.one_hot(tile_ids, num_classes=44).permute(0, 3, 1, 2).float()
z_q, indices = vqvae.encode(x_onehot)
```

---

## Block III — Dual-Stream Condition Encoder

**File:** `src/core/condition_encoder.py` (801 lines)
**Purpose:** Encode local (neighbor) + global (graph) context for conditioning the diffusion model.

### `DualStreamConditionEncoder`

| Method | Input | Output |
|--------|-------|--------|
| `forward(...)` | See below | `c: Tensor [B, output_dim]` (fused conditioning) |
| `encode_local_only(...)` | `neighbor_latents, boundary_constraints, position` | `c_local: Tensor` |
| `encode_global_only(...)` | `node_features, edge_index, [edge_features], [tpe]` | `c_global: Tensor [N, output_dim]` |

> **Note:** `forward()` returns `[B, output_dim]`, NOT `[B, T, output_dim]`.
> The `forward()` signature does NOT include a `z` parameter.

#### `encode_global_only` Parameters (most-used path)
```
node_features : Tensor [N, 5]           — Per-node features
edge_index    : Tensor [2, E]           — Edge connectivity
edge_features : Tensor [E, edge_dim]    — (Optional) Edge type features (Phase 3A: GATv2Conv)
tpe           : Tensor [N, 8]           — (Optional) Topological Positional Encoding

Returns: Tensor [N, output_dim]
```

#### `forward` Full Parameters
```
neighbor_latents     : Dict[str, Optional[Tensor]]  — N/S/E/W neighbor codes
boundary_constraints : Tensor [B, 8]    — Door requirements
position             : Tensor [B, 2]    — Grid position
node_features        : Tensor [N, 5]    — Graph node features
edge_index           : Tensor [2, E]    — Graph edges
edge_features        : Tensor [E, F] or None   — Edge type encodings
tpe                  : Tensor [N, 8] or None   — Topological PE
current_node_idx     : Optional[int]    — Return specific node's embedding

Returns: Tensor [B, output_dim]
```

### Constructor
```python
DualStreamConditionEncoder(
    latent_dim: int = 64,          # VQ-VAE latent dim
    node_feature_dim: int = 5,     # Node feature size
    hidden_dim: int = 256,         # GNN hidden dimension
    output_dim: int = 256,         # Conditioning vector size
    num_gnn_layers: int = 3,       # Number of GNN layers
    num_attention_heads: int = 8,  # Cross-attention heads
    dropout: float = 0.1,
)
```

### Factory
```python
create_condition_encoder(
    latent_dim: int = 64,
    output_dim: int = 256,
    **kwargs,
) → DualStreamConditionEncoder
```

### Architecture
```
Local Stream:    neighbors + boundaries + position → MLP → c_local
Global Stream:   node_features + edge_index → GNN (GATv2Conv or FallbackGNN) → c_global
Fusion:          CrossAttentionFusion(c_local, c_global) → c_fused → output_proj → c
```

### Internal Components
| Component | Purpose |
|-----------|---------|
| `LocalStreamEncoder` | Encodes neighbors + boundaries + position |
| `GlobalStreamEncoder` | GATv2Conv (or FallbackGNN fallback) with edge features |
| `CrossAttentionFusion` | Multi-head cross-attention between local and global |
| `FallbackGNN` | Adjacency-matrix GNN when `torch_geometric` unavailable |

---

## Block IV — Latent Diffusion Model

**File:** `src/core/latent_diffusion.py` (1132 lines)
**Purpose:** Generate new latent codes via denoising diffusion in VQ-VAE's latent space.

### `LatentDiffusionModel`

| Method | Input | Output |
|--------|-------|--------|
| `training_loss(z_0, context)` | `z_0: [B,D,H',W']`, `context: [B,ctx_dim]` | `loss: Tensor scalar` |
| `sample(context, shape, ...)` | `context: [B,ctx_dim]`, `shape: (B,D,H',W')`, `graph_data=None`, `return_intermediates=False` | `z_gen: [B,D,H',W']` (DDPM) |
| `ddim_sample(context, shape, ...)` | Same + `num_steps=50`, `eta=0.0`, `graph_data=None` | `z_gen: [B,D,H',W']` (DDIM, faster) |
| `q_sample(z_0, t, noise)` | `z_0, t: [B], noise` | `z_t: [B,D,H',W']` (forward diffusion) |
| `p_mean_variance(z_t, t, context)` | tensors | `(mean, variance, log_variance)` |

### Factory
```python
create_latent_diffusion(
    latent_dim: int = 64,             # Must match VQ-VAE
    context_dim: int = 256,           # Must match CondEncoder output_dim
    num_timesteps: int = 1000,        # Diffusion steps T
    prediction_type: str = 'epsilon', # 'epsilon' or 'v' (Phase 1C)
    cfg_dropout_prob: float = 0.1,    # Conditioning dropout for CFG training
    cfg_scale: float = 3.0,          # Classifier-Free Guidance scale (Phase 1A)
    min_snr_gamma: float = 5.0,       # Min-SNR-γ loss weighting (Phase 4B)
    **kwargs,                         # model_channels etc. passed through
) → LatentDiffusionModel
```

> **Note:** `cfg_scale` default is **3.0** (not 2.0). `model_channels` is NOT a named parameter —
> it's passed via `**kwargs`. Min-SNR-γ is **Phase 4B** (not 4C).

### Key Components

#### UNet Denoiser
```
Input:  x_t [B, D, H', W'], timestep t [B], context c [B, ctx_dim]
Output: predicted_noise [B, D, H', W']

Architecture: Encoder → Bottleneck → Decoder with skip connections
Channel multipliers: (1, 2, 4) by default
Attention at resolutions: (1, 2)
Skip connections: Each UpBlock pops num_res_blocks skips from encoder
```

#### Diffusion Process
```
Forward:   z_0 → z_t = √ᾱ_t · z_0 + √(1-ᾱ_t) · ε     (add noise)
Reverse:   z_t → z_{t-1}  (predict & remove noise)

Schedules: 'linear', 'cosine', 'quadratic'
Sampling:  DDPM (T steps), DDIM (<<T steps, deterministic when eta=0)
```

#### Enhancements
- **CFG (Phase 1A):** Trains with 10% unconditional dropout (`cfg_dropout_prob=0.1`); at inference, `ε_guided = ε_uncond + s·(ε_cond - ε_uncond)` where `s = cfg_scale = 3.0`.
- **v-prediction (Phase 1C):** Predicts velocity `v = √ᾱ_t·ε - √(1-ᾱ_t)·z_0` instead of noise.
- **Min-SNR-γ (Phase 4B):** Clips loss weights at high SNR timesteps to reduce gradient variance.

### Internal Components
| Component | Purpose |
|-----------|---------|
| `TimestepEmbedding` | Sinusoidal timestep → MLP embedding |
| `ResBlock` | GroupNorm + Conv2d + time conditioning |
| `AttentionBlock` | Self-attention + cross-attention + FFN |
| `DownBlock` | Encoder: ResBlocks + Attention + Downsample |
| `UpBlock` | Decoder: Upsample + skip concat + ResBlocks + Attention |
| `GradientGuidance` | LogicNet gradient guidance at sampling time |

---

## Block V — LogicNet

**File:** `src/core/logic_net.py` (828 lines)
**Purpose:** Differentiable constraint checker ensuring dungeon solvability.

### `LogicNet`

| Method | Input | Output |
|--------|-------|--------|
| `forward(z, graph_data=None)` | `z: [B, D, H', W']` (latent codes), `graph_data: Optional[Dict]` | `(loss: Tensor, info: Dict)` |
| `update_temperature(progress)` | `progress: float` ∈ [0, 1] | None (updates internal temperature) |

### Constructor
```python
LogicNet(
    latent_dim: int = 64,
    num_classes: int = 44,
    num_iterations: int = 20,          # Soft Bellman-Ford iterations
    temperature: float = 0.1,          # Softmax temperature for pathfinding
    reach_weight: float = 1.0,         # Weight for reachability loss
    lock_weight: float = 0.5,          # Weight for key-lock loss
    initial_temperature: float = 1.0,  # Annealing start (Phase 1D)
    final_temperature: float = 0.05,   # Annealing end (NOT 0.01)
)
```

### Output `info` Dictionary (actual keys from source)
```python
{
    'tile_logits': Tensor,        # [B, C, H, W] raw tile classification logits
    'walkability': Tensor,        # [B, 1, H, W] walkability mask
    'grid_distances': Tensor,     # [B, 1, H, W] soft shortest-path distances
    'grid_reachability': float,   # Mean reachability score
    'graph_distances': Tensor,    # [N] graph-level distances (if graph_data given)
    'grid_reach_loss': Tensor,    # Grid reachability loss
    'graph_reach_loss': Tensor,   # Graph reachability loss
    'lock_loss': Tensor,          # Key-lock consistency loss
    'total_loss': Tensor,         # Combined weighted loss
}
```

### Temperature Annealing (Phase 1D)
```
Formula: τ = τ_start × (τ_end / τ_start) ^ progress

progress = 0.0 → τ = 1.0    (soft, exploratory)
progress = 1.0 → τ = 0.05   (hard, strict)

This is EXPONENTIAL INTERPOLATION (not exponential decay).
Effect: Gradients from logic constraints become sharper over training.
```

### Internal Components
| Component | Purpose |
|-----------|---------|
| `SoftBellmanFord` | Differentiable shortest-path via relaxed Bellman-Ford |
| `GridPathfinder` | CNN-based soft grid pathfinding |
| `ReachabilityChecker` | Soft reachability scoring from distances |
| `KeyLockChecker` | Verifies key found before lock reached |
| `TileClassifier` | Latent z → tile logits via Conv network |
| `WalkabilityPredictor` | Tile logits → binary walkability mask |

### Integration
- **Training:** `logic_loss, info = logic_net(z_q)` added to diffusion loss.
- **Inference:** Used by `DungeonValidator` to score generated rooms.
- **`graph_data` dict keys:** `adjacency`, `edge_weights`, `start_idx`, `target_idx`, `key_lock_pairs`

---

## Block VI — MAP-Elites (Quality-Diversity)

**File:** `src/evaluation/map_elites.py` (994 lines)
**Purpose:** Maintain a diverse archive of high-quality dungeon solutions.

### `MAPElites`

| Method | Input | Output |
|--------|-------|--------|
| `add(dungeon, ...)` | Dungeon + opt precomputed fitness/features/metadata | `(was_added: bool, fitness: float, features: Tuple)` |
| `add_batch(dungeons)` | `List[dungeon]` | `(num_added: int, num_total: int)` |
| `get_diverse_set(n)` | `n: int` | `List[Elite]` |
| `get_best(n)` | `n: int` | `List[Elite]` (top-n by fitness) |
| `get_diversity_metrics()` | — | `Dict[str, float]` |
| `get_novelty_score(features)` | `features: Tuple[float,...]` | `float` |
| `save(path)` / `load(path)` | `str` | Persistence |

### Constructor
```python
MAPElites(
    feature_extractor: FeatureExtractor,
    fitness_fn: Callable[[Any], float],
    cells_per_dim: int = 10,
    feature_ranges: Optional[List[Tuple[float, float]]] = None,
    feature_dims: int = 2,
)
```

### Factory
```python
create_map_elites(
    feature_type: str = 'linearity_leniency',
    # Options: 'linearity_leniency', 'density_difficulty',
    #          'combined' (4D), 'cbs' (2D), 'full' (6D)
    fitness_fn: Optional[Callable] = None,
    cells_per_dim: int = 10,
    archive_type: str = 'grid',    # 'grid' or 'cvt'
    num_cells: int = 100,          # For CVT archive
) → MAPElites
```

### Archives

#### `EliteArchive` (Grid)
```
Constructor: EliteArchive(feature_dims=2, cells_per_dim=10, feature_ranges=None)
Storage:     Dict[Tuple[int,...], Elite]   — cell coords → best solution
Capacity:    cells_per_dim ^ feature_dims
Methods:     add(), get(), get_all_elites(), get_random(), clear(), coverage(), qd_score()
```

#### `CVTEliteArchive` (Centroidal Voronoi Tessellation)
```
Constructor: CVTEliteArchive(num_cells=100, feature_dims=2,
                             feature_ranges=None, num_cvt_samples=10000)
Storage:     Dict[int, Elite]   — centroid ID → best solution
Method:      k-means on uniform samples → Voronoi cells
```

### Feature Extractors

| Extractor | Dims | Features |
|-----------|------|----------|
| `LinearityLeniencyExtractor` | 2 | linearity, leniency |
| `DensityDifficultyExtractor` | 2 | density, difficulty |
| `CombinedFeatureExtractor` | 4 | linearity, leniency, density, difficulty |
| `CBSFeatureExtractor` | 2 | confusion_ratio, room_entropy |
| `FullFeatureExtractor` | 6 | linearity, leniency, density, difficulty, confusion_ratio, **room_entropy** |

> **Note:** Dim 6 of `FullFeatureExtractor` is `room_entropy`, NOT "symmetry score".

All extractors implement: `extract(graph: nx.DiGraph) → Tuple[float, ...]`

### `Elite` Data Structure
```python
@dataclass
class Elite:
    solution: Any                   # The dungeon (graph or tensor)
    fitness: float                  # Quality score
    features: Tuple[float, ...]     # Behavior descriptor
    cell: Tuple[int, ...]           # Archive cell coordinates
    metadata: Dict[str, Any]        # Extra info (seed, generation, etc.)
```

### `DiversityMetrics`
```python
@dataclass
class DiversityMetrics:
    coverage: float          # Fraction of archive cells filled
    qd_score: float          # Sum of all elite fitnesses
    mean_fitness: float      # Average fitness across elites
    max_fitness: float       # Best fitness
    num_elites: int          # Number of elites in archive
    feature_spread: Dict     # Spread per feature dimension
    uniformity: float        # How evenly distributed elites are
```

---

## Block VII — Symbolic Refiner (WFC Repair)

**File:** `src/core/symbolic_refiner.py` (1233 lines)
**Purpose:** Fix structural violations in generated rooms using Wave Function Collapse.

### `SymbolicRefiner`

| Method | Input | Output |
|--------|-------|--------|
| `repair_room(grid, start, goal)` | `grid: np.ndarray [H,W]` (tile IDs), `start: (r,c)`, `goal: (r,c)` | `(repaired_grid: np.ndarray [H,W], success: bool)` |
| `repair_dungeon(dungeon, validator)` | `dungeon: Any`, `validator: Optional[Any]` | `(repaired_dungeon: Any, success: bool)` |
| `analyze_failures(grid, start, goal)` | grid + coords | `List[FailurePoint]` |
| `wfc_repair(grid, region_mask)` | grid + mask | `repaired_grid: np.ndarray` |

> **Note:** `repair_dungeon` returns `(Any, bool)`, NOT `(rooms, List[RepairResult])`.
> There is NO `RepairResult` class. The actual WFC class is `ConstraintRepairEngine` (not `WFCEngine`).

### Constructor
```python
SymbolicRefiner(
    tile_types: Optional[List] = None,
    adjacency: Optional[Dict] = None,
    tile_weights: Optional[Dict] = None,
    learned_stats: Optional[LearnedTileStatistics] = None,
    max_repair_attempts: int = 5,
    margin: int = 2,
    adjacency_threshold: float = 0.01,
)
```

### Factory
```python
create_symbolic_refiner(
    tile_types: Optional[List] = None,
    max_repair_attempts: int = 5,          # NOT max_iterations
    learned_stats: Optional[LearnedTileStatistics] = None,
) → SymbolicRefiner
```

### `LearnedTileStatistics` (Phase 3B)
```
Purpose: Learn adjacency rules and tile weights from real dungeon data.

Methods:
  observe(room: np.ndarray)                    — Accumulate statistics from one room
  get_adjacency_rules(threshold=0.01)          → Dict  — Learned neighbor rules
  get_tile_weights()                           → Dict[int, float]  — Per-tile frequency weights
```

### `FailurePoint`
```python
@dataclass
class FailurePoint:
    position: Tuple[int, int]    # Where the failure occurred
    failure_type: str            # 'blocked', 'unreachable', 'missing_door', etc.
    required_item: Optional[str] # Item needed to fix
    metadata: Dict[str, Any]     # Additional context
```

### Internal Components
| Component | Purpose |
|-----------|---------|
| `TileType` (Enum) | Tile type constants for WFC |
| `WFCState` (line 121) | Wave Function Collapse state tracking |
| `ConstraintRepairEngine` | Main WFC engine (NOT `WFCEngine`) |
| `PathAnalyzer` | A* pathfinding failure analysis |
| `ConstraintPropagator` | Arc consistency post-WFC |

### WFC Repair Algorithm
```
1. Identify failure points (unreachable goals, blocked paths)
2. For each failure (up to max_repair_attempts=5):
   a. Determine affected region (with margin=2)
   b. Initialize WFC domain (constrained by learned_stats if available)
   c. Propagate constraints (adjacency rules)
   d. Collapse tiles (weighted by tile_weights from training data)
3. Verify structural integrity post-repair
4. Return repaired grid + success flag
```

---

## Training Pipeline

**File:** `src/train_diffusion.py` (713 lines)
**Purpose:** Orchestrate training of VQ-VAE + Diffusion + LogicNet with conditioning.

### `DiffusionTrainingConfig`

```python
DiffusionTrainingConfig(
    # Data
    data_dir: str = "Data/The Legend of Zelda",
    batch_size: int = 4,
    use_vglc: bool = True,
    vqvae_checkpoint: Optional[str] = None,

    # Model dimensions
    latent_dim: int = 64,
    model_channels: int = 128,
    context_dim: int = 256,
    num_timesteps: int = 1000,
    schedule_type: str = "cosine",

    # LogicNet
    num_logic_iterations: int = 30,
    guidance_scale: float = 1.0,

    # Training
    epochs: int = 100,
    learning_rate: float = 1e-4,
    alpha_visual: float = 1.0,     # Diffusion loss weight
    alpha_logic: float = 0.1,      # Solvability loss weight
    warmup_epochs: int = 5,        # Epochs before adding logic loss

    # Checkpointing
    checkpoint_dir: str = "./checkpoints",
    save_every: int = 10,
    device: str = "auto",
    quick: bool = False,           # If True: epochs=2
)
```

### `DiffusionTrainer`

| Method | Purpose |
|--------|---------|
| `train_step(real_maps, conditioning, include_logic_loss)` | Single training step |
| `train_epoch(dataloader, graph_data_available)` | Full epoch training |
| `validate(dataloader, num_samples)` | Validation using EMA weights |
| `save_checkpoint(path, metrics)` | Save all model states |
| `load_checkpoint(path)` | Restore from checkpoint |
| `encode_to_latent(x)` | **Converts [B,1,H,W] → [B,44,H,W] one-hot → VQ-VAE encode** |
| `get_dummy_conditioning(batch_size)` | Curriculum-based synthetic conditioning |

### Data Format Conversion (CRITICAL)
```
Data loader output:   [B, 1, H, W]  — normalized tile IDs in [0, 1]
VQ-VAE expects:       [B, 44, H, W] — one-hot encoded tiles

encode_to_latent() handles this conversion:
  1. Denormalize: tile_ids = (x * 43).round().long().clamp(0, 43)
  2. One-hot:     x_onehot = F.one_hot(tile_ids, 44).permute(0,3,1,2).float()
  3. Encode:      z_q, _ = vqvae.encode(x_onehot)
```

### Training Loop (per step)
```
1. Load batch: (room_tensors [B,1,H,W], graph_data)
2. Encode rooms:     z_q = encode_to_latent(rooms)        [Block II, with format conversion]
3. Build context:    c = cond_encoder.encode_global_only(
                         node_features, edge_index,
                         edge_features=...)               [Block III]
4. Diffusion loss:   L_diff = diffusion.training_loss(z_q, c)   [Block IV]
5. Logic loss:       L_logic, info = logic_net(z_sample)        [Block V]
6. Total loss:       L = α_vis · L_diff + α_logic · L_logic
7. Update EMA:       ema_diffusion ← exponential moving average (decay=0.9999)
8. Anneal temperature: logic_net.update_temperature(progress)
```

### Curriculum Conditioning
```
Phase 1 (< warmup_epochs):      Random noise (unconditional)
Phase 2 (warmup..2×warmup):     Simple 3-node linear graphs
Phase 3 (> 2×warmup):           Complex 5-12 node graphs with branching
```

---

## Generation Pipeline

**File:** `src/generate.py` (531 lines)
**Purpose:** Sample new dungeon rooms from trained models.

### Key Classes

| Class | Purpose |
|-------|---------|
| `LatentDiffusionWrapper` | Wraps VQ-VAE + Diffusion + CondEncoder for end-to-end sampling |
| `DungeonValidator` | Wraps LogicNet + A* for structural validation |
| `WFCRepair` | Wraps Block VII SymbolicRefiner for post-generation repair |

### `LatentDiffusionWrapper.sample()`
```
Input:  num_samples: int, device: torch.device
Output: room_grids: Tensor [B, 1, H, W] (argmax tile IDs, float)

Flow:
  1. Build conditioning (random or from graph)
  2. z_gen = diffusion.ddim_sample(c, shape, num_steps=50)
  3. logits = vqvae.decode(z_gen)          [B, 44, H, W]
  4. return logits.argmax(dim=1, keepdim=True).float()   [B, 1, H, W]
```

### Generation Flow
```
1. Load checkpoint (prefers EMA weights: ema_diffusion_state_dict)
2. Build conditioning c                          [Block III]
3. Sample z_gen via DDIM (50 steps)              [Block IV]
4. Decode to tile logits via VQ-VAE              [Block II]
5. Argmax → room_grid [B, H, W]
6. Validate with LogicNet                        [Block V]
7. Repair if needed via SymbolicRefiner          [Block VII]
8. Add to MAP-Elites archive                     [Block VI]
```

---

## Dimension Cheat Sheet

| Symbol | Default | Description |
|--------|---------|-------------|
| `B` | varies | Batch size |
| `C` | 44 | Tile classes (semantic vocabulary) |
| `H` | 11 | Room height (rows) |
| `W` | 16 | Room width (columns) |
| `D` | 64 | Latent dimension (VQ-VAE) |
| `K` | 512 | Codebook size (VQ-VAE) |
| `H'` | ~3 | Latent spatial height (H/4) |
| `W'` | ~4 | Latent spatial width (W/4) |
| `ctx_dim` | 256 | Conditioning dimension |
| `T` | 1000 | Diffusion timesteps |
| `N` | varies | Number of graph nodes / rooms |
| `E` | varies | Number of graph edges |
| `eta` | 0.0 | DDIM stochasticity (0=deterministic) |

---

## File Cross-Reference

| Block | Main File | Factory / Entry Point |
|-------|-----------|----------------------|
| I — DataAdapter | `src/data_processing/data_adapter.py` | `IntelligentDataAdapter(data_dir)` |
| II — VQ-VAE | `src/core/vqvae.py` | `create_vqvae(num_classes, codebook_size, latent_dim)` |
| III — CondEncoder | `src/core/condition_encoder.py` | `create_condition_encoder(latent_dim, output_dim)` |
| IV — Diffusion | `src/core/latent_diffusion.py` | `create_latent_diffusion(latent_dim, context_dim, ...)` |
| V — LogicNet | `src/core/logic_net.py` | `LogicNet(latent_dim, num_classes, num_iterations)` |
| VI — MAP-Elites | `src/evaluation/map_elites.py` | `create_map_elites(feature_type, fitness_fn, ...)` |
| VII — Refiner | `src/core/symbolic_refiner.py` | `create_symbolic_refiner(tile_types, max_repair_attempts, learned_stats)` |

### Supporting Files
| File | Purpose |
|------|---------|
| `src/train_diffusion.py` | Training orchestrator (all blocks) |
| `src/generate.py` | Generation/inference pipeline |
| `src/data/zelda_loader.py` | PyTorch Dataset/DataLoader for VGLC files |
| `src/core/definitions.py` | `TileID` IntEnum, `SEMANTIC_PALETTE`, `CHAR_TO_SEMANTIC` |
| `src/core/__init__.py` | Re-exports all core classes |
| `src/evaluation/__init__.py` | Re-exports MAP-Elites classes |
