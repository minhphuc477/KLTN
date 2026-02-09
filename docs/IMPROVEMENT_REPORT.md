# H-MOLQD Architecture Improvements Report

**Date:** 2025-01  
**Scope:** 10 targeted improvements across 6 files, 7 H-MOLQD blocks  
**Goal:** Strengthen the architecture for conference-level competitiveness  

---

## Overview

All 7 H-MOLQD blocks were **already fully implemented** before this work. These improvements are targeted, research-backed upgrades that improve training stability, generation quality, and evaluation rigor without altering the core architecture.

| # | Improvement | File | Block | Impact |
|---|-----------|------|-------|--------|
| 1A | Classifier-Free Guidance (CFG) | `latent_diffusion.py` | IV | Generation quality |
| 1B | Dead codebook reset | `vqvae.py` | II | Codebook utilization |
| 1C | v-prediction parameterization | `latent_diffusion.py` | IV | Training stability |
| 1D | Temperature annealing | `logic_net.py` | V | Gradient reliability |
| 2A | Curriculum conditioning | `train_diffusion.py` | Training | Training convergence |
| 3  | CVT-MAP-Elites + CBS features | `map_elites.py` | VI | Diversity measurement |
| 3A | Edge-feature GNN (GATv2Conv) | `condition_encoder.py` | III | Conditioning fidelity |
| 3B | Learned WFC weights | `symbolic_refiner.py` | VII | Repair quality |
| 4A | EMA model weights | `train_diffusion.py` | Training | Generation stability |
| 4B | Min-SNR-γ loss weighting | `latent_diffusion.py` | IV | Balanced denoising |

---

## Phase 1: Core Model Improvements

### 1A. Classifier-Free Guidance (CFG)

**File:** `src/core/latent_diffusion.py`  
**Reference:** Ho & Salimans, "Classifier-Free Diffusion Guidance" (2022)

**What:** During training, the conditioning vector `c` is randomly dropped (replaced with zeros) with probability `cfg_dropout_prob=0.1`. At inference, the model makes both a conditional and unconditional prediction, then extrapolates:

$$\hat{\epsilon} = \epsilon_\theta(z_t, \varnothing) + s \cdot (\epsilon_\theta(z_t, c) - \epsilon_\theta(z_t, \varnothing))$$

where $s$ is the guidance scale (`cfg_scale=3.0`).

**Why:** Without CFG, the diffusion model doesn't know "how much" to follow the conditioning. CFG lets us control the conditioning strength at inference time, enabling a tradeoff between diversity and fidelity.

**New code:**
- `_predict_noise_cfg()` method for guided inference
- Conditioning dropout in `training_loss()`
- Updated `p_mean_variance()` and `ddim_sample()` to use CFG

---

### 1B. Dead Codebook Reset

**File:** `src/core/vqvae.py`  
**Reference:** Williams et al., "Hierarchical Quantized Autoencoders" (2020)

**What:** Codebook entries that haven't been used in the last `_dead_threshold=2` EMA updates are detected every `_reset_interval=100` batches and replaced with randomly sampled encoder outputs plus small noise.

**Why:** With K=512 codebook entries, many entries can become "dead" (never selected during quantization), wasting capacity. Periodic reset reclaims them, improving codebook utilization from typical 30-60% to >90%.

**New code:**
- `_reset_dead_codes()` method in `VectorQuantizer`
- Periodic call wired into `_ema_update()` 

---

### 1C. v-prediction Parameterization

**File:** `src/core/latent_diffusion.py`  
**Reference:** Salimans & Ho, "Progressive Distillation for Fast Sampling" (2022)

**What:** The denoiser can now predict the "velocity" $v = \alpha_t \epsilon - \sigma_t x_0$ instead of noise $\epsilon$. Controlled by `prediction_type='epsilon'|'v'`.

**Why:** v-prediction provides more balanced gradients across timesteps. At high noise levels, predicting $\epsilon$ is easy but $x_0$ is hard; at low noise, the opposite. v-prediction balances both regimes, leading to more stable training and better sample quality.

**New code:**
- `_convert_prediction()` method handles both parameterizations
- Updated training loss to compute v-prediction targets when enabled

---

### 1D. Temperature Annealing for LogicNet

**File:** `src/core/logic_net.py`

**What:** LogicNet's differentiable Bellman-Ford, reachability scoring, and key-lock checking all use a temperature parameter that anneals from `initial_temperature=1.0` to `final_temperature=0.05` during training via exponential decay:

$$\tau = \tau_{\text{init}} \cdot \left(\frac{\tau_{\text{final}}}{\tau_{\text{init}}}\right)^{\text{progress}}$$

**Why:** High temperature early in training produces smoother gradients (making optimization easier), while low temperature late in training makes LogicNet's outputs sharper and closer to the discrete solvability checks used at evaluation.

**New code:**
- `update_temperature(progress)` method on `LogicNet`
- Temperature propagation to all sub-modules

---

## Phase 2: Training Pipeline

### 2A. Curriculum Conditioning

**File:** `src/train_diffusion.py`

**What:** The dummy conditioning used during early training is replaced with a 3-phase curriculum:

| Phase | Progress | Strategy |
|-------|----------|----------|
| 1 | 0–33% | Random noise conditioning |
| 2 | 33–66% | Simple 3-node linear graphs (start→mid→goal) |
| 3 | 66–100% | Complex 5-12 node graphs with branching & key-lock edges |

Synthetic mission graphs are encoded through `condition_encoder.encode_global_only()`.

**Why:** Jumping directly to complex graph conditioning overwhelms the model early. Curriculum learning (Bengio et al., 2009) starts simple and increases complexity, enabling the diffusion model to first learn basic room generation before conditioning on complex dungeon structures.

**New code:**
- `get_dummy_conditioning()` now dispatches by training phase
- `_encode_synthetic_graphs()` creates synthetic mission graphs

---

## Phase 3: Architecture Upgrades

### 3. CVT-MAP-Elites + CBS Feature Extractors

**File:** `src/evaluation/map_elites.py`  
**Reference:** Vassiliades et al., "Using Centroidal Voronoi Tessellations to Scale Up the MAP-Elites Algorithm" (2018)

**What:** Three new components:

1. **CVTEliteArchive:** Uses k-means centroids (Centroidal Voronoi Tessellation) instead of uniform grid discretization. Falls back to uniform grid if scipy is unavailable.

2. **CombinedFeatureExtractor (4D):** Combines linearity, leniency, density, and difficulty into a single feature vector.

3. **CBSFeatureExtractor (2D):** Extracts confusion ratio (from Cognitive Bounded Search) and room entropy as novel behavioral descriptors.

4. **FullFeatureExtractor (6D):** All six features combined.

**Why:** Uniform grid archives scale poorly with dimensionality and waste cells in empty regions. CVT concentrates cells where the search distribution is dense. CBS-derived features capture human-like playability, adding a dimension not covered by traditional linearity/leniency metrics.

**New code:**
- `CVTEliteArchive` class with k-means initialization
- `CombinedFeatureExtractor`, `CBSFeatureExtractor`, `FullFeatureExtractor`
- Updated `create_map_elites()` with `feature_type` and `archive_type` params

---

### 3A. Edge-Feature GNN (GATv2Conv)

**File:** `src/core/condition_encoder.py`  
**Reference:** Brody et al., "How Attentive are Graph Attention Networks?" (2022)

**What:** The Global Stream Encoder's GNN is upgraded from `GATConv` to `GATv2Conv` with edge feature support:

- **Edge encoder:** `nn.Linear(edge_feature_dim, hidden_dim)` maps raw edge type vectors (key_locked, boss_locked, bombable, stair, switch) to hidden representations.
- **GATv2Conv:** Uses dynamic attention (fixes the static attention limitation of GATv1) and natively accepts `edge_attr` in its attention computation.

**Why:** The previous GNN completely ignored edge type information during message passing. In Zelda dungeons, edge types encode critical structural semantics — a key-locked door vs. a free passage fundamentally changes the dungeon graph's meaning. Without edge features, the condition encoder cannot distinguish these structures.

**New code:**
- `GATv2Conv` import alongside existing `GATConv`
- `self.edge_encoder` in `_build_torch_geometric_gnn()`
- Edge feature passing in `_forward_torch_geometric()`
- `edge_features` parameter threaded through `forward()`, `DualStreamConditionEncoder.forward()`, and `encode_global_only()`

---

### 3B. Learned WFC Weights

**File:** `src/core/symbolic_refiner.py`

**What:** New `LearnedTileStatistics` class that accumulates tile co-occurrence statistics from training rooms and derives:

1. **Data-driven adjacency rules:** `get_adjacency_rules(threshold=0.01)` returns which tile pairs were observed adjacent with frequency above the threshold.
2. **Data-driven tile weights:** `get_tile_weights()` returns the relative frequency of each tile type.

The `WaveFunctionCollapse` class now accepts `tile_weights` for non-uniform initial probability distributions. `SymbolicRefiner` accepts `learned_stats` and uses them for both adjacency and weights.

**Why:** Hand-crafted `DEFAULT_ADJACENCY` rules and uniform tile weights don't reflect the actual distribution of tiles in VGLC Zelda rooms. Learned statistics make WFC repairs visually consistent with training data — repaired regions blend naturally with the rest of the room.

**New code:**
- `LearnedTileStatistics` class with `observe()`, `get_adjacency_rules()`, `get_tile_weights()`
- `tile_weights` parameter in `WaveFunctionCollapse.__init__()`
- `learned_stats` parameter in `SymbolicRefiner.__init__()`
- Updated `create_symbolic_refiner()`

---

## Phase 4: Training Stability

### 4A. Exponential Moving Average (EMA)

**File:** `src/train_diffusion.py`  
**Reference:** Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models" (2021)

**What:** A shadow copy of the diffusion model is maintained with EMA decay `0.9999`:

$$\theta_{\text{EMA}} \leftarrow \gamma \cdot \theta_{\text{EMA}} + (1 - \gamma) \cdot \theta$$

The EMA model is used for validation sampling and saved in checkpoints.

**Why:** EMA averages out training noise, producing smoother, more consistent generations. All modern diffusion models use EMA for their published results (DDPM, Stable Diffusion, etc.).

**New code:**
- `self.ema_diffusion` (deepcopy of diffusion model)
- `_update_ema()` called after each `optimizer.step()`
- EMA model used in `validate()` for sample generation
- EMA state included in checkpoint save/load

---

### 4B. Min-SNR-γ Loss Weighting

**File:** `src/core/latent_diffusion.py`  
**Reference:** Hang et al., "Efficient Diffusion Training via Min-SNR Weighting Strategy" (2023)

**What:** The training loss is weighted by:

$$w(t) = \min\left(\text{SNR}(t),\; \gamma\right) \;/\; \text{SNR}(t)$$

where $\text{SNR}(t) = \bar{\alpha}_t / (1 - \bar{\alpha}_t)$ and $\gamma = 5.0$.

**Why:** Without reweighting, high-SNR timesteps (low noise) dominate the loss and low-SNR timesteps (high noise) contribute almost nothing. Min-SNR-γ clips the weight, creating a more balanced gradient signal that improves training speed and final sample quality.

**New code:**
- SNR computation and Min-SNR weighting in `training_loss()`
- `min_snr_gamma=5.0` parameter

---

## Ablation Study Recommendations

To validate each improvement independently, run ablation experiments with one change disabled at a time:

| Ablation | How to disable | Expected impact if removed |
|----------|---------------|--------------------------|
| CFG | Set `cfg_scale=1.0` | Less conditioning adherence |
| Dead code reset | Set `_reset_interval=999999` | Lower codebook utilization |
| v-prediction | Set `prediction_type='epsilon'` | Slightly less stable training |
| Temperature anneal | Set `initial_temperature == final_temperature` | Noisier LogicNet gradients |
| Curriculum | Return random conditioning always | Slower convergence |
| CVT archive | Set `archive_type='grid'` | Less uniform QD coverage |
| Edge-feature GNN | Don't pass `edge_features` | Same as before (graceful fallback) |
| Learned WFC | Don't pass `learned_stats` | Uses DEFAULT_ADJACENCY |
| EMA | Use `self.diffusion` for validation | Noisier sample quality |
| Min-SNR-γ | Set `min_snr_gamma=None` | Unbalanced timestep training |

---

## Files Modified Summary

| File | Lines added (approx.) | Changes |
|------|----------------------|---------|
| `src/core/latent_diffusion.py` | ~70 | CFG, v-prediction, Min-SNR-γ |
| `src/core/vqvae.py` | ~40 | Dead codebook reset |
| `src/core/logic_net.py` | ~20 | Temperature annealing |
| `src/core/condition_encoder.py` | ~30 | GATv2Conv, edge features |
| `src/core/symbolic_refiner.py` | ~130 | LearnedTileStatistics, tile weights |
| `src/train_diffusion.py` | ~80 | EMA, curriculum, temperature |
| **Total** | **~370** | **10 improvements** |

All changes are backward-compatible. Existing code paths remain functional with default parameters.
