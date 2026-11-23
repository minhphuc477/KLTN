# G-LMDM: Graph-Conditioned Logic-Regularized Masked Discrete Model for Provably Solvable Dungeon Generation

[![Target Venue: CORE A* / IEEE TPAMI / IEEE ToG / NeurIPS](https://img.shields.io/badge/Target%20Venue-CORE%20A*%20%7C%20IEEE%20TPAMI%20%7C%20ToG-blue.svg)](PUBLICATION_BLUEPRINT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Validation: Schema 2.0 Attributed](https://img.shields.io/badge/Validation-Schema%202.0%20Attributed-orange.svg)](docs/PUBLICATION_GUIDANCE_AND_CARDS.md)
[![Solvability: StateSpaceA* Pareto](https://img.shields.io/badge/Solvability-StateSpaceA*%20Pareto-purple.svg)](docs/SOLVERS_AND_GUI_REFERENCE.md)

**Official Research Framework & Verification Benchmark for A*-Tier Venue Submission**  
*(IEEE Transactions on Pattern Analysis and Machine Intelligence [TPAMI] / IEEE Transactions on Games [T-G Flagship] / AAAI / NeurIPS)*

---

## Abstract & Scientific Overview

**G-LMDM** (`Graph-Conditioned Logic-Regularized Masked Discrete Model`) is an end-to-end, repair-aware neuro-symbolic generative framework designed for controllable, provably solvable action-adventure dungeon synthesis under discrete, small-data categorical regimes (*The Legend of Zelda*, Video Game Level Corpus [VGLC]).

Unlike conventional continuous-latent generators (such as continuous latent diffusion or VQ-VAE + UNet/DiT) that suffer from tokenization distortion, parameter bloat, and singleton-tile rarity erasure on small discrete tile corpora, **G-LMDM** natively formulates dungeon room generation as an **absorbing-state discrete masked diffusion process ($Q_t$)** operating directly on $16 \times 11$ discrete tokens with strict parameter efficiency ($3.58\text{M}$ training / $3.36\text{M}$ inference vs. $47.13\text{M}$ for continuous VQ-VAE + UNet baselines).

The framework establishes five core technical components:
1. **Multi-Scale Topological Conditioning:** GraphGPS with Relative Random Walk Probabilities (RRWP) fused with incident directional boundary tokens to prevent macro-to-micro semantic-geometric disconnect (`RG1 / PS1 / RO1 / RQ1 / H1`).
2. **Direct Differentiable Inductive Bias:** Direct $16 \times 11$ Soft Bellman-Ford value iteration (`LogicNet`, $\alpha_{\text{logic}} = 0.1$) penalizing disconnected cut-sets along critical progression paths during training (`RG3 / PS3 / RO3 / RQ3 / H3`).
3. **Attributed Four-Stage Accounting:** Formal four-stage attribution (`raw` $\to$ `constrained` $\to$ `repaired` $\to$ `compiled`), strictly reporting raw neural validity alongside repair burden (`repair_rate`, `tiles_repaired_mean`) to eliminate post-hoc repair confounding (`RG3 / PS3`).
4. **Exact State-Space Solvability Proofs:** An exact state-space oracle solver (`StateSpaceAStar`) with Pareto Dominance Pruning ($s_1 \succeq s_2$) certifying playability across non-monotonic fungible key economies, dynamic block geometry, and locked gates (`RG4 / PS4 / RO4 / RQ4 / H4`).
5. **Human-Centered Cognitive Probing:** Persona-Driven Cognitive Bounded Search (`P-CBS`) with Bayesian belief updates, Miller-Cowan working memory limits, and procedural personas to audit human playability and frustration under partial observability.

---

### Scientific Framework & Traceability Matrix (`RG1–RG4 -> PS1–PS4 -> RO1–RO4 -> RQ1–RQ4 -> H1–H4`)

All experiments strictly adhere to the 1-to-1 field-gap-to-evidence bijection defined across [`docs/publication_guidance_card.json`](docs/publication_guidance_card.json), [`docs/SOTA_COMPARISON_AND_BENCHMARKS.md`](docs/SOTA_COMPARISON_AND_BENCHMARKS.md), [`PUBLICATION_BLUEPRINT.md`](PUBLICATION_BLUEPRINT.md), and [`docs/TRACEABILITY_AND_CLAIM_VALIDATION.md`](docs/TRACEABILITY_AND_CLAIM_VALIDATION.md):

- **`RG1 (FG1) -> PS1 -> RO1 -> RQ1 (H1)`**: Address macro-to-micro semantic-geometric disconnect by isolating Block III `GraphGPS + RRWP` + directional neighbor/boundary conditioning (`G`) against unconditioned/neighbor-only (`N`) and `No-RRWP` ablations (`B1`).
- **`RG2 (FG2) -> PS2 -> RO2 -> RQ2 (H2)`**: Address small-data continuous compression & rarity erasure (`349` Dungeon `1–8` train rooms, `217` unique `16x11` grids, `43` unique `12x7` structural interiors; `88.60%` `WALL+FLOOR`, `0.68%` `ENEMY`, `0.05%` `STAIR`) by benchmarking absorbing-state discrete `G-LMDM` (`3.58M` train / `3.36M` infer; `84.8K` denoiser) against continuous `VQ-VAE + UNet[1,2]` (`47.13M`) and `DiT+Flow` (`22.44M`) on matched `4 GB` GPU capacity envelopes (`B2–B3`).
- **`RG3 (FG3) -> PS3 -> RO3 -> RQ3 (H3)`**: Address traversability-blind likelihood and post-hoc repair confounding by coupling direct `16x11` Soft Bellman-Ford `LogicNet` supervision (`G-L`/`G-LG`, $\alpha_{\text{logic}}=0.1$ vs $G$ and inference-only guidance; `B4`) with Schema 2.0 4-stage attribution (`raw -> constrained -> repaired -> compiled`, `repair_rate`, `tiles_repaired_mean`).
- **`RG4 (FG4) -> PS4 -> RO4 -> RQ4 (H4)`**: Prevent split leakage, omitted external baselines, and proxy/oracle conflation via unseen `Dungeon-9` holdout (`101` rooms), external symbolic baselines (`Random`, `WFC`, `Grammar-only`), paired Holm-corrected non-parametric inference, descriptive LODO (`B5`), Block I topology bounds, and strict separation between `StateSpaceAStar` solvability proofs and bounded `P-CBS` proxy diagnostics.

## Key Features

- **Evolutionary Topology Generation**: Search-based procedural content generation using genetic algorithms to evolve dungeon graphs that match tension curves
- **Neural-Symbolic Pipeline**: H-MOLQD components for VQ-VAE, graph conditioning, room generation, LogicNet guidance, symbolic repair, and stateful validation
- **VGLC Compliance**: Full compliance with Video Game Level Corpus standards for Zelda dungeon structure validation, including composite node labels, Boss-Goal subgraph validation, and centralized graph utilities
- **Interactive GUI**: Real-time visualization and validation environment for dungeon exploration with route export/loading and multi-algorithm pathfinding
- **Comprehensive Testing**: Automated pytest suite covering core neural, symbolic, evaluation, and VGLC compliance modules

## Architecture

### Proposed Primary Architecture: G-LMDM (Zero VQ-VAE / Direct Discrete Tokens)
1. **Stage 1 [Block I: Evolutionary Topology Director]:**
   - Implements evolutionary search over graph grammars to produce mission graph $G = (V, E, X_V, X_E)$.
   - Uses $(\mu+\lambda)$-ES with tournament selection and biased mutation to optimize tension curves.
2. **Stage 2 [Block III: Dual-Stream Condition Encoder]:**
   - Encodes global topological context via GraphGPS + RRWP (Relative Random Walk Probabilities).
   - Enforces exact local doorway alignment via hard boundary token clamping at $t=0$.
   - Projects mission graph nodes to room grid coordinates via `GraphToGridCrossAttention`.
3. **Stage 3 [G-LMDM Denoiser]:**
   - Absorbing-state discrete masked diffusion process ($Q_t$) operating natively on $16 \times 11$ categorical tokens.
   - **Completely eliminates Block II (VQ-VAE)**: zero continuous latent bottleneck ($z_q$), zero codebook distortion.
   - Ultra-lightweight footprint: $84.8\text{K}$ denoiser parameters ($3.58\text{M}$ training / $3.36\text{M}$ inference).
4. **Stage 4 [Block V: Direct 16x11 LogicNet Regularization]:**
   - Differentiable Soft Bellman-Ford value iteration applied directly to $16 \times 11$ walkability logits.
   - Penalizes corridor cut-sets between incident doorway anchors during training ($\alpha_{\text{logic}} = 0.1$).
5. **Stage 5 [Block VI: Global Dungeon Assembly & Attributed Symbolic Repair]:**
   - Strict/relaxed room spatial placement, boundary passage realization, and WFC/symbolic refiners.
   - Enforces Schema 2.0 4-stage attribution (`raw` $\to$ `constrained` $\to$ `repaired` $\to$ `compiled`).
6. **Stage 6 [Block VII: Dual-Tier Evaluation & Verification]:**
   - **Formal Oracle:** `StateSpaceAStar` with Pareto Dominance Pruning ($s_1 \succeq s_2$) certifying exact mathematical solvability.
   - **Behavioral Proxy:** Persona-Driven Cognitive Bounded Search (`P-CBS`) with 8 procedural personas for human-centered diagnostic probing.

### Comparative Baseline Architecture: H-MOLQD (Continuous Latent Baseline)
- Retained strictly for experimental comparison to validate Hypothesis $H_2$ / Research Gap $RG_2$ (Benchmark Tracks $B_2$–$B_3$):
  - **Block II (Continuous Latent Tokenizer):** `SemanticVQVAE` / `SemanticVQVAE2` ($17.37\text{M}$ parameters) compressing $16 \times 11 \to z_q \in \mathbb{R}^{4 \times 3 \times 64}$.
  - **Block IV (Continuous Latent Diffusion / Flow):** UNet $[1, 2]$ ($47.13\text{M}$ total) or DiT+Flow ($22.44\text{M}$ total) operating on continuous latents.

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

# Demo-ready launch with the Chapter 4 real generated artifact and live P-CBS solve
python gui_runner.py --advanced --load-txt examples\gui_demo_hard_real_model_level.txt --solver-algorithm 7 --solver-timeout 900

# In the GUI, press T to show the inferred topology overlay, then SPACE to solve live.

# Optional deterministic GUI AI generation seed
set KLTN_AI_SEED=42

# Optional checkpoint override for GUI AI generation:
set KLTN_CHECKPOINT_PATH=outputs\<run_name>\checkpoints\diffusion\best_model.pth

# Optional hardening: require checkpoint metadata sidecars for GUI AI generation
set KLTN_STRICT_CHECKPOINTS=1

# Controls:
# Arrow keys: Manual movement
# SPACE: Auto-solve with the selected solver
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
# Hyperparameter and batch preflight validation
python scripts/check_training_hyperparameters.py --config configs/zelda_hmolqd.yaml --probe-data

# Primary discrete G-LMDM training (GPS + RRWP + direct 16x11 Soft Bellman-Ford LogicNet)
python main.py train --config configs/zelda_hmolqd.yaml --stage masked_room --data-dir "Data/The Legend of Zelda"

# G-only discrete ablation without LogicNet
python main.py train --config configs/zelda_hmolqd_masked_small.yaml --stage masked_room --data-dir "Data/The Legend of Zelda"

# Full staged training via canonical CLI entrypoint
python main.py train --config configs/zelda_hmolqd.yaml --stage all --data-dir "Data/The Legend of Zelda"

# Only VQ-VAE pretraining
python main.py train --config configs/zelda_hmolqd.yaml --stage vqvae

# VQ-VAE with CoordConv + differentiable illegal adjacency penalty
python -m src.train_vqvae --config configs/zelda_hmolqd.yaml --data-dir "Data/The Legend of Zelda" --seed 42 --use-coordconv --mrf-penalty-weight 0.05

# Only diffusion (with pretrained VQ-VAE)
python main.py train --config configs/zelda_hmolqd.yaml --stage diffusion --diffusion-vqvae-checkpoint checkpoints/vqvae_pretrained.pth

# Training writes sidecar metadata (*.meta.json) next to checkpoints,
# which is validated when strict checkpoint mode is enabled in the pipeline.
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
# Attack 1 (Topology realism): verify grammar and cycle/progression rules
python -m pytest tests/test_topology_generation_fixes.py tests/test_wave3_pedagogical_rules.py tests/test_advanced_rules_integration.py -q

# Attack 2 (Evidence): run fixed-seed ablations and benchmark outputs
python scripts/run_ablation_study.py --plan-only --core-only --output results/ablation_plan_preview
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

- **Start Here (single docs entrypoint)**: [`docs/INDEX.md`](docs/INDEX.md)
- **Publication Blueprint (confirmatory protocol & claim gates)**: [`PUBLICATION_BLUEPRINT.md`](PUBLICATION_BLUEPRINT.md)
- **Experiment Plan & Run Tracker**: [`refine-logs/EXPERIMENT_PLAN.md`](refine-logs/EXPERIMENT_PLAN.md) & [`refine-logs/EXPERIMENT_TRACKER.md`](refine-logs/EXPERIMENT_TRACKER.md)
- **Publication Guidance & Cards (`PS/RO/RQ/H`)**: [`docs/PUBLICATION_GUIDANCE_AND_CARDS.md`](docs/PUBLICATION_GUIDANCE_AND_CARDS.md)
- **Claim Traceability (`PS -> RO -> RQ -> H -> Block`)**: [`docs/TRACEABILITY_AND_CLAIM_VALIDATION.md`](docs/TRACEABILITY_AND_CLAIM_VALIDATION.md)
- **Docs Folder Landing**: [`docs/README.md`](docs/README.md)
- **Current Architecture**: `docs/CURRENT_ARCHITECTURE.md`
- **Block I/O Reference**: `docs/BLOCK_IO_REFERENCE.md`
- **Canonical Model Rationale & Complexity**: `docs/CANONICAL_MODEL_RATIONALE_ABLATION_AND_COMPLEXITY_GUIDE.md`
- **Architecture Audit Research Notes**: `docs/architecture_audit_research_notes.md`
- **Claim Traceability**: `docs/TRACEABILITY_AND_CLAIM_VALIDATION.md`
- **Publication Guidance & Cards**: `docs/PUBLICATION_GUIDANCE_AND_CARDS.md`
- **Training Commands In Order**: `docs/TRAINING_COMMANDS_IN_ORDER.md`
- **Local Research Evidence Runbook**: `docs/LOCAL_RESEARCH_EVIDENCE_RUNBOOK.md`
- **Kaggle Training Suite**: `docs/KAGGLE_T4_X2_TRAINING.md`
- **Grammar Reference**: `docs/GRAMMAR_REFERENCE.md`
- **Solver + GUI Reference**: `docs/SOLVERS_AND_GUI_REFERENCE.md`
- **External Baselines & Benchmarks**: `docs/SOTA_COMPARISON_AND_BENCHMARKS.md`
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
