# Architecture Traceability and Claim Validation (2026-03-20)

## Scope

This report fulfills two requested tasks:

1. Architecture traceability matrix (claim -> concrete implementation evidence).
2. Claims that are overstated, partially true, or currently wired only in optional paths.

It also validates the external reviewer statement describing the system as a 3-tier evolutionary neuro-symbolic pipeline.

## 1) Architecture Traceability Matrix

| Claim | Status | Evidence |
|---|---|---|
| System has a canonical 7-block neural-symbolic path | TRUE | `src/pipeline/dungeon_pipeline.py` module header + `NeuralSymbolicDungeonPipeline.generate_dungeon(...)` |
| Block I topology generation is integrated with room generation | TRUE | `generate_dungeon(...)` creates `EvolutionaryTopologyGenerator` when `generate_topology=True` |
| Topology search optimizes target tension curve and descriptor objectives | TRUE | `src/generation/evolutionary_director.py` evaluator computes curve MSE/fitness and descriptor scoring (`leniency`, etc.) |
| Search includes a QD/Map-Elites style option | TRUE (OPTIONAL) | `search_strategy` supports `cvt_emitter`; uses `CVTEliteArchive` when available |
| Default topology strategy is MAP-Elites | FALSE | Default `search_strategy` is `ga`; only optional aliases map to `cvt_emitter` |
| Tier-2 representation uses VQ-VAE + latent diffusion in the main pipeline | TRUE | `src/core/vqvae.py`, `src/core/latent_diffusion.py`, and `generate_room(...)` in `dungeon_pipeline.py` |
| LCM-LoRA is part of the default core pipeline path | FALSE | The repo contains consistency-LoRA fast-sampler scaffolding in `src/optimization/lcm_lora.py`, but not a paper-faithful LCM-LoRA runtime on the default core path |
| Tier-3 refinement is WFC-based and symbolic | TRUE | `src/core/symbolic_refiner.py` contains WFC-based repair used by `dungeon_pipeline._create_refiner(...)` |
| Default runtime refiner is Weighted Bayesian WFC | FALSE | Core path instantiates `SymbolicRefiner`; Weighted Bayesian WFC is not directly invoked by `dungeon_pipeline.py` |
| Weighted Bayesian WFC exists and is used in research/benchmark probing | TRUE | `src/generation/weighted_bayesian_wfc.py`; used in `src/evaluation/benchmark_suite.py` via `integrate_weighted_wfc_into_pipeline(...)` |
| Architecture supports controllable benchmarking and ablation | TRUE | `scripts/run_ablation_study.py`, `src/evaluation/benchmark_suite.py` |

## 2) Claims That Need Tightening (Docs/Defense Risk)

### A. "Tier 1 is MAP-Elites grammar optimization" 

- Current reality:
  - Core Block I is GA by default (`search_strategy='ga'`).
  - CVT-emitter QD path is optional.
- Defense-safe wording:
  - "Tier 1 uses evolutionary grammar search with an optional CVT-emitter quality-diversity backend."

### B. "Tier 2 includes LCM-LoRA in the main path"

- Current reality:
  - The repo has a consistency-LoRA fast-sampler path and legacy LCM-LoRA naming.
  - Main thesis pipeline (`dungeon_pipeline.py`) does not default to this path.
  - A paper-faithful LCM-LoRA runtime is not currently implemented.
- Defense-safe wording:
  - "Tier 2 mainline uses VQ-VAE + latent diffusion; the repo contains an experimental consistency-LoRA fast-sampling path, not a full LCM-LoRA deployment."

### C. "Tier 3 default bridge is Weighted Bayesian WFC"

- Current reality:
  - Main generation path uses `SymbolicRefiner` WFC.
  - Weighted Bayesian WFC is present and exercised in benchmark probe workflows.
- Defense-safe wording:
  - "Tier 3 mainline uses symbolic WFC repair; a weighted Bayesian WFC variant is implemented for advanced/benchmark integration."

### D. "NO_EVOLUTION equals pure random graph baseline"

- Current reality:
  - In `run_ablation_study.py`, `NO_EVOLUTION` uses direct `MissionGrammar.generate(...)`, not pure random graph sampling.
- Action if needed:
  - Add a new `RANDOM_TOPOLOGY` config for a strict random baseline.

## Validation of the Provided Reviewer Statement

Overall verdict: MOSTLY VALID, with 3 important precision fixes.

### Valid parts

- Correctly recognizes a multi-tier neuro-symbolic architecture.
- Correctly identifies VQ-VAE + diffusion representation layer.
- Correctly identifies symbolic refinement and strong ablation opportunities.
- Correctly frames solvability/diversity/controllability as separable failure modes.

### Corrections required

1. MAP-Elites is not the always-on Tier-1 default.
2. Paper-faithful LCM-LoRA is not implemented on the default core execution path.
3. Weighted Bayesian WFC is not yet the default runtime refiner in `dungeon_pipeline.py`.

## Ablation Matrix Feasibility Against Current Code

| Proposed experiment | Feasibility now | Notes |
|---|---|---|
| Disable evolution and compare pacing control | HIGH | `NO_EVOLUTION` already exists (grammar direct generation, not random) |
| Representation ablation (diffusion vs categorical/codebook sweeps) | HIGH | `LATENT_DIFFUSION`, `LATENT_CATEGORICAL`, `VQ_CODEBOOK_*` already implemented |
| Bridge ablation (WFC off) | HIGH | `NO_WFC` exists via `apply_repair=False` |
| Heuristic ablation (pure WFC baseline) | MEDIUM | Need an explicit config that bypasses neural priors and runs standalone WFC generation |
| Random topology baseline for Tier-1 proof | MEDIUM | Requires adding `RANDOM_TOPOLOGY` config path |

## Recommended Defense-Safe One-Liner

"The architecture is a three-tier evolutionary neuro-symbolic system: macro topology search, micro neural prior generation, and symbolic repair. Each tier is independently ablated in code; weighted Bayesian WFC and a repo-specific consistency-LoRA fast-sampling path exist as advanced extensions."
