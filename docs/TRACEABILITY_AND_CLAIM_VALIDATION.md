# Traceability And Claim Validation

This reference traces architectural claims in the publication manuscript to their exact
implementation status in code.

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
| Topology search optimizes target tension curve and descriptor objectives | TRUE | `src/generation/evolutionary_director/` evaluator computes curve MSE/fitness and descriptor scoring (`leniency`, etc.) |
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
  - Main canonical pipeline (`dungeon_pipeline.py`) does not default to this path.
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
  - `RANDOM_TOPOLOGY` is now the strict random-topology baseline; it should be reported separately from `NO_EVOLUTION`.

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
| Heuristic ablation (pure WFC baseline) | HIGH | `PURE_WFC` bypasses neural priors and uses standalone Weighted Bayesian WFC rooms over the matched topology |
| Random topology baseline for Tier-1 proof | HIGH | `RANDOM_TOPOLOGY` exists as a strict start-to-goal random DAG baseline distinct from `NO_EVOLUTION` |

## Recommended Defense-Safe One-Liner

"The architecture is a three-tier evolutionary neuro-symbolic system: macro topology search, micro neural prior generation, and symbolic repair. Each tier is independently ablated in code; weighted Bayesian WFC and a repo-specific consistency-LoRA fast-sampling path exist as advanced extensions."

## 3. End-to-End Scientific Traceability Matrix

Every claim in [PUBLICATION_GUIDANCE_AND_CARDS.md](./PUBLICATION_GUIDANCE_AND_CARDS.md), [SOTA_COMPARISON_AND_BENCHMARKS.md](./SOTA_COMPARISON_AND_BENCHMARKS.md), and [publication_guidance_card.json](./publication_guidance_card.json) maps monotonically from a literature research gap (`RG`) and problem statement (`PS`) to a research objective (`RO`), research question (`RQ`), falsifiable hypothesis (`H`), architectural/experimental block, runnable artifact, and rejection gate.

| Research Gap | Problem Statement | Research Objective | Research Question | Hypothesis | Arch / Exp Block | Variants / Tracks | Configs & Code Modules | Primary Endpoints & Claim / Rejection Gates |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`RG1` (`FG1`)** Macro-to-micro semantic-geometric disconnect (Dormans 2010; Gutierrez & Schrum 2020; GRIT/RRWP Ma et al. 2023) | **`PS1`** Causal multi-scale conditioning deficit | **`RO1`** Isolate causal `GraphGPS+RRWP` + neighbor/boundary conditioner (`G`) vs `N` and `No-RRWP` | **`RQ1`** Does causal `GraphGPS+RRWP` conditioning improve pre-repair door alignment & role adherence on Dungeon 9? | **`H1`** `G` beats `N` and `No-RRWP` on boundary alignment & role consistency | `Block III`, `Block IV` / `B1` (`R001–R004`) | `N`, `No-RRWP`, `G` (discrete `MaskedRoom` conditioner ablation) | [zelda_hmolqd.yaml](../configs/zelda_hmolqd.yaml), [ablation_no_rrwp.yaml](../configs/ablation_no_rrwp.yaml), [ablation_neural_floor_conditioning.yaml](../configs/ablation_neural_floor_conditioning.yaml) | Raw door alignment, boundary match accuracy, mission-role adherence. Reject **`H1`** if `G` fails to improve over `N`/`No-RRWP`. |
| **`RG2` (`FG2`)** Continuous latent bottleneck & rarity erasure on tiny grids (Schrum et al. 2025; D3PM/MaskGIT/LayoutDM/MDLM) | **`PS2`** Continuous `VQ-VAE` distortion (`47.13M`/`22.44M`) vs discrete `G-LMDM` (`3.58M`) on `349` train rooms (`43` unique `12x7` interiors, `88.60%` wall/floor) | **`RO2`** Benchmark discrete absorbing-state `G-LMDM` vs continuous `VQ-VAE + UNet[1,2]` and `DiT+Flow` | **`RQ2`** Does discrete `G-LMDM` retain rare tiles (`<0.7%`) and beat continuous baselines on raw $R_q$ and cost? | **`H2`** Discrete `G-LMDM` beats continuous baselines on rarity retention & efficiency (or maps trade-off) | `Block II`, `Block IV` / `B2–B3` (`R005–R010`) | `G-LMDM` (`3.58M` train / `3.36M` infer), `UNet[1,2]` (`47.13M`), `DiT+Flow` (`22.44M`) | [zelda_hmolqd.yaml](../configs/zelda_hmolqd.yaml), [ablation_flow_matching.yaml](../configs/ablation_flow_matching.yaml), [ablation_no_fsq.yaml](../configs/ablation_no_fsq.yaml), [check_training_hyperparameters.py](../scripts/check_training_hyperparameters.py) | `rare_tile_recall`, raw $R_q$, `ms/room`, peak VRAM `<= 4 GB`. Reject **`H2`** if continuous backbones dominate raw $R_q$ cleanly. |
| **`RG3` (`FG3`)** Traversability-blind likelihood & unattributed repair (Tamar et al. 2016; Xu et al. 2018; Zhang et al. 2020; Cooper 2022/2023) | **`PS3`** Tokenwise CE misses 1-tile cut-sets & post-repair reporting hides neural defects | **`RO3`** Couple direct `16x11` Soft Bellman-Ford `LogicNet` (`G-LG`) & attribute `raw -> constrained -> repaired -> compiled` | **`RQ3`** Does direct `16x11` `LogicNet` (`G-LG`) raise raw solvability & reduce `repair_rate` / `tiles_repaired_mean`? | **`H3`** `G-LG` (`alpha_logic=0.1`) outperforms `G` & inference-only steering at matched diversity | `Block V`, `Block VI` / `B4` (`R011–R015`, `Schema 2.0`) | `G` ([zelda_hmolqd_masked_small.yaml](../configs/zelda_hmolqd_masked_small.yaml)), `G-L`, `G-LG` (`G-LMDM`), inference-only steering | [zelda_hmolqd.yaml](../configs/zelda_hmolqd.yaml), [zelda_hmolqd_masked_small.yaml](../configs/zelda_hmolqd_masked_small.yaml), [ablation_no_logicnet.yaml](../configs/ablation_no_logicnet.yaml), [ablation_inference_guidance_only.yaml](../configs/ablation_inference_guidance_only.yaml), [stage_bridge.py](../src/publication/stage_bridge.py) | Raw `oracle_solved_rate`, `repair_rate`, `tiles_repaired_mean`, diversity non-inferiority. Reject **`H3`** if `G-LG` fails Holm test or collapses diversity. |
| **`RG4` (`FG4`)** Dungeon-split leakage, omitted external baselines & proxy/oracle conflation (Withington et al. 2024; Khalifa et al. 2025) | **`PS4`** Room-level leakage, omitted symbolic baselines & proof-proxy conflation | **`RO4`** Enforce Dungeon-9 holdout, external symbolic baselines, paired Holm inference, descriptive LODO & proof separation | **`RQ4`** Do `G-LMDM` gains hold on unseen Dungeon 9 vs external baselines with `StateSpaceAStar` distinct from `P-CBS`? | **`H4`** Paired Dungeon-9 tests preserve gains & keep formal proofs separate from `P-CBS` proxies | `Block I`, `Block VII` / `B5` (`R016`) | Held-out Dungeon 9 paired test; descriptive LODO; `Random`, `WFC`, `Grammar-only`; `P-CBS` (`proxy_only`) | [confirmatory.py](../src/publication/confirmatory.py), [benchmark_suite.py](../src/evaluation/benchmark_suite.py), [run_ablation_study.py](../scripts/run_ablation_study.py), `experiments/paper_v1/` | Paired CIs, permutation p-values, Holm correction, `StateSpaceAStar` vs `P-CBS`. Gate: [validate_publication_readiness.py](../scripts/validate_publication_readiness.py). |

## 4. Cross-Document Mapping (`PUBLICATION_BLUEPRINT.md`, `EXPERIMENT_PLAN.md`, and Manuscript)

| Framework Layer | [PUBLICATION_BLUEPRINT.md](../PUBLICATION_BLUEPRINT.md) | [EXPERIMENT_PLAN.md](../refine-logs/EXPERIMENT_PLAN.md) & [EXPERIMENT_TRACKER.md](../refine-logs/EXPERIMENT_TRACKER.md) | Manuscript Reference (`REPORT_LATEX/IEEE_FULL_PAPER_VN.tex`, read-only) |
| :--- | :--- | :--- | :--- |
| **`RG1 / FG1`: Causal Conditioning (`PS1 -> RO1 -> RQ1 -> H1`)** | Block III `GraphGPS + RRWP` + directional neighbor/boundary conditioning | **`B1`** Causal conditioning ablations (`N`, `No-RRWP`, `G`) | Mission-graph conditioning framing |
| **`RG2 / FG2`: Discrete vs. Continuous (`PS2 -> RO2 -> RQ2 -> H2`)** | Discrete `G-LMDM` (`3.58M`) vs `VQ-VAE + UNet[1,2]` (`47.13M`) & `DiT+Flow` (`22.44M`) | **`B2–B3`** Discrete `G-LMDM` vs continuous `VQ-VAE` + `UNet[1,2]`/`DiT+Flow` on `349` train rooms (`43` unique `12x7` interiors) | Architecture & small-data latent bottleneck framing |
| **`RG3 / FG3`: Direct Logic & Repair Attribution (`PS3 -> RO3 -> RQ3 -> H3`)** | Block V direct `16x11` Soft Bellman-Ford `LogicNet` & Block VI Schema 2.0 attribution | **`B4`** `G` vs `G-L` vs `G-LG` vs inference-only steering + 4-stage `raw -> constrained -> repaired -> compiled` logs | Chapter 5 differentiable constraint & multi-stage attribution |
| **`RG4 / FG4`: Dungeon-9 Statistical & Oracle Proof (`PS4 -> RO4 -> RQ4 -> H4`)** | Dungeon-9 confirmatory suite, Block I bounds, Claim Gates (`Block I`, `Block VII`) | **`B5`** Held-out Dungeon-9 paired tests, external symbolic baselines (`Random`, `WFC`, `Grammar-only`), descriptive LODO | Chapter 5 formal oracle vs bounded-agent diagnostic layers |
