# IEEE ToG Blueprint and Room Generation Architecture

Last updated: 2026-03-20

This document maps the codebase to a journal-ready manuscript structure and provides a technical breakdown of the room generation block.

## 1) Target Paper Structure (IEEE Transactions on Games style)

### 1. Introduction

- Problem: Pure neural PCG has weak deterministic guarantees; pure symbolic methods can lack style priors.
- Gap: Reliable Zelda-style generation under small, highly constrained data.
- Contribution: A decoupled neuro-symbolic pipeline with explicit topology constraints, neural priors, and symbolic repair.

Code anchors:
- [src/pipeline/dungeon_pipeline.py](../src/pipeline/dungeon_pipeline.py)
- [src/generation/evolutionary_director.py](../src/generation/evolutionary_director.py)
- [src/generation/weighted_bayesian_wfc.py](../src/generation/weighted_bayesian_wfc.py)

### 2. Related Work

Recommended clusters:
- Mission/space decoupling and graph grammars.
- Search-based PCG and quality-diversity.
- PCGML for discrete level generation.

Suggested publications to cite:
- Dormans and Bakkes (2011), mission/space generation with graph grammars.
- Togelius et al. (2011), search-based procedural content generation survey.
- Summerville et al. (2016), VGLC corpus and representation standards.
- Khalifa et al. (2020), PCGRL (RL-based level generation).
- Vassiliades et al. (2017), CVT-MAP-Elites scaling behavior.
- Fontaine et al. (2020), CMA-ME improvements for QD.

Repo evidence summary already compiled in:
- [docs/SOTA_COMPARISON_AND_BENCHMARKS.md](SOTA_COMPARISON_AND_BENCHMARKS.md)

### 3. Proposed Architecture (Method)

#### 3.1 Topological Grammar and Evolutionary Search

- Grammar execution and rule application:
  - [src/generation/grammar.py](../src/generation/grammar.py#L1324)
- Evolutionary search and graph-genotype execution:
  - [src/generation/evolutionary_director.py](../src/generation/evolutionary_director.py#L106)
  - [src/generation/evolutionary_director.py](../src/generation/evolutionary_director.py#L2290)
  - [src/generation/evolutionary_director.py](../src/generation/evolutionary_director.py#L2640)

#### 3.2 Tokenized Latent Generation (VQ-VAE + Latent Sampling)

- Discrete semantic encoder/decoder:
  - [src/core/vqvae.py](../src/core/vqvae.py#L508)
  - [src/core/vqvae.py](../src/core/vqvae.py#L606)
  - [src/core/vqvae.py](../src/core/vqvae.py#L621)

#### 3.3 Neuro-Symbolic Bridge (Weighted Bayesian WFC)

- WFC configuration and integration:
  - [src/generation/weighted_bayesian_wfc.py](../src/generation/weighted_bayesian_wfc.py#L81)
  - [src/generation/weighted_bayesian_wfc.py](../src/generation/weighted_bayesian_wfc.py#L94)
  - [src/generation/weighted_bayesian_wfc.py](../src/generation/weighted_bayesian_wfc.py#L693)

### 4. Experimental Setup and Metrics

Baseline groups:
- Pure symbolic (WFC-centric variants)
- Pure neural / reduced symbolic
- Hybrid full pipeline

Core metrics and probes:
- Solvability and progression correctness
- Pacing/tension fit error
- Expressive range and diversity
- WFC robustness diagnostics (contradiction/restart/fallback)

Code anchors:
- [src/evaluation/benchmark_suite.py](../src/evaluation/benchmark_suite.py)
- [scripts/run_ablation_study.py](../scripts/run_ablation_study.py)

### 5. Results and Discussion

Recommended result framing:
- Ablation: show which module removals degrade solvability and pacing.
- Expressive range: show diversity coverage and novelty vs reference.
- Robustness: include WFC probe metrics and failure-mode analysis.

### 6. Conclusion

Suggested closing claim:
- A tokenized neuro-symbolic stack can preserve strict solvability/progression constraints while maintaining stylistic priors under low-data conditions.

## 2) Room Generation Block (Code-Level Breakdown)

The room generator transforms mission-node constraints into a playable 2D room using discrete tile semantics.

### Step 1: Boundary Injection from Topology

- Inputs: mission graph context and neighborhood constraints.
- The pipeline prepares graph-conditioned context per room before sampling.
- Anchor path: [src/pipeline/dungeon_pipeline.py](../src/pipeline/dungeon_pipeline.py#L570)

### Step 2: Neural Prior (Latent to Tile Probabilities)

- Latent sampled and decoded into tile logits/probability structure.
- Main generation entry:
  - [src/pipeline/dungeon_pipeline.py](../src/pipeline/dungeon_pipeline.py#L363)
- VQ-VAE decode side:
  - [src/core/vqvae.py](../src/core/vqvae.py#L621)

Conceptual output tensor shape is room-height x room-width x vocabulary-size for categorical tile beliefs.

### Step 3: Symbolic Refiner (Weighted Bayesian WFC)

- Hard constraints and adjacency validity are enforced during repair/refinement.
- Pipeline repair toggle and call site:
  - [src/pipeline/dungeon_pipeline.py](../src/pipeline/dungeon_pipeline.py#L513)
- Weighted WFC implementation:
  - [src/generation/weighted_bayesian_wfc.py](../src/generation/weighted_bayesian_wfc.py)

### Step 4: Semantic Decoupling via Entity Spawner

- After shell/layout validity, entities are injected on valid traversable tiles.
- Entity spawner entrypoints:
  - [src/generation/entity_spawner.py](../src/generation/entity_spawner.py#L70)
  - [src/generation/entity_spawner.py](../src/generation/entity_spawner.py#L119)
  - [src/generation/entity_spawner.py](../src/generation/entity_spawner.py#L648)

## 3) Practical Submission Checklist

- Fix manuscript to a single canonical architecture story from [docs/INDEX.md](INDEX.md).
- Use one benchmark protocol and report template from [docs/SOTA_COMPARISON_AND_BENCHMARKS.md](SOTA_COMPARISON_AND_BENCHMARKS.md).
- Include ablation table + expressive range figure + robustness probe table.
- Explicitly separate deterministic guarantees from learned style priors in claims.
