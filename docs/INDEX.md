# KLTN Documentation Index

Last updated: 2026-04-11

This is the single entrypoint for repository documentation. The top level of
`docs/` now contains only current reference material. Dated snapshots and
one-off research notes are archived under [`archive/`](archive/README.md).

## Canonical Start Here

1. [`../README.md`](../README.md) - repository overview, setup, and usage
2. [`../ARCHITECTURE_DIAGRAMS.md`](../ARCHITECTURE_DIAGRAMS.md) - high-level visual overview
3. [`CURRENT_ARCHITECTURE.md`](CURRENT_ARCHITECTURE.md) - current implementation map
4. [`CANONICAL_MODEL_RATIONALE_ABLATION_AND_COMPLEXITY_GUIDE.md`](CANONICAL_MODEL_RATIONALE_ABLATION_AND_COMPLEXITY_GUIDE.md) - thesis/report-ready explanation of the canonical model, config rationale, block-by-block flow, ablations, and complexity
5. [`TRACEABILITY_AND_CLAIM_VALIDATION.md`](TRACEABILITY_AND_CLAIM_VALIDATION.md) - claim-to-code traceability
6. [`TOPOLOGY_COMMANDS.md`](TOPOLOGY_COMMANDS.md) - canonical CLI guide for topology graphs, manual mission graphs, and fixed-graph audits
7. [`ARCHITECTURE_RESEARCH_AUDIT_2026_03_31.md`](ARCHITECTURE_RESEARCH_AUDIT_2026_03_31.md) - latest broad literature-backed architecture audit
8. [`ARCHITECTURE_RESEARCH_AUDIT_TOPOLOGY_SIGNAL_2026_04_04.md`](ARCHITECTURE_RESEARCH_AUDIT_TOPOLOGY_SIGNAL_2026_04_04.md) - focused topology-signal audit and semantic-anchor recommendations
9. [`STATEFUL_PUZZLE_ARCHITECTURE_AUDIT_2026_04_09.md`](STATEFUL_PUZZLE_ARCHITECTURE_AUDIT_2026_04_09.md) - research-backed audit and implementation notes for stateful puzzle templates, hidden repair assumptions, and runtime puzzle semantics
10. [`VQVAE_RESEARCH_AUDIT_2026_04_10.md`](VQVAE_RESEARCH_AUDIT_2026_04_10.md) - focused Block II audit covering literature, assumptions, complexity, validation protocol, and whether the current tokenizer actually needs an upgrade
11. [`VQVAE_PROTOCOL_RESULTS_2026_04_10.md`](VQVAE_PROTOCOL_RESULTS_2026_04_10.md) - held-out comparison of all completed VQ-VAE ablations and the current recommendation for which tokenizer to carry forward
12. [`DOWNSTREAM_CODEBOOK512_PROTOCOL_RESULTS_2026_04_11.md`](DOWNSTREAM_CODEBOOK512_PROTOCOL_RESULTS_2026_04_11.md) - end-to-end judgment of the downstream retrain performed on top of the codebook512 tokenizer

## Architecture And Core System Docs

- [`BLOCK_BY_BLOCK_ARCHITECTURE_AND_IMPLEMENTATION_AUDIT.md`](BLOCK_BY_BLOCK_ARCHITECTURE_AND_IMPLEMENTATION_AUDIT.md)
- [`BLOCK_IO_REFERENCE.md`](BLOCK_IO_REFERENCE.md)
- [`CBS_ARCHITECTURE.md`](CBS_ARCHITECTURE.md)
- [`GRAMMAR_REFERENCE.md`](GRAMMAR_REFERENCE.md)
- [`SOLVERS_AND_GUI_REFERENCE.md`](SOLVERS_AND_GUI_REFERENCE.md)
- [`TOPOLOGY_COMMANDS.md`](TOPOLOGY_COMMANDS.md)
- [`VGLC_COMPLIANCE_GUIDE.md`](VGLC_COMPLIANCE_GUIDE.md)
- [`ZELDA_SOLVER_DOCUMENTATION.md`](ZELDA_SOLVER_DOCUMENTATION.md)

## Benchmarking, Evaluation, And Research Guides

- [`SOTA_COMPARISON_AND_BENCHMARKS.md`](SOTA_COMPARISON_AND_BENCHMARKS.md)
- [`ARCHITECTURE_RESEARCH_AUDIT_2026_03_31.md`](ARCHITECTURE_RESEARCH_AUDIT_2026_03_31.md)
- [`ARCHITECTURE_RESEARCH_AUDIT_TOPOLOGY_SIGNAL_2026_04_04.md`](ARCHITECTURE_RESEARCH_AUDIT_TOPOLOGY_SIGNAL_2026_04_04.md)
- [`STATEFUL_PUZZLE_ARCHITECTURE_AUDIT_2026_04_09.md`](STATEFUL_PUZZLE_ARCHITECTURE_AUDIT_2026_04_09.md)
- [`VQVAE_RESEARCH_AUDIT_2026_04_10.md`](VQVAE_RESEARCH_AUDIT_2026_04_10.md)
- [`VQVAE_PROTOCOL_RESULTS_2026_04_10.md`](VQVAE_PROTOCOL_RESULTS_2026_04_10.md)
- [`DOWNSTREAM_CODEBOOK512_PROTOCOL_RESULTS_2026_04_11.md`](DOWNSTREAM_CODEBOOK512_PROTOCOL_RESULTS_2026_04_11.md)
- [`REALISM_PROFILE_WEIGHTING_AND_TUNING_GUIDE.md`](REALISM_PROFILE_WEIGHTING_AND_TUNING_GUIDE.md)
- [`MATCHED_BUDGET_BENCHMARK_PROTOCOL.md`](MATCHED_BUDGET_BENCHMARK_PROTOCOL.md)
- [`OOD_SCALING_AND_BLINDED_EVAL_PROTOCOL.md`](OOD_SCALING_AND_BLINDED_EVAL_PROTOCOL.md)
- [`IEEE_TOG_BLUEPRINT_AND_ROOM_GENERATION.md`](IEEE_TOG_BLUEPRINT_AND_ROOM_GENERATION.md)

## Supporting READMEs

- [`../src/pipeline/README.md`](../src/pipeline/README.md)
- [`../src/gui/README.md`](../src/gui/README.md)
- [`../kaggle/README.md`](../kaggle/README.md)

## Archived Snapshots

- [`archive/README.md`](archive/README.md)
- [`archive/2026-q1/README.md`](archive/2026-q1/README.md)
