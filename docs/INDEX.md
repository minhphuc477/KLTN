# KLTN Documentation Index

Last updated: 2026-06-30

This is the canonical navigation page for `docs/`. Top-level files should be
current references, executable protocols, or stable thesis/report materials.
Superseded notes belong under `docs/archive/`.

## Start Here

1. [`../README.md`](../README.md)
2. [`architecture_audit_research_notes.md`](architecture_audit_research_notes.md)
3. [`CURRENT_ARCHITECTURE.md`](CURRENT_ARCHITECTURE.md)
4. [`TRACEABILITY_AND_CLAIM_VALIDATION.md`](TRACEABILITY_AND_CLAIM_VALIDATION.md)
5. [`LOCAL_RESEARCH_EVIDENCE_RUNBOOK.md`](LOCAL_RESEARCH_EVIDENCE_RUNBOOK.md)

## Current Protocols

- Designer controllability and compute/sample-efficiency:
  [`DESIGNER_CONTROLLABILITY_AND_COMPUTE_PROTOCOL.md`](DESIGNER_CONTROLLABILITY_AND_COMPUTE_PROTOCOL.md)
- VQ-VAE-2, conditioning, LogicNet, repair, and P-CBS ablation protocol:
  [`VQVAE2_LOGICNET_REPAIR_ABLATION_PROTOCOL_2026_05_23.md`](VQVAE2_LOGICNET_REPAIR_ABLATION_PROTOCOL_2026_05_23.md)
- Kaggle full training suite for VQ-VAE/VQ-VAE-2 and downstream stages:
  [`KAGGLE_T4_X2_TRAINING.md`](KAGGLE_T4_X2_TRAINING.md)
- Kaggle research evidence runbook:
  [`KAGGLE_RESEARCH_EVIDENCE_RUNBOOK.md`](KAGGLE_RESEARCH_EVIDENCE_RUNBOOK.md)
- Local research evidence runbook:
  [`LOCAL_RESEARCH_EVIDENCE_RUNBOOK.md`](LOCAL_RESEARCH_EVIDENCE_RUNBOOK.md)
- Training hyperparameter/batch preflight command:
  [`FULL_TRAINING_ABLATION_AND_EVAL_COMMAND_BOOK_2026_04_18.md`](FULL_TRAINING_ABLATION_AND_EVAL_COMMAND_BOOK_2026_04_18.md)
- Thesis hyperparameter evidence and protocol justification:
  [`THESIS_HYPERPARAMETER_SEARCH_AND_PROTOCOL_JUSTIFICATION_2026_04_19.md`](THESIS_HYPERPARAMETER_SEARCH_AND_PROTOCOL_JUSTIFICATION_2026_04_19.md)
- Matched-budget benchmark:
  [`MATCHED_BUDGET_BENCHMARK_PROTOCOL.md`](MATCHED_BUDGET_BENCHMARK_PROTOCOL.md)
- OOD scaling and blinded evaluation:
  [`OOD_SCALING_AND_BLINDED_EVAL_PROTOCOL.md`](OOD_SCALING_AND_BLINDED_EVAL_PROTOCOL.md)
- PCG benchmark and external comparison:
  [`SOTA_COMPARISON_AND_BENCHMARKS.md`](SOTA_COMPARISON_AND_BENCHMARKS.md)
- CNN versus Bellman-Ford LogicNet ablation:
  [`pathfinder_ablation_protocol.md`](pathfinder_ablation_protocol.md)
- Human playtest provenance, synthetic structural OOD, and QD archive heatmaps:
  [`HUMAN_PLAYTEST_AND_OOD_QD_PROTOCOL.md`](HUMAN_PLAYTEST_AND_OOD_QD_PROTOCOL.md)

## Architecture And Rationale

- Current non-GUI audit ledger and publication boundary:
  [`architecture_audit_research_notes.md`](architecture_audit_research_notes.md)
- Live block map:
  [`CURRENT_ARCHITECTURE.md`](CURRENT_ARCHITECTURE.md)
- Report-writing rationale, ablations, and complexity:
  [`CANONICAL_MODEL_RATIONALE_ABLATION_AND_COMPLEXITY_GUIDE.md`](CANONICAL_MODEL_RATIONALE_ABLATION_AND_COMPLEXITY_GUIDE.md)
- Latest block-by-block gap audit:
  [`BLOCK_ARCHITECTURE_GAP_AUDIT_2026_05_23.md`](BLOCK_ARCHITECTURE_GAP_AUDIT_2026_05_23.md)
- Repo structure and cleanup audit:
  [`REPO_STRUCTURE_AND_CLEANUP_AUDIT_2026_05_24.md`](REPO_STRUCTURE_AND_CLEANUP_AUDIT_2026_05_24.md)

## Block References

- Block I topology commands:
  [`TOPOLOGY_COMMANDS.md`](TOPOLOGY_COMMANDS.md)
- VQ-VAE audit:
  [`VQVAE_RESEARCH_AUDIT_2026_04_10.md`](VQVAE_RESEARCH_AUDIT_2026_04_10.md)
- Learned staged-puzzle semantics:
  [`LEARNED_STAGE_PUZZLE_SEMANTICS_UPGRADE_2026_04_18.md`](LEARNED_STAGE_PUZZLE_SEMANTICS_UPGRADE_2026_04_18.md)
- Search algorithm audit:
  [`SEARCH_ALGORITHM_AUDIT_AND_RECOMMENDATION_2026_04_18.md`](SEARCH_ALGORITHM_AUDIT_AND_RECOMMENDATION_2026_04_18.md)
- P-CBS novelty and component notes:
  [`PCBS_RESEARCH_AND_PUBLICATION_GAP_AUDIT_2026_05_24.md`](PCBS_RESEARCH_AND_PUBLICATION_GAP_AUDIT_2026_05_24.md)

## Reference Docs

- [`BLOCK_IO_REFERENCE.md`](BLOCK_IO_REFERENCE.md)
- [`experiment_tracking.md`](experiment_tracking.md)
- [`gpu_validation.md`](gpu_validation.md)
- [`reproducibility_versioning.md`](reproducibility_versioning.md)
- [`GRAMMAR_REFERENCE.md`](GRAMMAR_REFERENCE.md)
- [`SOLVERS_AND_GUI_REFERENCE.md`](SOLVERS_AND_GUI_REFERENCE.md)
- [`VGLC_COMPLIANCE_GUIDE.md`](VGLC_COMPLIANCE_GUIDE.md)
- [`ZELDA_SOLVER_DOCUMENTATION.md`](ZELDA_SOLVER_DOCUMENTATION.md)

## Status

- Artifact/checkpoint status:
  [`ARTIFACT_AND_CHECKPOINT_STATUS_2026_04_18.md`](ARTIFACT_AND_CHECKPOINT_STATUS_2026_04_18.md)
- Current audit and evidence boundary:
  [`architecture_audit_research_notes.md`](architecture_audit_research_notes.md)

## Archive

- [`archive/README.md`](archive/README.md)
- [`archive/2026-q1/README.md`](archive/2026-q1/README.md)
- [`archive/2026-q2/README.md`](archive/2026-q2/README.md)

Cleanup note: test-like scripts now live in `tests/` as pytest coverage or in
`scripts/validate_*.py` as manual validation tools. One-off debug probes and
stale queue wrappers have been removed from `scripts/`.
The structure details are tracked in
[`REPO_STRUCTURE_AND_CLEANUP_AUDIT_2026_05_24.md`](REPO_STRUCTURE_AND_CLEANUP_AUDIT_2026_05_24.md).
