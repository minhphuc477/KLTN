# KLTN Documentation Index

Last updated: 2026-11-12

This is the canonical navigation page for `docs/`. Top-level files should be
current references, executable protocols, or domain guides.

## Start Here

1. [`../README.md`](../README.md)
2. [`SOTA_COMPARISON_AND_BENCHMARKS.md`](SOTA_COMPARISON_AND_BENCHMARKS.md) (external literature baseline table, `B1–B4 / A0–A4` Q1 matrix, $d_{\text{NN}}$ audit & `RG1–RG4`)
3. [`PUBLICATION_GUIDANCE_AND_CARDS.md`](PUBLICATION_GUIDANCE_AND_CARDS.md) (`RG1–RG4`, `PS1–PS4`, `RO1–RO4`, `RQ1–RQ4`, `H1–H4`)
4. [`../PUBLICATION_BLUEPRINT.md`](../PUBLICATION_BLUEPRINT.md) (operational confirmatory protocol & Section 13/28 claim gates)
5. [`../refine-logs/EXPERIMENT_PLAN.md`](../refine-logs/EXPERIMENT_PLAN.md) & [`../refine-logs/EXPERIMENT_TRACKER.md`](../refine-logs/EXPERIMENT_TRACKER.md) (`B1–B5`, `M0–M4`, `R001–R017`)
6. [`TRACEABILITY_AND_CLAIM_VALIDATION.md`](TRACEABILITY_AND_CLAIM_VALIDATION.md) (end-to-end `RG -> PS -> RO -> RQ -> H -> Block -> Config -> Metric` matrix)
7. [`CURRENT_ARCHITECTURE.md`](CURRENT_ARCHITECTURE.md) & [`architecture_audit_research_notes.md`](architecture_audit_research_notes.md)
8. [`LOCAL_RESEARCH_EVIDENCE_RUNBOOK.md`](LOCAL_RESEARCH_EVIDENCE_RUNBOOK.md)

## Publication Contract

- Canonical scientific framing (`RG1–RG4`, `PS1–PS4`, `RO1–RO4`, `RQ1–RQ4`, `H1–H4`),
  system boundary, claim language, data card, artifact card, metric contract,
  baseline taxonomy, failure taxonomy, reproducibility checklist, and IP note:
  [`PUBLICATION_GUIDANCE_AND_CARDS.md`](PUBLICATION_GUIDANCE_AND_CARDS.md)
- Machine-readable publication card:
  [`publication_guidance_card.json`](publication_guidance_card.json)
- Operational confirmatory evaluation & statistical design:
  [`../PUBLICATION_BLUEPRINT.md`](../PUBLICATION_BLUEPRINT.md)
- Validator:
  [`../scripts/validate_publication_readiness.py`](../scripts/validate_publication_readiness.py)

## Current Protocols And Runbooks

- Canonical dependency-ordered training commands:
  [`TRAINING_COMMANDS_IN_ORDER.md`](TRAINING_COMMANDS_IN_ORDER.md)
- Local research evidence runbook:
  [`LOCAL_RESEARCH_EVIDENCE_RUNBOOK.md`](LOCAL_RESEARCH_EVIDENCE_RUNBOOK.md)
- Kaggle notebook suite README:
  [`../kaggle/README.md`](../kaggle/README.md) & [`../kaggle/hmolqd_training_suite/README.md`](../kaggle/hmolqd_training_suite/README.md)
- PCG benchmark, literature gaps (`RG1–RG4`), and external comparison protocol:
  [`SOTA_COMPARISON_AND_BENCHMARKS.md`](SOTA_COMPARISON_AND_BENCHMARKS.md)
- Topology-QD archive generation and exact final-map materialization commands:
  [`TOPOLOGY_COMMANDS.md`](TOPOLOGY_COMMANDS.md)

## Architecture And Rationale

- Live block map and canonical config/backbone summary:
  [`CURRENT_ARCHITECTURE.md`](CURRENT_ARCHITECTURE.md)
- Report-writing rationale, ablations, and complexity analysis:
  [`CANONICAL_MODEL_RATIONALE_ABLATION_AND_COMPLEXITY_GUIDE.md`](CANONICAL_MODEL_RATIONALE_ABLATION_AND_COMPLEXITY_GUIDE.md)
- Current non-GUI audit ledger and publication claim boundary:
  [`architecture_audit_research_notes.md`](architecture_audit_research_notes.md)

## Reference Docs

- [`BLOCK_IO_REFERENCE.md`](BLOCK_IO_REFERENCE.md)
- [`GRAMMAR_REFERENCE.md`](GRAMMAR_REFERENCE.md)
- [`SOLVERS_AND_GUI_REFERENCE.md`](SOLVERS_AND_GUI_REFERENCE.md)
- [`VGLC_COMPLIANCE_GUIDE.md`](VGLC_COMPLIANCE_GUIDE.md)
- [`experiment_tracking.md`](experiment_tracking.md)
- [`gpu_validation.md`](gpu_validation.md)
- [`reproducibility_versioning.md`](reproducibility_versioning.md)

## Active Scripts By Role

- **Preflight & Publication Validation**:
  [`../scripts/check_training_hyperparameters.py`](../scripts/check_training_hyperparameters.py),
  [`../scripts/validate_publication_readiness.py`](../scripts/validate_publication_readiness.py),
  [`../scripts/validate_training_setup.py`](../scripts/validate_training_setup.py),
  [`../scripts/validate_all_features.py`](../scripts/validate_all_features.py),
  [`../scripts/audit_block0_data.py`](../scripts/audit_block0_data.py)
- **Latent Calibration & Split Locking (`RO1` / `B1–B3`)**:
  [`../scripts/calibrate_vqvae_latent_scale.py`](../scripts/calibrate_vqvae_latent_scale.py),
  [`../scripts/lock_dataset_split.py`](../scripts/lock_dataset_split.py),
  [`../scripts/train_logicnet_tile_classifier.py`](../scripts/train_logicnet_tile_classifier.py)
- **Causal Conditioning, LogicNet & Four-Stage Ablations (`RO2–RO3` / `B4`)**:
  [`../scripts/run_ablation_study.py`](../scripts/run_ablation_study.py),
  [`../scripts/run_conditioning_logicnet_repair_ablation.py`](../scripts/run_conditioning_logicnet_repair_ablation.py),
  [`../scripts/compare_logic_loss_ab.py`](../scripts/compare_logic_loss_ab.py),
  [`../scripts/run_fast_sampler_visual_audit.py`](../scripts/run_fast_sampler_visual_audit.py)
- **Held-Out Generalization, Topology Bounds & Oracle Separation (`RO4` / `B5` & Block I/VII)**:
  [`../scripts/run_dungeon9_holdout_protocol.py`](../scripts/run_dungeon9_holdout_protocol.py),
  [`../scripts/run_matched_budget_topology_benchmark.py`](../scripts/run_matched_budget_topology_benchmark.py),
  [`../scripts/run_ood_scaling_and_blinded_eval.py`](../scripts/run_ood_scaling_and_blinded_eval.py),
  [`../scripts/evaluate_pcbs_vs_astar.py`](../scripts/evaluate_pcbs_vs_astar.py),
  [`../scripts/statistical_validation_and_analysis.py`](../scripts/statistical_validation_and_analysis.py),
  [`../scripts/run_priority_research_suite.py`](../scripts/run_priority_research_suite.py)

Note: validation helpers live under `scripts/validate_*.py` and pytest tests under `tests/`.
