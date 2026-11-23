# Local Research Evidence Runbook

This runbook defines exact local training and verification commands for **G-LMDM** (`masked_room` primary discrete generator) and continuous `VQ-VAE + UNet/DiT` comparators under the locked `RG1–RG4 -> PS1–PS4 -> RO1–RO4 -> RQ1–RQ4 -> H1–H4` protocol.

## 1. Environment & Contract Check

```bash
python scripts/validate_publication_readiness.py --card docs/publication_guidance_card.json
python scripts/check_training_hyperparameters.py --config configs/zelda_hmolqd.yaml --output results/training_preflight --probe-data
python -m pytest tests/test_config_system.py tests/test_publication_protocol.py tests/test_neural_pipeline.py -q
```

## 2. Train Primary Model (`G-LMDM`) & Comparators

Use [`configs/zelda_hmolqd.yaml`](../configs/zelda_hmolqd.yaml) as the canonical source of truth (`masked_room`: `condition_gnn_type=gps`, `logic_net_enabled=true`, `alpha_logic=0.1`, `3.58M` total training / `3.36M` inference):

```bash
# Canonical G-LMDM (G-LG: GraphGPS+RRWP + Direct 16x11 Soft Bellman-Ford LogicNet)
python -m src.train_masked_room --config configs/zelda_hmolqd.yaml

# Ablation G without LogicNet (configs/zelda_hmolqd_masked_small.yaml)
python -m src.train_masked_room --config configs/zelda_hmolqd_masked_small.yaml

# Continuous VQ-VAE + UNet[1,2] / DiT+Flow Comparators (RQ2 / H2)
python -m src.train_vqvae --config configs/zelda_hmolqd.yaml
python -m src.train_diffusion --config configs/zelda_hmolqd.yaml
```

For quick 2-batch smoke runs:

```bash
python -m src.train_masked_room --config configs/zelda_hmolqd.yaml --quick
python -m src.train_diffusion --config configs/zelda_hmolqd.yaml --quick
```

## 3. Four-Stage Attribution & A* Solvability Evidence

Run hard oracle checks and Schema 2.0 4-stage attribution (`raw -> constrained -> repaired -> compiled`):

```bash
python scripts/validate_training_setup.py
python scripts/run_generated_graph_full_pipeline_eval.py --output-dir results/local_generated_graph_eval
python scripts/run_fixed_graph_multi_seed_audit.py --output-dir results/local_fixed_graph_audit
```

Core paper metrics to report (Dungeon-9 holdout, paired seeds):

- Stage-separated `raw_oracle_solved_rate`, `post_oracle_solved_rate` (`StateSpaceAStar`).
- Symbolic edit burden: `repair_rate` and `tiles_repaired_mean`.
- Rare-tile recall (`ENEMY`, `STAIR`, `BLOCK`, `ELEMENT`), pairwise diversity, controllability error, and `ms/room`.
- Failure classes kept separate: syntax invalid, unreachable, resource lock, timeout, indeterminate.

## 4. Conditioning, LogicNet & Repair Ablations (`B1`, `B4`, `B5`)

```bash
python scripts/run_conditioning_logicnet_repair_ablation.py --execute --output-dir results/local_logicnet_ablation
python scripts/run_ablation_study.py --output-dir results/ablation_suite
```

Report paired deltas (`N`, `No-RRWP`, `G`, `G-L`, `G-LG`, inference-only `logic_guidance_scale`, and external `Random`, `WFC`, `Grammar-only` baselines) at both `raw` (`apply_repair=false`) and `compiled` (`apply_repair=true`) stages.

## 5. P-CBS Bounded-Agent Proxy Diagnostics (Never Equated With Formal Proofs)

Run P-CBS only on oracle-solved maps:

```bash
python scripts/run_pcbs_persona_map_sweep.py --data-root "Data/The Legend of Zelda" --output-dir results/pcbs_persona_map_sweep --oracle-solved-only
python scripts/run_pcbs_component_ablation.py --data-root "Data/The Legend of Zelda" --output-dir results/pcbs_component_ablation
```

Report cognitive gap rate, confusion index, navigation entropy, cognitive load, and puzzle stall fraction strictly under `proxy_only` labeling unless calibrated against recruited human telemetry (`scripts/calibrate_pcbs_personas_from_telemetry.py`).

## 6. Minimum Evidence Bundle For Publication

Store under one timestamped result directory:

- Frozen config snapshot, git commit hash, and Dungeon 1–8 / Dungeon-9 manifest hashes.
- Checkpoint SHA-256 hashes and parameter counts.
- Schema 2.0 per-attempt records across `raw`, `constrained`, `repaired`, and `compiled` stages.
- Paired Holm-corrected non-parametric summary tables (`StateSpaceAStar` proofs separate from `P-CBS` proxy reports).
