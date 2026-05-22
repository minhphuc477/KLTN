# VQ-VAE-2 LogicNet Repair Ablation Protocol

Last updated: 2026-05-23

This protocol records the new experiment wiring added after reading the local
architecture and current literature. It is code-first: all heavy experiments are
defined now, but execution is left for a later compute run.

## Research Basis

- VQ-VAE-2 uses hierarchical discrete latents: a coarse top codebook for global
  structure and a bottom codebook for local detail.
- Value Iteration Networks and neural logic/program-style modules support the
  idea of differentiable planning guidance, but the evidence must show whether
  the guide improves generated validity before repair.
- Neural Logic Machines show why relational/logic modules should be tested on
  generalization and compositional structure, not only reconstruction loss.
- Pereira's dungeon work motivates explicit room/key/lock/linearity target
  tables and large-room stress discussion.
- P-CBS/persona-style validation is useful only when reported beside an oracle
  and calibrated as bounded-rationality effort rather than hard validity.

Sources:

- <https://arxiv.org/abs/1906.00446>
- <https://arxiv.org/abs/1711.00937>
- <https://arxiv.org/abs/1602.02867>
- <https://arxiv.org/abs/1904.11694>
- <https://antoniosliapis.com/papers/procedural_personas_as_critics_for_dungeon_generation.pdf>
- <https://repositorio.usp.br/item/002964434>
- <https://repositorio.usp.br/item/003032388>

## Implemented Code Hooks

- `src/core/vqvae.py`
  - adds `SemanticVQVAE2`
  - preserves `encode`, `decode`, `decode_indices`, `compute_loss`, and
    `get_codebook_usage` compatibility
  - exposes top/bottom codebook metrics
- `src/train_vqvae.py`
  - accepts `--architecture vqvae2`
  - records runtime, epoch-to-best, validation loss, codebook utilization, and
    checkpoint size in `vqvae_run_summary.json`
- `src/train_diffusion.py` and `src/pipeline/dungeon_pipeline.py`
  - load VQ-VAE/VQ-VAE-2 architecture metadata instead of assuming the
    single-level tokenizer
- `scripts/run_conditioning_logicnet_repair_ablation.py`
  - defines the full conditioning/LogicNet/repair paired matrix
- `src/evaluation/pcbs_validation.py`
  - adds readability, bounded-rationality, effort, and oracle/P-CBS delta fields
- `src/pipeline/dungeon_pipeline.py`
  - records `repair_count` and `repair_time_sec`

## VQ-VAE-2 Training Commands

Plan the single-level baseline and hierarchical tokenizer as paired Block-II
runs:

```powershell
python src\train_vqvae.py `
  --config configs\zelda_hmolqd.yaml `
  --architecture vqvae `
  --output-dir outputs\vqvae_baseline_v1
```

```powershell
python src\train_vqvae.py `
  --config configs\zelda_hmolqd.yaml `
  --architecture vqvae2 `
  --top-codebook-size 128 `
  --top-latent-dim 64 `
  --output-dir outputs\vqvae2_hierarchical_v1
```

Each run must export:

- `runtime_sec` / `wall_time_sec`
- `best_epoch` / `epoch_to_best`
- `val_loss`
- aggregate `codebook_utilization`
- `top_codebook_utilization`
- `bottom_codebook_utilization`
- `checkpoint_size_bytes` / `checkpoint_size_mb`

## Conditioning, LogicNet, Repair Matrix

Plan-only command:

```powershell
python scripts\run_conditioning_logicnet_repair_ablation.py `
  --output results\conditioning_logicnet_repair_ablation
```

Execute later:

```powershell
python scripts\run_conditioning_logicnet_repair_ablation.py `
  --execute `
  --config configs\zelda_hmolqd.yaml `
  --output results\conditioning_logicnet_repair_ablation `
  --seeds 42,43,44 `
  --vqvae-checkpoint outputs\vqvae2_hierarchical_v1\checkpoints\vqvae\vqvae_pretrained.pth `
  --diffusion-checkpoint outputs\YOUR_DIFFUSION_RUN\checkpoints\diffusion\best_model.pth `
  --logic-net-checkpoint outputs\YOUR_DIFFUSION_RUN\checkpoints\diffusion\best_model.pth
```

Variants:

- `full`
- `no_graph_tokens`
- `no_stage_tokens`

Each variant is crossed with:

- repair disabled/enabled
- LogicNet guidance disabled/enabled

Outputs:

- `conditioning_logicnet_repair_rows.csv`
- `conditioning_logicnet_repair_summary.csv`
- `conditioning_logicnet_repair_payload.json`
- `visual_sheet.png`
- `visual_sheet_manifest.json`

Required reported fields:

- pre-repair A* validity
- post-repair A* validity
- pre-repair P-CBS validity
- post-repair P-CBS validity
- pre/post semantic counts for start, goal, keys, locks, puzzles, enemies
- repair count
- repair time
- total tiles repaired
- LogicNet dungeon and room solvability
- readability / bounded-rationality / cognitive-effort indices

## LogicNet Improvement Target

Current code has two LogicNet paths:

- `src/core/logic_net.py`: integrated differentiable grid and dungeon-scope
  guidance path used by the pipeline.
- `src/ml/logic_net.py`: older standalone grid-oriented implementation.

The immediate improvement is not another architecture rewrite. The missing
evidence is a paired ON/OFF run proving whether LogicNet improves pre-repair
semantics and validity. If the ON/OFF table only improves post-repair validity,
the repair layer is doing the work and the LogicNet claim must be narrowed.

## P-CBS Reporting Upgrade

P-CBS now reports:

- `bounded_rationality_index`
- `readability_score`
- `cognitive_effort_index`
- `oracle_pcbs_path_delta`
- seed and timeout

These are derived fields for reporting. The raw P-CBS metrics remain in the
payload for statistical analysis and future calibration against human traces.
