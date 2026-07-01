# Learned Stage Puzzle Semantics Upgrade

Last updated: 2026-04-19

This note records the code-side upgrade that moves multi-step puzzle semantics
from a runtime-only hybrid scaffold into an explicit train/runtime conditioning
and semantic-supervision path.

## What Changed

The repo now has a shared `puzzle_stage_condition` contract.

It is built from validator-aware room semantics and reused in:

- room dataset graph samples
- diffusion graph conditioning
- masked-room graph conditioning
- runtime per-room graph context
- optional room-topology traversability priors
- an auxiliary learned puzzle-stage semantics head for:
  - diffusion
  - masked-room
  - fast-sampler distillation

Main code:

- `src/pipeline/room_topology_conditioning.py`
- `src/zelda_data/zelda_loader.py`
- `src/train_diffusion.py`
- `src/train_masked_room.py`
- `src/train_lcm.py`
- `src/pipeline/dungeon_pipeline.py`

## What It Learns

The new conditioning path can encode ordered room-local stages such as:

- `collect_key`
- `collect_item`
- `defeat_enemy`
- `push_block_to_switch`
- `step_on_puzzle`
- `reach_exit`

The conditioning path is deterministic and shape-safe:

- no conditioner parameter shapes were changed
- ordered stage tokens are appended to the conditioning sequence
- room-topology priors can optionally carry ordered stage traces
- generated room logits can now be supervised directly against:
  - gate family
  - sequence-required flag
  - stage count
  - ordered stage slots

## Why This Is The Correct Upgrade

Research direction that supports this change:

- graph-conditioned generation works better when structure is explicit rather
  than left implicit in the generator:
  - Graph2Plan: <https://arxiv.org/abs/2004.13204>
  - HouseDiffusion: <https://arxiv.org/abs/2211.13287>
- puzzle generation literature supports explicit state-space structure rather
  than hoping pure neural priors discover multi-step semantics reliably:
  - Procedural Generation of Initial States of Sokoban:
    <https://arxiv.org/abs/1907.02548>
- persona playtesting prior art still means P-CBS should be claimed as a
  bounded-rational validator, not as the first persona playtester:
  - Holmgard et al.:
    <https://arxiv.org/abs/1802.06881>

## Honest Claim Boundary

This patch closes the `code gap`, not the `evidence gap`.

After this patch, it is fair to say:

- the repo now contains a learned-stage-conditioning plus semantic-supervision
  path for multi-step puzzle semantics
- older checkpoints are outdated for any claim about learned staged puzzle
  semantics

It is still not fair to say:

- the current published artifacts prove learned staged puzzle semantics
- the model surpasses prior publications

Those claims still require:

1. retraining the staged-puzzle branch
2. rerunning the fixed-graph protocol
3. rerunning matched-budget baselines
4. rerunning the long P-CBS benchmark if persona evidence is needed

## Minimal Next Actions

1. Train:
   - `outputs/zelda_hmolqd_downstream_stageconditioned_semantics_v2`
2. Export:
   - `protocol_manual_compare_stageconditioned_semantics_v2`
   - `protocol_ablation_stageconditioned_semantics_v2`
3. Compare:
   - against `codebook512_puzzle_subtype_v1`
   - against matched-budget topology baselines
4. Only after those runs:
   - update the report claim boundary
   - decide whether the staged branch replaces the current thesis-safe branch
