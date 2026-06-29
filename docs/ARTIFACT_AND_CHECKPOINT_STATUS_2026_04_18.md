# Artifact And Checkpoint Status

Last updated: 2026-04-18

This file is the canonical management note for `outputs/` and `results/`.
Its purpose is operational clarity:

- which artifact families are current
- which are experimental only
- which are stale / superseded
- which patches require only re-export
- which patches require retraining

No checkpoint or output folder was deleted while preparing this note.

## Executive Assessment

The current thesis-safe checkpoint family is:

- tokenizer evidence:
  `outputs/vqvae_ablation_codebook512_v1`
- downstream branch:
  `outputs/zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1`

The current report-facing protocol artifacts for that branch are:

- manual compare:
  `outputs/zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1/protocol_manual_compare_statefulmultistep_v23`
- fixed-graph audit:
  `outputs/zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1/protocol_ablation_statefulmultistep_v23`
- baseline comparison:
  `outputs/zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1/protocol_ablation_statefulmultistep_v23/baseline_comparison`

Everything older than the above protocol paths should be treated as historical
evidence, not as the latest state of the stack.

## Status Legend

- `Current`: valid and preferred for current report / thesis use
- `Experimental`: valid but not the canonical final branch
- `Superseded`: historically useful, but not the latest evidence
- `Incomplete`: do not use for conclusions yet

## Output Family Assessment

| Path family | Status | Use it for | Alert |
|---|---|---|---|
| `outputs/vqvae_ablation_codebook512_v1` | Current | strongest tested tokenizer checkpoint family | Best tokenizer evidence, but still an ablation family rather than the YAML default |
| `outputs/zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1/checkpoints` | Current | canonical thesis-safe downstream checkpoints | Preferred room-generation checkpoint family |
| `outputs/zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1/protocol_ablation_statefulmultistep_v23` | Current | latest fixed-graph current-code protocol evidence | Preferred branch-comparison artifact on the patched stack |
| `outputs/zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1/protocol_manual_compare_statefulmultistep_v23` | Current but rerun recommended | latest manual side-by-side export path | Generated before the 2026-04-18 graph-role alias / strict-JSON fixes; regenerate before citing it as final evidence |
| `outputs/zelda_hmolqd_downstream_puzzle_structure_control_v1` | Experimental | training-time puzzle-structure control ablation family | Useful research branch, not the current best final branch |
| `outputs/zelda_hmolqd_puzzlecookbook_pdrop015_v1` | Incomplete | partial puzzle-dropout cookbook work | Only masked-room checkpoints are present; not a complete family |
| `outputs/zelda_hmolqd_puzzlecookbook_pdrop035_v1` | Incomplete | partial puzzle-dropout cookbook work | Only masked-room checkpoints are present; not a complete family |
| `outputs/zelda_hmolqd_puzzlecookbook_pdrop055_v1` | Incomplete | placeholder/incomplete cookbook run | No checkpoints present |
| `outputs/zelda_hmolqd_downstream_codebook512_v1` | Superseded | historical comparison only | Older downstream branch without the later puzzle-subtype/current-code stack |
| `outputs/zelda_hmolqd_downstream_codebook512_fastfix_v1` | Superseded | historical fast-sampler repair evidence | Do not use as latest branch judgment |
| `outputs/zelda_hmolqd_aux_topofocus_v1` | Superseded | auxiliary training history | Useful provenance only |
| `outputs/zelda_best_model_showcase_*` | Superseded | presentation / visual showcase | Not the canonical evidence path |
| `outputs/puzzle_scaffold_manual_nodes_v*` | Superseded | local scaffold debugging / iteration | Keep only as debugging history |
| `outputs/pytest_tmp_*` | Superseded | test scratch | Not report material |

## Results Family Assessment

| Path family | Status | Use it for | Alert |
|---|---|---|---|
| `results/stateful_puzzle_hparam_sweep_v2` | Current | current runtime puzzle-profile comparison | `baseline_default` is still the best profile |
| `results/matched_budget_topology_v1` | Current but not final-synchronized | matched-budget topology baseline reference | Still valid, but not rerun on every latest code-side runtime patch |
| `results/pcg_benchmark_alignment_v2` | Current but not final-synchronized | external PCG Benchmark alignment | Still valid, but not evidence of publication-surpassing performance |
| `results/room_branch_benchmark_quick_v2` | Current quick evidence | room-branch comparison | Quick slice only, not full final benchmark |
| `results/ablation_core_quick_v3` and `results/ablation_core_quick_part2_v1` | Current quick evidence | core-architecture ablations | Good support, but still quick-budget evidence |
| `results/pcbs_component_ablation_*_v3` | Current | P-CBS component-ablation evidence | Good for thesis claim boundary |
| `results/cbs_benchmark_levels1_9_variants12_all_personas_v2*` | Superseded / interim | historical long-form CBS benchmark | Predates latest benchmark-accounting cleanup; do not use as final publication-grade table |
| `results/pcbs_vs_astar_report_v2` | Current smoke/reference | one-off A* vs P-CBS report example | Good illustration, not the final full benchmark |

## Checkpoint Assessment

### Canonical checkpoints to keep in active use

Tokenizer:

- `outputs/vqvae_ablation_codebook512_v1/checkpoints/vqvae/...`

Canonical downstream branch:

- `outputs/zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1/checkpoints/diffusion/best_model.pth`
- `outputs/zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1/checkpoints/fast_sampler/fast_sampler_best.pth`
- `outputs/zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1/checkpoints/masked_room/masked_room_best.pth`

### Experimental checkpoints to keep, but not treat as final

- `outputs/zelda_hmolqd_downstream_puzzle_structure_control_v1/checkpoints/...`

Reason:

- this family is the only direct evidence for training-time puzzle-structure
  control
- it is not the current best overall final branch

### Partial / incomplete checkpoint families

- `outputs/zelda_hmolqd_puzzlecookbook_pdrop015_v1/checkpoints/...`
- `outputs/zelda_hmolqd_puzzlecookbook_pdrop035_v1/checkpoints/...`
- `outputs/zelda_hmolqd_puzzlecookbook_pdrop055_v1`

Clear alert:

- these are not complete cookbook families yet
- do not cite them as final tuning evidence

## Retraining Alerts

### Patches that do **not** require retraining

These are runtime / validator / reporting changes. Existing checkpoints can use
them immediately; you only need to rerun exports and evaluations.

- hybrid mechanical contract adoption
- graph-guided oracle reporting
- strict JSON export fixes
- OOM-safe export retry/backoff
- stateful multi-step puzzle validation / gating
- runtime interaction-sequence puzzle grammar
- report/markdown artifact sanitization

### Patches that **do** require retraining if you want the model itself to learn them

1. `puzzle_structure_dropout_prob` training-time conditioning

   Reason:

   - this changes the training signal so the network can explicitly learn
     puzzle-on / puzzle-off control
   - runtime stripping alone does not give you a learned conditional model

   Relevant branch:

   - `outputs/zelda_hmolqd_downstream_puzzle_structure_control_v1`

2. `puzzle_stage_conditioning_enabled` and `puzzle_stage_topology_enabled`
   staged-puzzle training path

   Reason:

   - this is the new train-time route for learned ordered multi-step puzzle
     semantics
   - it appends ordered puzzle-stage control tokens to graph conditioning
   - it can inject ordered stage traces into `room_topology_map`
   - old checkpoints do not contain evidence for this claim

   Clear alert:

   - every checkpoint trained before the 2026-04-18 staged-puzzle conditioning
     patch is `outdated for any claim about learned multi-step puzzle
     semantics`
   - those checkpoints remain usable for the current hybrid system, but not for
     the stronger learned-semantics claim

3. `puzzle_stage_semantics_loss_weight > 0`
   explicit semantic-supervision path

   Reason:

   - this is the stronger 2026-04-19 upgrade beyond token/trace conditioning
   - it adds a learned auxiliary head that predicts:
     - gate family
     - sequence-required flag
     - stage count
     - ordered stage slots
     from generated room logits during training
   - checkpoints trained before this patch do not support the stronger
     `explicitly supervised learned multi-step puzzle semantics` claim

4. Any future claim that the network itself learned staged multi-step puzzle
   semantics instead of relying only on hybrid runtime grammar

   Reason:

   - before the 2026-04-18 patch, staged multi-step mechanics were hybrid and
     validator-backed only
   - after the patch, the train-time route exists, but it still needs retrained
     checkpoints and rerun evidence

### Retraining recommendation, honestly stated

Current code does **not** force immediate retraining to remain usable.

However, retraining is recommended if the thesis or paper wants to claim:

- learned puzzle-structure control rather than runtime control
- learned staged multi-step puzzle semantics rather than hybrid grammar-backed
  semantics

## New Outdated-Checkpoint Alert

The following checkpoint families are now `architecturally outdated for the new
learned staged-puzzle claim` unless they are retrained with the new flags:

- `outputs/zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1/checkpoints/...`
- `outputs/zelda_hmolqd_downstream_puzzle_structure_control_v1/checkpoints/...`
- `outputs/zelda_hmolqd_downstream_stageconditioned_v1/checkpoints/...`
- all current `puzzlecookbook_*` families

Why:

- they can still run the hybrid runtime puzzle system
- they did not learn the new explicit semantic-supervision path
- they should not be described as evidence of `fully learned multi-step puzzle
  semantics`

## Re-Export Alerts

These folders are not invalid checkpoints, but their older report artifacts are
now stale because the code and validation contract moved forward:

- any protocol export under
  `outputs/zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1`
  older than:
  - `protocol_manual_compare_statefulmultistep_v23`
  - `protocol_ablation_statefulmultistep_v23`
- any protocol export under
  `outputs/zelda_hmolqd_downstream_puzzle_structure_control_v1`
  older than its latest comparison/audit folders

If you use those checkpoint families again, regenerate protocol artifacts rather
than quoting older exports.

Additional alert:

- `protocol_manual_compare_statefulmultistep_v23` should be regenerated after
  the 2026-04-18 graph-role alias and strict-JSON fixes if you want its manual
  compare outputs to count as final report evidence.

## Report-Writing Guidance

For the thesis/report, cite these as current:

- `docs/CURRENT_ARCHITECTURE.md`
- `docs/architecture_audit_research_notes.md`
- this file:
  `docs/ARTIFACT_AND_CHECKPOINT_STATUS_2026_04_18.md`
- `outputs/zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1/protocol_ablation_statefulmultistep_v23/summary.json`
- `outputs/zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1/protocol_ablation_statefulmultistep_v23/baseline_comparison/protocol_vs_baselines.md`

Do not describe these as current-final:

- older `protocol_*_v2/v3/v5/v8` exports
- `results/cbs_benchmark_levels1_9_variants12_all_personas_v2*`
- partial puzzle cookbook families

## Honest Remaining Gaps

The following gaps still exist and are not solved by documentation cleanup:

1. The current system is thesis-finalizable, but it still does **not** have
   evidence to claim it surpasses prior publications.
2. Stateful puzzle mechanics are hybrid and meaningful now, but not yet a full
   learned multi-object Sokoban-like model.
3. The long-form patched `1..9 x variants x personas` P-CBS benchmark still
   needs a clean final rerun if you want publication-grade persona tables.
4. The puzzle cookbook is not finished enough to support a strong hyperparameter
   selection claim across fully trained downstream branches.
