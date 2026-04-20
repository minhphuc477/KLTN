# Protocol vs Baselines

## Claim status

- can_claim_surpasses_publications: `False`
- The fixed-graph protocol measures room-generation stability on a fixed mission graph, not the full topology-generation task used by the matched-budget and PCG Benchmark baselines.
- The strongest external rows are mixed rather than dominant across all benchmark problems, especially on diversity and smaller-problem controlability.
- Strict puzzle/no-puzzle and no-fallback ablations are internal evidence, not direct external matched-budget replacements.

## Fixed-graph protocol

- `diffusion_cfg3_logic0_steps50`: repair=0.8333, overwrite=0.2500, room_unique=1.0000, room_pairwise_ncd=0.5984, room_ref_ncd=0.6048, pre_anchor=6.7500, post_anchor=0.0000, A*=0.0000, graph_guided=1.0000, softlock=1.0000, gauntlet=1.0000, hybrid_contract=1.0000, CBS=0.3333, CBS_confusion=n/a, time=53.79s
- `fast_cfg3_logic0_steps4`: repair=0.8333, overwrite=0.1944, room_unique=1.0000, room_pairwise_ncd=0.6297, room_ref_ncd=0.6229, pre_anchor=5.2500, post_anchor=0.0000, A*=0.0000, graph_guided=1.0000, softlock=1.0000, gauntlet=1.0000, hybrid_contract=1.0000, CBS=0.6667, CBS_confusion=n/a, time=187.76s
- `masked_room_full`: repair=0.6944, overwrite=0.1667, room_unique=1.0000, room_pairwise_ncd=0.6108, room_ref_ncd=0.5798, pre_anchor=4.5000, post_anchor=0.0000, A*=0.0000, graph_guided=1.0000, softlock=1.0000, gauntlet=1.0000, hybrid_contract=1.0000, CBS=0.6667, CBS_confusion=n/a, time=103.25s

## Matched-budget topology baselines

- `best_fitness`: method=ES, fitness=0.3172, overall_completeness=1.0000, linearity=0.3688, leniency=0.6526, generation_time_sec=12.27
- `best_completeness`: method=ES, fitness=0.3172, overall_completeness=1.0000, linearity=0.3688, leniency=0.6526, generation_time_sec=12.27
- `best_novelty`: method=MAP_ELITES, fitness=0.2831, overall_completeness=1.0000, linearity=0.3142, leniency=0.5998, generation_time_sec=10.50
- `best_generation_time`: method=RANDOM, fitness=0.1924, overall_completeness=0.9531, linearity=0.2840, leniency=0.7617, generation_time_sec=7.75

## PCG Benchmark alignment

- `zelda-v0`: best_quality=CORE_GA (0.7500), best_controlability=CORE_GA (0.5000), best_diversity=FULL_CVT (0.5000)
- `zelda-enemies-v0`: best_quality=FULL_GA (0.6250), best_controlability=CORE_GA (0.3750), best_diversity=FULL_CVT (0.5000)
- `zelda-large-v0`: best_quality=FULL_GA (1.0000), best_controlability=FULL_GA (1.0000), best_diversity=FULL_GA (0.2500)

## Notes

- Use the fixed-graph protocol to compare branch quality and validation stability under the same topology.
- Use matched-budget and PCG Benchmark outputs to judge topology-generation competitiveness.
- Do not collapse those two evidence layers into one scalar; they measure different stages of the stack.
