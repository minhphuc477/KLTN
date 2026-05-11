# Protocol vs Baselines

## Claim status

- can_claim_surpasses_publications: `False`
- The fixed-graph protocol measures room-generation stability on a fixed mission graph, not the full topology-generation task used by the matched-budget and PCG Benchmark baselines.
- The strongest external rows are mixed rather than dominant across all benchmark problems, especially on diversity and smaller-problem controlability.
- Strict puzzle/no-puzzle and no-fallback ablations are internal evidence, not direct external matched-budget replacements.

## Fixed-graph protocol

- `diffusion_cfg3_logic0_steps50`: repair=0.8889, overwrite=0.2361, room_unique=1.0000, room_pairwise_ncd=0.5879, room_ref_ncd=0.6255, pre_anchor=6.3750, post_anchor=0.0000, A*=0.0000, graph_guided=1.0000, softlock=1.0000, gauntlet=1.0000, hybrid_contract=1.0000, CBS=0.6667, CBS_confusion=n/a, time=183.53s
- `fast_cfg3_logic0_steps4`: repair=0.8333, overwrite=0.2222, room_unique=1.0000, room_pairwise_ncd=0.5791, room_ref_ncd=0.6238, pre_anchor=6.0000, post_anchor=0.0000, A*=0.0000, graph_guided=1.0000, softlock=1.0000, gauntlet=1.0000, hybrid_contract=1.0000, CBS=0.6667, CBS_confusion=n/a, time=175.13s
- `masked_room_full`: repair=0.8056, overwrite=0.0972, room_unique=1.0000, room_pairwise_ncd=0.5863, room_ref_ncd=0.5977, pre_anchor=2.6250, post_anchor=0.0000, A*=0.0000, graph_guided=1.0000, softlock=1.0000, gauntlet=1.0000, hybrid_contract=1.0000, CBS=0.6667, CBS_confusion=n/a, time=132.34s

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
