# Protocol vs Baselines

## Claim status

- can_claim_surpasses_publications: `False`
- The fixed-graph protocol measures room-generation stability on a fixed mission graph, not the full topology-generation task used by the matched-budget and PCG Benchmark baselines.
- The strongest external rows are mixed rather than dominant across all benchmark problems, especially on diversity and smaller-problem controlability.
- Strict puzzle/no-puzzle and no-fallback ablations are internal evidence, not direct external matched-budget replacements.

## Fixed-graph protocol

- `diffusion_cfg3_logic0_steps50`: repair=0.8333, overwrite=0.1528, pre_anchor=4.1250, post_anchor=0.0000, A*=1.0000, softlock=1.0000, gauntlet=1.0000, CBS=0.6667, CBS_confusion=9.4889, time=49.04s
- `fast_cfg3_logic0_steps4`: repair=0.8611, overwrite=0.2222, pre_anchor=6.0000, post_anchor=0.0000, A*=0.6667, softlock=1.0000, gauntlet=1.0000, CBS=0.6667, CBS_confusion=3.7889, time=82.22s
- `masked_room_full`: repair=0.7778, overwrite=0.2222, pre_anchor=6.0000, post_anchor=0.0000, A*=1.0000, softlock=1.0000, gauntlet=1.0000, CBS=1.0000, CBS_confusion=27.4122, time=62.84s
- `fast_cfg3_logic0_steps4_no_fallback`: repair=0.9444, overwrite=0.1250, pre_anchor=3.3750, post_anchor=0.0000, A*=1.0000, softlock=1.0000, gauntlet=1.0000, CBS=1.0000, CBS_confusion=4.5778, time=6.05s
- `masked_room_full_no_fallback`: repair=0.5278, overwrite=0.2639, pre_anchor=7.1250, post_anchor=0.0000, A*=0.6667, softlock=1.0000, gauntlet=1.0000, CBS=0.0000, CBS_confusion=nan, time=4.39s
- `diffusion_cfg3_logic0_steps50_pure_neural`: repair=0.8333, overwrite=0.1806, pre_anchor=4.8750, post_anchor=4.8750, A*=0.0000, softlock=0.6667, gauntlet=1.0000, CBS=0.3333, CBS_confusion=nan, time=49.60s
- `fast_cfg3_logic0_steps4_pure_neural_no_fallback`: repair=0.9444, overwrite=0.1250, pre_anchor=3.3750, post_anchor=3.3750, A*=0.3333, softlock=0.6667, gauntlet=1.0000, CBS=0.3333, CBS_confusion=5.1136, time=6.00s
- `masked_room_full_pure_neural_no_fallback`: repair=0.5278, overwrite=0.2639, pre_anchor=7.1250, post_anchor=7.1250, A*=0.3333, softlock=1.0000, gauntlet=1.0000, CBS=0.3333, CBS_confusion=5.5682, time=4.37s
- `diffusion_cfg3_logic0_steps50_no_puzzle`: repair=0.8333, overwrite=0.1250, pre_anchor=3.3750, post_anchor=0.0000, A*=1.0000, softlock=1.0000, gauntlet=1.0000, CBS=0.6667, CBS_confusion=9.4889, time=50.24s
- `fast_cfg3_logic0_steps4_no_puzzle`: repair=0.8611, overwrite=0.1111, pre_anchor=3.0000, post_anchor=0.0000, A*=1.0000, softlock=1.0000, gauntlet=1.0000, CBS=0.6667, CBS_confusion=3.7000, time=81.31s
- `masked_room_full_no_puzzle`: repair=0.6944, overwrite=0.2222, pre_anchor=6.0000, post_anchor=0.0000, A*=1.0000, softlock=1.0000, gauntlet=1.0000, CBS=1.0000, CBS_confusion=20.1541, time=49.83s

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
