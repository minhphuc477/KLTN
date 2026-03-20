# Ablation Study Report

## Configurations
- `FULL`: {'name': 'FULL', 'use_evolution': True, 'use_wfc': True, 'logic_guidance_scale': 1.0, 'latent_sampler': 'diffusion', 'categorical_codebook_size': None, 'use_tpe': True}
- `NO_EVOLUTION`: {'name': 'NO_EVOLUTION', 'use_evolution': False, 'use_wfc': True, 'logic_guidance_scale': 1.0, 'latent_sampler': 'diffusion', 'categorical_codebook_size': None, 'use_tpe': True}

## Summary Metrics

      config  n  success_rate  solvability_rate  confusion_ratio  confusion_index  path_optimal  tile_prior_kl  graph_edit_distance  generation_time_sec  novelty  reconstruction_error  constraint_valid_rate  room_repair_rate  tiles_repaired  diversity
        FULL  1           1.0               1.0              NaN      1999.066667           0.0       2.583446             0.348503            295.94395 0.097268              0.960227                    0.0             0.375           446.0        0.0
NO_EVOLUTION  0           0.0               0.0              NaN              NaN           0.0            NaN                  NaN                  NaN      NaN                   NaN                    NaN               NaN             NaN        0.0

## Paired Significance (vs FULL)

_No paired comparisons available_