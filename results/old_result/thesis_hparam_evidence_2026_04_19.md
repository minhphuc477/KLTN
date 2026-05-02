# Thesis Hyperparameter Evidence Summary

Generated from local artifacts on 2026-04-19T12:03:08+07:00.

## Protocol Alerts

- Diffusion branches currently running were launched before the held-out diffusion validation patch added on 2026-04-19.
- Completed VQ-VAE ablations use a deterministic 10% hold-out split and support tokenizer screening, but only single-seed results are available so far.
- For thesis-final Chapter 4 claims, rerun diffusion branches under the patched trainer and keep final comparisons on the fixed-graph multi-seed audit, not on in-training validation alone.

## VQ-VAE Ablations

| Variant | Codebook | Hidden | CoordConv | MRF | Best Epoch | Best Val Loss | Val Perplexity | Utilization | EMA Live Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline (256, CoordConv, MRF) | 256 | 96 | on | 0.05 | 294 | 6.93475e-05 | 40.4765 | 1 | 0.996094 |
| Codebook-128 | 128 | 96 | on | 0.05 | 161 | 8.82146e-05 | 35.8883 | 1 | 1 |
| Codebook-512 | 512 | 96 | on | 0.05 | 241 | 0.00252178 | 42.3387 | 1 | 0.896484 |
| Hidden-64 | 256 | 64 | on | 0.05 | 119 | 7.46959e-05 | 40.7899 | 1 | 0.996094 |
| No CoordConv | 256 | 96 | off | 0.05 | 203 | 0.0031889 | 39.9514 | 1 | 0.996094 |
| No MRF | 256 | 96 | on | 0 | 187 | 0.000233655 | 40.5697 | 1 | 0.996094 |

Best validation loss among the six completed VQ-VAE runs: **Baseline (256, CoordConv, MRF)** at epoch 294 with val_loss=6.93475e-05.

## Diffusion Interim Status

| Branch | Tokenizer | Best Epoch | Latest Epoch | Best Val Total | Best Val Diff | Best Val Logic | Best Solvability Proxy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline branch diffusion | Baseline VQ-VAE | 1 | 5 | 0.853 | 0.853 | 1.2086 | 0.2986 |
| Codebook-512 branch diffusion | Codebook-512 VQ-VAE | 4 | 4 | 0.8587 | 0.8587 | 1.1838 | 0.3061 |

These diffusion values are **interim only** because the runs were started before diffusion adopted a held-out validation split.
