# GPU Validation

The default GitHub Actions workflow remains CPU-only. It proves import,
configuration, and functional regressions on the portable path, but it does not
prove CUDA kernel compatibility or mixed-precision stability.

Use the manual `GPU smoke` workflow on a self-hosted runner labeled `gpu` after
dependency upgrades and before collecting final experiment evidence. The
workflow executes:

```powershell
python -m pytest tests/test_cuda_smoke.py -q
```

The CUDA smoke suite checks the differentiable soft-min path in FP16 and a
small latent-diffusion denoiser forward pass under CUDA autocast. Full training
and evaluation runs remain separate evidence-producing workloads.
