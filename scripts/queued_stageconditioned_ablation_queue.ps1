# Sequential queue for stageconditioned semantics ablation variants
# Runs diffusion -> fast_sampler -> masked_room for each variant sequentially.

$repoRoot = "F:\KLTN"
Set-Location $repoRoot
$env:PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128'
$env:KLTN_EXPORT_SEQUENTIAL='1'
$env:KLTN_EXPORT_MAX_BATCH_SIZE='1'

$cmds = @(
    # tokens_only: tokens enabled, topology disabled
    @{name='stageconditioned_tokens_only'; cmds=@(
        "python main.py train --config configs\zelda_hmolqd.yaml --stage diffusion --output-dir outputs\zelda_hmolqd_downstream_stageconditioned_semantics_tokens_only_v2 --diffusion-vqvae-checkpoint outputs\vqvae_ablation_codebook512_v2\checkpoints\vqvae\vqvae_pretrained.pth --diffusion-puzzle-structure-dropout-prob 0.35 --diffusion-puzzle-stage-conditioning-enabled --diffusion-puzzle-stage-token-scale 0.20 --no-diffusion-puzzle-stage-topology-enabled --diffusion-puzzle-stage-semantics-loss-weight 0.25 --diffusion-puzzle-stage-semantics-hidden-dim 96 --diffusion-puzzle-stage-semantics-max-sequence-length 6 --seed 42 --no-auto-resume --verbose",
        "python main.py train --config configs\zelda_hmolqd.yaml --stage fast_sampler --output-dir outputs\zelda_hmolqd_downstream_stageconditioned_semantics_tokens_only_v2 --fast-sampler-base-diffusion-checkpoint outputs\zelda_hmolqd_downstream_stageconditioned_semantics_tokens_only_v2\checkpoints\diffusion\best_model.pth --fast-sampler-puzzle-structure-dropout-prob 0.35 --fast-sampler-puzzle-stage-conditioning-enabled --fast-sampler-puzzle-stage-token-scale 0.20 --no-fast-sampler-puzzle-stage-topology-enabled --fast-sampler-puzzle-stage-semantics-loss-weight 0.25 --fast-sampler-puzzle-stage-semantics-hidden-dim 96 --fast-sampler-puzzle-stage-semantics-max-sequence-length 6 --seed 42 --no-auto-resume --verbose",
        "python main.py train --config configs\zelda_hmolqd.yaml --stage masked_room --output-dir outputs\zelda_hmolqd_downstream_stageconditioned_semantics_tokens_only_v2 --masked-room-puzzle-structure-dropout-prob 0.35 --masked-room-puzzle-stage-conditioning-enabled --masked-room-puzzle-stage-token-scale 0.20 --no-masked-room-puzzle-stage-topology-enabled --masked-room-puzzle-stage-semantics-loss-weight 0.25 --masked-room-puzzle-stage-semantics-hidden-dim 96 --masked-room-puzzle-stage-semantics-max-sequence-length 6 --seed 42 --no-auto-resume --verbose"
    )},

    # trace_only: topology enabled, token conditioning disabled
    @{name='stageconditioned_trace_only'; cmds=@(
        "python main.py train --config configs\zelda_hmolqd.yaml --stage diffusion --output-dir outputs\zelda_hmolqd_downstream_stageconditioned_semantics_trace_only_v2 --diffusion-vqvae-checkpoint outputs\vqvae_ablation_codebook512_v2\checkpoints\vqvae\vqvae_pretrained.pth --diffusion-puzzle-structure-dropout-prob 0.35 --no-diffusion-puzzle-stage-conditioning-enabled --diffusion-puzzle-stage-topology-enabled --diffusion-puzzle-stage-trace-decay 0.75 --diffusion-puzzle-stage-semantics-loss-weight 0.25 --diffusion-puzzle-stage-semantics-hidden-dim 96 --diffusion-puzzle-stage-semantics-max-sequence-length 6 --seed 42 --no-auto-resume --verbose",
        "python main.py train --config configs\zelda_hmolqd.yaml --stage fast_sampler --output-dir outputs\zelda_hmolqd_downstream_stageconditioned_semantics_trace_only_v2 --fast-sampler-base-diffusion-checkpoint outputs\zelda_hmolqd_downstream_stageconditioned_semantics_trace_only_v2\checkpoints\diffusion\best_model.pth --fast-sampler-puzzle-structure-dropout-prob 0.35 --no-fast-sampler-puzzle-stage-conditioning-enabled --fast-sampler-puzzle-stage-topology-enabled --fast-sampler-puzzle-stage-trace-decay 0.75 --fast-sampler-puzzle-stage-semantics-loss-weight 0.25 --fast-sampler-puzzle-stage-semantics-hidden-dim 96 --fast-sampler-puzzle-stage-semantics-max-sequence-length 6 --seed 42 --no-auto-resume --verbose",
        "python main.py train --config configs\zelda_hmolqd.yaml --stage masked_room --output-dir outputs\zelda_hmolqd_downstream_stageconditioned_semantics_trace_only_v2 --masked-room-puzzle-structure-dropout-prob 0.35 --no-masked-room-puzzle-stage-conditioning-enabled --masked-room-puzzle-stage-topology-enabled --masked-room-puzzle-stage-trace-decay 0.75 --masked-room-puzzle-stage-semantics-loss-weight 0.25 --masked-room-puzzle-stage-semantics-hidden-dim 96 --masked-room-puzzle-stage-semantics-max-sequence-length 6 --seed 42 --no-auto-resume --verbose"
    )},

    # loss010: semantics loss weight 0.10
    @{name='stageconditioned_loss010'; cmds=@(
        "python main.py train --config configs\zelda_hmolqd.yaml --stage diffusion --output-dir outputs\zelda_hmolqd_downstream_stageconditioned_semantics_loss010_v2 --diffusion-vqvae-checkpoint outputs\vqvae_ablation_codebook512_v2\checkpoints\vqvae\vqvae_pretrained.pth --diffusion-puzzle-structure-dropout-prob 0.35 --diffusion-puzzle-stage-conditioning-enabled --diffusion-puzzle-stage-token-scale 0.20 --diffusion-puzzle-stage-topology-enabled --diffusion-puzzle-stage-trace-decay 0.75 --diffusion-puzzle-stage-semantics-loss-weight 0.10 --diffusion-puzzle-stage-semantics-hidden-dim 96 --diffusion-puzzle-stage-semantics-max-sequence-length 6 --seed 42 --no-auto-resume --verbose",
        "python main.py train --config configs\zelda_hmolqd.yaml --stage fast_sampler --output-dir outputs\zelda_hmolqd_downstream_stageconditioned_semantics_loss010_v2 --fast-sampler-base-diffusion-checkpoint outputs\zelda_hmolqd_downstream_stageconditioned_semantics_loss010_v2\checkpoints\diffusion\best_model.pth --fast-sampler-puzzle-structure-dropout-prob 0.35 --fast-sampler-puzzle-stage-conditioning-enabled --fast-sampler-puzzle-stage-token-scale 0.20 --fast-sampler-puzzle-stage-topology-enabled --fast-sampler-puzzle-stage-trace-decay 0.75 --fast-sampler-puzzle-stage-semantics-loss-weight 0.10 --fast-sampler-puzzle-stage-semantics-hidden-dim 96 --fast-sampler-puzzle-stage-semantics-max-sequence-length 6 --seed 42 --no-auto-resume --verbose",
        "python main.py train --config configs\zelda_hmolqd.yaml --stage masked_room --output-dir outputs\zelda_hmolqd_downstream_stageconditioned_semantics_loss010_v2 --masked-room-puzzle-structure-dropout-prob 0.35 --masked-room-puzzle-stage-conditioning-enabled --masked-room-puzzle-stage-token-scale 0.20 --masked-room-puzzle-stage-topology-enabled --masked-room-puzzle-stage-trace-decay 0.75 --masked-room-puzzle-stage-semantics-loss-weight 0.10 --masked-room-puzzle-stage-semantics-hidden-dim 96 --masked-room-puzzle-stage-semantics-max-sequence-length 6 --seed 42 --no-auto-resume --verbose"
    )},

    # loss050: semantics loss weight 0.50
    @{name='stageconditioned_loss050'; cmds=@(
        "python main.py train --config configs\zelda_hmolqd.yaml --stage diffusion --output-dir outputs\zelda_hmolqd_downstream_stageconditioned_semantics_loss050_v2 --diffusion-vqvae-checkpoint outputs\vqvae_ablation_codebook512_v2\checkpoints\vqvae\vqvae_pretrained.pth --diffusion-puzzle-structure-dropout-prob 0.35 --diffusion-puzzle-stage-conditioning-enabled --diffusion-puzzle-stage-token-scale 0.20 --diffusion-puzzle-stage-topology-enabled --diffusion-puzzle-stage-trace-decay 0.75 --diffusion-puzzle-stage-semantics-loss-weight 0.50 --diffusion-puzzle-stage-semantics-hidden-dim 96 --diffusion-puzzle-stage-semantics-max-sequence-length 6 --seed 42 --no-auto-resume --verbose",
        "python main.py train --config configs\zelda_hmolqd.yaml --stage fast_sampler --output-dir outputs\zelda_hmolqd_downstream_stageconditioned_semantics_loss050_v2 --fast-sampler-base-diffusion-checkpoint outputs\zelda_hmolqd_downstream_stageconditioned_semantics_loss050_v2\checkpoints\diffusion\best_model.pth --fast-sampler-puzzle-structure-dropout-prob 0.35 --fast-sampler-puzzle-stage-conditioning-enabled --fast-sampler-puzzle-stage-token-scale 0.20 --fast-sampler-puzzle-stage-topology-enabled --fast-sampler-puzzle-stage-trace-decay 0.75 --fast-sampler-puzzle-stage-semantics-loss-weight 0.50 --fast-sampler-puzzle-stage-semantics-hidden-dim 96 --fast-sampler-puzzle-stage-semantics-max-sequence-length 6 --seed 42 --no-auto-resume --verbose",
        "python main.py train --config configs\zelda_hmolqd.yaml --stage masked_room --output-dir outputs\zelda_hmolqd_downstream_stageconditioned_semantics_loss050_v2 --masked-room-puzzle-structure-dropout-prob 0.35 --masked-room-puzzle-stage-conditioning-enabled --masked-room-puzzle-stage-token-scale 0.20 --masked-room-puzzle-stage-topology-enabled --masked-room-puzzle-stage-trace-decay 0.75 --masked-room-puzzle-stage-semantics-loss-weight 0.50 --masked-room-puzzle-stage-semantics-hidden-dim 96 --masked-room-puzzle-stage-semantics-max-sequence-length 6 --seed 42 --no-auto-resume --verbose"
    )}
)

foreach ($branch in $cmds) {
    Write-Host "=== Running branch: $($branch.name) ==="
    foreach ($c in $branch.cmds) {
        Write-Host "CMD: $c"
        Invoke-Expression $c
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Command failed with exit code $LASTEXITCODE; aborting queue."
            exit $LASTEXITCODE
        }
    }
}

Write-Host "All queued stageconditioned ablations completed."
