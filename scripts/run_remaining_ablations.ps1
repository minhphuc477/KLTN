$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot
$python = Join-Path $repoRoot ".venv-1\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}
$env:PYTORCH_CUDA_ALLOC_CONF = 'max_split_size_mb:128'
$env:KLTN_EXPORT_SEQUENTIAL = '1'
$env:KLTN_EXPORT_MAX_BATCH_SIZE = '1'

Write-Host "=== Polling for trace_only diffusion completion ==="
$meta_file = 'outputs\zelda_hmolqd_downstream_stageconditioned_semantics_trace_only_v2\checkpoints\diffusion\latest_resume.pth.meta.json'
$final_file = 'outputs\zelda_hmolqd_downstream_stageconditioned_semantics_trace_only_v2\checkpoints\diffusion\final_model.pth'

$check_count = 0
while ($true) {
    $check_count += 1
    if (Test-Path $final_file) {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Diffusion final_model found"
        break
    }
    if (Test-Path $meta_file) {
        $meta = Get-Content $meta_file -Raw | ConvertFrom-Json
        $epoch = $meta.extra.epoch
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Check #$check_count - epoch $epoch/100"
        if ($epoch -ge 100) {
            Write-Host "Epoch >= 100, breaking"
            break
        }
    }
    Start-Sleep -Seconds 120
}

Write-Host "=== Launching trace_only fast_sampler ==="
& $python main.py train --config configs\zelda_hmolqd.yaml --stage fast_sampler --output-dir outputs\zelda_hmolqd_downstream_stageconditioned_semantics_trace_only_v2 --fast-sampler-base-diffusion-checkpoint outputs\zelda_hmolqd_downstream_stageconditioned_semantics_trace_only_v2\checkpoints\diffusion\best_model.pth --fast-sampler-puzzle-structure-dropout-prob 0.35 --no-fast-sampler-puzzle-stage-conditioning-enabled --fast-sampler-puzzle-stage-topology-enabled --fast-sampler-puzzle-stage-trace-decay 0.75 --fast-sampler-puzzle-stage-semantics-loss-weight 0.25 --fast-sampler-puzzle-stage-semantics-hidden-dim 96 --fast-sampler-puzzle-stage-semantics-max-sequence-length 6 --seed 42 --no-auto-resume --verbose

Write-Host "=== Launching trace_only masked_room ==="
& $python main.py train --config configs\zelda_hmolqd.yaml --stage masked_room --output-dir outputs\zelda_hmolqd_downstream_stageconditioned_semantics_trace_only_v2 --masked-room-puzzle-structure-dropout-prob 0.35 --no-masked-room-puzzle-stage-conditioning-enabled --masked-room-puzzle-stage-topology-enabled --masked-room-puzzle-stage-trace-decay 0.75 --masked-room-puzzle-stage-semantics-loss-weight 0.25 --masked-room-puzzle-stage-semantics-hidden-dim 96 --masked-room-puzzle-stage-semantics-max-sequence-length 6 --seed 42 --no-auto-resume --verbose

Write-Host "=== trace_only COMPLETE ==="
Write-Host "=== Launching loss010 diffusion ==="
& $python main.py train --config configs\zelda_hmolqd.yaml --stage diffusion --output-dir outputs\zelda_hmolqd_downstream_stageconditioned_semantics_loss010_v2 --diffusion-vqvae-checkpoint outputs\vqvae_ablation_codebook512_v2\checkpoints\vqvae\vqvae_pretrained.pth --diffusion-puzzle-structure-dropout-prob 0.35 --diffusion-puzzle-stage-conditioning-enabled --diffusion-puzzle-stage-token-scale 0.20 --diffusion-puzzle-stage-topology-enabled --diffusion-puzzle-stage-trace-decay 0.75 --diffusion-puzzle-stage-semantics-loss-weight 0.10 --diffusion-puzzle-stage-semantics-hidden-dim 96 --diffusion-puzzle-stage-semantics-max-sequence-length 6 --seed 42 --no-auto-resume --verbose

Write-Host "=== loss010 diffusion complete - launching fast_sampler ==="
& $python main.py train --config configs\zelda_hmolqd.yaml --stage fast_sampler --output-dir outputs\zelda_hmolqd_downstream_stageconditioned_semantics_loss010_v2 --fast-sampler-base-diffusion-checkpoint outputs\zelda_hmolqd_downstream_stageconditioned_semantics_loss010_v2\checkpoints\diffusion\best_model.pth --fast-sampler-puzzle-structure-dropout-prob 0.35 --fast-sampler-puzzle-stage-conditioning-enabled --fast-sampler-puzzle-stage-token-scale 0.20 --fast-sampler-puzzle-stage-topology-enabled --fast-sampler-puzzle-stage-trace-decay 0.75 --fast-sampler-puzzle-stage-semantics-loss-weight 0.10 --fast-sampler-puzzle-stage-semantics-hidden-dim 96 --fast-sampler-puzzle-stage-semantics-max-sequence-length 6 --seed 42 --no-auto-resume --verbose

Write-Host "=== loss010 fast_sampler complete - launching masked_room ==="
& $python main.py train --config configs\zelda_hmolqd.yaml --stage masked_room --output-dir outputs\zelda_hmolqd_downstream_stageconditioned_semantics_loss010_v2 --masked-room-puzzle-structure-dropout-prob 0.35 --masked-room-puzzle-stage-conditioning-enabled --masked-room-puzzle-stage-token-scale 0.20 --masked-room-puzzle-stage-topology-enabled --masked-room-puzzle-stage-trace-decay 0.75 --masked-room-puzzle-stage-semantics-loss-weight 0.10 --masked-room-puzzle-stage-semantics-hidden-dim 96 --masked-room-puzzle-stage-semantics-max-sequence-length 6 --seed 42 --no-auto-resume --verbose

Write-Host "=== loss010 COMPLETE ==="
Write-Host "=== Launching loss050 diffusion ==="
& $python main.py train --config configs\zelda_hmolqd.yaml --stage diffusion --output-dir outputs\zelda_hmolqd_downstream_stageconditioned_semantics_loss050_v2 --diffusion-vqvae-checkpoint outputs\vqvae_ablation_codebook512_v2\checkpoints\vqvae\vqvae_pretrained.pth --diffusion-puzzle-structure-dropout-prob 0.35 --diffusion-puzzle-stage-conditioning-enabled --diffusion-puzzle-stage-token-scale 0.20 --diffusion-puzzle-stage-topology-enabled --diffusion-puzzle-stage-trace-decay 0.75 --diffusion-puzzle-stage-semantics-loss-weight 0.50 --diffusion-puzzle-stage-semantics-hidden-dim 96 --diffusion-puzzle-stage-semantics-max-sequence-length 6 --seed 42 --no-auto-resume --verbose

Write-Host "=== loss050 diffusion complete - launching fast_sampler ==="
& $python main.py train --config configs\zelda_hmolqd.yaml --stage fast_sampler --output-dir outputs\zelda_hmolqd_downstream_stageconditioned_semantics_loss050_v2 --fast-sampler-base-diffusion-checkpoint outputs\zelda_hmolqd_downstream_stageconditioned_semantics_loss050_v2\checkpoints\diffusion\best_model.pth --fast-sampler-puzzle-structure-dropout-prob 0.35 --fast-sampler-puzzle-stage-conditioning-enabled --fast-sampler-puzzle-stage-token-scale 0.20 --fast-sampler-puzzle-stage-topology-enabled --fast-sampler-puzzle-stage-trace-decay 0.75 --fast-sampler-puzzle-stage-semantics-loss-weight 0.50 --fast-sampler-puzzle-stage-semantics-hidden-dim 96 --fast-sampler-puzzle-stage-semantics-max-sequence-length 6 --seed 42 --no-auto-resume --verbose

Write-Host "=== loss050 fast_sampler complete - launching masked_room ==="
& $python main.py train --config configs\zelda_hmolqd.yaml --stage masked_room --output-dir outputs\zelda_hmolqd_downstream_stageconditioned_semantics_loss050_v2 --masked-room-puzzle-structure-dropout-prob 0.35 --masked-room-puzzle-stage-conditioning-enabled --masked-room-puzzle-stage-token-scale 0.20 --masked-room-puzzle-stage-topology-enabled --masked-room-puzzle-stage-trace-decay 0.75 --masked-room-puzzle-stage-semantics-loss-weight 0.50 --masked-room-puzzle-stage-semantics-hidden-dim 96 --masked-room-puzzle-stage-semantics-max-sequence-length 6 --seed 42 --no-auto-resume --verbose

Write-Host "=== loss050 COMPLETE ==="
Write-Host "=== ALL STAGECONDITIONED ABLATIONS FINISHED ==="
