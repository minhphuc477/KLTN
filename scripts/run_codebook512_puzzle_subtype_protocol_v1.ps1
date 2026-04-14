param(
    [string]$OutputDir = "outputs/zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1",
    [string]$VqvaeCheckpoint = "outputs/vqvae_ablation_codebook512_v1/checkpoints/vqvae/vqvae_pretrained.pth"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$CommandArgs
    )

    $logDir = Join-Path $OutputDir "protocol_logs"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $logFile = Join-Path $logDir ($Name + ".log")
    $cmdText = "python " + ($CommandArgs -join " ")
    "[$(Get-Date -Format o)] START $cmdText" | Tee-Object -FilePath $logFile -Append | Out-Host
    & python @CommandArgs 2>&1 | Tee-Object -FilePath $logFile -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Step '$Name' failed with exit code $LASTEXITCODE"
    }
    "[$(Get-Date -Format o)] DONE  $cmdText" | Tee-Object -FilePath $logFile -Append | Out-Host
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Invoke-Step "train_diffusion" @(
    "main.py","train",
    "--config","configs/zelda_hmolqd.yaml",
    "--stage","diffusion",
    "--output-dir",$OutputDir,
    "--diffusion-vqvae-checkpoint",$VqvaeCheckpoint,
    "--no-auto-resume",
    "--verbose"
)

Invoke-Step "train_fast_sampler" @(
    "main.py","train",
    "--config","configs/zelda_hmolqd.yaml",
    "--stage","fast_sampler",
    "--output-dir",$OutputDir,
    "--no-auto-resume",
    "--verbose"
)

Invoke-Step "train_masked_room" @(
    "main.py","train",
    "--config","configs/zelda_hmolqd.yaml",
    "--stage","masked_room",
    "--output-dir",$OutputDir,
    "--no-auto-resume",
    "--verbose"
)

Invoke-Step "compare_manual_default" @(
    "main.py","topology-compare-manual",
    "--run-dir",$OutputDir,
    "--output-dir",(Join-Path $OutputDir "protocol_manual_compare_default"),
    "--seed","20260411"
)

Invoke-Step "audit_hybrid_default" @(
    "main.py","topology-audit-fixed-graph",
    "--run-dir",$OutputDir,
    "--output-dir",(Join-Path $OutputDir "protocol_ablation_hybrid_default"),
    "--seeds","20260404","20260405"
)

Invoke-Step "audit_hybrid_weak_decode" @(
    "main.py","topology-audit-fixed-graph",
    "--run-dir",$OutputDir,
    "--output-dir",(Join-Path $OutputDir "protocol_ablation_hybrid_weak_decode"),
    "--seeds","20260404","20260405",
    "--semantic-marker-logit-bias","12",
    "--semantic-marker-suppression-bias","3"
)

Invoke-Step "audit_neural_only_constrained_decode" @(
    "main.py","topology-audit-fixed-graph",
    "--run-dir",$OutputDir,
    "--output-dir",(Join-Path $OutputDir "protocol_ablation_neural_only_constrained_decode"),
    "--seeds","20260404","20260405",
    "--no-deterministic-graph-marker-overlay-enabled"
)

Invoke-Step "audit_strict_pure_neural" @(
    "main.py","topology-audit-fixed-graph",
    "--run-dir",$OutputDir,
    "--output-dir",(Join-Path $OutputDir "protocol_ablation_strict_pure_neural"),
    "--seeds","20260404","20260405",
    "--no-semantic-constrained-decoding-enabled",
    "--no-deterministic-graph-marker-overlay-enabled"
)

Invoke-Step "audit_strict_pure_neural_no_fallback" @(
    "main.py","topology-audit-fixed-graph",
    "--run-dir",$OutputDir,
    "--output-dir",(Join-Path $OutputDir "protocol_ablation_strict_pure_neural_no_fallback"),
    "--seeds","20260404","20260405",
    "--no-semantic-constrained-decoding-enabled",
    "--no-deterministic-graph-marker-overlay-enabled",
    "--no-fast-sampler-teacher-fallback-enabled"
)
