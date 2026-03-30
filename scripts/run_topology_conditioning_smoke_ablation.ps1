param(
    [string]$ConfigPath = "configs/zelda_hmolqd.yaml",
    [string]$BaseOutputDir = "artifacts/ablations/topology_conditioning_smoke",
    [int]$Seed = 1337,
    [int]$BatchSize = 4,
    [int]$VqvaeEpochs = 2,
    [int]$MinSamplesPerEpoch = 64,
    [int]$DiffusionValidationSamples = 2
)

$ErrorActionPreference = "Stop"

function Invoke-Training {
    param(
        [string[]]$Arguments
    )

    Write-Host ""
    Write-Host ">>> python $($Arguments -join ' ')" -ForegroundColor Cyan
    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Training command failed with exit code $LASTEXITCODE"
    }
}

$config = Resolve-Path $ConfigPath
$baseDir = Join-Path (Get-Location) $BaseOutputDir
$vqvaeDir = Join-Path $baseDir "vqvae_bootstrap"
$additiveDir = Join-Path $baseDir "additive"
$spadeDir = Join-Path $baseDir "spade"
$vqvaeCheckpoint = Join-Path $vqvaeDir "checkpoints\vqvae\vqvae_pretrained.pth"

New-Item -ItemType Directory -Force -Path $baseDir | Out-Null

Invoke-Training @(
    "main.py", "train",
    "--config", $config,
    "--stage", "vqvae",
    "--output-dir", $vqvaeDir,
    "--seed", "$Seed",
    "--batch-size", "$BatchSize",
    "--min-samples-per-epoch", "$MinSamplesPerEpoch",
    "--vqvae-epochs", "$VqvaeEpochs",
    "--vqvae-save-every", "1"
)

if (-not (Test-Path $vqvaeCheckpoint)) {
    throw "Expected VQ-VAE checkpoint was not created: $vqvaeCheckpoint"
}

foreach ($mode in @("additive", "spade")) {
    $runDir = if ($mode -eq "additive") { $additiveDir } else { $spadeDir }
    Invoke-Training @(
        "main.py", "train",
        "--config", $config,
        "--stage", "diffusion",
        "--output-dir", $runDir,
        "--seed", "$Seed",
        "--batch-size", "$BatchSize",
        "--quick",
        "--diffusion-vqvae-checkpoint", $vqvaeCheckpoint,
        "--diffusion-condition-gnn-type", "gps",
        "--diffusion-topology-conditioning-mode", $mode,
        "--diffusion-validation-num-samples", "$DiffusionValidationSamples"
    )
}

Write-Host ""
Write-Host ">>> Summarizing diffusion runs" -ForegroundColor Cyan
@'
from pathlib import Path
import json
import torch

base = Path(r"__BASE_OUTPUT_DIR__")
results = []
for mode in ("additive", "spade"):
    run_dir = base / mode
    ckpt_path = run_dir / "checkpoints" / "diffusion" / "best_model.pth"
    metrics_path = run_dir / "checkpoints" / "diffusion" / "logs" / "diffusion_training_metrics.json"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint for {mode}: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    best_metrics = checkpoint.get("metrics") or {}
    history = []
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as handle:
            history = json.load(handle)
    final_metrics = history[-1] if history else {}
    results.append(
        {
            "mode": mode,
            "best_epoch": best_metrics.get("epoch"),
            "best_loss": best_metrics.get("loss"),
            "best_val_logic_loss": best_metrics.get("val_logic_loss"),
            "best_val_solvability_proxy": best_metrics.get("val_solvability_proxy"),
            "final_loss": final_metrics.get("loss"),
            "final_val_logic_loss": final_metrics.get("val_logic_loss"),
            "final_val_solvability_proxy": final_metrics.get("val_solvability_proxy"),
        }
    )

for row in results:
    print(json.dumps(row, indent=2))
'@.Replace("__BASE_OUTPUT_DIR__", $baseDir.Replace("\", "\\")) | python -
