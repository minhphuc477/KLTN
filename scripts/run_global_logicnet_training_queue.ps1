param(
    [string]$RepoRoot = "F:\KLTN",
    [string]$Python = "python",
    [string]$Suffix = "v4_global_logicnet",
    [string[]]$Branches = @("full", "tokens_only", "trace_only", "loss010", "loss050"),
    [ValidateSet("all", "train", "eval")]
    [string]$Mode = "train",
    [int]$BatchSize = 1,
    [string]$Device = "cuda",
    [int]$MinFreeGpuMb = 3300,
    [int]$PollSeconds = 300,
    [switch]$Force,
    [switch]$ResumeExisting,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Wait-GpuMemory {
    if ($Device -ne "cuda") {
        return
    }
    $nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($null -eq $nvidia) {
        Write-Host "[gpu] nvidia-smi not found; continuing without GPU queue wait."
        return
    }

    while ($true) {
        $total = (& nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | Select-Object -First 1)
        $used = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1)
        $free = [int]$total - [int]$used
        if ($free -ge $MinFreeGpuMb) {
            Write-Host "[gpu] free=${free}MiB >= ${MinFreeGpuMb}MiB; starting next step."
            return
        }
        Write-Host "[gpu] free=${free}MiB < ${MinFreeGpuMb}MiB; waiting ${PollSeconds}s."
        Start-Sleep -Seconds $PollSeconds
    }
}

function Invoke-Step {
    param(
        [string]$Label,
        [string[]]$CommandArgs,
        [switch]$NeedsGpu
    )

    Write-Host "[run] $Label"
    Write-Host "  $Python $($CommandArgs -join ' ')"
    if ($DryRun) {
        return
    }
    if ($NeedsGpu) {
        Wait-GpuMemory
    }
    & $Python @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Step '$Label' failed with exit code $LASTEXITCODE."
    }
}

function Get-BranchSpecs {
    return [ordered]@{
        full = @{
            Label = "global_logicnet_stageconditioned_$Suffix"
            OutputDir = "outputs\zelda_hmolqd_downstream_global_logicnet_$Suffix"
            Tokens = $true
            Topology = $true
            GenerationTopology = $true
            SemanticsLossWeight = "0.25"
        }
        tokens_only = @{
            Label = "global_logicnet_tokens_only_$Suffix"
            OutputDir = "outputs\zelda_hmolqd_downstream_global_logicnet_tokens_only_$Suffix"
            Tokens = $true
            Topology = $false
            GenerationTopology = $false
            SemanticsLossWeight = "0.25"
        }
        trace_only = @{
            Label = "global_logicnet_trace_only_$Suffix"
            OutputDir = "outputs\zelda_hmolqd_downstream_global_logicnet_trace_only_$Suffix"
            Tokens = $false
            Topology = $true
            GenerationTopology = $true
            SemanticsLossWeight = "0.25"
        }
        loss010 = @{
            Label = "global_logicnet_loss010_$Suffix"
            OutputDir = "outputs\zelda_hmolqd_downstream_global_logicnet_loss010_$Suffix"
            Tokens = $true
            Topology = $true
            GenerationTopology = $true
            SemanticsLossWeight = "0.10"
        }
        loss050 = @{
            Label = "global_logicnet_loss050_$Suffix"
            OutputDir = "outputs\zelda_hmolqd_downstream_global_logicnet_loss050_$Suffix"
            Tokens = $true
            Topology = $true
            GenerationTopology = $true
            SemanticsLossWeight = "0.50"
        }
    }
}

function Assert-OutputAvailable {
    param([hashtable]$Spec)

    $outputPath = Join-Path $RepoRoot $Spec.OutputDir
    if ((Test-Path $outputPath) -and (-not $Force) -and (-not $ResumeExisting)) {
        throw "Output already exists: $($Spec.OutputDir). Use -ResumeExisting to continue it or -Force to intentionally write into it."
    }
}

function Add-CommonArgs {
    param(
        [string[]]$ExtraArgs,
        [string]$Stage,
        [hashtable]$Spec
    )

    $argsOut = @(
        "main.py", "train",
        "--config", "configs\zelda_hmolqd.yaml",
        "--stage", $Stage,
        "--output-dir", $Spec.OutputDir,
        "--batch-size", "$BatchSize",
        "--device", $Device,
        "--seed", "42",
        "--verbose"
    )

    if ($ResumeExisting) {
        $argsOut += "--auto-resume"
    } else {
        $argsOut += "--no-auto-resume"
    }

    if ([bool]$Spec.GenerationTopology) {
        $argsOut += "--generation-puzzle-stage-topology-enabled"
    } else {
        $argsOut += "--no-generation-puzzle-stage-topology-enabled"
    }
    $argsOut += @("--generation-puzzle-stage-trace-decay", "0.75")
    $argsOut += $ExtraArgs
    return $argsOut
}

function Add-StagePuzzleArgs {
    param(
        [string[]]$BaseArgs,
        [string]$Prefix,
        [hashtable]$Spec
    )

    $argsOut = @($BaseArgs)
    $argsOut += @("--$Prefix-puzzle-structure-dropout-prob", "0.35")
    if ([bool]$Spec.Tokens) {
        $argsOut += "--$Prefix-puzzle-stage-conditioning-enabled"
    } else {
        $argsOut += "--no-$Prefix-puzzle-stage-conditioning-enabled"
    }
    $argsOut += @("--$Prefix-puzzle-stage-token-scale", "0.20")
    if ([bool]$Spec.Topology) {
        $argsOut += "--$Prefix-puzzle-stage-topology-enabled"
    } else {
        $argsOut += "--no-$Prefix-puzzle-stage-topology-enabled"
    }
    $argsOut += @(
        "--$Prefix-puzzle-stage-trace-decay", "0.75",
        "--$Prefix-puzzle-stage-semantics-loss-weight", $Spec.SemanticsLossWeight,
        "--$Prefix-puzzle-stage-semantics-hidden-dim", "96",
        "--$Prefix-puzzle-stage-semantics-max-sequence-length", "6"
    )
    return $argsOut
}

function Add-AllPuzzleConfigArgs {
    param(
        [string[]]$BaseArgs,
        [hashtable]$Spec
    )

    $argsOut = @($BaseArgs)
    $argsOut = Add-StagePuzzleArgs -Prefix "diffusion" -Spec $Spec -BaseArgs $argsOut
    $argsOut = Add-StagePuzzleArgs -Prefix "fast-sampler" -Spec $Spec -BaseArgs $argsOut
    $argsOut = Add-StagePuzzleArgs -Prefix "masked-room" -Spec $Spec -BaseArgs $argsOut
    return $argsOut
}

function Invoke-TrainingBranch {
    param([hashtable]$Spec)

    Assert-OutputAvailable -Spec $Spec

    $vqvaeCheckpoint = "outputs\vqvae_ablation_codebook512_v2\checkpoints\vqvae\vqvae_pretrained.pth"
    $diffusionCheckpoint = Join-Path $Spec.OutputDir "checkpoints\diffusion\best_model.pth"

    $diffArgs = Add-CommonArgs `
        -Stage "diffusion" `
        -Spec $Spec `
        -ExtraArgs @(
            "--diffusion-vqvae-checkpoint", $vqvaeCheckpoint,
            "--diffusion-logic-global-reach-weight", "1.0",
            "--diffusion-logic-global-room-weight", "0.25"
        )
    $diffArgs = Add-AllPuzzleConfigArgs -Spec $Spec -BaseArgs $diffArgs
    Invoke-Step -Label "$($Spec.Label) diffusion" -CommandArgs $diffArgs -NeedsGpu

    if ((-not $DryRun) -and (-not (Test-Path (Join-Path $RepoRoot $diffusionCheckpoint)))) {
        throw "Missing diffusion checkpoint after training: $diffusionCheckpoint"
    }

    $fastArgs = Add-CommonArgs `
        -Stage "fast_sampler" `
        -Spec $Spec `
        -ExtraArgs @("--fast-sampler-base-diffusion-checkpoint", $diffusionCheckpoint)
    $fastArgs = Add-AllPuzzleConfigArgs -Spec $Spec -BaseArgs $fastArgs
    Invoke-Step -Label "$($Spec.Label) fast_sampler" -CommandArgs $fastArgs -NeedsGpu

    $maskedArgs = Add-CommonArgs -Stage "masked_room" -Spec $Spec -ExtraArgs @()
    $maskedArgs = Add-AllPuzzleConfigArgs -Spec $Spec -BaseArgs $maskedArgs
    Invoke-Step -Label "$($Spec.Label) masked_room" -CommandArgs $maskedArgs -NeedsGpu
}

function Invoke-EvaluationBranch {
    param([hashtable]$Spec)

    $protocolDir = Join-Path $Spec.OutputDir "protocol_ablation_$($Spec.Label)"
    $manualDir = Join-Path $Spec.OutputDir "protocol_manual_compare_$($Spec.Label)"

    Invoke-Step `
        -Label "$($Spec.Label) fixed-graph audit" `
        -CommandArgs @(
            "scripts\run_fixed_graph_multi_seed_audit.py",
            "--run-dir", $Spec.OutputDir,
            "--output-dir", $protocolDir,
            "--seeds", "20260404", "20260405", "20260406"
        )

    Invoke-Step `
        -Label "$($Spec.Label) manual semantic compare" `
        -CommandArgs @(
            "scripts\export_semantic_anchor_end_to_end.py",
            "--run-dir", $Spec.OutputDir,
            "--output-dir", $manualDir,
            "--seed", "20260404"
        )
}

Set-Location $RepoRoot
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128"
$env:KLTN_EXPORT_SEQUENTIAL = "1"
$env:KLTN_EXPORT_MAX_BATCH_SIZE = "1"

$queueLog = Join-Path $RepoRoot "outputs\global_logicnet_queue_$Suffix.log"
Start-Transcript -Path $queueLog -Append | Out-Null

try {
    $specs = Get-BranchSpecs
    $selectedBranches = @()
    foreach ($branchGroup in $Branches) {
        foreach ($branchName in ([Regex]::Split($branchGroup, "[,\s]+"))) {
            $trimmed = $branchName.Trim()
            if ($trimmed -ne "") {
                $selectedBranches += $trimmed
            }
        }
    }

    Write-Host "[start] repo=$RepoRoot suffix=$Suffix mode=$Mode batch_size=$BatchSize device=$Device dry_run=$DryRun"

    foreach ($branch in $selectedBranches) {
        if (-not $specs.Contains($branch)) {
            throw "Unknown branch '$branch'. Known branches: $($specs.Keys -join ', ')"
        }
        $spec = $specs[$branch]
        Write-Host "[branch] $branch -> $($spec.OutputDir)"

        if ($Mode -in @("all", "train")) {
            Invoke-TrainingBranch -Spec $spec
        }
        if ($Mode -in @("all", "eval")) {
            Invoke-EvaluationBranch -Spec $spec
        }
    }

    Write-Host "[done] global LogicNet training queue completed."
} finally {
    Stop-Transcript | Out-Null
}
