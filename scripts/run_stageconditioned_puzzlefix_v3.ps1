param(
    [string]$RepoRoot = "F:\KLTN",
    [string]$Python = "python",
    [string]$Suffix = "v3_puzzlefix",
    [string[]]$Branches = @("full", "tokens_only", "trace_only", "loss010", "loss050"),
    [ValidateSet("all", "train", "eval")]
    [string]$Mode = "all",
    [switch]$Quick,
    [switch]$Force,
    [switch]$ResumeExisting,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [string]$Label,
        [string[]]$CommandArgs
    )

    Write-Host "[run] $Label"
    Write-Host "  $Python $($CommandArgs -join ' ')"
    if ($DryRun) {
        return
    }

    & $Python @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Step '$Label' failed with exit code $LASTEXITCODE."
    }
}

function Get-BranchSpecs {
    return [ordered]@{
        full = @{
            Label = "stageconditioned_semantics_$Suffix"
            OutputDir = "outputs\zelda_hmolqd_downstream_stageconditioned_semantics_$Suffix"
            Tokens = $true
            Topology = $true
            GenerationTopology = $true
            SemanticsLossWeight = "0.25"
        }
        tokens_only = @{
            Label = "stageconditioned_semantics_tokens_only_$Suffix"
            OutputDir = "outputs\zelda_hmolqd_downstream_stageconditioned_semantics_tokens_only_$Suffix"
            Tokens = $true
            Topology = $false
            GenerationTopology = $false
            SemanticsLossWeight = "0.25"
        }
        trace_only = @{
            Label = "stageconditioned_semantics_trace_only_$Suffix"
            OutputDir = "outputs\zelda_hmolqd_downstream_stageconditioned_semantics_trace_only_$Suffix"
            Tokens = $false
            Topology = $true
            GenerationTopology = $true
            SemanticsLossWeight = "0.25"
        }
        loss010 = @{
            Label = "stageconditioned_semantics_loss010_$Suffix"
            OutputDir = "outputs\zelda_hmolqd_downstream_stageconditioned_semantics_loss010_$Suffix"
            Tokens = $true
            Topology = $true
            GenerationTopology = $true
            SemanticsLossWeight = "0.10"
        }
        loss050 = @{
            Label = "stageconditioned_semantics_loss050_$Suffix"
            OutputDir = "outputs\zelda_hmolqd_downstream_stageconditioned_semantics_loss050_$Suffix"
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
        "--seed", "42",
        "--verbose"
    )

    if ($Quick) {
        $argsOut += "--quick"
        if ($Stage -eq "diffusion") {
            $argsOut += @("--diffusion-epochs", "1")
        } elseif ($Stage -eq "fast_sampler") {
            $argsOut += @("--fast-sampler-epochs", "1")
        } elseif ($Stage -eq "masked_room") {
            $argsOut += @("--masked-room-epochs", "1")
        }
    }

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
        -ExtraArgs @("--diffusion-vqvae-checkpoint", $vqvaeCheckpoint)
    $diffArgs = Add-AllPuzzleConfigArgs -Spec $Spec -BaseArgs $diffArgs
    Invoke-Step -Label "$($Spec.Label) diffusion" -CommandArgs $diffArgs

    if ((-not $DryRun) -and (-not (Test-Path (Join-Path $RepoRoot $diffusionCheckpoint)))) {
        throw "Missing diffusion checkpoint after training: $diffusionCheckpoint"
    }

    $fastArgs = Add-CommonArgs `
        -Stage "fast_sampler" `
        -Spec $Spec `
        -ExtraArgs @("--fast-sampler-base-diffusion-checkpoint", $diffusionCheckpoint)
    $fastArgs = Add-AllPuzzleConfigArgs -Spec $Spec -BaseArgs $fastArgs
    Invoke-Step -Label "$($Spec.Label) fast_sampler" -CommandArgs $fastArgs

    $maskedArgs = Add-CommonArgs -Stage "masked_room" -Spec $Spec -ExtraArgs @()
    $maskedArgs = Add-AllPuzzleConfigArgs -Spec $Spec -BaseArgs $maskedArgs
    Invoke-Step -Label "$($Spec.Label) masked_room" -CommandArgs $maskedArgs
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

    $summaryPath = Join-Path $protocolDir "summary.json"
    $matchedBudgetReport = "results\old_result\matched_budget_topology_v1\matched_budget_report.json"
    $pcgBenchmarkReport = "results\old_result\pcg_benchmark_alignment_v2\pcg_benchmark_alignment_report.json"
    if ((Test-Path (Join-Path $RepoRoot $matchedBudgetReport)) -and (Test-Path (Join-Path $RepoRoot $pcgBenchmarkReport))) {
        Invoke-Step `
            -Label "$($Spec.Label) baseline comparison" `
            -CommandArgs @(
                "scripts\compare_protocol_to_baselines.py",
                "--fixed-graph-summary", $summaryPath,
                "--matched-budget-report", $matchedBudgetReport,
                "--pcg-benchmark-report", $pcgBenchmarkReport,
                "--output-dir", (Join-Path $protocolDir "baseline_comparison")
            )
    } else {
        Write-Host "[skip] $($Spec.Label) baseline comparison reports are missing."
    }
}

Set-Location $RepoRoot
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128"
$env:KLTN_EXPORT_SEQUENTIAL = "1"
$env:KLTN_EXPORT_MAX_BATCH_SIZE = "1"

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

Write-Host "[start] repo=$RepoRoot suffix=$Suffix mode=$Mode quick=$Quick dry_run=$DryRun"

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

Write-Host "[done] stageconditioned puzzlefix queue completed."
