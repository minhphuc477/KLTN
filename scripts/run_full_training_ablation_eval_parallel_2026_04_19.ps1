param(
    [string]$RepoRoot = "F:\KLTN",
    [string]$GpuIds = "0,1",
    [int]$VqvaeParallelJobs = 6,
    [string]$CurrentOutputFolder = "outputs\zelda_hmolqd_downstream_stageconditioned_semantics_v2",
    [string]$ArchiveRoot = "outputs\old_output",
    [switch]$SkipArchive,
    [switch]$SkipVqvae,
    [switch]$SkipTraining,
    [switch]$SkipEvaluation,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Get-GpuList {
    param([string]$RawGpuIds)

    $result = @()
    foreach ($raw in ([Regex]::Split($RawGpuIds, "[,\s]+"))) {
        $id = $raw.Trim()
        if ($id -ne "") {
            $result += $id
        }
    }
    if ($result.Count -eq 0) {
        $result = @("0")
    }
    return $result
}

function Invoke-LauncherPhase {
    param(
        [string]$Phase,
        [string]$PhaseGpuIds,
        [int]$PhaseMaxParallelJobs = 0
    )

    $args = @(
        "-ExecutionPolicy", "Bypass",
        "-File", "scripts\run_parallel_training_suite_2026_04_17.ps1",
        "-Phase", $Phase,
        "-GpuIds", $PhaseGpuIds
    )
    if ($PhaseMaxParallelJobs -gt 0) {
        $args += @("-MaxParallelJobs", "$PhaseMaxParallelJobs")
    }

    Write-Host "[phase] $Phase"
    Write-Host "  command: powershell $($args -join ' ')"

    if ($DryRun) {
        return
    }

    & powershell @args
    if ($LASTEXITCODE -ne 0) {
        throw "Phase '$Phase' failed with exit code $LASTEXITCODE."
    }
}

function Invoke-PythonStep {
    param(
        [string]$Label,
        [string]$CommandText
    )

    Write-Host "[eval] $Label"
    Write-Host "  command: $CommandText"

    if ($DryRun) {
        return
    }

    Invoke-Expression $CommandText
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation step '$Label' failed with exit code $LASTEXITCODE."
    }
}

function Assert-NoRunningProcessUsingPath {
    param([string]$RelativePath)

    $escaped = [Regex]::Escape($RelativePath)
    $active = @(Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -and $_.CommandLine -match $escaped
    })
    if ($active.Count -gt 0) {
        $ids = @($active | ForEach-Object { $_.ProcessId }) -join ", "
        throw "Cannot archive '$RelativePath' because active processes still reference it. PIDs: $ids"
    }
}

function Archive-FolderIfExists {
    param(
        [string]$SourceRelativePath,
        [string]$ArchiveRelativeRoot
    )

    $source = Join-Path $RepoRoot $SourceRelativePath
    if (-not (Test-Path $source)) {
        Write-Host "[archive] skip (missing): $SourceRelativePath"
        return
    }

    Assert-NoRunningProcessUsingPath -RelativePath $SourceRelativePath

    $archiveBase = Join-Path $RepoRoot $ArchiveRelativeRoot
    $leaf = Split-Path -Path $source -Leaf
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $destination = Join-Path $archiveBase ("${leaf}_old_${stamp}")

    Write-Host "[archive] $SourceRelativePath -> $ArchiveRelativeRoot/$(Split-Path -Leaf $destination)"

    if ($DryRun) {
        return
    }

    New-Item -ItemType Directory -Path $archiveBase -Force | Out-Null
    Move-Item -Path $source -Destination $destination
}

$gpuList = @(Get-GpuList -RawGpuIds $GpuIds)
$primaryGpu = $gpuList[0]
$allGpus = ($gpuList -join ",")

Set-Location $RepoRoot
Write-Host "[start] repo=$RepoRoot gpus=$allGpus dry_run=$DryRun"

if (-not $SkipArchive) {
    Archive-FolderIfExists -SourceRelativePath $CurrentOutputFolder -ArchiveRelativeRoot $ArchiveRoot
}

if (-not $SkipVqvae) {
    # Allow oversubscribed parallel starts here so all VQ-VAE ablations can run together.
    Invoke-LauncherPhase -Phase "vqvae" -PhaseGpuIds $allGpus -PhaseMaxParallelJobs $VqvaeParallelJobs
}

if (-not $SkipTraining) {
    Invoke-LauncherPhase -Phase "diffusion-baseline" -PhaseGpuIds $primaryGpu
    Invoke-LauncherPhase -Phase "aux-baseline" -PhaseGpuIds $allGpus

    Invoke-LauncherPhase -Phase "diffusion-codebook512" -PhaseGpuIds $primaryGpu
    Invoke-LauncherPhase -Phase "aux-codebook512" -PhaseGpuIds $allGpus

    Invoke-LauncherPhase -Phase "diffusion-puzzle-control" -PhaseGpuIds $primaryGpu
    Invoke-LauncherPhase -Phase "aux-puzzle-control" -PhaseGpuIds $allGpus

    Invoke-LauncherPhase -Phase "diffusion-stageconditioned-semantics" -PhaseGpuIds $primaryGpu
    Invoke-LauncherPhase -Phase "aux-stageconditioned-semantics" -PhaseGpuIds $allGpus

    Invoke-LauncherPhase -Phase "puzzle-cookbook-diffusion" -PhaseGpuIds $allGpus
    Invoke-LauncherPhase -Phase "puzzle-cookbook-aux" -PhaseGpuIds $allGpus
}

if (-not $SkipEvaluation) {
    Invoke-PythonStep -Label "fixed-graph baseline v23" -CommandText "python scripts\run_fixed_graph_multi_seed_audit.py --run-dir outputs\zelda_hmolqd_downstream_baseline_puzzle_subtype_v2 --output-dir outputs\zelda_hmolqd_downstream_baseline_puzzle_subtype_v2\protocol_ablation_statefulmultistep_v23 --seeds 20260404 20260405 20260406"

    Invoke-PythonStep -Label "fixed-graph codebook512 v23" -CommandText "python scripts\run_fixed_graph_multi_seed_audit.py --run-dir outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v2 --output-dir outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v2\protocol_ablation_statefulmultistep_v23 --seeds 20260404 20260405 20260406"

    Invoke-PythonStep -Label "fixed-graph stageconditioned semantics v2" -CommandText "python scripts\run_fixed_graph_multi_seed_audit.py --run-dir outputs\zelda_hmolqd_downstream_stageconditioned_semantics_v2 --output-dir outputs\zelda_hmolqd_downstream_stageconditioned_semantics_v2\protocol_ablation_stageconditioned_semantics_v2 --seeds 20260404 20260405 20260406"

    Invoke-PythonStep -Label "manual compare baseline v23" -CommandText "python scripts\export_semantic_anchor_end_to_end.py --run-dir outputs\zelda_hmolqd_downstream_baseline_puzzle_subtype_v2 --output-dir outputs\zelda_hmolqd_downstream_baseline_puzzle_subtype_v2\protocol_manual_compare_statefulmultistep_v23 --seed 20260404"

    Invoke-PythonStep -Label "manual compare stageconditioned semantics v2" -CommandText "python scripts\export_semantic_anchor_end_to_end.py --run-dir outputs\zelda_hmolqd_downstream_stageconditioned_semantics_v2 --output-dir outputs\zelda_hmolqd_downstream_stageconditioned_semantics_v2\protocol_manual_compare_stageconditioned_semantics_v2 --seed 20260404"

    Invoke-PythonStep -Label "manual compare codebook512 v23" -CommandText "python scripts\export_semantic_anchor_end_to_end.py --run-dir outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v2 --output-dir outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v2\protocol_manual_compare_statefulmultistep_v23 --seed 20260404"

    Invoke-PythonStep -Label "stateful puzzle sweep v3" -CommandText "python scripts\run_stateful_puzzle_hparam_sweep.py --run-dir outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v2 --output-dir results\stateful_puzzle_hparam_sweep_v3 --seed 20260418"

    Invoke-PythonStep -Label "core ablation quick v4" -CommandText "python scripts\run_ablation_study.py --config configs\zelda_hmolqd.yaml --output-dir results\ablation_core_quick_v4 --quick --max-runtime-sec 7200 --vqvae-checkpoint outputs\vqvae_ablation_codebook512_v2\checkpoints\vqvae\vqvae_pretrained.pth"

    Invoke-PythonStep -Label "room branch benchmark quick v3" -CommandText "python scripts\run_room_branch_benchmark.py --config configs\zelda_hmolqd.yaml --output-dir results\room_branch_benchmark_quick_v3 --vqvae-checkpoint outputs\vqvae_ablation_codebook512_v2\checkpoints\vqvae\vqvae_pretrained.pth --quick"

    Invoke-PythonStep -Label "matched-budget topology v2" -CommandText "python scripts\run_matched_budget_topology_benchmark.py --output results\matched_budget_topology_v2 --samples-per-method 64 --seed 42"

    Invoke-PythonStep -Label "pcbs ablation balanced" -CommandText "python scripts\run_pcbs_component_ablation.py --levels 1,2,3 --persona balanced --output-dir results\pcbs_component_ablation_balanced_l123_v4"
    Invoke-PythonStep -Label "pcbs ablation explorer" -CommandText "python scripts\run_pcbs_component_ablation.py --levels 1,2,3 --persona explorer --output-dir results\pcbs_component_ablation_explorer_l123_v4"
    Invoke-PythonStep -Label "pcbs ablation novice" -CommandText "python scripts\run_pcbs_component_ablation.py --levels 1,2,3 --persona novice --output-dir results\pcbs_component_ablation_novice_l123_v1"

    Invoke-PythonStep -Label "cbs benchmark levels1-9 all personas" -CommandText "python scripts\run_cbs_benchmarks.py --levels 1,2,3,4,5,6,7,8,9 --variants 1,2 --all-personas --timeout-astar 200000 --timeout-cbs 50000 --seed 42 --output results\cbs_benchmark_levels1_9_variants12_all_personas_v_latest.csv"
}

Write-Host "[done] full command-book pipeline orchestration completed."
