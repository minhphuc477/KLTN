param(
    [ValidateSet(
        "vqvae",
        "diffusion-baseline",
        "aux-baseline",
        "diffusion-codebook512",
        "aux-codebook512",
        "diffusion-puzzle-control",
        "aux-puzzle-control",
        "diffusion-stageconditioned-semantics",
        "aux-stageconditioned-semantics",
        "puzzle-cookbook-diffusion",
        "puzzle-cookbook-aux"
    )]
    [string]$Phase = "vqvae",
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$GpuIds = "0",
    [int]$CpuThreads = 4,
    [int]$MaxParallelJobs = 0,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function New-JobSpec {
    param(
        [string]$Name,
        [string]$CommandText
    )

    @{
        Name = $Name
        CommandText = $CommandText.Trim()
    }
}

function Get-PhaseJobs {
    param(
        [string]$SelectedPhase
    )

    switch ($SelectedPhase) {
        "vqvae" {
            return @(
                (New-JobSpec "vqvae_baseline" 'python -m src.train_vqvae --config configs\zelda_hmolqd.yaml --save-dir outputs\vqvae_audit_baseline_v2\checkpoints\vqvae --data-dir "Data\The Legend of Zelda" --epochs 300 --hidden-dim 96 --latent-dim 64 --codebook-size 256 --use-coordconv --mrf-penalty-weight 0.05 --validation-fraction 0.1 --validation-max-batches 16 --best-checkpoint-metric val_loss --seed 42'),
                (New-JobSpec "vqvae_codebook128" 'python -m src.train_vqvae --config configs\zelda_hmolqd.yaml --save-dir outputs\vqvae_ablation_codebook128_v2\checkpoints\vqvae --data-dir "Data\The Legend of Zelda" --epochs 300 --hidden-dim 96 --latent-dim 64 --codebook-size 128 --use-coordconv --mrf-penalty-weight 0.05 --validation-fraction 0.1 --validation-max-batches 16 --best-checkpoint-metric val_loss --seed 42'),
                (New-JobSpec "vqvae_codebook512" 'python -m src.train_vqvae --config configs\zelda_hmolqd.yaml --save-dir outputs\vqvae_ablation_codebook512_v2\checkpoints\vqvae --data-dir "Data\The Legend of Zelda" --epochs 300 --hidden-dim 96 --latent-dim 64 --codebook-size 512 --use-coordconv --mrf-penalty-weight 0.05 --validation-fraction 0.1 --validation-max-batches 16 --best-checkpoint-metric val_loss --seed 42'),
                (New-JobSpec "vqvae_hidden64" 'python -m src.train_vqvae --config configs\zelda_hmolqd.yaml --save-dir outputs\vqvae_ablation_hidden64_v2\checkpoints\vqvae --data-dir "Data\The Legend of Zelda" --epochs 300 --hidden-dim 64 --latent-dim 64 --codebook-size 256 --use-coordconv --mrf-penalty-weight 0.05 --validation-fraction 0.1 --validation-max-batches 16 --best-checkpoint-metric val_loss --seed 42'),
                (New-JobSpec "vqvae_no_coordconv" 'python -m src.train_vqvae --config configs\zelda_hmolqd.yaml --save-dir outputs\vqvae_ablation_no_coordconv_v2\checkpoints\vqvae --data-dir "Data\The Legend of Zelda" --epochs 300 --hidden-dim 96 --latent-dim 64 --codebook-size 256 --no-use-coordconv --mrf-penalty-weight 0.05 --validation-fraction 0.1 --validation-max-batches 16 --best-checkpoint-metric val_loss --seed 42'),
                (New-JobSpec "vqvae_no_mrf" 'python -m src.train_vqvae --config configs\zelda_hmolqd.yaml --save-dir outputs\vqvae_ablation_no_mrf_v2\checkpoints\vqvae --data-dir "Data\The Legend of Zelda" --epochs 300 --hidden-dim 96 --latent-dim 64 --codebook-size 256 --use-coordconv --mrf-penalty-weight 0.0 --validation-fraction 0.1 --validation-max-batches 16 --best-checkpoint-metric val_loss --seed 42')
            )
        }
        "diffusion-baseline" {
            return @(
                (New-JobSpec "diffusion_baseline" 'python main.py train --config configs\zelda_hmolqd.yaml --stage diffusion --output-dir outputs\zelda_hmolqd_downstream_baseline_puzzle_subtype_v2 --diffusion-vqvae-checkpoint outputs\vqvae_audit_baseline_v2\checkpoints\vqvae\vqvae_pretrained.pth --seed 42 --no-auto-resume --verbose')
            )
        }
        "aux-baseline" {
            return @(
                (New-JobSpec "fast_sampler_baseline" 'python main.py train --config configs\zelda_hmolqd.yaml --stage fast_sampler --output-dir outputs\zelda_hmolqd_downstream_baseline_puzzle_subtype_v2 --fast-sampler-base-diffusion-checkpoint outputs\zelda_hmolqd_downstream_baseline_puzzle_subtype_v2\checkpoints\diffusion\best_model.pth --seed 42 --no-auto-resume --verbose'),
                (New-JobSpec "masked_room_baseline" 'python main.py train --config configs\zelda_hmolqd.yaml --stage masked_room --output-dir outputs\zelda_hmolqd_downstream_baseline_puzzle_subtype_v2 --seed 42 --no-auto-resume --verbose')
            )
        }
        "diffusion-codebook512" {
            return @(
                (New-JobSpec "diffusion_codebook512" 'python main.py train --config configs\zelda_hmolqd.yaml --stage diffusion --output-dir outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v2 --diffusion-vqvae-checkpoint outputs\vqvae_ablation_codebook512_v2\checkpoints\vqvae\vqvae_pretrained.pth --seed 42 --no-auto-resume --verbose')
            )
        }
        "aux-codebook512" {
            return @(
                (New-JobSpec "fast_sampler_codebook512" 'python main.py train --config configs\zelda_hmolqd.yaml --stage fast_sampler --output-dir outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v2 --fast-sampler-base-diffusion-checkpoint outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v2\checkpoints\diffusion\best_model.pth --seed 42 --no-auto-resume --verbose'),
                (New-JobSpec "masked_room_codebook512" 'python main.py train --config configs\zelda_hmolqd.yaml --stage masked_room --output-dir outputs\zelda_hmolqd_downstream_codebook512_puzzle_subtype_v2 --seed 42 --no-auto-resume --verbose')
            )
        }
        "diffusion-puzzle-control" {
            return @(
                (New-JobSpec "diffusion_puzzle_control" 'python main.py train --config configs\zelda_hmolqd.yaml --stage diffusion --output-dir outputs\zelda_hmolqd_downstream_puzzle_structure_control_v2 --diffusion-vqvae-checkpoint outputs\vqvae_ablation_codebook512_v2\checkpoints\vqvae\vqvae_pretrained.pth --diffusion-puzzle-structure-dropout-prob 0.35 --seed 42 --no-auto-resume --verbose')
            )
        }
        "aux-puzzle-control" {
            return @(
                (New-JobSpec "fast_sampler_puzzle_control" 'python main.py train --config configs\zelda_hmolqd.yaml --stage fast_sampler --output-dir outputs\zelda_hmolqd_downstream_puzzle_structure_control_v2 --fast-sampler-base-diffusion-checkpoint outputs\zelda_hmolqd_downstream_puzzle_structure_control_v2\checkpoints\diffusion\best_model.pth --fast-sampler-puzzle-structure-dropout-prob 0.35 --seed 42 --no-auto-resume --verbose'),
                (New-JobSpec "masked_room_puzzle_control" 'python main.py train --config configs\zelda_hmolqd.yaml --stage masked_room --output-dir outputs\zelda_hmolqd_downstream_puzzle_structure_control_v2 --masked-room-puzzle-structure-dropout-prob 0.35 --seed 42 --no-auto-resume --verbose')
            )
        }
        "diffusion-stageconditioned-semantics" {
            return @(
                (New-JobSpec "diffusion_stageconditioned_semantics" 'python main.py train --config configs\zelda_hmolqd.yaml --stage diffusion --output-dir outputs\zelda_hmolqd_downstream_stageconditioned_semantics_v2 --diffusion-vqvae-checkpoint outputs\vqvae_ablation_codebook512_v2\checkpoints\vqvae\vqvae_pretrained.pth --diffusion-puzzle-structure-dropout-prob 0.35 --diffusion-puzzle-stage-conditioning-enabled --diffusion-puzzle-stage-token-scale 0.20 --diffusion-puzzle-stage-topology-enabled --diffusion-puzzle-stage-trace-decay 0.75 --diffusion-puzzle-stage-semantics-loss-weight 0.25 --diffusion-puzzle-stage-semantics-hidden-dim 96 --diffusion-puzzle-stage-semantics-max-sequence-length 6 --seed 42 --no-auto-resume --verbose')
            )
        }
        "aux-stageconditioned-semantics" {
            return @(
                (New-JobSpec "fast_sampler_stageconditioned_semantics" 'python main.py train --config configs\zelda_hmolqd.yaml --stage fast_sampler --output-dir outputs\zelda_hmolqd_downstream_stageconditioned_semantics_v2 --fast-sampler-base-diffusion-checkpoint outputs\zelda_hmolqd_downstream_stageconditioned_semantics_v2\checkpoints\diffusion\best_model.pth --fast-sampler-puzzle-structure-dropout-prob 0.35 --fast-sampler-puzzle-stage-conditioning-enabled --fast-sampler-puzzle-stage-token-scale 0.20 --fast-sampler-puzzle-stage-topology-enabled --fast-sampler-puzzle-stage-trace-decay 0.75 --fast-sampler-puzzle-stage-semantics-loss-weight 0.25 --fast-sampler-puzzle-stage-semantics-hidden-dim 96 --fast-sampler-puzzle-stage-semantics-max-sequence-length 6 --seed 42 --no-auto-resume --verbose'),
                (New-JobSpec "masked_room_stageconditioned_semantics" 'python main.py train --config configs\zelda_hmolqd.yaml --stage masked_room --output-dir outputs\zelda_hmolqd_downstream_stageconditioned_semantics_v2 --masked-room-puzzle-structure-dropout-prob 0.35 --masked-room-puzzle-stage-conditioning-enabled --masked-room-puzzle-stage-token-scale 0.20 --masked-room-puzzle-stage-topology-enabled --masked-room-puzzle-stage-trace-decay 0.75 --masked-room-puzzle-stage-semantics-loss-weight 0.25 --masked-room-puzzle-stage-semantics-hidden-dim 96 --masked-room-puzzle-stage-semantics-max-sequence-length 6 --seed 42 --no-auto-resume --verbose')
            )
        }
        "puzzle-cookbook-diffusion" {
            return @(
                (New-JobSpec "puzzlecookbook_pdrop015" 'python main.py train --config configs\zelda_hmolqd.yaml --stage diffusion --output-dir outputs\zelda_hmolqd_puzzlecookbook_pdrop015_v1 --diffusion-vqvae-checkpoint outputs\vqvae_ablation_codebook512_v2\checkpoints\vqvae\vqvae_pretrained.pth --diffusion-puzzle-structure-dropout-prob 0.15 --seed 42 --no-auto-resume --verbose'),
                (New-JobSpec "puzzlecookbook_pdrop035" 'python main.py train --config configs\zelda_hmolqd.yaml --stage diffusion --output-dir outputs\zelda_hmolqd_puzzlecookbook_pdrop035_v1 --diffusion-vqvae-checkpoint outputs\vqvae_ablation_codebook512_v2\checkpoints\vqvae\vqvae_pretrained.pth --diffusion-puzzle-structure-dropout-prob 0.35 --seed 42 --no-auto-resume --verbose'),
                (New-JobSpec "puzzlecookbook_pdrop055" 'python main.py train --config configs\zelda_hmolqd.yaml --stage diffusion --output-dir outputs\zelda_hmolqd_puzzlecookbook_pdrop055_v1 --diffusion-vqvae-checkpoint outputs\vqvae_ablation_codebook512_v2\checkpoints\vqvae\vqvae_pretrained.pth --diffusion-puzzle-structure-dropout-prob 0.55 --seed 42 --no-auto-resume --verbose')
            )
        }
        "puzzle-cookbook-aux" {
            return @(
                (New-JobSpec "fast_sampler_pdrop015" 'python main.py train --config configs\zelda_hmolqd.yaml --stage fast_sampler --output-dir outputs\zelda_hmolqd_puzzlecookbook_pdrop015_v1 --fast-sampler-base-diffusion-checkpoint outputs\zelda_hmolqd_puzzlecookbook_pdrop015_v1\checkpoints\diffusion\best_model.pth --fast-sampler-puzzle-structure-dropout-prob 0.15 --seed 42 --no-auto-resume --verbose'),
                (New-JobSpec "masked_room_pdrop015" 'python main.py train --config configs\zelda_hmolqd.yaml --stage masked_room --output-dir outputs\zelda_hmolqd_puzzlecookbook_pdrop015_v1 --masked-room-puzzle-structure-dropout-prob 0.15 --seed 42 --no-auto-resume --verbose'),
                (New-JobSpec "fast_sampler_pdrop035" 'python main.py train --config configs\zelda_hmolqd.yaml --stage fast_sampler --output-dir outputs\zelda_hmolqd_puzzlecookbook_pdrop035_v1 --fast-sampler-base-diffusion-checkpoint outputs\zelda_hmolqd_puzzlecookbook_pdrop035_v1\checkpoints\diffusion\best_model.pth --fast-sampler-puzzle-structure-dropout-prob 0.35 --seed 42 --no-auto-resume --verbose'),
                (New-JobSpec "masked_room_pdrop035" 'python main.py train --config configs\zelda_hmolqd.yaml --stage masked_room --output-dir outputs\zelda_hmolqd_puzzlecookbook_pdrop035_v1 --masked-room-puzzle-structure-dropout-prob 0.35 --seed 42 --no-auto-resume --verbose'),
                (New-JobSpec "fast_sampler_pdrop055" 'python main.py train --config configs\zelda_hmolqd.yaml --stage fast_sampler --output-dir outputs\zelda_hmolqd_puzzlecookbook_pdrop055_v1 --fast-sampler-base-diffusion-checkpoint outputs\zelda_hmolqd_puzzlecookbook_pdrop055_v1\checkpoints\diffusion\best_model.pth --fast-sampler-puzzle-structure-dropout-prob 0.55 --seed 42 --no-auto-resume --verbose'),
                (New-JobSpec "masked_room_pdrop055" 'python main.py train --config configs\zelda_hmolqd.yaml --stage masked_room --output-dir outputs\zelda_hmolqd_puzzlecookbook_pdrop055_v1 --masked-room-puzzle-structure-dropout-prob 0.55 --seed 42 --no-auto-resume --verbose')
            )
        }
    }
}

function Start-RepoJob {
    param(
        [hashtable]$Job,
        [string]$GpuId,
        [string]$RepoRootPath,
        [int]$ThreadCount
    )

    $logDir = Join-Path $RepoRootPath "outputs\_launcher_logs"
    if (-not (Test-Path $logDir)) {
        New-Item -ItemType Directory -Path $logDir | Out-Null
    }
    $stdoutPath = Join-Path $logDir "$($Job.Name).out.log"
    $stderrPath = Join-Path $logDir "$($Job.Name).err.log"
    $statusPath = Join-Path $logDir "$($Job.Name).status.txt"
    Remove-Item -Path $statusPath -Force -ErrorAction SilentlyContinue

    $commandText = @"
Set-Location '$RepoRootPath'
`$ErrorActionPreference = 'Stop'
`$env:PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128'
`$env:OMP_NUM_THREADS='$ThreadCount'
`$env:MKL_NUM_THREADS='$ThreadCount'
`$env:KLTN_EXPORT_SEQUENTIAL='1'
`$env:KLTN_EXPORT_MAX_BATCH_SIZE='1'
"@

    if ($GpuId -ne "") {
        $commandText += "`n`$env:CUDA_VISIBLE_DEVICES='$GpuId'"
    }

    $commandText += @"

`$statusCode = 0
try {
    Invoke-Expression @'
$($Job.CommandText)
'@
    if (`$LASTEXITCODE -ne `$null) {
        `$statusCode = [int]`$LASTEXITCODE
    }
}
catch {
    `$statusCode = 1
}
[IO.File]::WriteAllText('$statusPath', "EXITCODE=`$statusCode`r`n")
exit `$statusCode
"@

    $encodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($commandText))

    if ($DryRun) {
        Write-Host "[dry-run][$GpuId] $($Job.Name): $($Job.CommandText)"
        return $null
    }

    Write-Host "Launching $($Job.Name) on GPU '$GpuId'"
    $process = Start-Process powershell -PassThru -WindowStyle Minimized -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-EncodedCommand", $encodedCommand
    ) -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath

    return @{
        Name = $Job.Name
        Gpu = $GpuId
        Process = $process
        Stdout = $stdoutPath
        Stderr = $stderrPath
        Status = $statusPath
        StartTime = Get-Date
    }
}

function Get-RecordedExitCode {
    param([string]$StatusPath)

    if (-not (Test-Path $StatusPath)) {
        return $null
    }

    try {
        $line = (Get-Content -Path $StatusPath -ErrorAction Stop | Select-Object -First 1)
        if ($line -match '^EXITCODE=(-?\d+)$') {
            return [int]$matches[1]
        }
    }
    catch {
        return $null
    }

    return $null
}

function Get-LastNonEmptyLogLine {
    param(
        [string]$Path,
        [int]$TailLines = 20
    )

    if (-not (Test-Path $Path)) {
        return ""
    }

    try {
        $lines = @(Get-Content -Path $Path -Tail $TailLines -ErrorAction Stop)
        $lines = @($lines | Where-Object { $_ -and $_.Trim() -ne "" })
        if ($lines.Count -eq 0) {
            return ""
        }
        return [string]$lines[-1]
    }
    catch {
        return ""
    }
}

function Write-RunningJobStatus {
    param(
        [array]$RunningHandles
    )

    if ($RunningHandles.Count -eq 0) {
        return
    }

    $statusParts = @()
    foreach ($handle in $RunningHandles) {
        $elapsedMinutes = [int]((Get-Date) - $handle.StartTime).TotalMinutes
        $tailLine = Get-LastNonEmptyLogLine -Path $handle.Stderr
        if ($tailLine -eq "") {
            $tailLine = Get-LastNonEmptyLogLine -Path $handle.Stdout
        }
        $tailLine = ($tailLine -replace '\s+', ' ').Trim()
        if ($tailLine.Length -gt 110) {
            $tailLine = $tailLine.Substring(0, 110) + "..."
        }

        if ($tailLine -ne "") {
            $statusParts += "$($handle.Name)@GPU$($handle.Gpu) pid=$($handle.Process.Id) ${elapsedMinutes}m :: $tailLine"
        }
        else {
            $statusParts += "$($handle.Name)@GPU$($handle.Gpu) pid=$($handle.Process.Id) ${elapsedMinutes}m"
        }
    }

    Write-Host ("Running jobs: " + ($statusParts -join " | "))
}

function Update-CompletedJobStatus {
    param(
        [array]$RunningHandles,
        [ref]$FailedHandles
    )

    $stillRunning = @()
    foreach ($handle in $RunningHandles) {
        if (-not $handle.Process.HasExited) {
            $stillRunning += $handle
            continue
        }

        $exitCode = Get-RecordedExitCode -StatusPath $handle.Status
        if ($null -eq $exitCode) {
            try {
                $exitCode = $handle.Process.ExitCode
            }
            catch {
                $exitCode = $null
            }
        }

        if ($null -eq $exitCode) {
            $errTailFallback = Get-LastNonEmptyLogLine -Path $handle.Stderr
            $outTailFallback = Get-LastNonEmptyLogLine -Path $handle.Stdout
            if ($errTailFallback -match '(?i)(traceback|error:|exception|failed with exit code)' -or $outTailFallback -match '(?i)(traceback|error:|exception|failed with exit code)') {
                $exitCode = 1
            }
            else {
                $exitCode = 0
            }
            Write-Host "  warning: missing explicit process exit code; inferred exit=$exitCode for $($handle.Name)."
        }

        if ($exitCode -eq 0) {
            Write-Host "[ok] $($handle.Name) finished on GPU '$($handle.Gpu)' (exit=$exitCode)."
            continue
        }

        Write-Host "[failed] $($handle.Name) finished on GPU '$($handle.Gpu)' (exit=$exitCode)."
        Write-Host "  stdout: $($handle.Stdout)"
        Write-Host "  stderr: $($handle.Stderr)"
        $errTail = Get-LastNonEmptyLogLine -Path $handle.Stderr
        if ($errTail -ne "") {
            $errTail = ($errTail -replace '\s+', ' ').Trim()
            Write-Host "  last log line: $errTail"
        }

        $FailedHandles.Value = @($FailedHandles.Value) + @($handle)
    }

    return @($stillRunning)
}

$jobs = @(Get-PhaseJobs -SelectedPhase $Phase)
if ($jobs.Count -eq 0) {
    throw "No jobs defined for phase '$Phase'."
}

$gpuList = @()
foreach ($rawId in ([Regex]::Split($GpuIds, "[,\s]+"))) {
    $normalized = $rawId.Trim()
    if ($normalized -ne "") {
        $gpuList += $normalized
    }
}
if ($gpuList.Count -eq 0) {
    $gpuList = @("")
}

$parallelLimit = $gpuList.Count
if ($MaxParallelJobs -gt 0) {
    $parallelLimit = $MaxParallelJobs
}
if ($parallelLimit -lt 1) {
    $parallelLimit = 1
}

if ($DryRun) {
    for ($idx = 0; $idx -lt $jobs.Count; $idx++) {
        $gpu = $gpuList[$idx % $gpuList.Count]
        [void](Start-RepoJob -Job $jobs[$idx] -GpuId $gpu -RepoRootPath $RepoRoot -ThreadCount $CpuThreads)
    }
    return
}

$running = @()
$failed = @()
foreach ($job in $jobs) {
    while ($running.Count -ge $parallelLimit) {
        Start-Sleep -Seconds 15
        $running = @(Update-CompletedJobStatus -RunningHandles $running -FailedHandles ([ref]$failed))
        Write-RunningJobStatus -RunningHandles $running
    }

    $usedGpus = @($running | ForEach-Object { $_.Gpu })
    $freeGpu = $null

    # Prefer a currently unused GPU first.
    foreach ($gpuCandidate in $gpuList) {
        if ($usedGpus -notcontains $gpuCandidate) {
            $freeGpu = $gpuCandidate
            break
        }
    }

    # If oversubscribing GPUs, place the next job on the least-loaded GPU.
    if ($null -eq $freeGpu) {
        $gpuLoad = @{}
        foreach ($gpuCandidate in $gpuList) {
            $gpuLoad[$gpuCandidate] = 0
        }
        foreach ($handle in $running) {
            if ($gpuLoad.ContainsKey($handle.Gpu)) {
                $gpuLoad[$handle.Gpu] += 1
            }
        }
        $freeGpu = ($gpuLoad.GetEnumerator() | Sort-Object Value, Name | Select-Object -First 1).Name
    }

    $handle = Start-RepoJob -Job $job -GpuId $freeGpu -RepoRootPath $RepoRoot -ThreadCount $CpuThreads
    if ($null -ne $handle) {
        $running += $handle
        Write-RunningJobStatus -RunningHandles $running
    }
}

while ($running.Count -gt 0) {
    Start-Sleep -Seconds 15
    $running = @(Update-CompletedJobStatus -RunningHandles $running -FailedHandles ([ref]$failed))
    Write-RunningJobStatus -RunningHandles $running
}

if ($failed.Count -gt 0) {
    $failedSummary = @($failed | ForEach-Object {
        $code = Get-RecordedExitCode -StatusPath $_.Status
        if ($null -eq $code) {
            try {
                $code = $_.Process.ExitCode
            }
            catch {
                $code = "?"
            }
        }
        "$($_.Name)(exit=$code)"
    })
    throw "Phase '$Phase' completed with failed jobs: $($failedSummary -join ', '). See outputs\\_launcher_logs for details."
}

Write-Host "Phase '$Phase' completed."
