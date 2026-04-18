param(
    [ValidateSet(
        "vqvae",
        "diffusion-codebook512",
        "aux-codebook512",
        "diffusion-puzzle-control",
        "aux-puzzle-control",
        "puzzle-cookbook-diffusion",
        "puzzle-cookbook-aux"
    )]
    [string]$Phase = "vqvae",
    [string]$RepoRoot = "F:\KLTN",
    [string]$GpuIds = "0",
    [int]$CpuThreads = 4,
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

    $commandParts = @(
        "Set-Location '$RepoRootPath'",
        "`$env:PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128'",
        "`$env:OMP_NUM_THREADS='$ThreadCount'",
        "`$env:MKL_NUM_THREADS='$ThreadCount'",
        "`$env:KLTN_EXPORT_SEQUENTIAL='1'",
        "`$env:KLTN_EXPORT_MAX_BATCH_SIZE='1'"
    )
    if ($GpuId -ne "") {
        $commandParts += "`$env:CUDA_VISIBLE_DEVICES='$GpuId'"
    }
    $commandParts += $Job.CommandText
    $commandParts += "if (`$LASTEXITCODE -ne `$null) { exit `$LASTEXITCODE }"
    $commandText = ($commandParts -join "; ")

    if ($DryRun) {
        Write-Host "[dry-run][$GpuId] $($Job.Name): $($Job.CommandText)"
        return $null
    }

    Write-Host "Launching $($Job.Name) on GPU '$GpuId'"
    $process = Start-Process powershell -PassThru -WindowStyle Minimized -ArgumentList @(
        "-Command",
        $commandText
    ) -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath

    return @{
        Name = $Job.Name
        Gpu = $GpuId
        Process = $process
        Stdout = $stdoutPath
        Stderr = $stderrPath
        StartTime = Get-Date
    }
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

        $exitCode = $handle.Process.ExitCode
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
foreach ($rawId in ($GpuIds -split ",")) {
    $normalized = $rawId.Trim()
    if ($normalized -ne "") {
        $gpuList += $normalized
    }
}
if ($gpuList.Count -eq 0) {
    $gpuList = @("")
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
    while ($running.Count -ge $gpuList.Count) {
        Start-Sleep -Seconds 15
        $running = @(Update-CompletedJobStatus -RunningHandles $running -FailedHandles ([ref]$failed))
        Write-RunningJobStatus -RunningHandles $running
    }

    $usedGpus = @($running | ForEach-Object { $_.Gpu })
    $freeGpu = $null
    foreach ($gpu in $gpuList) {
        if ($usedGpus -notcontains $gpu) {
            $freeGpu = $gpu
            break
        }
    }
    if ($null -eq $freeGpu) {
        $freeGpu = $gpuList[0]
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
    $failedSummary = @($failed | ForEach-Object { "$($_.Name)(exit=$($_.Process.ExitCode))" })
    throw "Phase '$Phase' completed with failed jobs: $($failedSummary -join ', '). See outputs\\_launcher_logs for details."
}

Write-Host "Phase '$Phase' completed."
