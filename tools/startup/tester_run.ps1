<#
.SYNOPSIS
One command for a tester: check the build, run the title offline with a per-commit shader cache, mail-sized zip at the end.

.DESCRIPTION
1. Checks that the build in -BuildDir was made from this checkout's current commit (the revision the build writes
   into generated\x2w_build_rev.cpp). A build from another commit is used as it is when the runtime sources (src, cmake,
   CMakeLists.txt, CMakePresets.json) are the same there and here, so a commit that changes only tools, tests or docs needs
   no rebuild; the summary says why (runtime_sources.py) and the manifest records it as build_check. Otherwise, or when a
   DLL run.ps1 copies is missing, it builds only the startup_runtime target (the eight DLLs and the
   revision stamp; -FullBuild builds everything) or, with -NoBuild, stops with a message.
2. Uses one shader cache folder per shader revision (out\startup\shader_cache_<rev>, from the x2w-startup-shader-N
   strings of gfx2hlsl.py): kept across runs and across branch updates that do not bump the revision. The --watch translator needs minutes to catch up, so an emptied cache would leave
   pipelines untranslated and the screen black. The summary says how many translations were present at start.
3. Runs run.ps1 with -Render -Translate -FakeUser -GpuMirror -ContinueAsserts 1000 and X2W_D3D12_BUDGET=1,
   the input script and, when run.ps1 knows it, the capture interval.
4. Packs report_<commit>_<yyyymmdd-hhmm>.zip with EVERY file the run wrote in its log folder (stdout, stderr, events,
   finished, threads, files, trap.log, translate, translate_err, window*.png, bink_luma.png ...), the debugger
   summary and manifest.json, and prints each path and size. Not included: the shader cache, memory dumps and game
   files. Over -LimitMB (default 20, so each part fits a mail) the zip is split into numbered parts
   (report_..._part01of03.zip); no log is dropped.

The launcher's job memory cap (run.ps1 -MemoryGiB, default 14) is set to 75% of physical RAM, at least 14 GiB: a run whose
commit passes the cap dies with error 1455. -MemoryGiB overrides it; the summary prints it and the manifest records it.

Lazy memory commit is ON by default (run.ps1 -LazyMemory): physical pages and the large startup blocks are committed on
first touch, which cuts the section commit. -NoLazyMemory turns it off. The manifest records it.

When the report is packed the script finishes the translation: one translator pass over the shader cache (no --watch), until
nothing is pending or -FinishTranslateSeconds (default 900, 0 = skip) is over, then prints "pending before/after, ok, refused".
The next run then starts with every request the title made already translated. The mail does not wait for this step.

-Deep also sets -TraceCalls and -TraceApi: a slower run with more detail in the logs.

Lazy slot 133 textures are ON for now: X2W_D3D12_LAZY_TEXTURES=1 creates read-only slot 133 textures on first use, which
keeps video memory from running out. -NoLazyTextures turns it off. The manifest records it.

-Input menus: D-pad down, D-pad down, A, one press every 2 s (each held 1.2 s) starting 5 s after the first pad poll, up to 300 s (input script
time counts from that poll, which the title reaches late).
-Input menus2: the same presses, but after every A the script waits until the presented picture has changed and stayed
still for 2 s (at most 90 s, then it goes on) before the next press, so nothing is pressed blind while a screen loads.
Any other value is a raw
X2W_INPUT_SCRIPT (docs/startup-runtime.md, "Input script").

.EXAMPLE
.\tools\startup\tester_run.ps1 -Exe D:\run\title.exe -DataDir G:\ -Seconds 600 -Input menus -CaptureEvery 15
#>
param(
    [Parameter(Mandatory)] [string]$Exe,
    [Parameter(Mandatory)] [string]$DataDir,
    [int]$Seconds = 600,
    [Alias("Input")] [string]$InputMode = "menus",   # $Input is a PowerShell automatic variable: the parameter is -InputMode, -Input is its alias
    [int]$CaptureEvery = 15,                    # seconds between window captures; 0 = only the final window.png
    [string]$BuildDir = "out/build/release",
    [string]$OutDir = "out/tester",
    [string]$ShaderCacheRoot = "out/startup",     # one folder per shader revision inside: shader_cache_<rev>
    [int]$LimitMB = 20,
    [switch]$FullBuild,                         # build every target (cmake --build with no --target) instead of only the run's
    [switch]$NoBuild,                           # stop instead of building when the build is not from this commit
    [switch]$Deep,                              # also -TraceCalls and -TraceApi: slower, more detailed
    [int]$MemoryGiB = 0,                        # job memory cap of the launcher; 0 = 75% of physical RAM (at least 14)
    [int]$FinishTranslateSeconds = 900,         # after the report is packed: one translator pass over the cache, at most this long; 0 = skip
    [switch]$NoLazyMemory,                      # lazy memory commit is on (run.ps1 -LazyMemory: X2W_XMEM_LAZY_COMMIT, _LAZY_STARTUP, _SHARED_VIEWS); this turns it off
    [switch]$NoLazyTextures,                    # lazy slot 133 textures are on (X2W_D3D12_LAZY_TEXTURES=1, PR #2773); this turns them off
    [switch]$NoStartupBelow40,                  # the 0x283 startup blocks are placed below 2^40 (X2W_XMEM_STARTUP_BELOW_40=1, #1758); this turns it off
    [switch]$NoLowAddresses,                   # the title starts with bottom-up ASLR off so its memory stays below 2^40 (X2W_LOW_ADDRESSES=1); this turns it off
    [switch]$NoTitleInputSnapshot,              # translated compute inputs in a placed buffer that does not alias title memory are read from title memory (X2W_TRANSLATED_TITLE_INPUT_SNAPSHOT=1, with X2W_TRANSLATED_INDIRECT_COPY_RANGES=1 for ranges a list copied); this turns both off
    [switch]$NoSysmemFallback,                 # a texture that runs out of video memory is created in system memory (X2W_D3D12_SYSMEM_FALLBACK=1); this turns it off
    [switch]$NoGpuTime,                        # GPU time per frame is logged every 300 frames (X2W_GPU_TIME=1, timestamp queries); this turns it off
    [int]$CommitHeadroomMiB = 256,              # last resort before error 1455: the run ends cleanly (logs packed) when the system commit has less than this free; 0 = off
    [int]$CommitWarnMiB = 2048                  # a warning line in the log when the system commit has less than this free; 0 = off
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$buildRoot = if ([System.IO.Path]::IsPathRooted($BuildDir)) { $BuildDir } else { Join-Path $repo $BuildDir }
$clock = [System.Diagnostics.Stopwatch]::StartNew()

# Windows PowerShell 5.1 turns a native command's stderr into error records when the script's own stderr is redirected
# (a caller that captures it), and "Stop" then ends the script on a harmless warning (cmake, git, python). Native commands
# run with "Continue" here and are judged by $LASTEXITCODE; a native command's stderr is shown, not captured.
function Invoke-Native([scriptblock]$Command) {
    $saved = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $Command | ForEach-Object { if ($_ -is [System.Management.Automation.ErrorRecord]) { Write-Host $_.ToString() } else { $_ } } }
    finally { $ErrorActionPreference = $saved }
}

function Get-CheckoutRevision {
    $head = (Invoke-Native { & git -C $repo rev-parse --short=8 HEAD } | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $head -notmatch "^[0-9a-f]+$") { throw "tester_run: $repo is not a git checkout" }
    $changes = Invoke-Native { & git -C $repo --no-optional-locks status --porcelain --untracked-files=no }
    return @{ Commit = $head; Dirty = [bool]$changes }
}

function Get-BuildRevision {
    $file = Join-Path $buildRoot "generated\x2w_build_rev.cpp"
    if (-not (Test-Path -LiteralPath $file)) { return $null }
    $text = Get-Content -LiteralPath $file -Raw
    if ($text -match '"([0-9a-f]+(?:-dirty)?|unknown)"') { return $Matches[1] }
    return $null
}

# `startup_runtime` builds exactly the eight DLLs run.ps1 copies next to the title (OUTPUT_NAMEs xmem, d3d12_xs, AcpHal,
# XCRDAPI, chad, xapu, xg_xs, GameInput) and the build revision stamp (cmake --build --preset runtime); a run needs nothing
# else, so the full build (hundreds of test executables) is skipped unless -FullBuild asks for it.
$runDlls = @("xmem", "d3d12_xs", "AcpHal", "XCRDAPI", "chad", "xapu", "xg_xs", "GameInput")

function Get-MissingDlls {
    return @($runDlls | Where-Object { -not (Test-Path -LiteralPath (Join-Path $buildRoot "$_.dll")) })
}

$python = Join-Path $repo "out\pe-env\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { $python = "python" }
$pack = Join-Path $PSScriptRoot "tester_pack.py"

$source = Get-CheckoutRevision
$wanted = if ($source.Dirty) { "$($source.Commit)-dirty" } else { $source.Commit }
$built = Get-BuildRevision
$missing = Get-MissingDlls
# A build from another commit is used as it is when the runtime sources (src, cmake, CMakeLists.txt, CMakePresets.json)
# are the same there and here: a commit that changes only tools, tests or docs needs no rebuild (runtime_sources.py).
$buildCheck = if ($built -eq $wanted) { "same commit" } else { $null }
if ($built -and $built -ne $wanted -and $missing.Count -eq 0) {
    $why = (Invoke-Native { & $python -B (Join-Path $PSScriptRoot "runtime_sources.py") $repo $built } | Out-String).Trim()
    if ($LASTEXITCODE -eq 0) { $buildCheck = $why; Write-Host "tester_run: using the build: $why" }
    elseif ($why) { Write-Host "tester_run: $why" }
}
if (-not $buildCheck -or $missing.Count -gt 0) {
    $have = if ($built) { $built } else { "no build" }
    if ($built -eq $wanted) { $have = "$built without " + ($missing -join ", ") + ".dll" }
    if ($NoBuild) {
        throw "tester_run: the build in $buildRoot is from $have, this checkout is $wanted. Build it (cmake --build `"$buildRoot`") or run without -NoBuild."
    }
    if (-not (Test-Path -LiteralPath (Join-Path $buildRoot "CMakeCache.txt"))) {
        if ($BuildDir -ne "out/build/release") { throw "tester_run: $buildRoot is not configured (run cmake -S `"$repo`" -B `"$buildRoot`" first)" }
        Write-Host "tester_run: configuring $buildRoot (cmake --preset release)"
        Push-Location $repo
        try { Invoke-Native { & cmake --preset release } } finally { Pop-Location }
        if ($LASTEXITCODE -ne 0) { throw "tester_run: the configure failed (cmake exit $LASTEXITCODE)" }
    }
    if ($FullBuild) {
        Write-Host "tester_run: build is from $have, checkout is ${wanted}: building everything (-FullBuild)"
        Invoke-Native { & cmake --build $buildRoot }
    } else {
        Write-Host "tester_run: build is from $have, checkout is ${wanted}: building startup_runtime (the run's targets)"
        Invoke-Native { & cmake --build $buildRoot --target startup_runtime }
    }
    if ($LASTEXITCODE -ne 0) { throw "tester_run: the build failed (cmake exit $LASTEXITCODE)" }
    $built = Get-BuildRevision
    $missing = Get-MissingDlls
    if ($missing.Count -gt 0) { throw "tester_run: after the build these DLLs are missing in ${buildRoot}: " + ($missing -join ", ") }
    if ($built -ne $wanted) {
        throw "tester_run: after the build the revision is $built, expected $wanted"
    }
    $buildCheck = "built here"
}

# One shader cache folder per shader revision (the x2w-startup-shader-N strings of gfx2hlsl.py), kept across runs and across
# branch updates that leave the revision alone: the --watch translator needs minutes to catch up, so an emptied cache
# leaves pipelines untranslated (a black screen). A revision bump starts a new folder.
$runScript = Join-Path $PSScriptRoot "run.ps1"
$runParams = (Get-Command $runScript).Parameters
$cacheRoot = if ([System.IO.Path]::IsPathRooted($ShaderCacheRoot)) { $ShaderCacheRoot } else { Join-Path $repo $ShaderCacheRoot }
$cacheTag = (Invoke-Native { & $python -B $pack cache-tag (Join-Path $PSScriptRoot "gfx2hlsl.py") } | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or -not $cacheTag) { throw "tester_run: cannot read the shader revisions of gfx2hlsl.py" }
$cache = Join-Path $cacheRoot "shader_cache_$cacheTag"
New-Item -ItemType Directory -Force $cache | Out-Null
$cacheAtStart = (Invoke-Native { & $python -B $pack cache-stats $cache } | Out-String).Trim() -split " "
# Pre-translation: after a revision bump the new cache folder is empty and the title would be black until the --watch
# translator caught up. The requests of the older shader_cache_* folders are copied under the new keys first (the translator
# of this checkout hashes them; no old result is reused), so the watcher the run starts has a backlog it works through in
# its process pool (all cores but one) while the title boots; what the title asks for meanwhile goes first. Never blocks the run.
$translateJobs = [Math]::Max(1, [Environment]::ProcessorCount - 1)
if (-not $env:X2W_TRANSLATE_JOBS) { $env:X2W_TRANSLATE_JOBS = "$translateJobs" }
$seedFile = Join-Path ([System.IO.Path]::GetTempPath()) "tester_seed_$PID.json"
$preTranslate = $null
# The re-keying pool is a child of this process: it runs at BELOW_NORMAL, like the --watch translator, and this process
# gets its own priority back afterwards.
$self = [System.Diagnostics.Process]::GetCurrentProcess()
$ownPriority = $self.PriorityClass
try {
    try { $self.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal } catch { Write-Warning "could not lower the priority: $($_.Exception.Message)" }
    $seedLine = (Invoke-Native { & $python -B $pack seed-cache $cacheRoot $cache --jobs $translateJobs --json $seedFile } | Out-String).Trim()
    if (Test-Path -LiteralPath $seedFile) { $preTranslate = Get-Content -Raw -LiteralPath $seedFile | ConvertFrom-Json }
} catch {
    $seedLine = "pre-translate: skipped ($($_.Exception.Message))"
} finally {
    try { $self.PriorityClass = $ownPriority } catch { }
}
Write-Host "tester_run: $seedLine"
Remove-Item -LiteralPath $seedFile -Force -ErrorAction SilentlyContinue
$script = (Invoke-Native { & $python -B $pack input-preset $InputMode $Seconds })
if ($LASTEXITCODE -ne 0) { throw "tester_run: the input script was refused (see above)" }

$stamp = Get-Date -Format "yyyyMMdd-HHmm"
$outRoot = if ([System.IO.Path]::IsPathRooted($OutDir)) { $OutDir } else { Join-Path $repo $OutDir }
$logDir = Join-Path $outRoot "run_$stamp"
New-Item -ItemType Directory -Force $logDir | Out-Null

$physicalBytes = [int64](Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
if ($MemoryGiB -le 0) { $MemoryGiB = [int](Invoke-Native { & $python -B $pack memory-gib $physicalBytes }) }
Write-Host ("tester_run: job memory cap {0} GiB ({1:N1} GiB physical RAM)" -f $MemoryGiB, ($physicalBytes / 1GB))
$env:X2W_D3D12_BUDGET = "1"
$LazyTextures = -not $NoLazyTextures
if ($LazyTextures) { $env:X2W_D3D12_LAZY_TEXTURES = "1" } else { Remove-Item Env:X2W_D3D12_LAZY_TEXTURES -ErrorAction SilentlyContinue }
$SysmemFallback = -not $NoSysmemFallback
if ($SysmemFallback) { $env:X2W_D3D12_SYSMEM_FALLBACK = "1" } else { Remove-Item Env:X2W_D3D12_SYSMEM_FALLBACK -ErrorAction SilentlyContinue }
$GpuTime = -not $NoGpuTime
if ($GpuTime) { $env:X2W_GPU_TIME = "1" } else { Remove-Item Env:X2W_GPU_TIME -ErrorAction SilentlyContinue }
$StartupBelow40 = -not $NoStartupBelow40
if ($StartupBelow40) { $env:X2W_XMEM_STARTUP_BELOW_40 = "1" } else { Remove-Item Env:X2W_XMEM_STARTUP_BELOW_40 -ErrorAction SilentlyContinue }
$LowAddresses = -not $NoLowAddresses
if ($LowAddresses) { $env:X2W_LOW_ADDRESSES = "1" } else { Remove-Item Env:X2W_LOW_ADDRESSES -ErrorAction SilentlyContinue }
$TitleInputSnapshot = -not $NoTitleInputSnapshot
if ($TitleInputSnapshot) { $env:X2W_TRANSLATED_TITLE_INPUT_SNAPSHOT = "1"; $env:X2W_TRANSLATED_INDIRECT_COPY_RANGES = "1" } else { Remove-Item Env:X2W_TRANSLATED_TITLE_INPUT_SNAPSHOT, Env:X2W_TRANSLATED_INDIRECT_COPY_RANGES -ErrorAction SilentlyContinue }
$runArgs = @{
    Exe = $Exe; DataDir = $DataDir; LogDir = $logDir; Seconds = $Seconds; BuildDir = $BuildDir
    MemoryGiB = $MemoryGiB; ShaderCache = $cache; ContinueAsserts = 1000; InputScript = $script
    Render = $true; Translate = $true; FakeUser = $true; GpuMirror = $true
}
if ($Deep) { $runArgs.TraceCalls = $true; $runArgs.TraceApi = $true }
if (-not $NoLazyMemory) { $runArgs.LazyMemory = $true }
$runArgs.NoNullIdentity = $true   # X2W_DRAW_NULL_IDENTITY=1 replaced the real address of an empty buffer slot in the log (Run I); the tester run keeps it
if ($CommitHeadroomMiB -gt 0) { $runArgs.CommitHeadroomMiB = $CommitHeadroomMiB }
if ($CommitWarnMiB -gt 0) { $runArgs.CommitWarnMiB = $CommitWarnMiB }
if ($CaptureEvery -gt 0) {
    if ($runParams.ContainsKey("CaptureEvery")) { $runArgs.CaptureEvery = $CaptureEvery }
    else { Write-Warning "run.ps1 has no -CaptureEvery here: only the final window.png is saved" }
}
$exitCode = 0
try {
    & $runScript @runArgs
    $exitCode = $LASTEXITCODE
} catch {
    Write-Warning "tester_run: the run stopped with an error: $($_.Exception.Message)"
    $exitCode = 1
}

# Manifest input: X2W_* variables only; the user's profile path is masked.
$envSet = [ordered]@{}
foreach ($item in (Get-ChildItem Env: | Where-Object { $_.Name -like "X2W_*" } | Sort-Object Name)) {
    $value = $item.Value
    if ($env:USERPROFILE) { $value = $value.Replace($env:USERPROFILE, "%USERPROFILE%") }
    $envSet[$item.Name] = $value
}
$video = @(Get-CimInstance Win32_VideoController | Where-Object { $_.Name -notmatch "Basic Display|Remote Display" })
$os = Get-CimInstance Win32_OperatingSystem
$manifest = [ordered]@{
    commit = $source.Commit; build_commit = $built; build_check = $buildCheck; dirty = $source.Dirty; seconds = $Seconds
    input_script = $script; env = $envSet
    gpu = (($video | ForEach-Object { $_.Name }) -join "; ")
    gpu_driver = (($video | ForEach-Object { $_.DriverVersion }) -join "; ")
    windows_build = $os.BuildNumber; wall_seconds = [math]::Round($clock.Elapsed.TotalSeconds, 1)
    exit_code = $exitCode; deep = [bool]$Deep; lazy_textures = [bool]$LazyTextures; sysmem_fallback = [bool]$SysmemFallback; gpu_time = [bool]$GpuTime; commit_headroom_mib = $CommitHeadroomMiB; commit_warn_mib = $CommitWarnMiB; startup_below_40 = [bool]$StartupBelow40; low_addresses = [bool]$LowAddresses; title_input_snapshot = [bool]$TitleInputSnapshot; lazy_memory = (-not $NoLazyMemory); memory_gib = $MemoryGiB; physical_ram_gib = [math]::Round($physicalBytes / 1GB, 1); shader_cache = (Split-Path -Leaf $cache)
    pretranslate = $preTranslate
    translations_at_start = [int]$cacheAtStart[0]; requests_at_start = [int]$cacheAtStart[1]; failed_at_start = [int]$cacheAtStart[2]
}
$manifestFile = Join-Path ([System.IO.Path]::GetTempPath()) "tester_manifest_${stamp}_$PID.json"
[System.IO.File]::WriteAllText($manifestFile, ($manifest | ConvertTo-Json -Depth 4), (New-Object System.Text.UTF8Encoding($false)))

$zip = Join-Path $outRoot ("report_{0}_{1}.zip" -f $source.Commit, $stamp)
$parts = @(Invoke-Native { & $python -B $pack pack $logDir $manifestFile $zip --limit-mb $LimitMB --cache $cache })
if ($LASTEXITCODE -ne 0) { throw "tester_run: packing the logs failed (see above); the logs stay in $logDir" }
Remove-Item -LiteralPath $manifestFile -Force
foreach ($line in $parts) {
    $path, $bytes = $line -split " (?=\d+$)"
    Write-Host ("Send this file: {0} ({1:N1} MB)" -f $path, ([double]$bytes / 1MB))
}
Write-Host ("Shader cache {0}: {1} translations were already present at start ({2} requests, {3} failed)." -f (Split-Path -Leaf $cache), $cacheAtStart[0], $cacheAtStart[1], $cacheAtStart[2])
Write-Host (Invoke-Native { & $python -B $pack first-poll $logDir })
if ($parts.Count -gt 1) { Write-Host "The report has $($parts.Count) parts: send all of them." }

# F1: the report is packed, now finish what the run left. One translator pass over the cache (no --watch) until nothing is
# pending or -FinishTranslateSeconds is over, so the next run starts with every request already translated.
if ($FinishTranslateSeconds -gt 0) {
    $before = (Invoke-Native { & $python -B $pack cache-stats $cache } | Out-String).Trim() -split " "
    # Pending = no result at all. A refusal is a result (c25: a refusal of this translator build is final): counting it as pending
    # printed "2616 pending" for requests that were all refused.
    $pendingBefore = [math]::Max(0, [int]$before[1] - [int]$before[0] - [int]$before[2])
    $finishOut = Join-Path $outRoot "finish_translate_$stamp.txt"
    $finishErr = Join-Path $outRoot "finish_translate_$stamp.err.txt"
    Write-Host ("tester_run: finishing the translation: {0} request(s) pending ({1} refused earlier, retried only if the translator changed), one pass, at most {2} s (the report above is already packed; do not close this window)" -f $pendingBefore, [int]$before[2], $FinishTranslateSeconds)
    $finishClock = [System.Diagnostics.Stopwatch]::StartNew()
    $translator = Start-Process -FilePath $python -PassThru -NoNewWindow -RedirectStandardOutput $finishOut -RedirectStandardError $finishErr `
        -ArgumentList @("-B", "`"$(Join-Path $PSScriptRoot 'gfx2hlsl.py')`"", "--once", "`"$cache`"")
    $null = $translator.Handle    # keeps the exit code readable after the process ends
    $capped = -not $translator.WaitForExit($FinishTranslateSeconds * 1000)
    if ($capped) { $null = Invoke-Native { & taskkill.exe /T /F /PID $translator.Id }; $null = $translator.WaitForExit(10000) }
    $after = (Invoke-Native { & $python -B $pack cache-stats $cache } | Out-String).Trim() -split " "
    $pendingAfter = [math]::Max(0, [int]$after[1] - [int]$after[0] - [int]$after[2])
    $note = if ($capped) { ", stopped at the cap" } elseif ($translator.ExitCode -ne 0) { ", translator exit code $($translator.ExitCode) (see $finishErr)" } else { "" }
    Write-Host ("tester_run: finish translation done in {0:N0} s: pending before {1}, after {2}; ok {3}, refused {4} in total ({5} new){6}" -f `
        $finishClock.Elapsed.TotalSeconds, $pendingBefore, $pendingAfter, [int]$after[0], [int]$after[2], ([int]$after[2] - [int]$before[2]), $note)
}
exit $exitCode
