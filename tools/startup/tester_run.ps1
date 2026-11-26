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
   finished, threads, files, trap.log, translate, translate_err, window*.png, bink_luma.png, memory.txt ...), the debugger
   summary and manifest.json, and prints each path and size. Not included: the shader cache, memory dumps and game
   files. Over -LimitMB (default 9, so each part fits a mail as it is: never re-zip; every part carries part_NN_of_MM.txt, which says what each part holds) the zip is split into numbered parts
   (report_..._part01of03.zip); no log is dropped.

The launcher's job memory cap (run.ps1 -MemoryGiB, default 14) is set to 75% of physical RAM, at least 14 GiB: a run whose
commit passes the cap dies with error 1455. -MemoryGiB overrides it; the summary prints it and the manifest records it.

Lazy memory commit is ON by default (run.ps1 -LazyMemory): physical pages and the large startup blocks are committed on
first touch, which cuts the section commit. -NoLazyMemory turns it off. The manifest records it.
The shader cache is repaired once before the run (tools/startup/shader_cache_repair_packed.py: translations made before the packed-position derivation are withdrawn so the translator redoes them, no revision bump); the console says "repaired N packed-position translations" and the manifest records cache_repaired (-1 = skipped). -NoCacheRepair skips it.

When the report is packed the script finishes the translation: one translator pass over the shader cache (no --watch), until
nothing is pending or -FinishTranslateSeconds (default 300, 0 = skip) is over, with at most 4 workers at BelowNormal (Run Q:
16 workers at 100% CPU), then prints "pending before/after, ok, refused". It is safe to stop: what is written is kept.
The next run then starts with every request the title made already translated. The mail does not wait for this step.

-Deep also sets -TraceCalls and -TraceApi: a slower run with more detail in the logs.

Lazy slot 133 textures are ON for now: X2W_D3D12_LAZY_TEXTURES=1 creates read-only slot 133 textures on first use, which
keeps video memory from running out. -NoLazyTextures turns it off. The manifest records it.

-Input accept (the default): ONLY the A button (mask 4), first at 35 s after the first pad poll, then every 10 s; a 0.3 s tap, every 3rd press held 1.5 s (movie skips); no press from 420 s on (gameplay) or after the run; never Start (it opens the pause menu in the game). -Input none (also off): no automatic presses at all, the title only sees the tester's own input. -Input menus: D-pad down, D-pad down, A, one press every second (each held 0.8 s) starting 5 s after the first pad poll, for the first 256 presses (about 260 s; input script
time counts from that poll, which the title reaches late).
-PrivacyBoxes N (default 2, with -Input menus): the privacy screen has N checkboxes and a Conferma button. The script does not depend on
the clock or on the state of the boxes: from 5 s it repeats passes of "up x(N+1), toggle at most one box with A, down to Conferma, A",
toggling a different single box in Gray-code order (none, box 1, box 2, box 1), so one pass in four reaches Conferma with both boxes
ticked whatever state they start in; Conferma changes the screen and the presses go nowhere. On the screens before it the same passes
only select items and end on the Avanti button, which opens the privacy screen. 0 keeps the plain DOWN,DOWN,A pattern.
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
    [Alias("Input")] [string]$InputMode = "accept",   # $Input is a PowerShell automatic variable: the parameter is -InputMode, -Input is its alias
    [int]$CaptureMenuEvery = 5,                 # during the menu phase (until the first loading-like picture) a picture this often; 0 = only -CaptureEvery
    [int]$CaptureEvery = 15,                    # seconds between window captures; 0 = only the final window.png
    [string]$BuildDir = "out/build/release",
    [string]$OutDir = "out/tester",
    [string]$ShaderCacheRoot = "out/startup",     # one folder per shader revision inside: shader_cache_<rev>
    [int]$LimitMB = 9,
    [switch]$FullBuild,                         # build every target (cmake --build with no --target) instead of only the run's
    [switch]$NoBuild,                           # stop instead of building when the build is not from this commit
    [switch]$Deep,                              # also -TraceCalls and -TraceApi: slower, more detailed
    [int]$MemoryGiB = 0,                        # job memory cap of the launcher; 0 = 75% of physical RAM (at least 14)
    [int]$FinishTranslateSeconds = 300,         # after the report is packed: one translator pass over the cache, at most this long; 0 = skip
    [int]$PrivacyBoxes = 2,                     # -Input menus: the privacy screen has this many checkboxes (2 in Run L/M); the script repeats clock-free passes (up x(N+1), at most one box toggled, down to Conferma, A) until the screen changes; 0 = the plain DOWN,DOWN,A pattern
    [switch]$NoCacheRepair,                     # skip the one-off repair of cached packed-position translations before the run (tools/startup/shader_cache_repair_packed.py)
    [switch]$NoLazyMemory,                      # lazy memory commit is on (run.ps1 -LazyMemory: X2W_XMEM_LAZY_COMMIT, _LAZY_STARTUP, _SHARED_VIEWS); this turns it off
    [switch]$NoLazyTextures,                    # lazy slot 133 textures are on (X2W_D3D12_LAZY_TEXTURES=1, PR #2773); this turns them off
    [switch]$NoStartupBelow40,                  # the 0x283 startup blocks are placed below 2^40 (X2W_XMEM_STARTUP_BELOW_40=1, #1758); this turns it off
    [switch]$NoLowAddresses,                   # the title starts with bottom-up ASLR off so its memory stays below 2^40 (X2W_LOW_ADDRESSES=1); this turns it off
    [switch]$NoTitleInputSnapshot,              # translated compute inputs in a placed buffer that does not alias title memory are read from title memory (X2W_TRANSLATED_TITLE_INPUT_SNAPSHOT=1, with X2W_TRANSLATED_INDIRECT_COPY_RANGES=1 for ranges a list copied); this turns both off
    [switch]$NoSysmemFallback,                 # a texture that runs out of video memory is created in system memory (X2W_D3D12_SYSMEM_FALLBACK=1); this turns it off
    [string[]]$ScanRoots = @(),                # folders searched after the run for files the title wrote (default: AppData, LocalAppData, Documents, Saved Games, ProgramData, the data folder): shows where it keeps a choice such as the privacy acceptance
    [switch]$FreshSave,                       # wipe the saved game folder first (out/startup/saves, or -SaveRoot): the title starts as a new profile and shows its legal/privacy screens again
    [string]$SaveRoot = "out/startup/saves",      # X2W_STORAGE_PATH for the title: its saves and persistent local storage, kept across runs so a screen accepted once is skipped next time
    [switch]$NoHddMode,                       # the DirectStorage stand-in plans, merges and prefetches the title's file reads for a hard disk (X2W_STORAGE_HDD=1); this turns it off
    [switch]$NoGpuTime,                        # GPU time per frame is logged every 300 frames (X2W_GPU_TIME=1, timestamp queries); this turns it off
    [int]$CommitHeadroomMiB = 256,              # last resort before error 1455: the run ends cleanly (logs packed) when the system commit has less than this free; 0 = off
    [int]$CommitWarnMiB = 2048,                 # a warning line in the log when the system commit has less than this free; 0 = off
    [int]$StallSeconds = 90,                    # stall watchdog: the run ends cleanly (zip packed) when frames presented, storage read, translator output and the window picture have all stood still this long; 0 = off
    [int]$StallMinSeconds = 150,                # the watchdog never ends a run before this many seconds
    [int]$MemoryStopPercent = 95                # the watchdog also ends the run when the committed memory is at or above this percent of the commit limit in 2 samples in a row; 0 = off
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$buildRoot = if ([System.IO.Path]::IsPathRooted($BuildDir)) { $BuildDir } else { Join-Path $repo $BuildDir }
$clock = [System.Diagnostics.Stopwatch]::StartNew()
$runStart = Get-Date

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
# Repair first (before the translator and before the counts below): results translated before the packed-position derivation
# are withdrawn so the translator makes them again (tools/startup/shader_cache_repair_packed.py; no revision bump, so the rest of the
# warm cache stays). -NoCacheRepair skips it. repaired = -1 in the manifest when skipped or when the tool failed.
$cacheRepaired = -1
if (-not $NoCacheRepair) {
    try {
        $repairTool = Join-Path $PSScriptRoot "shader_cache_repair_packed.py"
        $repairLine = (Invoke-Native { & $python -B $repairTool $cache } | Out-String).Trim()
        if ($LASTEXITCODE -eq 0 -and $repairLine -match '(\d+) stale results withdrawn') {
            $cacheRepaired = [int]$Matches[1]
            Write-Host "tester_run: repaired $cacheRepaired packed-position translations"
        } else {
            Write-Warning "tester_run: the cache repair tool gave no usable answer, cache left as it was: $repairLine"
        }
    } catch {
        Write-Warning "tester_run: cache repair skipped ($($_.Exception.Message))"
    }
}
$cacheAtStart = (Invoke-Native { & $python -B $pack cache-stats $cache } | Out-String).Trim() -split " "
# Pre-translation: after a revision bump the new cache folder is empty and the title would be black until the --watch
# translator caught up. The requests of the older shader_cache_* folders are copied under the new keys first (the translator
# of this checkout hashes them; no old result is reused), so the watcher the run starts has a backlog it works through
# while the title boots; what the title asks for meanwhile goes first. Never blocks the run. The re-keying uses all cores
# but one; the --watch pool beside the title keeps to 4 workers and leaves old refusals alone (gfx2hlsl.py watch_jobs),
# and the finish pass after the run uses every core. X2W_TRANSLATE_JOBS, when the tester sets it, overrides both pools.
$translateJobs = [Math]::Max(1, [Environment]::ProcessorCount - 1)
# The pre-run retry and the pass after the run keep to a few workers, so the PC stays usable (Run Q: the pass after the
# run took 16 workers at 100% CPU and the tester had to kill it).
$backgroundJobs = 4
# The --watch pool beside the title leaves old refusals for the pass after the run, so a translator lift would reach
# only the next run: every old refusal (a .fail an older build wrote) is translated again now, on 4 workers, for at
# most 180 s: the families recent builds lift first (gfx2hlsl.py LIFTED_REASONS), then graphics before compute.
# Never blocks the run.
# A cache without any refusal has nothing to retry: the translator is not even started (it takes seconds to load).
$anyFailure = Get-ChildItem -LiteralPath $cache -Filter "*.fail" -File -ErrorAction SilentlyContinue | Select-Object -First 1
# It is not the re-keying below ("the new cache already holds requests" is that step's skip): it always runs when the
# cache holds a .fail, and its line goes into the manifest as pre_run_retry.
$preRunRetry = "pre-run retry skipped: no refusal in the cache"
if ($anyFailure) { try {
    # The last run's stderr.txt ranks the retry by draws/dispatches (the per-key census lines), when there is one.
    $runsRoot = if ([System.IO.Path]::IsPathRooted($OutDir)) { $OutDir } else { Join-Path $repo $OutDir }
    $lastStderr = Get-ChildItem -LiteralPath $runsRoot -Directory -Filter "run_*" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | ForEach-Object { Join-Path $_.FullName "stderr.txt" } |
        Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    $retryArgs = @("--retry-lifted", $cache, "180", "--jobs", "$backgroundJobs")
    if ($lastStderr) { $retryArgs += @("--weights", $lastStderr) }
    $retryLine = (Invoke-Native { & $python -B (Join-Path $PSScriptRoot "gfx2hlsl.py") @retryArgs } | Out-String).Trim()
    $preRunRetry = if ($retryLine) { $retryLine } else { "pre-run retry: no output (translator exit code $LASTEXITCODE)" }
} catch {
    $preRunRetry = "pre-run retry skipped: $($_.Exception.Message)"
} }
Write-Host "tester_run: $preRunRetry"
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
$script = (Invoke-Native { & $python -B $pack input-preset $InputMode $Seconds --privacy-boxes $PrivacyBoxes })
if ($LASTEXITCODE -ne 0) { throw "tester_run: the input script was refused (see above)" }
# No script (-Input none): run.ps1 sets X2W_INPUT_SCRIPT only for a non-empty script, so clear one left in this shell's environment.
if (-not $script) { Remove-Item Env:X2W_INPUT_SCRIPT -ErrorAction SilentlyContinue }
# One intent label per timed press ("privacy pass 2: toggle box 1", ...), kept in the manifest next to input_script; analyze_run joins
# them with the "x2w-input script press item N" lines (item N = label N). Empty for scripts that are not the menus preset.
$inputLabels = @()
try {
    $labelJson = (Invoke-Native { & $python -B $pack input-labels $InputMode $Seconds --privacy-boxes $PrivacyBoxes } | Out-String).Trim()
    if ($LASTEXITCODE -eq 0 -and $labelJson) { $inputLabels = @((ConvertFrom-Json $labelJson) | ForEach-Object { $_ }) }
} catch { Write-Warning "tester_run: no input labels ($($_.Exception.Message))" }

$stamp = Get-Date -Format "yyyyMMdd-HHmm"
$outRoot = if ([System.IO.Path]::IsPathRooted($OutDir)) { $OutDir } else { Join-Path $repo $OutDir }
# A second run within the same minute gets its own folder: a reused one hands the stall watchdog the last run's
# finished.txt (it stops watching at once), its stop file and stall_stop.json, and packs the last run's files again.
if (Test-Path -LiteralPath (Join-Path $outRoot "run_$stamp")) { $stamp = Get-Date -Format "yyyyMMdd-HHmmss" }
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
$saveDir = if ([System.IO.Path]::IsPathRooted($SaveRoot)) { $SaveRoot } else { Join-Path $repo $SaveRoot }
if ($FreshSave -and (Split-Path -Leaf $saveDir) -notmatch "save") { throw "tester_run: -FreshSave only wipes a folder whose name contains save (got $saveDir)" }
if ($FreshSave -and (Test-Path -LiteralPath $saveDir)) { Remove-Item -LiteralPath $saveDir -Recurse -Force }
New-Item -ItemType Directory -Force $saveDir | Out-Null
$env:X2W_STORAGE_PATH = $saveDir.TrimEnd('\', '/') + '\'
$saveBefore = @(Get-ChildItem -LiteralPath $saveDir -Recurse -File -ErrorAction SilentlyContinue)
Write-Host ("tester_run: saves in {0} ({1} files kept from earlier runs{2})" -f $saveDir, $saveBefore.Count, $(if ($FreshSave) { ', -FreshSave wiped it' } else { '' }))
$HddMode = -not $NoHddMode
if ($HddMode) { $env:X2W_STORAGE_HDD = "1" } else { Remove-Item Env:X2W_STORAGE_HDD -ErrorAction SilentlyContinue }
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
if ($Deep) { $runArgs.TraceCalls = $true; $runArgs.TraceApi = $true; $runArgs.TraceFiles = 1 }
elseif ($runParams.ContainsKey("TraceFiles")) { $runArgs.TraceFiles = 0 }   # files.txt lists every open (one per ~5 ms on a hard disk): off unless -Deep
if (-not $NoLazyMemory) { $runArgs.LazyMemory = $true }
$runArgs.NoNullIdentity = $true   # X2W_DRAW_NULL_IDENTITY=1 replaced the real address of an empty buffer slot in the log (Run I); the tester run keeps it
if ($CommitHeadroomMiB -gt 0) { $runArgs.CommitHeadroomMiB = $CommitHeadroomMiB }
if ($CommitWarnMiB -gt 0) { $runArgs.CommitWarnMiB = $CommitWarnMiB }
if ($CaptureEvery -gt 0) {
    if ($runParams.ContainsKey("CaptureEvery")) { $runArgs.CaptureEvery = $CaptureEvery }
    if ($CaptureMenuEvery -gt 0 -and $runParams.ContainsKey("CaptureMenuEvery")) { $runArgs.CaptureMenuEvery = $CaptureMenuEvery }
    else { Write-Warning "run.ps1 has no -CaptureEvery here: only the final window.png is saved" }
}
# memory.txt in the run folder (so in the zip): every 30 s the system commit and the 8 processes holding the most of it,
# names and GiB only (memory_top.py), then one "end" line after the run. Never stops the run.
$memoryTop = Join-Path $PSScriptRoot "memory_top.py"
$memoryFile = Join-Path $logDir "memory.txt"
$memorySampler = $null
try {
    $memorySampler = Start-Process -FilePath $python -PassThru -NoNewWindow `
        -ArgumentList "-B", "`"$memoryTop`"", "`"$memoryFile`"", "--every", "30", "--top", "8", "--parent", "$PID"
} catch { Write-Warning "tester_run: memory sampling is off: $($_.Exception.Message)" }
# Manifest input: X2W_* variables only; the user's profile path is masked.
$envSet = [ordered]@{}
foreach ($item in (Get-ChildItem Env: | Where-Object { $_.Name -like "X2W_*" } | Sort-Object Name)) {
    $value = $item.Value
    if ($env:USERPROFILE) { $value = $value.Replace($env:USERPROFILE, "%USERPROFILE%") }
    $envSet[$item.Name] = $value
}
$video = @(Get-CimInstance Win32_VideoController | Where-Object { $_.Name -notmatch "Basic Display|Remote Display" })
$os = Get-CimInstance Win32_OperatingSystem
$zip = Join-Path $outRoot ("report_{0}_{1}.zip" -f $source.Commit, $stamp)
# The linker map of our d3d12_xs.dll names its functions in the zip; only a map written with this DLL (within a minute of it).
$mapArgs = @()
$dllFile = Join-Path $buildRoot "d3d12_xs.dll"
$mapFile = Join-Path $buildRoot "d3d12_xs.map"
if ((Test-Path -LiteralPath $dllFile) -and (Test-Path -LiteralPath $mapFile) -and
    ([math]::Abs(((Get-Item -LiteralPath $mapFile).LastWriteTimeUtc - (Get-Item -LiteralPath $dllFile).LastWriteTimeUtc).TotalSeconds) -lt 60)) {
    $mapArgs = @("--map", $mapFile)
}
# Janitor (tester_janitor.py), in a window of its own so that closing this console does not end it: if this script is stopped
# (Ctrl-C) or its window is closed before the report is packed, it ends the run, ends the translator and packs the report from
# this early manifest (interrupted = true). After a normal pack tester_run writes pack_done and it only sweeps leftovers.
$janitorFile = Join-Path ([System.IO.Path]::GetTempPath()) "tester_janitor_${stamp}_$PID.json"
try {
    $early = [ordered]@{
        commit = $source.Commit; build_commit = $built; dirty = $source.Dirty; seconds = $Seconds; input_script = $script; env = $envSet
        gpu = (($video | ForEach-Object { $_.Name }) -join "; "); gpu_driver = (($video | ForEach-Object { $_.DriverVersion }) -join "; ")
        windows_build = $os.BuildNumber; wall_seconds = 0; deep = [bool]$Deep; lazy_textures = [bool]$LazyTextures
        lazy_memory = (-not $NoLazyMemory); memory_gib = $MemoryGiB; shader_cache = (Split-Path -Leaf $cache)
        translations_at_start = [int]$cacheAtStart[0]; requests_at_start = [int]$cacheAtStart[1]; failed_at_start = [int]$cacheAtStart[2]
    }
    $job = [ordered]@{ manifest = $early; zip = $zip; limit_mb = $LimitMB; cache = $cache; map = $(if ($mapArgs.Count) { $mapArgs[1] } else { "" }) }
    [System.IO.File]::WriteAllText($janitorFile, ($job | ConvertTo-Json -Depth 5), (New-Object System.Text.UTF8Encoding($false)))
    $janitorArgs = @("-B", "`"$(Join-Path $PSScriptRoot 'tester_janitor.py')`"", "`"$logDir`"", "--parent", "$PID", "--pack-json", "`"$janitorFile`"", "--cache", "`"$cache`"")
    Start-Process -FilePath $python -WindowStyle Hidden -ArgumentList $janitorArgs `
        -RedirectStandardError "$janitorFile.err" -RedirectStandardOutput "$janitorFile.out" | Out-Null
} catch { Write-Warning "tester_run: the janitor is off (Ctrl-C or a closed window will not pack the report): $($_.Exception.Message)" }
# Stall watchdog (stall_watchdog.py): ends the run through the `stop` file when it is clearly stuck; the reason is printed as
# "tester_run: STALL STOP at N s: ..." and recorded as stop_reason in the manifest.
$stallFile = Join-Path $logDir "stall_stop.json"
$stallWatch = $null
if ($StallSeconds -gt 0 -or $MemoryStopPercent -gt 0) {
    try {
        $stallArgs = @("-B", "`"$(Join-Path $PSScriptRoot 'stall_watchdog.py')`"", "`"$logDir`"", "--parent", "$PID", "--seconds", "$Seconds",
            "--stall-seconds", "$StallSeconds", "--min-seconds", "$StallMinSeconds", "--memory-percent", "$MemoryStopPercent")
        $stallWatch = Start-Process -FilePath $python -PassThru -NoNewWindow -ArgumentList $stallArgs
    } catch { Write-Warning "tester_run: the stall watchdog is off: $($_.Exception.Message)" }
}
$exitCode = 0
try {
    & $runScript @runArgs
    $exitCode = $LASTEXITCODE
} catch {
    Write-Warning "tester_run: the run stopped with an error: $($_.Exception.Message)"
    $exitCode = 1
}
if ($stallWatch) { try { Stop-Process -Id $stallWatch.Id -Force -ErrorAction SilentlyContinue; [void]$stallWatch.WaitForExit(5000) } catch { } }
$stopReason = ""
if (Test-Path -LiteralPath $stallFile) {
    try { $stall = Get-Content -LiteralPath $stallFile -Raw | ConvertFrom-Json; $stopReason = "stall watchdog at $($stall.at_seconds) s: $($stall.reason)" } catch { }
}
# The end cause is always recorded (manifest end_cause): watchdog, timeout, stop request, title exited, killed from outside, crashed.
$endCause = ((Invoke-Native { & $python -B $pack end-cause $logDir }) -join " ").Trim()
Write-Host "tester_run: the run ended: $endCause"
if ($memorySampler) { try { Stop-Process -Id $memorySampler.Id -Force -ErrorAction SilentlyContinue; [void]$memorySampler.WaitForExit(5000) } catch { } }
Invoke-Native { & $python -B $memoryTop $memoryFile --once "end" --top 8 } | Out-Null

# Files written during this run outside the save folder (first 60, at most 20 s of searching, 4 levels deep): where the title keeps
# what it saves by itself, e.g. the privacy acceptance.
$scanRoots = if ($ScanRoots.Count) { $ScanRoots } else { @($env:LOCALAPPDATA, $env:APPDATA, [Environment]::GetFolderPath("MyDocuments"), (Join-Path $env:USERPROFILE "Saved Games"), $env:ProgramData, $DataDir) }
$scanClock = [System.Diagnostics.Stopwatch]::StartNew()
$changedFiles = @()
foreach ($root in $scanRoots) {
    if (-not $root -or -not (Test-Path -LiteralPath $root) -or $scanClock.Elapsed.TotalSeconds -gt 20) { continue }
    $changedFiles += @(Get-ChildItem -LiteralPath $root -Recurse -Depth 4 -File -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -ge $runStart -and $_.FullName.Substring($root.Length) -notmatch '\\(Temp|CrashDumps|Microsoft\\Windows|Packages|\.git)\\' -and -not $_.FullName.StartsWith($outRoot, [StringComparison]::OrdinalIgnoreCase) } |
        Select-Object -First 60 | ForEach-Object { "{0} ({1} bytes)" -f $_.FullName, $_.Length })
}
$saveAfter = @(Get-ChildItem -LiteralPath $saveDir -Recurse -File -ErrorAction SilentlyContinue)  # what the title's saves and persistent storage hold after this run
$manifest = [ordered]@{
    commit = $source.Commit; build_commit = $built; build_check = $buildCheck; dirty = $source.Dirty; seconds = $Seconds
    input_script = $script; env = $envSet
    gpu = (($video | ForEach-Object { $_.Name }) -join "; ")
    gpu_driver = (($video | ForEach-Object { $_.DriverVersion }) -join "; ")
    windows_build = $os.BuildNumber; wall_seconds = [math]::Round($clock.Elapsed.TotalSeconds, 1)
    exit_code = $exitCode; stop_reason = $stopReason; end_cause = $endCause; stall_seconds = $StallSeconds; stall_min_seconds = $StallMinSeconds; memory_stop_percent = $MemoryStopPercent; deep = [bool]$Deep; lazy_textures = [bool]$LazyTextures; sysmem_fallback = [bool]$SysmemFallback; gpu_time = [bool]$GpuTime; hdd_mode = [bool]$HddMode; commit_headroom_mib = $CommitHeadroomMiB; commit_warn_mib = $CommitWarnMiB; startup_below_40 = [bool]$StartupBelow40; low_addresses = [bool]$LowAddresses; title_input_snapshot = [bool]$TitleInputSnapshot; lazy_memory = (-not $NoLazyMemory); memory_gib = $MemoryGiB; physical_ram_gib = [math]::Round($physicalBytes / 1GB, 1); shader_cache = (Split-Path -Leaf $cache); save_root = $saveDir; fresh_save = [bool]$FreshSave; saves_at_start = $saveBefore.Count; saves_at_end = $saveAfter.Count; files_written = @($changedFiles | Select-Object -First 60); saves_files = @($saveAfter | Select-Object -First 40 | ForEach-Object { "{0} ({1} bytes)" -f $_.FullName.Substring($saveDir.Length).TrimStart([char]92), $_.Length })
    pretranslate = $preTranslate; pre_run_retry = $preRunRetry; cache_repaired = $cacheRepaired; input_labels = $inputLabels
    translations_at_start = [int]$cacheAtStart[0]; requests_at_start = [int]$cacheAtStart[1]; failed_at_start = [int]$cacheAtStart[2]
}
$manifestFile = Join-Path ([System.IO.Path]::GetTempPath()) "tester_manifest_${stamp}_$PID.json"
[System.IO.File]::WriteAllText($manifestFile, ($manifest | ConvertTo-Json -Depth 4), (New-Object System.Text.UTF8Encoding($false)))

$parts = @(Invoke-Native { & $python -B $pack pack $logDir $manifestFile $zip --limit-mb $LimitMB --cache $cache @mapArgs })
if ($LASTEXITCODE -ne 0) { throw "tester_run: packing the logs failed (see above); the logs stay in $logDir" }
Remove-Item -LiteralPath $manifestFile -Force
Set-Content -LiteralPath (Join-Path $logDir "pack_done") -Value "$PID"   # the janitor then only sweeps leftovers
Remove-Item -LiteralPath $janitorFile -Force -ErrorAction SilentlyContinue
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
    Write-Host "tester_run: safe to stop now, results kept (the report is packed; every translation already written stays in the cache)"
    Write-Host ("tester_run: finishing the translation: {0} request(s) pending ({1} refused earlier, retried only if the translator changed), one pass, at most {2} s (the report above is already packed; do not close this window)" -f $pendingBefore, [int]$before[2], $FinishTranslateSeconds)
    $finishClock = [System.Diagnostics.Stopwatch]::StartNew()
    $translator = Start-Process -FilePath $python -PassThru -NoNewWindow -RedirectStandardOutput $finishOut -RedirectStandardError $finishErr `
        -ArgumentList @("-B", "`"$(Join-Path $PSScriptRoot 'gfx2hlsl.py')`"", "--once", "`"$cache`"", "--jobs", "$backgroundJobs")
    $null = $translator.Handle    # keeps the exit code readable after the process ends
    try { $translator.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal } catch { }
    $capped = -not $translator.WaitForExit($FinishTranslateSeconds * 1000)
    if ($capped) { $null = Invoke-Native { & taskkill.exe /T /F /PID $translator.Id }; $null = $translator.WaitForExit(10000) }
    $after = (Invoke-Native { & $python -B $pack cache-stats $cache } | Out-String).Trim() -split " "
    $pendingAfter = [math]::Max(0, [int]$after[1] - [int]$after[0] - [int]$after[2])
    $note = if ($capped) { ", stopped at the cap" } elseif ($translator.ExitCode -ne 0) { ", translator exit code $($translator.ExitCode) (see $finishErr)" } else { "" }
    Write-Host ("tester_run: finish translation done in {0:N0} s: pending before {1}, after {2}; ok {3}, refused {4} in total ({5} new){6}" -f `
        $finishClock.Elapsed.TotalSeconds, $pendingBefore, $pendingAfter, [int]$after[0], [int]$after[2], ([int]$after[2] - [int]$before[2]), $note)
}
exit $exitCode
