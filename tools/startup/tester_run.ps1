<#
.SYNOPSIS
One command for a tester: check the build, run the title offline with a per-commit shader cache, mail-sized zip at the end.

.DESCRIPTION
1. Checks that the build in -BuildDir was made from this checkout's current commit (the revision the build writes
   into generated\x2w_build_rev.cpp). If not, or a DLL run.ps1 copies is missing, it builds only the startup_runtime target (the eight DLLs and the
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

-Deep also sets -TraceCalls and -TraceApi: a slower run with more detail in the logs.

Lazy slot 133 textures are ON for now: X2W_D3D12_LAZY_TEXTURES=1 creates read-only slot 133 textures on first use, which
keeps video memory from running out. -NoLazyTextures turns it off. The manifest records it.

-Input menus: D-pad down, D-pad down, A, one press every 6 s starting 5 s after the first pad poll, up to 500 s (input script
time counts from that poll, which the title reaches late). Any other value is a raw
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
    [switch]$NoLazyTextures                     # lazy slot 133 textures are on (X2W_D3D12_LAZY_TEXTURES=1, PR #2773); this turns them off
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$buildRoot = if ([System.IO.Path]::IsPathRooted($BuildDir)) { $BuildDir } else { Join-Path $repo $BuildDir }
$clock = [System.Diagnostics.Stopwatch]::StartNew()

function Get-CheckoutRevision {
    $head = (& git -C $repo rev-parse --short=8 HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $head -notmatch "^[0-9a-f]+$") { throw "tester_run: $repo is not a git checkout" }
    $changes = & git -C $repo --no-optional-locks status --porcelain --untracked-files=no
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

$source = Get-CheckoutRevision
$wanted = if ($source.Dirty) { "$($source.Commit)-dirty" } else { $source.Commit }
$built = Get-BuildRevision
$missing = Get-MissingDlls
if ($built -ne $wanted -or $missing.Count -gt 0) {
    $have = if ($built) { $built } else { "no build" }
    if ($built -eq $wanted) { $have = "$built without " + ($missing -join ", ") + ".dll" }
    if ($NoBuild) {
        throw "tester_run: the build in $buildRoot is from $have, this checkout is $wanted. Build it (cmake --build `"$buildRoot`") or run without -NoBuild."
    }
    if (-not (Test-Path -LiteralPath (Join-Path $buildRoot "CMakeCache.txt"))) {
        if ($BuildDir -ne "out/build/release") { throw "tester_run: $buildRoot is not configured (run cmake -S `"$repo`" -B `"$buildRoot`" first)" }
        Write-Host "tester_run: configuring $buildRoot (cmake --preset release)"
        Push-Location $repo
        try { & cmake --preset release } finally { Pop-Location }
        if ($LASTEXITCODE -ne 0) { throw "tester_run: the configure failed (cmake exit $LASTEXITCODE)" }
    }
    if ($FullBuild) {
        Write-Host "tester_run: build is from $have, checkout is ${wanted}: building everything (-FullBuild)"
        & cmake --build $buildRoot
    } else {
        Write-Host "tester_run: build is from $have, checkout is ${wanted}: building startup_runtime (the run's targets)"
        & cmake --build $buildRoot --target startup_runtime
    }
    if ($LASTEXITCODE -ne 0) { throw "tester_run: the build failed (cmake exit $LASTEXITCODE)" }
    $built = Get-BuildRevision
    $missing = Get-MissingDlls
    if ($missing.Count -gt 0) { throw "tester_run: after the build these DLLs are missing in ${buildRoot}: " + ($missing -join ", ") }
    if ($built -ne $wanted) {
        throw "tester_run: after the build the revision is $built, expected $wanted"
    }
}

# One shader cache folder per shader revision (the x2w-startup-shader-N strings of gfx2hlsl.py), kept across runs and across
# branch updates that leave the revision alone: the --watch translator needs minutes to catch up, so an emptied cache
# leaves pipelines untranslated (a black screen). A revision bump starts a new folder.
$runScript = Join-Path $PSScriptRoot "run.ps1"
$runParams = (Get-Command $runScript).Parameters
$cacheRoot = if ([System.IO.Path]::IsPathRooted($ShaderCacheRoot)) { $ShaderCacheRoot } else { Join-Path $repo $ShaderCacheRoot }

$python = Join-Path $repo "out\pe-env\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { $python = "python" }
$pack = Join-Path $PSScriptRoot "tester_pack.py"
$cacheTag = (& $python -B $pack cache-tag (Join-Path $PSScriptRoot "gfx2hlsl.py")).Trim()
if ($LASTEXITCODE -ne 0 -or -not $cacheTag) { throw "tester_run: cannot read the shader revisions of gfx2hlsl.py" }
$cache = Join-Path $cacheRoot "shader_cache_$cacheTag"
New-Item -ItemType Directory -Force $cache | Out-Null
$cacheAtStart = (& $python -B $pack cache-stats $cache) -split " "
$script = (& $python -B $pack input-preset $InputMode $Seconds)
if ($LASTEXITCODE -ne 0) { throw "tester_run: the input script was refused (see above)" }

$stamp = Get-Date -Format "yyyyMMdd-HHmm"
$outRoot = if ([System.IO.Path]::IsPathRooted($OutDir)) { $OutDir } else { Join-Path $repo $OutDir }
$logDir = Join-Path $outRoot "run_$stamp"
New-Item -ItemType Directory -Force $logDir | Out-Null

$env:X2W_D3D12_BUDGET = "1"
$LazyTextures = -not $NoLazyTextures
if ($LazyTextures) { $env:X2W_D3D12_LAZY_TEXTURES = "1" } else { Remove-Item Env:X2W_D3D12_LAZY_TEXTURES -ErrorAction SilentlyContinue }
$runArgs = @{
    Exe = $Exe; DataDir = $DataDir; LogDir = $logDir; Seconds = $Seconds; BuildDir = $BuildDir
    ShaderCache = $cache; ContinueAsserts = 1000; InputScript = $script
    Render = $true; Translate = $true; FakeUser = $true; GpuMirror = $true
}
if ($Deep) { $runArgs.TraceCalls = $true; $runArgs.TraceApi = $true }
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
    commit = $source.Commit; build_commit = $built; dirty = $source.Dirty; seconds = $Seconds
    input_script = $script; env = $envSet
    gpu = (($video | ForEach-Object { $_.Name }) -join "; ")
    gpu_driver = (($video | ForEach-Object { $_.DriverVersion }) -join "; ")
    windows_build = $os.BuildNumber; wall_seconds = [math]::Round($clock.Elapsed.TotalSeconds, 1)
    exit_code = $exitCode; deep = [bool]$Deep; lazy_textures = [bool]$LazyTextures; shader_cache = (Split-Path -Leaf $cache)
    translations_at_start = [int]$cacheAtStart[0]; requests_at_start = [int]$cacheAtStart[1]; failed_at_start = [int]$cacheAtStart[2]
}
$manifestFile = Join-Path ([System.IO.Path]::GetTempPath()) "tester_manifest_$stamp.json"
[System.IO.File]::WriteAllText($manifestFile, ($manifest | ConvertTo-Json -Depth 4), (New-Object System.Text.UTF8Encoding($false)))

$zip = Join-Path $outRoot ("report_{0}_{1}.zip" -f $source.Commit, $stamp)
$parts = @(& $python -B $pack pack $logDir $manifestFile $zip --limit-mb $LimitMB)
if ($LASTEXITCODE -ne 0) { throw "tester_run: packing the logs failed (see above); the logs stay in $logDir" }
Remove-Item -LiteralPath $manifestFile -Force
foreach ($line in $parts) {
    $path, $bytes = $line -split " (?=\d+$)"
    Write-Host ("Send this file: {0} ({1:N1} MB)" -f $path, ([double]$bytes / 1MB))
}
Write-Host ("Shader cache {0}: {1} translations were already present at start ({2} requests, {3} failed)." -f (Split-Path -Leaf $cache), $cacheAtStart[0], $cacheAtStart[1], $cacheAtStart[2])
Write-Host (& $python -B $pack first-poll $logDir)
if ($parts.Count -gt 1) { Write-Host "The report has $($parts.Count) parts: send all of them." }
exit $exitCode
