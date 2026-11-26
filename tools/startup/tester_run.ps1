<#
.SYNOPSIS
One command for a tester: check the build, run the title offline with a fresh shader cache, mail-sized zip at the end.

.DESCRIPTION
1. Checks that the build in -BuildDir was made from this checkout's current commit (the revision the build writes
   into generated\x2w_build_rev.cpp). If not, it builds (cmake --build) or, with -NoBuild, stops with a message.
2. Moves the old shader cache aside and starts with an empty one.
3. Runs run.ps1 with -Render -Translate -FakeUser -GpuMirror -ContinueAsserts 1000 and X2W_D3D12_BUDGET=1,
   the input script and, when run.ps1 knows it, the capture interval.
4. Packs report_<commit>_<yyyymmdd-hhmm>.zip with EVERY file the run wrote in its log folder (stdout, stderr, events,
   finished, threads, files, trap.log, translate, translate_err, window*.png, bink_luma.png ...), the debugger
   summary and manifest.json, and prints each path and size. Not included: the shader cache, memory dumps and game
   files. Over -LimitMB (default 20, so each part fits a mail) the zip is split into numbered parts
   (report_..._part01of03.zip); no log is dropped.

-Deep also sets -TraceCalls and -TraceApi: a slower run with more detail in the logs.

-Input menus: D-pad down, D-pad down, A, one press every 6 s starting 5 s after the first pad poll (input script
time counts from that poll, which the title reaches late: about 150 s after start in one tester run). Any other value is a raw
X2W_INPUT_SCRIPT (docs/startup-runtime.md, "Input script").

.EXAMPLE
.\tools\startup\tester_run.ps1 -Exe D:\run\title.exe -DataDir G:\ -Seconds 420 -Input menus -CaptureEvery 15
#>
param(
    [Parameter(Mandatory)] [string]$Exe,
    [Parameter(Mandatory)] [string]$DataDir,
    [int]$Seconds = 420,
    [string]$Input = "menus",
    [int]$CaptureEvery = 15,                    # seconds between window captures; 0 = only the final window.png
    [string]$BuildDir = "out/build/release",
    [string]$OutDir = "out/tester",
    [string]$ShaderCache = "out/startup/shader_cache",
    [int]$LimitMB = 20,
    [switch]$NoBuild,                           # stop instead of building when the build is not from this commit
    [switch]$Deep                               # also -TraceCalls and -TraceApi: slower, more detailed
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

$source = Get-CheckoutRevision
$wanted = if ($source.Dirty) { "$($source.Commit)-dirty" } else { $source.Commit }
$built = Get-BuildRevision
if ($built -ne $wanted) {
    $have = if ($built) { $built } else { "no build" }
    if ($NoBuild) {
        throw "tester_run: the build in $buildRoot is from $have, this checkout is $wanted. Build it (cmake --build `"$buildRoot`") or run without -NoBuild."
    }
    Write-Host "tester_run: build is from $have, checkout is $wanted: building"
    & cmake --build $buildRoot
    if ($LASTEXITCODE -ne 0) { throw "tester_run: the build failed (cmake exit $LASTEXITCODE)" }
    $built = Get-BuildRevision
    if ($built -ne $wanted) {
        throw "tester_run: after the build the revision is $built, expected $wanted"
    }
}

# A fresh shader cache: the old folder is kept beside it.
$cache = if ([System.IO.Path]::IsPathRooted($ShaderCache)) { $ShaderCache } else { Join-Path $repo $ShaderCache }
if (Test-Path -LiteralPath $cache) {
    Move-Item -LiteralPath $cache -Destination ("{0}.old-{1}" -f $cache, (Get-Date -Format "yyyyMMdd-HHmmss"))
}
New-Item -ItemType Directory -Force $cache | Out-Null

$python = Join-Path $repo "out\pe-env\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { $python = "python" }
$pack = Join-Path $PSScriptRoot "tester_pack.py"
$script = (& $python -B $pack input-preset $Input $Seconds)
if ($LASTEXITCODE -ne 0) { throw "tester_run: the input script was refused (see above)" }

$stamp = Get-Date -Format "yyyyMMdd-HHmm"
$outRoot = if ([System.IO.Path]::IsPathRooted($OutDir)) { $OutDir } else { Join-Path $repo $OutDir }
$logDir = Join-Path $outRoot "run_$stamp"
New-Item -ItemType Directory -Force $logDir | Out-Null

$env:X2W_D3D12_BUDGET = "1"
$runArgs = @{
    Exe = $Exe; DataDir = $DataDir; LogDir = $logDir; Seconds = $Seconds; BuildDir = $BuildDir
    ShaderCache = $cache; ContinueAsserts = 1000; InputScript = $script
    Render = $true; Translate = $true; FakeUser = $true; GpuMirror = $true
}
if ($Deep) { $runArgs.TraceCalls = $true; $runArgs.TraceApi = $true }
$runScript = Join-Path $PSScriptRoot "run.ps1"
if ($CaptureEvery -gt 0) {
    if ((Get-Command $runScript).Parameters.ContainsKey("CaptureEvery")) { $runArgs.CaptureEvery = $CaptureEvery }
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
    exit_code = $exitCode; deep = [bool]$Deep
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
Write-Host (& $python -B $pack first-poll $logDir)
if ($parts.Count -gt 1) { Write-Host "The report has $($parts.Count) parts: send all of them." }
exit $exitCode
