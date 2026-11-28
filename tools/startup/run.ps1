<#
.SYNOPSIS
Runs a console executable on the desktop stand-ins under the startup debugger (docs/startup-runtime.md).

.DESCRIPTION
Copies the stand-in DLLs from the build into -RunDir (a folder that holds the executable and the
package's other DLLs; prepare it yourself, never commit it), sets the permissive switches and starts
tools/startup/debug_launch.py. The run is offline: the debugger refuses sockets, NetBIOS and DNS.
Logs go to -LogDir; -LogDir.out holds the debugger's summary.

.EXAMPLE
.\tools\startup\run.ps1 -Exe D:\run\title.exe -DataDir G:\ -LogDir out\startup\run1 -Seconds 180 -Render -FakeUser
#>
param(
    [Parameter(Mandatory)] [string]$Exe,
    [Parameter(Mandatory)] [string]$DataDir,   # the package root the title opens its files from (its working directory)
    [Parameter(Mandatory)] [string]$LogDir,
    [int]$Seconds = 120,
    [string]$BuildDir = "out/build/release",
    [int]$MemoryGiB = 14,
    [int]$ContinueAsserts = 0,                 # main-image int3s to log and step over (title soft asserts)
    [string]$InputScript = "",                 # X2W_INPUT_SCRIPT, e.g. "4@70" = A at 70 s
    [string]$StoragePath = "",                 # X2W_STORAGE_PATH: root of the console drive the title sees (it appends x2w-gamesave)
    [string]$ShaderCache = "out/startup/shader_cache",
    [switch]$Render,                           # draw through translated pipelines
    [switch]$FakeUser,                         # answer the user API for one local user
    [switch]$GpuMirror,                        # map CPU-view pages into the paired GPU-view reservation
    [switch]$HostCpu,                          # rewrite instructions the host processor lacks (X2W_HOST_CPU)
    [switch]$Translate,                        # with -Render: translate new shader requests during the run (gfx2hlsl.py --watch)
    [switch]$FreshShaderCache,                 # with -Render: move a shader cache that is not empty aside first
    [string]$NativeShaders = "",               # with -Render: the directory of dxcompiler.dll and dxil.dll; compute requests are translated in the stand-in (X2W_NATIVE_SHADERS)
    [switch]$StartupBelow40,                   # place the 0x283 large-page startup blocks below 2^40 (X2W_XMEM_STARTUP_BELOW_40)
    [switch]$LowAddresses,                     # start the title with bottom-up ASLR off, so its own allocations stay below 2^40 (X2W_LOW_ADDRESSES, #1758)
    [switch]$NoNullIdentity,                   # with -Render: do not set X2W_DRAW_NULL_IDENTITY (an empty buffer slot then keeps its real address in the log)
    [switch]$LazyMemory,                       # commit physical pages and large startup blocks on first touch (X2W_XMEM_LAZY_COMMIT, X2W_XMEM_LAZY_STARTUP, X2W_XMEM_SHARED_VIEWS)
    [int]$TraceFiles = 1,                      # 1 = the debugger lists every file open in files.txt (--trace-files 1; slow on a hard disk), 0 = only denied/redirected ones
    [switch]$TraceCalls,                       # also break on runtime and movie-player calls (--trace-calls 1; slows the process)
    [double]$CaptureMenuEvery = 0,             # --capture-menu-every: pictures this often until the first loading-like picture (0 = off), then every -CaptureEvery
    [double]$CaptureEvery = 0,                 # --capture-every: save the title window as window_<sss>s.png every N seconds (0 = off)
    [int]$CommitHeadroomMiB = 0,               # --commit-headroom-mib: end the run cleanly when the system commit has less than this free (0 = off)
    [int]$CommitWarnMiB = 0,                   # --commit-warn-mib: log a warning when the system commit has less than this free (0 = off)
    [switch]$TraceApi                          # log runtime provider method calls (X2W_TRACE_API)
)
$ErrorActionPreference = "Stop"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$runDir = Split-Path -Parent (Resolve-Path $Exe)
$buildRoot = if ([System.IO.Path]::IsPathRooted($BuildDir)) { $BuildDir } else { Join-Path $repo $BuildDir }
foreach ($dll in "xmem", "d3d12_xs", "AcpHal", "XCRDAPI", "chad", "xapu", "xg_xs", "GameInput") {
    Copy-Item -LiteralPath (Join-Path $buildRoot "$dll.dll") -Destination $runDir -Force
}
$env:X2W_AUDIO_PERMISSIVE = "1"; $env:X2W_D3D12_DEVICE_NOOP = "1"; $env:X2W_D3D12_LIST_NOOP = "1"
$env:X2W_D3D12_PERMISSIVE = "1"; $env:X2W_XMEM_PERMISSIVE = "1"; $env:X2W_RUNTIME_PROXY = "1"
$config = Join-Path $DataDir "MicrosoftGame.config"
if (Test-Path -LiteralPath $config) {
    $text = Get-Content -LiteralPath $config -Raw
    if ($text -match "<TitleId>([^<]*)") { $env:X2W_TITLE_ID = $Matches[1] }
    if ($text -match "<SizeMB>([^<]*)") { $env:X2W_STORAGE_MB = $Matches[1] }
}
if ($StoragePath) { $env:X2W_STORAGE_PATH = $StoragePath }
if ($InputScript) { $env:X2W_INPUT_SCRIPT = $InputScript }
if ($FakeUser) { $env:X2W_FAKE_USER = "1" }
if (-not $FakeUser -and $env:X2W_FAKE_USER -ne "1") {
    Write-Warning "Without -FakeUser, user API calls go to the host gaming runtime (docs/startup-runtime.md)."
}
if ($GpuMirror) { $env:X2W_XMEM_GPU_MIRROR = "1" }
if ($HostCpu) { $env:X2W_HOST_CPU = "1" }
if ($TraceApi) { $env:X2W_TRACE_API = "1" }
if ($StartupBelow40) { $env:X2W_XMEM_STARTUP_BELOW_40 = "1" }
if ($LowAddresses) { $env:X2W_LOW_ADDRESSES = "1" }
if ($LazyMemory) {
    $env:X2W_XMEM_LAZY_COMMIT = "1"; $env:X2W_XMEM_LAZY_STARTUP = "1"; $env:X2W_XMEM_SHARED_VIEWS = "1"
}
$python = Join-Path $repo "out\pe-env\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { $python = "python" }
$revisionTool = Join-Path $PSScriptRoot "shader_cache_revision.py"
if ($FreshShaderCache -and -not $Render) { Write-Warning "-FreshShaderCache needs -Render (no shader cache without it); ignored" }
if ($Render) {
    if ($FreshShaderCache) {
        $cachePath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($ShaderCache)
        & $python -B $revisionTool $cachePath --fresh | ForEach-Object { Write-Host $_ }
    }
    New-Item -ItemType Directory -Force $ShaderCache | Out-Null
    $env:X2W_D3D12_RENDER = "1"
    if ($NoNullIdentity) { Remove-Item Env:X2W_DRAW_NULL_IDENTITY -ErrorAction SilentlyContinue } else { $env:X2W_DRAW_NULL_IDENTITY = "1" }
    $env:X2W_SHADER_CACHE = (Resolve-Path $ShaderCache).Path
    if ($NativeShaders) { $env:X2W_NATIVE_SHADERS = (Resolve-Path $NativeShaders).Path }
}
if ($NativeShaders -and -not $Render) { Write-Warning "-NativeShaders needs -Render (no shader requests without it); ignored" }
New-Item -ItemType Directory -Force $LogDir | Out-Null
# Requests keyed by another revision are all refused, and the translator's log does not say why: warn once,
# before the watcher starts (revision.txt of an earlier run) or after the run (the one this run's stand-ins wrote).
$checkRevision = {
    $note = & $python -B $revisionTool $env:X2W_SHADER_CACHE --translator (Join-Path $PSScriptRoot 'gfx2hlsl.py')
    if ($LASTEXITCODE -eq 0) { return $false }
    if ($LASTEXITCODE -eq 1) { Write-Warning ($note -join ' ') }
    else { Write-Warning "the shader cache revision was not checked (python exit $LASTEXITCODE)" }
    return $true
}
$revisionWarned = $false
$translator = $null
$watchShaders = $Translate -and $Render
if ($Translate -and -not $Render) { Write-Warning "-Translate needs -Render (no shader cache without it); ignored" }
if ($watchShaders) {
    $revisionWarned = & $checkRevision
    $translator = Start-Process -FilePath $python -PassThru -NoNewWindow `
        -ArgumentList "-B", "`"$(Join-Path $PSScriptRoot 'gfx2hlsl.py')`"", "--watch", "`"$env:X2W_SHADER_CACHE`"" `
        -RedirectStandardOutput (Join-Path $LogDir "translate.txt") -RedirectStandardError (Join-Path $LogDir "translate_err.txt")
    # The translator and its process pool (every core but one) must never run above the title. A venv or launcher python is a
    # process tree (launcher + real interpreter), so the class is set on the root and on every descendant, for about two
    # seconds so a child that appears a little later is caught; the pool workers the interpreter creates later inherit it.
    $treeOf = {
        param([int]$RootId)
        $ids = @($RootId)
        for ($i = 0; $i -lt $ids.Count; $i++) {
            foreach ($child in @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$($ids[$i])" -ErrorAction SilentlyContinue)) {
                if ($ids -notcontains [int]$child.ProcessId) { $ids += [int]$child.ProcessId }
            }
        }
        return $ids
    }
    $lowerTree = {
        param([int]$RootId)
        $ids = @(& $treeOf $RootId)
        foreach ($id in $ids) {
            try { (Get-Process -Id $id -ErrorAction Stop).PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal } catch { }
        }
        return $ids.Count
    }
    try {
        $deadline = [DateTime]::UtcNow.AddSeconds(2)
        do { $null = & $lowerTree $translator.Id; Start-Sleep -Milliseconds 250 } while ([DateTime]::UtcNow -lt $deadline -and -not $translator.HasExited)
    } catch { Write-Warning "could not lower the translator's priority: $($_.Exception.Message)" }
}
$traceArgs = @()
if ($TraceCalls) { $traceArgs = @("--trace-calls", "1") }
$captureArgs = @()
if ($CaptureEvery -ne 0) { $captureArgs = @("--capture-every", "$CaptureEvery") }
if ($CaptureEvery -ne 0 -and $CaptureMenuEvery -ne 0) { $captureArgs += @("--capture-menu-every", "$CaptureMenuEvery") }
if ($CommitHeadroomMiB -gt 0) { $captureArgs += @("--commit-headroom-mib", "$CommitHeadroomMiB") }
if ($CommitWarnMiB -gt 0) { $captureArgs += @("--commit-warn-mib", "$CommitWarnMiB") }
try {
    # Windows PowerShell 5.1 wraps redirected native stderr as an error record. Keep that
    # from aborting the launcher before it can report the debugger's exit code.
    & {
        $ErrorActionPreference = "Continue"
        & $python -B (Join-Path $PSScriptRoot "debug_launch.py") $Exe $DataDir $LogDir --timeout $Seconds --memory-gib $MemoryGiB `
            --trace-files $TraceFiles --continue-asserts $ContinueAsserts @traceArgs @captureArgs
        $script:debuggerExit = $LASTEXITCODE
    } *> "$LogDir.out"
} finally {
    # taskkill /T needs a live parent PID to reach the watcher's workers.
    if ($translator -and -not $translator.HasExited) {
        # Stop the workers as well. The watcher or a worker can exit during taskkill;
        # in Windows PowerShell 5.1 its redirected stderr would then abort the launcher.
        $treeIds = @(& $treeOf $translator.Id)
        & {
            $ErrorActionPreference = "Continue"
            & taskkill.exe /T /F /PID $translator.Id 2>&1 | Out-Null
        }
        # taskkill returns before the processes are gone, and the watcher and its workers hold translate.txt and
        # translate_err.txt open: wait until every one has ended, so the log files are free when this script exits
        # (a caller that removes the run folder right after would otherwise meet a sharing violation).
        foreach ($id in $treeIds) {
            try { [void](Get-Process -Id $id -ErrorAction Stop).WaitForExit(10000) } catch { }
        }
    }
}
if ($watchShaders -and -not $revisionWarned) { $null = & $checkRevision }
Get-Content "$LogDir.out" -TotalCount 5
exit $debuggerExit
